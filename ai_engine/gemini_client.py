"""
Gemini API client with deterministic 15 RPM rate limiting.

Rate limit strategy:
  - Maintain a deque of UTC timestamps for calls made in the last 60 seconds.
  - Before each call: evict timestamps older than 60s.
  - If len(deque) >= 15, sleep until the oldest timestamp is exactly 60s old.
  - This is deterministic (no busy-wait), correct under burst, and requires no
    external library.

Uses google-genai SDK (google.generativeai is deprecated as of early 2025).
"""

import time
from collections import deque
from datetime import datetime, timezone

from google import genai
from google.genai import types

from shared.config import settings
from shared.logger import get_logger

logger = get_logger(__name__)

EMBEDDING_MODEL = "models/gemini-embedding-001"
EMBEDDING_DIMS = 3072
RPM_LIMIT = 15
WINDOW_SECONDS = 60

# Text generation model priority — if primary hits daily quota exhaustion, try fallbacks in order.
GENERATE_MODELS = [
    "models/gemini-2.0-flash",
    "models/gemini-2.0-flash-lite",
    "models/gemini-2.5-flash",
]


class GeminiClient:
    """
    Singleton-safe Gemini client with built-in rate limiter.
    Instantiate once per process (ai_engine_runner creates one and passes it around).
    """

    def __init__(self):
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._call_timestamps: deque[float] = deque()

    def _enforce_rate_limit(self) -> None:
        """Block until a rate-limit slot is available. Uses deque timestamp approach."""
        now = time.monotonic()

        # Evict timestamps older than WINDOW_SECONDS
        while self._call_timestamps and (now - self._call_timestamps[0]) >= WINDOW_SECONDS:
            self._call_timestamps.popleft()

        if len(self._call_timestamps) >= RPM_LIMIT:
            # Sleep until the oldest call falls outside the 60s window
            oldest = self._call_timestamps[0]
            sleep_for = WINDOW_SECONDS - (now - oldest) + 0.05  # +50ms buffer
            logger.info("gemini_rate_limit_sleep", sleep_seconds=round(sleep_for, 2))
            time.sleep(max(0, sleep_for))

            # Re-evict after sleep
            now = time.monotonic()
            while self._call_timestamps and (now - self._call_timestamps[0]) >= WINDOW_SECONDS:
                self._call_timestamps.popleft()

        self._call_timestamps.append(time.monotonic())

    def embed(self, text: str, retries: int = 3) -> list[float]:
        """
        Generate an embedding vector for the given text.
        Returns a list of EMBEDDING_DIMS floats.
        Retries up to `retries` times on transient API errors.
        """
        last_exc = None
        for attempt in range(retries):
            try:
                self._enforce_rate_limit()
                result = self._client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=text,
                )
                vector = list(result.embeddings[0].values)
                logger.info(
                    "gemini_embed_ok",
                    dims=len(vector),
                    text_len=len(text),
                    queue_depth=len(self._call_timestamps),
                )
                return vector
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(
                    "gemini_embed_retry",
                    attempt=attempt + 1,
                    error=str(exc),
                    wait_seconds=wait,
                )
                time.sleep(wait)

        raise RuntimeError(f"Gemini embedding failed after {retries} retries: {last_exc}") from last_exc

    def batch_embed(self, texts: list[str], retries: int = 3) -> list[list[float]]:
        """
        Generate embedding vectors for multiple texts in a single API call.

        Uses batchEmbedContents when possible (1 rate-limit slot for the whole batch),
        which is more efficient than calling embed() in a loop when processing many
        anomalies at once.

        Falls back to sequential embed() calls if the SDK or API rejects the batch.

        Returns a list of vectors in the same order as `texts`.
        Raises RuntimeError if all retries are exhausted.
        """
        if not texts:
            return []

        last_exc = None
        for attempt in range(retries):
            try:
                self._enforce_rate_limit()
                result = self._client.models.embed_content(
                    model=EMBEDDING_MODEL,
                    contents=texts,
                )
                vectors = [list(emb.values) for emb in result.embeddings]
                logger.info(
                    "gemini_batch_embed_ok",
                    batch_size=len(texts),
                    dims=len(vectors[0]) if vectors else 0,
                    queue_depth=len(self._call_timestamps),
                )
                return vectors
            except Exception as exc:
                last_exc = exc
                exc_str = str(exc)
                # If the batch endpoint is unsupported (older SDK or quota issues),
                # fall back to sequential per-item embedding after the loop.
                if "INVALID_ARGUMENT" in exc_str or "NOT_FOUND" in exc_str:
                    logger.warning(
                        "gemini_batch_embed_fallback",
                        reason=exc_str[:200],
                        batch_size=len(texts),
                    )
                    break  # fall through to sequential fallback
                wait = 2 ** attempt
                logger.warning(
                    "gemini_batch_embed_retry",
                    attempt=attempt + 1,
                    error=exc_str,
                    wait_seconds=wait,
                )
                time.sleep(wait)
        else:
            # All retries exhausted without a break — raise the last error
            raise RuntimeError(
                f"Gemini batch embed failed after {retries} retries: {last_exc}"
            ) from last_exc

        # Sequential fallback: embed each text individually
        logger.info("gemini_batch_embed_sequential_fallback", batch_size=len(texts))
        return [self.embed(text, retries=retries) for text in texts]

    def generate_text(self, prompt: str, retries: int = 3) -> str:
        """
        Generate text via Gemini Flash (used by causality, prediction, evaluation engines).
        Tries each model in GENERATE_MODELS in order. If a model hits a daily quota
        exhaustion (RESOURCE_EXHAUSTED with limit: 0), it is skipped and the next
        fallback model is attempted. Per-minute rate limits are retried normally.
        Returns the response text as a string.
        """
        last_exc: Exception | None = None

        for model in GENERATE_MODELS:
            model_exhausted = False
            for attempt in range(retries):
                try:
                    self._enforce_rate_limit()
                    result = self._client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.2,
                        ),
                    )
                    if model != GENERATE_MODELS[0]:
                        logger.info("gemini_generate_fallback_used", model=model)
                    return result.text
                except Exception as exc:
                    last_exc = exc
                    exc_str = str(exc)
                    # Skip to next model on: daily quota exhaustion or model not found
                    quota_exhausted = "RESOURCE_EXHAUSTED" in exc_str and "limit: 0" in exc_str
                    not_found = "NOT_FOUND" in exc_str
                    if quota_exhausted or not_found:
                        logger.warning(
                            "gemini_generate_model_skip",
                            model=model,
                            reason="quota_exhausted" if quota_exhausted else "not_found",
                            fallback_next=True,
                        )
                        model_exhausted = True
                        break
                    wait = 2 ** attempt
                    logger.warning(
                        "gemini_generate_retry",
                        model=model,
                        attempt=attempt + 1,
                        error=exc_str,
                        wait_seconds=wait,
                    )
                    time.sleep(wait)

            if model_exhausted:
                continue  # try next model
            # If we exhausted retries without a quota error, stop trying further models
            break

        raise RuntimeError(f"Gemini generate failed on all models: {last_exc}") from last_exc
