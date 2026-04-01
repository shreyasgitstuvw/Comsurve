"""
Llama client via Ollama REST API.

Used for evaluation + feedback roles to prevent circular contamination:
  - Gemini: prediction_engine, causality_engine  (cloud, synthesis)
  - Llama:  evaluation_engine, feedback_controller (local, sandboxed)

Falls back to GeminiClient when Ollama is unavailable.
"""

import json
import os
from typing import Optional

from shared.logger import get_logger

logger = get_logger(__name__)

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "90"))


class LlamaClient:
    """
    Thin wrapper around the Ollama /api/generate endpoint.

    Usage:
        client = LlamaClient()
        if client.available:
            text = client.generate_text(prompt)
        else:
            # fall back to GeminiClient
    """

    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model
        self.base_url = OLLAMA_BASE_URL.rstrip("/")
        self._available: Optional[bool] = None  # lazy-checked on first use

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Return True if Ollama is running and the model is available."""
        try:
            import urllib.request
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
            return self.model in models or any(self.model in m for m in models)
        except Exception as exc:
            logger.debug("ollama_ping_failed", error=str(exc))
            return False

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = self.ping()
            logger.info("ollama_availability", available=self._available, model=self.model)
        return self._available

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate_text(self, prompt: str) -> str:
        """
        Call Ollama /api/generate (non-streaming).
        Returns the response text as a string (may be JSON).
        Raises RuntimeError if Ollama is unavailable or the call fails.
        """
        import urllib.request
        import urllib.error

        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,   # low temperature for evaluation consistency
                "num_predict": 2048,
            },
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=OLLAMA_TIMEOUT) as resp:
                raw = resp.read().decode()
            data = json.loads(raw)
            response_text = data.get("response", "")
            logger.debug("ollama_generate_ok", model=self.model, chars=len(response_text))
            return response_text
        except urllib.error.URLError as exc:
            self._available = False  # mark unavailable for remainder of session
            raise RuntimeError(f"Ollama unreachable: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"Ollama generate failed: {exc}") from exc


def get_evaluation_client():
    """
    Return the preferred client for evaluation/feedback (Llama if available,
    else Gemini as fallback). Always returns an object with .generate_text().
    """
    llama = LlamaClient()
    if llama.available:
        logger.info("evaluation_client_selected", client="llama", model=llama.model)
        return llama

    logger.info("evaluation_client_selected", client="gemini_fallback")
    from ai_engine.gemini_client import GeminiClient
    return GeminiClient()
