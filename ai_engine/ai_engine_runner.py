"""
AI Engine runner — the bridge between anomaly detection and signal correlation.

On each run (scheduled daily at 03:00):
  1. Poll anomaly_events WHERE status = 'new'
  2. Build context payloads for all pending anomalies
  3. Call Gemini batch embedding → one API call for the whole batch
  4. Upsert all vectors to Qdrant (commodity collections)
  5. Write EmbeddingCache rows to SQLite
  6. Mark all processed anomalies as 'processed'

If any step fails for an individual anomaly, it is re-queued (status reset to 'new')
so the next run picks it up. A permanent failure counter prevents infinite loops.

Batch embedding (F1) reduces Gemini API calls from N (one per anomaly) to 1 per run,
staying within the 15 RPM free-tier limit even during high-volume periods.
"""

import json
from datetime import datetime

from ai_engine.embedding_generator import build_context_payload
from ai_engine.gemini_client import GeminiClient
from ai_engine.qdrant_manager import QdrantManager
from shared.db import get_session, init_db
from shared.logger import get_logger
from shared.models import AnomalyEvent, EmbeddingCache, pack_vector

logger = get_logger(__name__)

MAX_RETRIES_PER_ANOMALY = 3  # stored in metadata_json; after this, skip permanently


def run(batch_size: int = 50) -> dict:
    """
    Process up to batch_size pending anomaly events using batch embedding.
    Returns a summary dict.
    """
    client = GeminiClient()
    qdrant = QdrantManager()

    processed = 0
    failed = 0
    skipped = 0
    pending_data: list[dict] = []

    try:
        # ── 1. Fetch pending anomalies ─────────────────────────────────────────
        with get_session() as session:
            pending = (
                session.query(AnomalyEvent)
                .filter(AnomalyEvent.status == "new")
                .order_by(AnomalyEvent.detected_at)
                .limit(batch_size)
                .all()
            )
            # Snapshot to plain dicts before session closes to avoid DetachedInstanceError
            pending_data = [
                {
                    "id": a.id,
                    "commodity": a.commodity,
                    "anomaly_type": a.anomaly_type,
                    "severity": a.severity,
                    "detected_at": a.detected_at,
                    "source_ids": a.source_ids,
                    "metadata_json": a.metadata_json,
                }
                for a in pending
            ]

        logger.info("ai_engine_run_start", pending_count=len(pending_data))

        # ── 2. Filter out permanently failed anomalies ─────────────────────────
        to_process: list[dict] = []
        for anomaly_data in pending_data:
            try:
                meta = json.loads(anomaly_data["metadata_json"] or "{}")
            except json.JSONDecodeError:
                meta = {}
            retry_count = meta.get("_embed_retries", 0)
            if retry_count >= MAX_RETRIES_PER_ANOMALY:
                logger.warning(
                    "ai_engine_skip_permanent_failure",
                    anomaly_id=anomaly_data["id"],
                    retries=retry_count,
                )
                skipped += 1
            else:
                anomaly_data["_meta"] = meta
                anomaly_data["_retry_count"] = retry_count
                to_process.append(anomaly_data)

        if not to_process:
            return {"processed": 0, "failed": 0, "skipped_permanent": skipped,
                    "total_pending": len(pending_data)}

        # ── 3. Mark all as embedding_queued ────────────────────────────────────
        ids_to_queue = [a["id"] for a in to_process]
        with get_session() as session:
            session.query(AnomalyEvent).filter(
                AnomalyEvent.id.in_(ids_to_queue)
            ).update({"status": "embedding_queued"}, synchronize_session=False)

        # ── 4. Build context payloads & batch-embed ────────────────────────────
        stubs = [_AnomalyStub(a) for a in to_process]
        contexts = [build_context_payload(stub) for stub in stubs]

        try:
            vectors = client.batch_embed(contexts)
        except Exception as exc:
            # If the whole batch fails, mark all as failed and bail
            logger.error("ai_engine_batch_embed_failed", error=str(exc),
                         batch_size=len(to_process))
            for anomaly_data in to_process:
                meta = anomaly_data["_meta"]
                meta["_embed_retries"] = anomaly_data["_retry_count"] + 1
                with get_session() as session:
                    session.query(AnomalyEvent).filter(
                        AnomalyEvent.id == anomaly_data["id"]
                    ).update({"status": "new", "metadata_json": json.dumps(meta)})
            return {"processed": 0, "failed": len(to_process),
                    "skipped_permanent": skipped, "total_pending": len(pending_data)}

        # ── 5. Write to Qdrant + EmbeddingCache per anomaly ───────────────────
        for anomaly_data, stub, context_text, vector in zip(
            to_process, stubs, contexts, vectors
        ):
            anomaly_id = anomaly_data["id"]
            try:
                qdrant_payload = {
                    "anomaly_event_id": anomaly_id,
                    "anomaly_type": anomaly_data["anomaly_type"],
                    "commodity": anomaly_data["commodity"],
                    "severity": anomaly_data["severity"],
                    "detected_at": anomaly_data["detected_at"].isoformat(),
                    "context_snippet": context_text[:300],
                }
                qdrant.upsert_embedding(
                    anomaly_event_id=anomaly_id,
                    commodity=anomaly_data["commodity"],
                    vector=vector,
                    payload=qdrant_payload,
                )

                with get_session() as session:
                    existing = (
                        session.query(EmbeddingCache)
                        .filter(EmbeddingCache.anomaly_event_id == anomaly_id)
                        .first()
                    )
                    if not existing:
                        session.add(EmbeddingCache(
                            anomaly_event_id=anomaly_id,
                            model="gemini-embedding-001",
                            vector_blob=pack_vector(vector),   # compact binary (12 KB)
                            vector_json=None,                  # no longer written for new rows
                            context_payload=context_text,
                            created_at=datetime.utcnow(),
                        ))
                    session.query(AnomalyEvent).filter(
                        AnomalyEvent.id == anomaly_id
                    ).update({"status": "processed"})

                logger.info(
                    "ai_engine_anomaly_processed",
                    anomaly_id=anomaly_id,
                    commodity=anomaly_data["commodity"],
                    anomaly_type=anomaly_data["anomaly_type"],
                )
                processed += 1

            except Exception as exc:
                logger.error(
                    "ai_engine_anomaly_failed",
                    anomaly_id=anomaly_id,
                    error=str(exc),
                    retry_count=anomaly_data["_retry_count"] + 1,
                )
                meta = anomaly_data["_meta"]
                meta["_embed_retries"] = anomaly_data["_retry_count"] + 1
                with get_session() as session:
                    session.query(AnomalyEvent).filter(
                        AnomalyEvent.id == anomaly_id
                    ).update({"status": "new", "metadata_json": json.dumps(meta)})
                failed += 1

    finally:
        qdrant.close()

    summary = {
        "processed": processed,
        "failed": failed,
        "skipped_permanent": skipped,
        "total_pending": len(pending_data),
    }
    logger.info("ai_engine_run_complete", **summary)
    return summary


class _AnomalyStub:
    """Lightweight stand-in for AnomalyEvent ORM object (avoids detached session issues)."""
    def __init__(self, data: dict):
        self.id = data["id"]
        self.commodity = data["commodity"]
        self.anomaly_type = data["anomaly_type"]
        self.severity = data["severity"]
        self.detected_at = data["detected_at"]
        self.source_ids = data["source_ids"]
        self.metadata_json = data["metadata_json"]


def run_full_ai_batch(batch_size: int = 50) -> dict:
    """
    Full AI batch: embed pending anomalies → correlate → return combined summary.
    This is what the scheduler calls.
    """
    from ai_engine.signal_correlator import run_signal_correlation

    embed_result = run(batch_size=batch_size)
    correlate_result = run_signal_correlation()
    return {"embedding": embed_result, "correlation": correlate_result}


if __name__ == "__main__":
    init_db()
    result = run_full_ai_batch()
    print(result)
