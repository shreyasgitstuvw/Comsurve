"""
Signal correlator — Phase 8.

For each anomaly that was just embedded (status='processed', no signal_alert yet):
  1. Search Qdrant for top-5 similar historical anomalies (cosine >= 0.70)
  2. Classify the alert: 'similar_historical' if matches found, 'novel_event' if not
  3. Write a SignalAlert row to signal_alerts
  4. Record the current price as price_at_alert (monitoring window starts here)

This runs immediately after ai_engine_runner in the daily batch.
"""

import json
from datetime import datetime

from sqlalchemy import text

from ai_engine.qdrant_manager import QdrantManager
from shared.commodity_registry import COMMODITY_LIST
from shared.db import get_session
from shared.logger import get_logger
from shared.models import AnomalyEvent, EmbeddingCache, SignalAlert, unpack_vector

logger = get_logger(__name__)

MIN_SIMILARITY = 0.55         # cosine floor — below this, treat as novel
CROSS_COMMODITY_SIMILARITY = 0.75  # higher bar for cross-commodity analogs
TOP_K = 8                 # max same-commodity analogs (was 5)


def _get_current_price(commodity: str) -> float | None:
    """Fetch the most recent price for this commodity from processed_features."""
    with get_session() as session:
        row = session.execute(
            text("""
                SELECT pf.value
                FROM processed_features pf
                JOIN raw_ingestion ri ON pf.raw_ingestion_id = ri.id
                WHERE ri.commodity = :commodity
                  AND pf.feature_type = 'price'
                ORDER BY ri.timestamp DESC
                LIMIT 1
            """),
            {"commodity": commodity},
        ).fetchone()
    return float(row[0]) if row and row[0] is not None else None


def _alert_already_exists(anomaly_event_id: int) -> bool:
    with get_session() as session:
        row = session.execute(
            text("SELECT id FROM signal_alerts WHERE anomaly_event_id = :id"),
            {"id": anomaly_event_id},
        ).fetchone()
    return row is not None


def run_signal_correlation() -> dict:
    """
    Correlate all recently processed anomalies that don't yet have a signal alert.
    Returns summary dict.
    """
    qdrant = QdrantManager()
    created = 0
    novel = 0
    skipped = 0

    try:
        # Find processed anomalies with embeddings but no signal_alert yet
        with get_session() as session:
            rows = session.execute(
                text("""
                    SELECT ae.id, ae.commodity, ae.anomaly_type, ae.severity,
                           ae.detected_at, ec.vector_blob, ec.vector_json
                    FROM anomaly_events ae
                    JOIN embeddings_cache ec ON ec.anomaly_event_id = ae.id
                    LEFT JOIN signal_alerts sa ON sa.anomaly_event_id = ae.id
                    WHERE ae.status = 'processed'
                      AND sa.id IS NULL
                    ORDER BY ae.detected_at DESC
                    LIMIT 100
                """),
            ).fetchall()

        logger.info("signal_correlator_start", candidates=len(rows))

        for row in rows:
            anomaly_id, commodity, anomaly_type, severity, detected_at, vector_blob, vector_json = row

            try:
                if vector_blob is not None:
                    vector = unpack_vector(vector_blob)
                elif vector_json:
                    vector = json.loads(vector_json)
                else:
                    logger.warning(
                        "signal_correlator_no_vector",
                        anomaly_id=anomaly_id,
                        reason="both vector_blob and vector_json are NULL",
                    )
                    skipped += 1
                    continue
            except Exception as exc:
                logger.warning(
                    "signal_correlator_vector_unpack_failed",
                    anomaly_id=anomaly_id,
                    error=str(exc),
                )
                skipped += 1
                continue

            # Same-commodity nearest-neighbor search
            similar = qdrant.search_similar(
                commodity=commodity,
                vector=vector,
                top_k=TOP_K,
                exclude_id=anomaly_id,
                min_score=MIN_SIMILARITY,
            )

            correlated_ids = [int(r.id) for r in similar]
            similarity_scores = [round(r.score, 4) for r in similar]

            # Cross-commodity search — higher threshold, other commodity collections only
            for other_commodity in COMMODITY_LIST:
                if other_commodity == commodity:
                    continue
                cross_similar = qdrant.search_similar(
                    commodity=other_commodity,
                    vector=vector,
                    top_k=3,
                    exclude_id=None,
                    min_score=CROSS_COMMODITY_SIMILARITY,
                )
                for r in cross_similar:
                    aid = int(r.id)
                    if aid not in correlated_ids:
                        correlated_ids.append(aid)
                        similarity_scores.append(round(r.score, 4))

            alert_type = "similar_historical" if correlated_ids else "novel_event"

            if alert_type == "novel_event":
                novel += 1

            price_at_alert = _get_current_price(commodity)

            with get_session() as session:
                session.add(SignalAlert(
                    anomaly_event_id=anomaly_id,
                    commodity=commodity,
                    alert_type=alert_type,
                    correlated_anomaly_ids=json.dumps(correlated_ids),
                    similarity_scores=json.dumps(similarity_scores),
                    price_at_alert=price_at_alert,
                    monitoring_complete=False,
                    created_at=datetime.utcnow(),
                ))

            logger.info(
                "signal_alert_created",
                anomaly_id=anomaly_id,
                commodity=commodity,
                alert_type=alert_type,
                correlated_count=len(correlated_ids),
                top_similarity=similarity_scores[0] if similarity_scores else None,
                price_at_alert=price_at_alert,
            )
            created += 1

    finally:
        qdrant.close()

    summary = {"alerts_created": created, "novel_events": novel, "skipped": skipped}

    # Run prediction engine immediately for new alerts so predictions are available
    # before the next daily batch (same-day predictions instead of +1 day lag)
    if created > 0:
        try:
            from ai_engine.prediction_engine import run_prediction_engine
            pred_result = run_prediction_engine()
            summary["predictions"] = pred_result
            logger.info("signal_correlator_triggered_predictions", **pred_result)
        except Exception as exc:
            logger.warning("signal_correlator_prediction_trigger_failed", error=str(exc))

    logger.info("signal_correlator_complete", **summary)
    return summary
