"""
Anomaly detector: scans processed_features for statistically significant deviations.

Methods:
  - Price: Z-score on 30-day rolling window of pct_change values
  - Sentiment: spike detection — compound score > |0.6| with 30-day mean baseline

The status field on AnomalyEvent is the queue mechanism for the AI engine:
  new → embedding_queued → processed
"""

import json
import math
import statistics
from datetime import datetime, timedelta

from sqlalchemy import text

from shared.commodity_registry import ANOMALY_THRESHOLDS
from shared.db import get_session
from shared.logger import get_logger
from shared.models import AnomalyEvent, ProcessedFeature, RawIngestion

logger = get_logger(__name__)

# Fallback thresholds (used if commodity not in ANOMALY_THRESHOLDS)
_DEFAULT_PRICE_ZSCORE = 2.0
_DEFAULT_SENTIMENT_SPIKE = 0.6
# Public aliases for use in tests and external callers
PRICE_ZSCORE_THRESHOLD = _DEFAULT_PRICE_ZSCORE
SENTIMENT_SPIKE_THRESHOLD = _DEFAULT_SENTIMENT_SPIKE
SENTIMENT_MEAN_WINDOW_DAYS = 30
PRICE_WINDOW_DAYS = 30
MIN_WINDOW_POINTS = 5           # need at least 5 points to compute meaningful Z-score


def _compute_zscore(value: float, window_values: list[float]) -> float | None:
    """Z-score of value relative to window_values. Returns None if window too small."""
    if len(window_values) < MIN_WINDOW_POINTS:
        return None
    mean = statistics.mean(window_values)
    stdev = statistics.stdev(window_values)
    if stdev == 0:
        return None
    return (value - mean) / stdev


def detect_price_anomalies(commodity: str) -> list[AnomalyEvent]:
    """Detect price spike anomalies via Z-score on 30-day pct_change window."""
    thresholds = ANOMALY_THRESHOLDS.get(commodity, {})
    price_zscore_threshold = thresholds.get("price_zscore", _DEFAULT_PRICE_ZSCORE)

    anomalies = []
    window_start = datetime.utcnow() - timedelta(days=PRICE_WINDOW_DAYS)

    with get_session() as session:
        # Get recent pct_change features for this commodity, ordered by time
        rows = session.execute(
            text("""
                SELECT pf.id, pf.value, pf.raw_ingestion_id, ri.timestamp, ri.data_type
                FROM processed_features pf
                JOIN raw_ingestion ri ON pf.raw_ingestion_id = ri.id
                WHERE ri.commodity = :commodity
                  AND pf.feature_type = 'pct_change'
                  AND ri.timestamp >= :window_start
                ORDER BY ri.timestamp ASC
            """),
            {"commodity": commodity, "window_start": window_start},
        ).fetchall()

        if len(rows) < MIN_WINDOW_POINTS:
            return []

        values = [r[1] for r in rows]

        # Slide over window: for each point, compute Z-score vs all preceding points.
        # SQL already limits to PRICE_WINDOW_DAYS so values[] is already time-bounded.
        for i, row in enumerate(rows):
            if i < MIN_WINDOW_POINTS:
                continue
            window = values[:i]
            z = _compute_zscore(row[1], window)
            if z is None or abs(z) < price_zscore_threshold:
                continue

            # Check if we already have this anomaly (same commodity + source + timestamp)
            existing = session.execute(
                text("""
                    SELECT id FROM anomaly_events
                    WHERE commodity = :commodity
                      AND anomaly_type = 'price_spike'
                      AND json_extract(metadata_json, '$.raw_ingestion_id') = :raw_id
                """),
                {"commodity": commodity, "raw_id": row[2]},
            ).fetchone()
            if existing:
                continue

            anomaly = AnomalyEvent(
                commodity=commodity,
                anomaly_type="price_spike",
                severity=abs(z),
                detected_at=datetime.utcnow(),
                source_ids=json.dumps([row[2]]),
                status="new",
                metadata_json=json.dumps({
                    "z_score": z,
                    "pct_change": row[1],
                    "data_type": row[4],
                    "raw_ingestion_id": row[2],
                    "data_timestamp": str(row[3]) if row[3] else None,
                }),
            )
            session.add(anomaly)
            anomalies.append(anomaly)

    return anomalies


def detect_sentiment_anomalies(commodity: str) -> list[AnomalyEvent]:
    """Detect sentiment spikes: current article compound score vs 30-day mean."""
    anomalies = []
    window_start = datetime.utcnow() - timedelta(days=SENTIMENT_MEAN_WINDOW_DAYS)

    with get_session() as session:
        rows = session.execute(
            text("""
                SELECT pf.id, pf.value, pf.raw_ingestion_id, ri.timestamp
                FROM processed_features pf
                JOIN raw_ingestion ri ON pf.raw_ingestion_id = ri.id
                WHERE ri.commodity = :commodity
                  AND pf.feature_type = 'sentiment_score'
                  AND ri.timestamp >= :window_start
                ORDER BY ri.timestamp ASC
            """),
            {"commodity": commodity, "window_start": window_start},
        ).fetchall()

        if len(rows) < MIN_WINDOW_POINTS:
            return []

        values = [r[1] for r in rows]

        for i, row in enumerate(rows):
            if i < MIN_WINDOW_POINTS:
                continue
            window = values[:i]
            z = _compute_zscore(row[1], window)
            if z is None or abs(z) < PRICE_ZSCORE_THRESHOLD:
                continue
            # Only flag strong sentiment — not neutral noise
            if abs(row[1]) < SENTIMENT_SPIKE_THRESHOLD:
                continue

            existing = session.execute(
                text("""
                    SELECT id FROM anomaly_events
                    WHERE commodity = :commodity
                      AND anomaly_type = 'sentiment_shift'
                      AND json_extract(metadata_json, '$.raw_ingestion_id') = :raw_id
                """),
                {"commodity": commodity, "raw_id": row[2]},
            ).fetchone()
            if existing:
                continue

            anomaly = AnomalyEvent(
                commodity=commodity,
                anomaly_type="sentiment_shift",
                severity=abs(z),
                detected_at=datetime.utcnow(),
                source_ids=json.dumps([row[2]]),
                status="new",
                metadata_json=json.dumps({
                    "z_score": z,
                    "compound_score": row[1],
                    "raw_ingestion_id": row[2],
                    "data_timestamp": str(row[3]) if row[3] else None,
                }),
            )
            session.add(anomaly)
            anomalies.append(anomaly)

    return anomalies


def run_anomaly_detection() -> dict:
    """Run all anomaly detectors for all commodities."""
    from shared.commodity_registry import COMMODITY_LIST

    total = 0
    by_commodity = {}

    for commodity in COMMODITY_LIST:
        price_anomalies = detect_price_anomalies(commodity)
        sentiment_anomalies = detect_sentiment_anomalies(commodity)
        count = len(price_anomalies) + len(sentiment_anomalies)
        total += count
        by_commodity[commodity] = count

    logger.info("anomaly_detection_complete", total_new=total, by_commodity=by_commodity)
    return {"total_new_anomalies": total, "by_commodity": by_commodity}
