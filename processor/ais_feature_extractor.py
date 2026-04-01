"""
AIS feature extractor.
Reads unprocessed raw_ingestion rows with data_type='ais' and writes
vessel activity features to processed_features.

Features computed per port per collection window:
  - vessel_count       : unique vessels observed in the zone
  - avg_sog_knots      : average speed over ground (proxy for activity level)
  - moored_ratio       : fraction of vessels that are moored/anchored (idle indicator)
  - vessel_count_7d_avg: rolling 7-day mean vessel count for the same port (for anomaly baseline)
"""

import json
from datetime import datetime, timedelta

from sqlalchemy import text

from shared.db import get_session
from shared.logger import get_logger
from shared.models import ProcessedFeature, RawIngestion

logger = get_logger(__name__)


def _get_rolling_mean(session, port_slug: str, before_ts: datetime, days: int = 7) -> float | None:
    """Compute mean vessel_count over the last `days` days for this port."""
    window_start = before_ts - timedelta(days=days)
    rows = session.execute(
        text("""
            SELECT pf.value
            FROM processed_features pf
            JOIN raw_ingestion ri ON pf.raw_ingestion_id = ri.id
            WHERE ri.symbol = :port_slug
              AND ri.data_type = 'ais'
              AND pf.feature_type = 'vessel_count'
              AND ri.timestamp >= :window_start
              AND ri.timestamp < :ts
            ORDER BY ri.timestamp ASC
        """),
        {"port_slug": port_slug, "window_start": window_start, "ts": before_ts},
    ).fetchall()

    if not rows:
        return None
    values = [r[0] for r in rows if r[0] is not None]
    return sum(values) / len(values) if values else None


def run_ais_feature_extraction() -> dict:
    """Process all unprocessed AIS raw_ingestion rows."""
    processed_count = 0
    feature_count = 0

    with get_session() as session:
        unprocessed = (
            session.query(RawIngestion)
            .filter(RawIngestion.processed == False)
            .filter(RawIngestion.data_type == "ais")
            .order_by(RawIngestion.timestamp)
            .all()
        )

        for row in unprocessed:
            try:
                data = json.loads(row.raw_json)
            except json.JSONDecodeError:
                row.processed = True
                continue

            port_slug = data.get("port_slug", row.symbol)
            vessel_count = data.get("vessel_count", 0)
            avg_sog = data.get("avg_sog_knots", 0.0)
            moored_count = data.get("moored_count", 0)
            moored_ratio = moored_count / vessel_count if vessel_count > 0 else 0.0

            rolling_mean = _get_rolling_mean(session, port_slug, row.timestamp, days=7)

            features = [
                ProcessedFeature(
                    raw_ingestion_id=row.id,
                    commodity=row.commodity,
                    feature_type="vessel_count",
                    value=float(vessel_count),
                    value_json=json.dumps({"port": port_slug}),
                    window="30min",
                    computed_at=datetime.utcnow(),
                ),
                ProcessedFeature(
                    raw_ingestion_id=row.id,
                    commodity=row.commodity,
                    feature_type="avg_sog_knots",
                    value=avg_sog,
                    value_json=json.dumps({"port": port_slug}),
                    window="30min",
                    computed_at=datetime.utcnow(),
                ),
                ProcessedFeature(
                    raw_ingestion_id=row.id,
                    commodity=row.commodity,
                    feature_type="moored_ratio",
                    value=moored_ratio,
                    value_json=json.dumps({"port": port_slug, "moored": moored_count, "total": vessel_count}),
                    window="30min",
                    computed_at=datetime.utcnow(),
                ),
            ]

            if rolling_mean is not None:
                features.append(ProcessedFeature(
                    raw_ingestion_id=row.id,
                    commodity=row.commodity,
                    feature_type="vessel_count_7d_avg",
                    value=rolling_mean,
                    value_json=json.dumps({"port": port_slug}),
                    window="7d",
                    computed_at=datetime.utcnow(),
                ))

            for f in features:
                session.add(f)
            row.processed = True
            processed_count += 1
            feature_count += len(features)

    logger.info("ais_feature_extraction_complete", processed_rows=processed_count, features_written=feature_count)
    return {"processed_rows": processed_count, "features_written": feature_count}
