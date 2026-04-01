"""
Rail anomaly detector.

Reads from processed_features (feature_type IN rail_length_km, rail_way_count)
and detects two anomaly types:

  rail_length_deviation
    Triggered when a corridor's total_length_km changes by more than
    LENGTH_DEVIATION_PCT between the two most recent weekly snapshots.
    A significant drop usually indicates a bridge closure or segment
    removal; a spike may reflect new OSM data or corridor extension.
    Severity = abs(pct_change) / 5  (scales with magnitude).

  rail_corridor_gap
    Triggered when a corridor that has historical snapshots shows no new
    data in the past GAP_THRESHOLD_DAYS days.  Indicates either an
    Overpass API outage or a genuine data gap that warrants investigation.
    Severity = 1.5 (fixed — magnitude cannot be assessed without new data).

Anomalies are written to anomaly_events with status='new' so the AI engine
picks them up for embedding and signal correlation.
"""

import json
from datetime import datetime, timedelta

from sqlalchemy import text

from shared.db import get_session
from shared.logger import get_logger

logger = get_logger(__name__)

LENGTH_DEVIATION_PCT = 5.0   # minimum % change in corridor length to flag
GAP_THRESHOLD_DAYS = 8       # days since last snapshot to flag as a gap
GAP_SEVERITY = 1.5           # fixed severity for data gaps


def detect_rail_anomalies() -> list[dict]:
    anomalies: list[dict] = []
    now = datetime.utcnow()

    with get_session() as session:

        # All corridors that have at least one processed rail_length_km feature
        corridors = session.execute(text("""
            SELECT DISTINCT
                json_extract(pf.value_json, '$.corridor_slug') AS corridor_slug,
                ri.commodity
            FROM processed_features pf
            JOIN raw_ingestion ri ON pf.raw_ingestion_id = ri.id
            WHERE pf.feature_type = 'rail_length_km'
        """)).fetchall()

        for corridor_slug, commodity in corridors:
            if not corridor_slug:
                continue

            # Two most recent snapshots for this corridor
            snapshots = session.execute(text("""
                SELECT pf.value, ri.timestamp, ri.id
                FROM processed_features pf
                JOIN raw_ingestion ri ON pf.raw_ingestion_id = ri.id
                WHERE pf.feature_type = 'rail_length_km'
                  AND json_extract(pf.value_json, '$.corridor_slug') = :slug
                ORDER BY ri.timestamp DESC
                LIMIT 2
            """), {"slug": corridor_slug}).fetchall()

            if not snapshots:
                continue

            latest_value, latest_ts_raw, latest_rid = snapshots[0]

            # Parse timestamp (SQLite may return string)
            if isinstance(latest_ts_raw, str):
                try:
                    latest_ts = datetime.fromisoformat(latest_ts_raw)
                except ValueError:
                    latest_ts = datetime.strptime(latest_ts_raw[:19], "%Y-%m-%d %H:%M:%S")
            else:
                latest_ts = latest_ts_raw

            # ── 1. Rail Corridor Gap ──────────────────────────────────────────
            days_since = (now - latest_ts).days
            if days_since > GAP_THRESHOLD_DAYS:
                existing = session.execute(text("""
                    SELECT id FROM anomaly_events
                    WHERE anomaly_type = 'rail_corridor_gap'
                      AND metadata_json LIKE :slug_pattern
                      AND detected_at >= :since_24h
                """), {
                    "slug_pattern": f'%"{corridor_slug}"%',
                    "since_24h": now - timedelta(hours=24),
                }).fetchone()

                if not existing:
                    session.execute(text("""
                        INSERT INTO anomaly_events
                            (commodity, anomaly_type, severity, detected_at,
                             source_ids, status, metadata_json)
                        VALUES
                            (:commodity, 'rail_corridor_gap', :severity,
                             CURRENT_TIMESTAMP, :source_ids, 'new', :meta)
                    """), {
                        "commodity": commodity,
                        "severity": GAP_SEVERITY,
                        "source_ids": json.dumps([latest_rid]),
                        "meta": json.dumps({
                            "corridor_slug": corridor_slug,
                            "days_since_last_snapshot": days_since,
                            "last_snapshot_ts": latest_ts.isoformat(),
                            "description": (
                                f"No rail data for {corridor_slug} in {days_since} days "
                                f"(threshold: {GAP_THRESHOLD_DAYS} days)"
                            ),
                        }),
                    })
                    anomalies.append({
                        "type": "rail_corridor_gap",
                        "corridor": corridor_slug,
                        "commodity": commodity,
                        "days_since": days_since,
                    })
                    logger.info("rail_corridor_gap_detected",
                                corridor=corridor_slug, days_since=days_since,
                                commodity=commodity)

            # ── 2. Rail Length Deviation ──────────────────────────────────────
            if len(snapshots) < 2:
                continue

            prev_value, _prev_ts, prev_rid = snapshots[1]
            if not latest_value or not prev_value or prev_value == 0:
                continue

            pct_change = (latest_value - prev_value) / prev_value * 100
            if abs(pct_change) < LENGTH_DEVIATION_PCT:
                continue

            existing = session.execute(text("""
                SELECT id FROM anomaly_events
                WHERE anomaly_type = 'rail_length_deviation'
                  AND metadata_json LIKE :slug_pattern
                  AND detected_at >= :since_24h
            """), {
                "slug_pattern": f'%"{corridor_slug}"%',
                "since_24h": now - timedelta(hours=24),
            }).fetchone()

            if existing:
                continue

            severity = round(abs(pct_change) / 5.0, 2)
            session.execute(text("""
                INSERT INTO anomaly_events
                    (commodity, anomaly_type, severity, detected_at,
                     source_ids, status, metadata_json)
                VALUES
                    (:commodity, 'rail_length_deviation', :severity,
                     CURRENT_TIMESTAMP, :source_ids, 'new', :meta)
            """), {
                "commodity": commodity,
                "severity": severity,
                "source_ids": json.dumps([latest_rid, prev_rid]),
                "meta": json.dumps({
                    "corridor_slug": corridor_slug,
                    "pct_change": round(pct_change, 2),
                    "latest_length_km": round(latest_value, 2),
                    "previous_length_km": round(prev_value, 2),
                    "description": (
                        f"{corridor_slug} length changed {pct_change:+.1f}% "
                        f"({prev_value:.0f} km → {latest_value:.0f} km)"
                    ),
                }),
            })

            anomalies.append({
                "type": "rail_length_deviation",
                "corridor": corridor_slug,
                "commodity": commodity,
                "pct_change": round(pct_change, 2),
                "severity": severity,
            })
            logger.info("rail_length_deviation_detected",
                        corridor=corridor_slug, pct_change=round(pct_change, 2),
                        latest_km=round(latest_value, 2), prev_km=round(prev_value, 2),
                        commodity=commodity)

    logger.info("rail_anomaly_detection_complete", new_anomalies=len(anomalies))
    return anomalies
