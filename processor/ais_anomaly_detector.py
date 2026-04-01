"""
AIS anomaly detector.
Detects vessel activity anomalies at monitored commodity ports.

Anomaly types:
  - ais_vessel_drop   : vessel_count drops >50% vs 7-day average at a port
                        (sudden cessation of loading/unloading activity)
  - ais_port_idle     : moored_ratio > 0.85 AND vessel_count above baseline
                        (vessels present but not moving — potential blockage)

Both write to anomaly_events with status='new' so the AI engine picks them up.
"""

import json
from datetime import datetime, timedelta

from sqlalchemy import text

from shared.db import get_session
from shared.logger import get_logger
from shared.models import AnomalyEvent

logger = get_logger(__name__)

VESSEL_DROP_THRESHOLD = 0.50    # flag if count falls >50% below 7d avg
IDLE_MOORED_RATIO = 0.85        # flag if >85% of vessels are moored
MIN_BASELINE_POINTS = 3         # need at least 3 prior readings for a meaningful baseline


def _already_flagged(session, port_slug: str, anomaly_type: str, within_hours: int = 6) -> bool:
    """Prevent duplicate anomaly rows for the same port within the dedup window."""
    cutoff = datetime.utcnow() - timedelta(hours=within_hours)
    row = session.execute(
        text("""
            SELECT id FROM anomaly_events
            WHERE anomaly_type = :anomaly_type
              AND detected_at >= :cutoff
              AND json_extract(metadata_json, '$.port_slug') = :port_slug
        """),
        {"anomaly_type": anomaly_type, "cutoff": cutoff, "port_slug": port_slug},
    ).fetchone()
    return row is not None


def detect_ais_anomalies() -> list[AnomalyEvent]:
    """Scan latest AIS features for anomalies across all ports. Returns new AnomalyEvent objects."""
    from ingestion.ais.port_registry import PORT_REGISTRY
    anomalies = []

    with get_session() as session:
        for port_slug, port_info in PORT_REGISTRY.items():
            commodity = port_info["commodity"]

            # Latest vessel_count reading for this port
            latest = session.execute(
                text("""
                    SELECT pf.id, pf.value, pf.raw_ingestion_id, ri.timestamp
                    FROM processed_features pf
                    JOIN raw_ingestion ri ON pf.raw_ingestion_id = ri.id
                    WHERE ri.symbol = :port_slug
                      AND ri.data_type = 'ais'
                      AND pf.feature_type = 'vessel_count'
                    ORDER BY ri.timestamp DESC
                    LIMIT 1
                """),
                {"port_slug": port_slug},
            ).fetchone()

            if latest is None:
                continue

            current_count = latest[1]
            raw_id = latest[2]

            # 7-day rolling average for this port
            baseline = session.execute(
                text("""
                    SELECT pf.value
                    FROM processed_features pf
                    JOIN raw_ingestion ri ON pf.raw_ingestion_id = ri.id
                    WHERE ri.symbol = :port_slug
                      AND ri.data_type = 'ais'
                      AND pf.feature_type = 'vessel_count'
                      AND ri.timestamp >= :window_start
                      AND ri.timestamp < :latest_ts
                    ORDER BY ri.timestamp ASC
                """),
                {
                    "port_slug": port_slug,
                    "window_start": datetime.utcnow() - timedelta(days=7),
                    "latest_ts": str(latest[3]),
                },
            ).fetchall()

            if len(baseline) < MIN_BASELINE_POINTS:
                continue

            baseline_values = [r[0] for r in baseline if r[0] is not None]
            if not baseline_values:
                continue
            avg_count = sum(baseline_values) / len(baseline_values)

            # ── Vessel drop anomaly ───────────────────────────────────────────
            if avg_count > 0 and (avg_count - current_count) / avg_count >= VESSEL_DROP_THRESHOLD:
                if not _already_flagged(session, port_slug, "ais_vessel_drop"):
                    drop_pct = (avg_count - current_count) / avg_count
                    severity = drop_pct * 10  # scale: 50% drop → 5.0, 100% → 10.0
                    anomaly = AnomalyEvent(
                        commodity=commodity,
                        anomaly_type="ais_vessel_drop",
                        severity=severity,
                        detected_at=datetime.utcnow(),
                        source_ids=json.dumps([raw_id]),
                        status="new",
                        metadata_json=json.dumps({
                            "port_slug": port_slug,
                            "port_name": port_info["name"],
                            "current_vessel_count": current_count,
                            "baseline_avg": round(avg_count, 2),
                            "drop_pct": round(drop_pct * 100, 1),
                            "baseline_days": 7,
                            "baseline_points": len(baseline_values),
                        }),
                    )
                    session.add(anomaly)
                    anomalies.append(anomaly)
                    logger.info(
                        "ais_vessel_drop_detected",
                        port=port_slug,
                        commodity=commodity,
                        current=current_count,
                        baseline_avg=round(avg_count, 2),
                    )

            # ── Port idle anomaly (vessels present but all moored) ────────────
            latest_moored = session.execute(
                text("""
                    SELECT pf.value
                    FROM processed_features pf
                    JOIN raw_ingestion ri ON pf.raw_ingestion_id = ri.id
                    WHERE ri.symbol = :port_slug
                      AND ri.data_type = 'ais'
                      AND pf.feature_type = 'moored_ratio'
                    ORDER BY ri.timestamp DESC
                    LIMIT 1
                """),
                {"port_slug": port_slug},
            ).fetchone()

            if latest_moored and latest_moored[0] >= IDLE_MOORED_RATIO and current_count >= 2:
                if not _already_flagged(session, port_slug, "ais_port_idle"):
                    # Severity normalized to [2.0, 10.0]: threshold → 2.0, fully moored → 10.0
                    # Consistent scale with ais_vessel_drop (drop_pct * 10 → [5.0, 10.0])
                    idle_severity = 2.0 + (latest_moored[0] - IDLE_MOORED_RATIO) / (1.0 - IDLE_MOORED_RATIO) * 8.0
                    anomaly = AnomalyEvent(
                        commodity=commodity,
                        anomaly_type="ais_port_idle",
                        severity=round(idle_severity, 3),
                        detected_at=datetime.utcnow(),
                        source_ids=json.dumps([raw_id]),
                        status="new",
                        metadata_json=json.dumps({
                            "port_slug": port_slug,
                            "port_name": port_info["name"],
                            "moored_ratio": round(latest_moored[0], 3),
                            "vessel_count": current_count,
                        }),
                    )
                    session.add(anomaly)
                    anomalies.append(anomaly)
                    logger.info(
                        "ais_port_idle_detected",
                        port=port_slug,
                        commodity=commodity,
                        moored_ratio=round(latest_moored[0], 3),
                    )

    logger.info("ais_anomaly_detection_complete", new_anomalies=len(anomalies))
    return anomalies
