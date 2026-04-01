"""
Satellite anomaly detector.

Reads from processed_features (feature_type IN satellite_s1_scene,
satellite_s2_cloud_cover, aircraft_count) and detects two anomaly types:

  satellite_scene_gap
    Triggered when a port has ZERO Sentinel-1 passes in the past 7 days.
    Sentinel-1 revisit is ~6 days (A+B combined), so a 7-day gap is unusual
    and may indicate acquisition planning changes or system issues.
    Severity = 3.0 (fixed — no rolling baseline yet).

  satellite_cloud_block
    Triggered when the mean Sentinel-2 cloud cover at an agricultural port
    (soybeans) exceeds 85% over the past 14 days.  Persistent cloud cover
    is a proxy for flooding / adverse weather affecting crops.
    Severity = (mean_cloud_cover - 85) / 5  (scales with how far above threshold).

  satellite_aircraft_surge
    Triggered when the aircraft count over a port in the latest snapshot
    exceeds 2× the 7-day rolling mean for that port.
    Severity = observed / mean (ratio).

Anomalies are written to anomaly_events with status='new' so the AI engine
picks them up for embedding and signal correlation.
"""

import json
from datetime import datetime, timedelta

from sqlalchemy import text

from shared.db import get_session
from shared.logger import get_logger

logger = get_logger(__name__)

S1_SCENE_GAP_SEVERITY = 3.0          # fixed severity for a complete scene gap
CLOUD_BLOCK_THRESHOLD_PCT = 85.0      # mean cloud cover threshold (soybeans only)
AIRCRAFT_SURGE_RATIO = 2.0            # ratio vs rolling mean to flag surge


def _get_port_slugs_for_commodity(commodity: str) -> list[str]:
    from ingestion.ais.port_registry import PORT_REGISTRY
    return [
        slug for slug, info in PORT_REGISTRY.items()
        if info["commodity"] == commodity
    ]


def detect_satellite_anomalies() -> list[dict]:
    anomalies: list[dict] = []
    now = datetime.utcnow()

    with get_session() as session:

        # ── 1. Sentinel-1 Scene Gap ───────────────────────────────────────────
        since_7d = now - timedelta(days=7)

        # Get all port slugs that had ANY S1 scene in the past 7 days
        active_ports_rows = session.execute(text("""
            SELECT DISTINCT json_extract(pf.value_json, '$.port_slug')
            FROM processed_features pf
            JOIN raw_ingestion ri ON pf.raw_ingestion_id = ri.id
            WHERE pf.feature_type = 'satellite_s1_scene'
              AND ri.timestamp >= :since
        """), {"since": since_7d}).fetchall()
        active_ports = {r[0] for r in active_ports_rows if r[0]}

        # Get all ports that EVER had an S1 scene (i.e., we've seen them before)
        all_seen_ports_rows = session.execute(text("""
            SELECT DISTINCT json_extract(pf.value_json, '$.port_slug')
            FROM processed_features pf
            WHERE pf.feature_type = 'satellite_s1_scene'
        """)).fetchall()
        all_seen_ports = {r[0] for r in all_seen_ports_rows if r[0]}

        # Ports with a known history but no scene in the past 7 days
        gap_ports = all_seen_ports - active_ports
        for port_slug in gap_ports:
            # Avoid duplicate anomaly within 24h
            existing = session.execute(text("""
                SELECT id FROM anomaly_events
                WHERE anomaly_type = 'satellite_scene_gap'
                  AND metadata_json LIKE :slug_pattern
                  AND detected_at >= :since_24h
            """), {
                "slug_pattern": f'%"{port_slug}"%',
                "since_24h": now - timedelta(hours=24),
            }).fetchone()
            if existing:
                continue

            # Find most recent source_ids for this port from S1 features
            src_rows = session.execute(text("""
                SELECT ri.id, ri.commodity
                FROM processed_features pf
                JOIN raw_ingestion ri ON pf.raw_ingestion_id = ri.id
                WHERE pf.feature_type = 'satellite_s1_scene'
                  AND json_extract(pf.value_json, '$.port_slug') = :slug
                ORDER BY ri.timestamp DESC
                LIMIT 5
            """), {"slug": port_slug}).fetchall()

            if not src_rows:
                continue

            commodity = src_rows[0][1]
            source_ids = [r[0] for r in src_rows]

            session.execute(text("""
                INSERT INTO anomaly_events
                    (commodity, anomaly_type, severity, detected_at,
                     source_ids, status, metadata_json)
                VALUES
                    (:commodity, 'satellite_scene_gap', :severity,
                     CURRENT_TIMESTAMP, :source_ids, 'new', :meta)
            """), {
                "commodity": commodity,
                "severity": S1_SCENE_GAP_SEVERITY,
                "source_ids": json.dumps(source_ids),
                "meta": json.dumps({
                    "port_slug": port_slug,
                    "window_days": 7,
                    "description": f"No Sentinel-1 SAR pass over {port_slug} in 7 days",
                }),
            })

            anomalies.append({"type": "satellite_scene_gap", "port": port_slug,
                               "commodity": commodity, "severity": S1_SCENE_GAP_SEVERITY})
            logger.info("satellite_scene_gap_detected", port=port_slug, commodity=commodity)

        # ── 2. Sentinel-2 Cloud Block (soybeans agricultural ports) ───────────
        since_14d = now - timedelta(days=14)
        soy_ports = _get_port_slugs_for_commodity("soybeans")

        for port_slug in soy_ports:
            cloud_rows = session.execute(text("""
                SELECT pf.value
                FROM processed_features pf
                JOIN raw_ingestion ri ON pf.raw_ingestion_id = ri.id
                WHERE pf.feature_type = 'satellite_s2_cloud_cover'
                  AND json_extract(pf.value_json, '$.port_slug') = :slug
                  AND ri.timestamp >= :since
            """), {"slug": port_slug, "since": since_14d}).fetchall()

            if len(cloud_rows) < 3:  # need at least 3 scenes to make a judgement
                continue

            values = [r[0] for r in cloud_rows if r[0] is not None]
            mean_cloud = sum(values) / len(values)

            if mean_cloud <= CLOUD_BLOCK_THRESHOLD_PCT:
                continue

            # Deduplicate within 24h
            existing = session.execute(text("""
                SELECT id FROM anomaly_events
                WHERE anomaly_type = 'satellite_cloud_block'
                  AND metadata_json LIKE :slug_pattern
                  AND detected_at >= :since_24h
            """), {
                "slug_pattern": f'%"{port_slug}"%',
                "since_24h": now - timedelta(hours=24),
            }).fetchone()
            if existing:
                continue

            src_rows = session.execute(text("""
                SELECT ri.id
                FROM processed_features pf
                JOIN raw_ingestion ri ON pf.raw_ingestion_id = ri.id
                WHERE pf.feature_type = 'satellite_s2_cloud_cover'
                  AND json_extract(pf.value_json, '$.port_slug') = :slug
                  AND ri.timestamp >= :since
                ORDER BY ri.timestamp DESC
                LIMIT 10
            """), {"slug": port_slug, "since": since_14d}).fetchall()

            source_ids = [r[0] for r in src_rows]
            severity = round((mean_cloud - CLOUD_BLOCK_THRESHOLD_PCT) / 5.0, 2)

            session.execute(text("""
                INSERT INTO anomaly_events
                    (commodity, anomaly_type, severity, detected_at,
                     source_ids, status, metadata_json)
                VALUES
                    ('soybeans', 'satellite_cloud_block', :severity,
                     CURRENT_TIMESTAMP, :source_ids, 'new', :meta)
            """), {
                "severity": severity,
                "source_ids": json.dumps(source_ids),
                "meta": json.dumps({
                    "port_slug": port_slug,
                    "mean_cloud_cover_pct": round(mean_cloud, 1),
                    "threshold_pct": CLOUD_BLOCK_THRESHOLD_PCT,
                    "window_days": 14,
                    "scene_count": len(values),
                    "description": (
                        f"Persistent cloud cover {mean_cloud:.0f}% at {port_slug} "
                        f"over 14 days — potential weather disruption"
                    ),
                }),
            })

            anomalies.append({"type": "satellite_cloud_block", "port": port_slug,
                               "mean_cloud": mean_cloud, "severity": severity})
            logger.info("satellite_cloud_block_detected",
                        port=port_slug, mean_cloud=mean_cloud)

        # ── 3. Aircraft Count Surge ───────────────────────────────────────────
        since_7d_aircraft = now - timedelta(days=7)

        # For each port, compare latest aircraft count to 7-day rolling mean
        port_latest = session.execute(text("""
            SELECT json_extract(pf.value_json, '$.port_slug'),
                   ri.commodity,
                   pf.value,
                   ri.id
            FROM processed_features pf
            JOIN raw_ingestion ri ON pf.raw_ingestion_id = ri.id
            WHERE pf.feature_type = 'aircraft_count'
              AND ri.id IN (
                  SELECT MAX(ri2.id)
                  FROM processed_features pf2
                  JOIN raw_ingestion ri2 ON pf2.raw_ingestion_id = ri2.id
                  WHERE pf2.feature_type = 'aircraft_count'
                  GROUP BY json_extract(pf2.value_json, '$.port_slug')
              )
        """)).fetchall()

        for port_slug, commodity, latest_count, latest_rid in port_latest:
            if latest_count is None or latest_count < 3:
                continue  # ignore tiny counts (0-2 aircraft is normal)

            # Rolling mean over past 7 days (excluding the current observation)
            mean_row = session.execute(text("""
                SELECT AVG(pf.value), COUNT(*)
                FROM processed_features pf
                JOIN raw_ingestion ri ON pf.raw_ingestion_id = ri.id
                WHERE pf.feature_type = 'aircraft_count'
                  AND json_extract(pf.value_json, '$.port_slug') = :slug
                  AND ri.timestamp >= :since
                  AND ri.id != :latest_rid
            """), {"slug": port_slug, "since": since_7d_aircraft,
                   "latest_rid": latest_rid}).fetchone()

            if not mean_row or mean_row[1] < 3 or not mean_row[0]:
                continue  # need at least 3 historical observations

            rolling_mean = float(mean_row[0])
            if rolling_mean < 1.0:
                continue  # avoid division issues / nonsensical ratios

            ratio = latest_count / rolling_mean
            if ratio < AIRCRAFT_SURGE_RATIO:
                continue

            existing = session.execute(text("""
                SELECT id FROM anomaly_events
                WHERE anomaly_type = 'satellite_aircraft_surge'
                  AND metadata_json LIKE :slug_pattern
                  AND detected_at >= :since_24h
            """), {
                "slug_pattern": f'%"{port_slug}"%',
                "since_24h": now - timedelta(hours=24),
            }).fetchone()
            if existing:
                continue

            severity = round(ratio, 2)
            session.execute(text("""
                INSERT INTO anomaly_events
                    (commodity, anomaly_type, severity, detected_at,
                     source_ids, status, metadata_json)
                VALUES
                    (:commodity, 'satellite_aircraft_surge', :severity,
                     CURRENT_TIMESTAMP, :source_ids, 'new', :meta)
            """), {
                "commodity": commodity,
                "severity": severity,
                "source_ids": json.dumps([latest_rid]),
                "meta": json.dumps({
                    "port_slug": port_slug,
                    "latest_aircraft_count": latest_count,
                    "rolling_mean_7d": round(rolling_mean, 1),
                    "surge_ratio": severity,
                    "description": (
                        f"Aircraft count {latest_count:.0f} at {port_slug} is "
                        f"{ratio:.1f}x the 7-day average ({rolling_mean:.1f})"
                    ),
                }),
            })

            anomalies.append({"type": "satellite_aircraft_surge", "port": port_slug,
                               "commodity": commodity, "ratio": ratio})
            logger.info("satellite_aircraft_surge_detected",
                        port=port_slug, count=latest_count, ratio=ratio)

    logger.info("satellite_anomaly_detection_complete",
                new_anomalies=len(anomalies))
    return anomalies
