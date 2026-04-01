"""
Satellite feature extractor.

Processes unprocessed satellite raw_ingestion rows (source IN sentinel1,
sentinel2, opensky) and writes per-scene features to processed_features:

  Sentinel-1 rows  →  feature_type='satellite_s1_scene'   value=1.0
                       value_json = {port_slug, orbit_direction, scene_name}

  Sentinel-2 rows  →  feature_type='satellite_s2_cloud_cover'  value=cloud_cover_pct
                       value_json = {port_slug, scene_name, cloud_cover_pct}

  OpenSky rows     →  feature_type='aircraft_count'   value=aircraft_count
                       value_json = {port_slug, states_sample}

The satellite_anomaly_detector later aggregates these per-port per-period to
look for scene gaps and persistent cloud blocks.
"""

import json

from sqlalchemy import text

from shared.db import get_session
from shared.logger import get_logger

logger = get_logger(__name__)

SATELLITE_SOURCES = {"sentinel1", "sentinel2", "opensky"}


def run_satellite_feature_extraction() -> dict:
    processed = 0
    features_written = 0

    with get_session() as session:
        rows = session.execute(text("""
            SELECT id, source, commodity, raw_json
            FROM raw_ingestion
            WHERE data_type IN ('satellite', 'aircraft')
              AND source IN ('sentinel1', 'sentinel2', 'opensky')
              AND processed = 0
        """)).fetchall()

        for row_id, source, commodity, raw_json_str in rows:
            try:
                raw = json.loads(raw_json_str)
            except Exception:
                raw = {}

            feature_rows: list[dict] = []

            if source == "sentinel1":
                feature_rows.append({
                    "feature_type": "satellite_s1_scene",
                    "value": 1.0,
                    "value_json": json.dumps({
                        "port_slug": raw.get("port_slug", ""),
                        "orbit_direction": raw.get("orbit_direction", ""),
                        "scene_name": raw.get("scene_name", ""),
                    }),
                })

            elif source == "sentinel2":
                cloud_cover = raw.get("cloud_cover_pct")
                if cloud_cover is None:
                    cloud_cover = 50.0  # conservative default if metadata missing
                feature_rows.append({
                    "feature_type": "satellite_s2_cloud_cover",
                    "value": float(cloud_cover),
                    "value_json": json.dumps({
                        "port_slug": raw.get("port_slug", ""),
                        "scene_name": raw.get("scene_name", ""),
                        "cloud_cover_pct": cloud_cover,
                    }),
                })

            elif source == "opensky":
                count = raw.get("aircraft_count", 0)
                feature_rows.append({
                    "feature_type": "aircraft_count",
                    "value": float(count),
                    "value_json": json.dumps({
                        "port_slug": raw.get("port_slug", ""),
                        "states_sample": raw.get("states_sample", []),
                    }),
                })

            for feat in feature_rows:
                session.execute(text("""
                    INSERT INTO processed_features
                        (raw_ingestion_id, commodity, feature_type,
                         value, value_json, window, computed_at)
                    VALUES
                        (:rid, :commodity, :ftype,
                         :value, :vj, '7d', CURRENT_TIMESTAMP)
                """), {
                    "rid": row_id,
                    "commodity": commodity,
                    "ftype": feat["feature_type"],
                    "value": feat["value"],
                    "vj": feat["value_json"],
                })
                features_written += 1

            # Mark raw row as processed
            session.execute(text("""
                UPDATE raw_ingestion SET processed = 1 WHERE id = :id
            """), {"id": row_id})
            processed += 1

    result = {
        "processed_rows": processed,
        "features_written": features_written,
    }
    logger.info("satellite_feature_extraction_complete", **result)
    return result
