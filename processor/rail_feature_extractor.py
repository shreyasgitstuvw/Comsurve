"""
Rail feature extractor.

Processes unprocessed rail raw_ingestion rows (source='osm_rail') and writes
two features per corridor to processed_features:

  feature_type='rail_way_count'     value=number of rail segments in corridor
  feature_type='rail_length_km'     value=total rail length in kilometres

These baseline values serve two purposes:
  1. Week-over-week length changes signal corridor construction or removal
     (e.g. a temporary bridge closure shortening total trackage)
  2. Gemini causality prompts are enriched with corridor context — when a
     news article mentions a rail disruption, the system can calculate what
     fraction of the corridor is affected
"""

import json

from sqlalchemy import text

from shared.db import get_session
from shared.logger import get_logger

logger = get_logger(__name__)


def run_rail_feature_extraction() -> dict:
    processed = 0
    features_written = 0

    with get_session() as session:
        rows = session.execute(text("""
            SELECT id, commodity, raw_json
            FROM raw_ingestion
            WHERE source = 'osm_rail'
              AND processed = 0
        """)).fetchall()

        for row_id, commodity, raw_json_str in rows:
            try:
                raw = json.loads(raw_json_str)
            except Exception:
                raw = {}

            way_count = raw.get("way_count", 0)
            total_km = raw.get("total_length_km", 0.0)
            corridor_slug = raw.get("corridor_slug", "")

            for ftype, value in [
                ("rail_way_count", float(way_count)),
                ("rail_length_km", float(total_km)),
            ]:
                session.execute(text("""
                    INSERT INTO processed_features
                        (raw_ingestion_id, commodity, feature_type,
                         value, value_json, window, computed_at)
                    VALUES
                        (:rid, :commodity, :ftype,
                         :value, :vj, 'static', CURRENT_TIMESTAMP)
                """), {
                    "rid": row_id,
                    "commodity": commodity,
                    "ftype": ftype,
                    "value": value,
                    "vj": json.dumps({"corridor_slug": corridor_slug}),
                })
                features_written += 1

            session.execute(text("""
                UPDATE raw_ingestion SET processed = 1 WHERE id = :id
            """), {"id": row_id})
            processed += 1

    result = {"processed_rows": processed, "features_written": features_written}
    logger.info("rail_feature_extraction_complete", **result)
    return result
