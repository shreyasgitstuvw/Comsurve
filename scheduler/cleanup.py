"""
Nightly cleanup job — keeps the SQLite DB lean.

What it does:
  1. Deletes raw_ingestion rows older than RETAIN_DAYS that have already been
     processed (i.e., at least one processed_features row exists for them).
     Unprocessed rows are never deleted so no data is lost.
  2. Deletes orphaned processed_features rows whose raw_ingestion parent was
     already removed by a prior cleanup run.
  3. Logs how many rows were removed.

Why: raw_ingestion accumulates quickly (news every 6h, prices hourly, AIS
every 30min) and is only needed for feature extraction. Once features are
written the raw payloads serve no purpose and inflate the DB unnecessarily.
"""

from datetime import datetime, timedelta

from sqlalchemy import text

from shared.db import get_session
from shared.logger import get_logger

logger = get_logger(__name__)

RETAIN_DAYS: int = 90   # keep raw rows for 90 days even after processing


def run_cleanup() -> dict:
    """
    Delete processed raw_ingestion rows older than RETAIN_DAYS.
    Returns a summary dict for job_history logging.
    """
    cutoff = datetime.utcnow() - timedelta(days=RETAIN_DAYS)

    with get_session() as session:
        # Count candidates before deletion
        candidate_count = session.execute(
            text("""
                SELECT COUNT(*)
                FROM raw_ingestion ri
                WHERE ri.timestamp < :cutoff
                  AND EXISTS (
                      SELECT 1 FROM processed_features pf
                      WHERE pf.raw_ingestion_id = ri.id
                  )
            """),
            {"cutoff": cutoff},
        ).scalar()

        # Delete processed raw rows beyond retention window
        deleted_raw = session.execute(
            text("""
                DELETE FROM raw_ingestion
                WHERE timestamp < :cutoff
                  AND id IN (
                      SELECT DISTINCT raw_ingestion_id
                      FROM processed_features
                  )
            """),
            {"cutoff": cutoff},
        ).rowcount

        # Delete orphaned processed_features (parent raw_ingestion gone)
        deleted_features = session.execute(
            text("""
                DELETE FROM processed_features
                WHERE raw_ingestion_id NOT IN (
                    SELECT id FROM raw_ingestion
                )
            """),
        ).rowcount

    summary = {
        "candidates": candidate_count,
        "deleted_raw_ingestion": deleted_raw,
        "deleted_processed_features": deleted_features,
        "cutoff_date": str(cutoff)[:10],
        "retain_days": RETAIN_DAYS,
    }
    logger.info("cleanup_complete", **summary)
    return summary
