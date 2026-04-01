"""
APScheduler job definitions for all MCEI services.

All jobs are wrapped in a try/except that logs to job_history so a single
job failure never crashes the scheduler process.

Schedule map (from commodity_registry.SCHEDULE_MAP):
  news              : every 6h
  price_realtime    : every 1h   (yfinance primary)
  price_historical  : daily 02:00
  ais               : every 30min
  processor         : every 30min, starts at :30 (offset from AIS)
  ai_engine         : daily 03:00 (embed + correlate)
  causality         : daily 04:00
  prediction        : daily 04:30 (after causality)
  evaluation        : daily 05:00 (after prediction)
"""

from scheduler.alerting import record_job_failure, record_job_success
from scheduler.job_history import record_end, record_error, record_start
from shared.logger import get_logger

logger = get_logger(__name__)


def _run_job(job_name: str, fn, *args, **kwargs):
    """Wrapper: records run in job_history, fires alerts, catches all exceptions."""
    run_id = record_start(job_name)
    try:
        result = fn(*args, **kwargs)
        record_end(run_id, summary=result if isinstance(result, dict) else None)
        record_job_success(job_name)
        logger.info("job_complete", job=job_name, result=result)
        return result
    except Exception as exc:
        error_str = str(exc)
        record_error(run_id, error_str)
        record_job_failure(job_name, error_str)
        logger.error("job_failed", job=job_name, error=error_str)


# ── Individual job functions (called by APScheduler) ─────────────────────────

def job_news():
    from ingestion.news.news_runner import run
    _run_job("news", run)


def job_price_realtime():
    from ingestion.price_realtime.price_realtime_runner import run
    _run_job("price_realtime", run)


def job_price_historical():
    from ingestion.price_historical.fred_ingestor import FredIngestor
    from ingestion.price_historical.eia_ingestor import EIAIngestor
    _run_job("fred", FredIngestor(lookback_days=7).run)
    _run_job("eia", EIAIngestor(lookback_days=7).run)


def job_ais():
    from ingestion.ais.aisstream_ingestor import AISStreamIngestor
    _run_job("ais", AISStreamIngestor(collection_seconds=60).run)


def job_processor():
    from processor.processor_runner import run
    _run_job("processor", run)


def job_ai_engine():
    from ai_engine.ai_engine_runner import run_full_ai_batch
    _run_job("ai_engine", run_full_ai_batch)


def job_causality():
    from ai_engine.causality_engine import run_causality_engine
    _run_job("causality", run_causality_engine)


def job_prediction():
    from ai_engine.prediction_engine import run_prediction_engine
    _run_job("prediction", run_prediction_engine)


def job_evaluation():
    from ai_engine.evaluation_engine import run_evaluation_engine
    _run_job("evaluation", run_evaluation_engine)


def job_satellite():
    from ingestion.satellite.satellite_runner import run
    _run_job("satellite", run)


def job_aircraft():
    from ingestion.aircraft.opensky_ingestor import OpenSkyIngestor
    _run_job("aircraft", OpenSkyIngestor().run)


def job_rail():
    from ingestion.rail.osm_overpass_ingestor import OSMRailIngestor
    _run_job("rail", OSMRailIngestor().run)


def job_cleanup():
    from scheduler.cleanup import run_cleanup
    return _run_job("cleanup", run_cleanup)


def job_qdrant_backup():
    from pathlib import Path
    from scripts.backup_qdrant import backup_qdrant_local, prune_old_backups
    from shared.config import settings

    backup_dir = Path("./qdrant_backups")
    qdrant_path = Path(settings.qdrant_path)
    created = backup_qdrant_local(backup_dir, qdrant_path)
    pruned = prune_old_backups(backup_dir, retention_days=7)
    return {"archives_created": len(created), "archives_pruned": pruned,
            "status": "ok" if created else "no_data"}
