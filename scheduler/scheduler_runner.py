"""
MCEI Scheduler — main entry point for autonomous operation.

Run with:
    cd mcei/
    python -m scheduler.scheduler_runner

Ctrl+C to stop gracefully.

All jobs are registered as BackgroundScheduler jobs (non-blocking).
The main thread blocks on scheduler.start() until KeyboardInterrupt.

Processor is deliberately offset +30 minutes from the top of the hour
so it runs after AIS data has been ingested (AIS fires on the hour).
"""

import signal
import sys
import time

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from scheduler.jobs import (
    job_ai_engine,
    job_aircraft,
    job_ais,
    job_causality,
    job_cleanup,
    job_evaluation,
    job_news,
    job_prediction,
    job_price_historical,
    job_price_realtime,
    job_processor,
    job_qdrant_backup,
    job_rail,
    job_satellite,
)
from shared.config import validate_secrets
from shared.db import init_db
from shared.logger import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


def build_scheduler() -> BackgroundScheduler:
    scheduler = BackgroundScheduler(
        job_defaults={
            "coalesce": True,       # if missed, run once (not multiple catch-up runs)
            "max_instances": 1,     # never overlap same job
            "misfire_grace_time": 300,  # 5 min grace window for missed fires
        }
    )

    # ── News: every 6 hours ───────────────────────────────────────────────────
    scheduler.add_job(
        job_news,
        trigger=IntervalTrigger(hours=6),
        id="news",
        name="News Ingestion (newsdata.io)",
    )

    # ── Price real-time: every 1 hour ─────────────────────────────────────────
    scheduler.add_job(
        job_price_realtime,
        trigger=IntervalTrigger(hours=1),
        id="price_realtime",
        name="Price Real-Time (yfinance / Commodities-API)",
    )

    # ── Price historical: daily at 02:00 ──────────────────────────────────────
    scheduler.add_job(
        job_price_historical,
        trigger=CronTrigger(hour=2, minute=0),
        id="price_historical",
        name="Price Historical (FRED + EIA)",
    )

    # ── AIS: every 30 minutes, on the hour and half-hour ─────────────────────
    scheduler.add_job(
        job_ais,
        trigger=CronTrigger(minute="0,30"),
        id="ais",
        name="AIS Ship Tracking (aisstream.io)",
    )

    # ── Processor: every 30 minutes, offset to :05 and :35 ───────────────────
    # Fires 5 minutes after AIS to ensure AIS data is written before processing
    scheduler.add_job(
        job_processor,
        trigger=CronTrigger(minute="5,35"),
        id="processor",
        name="Processor (features + anomaly detection + monitoring)",
    )

    # ── AI Engine: daily at 03:00 ─────────────────────────────────────────────
    scheduler.add_job(
        job_ai_engine,
        trigger=CronTrigger(hour=3, minute=0),
        id="ai_engine",
        name="AI Engine (Gemini embedding + signal correlation)",
    )

    # ── Causality: daily at 04:00 ─────────────────────────────────────────────
    scheduler.add_job(
        job_causality,
        trigger=CronTrigger(hour=4, minute=0),
        id="causality",
        name="Causality Engine (Gemini report generation)",
    )

    # ── Prediction: daily at 04:30 ────────────────────────────────────────────
    # Runs after causality to ensure reports are ready; generates scenario
    # predictions for any SignalAlerts that are still missing prediction_json.
    scheduler.add_job(
        job_prediction,
        trigger=CronTrigger(hour=4, minute=30),
        id="prediction",
        name="Prediction Engine (Gemini probabilistic scenario generation)",
    )

    # ── Evaluation: daily at 05:00 ────────────────────────────────────────────
    # Runs after prediction; evaluates completed monitoring windows against
    # actual outcomes and stores PredictionEvaluation + LearningUpdate rows.
    scheduler.add_job(
        job_evaluation,
        trigger=CronTrigger(hour=5, minute=0),
        id="evaluation",
        name="Evaluation Engine (post-event accuracy scoring + learning updates)",
    )

    # ── Satellite (S1 + S2): every 6 hours ───────────────────────────────────
    # Sentinel revisit is 5-12 days; 6h polling catches new acquisitions promptly
    scheduler.add_job(
        job_satellite,
        trigger=IntervalTrigger(hours=6),
        id="satellite",
        name="Satellite Ingestion (Sentinel-1 SAR + Sentinel-2 optical)",
    )

    # ── Aircraft (OpenSky ADS-B): every 30 minutes ────────────────────────────
    scheduler.add_job(
        job_aircraft,
        trigger=CronTrigger(minute="0,30"),
        id="aircraft",
        name="Aircraft Tracking (OpenSky ADS-B)",
    )

    # ── Rail (OSM Overpass): weekly Sunday 01:00 ──────────────────────────────
    # Rail geometry is semi-static; weekly refresh captures construction changes
    scheduler.add_job(
        job_rail,
        trigger=CronTrigger(day_of_week="sun", hour=1, minute=0),
        id="rail",
        name="Rail Corridor Geometry (OSM Overpass)",
    )

    # ── DB cleanup: daily at 01:30 ────────────────────────────────────────────
    # Removes processed raw_ingestion rows older than 90 days.
    # Runs before price_historical (02:00) to keep the DB lean.
    scheduler.add_job(
        job_cleanup,
        trigger=CronTrigger(hour=1, minute=30),
        id="cleanup",
        name="DB Cleanup (raw_ingestion 90-day retention)",
    )

    # ── Qdrant backup: daily at 02:30 ─────────────────────────────────────────
    # Runs between price_historical (02:00) and ai_engine (03:00).
    # Creates a compressed snapshot of qdrant_data/, prunes copies >7 days old.
    scheduler.add_job(
        job_qdrant_backup,
        trigger=CronTrigger(hour=2, minute=30),
        id="qdrant_backup",
        name="Qdrant Vector DB Backup (7-day retention)",
    )

    return scheduler


def print_job_table(scheduler: BackgroundScheduler) -> None:
    """Print registered jobs and their next run times."""
    print("\n" + "=" * 72)
    print("  MCEI SCHEDULER - registered jobs")
    print("=" * 72)
    print(f"  {'Job ID':20s}  {'Next Run':26s}  Name")
    print("-" * 72)
    for job in scheduler.get_jobs():
        next_run = str(job.next_run_time)[:19] if job.next_run_time else "not scheduled"
        print(f"  {job.id:20s}  {next_run:26s}  {job.name}")
    print("=" * 72 + "\n")


def main():
    validate_secrets(abort_on_critical=True)
    init_db()
    scheduler = build_scheduler()

    def _shutdown(signum, frame):
        logger.info("scheduler_shutdown_signal", signal=signum)
        print("\nShutting down scheduler...")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    scheduler.start()
    logger.info("scheduler_started", job_count=len(scheduler.get_jobs()))
    print_job_table(scheduler)
    print("Scheduler running. Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(30)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown(wait=False)
        print("Scheduler stopped.")


if __name__ == "__main__":
    main()
