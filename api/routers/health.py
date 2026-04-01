"""GET /health — DB row counts, recent job runs, version, uptime."""

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.dependencies import get_db
from shared.config import settings

router = APIRouter(prefix="/health", tags=["health"])

_START_TIME = time.monotonic()
_START_DT   = datetime.now(timezone.utc)

VERSION = "0.1.0"

TABLES = [
    "raw_ingestion",
    "processed_features",
    "anomaly_events",
    "embeddings_cache",
    "signal_alerts",
    "causality_reports",
    "prediction_evaluations",
    "learning_updates",
]


@router.get("")
def health(db: Session = Depends(get_db)):
    # Table row counts
    counts = {}
    for table in TABLES:
        row = db.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()
        counts[table] = row[0] if row else 0

    # Recent job runs
    try:
        job_rows = db.execute(
            text("""
                SELECT job_name, started_at, finished_at, status, error
                FROM job_runs
                ORDER BY started_at DESC
                LIMIT 50
            """)
        ).fetchall()
        recent_jobs = [
            {
                "job_name": r[0],
                "started_at": r[1],
                "finished_at": r[2],
                "status": r[3],
                "error": r[4],
            }
            for r in job_rows
        ]
    except Exception:
        recent_jobs = []

    # Last successful run per job name
    last_success: dict[str, str] = {}
    for j in recent_jobs:
        if j["status"] == "ok" and j["finished_at"]:
            name = j["job_name"]
            if name not in last_success:
                last_success[name] = j["finished_at"]

    uptime_seconds = int(time.monotonic() - _START_TIME)

    return {
        "status": "ok",
        "version": VERSION,
        "started_at": _START_DT.isoformat(),
        "uptime_seconds": uptime_seconds,
        "db_path": settings.db_path,
        "auth_enabled": bool(settings.mcei_api_key),
        "table_counts": counts,
        "last_job_success": last_success,
        "recent_jobs": recent_jobs[:20],
    }
