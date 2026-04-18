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

# Explicit allowlist — prevents any f-string table injection even if this list
# is ever constructed dynamically. Only names in this set reach SQL.
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
_ALLOWED_TABLES: frozenset[str] = frozenset(TABLES)


def _count_table(db: Session, table: str) -> int:
    """Count rows in a known-safe table. Raises if name is not in the allowlist."""
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"Unknown table: {table!r}")
    # Table names cannot be parameterised in SQLite; allowlist check above is the guard.
    row = db.execute(text(f"SELECT COUNT(*) FROM {table}")).fetchone()  # noqa: S608
    return row[0] if row else 0


def _qdrant_ok() -> bool:
    """Return True if Qdrant local store is reachable and has at least one collection."""
    try:
        from ai_engine.qdrant_manager import QdrantManager
        qm = QdrantManager()
        collections = qm._client.get_collections().collections
        qm.close()
        return len(collections) > 0
    except Exception:
        return False


def _ai_engine_stale(db: Session, staleness_hours: int = 30) -> bool:
    """
    Return True if the ai_engine job has not succeeded in the last staleness_hours.
    Indicates the embedding + correlation pipeline is behind.
    """
    try:
        row = db.execute(
            text("""
                SELECT finished_at FROM job_runs
                WHERE job_name = 'ai_engine' AND status = 'ok'
                ORDER BY finished_at DESC LIMIT 1
            """)
        ).fetchone()
        if not row or not row[0]:
            return True
        last_run = datetime.fromisoformat(str(row[0]))
        if last_run.tzinfo is None:
            last_run = last_run.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - last_run).total_seconds() / 3600
        return age_hours > staleness_hours
    except Exception:
        return True


@router.get("")
def health(db: Session = Depends(get_db)):
    # Table row counts (allowlist-guarded)
    counts = {}
    for table in TABLES:
        counts[table] = _count_table(db, table)

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
    qdrant_healthy = _qdrant_ok()
    ai_stale = _ai_engine_stale(db)

    # Overall status degrades if Qdrant is unreachable or AI pipeline is stale
    overall = "ok"
    if not qdrant_healthy:
        overall = "degraded"
    elif ai_stale:
        overall = "warning"

    return {
        "status": overall,
        "version": VERSION,
        "started_at": _START_DT.isoformat(),
        "uptime_seconds": uptime_seconds,
        "db_path": settings.db_path,
        "auth_enabled": bool(settings.mcei_api_key),
        "qdrant_healthy": qdrant_healthy,
        "ai_engine_stale": ai_stale,
        "table_counts": counts,
        "last_job_success": last_success,
        "recent_jobs": recent_jobs[:20],
    }
