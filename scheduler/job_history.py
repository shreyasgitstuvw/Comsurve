"""
Lightweight job run history logger.
Writes job start/end/error to a job_runs SQLite table for debugging and
system_health dashboard display.

Table is created lazily on first write — no schema migration needed.
"""

from datetime import datetime

from sqlalchemy import text

from shared.db import get_session
from shared.logger import get_logger

logger = get_logger(__name__)

_TABLE_CREATED = False


def _ensure_table() -> None:
    global _TABLE_CREATED
    if _TABLE_CREATED:
        return
    with get_session() as session:
        session.execute(text("""
            CREATE TABLE IF NOT EXISTS job_runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                job_name    TEXT NOT NULL,
                started_at  TEXT NOT NULL,
                finished_at TEXT,
                status      TEXT NOT NULL DEFAULT 'running',
                error       TEXT,
                summary_json TEXT
            )
        """))
    _TABLE_CREATED = True


def record_start(job_name: str) -> int:
    """Insert a job_runs row with status='running'. Returns the row id."""
    _ensure_table()
    with get_session() as session:
        result = session.execute(
            text("""
                INSERT INTO job_runs (job_name, started_at, status)
                VALUES (:name, :started, 'running')
            """),
            {"name": job_name, "started": datetime.utcnow().isoformat()},
        )
        return result.lastrowid


def record_end(run_id: int, summary: dict | None = None) -> None:
    """Mark a job run as completed."""
    import json
    _ensure_table()
    with get_session() as session:
        session.execute(
            text("""
                UPDATE job_runs
                SET finished_at = :finished, status = 'ok', summary_json = :summary
                WHERE id = :id
            """),
            {
                "finished": datetime.utcnow().isoformat(),
                "summary": json.dumps(summary) if summary else None,
                "id": run_id,
            },
        )


def record_error(run_id: int, error: str) -> None:
    """Mark a job run as failed."""
    _ensure_table()
    with get_session() as session:
        session.execute(
            text("""
                UPDATE job_runs
                SET finished_at = :finished, status = 'error', error = :error
                WHERE id = :id
            """),
            {"finished": datetime.utcnow().isoformat(), "error": error, "id": run_id},
        )


def get_recent_runs(limit: int = 50) -> list[dict]:
    """Return recent job_runs rows as dicts for the health dashboard."""
    import json
    _ensure_table()
    with get_session() as session:
        rows = session.execute(
            text("""
                SELECT job_name, started_at, finished_at, status, error, summary_json
                FROM job_runs
                ORDER BY started_at DESC
                LIMIT :limit
            """),
            {"limit": limit},
        ).fetchall()

    result = []
    for r in rows:
        result.append({
            "job_name": r[0],
            "started_at": r[1],
            "finished_at": r[2],
            "status": r[3],
            "error": r[4],
            "summary": json.loads(r[5]) if r[5] else None,
        })
    return result
