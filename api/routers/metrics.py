"""
GET /metrics — Prometheus text format metrics endpoint.

Exposes:
  - mcei_table_rows{table="..."} — row counts for all 8 MCEI tables
  - mcei_uptime_seconds          — API process uptime
  - mcei_job_last_success_ts{job="..."} — Unix timestamp of last successful run
  - mcei_anomalies_total{status="..."} — anomaly event counts by status
  - mcei_build_info{version="..."} — static build metadata gauge (value=1)

No external prometheus_client dependency required — the exposition format
(https://prometheus.io/docs/instrumenting/exposition_formats/) is plain text
and simple enough to write directly.

Scrape with:
    curl http://localhost:8000/metrics
    prometheus.yml: - targets: ['localhost:8000']
                      metrics_path: /metrics
"""

import time
from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse
from sqlalchemy import text
from sqlalchemy.orm import Session


from api.dependencies import get_db, require_api_key
from api.routers.health import TABLES, VERSION, _START_TIME, _count_table

router = APIRouter(prefix="/metrics", tags=["observability"])

_EXPECTED_JOBS = [
    "news", "price_realtime", "price_historical", "ais",
    "processor", "ai_engine", "causality", "prediction", "evaluation",
]


def _prom_line(name: str, labels: dict, value: float | int, comment: str = "") -> str:
    """Format a single Prometheus metric line."""
    label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
    suffix = f"{{{label_str}}}" if label_str else ""
    return f"{name}{suffix} {value}"


@router.get("", response_class=PlainTextResponse, include_in_schema=False,
            dependencies=[Depends(require_api_key)])
def prometheus_metrics(db: Session = Depends(get_db)) -> str:
    """Return all MCEI metrics in Prometheus text exposition format."""
    lines: list[str] = []

    # ── mcei_build_info ───────────────────────────────────────────────────────
    lines += [
        "# HELP mcei_build_info Static build information (value always 1)",
        "# TYPE mcei_build_info gauge",
        _prom_line("mcei_build_info", {"version": VERSION}, 1),
    ]

    # ── mcei_uptime_seconds ───────────────────────────────────────────────────
    uptime = int(time.monotonic() - _START_TIME)
    lines += [
        "# HELP mcei_uptime_seconds Seconds since the API process started",
        "# TYPE mcei_uptime_seconds counter",
        _prom_line("mcei_uptime_seconds", {}, uptime),
    ]

    # ── mcei_table_rows ───────────────────────────────────────────────────────
    lines += [
        "# HELP mcei_table_rows Number of rows in each SQLite table",
        "# TYPE mcei_table_rows gauge",
    ]
    for table in TABLES:
        count = _count_table(db, table)
        lines.append(_prom_line("mcei_table_rows", {"table": table}, count))

    # ── mcei_anomalies_total ──────────────────────────────────────────────────
    lines += [
        "# HELP mcei_anomalies_total Anomaly events grouped by status",
        "# TYPE mcei_anomalies_total gauge",
    ]
    for status in ("new", "embedding_queued", "processed"):
        row = db.execute(
            text("SELECT COUNT(*) FROM anomaly_events WHERE status = :s"),
            {"s": status},
        ).fetchone()
        lines.append(_prom_line("mcei_anomalies_total", {"status": status},
                                row[0] if row else 0))

    # ── mcei_job_last_success_ts ──────────────────────────────────────────────
    lines += [
        "# HELP mcei_job_last_success_ts Unix timestamp of each job's last successful run (0 = never)",
        "# TYPE mcei_job_last_success_ts gauge",
    ]
    try:
        job_rows = db.execute(
            text("""
                SELECT job_name, finished_at
                FROM job_runs
                WHERE status = 'ok' AND finished_at IS NOT NULL
                ORDER BY started_at DESC
                LIMIT 200
            """)
        ).fetchall()
        last_ts: dict[str, float] = {}
        for job_name, finished_at in job_rows:
            if job_name not in last_ts:
                try:
                    dt = datetime.fromisoformat(str(finished_at))
                    last_ts[job_name] = dt.timestamp()
                except (ValueError, TypeError):
                    pass
    except Exception:
        last_ts = {}

    for job in _EXPECTED_JOBS:
        ts = last_ts.get(job, 0)
        lines.append(_prom_line("mcei_job_last_success_ts", {"job": job}, ts))

    return "\n".join(lines) + "\n"
