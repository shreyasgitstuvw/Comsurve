#!/bin/bash
# MCEI container entrypoint.
# Routes to the correct process based on SERVICE env var.
#
# SERVICE values:
#   api        — FastAPI via uvicorn (default)
#   scheduler  — APScheduler background job runner
#   dashboard  — Streamlit dashboard
#
# The /data volume must be mounted before this runs.
# Alembic migrations are applied on every API startup to keep the schema current.

set -euo pipefail

SERVICE="${SERVICE:-api}"
LOG_DIR="${LOG_FILE%/*}"          # extract directory from LOG_FILE path
if [[ -n "${LOG_DIR}" && "${LOG_DIR}" != "${LOG_FILE}" ]]; then
    mkdir -p "${LOG_DIR}"
fi

mkdir -p "${DB_PATH%/*}" 2>/dev/null || true
mkdir -p "${QDRANT_PATH}" 2>/dev/null || true

case "${SERVICE}" in

    api)
        echo "[mcei] Running Alembic migrations..."
        alembic upgrade head

        echo "[mcei] Starting API server (uvicorn) on ${API_HOST:-0.0.0.0}:${API_PORT:-8000}"
        # Default 2 workers so one slow request can't block the /health probe.
        # Set UVICORN_WORKERS in .env for higher-traffic deployments.
        exec uvicorn api.main:app \
            --host "${API_HOST:-0.0.0.0}" \
            --port "${API_PORT:-8000}" \
            --workers "${UVICORN_WORKERS:-2}" \
            --log-level warning \
            --no-access-log
        ;;

    scheduler)
        echo "[mcei] Starting background scheduler..."
        exec python -m scheduler.scheduler_runner
        ;;

    dashboard)
        echo "[mcei] Starting Streamlit dashboard on port 8501..."
        exec streamlit run dashboard/app.py \
            --server.port 8501 \
            --server.address 0.0.0.0 \
            --server.headless true \
            --browser.gatherUsageStats false
        ;;

    *)
        echo "[mcei] ERROR: Unknown SERVICE='${SERVICE}'. Valid values: api, scheduler, dashboard" >&2
        exit 1
        ;;

esac
