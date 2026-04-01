# MCEI — Deployment Guide

Multimodal Commodity Event Intelligence Engine — production deployment reference.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Docker + Docker Compose | 24+ |
| Python (local dev only) | 3.11+ |
| Disk space | ≥ 5 GB (DB + Qdrant + logs) |

---

## Quick Start (Docker)

```bash
# 1. Clone and configure
cp .env.example .env
# Fill in GEMINI_API_KEY (required) and any other keys you have

# 2. Start API + Scheduler
docker compose up -d

# 3. Verify health
curl http://localhost:8000/health

# 4. (Optional) Start dashboard
docker compose --profile dashboard up -d dashboard
```

---

## Required Configuration

Only one key is truly required to start the system:

```
GEMINI_API_KEY=<your key>   # AI embedding, causality, prediction
```

For production, also set:

```
MCEI_API_KEY=<random 64-char hex>   # API authentication
# Generate: python -c "import secrets; print(secrets.token_hex(32))"

CORS_ORIGINS=https://your-dashboard-domain.com
```

All other keys enable additional data sources. The system degrades gracefully — missing sources are skipped with a warning at startup.

---

## Local Development (no Docker)

```bash
# 1. Create and activate virtualenv
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# edit .env

# 4. Initialise the database
alembic upgrade head

# 5. Run API
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

# 6. Run scheduler (separate terminal)
python -m scheduler.scheduler_runner

# 7. Run dashboard (separate terminal)
streamlit run dashboard/app.py
```

---

## Database Migrations

All schema changes are managed by Alembic.

```bash
# Apply all pending migrations (run on every deployment)
alembic upgrade head

# Check current migration version
alembic current

# Show pending migrations
alembic history --indicate-current

# Roll back one migration (emergency rollback)
alembic downgrade -1

# Roll back to a specific revision
alembic downgrade <revision_id>
```

**Migration checklist before deploying a new version:**
1. Back up `mcei.db`: `cp mcei.db mcei.db.bak`
2. Back up Qdrant: `python scripts/backup_qdrant.py`
3. Run `alembic upgrade head`
4. Restart services
5. Verify `/health` returns `"status": "ok"`

---

## Rollback Strategy

### Application rollback (Docker)

Pin image versions in `docker-compose.yml`. To roll back:

```bash
# 1. Stop current version
docker compose down

# 2. Edit docker-compose.yml to previous image tag (or use git revert)

# 3. Roll back database migration
alembic downgrade -1    # or to a specific revision

# 4. Restart
docker compose up -d

# 5. Verify
curl http://localhost:8000/health
```

### Database rollback

Every migration file in `alembic/versions/` contains a `downgrade()` function. Run in sequence:

```bash
# Show history
alembic history

# Roll back to specific safe point
alembic downgrade <target_revision>
```

### Qdrant rollback

Qdrant data is backed up nightly to `./qdrant_backups/`. To restore:

```bash
# Stop services
docker compose down

# Remove current Qdrant data
rm -rf qdrant_data/

# Extract backup
tar -xzf qdrant_backups/qdrant_snapshot_<timestamp>.tar.gz

# Rename extracted dir
mv qdrant_snapshot_<timestamp> qdrant_data

# Restart
docker compose up -d
```

---

## Verifying a Deployment

Run the smoke test script after any deployment:

```bash
# Full pipeline run (requires live API keys)
python scripts/run_pipeline_once.py

# Skip ingestion (if data already exists), still runs AI + DB health check
python scripts/run_pipeline_once.py --skip-ingest

# Skip AI (saves Gemini quota during verification)
python scripts/run_pipeline_once.py --skip-ai
```

Check the API:
```bash
# Health endpoint
curl http://localhost:8000/health

# Prometheus metrics
curl http://localhost:8000/metrics

# Anomaly list (requires API key if MCEI_API_KEY is set)
curl -H "X-API-Key: <your_key>" http://localhost:8000/anomalies
```

---

## Monitoring & Alerting

### Logs

```bash
# Docker logs
docker compose logs -f api
docker compose logs -f scheduler

# Structured log fields: timestamp, level, event, job, error, anomaly_id, etc.
```

### Prometheus scraping

The `/metrics` endpoint exposes Prometheus-format metrics. Configure your scraper:

```yaml
# prometheus.yml
scrape_configs:
  - job_name: mcei
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
    # Add auth header if MCEI_API_KEY is set:
    # authorization:
    #   credentials: <your_api_key>
```

Key metrics:
- `mcei_anomalies_total{status="new"}` — backlog of unprocessed anomalies
- `mcei_job_last_success_ts{job="ai_engine"}` — Unix timestamp of last successful run
- `mcei_table_rows{table="raw_ingestion"}` — total ingested records

### Webhook alerts

Set `MCEI_ALERT_WEBHOOK_URL` to receive alerts for:
- 3+ consecutive job failures
- Gemini daily quota exhaustion
- Any job silent for >26 hours

---

## Scheduler Job Schedule

| Job | Schedule | Description |
|---|---|---|
| price_realtime | Every 1h | yfinance → Commodities-API fallback |
| news | Every 6h | newsdata.io → newsapi.ai fallback |
| ais | Every 30min | aisstream.io vessel tracking |
| aircraft | Every 30min | OpenSky ADS-B |
| satellite | Every 6h | Sentinel-1 SAR + Sentinel-2 optical |
| processor | Every 30min | Feature extraction + anomaly detection |
| price_historical | Daily 02:00 | FRED + EIA |
| ai_engine | Daily 03:00 | Gemini embedding + signal correlation |
| causality | Daily 04:00 | Causal analysis reports |
| prediction | Daily 04:30 | Probabilistic price scenarios |
| evaluation | Daily 05:00 | Prediction accuracy scoring |
| qdrant_backup | Daily 02:30 | Qdrant snapshot (7-day retention) |
| rail | Weekly Sun 01:00 | OSM rail corridor geometry |

---

## Directory Structure

```
mcei/
├── api/              FastAPI application
├── ai_engine/        Gemini embedding, causality, prediction, evaluation
├── dashboard/        Streamlit dashboard
├── ingestion/        All data ingestors (news, price, AIS, satellite, aircraft, rail)
├── processor/        Feature extraction + anomaly detection
├── scheduler/        APScheduler job definitions + alerting
├── shared/           Models, DB, config, logger
├── alembic/          Database migrations
├── scripts/          Utility scripts (smoke test, backup, backfill)
├── tests/            Pytest test suite
├── reports/          Generated causality report JSON files
├── qdrant_data/      Qdrant vector DB (local embedded mode)
├── qdrant_backups/   Nightly Qdrant snapshots
├── .env              Secrets (never commit)
├── .env.example      Template — commit this
└── docker-compose.yml
```

---

## Common Issues

**Startup fails with "missing CRITICAL secrets"**
→ Add `GEMINI_API_KEY` to `.env`

**API returns 401 on all requests**
→ Supply `X-API-Key: <value>` header matching `MCEI_API_KEY` in `.env`

**`alembic upgrade head` fails**
→ Delete `mcei.db` and retry (first-time setup) or check migration history

**Qdrant data directory not found**
→ Run at least one AI engine job: `python -m ai_engine.ai_engine_runner`

**Dashboard shows "No data"**
→ Run `python scripts/run_pipeline_once.py` to seed the database
