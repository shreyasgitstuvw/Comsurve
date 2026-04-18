# MCEI — Multimodal Commodity Event Intelligence Engine

An autonomous AI pipeline for detecting, correlating, and predicting supply chain disruptions across LNG, Copper, and Soybeans markets. MCEI ingests realtime news, vessel tracking, satellite, price, and sentiment signals; embeds them with Gemini; detects anomalies; runs causal analysis; and generates calibrated directional price predictions with a self-improving feedback loop.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Data Sources                                                               │
│  News · yFinance · FRED/EIA · AIS · OpenSky · OSM Rail · Satellite         │
└────────────────────────┬────────────────────────────────────────────────────┘
                         │ Ingestion Layer (APScheduler)
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 1 · Ingestion   raw_ingestion table + Qdrant vectors                 │
└────────────────────────┬────────────────────────────────────────────────────┘
                         │ Processor (every 30 min)
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 2 · Feature Engineering   processed_features · embeddings_cache      │
└────────────────────────┬────────────────────────────────────────────────────┘
                         │ AI Engine (daily 03:00)
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 3 · Anomaly Detection     anomaly_events · signal_alerts             │
│  Stage 4 · Causality             causality_reports (Gemini CoT)             │
│  Stage 5 · Prediction            prediction_evaluations (Platt calibration) │
└────────────────────────┬────────────────────────────────────────────────────┘
                         │ Feedback Loop (daily 05:00)
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Self-improvement      learning_store · feedback_controller · PID adjustments│
└────────────────────────┬────────────────────────────────────────────────────┘
                         │ REST API
                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  FastAPI   /health · /anomalies · /alerts · /causality · /predictions       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5-Stage Pipeline

| Stage | Module | Description |
|-------|--------|-------------|
| 1 · Ingestion | `ingestion/` | Pulls news, prices, AIS, aircraft, rail, satellite into `raw_ingestion` |
| 2 · Processing | `processor/` | Feature engineering, z-score anomaly scoring, Gemini embeddings |
| 3 · AI Engine | `ai_engine/ai_engine_runner.py` | Embeds, correlates signals via Qdrant similarity search |
| 4 · Causality | `ai_engine/causality_engine.py` | Gemini CoT causal analysis → `causality_reports` |
| 5 · Prediction | `ai_engine/prediction_engine.py` | Analog-based directional forecasting with Platt-scaled confidence |

---

## Data Sources

| Source | Type | Cadence | Module |
|--------|------|---------|--------|
| NewsAPI / GNews | News headlines | Every 6h | `ingestion/news/` |
| yFinance | Realtime prices | Every 1h | `ingestion/price_realtime/` |
| FRED | Macro indicators | Daily 02:00 | `ingestion/price_historical/fred_ingestor.py` |
| EIA | Energy inventories | Daily 02:00 | `ingestion/price_historical/eia_ingestor.py` |
| AISStream | Vessel positions | Every 30 min | `ingestion/ais/` |
| OpenSky | Aircraft tracking | On schedule | `ingestion/aircraft/` |
| OSM Overpass | Rail infrastructure | On schedule | `ingestion/rail/` |
| Sentinel Hub | Satellite imagery | On schedule | `ingestion/satellite/` |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| AI / Embeddings | Google Gemini (`gemini-embedding-001` 3072-dim, `gemini-2.0-flash`) |
| Vector DB | Qdrant (local embedded, one collection per commodity) |
| Relational DB | SQLite + WAL mode, Alembic migrations |
| API | FastAPI + uvicorn, slowapi rate limiting |
| Scheduler | APScheduler 3.x, 14+ background jobs |
| ORM | SQLAlchemy 2.x |
| Testing | pytest, 334 tests |
| Containers | Docker + docker-compose |
| CI | GitHub Actions |

---

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/your-org/mcei.git
cd mcei

cp .env.example .env
# Edit .env — at minimum set GEMINI_API_KEY

docker-compose up -d
```

Services:
- **API**: http://localhost:8000
- **Health check**: http://localhost:8000/health

### Local Development

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env              # fill in API keys

# Apply DB migrations
alembic upgrade head

# Start the API
uvicorn api.main:app --reload --port 8000

# Start the scheduler (separate terminal)
python -m scheduler.scheduler_runner
```

---

## Configuration

Copy `.env.example` to `.env` and fill in values. Keys are grouped by priority:

### Critical (pipeline will not run without these)

| Key | Description |
|-----|-------------|
| `GEMINI_API_KEY` | Google AI Studio API key (embeddings + LLM calls) |
| `DB_PATH` | Path to SQLite database file (default: `./data/mcei.db`) |
| `QDRANT_PATH` | Path to Qdrant storage directory (default: `./data/qdrant`) |

### Degraded without these

| Key | Description |
|-----|-------------|
| `NEWS_API_KEY` | NewsAPI key for headline ingestion |
| `EIA_API_KEY` | U.S. Energy Information Administration API key |
| `FRED_API_KEY` | Federal Reserve Economic Data API key |
| `AISSTREAM_API_KEY` | AIS vessel tracking websocket key |
| `MCEI_API_KEY` | REST API authentication key (auth disabled if empty) |

### Optional

| Key | Description |
|-----|-------------|
| `OPENSKY_USER` / `OPENSKY_PASS` | OpenSky aircraft tracking credentials |
| `SENTINEL_HUB_CLIENT_ID` / `SECRET` | Satellite imagery credentials |
| `UVICORN_WORKERS` | API worker count (default: 2) |
| `SCHEDULER_HEARTBEAT_PATH` | Heartbeat file path for Docker healthcheck |

---

## API Reference

All endpoints return JSON. If `MCEI_API_KEY` is set, include `X-API-Key: <key>` header.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info and endpoint list |
| GET | `/health` | System health (DB counts, Qdrant, staleness) |
| GET | `/anomalies` | List anomaly events (filters: commodity, type, status, limit) |
| GET | `/anomalies/{id}` | Single anomaly event by ID |
| GET | `/alerts` | Signal alerts with predictions (filters: commodity, status, limit) |
| GET | `/alerts/{id}` | Single alert by ID |
| GET | `/causality` | Causality reports (filters: commodity, limit) |
| GET | `/predictions` | Prediction evaluations (filters: commodity, limit) |
| GET | `/metrics` | Aggregate counts and statistics (auth required) |

### `/health` Response Example

```json
{
  "status": "ok",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "last_job_success": "2026-04-18T05:00:00",
  "qdrant_healthy": true,
  "ai_engine_stale": false,
  "table_counts": {
    "raw_ingestion": 12450,
    "processed_features": 8320,
    "anomaly_events": 47,
    "embeddings_cache": 1280,
    "signal_alerts": 37,
    "causality_reports": 37,
    "prediction_evaluations": 37,
    "learning_updates": 12
  }
}
```

Status values: `ok` (all healthy), `warning` (ai_engine stale > 30h), `degraded` (Qdrant unreachable).

### Common Query Parameters

```
GET /anomalies?commodity=lng&anomaly_type=price_spike&status=new&limit=20
GET /alerts?commodity=copper&status=predicted&limit=10
GET /causality?commodity=soybeans&limit=5
```

---

## Scheduler Jobs

All jobs run via APScheduler with `coalesce=True`, `max_instances=1`, `misfire_grace_time=300`.

| Job | Schedule | Description |
|-----|----------|-------------|
| `news` | Every 6h | NewsAPI + GNews headline ingestion |
| `price_realtime` | Every 1h | yFinance spot price ingestion |
| `ais` | Every 30 min | AISStream vessel position ingestion |
| `processor` | Every 30 min (+:30 offset) | Feature engineering, z-score, embedding |
| `price_historical` | Daily 02:00 | FRED macro + EIA energy inventory pull |
| `ai_engine` | Daily 03:00 | Embed + correlate signals across Qdrant collections |
| `sqlite_backup` | Daily 03:30 | `VACUUM INTO` atomic SQLite backup, 7-day retention |
| `causality` | Daily 04:00 | Gemini CoT causality analysis on new anomalies |
| `prediction` | Daily 04:30 | Analog-based directional price prediction |
| `evaluation` | Daily 05:00 | Score prior predictions vs realised outcomes |
| `satellite` | On schedule | Sentinel Hub satellite imagery ingestion |
| `aircraft` | On schedule | OpenSky aircraft position ingestion |
| `rail` | On schedule | OSM Overpass rail infrastructure updates |
| `qdrant_backup` | On schedule | Qdrant snapshot backup, 7-day retention |
| `cleanup` | On schedule | Prune old raw_ingestion rows |

---

## Prediction & Recalibration Model

### Analog-Based Prediction

Each new anomaly event is embedded (Gemini `gemini-embedding-001`, 3072 dimensions) and matched against the historical `embeddings_cache` in Qdrant using cosine similarity. Up to 8 similar past events (threshold: 0.55) are retrieved as analogs; cross-commodity analogs at threshold 0.75 are also included.

The prediction prompt provides Gemini with:
- **Anomaly trajectory**: 7-day pre-event price z-score and sentiment trend
- **Market context**: 30-day price direction label
- **Top-8 analogs**: past similar events with their actual outcomes (weak analogs flagged at < 0.70 score)
- **Horizon guidance**: asset-specific typical resolution windows
- **Chain-of-thought reasoning**: STEP 1 explicit reasoning before JSON output

Output schema:

```json
{
  "direction": "up|down|neutral",
  "magnitude_pct": 2.5,
  "confidence": 0.82,
  "horizon_days": 14,
  "reasoning": "...",
  "invalidating_conditions": ["..."]
}
```

### Confidence Calibration

Raw LLM confidence undergoes two-stage calibration (`ai_engine/confidence_calibrator.py`):

1. **Platt scaling** (if ≥ 15 historical evaluation pairs exist): logistic regression `P = sigmoid(A·x + B)` fitted via gradient descent on `(raw_confidence, direction_correct)` pairs. Corrects systematic over/under-confidence.

2. **Shrinkage prior** (fallback for new commodities / cold start): `calibrated = 0.7 × raw + 0.15`. Pulls LLM confidence toward 0.5 prior, preventing extreme claims without evidence.

### Evaluation Scoring

Each prediction is evaluated against realised outcomes:

```
score = 0.4 × direction_correct
      + max(0, 0.3 − magnitude_error × 0.03)
      + calibration_score  # 0.3 if well-calibrated, 0.15 if under, 0.0 if over
```

### PID-Inspired Feedback Controller

`ai_engine/feedback_controller.py` reads the last N evaluations and computes three damped error signals (α=0.5 exponential smoothing):

- **P-channel**: adjusts confidence threshold (raise if direction error > 0.6, lower if < 0.2)
- **I-channel**: injects driver rules from repeated failure modes (demand miss → "always include demand analysis")
- **D-channel**: reduces analogy reliance when total error > 0.7 across 3+ evaluations

### Cross-Commodity Learning

Evaluations from other commodities sharing the same `anomaly_type` are prepended as soft priors. Same-commodity evaluations are weighted more heavily (placed last in the damped aggregation). Insights surfaced in the prediction prompt are drawn only from same-commodity history.

---

## Database Schema

| Table | Key Columns | Description |
|-------|-------------|-------------|
| `raw_ingestion` | commodity, source_type, content_hash, ingested_at | Deduped raw data from all sources |
| `processed_features` | commodity, feature_type, z_score, processed_at | Normalised numeric features |
| `anomaly_events` | commodity, anomaly_type, severity, status | Detected supply chain events |
| `embeddings_cache` | anomaly_event_id, embedding_json, qdrant_id | Gemini embeddings + Qdrant pointer |
| `signal_alerts` | anomaly_event_id, prediction_json, confidence | Per-event prediction output |
| `causality_reports` | anomaly_event_id, causal_factors_json, confidence | LLM causal chains |
| `prediction_evaluations` | signal_alert_id, accuracy_json, overall_score | Ground-truth evaluation |
| `learning_updates` | commodity, anomaly_type, adjustments_json | Applied feedback controller output |
| `job_history` | job_name, run_id, status, error, started_at | Scheduler run audit log |

---

## Testing

```bash
pytest                        # run all 334 tests
pytest tests/ -v              # verbose
pytest tests/test_confidence_calibrator.py  # single module
```

### Test Coverage by Module

| Module | Tests | Strategy |
|--------|-------|----------|
| `confidence_calibrator` | 17 | Pure functions + mocked DB |
| `feedback_controller` | 24 | Pure functions + parameterised |
| `learning_store` | 6 | In-memory SQLite fixture |
| `signal_correlator` | 20+ | Mocked Qdrant client |
| `causality_prediction` | 15+ | Mocked Gemini + DB |
| `api_endpoints` | 15 | FastAPI TestClient + in-memory SQLite |
| `evaluation_engine` | 12+ | Mocked DB sessions |
| `ingestion` | 80+ | Mocked HTTP + DB |
| `processor` | 40+ | Feature engineering unit tests |

---

## CI

GitHub Actions runs on every push and pull request to `main`:

```yaml
# .github/workflows/ci.yml
- pytest (334 tests)
- flake8 lint check
```

---

## Deployment Checklist

Before going to production:

- [ ] Set `GEMINI_API_KEY` in `.env`
- [ ] Set `MCEI_API_KEY` in `.env` (leave empty to disable auth)
- [ ] Set `NEWS_API_KEY`, `EIA_API_KEY`, `FRED_API_KEY`, `AISSTREAM_API_KEY`
- [ ] Run `alembic upgrade head` to apply all DB migrations
- [ ] Verify `docker-compose up -d` starts both `api` and `scheduler` services
- [ ] Confirm `/health` returns `{"status": "ok", "qdrant_healthy": true}`
- [ ] Monitor first scheduler cycle (processor → ai_engine → causality → prediction → evaluation)
- [ ] Check `job_history` table for any job failures after first run
- [ ] Configure Prometheus scraping at `/metrics` (auth required)

---

## Monitoring

The `/metrics` endpoint (requires API key) exposes aggregate counts suitable for Prometheus scraping or dashboard import.

The scheduler writes a heartbeat timestamp every 30 seconds to `$SCHEDULER_HEARTBEAT_PATH` (default: `/data/scheduler_heartbeat`). The Docker healthcheck fails if the heartbeat is older than 90 seconds:

```yaml
healthcheck:
  test: ["CMD", "python", "-c",
         "import time,os; age=time.time()-os.path.getmtime('/data/scheduler_heartbeat'); exit(0 if age<90 else 1)"]
  interval: 60s
  timeout: 5s
  retries: 3
  start_period: 45s
```

---

## Directory Structure

```
mcei/
├── ai_engine/
│   ├── ai_engine_runner.py        # Stage 3: embed + correlate
│   ├── causality_engine.py        # Stage 4: Gemini CoT causality
│   ├── confidence_calibrator.py   # Platt scaling + shrinkage prior
│   ├── evaluation_engine.py       # Score predictions vs reality
│   ├── feedback_controller.py     # PID-inspired error signal → adjustments
│   ├── learning_store.py          # Fetch evaluations + cross-commodity pool
│   ├── prediction_engine.py       # Stage 5: analog-based prediction
│   └── signal_correlator.py       # Qdrant similarity search
├── api/
│   ├── main.py                    # FastAPI app + startup logging
│   ├── dependencies.py            # get_db, require_api_key
│   └── routers/
│       ├── health.py              # /health with Qdrant + staleness checks
│       ├── anomalies.py           # /anomalies CRUD
│       ├── alerts.py              # /alerts CRUD
│       ├── causality.py           # /causality read
│       ├── predictions.py         # /predictions read
│       └── metrics.py             # /metrics aggregate counts
├── ingestion/
│   ├── news/                      # NewsAPI + GNews
│   ├── price_realtime/            # yFinance
│   ├── price_historical/          # FRED + EIA
│   ├── ais/                       # AISStream vessel tracking
│   ├── aircraft/                  # OpenSky
│   ├── rail/                      # OSM Overpass
│   └── satellite/                 # Sentinel Hub
├── processor/                     # Feature engineering + z-score
├── scheduler/
│   ├── scheduler_runner.py        # APScheduler main loop + heartbeat
│   ├── jobs.py                    # All job function definitions
│   ├── alerting.py                # Job success/failure alerting
│   └── job_history.py             # run audit log helpers
├── shared/
│   ├── config.py                  # Pydantic Settings
│   ├── db.py                      # SQLAlchemy engine + get_session
│   ├── models.py                  # ORM models
│   ├── logger.py                  # structlog configuration
│   ├── commodity_registry.py      # COMMODITY_LIST, SCHEDULE_MAP
│   └── qdrant_manager.py          # Qdrant client wrapper
├── scripts/
│   ├── backtest_engine.py         # Historical replay for prompt tuning
│   ├── backup_qdrant.py           # Qdrant snapshot backup
│   └── entrypoint.sh              # Docker entrypoint
├── tests/                         # 334 pytest tests
├── alembic/                       # DB migrations
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Gemini Quota Notes

MCEI uses the Gemini API with a built-in fallback chain:

1. `gemini-2.0-flash` (primary)
2. `gemini-2.0-flash-lite` (first fallback)
3. `gemini-2.5-flash` (second fallback)

Embedding calls use `gemini-embedding-001` exclusively (no fallback — embeddings must be dimensionally consistent). If the quota is exhausted, embedding jobs will be skipped for that cycle and retried at the next scheduled run.

Monitor quota usage at [Google AI Studio](https://aistudio.google.com/).

---

## License

MIT
