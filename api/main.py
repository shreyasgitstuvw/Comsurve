"""
MCEI FastAPI application.

Start with:
    uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

OpenAPI docs: http://127.0.0.1:8000/docs
"""

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from api.dependencies import require_api_key
from api.routers import anomalies, evaluations, health, metrics, predictions, prices, reports, signals
from shared.config import settings, validate_secrets
from shared.db import init_db
from shared.logger import configure_logging

configure_logging()
validate_secrets(abort_on_critical=True)
init_db()

# ── Rate limiter (120 req/min per IP by default) ──────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])

# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="MCEI — Multimodal Commodity Event Intelligence Engine",
    description=(
        "Supply chain disruption detection for LNG, Copper, and Soybeans. "
        "Correlates news, price, AIS vessel, and satellite signals through "
        "a 5-stage pipeline: anomaly detection → Gemini embedding → "
        "signal correlation → monitoring → causality report."
    ),
    version="0.1.0",
    dependencies=[Depends(require_api_key)],
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list(),
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(metrics.router)
app.include_router(anomalies.router)
app.include_router(signals.router)
app.include_router(reports.router)
app.include_router(prices.router)
app.include_router(predictions.router)
app.include_router(evaluations.router)


@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "MCEI API",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": [
            "/health", "/anomalies", "/signals", "/reports",
            "/prices/{commodity}", "/predictions", "/evaluations",
        ],
    }
