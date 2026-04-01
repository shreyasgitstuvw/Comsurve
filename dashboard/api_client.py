"""
Thin httpx wrapper for all MCEI API calls.
All dashboard pages import from here — no hardcoded URLs anywhere else.
Falls back gracefully when the API is offline.
"""

from datetime import datetime
from typing import Optional

import httpx

from shared.config import settings

BASE_URL = f"http://{settings.api_host}:{settings.api_port}"
TIMEOUT = 10


def _get(path: str, params: dict | None = None) -> dict | list | None:
    try:
        r = httpx.get(f"{BASE_URL}{path}", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def get_health() -> dict | None:
    return _get("/health")


def get_anomalies(
    commodity: str | None = None,
    status: str | None = None,
    anomaly_type: str | None = None,
    since: datetime | None = None,
    limit: int = 200,
) -> list[dict]:
    params = {"limit": limit}
    if commodity:
        params["commodity"] = commodity
    if status:
        params["status"] = status
    if anomaly_type:
        params["anomaly_type"] = anomaly_type
    if since:
        params["since"] = since.isoformat()
    result = _get("/anomalies", params)
    return result if isinstance(result, list) else []


def get_signals(
    commodity: str | None = None,
    alert_type: str | None = None,
    monitoring_complete: bool | None = None,
    limit: int = 100,
) -> list[dict]:
    params = {"limit": limit}
    if commodity:
        params["commodity"] = commodity
    if alert_type:
        params["alert_type"] = alert_type
    if monitoring_complete is not None:
        params["monitoring_complete"] = str(monitoring_complete).lower()
    result = _get("/signals", params)
    return result if isinstance(result, list) else []


def get_reports(commodity: str | None = None, limit: int = 50) -> list[dict]:
    params = {"limit": limit}
    if commodity:
        params["commodity"] = commodity
    result = _get("/reports", params)
    return result if isinstance(result, list) else []


def get_prices(commodity: str, window: str = "1m") -> dict | None:
    return _get(f"/prices/{commodity}", {"window": window})


def get_predictions(commodity: str | None = None, limit: int = 20) -> list[dict]:
    params: dict = {"limit": limit}
    if commodity:
        params["commodity"] = commodity
    result = _get("/predictions", params)
    return result if isinstance(result, list) else []


def get_evaluations(commodity: str | None = None, limit: int = 20) -> list[dict]:
    params: dict = {"limit": limit}
    if commodity:
        params["commodity"] = commodity
    result = _get("/evaluations", params)
    return result if isinstance(result, list) else []


def get_learning_updates(
    commodity: str | None = None,
    anomaly_type: str | None = None,
    limit: int = 50,
) -> list[dict]:
    params: dict = {"limit": limit}
    if commodity:
        params["commodity"] = commodity
    if anomaly_type:
        params["anomaly_type"] = anomaly_type
    result = _get("/evaluations/learning", params)
    return result if isinstance(result, list) else []
