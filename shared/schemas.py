"""
Pydantic schemas for API responses and inter-service contracts.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel


# ── Raw Ingestion ─────────────────────────────────────────────────────────────

class RawIngestionRecord(BaseModel):
    """Used by ingestors to pass data to base_ingestor.save_to_db()."""
    source: str
    commodity: str
    symbol: str
    timestamp: datetime
    data_type: str
    raw_json: str


# ── Processed Features ────────────────────────────────────────────────────────

class ProcessedFeatureOut(BaseModel):
    id: int
    raw_ingestion_id: int
    commodity: str
    feature_type: str
    value: float | None
    window: str | None
    computed_at: datetime

    class Config:
        from_attributes = True


# ── Anomaly Events ────────────────────────────────────────────────────────────

class AnomalyEventOut(BaseModel):
    id: int
    commodity: str
    anomaly_type: str
    severity: float
    detected_at: datetime
    source_ids: str           # JSON array string
    status: str
    metadata_json: str | None

    class Config:
        from_attributes = True


# ── Signal Alerts ─────────────────────────────────────────────────────────────

class SignalAlertOut(BaseModel):
    id: int
    anomaly_event_id: int
    commodity: str
    alert_type: str
    correlated_anomaly_ids: str
    similarity_scores: str
    price_at_alert: float | None
    price_1w: float | None
    price_2w: float | None
    price_1m: float | None
    monitoring_complete: bool
    created_at: datetime

    class Config:
        from_attributes = True


# ── Causality Reports ─────────────────────────────────────────────────────────

class CausalityReportOut(BaseModel):
    id: int
    signal_alert_id: int
    commodity: str
    report_json: str
    cause_category: str | None
    confidence_score: float | None
    price_impact_pct: float | None
    created_at: datetime

    class Config:
        from_attributes = True


# ── Health ────────────────────────────────────────────────────────────────────

class HealthOut(BaseModel):
    status: str
    table_counts: dict[str, int]
    db_path: str


# ── Price ─────────────────────────────────────────────────────────────────────

class PricePoint(BaseModel):
    timestamp: datetime
    commodity: str
    symbol: str
    value: float
    source: str
