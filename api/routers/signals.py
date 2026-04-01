"""GET /signals — query signal alerts."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.dependencies import get_db
from shared.models import SignalAlert

router = APIRouter(prefix="/signals", tags=["signals"])


@router.get("")
def list_signals(
    commodity: Optional[str] = Query(None),
    alert_type: Optional[str] = Query(None, description="similar_historical | novel_event"),
    monitoring_complete: Optional[bool] = Query(None),
    since: Optional[datetime] = Query(None),
    limit: int = Query(50, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    q = db.query(SignalAlert).order_by(SignalAlert.created_at.desc())
    if commodity:
        q = q.filter(SignalAlert.commodity == commodity)
    if alert_type:
        q = q.filter(SignalAlert.alert_type == alert_type)
    if monitoring_complete is not None:
        q = q.filter(SignalAlert.monitoring_complete == monitoring_complete)
    if since:
        q = q.filter(SignalAlert.created_at >= since)

    rows = q.offset(offset).limit(limit).all()
    return [
        {
            "id": r.id,
            "anomaly_event_id": r.anomaly_event_id,
            "commodity": r.commodity,
            "alert_type": r.alert_type,
            "correlated_anomaly_ids": r.correlated_anomaly_ids,
            "similarity_scores": r.similarity_scores,
            "price_at_alert": r.price_at_alert,
            "price_1w": r.price_1w,
            "price_2w": r.price_2w,
            "price_1m": r.price_1m,
            "monitoring_complete": r.monitoring_complete,
            "created_at": r.created_at,
        }
        for r in rows
    ]


@router.get("/{signal_id}")
def get_signal(signal_id: int, db: Session = Depends(get_db)):
    row = db.query(SignalAlert).filter(SignalAlert.id == signal_id).first()
    if not row:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Signal alert not found")
    return {
        "id": row.id,
        "anomaly_event_id": row.anomaly_event_id,
        "commodity": row.commodity,
        "alert_type": row.alert_type,
        "correlated_anomaly_ids": row.correlated_anomaly_ids,
        "similarity_scores": row.similarity_scores,
        "price_at_alert": row.price_at_alert,
        "price_1w": row.price_1w,
        "price_2w": row.price_2w,
        "price_1m": row.price_1m,
        "monitoring_complete": row.monitoring_complete,
        "created_at": row.created_at,
    }
