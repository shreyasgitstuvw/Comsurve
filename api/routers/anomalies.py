"""GET /anomalies — query anomaly events."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.dependencies import get_db
from shared.models import AnomalyEvent

router = APIRouter(prefix="/anomalies", tags=["anomalies"])


@router.get("")
def list_anomalies(
    commodity: Optional[str] = Query(None, description="lng | copper | soybeans"),
    status: Optional[str] = Query(None, description="new | embedding_queued | processed"),
    anomaly_type: Optional[str] = Query(None),
    since: Optional[datetime] = Query(None, description="ISO datetime filter on detected_at"),
    limit: int = Query(50, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    q = db.query(AnomalyEvent).order_by(AnomalyEvent.detected_at.desc())
    if commodity:
        q = q.filter(AnomalyEvent.commodity == commodity)
    if status:
        q = q.filter(AnomalyEvent.status == status)
    if anomaly_type:
        q = q.filter(AnomalyEvent.anomaly_type == anomaly_type)
    if since:
        q = q.filter(AnomalyEvent.detected_at >= since)

    rows = q.offset(offset).limit(limit).all()
    return [
        {
            "id": r.id,
            "commodity": r.commodity,
            "anomaly_type": r.anomaly_type,
            "severity": r.severity,
            "status": r.status,
            "detected_at": r.detected_at,
            "source_ids": r.source_ids,
            "metadata": r.metadata_json,
        }
        for r in rows
    ]


@router.get("/{anomaly_id}")
def get_anomaly(anomaly_id: int, db: Session = Depends(get_db)):
    row = db.query(AnomalyEvent).filter(AnomalyEvent.id == anomaly_id).first()
    if not row:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Anomaly not found")
    return {
        "id": row.id,
        "commodity": row.commodity,
        "anomaly_type": row.anomaly_type,
        "severity": row.severity,
        "status": row.status,
        "detected_at": row.detected_at,
        "source_ids": row.source_ids,
        "metadata": row.metadata_json,
    }
