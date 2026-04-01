"""GET /reports — query causality reports."""

import json
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.dependencies import get_db
from shared.models import CausalityReport

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("")
def list_reports(
    commodity: Optional[str] = Query(None),
    cause_category: Optional[str] = Query(None),
    limit: int = Query(20, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    q = db.query(CausalityReport).order_by(CausalityReport.created_at.desc())
    if commodity:
        q = q.filter(CausalityReport.commodity == commodity)
    if cause_category:
        q = q.filter(CausalityReport.cause_category == cause_category)

    rows = q.offset(offset).limit(limit).all()
    return [_format_report(r) for r in rows]


@router.get("/{report_id}")
def get_report(report_id: int, db: Session = Depends(get_db)):
    row = db.query(CausalityReport).filter(CausalityReport.id == report_id).first()
    if not row:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Report not found")
    return _format_report(row)


def _format_report(r: CausalityReport) -> dict:
    try:
        report_data = json.loads(r.report_json)
    except (json.JSONDecodeError, TypeError):
        report_data = {}
    return {
        "id": r.id,
        "signal_alert_id": r.signal_alert_id,
        "commodity": r.commodity,
        "cause_category": r.cause_category,
        "confidence_score": r.confidence_score,
        "price_impact_pct": r.price_impact_pct,
        "created_at": r.created_at,
        "cause": report_data.get("cause"),
        "mechanism": report_data.get("mechanism"),
        "summary": report_data.get("summary"),
        "supporting_signals": report_data.get("supporting_signals", []),
        "historical_precedents": report_data.get("historical_precedents", []),
        "full_report": report_data,
    }
