"""GET /predictions — signal alerts with prediction and evaluation data."""

import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.dependencies import get_db
from shared.models import PredictionEvaluation, SignalAlert

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("")
def list_predictions(
    commodity: Optional[str] = Query(None),
    prediction_type: Optional[str] = Query(None, description="directional | no_signal"),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    limit: int = Query(20, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """
    Returns signal_alerts that have a prediction, joined with prediction_evaluations.
    """
    q = (
        db.query(SignalAlert)
        .filter(SignalAlert.prediction_json.isnot(None))
        .order_by(SignalAlert.created_at.desc())
    )
    if commodity:
        q = q.filter(SignalAlert.commodity == commodity)
    if prediction_type:
        q = q.filter(SignalAlert.prediction_type == prediction_type)
    if date_from:
        q = q.filter(SignalAlert.created_at >= date_from)
    if date_to:
        q = q.filter(SignalAlert.created_at <= date_to)

    alerts = q.offset(offset).limit(limit).all()

    results = []
    for alert in alerts:
        # Fetch linked anomaly for anomaly_type
        anomaly_type = None
        if alert.anomaly_event:
            anomaly_type = alert.anomaly_event.anomaly_type

        # Parse prediction JSON
        try:
            pred_data = json.loads(alert.prediction_json) if alert.prediction_json else {}
        except (json.JSONDecodeError, TypeError):
            pred_data = {}

        # Fetch evaluation if present
        evaluation_out = None
        pe: Optional[PredictionEvaluation] = (
            db.query(PredictionEvaluation)
            .filter(PredictionEvaluation.signal_alert_id == alert.id)
            .first()
        )
        if pe:
            try:
                accuracy = json.loads(pe.prediction_accuracy_json) if pe.prediction_accuracy_json else {}
            except (json.JSONDecodeError, TypeError):
                accuracy = {}
            try:
                causal = json.loads(pe.causal_analysis_json) if pe.causal_analysis_json else {}
            except (json.JSONDecodeError, TypeError):
                causal = {}
            try:
                failure_modes = json.loads(pe.failure_modes_json) if pe.failure_modes_json else []
            except (json.JSONDecodeError, TypeError):
                failure_modes = []
            try:
                learning = json.loads(pe.learning_update_json) if pe.learning_update_json else {}
            except (json.JSONDecodeError, TypeError):
                learning = {}

            evaluation_out = {
                "prediction_accuracy": accuracy,
                "causal_analysis": causal,
                "failure_modes": failure_modes,
                "learning_update": learning,
                "actual_price_change_pct": pe.actual_price_change_pct,
                "overall_score": pe.overall_score,
                "evaluated_at": str(pe.created_at),
            }

        results.append({
            "signal_alert_id": alert.id,
            "commodity": alert.commodity,
            "anomaly_type": anomaly_type,
            "alert_type": alert.alert_type,
            "price_at_alert": alert.price_at_alert,
            "price_1m": alert.price_1m,
            "prediction_type": alert.prediction_type,
            "prediction_confidence": alert.prediction_confidence,
            "prediction_json": pred_data,
            "evaluation": evaluation_out,
            "overall_score": pe.overall_score if pe else None,
            "created_at": str(alert.created_at),
        })

    return results
