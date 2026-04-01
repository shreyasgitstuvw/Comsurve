"""GET /evaluations — prediction evaluations and learning updates."""

import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from api.dependencies import get_db
from shared.models import LearningUpdate, PredictionEvaluation

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.get("")
def list_evaluations(
    commodity: Optional[str] = Query(None),
    score_min: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum overall_score"),
    date_from: Optional[datetime] = Query(None),
    date_to: Optional[datetime] = Query(None),
    limit: int = Query(20, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
):
    """
    Returns PredictionEvaluations with their LearningUpdates, ordered newest first.
    """
    q = db.query(PredictionEvaluation).order_by(PredictionEvaluation.created_at.desc())
    if commodity:
        q = q.filter(PredictionEvaluation.commodity == commodity)
    if score_min is not None:
        q = q.filter(PredictionEvaluation.overall_score >= score_min)
    if date_from:
        q = q.filter(PredictionEvaluation.created_at >= date_from)
    if date_to:
        q = q.filter(PredictionEvaluation.created_at <= date_to)

    evaluations = q.offset(offset).limit(limit).all()

    results = []
    for pe in evaluations:
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
            learning_update = json.loads(pe.learning_update_json) if pe.learning_update_json else {}
        except (json.JSONDecodeError, TypeError):
            learning_update = {}

        # Fetch linked learning updates
        lu_rows = (
            db.query(LearningUpdate)
            .filter(LearningUpdate.prediction_evaluation_id == pe.id)
            .all()
        )
        learning_updates = [
            {
                "id": lu.id,
                "commodity": lu.commodity,
                "anomaly_type": lu.anomaly_type,
                "insight": lu.insight,
                "future_adjustment": lu.future_adjustment,
                "affected_signal_types": json.loads(lu.affected_signal_types or "[]"),
                "created_at": str(lu.created_at),
            }
            for lu in lu_rows
        ]

        results.append({
            "id": pe.id,
            "signal_alert_id": pe.signal_alert_id,
            "commodity": pe.commodity,
            "overall_score": pe.overall_score,
            "actual_price_change_pct": pe.actual_price_change_pct,
            "prediction_accuracy": accuracy,
            "causal_analysis": causal,
            "failure_modes": failure_modes,
            "learning_update": learning_update,
            "learning_updates": learning_updates,
            "created_at": str(pe.created_at),
        })

    return results


@router.get("/learning")
def list_learning_updates(
    commodity: Optional[str] = Query(None),
    anomaly_type: Optional[str] = Query(None),
    limit: int = Query(50, le=500),
    db: Session = Depends(get_db),
):
    """
    Returns all LearningUpdates — the distilled lessons from past prediction errors.
    Useful for understanding how the engine's signal interpretation evolves over time.
    """
    q = db.query(LearningUpdate).order_by(LearningUpdate.created_at.desc())
    if commodity:
        q = q.filter(LearningUpdate.commodity == commodity)
    if anomaly_type:
        q = q.filter(LearningUpdate.anomaly_type == anomaly_type)

    rows = q.limit(limit).all()

    return [
        {
            "id": lu.id,
            "prediction_evaluation_id": lu.prediction_evaluation_id,
            "commodity": lu.commodity,
            "anomaly_type": lu.anomaly_type,
            "insight": lu.insight,
            "future_adjustment": lu.future_adjustment,
            "affected_signal_types": json.loads(lu.affected_signal_types or "[]"),
            "created_at": str(lu.created_at),
        }
        for lu in rows
    ]
