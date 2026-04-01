"""
Learning store: reads learning_updates + prediction_evaluations from DB
and formats them into a {learning_section} string for injection into
PREDICTION_PROMPT.

Called at prediction time — not at evaluation time — so there is no
circular dependency.
"""

import json
from typing import Optional

from sqlalchemy import text

from ai_engine.feedback_controller import (
    ErrorSignal,
    ControlAdjustments,
    compute_error_signal,
    compute_control_adjustments,
)
from shared.db import get_session
from shared.logger import get_logger

logger = get_logger(__name__)

MAX_EVALUATIONS = 10   # cap lookback to avoid prompt bloat
MAX_INSIGHTS = 3       # max past insight bullets in prompt


def _fetch_evaluations(commodity: str, anomaly_type: str) -> list[dict]:
    """
    Fetch up to MAX_EVALUATIONS most recent evaluations for this
    commodity + anomaly_type pair, ordered oldest-first (for damping).
    """
    with get_session() as session:
        rows = session.execute(
            text("""
                SELECT
                    pe.prediction_accuracy_json,
                    pe.failure_modes_json,
                    lu.insight,
                    lu.future_adjustment
                FROM learning_updates lu
                JOIN prediction_evaluations pe
                    ON pe.id = lu.prediction_evaluation_id
                WHERE lu.commodity = :commodity
                  AND lu.anomaly_type = :anomaly_type
                ORDER BY lu.created_at DESC
                LIMIT :limit
            """),
            {
                "commodity": commodity,
                "anomaly_type": anomaly_type,
                "limit": MAX_EVALUATIONS,
            },
        ).fetchall()

    # Reverse so oldest is first (required by damped_aggregate)
    rows = list(reversed(rows))

    evaluations: list[dict] = []
    insights: list[tuple[str, str]] = []  # (insight, future_adjustment)

    for row in rows:
        accuracy_json, failure_json, insight, future_adj = row
        try:
            accuracy = json.loads(accuracy_json or "{}")
        except json.JSONDecodeError:
            accuracy = {}
        try:
            failure_modes = json.loads(failure_json or "[]")
        except json.JSONDecodeError:
            failure_modes = []

        evaluations.append({
            "direction_correct": accuracy.get("direction_correct"),
            "magnitude_error": float(accuracy.get("magnitude_error") or 0.0),
            "volatility_correct": accuracy.get("volatility_correct"),
            "failure_modes": failure_modes,
        })

        if insight:
            insights.append((insight, future_adj or ""))

    return evaluations, insights


def get_learning_context(commodity: str, anomaly_type: str) -> str:
    """
    Build the {learning_section} text for PREDICTION_PROMPT injection.

    Returns a compact, human-readable string.  If no evaluations exist,
    returns an empty string so the prompt is unchanged.
    """
    try:
        evaluations, insights = _fetch_evaluations(commodity, anomaly_type)
    except Exception as exc:
        logger.warning("learning_store_fetch_failed", commodity=commodity,
                       anomaly_type=anomaly_type, error=str(exc))
        return ""

    if not evaluations:
        return ""

    signal: ErrorSignal = compute_error_signal(evaluations, commodity, anomaly_type)
    adj: ControlAdjustments = compute_control_adjustments(signal)

    lines = []

    # Header + error signal summary
    lines.append(
        f"Based on {signal.n_evaluations} past evaluation(s) for "
        f"{commodity.upper()} / {anomaly_type}:"
    )
    lines.append(
        f"  Direction error rate: {signal.damped_direction_error:.0%}  |  "
        f"Magnitude error: {signal.damped_magnitude_error:.1f} pp  |  "
        f"Composite error: {signal.e_total:.2f}"
    )

    # Past insights (most recent MAX_INSIGHTS, oldest available)
    if insights:
        lines.append("\nKey lessons from past predictions:")
        for insight, future_adj in insights[-MAX_INSIGHTS:]:
            lines.append(f"  * {insight}")
            if future_adj:
                lines.append(f"    -> Adjustment: {future_adj}")

    # Control adjustments
    if not adj.is_empty():
        lines.append("\nRequired adjustments for this prediction:")
        lines.append(adj.to_text())

    result = "\n".join(lines)
    logger.info(
        "learning_context_built",
        commodity=commodity,
        anomaly_type=anomaly_type,
        n_evaluations=signal.n_evaluations,
        e_total=signal.e_total,
    )
    return result
