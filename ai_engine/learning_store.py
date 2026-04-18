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

MAX_EVALUATIONS = 10   # cap same-commodity lookback
MAX_CROSS = 3          # max cross-commodity evaluations prepended as soft prior
MAX_INSIGHTS = 3       # max past insight bullets in prompt


def _fetch_eval_rows(commodity: str, anomaly_type: str, limit: int) -> list:
    """Raw DB fetch for evaluation rows, newest-first."""
    with get_session() as session:
        return session.execute(
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
            {"commodity": commodity, "anomaly_type": anomaly_type, "limit": limit},
        ).fetchall()


def _rows_to_evals(rows) -> tuple[list[dict], list[tuple[str, str]]]:
    """Parse raw DB rows into evaluation dicts and insight tuples."""
    evaluations: list[dict] = []
    insights: list[tuple[str, str]] = []
    for accuracy_json, failure_json, insight, future_adj in rows:
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


def _fetch_evaluations(commodity: str, anomaly_type: str) -> tuple[list[dict], list[tuple[str, str]]]:
    """
    Fetch evaluations for this commodity+anomaly_type pair, then prepend a
    small cross-commodity sample (same anomaly_type, other commodities) as a
    soft prior — weighted lower by position in the damped aggregate.

    The cross-commodity rows are prepended (treated as older data) so the
    same-commodity rows remain the most recent and thus most influential.
    """
    from shared.commodity_registry import COMMODITY_LIST

    # Same-commodity rows (newest-first), then reversed to oldest-first
    same_rows = list(reversed(_fetch_eval_rows(commodity, anomaly_type, MAX_EVALUATIONS)))

    # Cross-commodity rows — one fetch per other commodity, oldest-first merged
    cross_rows = []
    for other in COMMODITY_LIST:
        if other == commodity:
            continue
        cross_rows.extend(_fetch_eval_rows(other, anomaly_type, MAX_CROSS))
    cross_rows = cross_rows[:MAX_CROSS]  # cap total cross-commodity contribution

    # Prepend cross rows so they appear "older" (more damped) than same-commodity rows
    all_rows = cross_rows + same_rows

    evals, insights = _rows_to_evals(all_rows)
    # Only surface insights from same-commodity rows for the prompt
    _, same_insights = _rows_to_evals(same_rows)
    return evals, same_insights


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
