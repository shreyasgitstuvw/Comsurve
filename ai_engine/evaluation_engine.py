"""
Post-Event Prediction Evaluation Engine.

For each CausalityReport where the linked SignalAlert has a non-null prediction_json
AND no PredictionEvaluation exists yet:
  1. Gather actual price outcomes (price_at_alert, 1w, 2w, 1m)
  2. Compute actual_price_change_pct
  3. Call Gemini with evaluation prompt
  4. Compute overall_score from accuracy components
  5. Store PredictionEvaluation + LearningUpdate rows
"""

import json
from datetime import datetime

from sqlalchemy import text

from ai_engine.llama_client import get_evaluation_client
from shared.db import get_session
from shared.logger import get_logger
from shared.models import LearningUpdate, PredictionEvaluation

logger = get_logger(__name__)

EVALUATION_PROMPT = """You are evaluating the accuracy of a commodity price prediction against actual outcomes.

PRE-EVENT PREDICTION (made {days_before} days ago):
{prediction_json}

ACTUAL PRICE OUTCOMES:
- Price at alert:    {price_at_alert}
- Price at +1 week:  {price_1w}  ({pct_1w:+.2f}% vs alert)
- Price at +2 weeks: {price_2w}  ({pct_2w:+.2f}% vs alert)
- Price at +1 month: {price_1m}  ({pct_1m:+.2f}% vs alert)
- Actual 1-month change: {actual_pct:+.2f}%

POST-EVENT CAUSAL ANALYSIS (what actually drove the price):
{causal_report_summary}

EVALUATION RULES:
- direction_correct: true if predicted direction matches actual; null if prediction_type was magnitude_only or no_signal
- magnitude_error: |predicted_midpoint_pct - actual_pct| (absolute percentage points)
- confidence_validity: "well_calibrated" if |confidence - accuracy| < 0.2, "overconfident" if confidence >> accuracy, "underconfident" if confidence << accuracy
- DO NOT suggest modifying global feature weights or model parameters
- Learning must be event-specific reasoning adjustments only

Return ONLY valid JSON:
{{
  "prediction_accuracy": {{
    "direction_correct": <true|false|null>,
    "magnitude_error": <float>,
    "volatility_correct": <true|false>,
    "confidence_validity": "<well_calibrated|overconfident|underconfident>"
  }},
  "causal_analysis": {{
    "correct_drivers": ["<drivers the prediction correctly identified>"],
    "missed_drivers": ["<important drivers the prediction missed>"],
    "overestimated_drivers": ["<drivers that were over-weighted in the prediction>"]
  }},
  "failure_modes": ["<specific reasoning gap 1>", "<specific reasoning gap 2>"],
  "learning_update": {{
    "insight": "<specific lesson about signal interpretation for {commodity}>",
    "future_adjustment": "<concrete change to scenario construction or driver identification logic>",
    "affected_signal_types": ["<signal_type_1>", "<signal_type_2>"]
  }}
}}"""


def _pct_change(base: float | None, current: float | None) -> float:
    """Return percentage change; 0.0 if either value is None or base is zero."""
    if base and current and base != 0:
        return (current - base) / base * 100
    return 0.0


def _compute_overall_score(accuracy: dict) -> float:
    """
    Composite 0.0-1.0 score:
      direction:   0.4 if correct, 0.2 if null, 0.0 if false
      magnitude:   max(0, 0.4 - magnitude_error * 0.04)
      calibration: 0.2 if well_calibrated, 0.1 if underconfident, 0.0 if overconfident
    """
    direction_correct = accuracy.get("direction_correct")
    if direction_correct is True:
        direction_score = 0.4
    elif direction_correct is None:
        direction_score = 0.2
    else:
        direction_score = 0.0

    mag_error = float(accuracy.get("magnitude_error") or 0.0)
    magnitude_score = max(0.0, 0.4 - mag_error * 0.04)

    calibration = accuracy.get("confidence_validity", "overconfident")
    if calibration == "well_calibrated":
        calibration_score = 0.2
    elif calibration == "underconfident":
        calibration_score = 0.1
    else:
        calibration_score = 0.0

    return round(direction_score + magnitude_score + calibration_score, 4)


def _days_between(alert_created_at, report_created_at) -> int:
    """Compute calendar days between alert creation and report creation."""
    try:
        if isinstance(alert_created_at, str):
            alert_created_at = datetime.fromisoformat(alert_created_at)
        if isinstance(report_created_at, str):
            report_created_at = datetime.fromisoformat(report_created_at)
        delta = report_created_at - alert_created_at
        return max(0, delta.days)
    except Exception:
        return 0


def _extract_causal_summary(report_json_str: str) -> str:
    """Pull a human-readable summary from the causality report JSON."""
    try:
        report = json.loads(report_json_str)
        parts = []
        if report.get("summary"):
            parts.append(report["summary"])
        if report.get("cause"):
            parts.append(f"Primary cause: {report['cause']}")
        if report.get("mechanism"):
            parts.append(f"Mechanism: {report['mechanism']}")
        sigs = report.get("supporting_signals", [])
        if sigs:
            parts.append(f"Supporting signals: {', '.join(sigs)}")
        return "\n".join(parts) if parts else "No causal summary available."
    except (json.JSONDecodeError, TypeError):
        return "Causal report unavailable."


def run_evaluation_engine() -> dict:
    """
    Evaluate all CausalityReports where the linked SignalAlert has a prediction
    but no PredictionEvaluation yet.
    Returns {"evaluations_generated": N, "failed": K}
    """
    client = get_evaluation_client()
    evaluations_generated = 0
    failed = 0

    with get_session() as session:
        rows = session.execute(
            text("""
                SELECT cr.id, cr.signal_alert_id, cr.commodity,
                       cr.report_json, cr.created_at AS report_created_at,
                       sa.prediction_json, sa.prediction_type, sa.prediction_confidence,
                       sa.price_at_alert, sa.price_1w, sa.price_2w, sa.price_1m,
                       sa.created_at AS alert_created_at,
                       ae.anomaly_type
                FROM causality_reports cr
                JOIN signal_alerts sa ON sa.id = cr.signal_alert_id
                JOIN anomaly_events ae ON ae.id = sa.anomaly_event_id
                LEFT JOIN prediction_evaluations pe ON pe.signal_alert_id = cr.signal_alert_id
                WHERE sa.prediction_json IS NOT NULL
                  AND pe.id IS NULL
                ORDER BY cr.created_at ASC
                LIMIT 20
            """),
        ).fetchall()

    logger.info("evaluation_engine_start", candidates=len(rows))

    for row in rows:
        (cr_id, alert_id, commodity,
         report_json_str, report_created_at,
         prediction_json_str, prediction_type, prediction_confidence,
         p_alert, p1w, p2w, p1m,
         alert_created_at,
         anomaly_type) = row

        try:
            # Compute price change percentages
            actual_pct = _pct_change(p_alert, p1m)
            pct_1w = _pct_change(p_alert, p1w)
            pct_2w = _pct_change(p_alert, p2w)
            pct_1m = actual_pct

            days_before = _days_between(alert_created_at, report_created_at)
            causal_summary = _extract_causal_summary(report_json_str)

            prompt = EVALUATION_PROMPT.format(
                days_before=days_before,
                prediction_json=prediction_json_str or "{}",
                price_at_alert=p_alert if p_alert is not None else "N/A",
                price_1w=p1w if p1w is not None else "N/A",
                pct_1w=pct_1w,
                price_2w=p2w if p2w is not None else "N/A",
                pct_2w=pct_2w,
                price_1m=p1m if p1m is not None else "N/A",
                pct_1m=pct_1m,
                actual_pct=actual_pct,
                causal_report_summary=causal_summary,
                commodity=commodity,
            )

            raw_response = client.generate_text(prompt)
            eval_dict = json.loads(raw_response)

            accuracy = eval_dict.get("prediction_accuracy", {})
            overall_score = _compute_overall_score(accuracy)

            learning = eval_dict.get("learning_update", {})

            with get_session() as session:
                pe = PredictionEvaluation(
                    signal_alert_id=alert_id,
                    commodity=commodity,
                    actual_price_change_pct=round(actual_pct, 4) if p_alert and p1m else None,
                    prediction_accuracy_json=json.dumps(accuracy),
                    causal_analysis_json=json.dumps(eval_dict.get("causal_analysis", {})),
                    failure_modes_json=json.dumps(eval_dict.get("failure_modes", [])),
                    learning_update_json=json.dumps(learning),
                    overall_score=overall_score,
                    created_at=datetime.utcnow(),
                )
                session.add(pe)
                session.flush()  # get pe.id

                lu = LearningUpdate(
                    prediction_evaluation_id=pe.id,
                    commodity=commodity,
                    anomaly_type=anomaly_type,
                    insight=learning.get("insight"),
                    future_adjustment=learning.get("future_adjustment"),
                    affected_signal_types=json.dumps(
                        learning.get("affected_signal_types", [])
                    ),
                    created_at=datetime.utcnow(),
                )
                session.add(lu)

            logger.info(
                "evaluation_generated",
                alert_id=alert_id,
                commodity=commodity,
                overall_score=overall_score,
                direction_correct=accuracy.get("direction_correct"),
            )
            evaluations_generated += 1

        except Exception as exc:
            logger.error(
                "evaluation_failed",
                alert_id=alert_id,
                commodity=commodity,
                error=str(exc),
            )
            failed += 1

    summary = {"evaluations_generated": evaluations_generated, "failed": failed}
    logger.info("evaluation_engine_complete", **summary)
    return summary


if __name__ == "__main__":
    from shared.db import init_db
    init_db()
    result = run_evaluation_engine()
    print(result)
