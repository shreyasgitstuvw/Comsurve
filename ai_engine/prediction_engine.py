"""
Pre-Event Scenario Prediction Engine — generates probabilistic price scenarios
for new SignalAlerts before their monitoring windows close.

For each SignalAlert that has no prediction_json yet:
  1. Gather anomaly details + historical analogs + context payload from embeddings_cache
  2. Build a structured Gemini prompt
  3. If confidence_score >= 0.6: store full prediction; else store prediction_type='no_signal'
  4. Update signal_alerts.prediction_json / prediction_type / prediction_confidence
"""

import json
from datetime import datetime

from sqlalchemy import text

from ai_engine.gemini_client import GeminiClient
from ai_engine.learning_store import get_learning_context
from shared.db import get_session
from shared.logger import get_logger

logger = get_logger(__name__)

PREDICTION_PROMPT = """You are a commodity market analyst specialising in supply chain disruption signals.

Analyse the pre-event signals below and generate a structured probabilistic scenario prediction.

STRICT RULES:
1. Only generate a substantive prediction if your overall confidence_score >= 0.6
2. If confidence < 0.6, set prediction_type to "no_signal" and return empty predicted_outcomes []
3. Prefer a magnitude-only prediction (direction_confidence: "low") over a low-confidence directional call
4. List only specific, observable supply/demand/geopolitical drivers — avoid generic placeholders

COMMODITY: {commodity}
ANOMALY TYPE: {anomaly_type} (severity: {severity:.2f})
DETECTED: {detected_at}

SIGNAL CONTEXT:
{signal_context}

HISTORICAL ANALOGS (similar past events):
{analogs_section}

FEEDBACK FROM PAST PREDICTIONS (damped learning signal):
{learning_section}

CURRENT PRICE: {current_price}

Return ONLY valid JSON matching this schema exactly:
{{
  "event_id": "{event_id}",
  "commodity": "{commodity}",
  "signal_summary": "<2-3 sentences describing what the signals collectively indicate>",
  "predicted_outcomes": [
    {{
      "scenario": "<Bullish|Bearish|Volatility Expected|Neutral>",
      "price_move": "<e.g. '+3% to +7%' or '±4% to ±8%'>",
      "probability": <0.0-1.0>,
      "direction_confidence": "<high|medium|low>",
      "time_horizon": "<1w|2w|1m>"
    }}
  ],
  "confidence_score": <float 0.0-1.0>,
  "prediction_type": "<directional|magnitude_only|no_signal>",
  "drivers": ["<specific driver 1>", "<specific driver 2>"],
  "historical_analogs": [<list of analog anomaly_event IDs as integers>]
}}"""


def _build_analogs_section(correlated_ids: list[int], similarity_scores: list[float]) -> str:
    """Fetch details of correlated historical anomalies to include in prompt."""
    if not correlated_ids:
        return "No similar historical events found (novel event)."

    lines = []
    with get_session() as session:
        for aid, score in zip(correlated_ids[:5], similarity_scores[:5]):
            row = session.execute(
                text("""
                    SELECT commodity, anomaly_type, severity, detected_at, metadata_json
                    FROM anomaly_events WHERE id = :id
                """),
                {"id": aid},
            ).fetchone()
            if not row:
                continue
            try:
                meta = json.loads(row[4] or "{}")
                pct = meta.get("pct_change", "N/A")
                if isinstance(pct, float):
                    pct = f"{pct * 100:+.2f}%"
            except (json.JSONDecodeError, KeyError):
                pct = "N/A"
            lines.append(
                f"  Event {aid} (similarity {score:.3f}): {row[1]} on {row[0].upper()} "
                f"at {str(row[3])[:10]}, severity={row[2]:.2f}, price_change={pct}"
            )
    return "\n".join(lines) if lines else "Historical analog data unavailable."


def _get_signal_context(anomaly_event_id: int) -> str:
    """
    Retrieve the context_payload stored by the embedding generator for this anomaly.
    Falls back to a minimal description if not available.
    """
    with get_session() as session:
        row = session.execute(
            text("""
                SELECT ec.context_payload, ae.anomaly_type, ae.commodity, ae.metadata_json
                FROM embeddings_cache ec
                JOIN anomaly_events ae ON ae.id = ec.anomaly_event_id
                WHERE ec.anomaly_event_id = :id
            """),
            {"id": anomaly_event_id},
        ).fetchone()

    if not row:
        return "Context payload unavailable."

    context_payload, anomaly_type, commodity, metadata_json = row
    if context_payload:
        return context_payload[:2000]  # cap at 2000 chars

    # Fallback: build minimal context from metadata
    try:
        meta = json.loads(metadata_json or "{}")
    except json.JSONDecodeError:
        meta = {}
    parts = [f"Anomaly type: {anomaly_type}", f"Commodity: {commodity}"]
    for k, v in meta.items():
        if not k.startswith("_"):
            parts.append(f"{k}: {v}")
    return "\n".join(parts)


def _compute_prediction_type(prediction_dict: dict) -> str:
    """Extract prediction_type from Gemini response, defaulting safely."""
    ptype = prediction_dict.get("prediction_type", "no_signal")
    valid = {"directional", "magnitude_only", "no_signal"}
    return ptype if ptype in valid else "no_signal"


def run_prediction_engine() -> dict:
    """
    Process all SignalAlerts that are missing prediction_json.
    Returns {"predictions_generated": N, "no_signal": M, "failed": K}
    """
    client = GeminiClient()
    predictions_generated = 0
    no_signal = 0
    failed = 0

    # Fetch all signal_alerts without a prediction yet
    with get_session() as session:
        rows = session.execute(
            text("""
                SELECT sa.id, sa.commodity, sa.alert_type,
                       sa.correlated_anomaly_ids, sa.similarity_scores,
                       sa.price_at_alert, sa.anomaly_event_id,
                       ae.anomaly_type, ae.severity, ae.detected_at
                FROM signal_alerts sa
                JOIN anomaly_events ae ON ae.id = sa.anomaly_event_id
                WHERE sa.prediction_json IS NULL
                ORDER BY sa.created_at DESC
                LIMIT 50
            """),
        ).fetchall()

    logger.info("prediction_engine_start", candidates=len(rows))

    for row in rows:
        (alert_id, commodity, alert_type,
         corr_ids_json, scores_json,
         price_at_alert, anomaly_event_id,
         anomaly_type, severity, detected_at) = row

        try:
            corr_ids = json.loads(corr_ids_json or "[]")
            scores = json.loads(scores_json or "[]")
        except json.JSONDecodeError:
            corr_ids, scores = [], []

        signal_context = _get_signal_context(anomaly_event_id)
        analogs_section = _build_analogs_section(corr_ids, scores)
        current_price = str(price_at_alert) if price_at_alert is not None else "N/A"
        learning_section = get_learning_context(commodity, anomaly_type)
        if not learning_section:
            learning_section = "No prior evaluations available for this signal type."

        prompt = PREDICTION_PROMPT.format(
            commodity=commodity,
            anomaly_type=anomaly_type,
            severity=float(severity),
            detected_at=str(detected_at),
            signal_context=signal_context,
            analogs_section=analogs_section,
            learning_section=learning_section,
            current_price=current_price,
            event_id=str(anomaly_event_id),
        )

        try:
            raw_response = client.generate_text(prompt)
            pred_dict = json.loads(raw_response)

            confidence = float(pred_dict.get("confidence_score", 0.0))
            prediction_type = _compute_prediction_type(pred_dict)

            # Enforce no_signal rule if confidence below threshold
            if confidence < 0.6:
                prediction_type = "no_signal"
                pred_dict["prediction_type"] = "no_signal"
                pred_dict["predicted_outcomes"] = []

            pred_dict.setdefault("event_id", str(anomaly_event_id))
            pred_dict.setdefault("commodity", commodity)

            # Persist to signal_alerts
            with get_session() as session:
                session.execute(
                    text("""
                        UPDATE signal_alerts
                        SET prediction_json = :pjson,
                            prediction_type = :ptype,
                            prediction_confidence = :pconf
                        WHERE id = :id
                    """),
                    {
                        "pjson": json.dumps(pred_dict),
                        "ptype": prediction_type,
                        "pconf": confidence,
                        "id": alert_id,
                    },
                )

            if prediction_type == "no_signal":
                no_signal += 1
                logger.info(
                    "prediction_no_signal",
                    alert_id=alert_id,
                    commodity=commodity,
                    confidence=confidence,
                )
            else:
                predictions_generated += 1
                logger.info(
                    "prediction_generated",
                    alert_id=alert_id,
                    commodity=commodity,
                    prediction_type=prediction_type,
                    confidence=confidence,
                )

        except Exception as exc:
            logger.error("prediction_failed", alert_id=alert_id, commodity=commodity, error=str(exc))
            failed += 1

    summary = {
        "predictions_generated": predictions_generated,
        "no_signal": no_signal,
        "failed": failed,
    }
    logger.info("prediction_engine_complete", **summary)
    return summary


if __name__ == "__main__":
    from shared.db import init_db
    init_db()
    result = run_prediction_engine()
    print(result)
