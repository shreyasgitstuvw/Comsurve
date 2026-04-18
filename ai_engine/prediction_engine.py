"""
Pre-Event Scenario Prediction Engine — generates probabilistic price scenarios
for new SignalAlerts before their monitoring windows close.

For each SignalAlert that has no prediction_json yet:
  1. Gather anomaly details + 7-day trajectory + 30-day market context + analogs + feedback
  2. Build a structured Gemini prompt with chain-of-thought reasoning step
  3. Apply confidence calibration (Platt scaling or shrinkage prior)
  4. If calibrated_confidence >= 0.6: store full prediction; else store prediction_type='no_signal'
  5. Update signal_alerts.prediction_json / prediction_type / prediction_confidence
"""

import json
from datetime import datetime, timedelta

from sqlalchemy import text

from ai_engine.confidence_calibrator import calibrate_confidence
from ai_engine.gemini_client import GeminiClient
from ai_engine.learning_store import get_learning_context
from shared.db import get_session
from shared.logger import get_logger

logger = get_logger(__name__)

# Analogs with cosine similarity below this are annotated as weak in the prompt
WEAK_ANALOG_THRESHOLD = 0.70

# Per-anomaly-type guidance on which time horizons are most predictable
HORIZON_GUIDANCE: dict[str, str] = {
    "ais_vessel_drop":       "AIS ship tracking best predicts 1–2 week outcomes (port logistics have a defined lag). Weight 1w and 2w outcomes higher; treat 1m as speculative.",
    "ais_cluster":           "AIS cluster signals best predict 2-week outcomes. Weight 2w highest.",
    "sentiment_shift":       "Sentiment shifts are typically mean-reverting. Weight 1w outcome highest; discount 2w and 1m.",
    "price_spike":           "Price spike anomalies best predict 1–2 week reversions. 1m persistence depends on fundamental driver continuity.",
    "price_historical":      "Historical price anomalies from FRED/EIA reflect structural shifts — weight 1m outcomes higher.",
    "rail_corridor_gap":     "Rail logistics gaps resolve in 2–3 weeks. Weight 2w outcomes highest.",
    "satellite_backscatter": "Satellite backscatter changes take 2–4 weeks to translate to prices. Weight 2w and 1m.",
}
_HORIZON_DEFAULT = "Weight all three time horizons equally unless signal characteristics suggest otherwise."


PREDICTION_PROMPT = """You are a commodity market analyst specialising in supply chain disruption signals.

Analyse the pre-event signals below and generate a structured probabilistic scenario prediction.

STRICT RULES:
1. Only generate a substantive prediction if your overall confidence_score >= 0.6
2. If confidence < 0.6, set prediction_type to "no_signal" and return empty predicted_outcomes []
3. Prefer a magnitude-only prediction (direction_confidence: "low") over a low-confidence directional call
4. List only specific, observable supply/demand/geopolitical drivers — avoid generic placeholders
5. "invalidating_conditions" must be concrete, observable events that would falsify this prediction

COMMODITY: {commodity}
ANOMALY TYPE: {anomaly_type}
DETECTED: {detected_at}

SIGNAL CONTEXT:
{signal_context}

ANOMALY TRAJECTORY (7-day pre-event trend):
{anomaly_trajectory}

RECENT MARKET CONTEXT (30-day price trend):
{market_context}

HISTORICAL ANALOGS (similar past events):
{analogs_section}

FEEDBACK FROM PAST PREDICTIONS (damped learning signal):
{learning_section}

CURRENT PRICE: {current_price}

HORIZON-SPECIFIC GUIDANCE:
{horizon_guidance}

STEP 1 — BRIEF REASONING (write 2-3 sentences, then give your JSON):
Consider: (a) How strong are the supply-side signals vs. demand-side? (b) Do the historical analogs support a directional call or only magnitude uncertainty? (c) What is the dominant source of uncertainty?

STEP 2 — OUTPUT (valid JSON only, immediately after your reasoning):
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
  "historical_analogs": [<list of analog anomaly_event IDs as integers>],
  "invalidating_conditions": ["<observable condition that would falsify this prediction>"]
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Context builders
# ─────────────────────────────────────────────────────────────────────────────

def _get_signal_context(anomaly_event_id: int) -> str:
    """Retrieve context_payload from embeddings_cache (up to 6000 chars)."""
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
        return context_payload[:6000]  # raised from 2000

    try:
        meta = json.loads(metadata_json or "{}")
    except json.JSONDecodeError:
        meta = {}
    parts = [f"Anomaly type: {anomaly_type}", f"Commodity: {commodity}"]
    for k, v in meta.items():
        if not k.startswith("_"):
            parts.append(f"{k}: {v}")
    return "\n".join(parts)


def _get_anomaly_trajectory(commodity: str, detected_at) -> str:
    """
    Fetch 7-day pre-event trend (price, z_score, sentiment) so the model can
    distinguish a spike-on-uptrend from a mean-reversion spike.
    """
    if isinstance(detected_at, str):
        try:
            detected_at = datetime.fromisoformat(detected_at)
        except ValueError:
            return "Trajectory data unavailable."

    window_start = detected_at - timedelta(days=7)

    with get_session() as session:
        rows = session.execute(
            text("""
                SELECT ri.timestamp, pf.feature_type, pf.value
                FROM processed_features pf
                JOIN raw_ingestion ri ON pf.raw_ingestion_id = ri.id
                WHERE ri.commodity = :commodity
                  AND ri.timestamp BETWEEN :start AND :end
                  AND pf.feature_type IN ('price', 'z_score', 'sentiment')
                ORDER BY ri.timestamp ASC
            """),
            {"commodity": commodity, "start": window_start, "end": detected_at},
        ).fetchall()

    if not rows:
        return "No pre-event data available within 7 days."

    by_type: dict[str, list] = {}
    for ts, ftype, val in rows:
        try:
            by_type.setdefault(ftype, []).append((str(ts)[:10], round(float(val), 4)))
        except (TypeError, ValueError):
            continue

    lines = ["7-day pre-event trend (oldest → newest):"]
    for ftype, points in by_type.items():
        sampled = points[::max(1, len(points) // 7)][-7:]
        pts_str = ", ".join(f"{d}: {v}" for d, v in sampled)
        lines.append(f"  {ftype}: {pts_str}")
    return "\n".join(lines)


def _get_market_context(commodity: str, detected_at) -> str:
    """Fetch 30-day price trend to give macro price momentum context."""
    if isinstance(detected_at, str):
        try:
            detected_at = datetime.fromisoformat(detected_at)
        except ValueError:
            return "Market context unavailable."

    window_start = detected_at - timedelta(days=30)

    with get_session() as session:
        rows = session.execute(
            text("""
                SELECT ri.timestamp, pf.value
                FROM processed_features pf
                JOIN raw_ingestion ri ON pf.raw_ingestion_id = ri.id
                WHERE ri.commodity = :commodity
                  AND ri.timestamp BETWEEN :start AND :end
                  AND pf.feature_type = 'price'
                ORDER BY ri.timestamp ASC
            """),
            {"commodity": commodity, "start": window_start, "end": detected_at},
        ).fetchall()

    if not rows:
        return "No 30-day price data available."

    prices = []
    for ts, val in rows:
        try:
            prices.append((str(ts)[:10], round(float(val), 4)))
        except (TypeError, ValueError):
            continue

    if len(prices) < 2:
        return f"Insufficient price history. Latest: {prices[0][1] if prices else 'N/A'}"

    first, last = prices[0][1], prices[-1][1]
    pct = (last - first) / first * 100 if first else 0.0
    trend = "upward" if pct > 2 else ("downward" if pct < -2 else "sideways")

    step = max(1, len(prices) // 8)
    sampled = prices[::step]
    pts_str = ", ".join(f"{d}: {v}" for d, v in sampled)
    return (
        f"30-day trend: {trend} ({pct:+.1f}% from {first} to {last})\n"
        f"  Sampled prices: {pts_str}"
    )


def _build_analogs_section(correlated_ids: list[int], similarity_scores: list[float]) -> str:
    """
    Fetch details of correlated historical anomalies (up to 8).
    Analogs with similarity < WEAK_ANALOG_THRESHOLD are annotated as weak.
    Cross-commodity analogs are naturally labelled by their commodity field.
    """
    if not correlated_ids:
        return "No similar historical events found (novel event)."

    lines = []
    with get_session() as session:
        for aid, score in zip(correlated_ids[:8], similarity_scores[:8]):
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
            annotation = " (weak analog)" if score < WEAK_ANALOG_THRESHOLD else ""
            lines.append(
                f"  Event {aid}{annotation} (similarity {score:.3f}): {row[1]} on "
                f"{row[0].upper()} at {str(row[3])[:10]}, "
                f"severity={row[2]:.2f}, price_change={pct}"
            )
    return "\n".join(lines) if lines else "Historical analog data unavailable."


def _compute_prediction_type(prediction_dict: dict) -> str:
    ptype = prediction_dict.get("prediction_type", "no_signal")
    valid = {"directional", "magnitude_only", "no_signal"}
    return ptype if ptype in valid else "no_signal"


# ─────────────────────────────────────────────────────────────────────────────
# Main engine
# ─────────────────────────────────────────────────────────────────────────────

def run_prediction_engine() -> dict:
    """
    Process all SignalAlerts that are missing prediction_json.
    Returns {"predictions_generated": N, "no_signal": M, "failed": K}
    """
    client = GeminiClient()
    predictions_generated = 0
    no_signal = 0
    failed = 0

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
        anomaly_trajectory = _get_anomaly_trajectory(commodity, detected_at)
        market_context = _get_market_context(commodity, detected_at)
        analogs_section = _build_analogs_section(corr_ids, scores)
        current_price = str(price_at_alert) if price_at_alert is not None else "N/A"
        learning_section = get_learning_context(commodity, anomaly_type)
        if not learning_section:
            learning_section = "No prior evaluations available for this signal type."
        horizon_guidance = HORIZON_GUIDANCE.get(anomaly_type, _HORIZON_DEFAULT)

        prompt = PREDICTION_PROMPT.format(
            commodity=commodity,
            anomaly_type=anomaly_type,
            severity=float(severity),
            detected_at=str(detected_at),
            signal_context=signal_context,
            anomaly_trajectory=anomaly_trajectory,
            market_context=market_context,
            analogs_section=analogs_section,
            learning_section=learning_section,
            current_price=current_price,
            horizon_guidance=horizon_guidance,
            event_id=str(anomaly_event_id),
        )

        try:
            raw_response = client.generate_text(prompt)

            # Strip chain-of-thought reasoning text before the JSON block
            json_start = raw_response.find("{")
            if json_start > 0:
                raw_response = raw_response[json_start:]

            pred_dict = json.loads(raw_response)

            raw_confidence = float(pred_dict.get("confidence_score", 0.0))
            calibrated_confidence = calibrate_confidence(raw_confidence, commodity)
            prediction_type = _compute_prediction_type(pred_dict)

            if calibrated_confidence < 0.6:
                prediction_type = "no_signal"
                pred_dict["prediction_type"] = "no_signal"
                pred_dict["predicted_outcomes"] = []

            pred_dict["confidence_score"] = calibrated_confidence
            pred_dict.setdefault("event_id", str(anomaly_event_id))
            pred_dict.setdefault("commodity", commodity)
            pred_dict.setdefault("invalidating_conditions", [])

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
                        "pconf": calibrated_confidence,
                        "id": alert_id,
                    },
                )

            if prediction_type == "no_signal":
                no_signal += 1
                logger.info(
                    "prediction_no_signal",
                    alert_id=alert_id,
                    commodity=commodity,
                    raw_confidence=raw_confidence,
                    calibrated_confidence=calibrated_confidence,
                )
            else:
                predictions_generated += 1
                logger.info(
                    "prediction_generated",
                    alert_id=alert_id,
                    commodity=commodity,
                    prediction_type=prediction_type,
                    raw_confidence=raw_confidence,
                    calibrated_confidence=calibrated_confidence,
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
