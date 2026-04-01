"""
Predictions page — Pre-event scenario predictions and post-event evaluation.
"""

import pandas as pd
import streamlit as st

from dashboard.api_client import get_predictions

PREDICTION_TYPE_ICONS = {
    "directional": "🎯",
    "magnitude_only": "↔️",
    "no_signal": "⬜",
}

DIRECTION_ICONS = {
    True: "✅",
    False: "❌",
    None: "—",
}


def _format_confidence(conf) -> str:
    if conf is None:
        return "N/A"
    return f"{conf:.0%}"


def _format_score(score) -> str:
    if score is None:
        return "N/A"
    return f"{score:.2f}"


def _format_pct(val) -> str:
    if val is None:
        return "N/A"
    return f"{val:+.2f}%"


def render(selected_commodity: str | None):
    st.subheader("Scenario Predictions & Evaluation")

    predictions = get_predictions(commodity=selected_commodity, limit=50)

    if not predictions:
        st.info(
            "No predictions yet.\n\n"
            "Predictions are generated automatically when new SignalAlerts are created "
            "during the daily AI batch. You can trigger them manually by running:\n"
            "```\npython -m ai_engine.prediction_engine\n```\n\n"
            "Evaluations are generated after a CausalityReport is produced for the same alert:\n"
            "```\npython -m ai_engine.evaluation_engine\n```"
        )
        return

    # ── Summary metrics ───────────────────────────────────────────────────────
    total = len(predictions)
    directional_count = sum(1 for p in predictions if p.get("prediction_type") == "directional")
    directional_pct = directional_count / total * 100 if total else 0

    confs = [p["prediction_confidence"] for p in predictions if p.get("prediction_confidence") is not None]
    avg_conf = sum(confs) / len(confs) if confs else None

    scores = [p["overall_score"] for p in predictions if p.get("overall_score") is not None]
    avg_score = sum(scores) / len(scores) if scores else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Predictions", total)
    col2.metric("Directional %", f"{directional_pct:.0f}%")
    col3.metric("Avg Confidence", _format_confidence(avg_conf))
    col4.metric("Avg Overall Score", _format_score(avg_score))

    st.markdown("---")

    # ── Summary table ─────────────────────────────────────────────────────────
    rows = []
    for p in predictions:
        pred_data = p.get("prediction_json") or {}
        outcomes = pred_data.get("predicted_outcomes", [])

        # Build scenario range: "Bull +3-7% / Bear -2-5%" style summary
        if outcomes:
            scenario_parts = [
                f"{o.get('scenario', '?')} {o.get('price_move', '?')}"
                for o in outcomes[:3]
            ]
            scenario_range = " | ".join(scenario_parts)
        else:
            scenario_range = "N/A"

        evaluation = p.get("evaluation") or {}
        accuracy = evaluation.get("prediction_accuracy", {})
        direction_correct = accuracy.get("direction_correct")

        rows.append({
            "Commodity": (p.get("commodity") or "").upper(),
            "Anomaly Type": p.get("anomaly_type") or "N/A",
            "Prediction Type": p.get("prediction_type") or "N/A",
            "Confidence": _format_confidence(p.get("prediction_confidence")),
            "Scenarios": len(outcomes),
            "Scenario Range": scenario_range,
            "Actual Change": _format_pct(
                evaluation.get("actual_price_change_pct")
                if evaluation else None
            ),
            "Direction Correct": DIRECTION_ICONS.get(direction_correct, "—"),
            "Overall Score": _format_score(p.get("overall_score")),
            "Date": str(p.get("created_at", ""))[:10],
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("---")

    # ── Detail cards for predictions with evaluations ─────────────────────────
    evaluated = [p for p in predictions if p.get("evaluation")]
    if not evaluated:
        st.info("No evaluated predictions yet. Evaluations appear once monitoring windows close.")
        return

    st.markdown("### Evaluation Detail Cards")

    for i, p in enumerate(evaluated):
        pred_data = p.get("prediction_json") or {}
        evaluation = p.get("evaluation") or {}
        accuracy = evaluation.get("prediction_accuracy", {})
        causal = evaluation.get("causal_analysis", {})
        failure_modes = evaluation.get("failure_modes", [])
        learning = evaluation.get("learning_update", {})

        outcomes = pred_data.get("predicted_outcomes", [])
        predicted_move = outcomes[0].get("price_move", "N/A") if outcomes else "N/A"
        direction_correct = accuracy.get("direction_correct")
        direction_icon = DIRECTION_ICONS.get(direction_correct, "—")
        score = p.get("overall_score")
        commodity = (p.get("commodity") or "").upper()

        label = (
            f"{direction_icon}  [{commodity}]  "
            f"{p.get('prediction_type', 'N/A')}  —  "
            f"Score: {_format_score(score)}  —  {str(p.get('created_at', ''))[:10]}"
        )

        with st.expander(label, expanded=(i == 0)):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Prediction Type", p.get("prediction_type") or "N/A")
            c2.metric("Confidence", _format_confidence(p.get("prediction_confidence")))
            c3.metric("Overall Score", _format_score(score))
            c4.metric(
                "Actual Change",
                _format_pct(evaluation.get("actual_price_change_pct")),
            )

            # Signal summary
            if pred_data.get("signal_summary"):
                st.info(pred_data["signal_summary"])

            # Predicted outcomes
            if outcomes:
                st.markdown("**Predicted Outcomes:**")
                for o in outcomes:
                    st.markdown(
                        f"- **{o.get('scenario', '?')}** — "
                        f"{o.get('price_move', '?')} "
                        f"(prob: {o.get('probability', 0):.0%}, "
                        f"confidence: {o.get('direction_confidence', '?')}, "
                        f"horizon: {o.get('time_horizon', '?')})"
                    )

            # Accuracy breakdown
            st.markdown("**Accuracy:**")
            acc_cols = st.columns(4)
            acc_cols[0].metric("Direction Correct", str(direction_correct))
            acc_cols[1].metric("Magnitude Error", f"{accuracy.get('magnitude_error', 0):.2f}pp")
            acc_cols[2].metric("Volatility Correct", str(accuracy.get("volatility_correct")))
            acc_cols[3].metric("Calibration", accuracy.get("confidence_validity", "N/A"))

            # Causal analysis
            correct_drivers = causal.get("correct_drivers", [])
            missed_drivers = causal.get("missed_drivers", [])
            overestimated_drivers = causal.get("overestimated_drivers", [])

            if correct_drivers:
                st.markdown("**Correct Drivers:** " + ", ".join(f"`{d}`" for d in correct_drivers))
            if missed_drivers:
                st.markdown("**Missed Drivers:** " + ", ".join(f"`{d}`" for d in missed_drivers))
            if overestimated_drivers:
                st.markdown("**Overestimated Drivers:** " + ", ".join(f"`{d}`" for d in overestimated_drivers))

            # Failure modes
            if failure_modes:
                st.markdown("**Failure Modes:**")
                for fm in failure_modes:
                    st.markdown(f"- {fm}")

            # Learning update
            if learning:
                st.markdown("**Learning Update:**")
                if learning.get("insight"):
                    st.markdown(f"- *Insight:* {learning['insight']}")
                if learning.get("future_adjustment"):
                    st.markdown(f"- *Future Adjustment:* {learning['future_adjustment']}")
                affected = learning.get("affected_signal_types", [])
                if affected:
                    st.markdown("- *Affected Signal Types:* " + ", ".join(f"`{s}`" for s in affected))
