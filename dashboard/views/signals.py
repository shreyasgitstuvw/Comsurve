"""
Signals page — signal alerts table with similarity scores and monitoring progress.
"""

import json

import pandas as pd
import streamlit as st

from dashboard.api_client import get_signals
from shared.commodity_registry import PRICE_DECIMALS


def _fmt_price(value, commodity: str | None) -> str:
    if value is None:
        return "—"
    decimals = PRICE_DECIMALS.get(commodity or "", 3)
    return f"{value:.{decimals}f}"


def render(selected_commodity: str | None):
    st.subheader("Signal Alerts")

    signals = get_signals(commodity=selected_commodity or None, limit=100)

    if not signals:
        st.info("No signal alerts yet. Run the AI engine after anomalies are detected.")
        return

    rows = []
    for s in signals:
        try:
            scores = json.loads(s.get("similarity_scores", "[]"))
            top_score = f"{max(scores):.3f}" if scores else "—"
            corr_count = len(json.loads(s.get("correlated_anomaly_ids", "[]")))
        except (json.JSONDecodeError, ValueError):
            top_score, corr_count = "—", 0

        comm = s.get("commodity")
        rows.append({
            "ID": s["id"],
            "Commodity": (comm or "").upper(),
            "Alert Type": s["alert_type"],
            "Top Similarity": top_score,
            "Precedents": corr_count,
            "Price @ Alert": _fmt_price(s.get("price_at_alert"), comm),
            "Price +1w": _fmt_price(s.get("price_1w"), comm),
            "Price +1m": _fmt_price(s.get("price_1m"), comm),
            "Monitoring": "Complete" if s["monitoring_complete"] else "Open",
            "Created": s["created_at"][:16],
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Detail expander for selected alert
    if signals:
        st.markdown("---")
        alert_ids = [s["id"] for s in signals]
        selected_id = st.selectbox("Inspect alert detail:", alert_ids)
        alert = next((s for s in signals if s["id"] == selected_id), None)
        if alert:
            comm = alert.get("commodity")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Commodity", (comm or "").upper())
                st.metric("Alert Type", alert["alert_type"])
                st.metric("Price at Alert", _fmt_price(alert.get("price_at_alert"), comm))
            with col2:
                st.metric("Price +1w", _fmt_price(alert.get("price_1w"), comm) if alert.get("price_1w") else "Pending")
                st.metric("Price +2w", _fmt_price(alert.get("price_2w"), comm) if alert.get("price_2w") else "Pending")
                st.metric("Price +1m", _fmt_price(alert.get("price_1m"), comm) if alert.get("price_1m") else "Pending")

            try:
                corr_ids = json.loads(alert.get("correlated_anomaly_ids", "[]"))
                scores = json.loads(alert.get("similarity_scores", "[]"))
                if corr_ids:
                    st.markdown("**Historical Precedents:**")
                    for aid, score in zip(corr_ids, scores):
                        st.write(f"  Anomaly #{aid} — similarity {score:.4f}")
                else:
                    st.write("No historical precedents (novel event).")
            except json.JSONDecodeError:
                pass
