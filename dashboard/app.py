"""
MCEI Streamlit Dashboard — main entry point.

Start with:
    streamlit run dashboard/app.py

Requires the FastAPI server running at localhost:8000.
"""

import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="MCEI — Commodity Event Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.views import backtest, overview, predictions, reports, signals, system_health

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("MCEI")
    st.caption("Multimodal Commodity Event Intelligence Engine")
    st.markdown("---")

    page = st.radio(
        "View",
        ["Overview", "Signals", "Reports", "Predictions", "Backtest", "System Health"],
        index=0,
    )

    st.markdown("---")
    st.markdown("**Commodity**")
    commodity_options = ["All", "lng", "copper", "soybeans"]
    selected_commodity = st.selectbox(
        "Filter by commodity",
        commodity_options,
        label_visibility="collapsed",
    )
    selected_commodity = None if selected_commodity == "All" else selected_commodity

    if page == "Overview":
        st.markdown("**Time Window**")
        window = st.selectbox(
            "Price window",
            ["1w", "1m", "3m", "1y"],
            index=1,
            label_visibility="collapsed",
        )
    else:
        window = "1m"

    st.markdown("---")
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()

    st.caption("Data sourced from FRED, EIA, newsdata.io, aisstream.io, yfinance via MCEI API")

# ── Main content ──────────────────────────────────────────────────────────────
if page == "Overview":
    commodities = ["lng", "copper", "soybeans"] if selected_commodity is None else [selected_commodity]
    for c in commodities:
        overview.render(c, window)
        if c != commodities[-1]:
            st.markdown("---")

elif page == "Signals":
    signals.render(selected_commodity)

elif page == "Reports":
    reports.render(selected_commodity)

elif page == "Predictions":
    predictions.render(selected_commodity)

elif page == "Backtest":
    backtest.render()

elif page == "System Health":
    system_health.render()
