"""
Overview page — price chart with anomaly event markers.
"""

import json
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.api_client import get_anomalies, get_prices

COMMODITY_LABELS = {
    "lng": "LNG — Henry Hub ($/MMBtu)",
    "copper": "Copper Futures ($/lb)",
    "soybeans": "Soybeans Futures (¢/bu)",
}

ANOMALY_COLORS = {
    "price_spike": "#ef4444",
    "sentiment_shift": "#f97316",
    "ais_vessel_drop": "#8b5cf6",
    "ais_port_idle": "#6366f1",
    "satellite_scene_gap": "#0ea5e9",
    "satellite_cloud_block": "#64748b",
    "satellite_aircraft_surge": "#10b981",
}

WINDOW_DAYS = {"1d": 1, "1w": 7, "1m": 30, "3m": 90, "1y": 365}


def render(selected_commodity: str, window: str):
    st.subheader(f"{selected_commodity.upper()} — {COMMODITY_LABELS.get(selected_commodity, 'Price')}")

    price_data = get_prices(selected_commodity, window)

    if not price_data or not price_data.get("prices"):
        st.warning(
            f"No price data for {selected_commodity.upper()} in the `{window}` window. "
            "Run `python scripts/run_pipeline_once.py` to ingest data."
        )
        return

    prices = price_data["prices"]
    df = pd.DataFrame(prices)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp", "source"])

    if len(df) < 2:
        st.info(
            f"Only {len(df)} data point(s) for {selected_commodity.upper()}. "
            "More data will accumulate as the scheduler runs."
        )
        if len(df) == 1:
            row = df.iloc[0]
            st.metric(
                label=f"Latest price ({row['source']})",
                value=f"{row['price']:.4f}",
                help=str(row["timestamp"]),
            )
        return

    # Anomaly events for overlay
    since_dt = datetime.utcnow() - timedelta(days=WINDOW_DAYS.get(window, 30))
    anomalies = get_anomalies(commodity=selected_commodity, since=since_dt, limit=200)

    fig = go.Figure()

    # Price line per source
    for source, grp in df.groupby("source"):
        fig.add_trace(go.Scatter(
            x=grp["timestamp"],
            y=grp["price"],
            mode="lines+markers" if len(grp) < 10 else "lines",
            name=source,
            line=dict(width=2),
            marker=dict(size=5),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br><b>%{y:.4f}</b><extra>" + source + "</extra>",
        ))

    # Anomaly markers
    for atype, color in ANOMALY_COLORS.items():
        subset = [a for a in anomalies if a["anomaly_type"] == atype]
        if not subset:
            continue

        marker_x, marker_y, hover_texts = [], [], []
        for a in subset:
            try:
                det = pd.to_datetime(a["detected_at"])
                # Snap to closest price row for y-position
                idx = (df["timestamp"] - det).abs().idxmin()
                y_val = float(df.loc[idx, "price"])
            except Exception:
                continue
            marker_x.append(det)
            marker_y.append(y_val * 1.015)   # offset slightly above line
            hover_texts.append(
                f"<b>{atype.replace('_', ' ').title()}</b><br>"
                f"Severity: {a['severity']:.2f}<br>"
                f"Status: {a['status']}<br>"
                f"{str(a['detected_at'])[:16]}"
            )

        if marker_x:
            fig.add_trace(go.Scatter(
                x=marker_x, y=marker_y,
                mode="markers",
                marker=dict(symbol="triangle-up", size=14, color=color,
                            line=dict(color="white", width=1)),
                name=atype.replace("_", " ").title(),
                hovertext=hover_texts,
                hoverinfo="text",
            ))

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=COMMODITY_LABELS.get(selected_commodity, "Price"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest", f"{df['price'].iloc[-1]:.4f}")
    col2.metric("High", f"{df['price'].max():.4f}")
    col3.metric("Low", f"{df['price'].min():.4f}")
    pct = (df["price"].iloc[-1] - df["price"].iloc[0]) / df["price"].iloc[0] * 100
    col4.metric("Change", f"{pct:+.2f}%")

    # Anomaly table
    if anomalies:
        with st.expander(f"{len(anomalies)} anomaly event(s) in this window", expanded=False):
            rows = [
                {
                    "Detected": str(a["detected_at"])[:16],
                    "Type": a["anomaly_type"],
                    "Severity": round(a["severity"], 3),
                    "Status": a["status"],
                }
                for a in anomalies[:20]
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
