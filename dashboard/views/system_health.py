"""
System health page — DB table counts + recent job run history.
"""

import pandas as pd
import streamlit as st

from dashboard.api_client import get_health

TABLE_ICONS = {
    "raw_ingestion": "Ingest",
    "processed_features": "Features",
    "anomaly_events": "Anomalies",
    "embeddings_cache": "Embeddings",
    "signal_alerts": "Signals",
    "causality_reports": "Reports",
    "prediction_evaluations": "Evaluations",
    "learning_updates": "Learnings",
}

STATUS_COLORS = {"ok": "green", "error": "red", "running": "orange"}


def render():
    st.subheader("System Health")

    health = get_health()

    if not health:
        st.error("API is offline. Start with: python -m uvicorn api.main:app --reload")
        return

    # DB table counts
    st.markdown("#### Database")
    counts = health.get("table_counts", {})
    cols = st.columns(len(counts))
    for i, (table, count) in enumerate(counts.items()):
        label = TABLE_ICONS.get(table, table)
        cols[i].metric(label, f"{count:,}")

    st.caption(f"DB: `{health.get('db_path', 'mcei.db')}`")

    # Recent job runs
    st.markdown("#### Recent Job Runs")
    jobs = health.get("recent_jobs", [])

    if not jobs:
        st.info("No job runs recorded yet. Start the scheduler to begin.")
        return

    rows = []
    for j in jobs:
        status = j.get("status", "?")
        icon = {"ok": "OK", "error": "!!", "running": ">>"}.get(status, "??")
        rows.append({
            "Status": f"{icon} {status}",
            "Job": j.get("job_name", ""),
            "Started": (j.get("started_at") or "")[:19],
            "Finished": (j.get("finished_at") or "—")[:19],
            "Error": (j.get("error") or "")[:80],
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
