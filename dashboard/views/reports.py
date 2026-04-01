"""
Reports page — causality report viewer.
"""

import pandas as pd
import streamlit as st

from dashboard.api_client import get_reports

CATEGORY_COLORS = {
    "supply_disruption": "🔴",
    "demand_shock": "🟠",
    "geopolitical": "🟣",
    "weather": "🔵",
    "regulatory": "🟢",
    "financial": "🟡",
    "unknown": "⚪",
}


def render(selected_commodity: str | None):
    st.subheader("Causality Reports")

    reports = get_reports(commodity=selected_commodity or None, limit=50)

    if not reports:
        st.info(
            "No causality reports yet.\n\n"
            "Reports are generated automatically once a signal alert's "
            "monitoring window closes (+1 month). "
            "You can test immediately by running:\n"
            "```\npython -m ai_engine.causality_engine\n```"
        )
        return

    # Summary table at the top
    rows = []
    for r in reports:
        cat = r.get("cause_category", "unknown")
        icon = CATEGORY_COLORS.get(cat, "⚪")
        impact = r.get("price_impact_pct")
        rows.append({
            "Commodity": r["commodity"].upper(),
            "Category": f"{icon} {cat.replace('_', ' ').title()}",
            "Confidence": f"{(r.get('confidence_score') or 0):.0%}",
            "Price Impact": f"{impact:+.1f}%" if impact is not None else "N/A",
            "Date": str(r["created_at"])[:10],
            "Summary": (r.get("summary") or "")[:80],
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.markdown("---")

    # Detail cards — expanded by default for the first report
    for i, r in enumerate(reports):
        cat = r.get("cause_category", "unknown")
        icon = CATEGORY_COLORS.get(cat, "⚪")
        confidence = r.get("confidence_score") or 0
        impact = r.get("price_impact_pct")
        impact_str = f"{impact:+.1f}%" if impact is not None else "N/A"

        with st.expander(
            f"{icon}  [{r['commodity'].upper()}]  {r.get('cause', 'Unknown cause')}  —  {str(r['created_at'])[:10]}",
            expanded=(i == 0),   # first report open, rest collapsed
        ):
            col1, col2, col3 = st.columns(3)
            col1.metric("Category", cat.replace("_", " ").title())
            col2.metric("Confidence", f"{confidence:.0%}")
            col3.metric("Price Impact (+1m)", impact_str)

            if r.get("summary"):
                st.info(r["summary"])

            if r.get("mechanism"):
                st.markdown(f"**Mechanism:** {r['mechanism']}")

            signals = r.get("supporting_signals") or []
            if signals:
                st.markdown("**Supporting signals:** " + ", ".join(
                    f"`{s}`" for s in signals
                ))

            precedents = r.get("historical_precedents") or []
            if precedents:
                st.markdown(f"**Historical precedents:** Anomaly IDs {precedents}")
            else:
                st.markdown("**Historical precedents:** None (novel event)")
