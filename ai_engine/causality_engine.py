"""
Causality engine — Phase 9.

For each signal_alert where monitoring_complete=True and no report exists yet:
  1. Assembles full context: anomaly details + correlated historical events
     + actual price outcomes (at_alert, +1w, +2w, +1m)
  2. Calls Gemini Flash with a structured JSON prompt
  3. Parses the JSON response into a CausalityReport
  4. Writes the report to causality_reports table
  5. Exports a human-readable JSON file to reports/

JSON report fields:
  cause          : primary causal factor (e.g. "supply disruption", "demand shock")
  cause_category : taxonomy bucket (supply_disruption | demand_shock | geopolitical |
                   weather | regulatory | financial | unknown)
  mechanism      : how the cause produced the price effect (1-3 sentences)
  price_impact_pct: actual price change from alert to +1m (%)
  confidence     : 0.0–1.0 self-reported Gemini confidence
  supporting_signals: list of signal types that contributed (news, ais, price_spike, etc.)
  historical_precedents: list of correlated anomaly IDs and their similarity scores
  summary        : one-sentence plain-English summary for dashboard display
"""

import json
import os
from datetime import datetime

from sqlalchemy import text

from ai_engine.gemini_client import GeminiClient
from shared.db import get_session, init_db
from shared.logger import get_logger
from shared.models import CausalityReport

logger = get_logger(__name__)

REPORTS_DIR = "reports"

CAUSALITY_PROMPT_TEMPLATE = """
You are a commodity market analyst specialising in supply chain disruptions.

Analyse the following event and produce a structured causal explanation in JSON.

=== COMMODITY CONTEXT ===
{commodity_context}

=== EVENT ===
{event_section}

=== HISTORICAL PRECEDENTS (similar past events) ===
{precedents_section}

=== PRICE OUTCOME ===
{price_section}

=== INSTRUCTIONS ===
Return ONLY valid JSON with exactly these fields:
{{
  "cause": "<primary causal factor in 5-10 words>",
  "cause_category": "<one of: supply_disruption | demand_shock | geopolitical | weather | regulatory | financial | unknown>",
  "mechanism": "<1-3 sentences explaining how the cause produced the price effect>",
  "price_impact_pct": <float: actual price change % from alert to +1m, or null if unavailable>,
  "confidence": <float 0.0-1.0: your confidence in this causal assessment>,
  "supporting_signals": [<list of strings: signal types that support this cause>],
  "historical_precedents": [<list of anomaly_event_ids that are similar>],
  "summary": "<one sentence plain-English summary for a trading dashboard>"
}}
Do not include any explanation outside the JSON object.
""".strip()

_COMMODITY_CONTEXT: dict[str, str] = {
    "lng": (
        "LNG (Liquefied Natural Gas): Henry Hub is the US benchmark ($/MMBtu). "
        "Key supply risks: terminal outages (Freeport, Sabine Pass, Corpus Christi), "
        "extreme weather freezing wellhead equipment, geopolitical events cutting "
        "pipeline flows, tanker diversions. Key demand drivers: temperature extremes, "
        "Asian spot buying, European storage targets. Price is highly seasonal and "
        "weather-sensitive; winter supply shocks are more impactful than summer ones."
    ),
    "copper": (
        "Copper: LME and COMEX are the benchmarks ($/metric ton and $/lb). "
        "Key supply risks: mine strikes or closures (Escondida, Codelco, Las Bambas), "
        "ore grade decline, flooding or seismic events at major Chilean/Peruvian mines, "
        "smelter constraints in China. Key demand drivers: Chinese construction and EV "
        "production, global infrastructure spending. Inventory at LME/COMEX warehouses "
        "is a leading indicator — drawdowns precede price spikes."
    ),
    "soybeans": (
        "Soybeans: CBOT futures (cents/bushel). Key supply risks: drought or flood in "
        "the US Corn Belt (Iowa, Illinois, Indiana), La Niña affecting Brazilian/Argentine "
        "harvests, export bans from Argentina. Key demand drivers: Chinese crush demand "
        "for animal feed, US biofuel mandates. Export inspection data and USDA crop "
        "progress reports are critical leading indicators. Basis (local vs futures) "
        "reflects regional supply tightness."
    ),
}


def _build_commodity_context(commodity: str) -> str:
    return _COMMODITY_CONTEXT.get(commodity, f"Commodity: {commodity.upper()}.")


def _build_event_section(alert_data: dict, anomaly_data: dict) -> str:
    lines = [
        f"Commodity: {alert_data['commodity'].upper()}",
        f"Anomaly type: {anomaly_data['anomaly_type']}",
        f"Severity (Z-score): {anomaly_data['severity']:.3f}",
        f"Detected at: {anomaly_data['detected_at']}",
        f"Alert type: {alert_data['alert_type']}",
    ]
    try:
        meta = json.loads(anomaly_data.get("metadata_json") or "{}")
        if "pct_change" in meta:
            lines.append(f"Price change at detection: {meta['pct_change']*100:+.2f}%")
        if "port_name" in meta:
            lines.append(f"Port affected: {meta['port_name']}")
        if "drop_pct" in meta:
            lines.append(f"Vessel count drop: {meta['drop_pct']}%")
        if "compound_score" in meta:
            lines.append(f"News sentiment score: {meta['compound_score']:.3f}")
    except (json.JSONDecodeError, KeyError):
        pass
    return "\n".join(lines)


def _build_precedents_section(correlated_ids: list[int], similarity_scores: list[float]) -> str:
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
                    pct = f"{pct*100:+.2f}%"
            except (json.JSONDecodeError, KeyError):
                pct = "N/A"
            lines.append(
                f"  Event {aid} (similarity {score:.3f}): {row[1]} on {row[0].upper()} "
                f"at {str(row[3])[:10]}, severity={row[2]:.2f}, price_change={pct}"
            )
    return "\n".join(lines) if lines else "Historical data unavailable."


def _build_price_section(alert_data: dict) -> str:
    p0 = alert_data.get("price_at_alert")
    p1w = alert_data.get("price_1w")
    p2w = alert_data.get("price_2w")
    p1m = alert_data.get("price_1m")

    def pct(base, current):
        if base and current and base != 0:
            return f"{(current - base) / base * 100:+.2f}%"
        return "N/A"

    return (
        f"Price at alert:   {p0}\n"
        f"Price at +1 week: {p1w}  ({pct(p0, p1w)} vs alert)\n"
        f"Price at +2 weeks:{p2w}  ({pct(p0, p2w)} vs alert)\n"
        f"Price at +1 month:{p1m}  ({pct(p0, p1m)} vs alert)"
    )


def _compute_price_impact(alert_data: dict) -> float | None:
    p0 = alert_data.get("price_at_alert")
    # Use best available endpoint: 1m preferred, then 2w, then 1w
    p_end = alert_data.get("price_1m") or alert_data.get("price_2w") or alert_data.get("price_1w")
    if p0 and p_end and p0 != 0:
        return round((p_end - p0) / p0 * 100, 2)
    return None


def run_causality_engine() -> dict:
    """Process all completed monitoring windows without existing reports."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    client = GeminiClient()
    generated = 0
    failed = 0

    with get_session() as session:
        rows = session.execute(
            text("""
                SELECT sa.id, sa.commodity, sa.alert_type,
                       sa.correlated_anomaly_ids, sa.similarity_scores,
                       sa.price_at_alert, sa.price_1w, sa.price_2w, sa.price_1m,
                       sa.created_at,
                       ae.anomaly_type, ae.severity, ae.detected_at, ae.metadata_json
                FROM signal_alerts sa
                JOIN anomaly_events ae ON ae.id = sa.anomaly_event_id
                LEFT JOIN causality_reports cr ON cr.signal_alert_id = sa.id
                WHERE (sa.monitoring_complete = 1 OR sa.price_1w IS NOT NULL)
                  AND cr.id IS NULL
                ORDER BY sa.created_at ASC
                LIMIT 20
            """),
        ).fetchall()

    logger.info("causality_engine_start", candidates=len(rows))

    for row in rows:
        (alert_id, commodity, alert_type,
         corr_ids_json, scores_json,
         p_alert, p1w, p2w, p1m,
         created_at,
         anomaly_type, severity, detected_at, metadata_json) = row

        alert_data = {
            "commodity": commodity, "alert_type": alert_type,
            "price_at_alert": p_alert, "price_1w": p1w,
            "price_2w": p2w, "price_1m": p1m,
        }
        anomaly_data = {
            "anomaly_type": anomaly_type, "severity": severity,
            "detected_at": str(detected_at), "metadata_json": metadata_json,
        }

        try:
            corr_ids = json.loads(corr_ids_json or "[]")
            scores = json.loads(scores_json or "[]")
        except json.JSONDecodeError:
            corr_ids, scores = [], []

        prompt = CAUSALITY_PROMPT_TEMPLATE.format(
            commodity_context=_build_commodity_context(commodity),
            event_section=_build_event_section(alert_data, anomaly_data),
            precedents_section=_build_precedents_section(corr_ids, scores),
            price_section=_build_price_section(alert_data),
        )

        try:
            raw_response = client.generate_text(prompt)
            report_dict = json.loads(raw_response)

            # Ensure required fields present
            report_dict.setdefault("cause", "unknown")
            report_dict.setdefault("cause_category", "unknown")
            report_dict.setdefault("confidence", 0.5)
            report_dict.setdefault("price_impact_pct", _compute_price_impact(alert_data))
            report_dict.setdefault("historical_precedents", corr_ids)
            report_dict.setdefault("supporting_signals", [anomaly_type])
            report_dict.setdefault("summary", f"{commodity.upper()} {anomaly_type} event analysis")

            # Enrich with metadata
            report_dict["_meta"] = {
                "signal_alert_id": alert_id,
                "commodity": commodity,
                "generated_at": datetime.utcnow().isoformat(),
            }

            price_impact = report_dict.get("price_impact_pct") or _compute_price_impact(alert_data)

            # Write to DB
            with get_session() as session:
                session.add(CausalityReport(
                    signal_alert_id=alert_id,
                    commodity=commodity,
                    report_json=json.dumps(report_dict, indent=2),
                    cause_category=report_dict.get("cause_category"),
                    confidence_score=float(report_dict.get("confidence", 0.5)),
                    price_impact_pct=price_impact,
                    created_at=datetime.utcnow(),
                ))

            # Export JSON file
            filename = f"{REPORTS_DIR}/report_alert{alert_id}_{commodity}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, "w") as f:
                json.dump(report_dict, f, indent=2)

            logger.info(
                "causality_report_generated",
                alert_id=alert_id,
                commodity=commodity,
                cause_category=report_dict.get("cause_category"),
                confidence=report_dict.get("confidence"),
                price_impact_pct=price_impact,
                file=filename,
            )
            generated += 1

        except Exception as exc:
            logger.error("causality_report_failed", alert_id=alert_id, error=str(exc))
            failed += 1

    summary = {"reports_generated": generated, "failed": failed, "candidates": len(rows)}
    logger.info("causality_engine_complete", **summary)
    return summary


if __name__ == "__main__":
    init_db()
    result = run_causality_engine()
    print(result)
