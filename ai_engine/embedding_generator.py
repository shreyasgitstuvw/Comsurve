"""
Context assembler + embedding generator for anomaly events.

For each AnomalyEvent, this module:
  1. Fetches the linked raw_ingestion rows (source_ids) to extract news text,
     price deltas, and AIS vessel counts.
  2. Builds a structured natural-language context payload that captures the
     multimodal signal (what happened, where, how severe, what data sources saw it).
  3. Calls GeminiClient.embed() to get a 3072-dim vector.

The text payload is designed so semantically similar events produce geometrically
close vectors — enabling Qdrant nearest-neighbor to surface historical precedents.
"""

import json
from datetime import datetime

from sqlalchemy import bindparam, text

from ai_engine.gemini_client import GeminiClient
from shared.db import get_session
from shared.logger import get_logger
from shared.models import AnomalyEvent

logger = get_logger(__name__)


def _fetch_source_data(source_ids: list[int]) -> dict:
    """
    Pull raw_ingestion rows for the given IDs and bucket them by data_type.
    Returns dict with keys: news, prices, ais.
    """
    if not source_ids:
        return {"news": [], "prices": [], "ais": []}

    with get_session() as session:
        rows = session.execute(
            text("""
                SELECT id, source, commodity, data_type, raw_json, timestamp
                FROM raw_ingestion
                WHERE id IN :ids
            """).bindparams(bindparam("ids", expanding=True)),
            {"ids": source_ids},
        ).fetchall()

    buckets: dict[str, list[dict]] = {"news": [], "prices": [], "ais": []}

    for row in rows:
        try:
            data = json.loads(row[4])
        except json.JSONDecodeError:
            continue

        data_type = row[3]
        if data_type == "news":
            buckets["news"].append({
                "title": data.get("title", ""),
                "description": data.get("description", "") or data.get("body", ""),
                "source": row[1],
                "timestamp": str(row[5]),
            })
        elif data_type in ("price_historical", "price_realtime"):
            price_val = data.get("value") or data.get("price_usd") or data.get("close")
            buckets["prices"].append({
                "source": row[1],
                "symbol": data.get("series_id") or data.get("symbol") or data.get("ticker", ""),
                "price": price_val,
                "timestamp": str(row[5]),
            })
        elif data_type == "ais":
            buckets["ais"].append({
                "port": data.get("port_name", ""),
                "vessel_count": data.get("vessel_count", 0),
                "avg_sog": data.get("avg_sog_knots", 0),
                "timestamp": str(row[5]),
            })

    return buckets


def build_context_payload(anomaly: AnomalyEvent) -> str:
    """
    Build a structured natural-language string describing the anomaly and its
    supporting data signals. This is the text that gets embedded.

    Format is designed to be:
    - Semantic: similar events have similar language patterns
    - Factual: numbers and sources included for specificity
    - Concise: under ~2000 tokens (well within Gemini's context limit)
    """
    try:
        meta = json.loads(anomaly.metadata_json or "{}")
    except json.JSONDecodeError:
        meta = {}

    try:
        source_ids = json.loads(anomaly.source_ids)
    except json.JSONDecodeError:
        source_ids = []

    source_data = _fetch_source_data(source_ids)

    lines = [
        f"COMMODITY: {anomaly.commodity.upper()}",
        f"ANOMALY TYPE: {anomaly.anomaly_type}",
        f"SEVERITY: {anomaly.severity:.3f} (Z-score or normalised magnitude)",
        f"DETECTED AT: {anomaly.detected_at.strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]

    # Type-specific context
    if anomaly.anomaly_type == "price_spike":
        pct = meta.get("pct_change", 0) * 100
        z = meta.get("z_score", anomaly.severity)
        lines += [
            f"PRICE EVENT: {anomaly.commodity} price moved {pct:+.2f}% "
            f"(Z-score: {z:.2f}, threshold: 2.0)",
            f"DATA SOURCE: {meta.get('data_type', 'price')}",
        ]

    elif anomaly.anomaly_type == "sentiment_shift":
        score = meta.get("compound_score", 0)
        z = meta.get("z_score", anomaly.severity)
        sentiment_label = "strongly negative" if score < -0.3 else ("strongly positive" if score > 0.3 else "neutral")
        lines += [
            f"SENTIMENT EVENT: News sentiment for {anomaly.commodity} is {sentiment_label} "
            f"(VADER compound: {score:.3f}, Z-score: {z:.2f})",
        ]

    elif anomaly.anomaly_type in ("ais_vessel_drop", "ais_port_idle"):
        port = meta.get("port_name", meta.get("port_slug", "unknown port"))
        lines += [
            f"PORT EVENT: {port}",
            f"CURRENT VESSEL COUNT: {meta.get('current_vessel_count', meta.get('vessel_count', 'N/A'))}",
        ]
        if anomaly.anomaly_type == "ais_vessel_drop":
            lines.append(
                f"BASELINE AVERAGE: {meta.get('baseline_avg', 'N/A')} vessels "
                f"(drop: {meta.get('drop_pct', 'N/A')}%)"
            )
        elif anomaly.anomaly_type == "ais_port_idle":
            lines.append(f"MOORED RATIO: {meta.get('moored_ratio', 'N/A'):.1%}")

    # News context (up to 3 articles)
    if source_data["news"]:
        lines += ["", "RELATED NEWS:"]
        for article in source_data["news"][:3]:
            title = article["title"]
            desc = article["description"][:200] if article["description"] else ""
            lines.append(f'  - [{article["timestamp"][:10]}] {title}')
            if desc:
                lines.append(f'    {desc}')

    # Price context (up to 2 data points)
    if source_data["prices"]:
        lines += ["", "PRICE DATA:"]
        for p in source_data["prices"][:2]:
            lines.append(f'  - {p["symbol"]} = {p["price"]} @ {str(p["timestamp"])[:10]}')

    # AIS context
    if source_data["ais"]:
        lines += ["", "PORT ACTIVITY:"]
        for a in source_data["ais"][:3]:
            lines.append(
                f'  - {a["port"]}: {a["vessel_count"]} vessels, avg SOG {a["avg_sog"]} kn'
            )

    return "\n".join(lines)


def generate_embedding(anomaly: AnomalyEvent, client: GeminiClient) -> tuple[list[float], str]:
    """
    Build context payload and generate embedding vector.
    Returns (vector, context_text).
    """
    context = build_context_payload(anomaly)
    logger.info(
        "embedding_context_built",
        anomaly_id=anomaly.id,
        anomaly_type=anomaly.anomaly_type,
        commodity=anomaly.commodity,
        context_chars=len(context),
    )
    vector = client.embed(context)
    return vector, context
