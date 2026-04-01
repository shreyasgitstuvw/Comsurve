"""
Feature extractor: reads unprocessed raw_ingestion rows and writes derived features
to processed_features.

Features computed:
  - news        → sentiment_score (VADER compound)
  - price_*     → pct_change, log_return (vs previous row for same symbol)
  - ais         → passed through from ais_feature_extractor (separate module)
  - satellite   → passed through from scene_processor (separate module, Phase 13)
"""

import json
import math
from datetime import datetime

from sqlalchemy import text

from processor.sentiment.vader_scorer import score_article
from shared.db import get_session
from shared.logger import get_logger
from shared.models import ProcessedFeature, RawIngestion

logger = get_logger(__name__)


def _extract_news_features(session, row: RawIngestion) -> list[ProcessedFeature]:
    try:
        data = json.loads(row.raw_json)
    except json.JSONDecodeError:
        return []

    title = data.get("title", "")
    description = data.get("description", "") or data.get("body", "")
    content = data.get("content", "")
    compound = score_article(title, description, content, commodity=row.commodity)

    return [ProcessedFeature(
        raw_ingestion_id=row.id,
        commodity=row.commodity,
        feature_type="sentiment_score",
        value=compound,
        computed_at=datetime.utcnow(),
    )]


def _extract_price_features(session, row: RawIngestion) -> list[ProcessedFeature]:
    try:
        data = json.loads(row.raw_json)
    except json.JSONDecodeError:
        return []

    # Extract price value from different source schemas
    price = (
        data.get("value")          # FRED / EIA
        or data.get("price_usd")   # Commodities-API
        or data.get("close")       # yfinance
    )
    if price is None:
        return []
    price = float(price)

    # Find previous price for the same commodity + data_type to compute returns
    prev = session.execute(
        text("""
            SELECT pf.value FROM processed_features pf
            JOIN raw_ingestion ri ON pf.raw_ingestion_id = ri.id
            WHERE ri.commodity = :commodity
              AND ri.data_type = :data_type
              AND ri.timestamp < :ts
              AND pf.feature_type = 'pct_change'
            ORDER BY ri.timestamp DESC
            LIMIT 1
        """),
        {"commodity": row.commodity, "data_type": row.data_type, "ts": row.timestamp},
    ).fetchone()

    # Get the raw price from the previous row to compute pct_change
    prev_raw = session.execute(
        text("""
            SELECT raw_json FROM raw_ingestion
            WHERE commodity = :commodity
              AND data_type = :data_type
              AND timestamp < :ts
            ORDER BY timestamp DESC
            LIMIT 1
        """),
        {"commodity": row.commodity, "data_type": row.data_type, "ts": row.timestamp},
    ).fetchone()

    features = []
    if prev_raw:
        try:
            prev_data = json.loads(prev_raw[0])
            prev_price = (
                prev_data.get("value")
                or prev_data.get("price_usd")
                or prev_data.get("close")
            )
            if prev_price and float(prev_price) != 0:
                prev_price = float(prev_price)
                pct_change = (price - prev_price) / prev_price
                log_return = math.log(price / prev_price) if price > 0 and prev_price > 0 else 0.0

                features.append(ProcessedFeature(
                    raw_ingestion_id=row.id,
                    commodity=row.commodity,
                    feature_type="pct_change",
                    value=pct_change,
                    computed_at=datetime.utcnow(),
                ))
                features.append(ProcessedFeature(
                    raw_ingestion_id=row.id,
                    commodity=row.commodity,
                    feature_type="log_return",
                    value=log_return,
                    computed_at=datetime.utcnow(),
                ))
        except (json.JSONDecodeError, ValueError, ZeroDivisionError):
            pass

    # Always store the raw price value as a feature for downstream use
    features.append(ProcessedFeature(
        raw_ingestion_id=row.id,
        commodity=row.commodity,
        feature_type="price",
        value=price,
        computed_at=datetime.utcnow(),
    ))

    return features


def run_feature_extraction() -> dict:
    """Process all unprocessed raw_ingestion rows and write features."""
    processed_count = 0
    feature_count = 0

    with get_session() as session:
        unprocessed = (
            session.query(RawIngestion)
            .filter(RawIngestion.processed == False)
            .filter(RawIngestion.data_type.in_(["news", "price_historical", "price_realtime"]))
            .order_by(RawIngestion.timestamp)
            .all()
        )

        for row in unprocessed:
            if row.data_type == "news":
                features = _extract_news_features(session, row)
            elif row.data_type in ("price_historical", "price_realtime"):
                features = _extract_price_features(session, row)
            else:
                features = []

            for f in features:
                session.add(f)
            row.processed = True
            processed_count += 1
            feature_count += len(features)

    logger.info("feature_extraction_complete", processed_rows=processed_count, features_written=feature_count)
    return {"processed_rows": processed_count, "features_written": feature_count}
