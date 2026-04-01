"""
Monitoring window tracker — Phase 8.

For each open signal_alert (monitoring_complete=False), checks whether the
+1w / +2w / +1m price checkpoints have elapsed and fills in the price outcome.

When all three checkpoints are filled, sets monitoring_complete=True,
which makes the alert eligible for causality engine processing (Phase 9).

Design:
  - Compares alert created_at against current time to decide which checkpoints are due.
  - Reads current price from processed_features (latest available).
  - Does NOT backfill — if a checkpoint is due but no price data exists yet,
    it is skipped and will be filled on the next run.
  - Run as part of the processor job (every 30 min).
"""

import json
from datetime import datetime, timedelta

from sqlalchemy import text

from shared.db import get_session
from shared.logger import get_logger
from shared.models import SignalAlert

logger = get_logger(__name__)

CHECKPOINTS = {
    "price_1w": timedelta(weeks=1),
    "price_2w": timedelta(weeks=2),
    "price_1m": timedelta(days=30),
}


def _get_price_near(commodity: str, target_dt: datetime, tolerance_hours: int = 12) -> float | None:
    """
    Fetch the closest available price to target_dt within ±tolerance_hours.

    Checks processed_features first (feature_type='price'), then falls back to
    raw_ingestion (price_realtime / price_historical data types) so that yfinance
    backfill data resolves checkpoints even before the feature extractor runs.
    Returns None if no data available within either source.
    """
    window_start = target_dt - timedelta(hours=tolerance_hours)
    window_end = target_dt + timedelta(hours=tolerance_hours)

    with get_session() as session:
        # Primary: processed_features price rows
        row = session.execute(
            text("""
                SELECT pf.value
                FROM processed_features pf
                JOIN raw_ingestion ri ON pf.raw_ingestion_id = ri.id
                WHERE ri.commodity = :commodity
                  AND pf.feature_type = 'price'
                  AND ri.timestamp BETWEEN :start AND :end
                ORDER BY ABS(julianday(ri.timestamp) - julianday(:target)) ASC
                LIMIT 1
            """),
            {"commodity": commodity, "start": window_start, "end": window_end, "target": target_dt},
        ).fetchone()

        if row and row[0] is not None:
            return float(row[0])

        # Fallback: parse price directly from raw_ingestion (handles backfilled yfinance data
        # that has not yet been through feature extraction).
        # The price field name varies by source: 'close' (yfinance), 'value' (FRED/EIA),
        # 'price_usd' (Commodities-API).
        raw_rows = session.execute(
            text("""
                SELECT ri.raw_json
                FROM raw_ingestion ri
                WHERE ri.commodity = :commodity
                  AND ri.data_type IN ('price_realtime', 'price_historical')
                  AND ri.timestamp BETWEEN :start AND :end
                ORDER BY ABS(julianday(ri.timestamp) - julianday(:target)) ASC
                LIMIT 5
            """),
            {"commodity": commodity, "start": window_start, "end": window_end, "target": target_dt},
        ).fetchall()

    for raw_row in (raw_rows or []):
        try:
            data = json.loads(raw_row[0])
            price = data.get("close") or data.get("value") or data.get("price_usd")
            if price is not None:
                return float(price)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    return None


def run_monitoring_window_check() -> dict:
    """
    Check all open signal_alerts and fill in elapsed price checkpoints.
    Returns summary dict.
    """
    now = datetime.utcnow()
    updated_count = 0
    completed_count = 0

    with get_session() as session:
        open_alerts = session.execute(
            text("""
                SELECT id, commodity, created_at, price_1w, price_2w, price_1m
                FROM signal_alerts
                WHERE monitoring_complete = 0
                ORDER BY created_at ASC
            """),
        ).fetchall()

    for alert_row in open_alerts:
        alert_id, commodity, created_at_raw, p1w, p2w, p1m = alert_row

        # Parse created_at (SQLite returns string)
        if isinstance(created_at_raw, str):
            try:
                created_at = datetime.fromisoformat(created_at_raw)
            except ValueError:
                created_at = datetime.strptime(created_at_raw[:19], "%Y-%m-%d %H:%M:%S")
        else:
            created_at = created_at_raw

        updates = {}
        current_values = {"price_1w": p1w, "price_2w": p2w, "price_1m": p1m}

        for field, delta in CHECKPOINTS.items():
            if current_values[field] is not None:
                continue  # already filled

            checkpoint_dt = created_at + delta
            if now >= checkpoint_dt:
                price = _get_price_near(commodity, checkpoint_dt, tolerance_hours=24)
                if price is not None:
                    updates[field] = price

        if not updates:
            continue

        # Merge with existing values to check if all three are now filled
        merged = {**current_values, **updates}
        all_filled = all(merged[f] is not None for f in CHECKPOINTS)

        set_clauses = ", ".join(f"{k} = :{k}" for k in updates)
        if all_filled:
            set_clauses += ", monitoring_complete = 1"

        with get_session() as session:
            session.execute(
                text(f"UPDATE signal_alerts SET {set_clauses} WHERE id = :id"),
                {**updates, "id": alert_id},
            )

        updated_count += 1
        if all_filled:
            completed_count += 1
            logger.info(
                "monitoring_window_complete",
                alert_id=alert_id,
                commodity=commodity,
                price_1w=merged["price_1w"],
                price_2w=merged["price_2w"],
                price_1m=merged["price_1m"],
            )
        else:
            filled = [k for k, v in updates.items()]
            logger.info("monitoring_checkpoint_filled", alert_id=alert_id, checkpoints=filled)

    summary = {
        "open_alerts_checked": len(open_alerts),
        "alerts_updated": updated_count,
        "monitoring_complete": completed_count,
    }
    logger.info("monitoring_window_check_complete", **summary)
    return summary
