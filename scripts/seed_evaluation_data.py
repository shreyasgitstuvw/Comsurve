"""
Seed script for evaluation coverage testing.

Creates 3 backdated signal_alerts (one per commodity: lng, copper, soybeans)
with all monitoring checkpoints pre-filled and prediction_json populated, then
runs causality + evaluation immediately.

Usage:
    cd mcei/
    python -m scripts.seed_evaluation_data

Idempotent — checks for existing seed rows before inserting (identified by
metadata_json containing "seed_data": true).

Why this is needed:
    The monitoring window requires +1w/+2w/+1m real time to elapse before
    monitoring_complete is set to 1. Alerts created in March 2026 won't
    qualify until late April. Seeding backdated alerts (35 days ago) makes
    all three checkpoints already elapsed, allowing causality + evaluation
    to run immediately and validate the full feedback loop.
"""

import json
import sys
from datetime import datetime, timedelta

from shared.db import get_session, init_db
from shared.logger import get_logger

logger = get_logger(__name__)

# Alerts backdated 35 days so all +1w/+2w/+1m checkpoints are already elapsed
SEED_DAYS_AGO = 35

# One seed event per commodity with realistic price levels and predictions
SEED_COMMODITIES = [
    {
        "commodity": "lng",
        "anomaly_type": "price_spike",
        "severity": 2.8,
        "metadata": {
            "pct_change": 0.087,
            "symbol": "HH",
            "window": "7d",
            "description": "Henry Hub spot price spike — 8.7% above 7-day rolling average",
            "seed_data": True,
        },
        # Approximate Henry Hub $/MMBtu levels
        "prices": {
            "price_at_alert": 2.48,
            "price_1w":       2.56,
            "price_2w":       2.51,
            "price_1m":       2.63,
        },
        # Pre-event prediction: bullish on supply disruption narrative
        "prediction": {
            "predicted_outcomes": [
                {
                    "scenario": "supply_disruption_continues",
                    "price_move": "+6 to +9%",
                    "probability": 0.55,
                    "direction_confidence": "high",
                    "time_horizon": "1m",
                },
                {
                    "scenario": "weather_normalises",
                    "price_move": "-2 to +1%",
                    "probability": 0.30,
                    "direction_confidence": "low",
                    "time_horizon": "1m",
                },
                {
                    "scenario": "demand_collapse",
                    "price_move": "-5 to -8%",
                    "probability": 0.15,
                    "direction_confidence": "high",
                    "time_horizon": "1m",
                },
            ],
            "reasoning": (
                "LNG price spike consistent with Gulf Coast export disruption; "
                "similar Q1-2024 event produced +7% over 30 days when disruption "
                "lasted >10 days."
            ),
            "key_risks": ["weather_normalisation", "EU_demand_response"],
        },
        "prediction_type": "directional",
        "prediction_confidence": 0.72,
    },
    {
        "commodity": "copper",
        "anomaly_type": "sentiment_shift",
        "severity": 1.9,
        "metadata": {
            "compound_score": -0.412,
            "window": "7d",
            "description": "Strongly negative news sentiment — Atacama mine labour dispute",
            "seed_data": True,
        },
        # Approximate copper $/lb levels
        "prices": {
            "price_at_alert": 4.12,
            "price_1w":       4.22,
            "price_2w":       4.35,
            "price_1m":       4.48,
        },
        # Pre-event prediction: bullish on supply squeeze narrative
        "prediction": {
            "predicted_outcomes": [
                {
                    "scenario": "strike_escalates",
                    "price_move": "+4 to +7%",
                    "probability": 0.45,
                    "direction_confidence": "high",
                    "time_horizon": "1m",
                },
                {
                    "scenario": "negotiated_settlement",
                    "price_move": "-1 to +2%",
                    "probability": 0.40,
                    "direction_confidence": "low",
                    "time_horizon": "1m",
                },
                {
                    "scenario": "production_halted",
                    "price_move": "+8 to +12%",
                    "probability": 0.15,
                    "direction_confidence": "high",
                    "time_horizon": "1m",
                },
            ],
            "reasoning": (
                "Negative sentiment driven by Chilean mine labour dispute; historical "
                "precedents show avg +5.5% over 30 days when disputes exceed 7 days."
            ),
            "key_risks": ["rapid_settlement", "Chinese_demand_slowdown"],
        },
        "prediction_type": "directional",
        "prediction_confidence": 0.68,
    },
    {
        "commodity": "soybeans",
        "anomaly_type": "ais_vessel_drop",
        "severity": 2.3,
        "metadata": {
            "drop_pct": 38,
            "port_name": "Santos",
            "window": "48h",
            "description": "Vessel count at Santos dropped 38% in 48h — potential port disruption",
            "seed_data": True,
        },
        # Approximate soybeans $/bushel levels
        "prices": {
            "price_at_alert": 10.24,
            "price_1w":        10.18,
            "price_2w":        10.05,
            "price_1m":         9.88,
        },
        # Pre-event prediction: bearish on export normalisation
        "prediction": {
            "predicted_outcomes": [
                {
                    "scenario": "port_congestion_resolves",
                    "price_move": "-1 to -3%",
                    "probability": 0.50,
                    "direction_confidence": "high",
                    "time_horizon": "1m",
                },
                {
                    "scenario": "export_ban_risk",
                    "price_move": "+3 to +6%",
                    "probability": 0.25,
                    "direction_confidence": "high",
                    "time_horizon": "1m",
                },
                {
                    "scenario": "weather_delay_short",
                    "price_move": "-0.5 to +0.5%",
                    "probability": 0.25,
                    "direction_confidence": "low",
                    "time_horizon": "1m",
                },
            ],
            "reasoning": (
                "Santos vessel drop consistent with port worker industrial action or "
                "storm closure; short-term supply squeeze then normalisation is most "
                "likely based on 2023-2024 precedents."
            ),
            "key_risks": ["prolonged_closure", "demand_side_weakness"],
        },
        "prediction_type": "directional",
        "prediction_confidence": 0.65,
    },
]


def _find_existing_ae(session, commodity: str, anomaly_type: str) -> int | None:
    """Return anomaly_event id if seed row already exists, else None."""
    from sqlalchemy import text
    row = session.execute(text("""
        SELECT id FROM anomaly_events
        WHERE commodity = :commodity
          AND anomaly_type = :anomaly_type
          AND metadata_json LIKE '%"seed_data": true%'
        ORDER BY id DESC LIMIT 1
    """), {"commodity": commodity, "anomaly_type": anomaly_type}).fetchone()
    return row[0] if row else None


def _find_existing_alert(session, ae_id: int) -> int | None:
    from sqlalchemy import text
    row = session.execute(text("""
        SELECT id FROM signal_alerts WHERE anomaly_event_id = :ae_id LIMIT 1
    """), {"ae_id": ae_id}).fetchone()
    return row[0] if row else None


def run_seed() -> dict:
    from sqlalchemy import text

    init_db()
    now = datetime.utcnow()
    alert_ts = now - timedelta(days=SEED_DAYS_AGO)
    # Day-floor for raw_ingestion unique constraint
    ri_ts = alert_ts.replace(hour=0, minute=0, second=0, microsecond=0)

    seeded_anomaly_ids: list[int] = []
    seeded_alert_ids: list[int] = []

    # ── Phase 1: anomaly_events ───────────────────────────────────────────────
    with get_session() as session:
        for cfg in SEED_COMMODITIES:
            commodity = cfg["commodity"]
            anomaly_type = cfg["anomaly_type"]

            existing_ae = _find_existing_ae(session, commodity, anomaly_type)
            if existing_ae:
                logger.info("seed_ae_exists", commodity=commodity, ae_id=existing_ae)
                seeded_anomaly_ids.append(existing_ae)
                continue

            # Minimal raw_ingestion row (source_ids reference)
            symbol = f"seed_{commodity}_{SEED_DAYS_AGO}d"
            session.execute(text("""
                INSERT OR IGNORE INTO raw_ingestion
                    (source, commodity, symbol, timestamp, data_type,
                     raw_json, ingested_at, processed)
                VALUES
                    ('seed_data', :commodity, :symbol, :ts, 'seed',
                     :raw_json, :now, 1)
            """), {
                "commodity": commodity,
                "symbol": symbol,
                "ts": ri_ts,
                "raw_json": json.dumps({"seed": True, "commodity": commodity}),
                "now": alert_ts,
            })

            ri_row = session.execute(text("""
                SELECT id FROM raw_ingestion
                WHERE source = 'seed_data' AND symbol = :symbol LIMIT 1
            """), {"symbol": symbol}).fetchone()
            ri_id = ri_row[0] if ri_row else 1

            session.execute(text("""
                INSERT INTO anomaly_events
                    (commodity, anomaly_type, severity, detected_at,
                     source_ids, status, metadata_json)
                VALUES
                    (:commodity, :atype, :severity, :detected_at,
                     :source_ids, 'processed', :meta)
            """), {
                "commodity": commodity,
                "atype": anomaly_type,
                "severity": cfg["severity"],
                "detected_at": alert_ts,
                "source_ids": json.dumps([ri_id]),
                "meta": json.dumps(cfg["metadata"]),
            })

            ae_row = session.execute(text("""
                SELECT id FROM anomaly_events
                WHERE commodity = :commodity
                  AND anomaly_type = :atype
                  AND metadata_json LIKE '%"seed_data": true%'
                ORDER BY id DESC LIMIT 1
            """), {"commodity": commodity, "atype": anomaly_type}).fetchone()
            ae_id = ae_row[0]
            seeded_anomaly_ids.append(ae_id)
            logger.info("ae_seeded", commodity=commodity, ae_id=ae_id)

    # ── Phase 2: signal_alerts ────────────────────────────────────────────────
    with get_session() as session:
        for i, cfg in enumerate(SEED_COMMODITIES):
            ae_id = seeded_anomaly_ids[i]

            existing_alert = _find_existing_alert(session, ae_id)
            if existing_alert:
                logger.info("seed_alert_exists", ae_id=ae_id, alert_id=existing_alert)
                seeded_alert_ids.append(existing_alert)
                continue

            # Cross-commodity correlated anomaly ids (the other two seeds)
            other_ae_ids = [aid for j, aid in enumerate(seeded_anomaly_ids) if j != i]
            similarity_scores = [round(0.82 - j * 0.07, 3) for j in range(len(other_ae_ids))]

            prices = cfg["prices"]
            session.execute(text("""
                INSERT INTO signal_alerts
                    (anomaly_event_id, commodity, alert_type,
                     correlated_anomaly_ids, similarity_scores,
                     price_at_alert, price_1w, price_2w, price_1m,
                     monitoring_complete, created_at,
                     prediction_json, prediction_type, prediction_confidence)
                VALUES
                    (:ae_id, :commodity, 'similar_historical',
                     :corr_ids, :scores,
                     :p0, :p1w, :p2w, :p1m,
                     1, :created_at,
                     :pred_json, :pred_type, :pred_conf)
            """), {
                "ae_id": ae_id,
                "commodity": cfg["commodity"],
                "corr_ids": json.dumps(other_ae_ids),
                "scores": json.dumps(similarity_scores),
                "p0":   prices["price_at_alert"],
                "p1w":  prices["price_1w"],
                "p2w":  prices["price_2w"],
                "p1m":  prices["price_1m"],
                "created_at": alert_ts,
                "pred_json": json.dumps(cfg["prediction"]),
                "pred_type": cfg["prediction_type"],
                "pred_conf": cfg["prediction_confidence"],
            })

            sa_row = session.execute(text("""
                SELECT id FROM signal_alerts WHERE anomaly_event_id = :ae_id
            """), {"ae_id": ae_id}).fetchone()
            sa_id = sa_row[0]
            seeded_alert_ids.append(sa_id)
            logger.info("alert_seeded", commodity=cfg["commodity"], alert_id=sa_id)

    logger.info("seed_complete",
                anomaly_ids=seeded_anomaly_ids,
                alert_ids=seeded_alert_ids)

    # ── Phase 3: causality + evaluation ──────────────────────────────────────
    from ai_engine.causality_engine import run_causality_engine
    causality_result = run_causality_engine()
    logger.info("causality_run_complete", **causality_result)

    return {
        "seeded_anomalies": len(seeded_anomaly_ids),
        "seeded_alerts": len(seeded_alert_ids),
        "anomaly_ids": seeded_anomaly_ids,
        "alert_ids": seeded_alert_ids,
        "causality": causality_result,
    }


if __name__ == "__main__":
    result = run_seed()
    print(json.dumps(result, indent=2))
    # Exit 1 only if nothing was seeded AND causality produced nothing
    causality = result.get("causality", {})
    ok = result["seeded_alerts"] > 0 or causality.get("reports_generated", 0) > 0
    sys.exit(0 if ok else 1)
