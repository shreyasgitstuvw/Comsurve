"""
Full E2E smoke test — runs all pipeline stages sequentially and validates output.

Usage:
    cd mcei/
    python scripts/run_pipeline_once.py [--skip-ai] [--skip-ingest]

Flags:
    --skip-ai      Skip Gemini embedding + signal correlation (saves API quota)
    --skip-ingest  Skip ingestion, run processor + AI only (useful if data already exists)

Exit codes:
    0 — all stages completed without error
    1 — one or more stages failed (details printed to stdout)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.db import init_db
from shared.logger import configure_logging, get_logger

configure_logging()
logger = get_logger("smoke_test")


def _run(results: dict, key: str, fn, *args, **kwargs):
    """Run fn and store result under key; catches all exceptions."""
    try:
        result = fn(*args, **kwargs)
        results[key] = result or {"status": "ok"}
    except Exception as exc:
        results[key] = {"status": "error", "error": str(exc)}
        logger.error("stage_failed", stage=key, error=str(exc))


def main():
    parser = argparse.ArgumentParser(description="MCEI full pipeline smoke test")
    parser.add_argument("--skip-ai", action="store_true",
                        help="Skip Gemini embedding and signal correlation")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip ingestion stages (run processor + AI only)")
    args = parser.parse_args()

    print("\n" + "=" * 68)
    print("  MCEI SMOKE TEST — full pipeline run")
    print("=" * 68)

    init_db()
    results: dict = {}

    # ── Stage 1: Ingestion ────────────────────────────────────────────────────
    if not args.skip_ingest:
        print("\n[1/5] Ingestion ...")

        logger.info("stage", name="fred")
        from ingestion.price_historical.fred_ingestor import FredIngestor
        _run(results, "fred", FredIngestor(lookback_days=30).run)

        logger.info("stage", name="eia")
        from ingestion.price_historical.eia_ingestor import EIAIngestor
        _run(results, "eia", EIAIngestor(lookback_days=30).run)

        logger.info("stage", name="price_realtime")
        from ingestion.price_realtime.price_realtime_runner import run as run_price_rt
        _run(results, "price_realtime", run_price_rt)

        logger.info("stage", name="news")
        from ingestion.news.news_runner import run as run_news
        _run(results, "news", run_news)

        logger.info("stage", name="ais")
        from ingestion.ais.aisstream_ingestor import AISStreamIngestor
        _run(results, "ais", AISStreamIngestor(collection_seconds=30).run)
    else:
        print("\n[1/5] Ingestion — SKIPPED")

    # ── Stage 2: Processor ────────────────────────────────────────────────────
    print("\n[2/5] Processor ...")
    from processor.processor_runner import run as run_processor
    _run(results, "processor", run_processor)

    # ── Stage 3: AI Engine (embed + correlate) ────────────────────────────────
    if not args.skip_ai:
        print("\n[3/5] AI Engine (embedding + correlation) ...")
        from ai_engine.ai_engine_runner import run_full_ai_batch
        _run(results, "ai_engine", run_full_ai_batch)
    else:
        print("\n[3/5] AI Engine — SKIPPED")

    # ── Stage 4: Causality + Prediction ──────────────────────────────────────
    if not args.skip_ai:
        print("\n[4/5] Causality + Prediction ...")
        from ai_engine.causality_engine import run_causality_engine
        _run(results, "causality", run_causality_engine)

        from ai_engine.prediction_engine import run_prediction_engine
        _run(results, "prediction", run_prediction_engine)
    else:
        print("\n[4/5] Causality + Prediction — SKIPPED")

    # ── Stage 5: DB health check ──────────────────────────────────────────────
    print("\n[5/5] DB health check ...")
    from sqlalchemy import text
    from shared.db import get_session
    from shared.models import (
        AnomalyEvent, EmbeddingCache, ProcessedFeature,
        RawIngestion, SignalAlert,
    )

    counts: dict[str, int] = {}
    try:
        with get_session() as session:
            for model, label in [
                (RawIngestion,      "raw_ingestion"),
                (ProcessedFeature,  "processed_features"),
                (AnomalyEvent,      "anomaly_events"),
                (EmbeddingCache,    "embeddings_cache"),
                (SignalAlert,       "signal_alerts"),
            ]:
                counts[label] = session.query(model).count()

            pending = session.query(AnomalyEvent).filter(
                AnomalyEvent.status == "new"
            ).count()
            processed = session.query(AnomalyEvent).filter(
                AnomalyEvent.status == "processed"
            ).count()
        results["db_health"] = {"status": "ok", "counts": counts,
                                 "anomalies_pending": pending,
                                 "anomalies_processed": processed}
    except Exception as exc:
        results["db_health"] = {"status": "error", "error": str(exc)}

    # ── Print summary ─────────────────────────────────────────────────────────
    failures = [k for k, v in results.items() if v.get("status") == "error"]

    print("\n" + "=" * 68)
    print("  RESULTS")
    print("=" * 68)
    for stage, result in results.items():
        status = result.get("status", "ok")
        icon = "✓" if status not in ("error",) else "✗"
        if stage == "db_health" and status == "ok":
            print(f"  {icon}  {stage:22s}  {result['counts']}")
            print(f"       {'':22s}  pending={result['anomalies_pending']}  "
                  f"processed={result['anomalies_processed']}")
        else:
            print(f"  {icon}  {stage:22s}  {result}")
    print("=" * 68)

    if failures:
        print(f"\n  {len(failures)} stage(s) failed: {', '.join(failures)}")
        print("  Check logs above for details.\n")
        return 1

    print("\n  All stages completed successfully.\n")

    raw_count = counts.get("raw_ingestion", 0)
    if raw_count == 0:
        print("  NOTE: raw_ingestion is empty — check API keys in .env\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
