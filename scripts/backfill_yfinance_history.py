"""
One-time backfill script:
  1. Removes FRED price_historical rows for copper and soybeans
     (monthly frequency + USD/metric-ton units conflict with yfinance futures quotes).
  2. Removes duplicate same-day yfinance price_realtime rows for copper/soybeans
     (identical close values stamped at utcnow — these pollute the price chart).
  3. Backfills 1 year of daily yfinance bars for copper and soybeans.
  4. Re-runs the processor so processed_features reflects the new data.

Safe to run multiple times — BaseIngestor uses INSERT OR IGNORE on
(source, symbol, timestamp) so re-runs will not double-insert.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.db import get_session, init_db
from sqlalchemy import text

from ingestion.price_historical.yfinance_historical_ingestor import YFinanceHistoricalIngestor
from processor.processor_runner import run as run_processor


def clean_bad_rows():
    with get_session() as s:
        # 1a. Delete processed_features for FRED copper/soybeans first (FK child rows)
        result = s.execute(text("""
            DELETE FROM processed_features
            WHERE raw_ingestion_id IN (
                SELECT id FROM raw_ingestion
                WHERE source = 'fred'
                  AND commodity IN ('copper', 'soybeans')
            )
        """))
        print(f"Deleted {result.rowcount} processed_features for FRED copper/soybeans")

        # 1b. Now safe to delete the parent raw_ingestion rows
        result = s.execute(text("""
            DELETE FROM raw_ingestion
            WHERE source = 'fred'
              AND commodity IN ('copper', 'soybeans')
        """))
        fred_deleted = result.rowcount
        print(f"Deleted {fred_deleted} FRED copper/soybeans raw_ingestion rows")

        # 2a. Delete processed_features for duplicate yfinance same-day rows
        result = s.execute(text("""
            DELETE FROM processed_features
            WHERE raw_ingestion_id IN (
                SELECT id FROM raw_ingestion
                WHERE source = 'yfinance'
                  AND commodity IN ('copper', 'soybeans')
                  AND id NOT IN (
                      SELECT MIN(id)
                      FROM raw_ingestion
                      WHERE source = 'yfinance'
                        AND commodity IN ('copper', 'soybeans')
                      GROUP BY commodity, DATE(timestamp)
                  )
            )
        """))
        print(f"Deleted {result.rowcount} processed_features for duplicate yfinance rows")

        # 2b. Remove the duplicate raw_ingestion rows (keep earliest per day)
        result = s.execute(text("""
            DELETE FROM raw_ingestion
            WHERE source = 'yfinance'
              AND commodity IN ('copper', 'soybeans')
              AND id NOT IN (
                  SELECT MIN(id)
                  FROM raw_ingestion
                  WHERE source = 'yfinance'
                    AND commodity IN ('copper', 'soybeans')
                  GROUP BY commodity, DATE(timestamp)
              )
        """))
        dup_deleted = result.rowcount
        print(f"Deleted {dup_deleted} duplicate yfinance same-day raw_ingestion rows")


def backfill():
    ingestor = YFinanceHistoricalIngestor(lookback_days=365)
    summary = ingestor.run()
    print(f"yfinance backfill: inserted={summary.get('inserted', 0)}, "
          f"skipped={summary.get('skipped', 0)}")
    return summary


def reprocess():
    print("Re-running processor...")
    result = run_processor()
    feats = result.get("features", {})
    print(f"Processor: processed={feats.get('processed_rows', 0)}, "
          f"features_written={feats.get('features_written', 0)}")
    anom = result.get("anomalies", {})
    print(f"Anomalies: {anom.get('total_new_anomalies', 0)} new")


if __name__ == "__main__":
    init_db()
    print("=== Step 1: Clean stale data ===")
    clean_bad_rows()
    print()
    print("=== Step 2: Backfill yfinance 1y daily history ===")
    backfill()
    print()
    print("=== Step 3: Re-run processor ===")
    reprocess()
    print()
    print("Done. Refresh the dashboard to see updated charts.")
