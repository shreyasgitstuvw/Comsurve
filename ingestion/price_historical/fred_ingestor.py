"""
FRED (Federal Reserve Economic Data) historical price ingestor.
Fetches daily/monthly commodity price series via FRED REST API.
Runs once daily at 02:00.

Series:
  lng      → DHHNGSP   (Henry Hub Natural Gas Spot, daily $/MMBtu)
  copper   → PCOPPUSDM (Global Copper Price, monthly USD/metric ton)
  soybeans → PSOYBUSDM (Global Soybean Price, monthly USD/metric ton)
"""

import json
from datetime import datetime, timedelta

import httpx

from ingestion.base_ingestor import BaseIngestor
from shared.commodity_registry import FRED_SERIES
from shared.config import settings
from shared.schemas import RawIngestionRecord

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


class FredIngestor(BaseIngestor):
    source = "fred"

    def __init__(self, lookback_days: int = 90):
        """
        lookback_days: how far back to fetch on each run.
        FRED deduplication is handled by the unique constraint in base_ingestor.
        """
        self.lookback_days = lookback_days

    def fetch(self) -> list[RawIngestionRecord]:
        if not settings.fred_api_key:
            raise ValueError("FRED_API_KEY not set in .env")

        observation_start = (datetime.utcnow() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
        records: list[RawIngestionRecord] = []

        with httpx.Client(timeout=30) as client:
            for commodity, series_id in FRED_SERIES.items():
                response = client.get(
                    FRED_BASE,
                    params={
                        "series_id": series_id,
                        "api_key": settings.fred_api_key,
                        "file_type": "json",
                        "observation_start": observation_start,
                        "sort_order": "asc",
                    },
                )
                response.raise_for_status()
                data = response.json()

                for obs in data.get("observations", []):
                    # FRED uses "." for missing values
                    if obs["value"] == ".":
                        continue
                    try:
                        ts = datetime.strptime(obs["date"], "%Y-%m-%d")
                        value = float(obs["value"])
                    except (ValueError, KeyError):
                        continue

                    records.append(RawIngestionRecord(
                        source=self.source,
                        commodity=commodity,
                        symbol=series_id,
                        timestamp=ts,
                        data_type="price_historical",
                        raw_json=json.dumps({
                            "series_id": series_id,
                            "date": obs["date"],
                            "value": value,
                            "units": data.get("units", ""),
                        }),
                    ))

        return records


if __name__ == "__main__":
    from shared.db import init_db
    init_db()
    ingestor = FredIngestor(lookback_days=30)
    summary = ingestor.run()
    print(summary)
