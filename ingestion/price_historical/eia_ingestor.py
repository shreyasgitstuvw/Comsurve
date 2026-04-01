"""
EIA (US Energy Information Administration) ingestor.
Fetches LNG export volumes and natural gas storage data.
Runs once daily at 02:00, alongside FRED.

Series:
  lng_exports → NG.N9133US2.W    (US LNG exports, Bcf/week)
  ng_storage  → NG.NW2_EPG0_SWO_R48_BCF.W (Working gas in storage, Bcf)
"""

import json
from datetime import datetime, timedelta

import httpx

from ingestion.base_ingestor import BaseIngestor
from shared.commodity_registry import EIA_SERIES
from shared.config import settings
from shared.schemas import RawIngestionRecord

EIA_BASE = "https://api.eia.gov/v2/seriesid/{series_id}"


class EIAIngestor(BaseIngestor):
    source = "eia"

    def __init__(self, lookback_days: int = 90):
        self.lookback_days = lookback_days

    def fetch(self) -> list[RawIngestionRecord]:
        if not settings.eia_api_key:
            raise ValueError("EIA_API_KEY not set in .env")

        start_date = (datetime.utcnow() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
        records: list[RawIngestionRecord] = []

        with httpx.Client(timeout=30) as client:
            for series_name, series_id in EIA_SERIES.items():
                url = EIA_BASE.format(series_id=series_id)
                response = client.get(
                    url,
                    params={
                        "api_key": settings.eia_api_key,
                        "data[0]": "value",
                        "start": start_date,
                        "sort[0][column]": "period",
                        "sort[0][direction]": "asc",
                        "length": 500,
                    },
                )
                response.raise_for_status()
                data = response.json()

                # EIA v2 API response structure
                response_data = data.get("response", {}).get("data", [])
                for obs in response_data:
                    try:
                        period = obs.get("period", "")
                        # EIA periods can be "2024-01" (monthly) or "2024-01-05" (weekly)
                        if len(period) == 7:
                            ts = datetime.strptime(period, "%Y-%m")
                        else:
                            ts = datetime.strptime(period, "%Y-%m-%d")
                        value = float(obs["value"])
                    except (ValueError, KeyError, TypeError):
                        continue

                    records.append(RawIngestionRecord(
                        source=self.source,
                        commodity="lng",
                        symbol=series_id,
                        timestamp=ts,
                        data_type="price_historical",
                        raw_json=json.dumps({
                            "series_name": series_name,
                            "series_id": series_id,
                            "period": period,
                            "value": value,
                            "units": obs.get("units", ""),
                        }),
                    ))

        return records


if __name__ == "__main__":
    from shared.db import init_db
    init_db()
    ingestor = EIAIngestor(lookback_days=30)
    summary = ingestor.run()
    print(summary)
