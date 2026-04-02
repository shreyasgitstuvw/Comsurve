"""
EIA (US Energy Information Administration) ingestor.
Fetches LNG export volumes and natural gas storage data.
Runs once daily at 02:00, alongside FRED.

Series:
  lng_exports → N9133US2   via /v2/natural-gas/move/expc/data/ (monthly, MMcf)
  ng_storage  → NG.NW2_EPG0_SWO_R48_BCF.W via /v2/seriesid/ (weekly, Bcf)
"""

import json
from datetime import datetime, timedelta

import httpx

from ingestion.base_ingestor import BaseIngestor
from shared.commodity_registry import EIA_EXPC_SERIES, EIA_SERIES
from shared.config import settings
from shared.schemas import RawIngestionRecord

EIA_BASE = "https://api.eia.gov/v2/seriesid/{series_id}"
EIA_EXPC_URL = "https://api.eia.gov/v2/natural-gas/move/expc/data/"


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
            # ── LNG exports via expc route ────────────────────────────────────
            for series_name, series_id in EIA_EXPC_SERIES.items():
                response = client.get(
                    EIA_EXPC_URL,
                    params={
                        "api_key": settings.eia_api_key,
                        "data[0]": "value",
                        "facets[series][]": series_id,
                        "start": start_date,
                        "sort[0][column]": "period",
                        "sort[0][direction]": "asc",
                        "length": 500,
                    },
                )
                response.raise_for_status()
                for obs in response.json().get("response", {}).get("data", []):
                    try:
                        period = obs.get("period", "")
                        ts = datetime.strptime(period, "%Y-%m") if len(period) == 7 else datetime.strptime(period, "%Y-%m-%d")
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

            # ── Other series via seriesid route ───────────────────────────────
            for series_name, series_id in EIA_SERIES.items():
                response = client.get(
                    EIA_BASE.format(series_id=series_id),
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
                for obs in response.json().get("response", {}).get("data", []):
                    try:
                        period = obs.get("period", "")
                        ts = datetime.strptime(period, "%Y-%m") if len(period) == 7 else datetime.strptime(period, "%Y-%m-%d")
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
