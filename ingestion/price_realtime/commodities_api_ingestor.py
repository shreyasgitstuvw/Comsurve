"""
Commodities-API real-time price ingestor (primary source).
Fetches current spot prices for LNG, Copper, Soybeans.
Runs every hour.
"""

import json
from datetime import datetime

import httpx

from ingestion.base_ingestor import BaseIngestor
from shared.commodity_registry import COMMODITIES_API_SYMBOLS
from shared.config import settings
from shared.schemas import RawIngestionRecord

COMMODITIES_API_BASE = "https://commodities-api.com/api/latest"


class CommoditiesAPIIngestor(BaseIngestor):
    source = "commodities_api"

    def fetch(self) -> list[RawIngestionRecord]:
        if not settings.commodities_api_key:
            raise ValueError("COMMODITIES_API_KEY not set in .env")

        symbols = ",".join(COMMODITIES_API_SYMBOLS.values())
        with httpx.Client(timeout=15) as client:
            response = client.get(
                COMMODITIES_API_BASE,
                params={
                    "access_key": settings.commodities_api_key,
                    "symbols": symbols,
                    "base": "USD",
                },
            )
            response.raise_for_status()
            data = response.json()

        if not data.get("success"):
            raise RuntimeError(f"Commodities-API error: {data.get('error', {})}")

        rates = data.get("data", {}).get("rates", {})
        ts = datetime.utcnow().replace(second=0, microsecond=0)
        records: list[RawIngestionRecord] = []

        for commodity, api_symbol in COMMODITIES_API_SYMBOLS.items():
            price = rates.get(api_symbol)
            if price is None:
                continue
            records.append(RawIngestionRecord(
                source=self.source,
                commodity=commodity,
                symbol=api_symbol,
                timestamp=ts,
                data_type="price_realtime",
                raw_json=json.dumps({
                    "symbol": api_symbol,
                    "price_usd": price,
                    "base": "USD",
                    "fetched_at": ts.isoformat(),
                }),
            ))

        return records
