"""
yfinance real-time price ingestor (fallback source).
Used when Commodities-API is unavailable or rate-limited.
Pulls last closing price from futures tickers.
"""

import json
from datetime import datetime

import yfinance as yf

from ingestion.base_ingestor import BaseIngestor
from shared.commodity_registry import YFINANCE_TICKERS
from shared.schemas import RawIngestionRecord


class YFinanceIngestor(BaseIngestor):
    source = "yfinance"

    def fetch(self) -> list[RawIngestionRecord]:
        records: list[RawIngestionRecord] = []
        ts = datetime.utcnow().replace(second=0, microsecond=0)

        for commodity, ticker_symbol in YFINANCE_TICKERS.items():
            ticker = yf.Ticker(ticker_symbol)
            # Use 5d window so we always get the last trading close even on weekends
            hist = ticker.history(period="5d")
            if hist.empty:
                continue
            last_row = hist.iloc[-1]
            price = float(last_row["Close"])

            records.append(RawIngestionRecord(
                source=self.source,
                commodity=commodity,
                symbol=ticker_symbol,
                timestamp=ts,
                data_type="price_realtime",
                raw_json=json.dumps({
                    "ticker": ticker_symbol,
                    "close": price,
                    "volume": int(last_row.get("Volume", 0)),
                    "fetched_at": ts.isoformat(),
                }),
            ))

        return records
