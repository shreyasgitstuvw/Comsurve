"""
yfinance historical price ingestor.
Fetches daily OHLC bars for the past N days using each bar's actual date
as the timestamp. Used as the primary historical source for copper and
soybeans (FRED series for these commodities are monthly and in different
units — USD/metric ton — which mismatches yfinance futures quotes).

LNG historical data stays on FRED (EIA daily series is daily-frequency and
in the same MMBtu units as the realtime feed).
"""

import json
from datetime import datetime, timezone

import yfinance as yf

from ingestion.base_ingestor import BaseIngestor
from shared.commodity_registry import YFINANCE_TICKERS
from shared.schemas import RawIngestionRecord

# Only backfill commodities where FRED data is monthly / wrong units.
# LNG is excluded because fred_ingestor already provides daily data.
HISTORICAL_COMMODITIES = {"copper", "soybeans"}


class YFinanceHistoricalIngestor(BaseIngestor):
    source = "yfinance_historical"

    def __init__(self, lookback_days: int = 365, commodities: set | None = None):
        self.lookback_days = lookback_days
        self.commodities = commodities or HISTORICAL_COMMODITIES

    def fetch(self) -> list[RawIngestionRecord]:
        records: list[RawIngestionRecord] = []
        period = f"{self.lookback_days}d"

        for commodity, ticker_symbol in YFINANCE_TICKERS.items():
            if commodity not in self.commodities:
                continue

            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period=period, interval="1d")
            if hist.empty:
                continue

            for bar_date, row in hist.iterrows():
                # bar_date is a pandas Timestamp (tz-aware from yfinance)
                if hasattr(bar_date, "to_pydatetime"):
                    ts = bar_date.to_pydatetime().replace(
                        hour=0, minute=0, second=0, microsecond=0, tzinfo=None
                    )
                else:
                    ts = datetime(bar_date.year, bar_date.month, bar_date.day)

                price = float(row["Close"])

                records.append(RawIngestionRecord(
                    source=self.source,
                    commodity=commodity,
                    symbol=ticker_symbol,
                    timestamp=ts,
                    data_type="price_historical",
                    raw_json=json.dumps({
                        "ticker": ticker_symbol,
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": price,
                        "volume": int(row.get("Volume", 0)),
                        "bar_date": ts.isoformat(),
                    }),
                ))

        return records
