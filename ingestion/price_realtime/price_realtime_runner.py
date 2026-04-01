"""
Price real-time runner with primary/fallback pattern.
yfinance is primary (no quota constraints, free, reliable).
Commodities-API is fallback (100 req/month free tier — preserved for when yfinance
is unavailable or returns stale data).
Logs which path was taken so fallback usage is auditable.
"""

from shared.db import init_db
from shared.logger import get_logger
from ingestion.price_realtime.commodities_api_ingestor import CommoditiesAPIIngestor
from ingestion.price_realtime.yfinance_ingestor import YFinanceIngestor

logger = get_logger(__name__)


def run() -> dict:
    primary = YFinanceIngestor()
    fallback = CommoditiesAPIIngestor()

    try:
        summary = primary.run()
        if summary.get("status") == "error":
            raise RuntimeError(summary["error"])
        summary["path"] = "primary_yfinance"
        return summary
    except Exception as primary_exc:
        logger.warning(
            "price_realtime_yfinance_failed",
            error=str(primary_exc),
            fallback="commodities_api",
        )
        summary = fallback.run()
        summary["path"] = "fallback_commodities_api"
        summary["primary_error"] = str(primary_exc)
        return summary


if __name__ == "__main__":
    init_db()
    result = run()
    print(result)
