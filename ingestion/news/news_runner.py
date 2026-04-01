"""
News ingestion runner with primary/fallback pattern and URL deduplication.
Mirrors price_realtime_runner.py's fallback approach.
"""

from ingestion.news.newsdata_ingestor import NewsdataIngestor
from ingestion.news.newsapi_ai_ingestor import NewsApiAIIngestor
from ingestion.news.rate_limiter import RateLimitExceeded
from shared.db import init_db
from shared.logger import get_logger

logger = get_logger(__name__)


def run() -> dict:
    primary = NewsdataIngestor()
    fallback = NewsApiAIIngestor()

    try:
        summary = primary.run()
        if summary.get("status") == "error":
            raise RuntimeError(summary["error"])
        summary["path"] = "primary"
        return summary
    except RateLimitExceeded as rle:
        logger.warning("news_rate_limit_reached", error=str(rle))
        return {"source": "newsdata", "status": "rate_limited", "error": str(rle)}
    except Exception as primary_exc:
        logger.warning(
            "news_primary_failed",
            error=str(primary_exc),
            fallback="newsapi_ai",
        )
        summary = fallback.run()
        summary["path"] = "fallback_newsapi_ai"
        summary["primary_error"] = str(primary_exc)
        return summary


if __name__ == "__main__":
    init_db()
    result = run()
    print(result)
