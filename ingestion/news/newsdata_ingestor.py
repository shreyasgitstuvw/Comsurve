"""
newsdata.io news ingestor (primary source).
Fetches recent news articles per commodity keyword.
Enforces 200 req/day quota via DailyRateLimiter.
Runs every 6 hours.
"""

import json
from datetime import datetime

import httpx

from ingestion.base_ingestor import BaseIngestor
from ingestion.news.rate_limiter import DailyRateLimiter, RateLimitExceeded
from shared.commodity_registry import NEWS_KEYWORDS
from shared.config import settings
from shared.schemas import RawIngestionRecord

NEWSDATA_BASE = "https://newsdata.io/api/1/news"
DAILY_LIMIT = 200


class NewsdataIngestor(BaseIngestor):
    source = "newsdata"

    def __init__(self):
        self.rate_limiter = DailyRateLimiter(source=self.source, daily_limit=DAILY_LIMIT)

    def fetch(self) -> list[RawIngestionRecord]:
        if not settings.newsdata_api_key:
            raise ValueError("NEWSDATA_API_KEY not set in .env")

        # Each commodity = 1 request. 3 commodities = 3 requests per run.
        requests_needed = len(NEWS_KEYWORDS)
        self.rate_limiter.check_and_increment(requests_needed)

        records: list[RawIngestionRecord] = []
        ts_now = datetime.utcnow()

        with httpx.Client(timeout=20) as client:
            for commodity, keywords in NEWS_KEYWORDS.items():
                query = " OR ".join(f'"{kw}"' for kw in keywords[:2])  # use top 2 keywords
                response = client.get(
                    NEWSDATA_BASE,
                    params={
                        "apikey": settings.newsdata_api_key,
                        "q": query,
                        "language": "en",
                        "category": "business,world",
                    },
                )
                response.raise_for_status()
                data = response.json()

                for article in data.get("results", []):
                    # Parse article publish time
                    pub_date_str = article.get("pubDate") or article.get("publishedAt", "")
                    try:
                        ts = datetime.strptime(pub_date_str, "%Y-%m-%d %H:%M:%S")
                    except (ValueError, TypeError):
                        ts = ts_now

                    url = article.get("link", "")
                    records.append(RawIngestionRecord(
                        source=self.source,
                        commodity=commodity,
                        symbol=url,          # URL used as symbol for dedup
                        timestamp=ts,
                        data_type="news",
                        raw_json=json.dumps({
                            "title": article.get("title", ""),
                            "description": article.get("description", ""),
                            "content": article.get("content", ""),
                            "url": url,
                            "source_name": article.get("source_id", ""),
                            "published_at": pub_date_str,
                            "keywords": keywords,
                        }),
                    ))

        return records
