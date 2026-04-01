"""
newsapi.ai fallback news ingestor.
Used when newsdata.io is unavailable or rate-limited.
Same interface as NewsdataIngestor.
"""

import json
from datetime import datetime

import httpx

from ingestion.base_ingestor import BaseIngestor
from shared.commodity_registry import NEWS_KEYWORDS
from shared.config import settings
from shared.schemas import RawIngestionRecord

NEWSAPI_AI_BASE = "https://eventregistry.org/api/v1/article/getArticles"


class NewsApiAIIngestor(BaseIngestor):
    source = "newsapi_ai"

    def fetch(self) -> list[RawIngestionRecord]:
        if not settings.newsapi_ai_key:
            raise ValueError("NEWSAPI_AI_KEY not set in .env")

        records: list[RawIngestionRecord] = []
        ts_now = datetime.utcnow()

        with httpx.Client(timeout=20) as client:
            for commodity, keywords in NEWS_KEYWORDS.items():
                payload = {
                    "action": "getArticles",
                    "keyword": keywords[:2],
                    "keywordOper": "OR",
                    "lang": "eng",
                    "sortBy": "date",
                    "sortByAsc": False,
                    "articlesPage": 1,
                    "articlesCount": 20,
                    "articlesSortBy": "date",
                    "resultType": "articles",
                    "dataType": ["news"],
                    "apiKey": settings.newsapi_ai_key,
                }
                response = client.post(NEWSAPI_AI_BASE, json=payload)
                response.raise_for_status()
                data = response.json()

                articles = data.get("articles", {}).get("results", [])
                for article in articles:
                    pub_date_str = article.get("dateTimePub", article.get("dateTime", ""))
                    try:
                        ts = datetime.strptime(pub_date_str[:19], "%Y-%m-%dT%H:%M:%S")
                    except (ValueError, TypeError):
                        ts = ts_now

                    url = article.get("url", "")
                    records.append(RawIngestionRecord(
                        source=self.source,
                        commodity=commodity,
                        symbol=url,
                        timestamp=ts,
                        data_type="news",
                        raw_json=json.dumps({
                            "title": article.get("title", ""),
                            "body": article.get("body", ""),
                            "url": url,
                            "source_name": article.get("source", {}).get("title", ""),
                            "published_at": pub_date_str,
                            "keywords": keywords,
                        }),
                    ))

        return records
