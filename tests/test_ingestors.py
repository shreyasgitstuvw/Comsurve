"""
E3 — Ingestor tests.

Tests:
  - BaseIngestor.save_to_db: idempotency (INSERT OR IGNORE), returns correct count
  - BaseIngestor.run: summary dict shape, error fallback
  - news_runner.run: primary success, primary error → fallback, RateLimitExceeded → rate_limited
  - price_realtime_runner.run: primary success, primary error → fallback
"""

import json
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from ingestion.base_ingestor import BaseIngestor
from shared.schemas import RawIngestionRecord


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_record(symbol: str = "LNG", timestamp: datetime | None = None) -> RawIngestionRecord:
    return RawIngestionRecord(
        source="test_source",
        commodity="lng",
        symbol=symbol,
        timestamp=timestamp or datetime(2024, 1, 1, 12, 0),
        data_type="price_realtime",
        raw_json=json.dumps({"price": 10.5}),
    )


class _ConcreteIngestor(BaseIngestor):
    """Minimal concrete ingestor for testing the base class."""
    source = "test_source"

    def __init__(self, records=None, raise_on_fetch=None):
        self._records = records or []
        self._raise = raise_on_fetch

    def fetch(self) -> list[RawIngestionRecord]:
        if self._raise:
            raise self._raise
        return self._records


def _patch_base_session(db_session):
    """Return a context-manager patch for ingestion.base_ingestor.get_session."""
    @contextmanager
    def _fake():
        try:
            yield db_session
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise
    return patch("ingestion.base_ingestor.get_session", side_effect=_fake)


# ── BaseIngestor.save_to_db ───────────────────────────────────────────────────

class TestSaveToDb:
    def test_empty_list_returns_zero(self, db_session):
        with _patch_base_session(db_session):
            ingestor = _ConcreteIngestor()
            assert ingestor.save_to_db([]) == 0

    def test_inserts_new_records(self, db_session):
        records = [_make_record("LNG"), _make_record("CU")]
        with _patch_base_session(db_session):
            ingestor = _ConcreteIngestor(records=records)
            count = ingestor.save_to_db(records)
        assert count == 2

    def test_idempotent_insert_or_ignore(self, db_session):
        """Second save of the same record returns 0 (duplicate silently ignored)."""
        record = [_make_record("LNG")]
        with _patch_base_session(db_session):
            ingestor = _ConcreteIngestor()
            first = ingestor.save_to_db(record)
        with _patch_base_session(db_session):
            second = ingestor.save_to_db(record)
        assert first == 1
        assert second == 0

    def test_partial_dedup(self, db_session):
        """One duplicate + one new record → returns 1."""
        rec_a = _make_record("LNG", datetime(2024, 1, 1))
        rec_b = _make_record("LNG", datetime(2024, 1, 2))  # different timestamp
        with _patch_base_session(db_session):
            _ConcreteIngestor().save_to_db([rec_a])
        with _patch_base_session(db_session):
            count = _ConcreteIngestor().save_to_db([rec_a, rec_b])
        assert count == 1


# ── BaseIngestor.run ──────────────────────────────────────────────────────────

class TestRun:
    def test_run_success_summary_keys(self, db_session):
        records = [_make_record()]
        with _patch_base_session(db_session):
            ingestor = _ConcreteIngestor(records=records)
            result = ingestor.run()

        assert result["status"] == "ok"
        assert result["source"] == "test_source"
        assert result["fetched"] == 1
        assert result["inserted"] == 1
        assert result["duplicates_skipped"] == 0
        assert "duration_ms" in result

    def test_run_deduplication_reflected_in_summary(self, db_session):
        records = [_make_record()]
        with _patch_base_session(db_session):
            _ConcreteIngestor(records=records).run()  # first run — inserts
        with _patch_base_session(db_session):
            result = _ConcreteIngestor(records=records).run()  # second run — dupe

        assert result["status"] == "ok"
        assert result["fetched"] == 1
        assert result["inserted"] == 0
        assert result["duplicates_skipped"] == 1

    def test_run_fetch_exception_returns_error_dict(self, db_session):
        with _patch_base_session(db_session):
            ingestor = _ConcreteIngestor(raise_on_fetch=RuntimeError("API down"))
            result = ingestor.run()

        assert result["status"] == "error"
        assert result["source"] == "test_source"
        assert "API down" in result["error"]
        assert "inserted" not in result  # not present on error path


# ── news_runner ───────────────────────────────────────────────────────────────

class TestNewsRunner:
    def test_primary_success_returns_primary_path(self):
        ok_summary = {"source": "newsdata", "status": "ok", "fetched": 5, "inserted": 5,
                      "duplicates_skipped": 0, "duration_ms": 100}

        with patch("ingestion.news.news_runner.NewsdataIngestor") as MockPrimary, \
             patch("ingestion.news.news_runner.NewsApiAIIngestor"):
            MockPrimary.return_value.run.return_value = ok_summary
            from ingestion.news import news_runner
            result = news_runner.run()

        assert result["path"] == "primary"
        assert result["status"] == "ok"
        assert result["source"] == "newsdata"

    def test_primary_error_triggers_fallback(self):
        error_summary = {"source": "newsdata", "status": "error", "error": "HTTP 503"}
        fallback_summary = {"source": "newsapi_ai", "status": "ok", "fetched": 3, "inserted": 3,
                            "duplicates_skipped": 0, "duration_ms": 80}

        with patch("ingestion.news.news_runner.NewsdataIngestor") as MockPrimary, \
             patch("ingestion.news.news_runner.NewsApiAIIngestor") as MockFallback:
            MockPrimary.return_value.run.return_value = error_summary
            MockFallback.return_value.run.return_value = fallback_summary
            from ingestion.news import news_runner
            result = news_runner.run()

        assert result["path"] == "fallback_newsapi_ai"
        assert "primary_error" in result
        assert "HTTP 503" in result["primary_error"]

    def test_rate_limit_returns_rate_limited_status(self):
        from ingestion.news.rate_limiter import RateLimitExceeded

        with patch("ingestion.news.news_runner.NewsdataIngestor") as MockPrimary, \
             patch("ingestion.news.news_runner.NewsApiAIIngestor"):
            MockPrimary.return_value.run.side_effect = RateLimitExceeded("quota hit")
            from ingestion.news import news_runner
            result = news_runner.run()

        assert result["status"] == "rate_limited"
        assert result["source"] == "newsdata"
        assert "quota hit" in result["error"]

    def test_primary_exception_not_rate_limit_triggers_fallback(self):
        fallback_summary = {"source": "newsapi_ai", "status": "ok", "fetched": 2, "inserted": 2,
                            "duplicates_skipped": 0, "duration_ms": 50}

        with patch("ingestion.news.news_runner.NewsdataIngestor") as MockPrimary, \
             patch("ingestion.news.news_runner.NewsApiAIIngestor") as MockFallback:
            MockPrimary.return_value.run.side_effect = ConnectionError("network error")
            MockFallback.return_value.run.return_value = fallback_summary
            from ingestion.news import news_runner
            result = news_runner.run()

        assert result["path"] == "fallback_newsapi_ai"


# ── price_realtime_runner ─────────────────────────────────────────────────────

class TestPriceRealtimeRunner:
    def test_primary_success_returns_yfinance_path(self):
        ok_summary = {"source": "yfinance", "status": "ok", "fetched": 3, "inserted": 3,
                      "duplicates_skipped": 0, "duration_ms": 50}

        with patch("ingestion.price_realtime.price_realtime_runner.YFinanceIngestor") as MockPrimary, \
             patch("ingestion.price_realtime.price_realtime_runner.CommoditiesAPIIngestor"):
            MockPrimary.return_value.run.return_value = ok_summary
            from ingestion.price_realtime import price_realtime_runner
            result = price_realtime_runner.run()

        assert result["path"] == "primary_yfinance"
        assert result["status"] == "ok"

    def test_primary_error_falls_back_to_commodities_api(self):
        error_summary = {"source": "yfinance", "status": "error", "error": "symbol not found"}
        fallback_summary = {"source": "commodities_api", "status": "ok", "fetched": 3, "inserted": 3,
                            "duplicates_skipped": 0, "duration_ms": 200}

        with patch("ingestion.price_realtime.price_realtime_runner.YFinanceIngestor") as MockPrimary, \
             patch("ingestion.price_realtime.price_realtime_runner.CommoditiesAPIIngestor") as MockFallback:
            MockPrimary.return_value.run.return_value = error_summary
            MockFallback.return_value.run.return_value = fallback_summary
            from ingestion.price_realtime import price_realtime_runner
            result = price_realtime_runner.run()

        assert result["path"] == "fallback_commodities_api"
        assert "primary_error" in result
        assert "symbol not found" in result["primary_error"]

    def test_primary_exception_triggers_fallback(self):
        fallback_summary = {"source": "commodities_api", "status": "ok", "fetched": 2, "inserted": 2,
                            "duplicates_skipped": 0, "duration_ms": 120}

        with patch("ingestion.price_realtime.price_realtime_runner.YFinanceIngestor") as MockPrimary, \
             patch("ingestion.price_realtime.price_realtime_runner.CommoditiesAPIIngestor") as MockFallback:
            MockPrimary.return_value.run.side_effect = RuntimeError("yfinance timeout")
            MockFallback.return_value.run.return_value = fallback_summary
            from ingestion.price_realtime import price_realtime_runner
            result = price_realtime_runner.run()

        assert result["path"] == "fallback_commodities_api"
        assert "yfinance timeout" in result["primary_error"]
