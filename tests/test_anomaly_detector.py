"""
Tests for processor/anomaly_detector.py (price + sentiment detectors).

All tests use an in-memory SQLite DB via the patch_db fixture from conftest.
No external services are called.
"""

import json
import statistics
from datetime import datetime, timedelta

import pytest
from sqlalchemy import text

from processor.anomaly_detector import (
    MIN_WINDOW_POINTS,
    PRICE_ZSCORE_THRESHOLD,
    SENTIMENT_SPIKE_THRESHOLD,
    _compute_zscore,
    detect_price_anomalies,
    detect_sentiment_anomalies,
)
from shared.models import AnomalyEvent, ProcessedFeature, RawIngestion


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_raw(session, commodity="lng", source="fred", symbol="LNG",
              data_type="price_historical", days_ago=0):
    """Insert a RawIngestion row and return its id."""
    ts = datetime.utcnow() - timedelta(days=days_ago)
    row = RawIngestion(
        source=source,
        commodity=commodity,
        symbol=symbol,
        timestamp=ts,
        data_type=data_type,
        raw_json="{}",
    )
    session.add(row)
    session.flush()
    return row.id


def _make_feature(session, raw_id, feature_type, value):
    pf = ProcessedFeature(
        raw_ingestion_id=raw_id,
        commodity="lng",
        feature_type=feature_type,
        value=value,
    )
    session.add(pf)
    session.flush()
    return pf


# ── Unit tests: _compute_zscore ───────────────────────────────────────────────

class TestComputeZscore:
    def test_returns_none_for_short_window(self):
        assert _compute_zscore(5.0, [1.0, 2.0]) is None

    def test_returns_none_for_zero_stdev(self):
        # All values identical → stdev = 0 → None (cannot compute meaningful Z)
        assert _compute_zscore(5.0, [3.0] * 10) is None

    def test_high_value_returns_positive_zscore(self):
        # Non-uniform window so stdev > 0
        window = [1.0, 1.2, 0.8, 1.1, 0.9, 1.0, 1.2, 0.8, 1.1, 0.9]
        z = _compute_zscore(100.0, window)
        assert z is not None and z > 2.0

    def test_low_value_returns_negative_zscore(self):
        window = [10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 10.3, 9.7, 10.0, 10.4]
        z = _compute_zscore(1.0, window)
        assert z is not None and z < -2.0

    def test_value_near_mean_returns_small_zscore(self):
        window = [1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        mean = statistics.mean(window)
        z = _compute_zscore(mean, window)
        assert z is not None and abs(z) < 0.1

    def test_minimum_window_size_accepted(self):
        window = [1.0, 2.0, 3.0, 4.0, 5.0]  # exactly MIN_WINDOW_POINTS
        assert len(window) == MIN_WINDOW_POINTS
        result = _compute_zscore(10.0, window)
        assert result is not None


# ── Integration tests: detect_price_anomalies ─────────────────────────────────

class TestDetectPriceAnomalies:
    def test_returns_empty_when_no_data(self, patch_db):
        result = detect_price_anomalies("lng")
        assert result == []

    def test_returns_empty_when_too_few_points(self, patch_db):
        session = patch_db
        for i in range(MIN_WINDOW_POINTS - 1):
            raw_id = _make_raw(session, days_ago=i)
            _make_feature(session, raw_id, "pct_change", 0.01)
        session.commit()

        result = detect_price_anomalies("lng")
        assert result == []

    def test_detects_price_spike(self, patch_db):
        session = patch_db
        # Insert baseline: 20 pct_change values with natural variance (not all identical)
        baseline = [0.01, 0.02, -0.01, 0.03, -0.02, 0.015, 0.005, 0.025,
                    -0.015, 0.02, 0.01, -0.005, 0.03, 0.01, -0.01, 0.02,
                    0.015, 0.005, 0.01, -0.02]
        for i, val in enumerate(baseline):
            raw_id = _make_raw(session, days_ago=len(baseline) - i)
            _make_feature(session, raw_id, "pct_change", val)

        # Insert a spike value far above the baseline (many stdev away)
        spike_raw_id = _make_raw(session, days_ago=0)
        _make_feature(session, spike_raw_id, "pct_change", 5.0)  # huge spike
        session.commit()

        result = detect_price_anomalies("lng")
        assert len(result) >= 1
        spike = result[-1]
        assert isinstance(spike, AnomalyEvent)
        assert spike.anomaly_type == "price_spike"
        assert spike.commodity == "lng"
        assert spike.severity > PRICE_ZSCORE_THRESHOLD

    def test_normal_variation_not_flagged(self, patch_db):
        session = patch_db
        # All values close together — no anomaly expected
        for i in range(25):
            raw_id = _make_raw(session, days_ago=25 - i)
            _make_feature(session, raw_id, "pct_change", 0.01 + i * 0.0001)
        session.commit()

        result = detect_price_anomalies("lng")
        assert result == []

    def test_deduplication_prevents_duplicate_anomaly(self, patch_db):
        session = patch_db
        baseline = [0.01, 0.02, -0.01, 0.03, -0.02, 0.015, 0.005, 0.025,
                    -0.015, 0.02, 0.01, -0.005, 0.03, 0.01, -0.01, 0.02,
                    0.015, 0.005, 0.01, -0.02]
        for i, val in enumerate(baseline):
            raw_id = _make_raw(session, days_ago=len(baseline) - i)
            _make_feature(session, raw_id, "pct_change", val)

        spike_raw_id = _make_raw(session, days_ago=0)
        _make_feature(session, spike_raw_id, "pct_change", 5.0)
        session.commit()

        result1 = detect_price_anomalies("lng")
        assert len(result1) >= 1

        # Second call should not create duplicate anomalies for the same raw_id
        result2 = detect_price_anomalies("lng")
        assert len(result2) == 0

    def test_isolation_by_commodity(self, patch_db):
        session = patch_db
        # Insert spike for 'copper', not for 'lng'
        for i in range(20):
            raw_id = _make_raw(session, commodity="copper", symbol="CU", days_ago=20 - i)
            _make_feature(session, raw_id, "pct_change", 0.01)

        spike_raw_id = _make_raw(session, commodity="copper", symbol="CU", days_ago=0)
        _make_feature(session, spike_raw_id, "pct_change", 5.0)
        session.commit()

        lng_anomalies = detect_price_anomalies("lng")
        assert lng_anomalies == []


# ── Integration tests: detect_sentiment_anomalies ─────────────────────────────

class TestDetectSentimentAnomalies:
    def test_returns_empty_when_no_data(self, patch_db):
        result = detect_sentiment_anomalies("lng")
        assert result == []

    def test_returns_empty_when_too_few_points(self, patch_db):
        session = patch_db
        for i in range(MIN_WINDOW_POINTS - 1):
            raw_id = _make_raw(session, data_type="news", days_ago=i)
            _make_feature(session, raw_id, "sentiment_score", 0.1)
        session.commit()

        result = detect_sentiment_anomalies("lng")
        assert result == []

    def test_detects_strong_negative_sentiment_spike(self, patch_db):
        session = patch_db
        # Baseline: 20 articles with slight natural variance around neutral
        baseline = [0.05, 0.08, 0.03, 0.06, -0.02, 0.04, 0.07, 0.01,
                    0.05, 0.09, 0.03, -0.01, 0.06, 0.04, 0.08, 0.02,
                    0.05, 0.07, 0.03, 0.06]
        for i, val in enumerate(baseline):
            raw_id = _make_raw(session, data_type="news", days_ago=len(baseline) - i)
            _make_feature(session, raw_id, "sentiment_score", val)

        # Spike: very negative article (far outside baseline, exceeds |0.6| threshold)
        spike_raw_id = _make_raw(session, data_type="news", days_ago=0)
        _make_feature(session, spike_raw_id, "sentiment_score", -0.95)
        session.commit()

        result = detect_sentiment_anomalies("lng")
        assert len(result) >= 1
        event = result[-1]
        assert event.anomaly_type == "sentiment_shift"
        assert event.severity > PRICE_ZSCORE_THRESHOLD

    def test_strong_sentiment_below_spike_threshold_not_flagged(self, patch_db):
        """A sentiment value that is a Z-score outlier but doesn't exceed SENTIMENT_SPIKE_THRESHOLD."""
        session = patch_db
        # All values near zero with tiny variance
        for i in range(20):
            raw_id = _make_raw(session, data_type="news", days_ago=20 - i)
            _make_feature(session, raw_id, "sentiment_score", 0.001 * i)

        # Outlier by Z-score but below |0.6| threshold
        spike_raw_id = _make_raw(session, data_type="news", days_ago=0)
        _make_feature(session, spike_raw_id, "sentiment_score", 0.3)  # below SENTIMENT_SPIKE_THRESHOLD
        session.commit()

        result = detect_sentiment_anomalies("lng")
        # May or may not flag; if flagged, severity must still exceed threshold
        for event in result:
            meta = json.loads(event.metadata_json)
            assert abs(meta["compound_score"]) >= SENTIMENT_SPIKE_THRESHOLD

    def test_deduplication_prevents_duplicate_sentiment_anomaly(self, patch_db):
        session = patch_db
        baseline = [0.05, 0.08, 0.03, 0.06, -0.02, 0.04, 0.07, 0.01,
                    0.05, 0.09, 0.03, -0.01, 0.06, 0.04, 0.08, 0.02,
                    0.05, 0.07, 0.03, 0.06]
        for i, val in enumerate(baseline):
            raw_id = _make_raw(session, data_type="news", days_ago=len(baseline) - i)
            _make_feature(session, raw_id, "sentiment_score", val)

        spike_raw_id = _make_raw(session, data_type="news", days_ago=0)
        _make_feature(session, spike_raw_id, "sentiment_score", -0.95)
        session.commit()

        result1 = detect_sentiment_anomalies("lng")
        assert len(result1) >= 1

        result2 = detect_sentiment_anomalies("lng")
        assert len(result2) == 0

    def test_metadata_fields_present(self, patch_db):
        session = patch_db
        baseline = [0.05, 0.08, 0.03, 0.06, -0.02, 0.04, 0.07, 0.01,
                    0.05, 0.09, 0.03, -0.01, 0.06, 0.04, 0.08, 0.02,
                    0.05, 0.07, 0.03, 0.06]
        for i, val in enumerate(baseline):
            raw_id = _make_raw(session, data_type="news", days_ago=len(baseline) - i)
            _make_feature(session, raw_id, "sentiment_score", val)

        spike_raw_id = _make_raw(session, data_type="news", days_ago=0)
        _make_feature(session, spike_raw_id, "sentiment_score", -0.95)
        session.commit()

        result = detect_sentiment_anomalies("lng")
        assert len(result) >= 1
        meta = json.loads(result[-1].metadata_json)
        assert "z_score" in meta
        assert "compound_score" in meta
        assert "raw_ingestion_id" in meta
