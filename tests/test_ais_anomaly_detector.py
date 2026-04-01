"""
Tests for processor/ais_anomaly_detector.py.

Covers:
  - ais_vessel_drop  : vessel count falls >50% below 7-day average
  - ais_port_idle    : moored_ratio > 0.85 with sufficient vessel count
  - deduplication    : same port not flagged twice within 6h

Uses the patch_db fixture from conftest.py which patches get_session
at the point of use in processor.ais_anomaly_detector.
"""

import json
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from processor.ais_anomaly_detector import (
    IDLE_MOORED_RATIO,
    MIN_BASELINE_POINTS,
    VESSEL_DROP_THRESHOLD,
    detect_ais_anomalies,
)
from shared.models import AnomalyEvent, ProcessedFeature, RawIngestion


# ── Minimal port registry for testing ─────────────────────────────────────────

_TEST_PORT = "sabine_pass"
_TEST_PORT_INFO = {
    "name": "Sabine Pass LNG",
    "commodity": "lng",
    "bbox": [29.68, -93.92, 29.80, -93.75],
    "country": "US",
}
_PORT_REGISTRY = {_TEST_PORT: _TEST_PORT_INFO}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _insert_vessel_count(session, port_slug, count, days_ago=0, hours_ago=0):
    ts = datetime.utcnow() - timedelta(days=days_ago, hours=hours_ago)
    raw = RawIngestion(
        source="ais",
        commodity=_TEST_PORT_INFO["commodity"],
        symbol=port_slug,
        timestamp=ts,
        data_type="ais",
        raw_json="{}",
    )
    session.add(raw)
    session.flush()
    pf = ProcessedFeature(
        raw_ingestion_id=raw.id,
        commodity=_TEST_PORT_INFO["commodity"],
        feature_type="vessel_count",
        value=float(count),
    )
    session.add(pf)
    session.flush()
    return raw.id


def _insert_moored_ratio(session, port_slug, ratio, existing_raw_id):
    pf = ProcessedFeature(
        raw_ingestion_id=existing_raw_id,
        commodity=_TEST_PORT_INFO["commodity"],
        feature_type="moored_ratio",
        value=ratio,
    )
    session.add(pf)
    session.flush()


def _run_detector(session):
    """Run detect_ais_anomalies with the test port registry."""
    with patch("ingestion.ais.port_registry.PORT_REGISTRY", _PORT_REGISTRY):
        return detect_ais_anomalies()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDetectAisAnomalies:

    def test_returns_empty_when_no_data(self, patch_db):
        result = _run_detector(patch_db)
        assert result == []

    def test_returns_empty_when_baseline_insufficient(self, patch_db):
        session = patch_db
        # Only 2 baseline points — below MIN_BASELINE_POINTS (3)
        for i in range(MIN_BASELINE_POINTS - 1):
            _insert_vessel_count(session, _TEST_PORT, 10, days_ago=i + 1)
        # Latest (current) reading — low vessel count
        _insert_vessel_count(session, _TEST_PORT, 1, days_ago=0)
        session.commit()

        result = _run_detector(session)
        drop_anomalies = [a for a in result if a.anomaly_type == "ais_vessel_drop"]
        assert drop_anomalies == []

    def test_detects_vessel_drop_anomaly(self, patch_db):
        session = patch_db
        # Baseline: 5 days of 10 vessels/day (> MIN_BASELINE_POINTS)
        for i in range(5):
            _insert_vessel_count(session, _TEST_PORT, 10, days_ago=i + 1)
        # Current: only 2 vessels (80% drop — well above 50% VESSEL_DROP_THRESHOLD)
        _insert_vessel_count(session, _TEST_PORT, 2, days_ago=0)
        session.commit()

        result = _run_detector(session)
        drop_anomalies = [a for a in result if a.anomaly_type == "ais_vessel_drop"]
        assert len(drop_anomalies) >= 1

        anomaly = drop_anomalies[0]
        assert anomaly.commodity == "lng"
        assert anomaly.severity > 0
        meta = json.loads(anomaly.metadata_json)
        assert meta["port_slug"] == _TEST_PORT
        assert meta["drop_pct"] >= 50.0

    def test_no_anomaly_for_small_vessel_drop(self, patch_db):
        session = patch_db
        # Baseline: 10 vessels/day
        for i in range(5):
            _insert_vessel_count(session, _TEST_PORT, 10, days_ago=i + 1)
        # Current: 8 vessels — only 20% drop, below 50% threshold
        _insert_vessel_count(session, _TEST_PORT, 8, days_ago=0)
        session.commit()

        result = _run_detector(session)
        drop_anomalies = [a for a in result if a.anomaly_type == "ais_vessel_drop"]
        assert drop_anomalies == []

    def test_no_anomaly_when_zero_baseline(self, patch_db):
        """Division-by-zero guard: if avg_count == 0, no drop anomaly."""
        session = patch_db
        for i in range(5):
            _insert_vessel_count(session, _TEST_PORT, 0, days_ago=i + 1)
        _insert_vessel_count(session, _TEST_PORT, 0, days_ago=0)
        session.commit()

        result = _run_detector(session)
        drop_anomalies = [a for a in result if a.anomaly_type == "ais_vessel_drop"]
        assert drop_anomalies == []

    def test_detects_port_idle_anomaly(self, patch_db):
        session = patch_db
        # Baseline vessel counts
        for i in range(MIN_BASELINE_POINTS + 1):
            _insert_vessel_count(session, _TEST_PORT, 10, days_ago=i + 1)
        # Current: 5 vessels present (count >= 2)
        latest_rid = _insert_vessel_count(session, _TEST_PORT, 5, days_ago=0)
        # 90% moored (above IDLE_MOORED_RATIO=0.85)
        _insert_moored_ratio(session, _TEST_PORT, 0.90, existing_raw_id=latest_rid)
        session.commit()

        result = _run_detector(session)
        idle_anomalies = [a for a in result if a.anomaly_type == "ais_port_idle"]
        assert len(idle_anomalies) >= 1
        # Formula: 2.0 + (moored_ratio - IDLE_MOORED_RATIO) / (1.0 - IDLE_MOORED_RATIO) * 8.0
        expected = 2.0 + (0.90 - IDLE_MOORED_RATIO) / (1.0 - IDLE_MOORED_RATIO) * 8.0
        assert idle_anomalies[0].severity == pytest.approx(expected, rel=0.01)

    def test_no_port_idle_when_moored_ratio_below_threshold(self, patch_db):
        session = patch_db
        for i in range(MIN_BASELINE_POINTS + 1):
            _insert_vessel_count(session, _TEST_PORT, 10, days_ago=i + 1)
        latest_rid = _insert_vessel_count(session, _TEST_PORT, 5, days_ago=0)
        _insert_moored_ratio(session, _TEST_PORT, 0.70, existing_raw_id=latest_rid)  # below 0.85
        session.commit()

        result = _run_detector(session)
        idle_anomalies = [a for a in result if a.anomaly_type == "ais_port_idle"]
        assert idle_anomalies == []

    def test_vessel_drop_severity_scales_with_drop_pct(self, patch_db):
        """100% drop (0 vessels) → severity ≈ 10.0."""
        session = patch_db
        for i in range(5):
            _insert_vessel_count(session, _TEST_PORT, 10, days_ago=i + 1)
        _insert_vessel_count(session, _TEST_PORT, 0, days_ago=0)
        session.commit()

        result = _run_detector(session)
        drop_anomalies = [a for a in result if a.anomaly_type == "ais_vessel_drop"]
        assert len(drop_anomalies) >= 1
        assert drop_anomalies[0].severity == pytest.approx(10.0, rel=0.01)

    def test_vessel_drop_deduplication(self, patch_db):
        """Same port not flagged twice within the dedup window."""
        session = patch_db
        for i in range(5):
            _insert_vessel_count(session, _TEST_PORT, 10, days_ago=i + 1)
        _insert_vessel_count(session, _TEST_PORT, 2, days_ago=0)
        session.commit()

        result1 = _run_detector(session)
        drop1 = [a for a in result1 if a.anomaly_type == "ais_vessel_drop"]
        assert len(drop1) >= 1

        # Second run — should be deduplicated (existing anomaly within 6h)
        result2 = _run_detector(session)
        drop2 = [a for a in result2 if a.anomaly_type == "ais_vessel_drop"]
        assert drop2 == []
