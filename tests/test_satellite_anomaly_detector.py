"""
Tests for processor/satellite_anomaly_detector.py — detect_satellite_anomalies().

Anomaly types under test:
  1. satellite_scene_gap    — port seen historically but no S1 scene in past 7 days
  2. satellite_cloud_block  — soybeans port mean cloud cover > 85% over 14 days
  3. satellite_aircraft_surge — latest aircraft_count >= 2× 7-day rolling mean

Isolation strategy
------------------
  - All DB access goes through the in-memory SQLite `db_session` fixture from conftest.py.
  - `get_session` is patched at "processor.satellite_anomaly_detector.get_session"
    using a local contextmanager that yields db_session and commits on success.
  - `_get_port_slugs_for_commodity` is patched at
    "processor.satellite_anomaly_detector._get_port_slugs_for_commodity"
    to avoid importing from ingestion.ais.port_registry.
"""

import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from shared.models import AnomalyEvent, ProcessedFeature, RawIngestion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw(db_session, *, data_type="satellite", commodity="lng",
               source="sentinel", symbol="s1", timestamp=None, processed=True):
    """Insert a minimal RawIngestion row and return it (flushed, id assigned)."""
    if timestamp is None:
        timestamp = datetime.utcnow()
    row = RawIngestion(
        source=source,
        commodity=commodity,
        symbol=symbol,
        timestamp=timestamp,
        data_type=data_type,
        raw_json="{}",
        ingested_at=datetime.utcnow(),
        processed=processed,
    )
    db_session.add(row)
    db_session.flush()
    return row


def _make_feature(db_session, *, raw_id, commodity="lng",
                   feature_type, value=None, value_json=None, window=None):
    """Insert a ProcessedFeature row and return it (flushed)."""
    row = ProcessedFeature(
        raw_ingestion_id=raw_id,
        commodity=commodity,
        feature_type=feature_type,
        value=value,
        value_json=value_json,
        window=window,
        computed_at=datetime.utcnow(),
    )
    db_session.add(row)
    db_session.flush()
    return row


def _fake_session_factory(db_session):
    """
    Return a contextmanager that behaves like get_session() but uses db_session.
    Commits on clean exit; rolls back on exception.
    """
    @contextmanager
    def _fake_session():
        try:
            yield db_session
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise
    return _fake_session


def _run_detector(db_session, mock_port_slugs=None):
    """
    Patch get_session and (optionally) _get_port_slugs_for_commodity,
    then call detect_satellite_anomalies().
    Returns the list of anomalies produced.
    """
    fake_session = _fake_session_factory(db_session)
    slugs = mock_port_slugs if mock_port_slugs is not None else []

    with patch(
        "processor.satellite_anomaly_detector.get_session",
        side_effect=fake_session,
    ), patch(
        "processor.satellite_anomaly_detector._get_port_slugs_for_commodity",
        return_value=slugs,
    ):
        from processor.satellite_anomaly_detector import detect_satellite_anomalies
        return detect_satellite_anomalies()


# ---------------------------------------------------------------------------
# Test 1 — Empty database returns empty list
# ---------------------------------------------------------------------------

class TestEmptyDatabase:
    def test_no_features_returns_empty_list(self, db_session):
        result = _run_detector(db_session)
        assert result == []


# ---------------------------------------------------------------------------
# Test 2 — satellite_scene_gap detection
# ---------------------------------------------------------------------------

class TestSatelliteSceneGap:
    """
    A port with S1 scenes OLDER than 7 days but NONE in the past 7 days
    should trigger a satellite_scene_gap anomaly with severity 3.0.
    """

    def _insert_old_scene(self, db_session, port_slug="test_port"):
        """Insert a satellite_s1_scene feature 30 days ago (outside 7-day window)."""
        raw = _make_raw(
            db_session,
            data_type="satellite",
            commodity="lng",
            timestamp=datetime.utcnow() - timedelta(days=30),
        )
        _make_feature(
            db_session,
            raw_id=raw.id,
            commodity="lng",
            feature_type="satellite_s1_scene",
            value_json=json.dumps({"port_slug": port_slug}),
        )
        db_session.commit()
        return raw

    def test_scene_gap_detected(self, db_session):
        self._insert_old_scene(db_session, port_slug="test_port")
        result = _run_detector(db_session)

        gap_anomalies = [a for a in result if a["type"] == "satellite_scene_gap"]
        assert len(gap_anomalies) >= 1

    def test_scene_gap_severity_is_3(self, db_session):
        self._insert_old_scene(db_session, port_slug="test_port")
        result = _run_detector(db_session)

        gap = next(a for a in result if a["type"] == "satellite_scene_gap")
        assert gap["severity"] == pytest.approx(3.0)

    def test_scene_gap_port_slug_in_metadata(self, db_session):
        self._insert_old_scene(db_session, port_slug="test_port")
        result = _run_detector(db_session)

        gap = next(a for a in result if a["type"] == "satellite_scene_gap")
        meta = {"port_slug": gap["port"]}
        assert meta.get("port_slug") == "test_port"

    def test_recent_scene_does_not_trigger_gap(self, db_session):
        """A scene timestamped 2 days ago is inside the 7-day window — no gap."""
        raw = _make_raw(
            db_session,
            data_type="satellite",
            commodity="lng",
            timestamp=datetime.utcnow() - timedelta(days=2),
        )
        _make_feature(
            db_session,
            raw_id=raw.id,
            commodity="lng",
            feature_type="satellite_s1_scene",
            value_json=json.dumps({"port_slug": "active_port"}),
        )
        db_session.commit()
        result = _run_detector(db_session)

        gap_anomalies = [a for a in result if a["type"] == "satellite_scene_gap"]
        assert all(
            a["port"] != "active_port"
            for a in gap_anomalies
        )

    def test_deduplication_no_double_insert(self, db_session):
        """Running the detector twice should not insert a second anomaly for the same port."""
        self._insert_old_scene(db_session, port_slug="dup_port")

        # First run
        _run_detector(db_session)

        # Insert an existing anomaly_events record within 24h to simulate deduplication
        existing = AnomalyEvent(
            commodity="lng",
            anomaly_type="satellite_scene_gap",
            severity=3.0,
            detected_at=datetime.utcnow() - timedelta(hours=1),
            source_ids="[]",
            status="new",
            metadata_json=json.dumps({"port_slug": "dup_port"}),
        )
        db_session.add(existing)
        db_session.commit()

        # Second run — should be deduplicated
        result2 = _run_detector(db_session)
        new_gaps = [
            a for a in result2
            if a["type"] == "satellite_scene_gap"
            and a["port"] == "dup_port"
        ]
        assert len(new_gaps) == 0, "Second run should not produce a duplicate gap anomaly"


# ---------------------------------------------------------------------------
# Test 3 — satellite_cloud_block detection
# ---------------------------------------------------------------------------

class TestSatelliteCloudBlock:
    """
    A soybeans port with mean S2 cloud cover > 85% over >=3 readings in 14 days
    triggers satellite_cloud_block. Severity = (mean - 85) / 5.
    """

    def _insert_cloud_readings(self, db_session, port_slug="soy_port",
                                cover_value=90.0, count=3):
        for i in range(count):
            raw = _make_raw(
                db_session,
                data_type="satellite",
                commodity="soybeans",
                source="sentinel2",
                symbol=f"s2_{i}",
                timestamp=datetime.utcnow() - timedelta(days=i + 1),
            )
            _make_feature(
                db_session,
                raw_id=raw.id,
                commodity="soybeans",
                feature_type="satellite_s2_cloud_cover",
                value=cover_value,
                value_json=json.dumps({"port_slug": port_slug}),
            )
        db_session.commit()

    def test_cloud_block_detected(self, db_session):
        self._insert_cloud_readings(db_session, port_slug="soy_port", cover_value=90.0)
        result = _run_detector(db_session, mock_port_slugs=["soy_port"])

        cloud_anomalies = [a for a in result if a["type"] == "satellite_cloud_block"]
        assert len(cloud_anomalies) >= 1

    def test_cloud_block_severity_formula(self, db_session):
        """Severity = (mean_cover - 85) / 5; with cover=90.0 → (90-85)/5 = 1.0."""
        self._insert_cloud_readings(db_session, port_slug="soy_port", cover_value=90.0)
        result = _run_detector(db_session, mock_port_slugs=["soy_port"])

        cloud = next(a for a in result if a["type"] == "satellite_cloud_block")
        assert cloud["severity"] == pytest.approx(1.0, abs=0.01)

    def test_cloud_block_insufficient_readings_no_alert(self, db_session):
        """Fewer than 3 readings should not trigger the anomaly."""
        self._insert_cloud_readings(db_session, port_slug="sparse_port",
                                    cover_value=95.0, count=2)
        result = _run_detector(db_session, mock_port_slugs=["sparse_port"])

        cloud_anomalies = [
            a for a in result
            if a["type"] == "satellite_cloud_block"
            and a["port"] == "sparse_port"
        ]
        assert len(cloud_anomalies) == 0

    def test_cloud_block_below_threshold_no_alert(self, db_session):
        """Mean cloud cover <= 85% should not trigger an anomaly."""
        self._insert_cloud_readings(db_session, port_slug="clear_port",
                                    cover_value=80.0, count=5)
        result = _run_detector(db_session, mock_port_slugs=["clear_port"])

        cloud_anomalies = [
            a for a in result
            if a["type"] == "satellite_cloud_block"
            and a["port"] == "clear_port"
        ]
        assert len(cloud_anomalies) == 0


# ---------------------------------------------------------------------------
# Test 4 — satellite_aircraft_surge detection
# ---------------------------------------------------------------------------

class TestSatelliteAircraftSurge:
    """
    A port where the latest aircraft_count >= 2× the 7-day rolling mean,
    the latest count >= 3, and the rolling mean >= 1.0 triggers
    satellite_aircraft_surge. Severity = ratio.
    """

    def _insert_aircraft_readings(self, db_session, port_slug="port1"):
        """
        Insert 3 historical readings (value=5.0, 6/5/4 days ago)
        and 1 latest reading (value=20.0, 1 hour ago).
        Ratio = 20 / 5 = 4.0 >= 2.0 → surge.
        """
        historical_values = [5.0, 5.0, 5.0]
        historical_offsets = [6, 5, 4]  # days ago

        for val, days_ago in zip(historical_values, historical_offsets):
            raw = _make_raw(
                db_session,
                data_type="satellite",
                commodity="lng",
                source="flightradar",
                symbol=f"aircraft_{days_ago}",
                timestamp=datetime.utcnow() - timedelta(days=days_ago),
            )
            _make_feature(
                db_session,
                raw_id=raw.id,
                commodity="lng",
                feature_type="aircraft_count",
                value=val,
                value_json=json.dumps({"port_slug": port_slug}),
            )

        # Latest (surge) reading
        raw_latest = _make_raw(
            db_session,
            data_type="satellite",
            commodity="lng",
            source="flightradar",
            symbol="aircraft_latest",
            timestamp=datetime.utcnow() - timedelta(hours=1),
        )
        _make_feature(
            db_session,
            raw_id=raw_latest.id,
            commodity="lng",
            feature_type="aircraft_count",
            value=20.0,
            value_json=json.dumps({"port_slug": port_slug}),
        )
        db_session.commit()

    def test_aircraft_surge_detected(self, db_session):
        self._insert_aircraft_readings(db_session, port_slug="port1")
        result = _run_detector(db_session)

        surge_anomalies = [a for a in result if a["type"] == "satellite_aircraft_surge"]
        assert len(surge_anomalies) >= 1

    def test_aircraft_surge_severity_is_ratio(self, db_session):
        """Severity = latest / rolling_mean = 20 / 5 = 4.0."""
        self._insert_aircraft_readings(db_session, port_slug="port1")
        result = _run_detector(db_session)

        surge = next(a for a in result if a["type"] == "satellite_aircraft_surge")
        assert surge["ratio"] == pytest.approx(4.0, abs=0.1)

    def test_aircraft_surge_insufficient_history_no_alert(self, db_session):
        """With only 2 historical observations, no anomaly should fire."""
        # Only 2 historical + 1 latest = 3 rows total, but baseline < 3
        for i in range(2):
            raw = _make_raw(
                db_session,
                data_type="satellite",
                commodity="lng",
                source="flightradar",
                symbol=f"a_hist_{i}",
                timestamp=datetime.utcnow() - timedelta(days=i + 3),
            )
            _make_feature(
                db_session,
                raw_id=raw.id,
                commodity="lng",
                feature_type="aircraft_count",
                value=5.0,
                value_json=json.dumps({"port_slug": "sparse_air_port"}),
            )

        raw_latest = _make_raw(
            db_session,
            data_type="satellite",
            commodity="lng",
            source="flightradar",
            symbol="a_latest",
            timestamp=datetime.utcnow() - timedelta(hours=1),
        )
        _make_feature(
            db_session,
            raw_id=raw_latest.id,
            commodity="lng",
            feature_type="aircraft_count",
            value=20.0,
            value_json=json.dumps({"port_slug": "sparse_air_port"}),
        )
        db_session.commit()

        result = _run_detector(db_session)
        surge_anomalies = [
            a for a in result
            if a["type"] == "satellite_aircraft_surge"
            and a["port"] == "sparse_air_port"
        ]
        assert len(surge_anomalies) == 0

    def test_aircraft_surge_below_2x_threshold_no_alert(self, db_session):
        """Ratio < 2.0 should not trigger a surge."""
        for i in range(3):
            raw = _make_raw(
                db_session,
                data_type="satellite",
                commodity="lng",
                source="flightradar",
                symbol=f"b_hist_{i}",
                timestamp=datetime.utcnow() - timedelta(days=i + 3),
            )
            _make_feature(
                db_session,
                raw_id=raw.id,
                commodity="lng",
                feature_type="aircraft_count",
                value=10.0,
                value_json=json.dumps({"port_slug": "normal_port"}),
            )

        raw_latest = _make_raw(
            db_session,
            data_type="satellite",
            commodity="lng",
            source="flightradar",
            symbol="b_latest",
            timestamp=datetime.utcnow() - timedelta(hours=1),
        )
        _make_feature(
            db_session,
            raw_id=raw_latest.id,
            commodity="lng",
            feature_type="aircraft_count",
            value=15.0,  # ratio = 1.5, below 2.0
            value_json=json.dumps({"port_slug": "normal_port"}),
        )
        db_session.commit()

        result = _run_detector(db_session)
        surge_anomalies = [
            a for a in result
            if a["type"] == "satellite_aircraft_surge"
            and a["port"] == "normal_port"
        ]
        assert len(surge_anomalies) == 0
