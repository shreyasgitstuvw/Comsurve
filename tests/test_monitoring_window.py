"""
Tests for processor/monitoring_window.py.

Covers:
  - _get_price_near: primary lookup, raw_ingestion fallback, no-data case
  - run_monitoring_window_check: no-op when no open alerts, checkpoint filling,
    partial completion, full completion (monitoring_complete=1)
"""

import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from sqlalchemy import text

from processor.monitoring_window import (
    CHECKPOINTS,
    _get_price_near,
    run_monitoring_window_check,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fake_session_factory(db_session):
    @contextmanager
    def _fake():
        try:
            yield db_session
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise
    return _fake


def _insert_raw(db_session, commodity, data_type, timestamp, raw_json, symbol="TEST"):
    db_session.execute(
        text("""
            INSERT INTO raw_ingestion
                (commodity, data_type, source, symbol, timestamp, raw_json, ingested_at, processed)
            VALUES (:c, :dt, :src, :sym, :ts, :rj, :ia, 0)
        """),
        {"c": commodity, "dt": data_type, "src": "test", "sym": symbol,
         "ts": timestamp, "rj": json.dumps(raw_json), "ia": datetime.utcnow()},
    )
    return db_session.execute(text("SELECT last_insert_rowid()")).scalar()


def _insert_processed(db_session, raw_id, commodity, feature_type, value, ts):
    db_session.execute(
        text("""
            INSERT INTO processed_features
                (raw_ingestion_id, commodity, feature_type, value, computed_at)
            VALUES (:rid, :c, :ft, :v, :ts)
        """),
        {"rid": raw_id, "c": commodity, "ft": feature_type, "v": value, "ts": ts},
    )


def _insert_anomaly(db_session, commodity="lng"):
    db_session.execute(
        text("""
            INSERT INTO anomaly_events
                (commodity, anomaly_type, severity, detected_at, source_ids, status)
            VALUES (:c, 'price_spike', 2.0, CURRENT_TIMESTAMP, '[]', 'new')
        """),
        {"c": commodity},
    )
    return db_session.execute(text("SELECT last_insert_rowid()")).scalar()


def _insert_alert(db_session, commodity, created_at,
                  p1w=None, p2w=None, p1m=None, monitoring_complete=0):
    anomaly_id = _insert_anomaly(db_session, commodity)
    db_session.execute(
        text("""
            INSERT INTO signal_alerts
                (anomaly_event_id, commodity, alert_type, correlated_anomaly_ids,
                 similarity_scores, created_at,
                 price_1w, price_2w, price_1m, monitoring_complete)
            VALUES (:aeid, :c, 'novel_event', '[]', '[]', :ts, :p1w, :p2w, :p1m, :mc)
        """),
        {
            "aeid": anomaly_id,
            "c": commodity,
            "ts": created_at,
            "p1w": p1w,
            "p2w": p2w,
            "p1m": p1m,
            "mc": monitoring_complete,
        },
    )
    return db_session.execute(text("SELECT last_insert_rowid()")).scalar()


# ══════════════════════════════════════════════════════════════════════════════
# _get_price_near
# ══════════════════════════════════════════════════════════════════════════════

class TestGetPriceNear:

    def _run(self, db_session, commodity, target_dt, tolerance_hours=12):
        fake = _fake_session_factory(db_session)
        with patch("processor.monitoring_window.get_session", side_effect=fake):
            return _get_price_near(commodity, target_dt, tolerance_hours)

    def test_returns_none_when_no_data(self, db_session):
        result = self._run(db_session, "lng", datetime.utcnow())
        assert result is None

    def test_returns_price_from_processed_features(self, db_session):
        target = datetime(2024, 6, 1, 12, 0, 0)
        raw_id = _insert_raw(db_session, "lng", "price_realtime", target, {"close": 3.45})
        _insert_processed(db_session, raw_id, "lng", "price", 3.45, target)
        db_session.commit()

        result = self._run(db_session, "lng", target)
        assert result == pytest.approx(3.45)

    def test_returns_none_when_price_outside_tolerance(self, db_session):
        target = datetime(2024, 6, 1, 12, 0, 0)
        far_ts = target + timedelta(hours=25)  # outside 12h window
        raw_id = _insert_raw(db_session, "lng", "price_realtime", far_ts, {"close": 3.45})
        _insert_processed(db_session, raw_id, "lng", "price", 3.45, far_ts)
        db_session.commit()

        result = self._run(db_session, "lng", target, tolerance_hours=12)
        assert result is None

    def test_falls_back_to_raw_ingestion_close_field(self, db_session):
        target = datetime(2024, 6, 1, 12, 0, 0)
        _insert_raw(db_session, "copper", "price_realtime", target, {"close": 4.12})
        db_session.commit()

        result = self._run(db_session, "copper", target)
        assert result == pytest.approx(4.12)

    def test_falls_back_to_raw_ingestion_value_field(self, db_session):
        target = datetime(2024, 6, 1, 12, 0, 0)
        _insert_raw(db_session, "soybeans", "price_historical", target, {"value": 1175.5})
        db_session.commit()

        result = self._run(db_session, "soybeans", target)
        assert result == pytest.approx(1175.5)

    def test_processed_features_takes_priority_over_raw(self, db_session):
        target = datetime(2024, 6, 1, 12, 0, 0)
        raw_id = _insert_raw(db_session, "lng", "price_realtime", target, {"close": 3.45})
        _insert_processed(db_session, raw_id, "lng", "price", 3.45, target)
        # Also insert a different raw row with different price (different symbol to avoid unique conflict)
        _insert_raw(db_session, "lng", "price_realtime", target, {"close": 9.99}, symbol="TEST2")
        db_session.commit()

        result = self._run(db_session, "lng", target)
        assert result == pytest.approx(3.45)  # processed wins

    def test_ignores_non_price_feature_types(self, db_session):
        target = datetime(2024, 6, 1, 12, 0, 0)
        raw_id = _insert_raw(db_session, "lng", "news", target, {})
        _insert_processed(db_session, raw_id, "lng", "sentiment_score", 0.5, target)
        db_session.commit()

        result = self._run(db_session, "lng", target)
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# run_monitoring_window_check
# ══════════════════════════════════════════════════════════════════════════════

class TestRunMonitoringWindowCheck:

    def _run(self, db_session):
        fake = _fake_session_factory(db_session)
        with patch("processor.monitoring_window.get_session", side_effect=fake):
            with patch("processor.monitoring_window._get_price_near") as mock_price:
                # Return a fixed price for any commodity / timestamp
                mock_price.return_value = 3.50
                result = run_monitoring_window_check()
        return result

    def _run_with_price_fn(self, db_session, price_fn):
        """Run with a custom price lookup function."""
        fake = _fake_session_factory(db_session)
        with patch("processor.monitoring_window.get_session", side_effect=fake):
            with patch("processor.monitoring_window._get_price_near", side_effect=price_fn):
                return run_monitoring_window_check()

    def test_returns_summary_dict(self, db_session):
        result = self._run(db_session)
        assert "alerts_updated" in result
        assert "monitoring_complete" in result
        assert "open_alerts_checked" in result

    def test_no_op_when_no_open_alerts(self, db_session):
        result = self._run(db_session)
        assert result["alerts_updated"] == 0
        assert result["monitoring_complete"] == 0
        assert result["open_alerts_checked"] == 0

    def test_skips_already_complete_alerts(self, db_session):
        old_ts = datetime.utcnow() - timedelta(days=40)
        _insert_alert(db_session, "lng", old_ts,
                      p1w=3.0, p2w=3.1, p1m=3.2, monitoring_complete=1)
        db_session.commit()

        result = self._run(db_session)
        assert result["open_alerts_checked"] == 0

    def test_fills_1w_price_when_due(self, db_session):
        # Alert created 8 days ago — 1w checkpoint (7 days) should fire
        created = datetime.utcnow() - timedelta(days=8)
        alert_id = _insert_alert(db_session, "lng", created)
        db_session.commit()

        self._run(db_session)

        row = db_session.execute(
            text("SELECT price_1w FROM signal_alerts WHERE id = :id"),
            {"id": alert_id},
        ).fetchone()
        assert row[0] == pytest.approx(3.50)

    def test_does_not_fill_future_checkpoint(self, db_session):
        # Alert created 1 day ago — no checkpoint due yet (1w = 7d, 2w = 14d, 1m = 30d)
        created = datetime.utcnow() - timedelta(days=1)
        alert_id = _insert_alert(db_session, "copper", created)
        db_session.commit()

        self._run(db_session)

        row = db_session.execute(
            text("SELECT price_1w, price_2w, price_1m FROM signal_alerts WHERE id = :id"),
            {"id": alert_id},
        ).fetchone()
        assert row[0] is None
        assert row[1] is None
        assert row[2] is None

    def test_sets_monitoring_complete_when_all_filled(self, db_session):
        # Alert created 35 days ago — all 3 checkpoints due
        created = datetime.utcnow() - timedelta(days=35)
        alert_id = _insert_alert(db_session, "soybeans", created)
        db_session.commit()

        self._run(db_session)

        row = db_session.execute(
            text("SELECT monitoring_complete FROM signal_alerts WHERE id = :id"),
            {"id": alert_id},
        ).fetchone()
        assert row[0] == 1

    def test_does_not_complete_when_price_unavailable(self, db_session):
        created = datetime.utcnow() - timedelta(days=35)
        alert_id = _insert_alert(db_session, "lng", created)
        db_session.commit()

        # Price lookup always returns None
        fake = _fake_session_factory(db_session)
        with patch("processor.monitoring_window.get_session", side_effect=fake):
            with patch("processor.monitoring_window._get_price_near", return_value=None):
                run_monitoring_window_check()

        row = db_session.execute(
            text("SELECT monitoring_complete FROM signal_alerts WHERE id = :id"),
            {"id": alert_id},
        ).fetchone()
        assert row[0] == 0

    def test_does_not_overwrite_existing_checkpoints(self, db_session):
        # Alert has price_1w already set; should not be overwritten
        created = datetime.utcnow() - timedelta(days=35)
        alert_id = _insert_alert(db_session, "lng", created, p1w=2.99)
        db_session.commit()

        self._run(db_session)

        row = db_session.execute(
            text("SELECT price_1w FROM signal_alerts WHERE id = :id"),
            {"id": alert_id},
        ).fetchone()
        assert row[0] == pytest.approx(2.99)  # original value preserved

    def test_multiple_alerts_processed_independently(self, db_session):
        old = datetime.utcnow() - timedelta(days=35)
        new = datetime.utcnow() - timedelta(days=1)
        alert_old = _insert_alert(db_session, "lng", old)
        alert_new = _insert_alert(db_session, "copper", new)
        db_session.commit()

        self._run(db_session)

        old_row = db_session.execute(
            text("SELECT monitoring_complete FROM signal_alerts WHERE id = :id"),
            {"id": alert_old},
        ).fetchone()
        new_row = db_session.execute(
            text("SELECT price_1w FROM signal_alerts WHERE id = :id"),
            {"id": alert_new},
        ).fetchone()
        assert old_row[0] == 1         # old alert completed
        assert new_row[0] is None      # new alert untouched

    def test_checkpoints_dict_has_three_entries(self):
        assert len(CHECKPOINTS) == 3
        assert "price_1w" in CHECKPOINTS
        assert "price_2w" in CHECKPOINTS
        assert "price_1m" in CHECKPOINTS
