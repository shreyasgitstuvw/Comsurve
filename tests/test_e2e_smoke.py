"""
End-to-end smoke tests (G.8).

These tests verify that the complete pipeline can be imported and key
entry-point functions execute without network calls, using an in-memory
SQLite DB. They do NOT call Gemini, yfinance, or any external API.

Purpose: catch import errors, wiring bugs, and fatal startup crashes
before they reach production.
"""

import json
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, pool, text
from sqlalchemy.orm import sessionmaker

from shared.models import (
    AnomalyEvent,
    Base,
    EmbeddingCache,
    RawIngestion,
    SignalAlert,
    pack_vector,
)


# ── Shared in-memory DB fixture ───────────────────────────────────────────────

@pytest.fixture()
def smoke_engine():
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=pool.StaticPool,
    )
    Base.metadata.create_all(eng)
    yield eng
    Base.metadata.drop_all(eng)
    eng.dispose()


@pytest.fixture()
def smoke_session(smoke_engine):
    Session = sessionmaker(bind=smoke_engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


def _fake_session_factory(session):
    @contextmanager
    def _fake():
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
    return _fake


# ── Stage 1: Anomaly Detection ────────────────────────────────────────────────

class TestAnomalyDetectionSmoke:

    def test_run_anomaly_detection_returns_dict(self, smoke_session):
        fake = _fake_session_factory(smoke_session)
        with patch("processor.anomaly_detector.get_session", side_effect=fake):
            from processor.anomaly_detector import run_anomaly_detection
            result = run_anomaly_detection()
        assert isinstance(result, dict)
        assert "total_new_anomalies" in result
        assert result["total_new_anomalies"] == 0  # empty DB → no anomalies

    def test_detect_price_anomalies_empty_db(self, smoke_session):
        fake = _fake_session_factory(smoke_session)
        with patch("processor.anomaly_detector.get_session", side_effect=fake):
            from processor.anomaly_detector import detect_price_anomalies
            result = detect_price_anomalies("lng")
        assert result == []

    def test_detect_ais_anomalies_empty_db(self, smoke_session):
        fake = _fake_session_factory(smoke_session)
        with patch("processor.ais_anomaly_detector.get_session", side_effect=fake):
            from processor.ais_anomaly_detector import detect_ais_anomalies
            result = detect_ais_anomalies()
        assert result == []

    def test_detect_satellite_anomalies_empty_db(self, smoke_session):
        fake = _fake_session_factory(smoke_session)
        with patch("processor.satellite_anomaly_detector.get_session", side_effect=fake):
            from processor.satellite_anomaly_detector import detect_satellite_anomalies
            result = detect_satellite_anomalies()
        assert result == []


# ── Stage 2: AI Engine (embedding) ────────────────────────────────────────────

class TestAIEngineSmoke:

    def test_embed_run_empty_db_returns_dict(self, smoke_session):
        fake = _fake_session_factory(smoke_session)
        mock_client = MagicMock()
        mock_client.batch_embed.return_value = []
        with patch("ai_engine.ai_engine_runner.get_session", side_effect=fake):
            with patch("ai_engine.ai_engine_runner.GeminiClient", return_value=mock_client):
                with patch("ai_engine.ai_engine_runner.QdrantManager"):
                    from ai_engine.ai_engine_runner import run
                    result = run()
        assert isinstance(result, dict)
        assert result.get("processed", 0) == 0

    def test_embed_run_processes_pending_anomaly(self, smoke_session):
        anomaly = AnomalyEvent(
            commodity="lng",
            anomaly_type="price_spike",
            severity=3.2,
            detected_at=datetime.utcnow(),
            status="new",
            source_ids="[]",
            metadata_json="{}",
        )
        smoke_session.add(anomaly)
        smoke_session.commit()

        fake = _fake_session_factory(smoke_session)
        mock_client = MagicMock()
        mock_client.batch_embed.return_value = [[0.1] * 3072]
        mock_qdrant = MagicMock()

        with patch("ai_engine.ai_engine_runner.get_session", side_effect=fake):
            with patch("ai_engine.ai_engine_runner.GeminiClient", return_value=mock_client):
                with patch("ai_engine.ai_engine_runner.QdrantManager", return_value=mock_qdrant):
                    from ai_engine.ai_engine_runner import run
                    result = run()

        assert result.get("processed", 0) >= 1


# ── Stage 3: Signal Correlation ───────────────────────────────────────────────

class TestSignalCorrelatorSmoke:

    def test_run_signal_correlation_empty_db(self, smoke_session):
        fake = _fake_session_factory(smoke_session)
        mock_qdrant = MagicMock()
        mock_qdrant.search_similar.return_value = []

        with patch("ai_engine.signal_correlator.get_session", side_effect=fake):
            with patch("ai_engine.signal_correlator.QdrantManager", return_value=mock_qdrant):
                from ai_engine.signal_correlator import run_signal_correlation
                result = run_signal_correlation()

        assert isinstance(result, dict)
        assert "alerts_created" in result
        assert result["alerts_created"] == 0


# ── Stage 4: Monitoring Window ────────────────────────────────────────────────

class TestMonitoringWindowSmoke:

    def test_run_monitoring_window_empty_db(self, smoke_session):
        fake = _fake_session_factory(smoke_session)
        with patch("processor.monitoring_window.get_session", side_effect=fake):
            from processor.monitoring_window import run_monitoring_window_check
            result = run_monitoring_window_check()
        assert isinstance(result, dict)
        assert result["open_alerts_checked"] == 0


# ── Stage 5: Causality Engine ─────────────────────────────────────────────────

class TestCausalityEngineSmoke:

    def test_run_causality_engine_empty_db(self, smoke_session, tmp_path):
        fake = _fake_session_factory(smoke_session)
        mock_client = MagicMock()
        mock_client.generate_text.return_value = json.dumps({
            "cause": "test", "cause_category": "unknown",
            "mechanism": "test", "confidence": 0.7,
            "supporting_signals": [], "historical_precedents": [], "summary": "test",
        })

        with patch("ai_engine.causality_engine.get_session", side_effect=fake):
            with patch("ai_engine.causality_engine.GeminiClient", return_value=mock_client):
                with patch("ai_engine.causality_engine.REPORTS_DIR", str(tmp_path)):
                    from ai_engine.causality_engine import run_causality_engine
                    result = run_causality_engine()

        assert isinstance(result, dict)
        assert result["reports_generated"] == 0  # no completed monitoring windows


# ── Scheduler jobs import cleanly ─────────────────────────────────────────────

class TestSchedulerJobsImport:

    def test_all_job_functions_importable(self):
        from scheduler.jobs import (
            job_ai_engine,
            job_ais,
            job_causality,
            job_cleanup,
            job_evaluation,
            job_news,
            job_prediction,
            job_price_historical,
            job_price_realtime,
            job_processor,
            job_qdrant_backup,
            job_rail,
            job_satellite,
        )
        for fn in [job_ai_engine, job_ais, job_causality, job_cleanup,
                   job_evaluation, job_news, job_prediction, job_price_historical,
                   job_price_realtime, job_processor, job_qdrant_backup,
                   job_rail, job_satellite]:
            assert callable(fn)

    def test_cleanup_job_runs_against_empty_db(self, smoke_session):
        fake = _fake_session_factory(smoke_session)
        with patch("scheduler.cleanup.get_session", side_effect=fake):
            from scheduler.cleanup import run_cleanup
            result = run_cleanup()
        assert result["deleted_raw_ingestion"] == 0
        assert result["deleted_processed_features"] == 0


# ── API health endpoint ────────────────────────────────────────────────────────

class TestAPIHealthSmoke:

    def test_health_endpoint_importable(self):
        from api.routers.health import router
        assert router is not None
