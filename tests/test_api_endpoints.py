"""
Tests for FastAPI endpoints.

Uses FastAPI's dependency_overrides to inject an in-memory SQLite session
without touching settings or the real DB.  Auth is disabled because
MCEI_API_KEY is not set in the test environment.
"""

from contextlib import contextmanager
from datetime import datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, pool
from sqlalchemy.orm import sessionmaker

from shared.models import AnomalyEvent, Base


# ── Shared in-memory DB fixture ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def mem_engine():
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
def db(mem_engine):
    SessionLocal = sessionmaker(bind=mem_engine)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()


@pytest.fixture(scope="module")
def client(mem_engine):
    """
    TestClient with get_db overridden to use the in-memory engine.
    api.main is imported here so module-level setup runs once with real settings
    (MCEI_API_KEY is empty in test env so auth is disabled).
    """
    from api.dependencies import get_db
    from api.main import app

    SessionLocal = sessionmaker(bind=mem_engine)

    def override_get_db():
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_has_required_fields(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "ok"
        assert "table_counts" in data
        assert "uptime_seconds" in data
        assert "version" in data
        assert "last_job_success" in data

    def test_health_table_counts_all_tables_present(self, client):
        resp = client.get("/health")
        counts = resp.json()["table_counts"]
        expected_tables = [
            "raw_ingestion", "processed_features", "anomaly_events",
            "embeddings_cache", "signal_alerts", "causality_reports",
            "prediction_evaluations", "learning_updates",
        ]
        for table in expected_tables:
            assert table in counts, f"Missing table count: {table}"

    def test_health_table_counts_are_integers(self, client):
        resp = client.get("/health")
        counts = resp.json()["table_counts"]
        for table, count in counts.items():
            assert isinstance(count, int), f"{table} count is not int: {count}"


# ── /anomalies ────────────────────────────────────────────────────────────────

class TestAnomaliesEndpoint:

    def _insert_event(self, db, commodity="lng", anomaly_type="price_spike",
                      severity=3.5, status="new"):
        event = AnomalyEvent(
            commodity=commodity,
            anomaly_type=anomaly_type,
            severity=severity,
            detected_at=datetime.utcnow(),
            source_ids="[1]",
            status=status,
            metadata_json="{}",
        )
        db.add(event)
        db.commit()
        return event

    def test_anomalies_empty_list(self, client, db):
        resp = client.get("/anomalies")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_anomalies_returns_inserted_row(self, client, db):
        event = self._insert_event(db, commodity="lng")
        resp = client.get("/anomalies?commodity=lng&anomaly_type=price_spike")
        assert resp.status_code == 200
        data = resp.json()
        ids = [d["id"] for d in data]
        assert event.id in ids

    def test_anomalies_filter_by_commodity(self, client, db):
        self._insert_event(db, commodity="copper")
        resp = client.get("/anomalies?commodity=copper")
        assert resp.status_code == 200
        data = resp.json()
        assert all(d["commodity"] == "copper" for d in data)

    def test_anomalies_filter_by_status(self, client, db):
        self._insert_event(db, commodity="lng", status="processed")
        resp = client.get("/anomalies?status=processed")
        assert resp.status_code == 200
        data = resp.json()
        assert all(d["status"] == "processed" for d in data)

    def test_anomalies_filter_by_type(self, client, db):
        self._insert_event(db, anomaly_type="sentiment_shift")
        resp = client.get("/anomalies?anomaly_type=sentiment_shift")
        assert resp.status_code == 200
        data = resp.json()
        assert all(d["anomaly_type"] == "sentiment_shift" for d in data)

    def test_get_anomaly_by_id(self, client, db):
        event = self._insert_event(db, commodity="soybeans")
        resp = client.get(f"/anomalies/{event.id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == event.id
        assert data["commodity"] == "soybeans"

    def test_get_anomaly_by_id_404(self, client):
        resp = client.get("/anomalies/9999999")
        assert resp.status_code == 404
        assert "detail" in resp.json()

    def test_anomalies_response_schema(self, client, db):
        """All required fields are present in each returned anomaly."""
        self._insert_event(db)
        resp = client.get("/anomalies?limit=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        row = data[0]
        for field in ["id", "commodity", "anomaly_type", "severity",
                      "status", "detected_at", "source_ids"]:
            assert field in row, f"Missing field: {field}"


# ── Root endpoint ─────────────────────────────────────────────────────────────

class TestRootEndpoint:

    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_has_service_name(self, client):
        resp = client.get("/")
        assert resp.json()["service"] == "MCEI API"

    def test_root_lists_endpoints(self, client):
        resp = client.get("/")
        endpoints = resp.json()["endpoints"]
        assert "/health" in endpoints
        assert "/anomalies" in endpoints
