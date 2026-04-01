"""
Extended API endpoint tests.

Covers routers not tested in test_api_endpoints.py:
  - GET /signals, /signals/{id}
  - GET /reports, /reports/{id}
  - GET /prices/{commodity}
  - GET /evaluations, /evaluations/learning
  - GET /predictions
  - GET /metrics (Prometheus text format)

Uses the same module-scoped client + per-test db fixtures as test_api_endpoints.py.
"""

import json
from contextlib import contextmanager
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, pool
from sqlalchemy.orm import sessionmaker

from shared.models import (
    AnomalyEvent,
    Base,
    CausalityReport,
    EmbeddingCache,
    LearningUpdate,
    PredictionEvaluation,
    ProcessedFeature,
    RawIngestion,
    SignalAlert,
)


# ── In-memory DB ──────────────────────────────────────────────────────────────

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


# ── Shared insert helpers ──────────────────────────────────────────────────────

def _insert_anomaly(db, commodity="lng", anomaly_type="price_spike", status="processed"):
    row = AnomalyEvent(
        commodity=commodity,
        anomaly_type=anomaly_type,
        severity=2.5,
        detected_at=datetime.utcnow(),
        source_ids="[]",
        status=status,
        metadata_json="{}",
    )
    db.add(row)
    db.flush()
    return row


def _insert_signal(db, anomaly, *, commodity="lng", alert_type="similar_historical",
                   monitoring_complete=False, price_at_alert=100.0):
    row = SignalAlert(
        anomaly_event_id=anomaly.id,
        commodity=commodity,
        alert_type=alert_type,
        correlated_anomaly_ids="[]",
        similarity_scores="[]",
        price_at_alert=price_at_alert,
        monitoring_complete=monitoring_complete,
        created_at=datetime.utcnow(),
    )
    db.add(row)
    db.flush()
    return row


def _insert_report(db, signal, *, commodity="lng"):
    row = CausalityReport(
        signal_alert_id=signal.id,
        commodity=commodity,
        report_json=json.dumps({
            "cause": "supply disruption",
            "cause_category": "supply_disruption",
            "mechanism": "Terminal outage.",
            "price_impact_pct": 4.5,
            "confidence": 0.80,
            "supporting_signals": ["price_spike"],
            "historical_precedents": [],
            "summary": "LNG price surge.",
        }),
        cause_category="supply_disruption",
        confidence_score=0.80,
        price_impact_pct=4.5,
        created_at=datetime.utcnow(),
    )
    db.add(row)
    db.flush()
    return row


def _insert_evaluation(db, signal, *, commodity="lng"):
    row = PredictionEvaluation(
        signal_alert_id=signal.id,
        commodity=commodity,
        actual_price_change_pct=5.2,
        prediction_accuracy_json=json.dumps({
            "direction_correct": True,
            "magnitude_error": 1.2,
            "volatility_correct": True,
            "confidence_validity": "well_calibrated",
        }),
        causal_analysis_json=json.dumps({
            "correct_drivers": ["supply"],
            "missed_drivers": [],
            "overestimated_drivers": [],
        }),
        failure_modes_json="[]",
        learning_update_json=json.dumps({
            "insight": "Supply signals are strong.",
            "future_adjustment": "Trust supply signals more.",
            "affected_signal_types": ["ais_vessel_drop"],
        }),
        overall_score=0.82,
        created_at=datetime.utcnow(),
    )
    db.add(row)
    db.flush()
    return row


def _insert_learning_update(db, evaluation, *, commodity="lng"):
    row = LearningUpdate(
        prediction_evaluation_id=evaluation.id,
        commodity=commodity,
        anomaly_type="price_spike",
        insight="Supply signals reliable.",
        future_adjustment="Weight supply higher.",
        affected_signal_types=json.dumps(["ais_vessel_drop"]),
        created_at=datetime.utcnow(),
    )
    db.add(row)
    db.flush()
    return row


def _insert_price_row(db, commodity="lng", price=105.0):
    raw = RawIngestion(
        source="yfinance",
        commodity=commodity,
        symbol="LNG",
        timestamp=datetime.utcnow(),
        data_type="price_realtime",
        raw_json="{}",
        ingested_at=datetime.utcnow(),
        processed=True,
    )
    db.add(raw)
    db.flush()
    feat = ProcessedFeature(
        raw_ingestion_id=raw.id,
        commodity=commodity,
        feature_type="price",
        value=price,
        computed_at=datetime.utcnow(),
    )
    db.add(feat)
    db.flush()
    return raw, feat


# ══════════════════════════════════════════════════════════════════════════════
# /signals
# ══════════════════════════════════════════════════════════════════════════════

class TestSignalsEndpoint:

    def test_signals_returns_200(self, client):
        resp = client.get("/signals")
        assert resp.status_code == 200

    def test_signals_returns_list(self, client):
        assert isinstance(client.get("/signals").json(), list)

    def test_signal_row_schema(self, client, db):
        anomaly = _insert_anomaly(db)
        _insert_signal(db, anomaly)
        db.commit()

        resp = client.get("/signals?limit=1")
        assert resp.status_code == 200
        row = resp.json()[0]
        for field in ("id", "anomaly_event_id", "commodity", "alert_type",
                      "monitoring_complete", "created_at"):
            assert field in row, f"Missing field: {field}"

    def test_signals_filter_by_commodity(self, client, db):
        anomaly = _insert_anomaly(db, commodity="copper")
        _insert_signal(db, anomaly, commodity="copper")
        db.commit()

        data = client.get("/signals?commodity=copper").json()
        assert all(r["commodity"] == "copper" for r in data)

    def test_signals_filter_by_alert_type(self, client, db):
        anomaly = _insert_anomaly(db)
        _insert_signal(db, anomaly, alert_type="novel_event")
        db.commit()

        data = client.get("/signals?alert_type=novel_event").json()
        assert all(r["alert_type"] == "novel_event" for r in data)

    def test_signals_filter_monitoring_complete(self, client, db):
        anomaly = _insert_anomaly(db)
        _insert_signal(db, anomaly, monitoring_complete=True)
        db.commit()

        data = client.get("/signals?monitoring_complete=true").json()
        assert all(r["monitoring_complete"] is True for r in data)

    def test_get_signal_by_id(self, client, db):
        anomaly = _insert_anomaly(db)
        signal = _insert_signal(db, anomaly)
        db.commit()

        resp = client.get(f"/signals/{signal.id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == signal.id

    def test_get_signal_by_id_404(self, client):
        resp = client.get("/signals/9999999")
        assert resp.status_code == 404

    def test_signals_limit_respected(self, client, db):
        anomaly = _insert_anomaly(db)
        for _ in range(5):
            _insert_signal(db, anomaly)
        db.commit()

        data = client.get("/signals?limit=2").json()
        assert len(data) <= 2


# ══════════════════════════════════════════════════════════════════════════════
# /reports
# ══════════════════════════════════════════════════════════════════════════════

class TestReportsEndpoint:

    def test_reports_returns_200(self, client):
        assert client.get("/reports").status_code == 200

    def test_reports_returns_list(self, client):
        assert isinstance(client.get("/reports").json(), list)

    def test_report_row_schema(self, client, db):
        anomaly = _insert_anomaly(db)
        signal = _insert_signal(db, anomaly, monitoring_complete=True)
        _insert_report(db, signal)
        db.commit()

        resp = client.get("/reports?limit=1")
        row = resp.json()[0]
        for field in ("id", "signal_alert_id", "commodity", "cause_category",
                      "confidence_score", "price_impact_pct", "created_at"):
            assert field in row, f"Missing: {field}"

    def test_report_cause_category_populated(self, client, db):
        anomaly = _insert_anomaly(db, commodity="soybeans")
        signal = _insert_signal(db, anomaly, commodity="soybeans", monitoring_complete=True)
        _insert_report(db, signal, commodity="soybeans")
        db.commit()

        data = client.get("/reports?commodity=soybeans").json()
        assert len(data) >= 1
        assert data[0]["cause_category"] == "supply_disruption"

    def test_reports_filter_by_commodity(self, client, db):
        anomaly = _insert_anomaly(db, commodity="soybeans")
        signal = _insert_signal(db, anomaly, commodity="soybeans", monitoring_complete=True)
        _insert_report(db, signal, commodity="soybeans")
        db.commit()

        data = client.get("/reports?commodity=soybeans").json()
        assert all(r["commodity"] == "soybeans" for r in data)

    def test_reports_filter_by_cause_category(self, client, db):
        anomaly = _insert_anomaly(db)
        signal = _insert_signal(db, anomaly, monitoring_complete=True)
        _insert_report(db, signal)
        db.commit()

        data = client.get("/reports?cause_category=supply_disruption").json()
        assert all(r["cause_category"] == "supply_disruption" for r in data)

    def test_get_report_by_id(self, client, db):
        anomaly = _insert_anomaly(db)
        signal = _insert_signal(db, anomaly, monitoring_complete=True)
        report = _insert_report(db, signal)
        db.commit()

        resp = client.get(f"/reports/{report.id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == report.id

    def test_get_report_by_id_404(self, client):
        assert client.get("/reports/9999999").status_code == 404

    def test_report_full_report_field_populated(self, client, db):
        anomaly = _insert_anomaly(db)
        signal = _insert_signal(db, anomaly, monitoring_complete=True)
        report = _insert_report(db, signal)
        db.commit()

        data = client.get(f"/reports/{report.id}").json()
        assert "full_report" in data
        assert "cause" in data["full_report"]


# ══════════════════════════════════════════════════════════════════════════════
# /prices/{commodity}
# ══════════════════════════════════════════════════════════════════════════════

class TestPricesEndpoint:

    def test_prices_returns_200(self, client):
        assert client.get("/prices/lng").status_code == 200

    def test_prices_response_schema(self, client):
        data = client.get("/prices/lng").json()
        for field in ("commodity", "window", "count", "prices"):
            assert field in data

    def test_prices_commodity_in_response(self, client):
        assert client.get("/prices/copper").json()["commodity"] == "copper"

    def test_prices_default_window_is_1m(self, client):
        assert client.get("/prices/lng").json()["window"] == "1m"

    def test_prices_window_override(self, client):
        data = client.get("/prices/lng?window=1w").json()
        assert data["window"] == "1w"

    def test_prices_invalid_commodity_400(self, client):
        assert client.get("/prices/gold").status_code == 400

    def test_prices_invalid_window_400(self, client):
        assert client.get("/prices/lng?window=5y").status_code == 400

    def test_prices_contains_data_when_inserted(self, client, db):
        _insert_price_row(db, commodity="lng", price=105.0)
        db.commit()

        data = client.get("/prices/lng?window=1d").json()
        assert data["count"] >= 1
        assert len(data["prices"]) == data["count"]

    def test_price_row_schema(self, client, db):
        _insert_price_row(db, commodity="copper", price=8500.0)
        db.commit()

        data = client.get("/prices/copper?window=1d").json()
        if data["prices"]:
            row = data["prices"][0]
            for field in ("timestamp", "source", "symbol", "price"):
                assert field in row

    def test_prices_all_commodities_valid(self, client):
        for commodity in ("lng", "copper", "soybeans"):
            assert client.get(f"/prices/{commodity}").status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# /evaluations
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluationsEndpoint:

    def test_evaluations_returns_200(self, client):
        assert client.get("/evaluations").status_code == 200

    def test_evaluations_returns_list(self, client):
        assert isinstance(client.get("/evaluations").json(), list)

    def test_evaluation_row_schema(self, client, db):
        anomaly = _insert_anomaly(db)
        signal = _insert_signal(db, anomaly, monitoring_complete=True)
        _insert_evaluation(db, signal)
        db.commit()

        resp = client.get("/evaluations?limit=1")
        row = resp.json()[0]
        for field in ("id", "signal_alert_id", "commodity", "overall_score",
                      "actual_price_change_pct", "prediction_accuracy",
                      "causal_analysis", "failure_modes", "learning_update",
                      "learning_updates", "created_at"):
            assert field in row, f"Missing: {field}"

    def test_evaluations_filter_by_commodity(self, client, db):
        anomaly = _insert_anomaly(db, commodity="copper")
        signal = _insert_signal(db, anomaly, commodity="copper", monitoring_complete=True)
        _insert_evaluation(db, signal, commodity="copper")
        db.commit()

        data = client.get("/evaluations?commodity=copper").json()
        assert all(r["commodity"] == "copper" for r in data)

    def test_evaluation_overall_score_populated(self, client, db):
        anomaly = _insert_anomaly(db)
        signal = _insert_signal(db, anomaly, monitoring_complete=True)
        _insert_evaluation(db, signal)
        db.commit()

        data = client.get("/evaluations?limit=1").json()
        assert data[0]["overall_score"] == pytest.approx(0.82, abs=0.01)

    def test_evaluation_prediction_accuracy_fields(self, client, db):
        anomaly = _insert_anomaly(db)
        signal = _insert_signal(db, anomaly, monitoring_complete=True)
        _insert_evaluation(db, signal)
        db.commit()

        data = client.get("/evaluations?limit=1").json()
        acc = data[0]["prediction_accuracy"]
        assert acc.get("direction_correct") is True

    def test_evaluations_learning_updates_sublist(self, client, db):
        anomaly = _insert_anomaly(db)
        signal = _insert_signal(db, anomaly, monitoring_complete=True)
        evaluation = _insert_evaluation(db, signal)
        _insert_learning_update(db, evaluation)
        db.commit()

        data = client.get("/evaluations?limit=1").json()
        lu_list = data[0]["learning_updates"]
        assert isinstance(lu_list, list)
        assert len(lu_list) >= 1
        assert lu_list[0]["insight"] == "Supply signals reliable."


# ══════════════════════════════════════════════════════════════════════════════
# /evaluations/learning
# ══════════════════════════════════════════════════════════════════════════════

class TestLearningUpdatesEndpoint:

    def test_learning_returns_200(self, client):
        assert client.get("/evaluations/learning").status_code == 200

    def test_learning_returns_list(self, client):
        assert isinstance(client.get("/evaluations/learning").json(), list)

    def test_learning_row_schema(self, client, db):
        anomaly = _insert_anomaly(db)
        signal = _insert_signal(db, anomaly, monitoring_complete=True)
        evaluation = _insert_evaluation(db, signal)
        _insert_learning_update(db, evaluation)
        db.commit()

        data = client.get("/evaluations/learning?limit=1").json()
        row = data[0]
        for field in ("id", "prediction_evaluation_id", "commodity",
                      "anomaly_type", "insight", "future_adjustment",
                      "affected_signal_types", "created_at"):
            assert field in row

    def test_learning_filter_by_commodity(self, client, db):
        anomaly = _insert_anomaly(db, commodity="soybeans")
        signal = _insert_signal(db, anomaly, commodity="soybeans", monitoring_complete=True)
        evaluation = _insert_evaluation(db, signal, commodity="soybeans")
        _insert_learning_update(db, evaluation, commodity="soybeans")
        db.commit()

        data = client.get("/evaluations/learning?commodity=soybeans").json()
        assert all(r["commodity"] == "soybeans" for r in data)

    def test_learning_filter_by_anomaly_type(self, client, db):
        anomaly = _insert_anomaly(db, anomaly_type="sentiment_shift")
        signal = _insert_signal(db, anomaly, monitoring_complete=True)
        evaluation = _insert_evaluation(db, signal)
        row = LearningUpdate(
            prediction_evaluation_id=evaluation.id,
            commodity="lng",
            anomaly_type="sentiment_shift",
            insight="Sentiment reliable.",
            future_adjustment="Trust it.",
            affected_signal_types="[]",
            created_at=datetime.utcnow(),
        )
        db.add(row)
        db.commit()

        data = client.get("/evaluations/learning?anomaly_type=sentiment_shift").json()
        assert all(r["anomaly_type"] == "sentiment_shift" for r in data)

    def test_learning_affected_signal_types_is_list(self, client, db):
        anomaly = _insert_anomaly(db)
        signal = _insert_signal(db, anomaly, monitoring_complete=True)
        evaluation = _insert_evaluation(db, signal)
        _insert_learning_update(db, evaluation)
        db.commit()

        data = client.get("/evaluations/learning?limit=1").json()
        assert isinstance(data[0]["affected_signal_types"], list)


# ══════════════════════════════════════════════════════════════════════════════
# /predictions
# ══════════════════════════════════════════════════════════════════════════════

class TestPredictionsEndpoint:

    def _insert_signal_with_prediction(self, db, commodity="lng"):
        anomaly = _insert_anomaly(db, commodity=commodity)
        prediction = json.dumps({
            "event_id": str(anomaly.id),
            "commodity": commodity,
            "signal_summary": "Strong price spike.",
            "predicted_outcomes": [
                {"scenario": "Bullish", "price_move": "+3% to +7%",
                 "probability": 0.65, "direction_confidence": "high",
                 "time_horizon": "1m"},
            ],
            "confidence_score": 0.75,
            "prediction_type": "directional",
            "key_drivers": ["supply disruption"],
            "risk_factors": [],
            "monitoring_horizon": "1m",
        })
        signal = SignalAlert(
            anomaly_event_id=anomaly.id,
            commodity=commodity,
            alert_type="similar_historical",
            correlated_anomaly_ids="[]",
            similarity_scores="[]",
            price_at_alert=100.0,
            monitoring_complete=False,
            prediction_json=prediction,
            prediction_type="directional",
            prediction_confidence=0.75,
            created_at=datetime.utcnow(),
        )
        db.add(signal)
        db.flush()
        return anomaly, signal

    def test_predictions_returns_200(self, client):
        assert client.get("/predictions").status_code == 200

    def test_predictions_returns_list(self, client):
        assert isinstance(client.get("/predictions").json(), list)

    def test_prediction_row_schema(self, client, db):
        self._insert_signal_with_prediction(db)
        db.commit()

        data = client.get("/predictions?limit=1").json()
        assert len(data) >= 1
        row = data[0]
        for field in ("signal_alert_id", "commodity", "prediction_type",
                      "prediction_confidence", "prediction_json",
                      "evaluation", "overall_score", "created_at"):
            assert field in row

    def test_predictions_filter_by_commodity(self, client, db):
        self._insert_signal_with_prediction(db, commodity="copper")
        db.commit()

        data = client.get("/predictions?commodity=copper").json()
        assert all(r["commodity"] == "copper" for r in data)

    def test_prediction_type_directional(self, client, db):
        self._insert_signal_with_prediction(db)
        db.commit()

        data = client.get("/predictions?limit=1").json()
        assert data[0]["prediction_type"] == "directional"

    def test_prediction_confidence_value(self, client, db):
        self._insert_signal_with_prediction(db)
        db.commit()

        data = client.get("/predictions?limit=1").json()
        assert data[0]["prediction_confidence"] == pytest.approx(0.75, abs=0.01)

    def test_prediction_json_contains_outcomes(self, client, db):
        self._insert_signal_with_prediction(db)
        db.commit()

        data = client.get("/predictions?limit=1").json()
        outcomes = data[0]["prediction_json"].get("predicted_outcomes", [])
        assert len(outcomes) >= 1

    def test_evaluation_none_when_not_evaluated(self, client, db):
        self._insert_signal_with_prediction(db)
        db.commit()

        data = client.get("/predictions?limit=1").json()
        # Evaluation should be None since no PredictionEvaluation was inserted
        assert data[0]["evaluation"] is None


# ══════════════════════════════════════════════════════════════════════════════
# /metrics (Prometheus text format)
# ══════════════════════════════════════════════════════════════════════════════

class TestMetricsEndpoint:

    def test_metrics_returns_200(self, client):
        assert client.get("/metrics").status_code == 200

    def test_metrics_content_type_plain_text(self, client):
        resp = client.get("/metrics")
        assert "text/plain" in resp.headers["content-type"]

    def test_metrics_contains_table_rows(self, client):
        text = client.get("/metrics").text
        assert "mcei_table_rows" in text

    def test_metrics_contains_uptime(self, client):
        assert "mcei_uptime_seconds" in client.get("/metrics").text

    def test_metrics_contains_build_info(self, client):
        assert "mcei_build_info" in client.get("/metrics").text

    def test_metrics_contains_anomalies_total(self, client):
        assert "mcei_anomalies_total" in client.get("/metrics").text

    def test_metrics_contains_job_success_ts(self, client):
        assert "mcei_job_last_success_ts" in client.get("/metrics").text

    def test_metrics_table_rows_all_tables_present(self, client):
        text = client.get("/metrics").text
        for table in ("raw_ingestion", "anomaly_events", "signal_alerts",
                      "causality_reports", "prediction_evaluations"):
            assert f'table="{table}"' in text, f"Missing table metric: {table}"

    def test_metrics_anomaly_statuses_present(self, client):
        text = client.get("/metrics").text
        for status in ("new", "embedding_queued", "processed"):
            assert f'status="{status}"' in text

    def test_metrics_expected_jobs_present(self, client):
        text = client.get("/metrics").text
        for job in ("news", "price_realtime", "ai_engine", "causality",
                    "prediction", "evaluation"):
            assert f'job="{job}"' in text
