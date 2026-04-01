"""
Tests for:
  - ai_engine/causality_engine.py  → run_causality_engine()
  - ai_engine/prediction_engine.py → run_prediction_engine()

Isolation strategy
------------------
  Causality engine:
    - get_session  patched at "ai_engine.causality_engine.get_session"
    - GeminiClient patched at "ai_engine.causality_engine.GeminiClient"
    - run_evaluation_engine patched at
      "ai_engine.causality_engine.run_evaluation_engine" (prevent chain execution)
    - os.makedirs patched to avoid creating the reports/ directory on disk
    - builtins.open patched to prevent writing report JSON files

  Prediction engine:
    - get_session  patched at "ai_engine.prediction_engine.get_session"
    - GeminiClient patched at "ai_engine.prediction_engine.GeminiClient"
"""

import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, mock_open

import pytest

from shared.models import (
    AnomalyEvent, CausalityReport, EmbeddingCache, SignalAlert
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _fake_session_factory(db_session):
    @contextmanager
    def _fake_session():
        try:
            yield db_session
            db_session.commit()
        except Exception:
            db_session.rollback()
            raise
    return _fake_session


def _insert_anomaly(db_session, *, commodity="lng", anomaly_type="price_spike",
                    severity=3.0, status="processed"):
    row = AnomalyEvent(
        commodity=commodity,
        anomaly_type=anomaly_type,
        severity=severity,
        detected_at=datetime(2024, 1, 10, 12, 0, 0),
        source_ids="[]",
        status=status,
        metadata_json="{}",
    )
    db_session.add(row)
    db_session.flush()
    return row


def _insert_signal_alert(db_session, anomaly, *, monitoring_complete=True,
                          price_at_alert=100.0, price_1m=105.0,
                          prediction_json=None):
    alert = SignalAlert(
        anomaly_event_id=anomaly.id,
        commodity=anomaly.commodity,
        alert_type="similar_historical",
        correlated_anomaly_ids="[]",
        similarity_scores="[]",
        price_at_alert=price_at_alert,
        price_1m=price_1m,
        monitoring_complete=monitoring_complete,
        created_at=datetime.utcnow(),
        prediction_json=prediction_json,
    )
    db_session.add(alert)
    db_session.flush()
    return alert


# ---------------------------------------------------------------------------
# Valid Gemini JSON responses used across multiple tests
# ---------------------------------------------------------------------------

CAUSALITY_RESPONSE = json.dumps({
    "cause": "supply disruption",
    "cause_category": "supply_disruption",
    "mechanism": "LNG terminal outage reduced export capacity.",
    "price_impact_pct": 5.2,
    "confidence": 0.85,
    "supporting_signals": ["price_spike"],
    "historical_precedents": [],
    "summary": "LNG price surge due to terminal outage.",
})

PREDICTION_RESPONSE = json.dumps({
    "event_id": "1",
    "commodity": "lng",
    "signal_summary": "Strong price spike detected.",
    "predicted_outcomes": [
        {
            "scenario": "Bullish",
            "price_move": "+3% to +7%",
            "probability": 0.65,
            "direction_confidence": "high",
            "time_horizon": "1w",
        }
    ],
    "confidence_score": 0.75,
    "prediction_type": "directional",
    "key_drivers": ["supply disruption"],
    "risk_factors": ["demand recovery"],
    "monitoring_horizon": "1m",
})

LOW_CONFIDENCE_PREDICTION = json.dumps({
    "event_id": "1",
    "commodity": "lng",
    "signal_summary": "Ambiguous signals.",
    "predicted_outcomes": [],
    "confidence_score": 0.45,
    "prediction_type": "directional",
    "key_drivers": [],
    "risk_factors": [],
    "monitoring_horizon": "1m",
})


# ===========================================================================
#  CAUSALITY ENGINE TESTS
# ===========================================================================

class TestCausalityEngine:

    def _run(self, db_session, gemini_response=CAUSALITY_RESPONSE,
             evaluation_return=None):
        """Run run_causality_engine() with all externals mocked."""
        fake_session = _fake_session_factory(db_session)
        mock_gemini_inst = MagicMock()
        mock_gemini_inst.generate_text.return_value = gemini_response
        mock_gemini_class = MagicMock(return_value=mock_gemini_inst)

        eval_mock = MagicMock(
            return_value=evaluation_return or {"evaluations_generated": 0}
        )

        with patch("ai_engine.causality_engine.get_session", side_effect=fake_session), \
             patch("ai_engine.causality_engine.GeminiClient", mock_gemini_class), \
             patch("ai_engine.evaluation_engine.run_evaluation_engine", eval_mock), \
             patch("os.makedirs"), \
             patch("builtins.open", mock_open()):
            from ai_engine.causality_engine import run_causality_engine
            return run_causality_engine()

    # --- Test 1: No eligible alerts → zero reports ---

    def test_no_eligible_alerts_returns_zero(self, db_session):
        result = self._run(db_session)
        assert result["reports_generated"] == 0

    def test_no_eligible_alerts_result_has_keys(self, db_session):
        result = self._run(db_session)
        assert "reports_generated" in result
        assert "failed" in result
        assert "candidates" in result

    def test_non_complete_alert_not_processed(self, db_session):
        """An alert with monitoring_complete=False must not generate a report."""
        anomaly = _insert_anomaly(db_session)
        _insert_signal_alert(db_session, anomaly, monitoring_complete=False)
        db_session.commit()

        result = self._run(db_session)
        assert result["reports_generated"] == 0

    # --- Test 2: Eligible alert → CausalityReport created ---

    def test_eligible_alert_creates_report(self, db_session):
        anomaly = _insert_anomaly(db_session, commodity="lng")
        alert = _insert_signal_alert(db_session, anomaly, monitoring_complete=True,
                                      price_at_alert=100.0, price_1m=105.0)
        db_session.commit()

        self._run(db_session)

        report = db_session.query(CausalityReport).filter_by(
            signal_alert_id=alert.id
        ).first()
        assert report is not None

    def test_report_fields_populated(self, db_session):
        anomaly = _insert_anomaly(db_session, commodity="lng")
        alert = _insert_signal_alert(db_session, anomaly, monitoring_complete=True)
        db_session.commit()

        self._run(db_session)

        report = db_session.query(CausalityReport).filter_by(
            signal_alert_id=alert.id
        ).first()
        assert report.cause_category == "supply_disruption"
        assert report.confidence_score == pytest.approx(0.85, abs=0.01)
        assert report.commodity == "lng"

    def test_reports_generated_count(self, db_session):
        anomaly = _insert_anomaly(db_session, commodity="lng")
        _insert_signal_alert(db_session, anomaly, monitoring_complete=True)
        db_session.commit()

        result = self._run(db_session)
        assert result["reports_generated"] == 1

    # --- Test 3: Alert already has a report → skipped ---

    def test_existing_report_not_duplicated(self, db_session):
        anomaly = _insert_anomaly(db_session, commodity="lng")
        alert = _insert_signal_alert(db_session, anomaly, monitoring_complete=True)

        # Pre-insert a CausalityReport for this alert
        existing_report = CausalityReport(
            signal_alert_id=alert.id,
            commodity="lng",
            report_json=CAUSALITY_RESPONSE,
            cause_category="supply_disruption",
            confidence_score=0.85,
            price_impact_pct=5.2,
            created_at=datetime.utcnow(),
        )
        db_session.add(existing_report)
        db_session.commit()

        result = self._run(db_session)
        assert result["reports_generated"] == 0
        count = db_session.query(CausalityReport).filter_by(
            signal_alert_id=alert.id
        ).count()
        assert count == 1

    # --- Test 4: Gemini returns invalid JSON → failed, no crash ---

    def test_invalid_gemini_json_does_not_crash(self, db_session):
        anomaly = _insert_anomaly(db_session, commodity="lng")
        _insert_signal_alert(db_session, anomaly, monitoring_complete=True)
        db_session.commit()

        result = self._run(db_session, gemini_response="NOT VALID JSON }{")
        assert result["reports_generated"] == 0
        assert result["failed"] == 1

    def test_invalid_json_no_report_in_db(self, db_session):
        anomaly = _insert_anomaly(db_session, commodity="lng")
        alert = _insert_signal_alert(db_session, anomaly, monitoring_complete=True)
        db_session.commit()

        self._run(db_session, gemini_response="BROKEN")

        report = db_session.query(CausalityReport).filter_by(
            signal_alert_id=alert.id
        ).first()
        assert report is None


# ===========================================================================
#  PREDICTION ENGINE TESTS
# ===========================================================================

class TestPredictionEngine:

    def _run(self, db_session, gemini_response=PREDICTION_RESPONSE):
        """Run run_prediction_engine() with all externals mocked."""
        fake_session = _fake_session_factory(db_session)
        mock_gemini_inst = MagicMock()
        mock_gemini_inst.generate_text.return_value = gemini_response
        mock_gemini_class = MagicMock(return_value=mock_gemini_inst)

        with patch("ai_engine.prediction_engine.get_session", side_effect=fake_session), \
             patch("ai_engine.prediction_engine.GeminiClient", mock_gemini_class):
            from ai_engine.prediction_engine import run_prediction_engine
            return run_prediction_engine()

    def _setup_alert_without_prediction(self, db_session, *, commodity="lng"):
        """Insert a minimal anomaly + signal_alert with prediction_json=None."""
        # Need an EmbeddingCache too, for _get_signal_context
        anomaly = _insert_anomaly(db_session, commodity=commodity)
        cache = EmbeddingCache(
            anomaly_event_id=anomaly.id,
            model="gemini-embedding-001",
            vector_json=json.dumps([0.1] * 3072),
            context_payload="LNG supply disruption signal context",
        )
        db_session.add(cache)
        db_session.flush()

        alert = _insert_signal_alert(
            db_session, anomaly,
            monitoring_complete=False,
            prediction_json=None,
        )
        db_session.commit()
        return anomaly, alert

    # --- Test 1: No unpredicted alerts → predictions_generated=0 ---

    def test_no_alerts_returns_zero(self, db_session):
        result = self._run(db_session)
        assert result["predictions_generated"] == 0

    def test_result_has_required_keys(self, db_session):
        result = self._run(db_session)
        for key in ("predictions_generated", "no_signal", "failed"):
            assert key in result, f"Missing key: {key}"

    def test_alert_with_existing_prediction_not_reprocessed(self, db_session):
        """Alert already having prediction_json must not be sent to Gemini again."""
        anomaly = _insert_anomaly(db_session, commodity="lng")
        _insert_signal_alert(
            db_session, anomaly,
            monitoring_complete=False,
            prediction_json=PREDICTION_RESPONSE,  # already filled
        )
        db_session.commit()

        result = self._run(db_session)
        assert result["predictions_generated"] == 0

    # --- Test 2: Alert without prediction → prediction_json gets written ---

    def test_prediction_json_written_to_db(self, db_session):
        anomaly, alert = self._setup_alert_without_prediction(db_session)

        self._run(db_session)

        db_session.expire(alert)
        refreshed = db_session.query(SignalAlert).filter_by(id=alert.id).first()
        assert refreshed.prediction_json is not None

    def test_prediction_type_stored(self, db_session):
        anomaly, alert = self._setup_alert_without_prediction(db_session)

        self._run(db_session)

        db_session.expire(alert)
        refreshed = db_session.query(SignalAlert).filter_by(id=alert.id).first()
        assert refreshed.prediction_type == "directional"

    def test_prediction_confidence_stored(self, db_session):
        anomaly, alert = self._setup_alert_without_prediction(db_session)

        self._run(db_session)

        db_session.expire(alert)
        refreshed = db_session.query(SignalAlert).filter_by(id=alert.id).first()
        assert refreshed.prediction_confidence == pytest.approx(0.75, abs=0.01)

    def test_predictions_generated_counter(self, db_session):
        self._setup_alert_without_prediction(db_session, commodity="lng")
        result = self._run(db_session)
        assert result["predictions_generated"] == 1

    # --- Test 3: Low confidence response → prediction_type='no_signal' ---

    def test_low_confidence_sets_no_signal_type(self, db_session):
        anomaly, alert = self._setup_alert_without_prediction(db_session)

        self._run(db_session, gemini_response=LOW_CONFIDENCE_PREDICTION)

        db_session.expire(alert)
        refreshed = db_session.query(SignalAlert).filter_by(id=alert.id).first()
        assert refreshed.prediction_type == "no_signal"

    def test_low_confidence_increments_no_signal_counter(self, db_session):
        self._setup_alert_without_prediction(db_session, commodity="lng")
        result = self._run(db_session, gemini_response=LOW_CONFIDENCE_PREDICTION)
        assert result["no_signal"] == 1
        assert result["predictions_generated"] == 0

    def test_low_confidence_prediction_json_still_stored(self, db_session):
        """Even for no_signal, the prediction_json must be written to the DB."""
        anomaly, alert = self._setup_alert_without_prediction(db_session)

        self._run(db_session, gemini_response=LOW_CONFIDENCE_PREDICTION)

        db_session.expire(alert)
        refreshed = db_session.query(SignalAlert).filter_by(id=alert.id).first()
        assert refreshed.prediction_json is not None
        stored = json.loads(refreshed.prediction_json)
        assert stored["prediction_type"] == "no_signal"

    # --- Malformed Gemini response → failed, no crash ---

    def test_invalid_gemini_json_does_not_crash(self, db_session):
        self._setup_alert_without_prediction(db_session, commodity="lng")
        result = self._run(db_session, gemini_response="BROKEN JSON")
        assert result["failed"] == 1

    def test_invalid_json_no_prediction_written(self, db_session):
        anomaly, alert = self._setup_alert_without_prediction(db_session)

        self._run(db_session, gemini_response="BROKEN JSON")

        db_session.expire(alert)
        refreshed = db_session.query(SignalAlert).filter_by(id=alert.id).first()
        assert refreshed.prediction_json is None
