"""
Tests for ai_engine/signal_correlator.py — run_signal_correlation().

Flow under test:
  1. Query anomaly_events (status='processed') JOIN embeddings_cache
     LEFT JOIN signal_alerts WHERE sa.id IS NULL
  2. json.loads(vector_json) → vector
  3. QdrantManager.search_similar(...) → similar results
  4. alert_type = 'similar_historical' or 'novel_event'
  5. _get_current_price(commodity) → latest price feature
  6. Insert SignalAlert row
  7. If alerts_created > 0, call run_prediction_engine (mocked out)

Isolation strategy
------------------
  - get_session patched at "ai_engine.signal_correlator.get_session"
  - QdrantManager patched at "ai_engine.signal_correlator.QdrantManager"
  - run_prediction_engine patched at "ai_engine.signal_correlator.run_prediction_engine"
    to prevent it from executing during correlation tests.
"""

import json
from contextlib import contextmanager
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from shared.models import AnomalyEvent, EmbeddingCache, ProcessedFeature, RawIngestion, SignalAlert


# ---------------------------------------------------------------------------
# Helpers
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


def _insert_processed_anomaly(db_session, *, commodity="lng",
                               anomaly_type="price_spike", severity=3.0,
                               detected_at=None, vector=None):
    """
    Insert an AnomalyEvent with status='processed' and a matching EmbeddingCache.
    Returns the AnomalyEvent ORM object (id assigned).
    """
    if detected_at is None:
        detected_at = datetime(2024, 3, 15)
    if vector is None:
        vector = [0.1] * 3072

    anomaly = AnomalyEvent(
        commodity=commodity,
        anomaly_type=anomaly_type,
        severity=severity,
        detected_at=detected_at,
        source_ids="[]",
        status="processed",
        metadata_json="{}",
    )
    db_session.add(anomaly)
    db_session.flush()

    cache = EmbeddingCache(
        anomaly_event_id=anomaly.id,
        model="gemini-embedding-001",
        vector_json=json.dumps(vector),
        context_payload="test context",
    )
    db_session.add(cache)
    db_session.commit()
    return anomaly


def _insert_price_feature(db_session, *, commodity="lng", price=100.0,
                           days_ago=0):
    """Insert a price ProcessedFeature so _get_current_price has data."""
    raw = RawIngestion(
        source="test_price_source",
        commodity=commodity,
        symbol=f"{commodity}_price_{days_ago}",
        timestamp=datetime.utcnow() - timedelta(days=days_ago),
        data_type="price_realtime",
        raw_json="{}",
        ingested_at=datetime.utcnow(),
        processed=True,
    )
    db_session.add(raw)
    db_session.flush()

    feat = ProcessedFeature(
        raw_ingestion_id=raw.id,
        commodity=commodity,
        feature_type="price",
        value=price,
        computed_at=datetime.utcnow(),
    )
    db_session.add(feat)
    db_session.commit()


def _make_mock_qdrant(similar_results=None):
    """Return a MagicMock QdrantManager with .search_similar configured."""
    mock_qdrant = MagicMock()
    mock_qdrant.search_similar.return_value = similar_results if similar_results is not None else []
    return mock_qdrant


def _run_correlator(db_session, mock_qdrant_instance=None, mock_prediction=None):
    """
    Patch all external dependencies and call run_signal_correlation().
    Returns the summary dict.
    """
    if mock_qdrant_instance is None:
        mock_qdrant_instance = _make_mock_qdrant([])

    fake_session = _fake_session_factory(db_session)

    mock_qdrant_class = MagicMock()
    mock_qdrant_class.return_value = mock_qdrant_instance

    prediction_mock = mock_prediction if mock_prediction is not None else MagicMock(
        return_value={"predictions_generated": 0}
    )

    with patch("ai_engine.signal_correlator.get_session", side_effect=fake_session), \
         patch("ai_engine.signal_correlator.QdrantManager", mock_qdrant_class), \
         patch("ai_engine.prediction_engine.run_prediction_engine", prediction_mock):
        from ai_engine.signal_correlator import run_signal_correlation
        return run_signal_correlation()


# ---------------------------------------------------------------------------
# Test 1 — No processed anomalies returns zero alerts
# ---------------------------------------------------------------------------

class TestNoProcessedAnomalies:
    def test_empty_db_returns_zero_alerts(self, db_session):
        result = _run_correlator(db_session)
        assert result["alerts_created"] == 0

    def test_result_has_required_keys(self, db_session):
        result = _run_correlator(db_session)
        assert "alerts_created" in result
        assert "novel_events" in result
        assert "skipped" in result

    def test_unprocessed_anomaly_is_skipped(self, db_session):
        """Anomaly with status='new' (not 'processed') must not generate an alert."""
        anomaly = AnomalyEvent(
            commodity="lng",
            anomaly_type="price_spike",
            severity=2.0,
            detected_at=datetime.utcnow(),
            source_ids="[]",
            status="new",
            metadata_json="{}",
        )
        db_session.add(anomaly)
        db_session.flush()
        cache = EmbeddingCache(
            anomaly_event_id=anomaly.id,
            model="gemini-embedding-001",
            vector_json=json.dumps([0.1] * 3072),
            context_payload="ctx",
        )
        db_session.add(cache)
        db_session.commit()

        result = _run_correlator(db_session)
        assert result["alerts_created"] == 0


# ---------------------------------------------------------------------------
# Test 2 — Similar historical match produces correct SignalAlert
# ---------------------------------------------------------------------------

class TestSimilarHistoricalAlert:
    def test_similar_match_creates_alert(self, db_session):
        anomaly = _insert_processed_anomaly(db_session, commodity="lng")
        mock_result = SimpleNamespace(id=99, score=0.85)
        mock_qdrant = _make_mock_qdrant([mock_result])

        _run_correlator(db_session, mock_qdrant_instance=mock_qdrant)

        alert = db_session.query(SignalAlert).filter_by(
            anomaly_event_id=anomaly.id
        ).first()
        assert alert is not None
        assert alert.alert_type == "similar_historical"

    def test_correlated_anomaly_ids_stored(self, db_session):
        anomaly = _insert_processed_anomaly(db_session, commodity="lng")
        mock_result = SimpleNamespace(id=99, score=0.85)
        mock_qdrant = _make_mock_qdrant([mock_result])

        _run_correlator(db_session, mock_qdrant_instance=mock_qdrant)

        alert = db_session.query(SignalAlert).filter_by(
            anomaly_event_id=anomaly.id
        ).first()
        correlated = json.loads(alert.correlated_anomaly_ids)
        assert 99 in correlated

    def test_similarity_scores_stored(self, db_session):
        anomaly = _insert_processed_anomaly(db_session, commodity="lng")
        mock_result = SimpleNamespace(id=99, score=0.85)
        mock_qdrant = _make_mock_qdrant([mock_result])

        _run_correlator(db_session, mock_qdrant_instance=mock_qdrant)

        alert = db_session.query(SignalAlert).filter_by(
            anomaly_event_id=anomaly.id
        ).first()
        scores = json.loads(alert.similarity_scores)
        assert pytest.approx(scores[0], abs=0.001) == 0.85

    def test_price_at_alert_captured(self, db_session):
        anomaly = _insert_processed_anomaly(db_session, commodity="lng")
        _insert_price_feature(db_session, commodity="lng", price=42.75)
        mock_result = SimpleNamespace(id=99, score=0.85)
        mock_qdrant = _make_mock_qdrant([mock_result])

        _run_correlator(db_session, mock_qdrant_instance=mock_qdrant)

        alert = db_session.query(SignalAlert).filter_by(
            anomaly_event_id=anomaly.id
        ).first()
        assert alert.price_at_alert == pytest.approx(42.75)

    def test_summary_alerts_created_count(self, db_session):
        _insert_processed_anomaly(db_session, commodity="lng")
        mock_result = SimpleNamespace(id=99, score=0.85)
        mock_qdrant = _make_mock_qdrant([mock_result])

        result = _run_correlator(db_session, mock_qdrant_instance=mock_qdrant)
        assert result["alerts_created"] == 1


# ---------------------------------------------------------------------------
# Test 3 — No Qdrant matches → novel_event
# ---------------------------------------------------------------------------

class TestNovelEvent:
    def test_no_matches_produces_novel_alert(self, db_session):
        anomaly = _insert_processed_anomaly(db_session, commodity="copper")
        mock_qdrant = _make_mock_qdrant([])  # empty results

        _run_correlator(db_session, mock_qdrant_instance=mock_qdrant)

        alert = db_session.query(SignalAlert).filter_by(
            anomaly_event_id=anomaly.id
        ).first()
        assert alert is not None
        assert alert.alert_type == "novel_event"

    def test_novel_event_counter_incremented(self, db_session):
        _insert_processed_anomaly(db_session, commodity="copper")
        mock_qdrant = _make_mock_qdrant([])

        result = _run_correlator(db_session, mock_qdrant_instance=mock_qdrant)
        assert result["novel_events"] == 1

    def test_novel_event_correlated_ids_empty(self, db_session):
        anomaly = _insert_processed_anomaly(db_session, commodity="copper")
        mock_qdrant = _make_mock_qdrant([])

        _run_correlator(db_session, mock_qdrant_instance=mock_qdrant)

        alert = db_session.query(SignalAlert).filter_by(
            anomaly_event_id=anomaly.id
        ).first()
        assert json.loads(alert.correlated_anomaly_ids) == []


# ---------------------------------------------------------------------------
# Test 4 — Anomaly already has a SignalAlert → skipped
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_existing_alert_not_duplicated(self, db_session):
        anomaly = _insert_processed_anomaly(db_session, commodity="lng")

        # Pre-insert a SignalAlert for this anomaly
        existing_alert = SignalAlert(
            anomaly_event_id=anomaly.id,
            commodity="lng",
            alert_type="novel_event",
            correlated_anomaly_ids="[]",
            similarity_scores="[]",
            monitoring_complete=False,
            created_at=datetime.utcnow(),
        )
        db_session.add(existing_alert)
        db_session.commit()

        mock_qdrant = _make_mock_qdrant([SimpleNamespace(id=77, score=0.90)])
        result = _run_correlator(db_session, mock_qdrant_instance=mock_qdrant)

        # No new alert should have been created
        assert result["alerts_created"] == 0
        count = db_session.query(SignalAlert).filter_by(
            anomaly_event_id=anomaly.id
        ).count()
        assert count == 1


# ---------------------------------------------------------------------------
# Test 5 — Malformed vector_json → skipped
# ---------------------------------------------------------------------------

class TestMalformedVector:
    def test_bad_json_skipped(self, db_session):
        anomaly = AnomalyEvent(
            commodity="lng",
            anomaly_type="price_spike",
            severity=1.5,
            detected_at=datetime.utcnow(),
            source_ids="[]",
            status="processed",
            metadata_json="{}",
        )
        db_session.add(anomaly)
        db_session.flush()

        bad_cache = EmbeddingCache(
            anomaly_event_id=anomaly.id,
            model="gemini-embedding-001",
            vector_json="NOT VALID JSON }{",
            context_payload="ctx",
        )
        db_session.add(bad_cache)
        db_session.commit()

        result = _run_correlator(db_session)

        assert result["skipped"] == 1
        assert result["alerts_created"] == 0

    def test_null_vector_json_skipped(self, db_session):
        anomaly = AnomalyEvent(
            commodity="lng",
            anomaly_type="price_spike",
            severity=1.5,
            detected_at=datetime.utcnow(),
            source_ids="[]",
            status="processed",
            metadata_json="{}",
        )
        db_session.add(anomaly)
        db_session.flush()

        null_cache = EmbeddingCache(
            anomaly_event_id=anomaly.id,
            model="gemini-embedding-001",
            vector_json=None,
            context_payload="ctx",
        )
        db_session.add(null_cache)
        db_session.commit()

        result = _run_correlator(db_session)

        assert result["skipped"] == 1


# ---------------------------------------------------------------------------
# Test 6 — Summary dict keys are always present
# ---------------------------------------------------------------------------

class TestSummaryDict:
    def test_all_keys_present_on_empty_run(self, db_session):
        result = _run_correlator(db_session)
        for key in ("alerts_created", "novel_events", "skipped"):
            assert key in result, f"Missing key: {key}"

    def test_predictions_key_present_when_alerts_created(self, db_session):
        """When alerts_created > 0, the 'predictions' key should appear in summary."""
        _insert_processed_anomaly(db_session, commodity="lng")
        mock_qdrant = _make_mock_qdrant([SimpleNamespace(id=50, score=0.80)])
        prediction_mock = MagicMock(return_value={"predictions_generated": 1})

        result = _run_correlator(
            db_session,
            mock_qdrant_instance=mock_qdrant,
            mock_prediction=prediction_mock,
        )
        assert "predictions" in result
        assert result["predictions"]["predictions_generated"] == 1
