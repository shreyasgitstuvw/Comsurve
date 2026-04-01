"""
E4 — AI engine tests.

Tests:
  - GeminiClient.batch_embed: happy path (one API call), INVALID_ARGUMENT → sequential fallback,
    empty input, retry exhaustion raises RuntimeError
  - build_context_payload: produces expected sections for each anomaly type
  - ai_engine_runner.run: full batch flow with mocked Gemini + Qdrant
"""

import json
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest

from shared.models import AnomalyEvent, EmbeddingCache


# ── GeminiClient.batch_embed ──────────────────────────────────────────────────

class TestBatchEmbed:
    """Unit tests for GeminiClient.batch_embed — the internal genai client is mocked."""

    def _make_client(self, embed_result=None, side_effect=None):
        """Return a GeminiClient with mocked internal _client."""
        # Patch genai.Client to avoid real API key requirement
        with patch("ai_engine.gemini_client.genai.Client"):
            from ai_engine.gemini_client import GeminiClient
            client = GeminiClient.__new__(GeminiClient)
            client._call_timestamps = __import__("collections").deque()
            client._client = MagicMock()

        if embed_result is not None:
            # Build a mock result matching the SDK's response shape:
            # result.embeddings = [obj with .values attribute]
            mock_embeddings = [SimpleNamespace(values=[float(i)] * 3072) for i in range(len(embed_result))]
            mock_result = SimpleNamespace(embeddings=mock_embeddings)
            client._client.models.embed_content.return_value = mock_result
        elif side_effect is not None:
            client._client.models.embed_content.side_effect = side_effect

        return client

    def test_empty_input_returns_empty_list(self):
        with patch("ai_engine.gemini_client.genai.Client"):
            from ai_engine.gemini_client import GeminiClient
            client = GeminiClient.__new__(GeminiClient)
            client._call_timestamps = __import__("collections").deque()
            client._client = MagicMock()
        result = client.batch_embed([])
        assert result == []
        client._client.models.embed_content.assert_not_called()

    def test_batch_happy_path_single_api_call(self):
        texts = ["text one", "text two", "text three"]
        client = self._make_client(embed_result=texts)
        result = client.batch_embed(texts)

        assert len(result) == 3
        # Single API call for the whole batch
        client._client.models.embed_content.assert_called_once()
        # Vectors have correct dimensionality
        assert all(len(v) == 3072 for v in result)

    def test_invalid_argument_falls_back_to_sequential(self):
        """INVALID_ARGUMENT error breaks out of retry loop and calls embed() per text."""
        with patch("ai_engine.gemini_client.genai.Client"):
            from ai_engine.gemini_client import GeminiClient
            client = GeminiClient.__new__(GeminiClient)
            client._call_timestamps = __import__("collections").deque()
            client._client = MagicMock()

        texts = ["a", "b"]
        batch_error = Exception("400 INVALID_ARGUMENT: batch not supported")
        single_embedding = SimpleNamespace(values=[1.0] * 3072)
        single_result = SimpleNamespace(embeddings=[single_embedding])

        # First call (batch) raises; subsequent calls (sequential) succeed
        client._client.models.embed_content.side_effect = [
            batch_error,
            single_result,
            single_result,
        ]
        result = client.batch_embed(texts)

        # Should have 2 vectors from sequential fallback
        assert len(result) == 2
        assert client._client.models.embed_content.call_count == 3  # 1 batch + 2 sequential

    def test_not_found_falls_back_to_sequential(self):
        """NOT_FOUND triggers the same sequential fallback path."""
        with patch("ai_engine.gemini_client.genai.Client"):
            from ai_engine.gemini_client import GeminiClient
            client = GeminiClient.__new__(GeminiClient)
            client._call_timestamps = __import__("collections").deque()
            client._client = MagicMock()

        single_embedding = SimpleNamespace(values=[0.5] * 3072)
        single_result = SimpleNamespace(embeddings=[single_embedding])
        client._client.models.embed_content.side_effect = [
            Exception("NOT_FOUND: model unknown"),
            single_result,
        ]
        result = client.batch_embed(["only one"])
        assert len(result) == 1

    def test_transient_error_retries_then_raises(self):
        """Non-INVALID_ARGUMENT errors retry up to `retries` times then raise."""
        with patch("ai_engine.gemini_client.genai.Client"), \
             patch("ai_engine.gemini_client.time.sleep"):  # don't actually sleep
            from ai_engine.gemini_client import GeminiClient
            client = GeminiClient.__new__(GeminiClient)
            client._call_timestamps = __import__("collections").deque()
            client._client = MagicMock()

        client._client.models.embed_content.side_effect = Exception("503 Service Unavailable")
        with pytest.raises(RuntimeError, match="batch embed failed"):
            client.batch_embed(["text"], retries=2)

        assert client._client.models.embed_content.call_count == 2


# ── build_context_payload ─────────────────────────────────────────────────────

class TestBuildContextPayload:
    """build_context_payload is called with an anomaly-like object; _fetch_source_data is mocked."""

    def _make_anomaly(self, anomaly_type: str, commodity: str = "lng",
                      severity: float = 3.5, metadata: dict | None = None) -> SimpleNamespace:
        meta = metadata or {}
        return SimpleNamespace(
            id=1,
            commodity=commodity,
            anomaly_type=anomaly_type,
            severity=severity,
            detected_at=datetime(2024, 3, 15, 6, 0),
            source_ids=json.dumps([1, 2]),
            metadata_json=json.dumps(meta),
        )

    def _call(self, anomaly, source_data=None):
        empty = {"news": [], "prices": [], "ais": []}
        with patch("ai_engine.embedding_generator._fetch_source_data",
                   return_value=source_data or empty):
            from ai_engine.embedding_generator import build_context_payload
            return build_context_payload(anomaly)

    def test_price_spike_contains_expected_sections(self):
        anomaly = self._make_anomaly(
            "price_spike", metadata={"pct_change": 0.12, "z_score": 3.2, "data_type": "price_realtime"}
        )
        text = self._call(anomaly)

        assert "COMMODITY: LNG" in text
        assert "ANOMALY TYPE: price_spike" in text
        assert "PRICE EVENT" in text
        assert "+12.00%" in text
        assert "Z-score: 3.20" in text

    def test_sentiment_shift_contains_sentiment_label(self):
        anomaly = self._make_anomaly(
            "sentiment_shift",
            metadata={"compound_score": -0.45, "z_score": 2.8},
        )
        text = self._call(anomaly)

        assert "SENTIMENT EVENT" in text
        assert "strongly negative" in text
        assert "VADER compound: -0.450" in text

    def test_ais_vessel_drop_contains_port_and_baseline(self):
        anomaly = self._make_anomaly(
            "ais_vessel_drop",
            commodity="lng",
            metadata={"port_name": "Sabine Pass", "current_vessel_count": 2,
                      "baseline_avg": 8.5, "drop_pct": 76.5},
        )
        text = self._call(anomaly)

        assert "PORT EVENT: Sabine Pass" in text
        assert "CURRENT VESSEL COUNT: 2" in text
        assert "BASELINE AVERAGE: 8.5" in text

    def test_ais_port_idle_shows_moored_ratio(self):
        anomaly = self._make_anomaly(
            "ais_port_idle",
            metadata={"port_slug": "rotterdam", "vessel_count": 5, "moored_ratio": 0.92},
        )
        text = self._call(anomaly)

        assert "PORT EVENT: rotterdam" in text
        assert "MOORED RATIO: 92.0%" in text

    def test_news_articles_included_in_output(self):
        anomaly = self._make_anomaly("price_spike", metadata={"pct_change": 0.05, "z_score": 2.1})
        source_data = {
            "news": [
                {"title": "LNG exports surge", "description": "Exports hit record",
                 "source": "reuters", "timestamp": "2024-03-15T05:00:00"},
            ],
            "prices": [],
            "ais": [],
        }
        text = self._call(anomaly, source_data=source_data)

        assert "RELATED NEWS:" in text
        assert "LNG exports surge" in text
        assert "Exports hit record" in text

    def test_missing_metadata_json_does_not_crash(self):
        anomaly = SimpleNamespace(
            id=99,
            commodity="copper",
            anomaly_type="price_spike",
            severity=2.1,
            detected_at=datetime(2024, 1, 1),
            source_ids="[]",
            metadata_json=None,  # NULL in DB
        )
        with patch("ai_engine.embedding_generator._fetch_source_data",
                   return_value={"news": [], "prices": [], "ais": []}):
            from ai_engine.embedding_generator import build_context_payload
            text = build_context_payload(anomaly)

        assert "COMMODITY: COPPER" in text

    def test_severity_formatted_to_three_decimals(self):
        anomaly = self._make_anomaly("price_spike", severity=2.123456)
        text = self._call(anomaly)
        assert "2.123" in text


# ── ai_engine_runner.run ──────────────────────────────────────────────────────

class TestAiEngineRunner:
    """Integration-level test for ai_engine_runner.run with mocked Gemini + Qdrant."""

    def _setup_db(self, db_session):
        """Insert a pending AnomalyEvent and return its id."""
        anomaly = AnomalyEvent(
            commodity="lng",
            anomaly_type="price_spike",
            severity=3.0,
            detected_at=datetime(2024, 3, 15, 6, 0),
            source_ids=json.dumps([]),
            status="new",
            metadata_json=json.dumps({}),
        )
        db_session.add(anomaly)
        db_session.commit()
        return anomaly.id

    @contextmanager
    def _patch_all(self, db_session):
        """Patch get_session at all ai_engine_runner import sites."""
        @contextmanager
        def _fake():
            try:
                yield db_session
                db_session.commit()
            except Exception:
                db_session.rollback()
                raise

        with patch("shared.db.get_session", side_effect=_fake), \
             patch("ai_engine.ai_engine_runner.get_session", side_effect=_fake):
            yield

    def test_processes_pending_anomaly(self, db_session):
        anomaly_id = self._setup_db(db_session)

        mock_qdrant = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.batch_embed.return_value = [[0.1] * 3072]

        with self._patch_all(db_session), \
             patch("ai_engine.ai_engine_runner.GeminiClient", return_value=mock_gemini), \
             patch("ai_engine.ai_engine_runner.QdrantManager", return_value=mock_qdrant), \
             patch("ai_engine.ai_engine_runner.build_context_payload", return_value="test context"):
            from ai_engine import ai_engine_runner
            result = ai_engine_runner.run(batch_size=10)

        assert result["processed"] == 1
        assert result["failed"] == 0
        mock_gemini.batch_embed.assert_called_once_with(["test context"])
        mock_qdrant.upsert_embedding.assert_called_once()

    def test_anomaly_marked_processed_in_db(self, db_session):
        anomaly_id = self._setup_db(db_session)

        mock_qdrant = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.batch_embed.return_value = [[0.0] * 3072]

        with self._patch_all(db_session), \
             patch("ai_engine.ai_engine_runner.GeminiClient", return_value=mock_gemini), \
             patch("ai_engine.ai_engine_runner.QdrantManager", return_value=mock_qdrant), \
             patch("ai_engine.ai_engine_runner.build_context_payload", return_value="ctx"):
            from ai_engine import ai_engine_runner
            ai_engine_runner.run(batch_size=10)

        db_session.expire_all()
        anomaly = db_session.query(AnomalyEvent).filter_by(id=anomaly_id).one()
        assert anomaly.status == "processed"

        cache = db_session.query(EmbeddingCache).filter_by(anomaly_event_id=anomaly_id).first()
        assert cache is not None
        assert cache.vector_blob is not None

    def test_batch_embed_failure_resets_anomalies_to_new(self, db_session):
        self._setup_db(db_session)

        mock_qdrant = MagicMock()
        mock_gemini = MagicMock()
        mock_gemini.batch_embed.side_effect = RuntimeError("Gemini API unreachable")

        with self._patch_all(db_session), \
             patch("ai_engine.ai_engine_runner.GeminiClient", return_value=mock_gemini), \
             patch("ai_engine.ai_engine_runner.QdrantManager", return_value=mock_qdrant), \
             patch("ai_engine.ai_engine_runner.build_context_payload", return_value="ctx"):
            from ai_engine import ai_engine_runner
            result = ai_engine_runner.run(batch_size=10)

        assert result["processed"] == 0
        assert result["failed"] == 1

        db_session.expire_all()
        anomaly = db_session.query(AnomalyEvent).first()
        assert anomaly.status == "new"
        meta = json.loads(anomaly.metadata_json)
        assert meta["_embed_retries"] == 1

    def test_no_pending_returns_zero_counts(self, db_session):
        # No anomalies in DB
        with self._patch_all(db_session), \
             patch("ai_engine.ai_engine_runner.GeminiClient"), \
             patch("ai_engine.ai_engine_runner.QdrantManager"):
            from ai_engine import ai_engine_runner
            result = ai_engine_runner.run()

        assert result["processed"] == 0
        assert result["failed"] == 0
        assert result["total_pending"] == 0

    def test_permanently_failed_anomaly_skipped(self, db_session):
        """Anomaly with _embed_retries >= MAX_RETRIES_PER_ANOMALY is skipped."""
        anomaly = AnomalyEvent(
            commodity="copper",
            anomaly_type="price_spike",
            severity=2.5,
            detected_at=datetime(2024, 3, 1),
            source_ids=json.dumps([]),
            status="new",
            metadata_json=json.dumps({"_embed_retries": 3}),
        )
        db_session.add(anomaly)
        db_session.commit()

        mock_gemini = MagicMock()
        mock_qdrant = MagicMock()

        with self._patch_all(db_session), \
             patch("ai_engine.ai_engine_runner.GeminiClient", return_value=mock_gemini), \
             patch("ai_engine.ai_engine_runner.QdrantManager", return_value=mock_qdrant):
            from ai_engine import ai_engine_runner
            result = ai_engine_runner.run()

        assert result["processed"] == 0
        assert result["skipped_permanent"] == 1
        mock_gemini.batch_embed.assert_not_called()
