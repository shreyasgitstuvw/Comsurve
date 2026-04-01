"""
Tests for embedding vector binary serialization round-trip (G.7).

Covers pack_vector / unpack_vector from shared.models, including:
  - exact round-trip
  - known-value encoding
  - blob size guarantee
  - tolerance for float32 precision loss
  - EmbeddingCache.to_vector() convenience accessor
"""

import json
import struct
from datetime import datetime

import pytest

from shared.models import _DIMS, _PACK_FMT, pack_vector, unpack_vector


# ══════════════════════════════════════════════════════════════════════════════
# pack_vector / unpack_vector round-trip
# ══════════════════════════════════════════════════════════════════════════════

class TestPackUnpackRoundTrip:

    def test_zeros_round_trip(self):
        v = [0.0] * _DIMS
        assert unpack_vector(pack_vector(v)) == v

    def test_ones_round_trip(self):
        v = [1.0] * _DIMS
        result = unpack_vector(pack_vector(v))
        assert all(abs(r - 1.0) < 1e-6 for r in result)

    def test_random_values_round_trip(self):
        import random
        rng = random.Random(42)
        v = [rng.gauss(0, 1) for _ in range(_DIMS)]
        result = unpack_vector(pack_vector(v))
        # float32 has ~7 significant decimal digits
        for orig, rt in zip(v, result):
            assert abs(orig - rt) < abs(orig) * 1e-5 + 1e-7

    def test_all_negative_round_trip(self):
        v = [-0.5] * _DIMS
        result = unpack_vector(pack_vector(v))
        assert all(abs(r + 0.5) < 1e-6 for r in result)

    def test_mixed_sign_values(self):
        v = [(-1) ** i * (i / _DIMS) for i in range(_DIMS)]
        result = unpack_vector(pack_vector(v))
        for orig, rt in zip(v, result):
            assert abs(orig - rt) < 1e-5

    def test_dimension_count_preserved(self):
        v = [float(i) / _DIMS for i in range(_DIMS)]
        result = unpack_vector(pack_vector(v))
        assert len(result) == _DIMS

    def test_blob_size_is_exactly_12288_bytes(self):
        v = [0.0] * _DIMS
        blob = pack_vector(v)
        assert len(blob) == _DIMS * 4  # float32 = 4 bytes each

    def test_blob_is_bytes_type(self):
        v = [1.0] * _DIMS
        blob = pack_vector(v)
        assert isinstance(blob, bytes)

    def test_unpack_returns_list(self):
        v = [0.1] * _DIMS
        result = unpack_vector(pack_vector(v))
        assert isinstance(result, list)

    def test_known_value_encoding(self):
        # Verify first two floats encode correctly at byte level
        v = [1.0, 2.0] + [0.0] * (_DIMS - 2)
        blob = pack_vector(v)
        # First 8 bytes should be float32 LE encoding of 1.0 and 2.0
        first, second = struct.unpack_from("<2f", blob)
        assert first == pytest.approx(1.0)
        assert second == pytest.approx(2.0)

    def test_little_endian_byte_order(self):
        # float32 LE of 1.0 = 0x3F800000 = bytes 00 00 80 3F
        v = [1.0] + [0.0] * (_DIMS - 1)
        blob = pack_vector(v)
        assert blob[:4] == b"\x00\x00\x80\x3f"

    def test_pack_wrong_length_raises(self):
        with pytest.raises(struct.error):
            pack_vector([1.0] * (_DIMS - 1))  # too short

    def test_unpack_wrong_length_raises(self):
        with pytest.raises(struct.error):
            unpack_vector(b"\x00" * 8)  # 2 floats, not 3072


# ══════════════════════════════════════════════════════════════════════════════
# EmbeddingCache.to_vector() accessor
# ══════════════════════════════════════════════════════════════════════════════

class TestEmbeddingCacheGetVector:
    """Tests for EmbeddingCache.get_vector() — the ORM accessor that returns the vector."""

    def test_get_vector_from_blob(self, db_session):
        from shared.models import AnomalyEvent, EmbeddingCache
        anomaly = AnomalyEvent(
            commodity="lng", anomaly_type="price_spike", severity=2.0,
            detected_at=datetime.utcnow(), source_ids="[]", status="new",
        )
        db_session.add(anomaly)
        db_session.flush()

        v = [0.5] * _DIMS
        ec = EmbeddingCache(
            anomaly_event_id=anomaly.id,
            model="test-model",
            vector_blob=pack_vector(v),
            vector_json=None,
        )
        db_session.add(ec)
        db_session.flush()

        result = ec.get_vector()
        assert len(result) == _DIMS
        assert all(abs(r - 0.5) < 1e-6 for r in result)

    def test_get_vector_from_json_when_blob_none(self, db_session):
        from datetime import datetime
        from shared.models import AnomalyEvent, EmbeddingCache
        anomaly = AnomalyEvent(
            commodity="copper", anomaly_type="price_spike", severity=2.0,
            detected_at=datetime.utcnow(), source_ids="[]", status="new",
        )
        db_session.add(anomaly)
        db_session.flush()

        v = [0.25] * _DIMS
        ec = EmbeddingCache(
            anomaly_event_id=anomaly.id,
            model="test-model",
            vector_blob=None,
            vector_json=json.dumps(v),
        )
        db_session.add(ec)
        db_session.flush()

        result = ec.get_vector()
        assert result == v

    def test_get_vector_blob_takes_priority_over_json(self, db_session):
        from datetime import datetime
        from shared.models import AnomalyEvent, EmbeddingCache
        anomaly = AnomalyEvent(
            commodity="soybeans", anomaly_type="price_spike", severity=2.0,
            detected_at=datetime.utcnow(), source_ids="[]", status="new",
        )
        db_session.add(anomaly)
        db_session.flush()

        blob_v = [0.1] * _DIMS
        json_v = [0.9] * _DIMS
        ec = EmbeddingCache(
            anomaly_event_id=anomaly.id,
            model="test-model",
            vector_blob=pack_vector(blob_v),
            vector_json=json.dumps(json_v),
        )
        db_session.add(ec)
        db_session.flush()

        result = ec.get_vector()
        assert all(abs(r - 0.1) < 1e-6 for r in result)

    def test_get_vector_returns_none_when_no_data(self, db_session):
        from datetime import datetime
        from shared.models import AnomalyEvent, EmbeddingCache
        anomaly = AnomalyEvent(
            commodity="lng", anomaly_type="sentiment_shift", severity=1.5,
            detected_at=datetime.utcnow(), source_ids="[]", status="new",
        )
        db_session.add(anomaly)
        db_session.flush()

        ec = EmbeddingCache(
            anomaly_event_id=anomaly.id,
            model="test-model",
            vector_blob=None,
            vector_json=None,
        )
        db_session.add(ec)
        db_session.flush()

        result = ec.get_vector()
        assert result is None
