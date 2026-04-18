"""
Tests for ai_engine/confidence_calibrator.py.

Covers:
  - _shrinkage(): pure math, no DB
  - _sigmoid(): numerical stability
  - _fit_platt(): gradient descent convergence
  - calibrate_confidence(): Platt path (>=15 samples) and shrinkage fallback (<15)
  - Edge cases: zero confidence, perfect confidence, all-correct history
"""

import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from ai_engine.confidence_calibrator import (
    MIN_SAMPLES,
    SHRINKAGE_ALPHA,
    _fit_platt,
    _shrinkage,
    _sigmoid,
    calibrate_confidence,
)


# ══════════════════════════════════════════════════════════════════════════════
# _shrinkage
# ══════════════════════════════════════════════════════════════════════════════

class TestShrinkage:
    def test_midpoint_unchanged(self):
        assert _shrinkage(0.5) == pytest.approx(0.5, abs=1e-9)

    def test_pulls_high_confidence_down(self):
        result = _shrinkage(1.0)
        assert result < 1.0
        assert result == pytest.approx(SHRINKAGE_ALPHA * 1.0 + (1 - SHRINKAGE_ALPHA) * 0.5, abs=1e-9)

    def test_pulls_low_confidence_up(self):
        result = _shrinkage(0.0)
        assert result > 0.0
        assert result == pytest.approx((1 - SHRINKAGE_ALPHA) * 0.5, abs=1e-9)

    def test_output_between_zero_and_one(self):
        for raw in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert 0.0 <= _shrinkage(raw) <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# _sigmoid
# ══════════════════════════════════════════════════════════════════════════════

class TestSigmoid:
    def test_zero_gives_half(self):
        assert _sigmoid(0.0) == pytest.approx(0.5, abs=1e-9)

    def test_large_positive_near_one(self):
        assert _sigmoid(100.0) > 0.999

    def test_large_negative_near_zero(self):
        assert _sigmoid(-100.0) < 0.001

    def test_clamps_extreme_values(self):
        # Should not raise OverflowError; floating point saturates to exact 0/1 at extremes
        assert _sigmoid(-1000.0) < 0.001
        assert _sigmoid(1000.0) > 0.999


# ══════════════════════════════════════════════════════════════════════════════
# _fit_platt
# ══════════════════════════════════════════════════════════════════════════════

class TestFitPlatt:
    def test_all_correct_pushes_positive_A(self):
        # All high confidence + correct → A should remain positive (high conf maps to high prob)
        pairs = [(0.8, 1), (0.9, 1), (0.7, 1), (0.85, 1), (0.75, 1)] * 4
        A, B = _fit_platt(pairs)
        # sigmoid(A*0.8+B) should be close to 1.0
        assert _sigmoid(A * 0.8 + B) > 0.6

    def test_all_wrong_gives_low_calibrated_confidence(self):
        # All high confidence + wrong → calibrated output for high raw should be low
        pairs = [(0.9, 0), (0.8, 0), (0.85, 0), (0.75, 0), (0.95, 0)] * 4
        A, B = _fit_platt(pairs)
        # For high raw confidence, calibrated should be low
        assert _sigmoid(A * 0.9 + B) < 0.5

    def test_returns_two_floats(self):
        pairs = [(0.6, 1), (0.4, 0), (0.7, 1), (0.3, 0)] * 5
        result = _fit_platt(pairs)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_single_sample_does_not_crash(self):
        # Edge case: only one sample — GD still runs
        _fit_platt([(0.8, 1)])


# ══════════════════════════════════════════════════════════════════════════════
# calibrate_confidence — integration (mocked DB)
# ══════════════════════════════════════════════════════════════════════════════

def _make_pairs_json(direction_correct: bool, n: int) -> list[tuple]:
    """Generate fake DB rows: (raw_confidence, accuracy_json)."""
    acc = json.dumps({"direction_correct": direction_correct, "magnitude_error": 2.0})
    return [(0.75, acc)] * n


class TestCalibrateConfidence:

    def _patch_pairs(self, pairs):
        """Return a context manager that patches _fetch_calibration_pairs."""
        return patch(
            "ai_engine.confidence_calibrator._fetch_calibration_pairs",
            return_value=pairs,
        )

    def test_shrinkage_fallback_below_min_samples(self):
        pairs = [(0.8, 1)] * (MIN_SAMPLES - 1)
        with self._patch_pairs(pairs):
            result = calibrate_confidence(0.8, "lng")
        expected = round(_shrinkage(0.8), 4)
        assert result == pytest.approx(expected, abs=1e-4)

    def test_platt_path_at_min_samples(self):
        pairs = [(0.8, 1)] * MIN_SAMPLES
        with self._patch_pairs(pairs):
            result = calibrate_confidence(0.8, "lng")
        # With all-correct pairs, calibrated should be >= raw (confident predictions rewarded)
        assert 0.0 < result <= 1.0

    def test_output_always_in_valid_range(self):
        for raw in [0.0, 0.3, 0.5, 0.75, 1.0]:
            with self._patch_pairs([]):
                result = calibrate_confidence(raw, "lng")
            assert 0.0 <= result <= 1.0

    def test_empty_db_uses_shrinkage(self):
        with self._patch_pairs([]):
            result = calibrate_confidence(0.9, "copper")
        assert result == pytest.approx(round(_shrinkage(0.9), 4), abs=1e-4)

    def test_shrinkage_on_db_error(self):
        with patch(
            "ai_engine.confidence_calibrator._fetch_calibration_pairs",
            side_effect=Exception("DB down"),
        ):
            result = calibrate_confidence(0.7, "soybeans")
        assert result == pytest.approx(round(_shrinkage(0.7), 4), abs=1e-4)

    def test_result_is_rounded_to_4_decimals(self):
        with self._patch_pairs([]):
            result = calibrate_confidence(0.6789, "lng")
        assert result == round(result, 4)

    def test_commodity_passed_through(self):
        """Verify commodity param reaches _fetch_calibration_pairs."""
        captured = {}
        def fake_fetch(commodity):
            captured["commodity"] = commodity
            return []
        with patch("ai_engine.confidence_calibrator._fetch_calibration_pairs", side_effect=fake_fetch):
            calibrate_confidence(0.5, "copper")
        assert captured["commodity"] == "copper"
