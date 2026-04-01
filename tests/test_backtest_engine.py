"""
Tests for scripts/backtest_engine.py.

Covers:
  - ScoringEngine (pure functions — no Gemini, no yfinance)
  - BacktestPromptBuilder (prompt structure)
  - FeedbackAccumulator (in-memory learning state)
  - compute_regime_metrics (report aggregation)

No external API calls are made in any test.
"""

import json
from datetime import date

import pytest

from scripts.backtest_engine import (
    NEUTRAL_BAND_PCT,
    BacktestEvent,
    BacktestPromptBuilder,
    FeedbackAccumulator,
    PredictionResult,
    ScoringEngine,
    W_CALIBRATION,
    W_DIRECTION,
    W_MAGNITUDE,
    compute_regime_metrics,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

def _event(*, commodity="lng", anomaly_type="price_spike", severity=2.5,
           as_of_date=date(2022, 1, 1), regime="normal",
           expected_direction="up"):
    return BacktestEvent(
        event_id=f"TEST-{commodity.upper()}-{as_of_date}",
        commodity=commodity,
        anomaly_type=anomaly_type,
        severity=severity,
        as_of_date=as_of_date,
        signal_context="Test context.",
        regime=regime,
        source="test",
        expected_direction=expected_direction,
    )


def _prediction_json(**overrides):
    base = {
        "event_id": "TEST",
        "commodity": "lng",
        "signal_summary": "Test signal.",
        "predicted_outcomes": [
            {
                "scenario": "Bullish",
                "price_move": "+3% to +7%",
                "probability": 0.65,
                "direction_confidence": "high",
                "time_horizon": "1m",
            }
        ],
        "confidence_score": 0.75,
        "prediction_type": "directional",
        "drivers": [],
        "historical_analogs": [],
    }
    base.update(overrides)
    return json.dumps(base)


def _no_signal_json():
    return json.dumps({
        "event_id": "TEST",
        "commodity": "lng",
        "signal_summary": "Ambiguous.",
        "predicted_outcomes": [],
        "confidence_score": 0.4,
        "prediction_type": "no_signal",
        "drivers": [],
        "historical_analogs": [],
    })


def _outcome_prices(*, p0=100.0, p1w=101.0, p2w=103.0, p1m=107.0):
    return {"1w": p1w, "2w": p2w, "1m": p1m}


# ══════════════════════════════════════════════════════════════════════════════
# ScoringEngine.parse_price_move_midpoint
# ══════════════════════════════════════════════════════════════════════════════

class TestParsePriceMoveMiddpoint:
    def test_positive_range(self):
        assert ScoringEngine.parse_price_move_midpoint("+3% to +7%") == pytest.approx(5.0)

    def test_negative_range(self):
        assert ScoringEngine.parse_price_move_midpoint("-5% to -8%") == pytest.approx(-6.5)

    def test_single_positive(self):
        assert ScoringEngine.parse_price_move_midpoint("+5%") == pytest.approx(5.0)

    def test_single_negative(self):
        assert ScoringEngine.parse_price_move_midpoint("-3%") == pytest.approx(-3.0)

    def test_magnitude_only_returns_none(self):
        assert ScoringEngine.parse_price_move_midpoint("±4% to ±8%") is None

    def test_empty_string_returns_none(self):
        assert ScoringEngine.parse_price_move_midpoint("") is None

    def test_none_returns_none(self):
        assert ScoringEngine.parse_price_move_midpoint(None) is None

    def test_mixed_sign_range(self):
        result = ScoringEngine.parse_price_move_midpoint("-2% to +4%")
        assert result == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════════
# ScoringEngine.direction_from_pct
# ══════════════════════════════════════════════════════════════════════════════

class TestDirectionFromPct:
    def test_above_band_is_up(self):
        assert ScoringEngine.direction_from_pct(NEUTRAL_BAND_PCT + 0.1) == "up"

    def test_below_band_is_down(self):
        assert ScoringEngine.direction_from_pct(-NEUTRAL_BAND_PCT - 0.1) == "down"

    def test_inside_band_is_neutral(self):
        assert ScoringEngine.direction_from_pct(0.0) == "neutral"
        assert ScoringEngine.direction_from_pct(NEUTRAL_BAND_PCT * 0.9) == "neutral"

    def test_none_returns_none(self):
        assert ScoringEngine.direction_from_pct(None) is None


# ══════════════════════════════════════════════════════════════════════════════
# ScoringEngine.compute_brier_score
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeBrierScore:
    def test_correct_prediction_low_brier(self):
        # Predicted up with prob=0.9, actual up → BS = (0.9-1)^2 = 0.01
        bs = ScoringEngine.compute_brier_score(0.9, "up", "up")
        assert bs == pytest.approx(0.01, abs=1e-9)

    def test_wrong_prediction_high_brier(self):
        # Predicted up with prob=0.9, actual down → BS = (0.9-0)^2 = 0.81
        bs = ScoringEngine.compute_brier_score(0.9, "up", "down")
        assert bs == pytest.approx(0.81, abs=1e-9)

    def test_neutral_actual_returns_none(self):
        assert ScoringEngine.compute_brier_score(0.8, "up", "neutral") is None

    def test_none_predicted_direction_returns_none(self):
        assert ScoringEngine.compute_brier_score(0.8, None, "up") is None

    def test_none_actual_direction_returns_none(self):
        assert ScoringEngine.compute_brier_score(0.8, "up", None) is None


# ══════════════════════════════════════════════════════════════════════════════
# ScoringEngine.compute_composite
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeComposite:
    def test_all_correct_no_magnitude_error(self):
        # direction_correct=True, magnitude_mae=0 → max possible score
        score = ScoringEngine.compute_composite(True, 0.0)
        expected = W_DIRECTION + W_MAGNITUDE + W_CALIBRATION / 2
        assert score == pytest.approx(expected, abs=1e-4)

    def test_wrong_direction(self):
        score = ScoringEngine.compute_composite(False, 0.0)
        # direction = 0.0, magnitude max, calibration partial
        assert score < ScoringEngine.compute_composite(True, 0.0)

    def test_none_direction_gets_half_credit(self):
        none_score = ScoringEngine.compute_composite(None, 0.0)
        true_score = ScoringEngine.compute_composite(True, 0.0)
        false_score = ScoringEngine.compute_composite(False, 0.0)
        assert false_score < none_score < true_score

    def test_high_magnitude_mae_reduces_score(self):
        low_mae  = ScoringEngine.compute_composite(True, 1.0)
        high_mae = ScoringEngine.compute_composite(True, 15.0)
        assert high_mae < low_mae

    def test_magnitude_mae_none_gets_partial_credit(self):
        score_none  = ScoringEngine.compute_composite(True, None)
        score_zero  = ScoringEngine.compute_composite(True, 0.0)
        # None magnitude → W_MAGNITUDE / 2; zero → W_MAGNITUDE
        assert score_none < score_zero


# ══════════════════════════════════════════════════════════════════════════════
# ScoringEngine.score (end-to-end)
# ══════════════════════════════════════════════════════════════════════════════

class TestScoringEngineScore:
    def test_correct_direction_detected(self):
        ev = _event(expected_direction="up")
        # predicted +5% → up; price went from 100 → 107 → +7% → actual up
        result = ScoringEngine.score(
            ev,
            raw_prediction=_prediction_json(),
            price_at_event=100.0,
            outcome_prices=_outcome_prices(p1m=107.0),
            is_dry_run=False,
        )
        assert result.direction_correct is True
        assert result.predicted_direction == "up"
        assert result.actual_direction == "up"

    def test_wrong_direction_detected(self):
        ev = _event(expected_direction="up")
        # price went from 100 → 95 → -5% → actual down
        result = ScoringEngine.score(
            ev,
            raw_prediction=_prediction_json(),
            price_at_event=100.0,
            outcome_prices=_outcome_prices(p1m=95.0),
            is_dry_run=False,
        )
        assert result.direction_correct is False

    def test_neutral_actual_direction_gives_none_correct(self):
        ev = _event()
        # price barely moved → inside neutral band → direction_correct=None
        result = ScoringEngine.score(
            ev,
            raw_prediction=_prediction_json(),
            price_at_event=100.0,
            outcome_prices=_outcome_prices(p1m=100.5),  # +0.5% < NEUTRAL_BAND_PCT
            is_dry_run=False,
        )
        assert result.direction_correct is None
        assert result.actual_direction == "neutral"

    def test_no_signal_prediction_type(self):
        ev = _event()
        result = ScoringEngine.score(
            ev,
            raw_prediction=_no_signal_json(),
            price_at_event=100.0,
            outcome_prices=_outcome_prices(),
            is_dry_run=False,
        )
        assert result.prediction_type == "no_signal"
        assert result.direction_correct is None  # no_signal → no direction

    def test_parse_error_on_broken_json(self):
        ev = _event()
        result = ScoringEngine.score(
            ev,
            raw_prediction="NOT JSON",
            price_at_event=100.0,
            outcome_prices=_outcome_prices(),
            is_dry_run=False,
        )
        assert result.parse_error is True

    def test_dry_run_flag_propagated(self):
        ev = _event()
        result = ScoringEngine.score(
            ev,
            raw_prediction=_no_signal_json(),
            price_at_event=None,
            outcome_prices={"1w": None, "2w": None, "1m": None},
            is_dry_run=True,
        )
        assert result.is_dry_run is True

    def test_actual_pct_none_when_no_price(self):
        ev = _event()
        result = ScoringEngine.score(
            ev,
            raw_prediction=_prediction_json(),
            price_at_event=None,
            outcome_prices={"1w": None, "2w": None, "1m": None},
            is_dry_run=False,
        )
        assert result.actual_pct_1m is None

    def test_composite_score_between_zero_and_one(self):
        ev = _event()
        result = ScoringEngine.score(
            ev,
            raw_prediction=_prediction_json(),
            price_at_event=100.0,
            outcome_prices=_outcome_prices(),
            is_dry_run=False,
        )
        assert 0.0 <= result.composite_score <= 1.0

    def test_fields_populated(self):
        ev = _event()
        result = ScoringEngine.score(
            ev,
            raw_prediction=_prediction_json(),
            price_at_event=100.0,
            outcome_prices=_outcome_prices(),
            is_dry_run=False,
        )
        assert result.event_id == ev.event_id
        assert result.commodity == "lng"
        assert result.regime == "normal"
        assert result.confidence_score == pytest.approx(0.75, abs=0.01)


# ══════════════════════════════════════════════════════════════════════════════
# BacktestPromptBuilder.build
# ══════════════════════════════════════════════════════════════════════════════

class TestBacktestPromptBuilder:
    def test_prompt_contains_commodity(self):
        ev = _event(commodity="copper")
        prompt = BacktestPromptBuilder.build(ev, 8500.0, [ev])
        assert "copper" in prompt.lower()

    def test_prompt_contains_as_of_date(self):
        ev = _event(as_of_date=date(2022, 3, 15))
        prompt = BacktestPromptBuilder.build(ev, 100.0, [ev])
        assert "2022-03-15" in prompt

    def test_prompt_contains_learning_section(self):
        ev = _event()
        learning = "Past errors: direction was wrong 60% of the time."
        prompt = BacktestPromptBuilder.build(ev, 100.0, [ev], learning_section=learning)
        assert "direction" in prompt

    def test_no_price_shows_na(self):
        ev = _event()
        prompt = BacktestPromptBuilder.build(ev, None, [ev])
        assert "N/A" in prompt

    def test_analogs_section_only_uses_prior_events(self):
        prior = _event(as_of_date=date(2020, 1, 1), commodity="lng")
        future = _event(as_of_date=date(2025, 1, 1), commodity="lng")
        current = _event(as_of_date=date(2022, 1, 1), commodity="lng")
        prompt = BacktestPromptBuilder.build(current, 100.0, [prior, current, future])
        # future event must not appear in analogs (as_of_date is after current)
        assert future.event_id not in prompt


# ══════════════════════════════════════════════════════════════════════════════
# FeedbackAccumulator
# ══════════════════════════════════════════════════════════════════════════════

def _result(*, commodity="lng", anomaly_type="price_spike",
            direction_correct=True, magnitude_mae=2.0,
            prediction_type="directional", composite=0.7):
    return PredictionResult(
        event_id="TEST",
        commodity=commodity,
        regime="normal",
        as_of_date="2022-01-01",
        expected_direction="up",
        price_at_event=100.0,
        price_1w=101.0,
        price_2w=103.0,
        price_1m=107.0,
        actual_pct_1m=7.0,
        actual_direction="up",
        prediction_type=prediction_type,
        confidence_score=0.75,
        predicted_direction="up",
        predicted_pct_midpoint=5.0,
        direction_correct=direction_correct,
        magnitude_mae=magnitude_mae,
        brier_score=0.01,
        composite_score=composite,
        is_dry_run=False,
    )


def _backtest_event(commodity="lng", anomaly_type="price_spike"):
    return _event(commodity=commodity, anomaly_type=anomaly_type)


class TestFeedbackAccumulator:
    def test_empty_accumulator_returns_empty_string(self):
        acc = FeedbackAccumulator()
        ev = _backtest_event()
        assert acc.get_learning_section(ev) == ""

    def test_after_record_learning_section_is_non_empty(self):
        acc = FeedbackAccumulator()
        ev = _backtest_event()
        acc.record(ev, _result())
        # Next event of same type should get a learning section
        ev2 = _backtest_event()
        section = acc.get_learning_section(ev2)
        assert len(section) > 0

    def test_different_commodity_not_shared(self):
        acc = FeedbackAccumulator()
        ev_lng = _backtest_event(commodity="lng")
        ev_cu  = _backtest_event(commodity="copper")
        acc.record(ev_lng, _result(commodity="lng"))
        # copper ledger should still be empty
        assert acc.get_learning_section(ev_cu) == ""

    def test_different_anomaly_type_not_shared(self):
        acc = FeedbackAccumulator()
        ev_spike = _backtest_event(anomaly_type="price_spike")
        ev_sent  = _backtest_event(anomaly_type="sentiment_shift")
        acc.record(ev_spike, _result(anomaly_type="price_spike"))
        assert acc.get_learning_section(ev_sent) == ""

    def test_multiple_records_accumulate(self):
        acc = FeedbackAccumulator()
        ev = _backtest_event()
        for _ in range(5):
            acc.record(ev, _result(direction_correct=False))
        section = acc.get_learning_section(ev)
        # 5 wrong directions should trigger adjustments
        assert "adjustment" in section.lower() or "error" in section.lower()


# ══════════════════════════════════════════════════════════════════════════════
# compute_regime_metrics
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeRegimeMetrics:
    def _results(self, regimes_and_correct):
        """Build minimal PredictionResult list for metric testing."""
        results = []
        for regime, direction_correct in regimes_and_correct:
            results.append(_result(direction_correct=direction_correct))
            results[-1] = PredictionResult(
                event_id="T", commodity="lng", regime=regime,
                as_of_date="2022-01-01", expected_direction="up",
                price_at_event=100.0, price_1w=101.0, price_2w=103.0,
                price_1m=107.0, actual_pct_1m=7.0, actual_direction="up",
                prediction_type="directional", confidence_score=0.7,
                predicted_direction="up", predicted_pct_midpoint=5.0,
                direction_correct=direction_correct,
                magnitude_mae=2.0, brier_score=0.04,
                composite_score=0.7, is_dry_run=False,
            )
        return results

    def test_all_regime_rows_present(self):
        results = self._results([
            ("normal", True), ("stress", False),
            ("crisis", True), ("black_swan", True),
        ])
        metrics = compute_regime_metrics(results)
        regime_names = {m["regime"] for m in metrics}
        assert "ALL" in regime_names

    def test_direction_accuracy_correct(self):
        # 3 correct out of 3 in normal regime → 100%
        results = self._results([("normal", True)] * 3)
        metrics = compute_regime_metrics(results)
        normal = next(m for m in metrics if m["regime"] == "normal")
        assert normal["direction_accuracy"] == pytest.approx(1.0, abs=0.01)

    def test_no_signal_rate_calculation(self):
        # Mix of directional and no_signal
        rows = []
        for i in range(4):
            pt = "no_signal" if i < 2 else "directional"
            rows.append(PredictionResult(
                event_id=str(i), commodity="lng", regime="normal",
                as_of_date="2022-01-01", expected_direction="up",
                price_at_event=100.0, price_1w=None, price_2w=None,
                price_1m=None, actual_pct_1m=None, actual_direction=None,
                prediction_type=pt, confidence_score=0.5,
                predicted_direction=None, predicted_pct_midpoint=None,
                direction_correct=None, magnitude_mae=None,
                brier_score=None, composite_score=0.3, is_dry_run=False,
            ))
        metrics = compute_regime_metrics(rows)
        normal = next(m for m in metrics if m["regime"] == "normal")
        assert normal["no_signal_rate"] == pytest.approx(0.5, abs=0.01)

    def test_empty_results_returns_empty(self):
        metrics = compute_regime_metrics([])
        assert metrics == []
