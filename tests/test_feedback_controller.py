"""
Tests for ai_engine/feedback_controller.py and ai_engine/learning_store.py.

feedback_controller: pure functions (no DB, no external calls).
learning_store: DB-dependent — tested with the db_session fixture.
"""

import json
from contextlib import contextmanager
from unittest.mock import patch

import pytest

from ai_engine.feedback_controller import (
    ALPHA,
    BOTH_HIGH_THRESHOLD,
    DIRECTION_HIGH_THRESHOLD,
    MAGNITUDE_HIGH_THRESHOLD,
    ControlAdjustments,
    ErrorSignal,
    _damped_aggregate,
    compute_control_adjustments,
    compute_error_signal,
)


# ══════════════════════════════════════════════════════════════════════════════
# _damped_aggregate
# ══════════════════════════════════════════════════════════════════════════════

class TestDampedAggregate:
    def test_empty_list_returns_zero(self):
        assert _damped_aggregate([]) == 0.0

    def test_single_value_returned_unchanged(self):
        assert _damped_aggregate([0.7]) == 0.7

    def test_two_values_applies_alpha(self):
        # result = ALPHA * v1 + (1-ALPHA) * v0
        result = _damped_aggregate([0.0, 1.0], alpha=0.5)
        assert result == pytest.approx(0.5, abs=1e-9)

    def test_persistent_high_value_outweighs_single_spike_with_old_heavy_alpha(self):
        # With alpha=0.3 (old-heavy), sustained early high propagates further
        # than a single spike at the end. Validates damping direction property.
        sustained_early = _damped_aggregate([1.0, 1.0, 1.0, 0.0], alpha=0.3)
        single_late     = _damped_aggregate([0.0, 0.0, 0.0, 1.0], alpha=0.3)
        assert sustained_early > single_late

    def test_global_alpha_0_5_balances_recency_and_history(self):
        # With ALPHA=0.5 (current default), sustained early and single late spike
        # are equally weighted — recency and history have equal influence.
        assert ALPHA == 0.5, "Update this test if ALPHA constant changes"
        sustained_early = _damped_aggregate([1.0, 1.0, 1.0, 0.0])
        single_late     = _damped_aggregate([0.0, 0.0, 0.0, 1.0])
        assert sustained_early == pytest.approx(single_late, abs=1e-6)

    def test_all_same_values_returns_that_value(self):
        assert _damped_aggregate([0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5, abs=1e-6)

    def test_alpha_zero_returns_first_value(self):
        # alpha=0: result = (1-0)*prev = prev, so only first value survives
        result = _damped_aggregate([0.1, 0.9, 0.9], alpha=0.0)
        assert result == pytest.approx(0.1, abs=1e-9)

    def test_alpha_one_returns_last_value(self):
        # alpha=1: result = 1*v + 0*prev = v, so only last value survives
        result = _damped_aggregate([0.1, 0.2, 0.9], alpha=1.0)
        assert result == pytest.approx(0.9, abs=1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# compute_error_signal
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeErrorSignal:

    def _ev(self, *, dc=True, me=0.0, vc=True, fm=None):
        return {
            "direction_correct": dc,
            "magnitude_error": me,
            "volatility_correct": vc,
            "failure_modes": fm or [],
        }

    def test_empty_evaluations_returns_zero_signal(self):
        signal = compute_error_signal([], "lng", "price_spike")
        assert signal.n_evaluations == 0
        assert signal.e_total == 0.0
        assert signal.damped_direction_error == 0.0
        assert signal.damped_magnitude_error == 0.0

    def test_perfect_predictions_give_near_zero_error(self):
        evs = [self._ev(dc=True, me=0.0, vc=True) for _ in range(5)]
        signal = compute_error_signal(evs, "lng", "price_spike")
        assert signal.damped_direction_error == pytest.approx(0.0, abs=1e-6)
        assert signal.damped_magnitude_error == pytest.approx(0.0, abs=1e-6)
        assert signal.e_total == pytest.approx(0.0, abs=1e-6)

    def test_all_wrong_direction_gives_high_error(self):
        evs = [self._ev(dc=False, me=0.0, vc=True) for _ in range(10)]
        signal = compute_error_signal(evs, "lng", "price_spike")
        assert signal.damped_direction_error > 0.9

    def test_none_direction_gives_partial_error(self):
        evs = [self._ev(dc=None, me=0.0, vc=True) for _ in range(5)]
        signal = compute_error_signal(evs, "lng", "price_spike")
        # direction_correct=None contributes 0.5 per evaluation
        assert 0.4 < signal.damped_direction_error < 0.6

    def test_large_magnitude_error_increases_e_total(self):
        evs_low = [self._ev(dc=True, me=1.0) for _ in range(5)]
        evs_high = [self._ev(dc=True, me=15.0) for _ in range(5)]
        sig_low = compute_error_signal(evs_low, "lng", "price_spike")
        sig_high = compute_error_signal(evs_high, "lng", "price_spike")
        assert sig_high.e_total > sig_low.e_total

    def test_n_evaluations_matches_input(self):
        evs = [self._ev() for _ in range(7)]
        signal = compute_error_signal(evs, "copper", "sentiment_shift")
        assert signal.n_evaluations == 7

    def test_failure_modes_deduplicated(self):
        evs = [self._ev(fm=["demand miss", "demand miss"]) for _ in range(3)]
        signal = compute_error_signal(evs, "soybeans", "price_spike")
        assert len(set(signal.failure_modes)) == len(signal.failure_modes)

    def test_failure_modes_capped_at_five(self):
        many_fm = [f"mode_{i}" for i in range(20)]
        evs = [self._ev(fm=many_fm)]
        signal = compute_error_signal(evs, "lng", "price_spike")
        assert len(signal.failure_modes) <= 5

    def test_commodity_and_anomaly_type_propagated(self):
        signal = compute_error_signal([], "copper", "ais_vessel_drop")
        assert signal.commodity == "copper"
        assert signal.anomaly_type == "ais_vessel_drop"


# ══════════════════════════════════════════════════════════════════════════════
# compute_control_adjustments
# ══════════════════════════════════════════════════════════════════════════════

class TestComputeControlAdjustments:

    def _signal(self, *, direction=0.0, magnitude=0.0, volatility=0.0,
                n=5, failure_modes=None):
        return ErrorSignal(
            commodity="lng",
            anomaly_type="price_spike",
            n_evaluations=n,
            damped_direction_error=direction,
            damped_magnitude_error=magnitude,
            damped_volatility_error=volatility,
            e_total=(0.4 * direction + 0.4 * min(1.0, magnitude / 10.0) + 0.2 * volatility),
            failure_modes=failure_modes or [],
        )

    def test_moderate_signal_no_adjustments(self):
        # direction=0.3 is between 0.2 and 0.6, magnitude=1.0 < 5.0 → no rules fire
        adj = compute_control_adjustments(self._signal(direction=0.3, magnitude=1.0, n=5))
        assert adj.is_empty()

    def test_high_direction_error_increases_confidence_threshold(self):
        adj = compute_control_adjustments(
            self._signal(direction=DIRECTION_HIGH_THRESHOLD + 0.1)
        )
        assert adj.confidence_threshold == "increase"

    def test_low_direction_error_decreases_confidence_threshold(self):
        adj = compute_control_adjustments(
            self._signal(direction=0.1, n=5)
        )
        assert adj.confidence_threshold == "decrease"

    def test_high_magnitude_widens_scenarios(self):
        adj = compute_control_adjustments(
            self._signal(magnitude=MAGNITUDE_HIGH_THRESHOLD + 1.0)
        )
        assert adj.scenario_complexity == "increase"

    def test_both_high_reduces_analogy_reliance(self):
        adj = compute_control_adjustments(
            self._signal(direction=BOTH_HIGH_THRESHOLD + 0.1,
                         magnitude=MAGNITUDE_HIGH_THRESHOLD + 1.0)
        )
        assert adj.analogy_reliance == "decrease"

    def test_demand_miss_injects_rule(self):
        adj = compute_control_adjustments(
            self._signal(failure_modes=["demand miss"])
        )
        assert any("demand" in r.lower() for r in adj.driver_rules)

    def test_sentiment_overweight_injects_rule(self):
        adj = compute_control_adjustments(
            self._signal(failure_modes=["sentiment overweight"])
        )
        assert any("sentiment" in r.lower() for r in adj.driver_rules)

    def test_inventory_miss_injects_rule(self):
        adj = compute_control_adjustments(
            self._signal(failure_modes=["inventory miss"])
        )
        assert any("inventor" in r.lower() or "stock" in r.lower() for r in adj.driver_rules)

    def test_to_text_returns_non_empty_when_adjustments_exist(self):
        adj = compute_control_adjustments(
            self._signal(direction=DIRECTION_HIGH_THRESHOLD + 0.1)
        )
        text = adj.to_text()
        assert len(text) > 0
        assert text != "No control adjustments required."

    def test_to_text_returns_no_adjustments_when_empty(self):
        # direction=0.3 (between thresholds), magnitude=1.0 → no rules fire
        adj = compute_control_adjustments(self._signal(direction=0.3, magnitude=1.0, n=5))
        assert adj.to_text() == "No control adjustments required."


# ══════════════════════════════════════════════════════════════════════════════
# ControlAdjustments.is_empty
# ══════════════════════════════════════════════════════════════════════════════

class TestControlAdjustmentsIsEmpty:
    def test_default_is_empty(self):
        assert ControlAdjustments().is_empty()

    def test_confidence_threshold_set_not_empty(self):
        adj = ControlAdjustments(confidence_threshold="increase")
        assert not adj.is_empty()

    def test_driver_rules_set_not_empty(self):
        adj = ControlAdjustments(driver_rules=["always include demand"])
        assert not adj.is_empty()

    def test_scenario_complexity_set_not_empty(self):
        adj = ControlAdjustments(scenario_complexity="increase")
        assert not adj.is_empty()

    def test_analogy_reliance_set_not_empty(self):
        adj = ControlAdjustments(analogy_reliance="decrease")
        assert not adj.is_empty()


# ══════════════════════════════════════════════════════════════════════════════
# learning_store.get_learning_context (DB-dependent)
# ══════════════════════════════════════════════════════════════════════════════

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


class TestLearningStoreGetLearningContext:

    def _run(self, db_session, commodity="lng", anomaly_type="price_spike"):
        fake_session = _fake_session_factory(db_session)
        with patch("ai_engine.learning_store.get_session", side_effect=fake_session):
            from ai_engine.learning_store import get_learning_context
            return get_learning_context(commodity, anomaly_type)

    def test_returns_empty_string_with_no_data(self, db_session):
        result = self._run(db_session)
        assert result == ""

    def test_returns_string_type(self, db_session):
        result = self._run(db_session)
        assert isinstance(result, str)


# ══════════════════════════════════════════════════════════════════════════════
# learning_store._fetch_evaluations — cross-commodity pool
# ══════════════════════════════════════════════════════════════════════════════

class TestLearningStoreCrossCommodity:
    """
    Verify that _fetch_evaluations pulls from other commodities (same anomaly_type)
    as a soft prior, prepended before same-commodity rows.
    """

    def _run_fetch(self, db_session, commodity="lng", anomaly_type="price_spike"):
        fake_session = _fake_session_factory(db_session)
        with patch("ai_engine.learning_store.get_session", side_effect=fake_session):
            from ai_engine.learning_store import _fetch_evaluations
            return _fetch_evaluations(commodity, anomaly_type)

    def test_no_data_returns_empty(self, db_session):
        evals, insights = self._run_fetch(db_session)
        assert evals == []
        assert insights == []

    def test_cross_commodity_rows_prepended_to_same_commodity(self, db_session):
        """
        _fetch_eval_rows is patched to return distinct lists per commodity.
        Cross-commodity rows must appear before same-commodity rows in the output.
        """
        import json as _json
        from unittest.mock import patch as _patch

        same_row = (
            _json.dumps({"direction_correct": True, "magnitude_error": 1.0}),
            _json.dumps([]),
            "same-commodity insight",
            "same adjustment",
        )
        cross_row = (
            _json.dumps({"direction_correct": False, "magnitude_error": 5.0}),
            _json.dumps([]),
            "cross-commodity insight",
            "cross adjustment",
        )

        def fake_fetch_rows(commodity, anomaly_type, limit):
            if commodity == "lng":
                return [same_row]
            return [cross_row]

        with _patch("ai_engine.learning_store._fetch_eval_rows", side_effect=fake_fetch_rows):
            from ai_engine.learning_store import _fetch_evaluations
            evals, insights = _fetch_evaluations("lng", "price_spike")

        # cross rows are prepended; same row is last (most influential in damping)
        assert len(evals) >= 2
        # Last eval should be same-commodity (direction_correct=True)
        assert evals[-1]["direction_correct"] is True
        # Earlier evals should include cross-commodity (direction_correct=False)
        assert any(e["direction_correct"] is False for e in evals[:-1])

    def test_insights_only_from_same_commodity(self, db_session):
        """
        Cross-commodity rows should NOT surface in the insights list
        (only same-commodity insights appear in the prompt).
        """
        import json as _json
        from unittest.mock import patch as _patch

        same_row = (
            _json.dumps({"direction_correct": True, "magnitude_error": 1.0}),
            _json.dumps([]),
            "same-only insight",
            "same adjustment",
        )
        cross_row = (
            _json.dumps({"direction_correct": False, "magnitude_error": 5.0}),
            _json.dumps([]),
            "cross insight should not appear",
            "cross adjustment",
        )

        def fake_fetch_rows(commodity, anomaly_type, limit):
            if commodity == "lng":
                return [same_row]
            return [cross_row]

        with _patch("ai_engine.learning_store._fetch_eval_rows", side_effect=fake_fetch_rows):
            from ai_engine.learning_store import _fetch_evaluations
            _, insights = _fetch_evaluations("lng", "price_spike")

        insight_texts = [i for i, _ in insights]
        assert "same-only insight" in insight_texts
        assert all("cross insight" not in t for t in insight_texts)
