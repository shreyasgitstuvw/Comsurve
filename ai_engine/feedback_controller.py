"""
PID-inspired adaptive feedback controller.

Reads historical PredictionEvaluation records and computes a damped error signal
that drives rule-based control adjustments injected into future prediction prompts.

Design:
  - Proportional: current error level → confidence threshold adjustment
  - Integral:     sustained bias over N evaluations → driver rule injection
  - Derivative:   direction-reversal across regimes → analogy reliance reduction

Error signal components (all normalised 0.0-1.0):
  e_direction  = fraction of recent evaluations with direction_correct == False
  e_magnitude  = mean |magnitude_error| / 10.0  (10 pp = saturates at 1.0)
  e_volatility = fraction where volatility_correct == False

Damping (exponential smoothing, oldest-first):
  result = values[0]
  for v in values[1:]:
      result = alpha * v + (1-alpha) * result

This weights recent errors more heavily while not discarding history.
"""

from dataclasses import dataclass, field
from typing import Optional


ALPHA = 0.5          # damping coefficient (recent weight); 0.5 = equal recency/history
W1, W2, W3 = 0.4, 0.4, 0.2   # direction, magnitude, volatility weights

# Thresholds that trigger control adjustments
DIRECTION_HIGH_THRESHOLD = 0.6   # >60% direction failures → raise confidence bar
MAGNITUDE_HIGH_THRESHOLD = 5.0   # mean error >5pp → widen scenario ranges
BOTH_HIGH_THRESHOLD = 0.55       # both direction+magnitude → reduce analogy reliance


@dataclass
class ErrorSignal:
    """Damped composite error signal for a commodity+anomaly_type pair."""
    commodity: str
    anomaly_type: str
    n_evaluations: int              # number of evaluations used
    damped_direction_error: float   # 0.0-1.0
    damped_magnitude_error: float   # absolute pp (not normalised for readability)
    damped_volatility_error: float  # 0.0-1.0
    e_total: float                  # composite 0.0-1.0
    failure_modes: list[str] = field(default_factory=list)


@dataclass
class ControlAdjustments:
    """
    Rule-based control variables for prompt engineering.
    None means "no adjustment needed for this variable".
    """
    confidence_threshold: Optional[str] = None   # "increase" | "decrease"
    scenario_complexity: Optional[str] = None    # "increase" (wider ranges)
    driver_rules: list[str] = field(default_factory=list)  # injected rules
    analogy_reliance: Optional[str] = None       # "decrease"

    def is_empty(self) -> bool:
        return (
            self.confidence_threshold is None
            and self.scenario_complexity is None
            and not self.driver_rules
            and self.analogy_reliance is None
        )

    def to_text(self) -> str:
        """Render as compact bullet-point text for prompt injection."""
        lines = []
        if self.confidence_threshold == "increase":
            lines.append("- Raise your internal confidence bar: require stronger corroborating signals before a directional call.")
        elif self.confidence_threshold == "decrease":
            lines.append("- You have been systematically under-confident; trust well-corroborated signals more.")

        if self.scenario_complexity == "increase":
            lines.append("- Widen scenario price-move ranges; past predictions underestimated actual magnitude.")

        for rule in self.driver_rules:
            lines.append(f"- {rule}")

        if self.analogy_reliance == "decrease":
            lines.append("- Reduce reliance on historical analogs; recent errors suggest this event type has lower historical repeatability.")

        return "\n".join(lines) if lines else "No control adjustments required."


# ------------------------------------------------------------------
# Core functions
# ------------------------------------------------------------------

def _damped_aggregate(values: list[float], alpha: float = ALPHA) -> float:
    """
    Exponential smoothing over a chronological sequence (oldest-first).
    Single value → returned unchanged.
    """
    if not values:
        return 0.0
    result = values[0]
    for v in values[1:]:
        result = alpha * v + (1 - alpha) * result
    return result


def compute_error_signal(
    evaluations: list[dict],
    commodity: str,
    anomaly_type: str,
) -> ErrorSignal:
    """
    Compute damped error signal from a list of evaluation dicts (oldest-first).

    Each dict must contain:
      "direction_correct"   : bool | None
      "magnitude_error"     : float
      "volatility_correct"  : bool | None
      "failure_modes"       : list[str]  (optional)
    """
    if not evaluations:
        return ErrorSignal(
            commodity=commodity,
            anomaly_type=anomaly_type,
            n_evaluations=0,
            damped_direction_error=0.0,
            damped_magnitude_error=0.0,
            damped_volatility_error=0.0,
            e_total=0.0,
        )

    direction_errors, magnitude_errors, volatility_errors = [], [], []
    all_failure_modes: list[str] = []

    for ev in evaluations:
        dc = ev.get("direction_correct")
        direction_errors.append(0.0 if dc is True else (0.5 if dc is None else 1.0))
        magnitude_errors.append(float(ev.get("magnitude_error") or 0.0))
        vc = ev.get("volatility_correct")
        volatility_errors.append(0.0 if vc is True else 1.0)
        all_failure_modes.extend(ev.get("failure_modes", []))

    dd = _damped_aggregate(direction_errors)
    dm = _damped_aggregate(magnitude_errors)
    dv = _damped_aggregate(volatility_errors)

    dm_normalised = min(1.0, dm / 10.0)
    e_total = W1 * dd + W2 * dm_normalised + W3 * dv

    # Deduplicate failure modes (most recent first)
    seen: set[str] = set()
    unique_modes: list[str] = []
    for fm in reversed(all_failure_modes):
        if fm not in seen:
            seen.add(fm)
            unique_modes.append(fm)
    unique_modes = unique_modes[:5]  # cap at 5

    return ErrorSignal(
        commodity=commodity,
        anomaly_type=anomaly_type,
        n_evaluations=len(evaluations),
        damped_direction_error=round(dd, 4),
        damped_magnitude_error=round(dm, 4),
        damped_volatility_error=round(dv, 4),
        e_total=round(e_total, 4),
        failure_modes=unique_modes,
    )


_DEMAND_PATTERNS = ("demand miss", "demand-side", "demand side", "demand reversal", "demand driver")
_SENTIMENT_PATTERNS = ("sentiment over", "over-weighted sentiment", "overweight sentiment",
                       "sentiment overestim", "over-estimate sentiment")
_INVENTORY_PATTERNS = ("inventory", "stock level", "storage level", "stockpile")
_SUPPLY_PATTERNS = ("supply miss", "supply-side", "supply constraint", "supply disruption missed")
_GEOPOLITICAL_PATTERNS = ("geopolit", "political risk", "sanctions missed", "trade policy")


def compute_control_adjustments(signal: ErrorSignal) -> ControlAdjustments:
    """
    Apply rule-based control logic to an ErrorSignal.
    Returns ControlAdjustments with non-null fields only where rules fire.
    """
    adj = ControlAdjustments()
    failure_text = " ".join(signal.failure_modes).lower()

    # P-channel: direction error threshold
    if signal.damped_direction_error > DIRECTION_HIGH_THRESHOLD:
        adj.confidence_threshold = "increase"
    elif signal.damped_direction_error < 0.2 and signal.n_evaluations >= 3:
        adj.confidence_threshold = "decrease"

    # P-channel: magnitude error (raw pp, not normalised)
    if signal.damped_magnitude_error > MAGNITUDE_HIGH_THRESHOLD:
        adj.scenario_complexity = "increase"

    # I-channel: sustained failure patterns → injected driver rules
    # Broader pattern matching than single keyword pairs to catch more failure modes.
    if any(p in failure_text for p in _DEMAND_PATTERNS):
        adj.driver_rules.append(
            "Always include an explicit demand-side scenario; past predictions missed demand-driven reversals."
        )
    if any(p in failure_text for p in _SENTIMENT_PATTERNS):
        adj.driver_rules.append(
            "Down-weight sentiment signals; past evaluations show sentiment was over-estimated as a price driver."
        )
    if any(p in failure_text for p in _INVENTORY_PATTERNS):
        adj.driver_rules.append(
            "Validate inventory/stock levels as a key driver; inventory effects were repeatedly missed."
        )
    if any(p in failure_text for p in _SUPPLY_PATTERNS):
        adj.driver_rules.append(
            "Include explicit supply-side constraints; past predictions underweighted supply-driven moves."
        )
    if any(p in failure_text for p in _GEOPOLITICAL_PATTERNS):
        adj.driver_rules.append(
            "Incorporate geopolitical risk explicitly; past evaluations identified political drivers as missed."
        )

    # D-channel: both direction AND magnitude persistently high → reduce analogy reliance.
    # High combined error means historical analogs are poor predictors for this regime.
    if (signal.damped_direction_error > BOTH_HIGH_THRESHOLD
            and signal.damped_magnitude_error > MAGNITUDE_HIGH_THRESHOLD):
        adj.analogy_reliance = "decrease"

    # D-channel extension: composite error alone very high → also reduce analogy reliance
    if signal.e_total > 0.7 and signal.n_evaluations >= 3:
        adj.analogy_reliance = "decrease"

    return adj
