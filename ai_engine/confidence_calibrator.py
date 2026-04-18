"""
Confidence calibration for LLM prediction outputs.

Two-mode operation:
1. Platt scaling (logistic regression on historical eval pairs) when >= MIN_SAMPLES exist
2. Shrinkage prior (pull toward 0.5) when data is insufficient

Platt scaling fits sigmoid(A*x + B) on (raw_confidence, direction_correct) pairs
using gradient descent — no numpy/sklearn dependency.

Called by prediction_engine.py before persisting to signal_alerts.
"""

import json
import math

from sqlalchemy import text

from shared.db import get_session
from shared.logger import get_logger

logger = get_logger(__name__)

MIN_SAMPLES = 15        # evaluations needed before Platt scaling activates
SHRINKAGE_ALPHA = 0.7   # weight on raw confidence; 1-SHRINKAGE_ALPHA pulls toward 0.5
_GD_ITER = 200          # gradient descent iterations for Platt fitting
_GD_LR = 0.1            # gradient descent learning rate


def _fetch_calibration_pairs(commodity: str) -> list[tuple[float, int]]:
    """
    Fetch (raw_confidence, direction_correct_int) pairs from past evaluations.
    Skips magnitude_only/no_signal rows (direction_correct is None).
    """
    with get_session() as session:
        rows = session.execute(
            text("""
                SELECT sa.prediction_confidence, pe.prediction_accuracy_json
                FROM prediction_evaluations pe
                JOIN signal_alerts sa ON sa.id = pe.signal_alert_id
                WHERE pe.commodity = :commodity
                  AND sa.prediction_confidence IS NOT NULL
                ORDER BY pe.created_at DESC
                LIMIT 100
            """),
            {"commodity": commodity},
        ).fetchall()

    pairs: list[tuple[float, int]] = []
    for raw_conf, acc_json in rows:
        try:
            acc = json.loads(acc_json or "{}")
            dc = acc.get("direction_correct")
            if dc is None:
                continue
            pairs.append((float(raw_conf), int(bool(dc))))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return pairs


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, x))))


def _fit_platt(pairs: list[tuple[float, int]]) -> tuple[float, float]:
    """Fit logistic P(correct) = sigmoid(A*x + B) via gradient descent."""
    A, B = 1.0, 0.0
    n = len(pairs)
    for _ in range(_GD_ITER):
        dA = dB = 0.0
        for x, y in pairs:
            p = _sigmoid(A * x + B)
            err = p - y
            dA += err * x
            dB += err
        A -= _GD_LR * dA / n
        B -= _GD_LR * dB / n
    return A, B


def _shrinkage(raw: float) -> float:
    """Pull raw LLM confidence toward the 0.5 uninformative prior."""
    return SHRINKAGE_ALPHA * raw + (1.0 - SHRINKAGE_ALPHA) * 0.5


def calibrate_confidence(raw_confidence: float, commodity: str) -> float:
    """
    Return calibrated confidence for a given commodity.

    Uses Platt scaling when >= MIN_SAMPLES evaluations are available;
    falls back to shrinkage prior otherwise.
    """
    try:
        pairs = _fetch_calibration_pairs(commodity)
        if len(pairs) >= MIN_SAMPLES:
            A, B = _fit_platt(pairs)
            calibrated = _sigmoid(A * raw_confidence + B)
            calibrated = round(max(0.01, min(0.99, calibrated)), 4)
            logger.info(
                "confidence_calibrated_platt",
                commodity=commodity,
                raw=raw_confidence,
                calibrated=calibrated,
                n_samples=len(pairs),
            )
            return calibrated
    except Exception as exc:
        logger.warning("platt_scaling_failed", commodity=commodity, error=str(exc))

    calibrated = round(_shrinkage(raw_confidence), 4)
    logger.debug(
        "confidence_calibrated_shrinkage",
        commodity=commodity,
        raw=raw_confidence,
        calibrated=calibrated,
    )
    return calibrated
