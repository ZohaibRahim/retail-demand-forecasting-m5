"""Frozen baseline definitions (METHODOLOGY section 8).

Every baseline receives only the history matrix ending at the forecast origin O
(shape: n_series x n_days, last column = day O). Forecast-window actuals are never an input.
"""

import numpy as np

from src import config

H = config.HORIZON


def _check(history: np.ndarray) -> np.ndarray:
    history = np.asarray(history, dtype=np.float64)
    if history.shape[1] < H:
        raise ValueError(f"Need at least {H} history days")
    return history


def lag28(history: np.ndarray) -> np.ndarray:
    """Target O+h gets the actual from day O+h-28, i.e. history days O-27..O in order."""
    return _check(history)[:, -H:].copy()


def trailing28_mean(history: np.ndarray) -> np.ndarray:
    """Mean of the final 28 observed days through O, held flat for all 28 targets."""
    last = _check(history)[:, -H:]
    return np.repeat(last.mean(axis=1, keepdims=True), H, axis=1)


def weekly_seasonal_naive(history: np.ndarray) -> np.ndarray:
    """Repeat the final known 7-day pattern (days O-6..O) four times."""
    pattern = _check(history)[:, -7:]
    return np.tile(pattern, H // 7)


BASELINES = {
    "lag28": lag28,
    "trailing28_mean": trailing28_mean,
    "weekly_seasonal_naive": weekly_seasonal_naive,
}
