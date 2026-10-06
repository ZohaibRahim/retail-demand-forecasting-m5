import numpy as np
import pandas as pd

from src.baselines import lag28, trailing28_mean, weekly_seasonal_naive
from src.validation import Fold, actual_matrix, history_matrix

ORIGIN = 100


def _wide(values):
    values = np.atleast_2d(values)
    return pd.DataFrame(values, index=[f"s{i}" for i in range(len(values))],
                        columns=np.arange(1, values.shape[1] + 1))


def test_lag28_uses_value_exactly_28_days_earlier():
    hist = np.arange(1, ORIGIN + 1, dtype=float)[None, :]  # value on day d is d
    pred = lag28(hist)
    # target O+h -> day O+h-28
    np.testing.assert_array_equal(pred[0], ORIGIN + np.arange(1, 29) - 28)


def test_trailing28_mean_uses_last_28_known_days_including_zeros():
    hist = np.zeros((1, ORIGIN))
    hist[0, :-28] = 50.0  # older history must not matter
    hist[0, -28:] = [0, 4] * 14  # zeros are genuine observations
    pred = trailing28_mean(hist)
    assert pred.shape == (1, 28)
    np.testing.assert_allclose(pred, 2.0)


def test_weekly_seasonal_repeats_final_known_week():
    hist = np.arange(1, ORIGIN + 1, dtype=float)[None, :]
    pred = weekly_seasonal_naive(hist)
    pattern = np.arange(ORIGIN - 6, ORIGIN + 1)
    np.testing.assert_array_equal(pred[0], np.tile(pattern, 4))


def test_weekly_baseline_does_not_consume_forecast_window_actuals():
    rng = np.random.default_rng(0)
    full = rng.integers(0, 10, (2, ORIGIN + 28)).astype(float)
    wide = _wide(full)
    fold = Fold("synthetic", ORIGIN)
    pred = weekly_seasonal_naive(history_matrix(wide, wide.index, ORIGIN))
    # validation day 8 must be the value from day O-6, not the actual from validation day 1
    np.testing.assert_array_equal(pred[:, 7], full[:, ORIGIN - 7])
    actual = actual_matrix(wide, wide.index, fold)
    full[:, ORIGIN:] = 1e6  # change every forecast-window actual
    wide2 = _wide(full)
    for fn in (lag28, trailing28_mean, weekly_seasonal_naive):
        np.testing.assert_array_equal(fn(history_matrix(wide, wide.index, ORIGIN)),
                                      fn(history_matrix(wide2, wide2.index, ORIGIN)))
    assert not np.array_equal(actual, actual_matrix(wide2, wide2.index, fold))
