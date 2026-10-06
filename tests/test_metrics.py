import numpy as np
import pandas as pd
import pytest

from src.metrics import improvement_pct, rmsse_scale, scaled_rmse_by_horizon, summarize
from src.validation import history_matrix


def test_scale_uses_training_history_only():
    rng = np.random.default_rng(0)
    full = rng.integers(0, 5, (1, 200)).astype(float)
    wide = pd.DataFrame(full, columns=np.arange(1, 201))
    q1 = rmsse_scale(history_matrix(wide, wide.index, 150))["q"].iloc[0]
    full[:, 150:] = 1000.0  # days after the origin
    wide2 = pd.DataFrame(full, columns=np.arange(1, 201))
    q2 = rmsse_scale(history_matrix(wide2, wide2.index, 150))["q"].iloc[0]
    assert q1 == q2


def test_scale_starts_at_first_nonzero_sale():
    x = np.array([[0, 0, 0, 2, 4, 2, 4] + [2, 4] * 15], dtype=float)
    res = rmsse_scale(x)
    diffs = np.diff(x[0, 3:])
    assert res["q"].iloc[0] == pytest.approx(np.mean(diffs**2))
    assert res["n_diffs"].iloc[0] == len(diffs)


def test_scale_uses_history_before_modelling_start():
    # first sale on day 10, long before any "d_1000"-style target start; all of it counts
    x = np.zeros((1, 1200))
    x[0, 9::2] = 3.0
    res = rmsse_scale(x)
    assert res["n_diffs"].iloc[0] == 1200 - 10
    assert res["q"].iloc[0] == pytest.approx(9.0)


def test_invalid_scales_detected():
    x = np.vstack(
        [
            np.zeros(60),  # never sold
            np.full(60, 5.0),  # constant: q = 0
            np.r_[np.zeros(31), [1, 0] * 14, 1.0],  # 29 obs from first sale -> 28 diffs: valid
            np.r_[np.zeros(32), [1, 0] * 14],  # 28 obs from first sale -> 27 diffs: invalid
        ]
    )
    res = rmsse_scale(x)
    assert list(res["valid"]) == [False, False, True, False]
    assert list(res["n_diffs"]) == [0, 59, 28, 27]


def test_scaled_rmse_by_horizon_hand_example():
    actual = np.array([[1.0, 2.0], [3.0, 4.0]])
    pred = np.array([[0.0, 2.0], [1.0, 6.0]])
    q = np.array([1.0, 4.0])
    # h1: (1/1 + 4/4)/2 = 1 -> 1 ; h2: (0/1 + 4/4)/2 = 0.5 -> sqrt(0.5)
    np.testing.assert_allclose(scaled_rmse_by_horizon(actual, pred, q), [1.0, np.sqrt(0.5)])


def test_summary_pools_mae_rmse_and_scores_rmsse_per_series():
    actual = np.array([[1.0, 2.0], [3.0, 4.0]])
    pred = np.array([[0.0, 2.0], [1.0, 6.0]])
    q = np.array([1.0, 4.0])
    s = summarize(actual, pred, q)
    assert s["mae"] == pytest.approx((1 + 0 + 2 + 2) / 4)
    assert s["rmse"] == pytest.approx(np.sqrt((1 + 0 + 4 + 4) / 4))
    per_series = [np.sqrt(0.5 / 1), np.sqrt(4 / 4)]
    assert s["mean_rmsse"] == pytest.approx(np.mean(per_series))
    assert s["max_rmsse"] == pytest.approx(1.0)
    assert s["n_series"] == 2


def test_improvement_pct_sign():
    assert improvement_pct(1.0, 0.8) == pytest.approx(20.0)
    assert improvement_pct(1.0, 1.1) == pytest.approx(-10.0)
