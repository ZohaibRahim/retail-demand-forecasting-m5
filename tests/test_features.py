import numpy as np
import pytest

from src.features import SALES_FEATURES, add_sales_features
from tests.synthetic import make_frame

N = 200


def _features(sales_a, sales_b=None, start_a=1, start_b=1):
    sales_b = np.arange(N) * 2.0 if sales_b is None else sales_b
    frame = make_frame({"A": sales_a, "B": sales_b}, {"A": start_a, "B": start_b})
    return add_sales_features(frame)


def _row(frame, sid, d):
    return frame[(frame["id"] == sid) & (frame["d"] == d)].iloc[0]


def test_lag_28_correct():
    f = _features(np.arange(1, N + 1, dtype=float))  # sales on day d equals d
    assert _row(f, "A", 100)["lag_28"] == 72
    assert np.isnan(_row(f, "A", 28)["lag_28"])
    assert _row(f, "A", 29)["lag_28"] == 1


@pytest.mark.parametrize("k", [35, 42, 49])
def test_longer_lags_correct(k):
    f = _features(np.arange(1, N + 1, dtype=float))
    assert _row(f, "A", 150)[f"lag_{k}"] == 150 - k


def test_rolling_stats_computed_after_shift_28():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 20, N).astype(float)
    f = _features(s)
    t = 150
    window = lambda w: s[t - 28 - w : t - 28]  # days t-28-w+1 .. t-28 (0-based slice)  # noqa: E731
    r = _row(f, "A", t)
    assert r["rolling_mean_7"] == pytest.approx(window(7).mean(), rel=1e-6)
    assert r["rolling_mean_28"] == pytest.approx(window(28).mean(), rel=1e-6)
    assert r["rolling_mean_56"] == pytest.approx(window(56).mean(), rel=1e-6)
    assert r["rolling_std_28"] == pytest.approx(window(28).std(ddof=1), rel=1e-5)
    # first full rolling_mean_56 needs days 1..56 shifted by 28 -> target day 84
    assert np.isnan(_row(f, "A", 83)["rolling_mean_56"])
    assert not np.isnan(_row(f, "A", 84)["rolling_mean_56"])


def test_sales_after_t_minus_28_cannot_change_features():
    rng = np.random.default_rng(1)
    s = rng.integers(0, 10, N).astype(float)
    t = 150
    base = _row(_features(s), "A", t)[SALES_FEATURES]
    changed = s.copy()
    changed[t - 28 :] = 9999.0  # every day after t-28 (day t-27 onward, 1-based)
    after = _row(_features(changed), "A", t)[SALES_FEATURES]
    np.testing.assert_array_equal(base.to_numpy(float), after.to_numpy(float))
    # sanity: changing day t-28 itself does change lag_28
    changed2 = s.copy()
    changed2[t - 29] = 9999.0
    assert _row(_features(changed2), "A", t)["lag_28"] == 9999.0


def test_prelaunch_zeros_are_not_demand():
    s = np.ones(N)
    s[:49] = 0.0  # days 1..49 are pre-launch zeros; launch on day 50
    f = _features(s, start_a=50)
    assert np.isnan(_row(f, "A", 77)["lag_28"])  # day 49 pre-launch
    assert _row(f, "A", 78)["lag_28"] == 1.0  # day 50 launched
    # rolling window touching pre-launch days is missing, not a diluted mean
    assert np.isnan(_row(f, "A", 80)["rolling_mean_7"])  # window days 46..52
    assert _row(f, "A", 84)["rolling_mean_7"] == 1.0  # window days 50..56
    # series B launched on day 1 is unaffected
    assert _row(f, "B", 80)["lag_28"] == 2 * 51
