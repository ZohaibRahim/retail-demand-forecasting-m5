"""Price features: a weekly price is usable for target t only if its week ended on or before t-28."""

import numpy as np
import pandas as pd
import pytest

from src.features import PRICE_FEATURES, add_price_features, cutoff_week_index
from tests.synthetic import make_calendar, make_prices

N_DAYS = 140
N_WEEKS = N_DAYS // 7
CAL = make_calendar(N_DAYS)


def _frame(ids=("A",)):
    f = pd.DataFrame([(i, d) for i in ids for d in range(1, N_DAYS + 1)], columns=["id", "d"])
    f["id"] = f["id"].astype("category")
    return f


def _feat(weekly, t, sid="A"):
    f = add_price_features(_frame(tuple(weekly)), make_prices(weekly), CAL)
    return f[(f["id"] == sid) & (f["d"] == t)].iloc[0][PRICE_FEATURES]


DISTINCT = [float(i + 1) for i in range(N_WEEKS)]  # week i has price i+1


def test_cutoff_week_uses_only_fully_ended_weeks():
    idx, _ = cutoff_week_index(np.array([97, 98, 100, 104, 105]), CAL)
    # t-28 = 69 (mid week 9: days 64-70)  -> week 8
    # t-28 = 70 (last day of week 9)      -> week 9
    # t-28 = 72 (mid week 10)             -> week 9
    # t-28 = 76 (mid week 10)             -> week 9
    # t-28 = 77 (last day of week 10)     -> week 10
    np.testing.assert_array_equal(idx, [8, 9, 9, 9, 10])
    assert cutoff_week_index(np.array([34]), CAL)[0][0] == -1  # t-28=6: no week finished yet


def test_future_prices_cannot_change_price_features():
    t = 100  # cutoff week 9
    base = _feat({"A": DISTINCT}, t)
    assert base["price_lag_28"] == 10.0
    future = DISTINCT[:10] + [999.0 + i for i in range(N_WEEKS - 10)]  # every week after week 9
    after = _feat({"A": future}, t)
    np.testing.assert_array_equal(base.to_numpy(float), after.to_numpy(float))


def test_weekly_boundary_mid_week_and_week_end():
    # t=100: t-28=72 falls inside week 10 (days 71-77); week 10 must not be used.
    changed_wk10 = DISTINCT.copy()
    changed_wk10[10] = 555.0
    assert _feat({"A": changed_wk10}, 100)["price_lag_28"] == 10.0
    # t=98: t-28=70 is the last day of week 9, so week 9 is fully known.
    assert _feat({"A": DISTINCT}, 98)["price_lag_28"] == 10.0
    # t=97: t-28=69 is mid week 9, so the previous week (8) is used.
    assert _feat({"A": DISTINCT}, 97)["price_lag_28"] == 9.0
    changed_wk9 = DISTINCT.copy()
    changed_wk9[9] = 777.0
    assert _feat({"A": changed_wk9}, 97)["price_lag_28"] == 9.0
    assert _feat({"A": changed_wk9}, 98)["price_lag_28"] == 777.0


def test_week_over_week_change_is_lagged():
    f = _feat({"A": DISTINCT}, 100)  # cutoff week 9 (price 10) vs week 8 (price 9)
    assert f["price_wow_change"] == pytest.approx(10 / 9 - 1)


def test_forward_fill_never_uses_future_value():
    gap = [1.0, 2.0, 3.0, 4.0, 5.0, None, None, None, 99.0] + [99.0] * (N_WEEKS - 9)
    # cutoff week 6 (t-28 = 49, last day of week 6) -> t = 77: forward-filled from week 4
    assert _feat({"A": gap}, 77)["price_lag_28"] == 5.0
    # series launching at week 3: before that there is no back-fill
    late = [None, None, None, 7.0] + [7.0] * (N_WEEKS - 4)
    f = _feat({"A": late}, 28 + 14)  # cutoff week 1
    assert np.isnan(f["price_lag_28"])
    assert np.isnan(f["price_rel_hist"])


def test_historical_reference_uses_observed_prices_only():
    prices = [1.0, 3.0, None, None, None] + [50.0] * (N_WEEKS - 5)
    # cutoff week 4 (t-28 = 35) -> t = 63; observed weeks 0,1 -> mean 2; last known price 3
    f = _feat({"A": prices}, 63)
    assert f["price_lag_28"] == 3.0
    assert f["price_rel_hist"] == pytest.approx(3.0 / 2.0)  # not 3 / mean(1,3,3,3,3)
    # changing future price weeks cannot alter the reference
    future = prices[:5] + [123.0] * (N_WEEKS - 5)
    assert _feat({"A": future}, 63)["price_rel_hist"] == pytest.approx(1.5)


def test_reference_is_per_series():
    f_a = _feat({"A": [1.0] * N_WEEKS, "B": [100.0] * N_WEEKS}, 100, "A")
    assert f_a["price_rel_hist"] == pytest.approx(1.0)
