import numpy as np
import pandas as pd
import pytest

from src import run
from src.metrics import score_models
from src.validation import (
    HOLDOUT,
    VALIDATION_FOLDS,
    eligibility,
    first_price_week,
    training_mask,
)
from tests.synthetic import make_calendar, week_id


def test_validation_fold_boundaries():
    expected = [(1829, 1830, 1857), (1857, 1858, 1885), (1885, 1886, 1913)]
    got = [(f.origin, f.start, f.end) for f in VALIDATION_FOLDS]
    assert got == expected
    for f in VALIDATION_FOLDS:
        assert len(f.days) == 28 and f.days[0] == f.start and f.days[-1] == f.end


def test_holdout_boundaries():
    assert (HOLDOUT.origin, HOLDOUT.start, HOLDOUT.end) == (1913, 1914, 1941)
    assert all(f.end <= HOLDOUT.origin for f in VALIDATION_FOLDS)


CAL = make_calendar(140)  # week i spans days 7i+1 .. 7i+7
ORIGIN = 100  # day 100 is inside week 14 (days 99-105)


def _price_rows(first_week: dict, through_week=19):
    rows = [
        {"id": sid, "wm_yr_wk": week_id(w), "sell_price": 1.0}
        for sid, fw in first_week.items()
        for w in range(fw, through_week + 1)
    ]
    return pd.DataFrame(rows)


def _scales(ids, valid=True):
    return pd.DataFrame({"q": 1.0, "n_diffs": 100, "valid": valid}, index=pd.Index(ids))


def test_active_start_never_uses_horizon_information():
    # A: first priced week 0 (ended day 7). B: week 14 straddles origin (days 99-105).
    # C: first priced week 15 is entirely inside the horizon.
    info = first_price_week(_price_rows({"A": 0, "B": 14, "C": 15}), CAL)
    ids = pd.Index(["A", "B", "C"])
    e = eligibility(ids, info, _scales(ids), ORIGIN)
    assert e.loc["B", "status"] == "not_yet_active"  # F2: week not fully ended by origin
    assert e.loc["C", "status"] == "not_yet_active"
    # adding/removing price weeks inside the horizon changes nothing at this origin
    info2 = first_price_week(_price_rows({"A": 0, "B": 14, "C": 15}, through_week=14), CAL)
    e2 = eligibility(ids, info2.reindex(ids), _scales(ids), ORIGIN)
    pd.testing.assert_series_equal(e["status"], e2["status"])


def test_84_day_minimum_active_history():
    # week 2 starts day 15 -> 86 active days at origin 100; week 3 starts day 22 -> 79 days
    info = first_price_week(_price_rows({"ok": 2, "short": 3}), CAL)
    ids = pd.Index(["ok", "short"])
    e = eligibility(ids, info, _scales(ids), ORIGIN)
    assert e.loc["ok", "status"] == "eligible" and e.loc["ok", "active_days"] == 86
    assert e.loc["short", "status"] == "insufficient_history"
    # exact boundary: 84 active days is enough, 83 is not
    edge = pd.DataFrame({"first_week_start": [17.0, 18.0], "first_week_end": [23.0, 24.0]}, index=["e84", "e83"])
    e = eligibility(edge.index, edge, _scales(edge.index), ORIGIN)
    assert list(e["status"]) == ["eligible", "insufficient_history"]


def test_invalid_scale_excluded_and_counted_separately():
    info = first_price_week(_price_rows({"A": 0, "B": 0}), CAL)
    ids = pd.Index(["A", "B"])
    sc = pd.DataFrame({"q": [1.0, 0.0], "n_diffs": [100, 100], "valid": [True, False]}, index=ids)
    e = eligibility(ids, info, sc, ORIGIN)
    assert list(e["status"]) == ["eligible", "invalid_scale"]


def test_training_mask_excludes_warmup_prelaunch_future_and_unfinished_first_week():
    frame = pd.DataFrame(
        {
            "d": [999, 1000, 1500, 1600, 1601, 1500, 1597],
            "first_week_start": [1.0, 1.0, 1550.0, 1.0, 1.0, 1597.0, 1597.0],
            "first_week_end": [7.0, 7.0, 1556.0, 7.0, 7.0, 1603.0, 1603.0],
        }
    )
    m = training_mask(frame, origin=1600)
    # warm-up, ok, pre-launch, ok (=origin), after origin, pre-launch, first week unfinished at origin
    assert list(m) == [False, True, False, True, False, False, False]


def test_every_model_scored_on_same_series_set():
    days = [1, 2]
    eligible = pd.Index(["a", "b"])
    actual = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=eligible, columns=days)
    scales = pd.Series([1.0, 1.0], index=eligible)
    preds = {
        "m1": pd.DataFrame([[1.0, 2.0], [3.0, 4.0], [9.0, 9.0]], index=["a", "b", "extra"], columns=days),
        "m2": pd.DataFrame([[0.0, 0.0], [0.0, 0.0]], index=eligible, columns=days),
    }
    res = score_models(preds, actual, scales, eligible)
    assert (res["n_series"] == 2).all()
    with pytest.raises(ValueError):
        score_models({"m3": preds["m2"].iloc[:1]}, actual, scales, eligible)


def test_holdout_requires_explicit_flag(local_tmp):
    with pytest.raises(run.HoldoutBlocked):
        run.check_holdout_allowed(False, local_tmp)
    run.check_holdout_allowed(True, local_tmp)  # allowed when flagged and no prior result


def test_holdout_cli_without_flag_exits_before_any_work(local_tmp, monkeypatch):
    monkeypatch.setattr(run.config, "RESULTS_DIR", local_tmp)
    called = []
    monkeypatch.setattr(run, "_run_tests", lambda: called.append("tests"))
    monkeypatch.setattr(run, "load_context", lambda **k: called.append("load"))
    with pytest.raises(SystemExit):
        run.main(["holdout"])
    assert called == []
    assert not (local_tmp / run.HOLDOUT_METRICS).exists()


def test_existing_holdout_result_not_overwritten(local_tmp):
    existing = local_tmp / run.HOLDOUT_METRICS
    existing.write_text("model,mean_rmsse\nlightgbm,0.5\n")
    with pytest.raises(run.HoldoutBlocked):
        run.check_holdout_allowed(True, local_tmp)
    assert existing.read_text() == "model,mean_rmsse\nlightgbm,0.5\n"
