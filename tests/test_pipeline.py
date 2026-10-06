"""End-to-end run of preprocess -> features -> baselines -> LightGBM on tiny synthetic M5-shaped files."""

import numpy as np
import pandas as pd
import pytest

from src import config, data, model, plots, run
from src.validation import HOLDOUT, VALIDATION_FOLDS

N_DAYS_SALES = 1941
N_DAYS_CAL = 1969


def _write_raw(raw):
    rng = np.random.default_rng(0)
    d = np.arange(1, N_DAYS_CAL + 1)
    dates = pd.date_range("2011-01-29", periods=N_DAYS_CAL)
    cal = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "wm_yr_wk": 11101 + (d - 1) // 7,
            "weekday": dates.day_name(),
            "wday": (d - 1) % 7 + 1,
            "month": dates.month,
            "year": dates.year,
            "d": [f"d_{i}" for i in d],
            "event_name_1": np.where(d % 50 == 0, "SuperBowl", None),
            "event_type_1": np.where(d % 50 == 0, "Sporting", None),
            "event_name_2": None,
            "event_type_2": None,
            "snap_CA": (d % 30 < 10).astype(int),
            "snap_TX": 0,
            "snap_WI": 0,
        }
    )
    cal.to_csv(raw / "calendar.csv", index=False)

    rows, prices = [], []
    launches = {"FOODS_3_001": 1, "FOODS_3_002": 500, "FOODS_3_003": 1200, "FOODS_3_004": 1850}
    for store, state in [("CA_1", "CA"), ("CA_2", "CA"), ("TX_1", "TX")]:
        for item, launch in launches.items():
            sales = rng.poisson(3, N_DAYS_SALES)
            launch_week_start = ((launch - 1) // 7) * 7 + 1
            sales[: launch_week_start - 1] = 0
            rows.append([f"{item}_{store}_evaluation", item, "FOODS_3", "FOODS", store, state, *sales])
            for w in range((launch - 1) // 7, (N_DAYS_CAL - 1) // 7 + 1):
                prices.append([store, item, 11101 + w, 2.0 + 0.1 * (w % 5)])
    rows.append(["HOBBIES_1_001_CA_1_evaluation", "HOBBIES_1_001", "HOBBIES_1", "HOBBIES", "CA_1", "CA",
                 *rng.poisson(1, N_DAYS_SALES)])
    cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"] + [f"d_{i}" for i in range(1, N_DAYS_SALES + 1)]
    pd.DataFrame(rows, columns=cols).to_csv(raw / "sales_train_evaluation.csv", index=False)
    pd.DataFrame(prices, columns=["store_id", "item_id", "wm_yr_wk", "sell_price"]).to_csv(
        raw / "sell_prices.csv", index=False
    )


@pytest.fixture
def synthetic_project(local_tmp, monkeypatch):
    raw, processed, results = local_tmp / "raw", local_tmp / "processed", local_tmp / "results"
    for p in (raw, processed, results):
        p.mkdir()
    _write_raw(raw)
    monkeypatch.setattr(config, "SALES_CSV", raw / "sales_train_evaluation.csv")
    monkeypatch.setattr(config, "CALENDAR_CSV", raw / "calendar.csv")
    monkeypatch.setattr(config, "PRICES_CSV", raw / "sell_prices.csv")
    monkeypatch.setattr(config, "PROCESSED_DIR", processed)
    monkeypatch.setattr(config, "LONG_CACHE", processed / "long.parquet")
    monkeypatch.setattr(config, "SCALE_WIDE_CACHE", processed / "scale.parquet")
    monkeypatch.setattr(config, "PRICES_CACHE", processed / "prices.parquet")
    monkeypatch.setattr(config, "CALENDAR_CACHE", processed / "calendar.parquet")
    monkeypatch.setattr(config, "RESULTS_DIR", results)
    data.build_cache()
    return local_tmp


def test_cache_scope_and_hidden_holdout(synthetic_project):
    long = data.load_cache()
    assert set(long["store_id"].unique()) == {"CA_1", "CA_2"}
    assert long["d"].min() == config.WARMUP_START_DAY
    assert long["d"].max() == config.LAST_PRE_HOLDOUT_DAY
    assert data.load_cache(include_holdout=True)["d"].max() == config.HOLDOUT_END_DAY
    assert max(data.load_scale_wide().columns) == config.LAST_PRE_HOLDOUT_DAY
    assert not long.duplicated(["id", "d"]).any()


def test_validation_fold_end_to_end(synthetic_project):
    ctx = run.load_context()
    assert ctx.frame["d"].max() == config.LAST_PRE_HOLDOUT_DAY
    fs = run.setup_fold(ctx, VALIDATION_FOLDS[0])
    st = fs.elig["status"]
    assert set(st[st.index.str.startswith("FOODS_3_004")]) == {"not_yet_active"}
    assert set(st[st.index.str.startswith("FOODS_3_001")]) == {"eligible"}
    params = model.lgb_params(7, 0.1)
    booster, pred, n_train, _ = run.train_and_predict(ctx, fs, params, 10)
    assert not pred.isna().any().any()
    res = run.score(fs, {"lightgbm": pred, **fs.baseline_preds})
    assert res["n_series"].nunique() == 1 and res["n_series"].iloc[0] == len(fs.eligible_ids)
    plots.feature_importance(model.feature_importance(booster), synthetic_project / "fi.png")


def test_holdout_setup_uses_hidden_days_only_when_requested(synthetic_project):
    ctx = run.load_context(include_holdout=True)
    fs = run.setup_fold(ctx, HOLDOUT)
    assert list(fs.actual.columns) == list(HOLDOUT.days)
    assert not fs.actual.isna().any().any()
    hidden = run.setup_fold(run.load_context(), HOLDOUT)
    assert hidden.actual.isna().all().all()  # default loader never exposes d_1914-d_1941


def test_holdout_command_end_to_end_then_refuses_rerun(synthetic_project, monkeypatch):
    import json

    monkeypatch.setattr(run, "_run_tests", lambda: None)
    (config.RESULTS_DIR / "best_config.json").write_text(
        json.dumps({"config": "T", "num_leaves": 7, "learning_rate": 0.1, "rounds": 5})
    )
    run.main(["holdout", "--run-holdout"])
    for name in ["holdout_metrics.csv", "feature_importance.csv", "feature_importance.png",
                 "holdout_daily_forecast.png", "scaled_rmse_by_horizon.csv", "scaled_rmse_by_horizon.png",
                 "holdout_series_rmsse.csv", "fold_eligibility.csv"]:
        assert (config.RESULTS_DIR / name).exists(), name
    res = pd.read_csv(config.RESULTS_DIR / "holdout_metrics.csv")
    assert set(res["model"]) == {"lightgbm", "lag28", "trailing28_mean", "weekly_seasonal_naive"}
    assert res["n_series"].nunique() == 1
    assert len(pd.read_csv(config.RESULTS_DIR / "scaled_rmse_by_horizon.csv")) == 28
    before = (config.RESULTS_DIR / "holdout_metrics.csv").read_bytes()
    with pytest.raises(SystemExit):
        run.main(["holdout", "--run-holdout"])
    assert (config.RESULTS_DIR / "holdout_metrics.csv").read_bytes() == before
