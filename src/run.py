"""CLI: inspect | preprocess | baselines | validate | tune | holdout."""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import baselines, config, data, metrics, model
from src.features import FEATURES, build_features
from src.validation import (
    HOLDOUT,
    VALIDATION_FOLDS,
    Fold,
    eligibility,
    eligibility_counts,
    first_price_week,
    history_matrix,
    training_mask,
)

HOLDOUT_METRICS = "holdout_metrics.csv"


class HoldoutBlocked(RuntimeError):
    pass


def check_holdout_allowed(run_holdout: bool, results_dir=None) -> None:
    """Hard gate: explicit flag required, and an existing holdout result is never overwritten."""
    results_dir = config.RESULTS_DIR if results_dir is None else results_dir
    if not run_holdout:
        raise HoldoutBlocked(
            "Final holdout evaluation requires explicit authorization: pass --run-holdout."
        )
    if (results_dir / HOLDOUT_METRICS).exists():
        raise HoldoutBlocked(
            f"{results_dir / HOLDOUT_METRICS} already exists; refusing to overwrite a holdout result."
        )


# ---------------------------------------------------------------- shared pipeline


@dataclass
class Context:
    frame: pd.DataFrame
    wide: pd.DataFrame
    active_info: pd.DataFrame


@dataclass
class FoldSetup:
    fold: Fold
    elig: pd.DataFrame
    eligible_ids: pd.Index
    scales: pd.Series
    actual: pd.DataFrame
    baseline_preds: dict


def load_context(include_holdout: bool = False) -> Context:
    t0 = time.time()
    cal = data.load_calendar()
    prices = data.load_prices()
    active_info = first_price_week(prices, cal)
    long = data.load_cache(include_holdout=include_holdout)
    long = data.attach_active_info(long, active_info)
    frame = build_features(long, prices, cal)
    wide = data.load_scale_wide()
    print(f"[context] {len(frame):,} rows, {frame['id'].nunique():,} series, "
          f"d_{frame['d'].min()}..d_{frame['d'].max()}, {len(FEATURES)} features "
          f"({time.time() - t0:.1f}s)")
    return Context(frame, wide, active_info)


def setup_fold(ctx: Context, fold: Fold) -> FoldSetup:
    ids = ctx.wide.index
    hist = history_matrix(ctx.wide, ids, fold.origin)
    scales = metrics.rmsse_scale(hist).set_index(ids)
    elig = eligibility(ids, ctx.active_info, scales, fold.origin)
    eligible_ids = elig.index[elig["status"] == "eligible"]

    rows = ctx.frame[ctx.frame["d"].between(fold.start, fold.end) & ctx.frame["id"].isin(eligible_ids)]
    actual = rows.pivot(index="id", columns="d", values="sales").reindex(index=eligible_ids, columns=fold.days)

    elig_hist = history_matrix(ctx.wide, eligible_ids, fold.origin)
    baseline_preds = {
        name: pd.DataFrame(fn(elig_hist), index=eligible_ids, columns=fold.days)
        for name, fn in baselines.BASELINES.items()
    }
    return FoldSetup(fold, elig, eligible_ids, scales["q"], actual, baseline_preds)


def train_and_predict(ctx: Context, fs: FoldSetup, params: dict, rounds: int):
    train = ctx.frame[training_mask(ctx.frame, fs.fold.origin)]
    assert train["d"].max() <= fs.fold.origin, "training rows leak past the origin"
    assert train["d"].min() >= config.TARGET_START_DAY
    t0 = time.time()
    booster = model.train_lgb(train, params, rounds)
    runtime = time.time() - t0
    rows = ctx.frame[ctx.frame["d"].between(fs.fold.start, fs.fold.end) & ctx.frame["id"].isin(fs.eligible_ids)]
    pred = pd.Series(booster.predict(rows[FEATURES]), index=rows.index)
    pred_df = (
        pd.DataFrame({"id": rows["id"], "d": rows["d"], "pred": pred})
        .pivot(index="id", columns="d", values="pred")
        .reindex(index=fs.eligible_ids, columns=fs.fold.days)
    )
    return booster, pred_df, len(train), runtime


def score(fs: FoldSetup, preds: dict) -> pd.DataFrame:
    res = metrics.score_models(preds, fs.actual, fs.scales, fs.eligible_ids)
    res.insert(0, "fold", fs.fold.name)
    return res


def top_rmsse_series(fs: FoldSetup, pred: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    a = fs.actual.to_numpy(float)
    p = pred.reindex(index=fs.eligible_ids, columns=fs.fold.days).to_numpy(float)
    q = fs.scales.reindex(fs.eligible_ids).to_numpy(float)
    r = metrics.per_series_rmsse(a, p, q)
    out = pd.DataFrame(
        {
            "rmsse": r,
            "q": q,
            "actual_mean": a.mean(axis=1),
            "pred_mean": p.mean(axis=1),
            "active_days": fs.elig.loc[fs.eligible_ids, "active_days"].to_numpy(),
        },
        index=fs.eligible_ids,
    )
    return out.sort_values("rmsse", ascending=False).head(k)


def print_table(df: pd.DataFrame) -> None:
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(df.to_string(index=False))


def _comparison_rows(res: pd.DataFrame, lgb_name: str) -> list:
    out = []
    m = res.set_index("model")["mean_rmsse"]
    for b in baselines.BASELINES:
        pct = metrics.improvement_pct(m[b], m[lgb_name])
        out.append(f"  vs {b}: {metrics.describe_improvement(b, pct)}")
    return out


# ---------------------------------------------------------------- commands


def cmd_inspect(_args):
    data.inspect_raw()


def cmd_preprocess(_args):
    data.build_cache()


def cmd_baselines(_args):
    ctx = load_context()
    rows, elig_rows = [], []
    for fold in VALIDATION_FOLDS:
        fs = setup_fold(ctx, fold)
        elig_rows.append(eligibility_counts(fs.elig, fold.name, fold.origin))
        rows.append(score(fs, fs.baseline_preds))
        for name, pred in fs.baseline_preds.items():
            print(f"[{fold.name}] {name} highest-RMSSE series:")
            print_table(top_rmsse_series(fs, pred, 3).reset_index())
    res = pd.concat(rows, ignore_index=True)
    avg = res.groupby("model", sort=False).mean(numeric_only=True).reset_index()
    avg.insert(0, "fold", "average")
    res = pd.concat([res, avg], ignore_index=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    res.to_csv(config.RESULTS_DIR / "baseline_metrics.csv", index=False)
    elig_df = pd.DataFrame(elig_rows)
    elig_df.to_csv(config.RESULTS_DIR / "fold_eligibility.csv", index=False)
    print("\n== Fold eligibility ==")
    print_table(elig_df)
    print("\n== Baseline validation metrics ==")
    print_table(res)


def cmd_validate(args):
    cfg = config.TUNING_CONFIGS[args.config]
    params = model.lgb_params(cfg["num_leaves"], cfg["learning_rate"])
    folds = [VALIDATION_FOLDS[i - 1] for i in args.folds]
    ctx = load_context()
    run_label = f"validate_{args.config}_r{args.rounds}"
    lgb_name = "lightgbm"
    all_rows = []
    for fold in folds:
        fs = setup_fold(ctx, fold)
        booster, pred, n_train, runtime = train_and_predict(ctx, fs, params, args.rounds)
        res = score(fs, {lgb_name: pred, **fs.baseline_preds})
        res.insert(0, "run", run_label)
        res["train_rows"] = n_train
        res["n_features"] = len(FEATURES)
        res["runtime_s"] = round(runtime, 1)
        all_rows.append(res)
        print(f"\n== {fold.name} (origin d_{fold.origin}) | {run_label} ==")
        print(f"training rows: {n_train:,} | eligible series: {len(fs.eligible_ids):,} | "
              f"features: {len(FEATURES)} | train time: {runtime:.1f}s")
        print_table(res.drop(columns=["run", "train_rows", "n_features", "runtime_s"]))
        print("\n".join(_comparison_rows(res, lgb_name)))
        print("top feature importances (gain):")
        print_table(model.feature_importance(booster).head(10))
        print("highest-RMSSE LightGBM series:")
        print_table(top_rmsse_series(fs, pred).reset_index())

    new = pd.concat(all_rows, ignore_index=True)
    path = config.RESULTS_DIR / "validation_metrics.csv"
    config.RESULTS_DIR.mkdir(exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        keep = ~(old["run"].eq(run_label) & old["fold"].isin(new["fold"]))
        new = pd.concat([old[keep], new], ignore_index=True)
    new.to_csv(path, index=False)
    print(f"\nsaved {path}")


def cmd_tune(_args):
    ctx = load_context()
    rows, importances = [], {}
    for fold in VALIDATION_FOLDS:
        fs = setup_fold(ctx, fold)
        base = score(fs, fs.baseline_preds).set_index("model")["mean_rmsse"]
        for name, cfg in config.TUNING_CONFIGS.items():
            params = model.lgb_params(cfg["num_leaves"], cfg["learning_rate"])
            booster, pred, n_train, runtime = train_and_predict(ctx, fs, params, cfg["rounds"])
            res = score(fs, {"lightgbm": pred}).iloc[0].to_dict()
            row = {"config": name, **{k: cfg[k] for k in ("num_leaves", "learning_rate", "rounds")}, **res}
            row.pop("model")
            row.update(train_rows=n_train, runtime_s=round(runtime, 1))
            for b, err in base.items():
                row[f"improvement_vs_{b}_pct"] = metrics.improvement_pct(err, row["mean_rmsse"])
            rows.append(row)
            importances[(name, fold.name)] = model.feature_importance(booster)
            print(f"[{fold.name}] config {name}: mean RMSSE {row['mean_rmsse']:.4f} "
                  f"(MAE {row['mae']:.4f}, RMSE {row['rmse']:.4f}, {runtime:.0f}s)")

    tuning = pd.DataFrame(rows)
    cols = ["config", "fold", "num_leaves", "learning_rate", "rounds", "mean_rmsse", "median_rmsse",
            "p95_rmsse", "max_rmsse", "mae", "rmse", "n_series", "train_rows", "runtime_s"]
    tuning = tuning[cols + [c for c in tuning.columns if c not in cols]]
    tuning.to_csv(config.RESULTS_DIR / "tuning_results.csv", index=False)

    summary = (
        tuning.groupby("config")["mean_rmsse"]
        .agg(avg_mean_rmsse="mean", std_mean_rmsse="std", min_fold="min", max_fold="max")
        .sort_values("avg_mean_rmsse")
    )
    best = summary.index[0]
    best_cfg = {
        "config": best,
        **config.TUNING_CONFIGS[best],
        "objective": "tweedie",
        "tweedie_variance_power": 1.5,
        "seed": config.SEED,
        "selection_metric": "average mean per-series RMSSE across validation folds 1-3",
        "avg_mean_rmsse": float(summary.loc[best, "avg_mean_rmsse"]),
        "std_mean_rmsse": float(summary.loc[best, "std_mean_rmsse"]),
        "fold_mean_rmsse": tuning[tuning["config"] == best].set_index("fold")["mean_rmsse"].to_dict(),
        "selected_using": "validation folds only (no holdout data)",
    }
    with open(config.RESULTS_DIR / "best_config.json", "w") as f:
        json.dump(best_cfg, f, indent=2)
    importances[(best, "fold_3")].to_csv(config.RESULTS_DIR / "validation_feature_importance.csv", index=False)

    print("\n== Tuning results ==")
    print_table(tuning[cols])
    print("\n== Config summary (selection: lowest average mean RMSSE) ==")
    print_table(summary.reset_index())
    print(f"\nselected config: {best} -> results/best_config.json")
    print("\nvalidation feature importance (selected config, fold_3, gain):")
    print_table(importances[(best, "fold_3")].head(15))


def _run_tests() -> None:
    print("[holdout] re-running test suite ...")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=config.ROOT)
    if r.returncode != 0:
        raise SystemExit("Tests failed; holdout aborted.")


def cmd_holdout(args):
    from src import plots

    try:
        check_holdout_allowed(args.run_holdout)
    except HoldoutBlocked as e:
        raise SystemExit(f"BLOCKED: {e}")
    _run_tests()
    with open(config.RESULTS_DIR / "best_config.json") as f:
        best = json.load(f)
    print(f"[holdout] frozen config: {best['config']} {best}")
    params = model.lgb_params(best["num_leaves"], best["learning_rate"])

    ctx = load_context(include_holdout=True)
    fs = setup_fold(ctx, HOLDOUT)
    booster, pred, n_train, runtime = train_and_predict(ctx, fs, params, best["rounds"])
    train_max = ctx.frame.loc[training_mask(ctx.frame, HOLDOUT.origin), "d"].max()
    assert train_max == config.LAST_PRE_HOLDOUT_DAY, f"final training ends at d_{train_max}"

    preds = {"lightgbm": pred, **fs.baseline_preds}
    res = score(fs, preds)
    m = res.set_index("model")["mean_rmsse"]
    for b in baselines.BASELINES:
        res.loc[res["model"] == "lightgbm", f"improvement_vs_{b}_pct"] = metrics.improvement_pct(m[b], m["lightgbm"])
    res["config"] = best["config"]
    res["train_rows"] = n_train
    check_holdout_allowed(True)
    with open(config.RESULTS_DIR / HOLDOUT_METRICS, "x", newline="") as f:
        res.to_csv(f, index=False)
    print(f"[holdout] raw metrics saved to results/{HOLDOUT_METRICS}")

    elig_path = config.RESULTS_DIR / "fold_eligibility.csv"
    holdout_elig = pd.DataFrame([eligibility_counts(fs.elig, "holdout", HOLDOUT.origin)])
    if elig_path.exists():
        prev = pd.read_csv(elig_path)
        holdout_elig = pd.concat([prev[prev["fold"] != "holdout"], holdout_elig], ignore_index=True)
    holdout_elig.to_csv(elig_path, index=False)

    imp = model.feature_importance(booster)
    imp.to_csv(config.RESULTS_DIR / "feature_importance.csv", index=False)
    plots.feature_importance(imp, config.RESULTS_DIR / "feature_importance.png")

    q = fs.scales.reindex(fs.eligible_ids).to_numpy(float)
    a = fs.actual.to_numpy(float)
    horizon = pd.DataFrame({"horizon": np.arange(1, config.HORIZON + 1)})
    for name, p in preds.items():
        horizon[name] = metrics.scaled_rmse_by_horizon(a, p.to_numpy(float), q)
    horizon.to_csv(config.RESULTS_DIR / "scaled_rmse_by_horizon.csv", index=False)
    plots.scaled_rmse_by_horizon(horizon, config.RESULTS_DIR / "scaled_rmse_by_horizon.png")

    cal = data.load_calendar().set_index("d")["date"]
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(cal.reindex(HOLDOUT.days).to_numpy()),
            "actual": a.sum(axis=0),
            "lightgbm": pred.to_numpy(float).sum(axis=0),
            "trailing28_mean": fs.baseline_preds["trailing28_mean"].to_numpy(float).sum(axis=0),
        }
    )
    plots.daily_forecast(daily, config.RESULTS_DIR / "holdout_daily_forecast.png")

    per_series = pd.DataFrame(
        {name: metrics.per_series_rmsse(a, p.to_numpy(float), q) for name, p in preds.items()},
        index=fs.eligible_ids,
    )
    per_series.insert(0, "q", q)
    per_series.insert(1, "actual_mean", a.mean(axis=1))
    per_series.insert(2, "lightgbm_pred_mean", pred.to_numpy(float).mean(axis=1))
    per_series.rename_axis("id").reset_index().to_csv(config.RESULTS_DIR / "holdout_series_rmsse.csv", index=False)
    long_preds = pd.concat(
        {name: p.rename_axis(index="id", columns="d").stack().rename("pred") for name, p in preds.items()},
        names=["model"],
    ).reset_index()
    long_preds.to_parquet(config.PROCESSED_DIR / "holdout_predictions.parquet", index=False)

    total_a, total_p = a.sum(), pred.to_numpy(float).sum()
    print(f"\n[holdout] training rows: {n_train:,}, train time {runtime:.1f}s, last training day d_{train_max}")
    print(f"[holdout] eligibility: {eligibility_counts(fs.elig, 'holdout', HOLDOUT.origin)}")
    print(f"[holdout] total units actual {total_a:,.0f} | LightGBM {total_p:,.0f} "
          f"({(total_p - total_a) / total_a * 100:+.1f}%)")
    print("\n== Scaled RMSE by forecast horizon ==")
    print_table(horizon)
    print("\n== Top 15 feature importances (gain) ==")
    print_table(imp.head(15))

    print("\n== Holdout metrics ==")
    print_table(res)
    print("\n".join(_comparison_rows(res, "lightgbm")))


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m src.run")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inspect").set_defaults(func=cmd_inspect)
    sub.add_parser("preprocess").set_defaults(func=cmd_preprocess)
    sub.add_parser("baselines").set_defaults(func=cmd_baselines)
    v = sub.add_parser("validate")
    v.add_argument("--rounds", type=int, default=100)
    v.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3], choices=[1, 2, 3])
    v.add_argument("--config", default="A", choices=sorted(config.TUNING_CONFIGS))
    v.set_defaults(func=cmd_validate)
    sub.add_parser("tune").set_defaults(func=cmd_tune)
    h = sub.add_parser("holdout")
    h.add_argument("--run-holdout", action="store_true", help="explicit authorization for final holdout")
    h.set_defaults(func=cmd_holdout)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
