import numpy as np
import pandas as pd

from src import config


def rmsse_scale(history: np.ndarray, min_diffs: int = config.MIN_SCALE_DIFFS) -> pd.DataFrame:
    """Per-series RMSSE denominator q_i from training history only.

    `history` is (n_series, n_days) and must end at the forecast origin; the caller is
    responsible for never passing days after the origin. The scale starts at each series'
    first non-zero sale: q_i = mean((y_t - y_{t-1})^2) over that portion.
    """
    x = np.asarray(history, dtype=np.float64)
    n_series, n_days = x.shape
    nonzero = x != 0
    has_sale = nonzero.any(axis=1)
    first = np.where(has_sale, nonzero.argmax(axis=1), n_days)

    diffs_sq = np.diff(x, axis=1) ** 2
    # diff column j is (day j+1) - (day j); keep it only when day j >= first non-zero day
    keep = np.arange(n_days - 1)[None, :] >= first[:, None]
    n_diffs = keep.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(keep, diffs_sq, 0.0).sum(axis=1) / n_diffs
    valid = (n_diffs >= min_diffs) & np.isfinite(q) & (q > 0)
    return pd.DataFrame({"q": q, "n_diffs": n_diffs, "valid": valid})


def per_series_rmsse(actual: np.ndarray, pred: np.ndarray, q: np.ndarray) -> np.ndarray:
    mse = np.mean((actual - pred) ** 2, axis=1)
    return np.sqrt(mse / q)


def summarize(actual: np.ndarray, pred: np.ndarray, q: np.ndarray) -> dict:
    """MAE/RMSE pooled over all series-days; RMSSE per series, then summarized (Decision G)."""
    err = actual - pred
    rmsse = per_series_rmsse(actual, pred, q)
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mean_rmsse": float(np.mean(rmsse)),
        "median_rmsse": float(np.median(rmsse)),
        "p95_rmsse": float(np.percentile(rmsse, 95)),
        "max_rmsse": float(np.max(rmsse)),
        "n_series": int(actual.shape[0]),
    }


def scaled_rmse_by_horizon(actual: np.ndarray, pred: np.ndarray, q: np.ndarray) -> np.ndarray:
    """For each horizon h: sqrt(mean over series of squared_error_{i,h} / q_i)."""
    scaled_sq = (actual - pred) ** 2 / q[:, None]
    return np.sqrt(scaled_sq.mean(axis=0))


def improvement_pct(baseline_error: float, model_error: float) -> float:
    return (baseline_error - model_error) / baseline_error * 100.0


def describe_improvement(baseline_name: str, pct: float) -> str:
    if pct >= 0:
        return f"{pct:.1f}% lower mean RMSSE than {baseline_name}"
    return f"{-pct:.1f}% higher mean RMSSE than {baseline_name} (worse)"


def score_models(
    predictions: dict,
    actual: pd.DataFrame,
    scales: pd.Series,
    eligible_ids: pd.Index,
) -> pd.DataFrame:
    """Score every model on exactly the same eligible series and target days.

    `predictions` maps model name -> DataFrame (index=series id, columns=target days).
    """
    act = actual.reindex(eligible_ids)
    if act.isna().any().any():
        raise ValueError("Missing actuals for eligible series")
    q = scales.reindex(eligible_ids).to_numpy(dtype=np.float64)
    rows = []
    for name, pred in predictions.items():
        p = pred.reindex(index=eligible_ids, columns=act.columns)
        if p.isna().any().any():
            raise ValueError(f"Model {name!r} is missing predictions for eligible series")
        rows.append({"model": name, **summarize(act.to_numpy(float), p.to_numpy(float), q)})
    return pd.DataFrame(rows)
