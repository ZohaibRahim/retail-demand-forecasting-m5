"""Feature engineering under the 28-day information boundary.

For a target day t:
- sales features use only sales on days <= t-28 (and never pre-launch days);
- price features use only weekly prices from weeks that fully ended on or before t-28.
Both rules depend only on t, so the same code is valid for training, validation and holdout rows.
"""

import numpy as np
import pandas as pd

from src import config
from src.validation import week_bounds

LAGS = (28, 35, 42, 49)
ROLL_SHIFT = 28
ROLL_MEANS = (7, 28, 56)
ROLL_STD = (28,)

SALES_FEATURES = [f"lag_{k}" for k in LAGS] + [
    *(f"rolling_mean_{w}" for w in ROLL_MEANS),
    *(f"rolling_std_{w}" for w in ROLL_STD),
]
PRICE_FEATURES = ["price_lag_28", "price_wow_change", "price_rel_hist"]
CALENDAR_FEATURES = [
    "wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
]
ID_FEATURES = ["item_id", "store_id"]
FEATURES = SALES_FEATURES + PRICE_FEATURES + CALENDAR_FEATURES + ID_FEATURES
CATEGORICAL = ["event_name_1", "event_type_1", "event_name_2", "event_type_2", *ID_FEATURES]


def _series_matrix(frame: pd.DataFrame, col: str) -> np.ndarray:
    """Reshape a column to (n_series, n_days); frame must be a complete grid sorted by (id, d)."""
    n_days = frame["d"].nunique()
    n_series = len(frame) // n_days
    if n_series * n_days != len(frame):
        raise ValueError("Frame is not a complete series x day grid")
    d = frame["d"].to_numpy().reshape(n_series, n_days)
    if not ((d == d[0]).all() and (np.diff(d[0]) == 1).all()):
        raise ValueError("Frame must be sorted by (id, d) with contiguous days")
    return frame[col].to_numpy(dtype=np.float64).reshape(n_series, n_days)


def _shift(mat: np.ndarray, k: int) -> np.ndarray:
    out = np.full_like(mat, np.nan)
    out[:, k:] = mat[:, :-k]
    return out


def _rolling(mat: np.ndarray, window: int, how: str) -> np.ndarray:
    roll = pd.DataFrame(mat.T).rolling(window, min_periods=window)
    return getattr(roll, how)().to_numpy().T


def mask_prelaunch_sales(frame: pd.DataFrame) -> pd.Series:
    """Sales before a series' first priced week are not demand: set them to NaN (Decision C)."""
    return frame["sales"].astype("float64").where(frame["d"].ge(frame["first_week_start"]))


def add_sales_features(frame: pd.DataFrame) -> pd.DataFrame:
    tmp = frame[["d"]].copy()
    tmp["sales_masked"] = mask_prelaunch_sales(frame)
    sales = _series_matrix(tmp, "sales_masked")
    out = {}
    for k in LAGS:
        out[f"lag_{k}"] = _shift(sales, k)
    shifted = _shift(sales, ROLL_SHIFT)
    for w in ROLL_MEANS:
        out[f"rolling_mean_{w}"] = _rolling(shifted, w, "mean")
    for w in ROLL_STD:
        out[f"rolling_std_{w}"] = _rolling(shifted, w, "std")
    for name, mat in out.items():
        frame[name] = mat.reshape(-1).astype(np.float32)
    return frame


def cutoff_week_index(days: np.ndarray, calendar: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Index (into chronologically sorted weeks) of the latest week ending on or before day-28.

    Returns (week_index, weeks). week_index is -1 when no week has fully ended by the cutoff.
    """
    wb = week_bounds(calendar)
    week_ends = wb["end"].to_numpy()
    idx = np.searchsorted(week_ends, np.asarray(days) - config.HORIZON, side="right") - 1
    return idx, wb.index.to_numpy()


def weekly_price_tables(prices: pd.DataFrame, ids: pd.Index, weeks: np.ndarray) -> dict:
    """Per-series weekly price tables over all weeks in chronological order (Decisions A, L)."""
    observed = (
        prices.dropna(subset=["sell_price"])
        .pivot_table(index="id", columns="wm_yr_wk", values="sell_price", aggfunc="first", observed=True)
        .reindex(index=ids, columns=weeks)
        .to_numpy(dtype=np.float64)
    )
    ff = pd.DataFrame(observed.T).ffill().to_numpy().T
    prev_ff = _shift(ff, 1)
    is_obs = ~np.isnan(observed)
    n_obs = np.cumsum(is_obs, axis=1)
    sum_obs = np.cumsum(np.where(is_obs, observed, 0.0), axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        expmean = np.where(n_obs > 0, sum_obs / n_obs, np.nan)
        wow = ff / prev_ff - 1.0
        rel = ff / expmean
    return {"price_lag_28": ff, "price_wow_change": wow, "price_rel_hist": rel}


def add_price_features(frame: pd.DataFrame, prices: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    ids = pd.Index(frame["id"].cat.categories if hasattr(frame["id"], "cat") else frame["id"].unique())
    week_idx, weeks = cutoff_week_index(frame["d"].to_numpy(), calendar)
    tables = weekly_price_tables(prices, ids, weeks)
    series_idx = ids.get_indexer(frame["id"])
    ok = week_idx >= 0
    safe_week = np.where(ok, week_idx, 0)
    for name, table in tables.items():
        vals = table[series_idx, safe_week]
        frame[name] = np.where(ok, vals, np.nan).astype(np.float32)
    return frame


def build_features(frame: pd.DataFrame, prices: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["id", "d"], kind="stable").reset_index(drop=True)
    frame = add_sales_features(frame)
    frame = add_price_features(frame, prices, calendar)
    return frame
