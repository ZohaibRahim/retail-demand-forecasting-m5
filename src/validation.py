from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config


@dataclass(frozen=True)
class Fold:
    name: str
    origin: int

    @property
    def start(self) -> int:
        return self.origin + 1

    @property
    def end(self) -> int:
        return self.origin + config.HORIZON

    @property
    def days(self) -> np.ndarray:
        return np.arange(self.start, self.end + 1)


VALIDATION_FOLDS = (
    Fold("fold_1", 1829),
    Fold("fold_2", 1857),
    Fold("fold_3", 1885),
)
HOLDOUT = Fold("holdout", 1913)


def week_bounds(calendar: pd.DataFrame) -> pd.DataFrame:
    """First and last day number of every wm_yr_wk, in chronological order."""
    wb = calendar.groupby("wm_yr_wk")["d"].agg(start="min", end="max").sort_values("start")
    return wb


def first_price_week(prices: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """Per series: start/end day of the earliest week that has an observed sell price."""
    wb = week_bounds(calendar)
    observed = prices.dropna(subset=["sell_price"])
    first_wk = observed.merge(wb, left_on="wm_yr_wk", right_index=True)
    first_wk = first_wk.sort_values("start").groupby("id", observed=True).first()
    return first_wk[["start", "end"]].rename(
        columns={"start": "first_week_start", "end": "first_week_end"}
    )


def active_at_origin(active_info: pd.DataFrame, origin: int) -> pd.Series:
    """F2 rule: the first priced week counts only once it has fully ended on or before origin."""
    return active_info["first_week_end"].le(origin).fillna(False)


def eligibility(
    ids: pd.Index,
    active_info: pd.DataFrame,
    scales: pd.DataFrame,
    origin: int,
) -> pd.DataFrame:
    """Status per candidate series at a forecast origin, applying exclusions in a fixed order."""
    info = active_info.reindex(ids)
    sc = scales.reindex(ids)
    active = active_at_origin(info, origin)
    active_days = origin - info["first_week_start"] + 1
    enough_history = active & active_days.ge(config.MIN_ACTIVE_DAYS)
    valid_scale = sc["valid"].fillna(False).astype(bool)

    status = pd.Series("eligible", index=ids, dtype=object)
    status[~valid_scale] = "invalid_scale"
    status[~enough_history] = "insufficient_history"
    status[~active] = "not_yet_active"
    return pd.DataFrame(
        {
            "status": status,
            "active_days": active_days.where(active),
            "q": sc["q"],
            "n_diffs": sc["n_diffs"],
        }
    )


def eligibility_counts(elig: pd.DataFrame, fold_name: str, origin: int) -> dict:
    vc = elig["status"].value_counts()
    return {
        "fold": fold_name,
        "origin": f"d_{origin}",
        "total_candidates": len(elig),
        "excluded_not_yet_active": int(vc.get("not_yet_active", 0)),
        "excluded_insufficient_history": int(vc.get("insufficient_history", 0)),
        "excluded_invalid_scale": int(vc.get("invalid_scale", 0)),
        "eligible": int(vc.get("eligible", 0)),
    }


def training_mask(frame: pd.DataFrame, origin: int) -> pd.Series:
    """Post-launch, non-warm-up target rows on or before origin (Decision D, F2)."""
    first_end = frame["first_week_end"]
    first_start = frame["first_week_start"]
    return (
        frame["d"].ge(config.TARGET_START_DAY)
        & frame["d"].le(origin)
        & first_end.le(origin)
        & frame["d"].ge(first_start)
    )


def history_matrix(wide: pd.DataFrame, ids, origin: int) -> np.ndarray:
    """Sales for days 1..origin (wide columns are day numbers) for the given series."""
    cols = [c for c in wide.columns if c <= origin]
    return wide.loc[ids, cols].to_numpy(dtype=np.float64)


def actual_matrix(wide: pd.DataFrame, ids, fold: Fold) -> np.ndarray:
    return wide.loc[ids, list(fold.days)].to_numpy(dtype=np.float64)
