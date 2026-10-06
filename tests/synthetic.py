import numpy as np
import pandas as pd

BASE_WEEK = 11101


def make_calendar(n_days: int) -> pd.DataFrame:
    """Day 1 is the first day of week 0; weeks are 7 days long (like M5's Sat-Fri weeks)."""
    d = np.arange(1, n_days + 1)
    return pd.DataFrame({"d": d, "wm_yr_wk": BASE_WEEK + (d - 1) // 7})


def week_id(i: int) -> int:
    return BASE_WEEK + i


def make_frame(sales: dict, first_week_start: dict) -> pd.DataFrame:
    """Long frame for several series sharing days 1..n."""
    parts = []
    for sid, values in sales.items():
        n = len(values)
        parts.append(
            pd.DataFrame(
                {
                    "id": sid,
                    "d": np.arange(1, n + 1),
                    "sales": np.asarray(values, dtype=float),
                    "first_week_start": float(first_week_start[sid]),
                }
            )
        )
    frame = pd.concat(parts, ignore_index=True)
    frame["id"] = frame["id"].astype("category")
    return frame


def make_prices(weekly: dict) -> pd.DataFrame:
    """weekly: series id -> list of prices per week index (None = no observed price)."""
    rows = []
    for sid, prices in weekly.items():
        for i, p in enumerate(prices):
            if p is not None:
                rows.append({"id": sid, "wm_yr_wk": week_id(i), "sell_price": float(p)})
    return pd.DataFrame(rows)
