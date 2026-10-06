"""Raw M5 inspection, filtering, reshaping and local Parquet caches."""

import numpy as np
import pandas as pd

from src import config

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CAL_COLS = [
    "d",
    "date",
    "wm_yr_wk",
    "wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap_CA",
]
EVENT_COLS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]


def _require_raw():
    missing = [p.name for p in (config.SALES_CSV, config.CALENDAR_CSV, config.PRICES_CSV) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing raw M5 files in {config.RAW_DIR}: {missing}. Download them manually."
        )


def _mb(df: pd.DataFrame) -> float:
    return df.memory_usage(deep=True).sum() / 1e6


def day_num(col: str) -> int:
    return int(col.split("_")[1])


def read_sales_wide_filtered() -> tuple[pd.DataFrame, int]:
    """Read sales in chunks, keeping only CA / FOODS_3 rows while still in wide format."""
    header = pd.read_csv(config.SALES_CSV, nrows=0).columns
    day_cols = [c for c in header if c.startswith("d_")]
    dtypes = {c: "int16" for c in day_cols}
    kept, n_raw = [], 0
    for chunk in pd.read_csv(config.SALES_CSV, dtype=dtypes, chunksize=5000):
        n_raw += len(chunk)
        kept.append(chunk[(chunk["state_id"] == config.STATE_ID) & (chunk["dept_id"] == config.DEPT_ID)])
    wide = pd.concat(kept, ignore_index=True)
    return wide, n_raw


def read_calendar() -> pd.DataFrame:
    cal = pd.read_csv(config.CALENDAR_CSV)
    cal["d"] = cal["d"].map(day_num).astype("int16")
    cal = cal[CAL_COLS].copy()
    for c in EVENT_COLS:
        cats = ["none", *sorted(cal[c].dropna().unique())]
        cal[c] = pd.Categorical(cal[c].fillna("none"), categories=cats)
    for c in ["wday", "month", "snap_CA"]:
        cal[c] = cal[c].astype("int8")
    cal["year"] = cal["year"].astype("int16")
    cal["wm_yr_wk"] = cal["wm_yr_wk"].astype("int32")
    return cal


def read_prices_filtered(item_ids, store_ids) -> pd.DataFrame:
    prices = pd.read_csv(config.PRICES_CSV, dtype={"wm_yr_wk": "int32", "sell_price": "float32"})
    prices = prices[prices["store_id"].isin(store_ids) & prices["item_id"].isin(item_ids)].copy()
    prices["id"] = prices["item_id"] + "_" + prices["store_id"] + "_evaluation"
    return prices.reset_index(drop=True)


def inspect_raw() -> None:
    _require_raw()
    wide, n_raw = read_sales_wide_filtered()
    day_cols = [c for c in wide.columns if c.startswith("d_")]
    ca_all = pd.read_csv(config.SALES_CSV, usecols=["state_id", "store_id", "dept_id"])
    ca_stores = sorted(ca_all.loc[ca_all["state_id"] == config.STATE_ID, "store_id"].unique())
    cal = pd.read_csv(config.CALENDAR_CSV)
    prices = pd.read_csv(config.PRICES_CSV)

    print("== Raw sales (sales_train_evaluation.csv) ==")
    print(f"raw rows: {n_raw:,}")
    print(f"day columns: {day_cols[0]} .. {day_cols[-1]} ({len(day_cols)} days)")
    print(f"CA stores: {ca_stores}")
    print(f"departments: {sorted(ca_all['dept_id'].unique())}")
    print(f"CA + FOODS_3 series: {len(wide):,}")
    print(f"CA + FOODS_3 unique items: {wide['item_id'].nunique():,}")
    print(f"series per store: {wide['store_id'].value_counts().sort_index().to_dict()}")
    print(f"filtered wide memory: {_mb(wide):.1f} MB")
    print("\n== calendar.csv ==")
    print(f"rows: {len(cal):,}, days d_1..{cal['d'].iloc[-1]}, dates {cal['date'].min()}..{cal['date'].max()}")
    print(f"columns: {list(cal.columns)}")
    print(f"weeks (wm_yr_wk): {cal['wm_yr_wk'].nunique()}, first weekday: {cal['weekday'].iloc[0]}")
    print(f"non-null event_name_1: {cal['event_name_1'].notna().sum()}, event_name_2: {cal['event_name_2'].notna().sum()}")
    print(f"calendar memory: {_mb(cal):.1f} MB")
    print("\n== sell_prices.csv ==")
    print(f"rows: {len(prices):,}, columns: {list(prices.columns)}")
    print(f"key (store_id, item_id, wm_yr_wk) duplicates: {prices.duplicated(['store_id', 'item_id', 'wm_yr_wk']).sum()}")
    print(f"weeks covered: {prices['wm_yr_wk'].min()}..{prices['wm_yr_wk'].max()}")
    print(f"prices memory: {_mb(prices):.1f} MB")


def build_cache() -> None:
    _require_raw()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    wide, _ = read_sales_wide_filtered()
    day_cols = [c for c in wide.columns if c.startswith("d_")]
    print(f"filtered CA/FOODS_3 wide: {wide.shape}, days {day_cols[0]}..{day_cols[-1]}")

    # Decision B: full-history wide matrix for RMSSE scaling, deliberately ending at d_1913.
    scale_cols = [c for c in day_cols if day_num(c) <= config.LAST_PRE_HOLDOUT_DAY]
    scale_wide = wide[["id", *scale_cols]].copy()
    scale_wide.to_parquet(config.SCALE_WIDE_CACHE, index=False)
    print(f"scale wide cache: {scale_wide.shape} -> {config.SCALE_WIDE_CACHE.name}")

    # Decision C: long frame from the d_900 warm-up buffer onward.
    long_cols = [c for c in day_cols if day_num(c) >= config.WARMUP_START_DAY]
    long = wide[["id", "item_id", "store_id", *long_cols]].melt(
        id_vars=["id", "item_id", "store_id"], var_name="d", value_name="sales"
    )
    long["d"] = long["d"].map(day_num).astype("int16")
    long["sales"] = long["sales"].astype("int16")
    for c in ["id", "item_id", "store_id"]:
        long[c] = long[c].astype("category")

    cal = read_calendar()
    cal.to_parquet(config.CALENDAR_CACHE, index=False)
    long = long.merge(cal.drop(columns=["date"]), on="d", how="left")
    long["is_warmup"] = long["d"] < config.TARGET_START_DAY

    prices = read_prices_filtered(wide["item_id"].unique(), wide["store_id"].unique())
    prices = prices[["id", "item_id", "store_id", "wm_yr_wk", "sell_price"]]
    prices.to_parquet(config.PRICES_CACHE, index=False)

    long = long.sort_values(["id", "d"]).reset_index(drop=True)
    long.to_parquet(config.LONG_CACHE, index=False)
    report_processed(long, prices)


def report_processed(long: pd.DataFrame, prices: pd.DataFrame) -> None:
    print("\n== Processed long cache ==")
    print(f"shape: {long.shape}")
    print(f"unique series: {long['id'].nunique():,}")
    print(f"day range: d_{long['d'].min()}..d_{long['d'].max()} "
          f"(warm-up d_{config.WARMUP_START_DAY}-d_{config.TARGET_START_DAY - 1}; "
          f"days > d_{config.LAST_PRE_HOLDOUT_DAY} hidden by default loader)")
    print(f"duplicate (id, d) keys: {long.duplicated(['id', 'd']).sum()}")
    print(f"memory: {_mb(long):.1f} MB")
    na = long.isna().sum()
    print(f"missing values: {na[na > 0].to_dict() or 'none'}")
    print(f"event_name_1 'none' share: {(long['event_name_1'] == 'none').mean():.3f}")
    print("\n== Filtered price table ==")
    print(f"rows: {len(prices):,}, series with any price: {prices['id'].nunique():,}, "
          f"weeks {prices['wm_yr_wk'].min()}..{prices['wm_yr_wk'].max()}")
    print(f"missing sell_price: {prices['sell_price'].isna().sum()}")
    print(f"memory: {_mb(prices):.1f} MB")


def load_cache(include_holdout: bool = False) -> pd.DataFrame:
    """Long frame. Target days after d_1913 are dropped unless the holdout path asks for them."""
    long = pd.read_parquet(config.LONG_CACHE)
    if not include_holdout:
        long = long[long["d"] <= config.LAST_PRE_HOLDOUT_DAY].reset_index(drop=True)
    return long


def load_scale_wide() -> pd.DataFrame:
    wide = pd.read_parquet(config.SCALE_WIDE_CACHE).set_index("id")
    wide.columns = [day_num(c) for c in wide.columns]
    return wide


def load_prices() -> pd.DataFrame:
    return pd.read_parquet(config.PRICES_CACHE)


def load_calendar() -> pd.DataFrame:
    return pd.read_parquet(config.CALENDAR_CACHE)


def attach_active_info(long: pd.DataFrame, active_info: pd.DataFrame) -> pd.DataFrame:
    info = active_info.reindex(long["id"].cat.categories)
    codes = long["id"].cat.codes.to_numpy()
    long["first_week_start"] = info["first_week_start"].to_numpy(dtype=np.float64)[codes]
    long["first_week_end"] = info["first_week_end"].to_numpy(dtype=np.float64)[codes]
    return long
