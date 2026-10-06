from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

SALES_CSV = RAW_DIR / "sales_train_evaluation.csv"
CALENDAR_CSV = RAW_DIR / "calendar.csv"
PRICES_CSV = RAW_DIR / "sell_prices.csv"

LONG_CACHE = PROCESSED_DIR / "m5_ca_foods3_d1000.parquet"
SCALE_WIDE_CACHE = PROCESSED_DIR / "m5_ca_foods3_scale_sales_wide.parquet"
PRICES_CACHE = PROCESSED_DIR / "m5_ca_foods3_prices.parquet"
CALENDAR_CACHE = PROCESSED_DIR / "calendar.parquet"

STATE_ID = "CA"
DEPT_ID = "FOODS_3"

HORIZON = 28
TARGET_START_DAY = 1000
WARMUP_START_DAY = 900
LAST_PRE_HOLDOUT_DAY = 1913
HOLDOUT_END_DAY = 1941

MIN_ACTIVE_DAYS = 84
MIN_SCALE_DIFFS = 28

SEED = 42

LGB_BASE_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "seed": SEED,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}

TUNING_CONFIGS = {
    "A": {"num_leaves": 31, "learning_rate": 0.05, "rounds": 300},
    "B": {"num_leaves": 63, "learning_rate": 0.05, "rounds": 300},
    "C": {"num_leaves": 31, "learning_rate": 0.03, "rounds": 500},
}
