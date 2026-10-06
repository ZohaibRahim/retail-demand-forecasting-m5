import lightgbm as lgb
import pandas as pd

from src import config
from src.features import CATEGORICAL, FEATURES


def lgb_params(num_leaves: int, learning_rate: float) -> dict:
    return {**config.LGB_BASE_PARAMS, "num_leaves": num_leaves, "learning_rate": learning_rate}


def train_lgb(train: pd.DataFrame, params: dict, rounds: int) -> lgb.Booster:
    """Fixed number of boosting rounds; no early stopping (Decision K)."""
    ds = lgb.Dataset(
        train[FEATURES],
        label=train["sales"].astype("float32"),
        categorical_feature=CATEGORICAL,
        free_raw_data=True,
    )
    return lgb.train(params, ds, num_boost_round=rounds)


def predict(booster: lgb.Booster, rows: pd.DataFrame) -> pd.Series:
    return pd.Series(booster.predict(rows[FEATURES]), index=rows.index)


def feature_importance(booster: lgb.Booster) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "feature": booster.feature_name(),
                "gain": booster.feature_importance("gain"),
                "split": booster.feature_importance("split"),
            }
        )
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
