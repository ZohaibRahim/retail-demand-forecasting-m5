# M5 Retail Demand Forecasting — CA / FOODS_3

> Work in progress. Every metric below is **TBD** until the corresponding stage has actually been run.

## 1. Business Problem

Can a machine-learning model forecast the next 28 days of item-store demand better than simple retail forecasting baselines?

This is an **offline historical backtest**, not a production forecasting system.

## 2. Dataset

[M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy): `sales_train_evaluation.csv`, `calendar.csv`, `sell_prices.csv`. The files are downloaded manually into `data/raw/` and are never committed.

## 3. Scope

- California stores, department `FOODS_3`. Series counts: TBD (computed from the data).
- Modelling targets from `d_1000` onward. Days `d_900`–`d_999` are used only as a feature warm-up buffer.

## 4. Forecasting Setup

Direct 28-day forecasts are made once at a forecast origin, using a single global LightGBM model across all item-store series.

## 5. Information Boundary / Leakage Prevention

- Sales features for target day `t` use only sales on or before `t − 28`. Pre-launch zeros are treated as missing, not as demand.
- Price features use only weekly prices from weeks that **fully ended** on or before `t − 28`, so price information is 28–34 days old.
- Target-date calendar variables (weekday, month, events, SNAP) are treated as known in advance.
- Holdout-period sales (`d_1914`–`d_1941`) are hidden from the default data loader.

## 6. Baselines

- Lag-28 seasonal naive
- Trailing-28 mean
- Weekly seasonal naive (the final known week, repeated four times)

## 7. Feature Engineering

TBD

## 8. Model

LightGBM, `objective=tweedie`, `tweedie_variance_power=1.5`, a fixed seed, and fixed boosting rounds (no early stopping).

## 9. Walk-Forward Validation

| Fold | Origin | Validation days |
|---|---|---|
| 1 | d_1829 | d_1830–d_1857 |
| 2 | d_1857 | d_1858–d_1885 |
| 3 | d_1885 | d_1886–d_1913 |
| Final holdout | d_1913 | d_1914–d_1941 |

## 10. Metrics

- **MAE and RMSE** are pooled over all eligible series × 28 days.
- **RMSSE** is computed per series (scaled by that series' own training-only naive error), then summarized as the mean (primary selection metric), median, 95th percentile, and maximum.
- This is **not** the official hierarchical M5 WRMSSE.

## 11. Holdout Results

TBD — the final holdout has not been run.

## 12. Forecast Chart

TBD

## 13. Feature Importance

TBD

## 14. Forecast-Horizon Analysis

TBD

## 15. Limitations

- Only CA / FOODS_3.
- Offline historical backtest.
- No full official hierarchical M5 WRMSSE.
- Point forecasts rather than probabilistic intervals.
- Deliberately restricted price information.
- Limited hyperparameter tuning.

## 16. What I Would Do Next

1. **Add known-ahead planned-price and promotion features supplied by the retailer at the forecast origin.** Real retailers often know future planned prices and promotions. This project intentionally adopted a simple 28-day historical information boundary instead.
2. TBD

## 17. How to Run

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
pytest
python -m src.run inspect
python -m src.run preprocess
python -m src.run baselines
python -m src.run validate --rounds 100 --folds 1
python -m src.run tune
python -m src.run holdout --run-holdout   # only with explicit authorization
```

## 18. Interview Discussion / Key Design Decisions

TBD
