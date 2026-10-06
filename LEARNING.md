# Learning Log

---

## Checkpoint 1 — Information boundary, baselines, metrics, eligibility (synthetic tests)

### What we built

These are the pure-logic building blocks of the project, all proven on small synthetic data before any real M5 data is touched:

- sales lag/rolling features;
- price features;
- the three baselines;
- RMSSE scaling and metrics;
- fold definitions;
- series eligibility;
- the holdout gate.

### Why it matters

In forecasting, the most common way to get impressive-but-fake results is *leakage*: letting the model see information it would not have had when the forecast was made. Every rule here encodes one question: "at the moment we forecast, what would we actually know?"

### Code I should understand

1. **`features.add_sales_features`** (`src/features.py`).
   - It reshapes sales into a `series × day` matrix.
   - `_shift(mat, k)` moves values `k` days later, so `lag_28` on day `t` is the sale from day `t−28`.
   - Rolling features are computed on the matrix *already shifted by 28*. So `rolling_mean_7` for day `t` averages days `t−34 … t−28`.
2. **`features.cutoff_week_index`**.
   - Prices are weekly. For target `t` it finds the latest week whose **last day** is ≤ `t−28`.
   - If `t−28` falls mid-week, that week is skipped and the previous one is used. That is why price information is 28–34 days old.
3. **`features.weekly_price_tables`**.
   - `price_ff` is the last known price, forward-filled only.
   - `price_expmean` is the running mean of *actually observed* prices, so filled-in gaps don't count as extra observations.
4. **`metrics.rmsse_scale`**.
   - `q_i` = mean of squared day-to-day changes, from the first non-zero sale up to the origin.
   - It needs at least 28 such changes and `q_i > 0`.
5. **`validation.eligibility` + `validation.training_mask`**.
   - A series is "active" only once its first priced week has *fully ended* by the origin.
   - It needs ≥ 84 active days to be evaluated.
   - Training uses every post-launch row from d_1000 to the origin.

### Leakage protection

- **Sales:** `_shift(sales, k)` with `k ≥ 28` in `add_sales_features`. Pre-launch zeros are masked to NaN by `mask_prelaunch_sales`.
- **Prices:** `cutoff_week_index` (`searchsorted` on week end days with `days − 28`).
- **Baselines:** they receive only `history_matrix(wide, ids, origin)`, which keeps columns ≤ origin. They never get the forecast window.
- **Scale:** `rmsse_scale` is given history ending at the origin. The wide scale file itself stops at d_1913.
- **Holdout:** `run.check_holdout_allowed` requires `--run-holdout` and refuses if `results/holdout_metrics.csv` exists. `data.load_cache()` drops days > d_1913 by default.

### Key assumptions

- A weekly price is only "known" after its week ends (conservative choice).
- A day before a series' first priced week is "not yet on sale", not "zero demand".
- Calendar events and SNAP days are known in advance.
- Zeros after launch are real outcomes. They may hide stockouts, but we do not remove them.

### Check yourself

1. For target day `t = 1900`, what is the latest sales day any feature may use? Why does that rule allow the same feature code to be used for training rows and forecast rows?
2. Why would counting forward-filled prices inside `price_expmean` distort `price_rel_hist`?

### 5-minute exercise

Sales for one series over its last 7 history days (days O−6 … O) are `[3, 0, 2, 5, 1, 0, 4]`. By hand, write the weekly-seasonal-naive forecasts for horizon days 1, 7, 8 and 15. Then explain why the forecast for day 8 is **not** the actual sales of forecast day 1. Check your answer against `tests/test_baselines.py::test_weekly_baseline_does_not_consume_forecast_window_actuals`.

---

## Checkpoint 2 — Real M5 data: inspection, preprocessing, baselines

### What we built

- **Validation:** we checked the three raw files against the expected M5 schema (21 checks, all passed).
- **Filtering:** we kept California × FOODS_3 *while the sales table is still wide*. That is 3,292 item-store series: 823 items × 4 stores (CA_1–CA_4).
- **Caches:**
  - The long `id × day` table runs from d_900 to d_1941: 3,430,264 rows, 79 MB.
  - The full-history wide matrix for RMSSE scaling runs from d_1 to d_1913.
  - A filtered weekly price table (733,895 rows) is stored separately.
- **Baselines:** we ran all three baselines over the three validation folds.

### Why it matters

M5 sales come as one row per item-store with 1,941 day columns ("wide"). Models need one row per item-store-day ("long"). Filtering before melting keeps memory small: 3,292 × 1,942 values instead of 30,490 × 1,942.

The baselines set the bar. A complex model that cannot beat "average of the last 28 days" is not worth deploying.

### Code I should understand

1. **`data.read_sales_wide_filtered`** reads the CSV in chunks and keeps only CA/FOODS_3 rows. Day columns are stored as `int16`.
2. **`data.build_cache`** writes the scale matrix only up to d_1913. That makes it *physically impossible* for the holdout days to enter an RMSSE denominator.
3. **`data.load_cache(include_holdout=False)`** is the default loader. It drops days after d_1913.
4. **`run.setup_fold`**:
   - computes scales from `history_matrix(..., origin)`;
   - computes eligibility;
   - computes the three baseline predictions;
   - builds the actuals table, all on the same `eligible_ids`.

### Leakage protection

- `history_matrix` slices columns `≤ origin` before any baseline or scale computation.
- `metrics.score_models` reindexes every model's predictions to the same `eligible_ids` and raises an error if any are missing. Models cannot be compared on different series.

### Key assumptions

- **Integrity check result:** no series has non-zero sales before its first priced week (0 units), and no series has price gaps after launch. So "first priced week = launch" fits this data well.
- **Eligible series:** 3,287 of 3,292 in every fold. The exclusions are 2–3 series launched within weeks of the origin. No series had an invalid scale.

### Real-data observations

- **Baseline ranking:** trailing-28 mean is clearly the strongest baseline (mean RMSSE ≈ 0.75), versus about 0.95–0.98 for the two naive baselines. A flat recent average beats "copy the past" because day-level retail sales are noisy, and averaging smooths that noise.
- **Extreme RMSSE values come from genuine behaviour:**
  - `FOODS_3_827_CA_4` has a tiny scale (`q ≈ 0.01`) because its history is almost all zeros, so selling a few units looks like a huge relative error.
  - `FOODS_3_382_CA_3` jumped from about 2 to about 31 units/day in fold 2, then collapsed in fold 3. Nothing was clipped.

### Check yourself

1. Why does a series with a near-constant sales history get an enormous RMSSE even when its absolute errors are tiny?
2. Why must the RMSSE scale be recomputed at each fold's origin instead of once for the whole dataset?

### 5-minute exercise

Open `results/baseline_metrics.csv`. For fold_2, compute by hand how much lower trailing-28's mean RMSSE is than lag-28's, using `(baseline − model) / baseline × 100`. Then explain in one sentence why lag-28 is *worse* at short horizons than its name suggests. Hint: is there anything "fresh" about a lag-28 value for horizon day 1?

---

## Checkpoint 3 — LightGBM smoke test, 3-fold validation, 9-fit tuning

### What we built

- **One global model:** a single LightGBM model across all 3,287 eligible series (not one model per series). It uses 21 features: 8 sales, 3 price, 8 calendar, and 2 identity features.
- **Training data:** each fold trains on every post-launch row from d_1000 to that fold's origin, about 2.6–2.8 M rows.
- **Runs:**
  - a 100-round smoke test on fold 1;
  - 100 rounds on all three folds;
  - configs A, B and C × 3 folds (9 fits).
- **Selection:** config **B** (63 leaves, learning rate 0.05, 300 rounds), chosen on average validation mean RMSSE only.

### Why it matters

Walk-forward validation imitates real use. Train on everything up to a date, forecast the next 28 days, move the date forward, and repeat. Three folds show whether a result is stable or a one-off.

### Code I should understand

1. **`validation.training_mask`** decides which rows are training rows:
   - `d ≥ 1000` (no warm-up rows);
   - `d ≤ origin`;
   - the first priced week has fully ended by the origin;
   - `d ≥` launch.
2. **`run.train_and_predict`**:
   - asserts that training never passes the origin;
   - trains for a fixed number of rounds (no early stopping);
   - predicts only the eligible series' 28 target rows.
3. **`model.lgb_params`** sets `objective="tweedie"` and `tweedie_variance_power=1.5`.
   - Tweedie treats demand as "often exactly zero, otherwise positive and skewed", which fits item-level grocery sales.
   - Predictions are on a log-link scale, so they are always positive.
4. **`run.cmd_tune`** ranks configs by the *average across folds* of mean per-series RMSSE and saves `best_config.json`. Nothing from d_1914+ is loaded.

### Leakage protection

- `train_and_predict` contains `assert train["d"].max() <= origin`.
- Features are built from the default loader, which already lacks d_1914+.
- Model selection reads only `tuning_results.csv`, which holds validation-fold results.

### Key assumptions and findings

- **Tuning barely matters.** The configs differ by ≤ 0.0026 in average mean RMSSE, about 10× smaller than the fold-to-fold standard deviation (0.028). B "wins", but the honest reading is that A, B and C are statistically indistinguishable.
- **LightGBM roughly ties the trailing-28 mean.**

  | Model | Fold 1 | Fold 2 | Fold 3 | Average |
  |---|---|---|---|---|
  | Config B | 0.7140 | 0.7579 | 0.7650 | 0.7456 |
  | Trailing-28 mean | 0.7127 | 0.7564 | 0.7712 | 0.7468 |

  That is about a 0.2% average improvement: slightly worse in folds 1–2, 0.8% better in fold 3. LightGBM is clearly better than lag-28 and weekly naive, by about 22–25%.
- **Where it wins and loses.** LightGBM beats trailing-28 on 52–60% of individual series and does better on medium- and high-volume series. It does worse on the lowest-volume quartile, which pulls its mean RMSSE up.
- **It under-forecasts total units:** −11% in fold 1, −5% in fold 2, −2% in fold 3. One likely reason is that every feature is at least 28 days old, so the model sees demand growth late.
- **Feature importance looks plausible.** `rolling_mean_56` dominates, followed by `item_id`, then the shorter rolling means, `lag_28`, and calendar features. Price features rank lower.
  - `item_id` has the most splits (823 levels). That is expected identity information, not leakage, but it is a capacity/overfitting risk worth mentioning.

### Check yourself

1. Why is it misleading to say "config B is the best" without mentioning that the configs differ by much less than the fold-to-fold variation?
2. LightGBM wins on most series but its mean RMSSE barely beats trailing-28. How can both be true?

### 5-minute exercise

From `results/tuning_results.csv`, compute config B's fold-to-fold range (max − min mean RMSSE). Compare it with the gap between configs A and B on their averages. Then write one sentence a hiring manager would accept as an honest summary of the tuning result.
