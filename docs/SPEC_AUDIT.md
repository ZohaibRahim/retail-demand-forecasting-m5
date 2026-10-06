# Specification Audit

Status: **APPROVED (2026-10-04).** All decisions A–L are final, and implementation approval has been given for all pre-holdout stages. The final holdout still requires separate explicit authorization.

Sources audited: `CLAUDE.md`, `docs/spec/METHODOLOGY.md`, `docs/spec/PROJECT_EXECUTION.md`.

Repository state at audit time (2026-10-04):

- Only `CLAUDE.md`, the two spec files, and an empty `LEARNING.md` exist.
- `data/raw/` is empty. `inspect` and `preprocess` cannot run until the three M5 CSVs are placed there.
- The folder is **not a git repository** yet. `.gitignore` will be created, but `git init` will not be run unless requested.
- Hard stop: October 9, which leaves 5 days.

Notation: `O` is the forecast origin, `t` is a target day, and `h = t − O ∈ {1..28}`.

---

## 1. Understanding of the core concepts

### 1.1 The 28-day information boundary

A forecast is made once at `O` for all of `O+1 … O+28`. For every target row `t` (training, validation, or holdout), a **sales-derived** or **price-derived** feature may only use raw data dated **≤ t − 28**. Because `t ≤ O + 28`, we get `t − 28 ≤ O`. As a result, the same feature code can be used for training and forecasting without any fold-specific branching, and no feature can see the horizon. The only exception is target-date calendar variables, which are treated as known in advance.

### 1.2 Active start vs RMSSE scaling start

| | Active start | RMSSE scaling start |
|---|---|---|
| Defined by | first week with a sell price (from `sell_prices.csv`) | first non-zero **sales** in training history at `O` |
| Used for | dropping pre-launch training rows; active-history length; evaluation eligibility | the starting point of the `q_i` denominator |
| Data allowed | price rows for weeks starting ≤ `O` | sales ≤ `O` only |

These are computed separately and are never merged.

### 1.3 84-day eligibility

`active_days(O) = O − active_start + 1 ≥ 84`. This equals 28 (shift) + 56 (longest rolling window). The check runs per fold, at that fold's origin.

Exclusions are counted in this order:

1. candidates;
2. not yet active;
3. insufficient active history;
4. invalid RMSSE scale;
5. eligible.

All four models are scored on the same eligible set.

### 1.4 Baselines (frozen before LightGBM tuning)

- **Lag-28:** `ŷ_t = y_{t−28}`. Since `t − 28 ≤ O`, this is always known. With 84-day eligibility, it is also always after activation.
- **Trailing-28 mean:** a constant equal to `mean(y_{O−27..O})`. Post-activation zeros are included. Eligibility guarantees at least 28 active days.
- **Weekly seasonal naive:** take the pattern `p = [y_{O−6}, …, y_O]` and set `ŷ_{O+k} = p[(k−1) mod 7]`. It is built directly from the origin pattern, **not** with `shift(7)` on forecast rows. A synthetic test checks that day 8 equals `y_{O−6}` and not the actual value for day 1.

### 1.5 Price cutoff

No price feature for `t` may use raw price information later than `t − 28`. The same rule applies to training, validation, and holdout rows. Forward-fill is chronological only and there is no back-fill. Features using price missingness or availability are not created. Because prices are weekly, a week's price is usable only if **the whole week ended on or before `t − 28`** (Decision A, §3).

### 1.6 Folds and holdout

| Name | Origin | Evaluated days |
|---|---|---|
| Fold 1 | d_1829 | d_1830–d_1857 |
| Fold 2 | d_1857 | d_1858–d_1885 |
| Fold 3 | d_1885 | d_1886–d_1913 |
| Holdout | d_1913 | d_1914–d_1941 |

Training uses an expanding window: training target rows are `≤ O`. There is no random splitting. The holdout runs only with `--run-holdout`, and it refuses to run if `results/holdout_metrics.csv` already exists.

---

## 2. Requirement → implementation map

### 2.1 Data scope and preprocessing

| Requirement | Location | Notes |
|---|---|---|
| Use the 3 M5 files from `data/raw/`; never download | `src/config.py` (paths), `src/data.py` | Raises a clear error if files are missing |
| Inspect raw files and report row count, day range, CA stores, CA+FOODS_3 count, unique items, calendar fields, price keys, memory | `data.inspect_raw()` → `run.py inspect` | Counts are computed from the data, never hardcoded |
| Filter `state_id=="CA"` & `dept_id=="FOODS_3"` **in wide format, before melt** | `data.load_filtered_sales_wide()` | |
| Keep `d_1000` onward as the **modelling-target scope** | `config.TARGET_START_DAY = 1000`, `config.WARMUP_START_DAY = 900` | The long cache holds d_900+. Rows d_900–d_999 are flagged `is_warmup=True`, are used only to build features, and are never training targets (Decision C) |
| Full-history sales for the RMSSE scale | `data.build_scale_sales_wide()` → `data/processed/m5_ca_foods3_scale_sales_wide.parquet` | Filtered wide matrix for d_1–d_1913, which is never melted. It deliberately stops at d_1913, so holdout actuals cannot reach any scale (Decision B) |
| Melt to long item-store-day | `data.melt_sales()` | |
| Join calendar and historical prices | `data.join_calendar()`, `data.join_prices()` | The raw weekly price is stored. Feature-time logic enforces the cutoff |
| Categorical dtypes, downcasting | `data.optimize_dtypes()` | |
| Report shape, unique series, min/max day, duplicate keys, memory, missing counts | `data.report_processed()` | |
| Parquet cache `data/processed/m5_ca_foods3_d1000.parquet` | `data.build_cache()`, `data.load_cache()` | `load_cache(max_day=1913)` is the default. Holdout actuals are masked unless the holdout path explicitly asks for them |

### 2.2 Active start and eligibility

| Requirement | Location |
|---|---|
| Active start = first day of first priced `wm_yr_wk` | `validation.compute_active_start(prices, calendar, origin)`. Which price weeks count as "known at `origin`" is pending Default F |
| Drop pre-activation training rows; keep post-activation zeros | `validation.training_rows(df, origin)` |
| ≥ 84 active days at origin | `validation.eligibility(..., min_active_days=config.MIN_ACTIVE_DAYS)` |
| Valid RMSSE scale required | `validation.eligibility` calls `metrics.rmsse_scale` |
| Per-fold exclusion counts | `validation.eligibility()` returns counts → `results/fold_eligibility.csv` |
| Same series set for all models | `validation.evaluate_fold()` scores all predictions on a single `eligible_ids` index |

### 2.3 Baselines

| Requirement | Location |
|---|---|
| Lag-28 | `baselines.lag28(sales_wide_or_long, origin)` |
| Trailing-28 mean | `baselines.trailing28_mean(...)` |
| Weekly seasonal naive (tiled pattern, not `shift(7)`) | `baselines.weekly_seasonal_naive(...)` |
| Run on 3 folds, save fold-level results and averages | `run.py baselines` → `results/baseline_metrics.csv` |

### 2.4 Features

| Requirement | Location |
|---|---|
| `lag_28/35/42/49` | `features.add_sales_features()`: `groupby(series).shift(k)` on a sorted, gap-free daily frame |
| `rolling_mean_7/28/56`, `rolling_std_28` computed on `shift(28)` | `features.add_sales_features()`, using `min_periods = window` |
| Price features using only weeks that **ended** ≤ `t−28` | `features.price_cutoff_week()`, `features.add_price_features()`. Exact rule in Decision A |
| Chronological forward-fill only | `features.add_price_features()`. Forward-fill happens at weekly level within each series, in week order. Weeks before the first price stay NaN |
| Pre-launch sales treated as missing for features | `features.mask_prelaunch_sales()` sets sales before active start to NaN before any shift or rolling (Decision C) |
| No `price_available` / stockout / future-missingness features | Enforced in `features.FEATURE_COLUMNS`, an explicit allow-list |
| Calendar: `wday, month, year, event_name_1/2, event_type_1/2, snap_CA` | `features.add_calendar_features()`. Missing events become the `"none"` category, and category sets are fixed across folds |
| Identity: `item_id`, `store_id` as categorical | `features.FEATURE_COLUMNS` / `model.CATEGORICAL` |

### 2.5 Metrics

| Requirement | Location |
|---|---|
| `q_i` from training history d_1…`O`, starting at the first non-zero sale, mean squared 1-step diff | `metrics.rmsse_scale(sales_wide, origin)`: vectorised NumPy over the wide matrix sliced to columns ≤ `O` (Decision B) |
| Valid scale: finite, `q_i > 0`, ≥ 28 observations from the first non-zero sale | `metrics.rmsse_scale` returns `(q, is_valid)` |
| MAE, RMSE (pooled over eligible series-days), mean / median / p95 / max per-series RMSSE, n series | `metrics.summarize()` |
| No MAPE | Not implemented anywhere |
| No clipping of extreme RMSSE values; investigate them instead | `metrics.summarize()` reports the raw values. `run.py` prints the top-5 extreme series per fold |
| `improvement_pct = (base − model)/base × 100`, with negative values reported as deterioration | `metrics.improvement_pct()` plus wording in `run.py` output |
| Scaled RMSE by forecast horizon, h = 1..28, for 4 models | `metrics.scaled_rmse_by_horizon()` → `results/scaled_rmse_by_horizon.csv/.png` |
| Never called "official WRMSSE" and never called "RMSSE by day" | README and plot labels |

### 2.6 LightGBM, validation, tuning

| Requirement | Location |
|---|---|
| Single global model, `objective=tweedie`, `tweedie_variance_power=1.5`, fixed seeds | `model.train_lgb()`, `config.LGB_BASE_PARAMS` (`seed`, `deterministic=True`, `bagging_seed`, `feature_fraction_seed`) |
| Fixed number of rounds and **no early stopping on validation folds** | `model.train_lgb(num_boost_round=...)` |
| Expanding-window walk-forward | `validation.get_folds()`, `validation.run_fold()` |
| Smoke test: 1 fold, 100 rounds, reporting training rows, eligible series, feature count, runtime, metrics, baseline comparison, top importances | `run.py validate --rounds 100 [--fold 1]` → `results/validation_metrics.csv` |
| Configs A/B/C × 3 folds = 9 fits | `config.TUNING_CONFIGS`, `run.py tune` → `results/tuning_results.csv` |
| Rank by average mean RMSSE and report fold standard deviation / range | `run.py tune` → `results/best_config.json` (validation only) |
| Pre-holdout checkpoint and STOP | `run.py tune` ends by printing the checkpoint summary. I stop and ask |

### 2.7 Holdout protection

| Requirement | Location |
|---|---|
| Requires `--run-holdout` | `run.py holdout`: exits non-zero without the flag. Guard logic lives in `run.check_holdout_allowed(flag, results_dir)` so it can be tested |
| Refuses if `results/holdout_metrics.csv` exists | Same guard |
| Pre-run checks: rerun tests, load frozen `best_config.json`, training ends at d_1913, boundary assertions | `run.py holdout` runs `pytest` through `subprocess` and asserts `max(train_day) == 1913` and `max(feature source day) ≤ t − 28` |
| All 3 baselines scored on the same eligible set | Uses the same `validation.evaluate_fold()` path |
| Save raw metrics before interpretation | `holdout_metrics.csv` is written first, then plots |

### 2.8 Outputs

| File | Produced by |
|---|---|
| `results/baseline_metrics.csv` | `baselines` |
| `results/fold_eligibility.csv` | `baselines` (3 folds). A holdout row is appended by `holdout` |
| `results/validation_metrics.csv` | `validate` |
| `results/tuning_results.csv`, `results/best_config.json` | `tune` |
| `results/holdout_metrics.csv`, `feature_importance.csv/.png`, `holdout_daily_forecast.png`, `scaled_rmse_by_horizon.csv/.png` | `holdout` (plots in `src/plots.py`) |

Feature importance uses `importance_type="gain"`, with split counts also saved in the CSV. The top validation importances for the pre-holdout checkpoint come from the selected config's fold-3 model. They are printed and saved to `results/validation_feature_importance.csv`. This is the only file added beyond the spec's list. It exists so the pre-holdout checkpoint can be reported without touching the holdout.

### 2.9 Tests (`pytest`, synthetic data only, no M5 files needed)

| # | Test | File |
|---|---|---|
| 1–4 | lag_28; lag_35/42/49; rolling after shift(28); perturbing sales after `t−28` leaves features unchanged | `tests/test_features.py` |
| 5–6 | Perturbing prices after the cutoff leaves price features unchanged; forward-fill never uses a future value | `tests/test_price_features.py` |
| 5a | **Weekly boundary test:** when `t−28` falls mid-week, changing that week's price leaves features unchanged and the previous week's price is used. When `t−28` is the last day of a week, that week's price is used | `tests/test_price_features.py` |
| 4a | Pre-launch zeros are not treated as demand: lag and rolling features that reach before active start are NaN. Warm-up rows (d_900–d_999) supply features but are never training targets | `tests/test_features.py` |
| 11a | The scale uses sales from before d_1000 when available, and never uses days after `O` | `tests/test_metrics.py` |
| 7–10 | Lag-28; trailing mean uses only known active history; weekly pattern tiling; weekly baseline ignores forecast-window actuals | `tests/test_baselines.py` |
| 11–14 | Scale uses training data only; scale starts at the first non-zero sale; zero/undefined scale is detected; hand-computed scaled-RMSE-by-horizon example | `tests/test_metrics.py` |
| 15–21 | Fold boundaries; holdout boundaries; active start ignores horizon data; 84-day rule; identical series sets across models; holdout blocked without the flag; no overwrite | `tests/test_splits.py` |

### 2.10 Repository, environment, docs

| Requirement | Location |
|---|---|
| Structure per PROJECT_EXECUTION §1 | As listed. No extra infrastructure |
| Python 3.11 where practical (3.11 is not installed on this machine, so the project-local `.venv` uses **3.12**); minimal `requirements.txt` | numpy, pandas, pyarrow, lightgbm, matplotlib, pytest. **scikit-learn and psutil are omitted** unless they are actually used (metrics are written by hand; memory is reported with `DataFrame.memory_usage`) |
| `.gitignore` | `.venv/ __pycache__/ .pytest_cache/ *.pyc data/raw/* data/processed/*`. `results/` is **not** ignored |
| README with TBD placeholders until holdout; 19 sections; required limitations; first "next step" = known-ahead planned prices/promotions | `README.md` (skeleton early, finalized after holdout) |
| LEARNING.md checkpoint after each major stage (7 required headings) | Appended after: preprocessing, features + tests, baselines, metrics/eligibility, LightGBM smoke test, tuning, holdout. Execution does not pause for reading |
| Interview prep: questions first, critique after the owner answers | After project completion only |

---

## 3. Approved decisions (A–D, approved 2026-10-04)

### Decision A: Conservative weekly-price cutoff

**Rule:** for target day `t`, a weekly price (`wm_yr_wk`) may be used only if **the entire week ended on or before `t − 28`**. The week containing `t − 28` is excluded unless `t − 28` is its last day. The rule is identical for training, validation, and holdout rows.

**Implementation (`features.py`):**

1. From `calendar.csv`, compute `week_end_day[W] = max(d)` for each `wm_yr_wk W`.
2. For each target day `t`, compute `cutoff_week(t) = max{ W : week_end_day[W] ≤ t − 28 }`. M5 weeks run Saturday to Friday, so this is the week of `t − 28` when `t − 28` is a Friday, and otherwise the previous week. The result is a day → week lookup table derived only from the calendar.
3. Prices are processed at **weekly** level per series, in week order:
   - `price_ff` = chronological forward-fill. No back-fill, so weeks before the first price stay NaN.
   - `price_prev_ff` = `price_ff` of the previous week.
   - `price_expmean` = expanding mean of `price_ff` over all weeks ≤ the current week (see Default L).
4. Target-row features join on `(series, cutoff_week(t))`:
   - `price_lag_28` = `price_ff` at the cutoff week
   - `price_wow_change` = `price_ff / price_prev_ff − 1`
   - `price_rel_hist` = `price_ff / price_expmean`

   All three depend only on weeks ≤ `cutoff_week(t)`, and all of those weeks ended ≤ `t − 28`.
5. Effective price age: 28–34 days, depending on weekday alignment.

**Tests:** #5 checks that perturbing every price in weeks after `cutoff_week(t)` leaves the features unchanged. #5a is the boundary test: with `t − 28` mid-week, changing that week's price leaves the features unchanged; with `t − 28` on a week's last day, that week's price is used. #6 checks that forward-fill never back-fills.

### Decision B: RMSSE scale from full available sales history

- For each fold, `q_i` is computed from sales **d_1 through that fold's origin `O`**, starting at the series' first non-zero sale in that range. The d_1000 cutoff limits the modelling targets, not the scaling history.
- Implementation: during `preprocess`, the CA/FOODS_3-filtered **wide** sales matrix for d_1–d_1913 is saved to `data/processed/m5_ca_foods3_scale_sales_wide.parquet` (excluded from git). It is never melted. `metrics.rmsse_scale()` slices the columns to ≤ `O` and computes the result in vectorised NumPy.
- The scale is fold-specific and uses training data only. The wide file stops at d_1913, so holdout actuals cannot reach any scale, including the holdout's own scale (origin d_1913).

### Decision C: Pre-launch as missing, with a warm-up buffer

- The long cache spans **d_900 onward**, giving 100 warm-up days ≥ the 84 needed. Rows d_900–d_999 carry `is_warmup=True`. They are used only as history for sales and price features and are **never** training or evaluation targets. Training targets start at d_1000.
- The cache keeps the spec's filename, `m5_ca_foods3_d1000.parquet`, because d_1000 remains the target scope. This is documented in the README.
- `features.mask_prelaunch_sales()` sets sales before each series' active start to NaN **before** any shift or rolling. Rolling windows use `min_periods = window`, so a window touching pre-launch days is NaN rather than a diluted mean. Series that launched recently get NaN features, which LightGBM handles natively.
- Leakage check: a series' active start comes from historical price-week existence. For any training row `t` ≤ `O`, masking depends only on whether a day before `t − 28` was pre-launch. Series that are not yet active at `O` contribute no training rows, so masking cannot carry information from the horizon.
- The 84-day rule still governs evaluation eligibility at each origin.

### Decision D: Training population

- The LightGBM training set for a fold is every **post-launch** CA/FOODS_3 row with target day in `[d_1000, O]`, whether or not the series is eligible for evaluation.
- Pre-launch rows, warm-up rows, and target rows after `O` are never included.
- LightGBM and all three baselines are scored on exactly the same fold-specific eligible-series set.

---

## 4. Approved decisions E–L (final, 2026-10-04)

### E: RMSSE scale validity (methodology-affecting) → **E2**

From the first non-zero sale through `O`, at least **29 observations** must exist, giving **≥ 28 one-step squared-difference terms**. The scale is also invalid if it is non-finite or ≤ 0. Invalid-scale series are reported as their own exclusion category. This is a project-specific stability rule, not an official M5 requirement.

### F: Active start (methodology-affecting) → **F2**

A weekly price row counts as historically known only once its entire `wm_yr_wk` has ended on or before the origin. `active_start(O)` = the first day of the earliest priced week whose `week_end_day ≤ O`. If no such week exists, the series is not yet active at `O`. This is the same "completed week" rule as Decision A.

For training-row filtering, a row with target day `t ≤ O` is post-launch if `t ≥ first day of the first priced week`, *provided that week ended ≤ `O`*. The few days of a first week still in progress at `O` are therefore excluded from training.

### G: MAE / RMSE (methodology-affecting) → **G1**

MAE and RMSE are **pooled** over every eligible series × every one of the 28 forecast days, in units sold. RMSSE is computed **per series** and then summarized across series (mean, median, p95, max). Mean per-series RMSSE remains the primary selection metric. The README states this distinction.

### H: Output files (implementation-only) → approved

`validation_metrics.csv` holds the smoke and validation runs with baseline comparisons. `tuning_results.csv` holds the 9 tuning fits.

### I: Scaled RMSE by horizon (implementation-only) → approved

Produced for the final holdout only.

### J: Hidden holdout actuals (implementation-only) → approved

`load_cache()` drops target days > d_1913 by default. Only the authorized holdout path can load them.

### K: No early stopping (implementation-only) → approved

The fixed round counts from the spec are used.

### L: Historical price reference (methodology-affecting) → **L1, clarified**

Computed per item-store series from the full `sell_prices.csv` history.

- `price_expmean(W)` = the expanding mean of **actually observed** (non-missing) weekly prices over weeks ≤ `W`.
- Forward-filled values are **not** counted in the mean.
- For target `t`, the value at `cutoff_week(t)` is used, so only weeks fully ended ≤ `t − 28` contribute.
- Forward-fill is used only for the last-known-price features (`price_lag_28`, `price_wow_change`) and never fills backward.

**Test 5b:** (1) changing future price weeks, after `cutoff_week(t)`, leaves `price_rel_hist` unchanged. (2) A series with a gap gets the same reference as computed from observed prices only, so forward-filled repetitions do not add weight.

No technically impossible requirements were found.

---

## 5. Planned execution order after approval

1. Scaffold: `.gitignore`, `requirements.txt`, `src/config.py`, README skeleton (TBD).
2. Tests and pure-logic modules (`features`, `baselines`, `metrics`, `validation`, holdout guard) built on synthetic data until `pytest` is green.
3. `inspect` and `preprocess`, once the raw CSVs are in `data/raw/`, then a LEARNING.md checkpoint.
4. `baselines` on 3 folds, then a checkpoint.
5. LightGBM smoke test (fold 1, 100 rounds), then a checkpoint.
6. `tune` (9 fits) → `best_config.json`, then a checkpoint.
7. **STOP** at the pre-holdout checkpoint and ask for holdout authorization.
