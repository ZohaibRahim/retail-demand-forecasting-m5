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
