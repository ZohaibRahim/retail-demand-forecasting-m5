# Final Forecasting Methodology

## 1. Objective

Build a rigorous retail demand forecasting project using the M5 Forecasting - Accuracy dataset.

Business question:

Can a machine-learning model forecast the next 28 days of item-store demand better than simple retail forecasting baselines?

This is an offline historical forecasting experiment designed to demonstrate:

- structured-data processing;
- time-series forecasting;
- feature engineering;
- LightGBM model training;
- leakage prevention;
- temporal validation;
- baseline comparison;
- model evaluation;
- business interpretation.

It is not a production forecasting system.

---

# 2. Dataset

Use these M5 files:

- sales_train_evaluation.csv
- calendar.csv
- sell_prices.csv

Expected location:

data/raw/

The user downloads these manually.

Never download them automatically unless explicitly requested.

Never commit them to Git.

---

# 3. Dataset scope

Filter sales data while it is still in wide format.

Use:

- state_id == "CA"
- dept_id == "FOODS_3"
- all California stores
- modelling history approximately from d_1000 onward

Do not hardcode expected series counts.

Calculate and report actual counts from the supplied dataset.

Filter before melting the wide sales table.

---

# 4. Forecast horizon

Forecast horizon:

28 days.

A forecast is treated as being created once at a forecast origin for all following 28 target days.

This information boundary must be respected throughout feature engineering and evaluation.

---

# 5. Validation periods

Use expanding-window walk-forward validation.

## Fold 1

Forecast origin:

d_1829

Validation:

d_1830 through d_1857

## Fold 2

Forecast origin:

d_1857

Validation:

d_1858 through d_1885

## Fold 3

Forecast origin:

d_1885

Validation:

d_1886 through d_1913

## Final holdout

Forecast origin:

d_1913

Holdout:

d_1914 through d_1941

Never use random train/test splitting.

The final holdout may NOT be evaluated until the user explicitly authorizes the holdout run.

If a holdout result already exists, never silently overwrite it.

---

# 6. Two distinct series-start concepts

Do not combine series activation with RMSSE scaling.

They serve different purposes.

## 6.1 Active start

Define an item-store series' active start from its first historical week with an available sell price.

Use active start for:

- removing pre-launch training observations;
- determining active-history length;
- eligibility for evaluation.

For each fold separately, determine active status using only information available on or before that fold's forecast origin.

Never use information inside the validation/holdout horizon to decide whether a series was already active.

Post-activation zero-sales days remain genuine observed outcomes and must remain in the dataset.

Do not remove a zero simply because it may have represented a stockout.

## 6.2 Minimum active history

A series must have at least:

84 days

of active history at the relevant forecast origin.

This threshold supports the longest required sales feature:

shift(28) followed by a 56-day rolling window.

Evaluate all compared models on exactly the same eligible-series set.

Report separately for every fold:

- total candidate series;
- not-yet-active exclusions;
- insufficient-active-history exclusions;
- invalid-RMSSE-scale exclusions;
- final eligible series count.

---

# 7. RMSSE scaling start

RMSSE scaling is separate from active start.

For each item-store series:

1. use training history available at that fold's forecast origin;
2. locate the first non-zero sales observation;
3. begin the scale calculation from that portion of the training history;
4. calculate the mean squared one-step naive difference in sales.

Conceptually:

q_i = mean((y_t - y_(t-1))²)

over the valid training portion.

The denominator must NEVER include validation or holdout observations.

For this project, a scale is considered valid only if:

- it is finite;
- q_i > 0;
- there are at least 28 training observations from the first non-zero sales observation onward.

The 28-observation requirement is a project-specific stability rule, not a claim about the official M5 definition.

If a series has an invalid scale, exclude it from the common evaluation set for all compared models in that fold.

---

# 8. Baselines

Freeze all baseline definitions before tuning LightGBM.

Implement three baselines.

## 8.1 Lag-28 seasonal naive

For target day t:

prediction = sales at t - 28

Every target gets the observation exactly 28 days earlier.

---

## 8.2 Trailing-28 mean

At forecast origin O:

calculate the mean of the final 28 observed active sales days available through O.

Use that constant value as the prediction for all 28 target days.

Post-activation zero-sales days remain part of this mean.

Do not use observations inside the forecast horizon.

---

## 8.3 Weekly seasonal naive

At forecast origin O:

take the final seven observed sales values available before/at O.

Repeat that exact seven-day sequence four times.

Do NOT implement this as a normal shift(7) over forecast-window rows.

For example, validation day 8 must not use the actual value from validation day 1.

Add a synthetic test proving this.

---

# 9. Sales-derived model features

All sales-derived model features must respect a minimum 28-day information delay.

Required lag features:

- lag_28
- lag_35
- lag_42
- lag_49

Required rolling features:

- rolling_mean_7
- rolling_mean_28
- rolling_mean_56
- rolling_std_28

Rolling features must first shift sales by 28 days.

Conceptually:

shifted_sales = sales.shift(28)

rolling_mean_7 =
shifted_sales.rolling(7).mean()

and similarly for the other rolling features.

For a target day t, changing any sales observation after t - 28 must not change a sales-derived feature for target t.

Test this using synthetic data.

---

# 10. Price information policy

Use a strict ex-ante information policy.

For target day t:

NO price-derived feature may use raw price information later than:

t - 28 days

Apply the identical rule to:

- historical training rows;
- validation rows;
- final holdout rows.

Do NOT use actual price values from inside the forecast horizon.

Do NOT use forecast-period price missingness.

Do NOT create:

- price_available;
- future-availability;
- stockout;
- future-price-missing;
- or equivalent features.

This project intentionally does not exploit M5's supplied future-period prices.

This is a modelling/framing choice, not a claim that genuinely known planned future prices would constitute leakage.

A real retailer may know planned prices and promotions ahead of time.

Adding known-ahead planned prices/promotions must be the first extension listed in README.

## Historical missing prices

Historical price information may be chronologically forward-filled using values available from earlier dates only.

Never backward-fill using future observations.

## Suggested price features

Implement sensible temporally safe features such as:

- price_lag_28;
- lagged historical week-over-week price change;
- lagged historical relative price.

A historical relative-price reference must itself use only past information, such as an expanding historical mean.

Do not use an all-period product mean.

## Required test

Create synthetic prices with distinctive future values.

For target day t:

modify raw prices later than t - 28.

Verify that NONE of the target row's price-derived features change.

---

# 11. Known-ahead calendar variables

Target-date calendar variables may be used because they are explicitly treated as known before the forecast is produced.

Appropriate variables include:

- wday;
- month;
- year;
- event_name_1;
- event_type_1;
- event_name_2;
- event_type_2;
- snap_CA.

Handle missing event categories appropriately.

---

# 12. Identity features

Use a single global model across eligible item-store observations.

Appropriate categorical identifiers include:

- item_id;
- store_id.

Do not train thousands of separate LightGBM models.

---

# 13. LightGBM model

Use LightGBM.

Objective:

tweedie

Explicitly configure:

tweedie_variance_power = 1.5

Do not tune Tweedie variance power during this project.

Use fixed random seeds wherever applicable.

Do not claim Tweedie is inherently superior.

Treat it as a modelling choice that must prove itself against the fixed baselines.

---

# 14. Limited hyperparameter tuning

Start with exactly these configurations.

## A

num_leaves = 31

learning_rate = 0.05

n_estimators / rounds = 300

## B

num_leaves = 63

learning_rate = 0.05

n_estimators / rounds = 300

## C

num_leaves = 31

learning_rate = 0.03

n_estimators / rounds = 500

Run all three across all three validation folds.

This produces nine main tuning fits.

Rank configurations using:

average mean per-series RMSSE across the three validation folds.

Also report fold-to-fold variability.

Do not add another configuration unless the existing results expose a meaningful unresolved modelling question.

Do not use final holdout performance to select parameters.

---

# 15. Metrics

Never use MAPE.

Report:

- MAE;
- RMSE;
- mean per-series RMSSE;
- median per-series RMSSE;
- 95th-percentile per-series RMSSE;
- maximum per-series RMSSE;
- number of eligible evaluated series.

Primary model-selection metric:

mean per-series RMSSE.

The median, 95th percentile, and maximum are diagnostics.

Do not silently clip or remove extreme RMSSE series.

Investigate extreme values.

Do NOT call our metric full official M5 WRMSSE.

Full hierarchical WRMSSE is out of scope unless explicitly requested later.

---

# 16. Improvement calculation

When comparing LightGBM against a baseline using mean RMSSE:

improvement_pct =
(baseline_error - model_error)
/
baseline_error
*
100

If the result is negative, describe it as deterioration or worse performance.

Do not call reduction in error an equivalent percentage increase in accuracy.

---

# 17. Forecast-horizon diagnostic

The direct model deliberately uses sales and price information that is at least 28 days old relative to every target.

This means simple baselines may have fresher sales information for early forecast horizons.

Do not redesign the project around recursive forecasting this week.

Instead calculate:

Scaled RMSE by Forecast Horizon.

For eligible series i and horizon h:

scaled_rmse_h =
sqrt(
    mean across series(
        squared_error_i,h / q_i
    )
)

where q_i is the training-only RMSSE squared-error scale.

Calculate this for forecast horizons 1 through 28 for:

- lag-28;
- trailing-28 mean;
- weekly seasonal naive;
- LightGBM.

Do NOT call this "RMSSE by day."

Document that averaging the 28 scaled-RMSE-by-horizon values does not necessarily equal headline mean per-series RMSSE because the aggregation order differs.

---

# 18. Final holdout rules

The final holdout is:

d_1914 through d_1941.

Before execution:

- rerun tests;
- confirm best_config.json;
- verify final training target period ends at d_1913;
- verify the feature information boundaries;
- confirm no prior holdout result exists.

Holdout evaluation must require explicit user authorization.

Require an explicit CLI option such as:

--run-holdout

If results/holdout_metrics.csv already exists:

STOP rather than overwrite it silently.

After seeing holdout results:

do not retune the model against them.

If methodology is changed afterward, the existing holdout can no longer be described as untouched.