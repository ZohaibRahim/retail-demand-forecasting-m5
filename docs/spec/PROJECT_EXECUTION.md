# Project Implementation and Execution Specification

# 1. Repository structure

Build approximately:

README.md
CLAUDE.md
LEARNING.md
requirements.txt
.gitignore

docs/
    spec/
        METHODOLOGY.md
        PROJECT_EXECUTION.md
    SPEC_AUDIT.md

data/
    raw/
        sales_train_evaluation.csv
        calendar.csv
        sell_prices.csv
    processed/

src/
    __init__.py
    config.py
    data.py
    features.py
    baselines.py
    metrics.py
    validation.py
    model.py
    plots.py
    run.py

tests/
    test_features.py
    test_price_features.py
    test_baselines.py
    test_metrics.py
    test_splits.py

results/
    baseline_metrics.csv
    validation_metrics.csv
    tuning_results.csv
    best_config.json
    fold_eligibility.csv
    holdout_metrics.csv
    feature_importance.csv
    feature_importance.png
    holdout_daily_forecast.png
    scaled_rmse_by_horizon.csv
    scaled_rmse_by_horizon.png

Do not create unnecessary infrastructure.

---

# 2. Python environment

Target Python 3.11 where practical.

requirements.txt should include only packages actually needed.

Likely dependencies:

- numpy
- pandas
- pyarrow
- lightgbm
- scikit-learn
- matplotlib
- pytest
- psutil if memory profiling actually uses it

Do not leave obviously unused dependencies in the final requirements file.

---

# 3. Git exclusions

.gitignore must exclude at minimum:

.venv/
__pycache__/
.pytest_cache/
*.pyc
data/raw/*
data/processed/*

Do NOT ignore final results and charts intended for the public repository.

Raw and processed M5 data must never be committed.

---

# 4. Preprocessing

Inspect the three raw files before modelling.

Report:

- raw sales row count;
- available day range;
- CA store IDs;
- actual CA + FOODS_3 series count;
- unique item count;
- relevant calendar fields;
- price-table keys;
- memory usage.

Filter the sales dataframe to:

state_id == CA
dept_id == FOODS_3

before melting.

Keep modelling history from approximately d_1000 onward.

Reshape to long item-store-day format.

Join:

calendar information

and historical price information.

Optimize dtypes where safe.

Use categorical dtypes for repeated identifiers where appropriate.

Report:

- processed shape;
- unique series;
- min/max day;
- duplicate item/store/day keys;
- memory use;
- relevant missing-value counts.

---

# 5. Parquet cache

Create:

data/processed/m5_ca_foods3_d1000.parquet

Use this processed cache for repeated modelling experiments so the complete CSV filtering/melting/join process is not repeated unnecessarily.

The cache is local and excluded from Git.

---

# 6. CLI

Provide a simple CLI.

Exact internal architecture may vary, but support equivalent commands to:

python -m src.run inspect

python -m src.run preprocess

python -m src.run baselines

python -m src.run validate --rounds 100

python -m src.run tune

python -m src.run holdout --run-holdout

Commands should provide useful progress/result summaries.

The holdout command must enforce the hard gate and overwrite protection.

---

# 7. Required synthetic/unit tests

Before trusting full-dataset results, implement synthetic tests covering at least:

## Sales features

1. lag_28 correctness;
2. lag_35/42/49 correctness;
3. rolling statistics are computed after shift(28);
4. changing sales later than t-28 cannot alter target t features.

## Price features

5. no price feature for target t can use information later than t-28;
6. historical forward-filling never uses a future value.

## Baselines

7. lag-28 baseline correctness;
8. trailing-28 mean uses only known active history;
9. weekly seasonal baseline repeats the final known seven-day pattern;
10. weekly baseline does not consume forecast-window actuals.

## Metrics

11. RMSSE scale uses training history only;
12. RMSSE scale starts from the first non-zero sales portion;
13. zero/undefined scale is detected;
14. scaled-RMSE-by-horizon calculation matches a hand-computable example.

## Splits and eligibility

15. validation boundaries match the specification exactly;
16. final holdout boundaries match exactly;
17. active-start eligibility never uses horizon information;
18. 84-day minimum history is enforced;
19. every compared model receives the same evaluation-series set.

## Holdout protection

20. holdout cannot accidentally execute without explicit authorization;
21. an existing holdout result is not silently overwritten.

Run pytest and fix failures before major evaluation runs.

---

# 8. Baseline stage

Before LightGBM tuning:

run all three baselines over all three validation folds.

Store fold-level metrics and averages.

Freeze their definitions.

Do not modify a baseline later merely because LightGBM struggles to beat it.

---

# 9. Initial LightGBM smoke test

Before expensive tuning, run a small model such as:

objective = tweedie
tweedie_variance_power = 1.5
learning_rate = 0.05
num_leaves = 31
rounds = 100

Run one validation fold first.

Report:

- training rows;
- eligible validation series;
- feature count;
- runtime;
- MAE;
- RMSE;
- mean RMSSE;
- baseline comparisons;
- top feature importances.

The goal is pipeline verification, not optimization.

---

# 10. Tuning stage

After the smoke test succeeds:

execute the three predefined configurations across the three validation folds.

Store every experiment in:

results/tuning_results.csv

Include at minimum:

- configuration;
- fold;
- mean RMSSE;
- median RMSSE;
- p95 RMSSE;
- max RMSSE;
- MAE;
- RMSE;
- eligible series;
- runtime if practical.

Create:

results/best_config.json

using validation results only.

---

# 11. Pre-holdout checkpoint

After tuning:

STOP.

Do not evaluate d_1914-d_1941.

Present:

- test status;
- baseline validation table;
- tuning table;
- selected configuration;
- fold variability;
- eligibility counts;
- top validation feature importances if available;
- any unresolved methodological issue.

Ask for explicit authorization before holdout execution.

This is the only mandatory modelling hard stop after initial specification approval.

---

# 12. Final holdout stage

Only after explicit authorization:

1. rerun tests;
2. load frozen best_config.json;
3. train using permitted data through d_1913;
4. evaluate d_1914-d_1941 exactly according to METHODOLOGY.md;
5. evaluate all three fixed baselines on the exact same eligible-series set;
6. save raw metrics before interpretation.

Generate:

results/holdout_metrics.csv
results/feature_importance.csv
results/feature_importance.png
results/holdout_daily_forecast.png
results/scaled_rmse_by_horizon.csv
results/scaled_rmse_by_horizon.png

Do not retune after seeing holdout results.

---

# 13. Main forecast chart

Create a 28-day holdout chart using total daily units aggregated across eligible series.

At minimum show:

- actual total units;
- LightGBM predicted total units.

A baseline may also be shown if it remains readable.

---

# 14. Feature importance

Create a top-15 or top-20 LightGBM feature-importance chart.

Also save raw feature importance.

Inspect the highest-ranked features for:

- business plausibility;
- unexpected identifiers;
- possible leakage;
- suspicious dominance.

If a suspicious feature appears, investigate before publishing results.

---

# 15. Model failure analysis

After holdout evaluation identify:

1. one eligible series where LightGBM clearly beats the baselines;
2. one where performance is similar;
3. one where LightGBM performs substantially worse.

Explain what occurred.

This analysis is for understanding rather than cherry-picking headline results.

Do not alter the overall result because of these examples.

---

# 16. README

Before final results exist, use TBD rather than fabricated metrics.

Final README should contain:

1. Project title
2. Business Problem
3. Dataset
4. Scope
5. Forecasting Setup
6. Information Boundary / Leakage Prevention
7. Baselines
8. Feature Engineering
9. Model
10. Walk-Forward Validation
11. Metrics
12. Holdout Results
13. Forecast Chart
14. Feature Importance
15. Forecast-Horizon Analysis
16. Limitations
17. What I Would Do Next
18. How to Run
19. Interview Discussion / Key Design Decisions

Limitations must explicitly include:

- only CA / FOODS_3;
- offline historical backtest;
- no full official hierarchical M5 WRMSSE;
- point forecasts rather than probabilistic intervals;
- deliberately restricted price information;
- limited hyperparameter tuning.

The FIRST item in "What I Would Do Next" must be:

Add known-ahead planned-price and promotion features supplied by the retailer at the forecast origin.

Explain that real retailers may know future planned prices and promotions, but this project intentionally adopted a simple 28-day historical information boundary.

---

# 17. LEARNING.md

The repository owner must learn the project while Claude Code builds it.

After every major stage APPEND a concise checkpoint to LEARNING.md.

Do not stop execution just to wait for the user to read it.

Each checkpoint should contain:

## What we built

Plain-English explanation.

## Why it matters

Why this step exists in a retail forecasting system.

## Code I should understand

Identify 3-5 important functions or code sections.

Explain important logic, especially temporal logic.

## Leakage protection

Identify exactly which code prevents future information from entering the model.

## Key assumptions

State important modelling assumptions.

## Check yourself

Give two short comprehension questions.

## 5-minute exercise

Give one code-reading, manual-calculation, or prediction exercise that takes less than five minutes.

Do not spend learning-document space explaining:

- argparse boilerplate;
- ordinary file I/O;
- plotting syntax;
- logging boilerplate;
- trivial Pandas syntax;

unless technically important.

Prioritize understanding:

1. M5 data layout;
2. item-store series;
3. forecast origin;
4. lag features;
5. rolling features;
6. information leakage;
7. all three baselines;
8. RMSSE;
9. walk-forward validation;
10. LightGBM;
11. Tweedie;
12. feature importance;
13. model failure modes.

---

# 18. Interview preparation

Do NOT initially write polished interview answers for the repository owner.

After the finished project, provide questions.

The owner should answer first.

Then critique their response for:

- technical accuracy;
- missing reasoning;
- unsupported claims;
- clarity;
- likely follow-up questions.

Only after their attempt should an improved answer be offered.

---

# 19. Time constraint and scope

Hard project stop:

October 9.

Priority order:

1. correct splits;
2. leakage prevention;
3. reproducible preprocessing;
4. baselines;
5. tests;
6. working LightGBM;
7. walk-forward validation;
8. limited tuning;
9. final holdout;
10. analysis;
11. README.

Do not add before project completion:

- LSTM;
- neural forecasting;
- Prophet;
- transformers;
- TimesFM;
- another boosting library;
- recursive forecasting;
- Optuna-scale searches;
- MLflow;
- databases;
- APIs;
- React;
- dashboards;
- Docker;
- cloud deployment.

Correct evaluation is more valuable than extra complexity.

---

# 20. Specification audit — mandatory first action

Before writing modelling implementation code:

1. Read CLAUDE.md.
2. Read METHODOLOGY.md.
3. Read this file.
4. Create docs/SPEC_AUDIT.md.

SPEC_AUDIT.md must contain a checklist mapping every major requirement to its planned implementation location.

At minimum cover:

- data scope;
- Parquet cache;
- active start;
- eligibility;
- all baselines;
- sales cutoff;
- price cutoff;
- calendar features;
- RMSSE;
- horizon metric;
- validation folds;
- LightGBM;
- tuning;
- tests;
- outputs;
- holdout protection;
- README;
- learning workflow.

Also report any contradiction, ambiguity, or technically impossible requirement.

After creating SPEC_AUDIT.md:

STOP.

Wait for user approval before implementation begins.

After that approval, continue automatically through all pre-holdout stages.