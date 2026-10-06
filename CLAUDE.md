# M5 Retail Demand Forecasting — Claude Code Instructions

This repository is a portfolio project for a retail AI/ML internship.

The authoritative project specifications are:

@docs/spec/METHODOLOGY.md
@docs/spec/PROJECT_EXECUTION.md

Read both imported specifications before making architectural, modelling, validation, metric, feature, or data-processing decisions.

## Authority

The imported specification files contain the FINAL project decisions.

Do not try to reconstruct requirements from previous conversations.

Do not silently change the methodology.

If implementation reality makes a specification impossible or technically incorrect:

1. stop that affected part of the work;
2. explain the issue;
3. identify the affected requirement;
4. recommend the smallest correction;
5. wait for approval only if the change would materially affect methodology, evaluation, leakage, metrics, model selection, or claimed results.

For ordinary implementation details, choose a sensible implementation and continue.

## Critical rules

Never:

- use MAPE;
- use random train/test splitting;
- use target sales from inside a 28-day forecast horizon as features;
- use price information newer than the defined 28-day information boundary;
- tune against the final holdout;
- evaluate the final holdout without explicit user authorization;
- silently overwrite existing holdout results;
- fabricate metrics;
- put placeholder metrics into README as if they were real;
- commit raw or processed M5 data;
- call this a production deployment;
- call our metrics official M5 WRMSSE unless full official WRMSSE is actually implemented;
- add unnecessary APIs, dashboards, Docker, cloud deployment, deep learning, recursive forecasting, or additional models before the required project is finished.

## Work style

Correct temporal evaluation is more important than model sophistication.

The project must remain achievable by October 9.

Continue automatically through all approved pre-holdout stages.

The final holdout is the only mandatory modelling hard gate.

Append concise learning checkpoints to LEARNING.md as specified in PROJECT_EXECUTION.md.

Before the first implementation begins, perform the specification audit described in PROJECT_EXECUTION.md and stop for approval.