# FAQ

## Does the pipeline support both simulated and real data?
Yes. Choose the config:
- simulated: `configs/education_paper.yaml`
- UCI: `configs/education_uci.yaml`

## Why Pareto analysis instead of a single fairness constraint?
Because in practice you must negotiate multiple goals (performance, fairness, explainability, cost, coverage). The Pareto frontier makes trade-offs explicit before selecting with MAS.

## Why ban the sensitive attribute from model features?
To avoid “direct use” of the attribute while keeping it available for *auditing* and fairness evaluation.

## Where do I find the Overleaf-ready artifacts?
`outputs/<run_id>/overleaf_dropin/`

## Why is Fairlearn optional?
Fairlearn introduces additional dependency constraints (notably SciPy pins) and is intended as an optional *strong baseline*, not a required core component.

