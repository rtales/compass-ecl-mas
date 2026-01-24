# Reproducibility & Artifacts

## What “reproducible” means here
Given the same config and environment, a user should be able to produce:
- candidate evaluation CSVs
- Pareto front and MAS selection
- paper-grade figures/tables for Overleaf
- a Decision Brief PDF (optional)

## Practical checklist
- **Environment**
  - Record Python version
  - Keep a dependency lock (`pip freeze > requirements-lock.txt`)
  - Prefer Docker for long-term artifact stability (see `docs/DOCKER.md`)
- **Configuration**
  - Commit / archive the exact YAML config used
  - Keep seed list for multi-seed aggregation
- **Artifacts**
  - Archive `outputs/<run_id>/bundle.zip`
  - Keep Overleaf drop-in artifacts together with the paper revision
- **Claims**
  - Report mean ± std across seeds (avoid single-run cherry-picking)

## What is inside bundle.zip
`bundle.zip` contains all key outputs needed for later inspection:
- metrics CSVs
- figures/tables
- metadata (run_id, config snapshot where available)
- the optional `decision_brief.pdf`

The Streamlit app can ingest `bundle.zip` to support “review-only” exploration.

