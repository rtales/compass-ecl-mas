# Overleaf Integration (paper drop-in)

Each run produces an **Overleaf drop-in** folder:
```
outputs/<run_id>/overleaf_dropin/
  figures/
  tables/
```

## How to use
1) Copy (or sync) `figures/` and `tables/` into your Overleaf project.
2) Ensure your LaTeX sources (`experiments.tex`, `results.tex`, and `main.tex`) reference the **same run_id** artifacts.
3) Re-compile.

## What gets generated
- Paper-grade Pareto plots (e.g., F1 vs ΔEOpp with coverage color)
- Tables (`tab_*.tex`) for:
  - MAS-selected candidate
  - baseline comparisons (including Fairlearn reductions if enabled)

## Common pitfalls
- Mixing tables from one run_id and figures from another
- Overwriting drop-in artifacts without updating the paper inputs

Recommendation: always version your runs by keeping `outputs/<run_id>/bundle.zip` archived for the exact submission.

