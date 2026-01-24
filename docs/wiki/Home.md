# Compass-ECL-MAS Wiki

Welcome! This wiki explains how to **run**, **reproduce**, and **interpret** Compass-ECL-MAS.

## Start here
- **Getting Started:** see [Getting-Started](Getting-Started.md)
- **Running experiments (CLI):** see [Running-Experiments](Running-Experiments.md)
- **Method overview (ECL → Pareto → MAS):** see [Method-ECL-MAS](Method-ECL-MAS.md)
- **Overleaf artifacts (figures/tables drop-in):** see [Overleaf-Integration](Overleaf-Integration.md)
- **Reproducibility & artifacts:** see [Reproducibility-and-Artifacts](Reproducibility-and-Artifacts.md)
- **Dataset provenance (simulated + UCI):** see [Data](Data.md)
- **Decision-maker report (Decision Brief PDF):** see [Decision-Brief](Decision-Brief.md)
- **Streamlit app:** see [Streamlit-App](Streamlit-App.md)
- **FAQ:** see [FAQ](FAQ.md)

## What you get after a run
A successful run creates a new folder `outputs/<run_id>/` containing:
- `metrics_candidates.csv` — evaluated candidates (all metrics)
- `pareto_front.csv` — Pareto-efficient candidates
- `mas_selected.csv` — the MAS-selected candidate
- `overleaf_dropin/` — paper-grade figures/tables ready for Overleaf
- `bundle.zip` — portable run bundle (used by the Streamlit “Review bundle.zip” feature)
- `decision_brief.pdf` — optional (requires `reportlab`)

