# Compass-ECL-MAS
**Constraint-based ethical governance for educational decision support in the public sector.**

This repository accompanies the paper **“Compass-ECL-MAS: A Constraint-Based Ethical Governance Layer for Educational Decision Support in the Public Sector”**. It operationalizes the **Compass Framework** (six ethical “laws”) via:

- **ECL (Ethical Constraint Layer):** enforceable constraints over *fairness* (Equal Opportunity gap), *explainability* proxies (e.g., entropy / feature budget), *cost/latency* proxies, and *abstention/coverage* (human review).
- **Pareto analysis:** explicit multi-objective trade-offs (performance × fairness × coverage × explainability × cost).
- **MAS (Moral Alignment Score):** auditable selection from the Pareto front using thresholds and tie-break rules.
- **Baselines:** unconstrained models, **Hardt-style** Equal Opportunity post-processing, and an optional **Fairlearn reductions** baseline (constrained optimization).
- **Artifacts-first reproducibility:** one CLI command produces CSVs, paper-grade figures/tables (Overleaf drop-in), and an optional **Decision Brief PDF** for decision-makers.

---

## Why this project exists

Public-sector education decisions (screening, risk triage, resource allocation) are **high-stakes**: optimizing accuracy alone can silently shift harms. Compass-ECL-MAS treats *ethical governance as a systems layer*:
1) make constraints explicit and measurable,  
2) compute the feasible trade-off set, and  
3) select and justify a policy choice with auditable rules.

---

## Quickstart (CLI)

### 1) Create an environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip
```

### 2) Install
```bash
# Core library
python -m pip install -e "."

# UI (Streamlit)
python -m pip install -e ".[ui]"

# Real dataset fetch/processing helpers (UCI)
python -m pip install -e ".[data]"

# Optional: strong constrained-optimization baseline (Fairlearn reductions)
python -m pip install -e ".[fairlearn]"

# Optional: generate the Decision Brief PDF
python -m pip install reportlab
```

> **macOS + Python 3.13 note:** `fairlearn==0.13.0` requires `scipy<1.16`.
> If SciPy gets upgraded, fix it with:
> ```bash
> python -m pip install "scipy<1.16.0" --force-reinstall
> ```

### 3) Run end-to-end
```bash
# Quick smoke test
python -m compass_ecl_mas.cli.run_all --config configs/education_quick.yaml

# Paper configuration (simulated, heavier)
python -m compass_ecl_mas.cli.run_all --config configs/education_paper.yaml

# Real-data configuration (UCI Student Performance)
python -m compass_ecl_mas.cli.run_all --config configs/education_uci.yaml
```

### Outputs
Each run writes to:
- `outputs/<run_id>/` — CSVs, plots, decision_brief.pdf (optional)
- `outputs/<run_id>/overleaf_dropin/` — paper-grade **figures + tables** ready to copy into Overleaf
- `outputs/<run_id>/bundle.zip` — portable bundle used by the Streamlit “Review bundle.zip” feature

Key files you’ll typically inspect:
- `metrics_candidates.csv`, `pareto_front.csv`, `mas_selected.csv`
- `baselines_no_ecl.csv`, `fairness_postproc.csv`, `fairlearn_reductions.csv` (if enabled)
- `decision_brief.pdf` (if reportlab installed)

---

## Streamlit app (interactive governance + artifact export)

```bash
streamlit run app/streamlit_app.py
```

The app supports two workflows:
- **Run**: configure constraints and run the pipeline from the UI
- **Review**: upload `bundle.zip` from a CLI run and explore:
  - F1 × ΔEOpp × Coverage trade-offs
  - which constraints bind
  - the MAS-selected policy candidate
  - Overleaf-ready figures/tables and exports

---

## Data

### Simulated education data (default)
The pipeline can generate reproducible simulated data (multi-seed) designed to stress *fairness vs. performance* trade-offs.

### Real dataset: UCI Student Performance (optional)
We include a processing script that converts UCI’s Student Performance dataset into the project’s input schema.

- Dataset repository: https://archive.ics.uci.edu/dataset/320/student%2Bperformance  
- Dataset DOI (UCI): **10.24432/C5TG7T**

Script:
```bash
python scripts/fetch_uci_student_performance.py --out data/uci_student/uci_student_processed.csv
```

**Important:** We do **not** ship raw UCI files in the repo; we ship a processed CSV plus the script and documentation so others can reproduce the transformation.

---

## Documentation

- Wiki pages are maintained under `docs/wiki/` and can be pushed to GitHub Wiki (see `docs/PUSH_WIKI.md`).
- See also:
  - `docs/DESIGN_RATIONALE.md` (design choices and trade-offs)
  - `docs/DOCKER.md` (containerized reproducibility)
  - `docs/CITATION.md` + `CITATION.cff` (how to cite)

---

## Reproducibility checklist (tl;dr)

- [ ] Fix random seeds (configs already do this; paper uses multi-seed)
- [ ] Keep a copy of `configs/*.yaml` used for a run
- [ ] Archive `outputs/<run_id>/bundle.zip` alongside the paper
- [ ] Record Python and dependency versions (`pip freeze > requirements-lock.txt`)
- [ ] Prefer Docker for long-term artifact stability (see `docs/DOCKER.md`)

---

## Contributing

Issues and PRs are welcome. Please:
- include a short reproduction snippet in bug reports,
- keep outputs out of commits (`outputs/` is gitignored),
- run formatting/linting if you have `.[dev]` installed.

---

## License

See `LICENSE`.

