# Running Experiments

## CLI entrypoint
All main artifacts are generated via:
```bash
python -m compass_ecl_mas.cli.run_all --config <config.yaml>
```

## Configs provided
- `configs/education_quick.yaml` — fast smoke test
- `configs/education_paper.yaml` — paper-style settings (simulated, heavier)
- `configs/education_uci.yaml` — real-data settings (UCI Student Performance)

## Multi-seed reporting (paper)
For journal-level reporting, prefer multi-seed runs and report mean ± std. The pipeline supports this via the config (seeds list) and aggregates tables/figures accordingly.

## Where outputs go
After each run:
- `outputs/<run_id>/` contains CSVs, figures, and the optional Decision Brief PDF
- `outputs/<run_id>/overleaf_dropin/` contains:
  - `figures/` (PNG)
  - `tables/` (LaTeX `tab_*.tex`)
- `outputs/<run_id>/bundle.zip` is a portable archive of the run

## Optional baselines
### 1) Hardt-style Equal Opportunity post-processing
Included by default in paper configs as a baseline.

### 2) Fairlearn reductions baseline
Install extras:
```bash
python -m pip install -e ".[fairlearn]"
```
Then re-run the pipeline with a config that enables the reductions baseline.

---

## Tips for debugging
- Always check the `run_id` printed at the end and verify that your Overleaf paper is using the **same** drop-in folder.
- If a baseline is skipped, look for a `[warn]` line; most are dependency-related.

