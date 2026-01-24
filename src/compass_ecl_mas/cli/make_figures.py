from __future__ import annotations

"""Generate paper-grade figures from an existing run directory.

Usage
-----
python -m compass_ecl_mas.cli.make_figures --run_dir outputs/<RUN_ID>

This is useful when you already have CSV outputs and just want to re-render
figures (e.g., after tweaking visualization choices).
"""

import argparse
from pathlib import Path

from compass_ecl_mas.viz.paper_figures import make_paper_figures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        required=True,
        help="Run directory produced by run_all.py (contains metrics_candidates.csv).",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    metrics = run_dir / "metrics_candidates.csv"
    pareto = run_dir / "pareto_front.csv"

    if not metrics.exists():
        raise FileNotFoundError(f"metrics_candidates.csv not found in: {run_dir}")

    outs = make_paper_figures(metrics_candidates_csv=metrics, pareto_front_csv=pareto if pareto.exists() else None, outdir=run_dir)
    print("[OK] Figures written:")
    for k, v in outs.items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    main()
