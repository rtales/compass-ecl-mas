from __future__ import annotations

"""Paper-grade figures used in the results section.

The main CLI (`compass_ecl_mas.cli.run_all`) calls `make_paper_figures` so that
one command produces both the quantitative outputs (CSVs/tables) and the
publication-ready figures.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_mkdir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _aggregate_over_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate candidate metrics across seeds (mean), keeping key columns."""
    # Columns we expect in metrics_candidates.csv
    keep = [
        "candidate_id",
        "model",
        "topk",
        "gamma",
        "expl_k",
        "auc",
        "f1",
        "eopp_gap",
        "abstention_rate",
        "coverage",
        "feasible_under_ecl",
        "postproc_enabled",
    ]
    cols = [c for c in keep if c in df.columns]
    g = df[cols].groupby("candidate_id", as_index=False)

    # For numeric columns -> mean; for non-numeric -> first
    numeric = [c for c in cols if c not in {"candidate_id", "model"}]
    out = g[numeric].mean(numeric_only=True)
    if "model" in cols:
        out["model"] = g["model"].first()["model"].values
    out = out.sort_values("f1", ascending=False)
    return out


def make_paper_figures(
    metrics_candidates_csv: Path,
    pareto_front_csv: Optional[Path],
    outdir: Path,
) -> dict[str, Path]:
    """Generate the two key journal-friendly trade-off figures.

    Figures requested:
      1) F1 × ΔEOpp with color = Coverage
      2) Coverage × F1 with color = ΔEOpp

    Notes
    -----
    - We aggregate candidate metrics across seeds before plotting.
    - We use the default matplotlib colormap (no hard-coded colors).
    """
    metrics_candidates_csv = Path(metrics_candidates_csv)
    if not metrics_candidates_csv.exists():
        raise FileNotFoundError(f"metrics_candidates.csv not found: {metrics_candidates_csv}")

    outdir = Path(outdir)
    fig_dir = _safe_mkdir(outdir / "figures")

    df = pd.read_csv(metrics_candidates_csv)
    df_agg = _aggregate_over_seeds(df)

    pareto = None
    if pareto_front_csv is not None:
        pareto_front_csv = Path(pareto_front_csv)
        if pareto_front_csv.exists():
            pareto = pd.read_csv(pareto_front_csv)

    # Optional: Fairlearn reductions baseline points (constrained optimization)
    fairlearn = None
    fl_csv = outdir / "fairlearn_reductions.csv"
    if fl_csv.exists():
        try:
            fairlearn = pd.read_csv(fl_csv)
        except Exception:
            fairlearn = None

    outputs: dict[str, Path] = {}

    # --- Figure 1: F1 vs EOpp gap, color = coverage ---
    fig = plt.figure()
    sc = plt.scatter(
        df_agg["eopp_gap"],
        df_agg["f1"],
        c=df_agg.get("coverage", df_agg.get("coverage", np.ones(len(df_agg)))),
        alpha=0.65,
    )
    plt.xlabel(r"$\Delta$EOpp (SES) $\downarrow$")
    plt.ylabel(r"F1 $\uparrow$")
    cb = plt.colorbar(sc)
    cb.set_label("Coverage")
    # Optional overlays (plotted first, legend only if labels exist)
    if pareto is not None and ("eopp_gap" in pareto.columns) and ("f1" in pareto.columns):
        plt.scatter(pareto["eopp_gap"], pareto["f1"], alpha=0.9, marker="x", label="Pareto front")

    if fairlearn is not None and ("eopp_gap" in fairlearn.columns) and ("f1" in fairlearn.columns):
        plt.scatter(fairlearn["eopp_gap"], fairlearn["f1"], alpha=0.9, marker="^", label="Fairlearn reductions")

    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend()

    p1 = fig_dir / "fig_f1_vs_eopp_color_coverage.png"
    fig.savefig(p1, dpi=240, bbox_inches="tight")
    plt.close(fig)
    outputs[p1.name] = p1

    # --- Figure 2: Coverage vs F1, color = EOpp gap ---
    fig = plt.figure()
    sc = plt.scatter(
        df_agg.get("coverage", 1.0 - df_agg.get("abstention_rate", 0.0)),
        df_agg["f1"],
        c=df_agg["eopp_gap"],
        alpha=0.65,
    )
    plt.xlabel(r"Coverage $\uparrow$")
    plt.ylabel(r"F1 $\uparrow$")
    cb = plt.colorbar(sc)
    cb.set_label(r"$\Delta$EOpp")

    p2 = fig_dir / "fig_coverage_vs_f1_color_eopp.png"
    fig.savefig(p2, dpi=240, bbox_inches="tight")
    plt.close(fig)
    outputs[p2.name] = p2

    return outputs