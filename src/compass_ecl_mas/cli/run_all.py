from __future__ import annotations

"""Run the full experiment suite and produce paper-ready artifacts.

This CLI is the main entrypoint to reproduce the results reported in the paper:
  - Synthetic data generation (education domain)
  - Baselines (no ECL)
  - Optional fairness post-processing baseline (Hardt-style Equal Opportunity)
  - ECL grid search (policy knobs)
  - Pareto frontier computation
  - MAS selection
  - Overleaf drop-in: figures + LaTeX tables
  - Reproducibility bundle (zip)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from ..pareto.mas import mas_select_post_pareto
from ..pareto.pareto import pareto_front
from ..pipeline import PostProcConfig, RunConfig, run_single
from ..reporting.paper_assets import (
    build_bundle,
    latex_table_baselines,
    latex_table_mas_selected,
    latex_table_fairlearn_reductions,
)
from ..data import load_dataset_from_config
from ..utils import ensure_dir, now_tag, sha1_of_dict

# Paper-grade figures (seed-aggregated trade-off plots)
from ..viz.paper_figures import make_paper_figures


def _mean_std(df: pd.DataFrame, cols: list[str]) -> dict:
    """Compute mean/std for a list of scalar metric columns."""
    out: dict[str, float] = {}
    for c in cols:
        out[c + "_mean"] = float(df[c].mean())
        out[c + "_std"] = float(df[c].std(ddof=0))
    return out


def _baseline_runs(df: pd.DataFrame, seed: int, model: str, top_k: int, expl_k: int) -> dict:
    """Run a baseline model without ECL.

    Notes
    -----
    - We set `abstention_enabled=False` and loosen constraints so feasibility does not gate the baseline.
    - We block sensitive attributes (e.g., SES/sex) from the predictive features for audit-only usage.
    """
    cfg = RunConfig(
        model=model,
        top_k=int(top_k),
        gamma=0.0,
        expl_k=int(expl_k),
        forbidden_features=["ses", "sex"],  # audit-only
        abstention_enabled=False,
        fairness_max_eopp_gap=1.0,
        explainability_max_features=999,
        explainability_max_entropy=999.0,
    )
    met, *_ = run_single(
        df,
        seed,
        cfg,
        fairness_group_col="ses",
        postproc=PostProcConfig(enabled=False),
    )
    met["baseline_type"] = f"{model.upper()} (no ECL)"
    return met


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to a YAML config (paper or quick).")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    run_id = f"{cfg['run']['name']}_{now_tag()}"
    out = ensure_dir(Path(cfg["run"]["out_dir"]) / run_id)

    # Overleaf drop-in structure
    figs = ensure_dir(out / "overleaf_dropin" / "figures")
    paper_assets_dir = ensure_dir(out / "overleaf_dropin" / "paper_assets")

    seeds = [int(s) for s in cfg["run"]["seeds"]]
    dcfg = cfg["data"]
    ecfg = cfg["ecl"]
    mcfg = cfg["mas"]
    policy = cfg["policy"]
    postcfg = cfg.get("baselines", {}).get("fairness_postprocessing", {"enabled": False})
    flcfg = cfg.get("baselines", {}).get("fairlearn_reductions", {"enabled": False})

    # Dataset probe (for captions/metadata). Works for both synthetic and CSV sources.
    try:
        df_probe = load_dataset_from_config(dcfg, seeds[0])
        n_samples = int(len(df_probe))
    except Exception:
        df_probe = None
        n_samples = None

    data_source = str(dcfg.get('source', 'synthetic')).lower()
    if data_source == 'synthetic':
        dataset_desc = 'simulated education dataset'
        dataset_size_str = f"$N={int(dcfg.get('n_students', n_samples or 0)):,}$"
    else:
        dataset_desc = 'real education dataset'
        dataset_size_str = f"$N={int(n_samples):,}$" if n_samples is not None else ""


    # --- collect: baselines (no ECL) ---
    baseline_rows: list[dict] = []
    for seed in seeds:
        df = load_dataset_from_config(dcfg, seed)
        for model in cfg["models"]["baselines"]:
            baseline_rows.append(
                _baseline_runs(
                    df,
                    seed,
                    model=model,
                    top_k=int(policy["topk_grid"][1]),
                    expl_k=int(policy["expl_k"]),
                )
            )
    df_baselines = pd.DataFrame(baseline_rows)
    df_baselines.to_csv(out / "baselines_no_ecl.csv", index=False)

    # Best baseline model by mean AUC across seeds
    best_base_id = (
        df_baselines.groupby("model", as_index=False)["auc"]
        .mean()
        .sort_values("auc", ascending=False)
        .iloc[0]["model"]
    )

    # --- fairness post-processing baseline (Hardt-style Equal Opportunity) ---
    post_rows: list[dict] = []
    if bool(postcfg.get("enabled", False)):
        for seed in seeds:
            df = load_dataset_from_config(dcfg, seed)
            cfg_post = RunConfig(
                model=str(best_base_id),
                top_k=int(policy["topk_grid"][1]),
                gamma=0.0,
                expl_k=int(policy["expl_k"]),
                forbidden_features=["ses", "sex"],
                abstention_enabled=False,
                fairness_max_eopp_gap=1.0,
                explainability_max_features=999,
                explainability_max_entropy=999.0,
            )
            _, _, met_pp, _ = run_single(
                df,
                seed,
                cfg_post,
                fairness_group_col="ses",
                postproc=PostProcConfig(enabled=True),
            )
            if met_pp is not None:
                met_pp["baseline_type"] = "EOpp post-proc (Hardt)"
                post_rows.append(met_pp)
    df_post = pd.DataFrame(post_rows) if post_rows else pd.DataFrame()

    # --- constrained optimization baseline (Fairlearn reductions: Equal Opportunity) ---
    df_fairlearn = pd.DataFrame()
    if bool(flcfg.get("enabled", False)):
        try:
            from ..baselines.fairlearn_reductions import FairlearnReductionsConfig, run_fairlearn_reductions_baseline
            eps_grid = tuple(float(x) for x in flcfg.get("eps_grid", [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]))
            fcfg = FairlearnReductionsConfig(
                method=str(flcfg.get("method", "exponentiated_gradient")),
                base_estimator=str(flcfg.get("base_estimator", "lr")),
                eps_grid=eps_grid,
                max_iter=int(flcfg.get("max_iter", 50)),
            )
            fl_rows = []
            for seed in seeds:
                df = load_dataset_from_config(dcfg, seed)
                fl_rows.append(
                    run_fairlearn_reductions_baseline(
                        df,
                        seed,
                        fairness_group_col="ses",
                        forbidden_features=("ses", "sex"),
                        top_k=int(policy["topk_grid"][1]),
                        cfg=fcfg,
                    )
                )
            if fl_rows:
                df_fairlearn = pd.concat(fl_rows, ignore_index=True)
                df_fairlearn.to_csv(out / "fairlearn_reductions.csv", index=False)
        except ImportError as e:
            print(f"[warn] Fairlearn reductions baseline not run (missing dependency). Install with: pip install '.[fairlearn]'  ({e})")
        except Exception as e:
            print(f"[warn] Could not run Fairlearn reductions baseline: {type(e).__name__}: {e}")


    # --- collect: ECL candidates grid (policy knobs) ---
    rows: list[dict] = []
    for seed in seeds:
        df = load_dataset_from_config(dcfg, seed)
        for model in cfg["models"]["baselines"]:
            for topk in policy["topk_grid"]:
                for gamma in policy["gamma_grid"]:
                    rc = RunConfig(
                        model=model,
                        top_k=int(topk),
                        gamma=float(gamma),
                        expl_k=int(policy["expl_k"]),
                        forbidden_features=list(ecfg["forbidden_features"]),
                        abstention_enabled=bool(ecfg["abstention_enabled"]),
                        fairness_max_eopp_gap=float(ecfg["fairness_max_eopp_gap"]),
                        explainability_max_features=int(ecfg["explainability_max_features"]),
                        explainability_max_entropy=float(ecfg["explainability_max_entropy"]),
                    )
                    met, *_ = run_single(
                        df,
                        seed,
                        rc,
                        fairness_group_col=str(ecfg["fairness_primary_attribute"]),
                        postproc=PostProcConfig(enabled=False),
                    )
                    met["seed"] = seed
                    rows.append(met)

    dfm = pd.DataFrame(rows)
    dfm.to_csv(out / "metrics_candidates.csv", index=False)

    # Pareto frontier (trade-off exploration)
    dfp = pareto_front(
        dfm,
        minimize_cols=["eopp_gap", "latency_ms", "expl_num_features", "expl_entropy"],
        maximize_cols=["auc"],
    )
    dfp.to_csv(out / "pareto_front.csv", index=False)

    # Best+ECL (no MAS): among feasible candidates, pick max AUC.
    feasible = dfm[dfm["feasible_under_ecl"] == True].copy()  # noqa: E712
    best_ecl_id = None
    if not feasible.empty:
        best_ecl_id = str(feasible.sort_values("auc", ascending=False).iloc[0]["candidate_id"])
    df_best_ecl = dfm[dfm["candidate_id"] == best_ecl_id].copy() if best_ecl_id else pd.DataFrame()
    df_best_ecl.to_csv(out / "best_ecl_no_mas.csv", index=False)

    # MAS selection (post-Pareto thresholds)
    selected = mas_select_post_pareto(
        candidates=dfm,
        pareto=dfp,
        perf_min_auc=float(mcfg["perf_min_auc"]),
        fairness_max_eopp_gap=float(mcfg["fairness_max_eopp_gap"]),
        explainability_max_features=int(mcfg["explainability_max_features"]),
        explainability_max_entropy=float(mcfg["explainability_max_entropy"]),
        cost_max_latency_ms=float(mcfg["cost_max_latency_ms"]),
    )
    mas_df = dfm[dfm["candidate_id"] == selected].copy() if selected else pd.DataFrame()
    mas_df.to_csv(out / "mas_selected.csv", index=False)

    # --- FIGURES (legacy names expected by the paper draft) ---
    # Figure: AUC vs ΔEOpp
    fig = plt.figure()
    plt.scatter(dfm["eopp_gap"], dfm["auc"], alpha=0.35, label="candidates")
    plt.scatter(dfp["eopp_gap"], dfp["auc"], alpha=0.90, label="pareto")
    if selected:
        r = dfm[dfm["candidate_id"] == selected].iloc[0]
        plt.scatter([r["eopp_gap"]], [r["auc"]], marker="*", s=260, label="MAS selected")
    plt.xlabel("EOpp gap (SES) ↓")
    plt.ylabel("AUC ↑")
    plt.legend()
    p1 = figs / "pareto_perf_fair.png"
    fig.savefig(p1, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Figure: AUC vs explainability proxy (#features)
    fig = plt.figure()
    plt.scatter(dfm["expl_num_features"], dfm["auc"], alpha=0.35, label="candidates")
    plt.scatter(dfp["expl_num_features"], dfp["auc"], alpha=0.90, label="pareto")
    if selected:
        r = dfm[dfm["candidate_id"] == selected].iloc[0]
        plt.scatter([r["expl_num_features"]], [r["auc"]], marker="*", s=260, label="MAS selected")
    plt.xlabel("Explainability proxy (#features) ↓")
    plt.ylabel("AUC ↑")
    plt.legend()
    p2 = figs / "pareto_perf_exp.png"
    fig.savefig(p2, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # --- Paper-grade figures (recommended for journals) ---
    # These are the two key plots you requested:
    #   (1) F1 × ΔEOpp with color = Coverage
    #   (2) Coverage × F1 with color = ΔEOpp
    paper_fig1 = None
    paper_fig2 = None
    try:
        make_paper_figures(
            metrics_candidates_csv=out / "metrics_candidates.csv",
            pareto_front_csv=(out / "pareto_front.csv") if (out / "pareto_front.csv").exists() else None,
            outdir=out,
        )
        generated_dir = out / "figures"
        paper_fig1 = generated_dir / "fig_f1_vs_eopp_color_coverage.png"
        paper_fig2 = generated_dir / "fig_coverage_vs_f1_color_eopp.png"

        # Copy into the Overleaf drop-in folder
        if paper_fig1.exists():
            (figs / paper_fig1.name).write_bytes(paper_fig1.read_bytes())
        if paper_fig2.exists():
            (figs / paper_fig2.name).write_bytes(paper_fig2.read_bytes())

        print(f"[ok] Paper-grade figures generated and copied to: {figs}")
    except Exception as e:
        print(f"[warn] Could not generate paper-grade figures: {e}")

    # --- TABLES (LaTeX drop-in) ---
    baseline_table_rows: list[dict] = []

    # LR/RF/GBDT (no ECL)
    for mname in ["lr", "rf", "gbdt"]:
        sub = df_baselines[df_baselines["model"] == mname]
        if sub.empty:
            continue
        row = {"method": f"{mname.upper()} (no ECL)"}
        row.update(_mean_std(sub, ["auc", "f1", "eopp_gap", "abstention_rate"]))
        baseline_table_rows.append(row)

    # fairness post-proc row
    if not df_post.empty:
        row = {"method": "EOpp post-proc (Hardt)"}
        row.update(_mean_std(df_post, ["auc", "f1", "eopp_gap", "abstention_rate"]))
        baseline_table_rows.append(row)

    # Best + ECL (no MAS)
    if best_ecl_id:
        sub = dfm[dfm["candidate_id"] == best_ecl_id]
        row = {"method": "Best + ECL (no MAS)"}
        row.update(_mean_std(sub, ["auc", "f1", "eopp_gap", "abstention_rate"]))
        baseline_table_rows.append(row)

    # ECL + MAS selected
    if selected:
        sub = dfm[dfm["candidate_id"] == selected]
        row = {"method": "ECL + MAS (selected)"}
        row.update(_mean_std(sub, ["auc", "f1", "eopp_gap", "abstention_rate"]))
        baseline_table_rows.append(row)

    tdf = pd.DataFrame(baseline_table_rows)
    if not tdf.empty:
        tdf = tdf.rename(
            columns={
                "eopp_gap_mean": "eopp_mean",
                "eopp_gap_std": "eopp_std",
                "abstention_rate_mean": "abst_mean",
                "abstention_rate_std": "abst_std",
            }
        )
    baselines_tex = latex_table_baselines(
        tdf,
        caption=(
            f"Baseline comparison on the {dataset_desc} ({dataset_size_str}; "
            f"mean$\\pm$std over {len(seeds)} seeds)."
        ),
        label="tab:baselines",
    )
    (paper_assets_dir / "tab_baselines.tex").write_text(baselines_tex, encoding="utf-8")

    # Fairlearn reductions (EOpp constrained optimization) — report trade-off curve over eps
    if not df_fairlearn.empty:
        fr_rows = []
        for eps, sub in df_fairlearn.groupby("eps", as_index=False):
            row = {"eps": float(eps)}
            row.update(_mean_std(sub, ["auc", "f1", "eopp_gap"]))
            fr_rows.append(row)
        fr_df = pd.DataFrame(fr_rows).sort_values("eps")
        fairlearn_tex = latex_table_fairlearn_reductions(
            fr_df,
            caption=(
                "Fairlearn reductions baseline (ExponentiatedGradient with Equal Opportunity constraints), "
                "sweeping $\\epsilon$ (mean$\\pm$std over seeds)."
            ),
            label="tab:fairlearn_reductions",
        )
        (paper_assets_dir / "tab_fairlearn_reductions.tex").write_text(fairlearn_tex, encoding="utf-8")


    # MAS selected + representative Pareto alternatives
    mas_rows: list[dict] = []
    if not dfp.empty:
        pareto_sorted_auc = dfp.sort_values("auc", ascending=False)
        pareto_sorted_fair = dfp.sort_values("eopp_gap", ascending=True)
        pareto_sorted_exp = dfp.sort_values("expl_entropy", ascending=True)

        def add_candidate(title: str, row: pd.Series) -> None:
            mas_rows.append(
                {
                    "candidate": title,
                    "auc": float(row["auc"]),
                    "eopp_gap": float(row["eopp_gap"]),
                    "expl_num_features": float(row["expl_num_features"]),
                    "expl_entropy": float(row["expl_entropy"]),
                }
            )

        if selected:
            r = dfm[dfm["candidate_id"] == selected].iloc[0]
            add_candidate("MAS selected", r)

        add_candidate("Pareto: max AUC", pareto_sorted_auc.iloc[0])
        add_candidate("Pareto: min EOpp gap", pareto_sorted_fair.iloc[0])
        add_candidate("Pareto: min expl. entropy", pareto_sorted_exp.iloc[0])

    mas_tex = latex_table_mas_selected(
        pd.DataFrame(mas_rows),
        caption=(
            "Selected MAS candidate and representative Pareto alternatives "
            "(single best points under different trade-off preferences)."
        ),
        label="tab:mas_selected",
    )
    (paper_assets_dir / "tab_mas_selected.tex").write_text(mas_tex, encoding="utf-8")

    # --- Bundle (zip) ---
    summary = {
        "run_id": run_id,
        "config_path": args.config,
        "config_sha1": sha1_of_dict(cfg),
        "data_source": data_source,
        "dataset_desc": dataset_desc,
        "n_samples": int(n_samples) if n_samples is not None else None,
        "n_students": int(dcfg.get("n_students")) if data_source == "synthetic" and dcfg.get("n_students") is not None else None,
        "seeds": list(seeds),
        "best_baseline_model": str(best_base_id),
        "best_ecl_no_mas_id": best_ecl_id,
        "mas_selected_id": selected,
        "notes": (
            "Synthetic data only; do not use for real student impact decisions."
            if data_source == "synthetic"
            else "Real dataset used for research; ensure appropriate governance, privacy, and institutional approvals before any deployment."
        ),
        "overleaf_expected_files": {
            "figures": [
                "figures/pareto_perf_fair.png",
                "figures/pareto_perf_exp.png",
                "figures/fig_f1_vs_eopp_color_coverage.png",
                "figures/fig_coverage_vs_f1_color_eopp.png",
            ],
            "tables": [
                "paper_assets/tab_baselines.tex",
                "paper_assets/tab_mas_selected.tex",
                "paper_assets/tab_fairlearn_reductions.tex",
            ],
        },
    }

    # --- DECISION BRIEF (PDF for non-technical stakeholders) ---
    decision_brief_pdf = None
    try:
        # Lazy import so the pipeline can still run if PDF deps are not installed.
        from compass_ecl_mas.reporting.decision_brief_pdf import (
            BriefInputs,
            generate_decision_brief_pdf,
        )
    except ImportError:
        print("[warn] Decision brief PDF not generated (missing dependency). Install with: pip install reportlab")
    else:
        try:
            brief_inputs = BriefInputs(
                run_id=run_id,
                dataset_desc=dataset_desc,
                dataset_size_str=dataset_size_str,
                seeds=seeds,
                cfg=cfg,
                out_dir=out,
            )
            decision_brief_pdf = generate_decision_brief_pdf(brief_inputs)
            print(f"[ok] Decision brief PDF generated: {decision_brief_pdf}")
        except Exception as e:
            print(f"[warn] Could not generate decision brief PDF: {e}")

    csvs = {
        "baselines_no_ecl": df_baselines,
        "metrics_candidates": dfm,
        "pareto_front": dfp,
        "mas_selected": mas_df,
    }
    if not df_post.empty:
        csvs["fairness_postproc"] = df_post

    if not df_fairlearn.empty:
        csvs["fairlearn_reductions"] = df_fairlearn

    files = {
        "overleaf_dropin/figures/pareto_perf_fair.png": p1,
        "overleaf_dropin/figures/pareto_perf_exp.png": p2,
    }

    if paper_fig1 and paper_fig1.exists():
        files[f"overleaf_dropin/figures/{paper_fig1.name}"] = figs / paper_fig1.name
    if paper_fig2 and paper_fig2.exists():
        files[f"overleaf_dropin/figures/{paper_fig2.name}"] = figs / paper_fig2.name

    if decision_brief_pdf is not None and hasattr(decision_brief_pdf, "exists") and decision_brief_pdf.exists():
        files["decision_brief.pdf"] = decision_brief_pdf

    latex = {
        "tab_baselines": baselines_tex,
        "tab_mas_selected": mas_tex,
    }

    bundle = build_bundle(out, summary, csvs=csvs, files=files, latex=latex)

    print(f"[OK] {run_id}")
    print(f"Outputs: {out}")
    print(f"Bundle: {bundle}")
    print("Overleaf drop-in folder:", out / "overleaf_dropin")


if __name__ == "__main__":
    main()