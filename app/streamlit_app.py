"""Compass-ECL-MAS Streamlit Lab

This UI is intentionally educational and audit-friendly.

Two recommended workflows:
1) Explore (single-seed): run quick interactive sweeps to understand trade-offs.
2) Review (paper-grade): run the CLI multi-seed pipeline, then upload the generated
   bundle.zip here to inspect artifacts (figures, tables, decision_brief.pdf) in one place.

The CLI pipeline remains the source of truth for paper-grade numbers.

"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict
from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from compass_ecl_mas.data import load_dataset_from_config
from compass_ecl_mas.pipeline import PostProcConfig, RunConfig, run_single
from compass_ecl_mas.pareto.mas import mas_select_post_pareto
from compass_ecl_mas.pareto.pareto import pareto_front
from compass_ecl_mas.reporting.decision_brief_pdf import BriefInputs, generate_decision_brief_pdf
from compass_ecl_mas.reporting.paper_assets import latex_table_baselines, latex_table_mas_selected

APP_TITLE = "🧭 Compass-ECL-MAS — Interactive Lab"

st.set_page_config(page_title="Compass-ECL-MAS — Lab", page_icon="🧭", layout="wide")
st.title(APP_TITLE)
st.caption(
    "Interactively explore ethical trade-offs (performance × fairness × coverage × explainability), "
    "then export audit artifacts. For paper-grade results, use the CLI multi-seed pipeline and upload its bundle.zip."
)

# ----------------------------
# Utilities
# ----------------------------
def _fmt_pct(x: float) -> str:
    return f"{100.0 * float(x):.1f}%"

def _safe_float(x, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default

def _bundle_find_member(z: zipfile.ZipFile, name: str, prefer_prefixes: Optional[List[str]] = None) -> Optional[str]:
    """Find a member inside a bundle.zip regardless of folder layout.

    We support both:
      - flat files (e.g., metrics_candidates.csv)
      - structured bundles (e.g., csv/metrics_candidates.csv, overleaf_dropin/figures/*.png, overleaf_dropin/paper_assets/*.tex)
    """
    members = z.namelist()
    if name in members:
        return name

    prefixes = ["csv/", "overleaf_dropin/figures/", "overleaf_dropin/paper_assets/", "figures/", "paper_assets/"]
    if prefer_prefixes:
        prefixes = list(prefer_prefixes) + [p for p in prefixes if p not in prefer_prefixes]

    for p in prefixes:
        candidate = f"{p}{name}"
        if candidate in members:
            return candidate

    # Fallback: any path that ends with the requested file name
    hits = [m for m in members if m.endswith("/" + name) or m.endswith(name)]
    if hits:
        hits.sort(key=lambda x: (len(x), x))
        return hits[0]
    return None


def _bundle_read_csv(z: zipfile.ZipFile, name: str) -> Optional[pd.DataFrame]:
    member = _bundle_find_member(z, name, prefer_prefixes=["csv/"])
    if not member:
        return None
    with z.open(member) as f:
        return pd.read_csv(f)


def _bundle_read_bytes(z: zipfile.ZipFile, name: str) -> Optional[bytes]:
    member = _bundle_find_member(z, name)
    if not member:
        return None
    with z.open(member) as f:
        return f.read()

def _render_pdf_download(pdf_bytes: bytes, filename: str = "decision_brief.pdf") -> None:
    st.download_button(
        "Download Decision Brief (PDF)",
        data=pdf_bytes,
        file_name=filename,
        mime="application/pdf",
        use_container_width=True,
    )
    st.caption("Tip: open the downloaded PDF in your system viewer for best rendering.")

@st.cache_data(show_spinner=False)
def _load_synth(
    seed: int,
    n_students: int,
    n_schools: int,
    measurement_noise: float,
    missingness: float,
    drift_strength: float,
) -> pd.DataFrame:
    """Synthetic dataset generator matching the paper's 'education_A' setup."""
    return load_dataset_from_config(
        {
            "source": "synthetic",
            "name": "education_A",
            "n_students": int(n_students),
            "n_schools": int(n_schools),
            "bias": {
                "measurement_noise_by_group": float(measurement_noise),
                "missingness_by_group": float(missingness),
                "drift_strength": float(drift_strength),
            },
        },
        seed=int(seed),
    )

def _load_real(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

def _explain_constraints() -> None:
    with st.expander("What do these constraints mean? (quick glossary)", expanded=False):
        st.markdown(
            """
- **Fairness (ΔEOpp / Equal Opportunity gap)**: absolute TPR difference between SES groups. Lower is better.
- **Coverage**: fraction of cases the model *does not abstain* on (i.e., auto-decisions). Higher is better.
- **Abstention (gamma)**: higher gamma → more conservative; more cases sent to human review (lower coverage).
- **Explainability proxies**:
  - **#features**: how many top features are surfaced in an explanation.
  - **Entropy**: concentration/uncertainty over the explanation feature-weights (lower is more concentrated).
- **Forbidden features**: sensitive attributes used for audit only (e.g., SES). They are removed from predictive features.
"""
        )

def _explain_actions() -> None:
    with st.expander("What does each action generate? Where do I find the outputs?", expanded=False):
        st.markdown(
            """
**Single Run**
- Runs one policy configuration on one seed (fast).
- Generates: metrics card + group audit table, plus optional Hardt-style post-processing baseline.

**Pareto Sweep**
- Sweeps **Top-K** and **gamma** to create many candidates.
- Computes the **Pareto front** and selects a policy with **MAS** (thresholds + tie-breakers).
- Generates in-app tables + a scatter plot.

**Export (bundle.zip)**
- Creates a *downloadable* `bundle.zip` containing:
  - `metrics_candidates.csv`, `pareto_front.csv`, `mas_selected.csv` (if selected)
  - `plots/*.png` (trade-off plots)
  - `paper_assets/*.tex` (LaTeX tables)
  - `decision_brief.pdf` (if ReportLab is installed)
- In the CLI pipeline, similar artifacts are written under `outputs/<run_id>/...`. In Streamlit, you download the zip.
"""
        )

def _has_labels_for_legend(ax) -> bool:
    handles, labels = ax.get_legend_handles_labels()
    return bool(handles)

def _plot_scatter(df: pd.DataFrame, x: str, y: str, *, highlight_id: Optional[str] = None, title: str = "") -> plt.Figure:
    fig = plt.figure()
    ax = plt.gca()
    ax.scatter(df[x], df[y], label="candidates")
    if highlight_id is not None and "candidate_id" in df.columns:
        sel = df[df["candidate_id"] == highlight_id]
        if len(sel) > 0:
            ax.scatter(sel[x], sel[y], marker="x", s=120, label="MAS-selected")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    if _has_labels_for_legend(ax):
        ax.legend()
    return fig

def _make_decision_brief_demo(
    *,
    data_source: str,
    seed: int,
    cfg: RunConfig,
    dfm: pd.DataFrame,
    dfp: pd.DataFrame,
    mas_df: pd.DataFrame,
    fig_bytes_1: bytes,
    fig_bytes_2: bytes,
) -> Optional[bytes]:
    """Generate a single-seed demo Decision Brief PDF in a temp dir."""
    try:
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            (out_dir / "overleaf_dropin" / "figures").mkdir(parents=True, exist_ok=True)

            out_dir.joinpath("metrics_candidates.csv").write_text(dfm.to_csv(index=False))
            out_dir.joinpath("pareto_front.csv").write_text(dfp.to_csv(index=False))
            if not mas_df.empty:
                out_dir.joinpath("mas_selected.csv").write_text(mas_df.to_csv(index=False))

            out_dir.joinpath("overleaf_dropin/figures/fig_f1_vs_eopp_color_coverage.png").write_bytes(fig_bytes_1)
            out_dir.joinpath("overleaf_dropin/figures/fig_coverage_vs_f1_color_eopp.png").write_bytes(fig_bytes_2)

            brief_inputs = BriefInputs(
                run_id="streamlit_demo",
                dataset_desc=f"{data_source} (single-seed demo)",
                dataset_size_str="N/A",
                seeds=[int(seed)],
                cfg={"ecl": dict(fairness_primary_attribute="ses")},
                out_dir=out_dir,
            )
            pdf_path = generate_decision_brief_pdf(brief_inputs)
            return pdf_path.read_bytes()
    except Exception:
        return None

def _build_bundle_zip(
    *,
    data_source: str,
    seed: int,
    cfg: RunConfig,
    selected: Optional[str],
    dfm: pd.DataFrame,
    dfp: pd.DataFrame,
    mas_df: pd.DataFrame,
    baselines_tex: str,
    mas_tex: str,
    plot1_bytes: bytes,
    plot2_bytes: bytes,
    decision_brief_bytes: Optional[bytes],
) -> bytes:
    summary = dict(
        data_source=data_source,
        seed=int(seed),
        cfg=asdict(cfg),
        selected=selected,
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("summary.json", json.dumps(summary, indent=2))
        z.writestr("metrics_candidates.csv", dfm.to_csv(index=False))
        z.writestr("pareto_front.csv", dfp.to_csv(index=False))
        if not mas_df.empty:
            z.writestr("mas_selected.csv", mas_df.to_csv(index=False))

        z.writestr("plots/f1_vs_eopp.png", plot1_bytes)
        z.writestr("plots/f1_vs_coverage.png", plot2_bytes)

        if decision_brief_bytes is not None:
            z.writestr("decision_brief.pdf", decision_brief_bytes)

        z.writestr("paper_assets/tab_baselines_demo.tex", baselines_tex)
        if mas_tex:
            z.writestr("paper_assets/tab_mas_demo.tex", mas_tex)

    return buffer.getvalue()

# ----------------------------
# Sidebar: Controls + Help
# ----------------------------
PRESETS: Dict[str, Dict] = {
    "Balanced (recommended)": dict(model="gbdt", top_k=1000, gamma=0.10, expl_k=8, abst=True, eopp=0.10, max_feat=10, max_ent=1.60),
    "Performance-first": dict(model="gbdt", top_k=1500, gamma=0.05, expl_k=10, abst=False, eopp=0.20, max_feat=20, max_ent=2.00),
    "Fairness-first": dict(model="lr", top_k=1000, gamma=0.12, expl_k=6, abst=True, eopp=0.06, max_feat=8, max_ent=1.40),
    "Conservative (more human review)": dict(model="gbdt", top_k=1000, gamma=0.18, expl_k=8, abst=True, eopp=0.10, max_feat=10, max_ent=1.60),
}

with st.sidebar:
    st.header("Controls")
    preset = st.selectbox(
        "Preset",
        list(PRESETS.keys()),
        index=0,
        help="Quickly loads a reasonable set of policy parameters you can tweak.",
    )
    if st.button("Apply preset", use_container_width=True):
        st.session_state.update(PRESETS[preset])
        st.success(f"Preset applied: {preset}")

    st.divider()
    st.subheader("Data")
    data_source = st.radio(
        "Data source",
        ["Synthetic (education_A)", "Real CSV (preprocessed)"],
        index=0,
        help="Synthetic mirrors the paper's simulation. Real CSV expects columns: features + ses (audit-only) + y (target).",
    )
    seed = st.selectbox("Seed", [0, 1, 2, 3, 4], index=0, help="Controls train/test split and (some) stochasticity.")

    csv_file = None
    if data_source.startswith("Real"):
        csv_file = st.file_uploader("Upload preprocessed CSV", type=["csv"], help="Use the adapter script to generate a compatible CSV.")
        st.caption("Required columns: `y` (target) and `ses` (audit-only sensitive attribute).")
        n_students, n_schools = 0, 0
        measurement_noise, missingness, drift_strength = 0.0, 0.0, 0.0
    else:
        n_students = st.select_slider("N students", options=[5000, 10000, 20000, 50000], value=20000, help="Total synthetic samples.")
        n_schools = st.slider("N schools", 10, 200, 50, 10, help="Controls hierarchy in the simulator.")
        with st.expander("Bias knobs (simulation)", expanded=False):
            measurement_noise = st.slider("measurement_noise_by_group", 0.0, 1.0, 0.25, 0.05, help="Higher → noisier measurements for one group.")
            missingness = st.slider("missingness_by_group", 0.0, 0.6, 0.25, 0.05, help="Higher → more missing values for one group.")
            drift_strength = st.slider("drift_strength", 0.0, 1.0, 0.20, 0.05, help="Higher → distribution drift across schools.")

    st.divider()
    st.subheader("Model + Policy")
    model = st.selectbox("Model", ["lr", "rf", "gbdt"], index=2, help="Baseline learner used to score candidates.")
    top_k = st.slider(
        "Top-K budget",
        10,
        5000,
        int(st.session_state.get("top_k", 1000)),
        10,
        help="Operational budget: maximum number of cases to auto-approve/allocate (higher can improve utility).",
    )
    gamma = st.slider(
        "Abstention gamma",
        0.0,
        0.30,
        float(st.session_state.get("gamma", 0.10)),
        0.01,
        help="Controls abstention (human review). Higher gamma → more conservative → lower coverage.",
    )
    expl_k = st.slider(
        "Explainability proxy top-k features",
        3,
        20,
        int(st.session_state.get("expl_k", 8)),
        1,
        help="Proxy for how many features appear in explanations (smaller is simpler).",
    )

    st.divider()
    st.subheader("ECL constraints (Primary: SES)")
    forbidden_features = st.multiselect(
        "Forbidden features (audit-only)",
        ["ses", "sex"],
        default=["ses"],
        help="These columns are removed from predictive features and used only for fairness auditing.",
    )
    abst = st.toggle(
        "Enable abstention (HUMAN_REVIEW)",
        value=bool(st.session_state.get("abst", True)),
        help="If enabled, uncertain cases can be deferred to human review to manage risk.",
    )
    eopp = st.slider(
        "Max ΔEOpp gap (SES)",
        0.0,
        0.30,
        float(st.session_state.get("eopp", 0.10)),
        0.01,
        help="Fairness constraint: maximum allowed Equal Opportunity gap between SES groups (lower is stricter).",
    )
    max_feat = st.slider(
        "Max explanation features",
        3,
        30,
        int(st.session_state.get("max_feat", 10)),
        1,
        help="Explainability constraint: maximum number of features surfaced in explanations.",
    )
    max_ent = st.slider(
        "Max explanation entropy",
        0.5,
        3.0,
        float(st.session_state.get("max_ent", 1.60)),
        0.05,
        help="Explainability constraint: maximum entropy of explanation weights (lower encourages concentrated explanations).",
    )

    st.divider()
    st.subheader("Baselines")
    postproc_enabled = st.toggle(
        "Fairness post-processing baseline (Hardt-style EOpp)",
        value=True,
        help="Runs a post-processing baseline that adjusts predictions to reduce ΔEOpp without changing the base model.",
    )
    fairlearn_enabled = st.toggle(
        "Fairlearn reductions baseline (Equal Opportunity constrained optimization)",
        value=False,
        help="Runs constrained optimization (ExponentiatedGradient) for a grid of ε bounds on ΔEOpp.",
    )
    eps_grid = st.multiselect(
        "Fairlearn ε grid (ΔEOpp bound)",
        [0.00, 0.02, 0.05, 0.10, 0.15, 0.20],
        default=[0.00, 0.05, 0.10],
        help="Each ε is a bound on allowed ΔEOpp. Smaller ε → stricter fairness → typically lower performance.",
    )

    st.divider()
    _explain_constraints()
    _explain_actions()

# ----------------------------
# Resolve dataset
# ----------------------------
if data_source.startswith("Real"):
    if csv_file is None:
        st.info("Upload a CSV to run on real data, or switch to synthetic.")
        st.stop()
    df = _load_real(csv_file)
else:
    df = _load_synth(seed, n_students, n_schools, measurement_noise, missingness, drift_strength)

required = {"y", "ses"}
missing = required - set(df.columns)
if missing:
    st.error(f"Dataset missing required columns: {sorted(list(missing))}.")
    st.stop()

cfg = RunConfig(
    model=str(model),
    top_k=int(top_k),
    gamma=float(gamma),
    expl_k=int(expl_k),
    forbidden_features=list(forbidden_features),
    abstention_enabled=bool(abst),
    fairness_max_eopp_gap=float(eopp),
    explainability_max_features=int(max_feat),
    explainability_max_entropy=float(max_ent),
)

# ----------------------------
# Main layout
# ----------------------------
tab_explore, tab_review = st.tabs(["Explore (run inside the app)", "Review (upload a CLI bundle.zip)"])

# ============
# Explore tab
# ============
with tab_explore:
    st.subheader("Explore (single-seed)")
    st.caption("Use this for intuition. For paper-grade numbers, run the CLI (multi-seed) and use the Review tab.")

    with st.expander("Dataset preview", expanded=False):
        st.write(f"Rows: {len(df):,} • Columns: {len(df.columns):,}")
        st.dataframe(df.head(25), use_container_width=True)

    colA, colB = st.columns([1.15, 0.85], gap="large")

    with colA:
        st.markdown("### 1) Single run")
        st.caption("Runs one policy configuration and shows key metrics + group audit table.")
        if st.button("Run single policy", type="primary", use_container_width=True):
            with st.status("Running single policy…", expanded=False) as status:
                met, audit, met_pp, audit_pp = run_single(
                    df, int(seed), cfg, fairness_group_col="ses", postproc=PostProcConfig(enabled=postproc_enabled)
                )
                st.session_state["single"] = dict(met=met, audit=audit, met_pp=met_pp, audit_pp=audit_pp)
                status.update(label="Single policy complete.", state="complete")

        if "single" in st.session_state:
            m = st.session_state["single"]["met"]
            c = st.columns(6)
            c[0].metric("AUC", f"{_safe_float(m.get('auc')):.3f}")
            c[1].metric("F1", f"{_safe_float(m.get('f1')):.3f}")
            c[2].metric("ΔEOpp (ses)", f"{_safe_float(m.get('eopp_gap')):.3f}")
            c[3].metric("Coverage", f"{_safe_float(m.get('coverage')):.3f}")
            c[4].metric("Expl entropy", f"{_safe_float(m.get('expl_entropy')):.3f}")
            c[5].metric("Latency (ms)", f"{_safe_float(m.get('latency_ms')):.1f}")

            st.markdown("**Group audit (SES)**")
            st.dataframe(st.session_state["single"]["audit"], use_container_width=True)

            if st.session_state["single"]["met_pp"] is not None:
                st.markdown("#### Post-processing fairness baseline (Hardt-style EOpp)")
                mpp = st.session_state["single"]["met_pp"]
                st.write(
                    {
                        "AUC": _safe_float(mpp.get("auc")),
                        "F1": _safe_float(mpp.get("f1")),
                        "ΔEOpp": _safe_float(mpp.get("eopp_gap")),
                    }
                )
                st.dataframe(st.session_state["single"]["audit_pp"], use_container_width=True)

        st.divider()

        st.markdown("### 2) Pareto sweep")
        st.caption("Sweeps Top-K and gamma, builds a candidate set, computes Pareto front, then selects with MAS.")
        K_grid = st.multiselect(
            "Top-K grid",
            [10, 25, 50, 75, 100, 150, 200, 500, 800, 1000, 1500, 2000],
            default=[800, 1000, 1500],
            help="Higher Top-K can improve utility but may worsen fairness or cost.",
        )
        G_grid = st.multiselect(
            "Gamma grid",
            [0.00, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20],
            default=[0.05, 0.10, 0.15],
            help="Higher gamma means more abstention (human review), usually improving fairness but reducing coverage.",
        )

        run_sweep = st.button("Run Pareto sweep", type="primary", use_container_width=True)
        if run_sweep:
            with st.status("Running sweep…", expanded=False) as status:
                rows = []
                total = len(K_grid) * len(G_grid)
                done = 0
                prog = st.progress(0.0)
                for K in K_grid:
                    for g in G_grid:
                        cfg2 = RunConfig(**{**asdict(cfg), "top_k": int(K), "gamma": float(g)})
                        met, *_ = run_single(
                            df, int(seed), cfg2, fairness_group_col="ses", postproc=PostProcConfig(enabled=False)
                        )
                        rows.append(met)
                        done += 1
                        prog.progress(done / max(total, 1))
                dfm = pd.DataFrame(rows)

                dfp = pareto_front(
                    dfm,
                    minimize_cols=["eopp_gap", "latency_ms", "expl_num_features", "expl_entropy"],
                    maximize_cols=["f1", "coverage"],
                )

                selected = mas_select_post_pareto(
                    dfp,
                    perf_min_auc=0.70,
                    fairness_max_eopp_gap=float(eopp),
                    explainability_max_features=int(max_feat),
                    explainability_max_entropy=float(max_ent),
                    cost_max_latency_ms=2000.0,
                )
                st.session_state["sweep"] = dict(dfm=dfm, dfp=dfp, selected=selected)
                status.update(label="Sweep complete.", state="complete")

        if "sweep" in st.session_state:
            dfm = st.session_state["sweep"]["dfm"]
            dfp = st.session_state["sweep"]["dfp"]
            selected = st.session_state["sweep"]["selected"]

            st.success("Sweep artifacts available below. Use Export to download a bundle.zip.")
            st.markdown("**MAS-selected candidate_id:** " + (f"`{selected}`" if selected else "`None (no feasible candidate)`"))

            with st.expander("Candidates (all)", expanded=False):
                st.dataframe(dfm.sort_values(["f1", "eopp_gap"], ascending=[False, True]), use_container_width=True)

            with st.expander("Pareto front", expanded=True):
                st.dataframe(dfp.sort_values(["f1", "eopp_gap"], ascending=[False, True]), use_container_width=True)

            st.markdown("#### Visual trade-off")
            fig = _plot_scatter(dfp, "eopp_gap", "f1", highlight_id=selected, title="Pareto (ΔEOpp vs F1)")
            st.pyplot(fig, clear_figure=True)

    with colB:
        st.markdown("### Output guide")
        st.info(
            "This tab runs in-memory and produces a downloadable `bundle.zip`.\n\n"
            "For paper-grade outputs (multi-seed, figures/tables, Overleaf drop-in, decision brief), "
            "run the CLI and then use the **Review** tab to inspect the `bundle.zip`."
        )

        st.markdown("### 3) Export (bundle.zip)")
        st.caption("Exports CSVs + plots + LaTeX tables + Decision Brief (if available).")

        if st.button("Build bundle.zip (demo)", use_container_width=True):
            if "sweep" not in st.session_state:
                st.warning("Run a Pareto sweep first.")
            else:
                dfm = st.session_state["sweep"]["dfm"]
                dfp = st.session_state["sweep"]["dfp"]
                selected = st.session_state["sweep"]["selected"]
                mas_df = dfm[dfm["candidate_id"] == selected].copy() if selected else pd.DataFrame()

                # Minimal baseline table (demo)
                baseline_rows = []
                best = dfm.sort_values("auc", ascending=False).iloc[0]
                baseline_rows.append(
                    dict(
                        method="Best candidate (single-seed demo)",
                        auc_mean=float(best["auc"]),
                        auc_std=0.0,
                        f1_mean=float(best["f1"]),
                        f1_std=0.0,
                        eopp_mean=float(best["eopp_gap"]),
                        eopp_std=0.0,
                        abst_mean=float(best.get("abstention_rate", 1.0 - best.get("coverage", 1.0))),
                        abst_std=0.0,
                    )
                )
                baselines_tex = latex_table_baselines(
                    pd.DataFrame(baseline_rows),
                    caption="Single-seed demo table (use CLI for multi-seed).",
                    label="tab:baselines_demo",
                )
                mas_tex = (
                    latex_table_mas_selected(mas_df, caption="MAS-selected policy (single-seed demo).", label="tab:mas_demo")
                    if not mas_df.empty
                    else ""
                )

                # Plots → bytes
                fig1 = _plot_scatter(dfm, "eopp_gap", "f1", highlight_id=selected, title="Candidates (ΔEOpp vs F1)")
                buf1 = io.BytesIO()
                fig1.savefig(buf1, format="png", bbox_inches="tight")
                plt.close(fig1)

                fig2 = plt.figure()
                ax2 = plt.gca()
                ax2.scatter(dfm["coverage"], dfm["f1"], label="candidates")
                ax2.set_xlabel("coverage")
                ax2.set_ylabel("f1")
                if _has_labels_for_legend(ax2):
                    ax2.legend()
                buf2 = io.BytesIO()
                fig2.savefig(buf2, format="png", bbox_inches="tight")
                plt.close(fig2)

                decision_brief_bytes = _make_decision_brief_demo(
                    data_source=data_source,
                    seed=int(seed),
                    cfg=cfg,
                    dfm=dfm,
                    dfp=dfp,
                    mas_df=mas_df,
                    fig_bytes_1=buf1.getvalue(),
                    fig_bytes_2=buf2.getvalue(),
                )

                bundle_bytes = _build_bundle_zip(
                    data_source=data_source,
                    seed=int(seed),
                    cfg=cfg,
                    selected=selected,
                    dfm=dfm,
                    dfp=dfp,
                    mas_df=mas_df,
                    baselines_tex=baselines_tex,
                    mas_tex=mas_tex,
                    plot1_bytes=buf1.getvalue(),
                    plot2_bytes=buf2.getvalue(),
                    decision_brief_bytes=decision_brief_bytes,
                )

                st.session_state["bundle"] = bundle_bytes
                st.success("bundle.zip is ready. Download below.")

        if "bundle" in st.session_state:
            st.download_button(
                "Download bundle.zip",
                data=st.session_state["bundle"],
                file_name="compass_ecl_mas_bundle.zip",
                mime="application/zip",
                use_container_width=True,
            )
            st.caption("Tip: upload this same file in the Review tab to inspect it here.")

# ============
# Review tab
# ============
with tab_review:
    st.subheader("Review (upload bundle.zip from a CLI run)")
    st.caption(
        "Upload `outputs/<run_id>/bundle.zip` from the CLI pipeline to inspect paper-grade artifacts "
        "(tables, figures, and the Decision Brief)."
    )

    bundle_file = st.file_uploader("Upload bundle.zip", type=["zip"])
    if not bundle_file:
        st.info("Upload a bundle.zip to review a run.")
    else:
        with zipfile.ZipFile(bundle_file, "r") as z:
            st.success("Bundle loaded.")

            with st.expander("Bundle contents (debug)", expanded=False):
                st.code("\n".join(sorted(z.namelist())), language="text")


            summary_bytes = _bundle_read_bytes(z, "summary.json")
            if summary_bytes:
                with st.expander("Run summary (summary.json)", expanded=False):
                    st.json(json.loads(summary_bytes.decode("utf-8")))

            col1, col2, col3 = st.columns(3)
            m_cand = _bundle_read_csv(z, "metrics_candidates.csv")
            p_front = _bundle_read_csv(z, "pareto_front.csv")
            m_sel = _bundle_read_csv(z, "mas_selected.csv")

            with col1:
                st.metric("Candidates", f"{len(m_cand):,}" if m_cand is not None else "—")
            with col2:
                st.metric("Pareto points", f"{len(p_front):,}" if p_front is not None else "—")
            with col3:
                st.metric("MAS selected", "yes" if (m_sel is not None and len(m_sel) > 0) else "no")

            st.divider()

            st.markdown("### Key artifacts")
            pdf_bytes = _bundle_read_bytes(z, "decision_brief.pdf")
            if pdf_bytes:
                st.markdown("#### Decision Brief")
                _render_pdf_download(pdf_bytes)
            else:
                st.info("No decision_brief.pdf found in this bundle.")

            st.markdown("#### Figures")
            fig_specs = [
                ("fig_f1_vs_eopp_color_coverage.png", "F1 vs ΔEOpp (color = coverage)"),
                ("fig_coverage_vs_f1_color_eopp.png", "Coverage vs F1 (color = ΔEOpp)"),
                ("pareto_perf_fair.png", "Pareto front: Performance × Fairness"),
                ("pareto_perf_exp.png", "Pareto front: Performance × Explainability"),
            ]
            found_any = False
            for fname, caption in fig_specs:
                b = _bundle_read_bytes(z, fname)
                if b:
                    st.image(b, caption=caption, use_container_width=True)
                    found_any = True
            if not found_any:
                st.info("No figures found in this bundle (expected under overleaf_dropin/figures/).")

            st.markdown("#### Tables (LaTeX)")
            tex_baselines = _bundle_read_bytes(z, "tab_baselines.tex")
            tex_mas = _bundle_read_bytes(z, "tab_mas_selected.tex")
            tex_fl = _bundle_read_bytes(z, "tab_fairlearn_reductions.tex")

            if tex_baselines:
                with st.expander("Baselines table (tab_baselines.tex)", expanded=False):
                    st.code(tex_baselines.decode("utf-8"), language="tex")
            if tex_mas:
                with st.expander("MAS-selected table (tab_mas_selected.tex)", expanded=False):
                    st.code(tex_mas.decode("utf-8"), language="tex")
            if tex_fl:
                with st.expander("Fairlearn reductions table (tab_fairlearn_reductions.tex)", expanded=False):
                    st.code(tex_fl.decode("utf-8"), language="tex")

            if not any([tex_baselines, tex_mas, tex_fl]):
                st.info("No LaTeX tables found in this bundle (expected under overleaf_dropin/paper_assets/).")

            st.divider()

            st.markdown("### Dataframes")
            if m_cand is not None:
                with st.expander("metrics_candidates.csv", expanded=False):
                    st.dataframe(m_cand, use_container_width=True)
            if p_front is not None:
                with st.expander("pareto_front.csv", expanded=False):
                    st.dataframe(p_front, use_container_width=True)
            if m_sel is not None:
                with st.expander("mas_selected.csv", expanded=False):
                    st.dataframe(m_sel, use_container_width=True)
