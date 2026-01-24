"""
Decision Brief PDF generator for Compass-ECL-MAS.

Creates a one–two page executive summary for public-sector decision makers:
- Recommended candidate (MAS-selected or best ECL-feasible fallback)
- Key metrics (performance, fairness, coverage, explainability, cost)
- Trade-off alternatives from Pareto front
- "Equity premium" (performance sacrificed for equity)
- Deployment guardrails + monitoring checklist

This module is intentionally dependency-light; it uses reportlab for PDF rendering.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image,
        PageBreak,
    )
except Exception as e:  # pragma: no cover
    raise ImportError(
        "reportlab is required to generate the Decision Brief PDF. "
        "Install it with: pip install reportlab"
    ) from e


@dataclass
class BriefInputs:
    run_id: str
    dataset_desc: str
    dataset_size_str: str
    seeds: list[int]
    cfg: dict[str, Any]
    out_dir: Path


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None


def _pick_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    return df.iloc[0]


def _fmt(x: Any, nd: int = 3) -> str:
    if x is None:
        return "—"
    try:
        if pd.isna(x):
            return "—"
    except Exception:
        pass
    try:
        if isinstance(x, (int,)):
            return f"{x:,}"
        if isinstance(x, (float,)):
            return f"{x:.{nd}f}"
    except Exception:
        return str(x)
    return str(x)


def _get_col(df: pd.DataFrame, *names: str) -> Optional[str]:
    if df is None:
        return None
    cols = set(df.columns)
    for n in names:
        if n in cols:
            return n
    return None


def _aggregate_best_baseline(df_baselines: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    """
    baselines_no_ecl.csv can be seed-level; aggregate by model to mean metrics,
    then pick the best by AUC (fallback: F1).
    """
    if df_baselines is None or df_baselines.empty:
        return None
    if "model" not in df_baselines.columns:
        return None

    auc_col = _get_col(df_baselines, "auc", "auc_mean")
    f1_col = _get_col(df_baselines, "f1", "f1_mean")
    eop_col = _get_col(df_baselines, "eopp_gap", "eopp_gap_mean")
    cov_col = _get_col(df_baselines, "coverage", "coverage_mean")

    grp = df_baselines.groupby("model", dropna=False)
    agg = grp[[c for c in [auc_col, f1_col, eop_col, cov_col] if c is not None]].mean(numeric_only=True)
    agg = agg.reset_index()

    if auc_col and auc_col in agg.columns:
        best = agg.loc[agg[auc_col].idxmax()]
    elif f1_col and f1_col in agg.columns:
        best = agg.loc[agg[f1_col].idxmax()]
    else:
        return None
    return best


def _pareto_alternatives(df_pf: Optional[pd.DataFrame]) -> list[pd.Series]:
    """
    Return up to 3 distinct alternatives from the Pareto front:
    - Highest performance (AUC then F1)
    - Lowest fairness gap
    - Highest coverage
    """
    if df_pf is None or df_pf.empty:
        return []

    auc_col = _get_col(df_pf, "auc_mean", "auc")
    f1_col = _get_col(df_pf, "f1_mean", "f1")
    eop_col = _get_col(df_pf, "eopp_gap_mean", "eopp_gap")
    cov_col = _get_col(df_pf, "coverage_mean", "coverage")

    candidates = []
    used = set()

    def add_best(idx):
        if idx is None:
            return
        row = df_pf.loc[idx]
        cid_col = _get_col(df_pf, "candidate_id", "id")
        cid = str(row[cid_col]) if cid_col else str(idx)
        if cid in used:
            return
        used.add(cid)
        candidates.append(row)

    if auc_col:
        add_best(df_pf[auc_col].idxmax())
    elif f1_col:
        add_best(df_pf[f1_col].idxmax())

    if eop_col:
        add_best(df_pf[eop_col].idxmin())

    if cov_col:
        add_best(df_pf[cov_col].idxmax())

    return candidates[:3]


def generate_decision_brief_pdf(inputs: BriefInputs) -> Path:
    out_dir = inputs.out_dir
    pdf_path = out_dir / "decision_brief.pdf"

    # Input CSVs
    df_mas = _safe_read_csv(out_dir / "mas_selected.csv")
    df_best = _safe_read_csv(out_dir / "best_ecl_no_mas.csv")
    df_pf = _safe_read_csv(out_dir / "pareto_front.csv")
    df_base = _safe_read_csv(out_dir / "baselines_no_ecl.csv")
    df_post = _safe_read_csv(out_dir / "fairness_postproc.csv")

    # IMPORTANT: do not use Python truthiness on a pandas Series (it is ambiguous).
    selected = _pick_row(df_mas)
    if selected is None:
        selected = _pick_row(df_best)
    selected_kind = "MAS-selected" if (df_mas is not None and not df_mas.empty) else ("Best ECL-feasible (no MAS)" if (df_best is not None and not df_best.empty) else "No ECL-feasible candidate")

    # Column mapping (handle both seed-level and aggregated outputs)
    auc_col = _get_col(df_mas if df_mas is not None and not df_mas.empty else df_best, "auc_mean", "auc")
    f1_col  = _get_col(df_mas if df_mas is not None and not df_mas.empty else df_best, "f1_mean", "f1")
    eop_col = _get_col(df_mas if df_mas is not None and not df_mas.empty else df_best, "eopp_gap_mean", "eopp_gap")
    cov_col = _get_col(df_mas if df_mas is not None and not df_mas.empty else df_best, "coverage_mean", "coverage")
    ent_col = _get_col(df_mas if df_mas is not None and not df_mas.empty else df_best, "expl_entropy_mean", "expl_entropy")
    nf_col  = _get_col(df_mas if df_mas is not None and not df_mas.empty else df_best, "expl_num_features_mean", "expl_num_features")
    lat_col = _get_col(df_mas if df_mas is not None and not df_mas.empty else df_best, "latency_ms_mean", "latency_ms")

    # Baseline best
    best_base = _aggregate_best_baseline(df_base)

    # Pareto alternatives
    alts = _pareto_alternatives(df_pf)

    # ECL constraints (for display)
    ecl = inputs.cfg.get("ecl", {}) if inputs.cfg else {}
    fairness_tau = ecl.get("fairness_max_eopp_gap", None)
    ent_max = ecl.get("explainability_max_entropy", None)
    nf_max = ecl.get("explainability_max_features", None)
    forb = ecl.get("forbidden_features", [])
    sens = ecl.get("fairness_primary_attribute", "ses")
    abst = ecl.get("abstention_enabled", True)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="H1", parent=styles["Heading1"], fontSize=16, spaceAfter=10))
    styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], fontSize=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="Body", parent=styles["BodyText"], fontSize=10, leading=13))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=12, textColor=colors.grey))

    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, leftMargin=0.75*inch, rightMargin=0.75*inch, topMargin=0.7*inch, bottomMargin=0.7*inch)
    story = []

    story.append(Paragraph("Compass-ECL-MAS — Decision Brief", styles["H1"]))
    story.append(Paragraph(f"<b>Run:</b> {inputs.run_id}", styles["Body"]))
    story.append(Paragraph(f"<b>Dataset:</b> {inputs.dataset_desc} ({inputs.dataset_size_str}; mean±std over {len(inputs.seeds)} seed(s))", styles["Body"]))
    story.append(Spacer(1, 10))

    # Recommendation box
    story.append(Paragraph("1) Recommendation", styles["H2"]))
    if selected is None:
        story.append(Paragraph(
            "No candidate satisfied the Ethical Constraint Layer (ECL) under the current thresholds. "
            "Recommendation: do not automate decisions; use human review and revisit constraints / representation before deployment.",
            styles["Body"]
        ))
    else:
        cid_col = _get_col(df_mas if df_mas is not None and not df_mas.empty else df_best, "candidate_id", "id")
        cid = str(selected[cid_col]) if cid_col else "selected_candidate"
        headline = f"<b>{selected_kind}:</b> {cid}"
        story.append(Paragraph(headline, styles["Body"]))

        # Metrics table
        data = [
            ["Metric", "Value"],
            ["AUC", _fmt(selected[auc_col]) if auc_col else "—"],
            ["F1", _fmt(selected[f1_col]) if f1_col else "—"],
            [f"ΔEOpp (audit on {sens})", _fmt(selected[eop_col]) if eop_col else "—"],
            ["Coverage (automation rate)", _fmt(selected[cov_col]) if cov_col else "—"],
            ["Explainability entropy", _fmt(selected[ent_col]) if ent_col else "—"],
            ["#Features in explanation", _fmt(selected[nf_col], nd=0) if nf_col else "—"],
            ["Latency (ms, proxy)", _fmt(selected[lat_col]) if lat_col else "—"],
        ]
        t = Table(data, colWidths=[2.5*inch, 3.0*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
            ("TEXTCOLOR", (0,0), (-1,0), colors.black),
            ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTNAME", (0,1), (-1,-1), "Helvetica"),
            ("FONTSIZE", (0,0), (-1,-1), 9),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(t)

        # Equity premium
        story.append(Spacer(1, 8))
        story.append(Paragraph("Equity premium (performance sacrificed for equity)", styles["H2"]))
        if best_base is None:
            story.append(Paragraph("Baseline results not available; equity premium could not be computed.", styles["Body"]))
        else:
            b_auc_col = _get_col(best_base.to_frame().T, "auc", "auc_mean")
            b_f1_col  = _get_col(best_base.to_frame().T, "f1", "f1_mean")
            b_eop_col = _get_col(best_base.to_frame().T, "eopp_gap", "eopp_gap_mean")
            prem_auc = (best_base[b_auc_col] - selected[auc_col]) if (auc_col and b_auc_col) else None
            prem_f1  = (best_base[b_f1_col]  - selected[f1_col])  if (f1_col and b_f1_col) else None
            gain_eop = (best_base[b_eop_col] - selected[eop_col]) if (eop_col and b_eop_col) else None
            model = str(best_base["model"]) if "model" in best_base.index else "best baseline"

            story.append(Paragraph(
                f"Compared to the best unconstrained baseline (<i>{model}</i>), the selected policy achieves "
                f"a reduction of ΔEOpp by {_fmt(gain_eop)} at a cost of ΔAUC={_fmt(prem_auc)} and ΔF1={_fmt(prem_f1)}.",
                styles["Body"]
            ))

    story.append(Spacer(1, 10))

    # Guardrails
    story.append(Paragraph("2) Governance guardrails (ECL)", styles["H2"]))
    bullets = []
    if fairness_tau is not None:
        bullets.append(f"Fairness: ΔEOpp ≤ {fairness_tau} (audit-only group: {sens}; sensitive attribute excluded from training/inference).")
    if ent_max is not None:
        bullets.append(f"Explainability: explanation entropy ≤ {ent_max}.")
    if nf_max is not None:
        bullets.append(f"Explainability: number of features in explanation ≤ {nf_max}.")
    if forb:
        bullets.append(f"Forbidden predictive features: {', '.join(map(str, forb))}.")
    bullets.append(f"Abstention enabled: {'yes' if abst else 'no'} (coverage controls automation budget).")

    story.append(Paragraph("<br/>".join([f"• {b}" for b in bullets]), styles["Body"]))
    story.append(Spacer(1, 8))

    # Alternatives
    story.append(Paragraph("3) Alternatives on the Pareto frontier", styles["H2"]))
    if not alts:
        story.append(Paragraph("Pareto frontier not available.", styles["Body"]))
    else:
        # build table
        pf = df_pf
        cid_col = _get_col(pf, "candidate_id", "id") or "candidate_id"
        auc_pf = _get_col(pf, "auc_mean", "auc")
        f1_pf  = _get_col(pf, "f1_mean", "f1")
        eop_pf = _get_col(pf, "eopp_gap_mean", "eopp_gap")
        cov_pf = _get_col(pf, "coverage_mean", "coverage")

        rows = [["Option", "candidate_id", "AUC", "F1", "ΔEOpp", "Coverage"]]
        labels = ["High performance", "Low disparity", "High automation"]
        for i, row in enumerate(alts):
            rows.append([
                labels[i] if i < len(labels) else f"Option {i+1}",
                str(row[cid_col]) if cid_col in pf.columns else "—",
                _fmt(row[auc_pf]) if auc_pf else "—",
                _fmt(row[f1_pf]) if f1_pf else "—",
                _fmt(row[eop_pf]) if eop_pf else "—",
                _fmt(row[cov_pf]) if cov_pf else "—",
            ])

        t2 = Table(rows, colWidths=[1.2*inch, 2.2*inch, 0.7*inch, 0.7*inch, 0.8*inch, 0.8*inch])
        t2.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.whitesmoke),
            ("GRID", (0,0), (-1,-1), 0.25, colors.lightgrey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE", (0,0), (-1,-1), 8.5),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(t2)

    story.append(Spacer(1, 10))

    # Monitoring checklist
    story.append(Paragraph("4) Practical deployment checklist", styles["H2"]))
    checklist = [
        "Run a time-bounded pilot (e.g., 4–8 weeks) before scaling.",
        "Monitor AUC/F1 and ΔEOpp monthly; report coverage and abstention volume to quantify the automation budget.",
        "Define rollback triggers (e.g., ΔEOpp above threshold for 2 consecutive periods, or performance drop beyond a preset margin).",
        "Document decision pathways: when the system abstains, what human workflow applies, and how appeals are handled.",
        "Re-evaluate constraints and thresholds with stakeholders (education experts, legal/compliance, and equity representatives).",
    ]
    story.append(Paragraph("<br/>".join([f"• {c}" for c in checklist]), styles["Body"]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "Note: this brief is generated automatically from run artifacts (CSVs/figures) to support auditing and governance.",
        styles["Small"]
    ))

    # Include a key figure if present
    figs_dir = out_dir / "overleaf_dropin" / "figures"
    preferred = [
        "pareto_perf_fair.png",
        "fig_f1_vs_eopp_color_coverage.png",
        "fig_coverage_vs_f1_color_eopp.png",
    ]
    fig_path = None
    for name in preferred:
        p = figs_dir / name
        if p.exists():
            fig_path = p
            break
    if fig_path is not None:
        story.append(PageBreak())
        story.append(Paragraph("Appendix: Key trade-off visualization", styles["H2"]))
        try:
            img = Image(str(fig_path))
            img._restrictSize(7.0*inch, 8.5*inch)
            story.append(img)
        except Exception:
            # If image cannot be loaded, ignore
            pass

    doc.build(story)
    return pdf_path
