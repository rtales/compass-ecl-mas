from __future__ import annotations
from pathlib import Path
import json, zipfile
import pandas as pd

def _pm(mean: float, std: float, nd: int = 3) -> str:
    return f"{mean:.{nd}f}\\pm{std:.{nd}f}"

def latex_table_baselines(df: pd.DataFrame, caption: str, label: str = "tab:baselines") -> str:
    """
    Expected columns:
      method, auc_mean, auc_std, f1_mean, f1_std, eopp_mean, eopp_std, abst_mean, abst_std
    """
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Method & AUC $\uparrow$ & F1 $\uparrow$ & $\Delta_{\text{EOpp}}$ $\downarrow$ & Abstain \% $\downarrow$ \\")
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        auc = _pm(float(r["auc_mean"]), float(r["auc_std"]), 3)
        f1 = _pm(float(r["f1_mean"]), float(r["f1_std"]), 3)
        eopp = _pm(float(r["eopp_mean"]), float(r["eopp_std"]), 3)
        abst = _pm(float(r["abst_mean"]) * 100.0, float(r["abst_std"]) * 100.0, 1)
        lines.append(f"{r['method']} & {auc} & {f1} & {eopp} & {abst} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

def latex_table_mas_selected(df: pd.DataFrame, caption: str, label: str = "tab:mas_selected") -> str:
    """
    Expected columns:
      candidate, auc, eopp_gap, expl_num_features, expl_entropy
    """
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"Candidate & AUC $\uparrow$ & EOpp gap $\downarrow$ & Expl.\ #feat $\downarrow$ & Expl.\ entropy $\downarrow$ \\")
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        lines.append(
            f"{r['candidate']} & {float(r['auc']):.3f} & {float(r['eopp_gap']):.3f} & {float(r['expl_num_features']):.0f} & {float(r['expl_entropy']):.3f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def latex_table_fairlearn_reductions(df: pd.DataFrame, caption: str, label: str = "tab:fairlearn_reductions") -> str:
    """Table for Fairlearn reductions trade-off curve over epsilon.

    Expected columns:
      eps, auc_mean, auc_std, f1_mean, f1_std, eopp_gap_mean, eopp_gap_std
    """
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{" + caption + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\begin{tabular}{cccc}")
    lines.append(r"\toprule")
    lines.append(r"$\epsilon$ & AUC $\uparrow$ & F1 $\uparrow$ & $\Delta_{\text{EOpp}}\downarrow$ \\")
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        eps = float(r["eps"])
        auc = _pm(float(r["auc_mean"]), float(r["auc_std"]), 3)
        f1 = _pm(float(r["f1_mean"]), float(r["f1_std"]), 3)
        eopp = _pm(float(r["eopp_gap_mean"]), float(r["eopp_gap_std"]), 3)
        lines.append(f"{eps:.2f} & {auc} & {f1} & {eopp} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)

def build_bundle(out_dir: Path, summary: dict, csvs: dict[str, pd.DataFrame], files: dict[str, Path], latex: dict[str, str]) -> Path:
    """
    Creates outputs/<run_id>/bundle.zip with:
      - summary.json
      - csv/*.csv
      - overleaf_dropin/figures/*
      - overleaf_dropin/paper_assets/*.tex
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / "bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("summary.json", json.dumps(summary, indent=2))
        for name, df in csvs.items():
            z.writestr(f"csv/{name}.csv", df.to_csv(index=False))
        for arcname, path in files.items():
            z.write(path, arcname)
        for name, tex in latex.items():
            z.writestr(f"overleaf_dropin/paper_assets/{name}.tex", tex)
    return bundle
