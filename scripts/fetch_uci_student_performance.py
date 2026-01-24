#!/usr/bin/env python3
"""Fetch and preprocess the UCI Student Performance dataset (UCI id=320).

Produces a CSV compatible with Compass-ECL-MAS:

- Numeric feature columns (one-hot encoded categoricals)
- ses (audit-only sensitive attribute)
- y   (binary target)

Defaults:
- y = 1 if final grade G3 < 10 ("at risk"), else 0
- ses derived from parental education proxy (Medu+Fedu), binarized at median
- Drops G1,G2,G3 from predictive features to avoid leakage

Usage:
  pip install ucimlrepo
  python fetch_uci_student_performance.py --out data/uci_student/uci_student_processed.csv
"""

import argparse
import os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--y-threshold", type=int, default=10,
                    help="G3 threshold for at-risk (y=1 if G3 < threshold)")
    ap.add_argument("--ses-quantile", type=float, default=0.5,
                    help="Quantile split for SES (default median)")
    ap.add_argument("--drop-leakage", action="store_true", default=True,
                    help="Drop G1/G2/G3 from predictive features (default True)")
    args = ap.parse_args()

    try:
        from ucimlrepo import fetch_ucirepo
    except Exception as e:
        raise SystemExit(
            "Missing dependency 'ucimlrepo'. Install with: pip install ucimlrepo\n"
            f"Original error: {e}"
        )

    ds = fetch_ucirepo(id=320)
    X = ds.data.features.copy()
    T = ds.data.targets.copy()

    df = pd.concat([X, T], axis=1)

    # Required columns
    for col in ("G3", "Medu", "Fedu"):
        if col not in df.columns:
            raise SystemExit(
                f"Expected column '{col}' not found. Columns (sample): {list(df.columns)[:30]}"
            )

    # Target
    df["y"] = (df["G3"].astype(float) < float(args.y_threshold)).astype(int)

    # SES (audit-only)
    ses_index = df["Medu"].astype(float) + df["Fedu"].astype(float)
    cut = float(ses_index.quantile(args.ses_quantile))
    df["ses"] = (ses_index <= cut).astype(int)  # 1 = lower-SES group

    # Predictive features
    drop_cols = ["y"]
    if args.drop_leakage:
        for c in ("G1", "G2", "G3"):
            if c in df.columns:
                drop_cols.append(c)

    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # IMPORTANT: do NOT include ses as predictive feature (audit-only)
    if "ses" in feature_df.columns:
        feature_df = feature_df.drop(columns=["ses"])

    X_enc = pd.get_dummies(feature_df, drop_first=True)

    out_df = pd.concat([X_enc, df[["ses", "y"]]], axis=1)

    # Guardrail: remove duplicate columns if any (e.g., upstream schema changes)
    out_df = out_df.loc[:, ~out_df.columns.duplicated()]

    # Force numeric dtypes (median imputer friendly)
    for c in out_df.columns:
        out_df[c] = pd.to_numeric(out_df[c], errors="coerce")

    # Sanity checks
    if not out_df["y"].dropna().isin([0, 1]).all():
        raise SystemExit("Column y is not binary after processing.")
    if not out_df["ses"].dropna().isin([0, 1]).all():
        raise SystemExit("Column ses is not binary after processing.")

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print(f"Wrote: {out_path}")
    print(f"Shape: {out_df.shape} (features={out_df.shape[1]-2} + ses + y)")
    print(f"y prevalence:   {out_df['y'].mean():.3f}")
    print(f"ses prevalence: {out_df['ses'].mean():.3f}")

if __name__ == "__main__":
    main()
