from __future__ import annotations
import pandas as pd

def pareto_front(df: pd.DataFrame, minimize_cols: list[str], maximize_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    values = df.reset_index(drop=True)
    keep = []
    for i in range(len(values)):
        a = values.iloc[i]
        dominated = False
        for j in range(len(values)):
            if i == j:
                continue
            b = values.iloc[j]
            if all(b[c] <= a[c] for c in minimize_cols) and all(b[c] >= a[c] for c in maximize_cols):
                if any(b[c] < a[c] for c in minimize_cols) or any(b[c] > a[c] for c in maximize_cols):
                    dominated = True
                    break
        if not dominated:
            keep.append(i)
    return values.iloc[keep].copy()
