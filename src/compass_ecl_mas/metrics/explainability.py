from __future__ import annotations
import numpy as np
import pandas as pd

def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.abs(v).astype(float)
    s = v.sum()
    if s <= 0:
        return np.ones_like(v) / len(v)
    return v / s

def explanation_proxy(model, X: pd.DataFrame, top_k: int = 8) -> dict:
    if hasattr(model, "coef_"):
        imp = np.abs(model.coef_).reshape(-1)
    elif hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_).reshape(-1)
    else:
        imp = np.ones(X.shape[1], dtype=float)

    k = min(int(top_k), len(imp))
    idx = np.argsort(-np.abs(imp))[:k]
    top = _normalize(imp[idx])
    entropy = float(-np.sum(top * np.log(top + 1e-12)))

    return {
        "expl_num_features": float(k),
        "expl_entropy": float(entropy),
        "expl_top_features": [str(X.columns[i]) for i in idx],
    }
