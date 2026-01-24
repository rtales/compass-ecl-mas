from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

def perf_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {"auc": float(roc_auc_score(y_true, y_prob)), "f1": float(f1_score(y_true, y_pred))}

def ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y_true = y_true.astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        m = (y_prob >= lo) & (y_prob < hi)
        if m.sum() == 0:
            continue
        acc = y_true[m].mean()
        conf = y_prob[m].mean()
        e += (m.sum() / len(y_true)) * abs(acc - conf)
    return float(e)
