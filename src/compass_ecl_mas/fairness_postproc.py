from __future__ import annotations
import numpy as np

def hardt_eopp_deterministic(y_true: np.ndarray, scores: np.ndarray, group: np.ndarray) -> dict:
    y_true = y_true.astype(int)
    groups = np.unique(group)

    def tprs_at_threshold(t):
        out = []
        for g in groups:
            m = (group == g) & (y_true == 1)
            if m.sum() == 0:
                out.append(0.0)
            else:
                out.append(float(((scores >= t) & m).sum() / m.sum()))
        return out

    base_tprs = tprs_at_threshold(0.5)
    target = float(np.min(base_tprs))  # equalize downward (conservative and feasible)

    thresholds = {}
    for g in groups:
        mpos = (group == g) & (y_true == 1)
        if mpos.sum() == 0:
            thresholds[int(g)] = 1.0
            continue
        s = np.sort(scores[mpos])[::-1]
        k = int(np.clip(round(target * len(s)), 0, len(s) - 1))
        thresholds[int(g)] = float(s[k]) if len(s) else 1.0

    return {"thresholds": thresholds, "target_tpr": target}

def apply_group_thresholds(scores: np.ndarray, group: np.ndarray, thresholds: dict[int, float]) -> np.ndarray:
    yhat = np.zeros_like(scores, dtype=int)
    for g, thr in thresholds.items():
        m = group == g
        yhat[m] = (scores[m] >= float(thr)).astype(int)
    return yhat
