from __future__ import annotations
import numpy as np

def _tpr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pos = (y_true == 1)
    denom = max(int(pos.sum()), 1)
    return float(((y_pred == 1) & pos).sum() / denom)

def _fnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    pos = (y_true == 1)
    denom = max(int(pos.sum()), 1)
    return float(((y_pred == 0) & pos).sum() / denom)

def eopp_gap(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> float:
    groups = np.unique(group)
    tprs = []
    for g in groups:
        m = group == g
        if m.sum() == 0:
            continue
        tprs.append(_tpr(y_true[m], y_pred[m]))
    if len(tprs) <= 1:
        return 0.0
    return float(np.max(tprs) - np.min(tprs))

def fnr_gap(y_true: np.ndarray, y_pred: np.ndarray, group: np.ndarray) -> float:
    groups = np.unique(group)
    fnrs = []
    for g in groups:
        m = group == g
        if m.sum() == 0:
            continue
        fnrs.append(_fnr(y_true[m], y_pred[m]))
    if len(fnrs) <= 1:
        return 0.0
    return float(np.max(fnrs) - np.min(fnrs))
