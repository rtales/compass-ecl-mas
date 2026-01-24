from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from .models.baselines import build_model
from .metrics.performance import perf_metrics, ece
from .metrics.fairness import eopp_gap, fnr_gap
from .metrics.explainability import explanation_proxy
from .metrics.cost import cost_metrics
from .ecl.engine import ECLEngine
from .fairness_postproc import hardt_eopp_deterministic, apply_group_thresholds


def split_train_test(df: pd.DataFrame, seed: int, test_frac: float = 0.30):
    rng = np.random.default_rng(seed + 123)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_te = int(len(df) * test_frac)
    te = idx[:n_te]
    tr = idx[n_te:]
    return tr, te


@dataclass
class RunConfig:
    model: str
    top_k: int
    gamma: float
    expl_k: int
    forbidden_features: list[str]
    abstention_enabled: bool
    fairness_max_eopp_gap: float
    explainability_max_features: int
    explainability_max_entropy: float


@dataclass
class PostProcConfig:
    enabled: bool = False
    method: str = "hardt_eopp_deterministic"


def compute_group_stats(y_true, y_pred, y_prob, group) -> pd.DataFrame:
    rows = []
    for g in np.unique(group):
        m = group == g
        if m.sum() == 0:
            continue
        yt, yp, yp_prob = y_true[m], y_pred[m], y_prob[m]
        pos = (yt == 1)
        tpr = ((yp == 1) & pos).sum() / max(pos.sum(), 1)
        fnr = ((yp == 0) & pos).sum() / max(pos.sum(), 1)
        pr = (yp == 1).mean()
        base = yt.mean()
        rows.append(
            {
                "group": int(g),
                "n": int(m.sum()),
                "base_rate": float(base),
                "pred_pos_rate": float(pr),
                "tpr": float(tpr),
                "fnr": float(fnr),
            }
        )
    return pd.DataFrame(rows).sort_values("group")


def _make_estimator(base_model):
    """
    Wrap the base sklearn estimator with a median imputer so estimators that do not
    accept NaN (e.g., LogisticRegression) can still be trained on simulated data
    with missingness.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", base_model),
        ]
    )


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def run_single(
    df: pd.DataFrame,
    seed: int,
    cfg: RunConfig,
    fairness_group_col: str = "ses",
    postproc: PostProcConfig | None = None,
):
    tr, te = split_train_test(df, seed)

    forbidden = set(cfg.forbidden_features)
    feature_cols = [c for c in df.columns if c not in ({"y"} | forbidden)]

    Xtr = df.loc[tr, feature_cols]
    ytr = df.loc[tr, "y"].to_numpy()
    Xte = df.loc[te, feature_cols]
    yte = df.loc[te, "y"].to_numpy()
    group = df.loc[te, fairness_group_col].to_numpy()

    base_model = build_model(cfg.model)
    est = _make_estimator(base_model)

    # Fit (pipeline handles NaN via imputer)
    t0 = time.time()
    est.fit(Xtr, ytr)
    train_ms = (time.time() - t0) * 1000.0

    # Predict probabilities (again: pipeline handles NaN via imputer)
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(Xte)[:, 1]
    else:
        scores = est.decision_function(Xte)
        p = _sigmoid(scores)

    # Ensure p in [0,1]
    if p.min() < 0 or p.max() > 1:
        p = _sigmoid(p)

    # Rank-based policy: choose Top-K as positive
    order = np.argsort(-p)
    yhat = np.zeros_like(yte)
    yhat[order[: cfg.top_k]] = 1

    # ECL engine
    engine = ECLEngine(
        forbidden_features=cfg.forbidden_features,
        abstention_enabled=cfg.abstention_enabled,
        abstention_gamma=cfg.gamma,
        fairness_max_eopp_gap=cfg.fairness_max_eopp_gap,
        explainability_max_features=cfg.explainability_max_features,
        explainability_max_entropy=cfg.explainability_max_entropy,
    )

    abst = engine.abstain_mask(p)
    abst_rate = float(abst.mean())

    perf = perf_metrics(yte, yhat, p)

    # Explainability proxy: use the fitted base estimator and the imputed test matrix
    imputer = est.named_steps["imputer"]
    fitted_model = est.named_steps["model"]
    Xte_imp = imputer.transform(Xte)
    Xte_imp_df = pd.DataFrame(Xte_imp, columns=feature_cols)

    expl = explanation_proxy(fitted_model, Xte_imp_df, top_k=int(cfg.expl_k))

    cst = cost_metrics(train_ms=train_ms, n_samples=len(Xte))
    fair = float(eopp_gap(yte, yhat, group))

    feasible = engine.is_feasible(
        float(perf["auc"]),
        fair,
        float(expl["expl_num_features"]),
        float(expl["expl_entropy"]),
    )

    metrics = {
        "candidate_id": f"{cfg.model}_K{cfg.top_k}_g{cfg.gamma:.3f}_e{cfg.expl_k}",
        "model": cfg.model,
        "topk": int(cfg.top_k),
        "gamma": float(cfg.gamma),
        "expl_k": int(cfg.expl_k),
        "auc": float(perf["auc"]),
        "f1": float(perf["f1"]),
        "eopp_gap": fair,
        "fnr_gap": float(fnr_gap(yte, yhat, group)),
        "ece": float(ece(yte, p)),
        "expl_num_features": float(expl["expl_num_features"]),
        "expl_entropy": float(expl["expl_entropy"]),
        "latency_ms": float(cst["latency_ms"]),
        "train_ms": float(cst["train_ms"]),
        "abstention_rate": abst_rate,
        "coverage": float(1.0 - abst_rate),
        "decidable_n": int((~abst).sum()),
        "feasible_under_ecl": bool(feasible),
        "postproc_enabled": False,
    }
    audit = compute_group_stats(yte, yhat, p, group)

    # Optional post-processing baseline (EOpp)
    metrics_pp = None
    audit_pp = None
    if postproc and postproc.enabled:
        pp = hardt_eopp_deterministic(y_true=yte, scores=p, group=group)
        yhat_pp = apply_group_thresholds(scores=p, group=group, thresholds=pp["thresholds"])
        perf_pp = perf_metrics(yte, yhat_pp, p)

        metrics_pp = {
            "candidate_id": f"{cfg.model}_K{cfg.top_k}_g{cfg.gamma:.3f}_e{cfg.expl_k}_postproc",
            "model": cfg.model,
            "topk": int(cfg.top_k),
            "gamma": float(cfg.gamma),
            "expl_k": int(cfg.expl_k),
            "auc": float(perf_pp["auc"]),
            "f1": float(perf_pp["f1"]),
            "eopp_gap": float(eopp_gap(yte, yhat_pp, group)),
            "fnr_gap": float(fnr_gap(yte, yhat_pp, group)),
            "ece": float(ece(yte, p)),
            "expl_num_features": float(expl["expl_num_features"]),
            "expl_entropy": float(expl["expl_entropy"]),
            "latency_ms": float(cst["latency_ms"]),
            "train_ms": float(train_ms),
            "abstention_rate": 0.0,
            "coverage": 1.0,
            "decidable_n": int(len(yte)),
            "feasible_under_ecl": False,
            "postproc_enabled": True,
        }
        audit_pp = compute_group_stats(yte, yhat_pp, p, group)

    return metrics, audit, metrics_pp, audit_pp
