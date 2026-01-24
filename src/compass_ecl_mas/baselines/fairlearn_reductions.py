from __future__ import annotations

"""Fairlearn reductions baseline (constrained optimization) for Equal Opportunity.

This module is intentionally isolated so it can be used as an *additional baseline*
without affecting the core ECL→Pareto→MAS pipeline.

We follow the 'audit-only sensitive feature' policy:
- sensitive attribute (e.g., SES) is NOT included in the predictive feature matrix X
- it is provided only as `sensitive_features` to enforce the fairness constraint
"""

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..pipeline import split_train_test
from ..metrics.performance import perf_metrics
from ..metrics.fairness import eopp_gap


class _PipelineSampleWeightWrapper(BaseEstimator, ClassifierMixin):
    """Adapter to make sklearn Pipeline compatible with estimators expecting `fit(..., sample_weight=...)`.

    Fairlearn's reductions algorithms may pass `sample_weight` to the base estimator's `fit`.
    sklearn Pipeline does not accept `sample_weight` directly; it must be routed to the final step
    using the `stepname__sample_weight` convention.
    """

    def __init__(self, pipeline: Pipeline, final_step_name: str = "clf"):
        self.pipeline = pipeline
        self.final_step_name = final_step_name

    def fit(self, X, y, sample_weight=None):
        if sample_weight is None:
            self.pipeline.fit(X, y)
        else:
            self.pipeline.fit(X, y, **{f"{self.final_step_name}__sample_weight": sample_weight})
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def predict_proba(self, X):
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X)
        raise AttributeError("Base pipeline does not support predict_proba")

    def get_params(self, deep=True):
        # Delegate so sklearn.clone works
        return {"pipeline": self.pipeline, "final_step_name": self.final_step_name}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

@dataclass
class FairlearnReductionsConfig:
    method: str = "exponentiated_gradient"
    base_estimator: str = "lr"
    eps_grid: tuple[float, ...] = (0.00, 0.02, 0.05, 0.10, 0.15, 0.20)
    max_iter: int = 50


def _build_base_estimator(name: str, seed: int):
    """Return a sklearn-compatible estimator for Fairlearn.

    We wrap the classifier in a small preprocessing pipeline so the baseline is robust:
    - SimpleImputer handles any missing values (NaN) deterministically
    - StandardScaler stabilizes LR optimization (dense inputs)
    """
    name = str(name).lower()
    if name in {"lr", "logreg", "logistic_regression"}:
        clf = LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            n_jobs=None,
            random_state=seed,
        )
        pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", clf),
            ]
        )
        return _PipelineSampleWeightWrapper(pipe, final_step_name="clf")
    raise ValueError(f"Unsupported Fairlearn base_estimator: {name}")


def _safe_probabilities(mitigator, Xte) -> Optional[np.ndarray]:
    """Try to obtain a probability-like score in [0,1] for ranking/top-k.

    Fairlearn mitigators may or may not implement predict_proba. When unavailable,
    we fall back to binary predictions as a degenerate score.
    """
    if hasattr(mitigator, "predict_proba"):
        try:
            p = mitigator.predict_proba(Xte)[:, 1]
            p = np.asarray(p, dtype=float)
            return np.clip(p, 0.0, 1.0)
        except Exception:
            pass
    # Fall back to predictions (0/1)
    try:
        yhat = mitigator.predict(Xte)
        p = np.asarray(yhat, dtype=float)
        return np.clip(p, 0.0, 1.0)
    except Exception:
        return None


def run_fairlearn_reductions_baseline(
    df: pd.DataFrame,
    seed: int,
    *,
    fairness_group_col: str = "ses",
    forbidden_features: Iterable[str] = ("ses", "sex"),
    top_k: Optional[int] = None,
    cfg: Optional[FairlearnReductionsConfig] = None,
) -> pd.DataFrame:
    """Run Fairlearn reductions baseline and return per-eps metrics.

    Parameters
    ----------
    df:
        Dataset with label column 'y' and sensitive attribute column.
    seed:
        Random seed (used for split and estimator).
    fairness_group_col:
        Sensitive attribute column to use for constraints and audit.
    forbidden_features:
        Columns excluded from predictive X (audit-only).
    top_k:
        If provided, we convert scores to decisions by selecting the top_k as positive.
        If None, we use the mitigator's default prediction.
    cfg:
        FairlearnReductionsConfig.

    Returns
    -------
    DataFrame with columns:
      seed, eps, f1, auc, eopp_gap, abstention_rate, coverage, model
    """
    cfg = cfg or FairlearnReductionsConfig()

    tr, te = split_train_test(df, seed)

    forbidden = set(forbidden_features)
    feature_cols = [c for c in df.columns if c not in (set(["y"]) | forbidden)]
    Xtr = df.loc[tr, feature_cols]
    ytr = df.loc[tr, "y"].to_numpy()
    Xte = df.loc[te, feature_cols]
    yte = df.loc[te, "y"].to_numpy()

    
    # Ensure numeric matrices for sklearn/fairlearn; coerce unexpected dtypes to NaN
    # (then impute inside the estimator pipeline).
    Xtr = Xtr.apply(pd.to_numeric, errors="coerce")
    Xte = Xte.apply(pd.to_numeric, errors="coerce")

    s_tr = df.loc[tr, fairness_group_col].to_numpy()
    s_te = df.loc[te, fairness_group_col].to_numpy()

    # Import fairlearn lazily (optional dependency)
    try:
        from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity
    except Exception as e:
        raise ImportError(
            "fairlearn is required for the reductions baseline. Install with: pip install '.[fairlearn]'\n"
            "(or pip install fairlearn)"
        ) from e

    # In fairlearn>=0.5, the constraint RHS is specified on the Moment via `difference_bound`.
    # We interpret each `eps` in `eps_grid` as the allowed Equal Opportunity (TPR) gap.

    rows = []
    for eps in cfg.eps_grid:
        # Re-seed numpy (and python random if used) each iteration for stability.
        # Fairlearn's ExponentiatedGradient does not consistently expose a `random_state`
        # parameter across versions, so we keep determinism by seeding the global RNGs and
        # ensuring the base estimator is seeded.
        np.random.seed(int(seed))
        try:
            import random as _py_random

            _py_random.seed(int(seed))
        except Exception:
            pass

        constraint = TruePositiveRateParity(difference_bound=float(eps))
        base = _build_base_estimator(cfg.base_estimator, seed)
        mitigator = ExponentiatedGradient(
            estimator=base,
            constraints=constraint,
            eps=0.01,  # optimization tolerance (NOT the fairness bound)
            max_iter=int(cfg.max_iter),
        )
        mitigator.fit(Xtr, ytr, sensitive_features=s_tr)

        # Score-like probabilities for ranking / top-k decisions
        p = _safe_probabilities(mitigator, Xte)

        if p is None:
            # If we cannot score, fall back to direct predictions
            yhat = np.asarray(mitigator.predict(Xte), dtype=int)
            perf = {"auc": float("nan"), "f1": float(f1_score(yte, yhat))}
        else:
            if top_k is None:
                yhat = (p >= 0.5).astype(int)
            else:
                order = np.argsort(-p)
                yhat = np.zeros_like(yte)
                yhat[order[: int(top_k)]] = 1

            perf = perf_metrics(yte, yhat, p)

        rows.append(
            {
                "seed": int(seed),
                "eps": float(eps),
                "auc": float(perf.get("auc", float("nan"))),
                "f1": float(perf.get("f1", float("nan"))),
                "eopp_gap": float(eopp_gap(yte, yhat, s_te)),
                "abstention_rate": 0.0,
                "coverage": 1.0,
                "model": "fairlearn_eg_lr",
            }
        )

    return pd.DataFrame(rows)
