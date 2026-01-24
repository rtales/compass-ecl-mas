from __future__ import annotations
import numpy as np

class ECLEngine:
    def __init__(self, forbidden_features, abstention_enabled, abstention_gamma, fairness_max_eopp_gap, explainability_max_features, explainability_max_entropy):
        self.forbidden_features = list(forbidden_features)
        self.abstention_enabled = bool(abstention_enabled)
        self.abstention_gamma = float(abstention_gamma)
        self.fairness_max_eopp_gap = float(fairness_max_eopp_gap)
        self.explainability_max_features = int(explainability_max_features)
        self.explainability_max_entropy = float(explainability_max_entropy)

    def abstain_mask(self, prob: np.ndarray) -> np.ndarray:
        if not self.abstention_enabled:
            return np.zeros_like(prob, dtype=bool)
        return (np.abs(prob - 0.5) < self.abstention_gamma)

    def is_feasible(self, auc: float, eopp_gap: float, expl_num_features: float, expl_entropy: float) -> bool:
        return (
            (float(eopp_gap) <= self.fairness_max_eopp_gap)
            and (float(expl_num_features) <= self.explainability_max_features)
            and (float(expl_entropy) <= self.explainability_max_entropy)
        )
