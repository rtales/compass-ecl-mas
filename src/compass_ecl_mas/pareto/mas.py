from __future__ import annotations
import pandas as pd

def mas_select_post_pareto(candidates: pd.DataFrame, pareto: pd.DataFrame, perf_min_auc: float, fairness_max_eopp_gap: float,
                          explainability_max_features: int, explainability_max_entropy: float, cost_max_latency_ms: float) -> str | None:
    if pareto.empty:
        return None
    feasible = pareto[
        (pareto["auc"] >= float(perf_min_auc)) &
        (pareto["eopp_gap"] <= float(fairness_max_eopp_gap)) &
        (pareto["expl_num_features"] <= float(explainability_max_features)) &
        (pareto["expl_entropy"] <= float(explainability_max_entropy)) &
        (pareto["latency_ms"] <= float(cost_max_latency_ms))
    ].copy()
    if feasible.empty:
        return None
    feasible = feasible.sort_values(by=["eopp_gap", "auc", "expl_entropy", "latency_ms"], ascending=[True, False, True, True])
    return str(feasible.iloc[0]["candidate_id"])
