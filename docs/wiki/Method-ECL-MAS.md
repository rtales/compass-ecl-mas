# Method: ECL → Pareto → MAS

This project is a **systems-style** contribution: we treat “ethical governance” as a *constraint and selection layer* around standard ML models.

## 1) Candidates
We generate candidate models / policies by varying:
- model family or hyperparameters (LR / RF / GBDT, etc.)
- abstention parameter(s) (coverage vs. deferral-to-human-review)
- constraint thresholds (fairness, explainability proxy, cost proxy)

Each candidate yields a vector of metrics (e.g., F1, ΔEOpp, coverage, explainability proxy, cost proxy).

## 2) Ethical Constraint Layer (ECL)
ECL encodes *quantified guardrails* inspired by Compass’s six principles (e.g., dignity → abstention/human review, transparency → explainability budget, harmony → group parity constraints).

ECL classifies candidates as:
- **feasible** (all constraints satisfied)
- **infeasible** (one or more constraints violated)

## 3) Pareto frontier
Among candidates, we compute the **Pareto-efficient set**: points where improving one objective necessarily worsens another.

This makes the “price” of ethical constraints visible:
- you can quantify the performance you sacrifice to reduce unfairness
- you can quantify the coverage you sacrifice to reduce risk

## 4) Moral Alignment Score (MAS)
MAS is a **selection rule** over Pareto-efficient candidates:
1) apply hard thresholds (must-pass constraints)
2) rank remaining candidates via a tie-break order (e.g., prioritize fairness then coverage, etc.)
3) output a selected policy candidate with an auditable rationale

## 5) Baselines
- **Unconstrained** models (no ECL)
- **Fairness post-processing** (Hardt et al., Equal Opportunity)
- **Fairlearn reductions** (optional) as a stronger constrained-optimization baseline

