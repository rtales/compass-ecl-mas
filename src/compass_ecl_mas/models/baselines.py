from __future__ import annotations
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

def build_model(name: str):
    name = name.lower().strip()
    if name == "lr":
        return LogisticRegression(max_iter=500, solver="lbfgs")
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    if name == "gbdt":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, random_state=0)
    raise ValueError(f"Unknown model: {name}")
