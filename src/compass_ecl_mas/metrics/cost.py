from __future__ import annotations

def cost_metrics(train_ms: float, n_samples: int) -> dict:
    latency_ms = float(train_ms) + float(n_samples) * 0.02
    return {"train_ms": float(train_ms), "latency_ms": float(latency_ms)}
