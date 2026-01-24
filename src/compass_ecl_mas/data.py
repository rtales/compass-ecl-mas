from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from .simulator.education import simulate_education_A

@dataclass(frozen=True)
class DataSpec:
    source: str = "synthetic"   # "synthetic" | "csv"
    name: str | None = None     # synthetic scenario name, e.g., education_A
    path: str | None = None     # csv path
    target_col: str = "y"
    group_col: str = "ses"      # audit-only sensitive attribute
    # Synthetic params
    n_students: int = 20000
    n_schools: int = 100
    bias: dict | None = None

def load_dataset_from_config(data_cfg: dict, seed: int) -> pd.DataFrame:
    """Load either synthetic or real CSV dataset based on config.

    Backward compatible with previous configs that only had synthetic params.
    """
    source = str(data_cfg.get("source", "synthetic")).lower()

    if source in ("synthetic", "simulated", "sim"):
        # Default to education_A simulator
        name = str(data_cfg.get("name", "education_A"))
        if name != "education_A":
            raise ValueError(f"Unknown synthetic dataset name: {name}. Only 'education_A' is supported.")
        return simulate_education_A(
            seed=int(seed),
            n_students=int(data_cfg.get("n_students", 20000)),
            n_schools=int(data_cfg.get("n_schools", 100)),
            bias=data_cfg.get("bias", None) or {},
        )

    if source in ("csv", "real"):
        path = data_cfg.get("path", None)
        if not path:
            raise ValueError("For data.source='csv', you must set data.path to a CSV file.")
        df = pd.read_csv(Path(path))
        target_col = str(data_cfg.get("target_col", "y"))
        group_col = str(data_cfg.get("group_col", "ses"))

        if target_col not in df.columns:
            raise ValueError(f"CSV missing target_col '{target_col}'. Columns: {list(df.columns)[:25]}")
        if group_col not in df.columns:
            raise ValueError(f"CSV missing group_col '{group_col}'. Columns: {list(df.columns)[:25]}")

        # Normalize schema to expected 'y' + group_col naming if needed
        if target_col != "y":
            df = df.rename(columns={target_col: "y"})
        if group_col != "ses":
            df = df.rename(columns={group_col: "ses"})

        return df

    raise ValueError(f"Unsupported data.source: {source}. Use 'synthetic' or 'csv'.")
