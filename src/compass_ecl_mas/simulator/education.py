from __future__ import annotations
import numpy as np
import pandas as pd

def simulate_education_A(seed: int, n_students: int = 20000, n_schools: int = 50, bias: dict | None = None) -> pd.DataFrame:
    '''
    Fully specified synthetic education scenario (privacy-safe).
    Protected attributes:
      - ses: 3 levels {0,1,2} (0=low SES)
      - sex: {0,1}
    Training features exclude protected attributes by default (ECL denylist).
    '''
    bias = bias or {}
    rng = np.random.default_rng(seed)

    ses = rng.choice([0, 1, 2], size=n_students, p=[0.35, 0.45, 0.20]).astype(int)
    sex = rng.choice([0, 1], size=n_students, p=[0.5, 0.5]).astype(int)
    school_id = rng.integers(0, n_schools, size=n_students)

    prev_grade = np.clip(rng.normal(0.0, 1.0, size=n_students), -2.5, 2.5)
    attendance = np.clip(rng.normal(0.0, 1.0, size=n_students), -2.5, 2.5)
    engagement = np.clip(rng.normal(0.0, 1.0, size=n_students), -2.5, 2.5)
    late_assign = np.clip(rng.normal(0.0, 1.0, size=n_students), -2.5, 2.5)

    drift_strength = float(bias.get("drift_strength", 0.0))
    school_shift = rng.normal(0.0, 1.0, size=n_schools)
    drift = drift_strength * school_shift[school_id]
    attendance = np.clip(attendance + 0.5 * drift, -3.0, 3.0)
    engagement = np.clip(engagement + 0.3 * drift, -3.0, 3.0)

    meas_noise = float(bias.get("measurement_noise_by_group", 0.0))
    noise_scale = meas_noise * (1.0 + (2 - ses) * 0.35)
    prev_grade = prev_grade + rng.normal(0.0, noise_scale)
    engagement = engagement + rng.normal(0.0, noise_scale)

    miss = float(bias.get("missingness_by_group", 0.0))
    miss_prob = np.clip(miss * (1.0 + (2 - ses) * 0.40), 0.0, 0.90)
    m = rng.random(n_students) < miss_prob
    engagement = engagement.astype(float)
    engagement[m] = np.nan

    logits = (
        -0.6 * prev_grade
        -0.5 * attendance
        -0.4 * np.nan_to_num(engagement, nan=0.0)
        +0.3 * late_assign
        +0.25 * (ses == 0)
        +0.10 * (ses == 1)
        + rng.normal(0.0, 0.5, size=n_students)
    )
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.random(n_students) < p).astype(int)

    return pd.DataFrame({
        "school_id": school_id.astype(int),
        "prev_grade": prev_grade.astype(float),
        "attendance": attendance.astype(float),
        "engagement": engagement.astype(float),
        "late_assign": late_assign.astype(float),
        "sex": sex.astype(int),
        "ses": ses.astype(int),
        "y": y.astype(int),
    })
