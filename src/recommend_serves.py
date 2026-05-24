"""Serve recommendation scoring and reliability calculation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + (z**2 / n)
    centre = p + z**2 / (2 * n)
    margin = z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))
    return ((centre - margin) / denom, (centre + margin) / denom)


def reliability_label(n: int) -> str:
    if n >= 30:
        return "High"
    if n >= 20:
        return "Medium"
    return "Low"


def compute_recommendation_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["serve_score"] = (
        0.50 * out["pred_win_prob"]
        + 0.20 * out["pred_weak_return_prob"]
        + 0.15 * out["pred_intended_rally_prob"]
        + 0.15 * out["reliability_score"]
    )
    return out.sort_values("serve_score", ascending=False)
