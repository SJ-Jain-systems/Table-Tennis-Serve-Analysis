"""Serve recommendation scoring and reliability calculation.

Mirrors ``notebooks/4_Serve_Recommendation.ipynb``. Candidate serves are scored
for a fixed match context by combining the model's predicted win probability
with leakage-aware historical combo performance and a sample-size reliability
correction:

    recommendation_score =
          0.70 * predicted_win_probability   (trained model)
        + 0.20 * combo_win_rate              (historical combo win rate)
        + 0.10 * combo_reliability           (capped sample-size correction)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Recommendation score weights (must match the README and notebook 4).
WIN_PROB_WEIGHT = 0.70
COMBO_WIN_RATE_WEIGHT = 0.20
RELIABILITY_WEIGHT = 0.10

# Sample-size cap for the reliability term (30 attempts -> fully reliable).
RELIABILITY_CAP = 30


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + (z**2 / n)
    centre = p + z**2 / (2 * n)
    margin = z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2)))
    return ((centre - margin) / denom, (centre + margin) / denom)


def reliability_label(n: int) -> str:
    """Bucket a combo's historical attempt count into a reliability tier."""
    if n >= RELIABILITY_CAP:
        return "High"
    if n >= 20:
        return "Medium"
    return "Low"


def build_combo_summary(history: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Historical attempts/win-rate per ``full_serve_combo`` and the overall rate."""
    combo_summary = (
        history.groupby("full_serve_combo")
        .agg(combo_attempts=("point_won", "count"), combo_win_rate=("point_won", "mean"))
        .reset_index()
    )
    overall_win_rate = history["point_won"].mean()
    return combo_summary, overall_win_rate


def compute_recommendation_score(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the weighted recommendation score and add reliability annotations.

    Expects ``predicted_win_probability``, ``combo_win_rate``,
    ``combo_reliability`` and ``combo_attempts`` columns (as produced by the
    notebook-4 scoring pipeline). Returns the frame sorted best-first with a
    ``recommendation_score``, ``reliability_label`` and Wilson CI on the
    historical win rate.
    """
    out = df.copy()
    out["recommendation_score"] = (
        WIN_PROB_WEIGHT * out["predicted_win_probability"]
        + COMBO_WIN_RATE_WEIGHT * out["combo_win_rate"]
        + RELIABILITY_WEIGHT * out["combo_reliability"]
    )

    attempts = out["combo_attempts"].fillna(0).astype(int)
    out["reliability_label"] = attempts.map(reliability_label)
    ci = [wilson_ci(int(round(wr * n)), n) for wr, n in zip(out["combo_win_rate"], attempts)]
    out["combo_win_rate_ci_low"] = [lo for lo, _ in ci]
    out["combo_win_rate_ci_high"] = [hi for _, hi in ci]

    return out.sort_values("recommendation_score", ascending=False)
