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

# Match-context fields that are validated against the values actually observed
# in the historical data (mirrors notebook 4's valid_domains/validate_context).
CONTEXT_CATEGORICAL_FIELDS = ["game_state", "opponent_skill_level", "opponent_style", "side"]


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


def compute_valid_domains(history: pd.DataFrame) -> dict[str, list[str]]:
    """Valid categorical values for each context field, derived from history.

    Mirrors notebook 4's ``valid_domains`` cell so a CLI-supplied match context
    is checked against the same domain the model was actually trained on,
    rather than a hardcoded list that can drift as new data is added.
    """
    return {
        field: sorted(history[field].dropna().unique().tolist())
        for field in CONTEXT_CATEGORICAL_FIELDS
        if field in history.columns
    }


def validate_context(context: dict, valid_domains: dict[str, list[str]]) -> None:
    """Raise ``ValueError`` listing every context field outside its valid domain.

    Mirrors notebook 4's ``validate_context`` cell. Guards against silently
    scoring a typo'd or out-of-domain categorical value, which
    ``OneHotEncoder(handle_unknown="ignore")`` would otherwise zero out
    without warning.
    """
    errors = []
    for field, valid_values in valid_domains.items():
        if field in context and context[field] not in valid_values:
            errors.append(
                f"  '{field}' = {context[field]!r} is not a valid value. "
                f"Valid options: {valid_values}"
            )
    if errors:
        raise ValueError("Invalid match context:\n" + "\n".join(errors))


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


def filter_by_reliability(df: pd.DataFrame, min_reliability: float = 0.5) -> pd.DataFrame:
    """Restrict ranked recommendations to combos with combo_reliability >= min_reliability.

    Mirrors the notebook 4 "High Confidence Filter" cell (``combo_reliability
    >= 0.5`` is equivalent to ``combo_attempts >= 15``, since
    ``combo_reliability = min(combo_attempts / RELIABILITY_CAP, 1)``). Passing
    ``min_reliability <= 0`` disables filtering. If the filter would remove
    every candidate (e.g. a genuinely novel context with no historical combo
    data at all), falls back to returning the unfiltered frame with a printed
    warning rather than silently returning an empty result.

    Assumes ``df`` is already sorted best-first (as returned by
    ``compute_recommendation_score``); sort order is preserved.
    """
    if min_reliability <= 0:
        return df

    filtered = df[df["combo_reliability"] >= min_reliability]

    if filtered.empty:
        print(
            f"[recommend] warning: combo_reliability >= {min_reliability} filtered out "
            f"all {len(df)} candidates (no combo has enough historical data). "
            f"Falling back to the unfiltered ranking -- treat these results with extra "
            f"caution (see the low-reliability caveat in the README)."
        )
        return df

    print(
        f"[recommend] filtered from {len(df)} to {len(filtered)} serve options "
        f"with combo_reliability >= {min_reliability} ({len(df) - len(filtered)} dropped)"
    )
    return filtered
