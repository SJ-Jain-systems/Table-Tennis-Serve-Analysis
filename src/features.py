"""Feature engineering for pre-point serve recommendation modeling.

This mirrors the transformation logic in ``notebooks/2_Feature_Engineering.ipynb``
exactly. Given the raw 25-column serve dataset it produces the 27 engineered
columns documented in the README, for a 52-column processed frame. Any change
to the notebook's feature logic must be reflected here (and vice versa).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Maps the opponent skill label to an ordinal value; unknown labels fall back to
# the "intermediate" level (2), matching notebook 2.
SKILL_MAP = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}

# Sample-size cap for the reliability term: a combo with >=30 "other" attempts is
# treated as fully reliable (1.0).
RELIABILITY_CAP = 30


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the engineered pre-point features to a raw serve dataframe.

    The returned frame contains the original raw columns plus 27 engineered
    columns, including the binary target ``point_won`` and the leakage-safe
    historical combination statistics (``combo_attempts``, ``combo_win_rate``,
    ``combo_reliability``).
    """
    engineered = df.copy()

    # --- Binary modeling target -------------------------------------------------
    engineered["point_won"] = (engineered["point_outcome"] == "won").astype(int)

    # --- Score-state features ---------------------------------------------------
    engineered["score_margin"] = engineered["server_score"] - engineered["receiver_score"]
    engineered["total_points_played_in_game"] = (
        engineered["server_score"] + engineered["receiver_score"]
    )
    engineered["is_tied"] = (engineered["server_score"] == engineered["receiver_score"]).astype(int)
    engineered["is_trailing"] = (engineered["server_score"] < engineered["receiver_score"]).astype(int)
    engineered["is_leading"] = (engineered["server_score"] > engineered["receiver_score"]).astype(int)
    engineered["is_late_game"] = (engineered["total_points_played_in_game"] >= 16).astype(int)
    engineered["is_deuce_or_later"] = (
        (engineered["server_score"] >= 10) & (engineered["receiver_score"] >= 10)
    ).astype(int)
    engineered["is_game_point_for_server"] = (
        (engineered["server_score"] >= 10) & (engineered["server_score"] > engineered["receiver_score"])
    ).astype(int)
    engineered["is_game_point_against_server"] = (
        (engineered["receiver_score"] >= 10) & (engineered["receiver_score"] > engineered["server_score"])
    ).astype(int)

    # --- Opponent encodings + interaction terms --------------------------------
    # Opponent style dummies are created before spin_x_looper because that
    # interaction depends on opponent_is_looper.
    engineered["opponent_is_looper"] = (engineered["opponent_style"] == "looper").astype(int)
    engineered["opponent_is_chopper"] = (engineered["opponent_style"] == "chopper").astype(int)
    engineered["opponent_is_attacker"] = (engineered["opponent_style"] == "attacker").astype(int)

    # Spin intensity amplified against looping opponents.
    engineered["spin_x_looper"] = engineered["spin_intensity"] * engineered["opponent_is_looper"]

    # Pressure distance: how far from a tie, regardless of direction.
    engineered["score_margin_abs"] = engineered["score_margin"].abs()

    # Unified high-pressure flag combining three correlated game-state conditions.
    engineered["is_high_pressure"] = (
        (engineered["is_deuce_or_later"] == 1)
        | (engineered["is_game_point_for_server"] == 1)
        | (engineered["is_game_point_against_server"] == 1)
    ).astype(int)

    engineered["opponent_skill_numeric"] = (
        engineered["opponent_skill_level"].map(SKILL_MAP).fillna(2)
    )

    # --- Serve combination identifiers -----------------------------------------
    engineered["serve_spin_combo"] = engineered["serve_type"] + "_" + engineered["spin_type"]
    engineered["serve_length_spin_combo"] = engineered["serve_length"] + "_" + engineered["spin_type"]
    engineered["serve_placement_combo"] = engineered["serve_type"] + "_" + engineered["placement_zone"]
    engineered["full_serve_combo"] = (
        engineered["serve_type"]
        + "_"
        + engineered["spin_type"]
        + "_"
        + engineered["serve_length"]
        + "_"
        + engineered["placement_zone"]
    )

    # --- Spin flags + interaction ----------------------------------------------
    engineered["is_heavy_spin"] = (engineered["spin_intensity"] >= 3).astype(int)
    engineered["is_low_spin"] = (engineered["spin_intensity"] <= 1).astype(int)
    engineered["spin_length_interaction"] = (
        engineered["spin_intensity"].astype(str) + "_" + engineered["serve_length"]
    )

    # --- Leakage-safe historical combo statistics ------------------------------
    engineered = add_combo_statistics(engineered)

    return engineered


def add_combo_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Add leave-one-out historical statistics for ``full_serve_combo``.

    ``combo_win_rate`` and ``combo_reliability`` summarize how a serve
    combination has historically performed. Each point's own outcome is excluded
    from the aggregate describing it (leave-one-out), so a combo attempted once
    cannot leak its own label. Combos with no "other" observations fall back to
    the dataset's overall win rate.
    """
    out = df.copy()

    combo_summary = (
        out.groupby("full_serve_combo")
        .agg(combo_attempts=("point_won", "count"), combo_win_sum=("point_won", "sum"))
        .reset_index()
    )
    out = out.merge(combo_summary, on="full_serve_combo", how="left")

    overall_win_rate = out["point_won"].mean()
    other_attempts = out["combo_attempts"] - 1
    other_wins = out["combo_win_sum"] - out["point_won"]

    out["combo_win_rate"] = np.where(
        other_attempts > 0,
        other_wins / other_attempts,
        overall_win_rate,
    )
    out["combo_reliability"] = np.minimum(np.maximum(other_attempts, 0) / RELIABILITY_CAP, 1)

    out = out.drop(columns=["combo_win_sum"])
    return out
