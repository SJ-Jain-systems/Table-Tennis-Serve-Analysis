"""Feature engineering for pre-point serve recommendation modeling."""
from __future__ import annotations

import pandas as pd

PRESSURE_STATES = {"deuce", "game_point", "trailing"}
ATTACKER_STYLES = {"looper", "attacker"}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    engineered = df.copy()

    if {"server_score", "receiver_score"}.issubset(engineered.columns):
        engineered["score_margin"] = engineered["server_score"] - engineered["receiver_score"]

    if "game_state" in engineered.columns:
        engineered["is_pressure_point"] = engineered["game_state"].isin(PRESSURE_STATES).astype(int)

    required = {"serve_length", "spin_type", "spin_intensity"}
    if required.issubset(engineered.columns):
        engineered["is_short_heavy_backspin"] = (
            (engineered["serve_length"] == "short")
            & (engineered["spin_type"] == "backspin")
            & (engineered["spin_intensity"].astype(str).isin({"3", "heavy"}))
        ).astype(int)

    combo_cols = ["serve_type", "spin_type", "placement_zone"]
    if set(combo_cols).issubset(engineered.columns):
        engineered["serve_combo"] = engineered[combo_cols].astype(str).agg("_".join, axis=1)

    if "opponent_style" in engineered.columns:
        engineered["opponent_is_attacker"] = engineered["opponent_style"].isin(ATTACKER_STYLES).astype(int)

    return engineered
