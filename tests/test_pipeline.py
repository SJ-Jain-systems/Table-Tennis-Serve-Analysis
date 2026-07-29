"""Minimal regression tests for the serve-analysis pipeline.

Run with: pytest
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import clean_data, features, recommend_serves, train_model
from src.train_model import build_preprocessor

RAW_PATH = Path("data/table_tennis_serves.csv")

# Post-serve / outcome columns that must never be used as model features.
# (intended_setup is intentionally excluded: it is the server's pre-serve intent,
# known before contact, and the project's leakage list treats it as usable.)
POST_SERVE_COLUMNS = {
    "return_type",
    "return_quality",
    "return_placement",
    "rally_length",
    "point_end_type",
    "point_outcome",
    "rally_type_achieved",
    "chop_rally_outcome",
}


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return pd.read_csv(RAW_PATH)


@pytest.fixture(scope="module")
def engineered_df(raw_df) -> pd.DataFrame:
    return features.add_features(raw_df)


def test_no_combo_leakage(engineered_df):
    """A combo appearing exactly once gets the global fallback win rate, not 0/1.

    This guards the leave-one-out computation: excluding a point's own outcome
    from its own combo statistic prevents the feature from leaking the label.
    """
    overall_win_rate = engineered_df["point_won"].mean()
    combo_counts = engineered_df.groupby("full_serve_combo")["point_won"].transform("count")
    single_use = engineered_df[combo_counts == 1]

    assert len(single_use) > 0, "expected at least one single-use combo in the dataset"
    # Every single-use combo falls back to the dataset-wide win rate ...
    assert np.allclose(single_use["combo_win_rate"], overall_win_rate)
    # ... and crucially is never the point's own 0/1 outcome.
    assert not single_use["combo_win_rate"].isin([0.0, 1.0]).any()


def test_no_post_serve_leakage():
    """The model feature set must contain no post-serve / outcome columns."""
    leaked = POST_SERVE_COLUMNS.intersection(train_model.MODEL_FEATURES)
    assert not leaked, f"post-serve columns leaked into MODEL_FEATURES: {sorted(leaked)}"


def test_pipeline_smoke(engineered_df):
    """clean -> features -> train path runs and yields probabilities in [0, 1]."""
    feature_cols = [c for c in train_model.MODEL_FEATURES if c in engineered_df.columns]
    assert feature_cols, "no model features present in engineered frame"

    X = engineered_df[feature_cols]
    y = engineered_df["point_won"]

    pipeline = Pipeline([
        ("prep", build_preprocessor(X)),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    pipeline.fit(X, y)
    probs = pipeline.predict_proba(X)[:, 1]

    assert probs.shape == (len(engineered_df),)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()


def test_validate_context_accepts_known_values():
    """A context using only observed categorical values passes validation."""
    history = pd.DataFrame({
        "game_state": ["neutral", "deuce"],
        "opponent_skill_level": ["advanced", "beginner"],
        "opponent_style": ["looper", "blocker"],
        "side": ["backhand_side", "forehand_side"],
    })
    domains = recommend_serves.compute_valid_domains(history)
    recommend_serves.validate_context(
        {"game_state": "neutral", "opponent_style": "looper"}, domains
    )  # should not raise


def test_validate_context_rejects_unknown_value():
    """An out-of-domain categorical value raises ValueError naming the field."""
    history = pd.DataFrame({
        "game_state": ["neutral", "deuce"],
        "opponent_skill_level": ["advanced", "beginner"],
        "opponent_style": ["looper", "looper"],
        "side": ["backhand_side", "forehand_side"],
    })
    domains = recommend_serves.compute_valid_domains(history)
    with pytest.raises(ValueError, match="opponent_style"):
        recommend_serves.validate_context({"opponent_style": "chopper"}, domains)


def test_filter_by_reliability_default_threshold():
    """combo_reliability >= 0.5 keeps only sufficiently-attempted combos, sorted order preserved."""
    df = pd.DataFrame({
        "combo_reliability": [0.9, 0.4, 0.6, 0.1],
        "recommendation_score": [0.8, 0.7, 0.6, 0.5],
    })
    filtered = recommend_serves.filter_by_reliability(df, min_reliability=0.5)
    assert list(filtered["combo_reliability"]) == [0.9, 0.6]
    assert list(filtered.index) == [0, 2]


def test_filter_by_reliability_zero_disables_filter():
    df = pd.DataFrame({"combo_reliability": [0.0, 0.1], "recommendation_score": [0.9, 0.5]})
    assert len(recommend_serves.filter_by_reliability(df, min_reliability=0.0)) == 2


def test_filter_by_reliability_empty_result_falls_back(capsys):
    """If every candidate is below threshold, fall back to the unfiltered frame with a warning."""
    df = pd.DataFrame({"combo_reliability": [0.1, 0.2], "recommendation_score": [0.9, 0.5]})
    result = recommend_serves.filter_by_reliability(df, min_reliability=0.9)
    assert len(result) == 2
    assert "warning" in capsys.readouterr().out.lower()


# --- Raw-data schema validation (clean_data.validate_raw_schema) -------------

def _valid_raw_row() -> dict:
    """A single row satisfying the raw-input contract, one value per column."""
    return {
        "serve_type": "backhand", "spin_type": "float/no-spin", "spin_intensity": 1,
        "serve_length": "long", "placement_zone": "middle_FH", "toss_height": "medium",
        "contact_point": "body_center", "match_id": 1, "game_number": 1,
        "server_score": 0, "receiver_score": 0, "game_state": "neutral",
        "opponent_id": 1, "opponent_skill_level": "intermediate", "opponent_style": "looper",
        "side": "forehand_side", "return_type": "loop", "return_quality": 3,
        "return_placement": "deep_BH", "rally_length": 6, "point_outcome": "lost",
        "point_end_type": "forced_error", "intended_setup": "force_weak_push",
        "rally_type_achieved": "attack_rally", "chop_rally_outcome": "not_applicable",
    }


def test_validate_raw_schema_accepts_shipped_dataset():
    """The dataset shipped in the repo satisfies its own input contract."""
    df = pd.read_csv(RAW_PATH)
    assert clean_data.validate_raw_schema(df) is df


def test_validate_raw_schema_rejects_missing_column():
    df = pd.DataFrame([_valid_raw_row()]).drop(columns=["placement_zone"])
    with pytest.raises(ValueError, match="placement_zone"):
        clean_data.validate_raw_schema(df)


def test_validate_raw_schema_rejects_out_of_range_numeric():
    row = _valid_raw_row()
    row["spin_intensity"] = 9  # scale is 1-3
    with pytest.raises(ValueError, match="spin_intensity"):
        clean_data.validate_raw_schema(pd.DataFrame([row]))


def test_validate_raw_schema_rejects_non_numeric_value():
    row = _valid_raw_row()
    row["rally_length"] = "long"
    with pytest.raises(ValueError, match="rally_length"):
        clean_data.validate_raw_schema(pd.DataFrame([row]))


def test_validate_raw_schema_rejects_unknown_outcome():
    row = _valid_raw_row()
    row["point_outcome"] = "tie"  # only won/lost allowed
    with pytest.raises(ValueError, match="point_outcome"):
        clean_data.validate_raw_schema(pd.DataFrame([row]))
