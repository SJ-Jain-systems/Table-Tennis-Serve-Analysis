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

from src import features, train_model
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
