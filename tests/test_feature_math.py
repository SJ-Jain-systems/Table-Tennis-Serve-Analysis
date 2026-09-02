"""Exact-value tests for the engineered feature math.

The existing suite covers leakage, schema validation, context validation and the
reliability filter, but never asserts the *numeric* output of ``add_features`` or
the ``wilson_ci`` helper. A silent arithmetic bug in a score-state flag, an
interaction term, or the Wilson interval would therefore pass CI unnoticed.

These tests build a tiny hand-constructed dataframe whose expected feature values
can be computed by hand, and assert ``features.add_features`` reproduces them
exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import features, recommend_serves


def _raw_row(**overrides) -> dict:
    """A minimal raw serve row; feature-relevant fields overridable per test."""
    row = {
        "serve_type": "backhand",
        "spin_type": "topspin",
        "spin_intensity": 2,
        "serve_length": "long",
        "placement_zone": "middle_fh",
        "toss_height": "medium",
        "contact_point": "body_center",
        "match_id": 1,
        "game_number": 1,
        "server_score": 0,
        "receiver_score": 0,
        "game_state": "neutral",
        "opponent_id": 1,
        "opponent_skill_level": "intermediate",
        "opponent_style": "looper",
        "side": "forehand_side",
        "return_type": "loop",
        "return_quality": 2,
        "return_placement": "deep_bh",
        "rally_length": 4,
        "point_outcome": "won",
        "point_end_type": "winner",
        "intended_setup": "force_weak_push",
        "rally_type_achieved": "attack_rally",
        "chop_rally_outcome": "not_applicable",
    }
    row.update(overrides)
    return row


def _engineer_one(**overrides) -> pd.Series:
    """Run add_features on a single-row frame and return that row."""
    df = pd.DataFrame([_raw_row(**overrides)])
    return features.add_features(df).iloc[0]


# --- Score-state features ----------------------------------------------------

def test_score_state_when_leading():
    row = _engineer_one(server_score=8, receiver_score=5)
    assert row["score_margin"] == 3
    assert row["score_margin_abs"] == 3
    assert row["total_points_played_in_game"] == 13
    assert row["is_leading"] == 1
    assert row["is_trailing"] == 0
    assert row["is_tied"] == 0
    assert row["is_late_game"] == 0  # 13 < 16


def test_score_state_when_trailing_and_late():
    row = _engineer_one(server_score=7, receiver_score=10)
    assert row["score_margin"] == -3
    assert row["score_margin_abs"] == 3
    assert row["is_trailing"] == 1
    assert row["is_leading"] == 0
    assert row["is_late_game"] == 1  # 17 >= 16
    # Receiver at 10 and ahead => game point against server.
    assert row["is_game_point_against_server"] == 1
    assert row["is_game_point_for_server"] == 0


def test_deuce_flags_and_high_pressure():
    row = _engineer_one(server_score=10, receiver_score=10)
    assert row["is_tied"] == 1
    assert row["is_deuce_or_later"] == 1
    assert row["is_game_point_for_server"] == 0
    assert row["is_game_point_against_server"] == 0
    # is_high_pressure is the OR of the three pressure flags; deuce alone trips it.
    assert row["is_high_pressure"] == 1


def test_game_point_for_server_sets_high_pressure():
    row = _engineer_one(server_score=10, receiver_score=8)
    assert row["is_game_point_for_server"] == 1
    assert row["is_deuce_or_later"] == 0  # receiver below 10
    assert row["is_high_pressure"] == 1


def test_no_pressure_early_game():
    row = _engineer_one(server_score=2, receiver_score=1)
    assert row["is_deuce_or_later"] == 0
    assert row["is_game_point_for_server"] == 0
    assert row["is_game_point_against_server"] == 0
    assert row["is_high_pressure"] == 0


# --- Opponent + interaction terms -------------------------------------------

def test_spin_x_looper_interaction():
    looper = _engineer_one(spin_intensity=3, opponent_style="looper")
    assert looper["opponent_is_looper"] == 1
    assert looper["spin_x_looper"] == 3  # spin_intensity * opponent_is_looper

    chopper = _engineer_one(spin_intensity=3, opponent_style="chopper")
    assert chopper["opponent_is_looper"] == 0
    assert chopper["spin_x_looper"] == 0  # zeroed out for non-loopers
    assert chopper["opponent_is_chopper"] == 1


def test_opponent_skill_numeric_maps_and_falls_back():
    assert _engineer_one(opponent_skill_level="beginner")["opponent_skill_numeric"] == 1
    assert _engineer_one(opponent_skill_level="expert")["opponent_skill_numeric"] == 4
    # Unknown label falls back to the intermediate level (2), per SKILL_MAP.
    assert _engineer_one(opponent_skill_level="unranked")["opponent_skill_numeric"] == 2


# --- Spin flags + combo identifiers -----------------------------------------

def test_spin_flags_thresholds():
    heavy = _engineer_one(spin_intensity=3)
    assert heavy["is_heavy_spin"] == 1 and heavy["is_low_spin"] == 0
    low = _engineer_one(spin_intensity=1)
    assert low["is_heavy_spin"] == 0 and low["is_low_spin"] == 1
    mid = _engineer_one(spin_intensity=2)
    assert mid["is_heavy_spin"] == 0 and mid["is_low_spin"] == 0


def test_serve_combo_identifiers():
    row = _engineer_one(
        serve_type="pendulum", spin_type="backspin",
        serve_length="short", placement_zone="wide_fh",
    )
    assert row["serve_spin_combo"] == "pendulum_backspin"
    assert row["serve_length_spin_combo"] == "short_backspin"
    assert row["serve_placement_combo"] == "pendulum_wide_fh"
    assert row["full_serve_combo"] == "pendulum_backspin_short_wide_fh"


def test_point_won_target_encoding():
    assert _engineer_one(point_outcome="won")["point_won"] == 1
    assert _engineer_one(point_outcome="lost")["point_won"] == 0


# --- Wilson score interval ---------------------------------------------------

def test_wilson_ci_symmetric_half():
    lo, hi = recommend_serves.wilson_ci(5, 10)
    # Known 95% Wilson interval for 5/10.
    assert lo == pytest.approx(0.2366, abs=1e-3)
    assert hi == pytest.approx(0.7634, abs=1e-3)
    # Symmetric about 0.5 for a symmetric proportion.
    assert (lo + hi) / 2 == pytest.approx(0.5, abs=1e-6)


def test_wilson_ci_zero_n_is_degenerate():
    assert recommend_serves.wilson_ci(0, 0) == (0.0, 0.0)


def test_wilson_ci_bounds_stay_in_unit_interval():
    for successes, n in [(0, 4), (4, 4), (1, 3), (99, 100)]:
        lo, hi = recommend_serves.wilson_ci(successes, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_wilson_ci_narrows_with_more_data():
    lo_small, hi_small = recommend_serves.wilson_ci(5, 10)
    lo_big, hi_big = recommend_serves.wilson_ci(50, 100)
    # Same point estimate (0.5), ten times the sample => a strictly tighter interval.
    assert (hi_big - lo_big) < (hi_small - lo_small)
    assert np.isclose((lo_small + hi_small) / 2, (lo_big + hi_big) / 2, atol=1e-6)
