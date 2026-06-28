"""End-to-end pipeline runner: clean -> features -> train -> recommend.

Run all steps:

    python -m src.run_pipeline            # or: python -m src.run_pipeline all

Run a single step:

    python -m src.run_pipeline features

Each step reads the previous step's on-disk output, mirroring the notebook
workflow (raw CSV -> processed features CSV -> saved model pkls -> ranked
serves), so the pipeline is reproducible from a fresh checkout.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

# Support both "python -m src.run_pipeline" (package import) and
# "python src/run_pipeline.py" (script import).
try:
    from . import clean_data, features, recommend_serves, train_model
except ImportError:  # pragma: no cover - script-style execution fallback
    import clean_data
    import features
    import recommend_serves
    import train_model

RAW_PATH = "data/table_tennis_serves.csv"
CLEANED_PATH = "data/processed/serves_cleaned.csv"
PROCESSED_PATH = "data/processed/table_tennis_serves_features.csv"
MODELS_DIR = "models"

# Pre-serve attributes that define a candidate serve (used by the recommend step).
SERVE_ATTRIBUTE_COLS = [
    "serve_type", "spin_type", "spin_intensity", "serve_length",
    "placement_zone", "toss_height", "contact_point", "intended_setup",
]

# Example match context for the recommend demonstration.
EXAMPLE_CONTEXT = {
    "game_number": 2,
    "server_score": 8,
    "receiver_score": 7,
    "game_state": "neutral",
    "opponent_skill_level": "advanced",
    "opponent_style": "looper",
    "side": "backhand_side",
}


def run_clean() -> None:
    """Standalone data-quality step: normalize strings and derive the target.

    Note: the feature step (``run_features``) intentionally reads the *raw* CSV,
    not this cleaned output, so that categorical casing is preserved exactly as
    in the notebooks. This step is kept as a separate normalization/QA artifact.
    """
    print("[clean] loading raw data ...")
    raw = clean_data.load_raw_data(RAW_PATH)
    cleaned = clean_data.clean_serves(raw)
    clean_data.save_processed_data(cleaned, CLEANED_PATH)
    print(f"[clean] wrote {CLEANED_PATH}  ({cleaned.shape[0]} rows x {cleaned.shape[1]} cols)")


def run_features() -> pd.DataFrame:
    """Build the 52-column engineered feature set from the raw CSV."""
    print("[features] engineering features ...")
    raw = pd.read_csv(RAW_PATH)
    engineered = features.add_features(raw)
    engineered.to_csv(PROCESSED_PATH, index=False)
    print(f"[features] wrote {PROCESSED_PATH}  ({engineered.shape[0]} rows x {engineered.shape[1]} cols)")
    return engineered


def run_train() -> None:
    """Train and grouped-CV evaluate all models, saving pkl artifacts."""
    print("[train] training models (GroupKFold CV) ...")
    df = pd.read_csv(PROCESSED_PATH)
    results = train_model.train_and_evaluate(df, models_dir=MODELS_DIR)
    summary = train_model.results_to_frame(results)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(summary.to_string(index=False))
    print(f"[train] saved model artifacts under {MODELS_DIR}/")


def build_candidate_features(serve_options: pd.DataFrame, context: dict) -> pd.DataFrame:
    """Reconstruct pre-point features for a candidate pool under a fixed context.

    Mirrors the candidate-construction logic in notebook 4.
    """
    rec = serve_options.copy()
    for key, value in context.items():
        rec[key] = value

    rec["score_margin"] = rec["server_score"] - rec["receiver_score"]
    rec["total_points_played_in_game"] = rec["server_score"] + rec["receiver_score"]
    rec["is_tied"] = (rec["server_score"] == rec["receiver_score"]).astype(int)
    rec["is_trailing"] = (rec["server_score"] < rec["receiver_score"]).astype(int)
    rec["is_leading"] = (rec["server_score"] > rec["receiver_score"]).astype(int)
    rec["is_late_game"] = (rec["total_points_played_in_game"] >= 16).astype(int)
    rec["is_deuce_or_later"] = ((rec["server_score"] >= 10) & (rec["receiver_score"] >= 10)).astype(int)
    rec["is_game_point_for_server"] = (
        (rec["server_score"] >= 10) & (rec["server_score"] > rec["receiver_score"])
    ).astype(int)
    rec["is_game_point_against_server"] = (
        (rec["receiver_score"] >= 10) & (rec["receiver_score"] > rec["server_score"])
    ).astype(int)

    rec["serve_spin_combo"] = rec["serve_type"] + "_" + rec["spin_type"]
    rec["serve_length_spin_combo"] = rec["serve_length"] + "_" + rec["spin_type"]
    rec["serve_placement_combo"] = rec["serve_type"] + "_" + rec["placement_zone"]
    rec["full_serve_combo"] = (
        rec["serve_type"] + "_" + rec["spin_type"] + "_" + rec["serve_length"] + "_" + rec["placement_zone"]
    )

    rec["is_heavy_spin"] = (rec["spin_intensity"] >= 3).astype(int)
    rec["is_low_spin"] = (rec["spin_intensity"] <= 1).astype(int)
    return rec


def run_recommend(context: dict | None = None, top_n: int = 10) -> pd.DataFrame:
    """Rank candidate serves for a match context using the saved primary model."""
    import joblib

    context = context or EXAMPLE_CONTEXT
    print("[recommend] scoring candidate serves ...")
    history = pd.read_csv(PROCESSED_PATH)
    model = joblib.load(f"{MODELS_DIR}/serve_win_probability_model.pkl")
    model_features = joblib.load(f"{MODELS_DIR}/model_features.pkl")

    serve_options = history[SERVE_ATTRIBUTE_COLS].drop_duplicates().reset_index(drop=True)
    combo_summary, overall_win_rate = recommend_serves.build_combo_summary(history)

    rec = build_candidate_features(serve_options, context)
    rec = rec.merge(combo_summary, on="full_serve_combo", how="left")
    rec["combo_attempts"] = rec["combo_attempts"].fillna(0)
    rec["combo_win_rate"] = rec["combo_win_rate"].fillna(overall_win_rate)
    rec["combo_reliability"] = np.minimum(rec["combo_attempts"] / recommend_serves.RELIABILITY_CAP, 1)
    rec["predicted_win_probability"] = model.predict_proba(rec[model_features])[:, 1]

    ranked = recommend_serves.compute_recommendation_score(rec)
    display_cols = [
        "serve_type", "spin_type", "spin_intensity", "serve_length", "placement_zone",
        "predicted_win_probability", "combo_win_rate", "combo_attempts",
        "reliability_label", "recommendation_score",
    ]
    print(f"[recommend] top {top_n} serves for context {context}:")
    print(ranked[display_cols].head(top_n).to_string(index=False))
    return ranked


STEPS = {
    "clean": run_clean,
    "features": run_features,
    "train": run_train,
    "recommend": run_recommend,
}


def run_all() -> None:
    run_clean()
    run_features()
    run_train()
    run_recommend()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    step = argv[0] if argv else "all"
    if step in ("all", "--all"):
        run_all()
    elif step in STEPS:
        STEPS[step]()
    else:
        print(f"Unknown step: {step!r}. Choose from: all, {', '.join(STEPS)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
