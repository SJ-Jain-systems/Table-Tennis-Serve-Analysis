"""Model training/evaluation with match-aware grouped validation.

Mirrors ``notebooks/3_Modeling.ipynb``: a majority-class baseline, a tuned LASSO
logistic regression, a tuned random forest, and a gradient boosting classifier,
all evaluated with ``GroupKFold`` (5 folds) grouped by ``match_id``. The trained
pipelines are saved under ``models/`` for the recommendation step.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, List

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Pre-point feature set used for modeling (matches notebook 3). Only information
# known before service contact is included; post-serve diagnostics are excluded
# to prevent leakage.
MODEL_FEATURES = [
    # Serve mechanics
    "serve_type", "spin_type", "spin_intensity", "serve_length", "placement_zone",
    "toss_height", "contact_point",
    # Game context
    "game_number", "server_score", "receiver_score", "game_state",
    # Opponent characteristics
    "opponent_skill_level", "opponent_style", "side", "intended_setup",
    # Engineered game-state flags
    "score_margin", "total_points_played_in_game",
    "is_tied", "is_trailing", "is_leading",
    "is_late_game", "is_deuce_or_later",
    "is_game_point_for_server", "is_game_point_against_server",
    # Serve combination identifiers
    "serve_spin_combo", "serve_length_spin_combo",
    "serve_placement_combo", "full_serve_combo",
    # Spin flags
    "is_heavy_spin", "is_low_spin",
    # Historical combination statistics
    "combo_attempts", "combo_win_rate", "combo_reliability",
]

N_SPLITS = 5
RANDOM_STATE = 42


@dataclass
class EvalResult:
    model_name: str
    accuracy: float
    accuracy_std: float
    roc_auc: float
    roc_auc_std: float


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """One-hot encode categoricals, standardize numerics (matches notebook 3)."""
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
    numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )


def _tune_lasso(preprocessor, X, y, groups, cv) -> Pipeline:
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            l1_ratio=1, solver="liblinear", max_iter=1000, class_weight="balanced"
        )),
    ])
    param_grid = {"classifier__C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]}
    search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    search.fit(X, y, groups=groups)
    return search.best_estimator_


def _tune_random_forest(preprocessor, X, y, groups, cv) -> Pipeline:
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"
        )),
    ])
    param_grid = {
        "classifier__max_depth": [3, 5, 7],
        "classifier__min_samples_leaf": [5, 10, 20],
    }
    search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
    search.fit(X, y, groups=groups)
    return search.best_estimator_


def _gradient_boosting(preprocessor) -> Pipeline:
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, random_state=RANDOM_STATE
        )),
    ])


def train_and_evaluate(
    df: pd.DataFrame,
    target_col: str = "point_won",
    models_dir: str = "models",
) -> List[EvalResult]:
    """Train and grouped-CV evaluate all models, then save pipelines to disk.

    Returns the per-model accuracy/ROC-AUC summary (the README results table).
    """
    feature_cols = [c for c in MODEL_FEATURES if c in df.columns]
    X = df[feature_cols]
    y = df[target_col]
    groups = df["match_id"]

    preprocessor = build_preprocessor(X)
    cv = GroupKFold(n_splits=N_SPLITS)

    results: List[EvalResult] = []

    # --- Baseline: always predict the majority class --------------------------
    baseline_accuracy = y.value_counts(normalize=True).max()
    results.append(EvalResult("Baseline", baseline_accuracy, 0.0, float("nan"), float("nan")))

    # --- Trainable models ------------------------------------------------------
    lasso_model = _tune_lasso(preprocessor, X, y, groups, cv)
    rf_model = _tune_random_forest(preprocessor, X, y, groups, cv)
    gb_model = _gradient_boosting(preprocessor)

    estimators: Dict[str, Pipeline] = {
        "LASSO (tuned)": lasso_model,
        "Random Forest (tuned)": rf_model,
        "Gradient Boosting": gb_model,
    }

    probs: Dict[str, np.ndarray] = {}
    for name, estimator in estimators.items():
        acc = cross_val_score(estimator, X, y, cv=cv, groups=groups, scoring="accuracy")
        auc = cross_val_score(estimator, X, y, cv=cv, groups=groups, scoring="roc_auc")
        results.append(
            EvalResult(name, acc.mean(), acc.std(), auc.mean(), auc.std())
        )
        probs[name] = cross_val_predict(
            estimator, X, y, cv=cv, groups=groups, method="predict_proba"
        )[:, 1]

    # --- Select the primary model by out-of-fold Brier score ------------------
    brier = {name: brier_score_loss(y, p) for name, p in probs.items()}
    best_name = min(brier, key=brier.get)

    # --- Refit on full data and persist ---------------------------------------
    os.makedirs(models_dir, exist_ok=True)
    for estimator in estimators.values():
        estimator.fit(X, y)

    joblib.dump(lasso_model, os.path.join(models_dir, "lasso_model.pkl"))
    joblib.dump(rf_model, os.path.join(models_dir, "rf_model.pkl"))
    joblib.dump(gb_model, os.path.join(models_dir, "gb_model.pkl"))
    joblib.dump(estimators[best_name], os.path.join(models_dir, "serve_win_probability_model.pkl"))
    joblib.dump(feature_cols, os.path.join(models_dir, "model_features.pkl"))

    return results


def results_to_frame(results: List[EvalResult]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": [r.model_name for r in results],
            "accuracy": [r.accuracy for r in results],
            "accuracy_std": [r.accuracy_std for r in results],
            "roc_auc": [r.roc_auc for r in results],
            "roc_auc_std": [r.roc_auc_std for r in results],
        }
    )


if __name__ == "__main__":
    processed = pd.read_csv("data/processed/table_tennis_serves_features.csv")
    summary = results_to_frame(train_and_evaluate(processed))
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print(summary.to_string(index=False))
