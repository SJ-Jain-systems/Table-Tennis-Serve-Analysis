"""Model training/evaluation utilities with match-level validation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PRE_POINT_FEATURES = [
    "serve_type",
    "spin_type",
    "spin_intensity",
    "serve_length",
    "placement_zone",
    "toss_height",
    "contact_point",
    "game_state",
    "server_score",
    "receiver_score",
    "opponent_skill_level",
    "opponent_style",
    "side",
    "score_margin",
    "is_pressure_point",
    "is_short_heavy_backspin",
    "serve_combo",
    "opponent_is_attacker",
]


@dataclass
class EvalResult:
    model_name: str
    accuracy: float
    roc_auc: float
    precision: float
    recall: float
    logloss: float


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def evaluate_with_logo(df: pd.DataFrame, target_col: str = "point_win") -> List[EvalResult]:
    modeling_df = df.dropna(subset=["match_id", target_col]).copy()
    feature_cols = [c for c in PRE_POINT_FEATURES if c in modeling_df.columns]
    X = modeling_df[feature_cols]
    y = modeling_df[target_col]
    groups = modeling_df["match_id"]

    preprocessor = build_preprocessor(X)

    models: Dict[str, object] = {
        "logistic_l1": LogisticRegression(penalty="l1", solver="liblinear", max_iter=500),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
    }

    results: List[EvalResult] = []
    logo = LeaveOneGroupOut()

    for model_name, estimator in models.items():
        y_true_all: List[int] = []
        y_prob_all: List[float] = []
        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            pipeline = Pipeline([("prep", preprocessor), ("model", estimator)])
            pipeline.fit(X_train, y_train)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            y_true_all.extend(y_test.tolist())
            y_prob_all.extend(y_prob.tolist())

        y_pred_all = (np.array(y_prob_all) >= 0.5).astype(int)
        results.append(
            EvalResult(
                model_name=model_name,
                accuracy=accuracy_score(y_true_all, y_pred_all),
                roc_auc=roc_auc_score(y_true_all, y_prob_all),
                precision=precision_score(y_true_all, y_pred_all, zero_division=0),
                recall=recall_score(y_true_all, y_pred_all, zero_division=0),
                logloss=log_loss(y_true_all, y_prob_all),
            )
        )

    return results
