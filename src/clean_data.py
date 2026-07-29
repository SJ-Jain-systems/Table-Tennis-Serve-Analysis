"""Data cleaning utilities for table tennis serve analysis."""
from __future__ import annotations

import pandas as pd

# --- Raw-input contract ------------------------------------------------------
# The pipeline reads a serve-level CSV whose shape the downstream steps
# (feature engineering, modeling, recommendation) assume without re-checking.
# ``validate_raw_schema`` turns a malformed or renamed input into a clear,
# early "column X missing / value out of range" error instead of a cryptic
# crash deep inside feature engineering.

REQUIRED_COLUMNS = [
    "serve_type", "spin_type", "spin_intensity", "serve_length", "placement_zone",
    "toss_height", "contact_point", "match_id", "game_number", "server_score",
    "receiver_score", "game_state", "opponent_id", "opponent_skill_level",
    "opponent_style", "side", "return_type", "return_quality", "return_placement",
    "rally_length", "point_outcome", "point_end_type", "intended_setup",
    "rally_type_achieved", "chop_rally_outcome",
]

# Inclusive (min, max) bounds for numeric columns; ``None`` means unbounded on
# that side. Ranges reflect the values observed in the shipped dataset (e.g.
# spin intensity is a 1-3 scale, return quality a 0-3 scale).
NUMERIC_RANGES = {
    "spin_intensity": (1, 3),
    "game_number": (1, None),
    "server_score": (0, None),
    "receiver_score": (0, None),
    "return_quality": (0, 3),
    "rally_length": (1, None),
    "match_id": (1, None),
    "opponent_id": (1, None),
}

# Categorical columns restricted to a known set of values (compared
# case-insensitively, since ``clean_serves`` later lower-cases string columns).
CATEGORICAL_DOMAINS = {
    "point_outcome": {"won", "lost"},
}


def validate_raw_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the raw serve dataset against the pipeline's input contract.

    Checks that all expected columns are present, that numeric columns are
    numeric and within their expected ranges, and that constrained categorical
    columns only contain known values. Raises ``ValueError`` naming the
    offending column on the first violation; returns ``df`` unchanged on success
    so it can be used inline (``df = validate_raw_schema(df)``).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Raw data is missing required column(s): {', '.join(missing)}"
        )

    for col, (low, high) in NUMERIC_RANGES.items():
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.isna().any():
            bad_count = int(numeric.isna().sum())
            raise ValueError(
                f"Column '{col}' must be numeric, but {bad_count} value(s) "
                "could not be parsed as numbers."
            )
        if low is not None and (numeric < low).any():
            raise ValueError(
                f"Column '{col}' has value(s) below the expected minimum {low} "
                f"(min found: {numeric.min()})."
            )
        if high is not None and (numeric > high).any():
            raise ValueError(
                f"Column '{col}' has value(s) above the expected maximum {high} "
                f"(max found: {numeric.max()})."
            )

    for col, allowed in CATEGORICAL_DOMAINS.items():
        observed = set(df[col].astype(str).str.strip().str.lower().unique())
        unexpected = observed - allowed
        if unexpected:
            raise ValueError(
                f"Column '{col}' contains unexpected value(s): "
                f"{', '.join(sorted(unexpected))}. Expected only: "
                f"{', '.join(sorted(allowed))}."
            )

    return df


def load_raw_data(path: str = "data/table_tennis_serves.csv") -> pd.DataFrame:
    """Load and schema-validate the raw serve-level dataset."""
    return validate_raw_schema(pd.read_csv(path))


def clean_serves(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleanup: normalize string columns and derive the binary target.

    The raw dataset encodes the outcome in ``point_outcome`` as ``"won"`` /
    ``"lost"``. The engineered pipeline (notebook 2, ``features.add_features``)
    and the models use a binary column named ``point_won``, so this derives the
    same column here for consistency.
    """
    cleaned = df.copy()
    object_cols = cleaned.select_dtypes(include="object").columns
    for col in object_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip().str.lower()

    if "point_outcome" in cleaned.columns:
        cleaned["point_won"] = (cleaned["point_outcome"] == "won").astype(int)

    return cleaned


def save_processed_data(
    df: pd.DataFrame,
    path: str = "data/processed/table_tennis_serves_features.csv",
) -> None:
    """Persist cleaned dataset to disk."""
    df.to_csv(path, index=False)


if __name__ == "__main__":
    raw = load_raw_data()
    cleaned = clean_serves(raw)
    save_processed_data(cleaned)
    print(f"Saved cleaned dataset with {len(cleaned)} rows.")
