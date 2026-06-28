"""Data cleaning utilities for table tennis serve analysis."""
from __future__ import annotations

import pandas as pd


def load_raw_data(path: str = "data/table_tennis_serves.csv") -> pd.DataFrame:
    """Load raw serve-level dataset."""
    return pd.read_csv(path)


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
