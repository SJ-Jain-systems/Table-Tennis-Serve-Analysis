"""Data cleaning utilities for table tennis serve analysis."""
from __future__ import annotations

import pandas as pd


def load_raw_data(path: str = "data/raw/table_tennis_serves.csv") -> pd.DataFrame:
    """Load raw serve-level dataset."""
    return pd.read_csv(path)


def clean_serves(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleanup: normalize string columns and enforce outcome target."""
    cleaned = df.copy()
    object_cols = cleaned.select_dtypes(include="object").columns
    for col in object_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip().str.lower()

    if "point_outcome" in cleaned.columns:
        cleaned["point_win"] = (cleaned["point_outcome"] == "win").astype(int)

    return cleaned


def save_processed_data(
    df: pd.DataFrame, path: str = "data/processed/serves_cleaned.csv"
) -> None:
    """Persist cleaned dataset to disk."""
    df.to_csv(path, index=False)


if __name__ == "__main__":
    raw = load_raw_data()
    cleaned = clean_serves(raw)
    save_processed_data(cleaned)
    print(f"Saved cleaned dataset with {len(cleaned)} rows.")
