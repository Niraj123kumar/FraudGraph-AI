"""Preprocessing for the Kaggle Credit Card Fraud dataset.

All transformers are fit on the training split only. The test split is
never resampled.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

RANDOM_SEED: int = 42
TARGET_COL: str = "Class"
SCALE_COLS: list[str] = ["Time", "Amount"]
FEATURE_COLS: list[str] = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
EXPECTED_COLS: list[str] = [*FEATURE_COLS, TARGET_COL]


@dataclass(frozen=True)
class PreparedData:
    """Train/test splits with the scaler fitted on the training split."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    scaler: RobustScaler


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load the raw CSV and validate its schema.

    Raises:
        ValueError: If expected columns are missing or any value is null.
    """
    df = pd.read_csv(path)
    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    if df[EXPECTED_COLS].isnull().any().any():
        raise ValueError("Dataset contains null values.")
    return df[EXPECTED_COLS].copy()


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split on the target column."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )


def fit_scaler(X_train: pd.DataFrame) -> RobustScaler:
    """Fit a RobustScaler on Time and Amount of the training data only."""
    scaler = RobustScaler()
    scaler.fit(X_train[SCALE_COLS])
    return scaler


def apply_scaler(X: pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
    """Return a copy of X with Time and Amount scaled by a fitted scaler."""
    X_out = X.copy()
    X_out[SCALE_COLS] = scaler.transform(X_out[SCALE_COLS])
    return X_out


def prepare_data(
    path: str | Path,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> PreparedData:
    """Load, split, and scale the dataset.

    The scaler is fit on the training split and applied to both splits.
    """
    df = load_raw(path)
    X_train, X_test, y_train, y_test = split_data(df, test_size, seed)
    scaler = fit_scaler(X_train)
    return PreparedData(
        X_train=apply_scaler(X_train, scaler),
        X_test=apply_scaler(X_test, scaler),
        y_train=y_train,
        y_test=y_test,
        scaler=scaler,
    )
