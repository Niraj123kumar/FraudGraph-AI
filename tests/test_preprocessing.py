"""Tests for src/data/preprocessing.py."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    EXPECTED_COLS,
    SCALE_COLS,
    TARGET_COL,
    apply_scaler,
    fit_scaler,
    load_raw,
    split_data,
)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 2000
    df = pd.DataFrame(rng.normal(size=(n, len(EXPECTED_COLS))),
                      columns=EXPECTED_COLS)
    df["Amount"] = rng.exponential(scale=80, size=n)
    df["Time"] = np.arange(n, dtype=float)
    df[TARGET_COL] = 0
    df.loc[rng.choice(n, size=20, replace=False), TARGET_COL] = 1
    return df


def test_split_is_stratified(synthetic_df: pd.DataFrame) -> None:
    _, _, y_train, y_test = split_data(synthetic_df)
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.002)


def test_split_is_deterministic(synthetic_df: pd.DataFrame) -> None:
    a = split_data(synthetic_df)
    b = split_data(synthetic_df)
    pd.testing.assert_frame_equal(a[0], b[0])
    pd.testing.assert_frame_equal(a[1], b[1])


def test_split_sizes_and_no_overlap(synthetic_df: pd.DataFrame) -> None:
    X_train, X_test, _, _ = split_data(synthetic_df, test_size=0.2)
    assert len(X_test) == 400
    assert len(X_train) + len(X_test) == len(synthetic_df)
    assert set(X_train.index).isdisjoint(X_test.index)


def test_scaler_fit_on_train_only(synthetic_df: pd.DataFrame) -> None:
    X_train, X_test, _, _ = split_data(synthetic_df)
    scaler = fit_scaler(X_train)
    np.testing.assert_allclose(
        scaler.center_, X_train[SCALE_COLS].median().to_numpy())
    # Test split is transformed with train statistics, so its median is not forced to 0.
    scaled_test = apply_scaler(X_test, scaler)
    assert scaled_test["Amount"].median() != pytest.approx(0.0, abs=1e-12)


def test_apply_scaler_does_not_mutate_input(synthetic_df: pd.DataFrame) -> None:
    X_train, _, _, _ = split_data(synthetic_df)
    original = X_train.copy()
    apply_scaler(X_train, fit_scaler(X_train))
    pd.testing.assert_frame_equal(X_train, original)


def test_load_raw_rejects_missing_columns(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    pd.DataFrame({"Time": [0.0], "Amount": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing expected columns"):
        load_raw(path)
