"""Read-only access to the labeled credit-card dataset for the API."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.preprocessing import FEATURE_COLS, TARGET_COL

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
DATA_SOURCE = "Kaggle mlg-ulb/creditcardfraud"
PARQUET_COLUMNS = FEATURE_COLS + [TARGET_COL]
PRECOMPUTED_PATH = PROJECT_ROOT / "models" / "transactions.parquet"


class DatasetUnavailableError(RuntimeError):
    """Raised when the transaction dataset is missing or invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parquet_path(path: Path) -> Path:
    return path.with_suffix(".parquet")


def _load_or_create_parquet(path: Path) -> pd.DataFrame:
    parquet = _parquet_path(path)
    source_mtime = path.stat().st_mtime_ns
    if parquet.is_file() and parquet.stat().st_mtime_ns >= source_mtime:
        try:
            return pd.read_parquet(parquet, columns=PARQUET_COLUMNS)
        except (OSError, ValueError, ImportError):
            parquet.unlink(missing_ok=True)

    frame = pd.read_csv(path, usecols=PARQUET_COLUMNS)
    temporary = parquet.with_name(f".{parquet.name}.{os.getpid()}.part")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(parquet)
    finally:
        temporary.unlink(missing_ok=True)
    return frame


@lru_cache(maxsize=4)
def _load_transaction_frame(
    path_string: str, mtime_ns: int, size: int
) -> tuple[pd.DataFrame, str]:
    """Load and validate the raw dataset once, preserving its source row order."""
    path = Path(path_string)
    if not path.is_file():
        raise DatasetUnavailableError("Transaction dataset is not available")

    try:
        dataset_sha256 = _sha256_file(path)
        frame = _load_or_create_parquet(path)
        numeric = frame[FEATURE_COLS + [TARGET_COL]].to_numpy(dtype=float)
    except (OSError, ValueError, TypeError) as exc:
        raise DatasetUnavailableError("Transaction dataset could not be loaded") from exc

    if not np.isfinite(numeric).all():
        raise DatasetUnavailableError("Transaction dataset contains non-finite values")
    if not set(frame[TARGET_COL].unique()).issubset({0, 1}):
        raise DatasetUnavailableError("Transaction labels must be binary")
    if (frame[["Time", "Amount"]] < 0).any().any():
        raise DatasetUnavailableError("Transaction time and amount must be non-negative")

    dataset_version = f"creditcard-{dataset_sha256[:16]}"
    frame = frame.copy()
    frame.insert(0, "source_row", np.arange(1, len(frame) + 1, dtype=np.int64))
    frame.insert(
        0,
        "transaction_id",
        [f"{dataset_version}:row:{row:09d}" for row in frame["source_row"]],
    )
    return frame, dataset_sha256


def load_transaction_frame(path_string: str = str(DATA_PATH)) -> tuple[pd.DataFrame, str]:
    """Load a cached frame and invalidate it when the source file changes."""
    path = Path(path_string)
    try:
        stat = path.stat()
    except OSError:
        raise DatasetUnavailableError("Transaction dataset is not available") from None
    return _load_transaction_frame(path_string, stat.st_mtime_ns, stat.st_size)


load_transaction_frame.cache_clear = _load_transaction_frame.cache_clear


def dataset_version_for(dataset_sha256: str) -> str:
    return f"creditcard-{dataset_sha256[:16]}"


def transaction_record(row: pd.Series, dataset_version: str) -> dict:
    return {
        "transaction_id": str(row["transaction_id"]),
        "dataset_version": dataset_version,
        "source_row": int(row["source_row"]),
        "Time": float(row["Time"]),
        "Amount": float(row["Amount"]),
        "Class": int(row[TARGET_COL]),
    }


def get_transaction_frame() -> tuple[pd.DataFrame, str, str]:
    if os.getenv("FRAUDGRAPH_LAPTOP_MODE", "1") == "1":
        if not PRECOMPUTED_PATH.is_file():
            raise DatasetUnavailableError(
                "Precomputed transactions.parquet is unavailable; build it on Colab and copy it into models/."
            )
        try:
            frame = pd.read_parquet(
                PRECOMPUTED_PATH,
                columns=["transaction_id", "source_row", *PARQUET_COLUMNS],
            )
            dataset_sha256 = hashlib.sha256(PRECOMPUTED_PATH.read_bytes()).hexdigest()
        except (OSError, ValueError, ImportError, KeyError) as exc:
            raise DatasetUnavailableError(
                "Precomputed transactions.parquet could not be loaded; rebuild it on Colab."
            ) from exc
        return frame, dataset_sha256, dataset_version_for(dataset_sha256)
    frame, dataset_sha256 = load_transaction_frame()
    return frame, dataset_sha256, dataset_version_for(dataset_sha256)
