from __future__ import annotations

import json
import importlib.metadata
import logging
from collections.abc import Callable, Mapping
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data.preprocessing import FEATURE_COLS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "transaction_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
FEATURE_NAMES = list(FEATURE_COLS)
logger = logging.getLogger(__name__)


def _running_library_versions() -> dict[str, str]:
    versions = {}
    for package in ("xgboost", "scikit-learn", "numpy", "pandas", "pyarrow"):
        key = package.replace("-", "_")
        try:
            versions[key] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[key] = "unknown"
    return versions


class ModelStore:
    """Lazily load and cache a model snapshot until its files change."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        metadata_path: Path = METADATA_PATH,
        model_loader: Callable[[Path], object] = joblib.load,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.model_loader = model_loader
        self._cache_key: tuple[int, int, int, int] | None = None
        self._model: object | None = None
        self._metadata: dict = {}

    def _file_key(self) -> tuple[int, int, int, int]:
        model_stat = self.model_path.stat()
        metadata_stat = self.metadata_path.stat()
        return (
            model_stat.st_mtime_ns,
            model_stat.st_size,
            metadata_stat.st_mtime_ns,
            metadata_stat.st_size,
        )

    def _load_snapshot(self) -> None:
        key = self._file_key()
        if self._cache_key == key:
            return
        with self.metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if not isinstance(metadata, dict):
            raise ValueError("Model metadata must be a JSON object")
        feature_names = metadata.get("feature_names")
        if not isinstance(feature_names, list) or not feature_names:
            raise ValueError("Model metadata does not contain a valid feature schema")
        if feature_names != FEATURE_NAMES:
            raise ValueError("Model metadata feature schema does not match the required feature order")
        try:
            threshold = float(metadata["threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Model metadata does not contain a valid threshold") from exc
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Model metadata contains an invalid threshold")
        recorded_versions = metadata.get("library_versions")
        if isinstance(recorded_versions, dict):
            mismatches = []
            for name, running in _running_library_versions().items():
                recorded = recorded_versions.get(name)
                if recorded is not None and recorded != running:
                    mismatches.append(f"{name}={recorded} (running {running})")
            if mismatches:
                logger.warning(
                    "Saved model library versions differ from the running environment: %s",
                    ", ".join(mismatches),
                )
        self._model = self.model_loader(self.model_path)
        self._metadata = metadata
        self._cache_key = key

    def get_model(self) -> object:
        self._load_snapshot()
        if self._model is None:
            raise FileNotFoundError("Saved prediction model is not available")
        return self._model

    def get_metadata(self) -> dict:
        self._load_snapshot()
        return self._metadata

    def get_feature_names(self) -> list[str]:
        return list(self.get_metadata()["feature_names"])

    def get_threshold(self) -> float:
        return float(self.get_metadata()["threshold"])

    def is_ready(self) -> bool:
        try:
            self._load_snapshot()
        except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
            return False
        return self._model is not None


model_store = ModelStore()


def coerce_feature_frame(features: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Convert model features to numeric values and identify invalid rows."""
    if list(features.columns) != FEATURE_NAMES:
        raise ValueError("Feature columns do not match the saved credit-card model schema")
    numeric = features.apply(pd.to_numeric, errors="coerce")
    invalid = ~np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))
    return numeric, invalid


def coerce_feature_values(features: Mapping[str, object]) -> dict[str, float]:
    """Convert one model feature record to finite numeric values."""
    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing:
        raise ValueError(f"Missing features: {', '.join(missing)}")
    frame = pd.DataFrame(
        [[features[name] for name in FEATURE_NAMES]],
        columns=FEATURE_NAMES,
    )
    numeric, invalid = coerce_feature_frame(frame)
    if invalid.any():
        bad_columns = [
            name for index, name in enumerate(FEATURE_NAMES) if invalid[0, index]
        ]
        raise ValueError(
            f"Feature values must be numeric and finite: {', '.join(bad_columns)}"
        )
    return {name: float(numeric.iloc[0][name]) for name in FEATURE_NAMES}


def _snapshot() -> tuple[object, float, list[str]]:
    model = model_store.get_model()
    threshold = model_store.get_threshold()
    feature_names = model_store.get_feature_names()
    if feature_names != FEATURE_NAMES:
        raise ValueError("Saved model metadata feature schema is incompatible")
    return model, threshold, feature_names


def _calibrate_probability(probability: float, metadata: dict) -> float:
    calibration = metadata.get("calibration", {})
    raw = calibration.get("raw_probabilities")
    calibrated = calibration.get("calibrated_probabilities")
    if not isinstance(raw, list) or not isinstance(calibrated, list) or len(raw) != len(calibrated) or not raw:
        return probability
    return float(np.interp(probability, np.asarray(raw, dtype=float), np.asarray(calibrated, dtype=float)))


def _risk_level(probability: float, metadata: dict) -> str:
    bands = metadata.get("risk_bands", {})
    low = float(bands.get("low_max_exclusive", 0.30))
    medium = float(bands.get("medium_max_exclusive", 0.70))
    return "low" if probability < low else "medium" if probability < medium else "high"


def predict_transaction(features: dict) -> dict:
    model, threshold, _ = _snapshot()
    metadata = model_store.get_metadata()
    input_data = pd.DataFrame([coerce_feature_values(features)], columns=FEATURE_NAMES)
    probability = float(model.predict_proba(input_data)[0][1])

    if not 0 <= probability <= 1:
        raise ValueError("Model returned an invalid probability")

    is_fraud = probability >= threshold
    risk_level = _risk_level(_calibrate_probability(probability, metadata), metadata)
    return {
        "probability": probability,
        "is_fraud": is_fraud,
        "risk_level": risk_level,
    }


def predict_transactions_batch(features: pd.DataFrame) -> list[dict]:
    """Run inference for one bounded batch in the saved model's exact schema."""
    model, threshold, _ = _snapshot()
    metadata = model_store.get_metadata()
    numeric, invalid = coerce_feature_frame(features)
    if invalid.any():
        bad_columns = [
            name for index, name in enumerate(FEATURE_NAMES)
            if invalid[:, index].any()
        ]
        raise ValueError(
            f"Feature values must be numeric and finite: {', '.join(bad_columns)}"
        )
    probabilities = model.predict_proba(numeric)[:, 1]
    results = []
    for raw_probability in probabilities:
        probability = float(raw_probability)
        if not 0 <= probability <= 1:
            raise ValueError("Model returned an invalid probability")
        risk_level = _risk_level(_calibrate_probability(probability, metadata), metadata)
        results.append({
            "probability": probability,
            "is_fraud": probability >= threshold,
            "risk_level": risk_level,
        })
    return results
