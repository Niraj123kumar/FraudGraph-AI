"""Train and publish the credit-card fraud model."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data.preprocessing import FEATURE_COLS, RANDOM_SEED, load_raw, split_data
from src.ml.evaluate import run_evaluation
from src.ml.evaluate import _calibration_details

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "models"
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

MODEL_CONFIG: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _library_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("xgboost", "scikit-learn", "numpy", "pandas", "pyarrow"):
        try:
            versions[package.replace("-", "_")] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package.replace("-", "_")] = "unknown"
    return versions


def _fit_model(X, y, seed: int, config: dict[str, Any]) -> XGBClassifier:
    positives = int(np.count_nonzero(np.asarray(y) == 1))
    negatives = int(np.count_nonzero(np.asarray(y) == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("Training data must contain both fraud and legitimate rows")
    params = {
        **config,
        "random_state": seed,
        "scale_pos_weight": negatives / positives,
    }
    model = XGBClassifier(**params)
    model.fit(X, y)
    return model


def _threshold_for_f1(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    thresholds = np.unique(np.concatenate((np.array([0.0, 1.0]), probabilities)))
    scores = [f1_score(y_true, probabilities >= threshold, zero_division=0) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])


def _threshold_for_precision(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    target_precision: float,
) -> float:
    if not 0 < target_precision <= 1:
        raise ValueError("target_precision must be greater than 0 and at most 1")
    thresholds = np.unique(np.concatenate((np.array([0.0, 1.0]), probabilities)))
    eligible = []
    for threshold in thresholds:
        predicted = probabilities >= threshold
        precision = precision_score(y_true, predicted, zero_division=0)
        if precision >= target_precision:
            eligible.append((recall_score(y_true, predicted, zero_division=0), threshold))
    if not eligible:
        raise ValueError("No validation threshold reaches target_precision")
    return float(max(eligible)[1])


def choose_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    objective: str = "f1",
    target_precision: float | None = None,
) -> float:
    if objective == "f1":
        return _threshold_for_f1(y_true, probabilities)
    if objective == "target_precision":
        if target_precision is None:
            raise ValueError("target_precision is required for that threshold objective")
        return _threshold_for_precision(y_true, probabilities, target_precision)
    raise ValueError(f"Unsupported threshold objective: {objective}")


def _test_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def train_model(
    data_path: Path,
    output_dir: Path,
    *,
    force: bool = False,
    threshold_objective: str = "f1",
    target_precision: float | None = None,
    seed: int = RANDOM_SEED,
) -> dict[str, Path]:
    """Train model artifacts in a temporary directory and publish them."""
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    artifact_names = ("transaction_model.joblib", "model_metadata.json", "evaluation_report.json")
    existing = [output_dir / name for name in artifact_names if (output_dir / name).exists()]
    if existing and not force:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing artifacts: {names}; use --force")

    frame = load_raw(data_path)
    X_train, X_test, y_train, y_test = split_data(frame, test_size=TEST_SIZE, seed=seed)
    X_fit, X_validation, y_fit, y_validation = train_test_split(
        X_train, y_train, test_size=VALIDATION_SIZE, stratify=y_train, random_state=seed
    )
    validation_model = _fit_model(X_fit, y_fit, seed, MODEL_CONFIG)
    threshold = choose_threshold(
        y_validation.to_numpy(),
        np.asarray(validation_model.predict_proba(X_validation)[:, 1], dtype=float),
        threshold_objective,
        target_precision,
    )
    calibration, _ = _calibration_details(
        y_validation.to_numpy(),
        np.asarray(validation_model.predict_proba(X_validation)[:, 1], dtype=float),
    )
    calibration_x = np.asarray(calibration["raw_probabilities"], dtype=float)
    calibration_y = np.asarray(calibration["calibrated_probabilities"], dtype=float)
    calibrated_cutoffs = np.interp([0.30, 0.70], calibration_x, calibration_y).tolist()
    final_model = _fit_model(X_train, y_train, seed, MODEL_CONFIG)
    test_probabilities = np.asarray(final_model.predict_proba(X_test)[:, 1], dtype=float)
    dataset_hash = _sha256_file(data_path)
    metadata = {
        "model_name": "xgb_weighted",
        "model_version": "trained-local",
        "training_date_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": seed,
        "threshold": threshold,
        "threshold_objective": threshold_objective,
        "target_precision": target_precision,
        "threshold_selection": "stratified validation split from training data; test set not used",
        "feature_names": FEATURE_COLS,
        "hyperparameters": {
            **MODEL_CONFIG,
            "scale_pos_weight": int(np.count_nonzero(y_train == 0))
            / int(np.count_nonzero(y_train == 1)),
        },
        "n_train": len(X_train),
        "n_validation_for_threshold": len(X_validation),
        "n_test": len(X_test),
        "dataset_sha256": dataset_hash,
        "library_versions": _library_versions(),
        "test_metrics": _test_metrics(y_test.to_numpy(), test_probabilities, threshold),
        "calibration": calibration,
        "risk_bands": {
            "low_max_exclusive": float(calibrated_cutoffs[0]),
            "medium_max_exclusive": float(calibrated_cutoffs[1]),
            "probability_space": "calibrated isotonic probability",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".fraudgraph-train-", dir=output_dir))
    try:
        model_path = temporary_dir / artifact_names[0]
        metadata_path = temporary_dir / artifact_names[1]
        report_path = temporary_dir / artifact_names[2]
        joblib.dump(final_model, model_path)
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        run_evaluation(data_path=data_path, model_path=model_path, metadata_path=metadata_path, report_path=report_path)
        for name in artifact_names:
            os.replace(temporary_dir / name, output_dir / name)
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)
    return {name: output_dir / name for name in artifact_names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out-dir", "--out", dest="output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--threshold-objective", choices=("f1", "target_precision"), default="f1")
    parser.add_argument("--target-precision", type=float)
    args = parser.parse_args()
    train_model(
        args.data,
        args.output_dir,
        force=args.force,
        threshold_objective=args.threshold_objective,
        target_precision=args.target_precision,
    )


if __name__ == "__main__":
    main()
