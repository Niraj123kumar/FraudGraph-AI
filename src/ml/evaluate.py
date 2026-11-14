"""Reproduce the saved model's test-only evaluation on the documented split."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from src.data.preprocessing import FEATURE_COLS, RANDOM_SEED, TARGET_COL, load_raw, split_data
from src.data.transactions import DATA_SOURCE, dataset_version_for

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "creditcard.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "transaction_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
REPORT_PATH = PROJECT_ROOT / "models" / "evaluation_report.json"
TEST_SIZE = 0.2


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unavailable_report(reason: str) -> dict:
    return {
        "available": False,
        "reason": reason,
        "evaluation_scope": "internal stratified holdout from the same dataset",
        "evaluation_date_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": DATA_SOURCE,
        "dataset_version": None,
        "dataset_sha256": None,
        "model_name": None,
        "model_version": None,
        "model_sha256": None,
        "threshold": None,
        "split": None,
        "sample_count": None,
        "positive_count": None,
        "negative_count": None,
        "feature_schema": FEATURE_COLS,
        "metrics": None,
        "confusion_matrix": None,
        "curves": None,
    }


def _threshold_curve_points(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    decision_threshold: float,
    max_points: int = 300,
) -> dict:
    """Create bounded ROC and PR samples from real scores, retaining model cutoff."""
    if len(y_true) != len(probabilities) or len(probabilities) == 0 or max_points < 4:
        raise ValueError("Evaluation curve inputs must align and contain at least four points")
    quantile_count = max_points - 3
    quantile_thresholds = np.quantile(
        probabilities, np.linspace(0.0, 1.0, num=quantile_count)
    )
    thresholds = np.unique(np.concatenate((
        quantile_thresholds,
        np.array([decision_threshold, np.nextafter(np.max(probabilities), np.inf), np.nextafter(np.min(probabilities), -np.inf)]),
    )))[::-1]

    positive_count = int(np.count_nonzero(y_true == 1))
    negative_count = int(len(y_true) - positive_count)
    roc = {"false_positive_rate": [], "true_positive_rate": []}
    precision_recall = {"recall": [], "precision": []}
    for threshold_value in thresholds:
        predicted = probabilities >= threshold_value
        true_positive = int(np.count_nonzero(predicted & (y_true == 1)))
        false_positive = int(np.count_nonzero(predicted & (y_true == 0)))
        true_negative = negative_count - false_positive
        false_negative = positive_count - true_positive
        recall_value = true_positive / positive_count if positive_count else 0.0
        roc["false_positive_rate"].append(false_positive / negative_count if negative_count else 0.0)
        roc["true_positive_rate"].append(recall_value)
        precision_recall["recall"].append(recall_value)
        precision_recall["precision"].append(true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0)
    return {"roc": roc, "precision_recall": precision_recall}


def _metadata_consistency(
    metadata: dict, model: object, threshold: float, y_test, probabilities: np.ndarray
) -> str | None:
    if metadata.get("feature_names") != FEATURE_COLS:
        return "Saved model metadata feature schema does not match the dataset"
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        return "Saved model threshold is invalid"

    try:
        booster = model.get_booster()
        model_features = booster.feature_names
    except AttributeError:
        model_features = getattr(model, "feature_names_in_", None)
    if model_features is not None and list(model_features) != FEATURE_COLS:
        return "Saved model feature schema does not match the dataset"

    if metadata.get("n_test") != len(y_test):
        return "Held-out split size does not match saved model metadata"

    predicted = probabilities >= threshold
    matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
    expected = metadata.get("test_metrics")
    if not isinstance(expected, dict):
        return "Saved model metadata does not contain reference test results"
    if expected.get("confusion_matrix") != matrix.tolist():
        return "Saved model predictions do not reproduce the recorded test confusion matrix"
    if not np.isclose(float(expected.get("threshold", -1)), threshold, atol=1e-12):
        return "Saved model threshold does not match the recorded test threshold"
    try:
        recorded_roc_auc = float(expected["roc_auc"])
    except (KeyError, TypeError, ValueError):
        return "Saved model metadata does not contain a valid recorded ROC-AUC"
    actual_roc_auc = float(roc_auc_score(y_test, probabilities))
    if not np.isclose(recorded_roc_auc, actual_roc_auc, atol=1e-6):
        return "Saved model predictions do not reproduce the recorded test ROC-AUC"
    return None


def _calibration_details(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    evaluation_y: np.ndarray | None = None,
    evaluation_probabilities: np.ndarray | None = None,
) -> tuple[dict, dict]:
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(probabilities, y_true)
    score_y = y_true if evaluation_y is None else evaluation_y
    score_probabilities = probabilities if evaluation_probabilities is None else evaluation_probabilities
    calibrated = np.asarray(calibrator.predict(score_probabilities), dtype=float)
    fraction, mean = calibration_curve(score_y, calibrated, n_bins=10, strategy="quantile")
    return (
        {
            "method": "isotonic",
            "raw_probabilities": [float(value) for value in calibrator.X_thresholds_],
            "calibrated_probabilities": [float(value) for value in calibrator.y_thresholds_],
        },
        {
            "brier_score": float(np.mean((calibrated - score_y) ** 2)),
            "reliability_curve": {
                "mean_predicted_probability": [float(value) for value in mean],
                "fraction_positive": [float(value) for value in fraction],
            },
        },
    )


def _threshold_for_f1(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    thresholds = np.unique(np.concatenate((np.array([0.0, 1.0]), probabilities)))
    scores = [
        f1_score(y_true, probabilities >= threshold, zero_division=0)
        for threshold in thresholds
    ]
    return float(thresholds[int(np.argmax(scores))])


def _internal_validation(
    frame,
    model,
    metadata: dict,
    threshold: float,
) -> tuple[dict, dict]:
    X = frame[FEATURE_COLS]
    y = frame[TARGET_COL].to_numpy()
    config = dict(metadata.get("hyperparameters", {}))
    config.pop("scale_pos_weight", None)
    config.pop("random_state", None)
    fold_metrics = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for fold, (train_indices, test_indices) in enumerate(splitter.split(X, y), start=1):
        positives = max(1, int(np.count_nonzero(y[train_indices] == 1)))
        negatives = int(np.count_nonzero(y[train_indices] == 0))
        fold_model = XGBClassifier(
            **config,
            scale_pos_weight=negatives / positives,
            random_state=RANDOM_SEED,
        )
        fold_model.fit(X.iloc[train_indices], y[train_indices])
        probabilities = np.asarray(fold_model.predict_proba(X.iloc[test_indices])[:, 1], dtype=float)
        fold_metrics.append({
            "fold": fold,
            "roc_auc": float(roc_auc_score(y[test_indices], probabilities)),
            "average_precision": float(average_precision_score(y[test_indices], probabilities)),
            "recall_at_threshold": float(recall_score(y[test_indices], probabilities >= threshold, zero_division=0)),
        })

    def summary(name: str) -> dict:
        values = np.asarray([item[name] for item in fold_metrics], dtype=float)
        return {"mean": float(values.mean()), "std": float(values.std(ddof=1)), "values": values.tolist()}

    time_order = np.argsort(frame["Time"].to_numpy())
    split_index = max(1, int(len(frame) * 0.8))
    time_train, time_test = time_order[:split_index], time_order[split_index:]
    time_train_indices, time_validation_indices = train_test_split(
        time_train,
        test_size=0.2,
        stratify=y[time_train],
        random_state=RANDOM_SEED,
    )
    config.pop("eval_metric", None)
    positives = max(1, int(np.count_nonzero(y[time_train_indices] == 1)))
    negatives = int(np.count_nonzero(y[time_train_indices] == 0))
    time_validation_model = XGBClassifier(
        **config,
        scale_pos_weight=negatives / positives,
        random_state=RANDOM_SEED,
    )
    time_validation_model.fit(X.iloc[time_train_indices], y[time_train_indices])
    validation_probabilities = np.asarray(
        time_validation_model.predict_proba(X.iloc[time_validation_indices])[:, 1],
        dtype=float,
    )
    time_threshold = _threshold_for_f1(
        y[time_validation_indices], validation_probabilities
    )
    positives = max(1, int(np.count_nonzero(y[time_train] == 1)))
    negatives = int(np.count_nonzero(y[time_train] == 0))
    time_model = XGBClassifier(
        **config,
        scale_pos_weight=negatives / positives,
        random_state=RANDOM_SEED,
    )
    time_model.fit(X.iloc[time_train], y[time_train])
    time_probabilities = np.asarray(time_model.predict_proba(X.iloc[time_test])[:, 1], dtype=float)
    time_report = {
        "strategy": "time-ordered split by Time; earliest 80% train, latest 20% test; threshold selected within earliest 80%",
        "training_count": int(len(time_train)),
        "threshold_selection_training_count": int(len(time_train_indices)),
        "threshold_selection_validation_count": int(len(time_validation_indices)),
        "test_count": int(len(time_test)),
        "threshold": float(time_threshold),
        "roc_auc": float(roc_auc_score(y[time_test], time_probabilities)),
        "average_precision": float(average_precision_score(y[time_test], time_probabilities)),
        "recall_at_threshold": float(recall_score(y[time_test], time_probabilities >= time_threshold, zero_division=0)),
    }
    return (
        {
            "label": "internal cross-validation; not an external validation estimate",
            "folds": fold_metrics,
            "roc_auc": summary("roc_auc"),
            "average_precision": summary("average_precision"),
            "recall_at_threshold": summary("recall_at_threshold"),
        },
        time_report,
    )


def run_evaluation(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
    report_path: Path = REPORT_PATH,
) -> dict:
    """Compute held-out metrics only when data, split, and saved model agree."""
    try:
        for path, label in (
            (data_path, "Dataset"),
            (model_path, "Saved model"),
            (metadata_path, "Model metadata"),
        ):
            if not path.is_file():
                raise FileNotFoundError(f"{label} is not available")

        dataset_sha256 = _sha256_file(data_path)
        model_sha256 = _sha256_file(model_path)
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        if not isinstance(metadata, dict):
            raise ValueError("Model metadata must be a JSON object")

        frame = load_raw(data_path)
        X_train, X_test, y_train, y_test = split_data(
            frame, test_size=TEST_SIZE, seed=RANDOM_SEED
        )
        if len(X_train) != metadata.get("n_train"):
            raise ValueError("Training partition size does not match saved metadata")

        threshold = float(metadata["threshold"])
        if metadata.get("feature_names") != FEATURE_COLS:
            raise ValueError("Saved model metadata feature schema does not match the dataset")

        model = joblib.load(model_path)
        if not callable(getattr(model, "predict_proba", None)):
            raise ValueError("Saved model does not provide probability predictions")
        probabilities = np.asarray(
            model.predict_proba(X_test[FEATURE_COLS])[:, 1], dtype=float
        )
        if probabilities.shape != (len(y_test),) or not np.isfinite(probabilities).all():
            raise ValueError("Saved model returned invalid probabilities")
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Saved model returned probabilities outside [0, 1]")

        mismatch = _metadata_consistency(metadata, model, threshold, y_test, probabilities)
        if mismatch:
            raise ValueError(mismatch)

        predictions = probabilities >= threshold
        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
        curve_points = _threshold_curve_points(y_test.to_numpy(), probabilities, threshold)
        _, X_validation, _, y_validation = train_test_split(
            X_train,
            y_train,
            test_size=0.2,
            stratify=y_train,
            random_state=RANDOM_SEED,
        )
        validation_probabilities = np.asarray(
            model.predict_proba(X_validation[FEATURE_COLS])[:, 1], dtype=float
        )
        calibration, calibration_metrics = _calibration_details(
            y_validation.to_numpy(),
            validation_probabilities,
            y_test.to_numpy(),
            probabilities,
        )
        cross_validation, time_ordered = _internal_validation(frame, model, metadata, threshold)
        report = {
            "available": True,
            "reason": None,
            "evaluation_scope": "internal stratified holdout from the same dataset; not external validation",
            "evaluation_date_utc": datetime.now(timezone.utc).isoformat(),
            "data_source": DATA_SOURCE,
            "dataset_version": dataset_version_for(dataset_sha256),
            "dataset_sha256": dataset_sha256,
            "model_name": metadata.get("model_name"),
            "model_version": metadata.get("model_version"),
            "model_sha256": model_sha256,
            "threshold": threshold,
            "split": {
                "strategy": "stratified random train/test split",
                "test_fraction": TEST_SIZE,
                "random_seed": RANDOM_SEED,
                "training_partition_count": len(X_train),
                "threshold_selection_validation_count": metadata.get(
                    "n_validation_for_threshold"
                ),
                "test_count": len(X_test),
                "transformations": "raw features, matching the API's saved-model prediction input",
            },
            "sample_count": len(y_test),
            "positive_count": int(y_test.sum()),
            "negative_count": int((y_test == 0).sum()),
            "feature_schema": FEATURE_COLS,
            "metrics": {
                "roc_auc": float(roc_auc_score(y_test, probabilities)),
                "average_precision": float(average_precision_score(y_test, probabilities)),
                "precision": float(precision_score(y_test, predictions, zero_division=0)),
                "recall": float(recall_score(y_test, predictions, zero_division=0)),
                "f1": float(f1_score(y_test, predictions, zero_division=0)),
            },
            "confusion_matrix": {
                "labels": [0, 1],
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "curves": curve_points,
            "internal_validation": {
                "five_fold_stratified": cross_validation,
                "time_ordered": time_ordered,
                "calibration": calibration_metrics,
                "calibration_mapping": calibration,
            },
        }
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        ImportError,
        EOFError,
        json.JSONDecodeError,
    ) as exc:
        report = _unavailable_report(str(exc))

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    result = run_evaluation()
    print(json.dumps(result, indent=2, allow_nan=False))
