import json

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import EXPECTED_COLS, FEATURE_COLS
from src.ml.predict import ModelStore
from src.ml.train import MODEL_CONFIG, train_model


@pytest.mark.heavy
def test_training_writes_reproducible_artifacts_and_model_loads(tmp_path):
    rng = np.random.default_rng(42)
    rows = 2000
    frame = pd.DataFrame(rng.normal(size=(rows, len(EXPECTED_COLS))), columns=EXPECTED_COLS)
    frame["Time"] = np.arange(rows, dtype=float)
    frame["Amount"] = rng.exponential(scale=80, size=rows)
    frame["Class"] = 0
    frame.loc[rng.choice(rows, size=100, replace=False), "Class"] = 1
    data_path = tmp_path / "creditcard.csv"
    output_dir = tmp_path / "models"
    frame.to_csv(data_path, index=False)

    artifacts = train_model(data_path, output_dir)

    assert all(path.is_file() for path in artifacts.values())
    metadata = json.loads(artifacts["model_metadata.json"].read_text(encoding="utf-8"))
    assert metadata["feature_names"] == FEATURE_COLS
    assert metadata["n_train"] == 1600
    assert metadata["n_validation_for_threshold"] == 320
    assert metadata["n_test"] == 400
    assert 0 <= metadata["threshold"] <= 1
    assert metadata["dataset_sha256"]
    assert metadata["test_metrics"]["threshold"] == metadata["threshold"]
    assert metadata["calibration"]["method"] == "isotonic"
    assert metadata["risk_bands"]["probability_space"] == "calibrated isotonic probability"
    report = json.loads(artifacts["evaluation_report.json"].read_text(encoding="utf-8"))
    assert set(report["internal_validation"]) >= {"five_fold_stratified", "time_ordered", "calibration"}
    assert len(report["internal_validation"]["five_fold_stratified"]["folds"]) == 5
    assert "brier_score" in report["internal_validation"]["calibration"]

    store = ModelStore(
        model_path=artifacts["transaction_model.joblib"],
        metadata_path=artifacts["model_metadata.json"],
    )
    result = store.get_model().predict_proba(frame[FEATURE_COLS].iloc[:1])[:, 1]
    assert result.shape == (1,)


@pytest.mark.heavy
def test_time_ordered_noise_has_near_random_auc(tmp_path, monkeypatch):
    rng = np.random.default_rng(7)
    rows = 3000
    frame = pd.DataFrame(rng.normal(size=(rows, len(EXPECTED_COLS))), columns=EXPECTED_COLS)
    frame["Time"] = np.arange(rows, dtype=float)
    frame["Amount"] = rng.normal(100, 20, rows)
    frame["Class"] = rng.integers(0, 2, rows)
    data_path = tmp_path / "noise.csv"
    frame.to_csv(data_path, index=False)
    monkeypatch.setitem(MODEL_CONFIG, "n_estimators", 40)
    artifacts = train_model(data_path, tmp_path / "models")
    report = json.loads(artifacts["evaluation_report.json"].read_text(encoding="utf-8"))
    assert 0.3 <= report["internal_validation"]["time_ordered"]["roc_auc"] <= 0.7
