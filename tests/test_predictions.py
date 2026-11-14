import json

import pandas as pd
import pytest
import src.ml.predict as predict_module

from src.ml.predict import (
    FEATURE_NAMES,
    ModelStore,
    model_store,
    predict_transaction,
    predict_transaction,
    predict_transactions_batch,
)


def _features(value: float = 0.0) -> dict[str, float]:
    return {name: value for name in FEATURE_NAMES}


def test_real_model_prediction_uses_saved_threshold():
    result = predict_transaction(_features())
    threshold = model_store.get_threshold()

    assert set(result) == {"probability", "is_fraud", "risk_level"}
    assert 0 <= result["probability"] <= 1
    assert result["is_fraud"] is (result["probability"] >= threshold)
    expected_risk = (
        "low"
        if result["probability"] < 0.30
        else "medium"
        if result["probability"] < 0.70
        else "high"
    )
    assert result["risk_level"] == expected_risk


def test_real_model_batch_prediction_returns_one_result_per_row():
    frame = pd.DataFrame([_features(), _features(1.0)], columns=FEATURE_NAMES)

    results = predict_transactions_batch(frame)

    assert len(results) == 2
    assert all(set(result) == {"probability", "is_fraud", "risk_level"} for result in results)


def test_model_store_injects_model_and_reloads_when_metadata_changes(tmp_path):
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    model_path.write_bytes(b"fake")
    metadata = {"feature_names": FEATURE_NAMES, "threshold": 0.4}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    fake_model = object()
    loaded_paths = []

    store = ModelStore(
        model_path=model_path,
        metadata_path=metadata_path,
        model_loader=lambda path: loaded_paths.append(path) or fake_model,
    )

    assert store.get_model() is fake_model
    assert store.get_model() is fake_model
    assert len(loaded_paths) == 1

    metadata["threshold"] = 0.6
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    assert store.get_threshold() == 0.6
    assert len(loaded_paths) == 2


def test_model_store_warns_on_library_version_mismatch_and_loads_old_metadata(tmp_path, caplog):
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    model_path.write_bytes(b"fake")
    metadata_path.write_text(json.dumps({
        "feature_names": FEATURE_NAMES,
        "threshold": 0.5,
        "library_versions": {"numpy": "0.0-not-running"},
    }), encoding="utf-8")
    store = ModelStore(model_path, metadata_path, model_loader=lambda _: object())
    store.get_model()
    assert "library versions differ" in caplog.text

    metadata_path.write_text(json.dumps({
        "feature_names": FEATURE_NAMES,
        "threshold": 0.5,
    }), encoding="utf-8")
    assert store.get_model() is not None


@pytest.mark.parametrize(
    "metadata",
    [
        {"feature_names": FEATURE_NAMES[:-1], "threshold": 0.5},
        {"feature_names": list(reversed(FEATURE_NAMES)), "threshold": 0.5},
        {"feature_names": FEATURE_NAMES},
    ],
)
def test_model_store_rejects_incompatible_metadata(tmp_path, metadata):
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    model_path.write_bytes(b"fake")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    store = ModelStore(model_path, metadata_path, model_loader=lambda _: object())

    with pytest.raises(ValueError, match="metadata"):
        store.get_model()


class _ProbabilityModel:
    def predict_proba(self, frame):
        return [[0.5, 0.5] for _ in range(len(frame))]


def test_calibrated_cutoffs_change_risk_bands_and_old_metadata_falls_back(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    model_path.write_bytes(b"fake")
    metadata = {
        "feature_names": FEATURE_NAMES,
        "threshold": 0.5,
        "calibration": {
            "method": "isotonic",
            "raw_probabilities": [0.0, 1.0],
            "calibrated_probabilities": [0.0, 1.0],
        },
        "risk_bands": {"low_max_exclusive": 0.2, "medium_max_exclusive": 0.4},
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    store = ModelStore(model_path, metadata_path, model_loader=lambda _: _ProbabilityModel())
    monkeypatch.setattr(predict_module, "model_store", store)
    assert predict_transaction(_features())["risk_level"] == "high"

    metadata_path.write_text(json.dumps({
        "feature_names": FEATURE_NAMES, "threshold": 0.5,
    }), encoding="utf-8")
    assert predict_transaction(_features())["risk_level"] == "medium"