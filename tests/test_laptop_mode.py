import json
import shutil

import pandas as pd
import pytest

from src.api import routes
from src.api.main import app
from src.data import datasets
from src.data import transactions
from src.ml import predict as predict_module
from src.ml.predict import ModelStore
from fastapi.testclient import TestClient


API_KEY = "test-api-key-with-at-least-32-characters"
client = TestClient(app)


@pytest.fixture(autouse=True)
def configured_api_key(monkeypatch):
    monkeypatch.setenv("FRAUDGRAPH_API_KEY", API_KEY)


def test_laptop_mode_blocks_raw_transaction_reads(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDGRAPH_LAPTOP_MODE", "1")
    monkeypatch.setattr(routes, "PRECOMPUTED_PATH", tmp_path / "missing.parquet")
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("raw CSV opened")))
    response = client.get("/transactions", headers={"X-API-Key": API_KEY})
    assert response.status_code == 503
    assert "precomputed" in response.json()["detail"].lower()


def test_laptop_mode_blocks_live_graph_build(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDGRAPH_LAPTOP_MODE", "1")
    source = tmp_path / "paysim.csv"
    source.write_text("not opened", encoding="utf-8")
    monkeypatch.setattr(routes, "PAYSIM_PATH", source)
    monkeypatch.setattr(routes, "GRAPH_PAYLOAD_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(routes, "load_paysim", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("graph built")))
    response = client.get("/graph/paysim", headers={"X-API-Key": API_KEY})
    assert response.status_code == 503
    assert "colab" in response.json()["detail"].lower()


def test_graph_live_build_flag_returns_503_without_payload(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDGRAPH_LAPTOP_MODE", "0")
    monkeypatch.setenv("FRAUDGRAPH_GRAPH_LIVE_BUILD", "0")
    source = tmp_path / "paysim.csv"
    source.write_text("not opened", encoding="utf-8")
    monkeypatch.setattr(routes, "PAYSIM_PATH", source)
    monkeypatch.setattr(routes, "GRAPH_PAYLOAD_PATH", tmp_path / "missing.json")
    response = client.get("/graph/paysim", headers={"X-API-Key": API_KEY})
    assert response.status_code == 503
    assert "precomputed" in response.json()["detail"].lower()


def test_laptop_mode_blocks_upload_analysis(monkeypatch):
    monkeypatch.setenv("FRAUDGRAPH_LAPTOP_MODE", "1")
    monkeypatch.setenv("FRAUDGRAPH_API_KEY", API_KEY)
    monkeypatch.setattr(datasets, "get_dataset", lambda *args, **kwargs: {
        "dataset_id": "upload", "row_count": 1,
        "supports": {"transaction_risk": True},
    })
    response = client.post("/datasets/upload/analysis", headers={"X-API-Key": API_KEY})
    assert response.status_code == 503


def test_precomputed_graph_is_served_and_stale_payload_is_ignored(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDGRAPH_LAPTOP_MODE", "1")
    source = tmp_path / "paysim.csv"
    source.write_text("source", encoding="utf-8")
    stat = source.stat()
    response = {
        "data_source": "PaySim synthetic financial dataset",
        "dataset_version": "paysim-test",
        "sampled": True,
        "sampling_ratio": 1.0,
        "focus_account": None,
        "neighborhood_truncated": False,
        "available_connections": 0,
        "nodes": [],
        "edges": [],
        "suspicious_accounts": [],
    }
    payload = {"metadata": {"source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns}, "response": response}
    payload_path = tmp_path / "graph_payload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(routes, "PAYSIM_PATH", source)
    monkeypatch.setattr(routes, "GRAPH_PAYLOAD_PATH", payload_path)
    served = client.get("/graph/paysim", headers={"X-API-Key": API_KEY})
    assert served.status_code == 200
    assert served.json()["dataset_version"] == "paysim-test"
    source.write_text("stale", encoding="utf-8")
    stale = client.get("/graph/paysim", headers={"X-API-Key": API_KEY})
    assert stale.status_code == 503


def test_laptop_api_uses_only_six_model_artifacts_without_raw_csv(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDGRAPH_LAPTOP_MODE", "1")
    models = tmp_path / "models"
    models.mkdir()
    project_models = routes.PROJECT_ROOT / "models"
    for name in ("transaction_model.joblib", "model_metadata.json", "evaluation_report.json"):
        shutil.copy2(project_models / name, models / name)
    frame = pd.DataFrame({
        "transaction_id": ["creditcard-test:row:000000001"],
        "source_row": [1],
        "Time": [1.0],
        **{name: [0.0] for name in routes.FEATURE_NAMES if name not in {"Time", "Amount"}},
        "Amount": [1.0],
        "Class": [0],
    })
    frame.to_parquet(models / "transactions.parquet", index=False)
    graph_response = {
        "data_source": "PaySim synthetic financial dataset",
        "dataset_version": "paysim-test",
        "sampled": False,
        "sampling_ratio": 1.0,
        "focus_account": None,
        "neighborhood_truncated": False,
        "available_connections": 0,
        "nodes": [],
        "edges": [],
        "suspicious_accounts": [],
    }
    (models / "graph_payload.json").write_text(json.dumps({
        "metadata": {"source_size": 0, "source_mtime_ns": 0},
        "response": graph_response,
    }), encoding="utf-8")
    (models / "graph_evaluation.json").write_text("{}", encoding="utf-8")
    store = ModelStore(models / "transaction_model.joblib", models / "model_metadata.json")
    monkeypatch.setattr(routes, "MODEL_PATH", models / "transaction_model.joblib")
    monkeypatch.setattr(routes, "REPORT_PATH", models / "evaluation_report.json")
    monkeypatch.setattr(routes, "GRAPH_REPORT_PATH", models / "graph_evaluation.json")
    monkeypatch.setattr(routes, "PRECOMPUTED_PATH", models / "transactions.parquet")
    monkeypatch.setattr(transactions, "PRECOMPUTED_PATH", models / "transactions.parquet")
    monkeypatch.setattr(routes, "GRAPH_PAYLOAD_PATH", models / "graph_payload.json")
    monkeypatch.setattr(routes, "PAYSIM_PATH", tmp_path / "absent-paysim.csv")
    monkeypatch.setattr(routes, "model_store", store)
    monkeypatch.setattr(predict_module, "model_store", store)
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("raw CSV opened")))
    headers = {"X-API-Key": API_KEY}

    assert client.get("/health", headers=headers).status_code == 200
    assert client.get("/model/info", headers=headers).status_code == 200
    assert client.post("/predict", headers=headers, json={name: 0.0 for name in routes.FEATURE_NAMES}).status_code == 200
    assert client.get("/transactions", headers=headers).status_code == 200
    assert client.get("/evaluation/report", headers=headers).status_code == 200
    assert client.get("/evaluation/graph", headers=headers).status_code == 200
    assert client.get("/graph/paysim", headers=headers).status_code == 200
