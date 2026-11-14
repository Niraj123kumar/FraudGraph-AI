import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes import validate_api_key_config
from src.data.preprocessing import FEATURE_COLS
from src.data.transactions import DATA_SOURCE, DatasetUnavailableError
from src.api import routes
from src.api import investigations
from src.data import datasets
from src import auth

client = TestClient(app)
API_KEY = "test-api-key-with-at-least-32-characters"


def _sample_frame():
    frame = pd.DataFrame(
        {
            **{name: [float(i + 1), float(i + 2), float(i + 3)] for i, name in enumerate(FEATURE_COLS)},
            "Time": [10.0, 3600.0, 7200.0],
            "Amount": [20.0, 50.0, 80.0],
            "Class": [0, 1, 0],
            "source_row": [1, 2, 3],
            "transaction_id": [
                "creditcard-testhash:row:000000001",
                "creditcard-testhash:row:000000002",
                "creditcard-testhash:row:000000003",
            ],
        }
    )
    return frame


def _configure_dataset(monkeypatch, frame=None):
    sample = frame if frame is not None else _sample_frame()
    monkeypatch.setattr(
        routes,
        "get_transaction_frame",
        lambda: (sample, "a" * 64, "creditcard-testhash"),
    )
    monkeypatch.setenv("FRAUDGRAPH_API_KEY", API_KEY)


def test_transaction_list_filters_and_marks_actual_label(monkeypatch):
    _configure_dataset(monkeypatch)
    response = client.get(
        "/transactions",
        params={
            "limit": 1,
            "class_label": 0,
            "min_amount": 10,
            "max_amount": 90,
            "sort_by": "Amount",
            "sort_order": "desc",
        },
        headers={"X-API-Key": API_KEY},
    )

    assert response.status_code == 200
    result = response.json()
    assert result["data_source"] == DATA_SOURCE
    assert result["total"] == 2
    assert result["items"][0]["source_row"] == 3
    assert result["items"][0]["Class"] == 0
    assert "probability" not in result["items"][0]
    assert "prediction" not in result["items"][0]


def test_transaction_detail_and_missing_id(monkeypatch):
    _configure_dataset(monkeypatch)
    detail = client.get(
        "/transactions/creditcard-testhash:row:000000002",
        headers={"X-API-Key": API_KEY},
    )
    missing = client.get(
        "/transactions/unknown",
        headers={"X-API-Key": API_KEY},
    )

    assert detail.status_code == 200
    assert detail.json()["Class"] == 1
    assert missing.status_code == 404


def test_dataset_transaction_prediction_uses_model_feature_schema(monkeypatch):
    _configure_dataset(monkeypatch)
    captured = {}

    def predict(features):
        captured.update(features)
        return {"probability": 0.42, "is_fraud": True, "risk_level": "medium"}

    monkeypatch.setattr(routes, "predict_transaction", predict)
    response = client.post(
        "/transactions/creditcard-testhash:row:000000002/predict",
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 200
    assert list(captured) == routes.FEATURE_NAMES
    assert "Class" not in captured
    assert response.json()["probability"] == 0.42


def test_predict_and_batch_use_real_model(monkeypatch):
    _configure_dataset(monkeypatch)
    transaction = {name: 0.0 for name in routes.FEATURE_NAMES}
    headers = {"X-API-Key": API_KEY}

    single = client.post("/predict", headers=headers, json=transaction)
    batch = client.post(
        "/predict/batch",
        headers=headers,
        json={"transactions": [transaction, transaction]},
    )

    assert single.status_code == 200
    assert batch.status_code == 200
    assert set(single.json()) == {"probability", "is_fraud", "risk_level"}
    assert batch.json()["count"] == 2
    assert len(batch.json()["predictions"]) == 2
    assert batch.json()["predictions"][0]["is_fraud"] is (
        single.json()["probability"] >= routes.model_store.get_threshold()
    )


def test_health_and_model_info_report_ready_model(monkeypatch):
    _configure_dataset(monkeypatch)
    headers = {"X-API-Key": API_KEY}

    health = client.get("/health", headers=headers)
    info = client.get("/model/info", headers=headers)

    assert health.status_code == 200
    assert health.json() == {"status": "ok", "model": "ready"}
    assert info.status_code == 200
    assert info.json()["feature_names"] == routes.FEATURE_NAMES
    assert info.json()["threshold"] == routes.model_store.get_threshold()


def test_batch_prediction_returns_503_when_model_is_unavailable(monkeypatch):
    _configure_dataset(monkeypatch)
    monkeypatch.setattr(
        routes,
        "predict_transaction",
        lambda _features: (_ for _ in ()).throw(
            FileNotFoundError("Saved prediction model is not available")
        ),
    )
    response = client.post(
        "/predict/batch",
        headers={"X-API-Key": API_KEY},
        json={"transactions": [_sample_frame().iloc[0][FEATURE_COLS].to_dict()]},
    )
    assert response.status_code == 503
    assert "model" in response.json()["detail"].lower()


def test_summary_and_payment_activity(monkeypatch):
    _configure_dataset(monkeypatch)
    headers = {"X-API-Key": API_KEY}

    summary = client.get("/transactions/summary", headers=headers)
    activity = client.get(
        "/transactions/payment-activity",
        params={"interval_seconds": 3600},
        headers=headers,
    )

    assert summary.status_code == 200
    assert summary.json()["record_count"] == 3
    assert summary.json()["positive_count"] == 1
    assert summary.json()["negative_count"] == 2
    assert activity.status_code == 200
    assert activity.json()["time_unit"] == "seconds_from_first_transaction"
    assert sum(bucket["transaction_count"] for bucket in activity.json()["buckets"]) == 3
    assert sum(bucket["fraud_count"] for bucket in activity.json()["buckets"]) == 1


def test_api_key_is_required(monkeypatch):
    _configure_dataset(monkeypatch)
    response = client.get("/transactions/summary")
    assert response.status_code == 401


@pytest.mark.parametrize("configured_key", [None, "short", "dev", "changeme", "test", "secret"])
def test_invalid_api_key_configuration_is_rejected(monkeypatch, configured_key):
    if configured_key is None:
        monkeypatch.delenv("FRAUDGRAPH_API_KEY", raising=False)
    else:
        monkeypatch.setenv("FRAUDGRAPH_API_KEY", configured_key)

    with pytest.raises(RuntimeError):
        validate_api_key_config()


def test_missing_dataset_returns_service_unavailable(monkeypatch):
    monkeypatch.setenv("FRAUDGRAPH_API_KEY", API_KEY)

    def missing_dataset():
        raise DatasetUnavailableError("Transaction dataset is not available")

    monkeypatch.setattr(routes, "get_transaction_frame", missing_dataset)
    response = client.get("/transactions", headers={"X-API-Key": API_KEY})
    assert response.status_code == 503


def test_invalid_amount_range_returns_422(monkeypatch):
    _configure_dataset(monkeypatch)
    response = client.get(
        "/transactions",
        params={"min_amount": 50, "max_amount": 10},
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 422


def test_persisted_investigation_workflow(monkeypatch, tmp_path):
    _configure_dataset(monkeypatch)
    monkeypatch.setattr(investigations, "DB_PATH", tmp_path / "cases.sqlite3")
    headers = {"X-API-Key": API_KEY}
    transaction_id = _sample_frame().iloc[1]["transaction_id"]

    created = client.post("/investigations", headers=headers, json={
        "transaction_id": transaction_id, "severity": "High", "notes": "Review linked transfers",
    })
    assert created.status_code == 201
    case = created.json()
    assert case["status"] == "Open"
    assert case["Class"] == 1
    assert case["notes"] == "Review linked transfers"

    changed = client.patch(f"/investigations/{case['id']}", headers=headers, json={
        "status": "In Review", "notes": "Analyst review started",
    })
    listed = client.get("/investigations", headers=headers)
    assert changed.status_code == 200
    assert changed.json()["status"] == "In Review"
    assert listed.json()["items"][0]["notes"] == "Analyst review started"


def test_investigation_patch_validation_and_unknown_id(monkeypatch, tmp_path):
    _configure_dataset(monkeypatch)
    monkeypatch.setattr(investigations, "DB_PATH", tmp_path / "cases.sqlite3")
    headers = {"X-API-Key": API_KEY}
    transaction_id = _sample_frame().iloc[0]["transaction_id"]
    created = client.post(
        "/investigations",
        headers=headers,
        json={"transaction_id": transaction_id},
    )
    case_id = created.json()["id"]

    invalid = client.patch(
        f"/investigations/{case_id}",
        headers=headers,
        json={"status": "Invalid"},
    )
    empty = client.patch(f"/investigations/{case_id}", headers=headers, json={})
    unknown = client.patch(
        "/investigations/CASE-UNKNOWN",
        headers=headers,
        json={"status": "Resolved"},
    )

    assert invalid.status_code == 422
    assert empty.status_code == 200
    assert empty.json()["id"] == case_id
    assert unknown.status_code == 404


def test_investigation_requires_existing_creditcard_row(monkeypatch, tmp_path):
    _configure_dataset(monkeypatch)
    monkeypatch.setattr(investigations, "DB_PATH", tmp_path / "cases.sqlite3")
    response = client.post("/investigations", headers={"X-API-Key": API_KEY}, json={
        "transaction_id": "paysim-not-a-creditcard-id", "severity": "Low", "notes": "",
    })
    assert response.status_code == 404


def test_paysim_graph_api_returns_explainable_account_links(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDGRAPH_API_KEY", API_KEY)
    source = tmp_path / "paysim.csv"
    source.write_text("test data", encoding="utf-8")
    monkeypatch.setattr(routes, "PAYSIM_PATH", source)
    monkeypatch.setattr(routes, "load_paysim", lambda *args, **kwargs: pd.DataFrame({
        "step": [1, 1, 1, 1, 1], "type": ["TRANSFER"] * 5, "amount": [10.0] * 5,
        "nameOrig": ["A"] * 5, "nameDest": ["B"] * 5, "isFraud": [1, 0, 0, 0, 0],
        "_source_row": [11, 12, 13, 14, 15],
    }))
    response = client.get("/graph/paysim", headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    result = response.json()
    assert result["data_source"] == "PaySim synthetic financial dataset"
    assert result["edges"][0]["fraud_count"] == 1
    assert result["edges"][0]["evidence"][0]["source_row"] == 11
    assert result["edges"][0]["evidence"][0]["simulator_fraud_label"] == 1
    flagged = next(account for account in result["suspicious_accounts"] if account["id"] == "A")
    assert any("High transfer activity" in reason for reason in flagged["reasons"])
    assert all("fraud" not in reason.lower() for reason in flagged["reasons"])


def test_paysim_graph_cache_reuses_valid_data_and_invalidates_on_file_change(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDGRAPH_API_KEY", API_KEY)
    source = tmp_path / "paysim.csv"
    source.write_text("first version", encoding="utf-8")
    monkeypatch.setattr(routes, "PAYSIM_PATH", source)
    calls = []
    frame = pd.DataFrame({
        "step": [1, 1, 1, 1, 1], "type": ["TRANSFER"] * 5,
        "amount": [10.0] * 5, "nameOrig": ["A"] * 5, "nameDest": list("BCDEF"),
        "isFraud": [1, 0, 0, 0, 0], "_source_row": [11, 12, 13, 14, 15],
    })

    def load(*args, **kwargs):
        calls.append(1)
        return frame.copy()

    monkeypatch.setattr(routes, "load_paysim", load)
    routes._cached_paysim_graph.cache_clear()
    headers = {"X-API-Key": API_KEY}
    first = client.get("/graph/paysim?limit=10&max_rows=1000", headers=headers).json()
    repeated = client.get("/graph/paysim?limit=10&max_rows=1000", headers=headers).json()
    assert len(calls) == 1
    assert first["edges"] == repeated["edges"]

    frame["isFraud"] = 0
    source.write_text("second version with changed size", encoding="utf-8")
    changed = client.get("/graph/paysim?limit=10&max_rows=1000", headers=headers).json()
    assert len(calls) == 2
    flagged_before = next(node for node in first["suspicious_accounts"] if node["id"] == "A")
    flagged_after = next(node for node in changed["suspicious_accounts"] if node["id"] == "A")
    assert flagged_before["risk_score"] == flagged_after["risk_score"]
    assert flagged_before["reasons"] == flagged_after["reasons"]
    assert changed["edges"][0]["fraud_count"] == 0
    routes._cached_paysim_graph.cache_clear()


def test_paysim_graph_focus_returns_bounded_real_neighborhood(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDGRAPH_API_KEY", API_KEY)
    source = tmp_path / "paysim.csv"
    source.write_text("fixture", encoding="utf-8")
    monkeypatch.setattr(routes, "PAYSIM_PATH", source)
    frame = pd.DataFrame({
        "step": [1] * 4, "type": ["TRANSFER"] * 4, "amount": [10.0] * 4,
        "nameOrig": ["A"] * 4, "nameDest": ["B", "C", "D", "E"],
        "isFraud": [0] * 4, "_source_row": [31, 32, 33, 34],
    })
    monkeypatch.setattr(routes, "load_paysim", lambda *args, **kwargs: frame.copy())
    routes._cached_paysim_graph.cache_clear()

    response = client.get("/graph/paysim?account_id=A&neighbor_limit=2", headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    result = response.json()
    assert result["focus_account"] == "A"
    assert result["neighborhood_truncated"] is True
    assert result["available_connections"] == 4
    assert len(result["edges"]) == 2
    assert all(edge["source"] == "A" for edge in result["edges"])
    assert all(edge["evidence"][0]["source_row"] in {31, 32, 33, 34} for edge in result["edges"])

def test_uploaded_csv_quality_preview_history_and_isolation(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDGRAPH_API_KEY", API_KEY)
    monkeypatch.setattr(datasets, "REGISTRY_PATH", tmp_path / "datasets.sqlite3")
    monkeypatch.setattr(datasets, "UPLOAD_DIR", tmp_path / "uploads")
    response = client.post(
        "/datasets/upload",
        headers={"X-API-Key": API_KEY},
        files={"file": ("accounts.csv", b"sender,receiver,amount\nA,B,10\nA,B,10\nC,,5\n", "text/csv")},
    )

    assert response.status_code == 201
    result = response.json()
    assert result["rows_read"] == result["rows_processed"] == 3
    assert result["rows_rejected"] == result["rows_skipped"] == 0
    assert result["total_missing_values"] == 1
    assert result["duplicate_rows"] == 1
    assert result["supports"]["transaction_risk"] is False
    assert result["supports"]["account_relationships"] is False
    assert result["supports"]["evaluation"] is False

    preview = client.get(f"/datasets/{result['dataset_id']}/preview", headers={"X-API-Key": API_KEY})
    history = client.get("/datasets", headers={"X-API-Key": API_KEY})
    incompatible = client.post(f"/datasets/{result['dataset_id']}/transactions/1/risk", headers={"X-API-Key": API_KEY})
    assert preview.status_code == 200
    assert preview.json()["items"][1]["sender"] == "A"
    assert history.json()["total"] == 1
    assert history.json()["items"][0]["dataset_id"] == result["dataset_id"]
    assert incompatible.status_code == 422

    second = client.post("/datasets/upload", headers={"X-API-Key": API_KEY}, files={"file": ("separate.csv", b"code,value\nX,99\n", "text/csv")})
    assert second.status_code == 201
    assert second.json()["dataset_id"] != result["dataset_id"]
    assert client.get(f"/datasets/{result['dataset_id']}/preview", headers={"X-API-Key": API_KEY}).json()["columns"] == ["sender", "receiver", "amount"]
    removed = client.delete(f"/datasets/{result['dataset_id']}", headers={"X-API-Key": API_KEY})
    assert removed.status_code == 204
    assert client.get(f"/datasets/{result['dataset_id']}", headers={"X-API-Key": API_KEY}).status_code == 404
    assert len(list((tmp_path / "uploads").glob("*.csv"))) == 1


def test_upload_limit_is_enforced_and_partial_files_are_removed(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDGRAPH_API_KEY", API_KEY)
    monkeypatch.setattr(datasets, "REGISTRY_PATH", tmp_path / "datasets.sqlite3")
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(datasets, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(datasets, "MAX_UPLOAD_BYTES", 12)
    response = client.post("/datasets/upload", headers={"X-API-Key": API_KEY},
                           files={"file": ("large.csv", b"column\n1234567890\n", "text/csv")})
    assert response.status_code == 413
    assert list(upload_dir.iterdir()) == []


def test_upload_limit_defaults_to_one_hundred_megabytes():
    assert datasets.MAX_UPLOAD_BYTES == 100 * 1024 * 1024


def test_upload_limit_uses_environment_override(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDGRAPH_API_KEY", API_KEY)
    monkeypatch.setenv("FRAUDGRAPH_MAX_UPLOAD_BYTES", "12")
    monkeypatch.setattr(datasets, "REGISTRY_PATH", tmp_path / "datasets.sqlite3")
    monkeypatch.setattr(datasets, "UPLOAD_DIR", tmp_path / "uploads")
    response = client.post(
        "/datasets/upload",
        headers={"X-API-Key": API_KEY},
        files={"file": ("large.csv", b"column\n1234567890\n", "text/csv")},
    )
    assert response.status_code == 413
    assert list((tmp_path / "uploads").iterdir()) == []


def test_uploaded_analysis_double_submit_schedules_once(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDGRAPH_API_KEY", API_KEY)
    monkeypatch.setattr(datasets, "REGISTRY_PATH", tmp_path / "datasets.sqlite3")
    dataset = {
        "dataset_id": "dataset-one",
        "row_count": 5,
        "supports": {"transaction_risk": True, "reasons": {"transaction_risk": None}},
    }
    monkeypatch.setattr(datasets, "get_dataset", lambda dataset_id, owner_id="legacy": dataset if dataset_id == "dataset-one" else None)
    submitted = []

    class FakeExecutor:
        def submit(self, background_tasks, job):
            submitted.append(job)

    monkeypatch.setattr(routes, "job_executor", FakeExecutor())
    first = client.post("/datasets/dataset-one/analysis", headers={"X-API-Key": API_KEY})
    second = client.post("/datasets/dataset-one/analysis", headers={"X-API-Key": API_KEY})
    assert first.status_code == second.status_code == 202
    assert first.json()["dataset_id"] == second.json()["dataset_id"] == "dataset-one"
    assert first.json()["status"] == second.json()["status"] == "queued"
    assert len(submitted) == 1


def test_graph_response_reports_sampling_ratio_and_guards(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDGRAPH_API_KEY", API_KEY)
    source = tmp_path / "paysim.csv"
    source.write_text("source", encoding="utf-8")
    monkeypatch.setattr(routes, "PAYSIM_PATH", source)
    monkeypatch.setattr(routes, "GRAPH_MAX_FILE_BYTES", 1)
    assert client.get("/graph/paysim", headers={"X-API-Key": API_KEY}).status_code == 413

    monkeypatch.setattr(routes, "GRAPH_MAX_FILE_BYTES", 10_000)
    monkeypatch.setattr(routes, "GRAPH_TIMEOUT_SECONDS", -1)
    routes._cached_paysim_graph.cache_clear()
    timed_out = client.get("/graph/paysim", headers={"X-API-Key": API_KEY})
    assert timed_out.status_code == 503

    monkeypatch.setattr(routes, "GRAPH_TIMEOUT_SECONDS", 30)
    sampled_frame = pd.DataFrame({
        "step": [1], "type": ["TRANSFER"], "amount": [10.0],
        "nameOrig": ["A"], "nameDest": ["B"], "isFraud": [0],
        "_source_row": [1],
    })
    sampled_frame.attrs.update({"sampled": True, "eligible_rows_seen": 4})
    monkeypatch.setattr(routes, "load_paysim", lambda *args, **kwargs: sampled_frame)
    routes._cached_paysim_graph.cache_clear()
    result = client.get("/graph/paysim", headers={"X-API-Key": API_KEY}).json()
    assert result["sampled"] is True
    assert result["sampling_ratio"] == pytest.approx(0.25)
    routes._cached_paysim_graph.cache_clear()


def test_uploaded_paysim_graph_uses_only_selected_uploaded_file(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDGRAPH_API_KEY", API_KEY)
    monkeypatch.setattr(datasets, "REGISTRY_PATH", tmp_path / "datasets.sqlite3")
    monkeypatch.setattr(datasets, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(datasets, "CHUNK_SIZE", 2)
    body = ("step,type,amount,nameOrig,nameDest,isFraud\n"
            "1,TRANSFER,10,A,B,0\n1,CASH_OUT,20,B,C,1\n1,TRANSFER,30,C,A,0\n")
    upload = client.post("/datasets/upload", headers={"X-API-Key": API_KEY},
                         files={"file": ("uploaded-paysim.csv", body.encode(), "text/csv")})
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]
    assert upload.json()["supports"]["account_relationships"] is True

    original = tmp_path / "separate-source.csv"
    original.write_text("not the uploaded data", encoding="utf-8")
    monkeypatch.setattr(routes, "PAYSIM_PATH", original)
    routes._cached_paysim_graph.cache_clear()
    graph = client.get(f"/graph/paysim?dataset_id={dataset_id}&account_id=A", headers={"X-API-Key": API_KEY})
    assert graph.status_code == 200
    result = graph.json()
    assert result["data_source"].endswith("uploaded-paysim.csv")
    assert all(edge["source"] == "A" or edge["target"] == "A" for edge in result["edges"])
    assert result["edges"][0]["evidence"]

    unavailable = client.get("/graph/paysim?dataset_id=00000000-0000-0000-0000-000000000000", headers={"X-API-Key": API_KEY})
    assert unavailable.status_code == 404
    routes._cached_paysim_graph.cache_clear()


def test_uploaded_dataset_prediction_uses_actual_ordered_csv_row(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDGRAPH_API_KEY", API_KEY)
    monkeypatch.setattr(datasets, "REGISTRY_PATH", tmp_path / "datasets.sqlite3")
    monkeypatch.setattr(datasets, "UPLOAD_DIR", tmp_path / "uploads")
    names = routes.FEATURE_NAMES + ["Class"]
    data = ",".join(names) + "\n" + ",".join(["1"] * len(names)) + "\n"
    uploaded = client.post("/datasets/upload", headers={"X-API-Key": API_KEY}, files={"file": ("credit-card.csv", data.encode(), "text/csv")})
    assert uploaded.status_code == 201
    result = uploaded.json()
    assert result["supports"]["transaction_risk"] is True

    captured = {}
    def predict(features):
        captured.update(features)
        return {"probability": 0.1, "is_fraud": False, "risk_level": "low"}
    monkeypatch.setattr(routes, "predict_transaction", predict)
    response = client.post(f"/datasets/{result['dataset_id']}/transactions/1/risk", headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    assert list(captured) == routes.FEATURE_NAMES
    assert response.json()["dataset_id"] == result["dataset_id"]
    assert response.json()["transaction_id"].startswith(result["dataset_id"] + ":row:")
    assert "Class" not in captured


@pytest.mark.parametrize("bad_value", ["not-a-number", "nan"])
def test_uploaded_dataset_prediction_rejects_invalid_feature(monkeypatch, tmp_path, bad_value):
    monkeypatch.setenv("FRAUDGRAPH_API_KEY", API_KEY)
    monkeypatch.setattr(datasets, "REGISTRY_PATH", tmp_path / "datasets.sqlite3")
    monkeypatch.setattr(datasets, "UPLOAD_DIR", tmp_path / "uploads")
    names = routes.FEATURE_NAMES
    values = ["1"] * len(names)
    values[1] = bad_value
    csv_text = ",".join(names) + "\n" + ",".join(values) + "\n"
    uploaded = client.post(
        "/datasets/upload",
        headers={"X-API-Key": API_KEY},
        files={"file": ("invalid-row.csv", csv_text.encode(), "text/csv")},
    )
    assert uploaded.status_code == 201

    response = client.post(
        f"/datasets/{uploaded.json()['dataset_id']}/transactions/1/risk",
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 422
    assert "V1" in response.json()["detail"]


def test_uploaded_dataset_batch_analysis_uses_uploaded_rows_and_separates_labels(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDGRAPH_API_KEY", API_KEY)
    monkeypatch.setattr(datasets, "REGISTRY_PATH", tmp_path / "datasets.sqlite3")
    monkeypatch.setattr(datasets, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(datasets, "PREDICTION_CHUNK_SIZE", 2)
    names = routes.FEATURE_NAMES
    columns = names + ["Class"]
    valid = ["1"] * len(names)
    valid_two = ["2"] * len(names)
    missing = valid.copy()
    missing[1] = ""
    incompatible_value = valid.copy()
    incompatible_value[2] = "not-a-number"
    csv_text = ",".join(columns) + "\n" + ",".join(valid + ["1"]) + "\n" + ",".join(valid_two + ["0"]) + "\n" + ",".join(missing + ["0"]) + "\n" + ",".join(incompatible_value + ["1"]) + "\n" + ",".join(valid_two + ["0"]) + "\n"
    upload = client.post("/datasets/upload", headers={"X-API-Key": API_KEY}, files={"file": ("batch.csv", csv_text.encode(), "text/csv")})
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]
    assert upload.json()["supports"]["transaction_risk"] is True

    observed_batches = []
    def fake_batch(features):
        observed_batches.append((list(features.columns), len(features), float(features.iloc[0]["Time"])))
        return [{"probability": 0.25, "is_fraud": False, "risk_level": "low"} for _ in range(len(features))]
    monkeypatch.setattr(routes, "predict_transactions_batch", fake_batch)
    started = client.post(f"/datasets/{dataset_id}/analysis", headers={"X-API-Key": API_KEY})
    assert started.status_code == 202
    status = client.get(f"/datasets/{dataset_id}/analysis", headers={"X-API-Key": API_KEY}).json()
    assert status["status"] == "completed"
    assert status["rows_scanned"] == 5
    assert status["rows_processed"] == 3
    assert status["rows_rejected"] == 2
    assert observed_batches == [(names, 2, 1.0), (names, 1, 2.0)]

    result = client.get(f"/datasets/{dataset_id}/predictions", headers={"X-API-Key": API_KEY}).json()
    assert result["dataset_id"] == dataset_id
    assert [item["source_row"] for item in result["items"]] == [1, 2, 5]
    assert [item["recorded_csv_label"] for item in result["items"]] == ["1", "0", "0"]
    assert result["items"][0]["is_fraud"] is False
    assert result["rows_processed"] == 3
    assert result["rows_rejected"] == 2

    second_upload = client.post("/datasets/upload", headers={"X-API-Key": API_KEY}, files={"file": ("separate-compatible.csv", (",".join(columns) + "\n" + ",".join(valid_two + ["0"]) + "\n").encode(), "text/csv")})
    second_id = second_upload.json()["dataset_id"]
    client.post(f"/datasets/{second_id}/analysis", headers={"X-API-Key": API_KEY})
    second_results = client.get(f"/datasets/{second_id}/predictions", headers={"X-API-Key": API_KEY}).json()
    assert second_results["dataset_id"] == second_id and second_results["total"] == 1
    assert result["dataset_id"] != second_results["dataset_id"] and result["total"] == 3

    incompatible_upload = client.post("/datasets/upload", headers={"X-API-Key": API_KEY}, files={"file": ("wrong.csv", b"Time,Amount,Class\n1,20,1\n", "text/csv")})
    assert incompatible_upload.status_code == 201
    rejected = client.post(f"/datasets/{incompatible_upload.json()['dataset_id']}/analysis", headers={"X-API-Key": API_KEY})
    assert rejected.status_code == 422
    assert "V1" in rejected.json()["detail"]


def test_prediction_run_queue_is_idempotent_and_restart_marks_jobs_retryable(monkeypatch, tmp_path):
    monkeypatch.setattr(datasets, "REGISTRY_PATH", tmp_path / "datasets.sqlite3")
    first, started = datasets.queue_prediction_run("dataset-one", 5)
    assert started is True and first["status"] == "queued"
    duplicate, started_again = datasets.queue_prediction_run("dataset-one", 5)
    assert started_again is False and duplicate["status"] == "queued"
    datasets.update_prediction_run("dataset-one", status="running")
    datasets.fail_interrupted_prediction_runs()
    recovered = datasets.get_prediction_run("dataset-one")
    assert recovered["status"] == "failed"
    assert "restarted" in recovered["error"]


def test_startup_recovery_only_fails_stale_running_jobs(monkeypatch, tmp_path):
    from datetime import datetime, timedelta, timezone

    monkeypatch.setattr(datasets, "REGISTRY_PATH", tmp_path / "datasets.sqlite3")
    datasets.queue_prediction_run("stale", 1)
    datasets.update_prediction_run(
        "stale",
        status="running",
        heartbeat_at=(datetime.now(timezone.utc) - timedelta(seconds=601)).isoformat(),
    )
    datasets.queue_prediction_run("fresh", 1)
    datasets.update_prediction_run(
        "fresh",
        status="running",
        heartbeat_at=datetime.now(timezone.utc).isoformat(),
    )
    datasets.fail_interrupted_prediction_runs()
    assert datasets.get_prediction_run("stale")["status"] == "failed"
    assert datasets.get_prediction_run("fresh")["status"] == "running"


def test_uploaded_csv_rejects_non_csv_and_malformed_rows(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDGRAPH_API_KEY", API_KEY)
    monkeypatch.setattr(datasets, "REGISTRY_PATH", tmp_path / "datasets.sqlite3")
    monkeypatch.setattr(datasets, "UPLOAD_DIR", tmp_path / "uploads")
    wrong_type = client.post("/datasets/upload", headers={"X-API-Key": API_KEY}, files={"file": ("data.json", b"{}", "application/json")})
    malformed = client.post("/datasets/upload", headers={"X-API-Key": API_KEY}, files={"file": ("data.csv", b"a,b\n1,2,3\n", "text/csv")})
    assert wrong_type.status_code == 415
    assert malformed.status_code == 422
    assert list((tmp_path / "uploads").glob("*.csv")) == []


def test_missing_model_reports_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDGRAPH_API_KEY", API_KEY)
    monkeypatch.setattr(routes, "MODEL_PATH", tmp_path / "missing-model.joblib")

    response = client.get("/model/info", headers={"X-API-Key": API_KEY})

    assert response.status_code == 503


def test_missing_evaluation_report_is_explicitly_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDGRAPH_API_KEY", API_KEY)
    monkeypatch.setattr(routes, "REPORT_PATH", tmp_path / "missing-report.json")

    response = client.get("/evaluation/report", headers={"X-API-Key": API_KEY})

    assert response.status_code == 200
    result = response.json()
    assert result["available"] is False
    assert result["metrics"] is None
    assert "not been generated" in result["reason"]


def test_graph_evaluation_report_is_authenticated_and_read_only(monkeypatch, tmp_path):
    monkeypatch.setenv("FRAUDGRAPH_API_KEY", API_KEY)
    report_path = tmp_path / "graph_evaluation.json"
    report_path.write_text('{"available": true, "metrics": {"roc_auc": 0.6}}', encoding="utf-8")
    monkeypatch.setattr(routes, "GRAPH_REPORT_PATH", report_path)
    response = client.get("/evaluation/graph", headers={"X-API-Key": API_KEY})
    assert response.status_code == 200
    assert response.json()["metrics"]["roc_auc"] == 0.6
    assert client.get("/evaluation/graph").status_code == 401


def test_per_user_keys_isolate_datasets_and_investigations(monkeypatch, tmp_path):
    monkeypatch.delenv("FRAUDGRAPH_SINGLE_KEY_MODE", raising=False)
    monkeypatch.setattr(auth, "AUTH_DB_PATH", tmp_path / "auth.sqlite3")
    monkeypatch.setattr(datasets, "REGISTRY_PATH", tmp_path / "datasets.sqlite3")
    monkeypatch.setattr(datasets, "UPLOAD_DIR", tmp_path / "uploads")
    monkeypatch.setattr(investigations, "DB_PATH", tmp_path / "investigations.sqlite3")
    key_a = auth.create_api_key("user-a")
    key_b = auth.create_api_key("user-b")
    headers_a = {"X-API-Key": key_a}
    headers_b = {"X-API-Key": key_b}

    upload = client.post(
        "/datasets/upload",
        headers=headers_a,
        files={"file": ("private.csv", b"value\n1\n", "text/csv")},
    )
    assert upload.status_code == 201
    dataset_id = upload.json()["dataset_id"]
    assert client.get(f"/datasets/{dataset_id}", headers=headers_a).status_code == 200
    assert client.get(f"/datasets/{dataset_id}", headers=headers_b).status_code == 404
    assert client.post(f"/datasets/{dataset_id}/analysis", headers=headers_b).status_code == 404
    datasets.queue_prediction_run(dataset_id, 1, "user-a")
    assert client.get(f"/datasets/{dataset_id}/analysis", headers=headers_b).status_code == 404
    assert client.get(f"/datasets/{dataset_id}/predictions", headers=headers_b).status_code == 404
    assert client.delete(f"/datasets/{dataset_id}", headers=headers_b).status_code == 404
    assert client.get("/datasets", headers=headers_b).json()["total"] == 0

    _configure_dataset(monkeypatch)
    created = client.post(
        "/investigations",
        headers=headers_a,
        json={"transaction_id": "creditcard-testhash:row:000000001"},
    )
    assert created.status_code == 201
    case_id = created.json()["id"]
    assert client.get("/investigations", headers=headers_b).json()["total"] == 0
    assert client.patch(
        f"/investigations/{case_id}",
        headers=headers_b,
        json={"notes": "takeover"},
    ).status_code == 404
    created_b = client.post(
        "/investigations",
        headers=headers_b,
        json={"transaction_id": "creditcard-testhash:row:000000002"},
    )
    assert created_b.status_code == 201
    assert client.delete(
        f"/investigations/{created_b.json()['id']}",
        headers=headers_a,
    ).status_code == 404
    assert client.get("/investigations", headers=headers_a).json()["total"] == 1


def test_revoked_per_user_key_is_rejected(monkeypatch, tmp_path):
    monkeypatch.delenv("FRAUDGRAPH_SINGLE_KEY_MODE", raising=False)
    monkeypatch.setattr(auth, "AUTH_DB_PATH", tmp_path / "auth.sqlite3")
    key_id, raw_key = auth.create_api_key_record("user-a")
    with auth.connect_auth() as db:
        stored = db.execute(
            "SELECT key_hash FROM api_keys WHERE key_id = ?", (key_id,)
        ).fetchone()
    assert stored["key_hash"] != raw_key
    assert len(stored["key_hash"]) == 64
    assert client.get("/datasets", headers={"X-API-Key": raw_key}).status_code == 200
    assert auth.revoke_api_key(key_id) is True
    assert client.get("/datasets", headers={"X-API-Key": raw_key}).status_code == 401


def test_background_job_and_results_keep_authenticated_owner(monkeypatch, tmp_path):
    monkeypatch.setattr(datasets, "REGISTRY_PATH", tmp_path / "datasets.sqlite3")
    run, queued = datasets.queue_prediction_run("dataset-owner", 1, "user-a")
    assert queued is True
    assert run["owner_id"] == "user-a"
    datasets.insert_dataset_predictions(
        "dataset-owner",
        [{
            "source_row": 1, "Time": 1.0, "Amount": 2.0, "probability": 0.1,
            "is_fraud": False, "risk_level": "low", "recorded_csv_label": None,
        }],
        "user-a",
    )
    items, total = datasets.get_dataset_predictions("dataset-owner", 10, 0, "user-a")
    assert total == 1 and items[0]["source_row"] == 1
    other_items, other_total = datasets.get_dataset_predictions("dataset-owner", 10, 0, "user-b")
    assert other_items == [] and other_total == 0
