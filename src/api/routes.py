import os
import secrets
import json
import uuid
import errno
import logging
import time
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import numpy as np
import pandas as pd
from typing import Literal
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, Header, HTTPException, Query, UploadFile

from src.api import schemas
from src.api.schemas import (
    BatchRequest,
    BatchResponse,
    DatasetSummaryResponse,
    EvaluationReportResponse,
    ModelInfoResponse,
    PaymentActivityResponse,
    PredictionResponse,
    RiskCutoffs,
    TransactionListResponse,
    TransactionFeatures,
)
from src.data import datasets
from src.data.datasets import DatasetValidationError
from src.data.preprocessing import TARGET_COL
from src.data.transactions import (
    DATA_SOURCE,
    DatasetUnavailableError,
    PRECOMPUTED_PATH,
    get_transaction_frame,
    transaction_record,
)
from src.ml.evaluate import REPORT_PATH
from src.ml.predict import (
    FEATURE_NAMES,
    MODEL_PATH,
    coerce_feature_frame,
    coerce_feature_values,
    model_store,
    predict_transactions_batch,
    predict_transaction,
)
from src.api import investigations
from src.api.jobs import InProcessJobExecutor
from src.auth import LEGACY_OWNER_ID, owner_for_key, single_key_mode_enabled
from src.graph.build_graph import aggregate_transfers, build_graph, load_paysim
from src.graph.graph_features import graph_node_metrics
from src.graph.suspicious_groups import score_accounts, suspicious_accounts
from src.graph.evaluate import GRAPH_REPORT_PATH

router = APIRouter()
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAYSIM_PATH = PROJECT_ROOT / "data" / "raw" / "paysim.csv"
PAYSIM_SAMPLE_SEED = 42
DEFAULT_API_KEYS = frozenset({"dev", "changeme", "test", "secret", "dev-local-key-change-me"})
job_executor = InProcessJobExecutor()
GRAPH_MAX_ROWS = int(os.getenv("FRAUDGRAPH_GRAPH_MAX_ROWS", "250000"))
GRAPH_MAX_FILE_BYTES = int(os.getenv("FRAUDGRAPH_GRAPH_MAX_FILE_BYTES", str(2 * 1024**3)))
GRAPH_TIMEOUT_SECONDS = float(os.getenv("FRAUDGRAPH_GRAPH_TIMEOUT_SECONDS", "30"))
GRAPH_PAYLOAD_PATH = PROJECT_ROOT / "models" / "graph_payload.json"


def _laptop_mode() -> bool:
    return os.getenv("FRAUDGRAPH_LAPTOP_MODE", "1") == "1"


def _live_graph_enabled() -> bool:
    return not _laptop_mode() and os.getenv("FRAUDGRAPH_GRAPH_LIVE_BUILD", "1") == "1"


def _precomputed_graph(path: Path) -> dict | None:
    if not GRAPH_PAYLOAD_PATH.is_file():
        return None
    try:
        payload = json.loads(GRAPH_PAYLOAD_PATH.read_text(encoding="utf-8"))
        metadata = payload["metadata"]
        if path.is_file():
            source_stat = path.stat()
            if (
                metadata["source_size"] != source_stat.st_size
                or metadata["source_mtime_ns"] != source_stat.st_mtime_ns
            ):
                return None
        elif not _laptop_mode():
            return None
        return payload["response"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def validate_api_key_config() -> str:
    """Return a valid server-side API key or fail startup."""
    api_key = os.getenv("FRAUDGRAPH_API_KEY")
    if not api_key:
        raise RuntimeError("FRAUDGRAPH_API_KEY must be configured before startup")
    if len(api_key) < 32:
        raise RuntimeError("FRAUDGRAPH_API_KEY must contain at least 32 characters")
    if api_key.strip().lower() in DEFAULT_API_KEYS:
        raise RuntimeError("FRAUDGRAPH_API_KEY uses a prohibited default value")
    return api_key


@lru_cache(maxsize=1)
def _cached_paysim_graph(path: str, mtime_ns: int, ctime_ns: int, size: int, inode: int, max_rows: int):
    """Cache the expensive, deterministic graph build for an unchanged CSV."""
    source = Path(path)
    rows = load_paysim(source, max_rows=max_rows, seed=PAYSIM_SAMPLE_SEED)
    edges = aggregate_transfers(rows)
    graph = build_graph(edges)
    digest = hashlib.sha256()
    with source.open("rb") as dataset_file:
        for chunk in iter(lambda: dataset_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return rows, edges, graph, digest.hexdigest()[:16]


def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    if single_key_mode_enabled():
        try:
            expected_key = validate_api_key_config()
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail="API authentication is not configured correctly") from exc
        if not x_api_key or not secrets.compare_digest(x_api_key, expected_key):
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return LEGACY_OWNER_ID

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    owner_id = owner_for_key(x_api_key)
    if owner_id is None:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return owner_id


@router.get("/datasets", response_model=schemas.DatasetListResponse, dependencies=[Depends(verify_api_key)])
def uploaded_datasets(owner_id: str = Depends(verify_api_key)):
    items = datasets.list_datasets(owner_id)
    return {"items": items, "total": len(items)}


@router.delete("/datasets/{dataset_id}", status_code=204, dependencies=[Depends(verify_api_key)])
def remove_uploaded_dataset(dataset_id: str, owner_id: str = Depends(verify_api_key)):
    try:
        deleted = datasets.delete_dataset(dataset_id, owner_id)
    except OSError as exc:
        raise HTTPException(status_code=500, detail="The uploaded CSV could not be safely removed") from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Uploaded dataset not found")
    _cached_paysim_graph.cache_clear()


@router.post("/datasets/upload", response_model=schemas.UploadedDataset, status_code=201, dependencies=[Depends(verify_api_key)])
async def upload_dataset(file: UploadFile = File(...), owner_id: str = Depends(verify_api_key)):
    filename = file.filename or ""
    if Path(filename).suffix.lower() != ".csv":
        raise HTTPException(status_code=415, detail="Choose a .csv file. Other file types are not supported.")
    datasets.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    generated_id = uuid.uuid4().hex
    staging_path = datasets.UPLOAD_DIR / f".{generated_id}.part"
    destination = datasets.UPLOAD_DIR / f"{generated_id}.csv"
    size = 0
    try:
        with staging_path.open("xb") as saved_file:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > datasets.max_upload_bytes():
                    limit_mb = datasets.max_upload_bytes() / (1024 * 1024)
                    detail = f"CSV files must be {limit_mb:g} MB or smaller."
                    if _laptop_mode():
                        detail += " Use Colab for larger files."
                    raise HTTPException(status_code=413, detail=detail)
                saved_file.write(chunk)
        info = datasets.inspect_csv(staging_path)
        staging_path.replace(destination)
        result = datasets.register_dataset(filename, destination, info, owner_id)
        return result
    except DatasetValidationError as exc:
        staging_path.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HTTPException:
        staging_path.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise
    except OSError as exc:
        staging_path.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        if exc.errno == errno.ENOSPC:
            raise HTTPException(status_code=507, detail="Not enough disk space to store or inspect this CSV. Free space and retry.") from exc
        raise HTTPException(status_code=500, detail="The uploaded CSV could not be stored safely.") from exc
    except BaseException:
        staging_path.unlink(missing_ok=True)
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@router.get("/datasets/{dataset_id}", response_model=schemas.UploadedDataset, dependencies=[Depends(verify_api_key)])
def uploaded_dataset_summary(dataset_id: str, owner_id: str = Depends(verify_api_key)):
    result = datasets.get_dataset(dataset_id, owner_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Uploaded dataset not found")
    result.pop("path", None)
    return result


@router.get("/datasets/{dataset_id}/preview", response_model=schemas.DatasetPreviewResponse, dependencies=[Depends(verify_api_key)])
def uploaded_dataset_preview(dataset_id: str, limit: int = Query(default=10, ge=1, le=50), offset: int = Query(default=0, ge=0), owner_id: str = Depends(verify_api_key)):
    try:
        result = datasets.get_preview(dataset_id, limit=limit, offset=offset, owner_id=owner_id)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="The stored CSV could not be previewed.") from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Uploaded dataset or CSV file not found")
    return result


def _analyze_uploaded_dataset(dataset_id: str, owner_id: str) -> None:
    dataset = datasets.get_dataset(dataset_id, owner_id)
    if dataset is None:
        datasets.update_prediction_run(dataset_id, owner_id, status="failed", error="Uploaded dataset was removed before analysis started.", completed_at=datetime.now(timezone.utc).isoformat())
        return
    started_at = datetime.now(timezone.utc).isoformat()
    datasets.update_prediction_run(
        dataset_id, owner_id, status="running", started_at=started_at, heartbeat_at=started_at
    )
    scanned = processed = rejected = skipped = 0
    try:
        path = dataset["path"]
        columns = dataset["columns"]
        usecols = list(FEATURE_NAMES) + (["Class"] if "Class" in columns else [])
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=datasets.PREDICTION_CHUNK_SIZE, on_bad_lines="error"):
            chunk_start = scanned
            scanned += len(chunk)
            feature_frame = chunk.loc[:, list(FEATURE_NAMES)]
            numeric, invalid = coerce_feature_frame(feature_frame)
            valid = ~invalid.any(axis=1)
            blank = feature_frame.isna().all(axis=1).to_numpy()
            skipped += int(blank.sum())
            rejected += int((~valid & ~blank).sum())
            valid_indices = np.flatnonzero(valid)
            if len(valid_indices):
                batch_features = numeric.iloc[valid_indices].reset_index(drop=True)
                batch_predictions = predict_transactions_batch(batch_features)
                label_values = chunk["Class"].iloc[valid_indices].tolist() if "Class" in chunk.columns else [None] * len(valid_indices)
                prediction_rows = []
                for relative_index, prediction, label in zip(valid_indices, batch_predictions, label_values):
                    prediction_rows.append({
                        "source_row": chunk_start + int(relative_index) + 1,
                        "Time": float(numeric.iloc[int(relative_index)]["Time"]),
                        "Amount": float(numeric.iloc[int(relative_index)]["Amount"]),
                        **prediction,
                        "recorded_csv_label": None if pd.isna(label) else str(label),
                    })
                datasets.insert_dataset_predictions(dataset_id, prediction_rows, owner_id)
                processed += len(prediction_rows)
            datasets.update_prediction_run(dataset_id, owner_id, rows_scanned=scanned, rows_processed=processed,
                                           rows_rejected=rejected, rows_skipped=skipped,
                                           heartbeat_at=datetime.now(timezone.utc).isoformat())
        datasets.update_prediction_run(dataset_id, owner_id, status="completed", rows_scanned=scanned,
                                       rows_processed=processed, rows_rejected=rejected,
                                       rows_skipped=skipped, error=None,
                                       completed_at=datetime.now(timezone.utc).isoformat(),
                                       heartbeat_at=None)
    except Exception:
        logger.exception("Uploaded dataset inference failed for dataset %s", dataset_id)
        datasets.update_prediction_run(dataset_id, owner_id, status="failed", rows_scanned=scanned,
                                       rows_processed=processed, rows_rejected=rejected,
                                       rows_skipped=skipped,
                                       error="Analysis failed while reading the uploaded CSV or running the saved model.",
                                       completed_at=datetime.now(timezone.utc).isoformat(),
                                       heartbeat_at=None)


@router.post("/datasets/{dataset_id}/analysis", response_model=schemas.UploadedAnalysisRun, status_code=202, dependencies=[Depends(verify_api_key)])
def start_uploaded_dataset_analysis(dataset_id: str, background_tasks: BackgroundTasks, owner_id: str = Depends(verify_api_key)):
    if _laptop_mode():
        raise HTTPException(status_code=503, detail="Uploaded dataset analysis must be precomputed on Colab.")
    dataset = datasets.get_dataset(dataset_id, owner_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Uploaded dataset not found")
    if not dataset["supports"]["transaction_risk"]:
        raise HTTPException(status_code=422, detail=dataset["supports"]["reasons"]["transaction_risk"])
    current = datasets.get_prediction_run(dataset_id, owner_id)
    if current and current["status"] in {"queued", "running"}:
        return current
    if current and current["status"] == "completed":
        return current
    run, queued = datasets.queue_prediction_run(dataset_id, dataset["row_count"], owner_id)
    if queued:
        job_executor.submit(background_tasks, lambda: _analyze_uploaded_dataset(dataset_id, owner_id))
    return run


@router.get("/datasets/{dataset_id}/analysis", response_model=schemas.UploadedAnalysisRun, dependencies=[Depends(verify_api_key)])
def uploaded_dataset_analysis_status(dataset_id: str, owner_id: str = Depends(verify_api_key)):
    if datasets.get_dataset(dataset_id, owner_id) is None:
        raise HTTPException(status_code=404, detail="Uploaded dataset not found")
    run = datasets.get_prediction_run(dataset_id, owner_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Fraud analysis has not been started for this dataset")
    return {"dataset_id": dataset_id, **run}


@router.get("/datasets/{dataset_id}/predictions", response_model=schemas.UploadedDatasetPredictionsResponse, dependencies=[Depends(verify_api_key)])
def uploaded_dataset_predictions(dataset_id: str, limit: int = Query(default=50, ge=1, le=500), offset: int = Query(default=0, ge=0), owner_id: str = Depends(verify_api_key)):
    dataset = datasets.get_dataset(dataset_id, owner_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Uploaded dataset not found")
    run = datasets.get_prediction_run(dataset_id, owner_id)
    if run is None or run["status"] != "completed":
        raise HTTPException(status_code=409, detail="Fraud analysis is not complete for this dataset")
    items, total = datasets.get_dataset_predictions(dataset_id, limit, offset, owner_id)
    return {"dataset_id": dataset_id, "filename": dataset["filename"],
            "rows_processed": run["rows_processed"], "rows_rejected": run["rows_rejected"],
            "rows_skipped": run["rows_skipped"], "items": items, "total": total,
            "limit": limit, "offset": offset,
            "model_name": model_store.get_metadata().get("model_name", "saved credit-card model"),
            "model_version": model_store.get_metadata().get("model_version", "unknown")}


@router.post("/datasets/{dataset_id}/transactions/{source_row}/risk", response_model=schemas.UploadedPredictionResponse, dependencies=[Depends(verify_api_key)])
def check_uploaded_transaction_risk(dataset_id: str, source_row: int, owner_id: str = Depends(verify_api_key)):
    dataset = datasets.get_dataset(dataset_id, owner_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Uploaded dataset not found")
    if not dataset["supports"]["transaction_risk"]:
        raise HTTPException(status_code=422, detail=dataset["supports"]["reasons"]["transaction_risk"])
    row = datasets.get_row(dataset_id, source_row, owner_id)
    if row is None:
        raise HTTPException(status_code=404, detail="CSV transaction row not found")
    _, transaction = row
    try:
        features = coerce_feature_values(transaction)
        prediction = predict_transaction(features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="The saved credit-card model is unavailable") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        **prediction,
        "transaction_id": f"{dataset_id}:row:{source_row:09d}",
        "dataset_id": dataset_id,
        "source_row": source_row,
        "model_name": model_store.get_metadata().get("model_name", "saved credit-card model"),
        "model_version": model_store.get_metadata().get("model_version", "unknown"),
        "transaction": transaction,
    }


@router.get("/health", dependencies=[Depends(verify_api_key)])
def health():
    model_ready = MODEL_PATH.is_file() and model_store.is_ready()
    return {"status": "ok", "model": "ready" if model_ready else "unavailable"}


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    dependencies=[Depends(verify_api_key)],
)
def model_info():
    if not MODEL_PATH.is_file() or not model_store.is_ready():
        raise HTTPException(status_code=503, detail="Prediction model is unavailable")
    current_metadata = model_store.get_metadata()
    threshold = model_store.get_threshold()
    feature_names = model_store.get_feature_names()
    risk_bands = current_metadata.get("risk_bands", {})
    return {
        "model_name": current_metadata.get("model_name", "xgb_weighted"),
        "model_version": current_metadata.get("model_version", "local"),
        "threshold": threshold,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "risk_cutoffs": {
            "low_max_exclusive": float(risk_bands.get("low_max_exclusive", 0.30)),
            "medium_max_exclusive": float(risk_bands.get("medium_max_exclusive", 0.70)),
        },
    }


@router.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
)
def predict(transaction: TransactionFeatures):
    try:
        return predict_transaction(transaction.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post(
    "/predict/batch",
    response_model=BatchResponse,
    dependencies=[Depends(verify_api_key)],
)
def predict_batch(request: BatchRequest):
    try:
        predictions = [
            predict_transaction(transaction.model_dump())
            for transaction in request.transactions
        ]

        return {"predictions": predictions, "count": len(predictions)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _dataset_or_503():
    if _laptop_mode() and not PRECOMPUTED_PATH.is_file():
        raise HTTPException(status_code=503, detail="Transactions must be precomputed on Colab and copied into models/.")
    try:
        return get_transaction_frame()
    except DatasetUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get(
    "/transactions",
    response_model=TransactionListResponse,
    dependencies=[Depends(verify_api_key)],
)
def list_transactions(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    search: str | None = Query(default=None, max_length=100),
    min_amount: float | None = Query(default=None, ge=0),
    max_amount: float | None = Query(default=None, ge=0),
    class_label: int | None = Query(default=None, ge=0, le=1),
    sort_by: Literal["source_row", "Time", "Amount", "Class"] = "source_row",
    sort_order: Literal["asc", "desc"] = "desc",
):
    if min_amount is not None and max_amount is not None and min_amount > max_amount:
        raise HTTPException(status_code=422, detail="min_amount must not exceed max_amount")

    frame, _, dataset_version = _dataset_or_503()
    filtered = frame
    if search:
        filtered = filtered[
            filtered["transaction_id"].str.contains(search, regex=False, case=False)
        ]
    if min_amount is not None:
        filtered = filtered[filtered["Amount"] >= min_amount]
    if max_amount is not None:
        filtered = filtered[filtered["Amount"] <= max_amount]
    if class_label is not None:
        filtered = filtered[filtered[TARGET_COL] == class_label]

    total = len(filtered)
    page = filtered.sort_values(
        sort_by, ascending=sort_order == "asc", kind="stable"
    ).iloc[offset : offset + limit]
    return {
        "data_source": DATA_SOURCE,
        "dataset_version": dataset_version,
        "label_field": "Class",
        "label_semantics": "This is the label recorded in the source CSV, not a model prediction.",
        "items": [transaction_record(row, dataset_version) for _, row in page.iterrows()],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get(
    "/transactions/summary",
    response_model=DatasetSummaryResponse,
    dependencies=[Depends(verify_api_key)],
)
def transaction_summary():
    frame, _, dataset_version = _dataset_or_503()
    count = len(frame)
    positives = int(frame[TARGET_COL].sum())
    return {
        "data_source": DATA_SOURCE,
        "dataset_version": dataset_version,
        "label_field": "Class",
        "label_semantics": "This is the label recorded in the source CSV.",
        "record_count": count,
        "positive_count": positives,
        "negative_count": count - positives,
        "positive_rate": positives / count if count else 0.0,
        "total_amount": float(frame["Amount"].sum()),
        "mean_amount": float(frame["Amount"].mean()) if count else 0.0,
        "time_min_seconds": float(frame["Time"].min()) if count else 0.0,
        "time_max_seconds": float(frame["Time"].max()) if count else 0.0,
    }


@router.get(
    "/transactions/payment-activity",
    response_model=PaymentActivityResponse,
    dependencies=[Depends(verify_api_key)],
)
def payment_activity(
    interval_seconds: int = Query(default=3600, ge=60, le=86_400),
    limit: int = Query(default=500, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
):
    frame, _, dataset_version = _dataset_or_503()
    grouped = frame.assign(
        _period=(frame["Time"] // interval_seconds) * interval_seconds
    ).groupby("_period", sort=True)
    buckets = grouped.agg(
        transaction_count=(TARGET_COL, "size"),
        fraud_count=(TARGET_COL, "sum"),
        total_amount=("Amount", "sum"),
        mean_amount=("Amount", "mean"),
    ).reset_index()
    page = buckets.iloc[offset : offset + limit]
    return {
        "data_source": DATA_SOURCE,
        "dataset_version": dataset_version,
        "time_unit": "seconds_from_first_transaction",
        "interval_seconds": interval_seconds,
        "buckets": [
            {
                "period_start_seconds": float(row["_period"]),
                "transaction_count": int(row["transaction_count"]),
                "fraud_count": int(row["fraud_count"]),
                "total_amount": float(row["total_amount"]),
                "mean_amount": float(row["mean_amount"]),
            }
            for _, row in page.iterrows()
        ],
    }


@router.post(
    "/transactions/{transaction_id}/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(verify_api_key)],
)
def predict_dataset_transaction(transaction_id: str):
    frame, _, _ = _dataset_or_503()
    match = frame[frame["transaction_id"] == transaction_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Transaction not found")
    row = match.iloc[0]
    features = {name: float(row[name]) for name in FEATURE_NAMES}
    try:
        return predict_transaction(features)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get(
    "/transactions/{transaction_id}",
    response_model=schemas.TransactionRecord,
    dependencies=[Depends(verify_api_key)],
)
def transaction_detail(transaction_id: str):
    frame, _, dataset_version = _dataset_or_503()
    match = frame[frame["transaction_id"] == transaction_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return transaction_record(match.iloc[0], dataset_version)


@router.get(
    "/evaluation/report",
    response_model=EvaluationReportResponse,
    dependencies=[Depends(verify_api_key)],
)
def evaluation_report():
    try:
        with REPORT_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        return {
            "available": False,
            "reason": "Evaluation report has not been generated",
            "evaluation_scope": "internal stratified holdout from the same dataset",
            "evaluation_date_utc": None,
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
            "feature_schema": FEATURE_NAMES,
            "metrics": None,
            "confusion_matrix": None,
            "curves": None,
        }
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail="Evaluation report is invalid") from exc


@router.get("/evaluation/graph", dependencies=[Depends(verify_api_key)])
def graph_evaluation_report():
    try:
        with GRAPH_REPORT_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Graph evaluation report has not been generated") from None
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=503, detail="Graph evaluation report is invalid") from exc


@router.get("/investigations", response_model=schemas.InvestigationCaseList, dependencies=[Depends(verify_api_key)])
def list_investigations(owner_id: str = Depends(verify_api_key)):
    items = investigations.list_cases(owner_id)
    return {"items": items, "total": len(items)}


@router.post("/investigations", response_model=schemas.InvestigationCase, status_code=201, dependencies=[Depends(verify_api_key)])
def create_investigation(request: schemas.InvestigationCaseCreate, owner_id: str = Depends(verify_api_key)):
    frame, _, version = _dataset_or_503()
    match = frame[frame["transaction_id"] == request.transaction_id]
    if match.empty:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return investigations.create_case(match.iloc[0], version, request.severity, request.notes, owner_id)


@router.patch("/investigations/{case_id}", response_model=schemas.InvestigationCase, dependencies=[Depends(verify_api_key)])
def patch_investigation(case_id: str, request: schemas.InvestigationCaseUpdate, owner_id: str = Depends(verify_api_key)):
    result = investigations.update_case(case_id, request.model_dump(exclude_unset=True), owner_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Investigation case not found")
    return result


@router.delete("/investigations/{case_id}", status_code=204, dependencies=[Depends(verify_api_key)])
def remove_investigation(case_id: str, owner_id: str = Depends(verify_api_key)):
    if not investigations.delete_case(case_id, owner_id):
        raise HTTPException(status_code=404, detail="Investigation case not found")


@router.get("/graph/paysim", response_model=schemas.TransactionGraphResponse, dependencies=[Depends(verify_api_key)])
def paysim_graph(
    limit: int = Query(default=100, ge=10, le=500),
    max_rows: int = Query(default=100000, ge=1000, le=GRAPH_MAX_ROWS),
    dataset_id: str | None = Query(default=None),
    account_id: str | None = Query(default=None, max_length=255),
    neighbor_limit: int = Query(default=24, ge=1, le=500),
    owner_id: str = Depends(verify_api_key),
):
    if dataset_id:
        dataset = datasets.get_dataset(dataset_id, owner_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail="Uploaded dataset not found")
        if not dataset["supports"]["account_relationships"]:
            raise HTTPException(status_code=422, detail=dataset["supports"]["reasons"]["account_relationships"])
        graph_path = dataset["path"]
        data_source = f"Uploaded PaySim-compatible CSV: {dataset['filename']}"
    else:
        graph_path = PAYSIM_PATH
        data_source = "PaySim synthetic financial dataset"
    if not dataset_id and not account_id:
        precomputed = _precomputed_graph(graph_path)
        if precomputed is not None:
            precomputed = dict(precomputed)
            precomputed["nodes"] = precomputed["nodes"][:limit]
            precomputed["suspicious_accounts"] = precomputed["suspicious_accounts"][:limit]
            precomputed["edges"] = precomputed["edges"][: limit * 2]
            return precomputed
    if not graph_path.is_file():
        raise HTTPException(status_code=503, detail="PaySim dataset is not available")
    if not _live_graph_enabled():
        raise HTTPException(
            status_code=503,
            detail="PaySim graph must be precomputed on Colab and copied into models/graph_payload.json.",
        )
    try:
        stat = graph_path.stat()
        if stat.st_size > GRAPH_MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail="PaySim dataset exceeds the configured graph size limit")
        started = time.monotonic()
        rows, edges_frame, graph, digest = _cached_paysim_graph(
            str(graph_path.resolve()), stat.st_mtime_ns, stat.st_ctime_ns,
            stat.st_size, stat.st_ino, max_rows
        )
        if time.monotonic() - started > GRAPH_TIMEOUT_SECONDS:
            raise HTTPException(status_code=503, detail="PaySim graph construction exceeded the configured time limit")
    except (OSError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=503, detail="PaySim graph could not be constructed") from exc

    metrics = graph_node_metrics(graph)
    scored = {item["id"]: item for item in score_accounts(graph)}
    ranked = suspicious_accounts(graph)

    def account_detail(account):
        if str(account) in scored:
            return scored[str(account)]
        raw = metrics[account]
        return {
            "id": str(account), "degree": raw["in_degree"] + raw["out_degree"],
            "in_degree": raw["in_degree"], "out_degree": raw["out_degree"],
            "total_amount": raw["total_amount"], "transaction_count": raw["transaction_count"],
            "fraud_count": raw["fraud_count"], "risk_score": 0.0, "reasons": [],
        }
    account_ids = {node["id"] for node in ranked[:limit]}
    focus_detail = None
    available_connections = 0
    neighborhood_truncated = False
    if account_id:
        if account_id not in graph:
            raise HTTPException(status_code=404, detail="Account is not present in the configured dataset sample")
        focus_detail = account_detail(account_id)
        incident = edges_frame[(edges_frame["source"].astype(str) == account_id) | (edges_frame["target"].astype(str) == account_id)]
        available_connections = len(incident)
        neighborhood_truncated = available_connections > neighbor_limit
        selected_frame = incident.nlargest(neighbor_limit, ["tx_count", "total_amount"])
        account_ids = {account_id}
    else:
        selected_frame = edges_frame[
            edges_frame["source"].astype(str).isin(account_ids) | edges_frame["target"].astype(str).isin(account_ids)
        ].nlargest(limit * 2, ["tx_count", "total_amount"])
    selected_edges = selected_frame.to_dict("records")
    row_lookup = rows.set_index("_source_row") if "_source_row" in rows.columns else None
    output_edges = []
    for edge in selected_edges[:limit * 2]:
        evidence = []
        if row_lookup is not None:
            for source_row in edge.get("evidence_rows", []) or []:
                if source_row not in row_lookup.index:
                    continue
                transaction = row_lookup.loc[source_row]
                evidence.append({
                    "source_row": int(source_row), "step": float(transaction["step"]),
                    "type": str(transaction["type"]), "amount": float(transaction["amount"]),
                    "simulator_fraud_label": int(transaction["isFraud"]),
                })
        output_edges.append({"source": str(edge["source"]), "target": str(edge["target"]),
                             "total_amount": float(edge["total_amount"]), "transaction_count": int(edge["tx_count"]),
                             "fraud_count": int(edge["fraud_count"]), "evidence": evidence})
    relevant_ids = account_ids | {edge[key] for edge in output_edges for key in ("source", "target")}
    nodes = ([focus_detail] if focus_detail else ranked[:limit]) + [account_detail(node_id) for node_id in sorted(relevant_ids - account_ids)]
    eligible_rows = int(rows.attrs.get("eligible_rows_seen", len(rows)))
    sampled = bool(rows.attrs.get("sampled", False))
    sampling_ratio = min(1.0, len(rows) / eligible_rows) if eligible_rows else 1.0
    return {"data_source": data_source, "dataset_version": f"paysim-{digest}",
            "sampled": sampled, "sampling_ratio": sampling_ratio, "nodes": nodes, "edges": output_edges,
            "suspicious_accounts": ranked[:limit], "focus_account": account_id,
            "neighborhood_truncated": neighborhood_truncated, "available_connections": available_connections}
