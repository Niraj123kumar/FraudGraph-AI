"""Persisted, isolated user CSV datasets and bounded quality inspection."""
from __future__ import annotations

import hashlib
import json
import csv
import sqlite3
import tempfile
import uuid
from datetime import datetime, timezone
import os
import warnings
from pathlib import Path

import pandas as pd

from src.ml.predict import FEATURE_NAMES, MODEL_PATH, model_store
from src.data.sqlite import connect_sqlite

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
REGISTRY_PATH = PROJECT_ROOT / "data" / "datasets.sqlite3"
MAX_UPLOAD_BYTES = int(os.getenv("FRAUDGRAPH_MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
CHUNK_SIZE = 50_000
PREDICTION_CHUNK_SIZE = 10_000
PAYSIM_REQUIRED_COLUMNS = ("step", "type", "amount", "nameOrig", "nameDest", "isFraud")
JOB_STALE_SECONDS = int(os.getenv("FRAUDGRAPH_JOB_STALE_SECONDS", "300"))


def max_upload_bytes() -> int:
    configured = os.getenv("FRAUDGRAPH_MAX_UPLOAD_BYTES")
    if configured is not None:
        return int(configured)
    if os.getenv("FRAUDGRAPH_LAPTOP_MODE", "1") == "1":
        return 20 * 1024 * 1024
    return MAX_UPLOAD_BYTES


class DatasetValidationError(ValueError):
    pass


def _connect() -> sqlite3.Connection:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = connect_sqlite(REGISTRY_PATH)
    db.execute("""CREATE TABLE IF NOT EXISTS datasets (
        id TEXT PRIMARY KEY, owner_id TEXT NOT NULL DEFAULT 'legacy', filename TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL,
        uploaded_at TEXT NOT NULL, row_count INTEGER NOT NULL, columns_json TEXT NOT NULL,
        dtypes_json TEXT NOT NULL, missing_json TEXT NOT NULL, duplicate_count INTEGER NOT NULL,
        mapping_json TEXT NOT NULL DEFAULT '{}'
    )""")
    columns = {row["name"] for row in db.execute("PRAGMA table_info(datasets)")}
    if "owner_id" not in columns:
        db.execute("ALTER TABLE datasets ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'legacy'")
    db.execute("""CREATE TABLE IF NOT EXISTS prediction_runs (
        dataset_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL DEFAULT 'legacy', status TEXT NOT NULL, rows_total INTEGER NOT NULL,
        rows_scanned INTEGER NOT NULL DEFAULT 0, rows_processed INTEGER NOT NULL DEFAULT 0,
        rows_rejected INTEGER NOT NULL DEFAULT 0, rows_skipped INTEGER NOT NULL DEFAULT 0,
        error TEXT, started_at TEXT, completed_at TEXT, heartbeat_at TEXT
    )""")
    columns = {row["name"] for row in db.execute("PRAGMA table_info(prediction_runs)")}
    if "heartbeat_at" not in columns:
        db.execute("ALTER TABLE prediction_runs ADD COLUMN heartbeat_at TEXT")
    columns = {row["name"] for row in db.execute("PRAGMA table_info(prediction_runs)")}
    if "owner_id" not in columns:
        db.execute("ALTER TABLE prediction_runs ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'legacy'")
    db.execute("""CREATE TABLE IF NOT EXISTS dataset_predictions (
        dataset_id TEXT NOT NULL, owner_id TEXT NOT NULL DEFAULT 'legacy', source_row INTEGER NOT NULL, time_value REAL NOT NULL, amount REAL NOT NULL, probability REAL NOT NULL,
        is_fraud INTEGER NOT NULL, risk_level TEXT NOT NULL, recorded_csv_label TEXT,
        PRIMARY KEY(dataset_id, source_row)
    )""")
    columns = {row["name"] for row in db.execute("PRAGMA table_info(dataset_predictions)")}
    if "owner_id" not in columns:
        db.execute("ALTER TABLE dataset_predictions ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'legacy'")
    return db


def queue_prediction_run(dataset_id: str, rows_total: int, owner_id: str = "legacy") -> dict:
    queued = False
    with _connect() as db:
        cursor = db.execute("""INSERT INTO prediction_runs
            (dataset_id, owner_id, status, rows_total, rows_scanned, rows_processed, rows_rejected, rows_skipped, error, started_at, completed_at)
            VALUES (?, ?, 'queued', ?, 0, 0, 0, 0, NULL, NULL, NULL)
            ON CONFLICT(dataset_id) DO UPDATE SET status='queued', rows_total=excluded.rows_total,
            owner_id=excluded.owner_id,
            rows_scanned=0, rows_processed=0, rows_rejected=0, rows_skipped=0, error=NULL,
            started_at=NULL, completed_at=NULL
            WHERE prediction_runs.status NOT IN ('queued', 'running', 'completed')""", (dataset_id, owner_id, rows_total))
        queued = cursor.rowcount > 0
        if queued:
            db.execute("DELETE FROM dataset_predictions WHERE dataset_id = ? AND owner_id = ?", (dataset_id, owner_id))
    return get_prediction_run(dataset_id, owner_id), queued


def update_prediction_run(dataset_id: str, owner_id: str = "legacy", **values) -> None:
    allowed = {
        "status", "rows_scanned", "rows_processed", "rows_rejected",
        "rows_skipped", "error", "started_at", "completed_at", "heartbeat_at",
    }
    fields = [(key, value) for key, value in values.items() if key in allowed]
    if not fields:
        return
    assignments = ", ".join(f"{key} = ?" for key, _ in fields)
    with _connect() as db:
        db.execute(f"UPDATE prediction_runs SET {assignments} WHERE dataset_id = ? AND owner_id = ?", [value for _, value in fields] + [dataset_id, owner_id])


def insert_dataset_predictions(dataset_id: str, rows: list[dict], owner_id: str = "legacy") -> None:
    with _connect() as db:
        db.executemany("""INSERT OR REPLACE INTO dataset_predictions
                (dataset_id, owner_id, source_row, time_value, amount, probability, is_fraud, risk_level, recorded_csv_label)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", [
                (dataset_id, owner_id, row["source_row"], row["Time"], row["Amount"], row["probability"], int(row["is_fraud"]), row["risk_level"], row["recorded_csv_label"])
                for row in rows
            ])


def get_prediction_run(dataset_id: str, owner_id: str = "legacy") -> dict | None:
    with _connect() as db:
        row = db.execute("SELECT * FROM prediction_runs WHERE dataset_id = ? AND owner_id = ?", (dataset_id, owner_id)).fetchone()
    return dict(row) if row else None


def fail_interrupted_prediction_runs() -> None:
    """Mark jobs left behind by a server restart so users can retry them."""
    completed_at = datetime.now(timezone.utc).isoformat()
    stale_before = datetime.now(timezone.utc).timestamp() - JOB_STALE_SECONDS
    with _connect() as db:
        rows = db.execute(
            "SELECT dataset_id, heartbeat_at, started_at FROM prediction_runs WHERE status = 'running'"
        ).fetchall()
        stale_ids = []
        for row in rows:
            heartbeat = row["heartbeat_at"] or row["started_at"]
            try:
                is_stale = datetime.fromisoformat(heartbeat).timestamp() < stale_before
            except (TypeError, ValueError):
                is_stale = True
            if is_stale:
                stale_ids.append(row["dataset_id"])
        if stale_ids:
            placeholders = ",".join("?" for _ in stale_ids)
            db.execute(f"""UPDATE prediction_runs SET status='failed',
            error='Backend restarted before this analysis completed; start it again to retry.',
            completed_at=?, heartbeat_at=NULL WHERE dataset_id IN ({placeholders})""",
                        [completed_at, *stale_ids])


def get_dataset_predictions(dataset_id: str, limit: int, offset: int, owner_id: str = "legacy") -> tuple[list[dict], int]:
    with _connect() as db:
        total = db.execute("SELECT COUNT(*) FROM dataset_predictions WHERE dataset_id = ? AND owner_id = ?", (dataset_id, owner_id)).fetchone()[0]
        rows = db.execute("""SELECT source_row, time_value AS Time, amount AS Amount, probability, is_fraud, risk_level, recorded_csv_label
            FROM dataset_predictions WHERE dataset_id = ? AND owner_id = ? ORDER BY source_row LIMIT ? OFFSET ?""",
            (dataset_id, owner_id, limit, offset)).fetchall()
    return [dict(row) for row in rows], int(total)


def _model_compatible(columns: list[str]) -> bool:
    return (
        MODEL_PATH.is_file()
        and model_store.is_ready()
        and all(name in columns for name in FEATURE_NAMES)
    )


def _model_support_reason(columns: list[str]) -> str:
    if not MODEL_PATH.is_file() or not model_store.is_ready():
        return "The saved credit-card model or its metadata is unavailable."
    missing = [name for name in FEATURE_NAMES if name not in columns]
    if missing:
        return f"Missing required credit-card model columns: {', '.join(missing)}. Required features are Time, V1–V28, and Amount."
    return "Required model columns are present; rows with missing or non-numeric feature values will be skipped during analysis."


def _supports(columns: list[str], missing: dict[str, int], dtypes: dict[str, str]) -> dict:
    compatible_model = _model_compatible(columns)
    paysim_compatible = (
        all(name in columns and missing.get(name, 0) == 0 for name in PAYSIM_REQUIRED_COLUMNS)
        and pd.api.types.is_numeric_dtype(dtypes.get("step", "object"))
        and pd.api.types.is_numeric_dtype(dtypes.get("amount", "object"))
        and pd.api.types.is_numeric_dtype(dtypes.get("isFraud", "object"))
    )
    return {
        "data_quality": True,
        "transaction_risk": compatible_model,
        "account_relationships": paysim_compatible,
        "evaluation": False,
        "reasons": {
            "transaction_risk": None if compatible_model else _model_support_reason(columns),
            "account_relationships": None if paysim_compatible else "Account connections need step, type, amount, sender, receiver, and simulator-label columns, with no missing values.",
            "evaluation": "Uploaded-file evaluation is not available. The model page reports only the saved, separate holdout evaluation.",
        },
    }


def inspect_csv(path: Path) -> dict:
    quality_db = None
    temp_db_path = None
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        temp_db_file = tempfile.NamedTemporaryFile(dir=UPLOAD_DIR, suffix=".quality.sqlite3", delete=False)
        temp_db_path = Path(temp_db_file.name)
        temp_db_file.close()
        quality_db = sqlite3.connect(temp_db_path)
        quality_db.execute("PRAGMA journal_mode=OFF")
        quality_db.execute("PRAGMA synchronous=OFF")
        quality_db.execute("CREATE TABLE row_hashes (digest TEXT PRIMARY KEY)")
        row_count = duplicate_count = 0
        columns: list[str] = []
        missing_counts: dict[str, int] = {}
        dtypes: dict[str, str] = {}
        preview: list[dict] = []
        digest = hashlib.sha256()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", pd.errors.ParserWarning)
                chunks = pd.read_csv(path, chunksize=CHUNK_SIZE, on_bad_lines="error", index_col=False)
                for chunk in chunks:
                    chunk_columns = [str(column) for column in chunk.columns]
                    if not columns:
                        columns = chunk_columns
                        if not columns:
                            raise DatasetValidationError("The CSV has no column names.")
                        if len(set(columns)) != len(columns):
                            raise DatasetValidationError("Column names must be unique.")
                        missing_counts = {column: 0 for column in columns}
                    elif chunk_columns != columns:
                        raise DatasetValidationError("The CSV column structure changes between rows.")
                    if len(chunk.columns) != len(set(chunk.columns)):
                        raise DatasetValidationError("Column names must be unique.")
                    if not chunk_columns:
                        raise DatasetValidationError("The CSV has no column names.")
                    if not dtypes:
                        dtypes = {str(name): str(dtype) for name, dtype in chunk.dtypes.items()}
                    else:
                        for name, dtype in chunk.dtypes.items():
                            key = str(name)
                            if dtypes.get(key) != str(dtype):
                                dtypes[key] = "mixed"
                    row_count += len(chunk)
                    for column in columns:
                        missing_counts[column] += int(chunk[column].isna().sum())
                    if len(preview) < 10:
                        preview_frame = chunk.head(10 - len(preview)).astype(object)
                        preview.extend(preview_frame.where(pd.notna(preview_frame), None).to_dict("records"))
                    hashes = pd.util.hash_pandas_object(chunk, index=False).astype("uint64")
                    digest.update(chunk.to_csv(index=False).encode("utf-8"))
                    before = quality_db.total_changes
                    quality_db.executemany(
                        "INSERT OR IGNORE INTO row_hashes(digest) VALUES (?)",
                        ((f"{int(value):016x}",) for value in hashes),
                    )
                    duplicate_count += len(chunk) - (quality_db.total_changes - before)
        except (pd.errors.EmptyDataError, pd.errors.ParserWarning) as exc:
            raise DatasetValidationError("The CSV has no column names.") from exc
        if row_count == 0:
            raise DatasetValidationError("The CSV contains a header but no transaction rows.")
        return {
            "row_count": row_count,
            "rows_read": row_count,
            "rows_processed": row_count,
            "rows_rejected": 0,
            "rows_skipped": 0,
            "columns": columns,
            "dtypes": dtypes,
            "missing_values": missing_counts,
            "total_missing_values": sum(missing_counts.values()),
            "duplicate_rows": duplicate_count,
            "preview": preview,
            "sha256": digest.hexdigest(),
            "supports": _supports(columns, missing_counts, dtypes),
        }
    except DatasetValidationError:
        raise
    except (OSError, csv.Error, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise DatasetValidationError("The CSV could not be read. Check its encoding and row structure.") from exc
    finally:
        if quality_db is not None:
            quality_db.close()
        if temp_db_path is not None:
            temp_db_path.unlink(missing_ok=True)


def register_dataset(filename: str, path: Path, info: dict, owner_id: str = "legacy") -> dict:
    dataset_id = str(uuid.uuid4())
    uploaded_at = datetime.now(timezone.utc).isoformat()
    with _connect() as db:
        db.execute(
            """INSERT INTO datasets
            (id, owner_id, filename, path, sha256, uploaded_at, row_count, columns_json, dtypes_json,
             missing_json, duplicate_count, mapping_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}')""",
            (dataset_id, owner_id, Path(filename).name[:255], str(path), info["sha256"], uploaded_at,
             info["row_count"], json.dumps(info["columns"]), json.dumps(info["dtypes"]),
             json.dumps(info["missing_values"]), info["duplicate_rows"]),
        )
    return {"dataset_id": dataset_id, "filename": Path(filename).name[:255], "uploaded_at": uploaded_at, **info}


def get_dataset(dataset_id: str, owner_id: str = "legacy") -> dict | None:
    try:
        uuid.UUID(dataset_id)
    except (ValueError, AttributeError):
        return None
    with _connect() as db:
        row = db.execute("SELECT * FROM datasets WHERE id = ? AND owner_id = ?", (dataset_id, owner_id)).fetchone()
    if row is None:
        return None
    columns = json.loads(row["columns_json"])
    dtypes = json.loads(row["dtypes_json"])
    missing = json.loads(row["missing_json"])
    return {
        "dataset_id": row["id"], "owner_id": row["owner_id"], "filename": row["filename"], "uploaded_at": row["uploaded_at"],
        "row_count": row["row_count"], "rows_read": row["row_count"], "rows_processed": row["row_count"],
        "rows_rejected": 0, "rows_skipped": 0, "columns": columns, "dtypes": dtypes,
        "missing_values": missing, "total_missing_values": sum(missing.values()),
        "duplicate_rows": row["duplicate_count"], "sha256": row["sha256"],
        "supports": _supports(columns, missing, dtypes), "path": Path(row["path"]),
    }


def list_datasets(owner_id: str = "legacy") -> list[dict]:
    with _connect() as db:
        ids = [row[0] for row in db.execute("SELECT id FROM datasets WHERE owner_id = ? ORDER BY uploaded_at DESC", (owner_id,))]
    return [{key: value for key, value in get_dataset(dataset_id, owner_id).items() if key not in {"path", "owner_id"}} for dataset_id in ids]


def delete_dataset(dataset_id: str, owner_id: str = "legacy") -> bool:
    dataset = get_dataset(dataset_id, owner_id)
    if dataset is None:
        return False
    path = dataset["path"]
    if path.parent.resolve() != UPLOAD_DIR.resolve():
        raise OSError("Stored upload path is outside the upload directory")
    path.unlink(missing_ok=True)
    with _connect() as db:
        db.execute("DELETE FROM dataset_predictions WHERE dataset_id = ? AND owner_id = ?", (dataset_id, owner_id))
        db.execute("DELETE FROM prediction_runs WHERE dataset_id = ? AND owner_id = ?", (dataset_id, owner_id))
        db.execute("DELETE FROM datasets WHERE id = ? AND owner_id = ?", (dataset_id, owner_id))
    return True


def get_preview(dataset_id: str, limit: int = 10, offset: int = 0, owner_id: str = "legacy") -> dict | None:
    dataset = get_dataset(dataset_id, owner_id)
    if dataset is None or not dataset["path"].is_file():
        return None
    frame = pd.read_csv(dataset["path"], skiprows=range(1, offset + 1), nrows=limit).astype(object)
    records = frame.where(pd.notna(frame), None).to_dict("records")
    return {"dataset_id": dataset_id, "columns": dataset["columns"], "offset": offset, "items": records}


def get_row(dataset_id: str, source_row: int, owner_id: str = "legacy") -> tuple[dict, dict] | None:
    dataset = get_dataset(dataset_id, owner_id)
    if dataset is None or not dataset["path"].is_file() or source_row < 1 or source_row > dataset["row_count"]:
        return None
    frame = pd.read_csv(dataset["path"], skiprows=range(1, source_row), nrows=1)
    if frame.empty:
        return None
    values = frame.iloc[0].astype(object).where(pd.notna(frame.iloc[0]), None).to_dict()
    return dataset, values
