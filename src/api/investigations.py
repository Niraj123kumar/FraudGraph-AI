"""SQLite-backed investigation case storage."""
from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from src.data.transactions import transaction_record
from src.data.sqlite import connect_sqlite

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.getenv("FRAUDGRAPH_CASES_DB", PROJECT_ROOT / "data" / "investigations.sqlite3"))


def _connect() -> sqlite3.Connection:
    connection = connect_sqlite(DB_PATH)
    connection.execute("""CREATE TABLE IF NOT EXISTS investigation_cases (
        id TEXT PRIMARY KEY, owner_id TEXT NOT NULL DEFAULT 'legacy', transaction_id TEXT NOT NULL, dataset_version TEXT NOT NULL,
        source_row INTEGER NOT NULL, time REAL NOT NULL, amount REAL NOT NULL,
        class_label INTEGER NOT NULL, severity TEXT NOT NULL, status TEXT NOT NULL,
        notes TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    )""")
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(investigation_cases)")}
    if "owner_id" not in columns:
        connection.execute("ALTER TABLE investigation_cases ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'legacy'")
    return connection


def list_cases(owner_id: str = "legacy") -> list[dict]:
    with _connect() as db:
        rows = db.execute("SELECT * FROM investigation_cases WHERE owner_id = ? ORDER BY updated_at DESC", (owner_id,)).fetchall()
    return [_serialize(row) for row in rows]


def create_case(row, dataset_version: str, severity: str, notes: str, owner_id: str = "legacy") -> dict:
    record = transaction_record(row, dataset_version)
    now = datetime.now(timezone.utc).isoformat()
    item = {**record, "id": f"CASE-{uuid.uuid4().hex[:12].upper()}", "owner_id": owner_id, "severity": severity,
            "status": "Open", "notes": notes, "created_at": now, "updated_at": now}
    with _connect() as db:
        db.execute("""INSERT INTO investigation_cases
            (id, owner_id, transaction_id, dataset_version, source_row, time, amount, class_label,
             severity, status, notes, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                   (item["id"], owner_id, record["transaction_id"], dataset_version, record["source_row"],
                    record["Time"], record["Amount"], record["Class"], severity, "Open", notes, now, now))
    return item


def update_case(case_id: str, changes: dict, owner_id: str = "legacy") -> dict | None:
    allowed = {key: value for key, value in changes.items() if value is not None}
    if not allowed:
        with _connect() as db:
            row = db.execute("SELECT * FROM investigation_cases WHERE id = ? AND owner_id = ?", (case_id, owner_id)).fetchone()
        return _serialize(row) if row else None
    allowed["updated_at"] = datetime.now(timezone.utc).isoformat()
    columns = {"status": "status", "severity": "severity", "notes": "notes", "updated_at": "updated_at"}
    clause = ", ".join(f"{columns[key]} = ?" for key in allowed)
    with _connect() as db:
        db.execute(f"UPDATE investigation_cases SET {clause} WHERE id = ? AND owner_id = ?", (*allowed.values(), case_id, owner_id))
        row = db.execute("SELECT * FROM investigation_cases WHERE id = ? AND owner_id = ?", (case_id, owner_id)).fetchone()
    return _serialize(row) if row else None


def delete_case(case_id: str, owner_id: str = "legacy") -> bool:
    with _connect() as db:
        cursor = db.execute(
            "DELETE FROM investigation_cases WHERE id = ? AND owner_id = ?",
            (case_id, owner_id),
        )
    return cursor.rowcount == 1


def _serialize(row: sqlite3.Row) -> dict:
    return {"id": row["id"], "transaction_id": row["transaction_id"], "dataset_version": row["dataset_version"],
            "source_row": row["source_row"], "Time": row["time"], "Amount": row["amount"], "Class": row["class_label"],
            "severity": row["severity"], "status": row["status"], "notes": row["notes"],
            "created_at": row["created_at"], "updated_at": row["updated_at"]}
