"""Hashed API-key storage and authentication."""

from __future__ import annotations

import hashlib
import os
import secrets
import sqlite3
import uuid
from pathlib import Path

from src.data.sqlite import connect_sqlite

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTH_DB_PATH = Path(os.getenv("FRAUDGRAPH_AUTH_DB", PROJECT_ROOT / "data" / "auth.sqlite3"))
LEGACY_OWNER_ID = "legacy"


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def connect_auth() -> sqlite3.Connection:
    db = connect_sqlite(AUTH_DB_PATH)
    db.execute(
        """CREATE TABLE IF NOT EXISTS api_keys (
            key_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, key_hash TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL, revoked_at TEXT
        )"""
    )
    return db


def create_api_key_record(owner_id: str) -> tuple[str, str]:
    if not owner_id.strip():
        raise ValueError("owner_id must not be empty")
    raw_key = secrets.token_urlsafe(32)
    from datetime import datetime, timezone

    key_id = uuid.uuid4().hex
    with connect_auth() as db:
        db.execute(
            "INSERT INTO api_keys VALUES (?, ?, ?, ?, NULL)",
            (key_id, owner_id, _hash_key(raw_key), datetime.now(timezone.utc).isoformat()),
        )
    return key_id, raw_key


def create_api_key(owner_id: str) -> str:
    return create_api_key_record(owner_id)[1]


def revoke_api_key(key_id: str) -> bool:
    with connect_auth() as db:
        cursor = db.execute(
            "UPDATE api_keys SET revoked_at = datetime('now') WHERE key_id = ? AND revoked_at IS NULL",
            (key_id,),
        )
    return cursor.rowcount == 1


def owner_for_key(raw_key: str) -> str | None:
    key_hash = _hash_key(raw_key)
    with connect_auth() as db:
        rows = db.execute(
            "SELECT owner_id, key_hash FROM api_keys WHERE revoked_at IS NULL"
        ).fetchall()
    for row in rows:
        if secrets.compare_digest(row["key_hash"], key_hash):
            return row["owner_id"]
    return None


def single_key_mode_enabled() -> bool:
    return os.getenv("FRAUDGRAPH_SINGLE_KEY_MODE", "").strip().lower() in {"1", "true", "yes"}
