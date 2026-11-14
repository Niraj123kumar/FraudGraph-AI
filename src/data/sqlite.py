"""Shared SQLite connection configuration."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path


def connect_sqlite(path: Path, timeout: float = 30.0) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=timeout)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=30000")
    connection.execute("PRAGMA foreign_keys=ON")
    for attempt in range(10):
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            break
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == 9:
                connection.close()
                raise
            time.sleep(0.05)
    return connection
