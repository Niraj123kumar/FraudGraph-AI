from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest

from src.api import investigations
from src.data import datasets
from src.data.preprocessing import EXPECTED_COLS


def _transaction_row():
    row = pd.Series({column: 1.0 for column in EXPECTED_COLS})
    row["Class"] = 0
    row["transaction_id"] = "creditcard-test:row:000000001"
    row["source_row"] = 1
    return row


@pytest.mark.heavy
def test_concurrent_investigation_updates_use_wal_without_lost_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(investigations, "DB_PATH", tmp_path / "cases.sqlite3")
    case = investigations.create_case(_transaction_row(), "creditcard-test", "Medium", "")

    def update(index):
        return investigations.update_case(
            case["id"],
            {"status": "In Review", "notes": f"note-{index}"},
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(update, range(16)))

    assert all(result is not None for result in results)
    stored = investigations.list_cases()[0]
    assert stored["status"] == "In Review"
    assert stored["notes"].startswith("note-")


@pytest.mark.heavy
def test_concurrent_dataset_registrations_do_not_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(datasets, "REGISTRY_PATH", tmp_path / "datasets.sqlite3")
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()
    info = {
        "row_count": 1,
        "columns": ["value"],
        "dtypes": {"value": "int64"},
        "missing_values": {"value": 0},
        "duplicate_rows": 0,
        "sha256": "a" * 64,
        "preview": [{"value": 1}],
        "rows_read": 1,
        "rows_processed": 1,
        "rows_rejected": 0,
        "rows_skipped": 0,
        "total_missing_values": 0,
        "supports": {
            "data_quality": True,
            "transaction_risk": False,
            "account_relationships": False,
            "evaluation": False,
            "reasons": {
                "transaction_risk": "unsupported",
                "account_relationships": "unsupported",
                "evaluation": "unsupported",
            },
        },
    }

    def register(index):
        path = upload_dir / f"{index}.csv"
        path.write_text("value\n1\n", encoding="utf-8")
        return datasets.register_dataset(path.name, path, info)["dataset_id"]

    with ThreadPoolExecutor(max_workers=8) as executor:
        identifiers = list(executor.map(register, range(20)))

    assert len(set(identifiers)) == 20
    assert len(datasets.list_datasets()) == 20
