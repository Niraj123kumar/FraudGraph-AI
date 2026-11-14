import pandas as pd
import pytest
import tracemalloc

from src.data import datasets
from src.data.preprocessing import EXPECTED_COLS
from src.data.transactions import DatasetUnavailableError, load_transaction_frame


def _credit_card_frame(rows=3):
    frame = pd.DataFrame(1.0, index=range(rows), columns=EXPECTED_COLS)
    frame["Class"] = [0, 1, 0][:rows]
    return frame


def test_transaction_validation_and_cache_invalidation(tmp_path):
    path = tmp_path / "creditcard.csv"
    frame = _credit_card_frame()
    frame.to_csv(path, index=False)
    load_transaction_frame.cache_clear()
    first, first_hash = load_transaction_frame(str(path))
    assert path.with_suffix(".parquet").is_file()
    frame.loc[0, "Amount"] = 99.0
    frame.to_csv(path, index=False)
    second, second_hash = load_transaction_frame(str(path))
    assert first_hash != second_hash
    assert first.iloc[0]["Amount"] != second.iloc[0]["Amount"]

    invalid = frame.copy()
    invalid["Class"] = 2
    invalid.to_csv(path, index=False)
    with pytest.raises(DatasetUnavailableError, match="binary"):
        load_transaction_frame(str(path))


@pytest.mark.heavy
def test_parquet_cache_is_reused_and_rebuilt_after_source_change(tmp_path, monkeypatch):
    path = tmp_path / "creditcard.csv"
    frame = _credit_card_frame()
    frame.to_csv(path, index=False)
    load_transaction_frame.cache_clear()
    first, _ = load_transaction_frame(str(path))
    parquet = path.with_suffix(".parquet")
    assert parquet.is_file()
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: pytest.fail("CSV should not be reread"))
    second, _ = load_transaction_frame(str(path))
    assert second.equals(first)

    updated = _credit_card_frame()
    updated.loc[0, "Amount"] = 123.0
    updated.to_csv(path, index=False)
    monkeypatch.undo()
    rebuilt, _ = load_transaction_frame(str(path))
    assert rebuilt.iloc[0]["Amount"] == 123.0


@pytest.mark.heavy
def test_large_csv_inspection_keeps_memory_bounded(tmp_path, monkeypatch):
    path = tmp_path / "large.csv"
    pd.DataFrame({"value": range(100_000)}).to_csv(path, index=False)
    monkeypatch.setattr(datasets, "CHUNK_SIZE", 5_000)
    tracemalloc.start()
    info = datasets.inspect_csv(path)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert info["row_count"] == 100_000
    assert peak < 40 * 1024 * 1024


def test_dataset_inspection_duplicates_row_retrieval_and_deletion(tmp_path, monkeypatch):
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(datasets, "UPLOAD_DIR", upload_dir)
    path = tmp_path / "input.csv"
    pd.DataFrame({"step": [1, 1], "type": ["TRANSFER", "TRANSFER"], "amount": [2, 2],
                  "nameOrig": ["A", "A"], "nameDest": ["B", "B"], "isFraud": [0, 0]}).to_csv(path, index=False)
    info = datasets.inspect_csv(path)
    assert info["row_count"] == 2
    assert info["duplicate_rows"] == 1
    stored = upload_dir / "stored.csv"
    stored.write_bytes(path.read_bytes())
    monkeypatch.setattr(datasets, "REGISTRY_PATH", tmp_path / "datasets.sqlite3")
    record = datasets.register_dataset("stored.csv", stored, info)
    result = datasets.get_row(record["dataset_id"], 2)
    assert result is not None
    assert result[1]["nameOrig"] == "A"
    assert datasets.delete_dataset(record["dataset_id"]) is True
    assert not stored.exists()
