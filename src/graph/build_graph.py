"""Build an account-transfer graph from the PaySim dataset.

Accounts are nodes. Repeated transfers between the same sender and receiver
are aggregated into one directed edge.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

RANDOM_SEED: int = 42
USE_COLS: list[str] = ["step", "type",
                       "amount", "nameOrig", "nameDest", "isFraud"]
# Fraud in PaySim occurs only in these types; PAYMENT/DEBIT go to merchants.
TRANSFER_TYPES: tuple[str, ...] = ("TRANSFER", "CASH_OUT")
EDGE_ATTRS: list[str] = ["total_amount", "tx_count",
                         "first_step", "last_step", "fraud_count", "evidence_rows"]


def load_paysim(
    path: str | Path,
    types: tuple[str, ...] = TRANSFER_TYPES,
    max_rows: int | None = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Load PaySim, keep selected transaction types, optionally sample rows.

    Args:
        path: Path to paysim.csv.
        types: Transaction types to keep.
        max_rows: If set and smaller than the filtered size, sample this many
            rows with a fixed seed. Sampling drops edges, so graph structure
            is partial; use it only for speed.

    Raises:
        ValueError: If expected columns are missing.
    """
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive")

    if max_rows is None:
        try:
            df = pd.read_csv(path, usecols=USE_COLS)
        except pd.errors.EmptyDataError as exc:
            raise ValueError("PaySim file is empty") from exc
        df["_source_row"] = np.arange(1, len(df) + 1, dtype=np.int64)
        result = df[df["type"].isin(types)].reset_index(drop=True)
        result = _validate_account_ids(result)
        result.attrs["sampled"] = False
        result.attrs["eligible_rows_seen"] = len(result)
        return result

    # Uniform reservoir sampling across the eligible rows avoids materializing
    # the full multi-million-row file. Keep only one CSV chunk plus the sample.
    rng = np.random.default_rng(seed)
    reservoir: pd.DataFrame | None = None
    seen = 0
    source_offset = 0
    try:
        chunks = pd.read_csv(path, usecols=USE_COLS, chunksize=250_000)
        for chunk in chunks:
            chunk["_source_row"] = np.arange(source_offset + 1, source_offset + len(chunk) + 1, dtype=np.int64)
            source_offset += len(chunk)
            eligible = chunk[chunk["type"].isin(types)]
            count = len(eligible)
            if not count:
                continue
            if seen + count <= max_rows:
                reservoir = eligible.copy() if reservoir is None else pd.concat((reservoir, eligible), ignore_index=True)
                seen += count
                continue

            keep_new = int(rng.hypergeometric(count, seen, max_rows))
            keep_old = max_rows - keep_new
            old_sample = reservoir.sample(
                n=keep_old,
                random_state=int(rng.integers(0, 2**32 - 1)),
            ) if keep_old and reservoir is not None else None
            new_sample = eligible.sample(
                n=keep_new,
                random_state=int(rng.integers(0, 2**32 - 1)),
            ) if keep_new else eligible.iloc[:0]
            reservoir = new_sample.reset_index(drop=True) if old_sample is None else pd.concat((old_sample, new_sample), ignore_index=True)
            seen += count
    except pd.errors.EmptyDataError as exc:
        raise ValueError("PaySim file is empty") from exc

    result = (reservoir if reservoir is not None else pd.DataFrame(columns=USE_COLS)).reset_index(drop=True)
    result = _validate_account_ids(result)
    result.attrs["sampled"] = seen > max_rows
    result.attrs["eligible_rows_seen"] = seen
    return result


def _validate_account_ids(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    invalid = (
        frame["nameOrig"].isna()
        | frame["nameDest"].isna()
        | frame["nameOrig"].astype(str).str.strip().eq("")
        | frame["nameDest"].astype(str).str.strip().eq("")
    )
    if invalid.any():
        raise ValueError("PaySim account IDs must be non-empty")
    return frame


def aggregate_transfers(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated sender->receiver transfers into one row per pair."""
    aggregation = {
        "total_amount": ("amount", "sum"),
        "tx_count": ("amount", "size"),
        "first_step": ("step", "min"),
        "last_step": ("step", "max"),
        "fraud_count": ("isFraud", "sum"),
    }
    if "_source_row" in df.columns:
        aggregation["evidence_rows"] = ("_source_row", lambda values: [int(value) for value in values.iloc[:10]])
    edges = df.groupby(["nameOrig", "nameDest"], sort=False).agg(**aggregation).reset_index()
    return edges.rename(columns={"nameOrig": "source", "nameDest": "target"})


def build_graph(edges: pd.DataFrame) -> nx.DiGraph:
    """Build a DiGraph from aggregated edges.

    Each node gets ``is_fraud_account`` = True if any incident edge carried a
    fraudulent transfer. This label is for validation only, not a feature.
    """
    graph = nx.from_pandas_edgelist(
        edges, "source", "target", edge_attr=[name for name in EDGE_ATTRS if name in edges.columns], create_using=nx.DiGraph
    )
    flagged = edges[edges["fraud_count"] > 0]
    fraud_nodes = set(flagged["source"]) | set(flagged["target"])
    nx.set_node_attributes(
        graph, {n: n in fraud_nodes for n in graph.nodes}, "is_fraud_account")
    return graph


def load_graph(
    path: str | Path = "data/raw/paysim.csv",
    max_rows: int | None = None,
    seed: int = RANDOM_SEED,
) -> nx.DiGraph:
    """Load PaySim and return the aggregated account-transfer graph."""
    return build_graph(aggregate_transfers(load_paysim(path, max_rows=max_rows, seed=seed)))
