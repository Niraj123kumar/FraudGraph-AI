"""Evaluate graph suspicion rankings against PaySim labels."""

from __future__ import annotations

import json
import argparse
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score

from src.graph.build_graph import aggregate_transfers, load_paysim
from src.graph.suspicious_groups import score_accounts

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_REPORT_PATH = PROJECT_ROOT / "models" / "graph_evaluation.json"


def run_graph_evaluation(
    data_path: Path,
    report_path: Path = GRAPH_REPORT_PATH,
    max_rows: int | None = None,
) -> dict:
    rows = load_paysim(data_path, max_rows=max_rows)
    scoring_rows = rows.copy()
    scoring_rows["isFraud"] = 0
    graph = nx.from_pandas_edgelist(
        aggregate_transfers(scoring_rows),
        "source",
        "target",
        edge_attr=True,
        create_using=nx.DiGraph,
    )
    labels = np.asarray(
        [int(any(rows.loc[rows["nameOrig"].eq(node) | rows["nameDest"].eq(node), "isFraud"])) for node in graph.nodes],
        dtype=int,
    )
    scored = {item["id"]: item["risk_score"] for item in score_accounts(graph)}
    scores = np.asarray([scored[str(node)] for node in graph.nodes], dtype=float)
    positive_rate = float(labels.mean()) if len(labels) else 0.0
    result = {
        "available": True,
        "evaluation_scope": "internal PaySim fraud labels used only for graph heuristic evaluation",
        "sampled": bool(rows.attrs.get("sampled", False)),
        "sampling_ratio": min(1.0, len(rows) / int(rows.attrs.get("eligible_rows_seen", len(rows)))) if len(rows) else 1.0,
        "node_count": int(len(labels)),
        "positive_node_count": int(labels.sum()),
        "metrics": {
            "precision_at_k": {
                str(k): float(labels[np.argsort(-scores)[: min(k, len(labels))]].mean()) if len(labels) else 0.0
                for k in (50, 100, 500)
            },
            "roc_auc": float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None,
            "random_baseline_precision_at_k": {str(k): positive_rate for k in (50, 100, 500)},
        },
        "warning": "Scores use graph structure only; fraud labels are evaluation-only.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paysim", type=Path, default=PROJECT_ROOT / "data" / "raw" / "paysim.csv")
    parser.add_argument("--out", type=Path, default=GRAPH_REPORT_PATH)
    args = parser.parse_args()
    print(json.dumps(run_graph_evaluation(args.paysim, args.out), indent=2))
