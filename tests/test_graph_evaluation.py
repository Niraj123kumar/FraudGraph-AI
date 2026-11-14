import json

import numpy as np
import pandas as pd

from src.graph.evaluate import run_graph_evaluation
import src.graph.evaluate as graph_evaluate


def test_graph_evaluation_persists_ranking_metrics(tmp_path):
    source = tmp_path / "paysim.csv"
    pd.DataFrame({
        "step": [1, 1, 2, 2, 3, 3],
        "type": ["TRANSFER"] * 6,
        "amount": [10, 10, 10, 10, 10, 10],
        "nameOrig": ["A", "A", "B", "C", "D", "E"],
        "nameDest": ["B", "C", "D", "D", "F", "F"],
        "isFraud": [1, 1, 0, 0, 0, 0],
    }).to_csv(source, index=False)
    report_path = tmp_path / "graph_evaluation.json"

    report = run_graph_evaluation(source, report_path)

    assert report["evaluation_scope"].startswith("internal")
    assert set(report["metrics"]["precision_at_k"]) == {"50", "100", "500"}
    assert "roc_auc" in report["metrics"]
    assert set(report["metrics"]["random_baseline_precision_at_k"]) == {"50", "100", "500"}
    assert json.loads(report_path.read_text()) == report


def test_graph_evaluation_uses_full_file_and_does_not_score_labels(tmp_path, monkeypatch):
    source = tmp_path / "paysim.csv"
    pd.DataFrame({
        "step": [1, 2, 3, 4], "type": ["TRANSFER"] * 4,
        "amount": [10, 20, 30, 40], "nameOrig": ["A", "B", "C", "D"],
        "nameDest": ["B", "C", "D", "E"], "isFraud": [1, 0, 1, 0],
    }).to_csv(source, index=False)
    observed = {}
    original_loader = graph_evaluate.load_paysim

    def loader(path, **kwargs):
        observed["max_rows"] = kwargs.get("max_rows")
        return original_loader(path, **kwargs)

    monkeypatch.setattr(graph_evaluate, "load_paysim", loader)
    seen_fraud_counts = []
    original_score = graph_evaluate.score_accounts

    def score(graph):
        seen_fraud_counts.extend(data.get("fraud_count") for _, _, data in graph.edges(data=True))
        return original_score(graph)

    monkeypatch.setattr(graph_evaluate, "score_accounts", score)
    report = run_graph_evaluation(source, tmp_path / "report.json")
    assert observed["max_rows"] is None
    assert all(value == 0 for value in seen_fraud_counts)
    assert report["sampled"] is False


def test_shuffled_fraud_labels_do_not_improve_graph_score(tmp_path):
    base = pd.DataFrame({
        "step": range(20), "type": ["TRANSFER"] * 20,
        "amount": [10] * 20, "nameOrig": [f"A{i}" for i in range(20)],
        "nameDest": ["Z"] * 20, "isFraud": [1] * 2 + [0] * 18,
    })
    shuffled = base.copy()
    shuffled["isFraud"] = np.random.default_rng(42).permutation(shuffled["isFraud"].to_numpy())
    base_path, shuffled_path = tmp_path / "base.csv", tmp_path / "shuffled.csv"
    base.to_csv(base_path, index=False)
    shuffled.to_csv(shuffled_path, index=False)
    original = run_graph_evaluation(base_path, tmp_path / "base.json")
    randomized = run_graph_evaluation(shuffled_path, tmp_path / "shuffled.json")
    assert randomized["metrics"]["precision_at_k"]["50"] == randomized["metrics"]["random_baseline_precision_at_k"]["50"]
    assert randomized["metrics"]["roc_auc"] <= 0.7
    assert original["metrics"]["precision_at_k"] == randomized["metrics"]["precision_at_k"]
