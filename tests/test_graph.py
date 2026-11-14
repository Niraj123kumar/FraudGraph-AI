"""Tests for the src/graph package."""

import pandas as pd
import pytest
import networkx as nx

from src.graph.build_graph import aggregate_transfers, build_graph, load_paysim
from src.graph.suspicious_groups import connected_component_groups, score_accounts


def _tiny_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "step": [1, 2, 3],
            "type": ["TRANSFER"] * 3,
            "amount": [100.0, 50.0, 10.0],
            "nameOrig": ["A", "A", "B"],
            "nameDest": ["B", "B", "C"],
            "isFraud": [0, 1, 0],
        }
    )


def test_repeated_transfers_are_aggregated() -> None:
    edges = aggregate_transfers(_tiny_df())
    ab = edges[(edges["source"] == "A") & (edges["target"] == "B")].iloc[0]
    assert len(edges) == 2
    assert ab["total_amount"] == 150.0
    assert ab["tx_count"] == 2
    assert (ab["first_step"], ab["last_step"]) == (1, 2)
    assert ab["fraud_count"] == 1


def test_graph_structure_and_fraud_flags() -> None:
    graph = build_graph(aggregate_transfers(_tiny_df()))
    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
    assert graph.has_edge("A", "B") and not graph.has_edge("B", "A")
    assert graph.nodes["A"]["is_fraud_account"] and graph.nodes["B"]["is_fraud_account"]
    assert not graph.nodes["C"]["is_fraud_account"]


def test_bounded_paysim_loader_filters_and_samples_deterministically(tmp_path) -> None:
    path = tmp_path / "paysim.csv"
    _tiny_df().assign(type=["TRANSFER", "PAYMENT", "CASH_OUT"]).to_csv(path, index=False)

    first = load_paysim(path, max_rows=1, seed=7)
    second = load_paysim(path, max_rows=1, seed=7)

    assert len(first) == 1
    assert first.iloc[0]["type"] in {"TRANSFER", "CASH_OUT"}
    assert first.equals(second)
    assert first.attrs["sampled"] is True
    assert first.attrs["eligible_rows_seen"] == 2
    assert len(first) / first.attrs["eligible_rows_seen"] == pytest.approx(0.5)


@pytest.mark.parametrize("graph", [nx.DiGraph(), nx.DiGraph([( "A", "A")])])
def test_graph_scoring_handles_empty_and_self_loop(graph) -> None:
    if graph.number_of_edges():
        graph["A"]["A"].update(tx_count=1, total_amount=10.0, first_step=1, last_step=1)
    scores = score_accounts(graph)
    assert len(scores) == graph.number_of_nodes()


def test_graph_scoring_handles_single_node() -> None:
    scores = score_accounts(nx.DiGraph([("A", "A")]))
    assert scores[0]["id"] == "A"
    assert scores[0]["out_degree"] == 1


def test_graph_scoring_flags_star_fan_out_and_burst() -> None:
    graph = nx.DiGraph()
    for index, target in enumerate(["B", "C", "D"], start=1):
        graph.add_edge("A", target, tx_count=1, total_amount=10.0, first_step=1, last_step=1)
    graph.add_edge("A", "E", tx_count=2, total_amount=10.0, first_step=1, last_step=1)
    account = next(item for item in score_accounts(graph) if item["id"] == "A")
    assert account["out_degree"] == 4
    assert account["score_components"]["fan_out"] > 0
    assert account["score_components"]["same_step_concentration"] == 1.0
    assert any("fan-out" in reason for reason in account["reasons"])


def test_graph_scoring_preserves_legacy_risk_score() -> None:
    graph = nx.DiGraph()
    graph.add_edge(
        "A",
        "B",
        tx_count=4,
        total_amount=1_500_000.0,
        first_step=10,
        last_step=10,
    )

    account = next(item for item in score_accounts(graph) if item["id"] == "A")

    assert account["score_components"]["same_step_concentration"] == 1.0
    assert account["risk_score"] == pytest.approx(0.25)


def test_connected_components_return_aggregate_groups() -> None:
    graph = nx.DiGraph([("A", "B"), ("C", "D")])
    for source, target in graph.edges:
        graph[source][target].update(tx_count=1, total_amount=10.0, first_step=1, last_step=1)
    groups = connected_component_groups(graph)
    assert len(groups) == 2
    assert {tuple(group["accounts"]) for group in groups} == {("A", "B"), ("C", "D")}


def test_paysim_loader_rejects_empty_missing_and_invalid_accounts(tmp_path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_paysim(empty)

    missing = tmp_path / "missing.csv"
    missing.write_text("step,type\n1,TRANSFER\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_paysim(missing)

    invalid = tmp_path / "invalid.csv"
    pd.DataFrame({
        "step": [1], "type": ["TRANSFER"], "amount": [1.0],
        "nameOrig": [""], "nameDest": ["B"], "isFraud": [0],
    }).to_csv(invalid, index=False)
    with pytest.raises(ValueError, match="account IDs"):
        load_paysim(invalid)


def test_paysim_sampling_never_exceeds_limit(tmp_path) -> None:
    path = tmp_path / "paysim.csv"
    pd.DataFrame({
        "step": range(20), "type": ["TRANSFER"] * 20, "amount": [1.0] * 20,
        "nameOrig": [f"A{i}" for i in range(20)], "nameDest": ["B"] * 20,
        "isFraud": [0] * 20,
    }).to_csv(path, index=False)
    sampled = load_paysim(path, max_rows=5)
    assert len(sampled) == 5
