"""Reusable account and edge metrics for transaction graphs."""

from __future__ import annotations

from typing import Any

import networkx as nx


def edge_transaction_count(edge: dict[str, Any]) -> int:
    return int(edge.get("tx_count", 0))


def edge_total_amount(edge: dict[str, Any]) -> float:
    return float(edge.get("total_amount", 0.0))


def node_metrics(graph: nx.DiGraph, account: Any) -> dict[str, Any]:
    """Return structural and flow metrics for one graph account."""
    incident = list(graph.in_edges(account, data=True)) + list(graph.out_edges(account, data=True))
    first_steps = [int(edge["first_step"]) for _, _, edge in incident if edge.get("first_step") is not None]
    last_steps = [int(edge["last_step"]) for _, _, edge in incident if edge.get("last_step") is not None]
    return {
        "in_degree": graph.in_degree(account),
        "out_degree": graph.out_degree(account),
        "transaction_count": sum(edge_transaction_count(edge) for _, _, edge in incident),
        "total_amount": sum(edge_total_amount(edge) for _, _, edge in incident),
        "fraud_count": sum(int(edge.get("fraud_count", 0)) for _, _, edge in incident),
        "first_step": min(first_steps) if first_steps else None,
        "last_step": max(last_steps) if last_steps else None,
    }


def graph_node_metrics(graph: nx.DiGraph) -> dict[Any, dict[str, Any]]:
    return {account: node_metrics(graph, account) for account in graph.nodes}