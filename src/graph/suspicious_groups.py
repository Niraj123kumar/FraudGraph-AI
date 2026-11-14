"""Pure graph-based account suspicion and connected-group scoring."""

from __future__ import annotations

from typing import Any

import networkx as nx

from src.graph.graph_features import graph_node_metrics


def _account_detail(account: Any, metrics: dict[str, Any]) -> dict[str, Any]:
    out_degree = metrics["out_degree"]
    transaction_count = metrics["transaction_count"]
    amount = metrics["total_amount"]
    reasons = []
    if out_degree >= 3:
        reasons.append(f"High outgoing fan-out: {out_degree} distinct accounts")
    if transaction_count >= 5:
        reasons.append(f"High transfer activity: {transaction_count} sampled transfers")
    if amount >= 1_000_000:
        reasons.append(f"High aggregate flow: {amount:,.2f} dataset units")
    first_step, last_step = metrics["first_step"], metrics["last_step"]
    same_step = transaction_count >= 3 and first_step is not None and last_step - first_step <= 1
    if same_step:
        reasons.append("Multiple transfers concentrated within one PaySim time step")
    components = {
        "fan_out": min(out_degree / 10, 1.0) * 0.4,
        "activity": min(transaction_count / 20, 1.0) * 0.3,
        "flow_amount": min(amount / 3_000_000, 1.0) * 0.3,
        "same_step_concentration": 1.0 if same_step else 0.0,
    }
    return {
        "id": str(account),
        "degree": metrics["in_degree"] + out_degree,
        "in_degree": metrics["in_degree"],
        "out_degree": out_degree,
        "total_amount": amount,
        "transaction_count": transaction_count,
        "fraud_count": metrics["fraud_count"],
        "risk_score": min(
            1.0,
            components["fan_out"]
            + components["activity"]
            + components["flow_amount"],
        ),
        "score_components": components,
        "reasons": reasons,
    }


def score_accounts(graph: nx.DiGraph) -> list[dict[str, Any]]:
    return [_account_detail(account, metrics) for account, metrics in graph_node_metrics(graph).items()]


def suspicious_accounts(graph: nx.DiGraph) -> list[dict[str, Any]]:
    return sorted(
        [account for account in score_accounts(graph) if account["reasons"]],
        key=lambda item: (item["risk_score"], item["transaction_count"]),
        reverse=True,
    )


def connected_component_groups(graph: nx.DiGraph) -> list[dict[str, Any]]:
    """Aggregate scores for weakly connected components."""
    scores = {item["id"]: item for item in score_accounts(graph)}
    groups = []
    for index, component in enumerate(nx.weakly_connected_components(graph), start=1):
        members = [scores[str(account)] for account in component]
        groups.append({
            "id": f"GROUP-{index}",
            "accounts": sorted(str(account) for account in component),
            "account_count": len(members),
            "total_amount": sum(item["total_amount"] for item in members),
            "transaction_count": sum(item["transaction_count"] for item in members),
            "risk_score": min(1.0, sum(item["risk_score"] for item in members)),
        })
    return sorted(groups, key=lambda group: group["risk_score"], reverse=True)