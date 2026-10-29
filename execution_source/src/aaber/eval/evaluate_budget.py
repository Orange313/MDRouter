from __future__ import annotations


def pareto_points(records: list[dict]) -> list[dict]:
    ordered = sorted(records, key=lambda r: (r.get("avg_latency_ms", 0.0), -r.get("recall@10", 0.0)))
    frontier: list[dict] = []
    best_recall = -1.0
    for record in ordered:
        recall = float(record.get("recall@10", 0.0))
        if recall > best_recall:
            frontier.append(record)
            best_recall = recall
    return frontier

