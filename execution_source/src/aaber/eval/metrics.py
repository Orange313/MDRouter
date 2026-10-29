from __future__ import annotations

import math
from statistics import mean
from typing import Iterable


def _top_k(ranked_ids: Iterable[str], k: int) -> list[str]:
    return list(ranked_ids)[:k]


def recall_at_k(ranked_ids: Iterable[str], positive_ids: set[str], k: int) -> float:
    if not positive_ids:
        return 0.0
    top = set(_top_k(ranked_ids, k))
    return 1.0 if top & positive_ids else 0.0


def evidence_recall_at_k(ranked_ids: Iterable[str], positive_ids: set[str], k: int) -> float:
    if not positive_ids:
        return 0.0
    top = set(_top_k(ranked_ids, k))
    return len(top & positive_ids) / len(positive_ids)


def mrr_at_k(ranked_ids: Iterable[str], positive_ids: set[str], k: int) -> float:
    if not positive_ids:
        return 0.0
    for idx, unit_id in enumerate(_top_k(ranked_ids, k), start=1):
        if unit_id in positive_ids:
            return 1.0 / idx
    return 0.0


def ndcg_at_k(ranked_ids: Iterable[str], positive_ids: set[str], k: int) -> float:
    if not positive_ids:
        return 0.0
    dcg = 0.0
    for idx, unit_id in enumerate(_top_k(ranked_ids, k), start=1):
        rel = 1.0 if unit_id in positive_ids else 0.0
        dcg += rel / math.log2(idx + 1)
    ideal_hits = min(len(positive_ids), k)
    idcg = sum(1.0 / math.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_rankings(
    rankings: dict[str, list[str]],
    qrels: dict[str, set[str]],
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    metrics: dict[str, list[float]] = {}
    for query_id, positives in qrels.items():
        ranked = rankings.get(query_id, [])
        for k in ks:
            metrics.setdefault(f"recall@{k}", []).append(recall_at_k(ranked, positives, k))
            metrics.setdefault(f"evidence_recall@{k}", []).append(evidence_recall_at_k(ranked, positives, k))
        metrics.setdefault("mrr@10", []).append(mrr_at_k(ranked, positives, 10))
        metrics.setdefault("ndcg@10", []).append(ndcg_at_k(ranked, positives, 10))
    return {name: mean(values) if values else 0.0 for name, values in metrics.items()}


def latency_stats(latencies_ms: Iterable[float]) -> dict[str, float]:
    values = sorted(float(x) for x in latencies_ms)
    if not values:
        return {"avg_latency_ms": 0.0, "p95_latency_ms": 0.0}
    p95_idx = min(len(values) - 1, math.ceil(0.95 * len(values)) - 1)
    return {"avg_latency_ms": mean(values), "p95_latency_ms": values[p95_idx]}


def cost_normalized_utility(metric: float, cost: float) -> float:
    return metric / max(cost, 1e-9)

