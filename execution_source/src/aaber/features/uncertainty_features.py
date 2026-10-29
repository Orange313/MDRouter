from __future__ import annotations

from aaber.common.text import score_entropy
from aaber.data.schema import EvidenceResult


def build_uncertainty_features(
    query_id: str,
    unit_id: str,
    score_index: dict[tuple[str, str], list[EvidenceResult]],
) -> dict[str, float]:
    query_results = [
        result
        for (qid, _), results in score_index.items()
        if qid == query_id
        for result in results
    ]
    scores = sorted((r.score for r in query_results), reverse=True)
    margin = scores[0] - scores[1] if len(scores) >= 2 else (scores[0] if scores else 0.0)
    unit_scores = [r.score for r in score_index.get((query_id, unit_id), [])]
    if unit_scores:
        avg = sum(unit_scores) / len(unit_scores)
        variance = sum((s - avg) ** 2 for s in unit_scores) / len(unit_scores)
    else:
        variance = 0.0
    return {
        "top1_top2_margin": float(margin),
        "retrieval_score_variance": float(variance),
        "score_entropy": float(score_entropy(scores[:10])),
    }

