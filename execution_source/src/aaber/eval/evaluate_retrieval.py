from __future__ import annotations

from aaber.data.schema import EvidenceResult
from aaber.eval.metrics import evaluate_rankings
from aaber.router.fusion import rankings_from_results


def evaluate_results(results: list[EvidenceResult], qrels: dict[str, set[str]]) -> dict[str, float]:
    return evaluate_rankings(rankings_from_results(results), qrels)

