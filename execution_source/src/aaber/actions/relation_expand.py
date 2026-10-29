from __future__ import annotations

from collections import defaultdict

from aaber.actions.base import RetrievalAction
from aaber.data.schema import EvidenceResult, EvidenceUnit, Query, Relation


class RelationExpandAction(RetrievalAction):
    name = "relation-expand"
    # One seed expansion measured 0.021 ms versus 14.363 ms for BM25.
    default_cost = 0.001

    def __init__(self, relations: list[Relation], units: list[EvidenceUnit]) -> None:
        self.units_by_id = {unit.unit_id: unit for unit in units}
        self.graph: dict[str, list[Relation]] = defaultdict(list)
        for relation in relations:
            self.graph[relation.src_unit_id].append(relation)
            self.graph[relation.dst_unit_id].append(
                Relation(
                    relation_id=relation.relation_id + "_rev",
                    src_unit_id=relation.dst_unit_id,
                    dst_unit_id=relation.src_unit_id,
                    relation_type=relation.relation_type,
                    score=relation.score,
                    evidence=relation.evidence,
                )
            )

    def execute(self, query: Query, seed_units: list[EvidenceUnit], top_k: int = 10) -> list[EvidenceResult]:
        scored: dict[str, float] = {}
        for seed in seed_units:
            for relation in self.graph.get(seed.unit_id, []):
                if relation.dst_unit_id not in self.units_by_id:
                    continue
                scored[relation.dst_unit_id] = max(scored.get(relation.dst_unit_id, 0.0), relation.score)
        ordered = sorted(scored.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            EvidenceResult(query.query_id, unit_id, score, self.name, rank=idx)
            for idx, (unit_id, score) in enumerate(ordered, start=1)
        ]
