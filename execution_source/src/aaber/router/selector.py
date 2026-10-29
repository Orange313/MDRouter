from __future__ import annotations

from collections import defaultdict

from aaber.data.schema import ActionCandidate


def greedy_select(
    candidates: list[ActionCandidate],
    budget: float,
    max_per_action: dict[str, int] | None = None,
    min_gain: float = 0.0,
    min_gain_by_action: dict[str, float] | None = None,
    selection_score: str = "gain_per_cost",
) -> list[ActionCandidate]:
    if selection_score not in {"gain_per_cost", "gain"}:
        raise ValueError(f"Unsupported selection_score: {selection_score}")
    max_per_action = max_per_action or {}
    min_gain_by_action = min_gain_by_action or {}
    valid = [
        c
        for c in candidates
        if c.estimated_cost >= 0
        and c.predicted_gain > min_gain_by_action.get(c.action_name, min_gain)
        and c.estimated_cost <= budget
    ]
    if selection_score == "gain":
        valid.sort(key=lambda c: (c.predicted_gain, -c.estimated_cost), reverse=True)
    else:
        valid.sort(
            key=lambda c: (
                c.predicted_gain / max(c.estimated_cost, 1e-9),
                c.predicted_gain,
            ),
            reverse=True,
        )

    selected: list[ActionCandidate] = []
    used = 0.0
    seen: set[tuple[str, str, str]] = set()
    action_counts: dict[str, int] = defaultdict(int)

    for candidate in valid:
        key = (candidate.query_id, candidate.unit_id, candidate.action_name)
        if key in seen:
            continue
        limit = max_per_action.get(candidate.action_name)
        if limit is not None and action_counts[candidate.action_name] >= limit:
            continue
        if used + candidate.estimated_cost > budget:
            continue
        selected.append(candidate)
        seen.add(key)
        action_counts[candidate.action_name] += 1
        used += candidate.estimated_cost
    return selected


def used_budget(candidates: list[ActionCandidate]) -> float:
    return sum(c.estimated_cost for c in candidates)
