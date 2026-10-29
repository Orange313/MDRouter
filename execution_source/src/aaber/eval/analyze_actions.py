from __future__ import annotations

from collections import Counter


def action_distribution(trace_rows: list[dict]) -> dict[str, float]:
    counts: Counter[str] = Counter()
    total = 0
    for row in trace_rows:
        for action in row.get("selected_actions", []):
            counts[action.get("action", "")] += 1
            total += 1
    if total == 0:
        return {}
    return {action: count / total for action, count in sorted(counts.items())}

