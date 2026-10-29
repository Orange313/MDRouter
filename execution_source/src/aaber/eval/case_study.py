from __future__ import annotations


def select_interesting_cases(trace_rows: list[dict], limit: int = 5) -> list[dict]:
    return sorted(
        trace_rows,
        key=lambda row: (len(row.get("selected_actions", [])), row.get("used_budget", 0.0)),
        reverse=True,
    )[:limit]

