from __future__ import annotations

from aaber.data.schema import EvidenceUnit, Query


def build_answer_prompt(query: Query, evidence: list[EvidenceUnit]) -> str:
    chunks = [f"Question: {query.question}", "", "Evidence:"]
    for idx, unit in enumerate(evidence, start=1):
        chunks.append(f"[{idx}] {unit.searchable_text}")
    chunks.append("")
    chunks.append("Answer using only the evidence above.")
    return "\n".join(chunks)

