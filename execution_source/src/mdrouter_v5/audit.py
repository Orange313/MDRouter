from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def unique_rows(rows, key):
    result = {}
    for row in rows:
        if row[key] in result:
            raise ValueError(f"Duplicate {key}: {row[key]}")
        result[row[key]] = row
    return result


def load_rankings(path: Path, expected: set[str], corpus: set[str]):
    grouped = {}
    for row in read_jsonl(path):
        qid = row["query_id"]
        if row["unit_id"] not in corpus:
            raise ValueError(f"Unknown ranked unit: {row['unit_id']}")
        if not math.isfinite(float(row["score"])):
            raise ValueError("Nonfinite score")
        grouped.setdefault(qid, []).append(row)
    if set(grouped) != expected:
        raise ValueError(f"Ranking query coverage differs: missing={expected-set(grouped)}, extra={set(grouped)-expected}")
    for qid, rows in grouped.items():
        rows.sort(key=lambda row: row["rank"])
        if [row["rank"] for row in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"Noncontiguous/duplicate ranks: {qid}")
        if len(set(row["unit_id"] for row in rows)) != len(rows):
            raise ValueError(f"Duplicate ranked units: {qid}")
        if len(rows) < min(10, len(corpus)):
            raise ValueError(f"Insufficient saved Top-10: {qid}")
    return {qid: [row["unit_id"] for row in rows] for qid, rows in grouped.items()}


def discount(rank: int, k=10):
    return 1 / math.log2(rank + 1) if 1 <= rank <= k else 0.0


def ndcg(ranking: list[str], positives: set[str], k=10):
    if not positives:
        raise ValueError("Missing positive qrels cannot be treated as zero utility")
    ideal = sum(discount(j, k) for j in range(1, min(len(positives), k) + 1))
    return sum(discount(j, k) for j, page in enumerate(ranking[:k], 1) if page in positives) / ideal


def displacement(reference: list[str], ranking: list[str], positives: set[str], k=10):
    """Exact B1 for one binary relevant page; B1-L upper bound otherwise."""
    ranks = {page: j for j, page in enumerate(ranking[:k], 1)}
    hits = [(j, page) for j, page in enumerate(reference[:k], 1) if page in positives]
    numerator = sum(max(0.0, discount(j, k) - discount(ranks.get(page, k + 1), k)) for j, page in hits)
    known_ideal = sum(discount(j, k) for j in range(1, max(1, len(hits)) + 1))
    return min(1.0, numerator / known_ideal)


def stable_number(qid, seed):
    return int(hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest()[:16], 16)


def visual_prefix(rows, risk_key, count, seed):
    if not 0 <= count <= len(rows):
        raise ValueError("Invalid quota")
    ordered = sorted(rows, key=lambda row: (-float(row[risk_key]), stable_number(row["query_id"], seed)))
    return {row["query_id"] for row in ordered[:count]}


def epsilon_visual(rows, risk_key, epsilon, seed):
    if epsilon < 0:
        raise ValueError("Negative risk budget")
    ordered = sorted(rows, key=lambda row: (-float(row[risk_key]), stable_number(row["query_id"], seed)))
    total, chosen = sum(max(0.0, float(row[risk_key])) for row in rows), set()
    for row in ordered:
        if total <= epsilon * len(rows) + 1e-12:
            break
        chosen.add(row["query_id"])
        total -= max(0.0, float(row[risk_key]))
    return chosen


def oracle_text(rows, epsilon):
    """Exact only for two plans with identical positive cost saving per query."""
    if epsilon < 0:
        raise ValueError("Negative risk budget")
    budget, total, chosen = epsilon * len(rows), 0.0, set()
    for row in sorted(rows, key=lambda row: (float(row["r"]), row["query_id"])):
        risk = float(row["r"])
        if not 0 <= risk <= 1:
            raise ValueError("Unbounded/invalid positive regret")
        if total + risk <= budget + 1e-12:
            chosen.add(row["query_id"])
            total += risk
        else:
            break
    return chosen


def assignment_hash(dataset, qids, visual):
    return hashlib.sha256(json.dumps([[dataset, qid, "V-Multi" if qid in visual else "T-BM25"] for qid in sorted(qids)], separators=(",", ":")).encode()).hexdigest()
