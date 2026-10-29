from __future__ import annotations

import hashlib
import itertools
import math
import re
from collections import defaultdict
from urllib.parse import urlparse

from aaber.data.schema import EvidenceUnit, Relation

REFERENCE_RE = re.compile(r"\b(fig(?:ure)?|table|chart|图|表)\s*\.?\s*\d*", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}|\d+(?:\.\d+)?%?")

STOPWORDS = {
    "about",
    "above",
    "after",
    "also",
    "and",
    "are",
    "because",
    "been",
    "being",
    "between",
    "can",
    "com",
    "could",
    "data",
    "does",
    "from",
    "had",
    "has",
    "have",
    "here",
    "http",
    "https",
    "infographic",
    "into",
    "its",
    "more",
    "most",
    "not",
    "one",
    "only",
    "our",
    "out",
    "page",
    "per",
    "source",
    "than",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "through",
    "use",
    "used",
    "using",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "www",
    "you",
    "your",
}


def build_adjacent_page_relations(units: list[EvidenceUnit]) -> list[Relation]:
    by_doc: dict[str, list[EvidenceUnit]] = defaultdict(list)
    for unit in units:
        if unit.unit_type == "page":
            by_doc[unit.doc_id].append(unit)

    relations: list[Relation] = []
    for doc_id, doc_units in by_doc.items():
        ordered = sorted(doc_units, key=lambda u: u.page_id)
        for left, right in zip(ordered, ordered[1:]):
            relation_id = f"adjacent_{left.unit_id}_{right.unit_id}"
            relations.append(
                Relation(
                    relation_id=relation_id,
                    src_unit_id=left.unit_id,
                    dst_unit_id=right.unit_id,
                    relation_type="adjacent_page",
                    score=0.5,
                    evidence=f"{doc_id} adjacent pages {left.page_id}->{right.page_id}",
                )
            )
    return relations


def build_reference_relations(units: list[EvidenceUnit]) -> list[Relation]:
    relations: list[Relation] = []
    caption_units = [u for u in units if u.caption]
    for src in units:
        if not REFERENCE_RE.search(src.searchable_text):
            continue
        for dst in caption_units:
            if src.doc_id != dst.doc_id or src.unit_id == dst.unit_id:
                continue
            relation_id = f"ref_{src.unit_id}_{dst.unit_id}"
            relations.append(
                Relation(
                    relation_id=relation_id,
                    src_unit_id=src.unit_id,
                    dst_unit_id=dst.unit_id,
                    relation_type="reference_link",
                    score=0.7,
                    evidence="reference keyword and caption in same document",
                )
            )
    return relations


def build_source_group_relations(units: list[EvidenceUnit], max_group_size: int = 8) -> list[Relation]:
    groups: dict[str, list[EvidenceUnit]] = defaultdict(list)
    for unit in units:
        image_filename = str(unit.metadata.get("image_filename") or "")
        for key in _source_group_keys(image_filename):
            groups[key].append(unit)

    relations: list[Relation] = []
    for key, group in sorted(groups.items()):
        unique = sorted({unit.unit_id: unit for unit in group}.values(), key=lambda unit: unit.unit_id)
        if not 2 <= len(unique) <= max_group_size:
            continue
        for left, right in itertools.combinations(unique, 2):
            relation_id = f"source_{_short_id(left.unit_id)}_{_short_id(right.unit_id)}"
            relations.append(
                Relation(
                    relation_id=relation_id,
                    src_unit_id=left.unit_id,
                    dst_unit_id=right.unit_id,
                    relation_type="same_source",
                    score=0.35,
                    evidence=f"shared source group: {key}",
                    metadata={"group": key},
                )
            )
    return relations


def build_text_similarity_relations(
    units: list[EvidenceUnit],
    max_neighbors: int = 2,
    min_score: float = 0.12,
    max_posting_size: int = 40,
) -> list[Relation]:
    if len(units) < 2:
        return []

    token_sets = {unit.unit_id: _tokenize_for_topic(unit) for unit in units}
    postings: dict[str, list[str]] = defaultdict(list)
    for unit_id, tokens in token_sets.items():
        for token in tokens:
            postings[token].append(unit_id)

    unit_count = len(units)
    max_df = max(2, min(max_posting_size, int(unit_count * 0.12)))
    allowed_terms = {
        token: ids
        for token, ids in postings.items()
        if 2 <= len(ids) <= max_df
    }
    if not allowed_terms:
        return []

    weights = {
        token: math.log((unit_count + 1.0) / (len(ids) + 0.5))
        for token, ids in allowed_terms.items()
    }
    norms: dict[str, float] = {}
    for unit_id, tokens in token_sets.items():
        norm = math.sqrt(sum(weights[token] ** 2 for token in tokens if token in weights))
        norms[unit_id] = norm

    pair_scores: dict[tuple[str, str], float] = defaultdict(float)
    for token, unit_ids in allowed_terms.items():
        weight = weights[token]
        for left, right in itertools.combinations(sorted(unit_ids), 2):
            pair_scores[(left, right)] += weight * weight

    neighbors: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for (left, right), dot in pair_scores.items():
        norm = norms.get(left, 0.0) * norms.get(right, 0.0)
        if norm <= 0.0:
            continue
        similarity = dot / norm
        if similarity < min_score:
            continue
        neighbors[left].append((similarity, right))
        neighbors[right].append((similarity, left))

    selected_pairs: dict[tuple[str, str], float] = {}
    for unit_id, items in neighbors.items():
        for similarity, neighbor_id in sorted(items, reverse=True)[:max_neighbors]:
            key = tuple(sorted((unit_id, neighbor_id)))
            selected_pairs[key] = max(selected_pairs.get(key, 0.0), similarity)

    relations: list[Relation] = []
    unit_by_id = {unit.unit_id: unit for unit in units}
    for (left_id, right_id), similarity in sorted(selected_pairs.items()):
        left = unit_by_id[left_id]
        right = unit_by_id[right_id]
        relation_id = f"topic_{_short_id(left_id)}_{_short_id(right_id)}"
        shared = sorted((token_sets[left_id] & token_sets[right_id]) & set(allowed_terms))[:8]
        relations.append(
            Relation(
                relation_id=relation_id,
                src_unit_id=left.unit_id,
                dst_unit_id=right.unit_id,
                relation_type="same_topic",
                score=min(0.9, 0.45 + similarity),
                evidence="weighted OCR token overlap",
                metadata={"similarity": round(similarity, 4), "shared_terms": shared},
            )
        )
    return relations


def build_default_relations(units: list[EvidenceUnit]) -> list[Relation]:
    seen: set[tuple[str, str, str]] = set()
    result: list[Relation] = []
    for relation in [
        *build_adjacent_page_relations(units),
        *build_reference_relations(units),
        *build_source_group_relations(units),
        *build_text_similarity_relations(units),
    ]:
        key = (relation.src_unit_id, relation.dst_unit_id, relation.relation_type)
        if key in seen:
            continue
        seen.add(key)
        result.append(relation)
    return result


def _source_group_keys(image_filename: str) -> list[str]:
    if not image_filename:
        return []
    parsed = urlparse(image_filename)
    keys: list[str] = []
    if parsed.netloc:
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        keys.append(f"host:{host}")
        parts = [part.lower() for part in parsed.path.split("/") if part]
        if len(parts) >= 2:
            keys.append(f"path:{host}/{'/'.join(parts[:2])}")
        elif parts:
            keys.append(f"path:{host}/{parts[0]}")
        return keys

    path = image_filename.replace("\\", "/")
    pieces = [piece.lower() for piece in path.split("/") if piece]
    if len(pieces) >= 2:
        keys.append(f"path:{'/'.join(pieces[:-1])}")
    stem = pieces[-1] if pieces else image_filename.lower()
    prefix = re.split(r"[_\-.]", stem)[0]
    if len(prefix) >= 5:
        keys.append(f"prefix:{prefix}")
    return keys


def _tokenize_for_topic(unit: EvidenceUnit) -> set[str]:
    text = " ".join(part for part in [unit.text, unit.caption] if part)
    tokens = set()
    for match in TOKEN_RE.finditer(text.lower()):
        token = match.group(0).strip("_-")
        if len(token) < 3 or token in STOPWORDS:
            continue
        if token.isdigit() and len(token) <= 2:
            continue
        tokens.add(token)
    return tokens


def _short_id(value: str, prefix_len: int = 24) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    if len(safe) <= prefix_len + 13:
        return safe
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()[:10]
    return f"{safe[:prefix_len]}_{digest}"
