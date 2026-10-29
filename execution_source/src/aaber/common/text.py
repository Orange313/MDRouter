from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def tokenize(text: str | None) -> list[str]:
    if not text:
        return []
    return [token.lower() for token in TOKEN_RE.findall(text)]


def token_counter(text: str | None) -> Counter[str]:
    return Counter(tokenize(text))


def keyword_overlap(a: str | None, b: str | None) -> float:
    left = set(tokenize(a))
    right = set(tokenize(b))
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left))


def cosine_counter(a: Counter[str], b: Counter[str]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def score_entropy(scores: Iterable[float]) -> float:
    vals = [max(0.0, s) for s in scores]
    total = sum(vals)
    if total <= 0:
        return 0.0
    probs = [v / total for v in vals if v > 0]
    return -sum(p * math.log(p + 1e-12) for p in probs)


def has_any(text: str | None, keywords: Iterable[str]) -> int:
    lower = (text or "").lower()
    return int(any(k.lower() in lower for k in keywords))

