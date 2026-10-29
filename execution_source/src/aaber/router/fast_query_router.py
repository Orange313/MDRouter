from __future__ import annotations

"""Low-overhead query-only features and prediction for Text--Visual routing.

The online path deliberately avoids query encoders, retrieval probes, candidate
generation, and model-runtime thread pools.  It is limited to tokenization,
fixed-dimensional lexical features, a NumPy ``float32`` ridge prediction, and
batch top-K selection.
"""

import hashlib
import math
import re
from collections import Counter
from typing import Any, Callable, Iterable

import numpy as np

from aaber.data.schema import EvidenceUnit, Query


TOKEN_RE = re.compile(r"[a-z0-9]+")
VISUAL_TERMS = frozenset(
    {
        "above",
        "below",
        "chart",
        "color",
        "column",
        "diagram",
        "figure",
        "graph",
        "image",
        "layout",
        "left",
        "picture",
        "right",
        "row",
        "table",
        "visual",
    }
)
COUNT_TERMS = frozenset(
    {"average", "count", "how", "many", "maximum", "minimum", "number", "percent", "percentage", "sum", "total"}
)
QUESTION_STARTS = ("how", "what", "which", "where", "when", "who", "why")


class FastLexicalFeatureEncoder:
    """Frozen, query-only lexical feature map with corpus-derived IDF values."""

    def __init__(self, word_hash_dim: int = 96, char_hash_dim: int = 32) -> None:
        if word_hash_dim < 1 or char_hash_dim < 1:
            raise ValueError("Hash dimensions must be positive.")
        self.word_hash_dim = word_hash_dim
        self.char_hash_dim = char_hash_dim
        self.idf: dict[str, float] = {}
        self.idf_default = 0.0
        self.feature_names = [
            "query_length_chars",
            "query_length_tokens",
            "avg_token_length",
            "digit_ratio",
            "punctuation_ratio",
            "has_question_mark",
            "has_visual_term",
            "visual_term_count",
            "has_count_term",
            "count_term_count",
            "idf_mean",
            "idf_max",
            "idf_min",
            "idf_known_fraction",
            "idf_high_fraction",
            *[f"starts_{word}" for word in QUESTION_STARTS],
            *[f"word_hash_{index:03d}" for index in range(word_hash_dim)],
            *[f"char_hash_{index:03d}" for index in range(char_hash_dim)],
        ]

    def fit_corpus(self, units: Iterable[EvidenceUnit]) -> None:
        """Precompute IDF from the frozen retrieval corpus, without qrels."""
        document_frequency: Counter[str] = Counter()
        document_count = 0
        for unit in units:
            tokens = set(tokenize(unit.searchable_text))
            document_frequency.update(tokens)
            document_count += 1
        if document_count == 0:
            raise ValueError("Cannot fit lexical IDF statistics on an empty corpus.")
        self.idf = {
            token: math.log((document_count + 1.0) / (count + 1.0)) + 1.0
            for token, count in document_frequency.items()
        }
        self.idf_default = math.log(document_count + 1.0) + 1.0

    def transform(self, query: Query) -> dict[str, float]:
        text = query.question.lower().strip()
        tokens = tokenize(text)
        token_set = set(tokens)
        char_count = max(1, len(text))
        numeric_chars = sum(char.isdigit() for char in text)
        punctuation = sum(not char.isalnum() and not char.isspace() for char in text)
        known_idfs = [self.idf[token] for token in tokens if token in self.idf]
        all_idfs = [self.idf.get(token, self.idf_default) for token in tokens]
        features = {name: 0.0 for name in self.feature_names}
        features.update(
            {
                "query_length_chars": float(len(text)),
                "query_length_tokens": float(len(tokens)),
                "avg_token_length": sum(len(token) for token in tokens) / max(1, len(tokens)),
                "digit_ratio": numeric_chars / char_count,
                "punctuation_ratio": punctuation / char_count,
                "has_question_mark": float("?" in text),
                "has_visual_term": float(bool(token_set & VISUAL_TERMS)),
                "visual_term_count": float(sum(token in VISUAL_TERMS for token in tokens)),
                "has_count_term": float(bool(token_set & COUNT_TERMS)),
                "count_term_count": float(sum(token in COUNT_TERMS for token in tokens)),
                "idf_mean": sum(all_idfs) / max(1, len(all_idfs)),
                "idf_max": max(all_idfs, default=0.0),
                "idf_min": min(all_idfs, default=0.0),
                "idf_known_fraction": len(known_idfs) / max(1, len(tokens)),
                "idf_high_fraction": sum(value >= 2.5 for value in all_idfs) / max(1, len(tokens)),
            }
        )
        for word in QUESTION_STARTS:
            features[f"starts_{word}"] = float(tokens[:1] == [word])
        self._add_hashed_tokens(features, tokens)
        self._add_hashed_char_ngrams(features, text)
        return features

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_family": "surface + corpus_idf + fixed_hash_ngrams",
            "word_hash_dim": self.word_hash_dim,
            "char_hash_dim": self.char_hash_dim,
            "feature_count": len(self.feature_names),
            "idf_terms": len(self.idf),
            "online_inputs": "query text and frozen corpus IDF map only",
            "excluded": "query encoder, retrieval probe, retrieval candidates, scores, OCR snippets, Visual output, qrels",
        }

    def _add_hashed_tokens(self, features: dict[str, float], tokens: list[str]) -> None:
        scale = 1.0 / math.sqrt(max(1, len(tokens)))
        for token in tokens:
            bucket, sign = stable_hash_bucket(f"word:{token}", self.word_hash_dim)
            features[f"word_hash_{bucket:03d}"] += sign * scale

    def _add_hashed_char_ngrams(self, features: dict[str, float], text: str) -> None:
        compact = "".join(tokenize(text))
        ngrams = [compact[index : index + 3] for index in range(max(0, len(compact) - 2))]
        scale = 1.0 / math.sqrt(max(1, len(ngrams)))
        for ngram in ngrams:
            bucket, sign = stable_hash_bucket(f"char3:{ngram}", self.char_hash_dim)
            features[f"char_hash_{bucket:03d}"] += sign * scale


class NumpyRidgeRiskRegressor:
    """A small ridge regressor with explicitly supplied, label-free columns."""

    def __init__(self, feature_names: list[str], l2: float = 8.0) -> None:
        if not feature_names:
            raise ValueError("NumpyRidgeRiskRegressor requires at least one feature.")
        self.feature_names = list(feature_names)
        self.l2 = float(l2)
        self.means = np.zeros(len(self.feature_names), dtype=np.float32)
        self.stds = np.ones(len(self.feature_names), dtype=np.float32)
        self.weights = np.zeros(len(self.feature_names), dtype=np.float32)
        self.bias = 0.0

    def fit(self, rows: list[dict[str, Any]], target: str) -> None:
        if not rows:
            raise ValueError("Cannot fit a risk regressor without training rows.")
        matrix = self._matrix(rows)
        values = np.asarray([float(row[target]) for row in rows], dtype=np.float32)
        self.means = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.stds = matrix.std(axis=0, dtype=np.float64).astype(np.float32)
        self.stds[self.stds < 1e-6] = 1.0
        normalized = (matrix - self.means) / self.stds
        self.bias = float(values.mean(dtype=np.float64))
        centered = values - self.bias
        identity = np.eye(normalized.shape[1], dtype=np.float32)
        gram = normalized.T @ normalized + self.l2 * identity
        rhs = normalized.T @ centered
        try:
            self.weights = np.linalg.solve(gram, rhs).astype(np.float32)
        except np.linalg.LinAlgError:
            self.weights = np.linalg.lstsq(gram, rhs, rcond=None)[0].astype(np.float32)

    def predict(self, row: dict[str, Any]) -> float:
        return float(self.predict_many([row])[0])

    def predict_many(self, rows: list[dict[str, Any]]) -> np.ndarray:
        if not rows:
            return np.asarray([], dtype=np.float32)
        normalized = (self._matrix(rows) - self.means) / self.stds
        return (normalized @ self.weights + self.bias).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": "numpy_float32_ridge",
            "feature_count": len(self.feature_names),
            "l2": self.l2,
            "feature_names": self.feature_names,
            "bias": self.bias,
            "online_compute": "one float32 matrix-vector product per query batch; no encoder or retrieval probe",
        }

    def _matrix(self, rows: list[dict[str, Any]]) -> np.ndarray:
        return np.asarray(
            [[float(row.get(name, 0.0) or 0.0) for name in self.feature_names] for row in rows],
            dtype=np.float32,
        )


def select_top_k_argpartition(
    rows: list[dict[str, Any]],
    score_key: str,
    count: int,
    tie_break: Callable[[dict[str, Any]], int],
) -> set[str]:
    """Exact deterministic top-K selection without a full sort of all queries."""
    count = max(0, min(count, len(rows)))
    if count == 0:
        return set()
    if count == len(rows):
        return {str(row["query_id"]) for row in rows}
    scores = np.asarray([float(row[score_key]) for row in rows], dtype=np.float32)
    provisional = np.argpartition(-scores, count - 1)[:count]
    cutoff = float(scores[provisional].min())
    strictly_above = np.flatnonzero(scores > cutoff).tolist()
    tied = np.flatnonzero(scores == cutoff).tolist()
    needed = count - len(strictly_above)
    chosen = strictly_above + sorted(tied, key=lambda index: tie_break(rows[index]))[:needed]
    return {str(rows[index]["query_id"]) for index in chosen}


def stable_hash_bucket(value: str, dimension: int) -> tuple[int, float]:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    number = int.from_bytes(digest, byteorder="big", signed=False)
    return number % dimension, 1.0 if number & 1 else -1.0


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())
