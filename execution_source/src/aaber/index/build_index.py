from __future__ import annotations

from pathlib import Path

from aaber.index.base import RetrieverIndex
from aaber.index.bm25_index import BM25Index
from aaber.index.cross_encoder_rerank_index import CrossEncoderRerankIndex
from aaber.index.dense_text_index import DenseTextIndex, SentenceTransformerTextIndex
from aaber.index.image_index import ImageTextIndex, TransformersImageTextIndex
from aaber.index.multivector_index import ColPaliEngineIndex, MultiVectorRerankIndex


INDEX_TYPES: dict[str, type[RetrieverIndex]] = {
    "bm25": BM25Index,
    "dense_text": DenseTextIndex,
    "sentence_transformer": SentenceTransformerTextIndex,
    "st_dense": SentenceTransformerTextIndex,
    "clip_image": ImageTextIndex,
    "image": ImageTextIndex,
    "clip_transformers": TransformersImageTextIndex,
    "siglip_image": TransformersImageTextIndex,
    "cross_encoder": CrossEncoderRerankIndex,
    "cross-rerank": CrossEncoderRerankIndex,
    "cross_rerank": CrossEncoderRerankIndex,
    "colpali": MultiVectorRerankIndex,
    "colqwen": MultiVectorRerankIndex,
    "colpali_engine": ColPaliEngineIndex,
}


def create_index(name: str, **kwargs) -> RetrieverIndex:
    key = name.lower()
    if key not in INDEX_TYPES:
        raise KeyError(f"Unknown index '{name}'. Available: {', '.join(sorted(INDEX_TYPES))}")
    if kwargs:
        try:
            return INDEX_TYPES[key](**kwargs)
        except TypeError:
            return INDEX_TYPES[key]()
    return INDEX_TYPES[key]()


def load_index(name: str, path: str | Path) -> RetrieverIndex:
    key = name.lower()
    if key not in INDEX_TYPES:
        raise KeyError(f"Unknown index '{name}'. Available: {', '.join(sorted(INDEX_TYPES))}")
    return INDEX_TYPES[key].load(path)
