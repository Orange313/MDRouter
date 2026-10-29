from __future__ import annotations

from pathlib import Path
from typing import Any

from aaber.data.dataset_base import FileDatasetAdapter
from aaber.data.irpapers_csv import IRPapersCSVAdapter
from aaber.data.vidore_parquet import VidoreDocVQATesseractParquetAdapter
from aaber.data.visrag_mpdocvqa import VisRAGMPDocVQAAdapter


def adapter_from_config(config: dict[str, Any]):
    dataset = str(config.get("dataset", "vidore"))
    source = config.get("source", {})
    if source.get("format") == "irpapers_csv":
        return IRPapersCSVAdapter(
            dataset=dataset,
            docs_csv=Path(source.get("docs_csv", "data/irpapers/raw/irpapers-docs.csv")),
            queries_csv=Path(source.get("queries_csv", "data/irpapers/raw/irpapers-queries.csv")),
            image_dir=Path(source.get("image_dir", "data/irpapers/pages")),
            source_name=str(source.get("name", "weaviate/IRPAPERS")),
            split_config=config.get("splits", source.get("splits", {})),
        )
    if source.get("format") == "visrag_mpdocvqa":
        return VisRAGMPDocVQAAdapter(
            dataset=dataset,
            corpus_parquet=Path(source.get("corpus_parquet", "data/visrag_mpdocvqa/raw/corpus/train-00000-of-00001.parquet")),
            queries_parquet=Path(source.get("queries_parquet", "data/visrag_mpdocvqa/raw/queries/train-00000-of-00001.parquet")),
            qrels_parquet=Path(source.get("qrels_parquet", "data/visrag_mpdocvqa/raw/qrels/train-00000-of-00001.parquet")),
            image_dir=Path(source.get("image_dir", "data/visrag_mpdocvqa/pages")),
            source_name=str(source.get("name", "openbmb/VisRAG-Ret-Test-MP-DocVQA")),
            split_config=config.get("splits", source.get("splits", {})),
        )
    if source.get("format") == "vidore_docvqa_tesseract_parquet" or source.get("parquet"):
        parquet_path = Path(source.get("parquet", "data/vidore_docvqa_test_subsampled_tesseract/raw/test-00000-of-00001.parquet"))
        image_dir = Path(source.get("image_dir", "data/vidore_docvqa_test_subsampled_tesseract/pages"))
        return VidoreDocVQATesseractParquetAdapter(
            dataset=dataset,
            parquet_path=parquet_path,
            image_dir=image_dir,
            source_name=str(source.get("name", "vidore/docvqa_test_subsampled_tesseract")),
            split_config=config.get("splits", source.get("splits", {})),
        )
    return FileDatasetAdapter(
        dataset=dataset,
        queries_path=Path(source.get("queries", f"data/{dataset}/queries.jsonl")),
        qrels_path=Path(source.get("qrels", f"data/{dataset}/qrels.jsonl")),
        units_path=Path(source.get("units", f"data/{dataset}/evidence_units.jsonl")),
        relations_path=Path(source.get("relations", f"data/{dataset}/relations.jsonl")),
    )
