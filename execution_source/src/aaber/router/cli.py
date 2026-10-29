from __future__ import annotations

import argparse
import logging
from pathlib import Path

from aaber.actions.base import RetrievalAction
from aaber.actions.cross_rerank import CrossEncoderRerankAction
from aaber.actions.dense_search import DenseSearchAction
from aaber.actions.image_search import ImageSearchAction
from aaber.actions.multivector_rerank import MultiVectorRerankAction, MultiVectorSearchAction
from aaber.actions.relation_expand import RelationExpandAction
from aaber.actions.simulate_actions import simulate_action_labels
from aaber.actions.text_search import TextSearchAction
from aaber.actions.vlm_verify import VLMVerifyAction
from aaber.common.config import load_config
from aaber.common.io import artifact_dataset_dir, ensure_dir, read_jsonl, write_json, write_jsonl
from aaber.common.logging import setup_logging
from aaber.data.build_relations import build_default_relations
from aaber.data.dataset_base import write_prepared_dataset
from aaber.data.schema import EvidenceResult, EvidenceUnit, Query, Relation
from aaber.data.vidore import adapter_from_config
from aaber.eval.metrics import evaluate_rankings
from aaber.features.build_features import build_candidate_features
from aaber.index.build_index import create_index, load_index
from aaber.index.multivector_index import MultiVectorRerankIndex
from aaber.index.precomputed_source_index import PrecomputedSourceIndex
from aaber.router.fusion import fuse_results, rankings_from_results
from aaber.router.inference import run_budgeted_routing
from aaber.router.lgbm_router import LightGBMActionGainRouter
from aaber.router.mlp_router import MLPActionGainRouter

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    parser = argparse.ArgumentParser(prog="aaber")
    parser.add_argument("--artifact-root", default="artifacts")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare-data")
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="all")
    p.add_argument("--dataset-override", default=None)
    p.add_argument("--split-seed", type=int, default=None)
    p.set_defaults(func=cmd_prepare_data)

    p = sub.add_parser("build-index")
    p.add_argument("--dataset", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--model-name", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output-index", default=None)
    p.set_defaults(func=cmd_build_index)

    p = sub.add_parser("run-baseline")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--retriever", required=True)
    p.add_argument("--top-k", type=int, default=10)
    p.set_defaults(func=cmd_run_baseline)

    p = sub.add_parser("build-candidates")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--retrievers", default="bm25,dense_text,clip_image")
    p.add_argument("--top-m", type=int, default=100)
    p.set_defaults(func=cmd_build_candidates)

    p = sub.add_parser("build-features")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--relation-types", default="all")
    p.add_argument("--relation-topic-min-sim", type=float, default=0.0)
    p.add_argument("--tag", default="")
    p.set_defaults(func=cmd_build_features)

    p = sub.add_parser("simulate-actions")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--actions", default="text-search,image-search,relation-expand,multivector-rerank")
    p.add_argument("--relation-types", default="all")
    p.add_argument("--relation-topic-min-sim", type=float, default=0.0)
    p.add_argument("--tag", default="")
    p.add_argument("--top-m", type=int, default=100)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--gain-metric", choices=["recall", "mrr", "ndcg"], default="ndcg")
    p.add_argument(
        "--fusion-mode",
        default="weighted_sum",
        choices=["weighted_sum", "sum", "max_norm", "rrf", "visual_dominant"],
        help="Fusion strategy used when computing before/after action gains.",
    )
    p.set_defaults(func=cmd_simulate_actions)

    p = sub.add_parser("train-router")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--router", choices=["lgbm", "mlp"], default="lgbm")
    p.add_argument("--target", default="gain")
    p.add_argument("--tag", default="")
    p.set_defaults(func=cmd_train_router)

    p = sub.add_parser("run-router")
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--router", choices=["lgbm", "mlp"], default="lgbm")
    p.add_argument("--budget", type=float, default=10.0)
    p.add_argument("--actions", default="text-search,image-search,relation-expand,multivector-rerank")
    p.add_argument("--relation-types", default="all")
    p.add_argument("--relation-topic-min-sim", type=float, default=0.0)
    p.add_argument("--tag", default="")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--min-gain", type=float, default=0.0)
    p.add_argument(
        "--min-gain-by-action",
        default="",
        help="Comma-separated thresholds, for example: image-search:0.05,text-search:0.0",
    )
    p.add_argument(
        "--suppress-image-if-text-gain-above",
        type=float,
        default=None,
        help="Drop image-search candidates when the query-level text-search predicted gain is at least this value.",
    )
    p.add_argument(
        "--fusion-mode",
        default="weighted_sum",
        choices=["weighted_sum", "sum", "max_norm", "rrf", "visual_dominant"],
        help="Fusion strategy for combining base candidates and selected action results.",
    )
    p.set_defaults(func=cmd_run_router)

    args = parser.parse_args(argv)
    args.func(args)


def cmd_prepare_data(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    if args.dataset_override:
        config["dataset"] = args.dataset_override
    if args.split_seed is not None:
        split_config = dict(config.get("splits", {}))
        split_config["seed"] = args.split_seed
        config["splits"] = split_config
    artifact_root = config.get("artifact_root", args.artifact_root)
    adapter = adapter_from_config(config)
    queries = adapter.load_queries(args.split)
    qrels = adapter.load_qrels(args.split)
    units = adapter.build_units(args.split)
    relations = adapter.build_relations(args.split)
    if config.get("auto_relations", True):
        relations = _merge_relations([*relations, *build_default_relations(units)])
    out_dir = artifact_dataset_dir(adapter.dataset, artifact_root)
    write_prepared_dataset(out_dir, queries, qrels, units, relations)
    LOGGER.info("Prepared %s: %d queries, %d units, %d relations", adapter.dataset, len(queries), len(units), len(relations))


def cmd_build_index(args: argparse.Namespace) -> None:
    dataset_dir = artifact_dataset_dir(args.dataset, args.artifact_root)
    units = _load_units(dataset_dir)
    if args.limit is not None:
        units = units[: max(0, args.limit)]
    kwargs = {"model_name": args.model_name} if args.model_name else {}
    index = create_index(args.index, **kwargs)
    index.build(units)
    out_dir = dataset_dir / "indexes" / (args.output_index or args.index)
    ensure_dir(out_dir)
    index.save(out_dir)
    LOGGER.info("Built %s index for %s with %d units", args.index, args.dataset, len(units))


def cmd_run_baseline(args: argparse.Namespace) -> None:
    dataset_dir = artifact_dataset_dir(args.dataset, args.artifact_root)
    queries, qrels, _units, _relations = _load_dataset(dataset_dir, args.split)
    index = load_index(args.retriever, dataset_dir / "indexes" / args.retriever)
    results: list[EvidenceResult] = []
    for query in queries:
        results.extend(index.search(query, top_k=args.top_k))
    rankings = rankings_from_results(results)
    metrics = evaluate_rankings(rankings, qrels)
    run_dir = dataset_dir / "runs" / f"baseline_{args.retriever}_{args.split}"
    ensure_dir(run_dir)
    write_jsonl(run_dir / "ranked_evidence.jsonl", [r.to_dict() for r in results])
    write_json(run_dir / "metrics.json", metrics)
    LOGGER.info("Baseline %s metrics: %s", args.retriever, metrics)


def cmd_build_candidates(args: argparse.Namespace) -> None:
    dataset_dir = artifact_dataset_dir(args.dataset, args.artifact_root)
    queries, _qrels, _units, _relations = _load_dataset(dataset_dir, args.split)
    retrievers = _split_csv(args.retrievers)
    all_results: list[EvidenceResult] = []
    for retriever in retrievers:
        index_path = dataset_dir / "indexes" / retriever
        if not index_path.exists():
            LOGGER.warning("Skipping missing index: %s", index_path)
            continue
        index = load_index(retriever, index_path)
        LOGGER.info("Building %s candidates for %s/%s (%d queries)", retriever, args.dataset, args.split, len(queries))
        for idx, query in enumerate(queries, start=1):
            all_results.extend(index.search(query, top_k=args.top_m))
            if idx == len(queries) or idx % 100 == 0:
                LOGGER.info(
                    "Built %s candidates for %s/%s: %d/%d queries",
                    retriever,
                    args.dataset,
                    args.split,
                    idx,
                    len(queries),
                )
    out = dataset_dir / "candidates" / f"coarse_candidates_{args.split}.jsonl"
    write_jsonl(out, [r.to_dict() for r in all_results])
    LOGGER.info("Wrote %d candidate rows to %s", len(all_results), out)


def cmd_build_features(args: argparse.Namespace) -> None:
    dataset_dir = artifact_dataset_dir(args.dataset, args.artifact_root)
    queries, _qrels, units, relations = _load_dataset(dataset_dir, args.split)
    relations = _filter_relations(relations, args.relation_types, args.relation_topic_min_sim)
    candidates = _load_candidates(dataset_dir, args.split)
    auxiliary_candidates = _load_auxiliary_candidates(dataset_dir, args.split)
    rows = build_candidate_features(queries, units, relations, candidates, auxiliary_candidates=auxiliary_candidates)
    out = _features_path(dataset_dir, args.split, args.tag)
    write_jsonl(out, rows)
    LOGGER.info("Wrote %d feature rows to %s", len(rows), out)


def cmd_simulate_actions(args: argparse.Namespace) -> None:
    dataset_dir = artifact_dataset_dir(args.dataset, args.artifact_root)
    queries, qrels, units, relations = _load_dataset(dataset_dir, args.split)
    relations = _filter_relations(relations, args.relation_types, args.relation_topic_min_sim)
    features = read_jsonl(_features_path(dataset_dir, args.split, args.tag))
    candidates = _load_candidates(dataset_dir, args.split)
    actions = _build_actions(dataset_dir, units, relations, _split_csv(args.actions), split=args.split)
    labels = simulate_action_labels(
        queries=queries,
        units=units,
        qrels=qrels,
        feature_rows=features,
        base_candidates=candidates,
        actions=list(actions.values()),
        top_k=args.top_k,
        gain_metric=args.gain_metric,
        fusion_mode=args.fusion_mode,
    )
    out = _action_labels_path(dataset_dir, args.split, args.tag)
    write_jsonl(out, labels)
    LOGGER.info("Wrote %d action label rows to %s", len(labels), out)


def cmd_train_router(args: argparse.Namespace) -> None:
    dataset_dir = artifact_dataset_dir(args.dataset, args.artifact_root)
    labels = read_jsonl(_action_labels_path(dataset_dir, args.split, args.tag))
    router = _create_router(args.router)
    router.fit(labels, target=args.target)
    out = _model_path(dataset_dir, args.router, args.tag)
    ensure_dir(out.parent)
    router.save(out)
    LOGGER.info("Trained %s router on %d rows with target=%s -> %s", args.router, len(labels), args.target, out)


def cmd_run_router(args: argparse.Namespace) -> None:
    dataset_dir = artifact_dataset_dir(args.dataset, args.artifact_root)
    queries, qrels, units, relations = _load_dataset(dataset_dir, args.split)
    relations = _filter_relations(relations, args.relation_types, args.relation_topic_min_sim)
    features = read_jsonl(_features_path(dataset_dir, args.split, args.tag))
    candidates = _load_candidates(dataset_dir, args.split)
    router = _load_router(args.router, _model_path(dataset_dir, args.router, args.tag))
    actions = _build_actions(dataset_dir, units, relations, _split_csv(args.actions), split=args.split)
    min_gain_by_action = _parse_action_thresholds(args.min_gain_by_action)
    results, traces = run_budgeted_routing(
        queries=queries,
        units=units,
        feature_rows=features,
        base_candidates=candidates,
        actions=actions,
        router=router,
        budget=args.budget,
        top_k=args.top_k,
        max_per_action={"vlm-verify": 3, "multivector-rerank": 1, "relation-expand": 2},
        min_gain=args.min_gain,
        min_gain_by_action=min_gain_by_action,
        suppress_image_if_text_gain_above=args.suppress_image_if_text_gain_above,
        fusion_mode=args.fusion_mode,
    )
    rankings = rankings_from_results(results)
    metrics = evaluate_rankings(rankings, qrels)
    run_dir = dataset_dir / "runs" / _run_name(
        args.router,
        args.budget,
        args.split,
        args.tag,
        args.min_gain,
        min_gain_by_action,
        args.suppress_image_if_text_gain_above,
        args.fusion_mode,
    )
    ensure_dir(run_dir)
    write_jsonl(run_dir / "ranked_evidence.jsonl", [r.to_dict() for r in results])
    write_jsonl(run_dir / "routing_trace.jsonl", [t.to_dict() for t in traces])
    write_json(run_dir / "metrics.json", metrics)
    action_stats = _action_stats(traces)
    write_json(run_dir / "action_stats.json", action_stats)
    LOGGER.info("Router metrics: %s", metrics)
    LOGGER.info("Action stats: %s", action_stats)


def _load_dataset(dataset_dir: Path, split: str = "all") -> tuple[list[Query], dict[str, set[str]], list[EvidenceUnit], list[Relation]]:
    all_queries = [Query.from_dict(row) for row in read_jsonl(dataset_dir / "queries.jsonl")]
    if split != "all" and any(query.metadata.get("split") is not None for query in all_queries):
        queries = [query for query in all_queries if query.metadata.get("split") == split]
    else:
        queries = all_queries
    query_ids = {query.query_id for query in queries}
    qrels = {
        str(row["query_id"]): {str(x) for x in row.get("positive_unit_ids", row.get("positive_page_ids", []))}
        for row in read_jsonl(dataset_dir / "qrels.jsonl")
        if str(row["query_id"]) in query_ids
    }
    units = _load_units(dataset_dir)
    relations = [Relation.from_dict(row) for row in read_jsonl(dataset_dir / "units" / "relations.jsonl")]
    return queries, qrels, units, relations


def _load_units(dataset_dir: Path) -> list[EvidenceUnit]:
    return [EvidenceUnit.from_dict(row) for row in read_jsonl(dataset_dir / "units" / "evidence_units.jsonl")]


def _load_candidates(dataset_dir: Path, split: str) -> list[EvidenceResult]:
    path = dataset_dir / "candidates" / f"coarse_candidates_{split}.jsonl"
    rows = read_jsonl(path)
    if not rows:
        LOGGER.warning("No candidates found at %s", path)
    return [EvidenceResult.from_dict(row) for row in rows]


def _load_auxiliary_candidates(dataset_dir: Path, split: str) -> list[EvidenceResult]:
    rows: list[dict] = []
    for path in [dataset_dir / "candidates" / f"image_candidates_{split}.jsonl"]:
        if path.exists():
            rows.extend(read_jsonl(path))
    return [EvidenceResult.from_dict(row) for row in rows]


def _build_actions(
    dataset_dir: Path,
    units: list[EvidenceUnit],
    relations: list[Relation],
    action_names: list[str],
    split: str | None = None,
) -> dict[str, RetrievalAction]:
    actions: dict[str, RetrievalAction] = {}
    for name in action_names:
        if name == "text-search":
            index = _load_first_available_index(dataset_dir, ["bm25", "sentence_transformer", "dense_text"])
            actions[name] = TextSearchAction(index)
        elif name == "dense-search":
            index = _load_first_available_index(dataset_dir, ["sentence_transformer"])
            actions[name] = DenseSearchAction(index)
        elif name == "image-search":
            index = _load_precomputed_source_index(
                dataset_dir=dataset_dir,
                split=split,
                source_actions={"clip_transformers", "siglip_image", "clip_image", "image"},
                name="precomputed_image_search",
            )
            if index is None:
                index = _load_first_available_index(dataset_dir, ["clip_transformers", "siglip_image", "clip_image", "image"])
            actions[name] = ImageSearchAction(index)
        elif name == "cross-rerank":
            index = _load_first_available_index(dataset_dir, ["cross_encoder", "cross-rerank", "cross_rerank"])
            actions[name] = CrossEncoderRerankAction(index)
        elif name == "relation-expand":
            actions[name] = RelationExpandAction(relations, units)
        elif name == "multivector-rerank":
            index = _load_first_available_index(dataset_dir, ["colpali_engine", "colpali", "colqwen"])
            if not hasattr(index, "rerank"):
                fallback = MultiVectorRerankIndex()
                fallback.build(units)
                index = fallback
            actions[name] = MultiVectorRerankAction(index)
        elif name == "multivector-search":
            index = _load_first_available_index(dataset_dir, ["colpali_engine", "colpali", "colqwen"])
            actions[name] = MultiVectorSearchAction(index)
        elif name == "vlm-verify":
            actions[name] = VLMVerifyAction()
        else:
            raise KeyError(f"Unknown action: {name}")
    return actions


def _load_first_available_index(dataset_dir: Path, names: list[str]):
    for name in names:
        path = dataset_dir / "indexes" / name
        if path.exists():
            return load_index(name, path)
    fallback = create_index(names[0])
    units = _load_units(dataset_dir)
    fallback.build(units)
    return fallback


def _load_precomputed_source_index(
    dataset_dir: Path,
    split: str | None,
    source_actions: set[str],
    name: str,
) -> PrecomputedSourceIndex | None:
    candidate_dir = dataset_dir / "candidates"
    if split:
        paths = [
            candidate_dir / f"image_candidates_{split}.jsonl",
            candidate_dir / f"coarse_candidates_{split}.jsonl",
        ]
    else:
        paths = sorted(candidate_dir.glob("image_candidates_*.jsonl"))
        paths.extend(sorted(candidate_dir.glob("coarse_candidates_*.jsonl")))
    index = PrecomputedSourceIndex.from_candidate_files(paths, source_actions=source_actions, name=name)
    return index if index.has_results() else None


def _create_router(name: str):
    if name == "mlp":
        return MLPActionGainRouter()
    return LightGBMActionGainRouter()


def _load_router(name: str, path: Path):
    if name == "mlp":
        return MLPActionGainRouter.load(path)
    return LightGBMActionGainRouter.load(path)


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_action_thresholds(value: str) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for item in _split_csv(value):
        if ":" not in item:
            raise ValueError(f"Invalid --min-gain-by-action item '{item}', expected action:threshold")
        action, threshold = item.split(":", 1)
        action = action.strip()
        if not action:
            raise ValueError(f"Invalid --min-gain-by-action item '{item}', action is empty")
        thresholds[action] = float(threshold)
    return thresholds


def _action_stats(traces) -> dict[str, float]:
    counts: dict[str, float] = {}
    total_queries = max(1, len(traces))
    used_budget = 0.0
    for trace in traces:
        used_budget += trace.used_budget
        for item in trace.selected_actions:
            action = item["action"]
            counts[action] = counts.get(action, 0.0) + 1.0
    stats = {f"{name}_calls_per_query": count / total_queries for name, count in sorted(counts.items())}
    stats["avg_used_budget"] = used_budget / total_queries
    return stats


def _merge_relations(relations: list[Relation]) -> list[Relation]:
    seen: set[tuple[str, str, str]] = set()
    merged: list[Relation] = []
    for relation in relations:
        key = (relation.src_unit_id, relation.dst_unit_id, relation.relation_type)
        if key in seen:
            continue
        seen.add(key)
        merged.append(relation)
    return merged


def _filter_relations(relations: list[Relation], relation_types: str, topic_min_similarity: float = 0.0) -> list[Relation]:
    requested = [item.strip() for item in relation_types.split(",") if item.strip()]
    if not requested or requested == ["all"]:
        filtered = relations
    elif requested == ["none"]:
        return []
    else:
        allowed = set(requested)
        filtered = [relation for relation in relations if relation.relation_type in allowed]
    if topic_min_similarity <= 0.0:
        return filtered
    result: list[Relation] = []
    for relation in filtered:
        if relation.relation_type != "same_topic":
            result.append(relation)
            continue
        similarity = float(relation.metadata.get("similarity", 0.0) or 0.0)
        if similarity >= topic_min_similarity:
            result.append(relation)
    return result


def _features_path(dataset_dir: Path, split: str, tag: str = "") -> Path:
    return dataset_dir / "features" / f"query_unit_features_{split}{_tag_suffix(tag)}.jsonl"


def _action_labels_path(dataset_dir: Path, split: str, tag: str = "") -> Path:
    return dataset_dir / "action_labels" / f"action_labels_{split}{_tag_suffix(tag)}.jsonl"


def _model_path(dataset_dir: Path, router: str, tag: str = "") -> Path:
    return dataset_dir / "models" / f"{router}_router{_tag_suffix(tag)}.json"


def _run_name(
    router: str,
    budget: float,
    split: str,
    tag: str = "",
    min_gain: float = 0.0,
    min_gain_by_action: dict[str, float] | None = None,
    suppress_image_if_text_gain_above: float | None = None,
    fusion_mode: str = "weighted_sum",
) -> str:
    suffix = _tag_suffix(tag)
    if min_gain > 0:
        suffix += _tag_suffix(f"mg{min_gain:g}")
    for action, threshold in sorted((min_gain_by_action or {}).items()):
        if threshold > 0:
            suffix += _tag_suffix(f"mg_{action}_{threshold:g}")
    if suppress_image_if_text_gain_above is not None:
        suffix += _tag_suffix(f"textguard_{suppress_image_if_text_gain_above:g}")
    normalized_fusion = fusion_mode.lower().replace("-", "_")
    if normalized_fusion not in {"weighted_sum", "sum"}:
        suffix += _tag_suffix(f"fusion_{normalized_fusion}")
    return f"router_{router}_b{budget:g}_{split}{suffix}"


def _tag_suffix(tag: str) -> str:
    if not tag:
        return ""
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in tag).strip("_")
    return f"_{safe}" if safe else ""


if __name__ == "__main__":
    main()
