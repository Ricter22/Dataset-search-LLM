import argparse
import csv
import importlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from datasets import load_dataset
from dotenv import load_dotenv
from target_benchmark.evaluators import TARGET
from target_benchmark.retrievers import AbsCustomEmbeddingRetriever

import preprocessing_utils as utils


ARTIFACT_FILES = {
    ("ottqa", "queryopt_sem"): "query_optimization_ottqa_result_recall_1_3_5_10.json",
    ("ottqa", "queryopt_nosem"): "query_optimization_ottqa_result_nosem_recall_1_3_5_10.json",
    ("fetaqa", "queryopt_sem"): "query_optimization_result_recall_1_3_5.json",
    ("fetaqa", "queryopt_nosem"): "query_optimization_fetaqa_result_nosem_recall_1_3_5_10.json",
}

COLLECTION_NAMES = {
    ("fetaqa", "noquery_sem"): "test_instructions_fetaqa",
    ("fetaqa", "noquery_nosem"): "test_instructions_fetaqa_no_sem_profile",
    ("ottqa", "noquery_sem"): "test_instructions_ottqa",
    ("ottqa", "noquery_nosem"): "test_instructions_ottqa_no_sem_profile",
}

QUERYOPT_VARIANTS = {"queryopt_sem", "queryopt_nosem"}
NOQUERY_VARIANTS = {"noquery_sem", "noquery_nosem"}
ALL_VARIANTS = ["queryopt_sem", "queryopt_nosem", "noquery_sem", "noquery_nosem"]
VALID_K = [1, 3, 5, 10]


class ReplayRetriever(AbsCustomEmbeddingRetriever):
    def __init__(self, query_to_results: Dict[str, List[Tuple[str, str]]]):
        super().__init__(expected_corpus_format="nested array")
        self.query_to_results = query_to_results

    def embed_corpus(self, dataset_name: str, corpus: Iterable[Dict]) -> None:
        # Tables are already embedded in the persisted ChromaDB.
        return

    def retrieve(self, query: str, dataset_name: str, top_k: int, **kwargs) -> List[Tuple]:
        return self.query_to_results.get(query, [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover TARGET results from existing artifacts and collections.")
    parser.add_argument("--datasets", nargs="+", default=["fetaqa", "ottqa"], choices=["fetaqa", "ottqa"])
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10], choices=VALID_K)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--chroma-path", default="/chroma_db")
    parser.add_argument("--output-dir", default="target_recovery_results")
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        help="Embedding model used to query baseline collections (must match indexed embedding space).",
    )
    parser.add_argument(
        "--embedding-api-base",
        default=os.getenv("EMBEDDING_API_BASE"),
        help="OpenAI-compatible API base for embeddings (e.g., https://openrouter.ai/api/v1).",
    )
    parser.add_argument(
        "--openrouter-site-url",
        default=os.getenv("OPENROUTER_SITE_URL"),
        help="Optional HTTP-Referer header for OpenRouter.",
    )
    parser.add_argument(
        "--openrouter-app-name",
        default=os.getenv("OPENROUTER_APP_NAME", "target-recovery"),
        help="Optional X-Title header for OpenRouter.",
    )
    parser.add_argument("--smoke-only", action="store_true")
    return parser.parse_args()


def normalize_tuple_list(values: List[List[str]]) -> List[Tuple[str, str]]:
    if not values:
        return []
    return [(str(x[0]), str(x[1])) for x in values]


def load_queryopt_artifact(path: Path) -> Dict[int, Dict[str, List[Tuple[str, str]]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    query_maps: Dict[int, Dict[str, List[Tuple[str, str]]]] = {1: {}, 3: {}, 5: {}, 10: {}}
    for batch in data:
        for entry in batch:
            query = entry[0]
            payload = entry[1]
            if isinstance(payload, dict):
                for k_key, tuples_list in payload.items():
                    k_int = int(k_key)
                    query_maps[k_int][query] = normalize_tuple_list(tuples_list)
            elif isinstance(payload, list):
                # Legacy artifact shape containing only a single result list (treat as k=5).
                query_maps[5][query] = normalize_tuple_list(payload)
    return query_maps


def save_baseline_cache(path: Path, recall_map: Dict[int, Dict[str, List[Tuple[str, str]]]]) -> None:
    serializable = {}
    for k, query_map in recall_map.items():
        serializable[str(k)] = {q: [[db, tid] for db, tid in pairs] for q, pairs in query_map.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def load_baseline_cache(path: Path) -> Dict[int, Dict[str, List[Tuple[str, str]]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    recall_map: Dict[int, Dict[str, List[Tuple[str, str]]]] = {1: {}, 3: {}, 5: {}, 10: {}}
    for k_key, query_map in raw.items():
        k_int = int(k_key)
        for query, pairs in query_map.items():
            recall_map[k_int][query] = normalize_tuple_list(pairs)
    return recall_map


def run_preflight_checks(
    selected_datasets: List[str], split: str, chroma_client: chromadb.PersistentClient, chroma_path: str
) -> Dict[str, List[str]]:
    info = {"warnings": [], "errors": []}

    for dataset in selected_datasets:
        for variant in QUERYOPT_VARIANTS:
            artifact_file = ARTIFACT_FILES[(dataset, variant)]
            artifact_path = Path(artifact_file)
            if not artifact_path.exists():
                info["errors"].append(f"Missing artifact file: {artifact_path}")

    for dataset in selected_datasets:
        for variant in NOQUERY_VARIANTS:
            collection_name = COLLECTION_NAMES[(dataset, variant)]
            try:
                collection = chroma_client.get_collection(name=collection_name)
                count = collection.count()
                if count == 0:
                    info["errors"].append(f"Collection '{collection_name}' is empty (chroma path: {chroma_path}).")
            except Exception as exc:
                info["errors"].append(f"Missing/inaccessible collection '{collection_name}': {exc}")

    for dataset in selected_datasets:
        ds = load_dataset(f"target-benchmark/{dataset}-queries", split=split)
        if len(ds) == 0:
            info["errors"].append(f"Dataset target-benchmark/{dataset}-queries ({split}) has zero queries.")

    return info


def build_baseline_recall_map(
    collection,
    queries: List[str],
    log_prefix: str,
) -> Dict[int, Dict[str, List[Tuple[str, str]]]]:
    recall_map: Dict[int, Dict[str, List[Tuple[str, str]]]] = {1: {}, 3: {}, 5: {}, 10: {}}
    total = len(queries)
    for i, query in enumerate(queries, 1):
        response = collection.query(query_texts=[query], include=["metadatas"], n_results=100)
        tuples_list: List[Tuple[str, str]] = []
        for batch in response["metadatas"]:
            for metadata in batch:
                tuples_list.append((str(metadata["database_id"]), str(metadata["table_id"])))

        recall_map[1][query] = utils.top_k_tuples(tuples_list, 1)
        recall_map[3][query] = utils.top_k_tuples(tuples_list, 3)
        recall_map[5][query] = utils.top_k_tuples(tuples_list, 5)
        recall_map[10][query] = utils.top_k_tuples(tuples_list, 10)

        if i % 200 == 0 or i == total:
            print(f"[baseline-cache] {log_prefix}: {i}/{total}")
    return recall_map


def expected_rows_for_run(datasets: List[str], k_values: List[int]) -> int:
    rows = 0
    for dataset in datasets:
        for variant in ALL_VARIANTS:
            for k in k_values:
                rows += 1
                if dataset == "fetaqa" and variant == "queryopt_sem" and k == 10:
                    # Included as not_available summary row.
                    continue
    return rows


def count_empty_retrieval_rows(path: Path) -> int:
    if not path.exists():
        return -1
    empty_count = 0
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            row = json.loads(line)
            if not row.get("retrieval_results"):
                empty_count += 1
    return empty_count


def flatten_summary_row(
    run_id: str,
    dataset: str,
    variant: str,
    k: int,
    status: str,
    reason: str,
    performance: Dict,
    empty_retrieval_rows: int,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "run_id": run_id,
        "dataset": dataset,
        "variant": variant,
        "k": k,
        "status": status,
        "reason": reason,
        "empty_retrieval_rows": empty_retrieval_rows,
    }
    if performance is None:
        for metric in [
            "accuracy",
            "precision",
            "recall",
            "capped_recall",
            "retrieval_duration_process",
            "avg_retrieval_duration_process",
            "retrieval_duration_wall_clock",
            "avg_retrieval_duration_wall_clock",
            "embedding_creation_duration_process",
            "avg_embedding_creation_duration_process",
            "embedding_creation_duration_wall_clock",
            "avg_embedding_creation_duration_wall_clock",
            "embedding_size",
            "avg_embedding_size",
        ]:
            row[metric] = None
        return row

    task_blob = performance["Table Retrieval Task"][dataset]
    retrieval_metrics = task_blob["retrieval_performance"]
    embedding_stats = task_blob["embedding_statistics"]
    row["accuracy"] = retrieval_metrics.get("accuracy")
    row["precision"] = retrieval_metrics.get("precision")
    row["recall"] = retrieval_metrics.get("recall")
    row["capped_recall"] = retrieval_metrics.get("capped_recall")
    row["retrieval_duration_process"] = retrieval_metrics.get("retrieval_duration_process")
    row["avg_retrieval_duration_process"] = retrieval_metrics.get("avg_retrieval_duration_process")
    row["retrieval_duration_wall_clock"] = retrieval_metrics.get("retrieval_duration_wall_clock")
    row["avg_retrieval_duration_wall_clock"] = retrieval_metrics.get("avg_retrieval_duration_wall_clock")
    row["embedding_creation_duration_process"] = embedding_stats.get("embedding_creation_duration_process")
    row["avg_embedding_creation_duration_process"] = embedding_stats.get("avg_embedding_creation_duration_process")
    row["embedding_creation_duration_wall_clock"] = embedding_stats.get("embedding_creation_duration_wall_clock")
    row["avg_embedding_creation_duration_wall_clock"] = embedding_stats.get("avg_embedding_creation_duration_wall_clock")
    row["embedding_size"] = embedding_stats.get("embedding_size")
    row["avg_embedding_size"] = embedding_stats.get("avg_embedding_size")
    return row


def run_target(
    dataset: str,
    k: int,
    query_map: Dict[str, List[Tuple[str, str]]],
    split: str,
    batch_size: int,
    output_dir: Path,
    run_id: str,
) -> Tuple[Dict, int]:
    retrieval_dir = output_dir / "retrieval_results" / run_id
    downstream_dir = output_dir / "downstream_results" / run_id
    if retrieval_dir.exists():
        shutil.rmtree(retrieval_dir)
    if downstream_dir.exists():
        shutil.rmtree(downstream_dir)

    retriever = ReplayRetriever(query_map)
    target = TARGET(("Table Retrieval Task", dataset), persist_log=False)
    performance = target.run(
        retriever=retriever,
        split=split,
        batch_size=batch_size,
        top_k=k,
        retrieval_results_dir=str(retrieval_dir),
        downstream_results_dir=str(downstream_dir),
    )
    retrieval_file = retrieval_dir / dataset / f"{k}.jsonl"
    empty_rows = count_empty_retrieval_rows(retrieval_file)
    return performance, empty_rows


def patch_target_utf8_persistence() -> None:
    def append_results_utf8(results, path_to_persistence):
        if not path_to_persistence:
            return
        path_to_persistence.touch()
        with open(path_to_persistence, "a", encoding="utf-8") as file:
            for result in results:
                file.write(result.model_dump_json() + "\n")

    tasks_utils_module = importlib.import_module("target_benchmark.tasks.utils")
    abs_task_module = importlib.import_module("target_benchmark.tasks.AbsTask")
    tasks_utils_module.append_results = append_results_utf8
    abs_task_module.append_results = append_results_utf8


def resolve_embedding_provider(args: argparse.Namespace) -> Dict[str, object]:
    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if openai_key:
        return {
            "provider": "openai",
            "api_key": openai_key,
            "api_base": args.embedding_api_base,
            "model_name": args.embedding_model,
            "default_headers": None,
        }

    if openrouter_key:
        default_headers = {}
        if args.openrouter_site_url:
            default_headers["HTTP-Referer"] = args.openrouter_site_url
        if args.openrouter_app_name:
            default_headers["X-Title"] = args.openrouter_app_name
        return {
            "provider": "openrouter",
            "api_key": openrouter_key,
            "api_base": args.embedding_api_base or "https://openrouter.ai/api/v1",
            "model_name": args.embedding_model,
            "default_headers": default_headers or None,
        }

    return {
        "provider": None,
        "api_key": None,
        "api_base": None,
        "model_name": args.embedding_model,
        "default_headers": None,
    }


def main() -> None:
    args = parse_args()
    load_dotenv()
    embedding_cfg = resolve_embedding_provider(args)
    if (not args.smoke_only) and (not embedding_cfg["api_key"]):
        raise EnvironmentError(
            "Missing API key for baseline retrieval. Set OPENAI_API_KEY or OPENROUTER_API_KEY."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=args.chroma_path)
    patch_target_utf8_persistence()
    if embedding_cfg["provider"]:
        print(
            f"[embeddings] Provider={embedding_cfg['provider']} model={embedding_cfg['model_name']} "
            f"api_base={embedding_cfg['api_base'] or 'default'}"
        )

    print("[preflight] Running artifact/collection/dataset checks...")
    preflight = run_preflight_checks(args.datasets, args.split, chroma_client, args.chroma_path)
    for warning in preflight["warnings"]:
        print(f"[preflight][warning] {warning}")
    if preflight["errors"]:
        for err in preflight["errors"]:
            print(f"[preflight][error] {err}")
        raise RuntimeError("Preflight failed.")
    print("[preflight] OK")

    queryopt_maps: Dict[Tuple[str, str], Dict[int, Dict[str, List[Tuple[str, str]]]]] = {}
    for dataset in args.datasets:
        for variant in QUERYOPT_VARIANTS:
            artifact_path = Path(ARTIFACT_FILES[(dataset, variant)])
            queryopt_maps[(dataset, variant)] = load_queryopt_artifact(artifact_path)
            print(f"[artifact] Loaded {artifact_path}")

    # Smoke run (fast replay path) before expensive baseline cache building.
    smoke_run_id = "ottqa__queryopt_sem__k5"
    smoke_done = False
    smoke_performance = None
    smoke_empty_rows = None
    if "ottqa" in args.datasets and 5 in args.k:
        smoke_map = queryopt_maps[("ottqa", "queryopt_sem")][5]
        print("[smoke] Running ottqa/queryopt_sem/k=5...")
        smoke_performance, smoke_empty_rows = run_target(
            dataset="ottqa",
            k=5,
            query_map=smoke_map,
            split=args.split,
            batch_size=args.batch_size,
            output_dir=output_dir,
            run_id=smoke_run_id,
        )
        smoke_done = True
        print("[smoke] Completed")

    if args.smoke_only:
        raw = {}
        rows = []
        if smoke_done:
            raw[smoke_run_id] = {
                "dataset": "ottqa",
                "variant": "queryopt_sem",
                "k": 5,
                "status": "ok",
                "reason": "",
                "performance": smoke_performance,
                "empty_retrieval_rows": smoke_empty_rows,
            }
            rows.append(
                flatten_summary_row(
                    run_id=smoke_run_id,
                    dataset="ottqa",
                    variant="queryopt_sem",
                    k=5,
                    status="ok",
                    reason="",
                    performance=smoke_performance,
                    empty_retrieval_rows=smoke_empty_rows,
                )
            )
        else:
            print("[smoke] Skipped because ottqa and k=5 were not both requested.")
        raw_path = output_dir / "metrics_raw.json"
        csv_path = output_dir / "metrics_summary.csv"
        raw_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        if rows:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        print(f"[done] Smoke-only outputs written to: {output_dir}")
        return

    baseline_maps: Dict[Tuple[str, str], Dict[int, Dict[str, List[Tuple[str, str]]]]] = {}
    for dataset in args.datasets:
        ds = load_dataset(f"target-benchmark/{dataset}-queries", split=args.split)
        queries = ds["query"]
        for variant in NOQUERY_VARIANTS:
            collection_name = COLLECTION_NAMES[(dataset, variant)]
            cache_path = output_dir / "baseline_cache" / f"{dataset}__{variant}.json"
            if cache_path.exists():
                baseline_maps[(dataset, variant)] = load_baseline_cache(cache_path)
                print(f"[baseline-cache] Loaded existing cache: {cache_path}")
                continue

            collection = chroma_client.get_collection(
                name=collection_name,
                embedding_function=OpenAIEmbeddingFunction(
                    api_key=embedding_cfg["api_key"],
                    model_name=str(embedding_cfg["model_name"]),
                    api_base=embedding_cfg["api_base"],
                    default_headers=embedding_cfg["default_headers"],
                ),
            )
            print(f"[baseline-cache] Building {dataset}/{variant} from collection '{collection_name}'...")
            recall_map = build_baseline_recall_map(
                collection=collection,
                queries=queries,
                log_prefix=f"{dataset}/{variant}",
            )
            baseline_maps[(dataset, variant)] = recall_map
            save_baseline_cache(cache_path, recall_map)
            print(f"[baseline-cache] Saved: {cache_path}")

    raw_results: Dict[str, Dict] = {}
    summary_rows: List[Dict[str, object]] = []
    empty_failures: List[str] = []

    for dataset in args.datasets:
        for variant in ALL_VARIANTS:
            for k in args.k:
                run_id = f"{dataset}__{variant}__k{k}"

                if dataset == "fetaqa" and variant == "queryopt_sem" and k == 10:
                    reason = "not_available: no precomputed fetaqa semantic query-opt recall@10 artifact"
                    raw_results[run_id] = {
                        "dataset": dataset,
                        "variant": variant,
                        "k": k,
                        "status": "not_available",
                        "reason": reason,
                        "performance": None,
                        "empty_retrieval_rows": None,
                    }
                    summary_rows.append(
                        flatten_summary_row(
                            run_id=run_id,
                            dataset=dataset,
                            variant=variant,
                            k=k,
                            status="not_available",
                            reason=reason,
                            performance=None,
                            empty_retrieval_rows=-1,
                        )
                    )
                    continue

                if smoke_done and run_id == smoke_run_id:
                    performance = smoke_performance
                    empty_rows = smoke_empty_rows
                else:
                    if variant in QUERYOPT_VARIANTS:
                        query_map = queryopt_maps[(dataset, variant)][k]
                    else:
                        query_map = baseline_maps[(dataset, variant)][k]
                    print(f"[run] {run_id}")
                    performance, empty_rows = run_target(
                        dataset=dataset,
                        k=k,
                        query_map=query_map,
                        split=args.split,
                        batch_size=args.batch_size,
                        output_dir=output_dir,
                        run_id=run_id,
                    )

                status = "ok"
                reason = ""
                if empty_rows is not None and empty_rows > 0:
                    status = "warning"
                    reason = f"{empty_rows} queries returned empty retrieval_results"
                    empty_failures.append(run_id)

                raw_results[run_id] = {
                    "dataset": dataset,
                    "variant": variant,
                    "k": k,
                    "status": status,
                    "reason": reason,
                    "performance": performance,
                    "empty_retrieval_rows": empty_rows,
                }
                summary_rows.append(
                    flatten_summary_row(
                        run_id=run_id,
                        dataset=dataset,
                        variant=variant,
                        k=k,
                        status=status,
                        reason=reason,
                        performance=performance,
                        empty_retrieval_rows=empty_rows,
                    )
                )

    expected_rows = expected_rows_for_run(args.datasets, args.k)
    if len(summary_rows) != expected_rows:
        raise RuntimeError(f"Unexpected summary row count: {len(summary_rows)} (expected {expected_rows})")

    raw_path = output_dir / "metrics_raw.json"
    csv_path = output_dir / "metrics_summary.csv"
    raw_path.write_text(json.dumps(raw_results, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[done] Wrote {len(summary_rows)} rows to {csv_path}")
    print(f"[done] Wrote raw metrics to {raw_path}")
    if empty_failures:
        print(f"[warning] Runs with empty retrieval rows: {', '.join(empty_failures)}")


if __name__ == "__main__":
    main()
