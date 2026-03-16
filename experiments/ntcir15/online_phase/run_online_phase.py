#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from config import RUN_CONFIG_BY_NAME, RUN_CONFIGS, RunConfig
from prompts import (
    background_doc_prompt,
    keyword_to_complex_prompt,
    rerank_prompt,
    subquery_decomposition_prompt,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_duration(seconds: float) -> str:
    if seconds < 0 or not math.isfinite(seconds):
        return "unknown"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (SCRIPT_DIR / raw).resolve()
    return path


def safe_json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def openai_client() -> OpenAI:
    root_env = (SCRIPT_DIR.parent.parent / ".env").resolve()
    if root_env.exists():
        load_dotenv(dotenv_path=root_env, override=False)
    else:
        load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set OPENROUTER_API_KEY in .env "
            "(or OPENAI_API_KEY for backward compatibility)."
        )

    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    site_url = os.getenv("OPENROUTER_SITE_URL", "").strip()
    site_name = os.getenv("OPENROUTER_SITE_NAME", "").strip()

    kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
    if site_url:
        kwargs["default_headers"] = {"HTTP-Referer": site_url}
    if site_name:
        headers = kwargs.setdefault("default_headers", {})
        headers["X-Title"] = site_name

    if "openrouter.ai" in str(base_url).lower() and not str(api_key).startswith("sk-or-v1-"):
        print(
            "[auth:warning] OPENROUTER_BASE_URL is set but API key is not in sk-or-v1 format.",
            flush=True,
        )

    return OpenAI(**kwargs)


class OpenAIEmbeddingFunction:
    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.model = model
        self.client = openai_client()

    def __call__(self, input: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(model=self.model, input=input)
        return [item.embedding for item in response.data]

    def embed_query(self, input: str | list[str]) -> list[float] | list[list[float]]:
        if isinstance(input, str):
            return self([input])[0]
        return self(input)

    @staticmethod
    def name() -> str:
        return "openai_embedding_function"

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "OpenAIEmbeddingFunction":
        model = str(config.get("model", "text-embedding-3-small"))
        return OpenAIEmbeddingFunction(model=model)

    def get_config(self) -> dict[str, Any]:
        return {"model": self.model}


def chat_completion(
    prompt: str,
    model: str,
    *,
    max_retries: int = 3,
    retry_wait: float = 2.0,
) -> str:
    client = openai_client()
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            if model == "o1-mini":
                messages = [{"role": "user", "content": prompt}]
            else:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
            response = client.chat.completions.create(model=model, messages=messages)
            return str(response.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt >= max_retries:
                break
            time.sleep(retry_wait * attempt)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Chat completion failed without exception.")


def run_api_preflight(model: str) -> str:
    reply = chat_completion("Reply with exactly: OK", model=model, max_retries=1)
    if "OK" not in reply.upper():
        raise RuntimeError(f"Unexpected preflight response: {reply!r}")
    return reply


def load_topics(path: Path) -> dict[str, str]:
    topics: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split("\t", 1)
            if len(parts) != 2:
                continue
            qid, query_text = parts[0].strip(), parts[1].strip()
            if qid and query_text:
                topics[qid] = query_text
    return dict(sorted(topics.items(), key=lambda x: x[0]))


def parse_numbered_list(raw_text: str) -> list[str]:
    lines = [ln.strip() for ln in str(raw_text).splitlines() if ln.strip()]
    pattern = re.compile(r"^\s*(?:\d+[\)\.\-:]|\-|\*)\s*(.+?)\s*$")
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        m = pattern.match(line)
        item = (m.group(1) if m else line).strip().strip('"').strip("'")
        if not item:
            continue
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def parse_rerank_response(raw_text: str) -> list[dict[str, Any]]:
    text = str(raw_text).strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    def _from_payload(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, dict):
            payload = payload.get("ranking", [])
        if not isinstance(payload, list):
            return []
        normalized: list[dict[str, Any]] = []
        for item in payload:
            if isinstance(item, str):
                normalized.append({"dataset_id": item.strip(), "score": None})
                continue
            if not isinstance(item, dict):
                continue
            dataset_id = str(item.get("dataset_id", "")).strip()
            if not dataset_id:
                continue
            score = item.get("score")
            try:
                score = float(score) if score is not None else None
            except Exception:  # noqa: BLE001
                score = None
            reason = str(item.get("reason", "")).strip()
            normalized.append({"dataset_id": dataset_id, "score": score, "reason": reason})
        return normalized

    try:
        parsed = json.loads(text)
        items = _from_payload(parsed)
        if items:
            return items
    except Exception:  # noqa: BLE001
        pass

    line_pattern = re.compile(
        r"^\s*\d+\s*[\)\.\-:]?\s*([0-9a-fA-F\-]{8,})\s*(?:[\|\-:]\s*([01](?:\.\d+)?))?\s*$"
    )
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        m = line_pattern.match(line.strip())
        if not m:
            continue
        did = m.group(1)
        score = float(m.group(2)) if m.group(2) is not None else None
        out.append({"dataset_id": did, "score": score, "reason": ""})
    return out


def trim_text(value: str, max_chars: int) -> str:
    raw = str(value or "")
    if max_chars <= 0 or len(raw) <= max_chars:
        return raw
    return raw[:max_chars] + " ...[truncated]"


def select_configs(run_all: bool, single_config: str | None) -> list[RunConfig]:
    if single_config:
        cfg = RUN_CONFIG_BY_NAME.get(single_config)
        if cfg is None:
            raise ValueError(f"Unknown config: {single_config}")
        return [cfg]
    if run_all:
        return RUN_CONFIGS
    return [RUN_CONFIG_BY_NAME["baseline"]]


def build_query_sets(
    topics: dict[str, str],
    *,
    query_set: str,
    generate_complex: bool,
    complex_cache_path: Path,
    llm_model: str,
    dry_run: bool,
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    if query_set in {"original", "both"}:
        output["original"] = dict(topics)

    if query_set in {"complex", "both"}:
        cache: dict[str, Any] = {}
        if complex_cache_path.exists():
            with complex_cache_path.open("r", encoding="utf-8") as fh:
                loaded = json.load(fh)
                if isinstance(loaded, dict):
                    cache = loaded
        if not generate_complex:
            missing = [qid for qid in topics if not str(cache.get(qid, {}).get("complex", "")).strip()]
            if missing:
                raise RuntimeError(
                    f"Complex cache missing {len(missing)} queries and generation disabled."
                )
        total = len(topics)
        started = perf_counter()
        done = 0
        for qid, query_text in topics.items():
            done += 1
            existing = cache.get(qid, {})
            if str(existing.get("complex", "")).strip():
                continue
            if dry_run:
                complex_query = f"Find comprehensive tabular datasets about: {query_text}"
            else:
                complex_query = chat_completion(
                    keyword_to_complex_prompt(query_text),
                    model=llm_model,
                ).strip()
            cache[qid] = {
                "original": query_text,
                "complex": complex_query,
                "model": llm_model,
                "generated_at_utc": utc_now(),
            }
            safe_json_dump(complex_cache_path, cache)
            elapsed = perf_counter() - started
            rate = done / elapsed if elapsed > 0 else 0
            eta = (total - done) / rate if rate > 0 else float("inf")
            print(
                (
                    f"[complex] {done}/{total} | "
                    f"elapsed={format_duration(elapsed)} | eta={format_duration(eta)}"
                ),
                flush=True,
            )
        output["complex"] = {
            qid: str(cache[qid]["complex"]).strip()
            for qid in sorted(topics.keys())
        }
    return output


def open_collection(chroma_path: Path, collection_name: str) -> Any:
    import chromadb

    client = chromadb.PersistentClient(path=str(chroma_path))
    embedding_function = OpenAIEmbeddingFunction()
    try:
        return client.get_collection(name=collection_name, embedding_function=embedding_function)
    except Exception as exc:  # noqa: BLE001
        available: list[str] = []
        try:
            available = sorted([c.name for c in client.list_collections()])
        except Exception:  # noqa: BLE001
            available = []
        if "NotFoundError" in type(exc).__name__:
            raise RuntimeError(
                f"Collection '{collection_name}' not found in {chroma_path}. "
                f"Available collections: {available}"
            ) from exc
        raise


def retrieve_candidates_for_subquery(
    collection: Any,
    subquery: str,
    n_results: int,
) -> list[dict[str, Any]]:
    raw = collection.query(
        query_texts=[subquery],
        n_results=n_results,
        include=["metadatas", "distances", "documents"],
    )
    metadatas = raw.get("metadatas", [[]])[0] if raw.get("metadatas") else []
    distances = raw.get("distances", [[]])[0] if raw.get("distances") else []
    documents = raw.get("documents", [[]])[0] if raw.get("documents") else []
    ids = raw.get("ids", [[]])[0] if raw.get("ids") else []

    out: list[dict[str, Any]] = []
    for idx, md in enumerate(metadatas):
        metadata = md if isinstance(md, dict) else {}
        dataset_id = str(metadata.get("dataset_id") or metadata.get("dataset") or "").strip()
        if not dataset_id:
            continue
        distance = float(distances[idx]) if idx < len(distances) else 999.0
        vector_score = 1.0 / (1.0 + max(distance, 0.0))
        out.append(
            {
                "dataset_id": dataset_id,
                "distance": distance,
                "vector_score": vector_score,
                "metadata": metadata,
                "document": str(documents[idx]) if idx < len(documents) else "",
                "doc_id": str(ids[idx]) if idx < len(ids) else "",
            }
        )
    return out


def dedup_candidates(all_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_dataset: dict[str, dict[str, Any]] = {}
    for hit in all_hits:
        did = str(hit.get("dataset_id", "")).strip()
        if not did:
            continue
        prev = by_dataset.get(did)
        if prev is None or float(hit["distance"]) < float(prev["distance"]):
            by_dataset[did] = dict(hit)
    return sorted(by_dataset.values(), key=lambda x: float(x["distance"]))


def rerank_candidates(
    query_text: str,
    candidates: list[dict[str, Any]],
    *,
    llm_model: str,
    max_data_profile_chars: int,
    max_semantic_profile_chars: int,
    dry_run: bool,
) -> list[dict[str, Any]]:
    if not candidates:
        return []

    candidates_for_prompt: list[dict[str, Any]] = []
    for item in candidates:
        md = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
        candidates_for_prompt.append(
            {
                "dataset_id": item["dataset_id"],
                "vector_score": round(float(item.get("vector_score", 0.0)), 6),
                "distance": round(float(item.get("distance", 999.0)), 6),
                "domain": str(md.get("domain", "")),
                "dataset_name": str(md.get("dataset_name", "")),
                "data_profile": trim_text(str(md.get("data_profile", "")), max_data_profile_chars),
                "semantic_profile": trim_text(
                    str(md.get("semantic_profile", "")),
                    max_semantic_profile_chars,
                ),
            }
        )

    if dry_run:
        ordered = sorted(candidates, key=lambda x: float(x.get("vector_score", 0.0)), reverse=True)
        return [
            {
                **item,
                "final_score": float(item.get("vector_score", 0.0)),
                "rank_reason": "dry_run_vector_score",
            }
            for item in ordered
        ]

    response_text = chat_completion(
        rerank_prompt(query_text, candidates_for_prompt),
        model=llm_model,
    )
    parsed = parse_rerank_response(response_text)

    by_id: dict[str, dict[str, Any]] = {str(c["dataset_id"]): c for c in candidates}
    used: set[str] = set()
    reranked: list[dict[str, Any]] = []
    for idx, row in enumerate(parsed):
        did = str(row.get("dataset_id", "")).strip()
        if did not in by_id or did in used:
            continue
        base = dict(by_id[did])
        llm_score = row.get("score")
        if llm_score is None:
            llm_score = max(0.0, 1.0 - (idx / max(1, len(candidates))))
        final_score = float(llm_score)
        base["final_score"] = final_score
        base["rank_reason"] = str(row.get("reason", "")).strip()
        reranked.append(base)
        used.add(did)

    if not reranked:
        ordered = sorted(candidates, key=lambda x: float(x.get("vector_score", 0.0)), reverse=True)
        return [
            {
                **item,
                "final_score": float(item.get("vector_score", 0.0)),
                "rank_reason": "fallback_vector_score",
            }
            for item in ordered
        ]

    for item in sorted(candidates, key=lambda x: float(x.get("vector_score", 0.0)), reverse=True):
        did = str(item.get("dataset_id", ""))
        if did in used:
            continue
        remainder = dict(item)
        remainder["final_score"] = float(item.get("vector_score", 0.0)) * 0.5
        remainder["rank_reason"] = "llm_not_returned_dataset"
        reranked.append(remainder)

    return sorted(reranked, key=lambda x: float(x.get("final_score", 0.0)), reverse=True)


def optimize_query(
    query_text: str,
    *,
    llm_model: str,
    dry_run: bool,
) -> tuple[str, list[str], str]:
    if dry_run:
        return (
            "dry_run_background",
            [query_text, f"{query_text} time trends", f"{query_text} regional breakdown"],
            "dry_run_subqueries",
        )
    background = chat_completion(background_doc_prompt(query_text), model=llm_model)
    subqueries_raw = chat_completion(
        subquery_decomposition_prompt(query_text, background),
        model=llm_model,
    )
    subqueries = parse_numbered_list(subqueries_raw)
    if not subqueries:
        subqueries = [query_text]
    return background, subqueries, subqueries_raw


def process_single_query(
    *,
    query_id: str,
    query_text: str,
    run_config: RunConfig,
    collection: Any,
    llm_model: str,
    n_results_per_subquery: int,
    final_topk: int,
    max_data_profile_chars: int,
    max_semantic_profile_chars: int,
    dry_run: bool,
) -> dict[str, Any]:
    if run_config.use_query_optimization:
        background_doc, subqueries, subqueries_raw = optimize_query(
            query_text,
            llm_model=llm_model,
            dry_run=dry_run,
        )
    else:
        background_doc = ""
        subqueries = [query_text]
        subqueries_raw = ""

    all_hits: list[dict[str, Any]] = []
    retrieval_by_subquery: list[dict[str, Any]] = []
    for sq in subqueries:
        hits = retrieve_candidates_for_subquery(collection, sq, n_results_per_subquery) if not dry_run else []
        all_hits.extend(hits)
        retrieval_by_subquery.append(
            {
                "subquery": sq,
                "hits_count": len(hits),
                "hits": hits,
            }
        )

    deduped = dedup_candidates(all_hits)
    reranked = rerank_candidates(
        query_text=query_text,
        candidates=deduped,
        llm_model=llm_model,
        max_data_profile_chars=max_data_profile_chars,
        max_semantic_profile_chars=max_semantic_profile_chars,
        dry_run=dry_run,
    )
    final_ranking = reranked[: max(1, final_topk)]

    return {
        "query_id": query_id,
        "query_text": query_text,
        "use_query_optimization": run_config.use_query_optimization,
        "background_document": background_doc,
        "subqueries_raw": subqueries_raw,
        "subqueries": subqueries,
        "retrieval_by_subquery": retrieval_by_subquery,
        "raw_hits_count": len(all_hits),
        "deduped_count": len(deduped),
        "final_ranking": final_ranking,
        "processed_at_utc": utc_now(),
    }


def compute_run_diagnostics(results_by_query: dict[str, dict[str, Any]]) -> dict[str, int]:
    retrieved_candidates_total = 0
    ranked_docs_total = 0
    queries_with_any_hits = 0
    queries_with_ranking = 0
    for result in results_by_query.values():
        raw_hits = int(result.get("raw_hits_count", 0) or 0)
        ranking = result.get("final_ranking", [])
        ranking_count = len(ranking) if isinstance(ranking, list) else 0
        retrieved_candidates_total += raw_hits
        ranked_docs_total += ranking_count
        if raw_hits > 0:
            queries_with_any_hits += 1
        if ranking_count > 0:
            queries_with_ranking += 1
    return {
        "retrieved_candidates_total": retrieved_candidates_total,
        "ranked_docs_total": ranked_docs_total,
        "queries_with_any_hits": queries_with_any_hits,
        "queries_with_ranking": queries_with_ranking,
    }


def write_run_trec(
    run_dir: Path,
    run_id: str,
    results_by_query: dict[str, dict[str, Any]],
    final_topk: int,
) -> tuple[Path, int]:
    run_path = run_dir / "run.trec"
    lines: list[str] = []
    for qid in sorted(results_by_query.keys()):
        rows = results_by_query[qid].get("final_ranking", [])
        rank = 1
        for item in rows[:final_topk]:
            did = str(item.get("dataset_id", "")).strip()
            if not did:
                continue
            score = float(item.get("final_score", 0.0))
            lines.append(f"{qid} Q0 {did} {rank} {score:.6f} {run_id}")
            rank += 1
    run_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return run_path, len(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run NTCIR-15 online phase on original and/or complex queries with 4 ablation configs."
        )
    )
    parser.add_argument("--queries-path", default="../evaluation_ready_pack/topics.tsv")
    parser.add_argument("--qrels-path", default="../evaluation_ready_pack/qrels_filtered_unprocessable.txt")
    parser.add_argument("--chroma-path", default="../offline_phase/chroma_ntcir15_eval")
    parser.add_argument(
        "--collection-with-semantic",
        default="instructions_ntcir15_eval_with_semantic",
    )
    parser.add_argument(
        "--collection-without-semantic",
        default="instructions_ntcir15_eval_without_semantic",
    )
    parser.add_argument("--artifacts-dir", default="./artifacts")
    parser.add_argument("--query-set", choices=["original", "complex", "both"], default="both")
    parser.add_argument(
        "--generate-complex-queries",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--complex-cache-path", default="./artifacts/complex_queries.json")
    parser.add_argument(
        "--run-all-configs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When true runs baseline/semantic_only/queryopt_only/full.",
    )
    parser.add_argument(
        "--config",
        choices=sorted(list(RUN_CONFIG_BY_NAME.keys())),
        default=None,
        help="Run a single config; overrides --run-all-configs.",
    )
    parser.add_argument("--n-results-per-subquery", type=int, default=5)
    parser.add_argument("--final-topk", type=int, default=10)
    parser.add_argument("--llm-model", default="gpt-5-mini")
    parser.add_argument("--max-data-profile-chars", type=int, default=2400)
    parser.add_argument("--max-semantic-profile-chars", type=int, default=1200)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--fail-if-empty-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if a run writes an empty run.trec.",
    )
    parser.add_argument(
        "--require-collection-nonempty",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if the selected collection has zero indexed documents.",
    )
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    queries_path = resolve_path(args.queries_path)
    qrels_path = resolve_path(args.qrels_path)
    chroma_path = resolve_path(args.chroma_path)
    artifacts_dir = resolve_path(args.artifacts_dir)
    complex_cache_path = resolve_path(args.complex_cache_path)

    if not queries_path.exists():
        raise FileNotFoundError(f"queries-path not found: {queries_path}")
    if not qrels_path.exists():
        raise FileNotFoundError(f"qrels-path not found: {qrels_path}")
    if not args.dry_run and not chroma_path.exists():
        raise FileNotFoundError(f"chroma-path not found: {chroma_path}")

    topics = load_topics(queries_path)
    if not topics:
        raise RuntimeError(f"No topics found in {queries_path}")

    print(f"[online] loaded {len(topics)} queries from {queries_path}", flush=True)
    if len(topics) != 10:
        print(
            f"[online:warning] expected 10 selected queries, found {len(topics)}",
            flush=True,
        )
    if args.dry_run:
        print(
            "[online:warning] DRY-RUN mode enabled. Retrieval is simulated and run.trec can be empty.",
            flush=True,
        )

    if not args.dry_run:
        print(f"[preflight] checking API auth using model={args.llm_model}", flush=True)
        preflight_reply = run_api_preflight(args.llm_model)
        print(f"[preflight] success: {preflight_reply}", flush=True)
    if args.preflight_only:
        return 0

    query_sets = build_query_sets(
        topics,
        query_set=args.query_set,
        generate_complex=args.generate_complex_queries,
        complex_cache_path=complex_cache_path,
        llm_model=args.llm_model,
        dry_run=args.dry_run,
    )
    selected_configs = select_configs(args.run_all_configs, args.config)
    runs_root = artifacts_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    global_summary: dict[str, Any] = {
        "run_started_at_utc": utc_now(),
        "queries_path": str(queries_path),
        "qrels_path": str(qrels_path),
        "chroma_path": str(chroma_path),
        "query_sets": list(query_sets.keys()),
        "configs": [cfg.name for cfg in selected_configs],
        "n_results_per_subquery": int(args.n_results_per_subquery),
        "final_topk": int(args.final_topk),
        "llm_model": args.llm_model,
        "dry_run": bool(args.dry_run),
        "resume": bool(args.resume),
        "fail_if_empty_run": bool(args.fail_if_empty_run),
        "require_collection_nonempty": bool(args.require_collection_nonempty),
        "status": "running",
        "runs": {},
    }

    for query_set_name, query_map in query_sets.items():
        for cfg in selected_configs:
            run_id = f"{query_set_name}__{cfg.name}"
            run_dir = runs_root / run_id
            query_dir = run_dir / "query_results"
            query_dir.mkdir(parents=True, exist_ok=True)

            if cfg.use_semantic:
                collection_name = args.collection_with_semantic
            else:
                collection_name = args.collection_without_semantic

            collection = None
            collection_count = 0
            if not args.dry_run:
                collection = open_collection(chroma_path, collection_name)
                collection_count = int(collection.count())
                if args.require_collection_nonempty and collection_count <= 0:
                    message = (
                        f"Collection '{collection_name}' is empty (count=0). "
                        "Run offline indexing first or disable --require-collection-nonempty."
                    )
                    global_summary["status"] = "failed"
                    global_summary["error"] = message
                    safe_json_dump(artifacts_dir / "online_run_summary.json", global_summary)
                    raise RuntimeError(message)

            run_started = perf_counter()
            run_results: dict[str, dict[str, Any]] = {}
            success = 0
            failed = 0

            ordered_queries = sorted(query_map.items(), key=lambda x: x[0])
            total = len(ordered_queries)
            print(
                (
                    f"[run:{run_id}] starting {total} queries | "
                    f"semantic={cfg.use_semantic} | query_opt={cfg.use_query_optimization} | "
                    f"collection={collection_name} | collection_count={collection_count}"
                ),
                flush=True,
            )

            for idx, (qid, query_text) in enumerate(ordered_queries, start=1):
                query_out = query_dir / f"{qid}.json"
                if args.resume and query_out.exists():
                    with query_out.open("r", encoding="utf-8") as fh:
                        existing = json.load(fh)
                    if str(existing.get("status", "")).lower() == "ok":
                        run_results[qid] = existing
                        success += 1
                        continue
                t0 = perf_counter()
                try:
                    result = process_single_query(
                        query_id=qid,
                        query_text=query_text,
                        run_config=cfg,
                        collection=collection,
                        llm_model=args.llm_model,
                        n_results_per_subquery=args.n_results_per_subquery,
                        final_topk=args.final_topk,
                        max_data_profile_chars=args.max_data_profile_chars,
                        max_semantic_profile_chars=args.max_semantic_profile_chars,
                        dry_run=args.dry_run,
                    )
                    result["status"] = "ok"
                    result["error"] = ""
                    success += 1
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "query_id": qid,
                        "query_text": query_text,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "processed_at_utc": utc_now(),
                        "subqueries": [],
                        "retrieval_by_subquery": [],
                        "final_ranking": [],
                    }
                    failed += 1
                result["elapsed_seconds"] = round(perf_counter() - t0, 3)
                safe_json_dump(query_out, result)
                run_results[qid] = result

                elapsed = perf_counter() - run_started
                rate = idx / elapsed if elapsed > 0 else 0.0
                eta = (total - idx) / rate if rate > 0 else float("inf")
                print(
                    (
                        f"[run:{run_id}] {idx}/{total} | ok={success} fail={failed} | "
                        f"elapsed={format_duration(elapsed)} | eta={format_duration(eta)}"
                    ),
                    flush=True,
                )

            diagnostics = compute_run_diagnostics(run_results)
            run_trec_path, run_trec_lines = write_run_trec(run_dir, run_id, run_results, args.final_topk)
            run_summary = {
                "run_id": run_id,
                "query_set": query_set_name,
                "config_name": cfg.name,
                "use_semantic": cfg.use_semantic,
                "use_query_optimization": cfg.use_query_optimization,
                "collection_name": collection_name,
                "collection_count": collection_count,
                "queries_total": len(query_map),
                "queries_ok": success,
                "queries_failed": failed,
                "retrieved_candidates_total": diagnostics["retrieved_candidates_total"],
                "ranked_docs_total": diagnostics["ranked_docs_total"],
                "queries_with_any_hits": diagnostics["queries_with_any_hits"],
                "queries_with_ranking": diagnostics["queries_with_ranking"],
                "run_trec_lines": run_trec_lines,
                "run_trec_path": str(run_trec_path),
                "started_at_utc": utc_now(),
                "completed_in_seconds": round(perf_counter() - run_started, 3),
            }
            safe_json_dump(run_dir / "run_summary.json", run_summary)
            global_summary["runs"][run_id] = run_summary
            safe_json_dump(artifacts_dir / "online_run_summary.json", global_summary)
            print(
                (
                    f"[run:{run_id}] summary | candidates={diagnostics['retrieved_candidates_total']} | "
                    f"ranked={diagnostics['ranked_docs_total']} | run_lines={run_trec_lines}"
                ),
                flush=True,
            )

            if args.fail_if_empty_run and run_trec_lines == 0:
                message = (
                    f"Run '{run_id}' generated an empty run.trec. "
                    "Disable --dry-run or pass --no-fail-if-empty-run to allow empty output."
                )
                global_summary["status"] = "failed"
                global_summary["error"] = message
                safe_json_dump(artifacts_dir / "online_run_summary.json", global_summary)
                raise RuntimeError(message)
            print(
                f"[run:{run_id}] completed in {format_duration(perf_counter() - run_started)}",
                flush=True,
            )

    global_summary["run_finished_at_utc"] = utc_now()
    global_summary["status"] = "completed"
    safe_json_dump(artifacts_dir / "online_run_summary.json", global_summary)
    print("[online] all requested runs completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
