#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (SCRIPT_DIR / raw).resolve()
    return path


def safe_json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(str(text).lower())


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
            qid = parts[0].strip()
            query_text = parts[1].strip()
            if qid and query_text:
                topics[qid] = query_text
    return dict(sorted(topics.items(), key=lambda x: x[0]))


def load_complex_queries(path: Path, expected_qids: set[str]) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid complex cache format: {path}")

    missing: list[str] = []
    out: dict[str, str] = {}
    for qid in sorted(expected_qids):
        entry = payload.get(qid)
        if not isinstance(entry, dict):
            missing.append(qid)
            continue
        complex_text = str(entry.get("complex", "")).strip()
        if not complex_text:
            missing.append(qid)
            continue
        out[qid] = complex_text

    if missing:
        raise RuntimeError(
            "Complex cache missing queries: " + ", ".join(missing)
        )
    return out


@dataclass
class BM25Doc:
    dataset_id: str
    tf: Counter[str]
    length: int


class BM25Index:
    def __init__(self, docs: list[BM25Doc], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1 = float(k1)
        self.b = float(b)
        self.N = len(docs)
        self.avgdl = (
            sum(doc.length for doc in docs) / float(self.N) if self.N > 0 else 0.0
        )
        self.df: dict[str, int] = defaultdict(int)
        for doc in docs:
            for term in doc.tf.keys():
                self.df[term] += 1
        self.idf: dict[str, float] = {}
        for term, df in self.df.items():
            # BM25 Okapi with +1 smoothing for stable positive idf.
            self.idf[term] = math.log(1.0 + ((self.N - df + 0.5) / (df + 0.5)))

    def score(self, query_tokens: list[str]) -> list[tuple[str, float]]:
        if not self.docs or not query_tokens:
            return []
        query_terms = Counter(query_tokens)
        scores: list[tuple[str, float]] = []
        for doc in self.docs:
            dl = float(doc.length)
            if dl <= 0:
                continue
            denom_norm = self.k1 * (1.0 - self.b + self.b * (dl / max(self.avgdl, 1e-9)))
            score = 0.0
            for term, qtf in query_terms.items():
                tf = float(doc.tf.get(term, 0))
                if tf <= 0.0:
                    continue
                idf = self.idf.get(term, 0.0)
                term_score = idf * ((tf * (self.k1 + 1.0)) / (tf + denom_norm))
                score += term_score * float(qtf)
            if score > 0.0:
                scores.append((doc.dataset_id, score))
        return scores


def load_dataset_info(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def build_doc_text(row: dict[str, Any]) -> str:
    fields = [
        str(row.get("dataset_name", "")).strip(),
        str(row.get("domain", "")).strip(),
        str(row.get("data_profile", "")).strip(),
        str(row.get("semantic_profile", "")).strip(),
    ]
    return "\n".join(part for part in fields if part)


def build_index(dataset_rows: list[dict[str, Any]]) -> BM25Index:
    docs: list[BM25Doc] = []
    seen_ids: set[str] = set()
    for row in dataset_rows:
        did = str(row.get("dataset_id", "")).strip()
        if not did or did in seen_ids:
            continue
        text = build_doc_text(row)
        tokens = tokenize(text)
        if not tokens:
            continue
        docs.append(BM25Doc(dataset_id=did, tf=Counter(tokens), length=len(tokens)))
        seen_ids.add(did)
    return BM25Index(docs)


def rank_queries(
    index: BM25Index,
    queries: dict[str, str],
    *,
    topk: int,
) -> dict[str, list[tuple[str, float]]]:
    out: dict[str, list[tuple[str, float]]] = {}
    for qid, qtext in sorted(queries.items(), key=lambda x: x[0]):
        tokens = tokenize(qtext)
        scored = index.score(tokens)
        # Deterministic ordering by score desc then dataset_id asc.
        scored = sorted(scored, key=lambda x: (-x[1], x[0]))[: max(1, topk)]
        out[qid] = scored
    return out


def write_run_trec(
    run_dir: Path,
    run_id: str,
    ranking_by_qid: dict[str, list[tuple[str, float]]],
) -> tuple[Path, int]:
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "run.trec"
    lines: list[str] = []
    for qid in sorted(ranking_by_qid.keys()):
        rows = ranking_by_qid[qid]
        for idx, (did, score) in enumerate(rows, start=1):
            did = str(did).strip()
            if not did:
                continue
            lines.append(f"{qid} Q0 {did} {idx} {score:.6f} {run_id}")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out_path, len(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run lexical BM25 baseline for NTCIR-15 online phase."
    )
    parser.add_argument("--topics-path", default="../evaluation_ready_pack/topics.tsv")
    parser.add_argument("--complex-cache-path", default="./artifacts/complex_queries.json")
    parser.add_argument(
        "--dataset-info-path",
        default="../offline_phase/artifacts/dataset_info_evaluation.jsonl",
    )
    parser.add_argument("--artifacts-dir", default="./artifacts_live")
    parser.add_argument("--query-set", choices=["original", "complex", "both"], default="both")
    parser.add_argument("--topk", type=int, default=10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    topics_path = resolve_path(args.topics_path)
    complex_cache_path = resolve_path(args.complex_cache_path)
    dataset_info_path = resolve_path(args.dataset_info_path)
    artifacts_dir = resolve_path(args.artifacts_dir)

    if not topics_path.exists():
        raise FileNotFoundError(f"topics-path not found: {topics_path}")
    if not dataset_info_path.exists():
        raise FileNotFoundError(f"dataset-info-path not found: {dataset_info_path}")
    if args.query_set in {"complex", "both"} and not complex_cache_path.exists():
        raise FileNotFoundError(f"complex-cache-path not found: {complex_cache_path}")

    topics = load_topics(topics_path)
    if not topics:
        raise RuntimeError(f"No topics found in {topics_path}")

    dataset_rows = load_dataset_info(dataset_info_path)
    if not dataset_rows:
        raise RuntimeError(f"No dataset rows found in {dataset_info_path}")

    index = build_index(dataset_rows)
    if index.N <= 0:
        raise RuntimeError("BM25 index is empty after parsing dataset_info.")

    query_sets: dict[str, dict[str, str]] = {}
    if args.query_set in {"original", "both"}:
        query_sets["original"] = topics
    if args.query_set in {"complex", "both"}:
        query_sets["complex"] = load_complex_queries(complex_cache_path, set(topics.keys()))

    runs_root = artifacts_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    global_summary: dict[str, Any] = {
        "run_started_at_utc": utc_now(),
        "topics_path": str(topics_path),
        "complex_cache_path": str(complex_cache_path),
        "dataset_info_path": str(dataset_info_path),
        "artifacts_dir": str(artifacts_dir),
        "query_set": args.query_set,
        "topk": int(args.topk),
        "bm25": {"k1": 1.5, "b": 0.75},
        "corpus_size": int(index.N),
        "runs": {},
        "status": "running",
    }

    for set_name, queries in query_sets.items():
        run_id = f"{set_name}__bm25"
        run_dir = runs_root / run_id
        ranking = rank_queries(index, queries, topk=args.topk)
        run_path, line_count = write_run_trec(run_dir, run_id, ranking)

        run_summary = {
            "run_id": run_id,
            "query_set": set_name,
            "queries_total": len(queries),
            "corpus_size": int(index.N),
            "run_trec_lines": int(line_count),
            "run_trec_path": str(run_path),
            "bm25": {"k1": 1.5, "b": 0.75, "tokenizer": "lowercase_alnum_regex"},
            "topk": int(args.topk),
            "dataset_info_path": str(dataset_info_path),
            "topics_path": str(topics_path),
            "complex_cache_path": str(complex_cache_path),
            "generated_at_utc": utc_now(),
        }
        safe_json_dump(run_dir / "run_summary.json", run_summary)
        global_summary["runs"][run_id] = run_summary
        print(
            f"[bm25:{run_id}] queries={len(queries)} corpus={index.N} run_lines={line_count}",
            flush=True,
        )

    global_summary["run_finished_at_utc"] = utc_now()
    global_summary["status"] = "completed"
    safe_json_dump(artifacts_dir / "bm25_run_summary.json", global_summary)
    print("[bm25] completed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
