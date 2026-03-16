#!/usr/bin/env python3
"""Select high-signal tabular queries and emit transparent selection metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class QueryMetrics:
    query_id: str
    query_text: str
    judged_total: int
    judged_tabular: int
    judged_tabular_ratio: float
    rel_l1plus_total: int
    rel_l1plus_tabular: int
    rel_tabular_ratio: float


def load_topics(path: Path) -> list[tuple[str, str]]:
    topics: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh, delimiter="\t")
        for row in reader:
            if not row:
                continue
            query_id = str(row[0]).strip()
            query_text = str(row[1]).strip() if len(row) > 1 else ""
            if query_id:
                topics.append((query_id, query_text))
    return topics


def load_tabular_dataset_ids(path: Path) -> set[str]:
    dataset_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            content = line.strip()
            if not content:
                continue
            obj = json.loads(content)
            dataset_id = str(obj.get("id", "")).strip()
            if dataset_id:
                dataset_ids.add(dataset_id)
    return dataset_ids


def parse_qrels(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            rows.append((parts[0], parts[1], parts[2]))
    return rows


def label_to_level(label: str) -> int | None:
    label = label.strip().upper()
    if not label.startswith("L"):
        return None
    try:
        return int(label[1:])
    except ValueError:
        return None


def compute_query_metrics(
    topics: Iterable[tuple[str, str]],
    qrels_rows: Iterable[tuple[str, str, str]],
    tabular_dataset_ids: set[str],
    min_rel_level: int,
) -> tuple[list[QueryMetrics], dict[str, set[str]], dict[str, set[str]]]:
    judged_all: dict[str, set[str]] = defaultdict(set)
    judged_tabular: dict[str, set[str]] = defaultdict(set)
    rel_all: dict[str, set[str]] = defaultdict(set)
    rel_tabular: dict[str, set[str]] = defaultdict(set)

    for query_id, dataset_id, label in qrels_rows:
        judged_all[query_id].add(dataset_id)
        is_tabular = dataset_id in tabular_dataset_ids
        if is_tabular:
            judged_tabular[query_id].add(dataset_id)

        level = label_to_level(label)
        if level is None or level < min_rel_level:
            continue

        rel_all[query_id].add(dataset_id)
        if is_tabular:
            rel_tabular[query_id].add(dataset_id)

    metrics: list[QueryMetrics] = []
    for query_id, query_text in topics:
        judged_total = len(judged_all.get(query_id, set()))
        judged_tab = len(judged_tabular.get(query_id, set()))
        rel_total = len(rel_all.get(query_id, set()))
        rel_tab = len(rel_tabular.get(query_id, set()))
        judged_ratio = (judged_tab / judged_total) if judged_total else 0.0
        rel_ratio = (rel_tab / rel_total) if rel_total else 0.0
        metrics.append(
            QueryMetrics(
                query_id=query_id,
                query_text=query_text,
                judged_total=judged_total,
                judged_tabular=judged_tab,
                judged_tabular_ratio=judged_ratio,
                rel_l1plus_total=rel_total,
                rel_l1plus_tabular=rel_tab,
                rel_tabular_ratio=rel_ratio,
            )
        )
    return metrics, judged_tabular, rel_tabular


def select_queries_by_target(
    ranked_metrics: list[QueryMetrics],
    judged_tabular: dict[str, set[str]],
    rel_tabular: dict[str, set[str]],
    target_corpus_size: int,
) -> tuple[list[QueryMetrics], list[dict[str, int]]]:
    selected: list[QueryMetrics] = []
    trace: list[dict[str, int]] = []
    union_judged: set[str] = set()
    union_rel: set[str] = set()

    for metric in ranked_metrics:
        qid = metric.query_id
        judged_ids = judged_tabular.get(qid, set())
        rel_ids = rel_tabular.get(qid, set())
        marginal_judged = len(judged_ids - union_judged)
        marginal_rel = len(rel_ids - union_rel)

        selected.append(metric)
        union_judged |= judged_ids
        union_rel |= rel_ids

        trace.append(
            {
                "step": len(selected),
                "query_id": qid,
                "marginal_new_judged_tabular": marginal_judged,
                "cumulative_judged_tabular_union": len(union_judged),
                "marginal_new_rel_tabular": marginal_rel,
                "cumulative_rel_tabular_union": len(union_rel),
            }
        )
        if len(union_judged) >= target_corpus_size:
            break

    return selected, trace


def write_query_metrics(path: Path, ranked_metrics: list[QueryMetrics]) -> None:
    fieldnames = [
        "rank",
        "query_id",
        "query_text",
        "judged_total",
        "judged_tabular",
        "judged_tabular_ratio",
        "rel_l1plus_total",
        "rel_l1plus_tabular",
        "rel_tabular_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for idx, metric in enumerate(ranked_metrics, start=1):
            writer.writerow(
                {
                    "rank": idx,
                    "query_id": metric.query_id,
                    "query_text": metric.query_text,
                    "judged_total": metric.judged_total,
                    "judged_tabular": metric.judged_tabular,
                    "judged_tabular_ratio": f"{metric.judged_tabular_ratio:.6f}",
                    "rel_l1plus_total": metric.rel_l1plus_total,
                    "rel_l1plus_tabular": metric.rel_l1plus_tabular,
                    "rel_tabular_ratio": f"{metric.rel_tabular_ratio:.6f}",
                }
            )


def write_trace(path: Path, trace_rows: list[dict[str, int]]) -> None:
    fieldnames = [
        "step",
        "query_id",
        "marginal_new_judged_tabular",
        "cumulative_judged_tabular_union",
        "marginal_new_rel_tabular",
        "cumulative_rel_tabular_union",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in trace_rows:
            writer.writerow(row)


def write_selection_json(
    path: Path,
    selected: list[QueryMetrics],
    trace_rows: list[dict[str, int]],
    target_corpus_size: int,
    min_rel_level: int,
) -> None:
    achieved = trace_rows[-1]["cumulative_judged_tabular_union"] if trace_rows else 0
    payload = {
        "selection_method": "greedy_by_tabular_relevance",
        "target_corpus_size": target_corpus_size,
        "achieved_corpus_size": achieved,
        "min_rel_level": min_rel_level,
        "selected_queries": [m.query_id for m in selected],
        "selection_metrics_version": 1,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select NTCIR-15 queries by tabular relevance and emit selection metrics."
    )
    parser.add_argument("--topics", required=True, help="Topics TSV path.")
    parser.add_argument("--qrels", required=True, help="Qrels path.")
    parser.add_argument("--tabular-collection", required=True, help="Tabular JSONL collection path.")
    parser.add_argument("--target-corpus-size", type=int, default=250, help="Target tabular judged union size.")
    parser.add_argument("--min-rel-level", type=int, default=1, help="Minimum relevance level considered relevant.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--summary", action="store_true", help="Print summary.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    topics_path = Path(args.topics)
    qrels_path = Path(args.qrels)
    tabular_collection_path = Path(args.tabular_collection)
    out_dir = Path(args.out_dir)

    for path in (topics_path, qrels_path, tabular_collection_path):
        if not path.exists():
            parser.error(f"Input file not found: {path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    topics = load_topics(topics_path)
    tabular_ids = load_tabular_dataset_ids(tabular_collection_path)
    qrels_rows = parse_qrels(qrels_path)

    metrics, judged_tabular, rel_tabular = compute_query_metrics(
        topics=topics,
        qrels_rows=qrels_rows,
        tabular_dataset_ids=tabular_ids,
        min_rel_level=max(0, args.min_rel_level),
    )

    ranked = sorted(
        metrics,
        key=lambda m: (
            -m.rel_l1plus_tabular,
            -m.judged_tabular,
            -m.judged_tabular_ratio,
            m.query_id,
        ),
    )

    selected, trace_rows = select_queries_by_target(
        ranked_metrics=ranked,
        judged_tabular=judged_tabular,
        rel_tabular=rel_tabular,
        target_corpus_size=max(0, args.target_corpus_size),
    )

    query_metrics_path = out_dir / "query_metrics.csv"
    trace_path = out_dir / "selection_trace.csv"
    selection_path = out_dir / "selected_queries.json"

    write_query_metrics(query_metrics_path, ranked)
    write_trace(trace_path, trace_rows)
    write_selection_json(
        selection_path,
        selected=selected,
        trace_rows=trace_rows,
        target_corpus_size=max(0, args.target_corpus_size),
        min_rel_level=max(0, args.min_rel_level),
    )

    if args.summary:
        achieved = trace_rows[-1]["cumulative_judged_tabular_union"] if trace_rows else 0
        print(f"Topics parsed: {len(topics)}")
        print(f"Ranked queries: {len(ranked)}")
        print(f"Selected queries: {len(selected)}")
        print(f"Target corpus size: {args.target_corpus_size}")
        print(f"Achieved corpus size: {achieved}")
        print(f"Selection JSON: {selection_path}")
        print(f"Query metrics: {query_metrics_path}")
        print(f"Selection trace: {trace_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
