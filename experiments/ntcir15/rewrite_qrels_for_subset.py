#!/usr/bin/env python3
"""Rewrite qrels to selected queries and subset dataset IDs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_selected_queries(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    selected = payload.get("selected_queries")
    if not isinstance(selected, list):
        raise ValueError("selected_queries must be a list in selection JSON.")
    return {str(item).strip() for item in selected if str(item).strip()}


def load_dataset_ids(path: Path) -> set[str]:
    dataset_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            dataset_id = line.strip()
            if dataset_id:
                dataset_ids.add(dataset_id)
    return dataset_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rewrite qrels for selected queries and subset datasets.")
    parser.add_argument("--qrels", required=True, help="Input qrels path.")
    parser.add_argument("--selection-json", required=True, help="selected_queries.json path.")
    parser.add_argument("--subset-dataset-ids", required=True, help="Subset dataset IDs TXT path.")
    parser.add_argument("--out-qrels", required=True, help="Output qrels path.")
    parser.add_argument("--report-json", required=True, help="Output rewrite report JSON path.")
    parser.add_argument("--summary", action="store_true", help="Print summary.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    qrels_path = Path(args.qrels)
    selection_path = Path(args.selection_json)
    subset_ids_path = Path(args.subset_dataset_ids)
    out_qrels_path = Path(args.out_qrels)
    report_path = Path(args.report_json)

    for path in (qrels_path, selection_path, subset_ids_path):
        if not path.exists():
            parser.error(f"Input file not found: {path}")

    selected_queries = load_selected_queries(selection_path)
    subset_ids = load_dataset_ids(subset_ids_path)

    kept_lines: list[str] = []
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    input_lines = 0
    dropped_lines = 0

    with qrels_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            input_lines += 1
            stripped = line.strip()
            if not stripped:
                dropped_lines += 1
                continue
            parts = stripped.split()
            if len(parts) < 3:
                dropped_lines += 1
                continue
            query_id, dataset_id, label = parts[0], parts[1], parts[2]
            if query_id not in selected_queries or dataset_id not in subset_ids:
                dropped_lines += 1
                continue
            kept_lines.append(f"{query_id} {dataset_id} {label}")
            label_counts[query_id][label] += 1

    out_qrels_path.parent.mkdir(parents=True, exist_ok=True)
    with out_qrels_path.open("w", encoding="utf-8") as fh:
        for line in kept_lines:
            fh.write(f"{line}\n")

    report = {
        "input_lines": input_lines,
        "kept_lines": len(kept_lines),
        "dropped_lines": dropped_lines,
        "selected_queries_count": len(selected_queries),
        "subset_dataset_ids_count": len(subset_ids),
        "per_query_label_counts": {
            query_id: dict(counter) for query_id, counter in sorted(label_counts.items())
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    if args.summary:
        print(f"Input qrels lines: {input_lines}")
        print(f"Kept lines: {len(kept_lines)}")
        print(f"Dropped lines: {dropped_lines}")
        print(f"Output qrels: {out_qrels_path}")
        print(f"Report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
