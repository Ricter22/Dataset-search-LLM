#!/usr/bin/env python3
"""Rewrite qrels by excluding dataset IDs marked unprocessable in dataset info."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as fh:
        payload = pickle.load(fh)
    if not isinstance(payload, list):
        raise ValueError("dataset-info pickle must contain a list of records")
    out: list[dict[str, Any]] = []
    for item in payload:
        if isinstance(item, dict):
            out.append(item)
    return out


def collect_excluded_dataset_ids(records: list[dict[str, Any]]) -> set[str]:
    excluded: set[str] = set()
    for record in records:
        dataset_id = str(record.get("dataset_id", "")).strip()
        if not dataset_id:
            continue
        if record.get("processable") is False or str(record.get("status", "")).lower() == "excluded":
            excluded.add(dataset_id)
    return excluded


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rewrite qrels excluding unprocessable datasets.")
    parser.add_argument(
        "--dataset-info-path",
        default="./artifacts/dataset_info_evaluation.pkl",
        help="Path to dataset info pickle produced by offline pipeline.",
    )
    parser.add_argument(
        "--qrels-path",
        default="../evaluation_ready_pack/qrels.txt",
        help="Input qrels path.",
    )
    parser.add_argument(
        "--out-qrels",
        default="../evaluation_ready_pack/qrels_filtered_unprocessable.txt",
        help="Output filtered qrels path.",
    )
    parser.add_argument(
        "--report-json",
        default="./artifacts/qrels_exclusion_report.json",
        help="Output report path.",
    )
    parser.add_argument("--summary", action="store_true", help="Print a short summary.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dataset_info_path = Path(args.dataset_info_path).resolve()
    qrels_path = Path(args.qrels_path).resolve()
    out_qrels_path = Path(args.out_qrels).resolve()
    report_path = Path(args.report_json).resolve()

    if not dataset_info_path.exists():
        parser.error(f"dataset-info not found: {dataset_info_path}")
    if not qrels_path.exists():
        parser.error(f"qrels not found: {qrels_path}")

    records = load_records(dataset_info_path)
    excluded_ids = collect_excluded_dataset_ids(records)

    input_lines = 0
    kept_lines: list[str] = []
    removed_lines: list[str] = []
    removed_query_counts: Counter[str] = Counter()
    removed_dataset_counts: Counter[str] = Counter()

    with qrels_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            input_lines += 1
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            query_id, dataset_id = parts[0], parts[1]
            if dataset_id in excluded_ids:
                removed_lines.append(line)
                removed_query_counts[query_id] += 1
                removed_dataset_counts[dataset_id] += 1
            else:
                kept_lines.append(line)

    out_qrels_path.parent.mkdir(parents=True, exist_ok=True)
    with out_qrels_path.open("w", encoding="utf-8") as fh:
        for line in kept_lines:
            fh.write(f"{line}\n")

    report = {
        "input_qrels_path": str(qrels_path),
        "output_qrels_path": str(out_qrels_path),
        "dataset_info_path": str(dataset_info_path),
        "input_lines": input_lines,
        "kept_lines": len(kept_lines),
        "removed_lines": len(removed_lines),
        "excluded_dataset_ids_count": len(excluded_ids),
        "excluded_dataset_ids": sorted(excluded_ids),
        "affected_query_ids_count": len(removed_query_counts),
        "affected_query_ids": sorted(removed_query_counts),
        "removed_by_query": dict(sorted(removed_query_counts.items())),
        "removed_by_dataset_id": dict(sorted(removed_dataset_counts.items())),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    if args.summary:
        print(f"Excluded dataset IDs: {len(excluded_ids)}")
        print(f"Removed qrels lines: {len(removed_lines)}")
        print(f"Kept qrels lines: {len(kept_lines)}")
        print(f"Filtered qrels: {out_qrels_path}")
        print(f"Report: {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
