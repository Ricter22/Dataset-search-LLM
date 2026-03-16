#!/usr/bin/env python3
"""Build a tabular subset from selected queries and download selected resources."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from filter_convertible_datasets import detect_resource_format


FORMAT_PRIORITY = {
    "csv": 0,
    "tsv": 1,
    "parquet": 2,
    "json": 3,
    "excel": 4,
    "feather": 5,
    "orc": 6,
    "hdf": 7,
    "stata": 8,
    "sas": 9,
    "spss": 10,
    "sqlite": 11,
}


def load_selection(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    selected = payload.get("selected_queries")
    if not isinstance(selected, list):
        raise ValueError("selected_queries must be a list in selection JSON.")
    out: list[str] = []
    for qid in selected:
        q = str(qid).strip()
        if q:
            out.append(q)
    return out


def parse_qrels(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            rows.append((parts[0], parts[1], parts[2]))
    return rows


def dataset_ids_for_queries(qrels_rows: list[tuple[str, str, str]], selected_queries: set[str]) -> set[str]:
    by_query: dict[str, set[str]] = defaultdict(set)
    for query_id, dataset_id, _ in qrels_rows:
        if query_id in selected_queries:
            by_query[query_id].add(dataset_id)
    merged: set[str] = set()
    for dataset_ids in by_query.values():
        merged |= dataset_ids
    return merged


def load_tabular_collection(path: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            content = line.strip()
            if not content:
                continue
            obj = json.loads(content)
            dataset_id = str(obj.get("id", "")).strip()
            if dataset_id:
                by_id[dataset_id] = obj
    return by_id


def choose_one_resource(resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for idx, resource in enumerate(resources):
        detected = detect_resource_format(resource) or ""
        fmt_rank = FORMAT_PRIORITY.get(detected, 10_000)
        scored.append((fmt_rank, idx, resource))
    if not scored:
        return []
    scored.sort(key=lambda t: (t[0], t[1]))
    return [scored[0][2]]


def filter_resources(
    resources: list[Any], resource_policy: str, resource_cap: int
) -> list[dict[str, Any]]:
    typed = [r for r in resources if isinstance(r, dict)]
    if not typed:
        return []
    if resource_policy == "all":
        return typed
    if resource_policy == "one":
        return choose_one_resource(typed)
    # cap policy
    cap = max(1, resource_cap)
    return typed[:cap]


def build_subset_collection(
    selected_dataset_ids: set[str],
    collection_by_id: dict[str, dict[str, Any]],
    resource_policy: str,
    resource_cap: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    subset_rows: list[dict[str, Any]] = []
    resource_formats: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    missing_ids = 0

    for dataset_id in sorted(selected_dataset_ids):
        obj = collection_by_id.get(dataset_id)
        if obj is None:
            missing_ids += 1
            continue
        resources = obj.get("data")
        if not isinstance(resources, list):
            continue
        kept = filter_resources(resources, resource_policy=resource_policy, resource_cap=resource_cap)
        if not kept:
            continue
        cloned = dict(obj)
        cloned["data"] = kept
        subset_rows.append(cloned)

        for resource in kept:
            fmt = detect_resource_format(resource) or "unknown"
            resource_formats[fmt] += 1
            url = str(resource.get("data_url", "")).strip()
            if url:
                domain = url.split("/")[2] if "://" in url and len(url.split("/")) > 2 else "unknown-domain"
                domains[domain.lower()] += 1

    report = {
        "selected_dataset_ids_requested": len(selected_dataset_ids),
        "datasets_missing_from_tabular_collection": missing_ids,
        "subset_dataset_count": len(subset_rows),
        "subset_resource_count": sum(len(r.get("data", [])) for r in subset_rows),
        "resource_policy": resource_policy,
        "resource_cap": resource_cap if resource_policy == "cap" else None,
        "resource_format_counts": dict(resource_formats),
        "top_domains": domains.most_common(20),
    }
    return subset_rows, report


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_download(
    downloader_path: Path,
    subset_jsonl: Path,
    out_dir: Path,
    manifest: Path,
    workers: int,
    retries: int,
    timeout: float,
    domain_delay_ms: int,
    limit: int | None,
    overwrite: bool,
    no_resume: bool,
    summary: bool,
) -> None:
    cmd = [
        sys.executable,
        str(downloader_path),
        "--input",
        str(subset_jsonl),
        "--out-dir",
        str(out_dir),
        "--manifest",
        str(manifest),
        "--workers",
        str(max(1, workers)),
        "--retries",
        str(max(1, retries)),
        "--timeout",
        str(max(1.0, timeout)),
        "--domain-delay-ms",
        str(max(0, domain_delay_ms)),
    ]
    if limit is not None:
        cmd.extend(["--limit", str(max(1, limit))])
    if overwrite:
        cmd.append("--overwrite")
    if no_resume:
        cmd.append("--no-resume")
    if summary:
        cmd.append("--summary")
    subprocess.run(cmd, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a tabular subset collection from selected queries and optionally "
            "download resources with existing downloader."
        )
    )
    parser.add_argument("--selection-json", required=True, help="selected_queries.json path.")
    parser.add_argument("--qrels", required=True, help="Qrels path.")
    parser.add_argument("--tabular-collection", required=True, help="Tabular collection JSONL path.")
    parser.add_argument("--out-jsonl", required=True, help="Subset collection JSONL output path.")
    parser.add_argument("--subset-dataset-ids", required=True, help="Output TXT of dataset IDs in subset.")
    parser.add_argument("--report-json", required=True, help="Output subset report JSON path.")
    parser.add_argument(
        "--resource-policy",
        choices=["one", "all", "cap"],
        default="one",
        help="Resource selection policy per dataset (default: one).",
    )
    parser.add_argument("--resource-cap", type=int, default=1, help="Cap for --resource-policy cap.")
    parser.add_argument("--download-out-dir", required=True, help="Directory for downloaded resources.")
    parser.add_argument("--manifest", required=True, help="Manifest CSV path for downloader.")
    parser.add_argument("--downloader-script", default="download_datasets.py", help="Path to downloader script.")
    parser.add_argument("--workers", type=int, default=12, help="Downloader workers.")
    parser.add_argument("--retries", type=int, default=4, help="Downloader retries.")
    parser.add_argument("--timeout", type=float, default=60.0, help="Downloader timeout seconds.")
    parser.add_argument("--domain-delay-ms", type=int, default=200, help="Downloader domain delay ms.")
    parser.add_argument("--download-limit", type=int, default=None, help="Optional downloader --limit.")
    parser.add_argument("--overwrite", action="store_true", help="Downloader overwrite flag.")
    parser.add_argument("--no-resume", action="store_true", help="Downloader no-resume flag.")
    parser.add_argument("--no-download", action="store_true", help="Build subset files only; skip download step.")
    parser.add_argument("--summary", action="store_true", help="Print summary.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    selection_path = Path(args.selection_json)
    qrels_path = Path(args.qrels)
    tabular_collection_path = Path(args.tabular_collection)
    out_jsonl = Path(args.out_jsonl)
    subset_ids_txt = Path(args.subset_dataset_ids)
    report_json = Path(args.report_json)
    downloader_script = Path(args.downloader_script)

    for path in (selection_path, qrels_path, tabular_collection_path):
        if not path.exists():
            parser.error(f"Input file not found: {path}")
    if not args.no_download and not downloader_script.exists():
        parser.error(f"Downloader script not found: {downloader_script}")

    selected_queries = set(load_selection(selection_path))
    if not selected_queries:
        parser.error("No selected queries in selection JSON.")

    qrels_rows = parse_qrels(qrels_path)
    requested_dataset_ids = dataset_ids_for_queries(qrels_rows, selected_queries)
    collection_by_id = load_tabular_collection(tabular_collection_path)
    tabular_ids = set(collection_by_id.keys())
    selected_dataset_ids = requested_dataset_ids & tabular_ids

    subset_rows, report = build_subset_collection(
        selected_dataset_ids=selected_dataset_ids,
        collection_by_id=collection_by_id,
        resource_policy=args.resource_policy,
        resource_cap=args.resource_cap,
    )

    write_jsonl(out_jsonl, subset_rows)
    subset_ids_txt.parent.mkdir(parents=True, exist_ok=True)
    subset_ids_from_written_rows = sorted(
        str(row.get("id", "")).strip() for row in subset_rows if str(row.get("id", "")).strip()
    )
    with subset_ids_txt.open("w", encoding="utf-8") as fh:
        for dataset_id in subset_ids_from_written_rows:
            fh.write(f"{dataset_id}\n")

    report["selected_queries"] = sorted(selected_queries)
    report["requested_dataset_ids_from_qrels"] = len(requested_dataset_ids)
    report["selected_dataset_ids_after_tabular_filter"] = len(selected_dataset_ids)
    report["selected_dataset_ids_written"] = len(subset_ids_from_written_rows)
    report["out_jsonl"] = str(out_jsonl)
    report["subset_dataset_ids_txt"] = str(subset_ids_txt)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    with report_json.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    if not args.no_download:
        run_download(
            downloader_path=downloader_script,
            subset_jsonl=out_jsonl,
            out_dir=Path(args.download_out_dir),
            manifest=Path(args.manifest),
            workers=args.workers,
            retries=args.retries,
            timeout=args.timeout,
            domain_delay_ms=args.domain_delay_ms,
            limit=args.download_limit,
            overwrite=args.overwrite,
            no_resume=args.no_resume,
            summary=args.summary,
        )

    if args.summary:
        print(f"Selected queries: {len(selected_queries)}")
        print(f"Requested dataset IDs from qrels: {len(requested_dataset_ids)}")
        print(f"Selected tabular dataset IDs: {len(selected_dataset_ids)}")
        print(f"Subset datasets written: {len(subset_rows)}")
        print(f"Subset JSONL: {out_jsonl}")
        print(f"Subset IDs TXT: {subset_ids_txt}")
        print(f"Report JSON: {report_json}")
        if args.no_download:
            print("Download step: skipped (--no-download)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
