#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

from evaluate_runs import (
    METRIC_FIELDS,
    count_nonempty_lines,
    parse_qrels,
    parse_run,
    per_query_manual_scores,
    resolve_path,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute bootstrap confidence intervals and pairwise significance from saved run files."
    )
    parser.add_argument("--qrels-path", default="../evaluation_ready_pack/qrels_filtered_unprocessable.txt")
    parser.add_argument("--runs-dir", default="./artifacts_live/runs")
    parser.add_argument("--output-dir", default="./artifacts_live/eval_bootstrap")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--fail-on-empty-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if any run.trec is empty.",
    )
    return parser


def generate_bootstrap_indices(n_queries: int, n_samples: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(n_queries) for _ in range(n_queries)] for _ in range(n_samples)]


def percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    ordered = sorted(float(v) for v in values)
    pos = probability * (len(ordered) - 1)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    weight = pos - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def bootstrap_means(values: list[float], bootstrap_indices: list[list[int]]) -> list[float]:
    n_queries = len(values)
    if n_queries == 0:
        return []

    means: list[float] = []
    denom = float(n_queries)
    for indices in bootstrap_indices:
        total = 0.0
        for idx in indices:
            total += values[idx]
        means.append(total / denom)
    return means


def summarize_metric(values: list[float], bootstrap_indices: list[list[int]]) -> dict[str, float]:
    point_estimate = sum(values) / float(len(values)) if values else 0.0
    query_std = statistics.stdev(values) if len(values) > 1 else 0.0
    boot_means = bootstrap_means(values, bootstrap_indices)
    return {
        "point_estimate": point_estimate,
        "query_std": query_std,
        "ci_low": percentile(boot_means, 0.025),
        "ci_high": percentile(boot_means, 0.975),
    }


def compare_metric(
    run_a_values: list[float],
    run_b_values: list[float],
    bootstrap_indices: list[list[int]],
) -> dict[str, float]:
    diffs = [a - b for a, b in zip(run_a_values, run_b_values)]
    observed_delta = sum(diffs) / float(len(diffs)) if diffs else 0.0
    boot_deltas = bootstrap_means(diffs, bootstrap_indices)
    prob_le_zero = sum(1 for delta in boot_deltas if delta <= 0.0) / float(len(boot_deltas) or 1)
    prob_ge_zero = sum(1 for delta in boot_deltas if delta >= 0.0) / float(len(boot_deltas) or 1)
    p_value = min(1.0, 2.0 * min(prob_le_zero, prob_ge_zero))
    return {
        "observed_delta": observed_delta,
        "delta_ci_low": percentile(boot_deltas, 0.025),
        "delta_ci_high": percentile(boot_deltas, 0.975),
        "p_value": p_value,
    }


def adjust_pvalues_bh(p_values: list[float]) -> list[float]:
    n = len(p_values)
    if n == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [1.0] * n
    running_min = 1.0
    for rank in range(n, 0, -1):
        idx, value = indexed[rank - 1]
        candidate = min(1.0, value * n / float(rank))
        running_min = min(running_min, candidate)
        adjusted[idx] = running_min
    return adjusted


def round_float(value: float) -> float:
    return round(float(value), 6)


def collect_run_data(
    qrels: dict[str, dict[str, int]],
    run_files: list[Path],
    k: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[float]]], list[str]]:
    query_ids = sorted(qrels.keys())
    run_rows: list[dict[str, Any]] = []
    metric_vectors: dict[str, dict[str, list[float]]] = {}

    for run_file in run_files:
        run_name = run_file.parent.name
        run = parse_run(run_file)
        query_scores = per_query_manual_scores(qrels, run, k)
        metric_vectors[run_name] = {
            metric: [float(query_scores[qid][metric]) for qid in query_ids]
            for metric in METRIC_FIELDS
        }

        for qid in query_ids:
            score_row = {"run_name": run_name, "run_file": str(run_file), "query_id": qid}
            for metric in METRIC_FIELDS:
                score_row[metric] = round_float(query_scores[qid][metric])
            run_rows.append(score_row)
    return run_rows, metric_vectors, query_ids


def build_summary_rows(
    run_files: list[Path],
    metric_vectors: dict[str, dict[str, list[float]]],
    bootstrap_indices: list[list[int]],
    query_count: int,
    bootstrap_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    for run_file in run_files:
        run_name = run_file.parent.name
        row: dict[str, Any] = {
            "run_name": run_name,
            "run_file": str(run_file),
            "query_count": query_count,
            "bootstrap_samples": bootstrap_samples,
            "seed": seed,
        }
        for metric in METRIC_FIELDS:
            stats = summarize_metric(metric_vectors[run_name][metric], bootstrap_indices)
            row[f"{metric}_point_estimate"] = round_float(stats["point_estimate"])
            row[f"{metric}_query_std"] = round_float(stats["query_std"])
            row[f"{metric}_ci_low"] = round_float(stats["ci_low"])
            row[f"{metric}_ci_high"] = round_float(stats["ci_high"])
        summary_rows.append(row)
    return sorted(summary_rows, key=lambda item: item["run_name"])


def build_pairwise_rows(
    run_files: list[Path],
    metric_vectors: dict[str, dict[str, list[float]]],
    bootstrap_indices: list[list[int]],
    query_count: int,
    alpha: float,
) -> list[dict[str, Any]]:
    pairwise_rows: list[dict[str, Any]] = []
    run_names = [run_file.parent.name for run_file in run_files]

    for run_a, run_b in combinations(run_names, 2):
        for metric in METRIC_FIELDS:
            comparison = compare_metric(
                metric_vectors[run_a][metric],
                metric_vectors[run_b][metric],
                bootstrap_indices,
            )
            observed_delta = comparison["observed_delta"]
            if observed_delta > 0:
                better_run = run_a
            elif observed_delta < 0:
                better_run = run_b
            else:
                better_run = "tie"

            pairwise_rows.append(
                {
                    "metric": metric,
                    "run_a": run_a,
                    "run_b": run_b,
                    "query_count": query_count,
                    "observed_delta": round_float(observed_delta),
                    "delta_ci_low": round_float(comparison["delta_ci_low"]),
                    "delta_ci_high": round_float(comparison["delta_ci_high"]),
                    "p_value": round_float(comparison["p_value"]),
                    "better_run": better_run,
                    "significant_raw": comparison["p_value"] <= alpha,
                    "_raw_p_value": float(comparison["p_value"]),
                }
            )

    for metric in METRIC_FIELDS:
        metric_rows = [row for row in pairwise_rows if row["metric"] == metric]
        adjusted = adjust_pvalues_bh([float(row["_raw_p_value"]) for row in metric_rows])
        for row, adjusted_p in zip(metric_rows, adjusted):
            row["p_value_bh"] = round_float(adjusted_p)
            row["significant_bh"] = adjusted_p <= alpha
            del row["_raw_p_value"]

    return sorted(pairwise_rows, key=lambda item: (item["metric"], item["run_a"], item["run_b"]))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = build_parser().parse_args()
    qrels_path = resolve_path(args.qrels_path)
    runs_dir = resolve_path(args.runs_dir)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not qrels_path.exists():
        raise FileNotFoundError(f"qrels not found: {qrels_path}")
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs dir not found: {runs_dir}")

    qrels = parse_qrels(qrels_path)
    if not qrels:
        raise RuntimeError(f"No qrels parsed from {qrels_path}. Check qrels format/path.")

    run_files = sorted(runs_dir.rglob("run.trec"))
    if not run_files:
        raise RuntimeError(f"No run.trec files found under {runs_dir}")

    empty_run_files = [str(run_file) for run_file in run_files if count_nonempty_lines(run_file) == 0]
    if args.fail_on_empty_run and empty_run_files:
        listed = "\n".join(f"- {path}" for path in empty_run_files)
        raise RuntimeError(
            "Found empty run.trec files. Re-run online phase without --dry-run "
            "or pass --no-fail-on-empty-run.\n"
            f"{listed}"
        )

    per_query_rows, metric_vectors, query_ids = collect_run_data(qrels, run_files, args.k)
    bootstrap_indices = generate_bootstrap_indices(len(query_ids), args.bootstrap_samples, args.seed)
    summary_rows = build_summary_rows(
        run_files,
        metric_vectors,
        bootstrap_indices,
        len(query_ids),
        args.bootstrap_samples,
        args.seed,
    )
    pairwise_rows = build_pairwise_rows(
        run_files,
        metric_vectors,
        bootstrap_indices,
        len(query_ids),
        args.alpha,
    )

    per_query_csv = output_dir / "per_query_metrics.csv"
    summary_csv = output_dir / "bootstrap_summary.csv"
    summary_json = output_dir / "bootstrap_summary.json"
    pairwise_csv = output_dir / "pairwise_significance.csv"
    pairwise_json = output_dir / "pairwise_significance.json"

    write_csv(
        per_query_csv,
        per_query_rows,
        ["run_name", "run_file", "query_id", *METRIC_FIELDS],
    )

    summary_fieldnames = [
        "run_name",
        "run_file",
        "query_count",
        "bootstrap_samples",
        "seed",
    ]
    for metric in METRIC_FIELDS:
        summary_fieldnames.extend(
            [
                f"{metric}_point_estimate",
                f"{metric}_query_std",
                f"{metric}_ci_low",
                f"{metric}_ci_high",
            ]
        )
    write_csv(summary_csv, summary_rows, summary_fieldnames)

    with summary_json.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "qrels_path": str(qrels_path),
                "runs_dir": str(runs_dir),
                "output_dir": str(output_dir),
                "query_count": len(query_ids),
                "bootstrap_samples": args.bootstrap_samples,
                "seed": args.seed,
                "alpha": args.alpha,
                "rows": summary_rows,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    write_csv(
        pairwise_csv,
        pairwise_rows,
        [
            "metric",
            "run_a",
            "run_b",
            "query_count",
            "observed_delta",
            "delta_ci_low",
            "delta_ci_high",
            "p_value",
            "p_value_bh",
            "better_run",
            "significant_raw",
            "significant_bh",
        ],
    )

    with pairwise_json.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "qrels_path": str(qrels_path),
                "runs_dir": str(runs_dir),
                "output_dir": str(output_dir),
                "query_count": len(query_ids),
                "bootstrap_samples": args.bootstrap_samples,
                "seed": args.seed,
                "alpha": args.alpha,
                "rows": pairwise_rows,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[bootstrap] processed {len(run_files)} runs across {len(query_ids)} queries", flush=True)
    print(f"[bootstrap] per-query csv: {per_query_csv}", flush=True)
    print(f"[bootstrap] summary csv: {summary_csv}", flush=True)
    print(f"[bootstrap] summary json: {summary_json}", flush=True)
    print(f"[bootstrap] pairwise csv: {pairwise_csv}", flush=True)
    print(f"[bootstrap] pairwise json: {pairwise_json}", flush=True)
    print(
        "[bootstrap] warning: NTCIR-15 uses only 10 queries here, so confidence intervals may be wide and "
        "significance tests may be underpowered.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
