#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
METRIC_FIELDS = ("map", "ndcg_cut_10", "recall_10", "precision_10")


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (SCRIPT_DIR / raw).resolve()
    return path


def parse_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split()
            if len(parts) < 3:
                continue

            # Supported:
            # - qid dataset_id L0/L1/L2
            # - qid 0 dataset_id 0/1/2
            if len(parts) >= 4 and (parts[2].startswith("L") or parts[3].isdigit() or parts[3].startswith("L")):
                qid = parts[0]
                did = parts[2]
                raw_rel = parts[3]
            else:
                qid = parts[0]
                did = parts[1]
                raw_rel = parts[2]

            if raw_rel.startswith("L") and raw_rel[1:].isdigit():
                rel = int(raw_rel[1:])
            elif raw_rel.isdigit():
                rel = int(raw_rel)
            else:
                rel = 0
            qrels.setdefault(qid, {})[did] = rel
    return qrels


def parse_run(path: Path) -> dict[str, list[tuple[str, float, int]]]:
    run: dict[str, list[tuple[str, float, int]]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split()
            if len(parts) < 6:
                continue
            qid = parts[0]
            did = parts[2]
            try:
                rank = int(parts[3])
            except Exception:  # noqa: BLE001
                rank = 999999
            try:
                score = float(parts[4])
            except Exception:  # noqa: BLE001
                score = 0.0
            run.setdefault(qid, []).append((did, score, rank))
    for qid, rows in run.items():
        run[qid] = sorted(rows, key=lambda x: (x[2], -x[1]))
    return run


def count_nonempty_lines(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def dcg(rels: list[int], k: int) -> float:
    total = 0.0
    for idx, rel in enumerate(rels[:k], start=1):
        gain = (2**rel - 1)
        total += gain / math.log2(idx + 1)
    return total


def ndcg_at_k(pred_docs: list[str], qrel_map: dict[str, int], k: int) -> float:
    pred_rels = [qrel_map.get(d, 0) for d in pred_docs[:k]]
    ideal = sorted(qrel_map.values(), reverse=True)
    idcg = dcg(ideal, k)
    if idcg <= 0:
        return 0.0
    return dcg(pred_rels, k) / idcg


def precision_at_k(pred_docs: list[str], qrel_map: dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    hit = sum(1 for d in pred_docs[:k] if qrel_map.get(d, 0) > 0)
    return hit / float(k)


def recall_at_k(pred_docs: list[str], qrel_map: dict[str, int], k: int) -> float:
    total_rel = sum(1 for _, rel in qrel_map.items() if rel > 0)
    if total_rel <= 0:
        return 0.0
    hit = sum(1 for d in pred_docs[:k] if qrel_map.get(d, 0) > 0)
    return hit / float(total_rel)


def average_precision(pred_docs: list[str], qrel_map: dict[str, int]) -> float:
    total_rel = sum(1 for _, rel in qrel_map.items() if rel > 0)
    if total_rel <= 0:
        return 0.0
    hit = 0
    acc = 0.0
    for idx, did in enumerate(pred_docs, start=1):
        if qrel_map.get(did, 0) > 0:
            hit += 1
            acc += hit / float(idx)
    return acc / float(total_rel)


def score_query(
    qrel_map: dict[str, int],
    docs: list[str],
    k: int,
) -> dict[str, float]:
    return {
        "map": average_precision(docs, qrel_map),
        "ndcg_cut_10": ndcg_at_k(docs, qrel_map, k),
        "recall_10": recall_at_k(docs, qrel_map, k),
        "precision_10": precision_at_k(docs, qrel_map, k),
    }


def per_query_manual_scores(
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[tuple[str, float, int]]],
    k: int,
) -> dict[str, dict[str, float]]:
    query_scores: dict[str, dict[str, float]] = {}
    for qid in sorted(qrels.keys()):
        qrel_map = qrels.get(qid, {})
        rows = run.get(qid, [])
        docs = [did for did, _, _ in rows]
        query_scores[qid] = score_query(qrel_map, docs, k)
    return query_scores


def aggregate_query_scores(query_scores: dict[str, dict[str, float]]) -> dict[str, float]:
    if not query_scores:
        return {metric: 0.0 for metric in METRIC_FIELDS}

    qids = sorted(query_scores.keys())
    n = float(len(qids))
    return {
        metric: sum(float(query_scores[qid].get(metric, 0.0)) for qid in qids) / n
        for metric in METRIC_FIELDS
    }


def evaluate_manual(
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[tuple[str, float, int]]],
    k: int,
) -> dict[str, float]:
    return aggregate_query_scores(per_query_manual_scores(qrels, run, k))


def evaluate_with_pytrec(
    qrels: dict[str, dict[str, int]],
    run: dict[str, list[tuple[str, float, int]]],
) -> dict[str, float] | None:
    try:
        import pytrec_eval
    except Exception:  # noqa: BLE001
        return None

    run_for_eval: dict[str, dict[str, float]] = {}
    for qid, rows in run.items():
        run_for_eval[qid] = {}
        for did, score, _rank in rows:
            prev = run_for_eval[qid].get(did)
            if prev is None or score > prev:
                run_for_eval[qid][did] = float(score)

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        {"map", "ndcg_cut_10", "recall_10", "P_10"},
    )
    query_scores = evaluator.evaluate(run_for_eval)
    if not query_scores:
        return None

    n = float(len(query_scores))
    agg = {
        "map": 0.0,
        "ndcg_cut_10": 0.0,
        "recall_10": 0.0,
        "precision_10": 0.0,
    }
    for qid, scores in query_scores.items():
        agg["map"] += float(scores.get("map", 0.0))
        agg["ndcg_cut_10"] += float(scores.get("ndcg_cut_10", 0.0))
        agg["recall_10"] += float(scores.get("recall_10", 0.0))
        agg["precision_10"] += float(scores.get("P_10", 0.0))

    return {k: v / n for k, v in agg.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate online-phase run files against qrels.")
    parser.add_argument("--qrels-path", default="../evaluation_ready_pack/qrels_filtered_unprocessable.txt")
    parser.add_argument("--runs-dir", default="./artifacts/runs")
    parser.add_argument("--output-dir", default="./artifacts/eval")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--fail-on-empty-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if any run.trec is empty.",
    )
    return parser


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

    empty_run_files: list[str] = []
    run_line_count: dict[str, int] = {}
    for run_file in run_files:
        lines = count_nonempty_lines(run_file)
        run_line_count[str(run_file)] = lines
        if lines == 0:
            empty_run_files.append(str(run_file))
    if args.fail_on_empty_run and empty_run_files:
        listed = "\n".join(f"- {p}" for p in empty_run_files)
        raise RuntimeError(
            "Found empty run.trec files. Re-run online phase without --dry-run "
            "or pass --no-fail-on-empty-run.\n"
            f"{listed}"
        )

    rows: list[dict[str, Any]] = []
    for run_file in run_files:
        run_name = run_file.parent.name
        run_lines = run_line_count[str(run_file)]
        run = parse_run(run_file)
        scores = evaluate_with_pytrec(qrels, run)
        source = "pytrec_eval"
        if scores is None:
            scores = evaluate_manual(qrels, run, args.k)
            source = "manual"

        row = {
            "run_name": run_name,
            "run_file": str(run_file),
            "run_lines": run_lines,
            "empty_run": run_lines == 0,
            "metric_source": source,
            "map": round(float(scores["map"]), 6),
            "ndcg_cut_10": round(float(scores["ndcg_cut_10"]), 6),
            "recall_10": round(float(scores["recall_10"]), 6),
            "precision_10": round(float(scores["precision_10"]), 6),
        }
        rows.append(row)

    rows = sorted(rows, key=lambda x: (-x["ndcg_cut_10"], -x["map"], x["run_name"]))

    csv_path = output_dir / "evaluation_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "run_name",
                "run_file",
                "run_lines",
                "empty_run",
                "metric_source",
                "map",
                "ndcg_cut_10",
                "recall_10",
                "precision_10",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "evaluation_summary.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "qrels_path": str(qrels_path),
                "runs_dir": str(runs_dir),
                "qrels_queries_count": len(qrels),
                "empty_run_files": empty_run_files,
                "runs_with_any_predictions": sum(1 for x in rows if int(x["run_lines"]) > 0),
                "rows": rows,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[eval] processed {len(rows)} run files", flush=True)
    print(f"[eval] csv: {csv_path}", flush=True)
    print(f"[eval] json: {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
