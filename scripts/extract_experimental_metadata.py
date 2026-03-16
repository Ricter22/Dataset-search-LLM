from __future__ import annotations

import csv
import json
import re
import sqlite3
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "EXPERIMENT_METADATA.md"


@dataclass(frozen=True)
class Row:
    field: str
    target_value: str
    ntcir_value: str
    evidence_type: str
    source_path: str
    notes: str


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def escape_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def load_target_model() -> str:
    notebook_path = ROOT / "legacy" / "target" / "TARGET_test.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    pattern = re.compile(r'\bmodel\s*=\s*"([^"]+)"')
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        match = pattern.search(source)
        if match:
            return match.group(1)
    raise RuntimeError("TARGET model variable not found in TARGET_test.ipynb")


def parse_metrics_summary() -> dict[str, dict[str, str]]:
    path = ROOT / "experiments" / "target" / "target_recovery_results_full" / "metrics_summary.csv"
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return {row["run_id"]: row for row in reader}


def load_ntcir_run_summary() -> dict:
    path = ROOT / "experiments" / "ntcir15" / "offline_phase" / "artifacts" / "run_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_ntcir_online_live_summary() -> dict:
    path = ROOT / "experiments" / "ntcir15" / "online_phase" / "artifacts_live" / "online_run_summary.json"
    return json.loads(path.read_text(encoding="utf-8"))


def iter_query_results(run_name: str) -> Iterable[dict]:
    query_dir = (
        ROOT
        / "experiments"
        / "ntcir15"
        / "online_phase"
        / "artifacts_live"
        / "runs"
        / run_name
        / "query_results"
    )
    for path in sorted(query_dir.glob("*.json")):
        yield json.loads(path.read_text(encoding="utf-8"))


def load_ntcir_instruction_counts() -> tuple[list[int], list[int]]:
    path = ROOT / "experiments" / "ntcir15" / "offline_phase" / "artifacts" / "dataset_info_evaluation.jsonl"
    with_semantic: list[int] = []
    without_semantic: list[int] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if row.get("processable") is not True:
                continue
            with_block = row.get("instructions_with_semantic")
            without_block = row.get("instructions_without_semantic")
            if isinstance(with_block, dict):
                with_semantic.append(len(with_block.get("queries", [])))
            if isinstance(without_block, dict):
                without_semantic.append(len(without_block.get("queries", [])))
    return with_semantic, without_semantic


def extract_chroma_space(sqlite_path: Path, collection_name: str) -> str:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        row = conn.execute(
            "SELECT * FROM collections WHERE name = ? LIMIT 1",
            (collection_name,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise RuntimeError(f"Collection {collection_name!r} not found in {sqlite_path}")

    for value in row:
        if not isinstance(value, str):
            continue
        match = re.search(r'"space"\s*:\s*"([^"]+)"', value)
        if match:
            return match.group(1)
    raise RuntimeError(f'Unable to extract "space" from {sqlite_path} collection {collection_name}')


def format_seconds(seconds: float) -> str:
    return f"{seconds:.2f}s"


def parse_iso8601(value: str) -> datetime:
    return datetime.fromisoformat(value)


def compute_ntcir_latency_stats(live_summary: dict) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for run_name in ("original__full", "complex__full"):
        per_query = [float(row["elapsed_seconds"]) for row in iter_query_results(run_name)]
        run_total = float(live_summary["runs"][run_name]["completed_in_seconds"])
        queries_total = int(live_summary["runs"][run_name]["queries_total"])
        per_query_mean = statistics.mean(per_query)
        per_query_run_total = run_total / queries_total
        out[run_name] = {
            "per_query_mean": per_query_mean,
            "per_query_run_total": per_query_run_total,
            "min": min(per_query),
            "max": max(per_query),
        }
    return out


def compute_ntcir_candidate_stats(live_summary: dict) -> dict[str, float]:
    dedup_counts: list[int] = []
    raw_counts: list[int] = []
    subquery_counts: list[int] = []
    for run_name in ("original__full", "complex__full"):
        run_raw_total = 0
        for row in iter_query_results(run_name):
            dedup_counts.append(int(row.get("deduped_count", 0)))
            raw = int(row.get("raw_hits_count", 0))
            run_raw_total += raw
            raw_counts.append(raw)
            subquery_counts.append(len(row.get("subqueries", [])))

        expected_raw = int(live_summary["runs"][run_name]["retrieved_candidates_total"])
        if run_raw_total != expected_raw:
            raise RuntimeError(
                f"NTCIR raw-hit mismatch for {run_name}: query_results={run_raw_total}, summary={expected_raw}"
            )

    return {
        "dedup_avg": statistics.mean(dedup_counts),
        "dedup_min": min(dedup_counts),
        "dedup_max": max(dedup_counts),
        "dedup_original_avg": statistics.mean(
            [int(row.get("deduped_count", 0)) for row in iter_query_results("original__full")]
        ),
        "dedup_complex_avg": statistics.mean(
            [int(row.get("deduped_count", 0)) for row in iter_query_results("complex__full")]
        ),
        "raw_original_avg": statistics.mean(
            [int(row.get("raw_hits_count", 0)) for row in iter_query_results("original__full")]
        ),
        "raw_complex_avg": statistics.mean(
            [int(row.get("raw_hits_count", 0)) for row in iter_query_results("complex__full")]
        ),
        "subq_original_avg": statistics.mean(
            [len(row.get("subqueries", [])) for row in iter_query_results("original__full")]
        ),
        "subq_complex_avg": statistics.mean(
            [len(row.get("subqueries", [])) for row in iter_query_results("complex__full")]
        ),
        "subq_min": min(subquery_counts),
        "subq_max": max(subquery_counts),
    }


def build_rows() -> list[Row]:
    target_model = load_target_model()
    target_metrics = parse_metrics_summary()
    ntcir_offline = load_ntcir_run_summary()
    ntcir_online_live = load_ntcir_online_live_summary()
    ntcir_with_semantic, ntcir_without_semantic = load_ntcir_instruction_counts()
    ntcir_latency = compute_ntcir_latency_stats(ntcir_online_live)
    ntcir_candidates = compute_ntcir_candidate_stats(ntcir_online_live)

    expected_with = int(ntcir_offline["stages"]["index"]["with_semantic"]["inserted"])
    expected_without = int(ntcir_offline["stages"]["index"]["without_semantic"]["inserted"])
    if sum(ntcir_with_semantic) != expected_with:
        raise RuntimeError(
            f"NTCIR with-semantic query count mismatch: records={sum(ntcir_with_semantic)}, summary={expected_with}"
        )
    if sum(ntcir_without_semantic) != expected_without:
        raise RuntimeError(
            f"NTCIR without-semantic query count mismatch: records={sum(ntcir_without_semantic)}, summary={expected_without}"
        )

    target_space = extract_chroma_space(
        ROOT / "experiments" / "target" / "chroma" / "chroma.sqlite3",
        "instructions_test",
    )
    ntcir_space = extract_chroma_space(
        ROOT / "experiments" / "ntcir15" / "offline_phase" / "chroma_ntcir15_eval" / "chroma.sqlite3",
        "instructions_ntcir15_eval_with_semantic",
    )

    target_offline_path = ROOT / "experiments" / "target" / "preprocessing_utils.py"
    target_notebook_path = ROOT / "legacy" / "target" / "TARGET_test.ipynb"
    target_recovery_path = ROOT / "experiments" / "target" / "rerun_target_recovery.py"
    target_metrics_path = ROOT / "experiments" / "target" / "target_recovery_results_full" / "metrics_summary.csv"
    target_sqlite_path = ROOT / "experiments" / "target" / "chroma" / "chroma.sqlite3"

    ntcir_offline_script = ROOT / "experiments" / "ntcir15" / "offline_phase" / "db_population_evaluation_pack.py"
    ntcir_offline_summary_path = ROOT / "experiments" / "ntcir15" / "offline_phase" / "artifacts" / "run_summary.json"
    ntcir_dataset_info_path = ROOT / "experiments" / "ntcir15" / "offline_phase" / "artifacts" / "dataset_info_evaluation.jsonl"
    ntcir_prompts_path = ROOT / "experiments" / "ntcir15" / "online_phase" / "prompts.py"
    ntcir_online_script = ROOT / "experiments" / "ntcir15" / "online_phase" / "run_online_phase.py"
    ntcir_online_summary_path = ROOT / "experiments" / "ntcir15" / "online_phase" / "artifacts_live" / "online_run_summary.json"
    ntcir_sqlite_path = ROOT / "experiments" / "ntcir15" / "offline_phase" / "chroma_ntcir15_eval" / "chroma.sqlite3"

    target_k5_ottqa = target_metrics["ottqa__queryopt_sem__k5"]
    target_k5_fetaqa = target_metrics["fetaqa__queryopt_sem__k5"]

    ntcir_index_started = parse_iso8601(ntcir_offline["run_started_at_utc"])
    ntcir_index_finished = parse_iso8601(ntcir_offline["run_finished_at_utc"])
    ntcir_index_seconds = (ntcir_index_finished - ntcir_index_started).total_seconds()

    rows = [
        Row(
            field="LLM name/version for pseudoquery generation",
            target_value=target_model,
            ntcir_value=str(ntcir_offline["instruction_model"]),
            evidence_type="TARGET: code-backed; NTCIR-15: artifact-backed",
            source_path=(
                f"TARGET: `{rel(target_notebook_path)}`<br>"
                f"NTCIR-15: `{rel(ntcir_offline_summary_path)}`"
            ),
            notes="TARGET recovered runs replay prebuilt artifacts; the notebook model variable is the only checked-in model binding for pseudoquery generation.",
        ),
        Row(
            field="LLM for query optimization",
            target_value=target_model,
            ntcir_value=str(ntcir_online_live["llm_model"]),
            evidence_type="TARGET: code-backed; NTCIR-15: artifact-backed",
            source_path=(
                f"TARGET: `{rel(target_notebook_path)}`, `{rel(target_offline_path)}`<br>"
                f"NTCIR-15: `{rel(ntcir_online_summary_path)}`, `{rel(ntcir_online_script)}`"
            ),
            notes="TARGET query optimization is implemented as background generation plus subquery decomposition in `get_subqueries_from_query`.",
        ),
        Row(
            field="LLM for reranking",
            target_value="Not used / no reranking stage in recovered TARGET runs",
            ntcir_value=str(ntcir_online_live["llm_model"]),
            evidence_type="TARGET: artifact-backed; NTCIR-15: artifact-backed",
            source_path=(
                f"TARGET: `{rel(target_recovery_path)}`, `{rel(target_metrics_path)}`<br>"
                f"NTCIR-15: `{rel(ntcir_online_script)}`, `{rel(ntcir_prompts_path)}`, `{rel(ntcir_online_summary_path)}`"
            ),
            notes="NTCIR-15 reranking uses the same live `llm_model` as query optimization in the checked-in live runs.",
        ),
        Row(
            field="Embedding model name/version",
            target_value="text-embedding-3-small",
            ntcir_value="text-embedding-3-small",
            evidence_type="TARGET: code-backed; NTCIR-15: code-backed",
            source_path=(
                f"TARGET: `{rel(target_offline_path)}`, `{rel(target_recovery_path)}`<br>"
                f"NTCIR-15: `{rel(ntcir_offline_script)}`, `{rel(ntcir_online_script)}`"
            ),
            notes="No more specific model revision is persisted in the repo.",
        ),
        Row(
            field="Vector store / nearest-neighbor backend",
            target_value="Chroma PersistentClient with HNSW vector index",
            ntcir_value="Chroma PersistentClient with HNSW vector index",
            evidence_type="TARGET: code-backed; NTCIR-15: artifact-backed",
            source_path=(
                f"TARGET: `{rel(target_notebook_path)}`, `{rel(target_sqlite_path)}`<br>"
                f"NTCIR-15: `{rel(ntcir_online_summary_path)}`, `{rel(ntcir_sqlite_path)}`"
            ),
            notes="TARGET recovery code points at `/chroma_db`; the checked-in target-adjacent Chroma SQLite also stores HNSW config.",
        ),
        Row(
            field="temperature",
            target_value="Not explicitly set in chat completions; SDK/provider default",
            ntcir_value="Not explicitly set in chat completions; SDK/provider default",
            evidence_type="TARGET: code-backed; NTCIR-15: code-backed",
            source_path=(
                f"TARGET: `{rel(target_offline_path)}`<br>"
                f"NTCIR-15: `{rel(ntcir_offline_script)}`, `{rel(ntcir_online_script)}`"
            ),
            notes="Do not interpret unset parameters as a known numeric value.",
        ),
        Row(
            field="max tokens",
            target_value="Not explicitly set in actual generation calls; SDK/provider default",
            ntcir_value="Not explicitly set in actual generation calls; SDK/provider default",
            evidence_type="TARGET: code-backed; NTCIR-15: code-backed",
            source_path=(
                f"TARGET: `{rel(target_offline_path)}`<br>"
                f"NTCIR-15: `{rel(ntcir_offline_script)}`, `{rel(ntcir_online_script)}`"
            ),
            notes="NTCIR-15 only sets `max_tokens=16` for API preflight, not for the real experiment calls.",
        ),
        Row(
            field="Whether the same prompt template is used across datasets",
            target_value="Yes",
            ntcir_value="Yes",
            evidence_type="TARGET: code-backed; NTCIR-15: code-backed",
            source_path=(
                f"TARGET: `{rel(target_offline_path)}`<br>"
                f"NTCIR-15: `{rel(ntcir_offline_script)}`, `{rel(ntcir_prompts_path)}`"
            ),
            notes="Both tracks reuse shared prompt-builder functions rather than per-dataset templates.",
        ),
        Row(
            field="Whether prompts are in appendix / supplementary material",
            target_value="Unavailable from repo; requires manuscript source",
            ntcir_value="Unavailable from repo; requires manuscript source",
            evidence_type="TARGET: unavailable; NTCIR-15: unavailable",
            source_path="Manuscript source not included in this repository",
            notes="Repo only contains implementation prompts, not paper appendix metadata.",
        ),
        Row(
            field="Number of pseudoqueries per dataset",
            target_value="10 per dataset",
            ntcir_value=(
                f"Variable-length. With semantic (full-system collection): "
                f"{statistics.mean(ntcir_with_semantic):.2f} avg, {min(ntcir_with_semantic)}-{max(ntcir_with_semantic)} range "
                f"({sum(ntcir_with_semantic)} total across {len(ntcir_with_semantic)} datasets)."
            ),
            evidence_type="TARGET: code-backed; NTCIR-15: artifact-backed",
            source_path=(
                f"TARGET: `{rel(target_offline_path)}`<br>"
                f"NTCIR-15: `{rel(ntcir_dataset_info_path)}`, `{rel(ntcir_offline_summary_path)}`"
            ),
            notes="NTCIR-15 prompt says 'Generate as many queries as required', so the count is not fixed.",
        ),
        Row(
            field="Top-K retrieved pseudoqueries per subquery",
            target_value="100",
            ntcir_value=str(ntcir_online_live["n_results_per_subquery"]),
            evidence_type="TARGET: code-backed; NTCIR-15: artifact-backed",
            source_path=(
                f"TARGET: `{rel(target_offline_path)}`<br>"
                f"NTCIR-15: `{rel(ntcir_online_summary_path)}`, `{rel(ntcir_online_script)}`"
            ),
            notes="TARGET retrieves 100 Chroma hits per subquery before frequency aggregation; NTCIR-15 retrieves 5 hits per subquery before deduplication and reranking.",
        ),
        Row(
            field="Max number of subqueries N, if capped",
            target_value="10 subqueries, explicitly capped by prompt",
            ntcir_value=(
                "No explicit cap in code. "
                f"Observed in live `full` runs: {ntcir_candidates['subq_min']}-{ntcir_candidates['subq_max']} subqueries/query "
                f"({ntcir_candidates['subq_original_avg']:.1f} avg original, {ntcir_candidates['subq_complex_avg']:.1f} avg complex)."
            ),
            evidence_type="TARGET: code-backed; NTCIR-15: code-backed + artifact-backed",
            source_path=(
                f"TARGET: `{rel(target_offline_path)}`<br>"
                f"NTCIR-15: `{rel(ntcir_prompts_path)}`, `{rel(ntcir_online_script)}`, `{rel(ntcir_online_summary_path)}`"
            ),
            notes="NTCIR-15 live counts come from `artifacts_live/runs/*/query_results/*.json`.",
        ),
        Row(
            field="Candidate set size before reranking",
            target_value="N/A; no reranking stage",
            ntcir_value=(
                f"{ntcir_candidates['dedup_avg']:.1f} avg unique candidates/query in live `full` runs "
                f"({ntcir_candidates['dedup_original_avg']:.1f} original, {ntcir_candidates['dedup_complex_avg']:.1f} complex; "
                f"{ntcir_candidates['dedup_min']}-{ntcir_candidates['dedup_max']} range)."
            ),
            evidence_type="TARGET: artifact-backed; NTCIR-15: artifact-backed",
            source_path=(
                f"TARGET: `{rel(target_metrics_path)}`<br>"
                f"NTCIR-15: `{rel(ntcir_online_summary_path)}`, `experiments/ntcir15/online_phase/artifacts_live/runs/*/query_results/*.json`"
            ),
            notes="NTCIR-15 candidate-set size is the deduplicated set that is passed into the reranker.",
        ),
        Row(
            field="Similarity metric",
            target_value=target_space.upper(),
            ntcir_value=ntcir_space.upper(),
            evidence_type="TARGET: artifact-backed; NTCIR-15: artifact-backed",
            source_path=(
                f"TARGET: `{rel(target_sqlite_path)}`<br>"
                f"NTCIR-15: `{rel(ntcir_sqlite_path)}`"
            ),
            notes='Chroma SQLite collection config stores `space="l2"` for the HNSW vector index.',
        ),
        Row(
            field="How tables are cleaned",
            target_value=(
                "First row is promoted to header; on multi-line-header failure, header rows are merged, "
                "apostrophes are removed, whitespace is normalized to `_`, names are lowercased, and duplicates are disambiguated."
            ),
            ntcir_value=(
                "Files are robustly parsed from CSV/TSV/TXT/XLS/XLSX/JSON/HTML/ZIP/GZ; bad CSV lines are skipped; "
                "large streamable files use hybrid sampled profiling; complex values are sanitized to JSON/text for profiling."
            ),
            evidence_type="TARGET: code-backed; NTCIR-15: code-backed",
            source_path=(
                f"TARGET: `{rel(target_offline_path)}`<br>"
                f"NTCIR-15: `{rel(ntcir_offline_script)}`"
            ),
            notes="Neither track performs aggressive cell-level normalization beyond what is described here.",
        ),
        Row(
            field="Type detection rules",
            target_value=(
                "Uses `datamart_profiler` structural types, then `pandas.is_numeric_dtype` and "
                "`pandas.is_datetime64_any_dtype`; all other columns are treated as categorical/text-like."
            ),
            ntcir_value=(
                "Uses the same `datamart_profiler` structural type summary and pandas numeric/datetime checks; "
                "global scans also accumulate numeric counts and per-column null/non-null stats."
            ),
            evidence_type="TARGET: code-backed; NTCIR-15: code-backed",
            source_path=(
                f"TARGET: `{rel(target_offline_path)}`<br>"
                f"NTCIR-15: `{rel(ntcir_offline_script)}`"
            ),
            notes="Coverage/range summaries are emitted when profiler metadata includes them.",
        ),
        Row(
            field="Handling missing values",
            target_value=(
                "No explicit blanket imputation in wrapper code. Categorical top-value summaries use `dropna()`, "
                "so missing entries are excluded from frequency counts; other handling is delegated to parsers/profiler."
            ),
            ntcir_value=(
                "No blanket imputation in wrapper code. Global scan reports missing counts and percentages per column; "
                "categorical frequency summaries use `dropna()` and value counts without nulls."
            ),
            evidence_type="TARGET: code-backed; NTCIR-15: code-backed",
            source_path=(
                f"TARGET: `{rel(target_offline_path)}`<br>"
                f"NTCIR-15: `{rel(ntcir_offline_script)}`"
            ),
            notes="Notebook outputs also show `datamart_profiler` internals warning about internal `fillna('')`, but that is not an explicit wrapper rule.",
        ),
        Row(
            field="Offline indexing cost",
            target_value="Unavailable from checked-in TARGET artifacts",
            ntcir_value=(
                f"{format_seconds(ntcir_index_seconds)} total for the checked-in index stage "
                f"({ntcir_offline['coverage']['records_total']} discovered, {ntcir_offline['coverage']['processable_true']} processable datasets)."
            ),
            evidence_type="TARGET: unavailable; NTCIR-15: artifact-backed",
            source_path=(
                f"TARGET: `{rel(target_notebook_path)}`<br>"
                f"NTCIR-15: `{rel(ntcir_offline_summary_path)}`"
            ),
            notes="TARGET notebook contains commented timing prints, but no persisted offline indexing runtime.",
        ),
        Row(
            field="Average online latency",
            target_value=(
                "Recovery replay only, not original end-to-end LLM latency. "
                f"`queryopt_sem` k=5 totals: OTT-QA {float(target_k5_ottqa['retrieval_duration_wall_clock']):.5f}s/run, "
                f"FetaQA {float(target_k5_fetaqa['retrieval_duration_wall_clock']):.5f}s/run; "
                "per-query averages round to about 0.00001s/query in the persisted metrics."
            ),
            ntcir_value=(
                f"Live `full` runs: original {ntcir_latency['original__full']['per_query_mean']:.2f}s/query "
                f"(run-total cross-check {ntcir_latency['original__full']['per_query_run_total']:.2f}s), "
                f"complex {ntcir_latency['complex__full']['per_query_mean']:.2f}s/query "
                f"(run-total cross-check {ntcir_latency['complex__full']['per_query_run_total']:.2f}s)."
            ),
            evidence_type="TARGET: artifact-backed; NTCIR-15: artifact-backed",
            source_path=(
                f"TARGET: `{rel(target_metrics_path)}`<br>"
                f"NTCIR-15: `{rel(ntcir_online_summary_path)}`, `experiments/ntcir15/online_phase/artifacts_live/runs/*/query_results/*.json`"
            ),
            notes="The checked-in `experiments/ntcir15/online_phase/artifacts/online_run_summary.json` is dry-run and is intentionally excluded.",
        ),
        Row(
            field="Approximate token usage",
            target_value=(
                "Not persisted. Rough proxy only: offline indexing uses about 2 LLM calls per dataset "
                "(semantic profile + pseudoqueries), and online query optimization uses about 2 LLM calls per user query "
                "(background + subquery decomposition)."
            ),
            ntcir_value=(
                "Not persisted. Rough proxy only: offline indexing uses about 2 LLM calls per processable dataset "
                "(semantic + instructions); live `full` retrieval uses about 3 LLM calls per query "
                "(background + subquery decomposition + rerank), plus 1 extra call/query when complex queries are generated."
            ),
            evidence_type="TARGET: code-inferred; NTCIR-15: code-inferred",
            source_path=(
                f"TARGET: `{rel(target_offline_path)}`, `{rel(target_notebook_path)}`<br>"
                f"NTCIR-15: `{rel(ntcir_offline_script)}`, `{rel(ntcir_online_script)}`"
            ),
            notes="These are call-count proxies, not actual prompt/completion token totals.",
        ),
    ]
    return rows


def build_markdown(rows: list[Row]) -> str:
    lines = [
        "# Experimental Metadata for TARGET and NTCIR-15",
        "",
        "| Field | TARGET value | NTCIR-15 value | Evidence type | Source path | Notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                escape_cell(value)
                for value in (
                    row.field,
                    row.target_value,
                    row.ntcir_value,
                    row.evidence_type,
                    row.source_path,
                    row.notes,
                )
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    rows = build_rows()
    OUTPUT_PATH.write_text(build_markdown(rows), encoding="utf-8")
    print(f"[done] Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





