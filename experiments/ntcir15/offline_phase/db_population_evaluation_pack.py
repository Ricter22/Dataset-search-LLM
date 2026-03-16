#!/usr/bin/env python3
"""Populate Chroma DB from evaluation_ready_pack with dual instruction variants.

This script replicates the benchmark notebook logic for all datasets that can be
loaded into a pandas DataFrame and builds two collections:
1) instructions generated with semantic profile context
2) instructions generated without semantic profile context
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import pickle
import random
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import datamart_profiler
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


SCRIPT_DIR = Path(__file__).resolve().parent
ZIP_MAGIC = b"PK\x03\x04"
OLE_MAGIC = b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"
GZIP_MAGIC = b"\x1F\x8B"

TEMPLATE_SEMANTIC = """
{'Temporal':
    {
        'isTemporal': Does this column contain temporal information? Yes or No,
        'resolution': If Yes, specify the resolution (Year, Month, Day, Hour, etc.).
    },
 'Spatial': {'isSpatial': Does this column contain spatial information? Yes or No,
             'resolution': If Yes, specify the resolution (Country, State, City, Coordinates, etc.).},
 'Entity Type': What kind of entity does the column describe? (e.g., Person, Location, Organization, Product),
 'Domain-Specific Types': What domain is this column from (e.g., Financial, Healthcare, E-commerce, Climate, Demographic),
 'Function/Usage Context': How might the data be used (e.g., Aggregation Key, Ranking/Scoring, Interaction Data, Measurement).}
"""

RESPONSE_EXAMPLE_SEMANTIC = """
{
"Domain-Specific Types": "General",
"Entity Type": "Temporal Entity",
"Function/Usage Context": "Aggregation Key",
"Spatial": {"isSpatial": false,
            "resolution": ""},
"Temporal": {"isTemporal": true,
            "resolution": "Year"}
}
"""

RESPONSE_QUERIES_INSTRUCTIONS = """
{
  "queries": [
    {
      "query": "Select data ..."
    },
    {
      "query": "Find datasets ..."
    },
    {
      "query": "Show me data ..."
    }
  ]
}
"""


class UnprocessableDatasetError(Exception):
    """Raised when a dataset file exists but cannot be transformed into tabular data."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail.strip()
        msg = reason if not self.detail else f"{reason}: {self.detail}"
        super().__init__(msg)


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


def should_log_progress(
    index: int,
    total: int,
    last_log_ts: float,
    now_ts: float,
    log_every: int,
) -> bool:
    if index == 1 or index == total:
        return True
    if log_every > 0 and index % log_every == 0:
        return True
    if (now_ts - last_log_ts) >= 30:
        return True
    return False


def print_progress(
    stage: str,
    index: int,
    total: int,
    started_at: float,
    stats: dict[str, Any],
) -> None:
    elapsed = perf_counter() - started_at
    rate = index / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total - index)
    eta = (remaining / rate) if rate > 0 else float("inf")
    pct = (index / total * 100.0) if total > 0 else 100.0
    stats_text = ", ".join(f"{k}={v}" for k, v in stats.items())
    print(
        (
            f"[{stage}] {index}/{total} ({pct:.1f}%) | "
            f"elapsed={format_duration(elapsed)} | "
            f"eta={format_duration(eta)} | "
            f"rate={rate:.2f}/s | {stats_text}"
        ),
        flush=True,
    )


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return path


def digest_text(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


def discover_files(input_root: Path, limit: int | None) -> list[Path]:
    files = sorted([p for p in input_root.rglob("*") if p.is_file()])
    if limit is not None:
        return files[: max(1, limit)]
    return files


def infer_domain_dataset(file_path: Path, input_root: Path) -> tuple[str, str, str]:
    rel = file_path.relative_to(input_root)
    parts = rel.parts
    domain = parts[0] if len(parts) >= 1 else "unknown-domain"
    dataset_id = parts[1] if len(parts) >= 2 else f"generated-{digest_text(str(rel))}"
    dataset_name = str(rel).replace("\\", "/")
    return domain, dataset_id, dataset_name


def safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return str(obj)


def append_error(errors_path: Path, payload: dict[str, Any]) -> None:
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    with errors_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def create_global_stats() -> dict[str, Any]:
    return {"rows_total": 0, "columns": {}}


def _ensure_column_stats(global_stats: dict[str, Any], column: str) -> dict[str, Any]:
    columns = global_stats["columns"]
    if column not in columns:
        columns[column] = {
            "non_null": 0,
            "nulls": 0,
            "numeric_count": 0,
            "sum": 0.0,
            "sumsq": 0.0,
            "min": None,
            "max": None,
        }
    return columns[column]


def update_global_stats_with_df(global_stats: dict[str, Any], df: pd.DataFrame) -> None:
    row_count = int(len(df))
    global_stats["rows_total"] += row_count

    for col in df.columns:
        col_name = str(col)
        stats = _ensure_column_stats(global_stats, col_name)
        series = df[col]

        non_null = int(series.notna().sum())
        nulls = row_count - non_null
        stats["non_null"] += non_null
        stats["nulls"] += nulls

        numeric = pd.to_numeric(series, errors="coerce")
        valid = numeric.dropna()
        n_numeric = int(valid.shape[0])
        if n_numeric == 0:
            continue

        stats["numeric_count"] += n_numeric
        s = float(valid.sum())
        stats["sum"] += s
        stats["sumsq"] += float((valid * valid).sum())

        vmin = float(valid.min())
        vmax = float(valid.max())
        if stats["min"] is None or vmin < stats["min"]:
            stats["min"] = vmin
        if stats["max"] is None or vmax > stats["max"]:
            stats["max"] = vmax


def infer_file_size_bytes(file_path: Path) -> int:
    try:
        return int(file_path.stat().st_size)
    except Exception:  # noqa: BLE001
        return 0


def read_file_prefix(file_path: Path, size: int = 4096) -> bytes:
    try:
        with file_path.open("rb") as fh:
            return fh.read(max(1, size))
    except Exception:  # noqa: BLE001
        return b""


def has_zip_signature(file_path: Path) -> bool:
    return read_file_prefix(file_path, 4).startswith(ZIP_MAGIC)


def looks_like_html(file_path: Path) -> bool:
    prefix = read_file_prefix(file_path, 2048).lstrip().lower()
    if not prefix:
        return False
    return (
        prefix.startswith(b"<!doctype html")
        or prefix.startswith(b"<html")
        or b"<html" in prefix[:400]
    )


def looks_like_whitespace_only(file_path: Path, max_scan_bytes: int = 65536) -> bool:
    try:
        with file_path.open("rb") as fh:
            chunk = fh.read(max(1, max_scan_bytes))
            if not chunk:
                return True
            return len(chunk.strip()) == 0
    except Exception:  # noqa: BLE001
        return False


def sanitize_value_for_profile(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (dict, list, tuple, set)):
        return safe_json(value)
    return str(value)


def is_streamable_for_hybrid(file_path: Path) -> bool:
    ext = file_path.suffix.lower()
    if ext not in {".csv", ".txt", ".gz"}:
        return False
    if ext == ".gz":
        return True
    prefix = read_file_prefix(file_path, 1024)
    if not prefix:
        return False
    if prefix.startswith(ZIP_MAGIC):
        return False
    return True


def iter_stream_chunks(file_path: Path, chunk_size: int):
    ext = file_path.suffix.lower()
    base_kwargs: dict[str, Any] = {
        "chunksize": max(1, chunk_size),
        "engine": "python",
        "sep": None,
        "on_bad_lines": "skip",
    }
    if ext == ".gz":
        base_kwargs["compression"] = "infer"
    if ext not in {".csv", ".txt", ".gz"}:
        raise ValueError(f"Hybrid streaming is not supported for extension: {ext}")
    return pd.read_csv(file_path, **base_kwargs)


def stream_global_stats_and_sample(
    file_path: Path,
    chunk_size: int,
    sample_rows: int,
    sample_seed: int,
    dataset_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    global_stats = create_global_stats()
    scan_started_at = perf_counter()
    last_scan_log = scan_started_at

    # Pass 1: full scan for global stats.
    chunks_seen = 0
    for chunk in iter_stream_chunks(file_path, chunk_size):
        chunks_seen += 1
        update_global_stats_with_df(global_stats, chunk)
        now_ts = perf_counter()
        if chunks_seen == 1 or now_ts - last_scan_log >= 30:
            print(
                (
                    f"[profile:scan] {dataset_name} | pass=1 | chunks={chunks_seen} | "
                    f"rows_scanned={global_stats['rows_total']} | "
                    f"elapsed={format_duration(now_ts - scan_started_at)}"
                ),
                flush=True,
            )
            last_scan_log = now_ts

    rows_total = int(global_stats["rows_total"])
    if rows_total <= 0:
        return pd.DataFrame(), global_stats

    # Pass 2: random chunk sampling.
    target_rows = max(1, sample_rows)
    fraction = min(1.0, float(target_rows) / float(rows_total))
    sampled_parts: list[pd.DataFrame] = []
    sample_started_at = perf_counter()
    last_sample_log = sample_started_at

    chunks_seen = 0
    for chunk in iter_stream_chunks(file_path, chunk_size):
        chunks_seen += 1
        if fraction >= 1.0:
            part = chunk
        else:
            random_state = random.Random(sample_seed + chunks_seen).randint(0, 2**31 - 1)
            part = chunk.sample(frac=fraction, random_state=random_state)
        if not part.empty:
            sampled_parts.append(part)
        now_ts = perf_counter()
        if chunks_seen == 1 or now_ts - last_sample_log >= 30:
            sampled_so_far = sum(len(df) for df in sampled_parts)
            print(
                (
                    f"[profile:scan] {dataset_name} | pass=2 | chunks={chunks_seen} | "
                    f"sampled_rows={sampled_so_far} | "
                    f"elapsed={format_duration(now_ts - sample_started_at)}"
                ),
                flush=True,
            )
            last_sample_log = now_ts

    if not sampled_parts:
        return pd.DataFrame(), global_stats

    sampled_df = pd.concat(sampled_parts, ignore_index=True)
    if sampled_df.shape[0] > target_rows:
        sampled_df = sampled_df.sample(n=target_rows, random_state=sample_seed).reset_index(
            drop=True
        )
    return sampled_df, global_stats


def append_global_profile_section(
    profile_text: str,
    global_stats: dict[str, Any],
    mode: str,
    sampled_rows: int,
    sampling_seed: int | None,
) -> str:
    rows_total = int(global_stats.get("rows_total", 0))
    columns = global_stats.get("columns", {})

    lines = [
        "",
        "Profiling context:",
        f"- profile_mode: {mode}",
        f"- total_rows_global_scan: {rows_total}",
        f"- sampled_rows_for_deep_profile: {sampled_rows}",
    ]
    if sampling_seed is not None:
        lines.append(f"- sampling_seed: {sampling_seed}")

    for col_name, col_stats in columns.items():
        non_null = int(col_stats.get("non_null", 0))
        nulls = int(col_stats.get("nulls", 0))
        missing_pct = (100.0 * nulls / rows_total) if rows_total > 0 else 0.0
        line = (
            f"**{col_name} (global scan)**: non-null {non_null}/{rows_total}, "
            f"missing {nulls} ({missing_pct:.2f}%). "
        )

        numeric_count = int(col_stats.get("numeric_count", 0))
        if numeric_count > 0:
            num_sum = float(col_stats.get("sum", 0.0))
            num_sumsq = float(col_stats.get("sumsq", 0.0))
            mean = num_sum / numeric_count
            variance = max(0.0, (num_sumsq / numeric_count) - (mean * mean))
            std = math.sqrt(variance)
            nmin = col_stats.get("min")
            nmax = col_stats.get("max")
            line += (
                f"Numeric(global): count {numeric_count}, mean {mean:.6g}, "
                f"std {std:.6g}, min {nmin}, max {nmax}. "
            )
        lines.append(line)

    return profile_text + "\n" + "\n".join(lines)


def load_dataframe(file_path: Path) -> pd.DataFrame:
    ext = file_path.suffix.lower()

    def read_csv_robust(source: Any) -> pd.DataFrame:
        attempts: list[dict[str, Any]] = []
        for encoding in [None, "utf-8", "utf-8-sig", "latin1", "cp1252"]:
            for sep in [None, ",", "\t", ";", "|"]:
                kwargs: dict[str, Any] = {
                    "engine": "python",
                    "on_bad_lines": "skip",
                }
                if encoding is not None:
                    kwargs["encoding"] = encoding
                kwargs["sep"] = sep
                attempts.append(kwargs)

        last_exc: Exception | None = None
        for kwargs in attempts:
            try:
                if hasattr(source, "seek"):
                    source.seek(0)
                return pd.read_csv(source, **kwargs)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc

        if last_exc is None:
            raise ValueError(f"Unable to parse CSV: {file_path}")
        raise last_exc

    def read_excel_robust(path_or_buffer: Any, extension_hint: str) -> pd.DataFrame:
        if extension_hint == ".xls":
            try:
                return pd.read_excel(path_or_buffer, engine="xlrd")
            except ImportError as exc:
                raise UnprocessableDatasetError("missing_dependency_xlrd", str(exc)) from exc
        try:
            return pd.read_excel(path_or_buffer, engine="openpyxl")
        except ImportError:
            return pd.read_excel(path_or_buffer)

    def read_json_robust(path_or_buffer: Any) -> pd.DataFrame:
        try:
            if hasattr(path_or_buffer, "seek"):
                path_or_buffer.seek(0)
            return pd.read_json(path_or_buffer)
        except ValueError:
            if hasattr(path_or_buffer, "seek"):
                path_or_buffer.seek(0)
            return pd.read_json(path_or_buffer, lines=True)

    def read_html_table(path_or_buffer: Any) -> pd.DataFrame:
        try:
            tables = pd.read_html(path_or_buffer)
        except ImportError as exc:
            raise UnprocessableDatasetError("missing_dependency_html5lib", str(exc)) from exc
        except ValueError as exc:
            raise UnprocessableDatasetError("html_non_tabular", str(exc)) from exc
        if not tables:
            raise UnprocessableDatasetError("html_non_tabular", "No HTML tables found")
        return tables[0]

    def read_zip_tabular(path: Path) -> pd.DataFrame:
        candidate_exts = [".csv", ".tsv", ".txt", ".xlsx", ".xls", ".json"]
        with zipfile.ZipFile(path, "r") as zf:
            members = [m for m in zf.namelist() if not m.endswith("/")]
            ordered: list[str] = []
            for ext_candidate in candidate_exts:
                for member in members:
                    if Path(member).suffix.lower() == ext_candidate:
                        ordered.append(member)
            if not ordered:
                raise UnprocessableDatasetError(
                    "zip_without_tabular_member",
                    f"No tabular members in archive ({len(members)} entries)",
                )

            last_exc: Exception | None = None
            for member in ordered:
                member_ext = Path(member).suffix.lower()
                try:
                    payload = zf.read(member)
                    if not payload:
                        continue
                    buffer = io.BytesIO(payload)
                    if member_ext in {".csv", ".tsv", ".txt"}:
                        return read_csv_robust(buffer)
                    if member_ext in {".xlsx", ".xls"}:
                        return read_excel_robust(buffer, member_ext)
                    if member_ext == ".json":
                        return read_json_robust(buffer)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    continue
            if last_exc is not None:
                raise last_exc
            raise UnprocessableDatasetError("zip_tabular_parse_failed", "Unable to parse archive members")

    if ext == ".csv":
        if looks_like_whitespace_only(file_path):
            raise UnprocessableDatasetError("empty_content", "CSV contains only whitespace/newlines")
        if has_zip_signature(file_path):
            return read_zip_tabular(file_path)
        return read_csv_robust(file_path)
    if ext == ".xls":
        if looks_like_html(file_path):
            return read_html_table(file_path)
        return read_excel_robust(file_path, ext)
    if ext == ".xlsx":
        if looks_like_html(file_path):
            return read_html_table(file_path)
        return read_excel_robust(file_path, ext)
    if ext == ".json":
        return read_json_robust(file_path)
    if ext == ".txt":
        if has_zip_signature(file_path):
            return read_zip_tabular(file_path)
        try:
            return read_csv_robust(file_path)
        except Exception:  # noqa: BLE001
            lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            return pd.DataFrame({"text": lines})
    if ext == ".gz":
        try:
            return pd.read_csv(file_path, compression="infer", low_memory=False, engine="python")
        except Exception:  # noqa: BLE001
            try:
                return pd.read_json(file_path, compression="infer")
            except Exception:  # noqa: BLE001
                return pd.read_json(file_path, lines=True, compression="infer")

    if has_zip_signature(file_path):
        return read_zip_tabular(file_path)
    if looks_like_html(file_path):
        return read_html_table(file_path)

    readers = [
        read_csv_robust,
        lambda p: read_excel_robust(p, ext),
        read_json_robust,
    ]
    last_exc: Exception | None = None
    for reader in readers:
        try:
            return reader(file_path)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
    if last_exc is None:
        raise ValueError(f"Unsupported file type: {file_path}")
    raise last_exc


def build_data_profile_from_df(df: pd.DataFrame, dataset_title: str) -> str:
    metadata = datamart_profiler.process_dataset(df)
    profile_summary: list[str] = []

    columns_meta = metadata.get("columns", []) if isinstance(metadata, dict) else []
    for column_meta in columns_meta:
        col_name = str(column_meta.get("name", "unknown"))
        column_summary = f"**{col_name}**: "

        structural_type = str(column_meta.get("structural_type", "Unknown"))
        column_summary += f"Data is of type {structural_type.split('/')[-1].lower()}. "

        if "num_distinct_values" in column_meta:
            column_summary += (
                f"There are {column_meta.get('num_distinct_values', 'unknown')} unique values. "
            )

        series = df[col_name] if col_name in df.columns else None
        if series is not None and pd.api.types.is_numeric_dtype(series):
            mean_value = series.mean()
            max_value = series.max()
            min_value = series.min()
            column_summary += "This column is numeric. "
            column_summary += f"Mean: {mean_value}, Max: {max_value}, Min: {min_value}. "
        elif series is not None and pd.api.types.is_datetime64_any_dtype(series):
            min_date = series.min()
            max_date = series.max()
            column_summary += "This column is of datetime type. "
            column_summary += f"Date range: from {min_date} to {max_date}. "
        elif series is not None:
            safe_series = series.dropna().map(sanitize_value_for_profile)
            value_counts = safe_series.value_counts(dropna=True)
            if len(value_counts) > 0:
                top_categories = [str(v) for v in value_counts.nlargest(3).index.tolist()]
                if top_categories:
                    column_summary += f"Top 3 frequent values: {', '.join(top_categories)}. "

        if "coverage" in column_meta and isinstance(column_meta["coverage"], list):
            low = 0
            high = 0
            for item in column_meta["coverage"]:
                try:
                    gte = item["range"]["gte"]
                    lte = item["range"]["lte"]
                    if gte < low:
                        low = gte
                    if lte > high:
                        high = lte
                except Exception:  # noqa: BLE001
                    continue
            column_summary += f"Coverage spans from {low} to {high}. "

        profile_summary.append(column_summary)

    final_profile = (
        f"The key data profile information for the dataset {dataset_title} includes:\n"
        + "\n".join(profile_summary)
    )
    return final_profile


def normalize_queries(payload: Any) -> dict[str, list[dict[str, str]]]:
    out: list[dict[str, str]] = []
    queries = payload.get("queries", []) if isinstance(payload, dict) else []
    if isinstance(queries, list):
        for item in queries:
            if isinstance(item, dict):
                q = str(item.get("query", "")).strip()
                if q:
                    out.append({"query": q})
            elif isinstance(item, str):
                q = item.strip()
                if q:
                    out.append({"query": q})
    return {"queries": out}


def openai_client() -> OpenAI:
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

    headers: dict[str, str] = {}
    if site_url:
        headers["HTTP-Referer"] = site_url
    if site_name:
        headers["X-Title"] = site_name

    kwargs: dict[str, Any] = {"api_key": api_key, "base_url": base_url}
    if headers:
        kwargs["default_headers"] = headers

    if "openrouter.ai" in str(base_url).lower() and not str(api_key).startswith("sk-or-v1-"):
        print(
            "[auth:warning] OPENROUTER_BASE_URL is set but API key is not in sk-or-v1 format.",
            flush=True,
        )

    return OpenAI(**kwargs)


def run_api_preflight(model: str) -> str:
    client = openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=16,
    )
    content = response.choices[0].message.content
    return str(content or "").strip()


def call_openai_api(prompt: str, model: str) -> Any:
    client = openai_client()
    if model == "o1-mini":
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    return client.chat.completions.create(model=model, messages=messages)


def generate_prompt(data_profile: str, template: str, response_example: str) -> str:
    return f"""
        You are a dataset semantic analyzer. Based on the data profile provided, classify the columns into multiple semantic types.
        Please group the semantic types under the following categories:
        'Temporal', 'Spatial', 'Entity Type', 'Data Format', 'Domain-Specific Types', 'Function/Usage Context'.
        Following is the template {template}
        Please follow these rules:
        1. The output must be a valid JSON object that can be directly loaded by json.loads. Example response is {response_example}
        2. All keys from the template must be present in the response.
        3. All keys and string values must be enclosed in double quotes.
        4. There must be no trailing commas.
        5. Use booleans (true/false) and numbers without quotes.
        6. Do not include any additional information or context in the response.
        7. If you are unsure about a specific category, you can leave it as an empty string.

        Data Profile: {data_profile}
        """


def generate_prompt_instructions(
    dataset_title: str, semantic_profile: str, final_profile_summary: str
) -> str:
    return f"""
    You are the dataset owner of {dataset_title}. You need to provide instructions to the users on how to discover effectively the dataset in a DataSpace platform.
    The user is provided with a prompt interface, so it can ask natural language questions to find the dataset.
    The dataset contains the following semantic types: {semantic_profile} /n
    The data profile is the following: {final_profile_summary} /n
    The final users don't have access to the dataset content. So, provide instructions (queries) that they could use to find this dataset.

    One example of a query could be: "Find a dataset with entries about cannabis strains and their effects"

    Generate as many queries as required to cover the entire dataset content and structure.
    Reason step by step:
    1. First understand the dataset content and structure.
    2. Formulate general queries to show the dataset content.
    3. Formulate precise queries to highlight specific findings or limitations.
    4. Formulate queries that consider interactions between different columns.
    5. Formulate queries that consider the dataset's temporal and spatial aspects.

    The output must be a valid JSON object that can be directly loaded by json.loads. It should be a list of queries. An example response is: {RESPONSE_QUERIES_INSTRUCTIONS}
    """


def json_to_dict(raw_json: str) -> dict[str, Any]:
    cleaned = raw_json.replace("```json\n", "").replace("\n```", "").strip()
    parsed = json.loads(cleaned)
    if isinstance(parsed, dict):
        return parsed
    return {"queries": []}


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


def generate_semantic_profile(data_profile: str, model: str, dry_run: bool) -> str:
    if dry_run:
        return '{"Temporal":{"isTemporal":false,"resolution":""},"Spatial":{"isSpatial":false,"resolution":""},"Entity Type":"","Domain-Specific Types":"","Function/Usage Context":""}'
    prompt = generate_prompt(data_profile, TEMPLATE_SEMANTIC, RESPONSE_EXAMPLE_SEMANTIC)
    response = call_openai_api(prompt, model)
    return response.choices[0].message.content


def generate_instruction_queries(
    dataset_name: str,
    data_profile: str,
    semantic_profile: str,
    model: str,
    use_semantic: bool,
    dry_run: bool,
) -> dict[str, list[dict[str, str]]]:
    semantic_for_prompt = semantic_profile if use_semantic else "{}"
    if dry_run:
        label = "with semantic profile" if use_semantic else "without semantic profile"
        return {"queries": [{"query": f"Find a dataset like {dataset_name} ({label})."}]}

    prompt = generate_prompt_instructions(dataset_name, semantic_for_prompt, data_profile)
    response = call_openai_api(prompt, model)
    parsed = json_to_dict(response.choices[0].message.content)
    return normalize_queries(parsed)


def load_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("rb") as fh:
        content = pickle.load(fh)
    if isinstance(content, list):
        return content
    return []


def write_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(records, fh)

    jsonl_path = path.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for record in records:
            safe_record = dict(record)
            safe_record["file_path"] = str(safe_record.get("file_path", ""))
            fh.write(json.dumps(safe_record, ensure_ascii=False) + "\n")


def upsert_discovered_records(
    existing: list[dict[str, Any]],
    files: list[Path],
    input_root: Path,
) -> list[dict[str, Any]]:
    by_path: dict[str, dict[str, Any]] = {}
    for record in existing:
        fp = str(record.get("file_path", "")).strip()
        if fp:
            by_path[fp] = record

    for file_path in files:
        key = str(file_path.resolve())
        domain, dataset_id, dataset_name = infer_domain_dataset(file_path, input_root)
        defaults: dict[str, Any] = {
            "dataset_id": dataset_id,
            "domain": domain,
            "dataset_name": dataset_name,
            "file_path": key,
            "file_format": file_path.suffix.lower(),
            "rows": None,
            "cols": None,
            "file_size_bytes": infer_file_size_bytes(file_path),
            "profile_mode": None,
            "profile_rows_total": None,
            "profile_rows_sampled": None,
            "profile_sampling_seed": None,
            "data_profile": None,
            "semantic_profile": None,
            "instructions_with_semantic": None,
            "instructions_without_semantic": None,
            "processable": None,
            "exclusion_reason": None,
            "status": "new",
            "error_stage": None,
            "error_message": None,
            "updated_at_utc": utc_now(),
        }
        if key not in by_path:
            by_path[key] = defaults
        else:
            for field, value in defaults.items():
                if field in {"dataset_id", "domain", "dataset_name", "file_format", "file_path"}:
                    by_path[key][field] = value
                elif field == "file_size_bytes":
                    by_path[key][field] = value
                else:
                    by_path[key].setdefault(field, value)

    return sorted(by_path.values(), key=lambda r: str(r.get("dataset_name", "")))


def normalize_record_states(records: list[dict[str, Any]]) -> None:
    for record in records:
        processable = record.get("processable")
        has_profile = bool(record.get("data_profile"))
        status = str(record.get("status", "")).lower()

        if processable is None:
            if has_profile:
                record["processable"] = True
            elif status == "excluded":
                record["processable"] = False

        if record.get("processable") is False and not record.get("exclusion_reason"):
            error_message = str(record.get("error_message") or "").strip()
            record["exclusion_reason"] = error_message or "unprocessable"


def stage_profile(
    records: list[dict[str, Any]],
    dataset_info_path: Path,
    errors_path: Path,
    resume: bool,
    log_every: int,
    large_file_size_mb: int,
    sample_rows: int,
    sample_seed: int,
    chunk_size: int,
    force_full_profile: bool,
) -> dict[str, int]:
    total = len(records)
    started_at = perf_counter()
    last_log_ts = started_at
    print(f"[profile] starting {total} records", flush=True)
    stats = {"processed": 0, "excluded": 0, "skipped": 0, "failed": 0}
    for idx, record in enumerate(records, start=1):
        if resume and (record.get("data_profile") or record.get("processable") is False):
            stats["skipped"] += 1
            now_ts = perf_counter()
            if should_log_progress(idx, total, last_log_ts, now_ts, log_every):
                print_progress("profile", idx, total, started_at, stats)
                last_log_ts = now_ts
            continue

        file_path = Path(str(record.get("file_path", "")))
        dataset_name = str(record.get("dataset_name", file_path.name))
        file_size_bytes = infer_file_size_bytes(file_path)
        record["file_size_bytes"] = file_size_bytes
        threshold_bytes = max(1, int(large_file_size_mb)) * 1024 * 1024
        use_hybrid = (
            (not force_full_profile)
            and (file_size_bytes > threshold_bytes)
            and is_streamable_for_hybrid(file_path)
        )
        mode = "hybrid_sampled" if use_hybrid else "full"
        print(
            (
                f"[profile:file] {idx}/{total} | mode={mode} | "
                f"size_mb={file_size_bytes / (1024 * 1024):.2f} | {dataset_name}"
            ),
            flush=True,
        )
        try:
            dataset_title = file_path.stem
            if use_hybrid:
                sampled_df, global_stats = stream_global_stats_and_sample(
                    file_path=file_path,
                    chunk_size=max(1, chunk_size),
                    sample_rows=max(1, sample_rows),
                    sample_seed=sample_seed,
                    dataset_name=dataset_name,
                )
                if sampled_df is None or sampled_df.empty:
                    raise ValueError("Hybrid sampled DataFrame is empty")
                base_profile = build_data_profile_from_df(sampled_df, dataset_title)
                profile = append_global_profile_section(
                    base_profile,
                    global_stats=global_stats,
                    mode="hybrid_sampled",
                    sampled_rows=int(sampled_df.shape[0]),
                    sampling_seed=sample_seed,
                )
                rows_total = int(global_stats.get("rows_total", sampled_df.shape[0]))
                cols_total = len(global_stats.get("columns", {})) or int(sampled_df.shape[1])
                record["rows"] = rows_total
                record["cols"] = cols_total
                record["profile_mode"] = "hybrid_sampled"
                record["profile_rows_total"] = rows_total
                record["profile_rows_sampled"] = int(sampled_df.shape[0])
                record["profile_sampling_seed"] = int(sample_seed)
            else:
                df = load_dataframe(file_path)
                if df is None or df.empty:
                    raise ValueError("Parsed DataFrame is empty")
                global_stats = create_global_stats()
                update_global_stats_with_df(global_stats, df)
                base_profile = build_data_profile_from_df(df, dataset_title)
                profile = append_global_profile_section(
                    base_profile,
                    global_stats=global_stats,
                    mode="full",
                    sampled_rows=int(df.shape[0]),
                    sampling_seed=None,
                )
                record["rows"] = int(df.shape[0])
                record["cols"] = int(df.shape[1])
                record["profile_mode"] = "full"
                record["profile_rows_total"] = int(df.shape[0])
                record["profile_rows_sampled"] = int(df.shape[0])
                record["profile_sampling_seed"] = None

            if profile is None or not str(profile).strip():
                raise UnprocessableDatasetError("empty_profile", "Generated profile text is empty")
            record["data_profile"] = profile
            record["processable"] = True
            record["exclusion_reason"] = None
            record["status"] = "ok"
            record["error_stage"] = None
            record["error_message"] = None
            record["updated_at_utc"] = utc_now()
            stats["processed"] += 1
        except UnprocessableDatasetError as exc:
            is_dependency_issue = str(exc.reason).startswith("missing_dependency_")
            record["rows"] = None
            record["cols"] = None
            record["data_profile"] = None
            record["semantic_profile"] = None
            record["instructions_with_semantic"] = None
            record["instructions_without_semantic"] = None
            record["error_stage"] = "profile"
            record["error_message"] = f"{type(exc).__name__}: {exc}"
            record["updated_at_utc"] = utc_now()
            if is_dependency_issue:
                record["processable"] = None
                record["exclusion_reason"] = None
                record["status"] = "error"
                stats["failed"] += 1
            else:
                record["processable"] = False
                record["exclusion_reason"] = exc.reason
                record["status"] = "excluded"
                stats["excluded"] += 1
            append_error(
                errors_path,
                {
                    "time": utc_now(),
                    "stage": "profile",
                    "dataset_name": record.get("dataset_name"),
                    "dataset_id": record.get("dataset_id"),
                    "file_path": str(file_path),
                    "error": record["error_message"],
                    "traceback": traceback.format_exc(),
                },
            )
        except Exception as exc:  # noqa: BLE001
            record["status"] = "error"
            record["error_stage"] = "profile"
            record["error_message"] = f"{type(exc).__name__}: {exc}"
            record["processable"] = None
            record["exclusion_reason"] = None
            record["updated_at_utc"] = utc_now()
            stats["failed"] += 1
            append_error(
                errors_path,
                {
                    "time": utc_now(),
                    "stage": "profile",
                    "dataset_name": record.get("dataset_name"),
                    "dataset_id": record.get("dataset_id"),
                    "file_path": str(file_path),
                    "error": record["error_message"],
                    "traceback": traceback.format_exc(),
                },
            )
        now_ts = perf_counter()
        if should_log_progress(idx, total, last_log_ts, now_ts, log_every):
            print_progress("profile", idx, total, started_at, stats)
            last_log_ts = now_ts
        write_records(dataset_info_path, records)
    write_records(dataset_info_path, records)
    print(
        f"[profile] completed in {format_duration(perf_counter() - started_at)}",
        flush=True,
    )
    return stats


def stage_semantic(
    records: list[dict[str, Any]],
    dataset_info_path: Path,
    errors_path: Path,
    resume: bool,
    semantic_model: str,
    dry_run: bool,
    log_every: int,
) -> dict[str, int]:
    total = len(records)
    started_at = perf_counter()
    last_log_ts = started_at
    print(f"[semantic] starting {total} records", flush=True)
    stats = {"processed": 0, "skipped": 0, "failed": 0}
    for idx, record in enumerate(records, start=1):
        if (record.get("processable") is False) or (not record.get("data_profile")):
            stats["skipped"] += 1
            now_ts = perf_counter()
            if should_log_progress(idx, total, last_log_ts, now_ts, log_every):
                print_progress("semantic", idx, total, started_at, stats)
                last_log_ts = now_ts
            continue
        if resume and record.get("semantic_profile"):
            stats["skipped"] += 1
            now_ts = perf_counter()
            if should_log_progress(idx, total, last_log_ts, now_ts, log_every):
                print_progress("semantic", idx, total, started_at, stats)
                last_log_ts = now_ts
            continue

        try:
            semantic_profile = generate_semantic_profile(
                str(record.get("data_profile")), semantic_model, dry_run
            )
            record["semantic_profile"] = semantic_profile
            record["processable"] = True
            record["exclusion_reason"] = None
            record["status"] = "ok"
            record["error_stage"] = None
            record["error_message"] = None
            record["updated_at_utc"] = utc_now()
            stats["processed"] += 1
        except Exception as exc:  # noqa: BLE001
            record["status"] = "error"
            record["error_stage"] = "semantic"
            record["error_message"] = f"{type(exc).__name__}: {exc}"
            record["updated_at_utc"] = utc_now()
            stats["failed"] += 1
            append_error(
                errors_path,
                {
                    "time": utc_now(),
                    "stage": "semantic",
                    "dataset_name": record.get("dataset_name"),
                    "dataset_id": record.get("dataset_id"),
                    "error": record["error_message"],
                    "traceback": traceback.format_exc(),
                },
            )
        now_ts = perf_counter()
        if should_log_progress(idx, total, last_log_ts, now_ts, log_every):
            print_progress("semantic", idx, total, started_at, stats)
            last_log_ts = now_ts
        write_records(dataset_info_path, records)
    write_records(dataset_info_path, records)
    print(
        f"[semantic] completed in {format_duration(perf_counter() - started_at)}",
        flush=True,
    )
    return stats


def stage_instructions(
    records: list[dict[str, Any]],
    dataset_info_path: Path,
    errors_path: Path,
    resume: bool,
    instruction_model: str,
    dry_run: bool,
    log_every: int,
) -> dict[str, int]:
    total = len(records)
    started_at = perf_counter()
    last_log_ts = started_at
    print(f"[instructions] starting {total} records", flush=True)
    stats = {
        "processed_with_semantic": 0,
        "processed_without_semantic": 0,
        "skipped": 0,
        "failed": 0,
    }
    for idx, record in enumerate(records, start=1):
        if (record.get("processable") is False) or (not record.get("data_profile")):
            stats["skipped"] += 1
            now_ts = perf_counter()
            if should_log_progress(idx, total, last_log_ts, now_ts, log_every):
                print_progress("instructions", idx, total, started_at, stats)
                last_log_ts = now_ts
            continue

        try:
            if not (resume and record.get("instructions_without_semantic")):
                without_semantic = generate_instruction_queries(
                    dataset_name=str(record.get("dataset_name")),
                    data_profile=str(record.get("data_profile")),
                    semantic_profile=str(record.get("semantic_profile") or ""),
                    model=instruction_model,
                    use_semantic=False,
                    dry_run=dry_run,
                )
                record["instructions_without_semantic"] = without_semantic
                stats["processed_without_semantic"] += 1

            if record.get("semantic_profile"):
                if not (resume and record.get("instructions_with_semantic")):
                    with_semantic = generate_instruction_queries(
                        dataset_name=str(record.get("dataset_name")),
                        data_profile=str(record.get("data_profile")),
                        semantic_profile=str(record.get("semantic_profile") or ""),
                        model=instruction_model,
                        use_semantic=True,
                        dry_run=dry_run,
                    )
                    record["instructions_with_semantic"] = with_semantic
                    stats["processed_with_semantic"] += 1
            else:
                # Keep explicit null when semantic profile is unavailable.
                if not record.get("instructions_with_semantic"):
                    record["instructions_with_semantic"] = None

            record["processable"] = True
            record["exclusion_reason"] = None
            record["status"] = "ok"
            record["error_stage"] = None
            record["error_message"] = None
            record["updated_at_utc"] = utc_now()
        except Exception as exc:  # noqa: BLE001
            record["status"] = "error"
            record["error_stage"] = "instructions"
            record["error_message"] = f"{type(exc).__name__}: {exc}"
            record["updated_at_utc"] = utc_now()
            stats["failed"] += 1
            append_error(
                errors_path,
                {
                    "time": utc_now(),
                    "stage": "instructions",
                    "dataset_name": record.get("dataset_name"),
                    "dataset_id": record.get("dataset_id"),
                    "error": record["error_message"],
                    "traceback": traceback.format_exc(),
                },
            )
        now_ts = perf_counter()
        if should_log_progress(idx, total, last_log_ts, now_ts, log_every):
            print_progress("instructions", idx, total, started_at, stats)
            last_log_ts = now_ts
        write_records(dataset_info_path, records)
    write_records(dataset_info_path, records)
    print(
        f"[instructions] completed in {format_duration(perf_counter() - started_at)}",
        flush=True,
    )
    return stats


def stringify_metadata_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return safe_json(value)


def build_query_metadata(record: dict[str, Any], variant: str, query_index: int) -> dict[str, Any]:
    metadata = {
        "dataset_id": record.get("dataset_id"),
        "domain": record.get("domain"),
        "dataset_name": record.get("dataset_name"),
        "file_format": record.get("file_format"),
        "file_size_bytes": record.get("file_size_bytes"),
        "rows": record.get("rows"),
        "cols": record.get("cols"),
        "profile_mode": record.get("profile_mode"),
        "profile_rows_total": record.get("profile_rows_total"),
        "profile_rows_sampled": record.get("profile_rows_sampled"),
        "variant": variant,
        "query_index": query_index,
        "processable": record.get("processable"),
        "exclusion_reason": record.get("exclusion_reason"),
        "data_profile": record.get("data_profile"),
        "semantic_profile": record.get("semantic_profile"),
    }
    return {k: stringify_metadata_value(v) for k, v in metadata.items()}


def load_existing_ids(collection: Any, page_size: int = 1000) -> set[str]:
    existing: set[str] = set()
    offset = 0
    while True:
        batch = collection.get(include=[], limit=page_size, offset=offset)
        ids = batch.get("ids", [])
        if not ids:
            break
        existing.update([str(i) for i in ids])
        if len(ids) < page_size:
            break
        offset += len(ids)
    return existing


def index_variant(
    collection: Any,
    records: list[dict[str, Any]],
    key: str,
    variant: str,
    resume: bool,
    log_every: int,
) -> dict[str, int]:
    total = 0
    for record in records:
        if record.get("processable") is False:
            continue
        instructions = record.get(key)
        if isinstance(instructions, dict):
            queries = instructions.get("queries", [])
            if isinstance(queries, list):
                total += len(queries)

    started_at = perf_counter()
    last_log_ts = started_at
    print(f"[index:{variant}] starting {total} candidate queries", flush=True)

    stats = {"inserted": 0, "skipped": 0, "failed": 0}
    existing_ids = load_existing_ids(collection) if resume else set()
    seen = 0

    for record in records:
        if record.get("processable") is False:
            continue
        instructions = record.get(key)
        if not isinstance(instructions, dict):
            continue
        queries = instructions.get("queries", [])
        if not isinstance(queries, list):
            continue

        for idx, item in enumerate(queries):
            seen += 1
            query = ""
            if isinstance(item, dict):
                query = str(item.get("query", "")).strip()
            elif isinstance(item, str):
                query = item.strip()

            if not query:
                stats["skipped"] += 1
            else:
                doc_id = f"{record.get('dataset_id')}::{variant}::q{idx}"
                if resume and doc_id in existing_ids:
                    stats["skipped"] += 1
                else:
                    try:
                        # Embedding is derived only from this single instruction string.
                        collection.add(
                            documents=[query],
                            ids=[doc_id],
                            metadatas=[build_query_metadata(record, variant, idx)],
                        )
                        existing_ids.add(doc_id)
                        stats["inserted"] += 1
                    except Exception:  # noqa: BLE001
                        stats["failed"] += 1

            now_ts = perf_counter()
            if should_log_progress(seen, max(1, total), last_log_ts, now_ts, log_every):
                print_progress(f"index:{variant}", seen, max(1, total), started_at, stats)
                last_log_ts = now_ts

    if seen == 0:
        print_progress(f"index:{variant}", 0, 1, started_at, stats)
    print(
        f"[index:{variant}] completed in {format_duration(perf_counter() - started_at)}",
        flush=True,
    )
    return stats


def stage_index(
    records: list[dict[str, Any]],
    chroma_path: Path,
    collection_with_semantic: str,
    collection_without_semantic: str,
    resume: bool,
    dry_run: bool,
    log_every: int,
) -> dict[str, Any]:
    if dry_run:
        with_semantic_queries = 0
        without_semantic_queries = 0
        for record in records:
            with_block = record.get("instructions_with_semantic")
            without_block = record.get("instructions_without_semantic")
            if isinstance(with_block, dict):
                with_semantic_queries += len(with_block.get("queries", []))
            if isinstance(without_block, dict):
                without_semantic_queries += len(without_block.get("queries", []))
        return {
            "dry_run": True,
            "with_semantic": {"candidate_queries": with_semantic_queries},
            "without_semantic": {"candidate_queries": without_semantic_queries},
        }

    import chromadb

    chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_path))
    embedding_function = OpenAIEmbeddingFunction()

    with_collection = client.get_or_create_collection(
        name=collection_with_semantic,
        embedding_function=embedding_function,
    )
    without_collection = client.get_or_create_collection(
        name=collection_without_semantic,
        embedding_function=embedding_function,
    )

    with_stats = index_variant(
        collection=with_collection,
        records=records,
        key="instructions_with_semantic",
        variant="with_semantic",
        resume=resume,
        log_every=log_every,
    )
    without_stats = index_variant(
        collection=without_collection,
        records=records,
        key="instructions_without_semantic",
        variant="without_semantic",
        resume=resume,
        log_every=log_every,
    )
    return {"dry_run": False, "with_semantic": with_stats, "without_semantic": without_stats}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replicate benchmark DB-population logic on evaluation_ready_pack with dual "
            "instruction variants and dual Chroma collections."
        )
    )
    parser.add_argument(
        "--stage",
        choices=["profile", "semantic", "instructions", "index", "all"],
        default="all",
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--input-root",
        default="../evaluation_ready_pack/datasets",
        help="Input dataset root directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="./artifacts",
        help="Output directory for artifacts.",
    )
    parser.add_argument(
        "--dataset-info-path",
        default="./artifacts/dataset_info_evaluation.pkl",
        help="Path to dataset info pickle checkpoint.",
    )
    parser.add_argument(
        "--errors-path",
        default="./artifacts/processing_errors.jsonl",
        help="Path for error logs (jsonl).",
    )
    parser.add_argument(
        "--chroma-path",
        default="./chroma_ntcir15_eval",
        help="Persistent Chroma directory.",
    )
    parser.add_argument(
        "--collection-with-semantic",
        default="instructions_ntcir15_eval_with_semantic",
        help="Chroma collection name for semantic-enabled instructions.",
    )
    parser.add_argument(
        "--collection-without-semantic",
        default="instructions_ntcir15_eval_without_semantic",
        help="Chroma collection name for semantic-disabled instructions.",
    )
    parser.add_argument(
        "--semantic-model",
        default="gpt-5-mini",
        help="OpenAI model for semantic profiling.",
    )
    parser.add_argument(
        "--instruction-model",
        default="gpt-5-mini",
        help="OpenAI model for instruction generation.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on discovered files.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Reserved for future parallel profiling support.",
    )
    parser.add_argument(
        "--large-file-size-mb",
        type=int,
        default=100,
        help="If file size exceeds this threshold, use hybrid profiling mode.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=200000,
        help="Target sample rows used for deep profiling in hybrid mode.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for hybrid sampling.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Chunk size for global stats scan and hybrid sampling.",
    )
    parser.add_argument(
        "--force-full-profile",
        action="store_true",
        help="Disable hybrid mode and always run full profiling.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Progress log interval in processed items (also logs every 30s).",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate API auth/connectivity and exit without running stages.",
    )
    parser.add_argument(
        "--fail-on-auth-error",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Abort run if API preflight fails.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip entries already completed.")
    parser.add_argument("--dry-run", action="store_true", help="Do not call OpenAI/Chroma.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_root = resolve_path(args.input_root)
    output_dir = resolve_path(args.output_dir)
    dataset_info_path = resolve_path(args.dataset_info_path)
    errors_path = resolve_path(args.errors_path)
    chroma_path = resolve_path(args.chroma_path)
    summary_path = output_dir / "run_summary.json"

    if not input_root.exists():
        parser.error(f"Input root not found: {input_root}")
    if not input_root.is_dir():
        parser.error(f"Input root must be a directory: {input_root}")

    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_info_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    files = discover_files(input_root, args.limit)
    records = load_records(dataset_info_path)
    records = upsert_discovered_records(records, files, input_root)
    normalize_record_states(records)
    write_records(dataset_info_path, records)

    summary: dict[str, Any] = {
        "run_started_at_utc": utc_now(),
        "stage": args.stage,
        "input_root": str(input_root),
        "discovered_files": len(files),
        "records": len(records),
        "resume": bool(args.resume),
        "dry_run": bool(args.dry_run),
        "semantic_model": args.semantic_model,
        "instruction_model": args.instruction_model,
        "workers": int(args.workers),
        "log_every": int(args.log_every),
        "large_file_size_mb": int(args.large_file_size_mb),
        "sample_rows": int(args.sample_rows),
        "sample_seed": int(args.sample_seed),
        "chunk_size": int(args.chunk_size),
        "force_full_profile": bool(args.force_full_profile),
        "preflight_only": bool(args.preflight_only),
        "fail_on_auth_error": bool(args.fail_on_auth_error),
        "stages": {},
    }

    need_openai_preflight = (
        (not bool(args.dry_run))
        and (bool(args.preflight_only) or args.stage in {"semantic", "instructions", "index", "all"})
    )
    preflight_ok = True
    if need_openai_preflight:
        preflight_model = str(args.semantic_model or args.instruction_model)
        print(f"[preflight] checking API auth using model={preflight_model}", flush=True)
        try:
            preflight_reply = run_api_preflight(preflight_model)
            print("[preflight] success", flush=True)
            summary["preflight"] = {
                "ok": True,
                "model": preflight_model,
                "reply": preflight_reply,
            }
        except Exception as exc:  # noqa: BLE001
            preflight_ok = False
            msg = f"{type(exc).__name__}: {exc}"
            print(f"[preflight] failed: {msg}", flush=True)
            summary["preflight"] = {
                "ok": False,
                "model": preflight_model,
                "error": msg,
            }
            append_error(
                errors_path,
                {
                    "time": utc_now(),
                    "stage": "preflight",
                    "error": msg,
                    "traceback": traceback.format_exc(),
                },
            )
            if bool(args.fail_on_auth_error):
                summary["run_finished_at_utc"] = utc_now()
                with summary_path.open("w", encoding="utf-8") as fh:
                    json.dump(summary, fh, ensure_ascii=False, indent=2)
                print(json.dumps(summary, ensure_ascii=False, indent=2))
                return 2

    if bool(args.preflight_only):
        summary["run_finished_at_utc"] = utc_now()
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if preflight_ok else 2

    if args.stage in {"profile", "all"}:
        summary["stages"]["profile"] = stage_profile(
            records=records,
            dataset_info_path=dataset_info_path,
            errors_path=errors_path,
            resume=args.resume,
            log_every=max(1, int(args.log_every)),
            large_file_size_mb=max(1, int(args.large_file_size_mb)),
            sample_rows=max(1, int(args.sample_rows)),
            sample_seed=int(args.sample_seed),
            chunk_size=max(1, int(args.chunk_size)),
            force_full_profile=bool(args.force_full_profile),
        )

    if args.stage in {"semantic", "all"}:
        summary["stages"]["semantic"] = stage_semantic(
            records=records,
            dataset_info_path=dataset_info_path,
            errors_path=errors_path,
            resume=args.resume,
            semantic_model=args.semantic_model,
            dry_run=args.dry_run,
            log_every=max(1, int(args.log_every)),
        )

    if args.stage in {"instructions", "all"}:
        summary["stages"]["instructions"] = stage_instructions(
            records=records,
            dataset_info_path=dataset_info_path,
            errors_path=errors_path,
            resume=args.resume,
            instruction_model=args.instruction_model,
            dry_run=args.dry_run,
            log_every=max(1, int(args.log_every)),
        )

    if args.stage in {"index", "all"}:
        summary["stages"]["index"] = stage_index(
            records=records,
            chroma_path=chroma_path,
            collection_with_semantic=args.collection_with_semantic,
            collection_without_semantic=args.collection_without_semantic,
            resume=args.resume,
            dry_run=args.dry_run,
            log_every=max(1, int(args.log_every)),
        )

    coverage = {
        "records_total": len(records),
        "processable_true": sum(1 for r in records if r.get("processable") is True),
        "processable_false_excluded": sum(1 for r in records if r.get("processable") is False),
        "has_data_profile": sum(1 for r in records if bool(r.get("data_profile"))),
        "has_semantic_profile": sum(1 for r in records if bool(r.get("semantic_profile"))),
        "has_instructions_with_semantic": sum(
            1 for r in records if isinstance(r.get("instructions_with_semantic"), dict)
        ),
        "has_instructions_without_semantic": sum(
            1 for r in records if isinstance(r.get("instructions_without_semantic"), dict)
        ),
    }
    summary["coverage"] = coverage

    summary["run_finished_at_utc"] = utc_now()
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
