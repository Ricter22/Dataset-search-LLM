#!/usr/bin/env python3
"""Filter JSONL dataset metadata to dataframe-friendly resources."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


DEFAULT_INCLUDE_FORMATS = {
    "csv",
    "excel",
    "tsv",
    "json",
    "parquet",
    "feather",
    "orc",
    "hdf",
    "stata",
    "sas",
    "spss",
    "sqlite",
}

FORMAT_ALIASES = {
    "csv": "csv",
    "csv (comma-separated values)": "csv",
    "ascii (csv)": "csv",
    "excel": "excel",
    "xls": "excel",
    "xlsx": "excel",
    "xlxs": "excel",
    "tsv": "tsv",
    "json": "json",
    "jsonl": "json",
    "ndjson": "json",
    "parquet": "parquet",
    "feather": "feather",
    "orc": "orc",
    "hdf": "hdf",
    "hdf5": "hdf",
    "h5": "hdf",
    "stata": "stata",
    "dta": "stata",
    "sas": "sas",
    "sas7bdat": "sas",
    "xpt": "sas",
    "spss": "spss",
    "sav": "spss",
    "sqlite": "sqlite",
    "sqlite3": "sqlite",
    "db": "sqlite",
}

EXT_ALIASES = {
    "csv": "csv",
    "tsv": "tsv",
    "xls": "excel",
    "xlsx": "excel",
    "xlxs": "excel",
    "sheet": "excel",
    "json": "json",
    "jsonl": "json",
    "ndjson": "json",
    "parquet": "parquet",
    "feather": "feather",
    "orc": "orc",
    "hdf": "hdf",
    "hdf5": "hdf",
    "h5": "hdf",
    "dta": "stata",
    "sas7bdat": "sas",
    "xpt": "sas",
    "sav": "spss",
    "sqlite": "sqlite",
    "sqlite3": "sqlite",
    "db": "sqlite",
}


def _clean_token(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def canonicalize_format(value: str | None) -> str | None:
    if not value:
        return None
    token = _clean_token(value)
    if token in FORMAT_ALIASES:
        return FORMAT_ALIASES[token]
    token_no_spaces = token.replace(" ", "")
    return FORMAT_ALIASES.get(token_no_spaces)


def canonicalize_extension(ext: str | None) -> str | None:
    if not ext:
        return None
    token = ext.lower().lstrip(".")
    return EXT_ALIASES.get(token)


def extension_from_path_like(path_like: str | None) -> str | None:
    if not path_like:
        return None
    suffixes = Path(path_like).suffixes
    if not suffixes:
        return None
    # Check all suffixes from right to left, so ".text.csv" resolves to csv.
    for suffix in reversed(suffixes):
        canonical = canonicalize_extension(suffix)
        if canonical:
            return canonical
    return None


def extension_from_url(url: str | None) -> str | None:
    if not url:
        return None
    parsed = urlparse(url)
    path = unquote(parsed.path or "")
    return extension_from_path_like(path)


def detect_resource_format(resource: dict[str, Any]) -> str | None:
    raw_format = resource.get("data_format")
    if isinstance(raw_format, str):
        canonical = canonicalize_format(raw_format)
        if canonical:
            return canonical

    filename = resource.get("data_filename")
    if isinstance(filename, str):
        canonical = extension_from_path_like(filename)
        if canonical:
            return canonical

    data_url = resource.get("data_url")
    if isinstance(data_url, str):
        canonical = extension_from_url(data_url)
        if canonical:
            return canonical

    return None


def parse_include_formats(raw_formats: str | None) -> set[str]:
    if not raw_formats:
        return set(DEFAULT_INCLUDE_FORMATS)

    include_formats: set[str] = set()
    for item in raw_formats.split(","):
        normalized = canonicalize_format(item) or canonicalize_extension(item)
        if normalized:
            include_formats.add(normalized)
        else:
            cleaned = _clean_token(item)
            if cleaned:
                include_formats.add(cleaned)
    return include_formats


def filter_jsonl(input_path: Path, output_path: Path, include_formats: set[str]) -> dict[str, int]:
    stats = {
        "lines_read": 0,
        "datasets_parsed": 0,
        "datasets_kept": 0,
        "resources_kept": 0,
        "parse_errors": 0,
    }

    with input_path.open("r", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            stats["lines_read"] += 1
            content = line.strip()
            if not content:
                continue

            try:
                obj = json.loads(content)
            except json.JSONDecodeError:
                stats["parse_errors"] += 1
                continue

            if not isinstance(obj, dict):
                continue
            stats["datasets_parsed"] += 1

            resources = obj.get("data")
            if not isinstance(resources, list):
                continue

            kept_resources = []
            for resource in resources:
                if not isinstance(resource, dict):
                    continue
                detected_format = detect_resource_format(resource)
                if detected_format in include_formats:
                    kept_resources.append(resource)

            if not kept_resources:
                continue

            filtered_obj = dict(obj)
            filtered_obj["data"] = kept_resources
            outfile.write(json.dumps(filtered_obj, ensure_ascii=False) + "\n")

            stats["datasets_kept"] += 1
            stats["resources_kept"] += len(kept_resources)

    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter JSONL dataset metadata to only datasets containing CSV "
            "or dataframe-friendly resource formats."
        )
    )
    parser.add_argument("--input", required=True, help="Input JSONL file path.")
    parser.add_argument("--output", required=True, help="Output JSONL file path.")
    parser.add_argument(
        "--formats",
        help=(
            "Optional comma-separated accepted formats/aliases. "
            "Example: csv,excel,tsv,json,parquet"
        ),
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary stats after processing.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    include_formats = parse_include_formats(args.formats)

    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    stats = filter_jsonl(input_path=input_path, output_path=output_path, include_formats=include_formats)

    if args.summary:
        print(f"Input: {input_path}")
        print(f"Output: {output_path}")
        print(f"Included formats: {', '.join(sorted(include_formats))}")
        print(f"Lines read: {stats['lines_read']}")
        print(f"Datasets parsed: {stats['datasets_parsed']}")
        print(f"Datasets kept: {stats['datasets_kept']}")
        print(f"Resources kept: {stats['resources_kept']}")
        print(f"Parse errors: {stats['parse_errors']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
