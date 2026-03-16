#!/usr/bin/env python3
"""Download dataset resources from filtered JSONL metadata with resume support."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, unquote
from urllib.request import Request, urlopen


TERMINAL_OK = "ok"
TERMINAL_FAILED = "failed"
TERMINAL_SKIPPED = "skipped"

MANIFEST_COLUMNS = [
    "dataset_id",
    "resource_index",
    "resource_url",
    "declared_format",
    "domain",
    "filename",
    "local_path",
    "status",
    "http_status",
    "bytes",
    "attempts",
    "error",
    "downloaded_at_utc",
]

RETRYABLE_HTTP = {408, 425, 429, 500, 502, 503, 504}
NON_RETRYABLE_HTTP = {400, 401, 403, 404, 405, 410, 422}

EXT_BY_FORMAT = {
    "csv": "csv",
    "excel": "xlsx",
    "tsv": "tsv",
    "json": "json",
    "parquet": "parquet",
    "feather": "feather",
    "orc": "orc",
    "hdf": "h5",
    "stata": "dta",
    "sas": "sas7bdat",
    "spss": "sav",
    "sqlite": "sqlite",
}


@dataclass(frozen=True)
class Task:
    dataset_id: str
    resource_index: int
    resource_url: str
    declared_format: str
    domain: str
    filename: str
    local_path: Path

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.dataset_id, str(self.resource_index), self.resource_url)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str, fallback: str) -> str:
    cleaned = value.strip()
    if not cleaned:
        cleaned = fallback
    cleaned = cleaned.replace("\\", "_").replace("/", "_")
    cleaned = re.sub(r"[:*?\"<>|]", "_", cleaned)
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned.strip("._")
    return cleaned or fallback


def infer_filename(resource: dict[str, Any], dataset_id: str, idx: int) -> str:
    url = str(resource.get("data_url", "")).strip()
    declared_fmt = str(resource.get("data_format", "")).strip().lower()
    url_path = unquote(urlparse(url).path)
    from_url = Path(url_path).name
    from_meta = str(resource.get("data_filename", "")).strip()
    preferred = from_url or from_meta

    ext = Path(preferred).suffix.lstrip(".").lower() if preferred else ""
    if not ext:
        ext = EXT_BY_FORMAT.get(declared_fmt, "bin")

    base = Path(preferred).stem if preferred else f"resource_{idx}"
    base = safe_name(base, f"resource_{idx}")

    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    return f"{idx:03d}_{base}_{digest}.{ext}"


def load_tasks(input_path: Path, out_dir: Path, limit: int | None = None) -> list[Task]:
    tasks: list[Task] = []
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            dataset_id = str(obj.get("id", "")).strip()
            resources = obj.get("data", [])
            if not isinstance(resources, list):
                continue

            for idx, resource in enumerate(resources):
                if not isinstance(resource, dict):
                    continue
                url = str(resource.get("data_url", "")).strip()
                if not url:
                    continue
                domain = urlparse(url).netloc.lower() or "unknown-domain"
                filename = infer_filename(resource, dataset_id, idx)
                local_path = out_dir / domain / dataset_id / filename
                task = Task(
                    dataset_id=dataset_id,
                    resource_index=idx,
                    resource_url=url,
                    declared_format=str(resource.get("data_format", "")).strip().lower(),
                    domain=domain,
                    filename=filename,
                    local_path=local_path,
                )
                tasks.append(task)
                if limit is not None and len(tasks) >= limit:
                    return tasks
    return tasks


def load_manifest_latest(manifest_path: Path) -> dict[tuple[str, str, str], dict[str, str]]:
    latest: dict[tuple[str, str, str], dict[str, str]] = {}
    if not manifest_path.exists():
        return latest
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = (row["dataset_id"], row["resource_index"], row["resource_url"])
            latest[key] = row
    return latest


def write_manifest(manifest_path: Path, rows: list[dict[str, Any]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class DomainRateLimiter:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = delay_seconds
        self.lock = threading.Lock()
        self.next_allowed: dict[str, float] = {}

    def wait(self, domain: str) -> None:
        if self.delay_seconds <= 0:
            return
        while True:
            with self.lock:
                now = time.monotonic()
                target = self.next_allowed.get(domain, now)
                if now >= target:
                    self.next_allowed[domain] = now + self.delay_seconds
                    return
                sleep_for = target - now
            time.sleep(min(sleep_for, 0.1))


def should_retry(http_status: int | None, error_message: str) -> bool:
    if http_status is not None:
        if http_status in NON_RETRYABLE_HTTP:
            return False
        if http_status in RETRYABLE_HTTP:
            return True
    transient_markers = ("timed out", "temporary", "connection reset", "connection aborted")
    em = error_message.lower()
    return any(marker in em for marker in transient_markers)


def download_with_retries(task: Task, retries: int, timeout: float, limiter: DomainRateLimiter) -> dict[str, Any]:
    last_err = ""
    last_status: int | None = None
    attempts = 0

    task.local_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = task.local_path.with_suffix(task.local_path.suffix + ".part")

    for attempt in range(1, retries + 1):
        attempts = attempt
        limiter.wait(task.domain)
        try:
            req = Request(task.resource_url, headers={"User-Agent": "dataset-downloader/1.0"})
            with urlopen(req, timeout=timeout) as resp, tmp_path.open("wb") as out:
                status = getattr(resp, "status", 200)
                total = 0
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
                    total += len(chunk)

            if total == 0 and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
                raise ValueError("Downloaded empty file")

            os.replace(tmp_path, task.local_path)
            return {
                "status": TERMINAL_OK,
                "http_status": status,
                "bytes": total,
                "attempts": attempts,
                "error": "",
            }
        except HTTPError as e:
            last_status = int(getattr(e, "code", 0) or 0) or None
            last_err = f"HTTPError: {e}"
        except URLError as e:
            last_status = None
            last_err = f"URLError: {e}"
        except Exception as e:  # noqa: BLE001
            last_status = None
            last_err = f"{type(e).__name__}: {e}"
        finally:
            tmp_path.unlink(missing_ok=True)

        if attempt < retries and should_retry(last_status, last_err):
            sleep_s = min(30.0, 0.5 * (2 ** (attempt - 1)))
            sleep_s *= 1.0 + ((attempt % 3) * 0.1)
            time.sleep(sleep_s)
        elif attempt < retries:
            break

    return {
        "status": TERMINAL_FAILED,
        "http_status": last_status if last_status is not None else "",
        "bytes": 0,
        "attempts": attempts,
        "error": last_err,
    }


def row_from_task(task: Task, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_id": task.dataset_id,
        "resource_index": task.resource_index,
        "resource_url": task.resource_url,
        "declared_format": task.declared_format,
        "domain": task.domain,
        "filename": task.filename,
        "local_path": str(task.local_path),
        "status": result["status"],
        "http_status": result.get("http_status", ""),
        "bytes": result.get("bytes", 0),
        "attempts": result.get("attempts", 0),
        "error": result.get("error", ""),
        "downloaded_at_utc": utc_now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download dataset resources from filtered JSONL metadata.")
    parser.add_argument("--input", required=True, help="Input JSONL file.")
    parser.add_argument("--out-dir", required=True, help="Directory for downloaded raw files.")
    parser.add_argument("--manifest", required=True, help="Manifest CSV path.")
    parser.add_argument("--workers", type=int, default=12, help="Concurrent worker threads (default: 12).")
    parser.add_argument("--retries", type=int, default=4, help="Max attempts per resource (default: 4).")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds (default: 60).")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N resources.")
    parser.add_argument(
        "--domain-delay-ms",
        type=int,
        default=200,
        help="Minimum delay between requests to same domain in ms (default: 200).",
    )
    parser.add_argument("--summary", action="store_true", help="Print summary.")
    parser.add_argument("--overwrite", action="store_true", help="Redownload even if successful already.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore prior manifest status.")

    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    manifest_path = Path(args.manifest)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    tasks = load_tasks(input_path=input_path, out_dir=out_dir, limit=args.limit)
    previous = {} if args.no_resume else load_manifest_latest(manifest_path)
    latest_rows: dict[tuple[str, str, str], dict[str, Any]] = dict(previous)

    to_download: list[Task] = []
    skipped = 0
    for task in tasks:
        old = previous.get(task.key)
        already_ok = bool(old and old.get("status") == TERMINAL_OK and Path(old.get("local_path", "")).exists())
        if already_ok and not args.overwrite:
            skipped += 1
            latest_rows[task.key] = row_from_task(
                task,
                {"status": TERMINAL_SKIPPED, "http_status": old.get("http_status", ""), "bytes": old.get("bytes", 0), "attempts": 0, "error": ""},
            )
        else:
            to_download.append(task)

    limiter = DomainRateLimiter(delay_seconds=max(args.domain_delay_ms, 0) / 1000.0)
    ok_count = 0
    fail_count = 0

    if to_download:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(download_with_retries, task, max(1, args.retries), max(1.0, args.timeout), limiter): task
                for task in to_download
            }
            for fut in as_completed(futures):
                task = futures[fut]
                result = fut.result()
                latest_rows[task.key] = row_from_task(task, result)
                if result["status"] == TERMINAL_OK:
                    ok_count += 1
                else:
                    fail_count += 1

    rows = [latest_rows[k] for k in sorted(latest_rows.keys())]
    write_manifest(manifest_path, rows)

    if args.summary:
        total = len(tasks)
        print(f"Discovered resources: {total}")
        print(f"Downloaded: {ok_count}")
        print(f"Skipped: {skipped}")
        print(f"Failed: {fail_count}")
        print(f"Manifest: {manifest_path}")
        print(f"Output dir: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
