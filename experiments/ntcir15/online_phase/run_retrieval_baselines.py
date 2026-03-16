#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path

    cwd_candidate = (Path.cwd() / path).resolve()
    script_candidate = (SCRIPT_DIR / path).resolve()

    if cwd_candidate.exists():
        return cwd_candidate
    return script_candidate


def safe_json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def load_topics(path: Path) -> dict[str, str]:
    topics: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split("\t", 1)
            if len(parts) != 2:
                continue
            qid = parts[0].strip()
            query_text = parts[1].strip()
            if qid and query_text:
                topics[qid] = query_text
    return dict(sorted(topics.items(), key=lambda x: x[0]))


def load_complex_queries(path: Path, expected_qids: set[str]) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid complex cache format: {path}")

    missing: list[str] = []
    out: dict[str, str] = {}
    for qid in sorted(expected_qids):
        entry = payload.get(qid)
        if not isinstance(entry, dict):
            missing.append(qid)
            continue
        complex_text = str(entry.get("complex", "")).strip()
        if not complex_text:
            missing.append(qid)
            continue
        out[qid] = complex_text

    if missing:
        raise RuntimeError("Complex cache missing queries: " + ", ".join(missing))
    return out


def load_dataset_info(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def build_text_document(row: dict[str, Any]) -> str:
    fields = [
        str(row.get("dataset_name", "")).strip(),
        str(row.get("domain", "")).strip(),
        str(row.get("data_profile", "")).strip(),
        str(row.get("semantic_profile", "")).strip(),
    ]
    return "\n".join(part for part in fields if part)


@dataclass(frozen=True)
class CorpusDoc:
    dataset_id: str
    text_document: str
    dataset_name: str
    file_path: str


def build_corpus_docs(rows: list[dict[str, Any]]) -> list[CorpusDoc]:
    docs: list[CorpusDoc] = []
    seen: set[str] = set()
    for row in rows:
        dataset_id = str(row.get("dataset_id", "")).strip()
        if not dataset_id or dataset_id in seen:
            continue
        text_document = build_text_document(row)
        if not text_document:
            continue
        docs.append(
            CorpusDoc(
                dataset_id=dataset_id,
                text_document=text_document,
                dataset_name=str(row.get("dataset_name", "")).strip(),
                file_path=str(row.get("file_path", "")).strip(),
            )
        )
        seen.add(dataset_id)
    return docs


def resolve_dataset_file(doc: CorpusDoc, datasets_root: Path) -> Path | None:
    raw_file_path = Path(doc.file_path) if doc.file_path else None
    if raw_file_path is not None and raw_file_path.exists():
        return raw_file_path
    if doc.dataset_name:
        fallback = (datasets_root / doc.dataset_name).resolve()
        if fallback.exists():
            return fallback
    return None


def truncate_text(value: str, max_chars: int) -> str:
    raw = str(value)
    if max_chars <= 0 or len(raw) <= max_chars:
        return raw
    return raw[:max_chars] + " ...[truncated]"


def _read_table_preview(path: Path, max_rows: int, max_cols: int) -> pd.DataFrame | None:
    ext = path.suffix.lower()
    try:
        if ext in {".csv", ".txt"}:
            df = pd.read_csv(
                path,
                nrows=max_rows,
                sep=None,
                engine="python",
                on_bad_lines="skip",
            )
        elif ext == ".gz":
            df = pd.read_csv(
                path,
                nrows=max_rows,
                compression="infer",
                sep=None,
                engine="python",
                on_bad_lines="skip",
            )
        elif ext in {".xls", ".xlsx"}:
            df = pd.read_excel(path, nrows=max_rows)
        elif ext == ".json":
            try:
                df = pd.read_json(path, lines=True)
            except Exception:  # noqa: BLE001
                df = pd.read_json(path)
            df = df.head(max_rows)
        else:
            return None
    except Exception:  # noqa: BLE001
        return None

    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    return df.iloc[:, : max(1, max_cols)]


def serialize_table(df: pd.DataFrame, max_chars: int) -> str:
    headers = [str(col) for col in df.columns]
    lines = ["Columns: " + " | ".join(headers)]
    for _, row in df.iterrows():
        values = [str(row[col]) for col in df.columns]
        lines.append(" | ".join(values))
    return truncate_text("\n".join(lines), max_chars=max_chars)


def build_table_serializations(
    docs: list[CorpusDoc],
    *,
    datasets_root: Path,
    cache_path: Path,
    max_rows: int,
    max_cols: int,
    max_chars: int,
) -> dict[str, str]:
    cache: dict[str, dict[str, Any]] = {}
    if cache_path.exists():
        try:
            loaded = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for did, row in loaded.items():
                    if isinstance(row, dict):
                        cache[str(did)] = row
        except Exception:  # noqa: BLE001
            cache = {}

    updated = False
    out: dict[str, str] = {}
    total = len(docs)
    for idx, doc in enumerate(docs, start=1):
        cached = cache.get(doc.dataset_id, {})
        cached_serialized = str(cached.get("serialized", "")).strip()
        if cached_serialized:
            out[doc.dataset_id] = cached_serialized
            if idx % 25 == 0 or idx == total:
                print(f"[tapas:cache] {idx}/{total} ready", flush=True)
            continue

        file_path = resolve_dataset_file(doc, datasets_root)
        if file_path is None:
            if idx % 25 == 0 or idx == total:
                print(f"[tapas:cache] {idx}/{total} ready", flush=True)
            continue

        preview = _read_table_preview(file_path, max_rows=max_rows, max_cols=max_cols)
        if preview is None:
            if idx % 25 == 0 or idx == total:
                print(f"[tapas:cache] {idx}/{total} ready", flush=True)
            continue

        serialized = serialize_table(preview, max_chars=max_chars).strip()
        if not serialized:
            if idx % 25 == 0 or idx == total:
                print(f"[tapas:cache] {idx}/{total} ready", flush=True)
            continue

        cache[doc.dataset_id] = {
            "dataset_name": doc.dataset_name,
            "source_file": str(file_path),
            "serialized": serialized,
            "generated_at_utc": utc_now(),
        }
        out[doc.dataset_id] = serialized
        updated = True
        if idx % 25 == 0 or idx == total:
            print(f"[tapas:cache] {idx}/{total} ready", flush=True)

    if updated:
        safe_json_dump(cache_path, cache)
    return out


def write_run_trec(
    run_dir: Path,
    run_id: str,
    ranking_by_qid: dict[str, list[tuple[str, float]]],
    valid_ids: set[str],
) -> tuple[Path, int, int]:
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / "run.trec"
    lines: list[str] = []
    invalid_ids = 0
    for qid in sorted(ranking_by_qid.keys()):
        rows = ranking_by_qid[qid]
        rank = 1
        for did, score in rows:
            dataset_id = str(did).strip()
            if not dataset_id:
                continue
            if dataset_id not in valid_ids:
                invalid_ids += 1
                continue
            lines.append(f"{qid} Q0 {dataset_id} {rank} {float(score):.6f} {run_id}")
            rank += 1
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out_path, len(lines), invalid_ids


class DenseEncoderMixin:
    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self.torch = None
        self.np = None
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self) -> None:
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.np = np
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _encode_texts(
        self,
        texts: list[str],
        *,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> Any:
        if not texts:
            return self.np.zeros((0, 1), dtype="float32")

        vectors: list[Any] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with self.torch.no_grad():
                outputs = self.model(**encoded)
                hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.detach().cpu().numpy().astype("float32"))
        return self.np.vstack(vectors)


class DenseBgeBackend(DenseEncoderMixin):
    key = "dense"
    run_name = "dense_bge"
    model_name = "BAAI/bge-base-en-v1.5"

    def __init__(self, device: str) -> None:
        super().__init__(model_name=self.model_name, device=device)
        import faiss

        self.faiss = faiss
        self.index = None
        self.doc_ids: list[str] = []

    def prepare(self, docs: list[CorpusDoc]) -> dict[str, Any]:
        texts = [d.text_document for d in docs]
        self.doc_ids = [d.dataset_id for d in docs]
        matrix = self._encode_texts(texts)
        if matrix.shape[0] <= 0:
            raise RuntimeError("Dense index received zero documents.")
        self.index = self.faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)
        return {"documents_indexed": int(matrix.shape[0]), "dim": int(matrix.shape[1])}

    def rank(
        self,
        queries: dict[str, str],
        *,
        topk: int,
    ) -> dict[str, list[tuple[str, float]]]:
        qids = sorted(queries.keys())
        qtexts = [queries[qid] for qid in qids]
        qvec = self._encode_texts(qtexts, max_length=256)
        if qvec.shape[0] <= 0:
            return {qid: [] for qid in qids}
        scores, indices = self.index.search(qvec, max(1, topk))
        out: dict[str, list[tuple[str, float]]] = {}
        for row, qid in enumerate(qids):
            ranked: list[tuple[str, float]] = []
            seen: set[str] = set()
            for col in range(indices.shape[1]):
                idx = int(indices[row, col])
                if idx < 0 or idx >= len(self.doc_ids):
                    continue
                did = self.doc_ids[idx]
                if did in seen:
                    continue
                ranked.append((did, float(scores[row, col])))
                seen.add(did)
            out[qid] = ranked
        return out


class ColbertLateInteractionBackend(DenseEncoderMixin):
    key = "colbert"
    run_name = "colbertv2"
    model_name = "colbert-ir/colbertv2.0"

    def __init__(self, device: str) -> None:
        super().__init__(model_name=self.model_name, device=device)
        self.doc_ids: list[str] = []
        self.doc_token_vectors: list[Any] = []

    def _encode_tokens(self, text: str, max_length: int) -> Any:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=max_length,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with self.torch.no_grad():
            outputs = self.model(**encoded)
            hidden = outputs.last_hidden_state[0]
            mask = encoded["attention_mask"][0].bool()
            hidden = hidden[mask]
            hidden = self.torch.nn.functional.normalize(hidden, p=2, dim=1)
        return hidden.detach().cpu()

    def prepare(self, docs: list[CorpusDoc]) -> dict[str, Any]:
        self.doc_ids = []
        self.doc_token_vectors = []
        for idx, doc in enumerate(docs, start=1):
            token_matrix = self._encode_tokens(doc.text_document, max_length=220)
            if token_matrix.numel() <= 0:
                continue
            self.doc_ids.append(doc.dataset_id)
            self.doc_token_vectors.append(token_matrix)
            if idx % 25 == 0 or idx == len(docs):
                print(f"[colbert:index] {idx}/{len(docs)}", flush=True)
        if not self.doc_ids:
            raise RuntimeError("ColBERT index received zero documents.")
        return {"documents_indexed": len(self.doc_ids)}

    def _score(self, query_vec: Any, doc_vec: Any) -> float:
        sim = self.torch.matmul(query_vec, doc_vec.transpose(0, 1))
        maxsim = sim.max(dim=1).values
        return float(maxsim.sum().item())

    def rank(
        self,
        queries: dict[str, str],
        *,
        topk: int,
    ) -> dict[str, list[tuple[str, float]]]:
        out: dict[str, list[tuple[str, float]]] = {}
        for i, qid in enumerate(sorted(queries.keys()), start=1):
            query_vec = self._encode_tokens(queries[qid], max_length=64)
            if query_vec.numel() <= 0:
                out[qid] = []
                continue
            scores: list[tuple[str, float]] = []
            for did, doc_vec in zip(self.doc_ids, self.doc_token_vectors):
                score = self._score(query_vec, doc_vec)
                scores.append((did, score))
            scores = sorted(scores, key=lambda x: (-x[1], x[0]))[: max(1, topk)]
            out[qid] = scores
            if i % 3 == 0 or i == len(queries):
                print(f"[colbert:search] {i}/{len(queries)}", flush=True)
        return out


class SpladeBackend:
    key = "splade"
    run_name = "splade"
    model_name = "naver/splade-cocondenser-ensembledistil"

    def __init__(self, device: str, max_terms: int = 256) -> None:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_terms = int(max_terms)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.doc_ids: list[str] = []
        self.doc_sparse: list[dict[int, float]] = []

    def _sparse_representation(self, text: str, max_length: int) -> dict[int, float]:
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with self.torch.no_grad():
            logits = self.model(**encoded).logits
            weights = self.torch.log1p(self.torch.relu(logits))
            masked = weights * encoded["attention_mask"].unsqueeze(-1)
            sparse_vec = masked.max(dim=1).values.squeeze(0)
            nonzero = self.torch.nonzero(sparse_vec > 0, as_tuple=False).squeeze(-1)
            if nonzero.numel() <= 0:
                return {}
            if nonzero.numel() > self.max_terms:
                vals = sparse_vec[nonzero]
                top_vals, top_idx = self.torch.topk(vals, k=self.max_terms)
                term_ids = nonzero[top_idx]
                values = top_vals
            else:
                term_ids = nonzero
                values = sparse_vec[nonzero]
            return {
                int(term_id.item()): float(val.item())
                for term_id, val in zip(term_ids, values)
            }

    @staticmethod
    def _dot_sparse(left: dict[int, float], right: dict[int, float]) -> float:
        if len(left) > len(right):
            left, right = right, left
        score = 0.0
        for term, val in left.items():
            other = right.get(term)
            if other is not None:
                score += val * other
        return score

    def prepare(self, docs: list[CorpusDoc]) -> dict[str, Any]:
        self.doc_ids = []
        self.doc_sparse = []
        for idx, doc in enumerate(docs, start=1):
            sparse = self._sparse_representation(doc.text_document, max_length=256)
            if not sparse:
                continue
            self.doc_ids.append(doc.dataset_id)
            self.doc_sparse.append(sparse)
            if idx % 25 == 0 or idx == len(docs):
                print(f"[splade:index] {idx}/{len(docs)}", flush=True)
        if not self.doc_ids:
            raise RuntimeError("SPLADE index received zero documents.")
        return {"documents_indexed": len(self.doc_ids), "max_terms": self.max_terms}

    def rank(
        self,
        queries: dict[str, str],
        *,
        topk: int,
    ) -> dict[str, list[tuple[str, float]]]:
        out: dict[str, list[tuple[str, float]]] = {}
        total = len(queries)
        for idx, qid in enumerate(sorted(queries.keys()), start=1):
            q_sparse = self._sparse_representation(queries[qid], max_length=128)
            if not q_sparse:
                out[qid] = []
                continue
            scores: list[tuple[str, float]] = []
            for did, d_sparse in zip(self.doc_ids, self.doc_sparse):
                score = self._dot_sparse(q_sparse, d_sparse)
                if score > 0.0:
                    scores.append((did, score))
            scores = sorted(scores, key=lambda x: (-x[1], x[0]))[: max(1, topk)]
            out[qid] = scores
            if idx % 3 == 0 or idx == total:
                print(f"[splade:search] {idx}/{total}", flush=True)
        return out


class TapasSerializedBackend:
    key = "tapas"
    run_name = "tapas_base"
    model_name = "google/tapas-base"

    def __init__(self, device: str) -> None:
        import numpy as np
        import torch
        from transformers import AutoModel, TapasTokenizer

        self.np = np
        self.torch = torch
        self.device = device
        self.tokenizer = TapasTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        self.index = None
        self.faiss = None
        self.doc_ids: list[str] = []
        self.doc_tables: list[pd.DataFrame] = []
        self.query_table = pd.DataFrame({"value": ["query"]})

        import faiss

        self.faiss = faiss

    @staticmethod
    def _deserialize_table(serialized: str) -> pd.DataFrame | None:
        lines = [line.strip() for line in str(serialized).splitlines() if line.strip()]
        if not lines:
            return None
        header_line = lines[0]
        if not header_line.lower().startswith("columns:"):
            return None
        raw_headers = header_line.split(":", 1)[1]
        headers = [part.strip() for part in raw_headers.split("|") if part.strip()]
        if not headers:
            return None
        rows: list[list[str]] = []
        for line in lines[1:]:
            cells = [part.strip() for part in line.split("|")]
            if not cells:
                continue
            if len(cells) < len(headers):
                cells += [""] * (len(headers) - len(cells))
            rows.append(cells[: len(headers)])
        if not rows:
            rows = [[""] * len(headers)]
        return pd.DataFrame(rows, columns=headers)

    def _pool_last_hidden_state(self, outputs: Any) -> Any:
        hidden = outputs.last_hidden_state
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(0)
        cls = hidden[:, 0, :]
        return self.torch.nn.functional.normalize(cls, p=2, dim=1)

    def _embed_table(self, table_df: pd.DataFrame) -> Any:
        encoded = self.tokenizer(
            table=table_df,
            queries=["table retrieval representation"],
            return_tensors="pt",
            truncation=True,
            padding="max_length",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with self.torch.no_grad():
            outputs = self.model(**encoded)
            pooled = self._pool_last_hidden_state(outputs)
        return pooled.detach().cpu().numpy().astype("float32")

    def _encode_query_texts(self, queries: list[str]) -> Any:
        if not queries:
            return self.np.zeros((0, 1), dtype="float32")

        vectors: list[Any] = []
        for query in queries:
            encoded = self.tokenizer(
                table=self.query_table,
                queries=[query],
                return_tensors="pt",
                truncation=True,
                padding="max_length",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with self.torch.no_grad():
                outputs = self.model(**encoded)
                pooled = self._pool_last_hidden_state(outputs)
            vectors.append(pooled.detach().cpu().numpy().astype("float32"))
        return self.np.vstack(vectors)

    def prepare(self, serialized_by_id: dict[str, str]) -> dict[str, Any]:
        self.doc_ids = []
        self.doc_tables = []
        for did in sorted(serialized_by_id.keys()):
            table_df = self._deserialize_table(serialized_by_id[did])
            if table_df is None or table_df.empty:
                continue
            self.doc_ids.append(did)
            self.doc_tables.append(table_df)

        matrix_parts: list[Any] = []
        for idx, table_df in enumerate(self.doc_tables, start=1):
            matrix_parts.append(self._embed_table(table_df))
            if idx % 20 == 0 or idx == len(self.doc_tables):
                print(f"[tapas:index] {idx}/{len(self.doc_tables)}", flush=True)

        if not matrix_parts:
            raise RuntimeError("TAPAS index received zero serialized tables.")
        matrix = self.np.vstack(matrix_parts)
        if matrix.shape[0] <= 0:
            raise RuntimeError("TAPAS index received zero serialized tables.")
        self.index = self.faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)
        return {"documents_indexed": int(matrix.shape[0]), "dim": int(matrix.shape[1])}

    def rank(
        self,
        queries: dict[str, str],
        *,
        topk: int,
    ) -> dict[str, list[tuple[str, float]]]:
        qids = sorted(queries.keys())
        qtexts = [queries[qid] for qid in qids]
        qvec = self._encode_query_texts(qtexts)
        if qvec.shape[0] <= 0:
            return {qid: [] for qid in qids}
        scores, indices = self.index.search(qvec, max(1, topk))
        out: dict[str, list[tuple[str, float]]] = {}
        for row, qid in enumerate(qids):
            ranked: list[tuple[str, float]] = []
            seen: set[str] = set()
            for col in range(indices.shape[1]):
                idx = int(indices[row, col])
                if idx < 0 or idx >= len(self.doc_ids):
                    continue
                did = self.doc_ids[idx]
                if did in seen:
                    continue
                ranked.append((did, float(scores[row, col])))
                seen.add(did)
            out[qid] = ranked
        return out


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        return "cpu"
    return "cpu"


def parse_baselines(raw: str) -> list[str]:
    values = [x.strip().lower() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("At least one baseline must be provided.")
    allowed = {"dense", "colbert", "splade", "tapas"}
    bad = [x for x in values if x not in allowed]
    if bad:
        raise ValueError(f"Unknown baselines: {bad}. Allowed={sorted(allowed)}")
    ordered: list[str] = []
    seen: set[str] = set()
    for item in values:
        if item in seen:
            continue
        ordered.append(item)
        seen.add(item)
    return ordered


def build_query_sets(
    topics: dict[str, str],
    *,
    query_set: str,
    complex_cache_path: Path,
) -> dict[str, dict[str, str]]:
    query_sets: dict[str, dict[str, str]] = {}
    if query_set in {"original", "both"}:
        query_sets["original"] = topics
    if query_set in {"complex", "both"}:
        query_sets["complex"] = load_complex_queries(complex_cache_path, set(topics.keys()))
    return query_sets


def create_backend(name: str, device: str) -> Any:
    if name == "dense":
        return DenseBgeBackend(device=device)
    if name == "colbert":
        return ColbertLateInteractionBackend(device=device)
    if name == "splade":
        return SpladeBackend(device=device)
    if name == "tapas":
        return TapasSerializedBackend(device=device)
    raise ValueError(f"Unsupported baseline: {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run non-BM25 retrieval baselines for NTCIR-15 with BM25-compatible artifacts."
    )
    parser.add_argument("--topics-path", default="../evaluation_ready_pack/topics.tsv")
    parser.add_argument("--complex-cache-path", default="./artifacts/complex_queries.json")
    parser.add_argument(
        "--dataset-info-path",
        default="../offline_phase/artifacts/dataset_info_evaluation.jsonl",
    )
    parser.add_argument(
        "--datasets-root",
        default="../evaluation_ready_pack/datasets",
        help="Root folder containing real tabular files used by TAPAS serialization.",
    )
    parser.add_argument(
        "--table-cache-path",
        default="./artifacts/table_serializations_cache.json",
    )
    parser.add_argument("--artifacts-dir", default="./artifacts_live")
    parser.add_argument("--query-set", choices=["original", "complex", "both"], default="both")
    parser.add_argument("--baselines", default="dense,colbert,splade,tapas")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--skip-unavailable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip unavailable backends (missing deps/model download failure) and continue.",
    )
    parser.add_argument("--table-max-rows", type=int, default=20)
    parser.add_argument("--table-max-cols", type=int, default=20)
    parser.add_argument("--table-max-chars", type=int, default=12000)
    return parser


def main() -> int:
    args = build_parser().parse_args()

    topics_path = resolve_path(args.topics_path)
    complex_cache_path = resolve_path(args.complex_cache_path)
    dataset_info_path = resolve_path(args.dataset_info_path)
    datasets_root = resolve_path(args.datasets_root)
    table_cache_path = resolve_path(args.table_cache_path)
    artifacts_dir = resolve_path(args.artifacts_dir)

    if not topics_path.exists():
        raise FileNotFoundError(f"topics-path not found: {topics_path}")
    if not dataset_info_path.exists():
        raise FileNotFoundError(f"dataset-info-path not found: {dataset_info_path}")
    if args.query_set in {"complex", "both"} and not complex_cache_path.exists():
        raise FileNotFoundError(f"complex-cache-path not found: {complex_cache_path}")
    if not datasets_root.exists():
        raise FileNotFoundError(f"datasets-root not found: {datasets_root}")

    baselines = parse_baselines(args.baselines)
    topics = load_topics(topics_path)
    if not topics:
        raise RuntimeError(f"No topics found in {topics_path}")
    query_sets = build_query_sets(
        topics,
        query_set=args.query_set,
        complex_cache_path=complex_cache_path,
    )

    rows = load_dataset_info(dataset_info_path)
    docs = build_corpus_docs(rows)
    if not docs:
        raise RuntimeError("Corpus is empty after parsing dataset info.")
    valid_ids = {d.dataset_id for d in docs}

    table_serializations: dict[str, str] = {}
    if "tapas" in baselines:
        print("[tapas] building/loading table serialization cache", flush=True)
        table_serializations = build_table_serializations(
            docs,
            datasets_root=datasets_root,
            cache_path=table_cache_path,
            max_rows=max(1, int(args.table_max_rows)),
            max_cols=max(1, int(args.table_max_cols)),
            max_chars=max(400, int(args.table_max_chars)),
        )
        print(
            f"[tapas] table cache ready for {len(table_serializations)} datasets",
            flush=True,
        )

    device = choose_device(args.device)
    runs_root = artifacts_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    global_summary: dict[str, Any] = {
        "run_started_at_utc": utc_now(),
        "topics_path": str(topics_path),
        "complex_cache_path": str(complex_cache_path),
        "dataset_info_path": str(dataset_info_path),
        "datasets_root": str(datasets_root),
        "table_cache_path": str(table_cache_path),
        "artifacts_dir": str(artifacts_dir),
        "baselines_requested": baselines,
        "query_set": args.query_set,
        "query_sets": list(query_sets.keys()),
        "topk": int(args.topk),
        "device": device,
        "skip_unavailable": bool(args.skip_unavailable),
        "corpus_size": len(docs),
        "table_serializations_count": len(table_serializations),
        "runs": {},
        "backend_status": {},
        "status": "running",
    }

    for baseline in baselines:
        print(f"[baseline:{baseline}] initializing", flush=True)
        backend = None
        backend_error = ""
        prepare_info: dict[str, Any] = {}
        try:
            backend = create_backend(baseline, device=device)
            if baseline == "tapas":
                prepare_info = backend.prepare(table_serializations)
            else:
                prepare_info = backend.prepare(docs)
            global_summary["backend_status"][baseline] = {
                "status": "ready",
                "prepare_info": prepare_info,
                "error": "",
            }
            print(f"[baseline:{baseline}] ready", flush=True)
        except Exception as exc:  # noqa: BLE001
            backend = None
            backend_error = f"{type(exc).__name__}: {exc}"
            global_summary["backend_status"][baseline] = {
                "status": "skipped_unavailable" if args.skip_unavailable else "failed",
                "prepare_info": {},
                "error": backend_error,
            }
            print(f"[baseline:{baseline}] unavailable: {backend_error}", flush=True)
            if not args.skip_unavailable:
                global_summary["status"] = "failed"
                global_summary["error"] = backend_error
                safe_json_dump(artifacts_dir / "retrieval_baselines_run_summary.json", global_summary)
                raise

        run_label = (
            backend.run_name
            if backend is not None and hasattr(backend, "run_name")
            else {
                "dense": "dense_bge",
                "colbert": "colbertv2",
                "splade": "splade",
                "tapas": "tapas_base",
            }[baseline]
        )

        for query_set_name, query_map in query_sets.items():
            run_id = f"{query_set_name}__{run_label}"
            run_dir = runs_root / run_id
            run_dir.mkdir(parents=True, exist_ok=True)

            if backend is None:
                run_summary = {
                    "run_id": run_id,
                    "baseline": baseline,
                    "backend_name": run_label,
                    "query_set": query_set_name,
                    "queries_total": len(query_map),
                    "corpus_size": len(docs),
                    "topk": int(args.topk),
                    "status": "skipped_unavailable",
                    "error": backend_error,
                    "generated_at_utc": utc_now(),
                    "run_trec_lines": 0,
                    "run_trec_path": "",
                    "invalid_dataset_ids_filtered": 0,
                    "prepare_info": {},
                }
                safe_json_dump(run_dir / "run_summary.json", run_summary)
                global_summary["runs"][run_id] = run_summary
                continue

            try:
                print(
                    f"[run:{run_id}] ranking {len(query_map)} queries",
                    flush=True,
                )
                ranking = backend.rank(query_map, topk=max(1, int(args.topk)))
                run_path, line_count, invalid_filtered = write_run_trec(
                    run_dir,
                    run_id,
                    ranking,
                    valid_ids,
                )
                run_summary = {
                    "run_id": run_id,
                    "baseline": baseline,
                    "backend_name": run_label,
                    "query_set": query_set_name,
                    "queries_total": len(query_map),
                    "corpus_size": len(docs),
                    "topk": int(args.topk),
                    "status": "ok",
                    "error": "",
                    "generated_at_utc": utc_now(),
                    "run_trec_lines": int(line_count),
                    "run_trec_path": str(run_path),
                    "invalid_dataset_ids_filtered": int(invalid_filtered),
                    "prepare_info": prepare_info,
                }
                safe_json_dump(run_dir / "run_summary.json", run_summary)
                global_summary["runs"][run_id] = run_summary
                print(
                    f"[run:{run_id}] completed with {line_count} lines",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {exc}"
                run_summary = {
                    "run_id": run_id,
                    "baseline": baseline,
                    "backend_name": run_label,
                    "query_set": query_set_name,
                    "queries_total": len(query_map),
                    "corpus_size": len(docs),
                    "topk": int(args.topk),
                    "status": "failed",
                    "error": err,
                    "generated_at_utc": utc_now(),
                    "run_trec_lines": 0,
                    "run_trec_path": "",
                    "invalid_dataset_ids_filtered": 0,
                    "prepare_info": prepare_info,
                }
                safe_json_dump(run_dir / "run_summary.json", run_summary)
                global_summary["runs"][run_id] = run_summary
                if not args.skip_unavailable:
                    global_summary["status"] = "failed"
                    global_summary["error"] = err
                    safe_json_dump(
                        artifacts_dir / "retrieval_baselines_run_summary.json",
                        global_summary,
                    )
                    raise RuntimeError(
                        f"Run failed for {run_id} and --skip-unavailable is disabled: {err}"
                    ) from exc

    statuses = [str(r.get("status", "")) for r in global_summary["runs"].values()]
    if statuses and all(x == "skipped_unavailable" for x in statuses):
        global_summary["status"] = "completed_with_only_skipped_runs"
    elif any(x == "failed" for x in statuses):
        global_summary["status"] = "completed_with_failures"
    else:
        global_summary["status"] = "completed"
    global_summary["run_finished_at_utc"] = utc_now()
    safe_json_dump(artifacts_dir / "retrieval_baselines_run_summary.json", global_summary)
    print(f"[retrieval-baselines] status={global_summary['status']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
