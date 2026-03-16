# Retrieval Baselines (NTCIR-15)

Unified runner for non-BM25 retrieval baselines:
- `dense` (`BAAI/bge-base-en-v1.5` + FAISS)
- `colbert` (`colbert-ir/colbertv2.0`, late interaction)
- `splade` (`naver/splade-cocondenser-ensembledistil`)
- `tapas` (`google/tapas-base` over serialized real table snippets)

## Install

```powershell
pip install -r .\requirements_retrieval_baselines_core.txt
pip install -r .\requirements_retrieval_baselines_dense.txt
pip install -r .\requirements_retrieval_baselines_colbert.txt
pip install -r .\requirements_retrieval_baselines_splade.txt
pip install -r .\requirements_retrieval_baselines_tapas.txt
```

## Run

From `NTCIR-15\online_phase`:

```powershell
python .\run_retrieval_baselines.py `
  --artifacts-dir .\artifacts_live `
  --query-set both `
  --baselines dense,colbert,splade,tapas `
  --topk 10 `
  --skip-unavailable
```

Runs are written to `artifacts_live\runs\{query_set}__{baseline_name}`.

## Evaluate

```powershell
python .\evaluate_runs.py `
  --qrels-path ..\evaluation_ready_pack\qrels_filtered_unprocessable.txt `
  --runs-dir .\artifacts_live\runs `
  --output-dir .\artifacts_live\eval `
  --fail-on-empty-run
```
