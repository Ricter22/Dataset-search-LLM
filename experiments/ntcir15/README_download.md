# Dataset Download Workflow

This workflow downloads all resources listed in the filtered JSONL file and tracks outcomes in a CSV manifest.

## Script

- `download_datasets.py`

## Quick test (first 20 resources)

```powershell
python .\download_datasets.py `
  --input ".\data_search_e_collection_tabular.jsonl" `
  --out-dir ".\datasets_raw" `
  --manifest ".\download_manifest.csv" `
  --limit 20 `
  --summary
```

## Full run

```powershell
python .\download_datasets.py `
  --input ".\data_search_e_collection_tabular.jsonl" `
  --out-dir ".\datasets_raw" `
  --manifest ".\download_manifest.csv" `
  --workers 12 `
  --retries 4 `
  --timeout 60 `
  --domain-delay-ms 200 `
  --summary
```

## Resume behavior

- By default, rows that were previously successful and whose file still exists are skipped.
- Use `--overwrite` to force redownload.
- Use `--no-resume` to ignore previous manifest state.

## Manifest columns

- `dataset_id`
- `resource_index`
- `resource_url`
- `declared_format`
- `domain`
- `filename`
- `local_path`
- `status` (`ok`, `failed`, `skipped`)
- `http_status`
- `bytes`
- `attempts`
- `error`
- `downloaded_at_utc`
