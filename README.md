# Dataset-search-LLM

Cleaned repository for the paper _A Tabular Data Findability Approach Using Retrieval-Augmented Large Language Models_.

The repository is organized around two experiment tracks:

- `experiments/ntcir15/`: scripted NTCIR-15 download, indexing, retrieval, and evaluation pipeline plus the retained machine-readable inputs used for the selected subset.
- `experiments/target/`: retained TARGET code for preprocessing, profile generation, pseudoquery generation, query decomposition, and recovery/replay.
- `legacy/`: archival notebooks and exploratory material kept for reference only.

## Repository Layout

- `experiments/ntcir15/offline_phase/db_population_evaluation_pack.py`
- `experiments/ntcir15/online_phase/run_online_phase.py`
- `experiments/ntcir15/online_phase/run_bm25_baseline.py`
- `experiments/target/preprocessing_utils.py`
- `experiments/target/rerun_target_recovery.py`
- `tests/test_bootstrap_significance.py`
- `scripts/extract_experimental_metadata.py`

## Notes

- Generated artifacts, vector stores, downloaded datasets, and local run outputs are intentionally not versioned.
- `legacy/` is not the main public entrypoint. It preserves notebook-based exploratory work only.
- TARGET benchmark data is not shipped in this repository. NTCIR-15 retains only lightweight selection and evaluation inputs in git.
- Use `environment.yml` as the base environment definition. Keep API credentials in a local `.env` file.
