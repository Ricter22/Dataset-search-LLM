# NTCIR-15

This directory contains the retained scripted NTCIR-15 pipeline.

- `offline_phase/`: dataset profiling, pseudoquery generation, and Chroma indexing code.
- `online_phase/`: retrieval, reranking, evaluation, and baseline scripts.
- Top-level `*.py`, `*.jsonl`, `*.tsv`, and `*.txt` files: selected queries, qrels, manifests, and subset-building utilities kept in git.
- `evaluation_ready_pack/` and `eval_subset/`: lightweight evaluation inputs and selection metadata. Downloaded datasets and generated indexes are not versioned.

Generated run outputs, local vector stores, and downloaded dataset directories are intentionally excluded from git.
