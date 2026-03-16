# TARGET

This directory keeps the TARGET-side code that is still useful for the paper-facing repository.

- `preprocessing_utils.py`: existing preprocessing logic for table cleanup, data profiles, semantic profiles, pseudoquery generation, background summaries, and query decomposition.
- `rerun_target_recovery.py`: existing recovery/replay script for TARGET evaluation runs.

Notebook wrappers and exploratory executions were moved to `legacy/target/`. Generated TARGET artifacts such as local Chroma stores, cached JSON outputs, and recovery result folders are intentionally not versioned.
