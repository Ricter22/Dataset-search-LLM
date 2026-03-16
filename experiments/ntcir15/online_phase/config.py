from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig:
    name: str
    use_semantic: bool
    use_query_optimization: bool


RUN_CONFIGS: list[RunConfig] = [
    RunConfig(name="baseline", use_semantic=False, use_query_optimization=False),
    RunConfig(name="semantic_only", use_semantic=True, use_query_optimization=False),
    RunConfig(name="queryopt_only", use_semantic=False, use_query_optimization=True),
    RunConfig(name="full", use_semantic=True, use_query_optimization=True),
]

RUN_CONFIG_BY_NAME: dict[str, RunConfig] = {cfg.name: cfg for cfg in RUN_CONFIGS}

