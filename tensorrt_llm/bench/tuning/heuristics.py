from __future__ import annotations

from typing import Any, Dict, Protocol

from tensorrt_llm.bench.benchmark.tuning.dataclasses import (
    ScenarioSpecification, TuningConstraints, WorldConfig)


class ScenarioProtocol(Protocol):

    @classmethod
    def get_settings(cls, scenario: ScenarioSpecification, world: WorldConfig,
                     tuning: TuningConstraints) -> Dict[str, Any]:
        ...


class DefaultScenario(ScenarioProtocol):

    @classmethod
    def get_settings(cls, scenario: ScenarioSpecification, world: WorldConfig,
                     tuning: TuningConstraints) -> Dict[str, Any]:
        return {
            "model":
            scenario.environment.model,
            "pipeline_parallel_size":
            world.pp_size,
            "tensor_parallel_size":
            world.tp_size,
            "gpus_per_node":
            world.gpus_per_node,
            "moe_expert_parallel_size":
            world.ep_size,
            "moe_cluster_parallel_size":
            world.cluster_size,
            "trust_remote_code":
            True,
            "batching_type":
            "INFLIGHT",
            "kv_cache_config":
            dict(kv_cache_free_gpu_mem_fraction=scenario.
                 kv_cache_free_gpu_mem_fraction, ),
        }

