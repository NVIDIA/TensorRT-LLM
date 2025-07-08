from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Protocol

from tensorrt_llm.bench.dataclasses.scenario import ScenarioSpecification


class HueristicProtocol(Protocol):

    @classmethod
    def get_settings(cls, scenario: ScenarioSpecification) -> Dict[str, Any]:
        ...


class DefaultLlmHeuristic(HueristicProtocol):
    """Default settings for the TensorRT backend."""
    @classmethod
    def get_settings(cls, scenario: ScenarioSpecification) -> Dict[str, Any]:
        print(scenario)
        world = scenario.world
        settings = {
            "model":
            scenario.environment.model,
            "pipeline_parallel_size":
            world.pp,
            "tensor_parallel_size":
            world.tp,
            "gpus_per_node":
            world.gpus_per_node,
            "moe_expert_parallel_size":
            world.ep,
            "moe_cluster_parallel_size":
            world.cluster_size,
            "trust_remote_code":
            True,
            "batching_type":
            "INFLIGHT",
            "kv_cache_config": {
                "kv_cache_free_gpu_mem_fraction": scenario.llm_config.kv_cache_free_gpu_mem_fraction
            }
        }
        return defaultdict(dict, settings)