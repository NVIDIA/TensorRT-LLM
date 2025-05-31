from __future__ import annotations

from typing import Any, Dict, Protocol

from tensorrt_llm.bench.benchmark.tuning.dataclasses import (
    ScenarioSpecification, TuningConstraints, WorldConfig)
from tensorrt_llm.llmapi.llm_args import BatchingType


class ScenarioProtocol(Protocol):

    def get_settings(self, scenario: ScenarioSpecification, world: WorldConfig,
                     tuning: TuningConstraints) -> Dict[str, Any]:
        ...


class DefaultScenario(ScenarioProtocol):

    def get_settings(self, scenario: ScenarioSpecification, world: WorldConfig,
                     tuning: TuningConstraints) -> Dict[str, Any]:
        return {
            "model": scenario.environment.model,
            "pipeline_parallel_size": world.pp_size,
            "tensor_parallel_size": world.tp_size,
            "gpus_per_node": world.gpus_per_node,
            "moe_expert_parallel_size": world.ep_size,
            "moe_cluster_parallel_size": world.cluster_size,
            "trust_remote_code": True,
            "batching_type": "INFLIGHT",
        }


class MaxThroughputScenario(DefaultScenario):

    def get_settings(self, scenario: ScenarioSpecification, world: WorldConfig,
                     tuning: TuningConstraints) -> Dict[str, Any]:
        llm_args = super().get_settings(scenario, world, tuning)
        llm_args |= {
            "scheduler_config":
            dict(
                capacity_scheduler_policy="MAX_UTILIZATION",
                dynamic_batch_config=dict(
                    enable_batch_size_tuning=True,
                    enable_max_num_tokens_tuning=True,
                    dynamic_batch_moving_average_window=128,
                ),
            ),
            "kv_cache_config":
            dict(
                kv_cache_free_gpu_mem_fraction=scenario.
                kv_cache_free_gpu_mem_fraction,
                kv_cache_reuse=False,
            ),
            "extended_runtime_perf_knob_config":
            dict(
                multi_block_mode=True,
                enable_context_fmha_fp32_acc=False,
            ),
        }
        return llm_args
        llm_args = {
            "scheduler_config":
            self.settings_config.get_scheduler_config(),
            "model":
            model,
            "skip_tokenizer_init":
            True,
            "pipeline_parallel_size":
            self.world_config.pp_size,
            "tensor_parallel_size":
            self.world_config.tp_size,
            "gpus_per_node":
            self.world_config.gpus_per_node,
            "moe_expert_parallel_size":
            self.world_config.ep_size,
            "moe_cluster_parallel_size":
            self.world_config.cluster_size,
            "trust_remote_code":
            True,
            "kv_cache_config":
            self.settings_config.get_kvcache_config(),
            "enable_chunked_prefill":
            self.settings_config.chunking,
            "extended_runtime_perf_knob_config":
            self.performance_options.get_perf_config(),
            "decoding_config":
            self.decoding_config.get_decoding_config(),
            "batching_type":
            BatchingType.INFLIGHT,
            "max_batch_size":
            self.settings_config.max_batch_size,
            "max_num_tokens":
            self.settings_config.max_num_tokens,
        }
