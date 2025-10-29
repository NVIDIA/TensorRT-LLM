from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import (BaseModel, Field, PositiveFloat, field_validator,
                      model_validator)

import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm._torch.pyexecutor.config import PyTorchConfig
from tensorrt_llm.llmapi import (BatchingType, CapacitySchedulerPolicy,
                                 ContextChunkingPolicy, DynamicBatchConfig,
                                 ExtendedRuntimePerfKnobConfig, KvCacheConfig,
                                 SchedulerConfig)
from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_options
from tensorrt_llm.models.modeling_utils import SpeculativeDecodingMode

SPECULATIVE_MAP = {
    SpeculativeDecodingMode.NONE: lambda *args: None,
    SpeculativeDecodingMode.MEDUSA: trtllm.DecodingMode.Medusa,
}


class RuntimeConfig(BaseModel):
    model: str
    model_path: Optional[Path] = None
    engine_dir: Optional[Path] = None
    sw_version: str
    settings_config: ExecutorSettingsConfig
    # TODO: this is a dict corresponding to the Mapping class, the type should be
    # changed to Mapping after the Mapping class is migrated to a Pydantic model.
    mapping: Dict[str, Any]
    decoding_config: Optional[DecodingConfig] = None
    performance_options: PerformanceOptions
    backend: Literal["pytorch", "_autodeploy", None] = None
    extra_llm_api_options: Optional[str] = None
    iteration_log: Optional[Path] = None

    def get_llm_args(self) -> Dict:
        model = self.engine_dir or self.model_path or self.model

        llm_args = {
            "scheduler_config":
            self.settings_config.get_scheduler_config(),
            "model":
            model,
            "skip_tokenizer_init":
            True,
            "pipeline_parallel_size":
            self.mapping["pp_size"],
            "tensor_parallel_size":
            self.mapping["tp_size"],
            "gpus_per_node":
            self.mapping["gpus_per_node"],
            "moe_expert_parallel_size":
            self.mapping["moe_ep_size"],
            "moe_cluster_parallel_size":
            self.mapping["moe_cluster_size"],
            "trust_remote_code":
            True,
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

        backend_config_map = {
            "pytorch": self.performance_options.get_pytorch_perf_config,
            "_autodeploy": self.performance_options.get_autodeploy_perf_config
        }

        if self.backend in backend_config_map:
            llm_args.update(backend_config_map[self.backend]())

        kv_cache_config = self.settings_config.get_kvcache_config().__dict__
        backend_cache_config = llm_args.pop("kv_cache_config", {})
        llm_args["kv_cache_config"] = backend_cache_config | kv_cache_config

        updated_llm_args = update_llm_args_with_extra_options(
            llm_args, self.extra_llm_api_options)

        if self.backend == "pytorch":
            cuda_graph_config = updated_llm_args.pop(
                "cuda_graph_config", llm_args["cuda_graph_config"])
            if cuda_graph_config:
                # Use runtime max_batch_size as cuda_graph_config.max_batch_size
                # if both max_batch_size and batch_sizes are not set.
                batch_sizes_set = cuda_graph_config.get("batch_sizes",
                                                        None) is not None
                max_batch_size_set = cuda_graph_config.get(
                    "max_batch_size", None) is not None
                if not batch_sizes_set and not max_batch_size_set:
                    cuda_graph_config[
                        "max_batch_size"] = self.settings_config.max_batch_size
            updated_llm_args["cuda_graph_config"] = cuda_graph_config

        return updated_llm_args

    @model_validator(mode="after")
    def validate_full_config(self) -> RuntimeConfig:
        # TODO: Check engine to make sure it can support Medusa.
        return self


@dataclass
class PerformanceOptions:
    cuda_graphs: bool = False
    multi_block_mode: bool = True
    cuda_graph_cache_size: int = 1000
    pytorch_config: Dict[str, Any] = Field(default_factory=dict)

    def get_perf_config(self) -> ExtendedRuntimePerfKnobConfig:
        config = ExtendedRuntimePerfKnobConfig()
        config.cuda_graph_mode = self.cuda_graphs
        config.multi_block_mode = self.multi_block_mode
        config.cuda_graph_cache_size = self.cuda_graph_cache_size

        return config

    def get_pytorch_perf_config(self) -> PyTorchConfig:
        return self.pytorch_config

    def get_autodeploy_perf_config(self) -> Dict:
        AutoDeployPerfConfig = dict
        ad_config = AutoDeployPerfConfig()
        return ad_config


class DecodingConfig(BaseModel):
    medusa_choices: Optional[List[List[int]]] = None
    decoding_mode: SpeculativeDecodingMode = SpeculativeDecodingMode.NONE

    @field_validator("decoding_mode")
    @classmethod
    def decoding_mode_validator(
        cls, value: Union[str, int,
                          SpeculativeDecodingMode]) -> SpeculativeDecodingMode:
        return SpeculativeDecodingMode(value)

    @model_validator(mode="after")
    def validate_speculative_decoding(self) -> DecodingConfig:
        if self.medusa_choices and self.decoding_mode != SpeculativeDecodingMode.MEDUSA:
            raise RuntimeError(
                "Attempting to use set Medusa choices with a non-Medusa engine."
                " Verify that you are using a Medusa engine.")

        return self

    def get_decoding_config(self) -> trtllm.DecodingConfig:
        """Create a populated TRT-LLM DecodingConfig."""
        kwargs = {"decoding_mode": SPECULATIVE_MAP[self.decoding_mode]()}

        if self.medusa_choices is not None:
            kwargs["medusa_choices"] = self.medusa_choices

        return trtllm.DecodingConfig(**kwargs)


class ExecutorSettingsConfig(BaseModel):
    chunking: bool = True
    scheduler_policy: CapacitySchedulerPolicy = Field(
        default=CapacitySchedulerPolicy.MAX_UTILIZATION)
    max_batch_size: int
    max_num_tokens: int
    kv_cache_percent: PositiveFloat = Field(default=.90, le=1.0)
    kv_cache_reuse: bool = False
    dynamic_max_batch_size: bool = True
    dynamic_max_num_tokens: bool = False  # Will enable after more validation.

    def get_dynamic_config(self) -> DynamicBatchConfig:
        window_size = 128 if self.dynamic_max_batch_size else 0
        return DynamicBatchConfig(
            enable_batch_size_tuning=self.dynamic_max_batch_size,
            enable_max_num_tokens_tuning=self.dynamic_max_num_tokens,
            dynamic_batch_moving_average_window=window_size,
        )

    def get_kvcache_config(self) -> KvCacheConfig:
        return KvCacheConfig(
            free_gpu_memory_fraction=self.kv_cache_percent,
            enable_block_reuse=False,
        )

    def get_scheduler_config(self) -> SchedulerConfig:
        if self.chunking:
            return SchedulerConfig(
                capacity_scheduler_policy=self.scheduler_policy,
                context_chunking_policy=ContextChunkingPolicy.
                FIRST_COME_FIRST_SERVED,
                dynamic_batch_config=self.get_dynamic_config(),
            )
        else:
            return SchedulerConfig(
                capacity_scheduler_policy=self.scheduler_policy,
                dynamic_batch_config=self.get_dynamic_config())
