from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import List, Optional, Union

from pydantic import (BaseModel, Field, PositiveFloat, field_validator,
                      model_validator)

import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm.bench.dataclasses.enums import IFBSchedulingPolicy
from tensorrt_llm.llmapi.llm_utils import LlmArgs
from tensorrt_llm.models.modeling_utils import SpeculativeDecodingMode

SPECULATIVE_MAP = {
    SpeculativeDecodingMode.NONE: lambda *args: None,
    SpeculativeDecodingMode.MEDUSA: trtllm.DecodingMode.Medusa,
}


class RuntimeConfig(BaseModel):
    model: str
    engine_dir: Path
    sw_version: str
    settings_config: ExecutorSettingsConfig
    world_config: ExecutorWorldConfig
    decoding_config: DecodingConfig
    performance_options: PerformanceOptions

    def get_config(self) -> trtllm.ExecutorConfig:
        return trtllm.ExecutorConfig(
            batching_type=trtllm.BatchingType.INFLIGHT,
            decoding_config=self.decoding_config.get_decoding_config(),
            enable_chunked_context=self.settings_config.chunking,
            extended_runtime_perf_knob_config=self.performance_options.
            get_perf_config(),
            iter_stats_max_iterations=0,
            kv_cache_config=self.settings_config.get_kvcache_config(),
            max_batch_size=self.settings_config.max_batch_size,
            max_num_tokens=self.settings_config.max_num_tokens,
            parallel_config=self.world_config.get_parallel_config(),
            request_stats_max_iterations=0,
            scheduler_config=self.settings_config.get_scheduler_config(),
        )

    def get_llm_args(self) -> LlmArgs:
        return LlmArgs(
            scheduler_config=self.settings_config.get_scheduler_config(),
            model=self.engine_dir,
            skip_tokenizer_init=True,
            pipeline_parallel_size=self.world_config.pp_size,
            tensor_parallel_size=self.world_config.tp_size,
            trust_remote_code=True,
            kv_cache_config=self.settings_config.get_kvcache_config(),
            enable_chunked_prefill=self.settings_config.chunking,
            extended_runtime_perf_knob_config=self.performance_options.
            get_perf_config(),
            decoding_config=self.decoding_config.get_decoding_config(),
            batching_type=trtllm.BatchingType.INFLIGHT,
            max_batch_size=self.settings_config.max_batch_size,
            max_num_tokens=self.settings_config.max_num_tokens,
        )

    @model_validator(mode="after")
    def validate_full_config(self) -> RuntimeConfig:
        # TODO: Check engine to make sure it can support Medusa.
        return self


class PerformanceOptions(BaseModel):
    cuda_graphs: bool = False
    multi_block_mode: bool = True
    cuda_graph_cache_size: int = 1000

    def get_perf_config(self) -> trtllm.ExtendedRuntimePerfKnobConfig:
        config = trtllm.ExtendedRuntimePerfKnobConfig()
        config.cuda_graph_mode = self.cuda_graphs
        config.multi_block_mode = self.multi_block_mode
        config.cuda_graph_cache_size = self.cuda_graph_cache_size

        return config


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


class ExecutorWorldConfig(BaseModel):
    pp_size: int = 1
    tp_size: int = 1
    world_size: int = 1
    gpus_per_node: int = 8
    leader_mode: bool = False

    @model_validator(mode="after")
    def validate_world_size(self) -> ExecutorWorldConfig:
        parallel_world = self.pp_size * self.tp_size
        num_gpus = self.world_size * self.gpus_per_node
        valid_world = bool(num_gpus >= parallel_world)

        if not valid_world:
            raise ValueError(
                f"World configuration is invalid, TP * PP ({parallel_world})"
                "does not equal the total number of available GPUs"
                f"({num_gpus}).")

        return self

    def _get_tensorrt_llm_executor_worker_path(self) -> Path:
        module_path = find_spec("tensorrt_llm").loader.get_filename()
        exec_path = Path(module_path).parent / 'bin' / 'executorWorker'
        return exec_path.absolute()

    def get_parallel_config(self) -> trtllm.ParallelConfig:
        if self.leader_mode:
            comm_mode = trtllm.CommunicationMode.LEADER
            orchestrator_config = None
        else:
            comm_mode = trtllm.CommunicationMode.ORCHESTRATOR
            orchestrator_config = trtllm.OrchestratorConfig(
                True, str(self._get_tensorrt_llm_executor_worker_path()))

        return trtllm.ParallelConfig(
            trtllm.CommunicationType.MPI,
            comm_mode,
            orchestrator_config=orchestrator_config,
        )


class ExecutorSettingsConfig(BaseModel):
    chunking: bool = True
    scheduler_policy: IFBSchedulingPolicy = IFBSchedulingPolicy.MAX_UTILIZTION
    max_batch_size: int
    max_num_tokens: int
    kv_cache_percent: PositiveFloat = Field(default=.90, le=1.0)
    kv_cache_reuse: bool = False
    dynamic_max_batch_size: bool = True

    def get_dynamic_config(self) -> trtllm.DynamicBatchConfig:
        window_size = 128 if self.dynamic_max_batch_size else 0
        return trtllm.DynamicBatchConfig(self.dynamic_max_batch_size,
                                         window_size)

    def get_kvcache_config(self) -> trtllm.KvCacheConfig:
        return trtllm.KvCacheConfig(
            free_gpu_memory_fraction=self.kv_cache_percent,
            enable_block_reuse=False,
        )

    def get_scheduler_config(self) -> trtllm.SchedulerConfig:
        return trtllm.SchedulerConfig(
            capacity_scheduler_policy=self.scheduler_policy.value,
            context_chunking_policy=trtllm.ContextChunkingPolicy.
            FIRST_COME_FIRST_SERVED,
        )
