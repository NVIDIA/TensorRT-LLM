from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Any, List, Optional, Union

from pydantic import (BaseModel, Field, PositiveFloat, computed_field,
                      field_validator, model_validator)

import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm.bench.enums import IFBSchedulingPolicy
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
            scheduler_config=self.settings_config.get_scheduler_config(),
            kv_cache_config=self.settings_config.get_kvcache_config(),
            parallel_config=self.world_config.get_parallel_config(),
            batching_type=trtllm.BatchingType.INFLIGHT,
            iter_stats_max_iterations=0,
            request_stats_max_iterations=0,
            max_batch_size=self.settings_config.max_batch_size,
            max_num_tokens=self.settings_config.max_num_tokens,
            enable_chunked_context=self.settings_config.chunking,
            extended_runtime_perf_knob_config=self.performance_options.
            get_perf_config(),
            decoding_config=self.decoding_config.get_decoding_config(),
        )

    @model_validator(mode="after")
    def validate_full_config(self) -> RuntimeConfig:
        # TODO: Check engine to make sure it can support Medusa.
        return self


class PerformanceOptions(BaseModel):
    cuda_graphs: bool = False
    multi_block_mode: bool = False
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

    def get_kvcache_config(self) -> trtllm.KvCacheConfig:
        return trtllm.KvCacheConfig(
            free_gpu_memory_fraction=self.kv_cache_percent, )

    def get_scheduler_config(self) -> trtllm.SchedulerConfig:
        return trtllm.SchedulerConfig(
            capacity_scheduler_policy=self.scheduler_policy.value,
            context_chunking_policy=trtllm.ContextChunkingPolicy.
            FIRST_COME_FIRST_SERVED,
        )


class RequestRecord(BaseModel):
    id: int = -1
    num_input_tokens: int = -1
    tokens: List[int] = []
    error_tokens: int = 0
    start_timestamp: int = -1
    first_token_timestamp: int = -1
    end_timestamp: int = -1
    decode_iteration: int = 0

    def register_event(self, is_error: bool, is_final: bool, timestamp: int,
                       decoding_iter: int, tokens: List[int]) -> None:
        if is_final:
            self.end_timestamp = timestamp
        elif self.first_token_timestamp == -1:
            self.first_token_timestamp = timestamp

        if is_error:
            self.error_tokens += 1

        self.tokens += tokens
        self.decode_iteration = decoding_iter

    @computed_field
    def num_output_tokens(self) -> int:
        return len(self.tokens)

    @computed_field
    def num_generated_tokens(self) -> int:
        return self.num_output_tokens - 1

    @computed_field
    def generation_time(self) -> int:
        return self.end_to_end_latency - self.time_to_first_token

    @computed_field
    def time_to_first_token(self) -> int:
        return (self.first_token_timestamp -
                self.start_timestamp if self.first_token_timestamp > 0 else 0.0)

    @computed_field
    def intertoken_latency(self) -> float:
        return ((self.end_timestamp - self.first_token_timestamp) /
                self.num_generated_tokens
                if self.num_generated_tokens > 0 else 0.0)

    @computed_field
    def end_to_end_latency(self) -> int:
        return self.end_timestamp - self.start_timestamp

    @computed_field
    def total_token_throughput(self) -> float:
        return self.num_output_tokens / self.end_to_end_latency

    @computed_field
    def output_token_throughput(self) -> float:
        return (self.num_generated_tokens / self.generation_time)


class PercentileStats(BaseModel):
    p50: float
    p95: float
    p99: float
    minimum: float
    maximum: float
    average: float

    @classmethod
    def from_iterable(cls, values: List[Any]) -> PercentileStats:
        length = len(values)
        sorted_values = sorted(values)
        return cls(
            p50=sorted_values[int(length * 0.50)],
            p95=sorted_values[int(length * 0.95)],
            p99=sorted_values[int(length * 0.99)],
            average=float(sum(values)) / length,
            minimum=min(values),
            maximum=max(values),
        )


class BenchmarkStatistics(BaseModel):
    # Time-related Properties
    total_latency_ns: float

    # Token-related Properties
    total_output_tokens: int
    total_input_tokens: int

    # General Information
    num_requests: int
    issue_rate_ns: float

    # Speculative Information
    acceptance_rate: float

    # Percentile-related Statistics
    request_latency_percentiles: Optional[PercentileStats] = None
    token_percentiles: Optional[PercentileStats] = None
    itl_percentiles: Optional[PercentileStats] = None
    ttft_percentiles: Optional[PercentileStats] = None
    generation_tp_percentiles: Optional[PercentileStats] = None
    generation_latency_percentiles: Optional[PercentileStats] = None
    acceptance_percentiles: Optional[PercentileStats] = None

    @computed_field
    def generation_tokens(self) -> int:
        return int(self.total_output_tokens - self.num_requests)

    @computed_field
    def token_throughput_ns(self) -> float:
        return float(self.total_output_tokens) / self.total_latency_ns

    @computed_field
    def generation_token_throughput_ns(self) -> float:
        return self.generation_tp_percentiles.average

    @computed_field
    def request_throughput_ns(self) -> float:
        return float(self.num_requests) / self.total_latency_ns

    @computed_field
    def average_input_length(self) -> float:
        return float(self.total_input_tokens) / self.num_requests

    @computed_field
    def average_output_length(self) -> float:
        return float(self.total_output_tokens) / self.num_requests
