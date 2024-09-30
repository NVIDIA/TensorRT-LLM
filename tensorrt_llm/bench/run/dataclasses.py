from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Any, List, Optional

from pydantic import (BaseModel, Field, PositiveFloat, computed_field,
                      model_validator)

import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm.bench.enums import IFBSchedulingPolicy


class RuntimeConfig(BaseModel):
    model: str
    engine_dir: Path
    sw_version: str
    settings_config: ExecutorSettingsConfig
    world_config: ExecutorWorldConfig

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
        )


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

    def register_event(self, is_error: bool, is_final: bool, timestamp: int,
                       tokens: List[int]) -> None:
        if is_final:
            self.end_timestamp = timestamp
        elif self.first_token_timestamp == -1:
            self.first_token_timestamp = timestamp

        if is_error:
            self.error_tokens += 1

        self.tokens += tokens

    @computed_field
    def num_output_tokens(self) -> int:
        return len(self.tokens)

    @computed_field
    def num_generated_tokens(self) -> int:
        return self.num_output_tokens - 1

    @computed_field
    def generation_time(self) -> int:
        return self.end_timestamp - self.time_to_first_token

    @computed_field
    def time_to_first_token(self) -> int:
        return self.first_token_timestamp - self.start_timestamp

    @computed_field
    def intertoken_latency(self) -> float:
        return (self.end_timestamp -
                self.first_token_timestamp) / self.num_generated_tokens

    @computed_field
    def end_to_end_latency(self) -> int:
        return self.end_timestamp - self.start_timestamp

    @computed_field
    def total_token_throughput(self) -> float:
        return self.num_output_tokens / self.end_to_end_latency

    @computed_field
    def output_token_throughput(self) -> float:
        return self.num_output_tokens / self.generation_time


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
    total_latency_ns: float
    total_output_tokens: int
    total_input_tokens: int
    num_requests: int
    issue_rate_ns: float

    request_percentiles: Optional[PercentileStats] = None
    token_percentiles: Optional[PercentileStats] = None
    itl_percentiles: Optional[PercentileStats] = None
    ttft_percentiles: Optional[PercentileStats] = None

    @computed_field
    def token_throughput_ns(self) -> float:
        return float(self.total_output_tokens) / self.total_latency_ns

    @computed_field
    def request_throughput_ns(self) -> float:
        return float(self.num_requests) / self.total_latency_ns

    @computed_field
    def average_input_length(self) -> float:
        return float(self.total_input_tokens) / self.num_requests

    @computed_field
    def average_output_length(self) -> float:
        return float(self.total_output_tokens) / self.num_requests
