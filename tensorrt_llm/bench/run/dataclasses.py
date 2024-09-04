from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Any, List

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


class ResponseRecord(BaseModel):
    request_id: int
    timestamp: float
    output_tokens: List[int]
    is_final: bool
    has_error: bool


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
        return cls(
            p50=values[int(length * 0.50)],
            p95=values[int(length * 0.95)],
            p99=values[int(length * 0.99)],
            average=float(sum(values)) / length,
            minimum=min(values),
            maximum=max(values),
        )


class RequestStats(BaseModel):
    request_id: int
    input_tokens: int
    time_log: List[float] = Field(default_factory=list, init=False)
    error_responses: int = Field(default=0, init=False)
    num_responses: int = Field(default=0, init=False)
    num_tokens: int = Field(default=0, init=False)

    @computed_field
    def first_token_latency(self) -> float:
        try:
            return self.time_log[1] - self.time_log[0]
        except IndexError:
            return 0

    @computed_field
    def request_latency(self) -> float:
        return max(self.time_log) - min(self.time_log)

    def register_event(self, is_error: bool, is_response: bool,
                       timestamp: float, num_tokens: int) -> None:
        self.time_log.append(timestamp)
        self.error_responses += 1 if is_error else 0
        self.num_responses += 1 if is_response else 0
        self.num_tokens += num_tokens


class BenchmarkStatistics(BaseModel):
    total_latency_ns: float
    total_output_tokens: int
    total_input_tokens: int
    num_requests: int
    issue_rate_ns: float

    request_percentiles: PercentileStats = None
    token_percentiles: PercentileStats = None

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
