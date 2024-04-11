from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, computed_field
from utils import VALID_MODELS
from utils.enums import ComputeDtypeEnum, KVCacheDtypeEnum, QuantizationAlgo


class BenchmarkResults(BaseModel):
    """High level report out for a benchmark."""

    benchmark_cmd: str = ""
    binary: str
    build_cmd: str = ""
    first_token_latency: float
    inflight_batching: bool
    kv_mem_fraction: float
    latency_units: str
    max_batch_size: int
    max_tokens_in_cache: int
    model: VALID_MODELS
    peak_gpu_mem_units: str
    peak_gpu_mem: float
    scheduler: Literal["Static", "No evict", "Max Utilization"]
    throughput_units: str
    throughput: float
    time_per_output_token: float
    total_input_tokens: int
    total_latency: float
    total_output_tokens: int

    def get_summary(self, config: BenchmarkConfig) -> str:
        """Generate the summary information.

        Args:
            config (BenchmarkConfig): Configuration for the run that generated
            this result.

        Returns:
            str: Summary output for printing.
        """
        return (
            "===========================================================\n"
            "= METADATA\n"
            "===========================================================\n"
            f"Model:\t\t\t{config.model}\n"
            f"TP Size:\t\t{config.tensor_parallel}\n"
            f"PP Size:\t\t{config.pipeline_parallel}\n"
            f"Scheduling Policy:\t{self.scheduler}\n"
            f"In-flight Batcher?:\t{self.inflight_batching}\n"
            f"Dtype:\t\t\t{config.dtype.value}\n"
            f"KV Cache Dtype:\t\t{config.cache_dtype.value}\n"
            f"KV Cache Size (tokens):\t{self.max_tokens_in_cache}\n"
            f"Quantization:\t\t{config.quantization.value}\n"
            f"KV Memory Percentage:\t{self.kv_mem_fraction * 100}%\n"
            f"\n"
            "===========================================================\n"
            "= ENGINE DETAILS\n"
            "===========================================================\n"
            f"Engine Directory:\t{config.engine_path}\n"
            f"Max Batch Size:\t\t{self.max_batch_size}\n"
            f"Total Input Length:\t{self.total_input_tokens}\n"
            f"Total Output Length:\t{self.total_input_tokens}\n"
            f"\n"
            "===========================================================\n"
            "= STATISTICS\n"
            "===========================================================\n"
            f"Throughput ({self.throughput_units}):\t{self.throughput}\n"
            f"Total Latency ({self.latency_units}):\t\t{self.total_latency}\n"
            f"First Token Latency ({self.latency_units}):\t{self.first_token_latency}\n"
            f"Token-to-token Latency ({self.latency_units}):\t{self.time_per_output_token}\n"
            f"Peak GPU Memory Usage ({self.peak_gpu_mem_units}):\t{self.peak_gpu_mem}\n"
            f"\n"
            "===========================================================\n"
            "= COMMANDS\n"
            "===========================================================\n"
            f"Build: {self.build_cmd}\n"
            f"Benchmark: {self.benchmark_cmd}\n")


class BenchmarkConfig(BaseModel):
    """Basic configuration of a benchmark."""

    model: VALID_MODELS
    workspace: Path
    dtype: ComputeDtypeEnum
    cache_dtype: KVCacheDtypeEnum
    quantization: QuantizationAlgo
    tensor_parallel: int
    pipeline_parallel: int

    @computed_field
    def engine_path(self) -> Path:
        """Path to the engine workspace."""
        return Path(self.workspace.absolute(), self.model.lower())

    @computed_field
    def world_size(self) -> int:
        """Total world size needed to run the model."""
        return self.tensor_parallel * self.pipeline_parallel
