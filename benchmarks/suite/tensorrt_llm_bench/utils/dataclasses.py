from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional, Union, get_args

from pydantic import (BaseModel, Field, ValidationError, computed_field,
                      field_validator, model_validator)
from transformers import AutoConfig
from utils import VALID_MODELS, VALID_SCHEDULING_POLICIES
from utils.enums import (ComputeDtypeEnum, KVCacheDtypeEnum, ModelArchitecture,
                         QuantizationAlgo)


class InferenceRequest(BaseModel):
    task_id: int
    prompt: Optional[str] = None
    output_tokens: int
    logits: Optional[List[int]] = None

    @model_validator(mode="after")
    def verify_prompt_and_logits(self) -> InferenceRequest:
        if self.prompt is None and self.logits is None:
            raise ValueError(
                f"Both prompt and logits for {self.task_id} are both None.")
        return self


class DatasetMetadata(BaseModel):
    max_isl: int
    max_osl: int
    num_requests: int


class BenchmarkResults(BaseModel):
    """High level report out for a benchmark."""

    benchmark_cmd: str = ""
    binary: str = ""
    build_cmd: str = ""
    first_token_latency: float
    inflight_batching: bool
    kv_mem_fraction: float
    latency_units: str
    max_batch_size: int
    max_tokens: int = 0
    model: Union[VALID_MODELS, Path]
    peak_gpu_mem_units: str
    peak_gpu_mem: float
    scheduler: Literal["Static", "No Evict", "Max Utilization"]
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
            f"Quantization:\t\t{config.quantization.value}\n"
            f"KV Memory Percentage:\t{self.kv_mem_fraction * 100}%\n"
            f"\n"
            "===========================================================\n"
            "= ENGINE DETAILS\n"
            "===========================================================\n"
            f"Engine Directory:\t{config.engine_path}\n"
            f"Max Batch Size:\t\t{self.max_batch_size}\n"
            f"Total Input Length:\t{self.total_input_tokens}\n"
            f"Total Output Length:\t{self.total_output_tokens}\n"
            f"Max Tokens:\t\t{self.max_tokens}\n"
            f"\n"
            "===========================================================\n"
            "= STATISTICS\n"
            "===========================================================\n"
            f"Throughput ({self.throughput_units}):\t{self.throughput}\n"
            f"Total Latency ({self.latency_units}):"
            f"\t\t{self.total_latency * 1000.0:.4f}\n"
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

    model: Union[VALID_MODELS, Path]
    workspace: Path
    max_batch_size: int
    dtype: ComputeDtypeEnum
    cache_dtype: KVCacheDtypeEnum
    quantization: QuantizationAlgo
    tensor_parallel: int
    pipeline_parallel: int
    max_tokens: int = 0
    kv_cache_mem_percentage: float = .9
    engine_isl: int = 0
    engine_osl: int = 0
    chunking: bool = False
    build_overrides: List[str] = Field(default_factory=list)
    scheduling_policy: Literal[VALID_SCHEDULING_POLICIES] = "static"

    @field_validator("model", mode="before")
    @classmethod
    def validate_model(cls, value) -> Union[VALID_MODELS, Path]:
        if value in get_args(VALID_MODELS):
            return value

        path = Path(value)
        config = AutoConfig.from_pretrained(str(path.absolute()))
        for arch in config.architectures:
            _ = ModelArchitecture(arch)

        return path

    @field_validator("quantization", mode="before")
    @classmethod
    def validate_quantization(cls, value) -> QuantizationAlgo:
        return QuantizationAlgo(value)

    @field_validator("cache_dtype", mode="before")
    @classmethod
    def validate_kvcache_dtype(cls, value) -> KVCacheDtypeEnum:
        return KVCacheDtypeEnum(value)

    @field_validator("kv_cache_mem_percentage", mode="after")
    @classmethod
    def validate_kv_cache_mem_fraction(cls, value: float) -> float:
        if 0 < value < 1.0:
            return value
        else:
            raise ValidationError(
                "KV cache memory percentage must be between 0 and 1.0.")

    @field_validator("build_overrides", mode="before")
    @classmethod
    def validate_build_overrides(cls, value) -> List[str]:
        # If we encounter a list, scan it to make sure all entries are strings.
        if isinstance(value, list):
            if not all([isinstance(x, str) for x in value]):
                raise ValidationError(
                    "Found a non-string entry in list of options.")
            return value
        elif isinstance(value, str):
            # Handle the case where we receive a single string of command
            # options.
            overrides = []
            if value:
                overrides = [str(x) for x in value.split()]
            return overrides
        else:
            raise ValidationError(
                "Invalid value specified for build overrides.")

    @computed_field
    def engine_path(self) -> Path:
        """Path to the engine workspace."""
        if self.model in get_args(VALID_MODELS):
            return Path(self.workspace.absolute(), self.model.lower())
        else:
            return Path(self.workspace.absolute(), "engine")

    @computed_field
    def world_size(self) -> int:
        """Total world size needed to run the model."""
        return self.tensor_parallel * self.pipeline_parallel
