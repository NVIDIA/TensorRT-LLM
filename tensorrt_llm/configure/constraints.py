from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel, Field, NonNegativeInt, PositiveFloat, PositiveInt, model_validator


class GPU(StrEnum):
    GB200 = "GB200"
    H200_SXM = "H200_SXM"


class BaseConstraints(BaseModel, ABC):
    """Base class for all constraints containing common fields.

    A set of constraints fully defines the requirements for a specific inference workload
    and the goals for optimization (e.g. SLA targets).
    """

    model: str = Field(description="HuggingFace ID of the model being deployed")
    gpu: GPU = Field(description="GPU SKU used in the deployment")
    num_gpus: PositiveInt = Field(description="Number of GPUs available in the deployment")

    @classmethod
    @abstractmethod
    def _get_cli_description(cls) -> str:
        """Get a description of the constraints which will be shown in CLI help messages."""


class BenchmarkConstraints(BaseConstraints):
    isl: NonNegativeInt = Field(description="Target input sequence length")
    osl: NonNegativeInt = Field(description="Target output sequence length")
    concurrency: PositiveInt = Field(description="Target number of concurrent requests")
    # TODO: make this optional and add logic to choose best parallelization mapping automatically
    tp_size: PositiveInt = Field(description="Specific tensor parallel size that should be used")

    @classmethod
    def _get_cli_description(cls) -> str:
        return "Optimize TensorRT LLM for a benchmark workload with a specific number of concurrent requests."


class ThroughputLatencyConstraints(BaseConstraints):
    tps_per_gpu: Optional[PositiveFloat] = Field(
        default=None,
        description="Target minimum throughput per GPU in tokens per second",
    )
    tps_per_user: Optional[PositiveFloat] = Field(
        default=None, description="Target minimum throughput per user in tokens per second."
    )
    ttft: Optional[PositiveFloat] = Field(
        default=None,
        description="Target maximum time to first token in seconds.",
    )

    @classmethod
    def _get_cli_description(cls) -> str:
        return "Optimize TensorRT LLM to meet a throughput and/or latency SLA."

    @model_validator(mode="after")
    def validate_has_at_least_one_constraint(self) -> "ThroughputLatencyConstraints":
        if not any([self.tps_per_gpu, self.tps_per_user, self.ttft]):
            raise ValueError(
                "At least one of target throughput per GPU, target throughput per user, or target time to first token "
                "must be specified."
            )
        return self
