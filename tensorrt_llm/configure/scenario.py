from enum import StrEnum
from typing import Optional

from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)


class GPU(StrEnum):
    GB200 = "GB200"
    H200_SXM = "H200_SXM"


class BaseScenario(BaseModel):
    """Base class for all scenarios containing common fields.

    A scenario fully defines a specific inference workload and the goals for optimization
    (e.g. SLA targets).
    """

    model: str = Field(description="HuggingFace ID of the model being deployed")
    gpu: GPU = Field(description="GPU SKU used in the deployment")
    num_gpus: PositiveInt = Field(description="Number of GPUs available in the deployment")


class BenchmarkScenario(BaseScenario):
    isl: NonNegativeInt = Field(
        description="Target input sequence length",
        validation_alias=AliasChoices("isl", "target_isl", "input_sequence_length"),
    )
    osl: NonNegativeInt = Field(
        description="Target output sequence length",
        validation_alias=AliasChoices("osl", "target_osl", "output_sequence_length"),
    )
    concurrency: PositiveInt = Field(
        description="Target number of concurrent requests",
        validation_alias=AliasChoices("concurrency", "target_concurrency"),
    )
    # TODO: make this optional and add logic to choose best parallelization mapping automatically
    tensor_parallel_size: PositiveInt = Field(
        description="Specific tensor parallel size that should be used",
        validation_alias=AliasChoices("tensor_parallel_size", "tp"),
    )


class ThroughputLatencySLAScenario(BaseScenario):
    tps_per_gpu: PositiveFloat = Field(
        description="Target throughput per GPU in tokens per second",
        validation_alias=AliasChoices("tps_per_gpu", "target_tps_per_gpu", "min_tps_per_gpu"),
    )
    tps_per_user: Optional[PositiveFloat] = Field(
        default=None,
        description="Target throughput per user in tokens per second. Mutually exclusive with target time to first "
        "token.",
        validation_alias=AliasChoices("tps_per_user", "target_tps_per_user", "min_tps_per_user"),
    )
    ttft: Optional[PositiveFloat] = Field(
        default=None,
        description="Target time to first token in seconds. Mutually exclusive with target throughput per user.",
        validation_alias=AliasChoices("ttft", "target_ttft", "max_ttft"),
    )

    @model_validator(mode="after")
    def validate_mutually_exclusive_latency_sla(self) -> "ThroughputLatencySLAScenario":
        if self.tps_per_user is not None and self.ttft is not None:
            raise ValueError(
                "Target throughput per user and target time to first token cannot be specified together."
            )
        return self
