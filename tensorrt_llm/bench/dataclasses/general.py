from __future__ import annotations

from typing import Any, List, Optional, Union

from pydantic import (AliasChoices, BaseModel, Field, computed_field,
                      model_validator)

from tensorrt_llm.bench.dataclasses.statistics import PercentileStats
from tensorrt_llm.executor.request import LoRARequest


class InferenceRequest(BaseModel):
    task_id: int
    prompt: Optional[Union[str, Any]] = None
    output_tokens: int
    input_ids: Optional[List[int]] = Field(
        alias=AliasChoices("input_ids", "logits"))
    lora_request: Optional[LoRARequest] = None

    @model_validator(mode="after")
    def verify_prompt_and_logits(self) -> InferenceRequest:
        if self.prompt is None and self.input_ids is None:
            raise ValueError(
                f"Both prompt and input_ids for {self.task_id} are both None.")
        return self


class DatasetMetadata(BaseModel):
    isl_stats: PercentileStats
    osl_stats: PercentileStats
    seq_len_stats: PercentileStats
    num_requests: int

    @computed_field
    @property
    def max_isl(self) -> int:
        return int(self.isl_stats.maximum)

    @computed_field
    @property
    def max_osl(self) -> int:
        return int(self.osl_stats.maximum)

    @computed_field
    @property
    def max_sequence_length(self) -> int:
        return int(self.seq_len_stats.maximum)

    @computed_field
    @property
    def avg_isl(self) -> int:
        return int(self.isl_stats.average)

    @computed_field
    @property
    def avg_osl(self) -> int:
        return int(self.osl_stats.average)

    @computed_field
    @property
    def avg_sequence_length(self) -> int:
        return int(self.seq_len_stats.average)
