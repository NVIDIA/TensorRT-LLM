from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Union

from pydantic import (AliasChoices, BaseModel, Field, computed_field,
                      model_validator)

from tensorrt_llm.bench.dataclasses.statistics import PercentileStats
from tensorrt_llm.executor.request import LoRARequest


class BenchmarkEnvironment(BaseModel):
    model: str
    checkpoint_path: Optional[Path]
    workspace: Path


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
    dataset_path: Optional[Path] = None

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

    def _format_number(self, value: float) -> str:
        """Format number to fit within 9 characters including decimal."""
        if value >= 100000:
            return f"{value:9.2e}".ljust(
                9
            )  # Scientific notation for large numbers, padded to 9 characters
        return f"{value:9.4f}".ljust(
            9)  # Fixed point for smaller numbers, padded to 9 characters

    def get_summary_for_print(self) -> str:
        form = self._format_number
        return (
            "\n===========================================================\n"
            "= DATASET DETAILS\n"
            "===========================================================\n"
            f"Dataset Path:         {self.dataset_path}\n"
            f"Number of Sequences:  {self.num_requests}\n"
            "\n-- Percentiles statistics ---------------------------------\n\n"
            "        Input              Output           Seq. Length\n"
            "-----------------------------------------------------------\n"
            f"MIN:  {form(self.isl_stats.minimum)}          {form(self.osl_stats.minimum)}          {form(self.seq_len_stats.minimum)}\n"
            f"MAX:  {form(self.isl_stats.maximum)}          {form(self.osl_stats.maximum)}          {form(self.seq_len_stats.maximum)}\n"
            f"AVG:  {form(self.isl_stats.average)}          {form(self.osl_stats.average)}          {form(self.seq_len_stats.average)}\n"
            f"P50:  {form(self.isl_stats.p50)}          {form(self.osl_stats.p50)}          {form(self.seq_len_stats.p50)}\n"
            f"P90:  {form(self.isl_stats.p90)}          {form(self.osl_stats.p90)}          {form(self.seq_len_stats.p90)}\n"
            f"P95:  {form(self.isl_stats.p95)}          {form(self.osl_stats.p95)}          {form(self.seq_len_stats.p95)}\n"
            f"P99:  {form(self.isl_stats.p99)}          {form(self.osl_stats.p99)}          {form(self.seq_len_stats.p99)}\n"
            "===========================================================\n")
