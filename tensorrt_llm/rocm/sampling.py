# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Portable, strictly validated decoding parameters and output containers."""

from dataclasses import dataclass
from typing import Literal

from pydantic import Field, model_validator

from tensorrt_llm._config import StrictBaseModel


class SamplingParams(StrictBaseModel):
    """ROCm decoding parameters; unsupported upstream options are rejected."""

    max_tokens: int = Field(
        default=64, gt=0, description="Maximum generated tokens per completion."
    )
    min_tokens: int = Field(
        default=0, ge=0, description="Minimum generated tokens before stopping."
    )
    temperature: float = Field(
        default=1.0,
        ge=0,
        allow_inf_nan=False,
        description="Sampling temperature; zero selects greedy decoding.",
    )
    top_p: float = Field(default=1.0, gt=0, le=1, description="Nucleus sampling probability.")
    top_k: int = Field(
        default=0, ge=0, description="Top-k sampling cutoff; zero disables filtering."
    )
    repetition_penalty: float = Field(
        default=1.0, gt=0, allow_inf_nan=False, description="Hugging Face repetition penalty."
    )
    n: int = Field(default=1, gt=0, description="Number of completions for each prompt.")
    beam_width: int = Field(default=1, gt=0, description="Number of decoding beams.")
    seed: int | None = Field(
        default=None, ge=0, strict=True, description="Optional reproducible RNG seed."
    )
    stop: str | list[str] | None = Field(
        default=None, description="Stop strings in generated text, not prompts."
    )
    end_id: int | None = Field(
        default=None, ge=0, description="Override the end-of-sequence token ID."
    )
    pad_id: int | None = Field(default=None, ge=0, description="Override the padding token ID.")
    ignore_eos: bool = Field(
        default=False, description="Ignore EOS and continue to a stop string or length limit."
    )

    @model_validator(mode="after")
    def _check_constraints(self) -> "SamplingParams":
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens cannot exceed max_tokens")
        if self.beam_width > 1 and self.n > self.beam_width:
            raise ValueError("n cannot exceed beam_width for beam decoding")
        if self.temperature == 0 and self.beam_width == 1 and self.n > 1:
            raise ValueError("Multiple greedy completions require beam_width >= n")
        strings = [self.stop] if isinstance(self.stop, str) else (self.stop or [])
        if any(not string for string in strings):
            raise ValueError("Stop strings must not be empty")
        return self


@dataclass
class CompletionOutput:
    index: int
    text: str
    token_ids: list[int]
    finish_reason: Literal["stop", "length"]
    cumulative_logprob: float | None = None


@dataclass
class RequestOutput:
    request_id: int
    prompt: str
    prompt_token_ids: list[int]
    outputs: list[CompletionOutput]
    finished: bool = True


__all__ = ["SamplingParams", "CompletionOutput", "RequestOutput"]
