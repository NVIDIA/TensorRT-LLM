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

# Adapted from
# https://github.com/vllm-project/vllm/blob/4db5176d9758b720b05460c50ace3c01026eb158/vllm/entrypoints/openai/protocol.py

import time
import uuid
from typing import Dict, List, Optional, Union

from pydantic import Field, field_validator, model_validator
from typing_extensions import Annotated

from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.kv_hints import KvHintsEnvelope
from tensorrt_llm.llmapi import SamplingParams
from tensorrt_llm.sampling_params import (check_logprobs_limit,
                                          validate_thinking_token_budget)
from tensorrt_llm.serve.openai_protocol.common import (
    ConversationParams, DisaggregatedParams, EarlyStopping, OpenAIBaseModel,
    ResponseFormat, SpeculativeDecodingStats, StreamOptions, UsageInfo,
    _OmitsAbsentSpecDecodeStats)
from tensorrt_llm.serve.openai_protocol.sampling import (
    _early_stopping_to_int, _logit_bias_to_embedding_bias,
    _record_sampling_params_request_fields,
    _response_format_to_guided_decoding_params)


class CompletionLogProbs(OpenAIBaseModel):
    text_offset: List[int] = Field(default_factory=list)
    token_logprobs: List[Optional[float]] = Field(default_factory=list)
    tokens: List[str] = Field(default_factory=list)
    top_logprobs: List[Optional[Dict[str, float]]] = Field(default_factory=list)


class CompletionResponseChoice(_OmitsAbsentSpecDecodeStats):
    index: int
    text: str
    token_ids: Optional[List[int]] = None
    logprobs: Optional[CompletionLogProbs] = None
    context_logits: Optional[Union[List[float], List[List[
        float]]]] = None  # For reward models, the output is score logits instead of text.
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"),
    )
    disaggregated_params: Optional[DisaggregatedParams] = Field(default=None)
    avg_decoded_tokens_per_iter: Optional[float] = Field(default=None)
    speculative_decoding: Optional[SpeculativeDecodingStats] = Field(
        default=None)


class CompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{str(uuid.uuid4().hex)}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseChoice]
    usage: UsageInfo
    # Add prompt_tokens_ids to the response to remove the tokenization
    # in the generation server in disaggreated serving
    prompt_token_ids: Optional[Union[List[List[int]], List[int]]] = None


class CompletionResponseStreamChoice(_OmitsAbsentSpecDecodeStats):
    index: int
    text: str
    token_ids: Optional[List[int]] = None
    logprobs: Optional[CompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "The stop string or token id that caused the completion "
            "to stop, None if the completion finished for some other reason "
            "including encountering the EOS token"),
    )
    avg_decoded_tokens_per_iter: Optional[float] = Field(default=None)
    speculative_decoding: Optional[SpeculativeDecodingStats] = Field(
        default=None)


class CompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{str(uuid.uuid4().hex)}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


class CompletionRequest(OpenAIBaseModel):
    kv_hints: Optional[KvHintsEnvelope] = Field(
        default=None,
        repr=False,
        description=
        "Prototype KV hint metadata; transported without executing actions.",
    )
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    model: str
    prompt: Union[List[int], List[List[int]], str, List[str]]
    best_of: Optional[int] = None
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    seed: Optional[int] = Field(default=None)
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    suffix: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    user: Optional[str] = None
    lora_request: Optional[LoRARequest] = None
    prompt_ignore_length: Optional[int] = 0

    # doc: begin-completion-sampling-params
    use_beam_search: bool = False
    top_k: int = 0
    top_p_min: float = 0.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    # Unset by default so the engine picks its own beam-search defaults, as it
    # does for requests coming through the Python API. Sending concrete values
    # here would put every served request on the length-normalized, exhaustive
    # beam-search path.
    length_penalty: Optional[float] = None
    early_stopping: Optional[EarlyStopping] = None
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    truncate_prompt_tokens: Optional[Annotated[int, Field(ge=1)]] = None
    return_context_logits: bool = False
    detokenize: bool = True
    thinking_token_budget: Optional[int] = None
    # doc: end-completion-sampling-params

    # doc: begin-completion-extra-params
    add_special_tokens: bool = Field(
        default=True,
        description=(
            "If true (the default), special tokens (e.g. BOS) will be added to "
            "the prompt."),
    )
    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description=
        ("Similar to chat completion, this parameter specifies the format of output. "
         "{'type': 'text'}, {'type': 'json'}, {'type': 'json_object'}, {'type': 'regex'}, "
         "{'type': 'ebnf'}, {'type': 'structural_tag'} are supported."),
    )

    disaggregated_params: Optional[DisaggregatedParams] = Field(
        default=None,
        description=("Parameters for disaggregated serving"),
    )
    priority: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=
        ("Scheduling priority in [0.0, 1.0]; higher is served first. Only honored "
         "when the engine runs with scheduler_config.waiting_queue_policy=priority. "
         "If unset, the engine default (0.5) is used."),
    )
    conversation_params: Optional[ConversationParams] = Field(
        default=None,
        description=("Parameters for multi-turn conversation routing"),
    )

    # doc: end-completion-extra-params

    def to_sampling_params(self,
                           vocab_size: Optional[int] = None,
                           gather_generation_logits: bool = False,
                           backend: Optional[str] = None) -> SamplingParams:
        sampling_logprobs = None
        if self.logprobs:
            if backend == "pytorch" or gather_generation_logits:
                sampling_logprobs = self.logprobs
            elif self.logprobs > 1:
                raise ValueError(
                    "`logprobs` must be 1 or `gather_generation_logits` must be `True` to use `logprobs` > 1"
                )

        sampling_params = SamplingParams(
            best_of=self.best_of,
            frequency_penalty=self.frequency_penalty,
            logprobs=sampling_logprobs,
            max_tokens=self.max_tokens,
            n=self.n,
            presence_penalty=self.presence_penalty,
            seed=self.seed,
            stop=self.stop,
            temperature=(self.temperature
                         if self.temperature is not None else 1.0),
            top_p=(self.top_p if self.top_p is not None else 1.0),
            prompt_ignore_length=self.prompt_ignore_length,

            # completion-sampling-params
            use_beam_search=self.use_beam_search,
            top_k=self.top_k,
            top_p_min=self.top_p_min if self.top_p_min > 0 else None,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            early_stopping=_early_stopping_to_int(self.early_stopping),
            stop_token_ids=self.stop_token_ids,
            include_stop_str_in_output=self.include_stop_str_in_output,
            ignore_eos=self.ignore_eos,
            min_tokens=self.min_tokens,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            truncate_prompt_tokens=self.truncate_prompt_tokens,
            return_context_logits=self.return_context_logits,
            guided_decoding=_response_format_to_guided_decoding_params(
                self.response_format),
            detokenize=self.detokenize,
            thinking_token_budget=self.thinking_token_budget,

            # logits_bias
            embedding_bias=_logit_bias_to_embedding_bias(
                self.logit_bias, vocab_size),

            # completion-extra-params
            add_special_tokens=self.add_special_tokens,
        )
        _record_sampling_params_request_fields(self, sampling_params)
        return sampling_params

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        check_logprobs_limit("logprobs", data.get("logprobs"))
        return data

    @field_validator("thinking_token_budget", mode="before")
    @classmethod
    def check_thinking_token_budget(cls, value):
        return validate_thinking_token_budget(value)

    @model_validator(mode="before")
    @classmethod
    def validate_stream_options(cls, data):
        if data.get("stream_options") and not data.get("stream"):
            raise ValueError(
                "Stream options can only be defined when stream is true.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_suffix(cls, data):
        if data.get("suffix"):
            raise ValueError("suffix is not supported")
        return data
