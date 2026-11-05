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
from typing import Any, Callable, Dict, List, Literal, Optional, Union

from openai.types.chat import ChatCompletionAssistantMessageParam
from openai.types.chat import \
    ChatCompletionContentPartParam as OpenAIChatCompletionContentPartParam
from openai.types.chat import \
    ChatCompletionMessageParam as OpenAIChatCompletionMessageParam
from openai_harmony import ReasoningEffort
from pydantic import ConfigDict, Field, field_validator, model_validator
from typing_extensions import Annotated, Required, TypedDict

from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.inputs.media_io import MediaModality
from tensorrt_llm.kv_hints import KvHintsEnvelope
from tensorrt_llm.llmapi import SamplingParams
from tensorrt_llm.sampling_params import (check_logprobs_limit,
                                          validate_thinking_token_budget)
from tensorrt_llm.scheduling_params import AgentHierarchy
from tensorrt_llm.serve.openai_protocol.common import (
    ConversationParams, DisaggregatedParams, EarlyStopping, OpenAIBaseModel,
    ResponseFormat, SpeculativeDecodingStats, StreamOptions, UsageInfo,
    _OmitsAbsentSpecDecodeStats)
from tensorrt_llm.serve.openai_protocol.sampling import (
    _early_stopping_to_int, _logit_bias_to_embedding_bias,
    _record_sampling_params_request_fields,
    _response_format_to_guided_decoding_params)
from tensorrt_llm.serve.openai_protocol.tools import (
    ChatCompletionNamedToolChoiceParam, ChatCompletionToolsParam, DeltaToolCall,
    ToolCall)


class ChatMessage(OpenAIBaseModel):
    role: str
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    reasoning: Optional[str] = None
    tool_calls: List[ToolCall] = Field(default_factory=list)


class ChatCompletionLogProb(OpenAIBaseModel):
    token: str
    logprob: float = -9999.0
    bytes: Optional[List[int]] = None


class ChatCompletionLogProbsContent(ChatCompletionLogProb):
    top_logprobs: Optional[List[ChatCompletionLogProb]] = None


class CustomChatCompletionContentPartParam(TypedDict, total=False):
    __pydantic_config__ = ConfigDict(extra="allow")  # type: ignore

    type: Required[str]
    """The type of the content part."""


ChatCompletionContentPartParam = Union[OpenAIChatCompletionContentPartParam,
                                       CustomChatCompletionContentPartParam]


class CustomChatCompletionMessageParam(TypedDict, total=False):
    """Enables custom roles in the Chat Completion API."""

    # This is so custom fields not in any of the `ChatCompletionMessage<XYZ>Param` defined by OpenAI
    # are still allowed.
    # Examples include: assistant messages with `reasoning` / `reasoning_content`.
    __pydantic_config__ = ConfigDict(extra="allow")  # type: ignore

    role: Required[str]
    """The role of the message's author."""

    content: Union[str, List[ChatCompletionContentPartParam], None]
    """The contents of the message."""

    name: str
    """An optional name for the participant.

    Provides the model information to differentiate between participants of the
    same role.
    """


class ReasoningAssistantMessage(ChatCompletionAssistantMessageParam):
    """Assistant message that includes reasoning tokens."""
    reasoning: Optional[str]
    # NOTE: some older benchmarks and chat templates assume the below, which has been deprecated
    # in other inference frameworks in favor of the above `reasoning` field.
    reasoning_content: Optional[str]


class DynamicToolsSystemMessageParam(TypedDict, total=False):
    """System message carrying message-level (dynamic) tool declarations.

    Kimi-style templates render such messages as an in-conversation tool
    declare block. Must come first in `ChatCompletionMessageParam`: the
    stock OpenAI system-message TypedDict otherwise wins smart-union scoring
    and silently drops the `tools` key.
    """
    __pydantic_config__ = ConfigDict(extra="allow")  # type: ignore

    role: Required[Literal["system"]]
    tools: Required[List[dict]]
    content: Union[str, List[ChatCompletionContentPartParam], None]
    name: str


ChatCompletionMessageParam = Union[DynamicToolsSystemMessageParam,
                                   OpenAIChatCompletionMessageParam,
                                   CustomChatCompletionMessageParam,
                                   ReasoningAssistantMessage]


class ChatCompletionLogProbs(OpenAIBaseModel):
    content: Optional[List[ChatCompletionLogProbsContent]] = None


class ChatCompletionResponseChoice(_OmitsAbsentSpecDecodeStats):
    index: int
    message: ChatMessage
    logprobs: Optional[ChatCompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = None
    # TODO: progressivly add more info like input_ids, specific_token_ids, mrope, mm_hashes, etc
    # TODO: and use a JSON-safe handle to refer to the server-side output
    mm_embedding_handle: Optional[Dict[str, Any]] = None

    disaggregated_params: Optional[DisaggregatedParams] = Field(default=None)
    avg_decoded_tokens_per_iter: Optional[float] = Field(default=None)
    speculative_decoding: Optional[SpeculativeDecodingStats] = Field(
        default=None)


class ChatCompletionResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{str(uuid.uuid4().hex)}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: UsageInfo
    # Add prompt_tokens_ids to the response to remove the tokenization
    # in the generation server in disaggreated serving
    prompt_token_ids: Optional[List[int]] = None
    # base64 int32 buffer alternative to prompt_token_ids; set by the context
    # worker so the orchestrator can relay a string instead of the int list.
    prompt_token_ids_b64: Optional[str] = None


class DeltaMessage(OpenAIBaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    # For GPT-OSS style reasoning
    reasoning: Optional[str] = None
    tool_calls: Optional[List[DeltaToolCall]] = None


class ChatCompletionResponseStreamChoice(_OmitsAbsentSpecDecodeStats):
    index: int
    delta: DeltaMessage
    logprobs: Optional[ChatCompletionLogProbs] = None
    finish_reason: Optional[str] = None
    stop_reason: Optional[Union[int, str]] = None
    avg_decoded_tokens_per_iter: Optional[float] = Field(default=None)
    speculative_decoding: Optional[SpeculativeDecodingStats] = Field(
        default=None)


class ChatCompletionStreamResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{str(uuid.uuid4().hex)}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseStreamChoice]
    usage: Optional[UsageInfo] = Field(default=None)


class ChatCompletionThinkingParam(OpenAIBaseModel):
    """Kimi/Moonshot `thinking` extension controlling reasoning output.

    `keep` is fixed to `"all"` when thinking is enabled and ignored when
    disabled; `effort` is only meaningful when enabled. An explicit
    `effort` wins over a request-level `reasoning_effort`, which applies
    only when `effort` is absent (and never for a disabled request).
    """
    type: Literal["enabled", "disabled"] = "enabled"
    keep: Optional[Literal["all"]] = None
    effort: Optional[Literal["low", "high", "max"]] = None


class ChatCompletionRequest(OpenAIBaseModel):
    kv_hints: Optional[KvHintsEnvelope] = Field(
        default=None,
        repr=False,
        description=
        "Prototype KV hint metadata; transported without executing actions.",
    )
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: List[ChatCompletionMessageParam]
    # Add prompt_tokens_ids to the request to remove the tokenization
    # in the generation server in disaggreated serving
    prompt_token_ids: Optional[List[int]] = None
    # base64 int32 buffer relayed by the orchestrator from the ctx response;
    # decoded back to prompt_token_ids on the generation worker. Not for clients.
    prompt_token_ids_b64: Optional[str] = None
    model: str
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = 0
    max_completion_tokens: Optional[int] = Field(default=None,
                                                 validation_alias='max_tokens')
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = Field(None)
    service_tier: Optional[Literal["auto", "default", "flex", "scale",
                                   "priority", "fast"]] = Field(
                                       default=None,
                                       description=(
                                           "Accepted for OpenAI compatibility "
                                           "and ignored: the server has no "
                                           "service tiers. Scheduling priority "
                                           "is the separate `priority` field."))
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    # Standard OpenAI field, accepted for compatibility and otherwise ignored:
    # the server keeps no copy of a chat completion, so there is nothing to
    # store or to leave unstored.
    store: Optional[bool] = None
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List[ChatCompletionToolsParam]] = None
    tool_choice: Optional[Union[Literal["none", "auto", "required"],
                                ChatCompletionNamedToolChoiceParam]] = "none"
    # Standard OpenAI field, accepted for compatibility. `false` is not
    # enforced: the engine does not restrict how many tool calls the model
    # emits per turn, so parallel emission remains model behavior either way.
    parallel_tool_calls: Optional[bool] = None
    user: Optional[str] = None
    reasoning_effort: Optional[ReasoningEffort | Literal[
        "low", "medium", "high", "max", "none"]] = Field(
            default=ReasoningEffort.LOW,
            description=(
                "The level of reasoning effort to use. Controls how much "
                "reasoning is shown in the model's response. Options: "
                "'low', 'medium', 'high' (harmony/gpt-oss), plus 'max' and "
                "'none' for models with Kimi-style thinking control."),
        )
    # Kimi/Moonshot extension: structured control of reasoning output. The
    # serving layer maps it into the chat-template kwargs for models whose
    # template understands it (e.g. kimi_k3); other models ignore it.
    thinking: Optional[ChatCompletionThinkingParam] = None
    thinking_token_budget: Optional[int] = None
    prompt_ignore_length: Optional[int] = 0
    # Match Codex CLI's string-map contract; widen only for a client that needs
    # other value types, even though the metadata is otherwise ignored.
    client_metadata: Optional[Dict[str, str]] = Field(
        default=None,
        description=("Not an OpenAI field: opaque per-request metadata that "
                     "agent clients attach, a map of strings to strings. "
                     "Accepted and ignored."))

    # doc: begin-chat-completion-sampling-params
    best_of: Optional[int] = None
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
    lora_request: Optional[LoRARequest] = None
    # doc: end-chat-completion-sampling-params

    # doc: begin-chat-completion-extra-params
    echo: bool = Field(
        default=False,
        description=(
            "If true, the new message will be prepended with the last message "
            "if they belong to the same role."),
    )
    add_generation_prompt: bool = Field(
        default=True,
        description=
        ("If true, the generation prompt will be added to the chat template. "
         "This is a parameter used by chat template in tokenizer config of the "
         "model."),
    )
    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."),
    )
    documents: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description=
        ("A list of dicts representing documents that will be accessible to "
         "the model if it is performing RAG (retrieval-augmented generation)."
         " If the template does not support RAG, this argument will have no "
         "effect. We recommend that each document should be a dict containing "
         "\"title\" and \"text\" keys."),
    )
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "If this is not passed, the model's default chat template will be "
            "used instead."),
    )
    chat_template_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the template renderer. "
                     "Will be accessible by the chat template."),
    )
    injected_chat_template_kwargs: Optional[List[str]] = Field(
        default=None,
        description=(
            "Server-populated. Keys of `chat_template_kwargs` that the server "
            "derived from API-level fields (for example the Anthropic "
            "`thinking` and `context_management` controls) rather than the "
            "caller. The unused-kwargs guard exempts them, since the caller "
            "cannot remove a control the server added. Relayed with the "
            "request so a disaggregated worker rendering the prompt applies "
            "the same exemption."),
    )

    media_io_kwargs: Optional[Dict[MediaModality, Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Per-request override for the server's `--media_io_kwargs`. "
            "Shape: `{modality: {kwarg: value}}` with modality in "
            "{\"image\", \"video\", \"audio\"}; unknown modality keys are "
            "rejected. Per modality, request kwargs are shallow-merged "
            "onto the server defaults (request wins per key). For "
            "`video`, overriding only one of `fps`/`num_frames` drops "
            "the other from the server default so the loader's built-in "
            "is used. "
            "Example: `{\"video\": {\"num_frames\": 32}}`."),
    )

    disaggregated_params: Optional[DisaggregatedParams] = Field(
        default=None,
        description=("Parameters for disaggregated serving"),
    )
    conversation_params: Optional[ConversationParams] = Field(
        default=None,
        description=("Parameters for multi-turn conversation routing"),
    )

    cache_salt: Optional[str] = Field(
        default=None,
        description=
        ("If specified, KV cache will be salted with the provided string "
         "to limit the kv cache reuse on with the requests having the same string."
         ))
    priority: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=
        ("Scheduling priority in [0.0, 1.0]; higher is served first. Only honored "
         "when the engine runs with scheduler_config.waiting_queue_policy=priority. "
         "If unset, the engine default (0.5) is used."),
    )

    agent_hierarchy: Optional[AgentHierarchy] = Field(
        default=None, description="Agent hierarchy ")

    mm_processor_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description=
        "Per-request kwargs forwarded to the multimodal HF processor (e.g. num_frames for video models).",
    )

    # doc: end-chat-completion-extra-params

    def to_sampling_params(
        self,
        vocab_size: Optional[int] = None,
        gather_generation_logits: bool = False,
        reasoning_parser: Optional[str] = None,
        backend: Optional[str] = None,
        *,
        structured_output_format: Optional[
            Callable[[Dict[str, Any], Optional[Dict[str, Any]]],
                     Optional[Dict[str, Any]]]] = None
    ) -> SamplingParams:
        """Build the engine's sampling parameters for this request.

        ``structured_output_format`` places the ``response_format`` constraint
        through a serving extension's hook instead of ``reasoning_parser``
        (see ``_response_format_to_guided_decoding_params``).
        """
        sampling_logprobs = None
        if self.logprobs:
            logprobs = 1 if not self.top_logprobs else self.top_logprobs
            if backend == "pytorch" or gather_generation_logits:
                sampling_logprobs = logprobs
            elif self.top_logprobs:
                raise ValueError(
                    "`gather_generation_logits` must be `True` to use `top_logprobs`"
                )

        sampling_params = SamplingParams(
            frequency_penalty=self.frequency_penalty,
            logprobs=sampling_logprobs,
            max_tokens=self.max_completion_tokens,
            n=self.n,
            presence_penalty=self.presence_penalty,
            seed=self.seed,
            stop=self.stop,
            temperature=(self.temperature
                         if self.temperature is not None else 1.0),
            prompt_ignore_length=self.prompt_ignore_length,

            # chat-completion-sampling-params
            best_of=self.best_of,
            use_beam_search=self.use_beam_search,
            top_k=self.top_k,
            top_p=(self.top_p if self.top_p is not None else 1.0),
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
            guided_decoding=_response_format_to_guided_decoding_params(
                self.response_format,
                reasoning_parser=reasoning_parser,
                chat_template_kwargs=self.chat_template_kwargs,
                structured_output_format=structured_output_format),
            thinking_token_budget=self.thinking_token_budget,

            # logits_bias
            embedding_bias=_logit_bias_to_embedding_bias(
                self.logit_bias, vocab_size),

            # chat-completion-extra-params
            add_special_tokens=self.add_special_tokens,
        )
        _record_sampling_params_request_fields(self, sampling_params)
        return sampling_params

    @model_validator(mode='before')
    @classmethod
    def validate_stream_options(cls, values):
        if (values.get('stream_options') is not None
                and not values.get('stream')):
            raise ValueError("stream_options can only be set if stream is true")
        return values

    @model_validator(mode="before")
    @classmethod
    def check_tool_choice(cls, data):
        if not isinstance(data, dict):
            return data
        has_dynamic_tools = any(
            isinstance(msg, dict) and msg.get("role") == "system"
            and msg.get("tools") for msg in data.get("messages") or [])
        if "tool_choice" not in data and (data.get("tools")
                                          or has_dynamic_tools):
            data["tool_choice"] = "auto"
        # "none" and "auto" are meaningful without tools. "required" needs a
        # non-empty tool set — request-level or message-level (dynamic)
        # tools both count. A named function must be a request-level tool.
        if "tool_choice" in data and data["tool_choice"] not in ("none",
                                                                 "auto"):
            satisfied = data.get("tools") or (data["tool_choice"] == "required"
                                              and has_dynamic_tools)
            if not satisfied:
                raise ValueError(
                    "When using `tool_choice`, `tools` must be set.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_message_tools_role(cls, data):
        """Message-level tool declarations ride on system messages only.

        Union validation strips unknown keys from non-system messages, so
        this raw-payload validator is the only layer that can reject the
        misuse loudly instead of silently dropping the client's tools. The
        rest of the dynamic-tools contract is model-specific and validated
        in the serving layer (kimi_k3 only).
        """
        if not isinstance(data, dict):
            return data
        for message in data.get("messages") or []:
            if (isinstance(message, dict) and message.get("tools") is not None
                    and message.get("role") != "system"):
                raise ValueError(
                    "Message-level `tools` are only allowed on system "
                    "messages.")
        return data

    @model_validator(mode="before")
    @classmethod
    def check_logprobs(cls, data):
        if (top_logprobs := data.get("top_logprobs")) is not None:
            check_logprobs_limit("top_logprobs", top_logprobs)
            if not data.get("logprobs"):
                raise ValueError(
                    "logprobs must be true when using top_logprobs")
        return data

    @field_validator("thinking_token_budget", mode="before")
    @classmethod
    def check_thinking_token_budget(cls, value):
        return validate_thinking_token_budget(value)

    @model_validator(mode="before")
    @classmethod
    def check_suffix(cls, data):
        if data.get("suffix"):
            raise ValueError("suffix is not supported")
        return data

    @field_validator("cache_salt")
    @classmethod
    def check_cache_salt_support(cls, v):
        if v is not None:
            if not isinstance(v, str) or not v.strip():
                raise ValueError(
                    "Parameter 'cache_salt' must be a non-empty string if provided."
                )
        return v
