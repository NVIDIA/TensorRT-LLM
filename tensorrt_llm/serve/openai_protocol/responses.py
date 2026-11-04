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

import base64
import time
import uuid
from typing import Any, List, Literal, Optional, Union

from openai.types.responses import (
    ResponseCodeInterpreterCallCodeDeltaEvent,
    ResponseCodeInterpreterCallCodeDoneEvent,
    ResponseCodeInterpreterCallCompletedEvent,
    ResponseCodeInterpreterCallInProgressEvent,
    ResponseCodeInterpreterCallInterpretingEvent, ResponseCompletedEvent,
    ResponseContentPartAddedEvent, ResponseContentPartDoneEvent,
    ResponseCreatedEvent, ResponseFunctionToolCall, ResponseInProgressEvent,
    ResponseInputItemParam, ResponseOutputItem, ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent, ResponsePrompt, ResponseReasoningItem,
    ResponseReasoningTextDeltaEvent, ResponseReasoningTextDoneEvent,
    ResponseStatus, ResponseTextConfig, ResponseWebSearchCallCompletedEvent,
    ResponseWebSearchCallInProgressEvent, ResponseWebSearchCallSearchingEvent)
from openai.types.responses.response import IncompleteDetails, ToolChoice
from openai.types.responses.tool import Tool
from openai.types.shared import Metadata, Reasoning
from pydantic import (ConfigDict, Field, field_validator, model_serializer,
                      model_validator)
from typing_extensions import TypeAlias

from tensorrt_llm.kv_hints import KvHintsEnvelope
from tensorrt_llm.llmapi import SamplingParams
from tensorrt_llm.sampling_params import validate_thinking_token_budget
from tensorrt_llm.serve.openai_protocol.common import (ConversationParams,
                                                       DisaggregatedParams,
                                                       OpenAIBaseModel)
from tensorrt_llm.serve.openai_protocol.sampling import (
    _record_sampling_params_request_fields,
    _response_format_text_config_to_guided_decoding_params)

# The trailing dict keeps a client's own item types from failing the request
# outright. Codex multi-agent sessions carry "agent_message" items, which no
# SDK model describes; without a permissive member the union rejects the whole
# input and every request from a spawned agent comes back 422, so agent
# collaboration cannot work at all. Known shapes still match their typed
# member first; see _response_output_item_to_chat_completion_message for how
# an unrecognised item is replayed.
ResponseInputOutputItem: TypeAlias = Union[ResponseInputItemParam,
                                           ResponseReasoningItem,
                                           ResponseFunctionToolCall, dict[str,
                                                                          Any]]

# Roles whose message items map to EasyInputMessageParam / Message, both of
# which forbid ``id``. An assistant message maps to ResponseOutputMessageParam
# instead, which *requires* ``id`` and ``status`` - stripping ``id`` there
# leaves the item matching no variant of the input union, so a conversation
# breaks as soon as it carries one assistant turn.
#
# Module level on purpose: pydantic turns a leading-underscore class attribute
# into a ModelPrivateAttr, so referring to it from a validator would raise
# "argument of type 'ModelPrivateAttr' is not iterable" at request time.
_ID_STRIPPED_ROLES = ("user", "system", "developer")


def _drop_explicit_nulls(value, _depth=0):
    """Recursively drop dict keys whose value is an explicit ``null``.

    On the OpenAI wire an unset optional field is omitted, so an explicit
    ``null`` carries no information — but the vendored Responses item types
    validate ``null`` differently from omitted and reject it. Dropping
    null-valued keys restores the omit-unset wire shape for clients that
    serialize every unset optional as ``null`` (litellm among them).
    """
    if _depth > 12:
        return value
    if isinstance(value, dict):
        return {
            k: _drop_explicit_nulls(v, _depth + 1)
            for k, v in value.items() if v is not None
        }
    if isinstance(value, list):
        return [_drop_explicit_nulls(v, _depth + 1) for v in value]
    return value


def _materialize_validator_iterators(value, _depth=0):
    """Recursively replace pydantic ValidatorIterator objects with lists.

    Matched by type name rather than by import: the class lives in the
    pydantic_core extension module and is not part of its public API.
    """
    if _depth > 12:
        return value
    if type(value).__name__ == "ValidatorIterator":
        return [_materialize_validator_iterators(v, _depth + 1) for v in value]
    if isinstance(value, dict):
        for key, item in value.items():
            value[key] = _materialize_validator_iterators(item, _depth + 1)
        return value
    if isinstance(value, list):
        return [_materialize_validator_iterators(v, _depth + 1) for v in value]
    return value


class ResponsesRequest(OpenAIBaseModel):
    kv_hints: Optional[KvHintsEnvelope] = Field(
        default=None,
        repr=False,
        description=
        "Prototype KV hint metadata; transported without executing actions.",
    )
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/responses/create
    #
    # Unlike the rest of the OpenAI models this one accepts unknown fields.
    # Real Responses API clients attach evolving telemetry and routing keys -
    # Codex CLI sends client_metadata and prompt_cache_key, for instance - and
    # rejecting the whole request over a field the server would ignore anyway
    # makes those clients unusable for no benefit.
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    background: Optional[bool] = False
    include: Optional[list[
        Literal[
            "code_interpreter_call.outputs",
            "computer_call_output.output.image_url",
            "file_search_call.results",
            "message.input_image.image_url",
            "message.output_text.logprobs",
            "reasoning.encrypted_content",
        ],
    ]] = None
    input: Union[str, list[ResponseInputOutputItem]]
    instructions: Optional[str] = None

    @field_validator("input", mode="before")
    @classmethod
    def _drop_unsupported_input_item_keys(cls, value):
        """Strip per-item keys the vendored Responses item types reject.

        Clients echo back items exactly as the server emitted them, so input
        message items arrive carrying the ``id`` the server assigned, which
        ``EasyInputMessageParam`` / ``Message`` forbid. That fails the whole
        request over a field with no prompt content.

        Assistant messages and tool-call items keep their ids: the former
        needs it to validate, the latter uses it to pair calls with results.
        """
        if not isinstance(value, list):
            return value

        # Must run before the per-item shaping below: the item types reject
        # explicit nulls that mean "unset" (see _drop_explicit_nulls).
        value = [_drop_explicit_nulls(item) for item in value]

        def _with_annotations(part):
            """output_text requires annotations; clients often omit it."""
            if (isinstance(part, dict) and part.get("type") == "output_text"
                    and "annotations" not in part):
                return {**part, "annotations": []}
            return part

        cleaned = []
        for item in value:
            # A client may send a bare content part as a top-level item, not
            # only nested inside a message.
            item = _with_annotations(item)
            if isinstance(item, dict) and item.get("type") in (None, "message"):
                role = item.get("role")
                if "id" in item and role in _ID_STRIPPED_ROLES:
                    item = {k: v for k, v in item.items() if k != "id"}
                elif role == "assistant":
                    # ResponseOutputMessageParam requires status, and each
                    # output_text part requires annotations. Clients rebuild an
                    # assistant turn from the streamed deltas rather than from
                    # the server's content-part objects, so both routinely
                    # arrive absent and fail the whole request. Default them
                    # rather than reject a conversation over fields that carry
                    # no prompt content.
                    item = dict(item)
                    item.setdefault("status", "completed")
                    content = item.get("content")
                    if isinstance(content, list):
                        item["content"] = [
                            {
                                **part, "annotations": part.get(
                                    "annotations", [])
                            } if isinstance(part, dict)
                            and part.get("type") == "output_text" else part
                            for part in content
                        ]
            cleaned.append(item)
        return cleaned

    @field_validator("input", mode="after")
    @classmethod
    def _materialize_lazy_item_content(cls, value):
        """Force lazily-validated sequences anywhere in ``input`` into lists.

        Several vendored Responses item types declare sequence fields as
        ``Iterable[...]`` - ``ResponseOutputMessageParam.content`` and
        ``ResponseOutputTextParam.annotations`` among them - and pydantic
        validates an Iterable lazily into a ``ValidatorIterator``. That object
        cannot be pickled, so a request carrying structured input dies with

            cannot pickle 'pydantic_core._pydantic_core.ValidatorIterator'

        the moment it is handed to a postprocess worker. It is also
        single-consumption, so even with workers disabled anything that reads
        the field twice sees an empty sequence the second time.

        The walk has to be recursive: the lazy fields are nested inside items,
        e.g. input[1].content[0].annotations.
        """
        return _materialize_validator_iterators(value)

    max_output_tokens: Optional[int] = None
    max_tool_calls: Optional[int] = None
    metadata: Optional[Metadata] = None
    model: str
    parallel_tool_calls: Optional[bool] = False
    previous_response_id: Optional[str] = None
    prompt: Optional[ResponsePrompt] = None
    reasoning: Optional[Reasoning] = None
    thinking_token_budget: Optional[int] = None
    service_tier: Literal["auto", "default", "flex", "scale",
                          "priority"] = "auto"
    store: Optional[bool] = True
    stream: Optional[bool] = False
    temperature: Optional[float] = None
    text: Optional[ResponseTextConfig] = None
    tool_choice: ToolChoice = "auto"
    tools: list[Tool] = Field(default_factory=list)
    top_logprobs: Optional[int] = 0
    top_p: Optional[float] = None
    truncation: Optional[Literal["auto", "disabled"]] = "disabled"
    user: Optional[str] = None

    request_id: str = Field(
        default_factory=lambda: f"resp_{str(uuid.uuid4().hex)}",
        description=(
            "The request_id related to this request. If the caller does "
            "not set it, a random_uuid will be generated. This id is used "
            "through out the inference process and return in response."),
    )

    # doc: begin-responses-extra-params
    # TensorRT-LLM extensions carrying the disaggregated-serving handoff between
    # the orchestrator and the workers; clients do not set them.
    disaggregated_params: Optional[DisaggregatedParams] = Field(
        default=None,
        description=("Parameters for disaggregated serving"),
    )
    conversation_params: Optional[ConversationParams] = Field(
        default=None,
        description=("Parameters for multi-turn conversation routing"),
    )
    # The context worker's tokenized prompt, so the generation worker does not
    # render it again; the _b64 form is a base64 int32 buffer.
    prompt_token_ids: Optional[List[int]] = None
    prompt_token_ids_b64: Optional[str] = None

    # doc: end-responses-extra-params

    @field_validator("prompt_token_ids_b64")
    @classmethod
    def _check_prompt_token_ids_b64(cls, value: Optional[str]) -> Optional[str]:
        # Decoded during routing and preprocessing, after validation; a bad
        # value must be a 400 here rather than an internal error there.
        if not value:
            return value
        try:
            raw = base64.b64decode(value, validate=True)
        except ValueError as error:
            raise ValueError(
                "prompt_token_ids_b64 is not valid base64") from error
        if len(raw) % 4:
            raise ValueError(
                "prompt_token_ids_b64 is not a buffer of int32 token ids")
        return value

    def relayed_prompt_token_ids(self) -> Optional[List[int]]:
        """The prompt the orchestrator relayed from the context worker, if any.

        Only a generation-only handoff carries one. On any other request the
        fields are ignored, so the prompt is rendered from the request and the
        response echoes what was actually prefilled.
        """
        params = self.disaggregated_params
        if params is None or params.request_type != "generation_only":
            return None
        if self.prompt_token_ids is not None:
            return self.prompt_token_ids
        if not self.prompt_token_ids_b64:
            return None
        import numpy as np
        return np.frombuffer(base64.b64decode(self.prompt_token_ids_b64),
                             dtype=np.int32).tolist()

    _DEFAULT_SAMPLING_PARAMS = {
        "temperature": 1.0,
        "top_p": 1.0,
    }

    def to_sampling_params(
        self,
        default_sampling_params: Optional[dict] = None,
        reasoning_parser: Optional[str] = None,
    ) -> SamplingParams:
        max_tokens = None
        if self.max_output_tokens is not None:
            max_tokens = self.max_output_tokens

        default_sampling_params = default_sampling_params or {}
        if (temperature := self.temperature) is None:
            temperature = default_sampling_params.get(
                "temperature", self._DEFAULT_SAMPLING_PARAMS["temperature"])
        if (top_p := self.top_p) is None:
            top_p = default_sampling_params.get(
                "top_p", self._DEFAULT_SAMPLING_PARAMS["top_p"])
        stop_token_ids = default_sampling_params.get("stop_token_ids", None)

        # Structured output
        guided_decoding = None
        if self.text is not None and self.text.format is not None:
            guided_decoding = _response_format_text_config_to_guided_decoding_params(
                self.text.format, reasoning_parser=reasoning_parser)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            logprobs=self.top_logprobs,
            stop_token_ids=stop_token_ids,
            guided_decoding=guided_decoding,
            thinking_token_budget=self.thinking_token_budget,
        )
        _record_sampling_params_request_fields(self, sampling_params)
        return sampling_params

    @model_validator(mode="before")
    @classmethod
    def _null_top_level_fields_mean_unset(cls, data):
        """Drop explicitly-null top-level fields so they take their defaults.

        Shallow on purpose: ``input`` items are scrubbed recursively by their
        own validator, and the remaining structured fields are Optional
        throughout. A null required field (``model``, ``input``) still
        reports as missing.
        """
        if not isinstance(data, dict):
            return data
        return {k: v for k, v in data.items() if v is not None}

    @model_validator(mode="before")
    @classmethod
    def hoist_additional_tools(cls, data: Any) -> Any:
        """Move the tools of an ``additional_tools`` input item into ``tools``.

        Codex declares its tools that way; hoisted here, they are validated as
        tools and resolve when the reply's calls are parsed. Malformed shapes
        are left in place for validation to reject.
        """
        if not isinstance(data, dict):
            return data
        items = data.get("input")
        tools = data.get("tools")
        if not isinstance(items, list) or not isinstance(
                tools, (list, type(None))):
            return data
        hoisted = []
        kept = []
        for item in items:
            if (isinstance(item, dict)
                    and item.get("type") == "additional_tools"
                    and isinstance(item.get("tools"), list)):
                hoisted.extend(item["tools"])
            else:
                kept.append(item)
        if not hoisted:
            return data
        data = dict(data)
        data["input"] = kept
        data["tools"] = list(data.get("tools") or []) + hoisted
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_background(cls, data):
        if not data.get("background"):
            return data
        if not data.get("store", True):
            raise ValueError("background can only be used when `store` is true")
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_prompt(cls, data):
        if data.get("prompt") is not None:
            raise ValueError("prompt template is not supported")
        return data

    @field_validator("thinking_token_budget", mode="before")
    @classmethod
    def check_thinking_token_budget(cls, value):
        return validate_thinking_token_budget(value)


class InputTokensDetails(OpenAIBaseModel):
    cached_tokens: int
    # Required by the openai SDK models that re-validate this payload when it
    # is embedded in a streaming event. Omitting it fails validation while the
    # response is being streamed, which truncates the stream with no
    # terminating event and leaves the client waiting forever. Prompt cache
    # writes are not tracked separately, so this is reported as zero.
    cache_write_tokens: int = 0


class OutputTokensDetails(OpenAIBaseModel):
    reasoning_tokens: int


class ResponseUsage(OpenAIBaseModel):
    input_tokens: int
    input_tokens_details: InputTokensDetails
    output_tokens: int
    output_tokens_details: OutputTokensDetails
    total_tokens: int


StreamingResponsesResponse: TypeAlias = Union[
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponseCompletedEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseContentPartAddedEvent,
    ResponseContentPartDoneEvent,
    ResponseReasoningTextDeltaEvent,
    ResponseReasoningTextDoneEvent,
    ResponseCodeInterpreterCallInProgressEvent,
    ResponseCodeInterpreterCallCodeDeltaEvent,
    ResponseWebSearchCallInProgressEvent,
    ResponseWebSearchCallSearchingEvent,
    ResponseWebSearchCallCompletedEvent,
    ResponseCodeInterpreterCallCodeDoneEvent,
    ResponseCodeInterpreterCallInterpretingEvent,
    ResponseCodeInterpreterCallCompletedEvent,
]

_RESPONSES_HANDOFF_FIELDS = (
    "disaggregated_params",
    "prompt_token_ids",
    "prompt_token_ids_b64",
    "finish_reason",
)


class ResponsesResponse(OpenAIBaseModel):
    id: str = Field(default_factory=lambda: f"resp_{str(uuid.uuid4().hex)}")
    created_at: int = Field(default_factory=lambda: int(time.time()))
    # error: Optional[ResponseError] = None
    incomplete_details: Optional[IncompleteDetails] = None
    instructions: Optional[str] = None
    metadata: Optional[Metadata] = None
    model: str
    object: Literal["response"] = "response"
    output: list[ResponseOutputItem]
    parallel_tool_calls: bool
    temperature: float
    tool_choice: ToolChoice
    tools: list[Tool]
    top_p: float
    background: bool
    max_output_tokens: Optional[int] = None
    max_tool_calls: Optional[int] = None
    previous_response_id: Optional[str] = None
    prompt: Optional[ResponsePrompt] = None
    reasoning: Optional[Reasoning] = None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"]
    status: ResponseStatus
    text: Optional[ResponseTextConfig] = None
    top_logprobs: int
    truncation: Literal["auto", "disabled"]
    usage: Optional[ResponseUsage] = None
    user: Optional[str] = None

    # TensorRT-LLM extensions, set only on a context-only response: the
    # disaggregated handoff, and the engine's raw finish reason, which tells the
    # orchestrator whether a generation phase is pending ("length" and
    # "not_finished" both map to status "incomplete").
    disaggregated_params: Optional[DisaggregatedParams] = Field(default=None)
    prompt_token_ids: Optional[List[int]] = None
    prompt_token_ids_b64: Optional[str] = None
    finish_reason: Optional[str] = None

    @model_serializer(mode="wrap")
    def _omit_absent_handoff(self, handler: Any) -> Any:
        # The handoff fields are unset on every response a client sees, so they
        # are left out rather than serialized as null in each body and event.
        data = handler(self)
        if isinstance(data, dict):
            for key in _RESPONSES_HANDOFF_FIELDS:
                if data.get(key, ...) is None:
                    data.pop(key)
        return data

    @classmethod
    def from_request(
        cls,
        request: ResponsesRequest,
        sampling_params: SamplingParams,
        model_name: str,
        created_time: int,
        output: list[ResponseOutputItem],
        status: ResponseStatus,
        usage: Optional[ResponseUsage] = None,
    ) -> "ResponsesResponse":
        return cls(
            id=request.request_id,
            created_at=created_time,
            instructions=request.instructions,
            metadata=request.metadata,
            model=model_name,
            output=output,
            parallel_tool_calls=request.parallel_tool_calls,
            temperature=sampling_params.temperature,
            tool_choice=request.tool_choice,
            tools=request.tools,
            top_p=sampling_params.top_p,
            background=request.background,
            max_output_tokens=sampling_params.max_tokens,
            max_tool_calls=request.max_tool_calls,
            previous_response_id=request.previous_response_id,
            prompt=request.prompt,
            reasoning=request.reasoning,
            service_tier=request.service_tier,
            status=status,
            text=request.text,
            top_logprobs=sampling_params.logprobs,
            truncation=request.truncation,
            user=request.user,
            usage=usage,
        )


class ResponsesStreamResponse(OpenAIBaseModel):
    response: ResponsesResponse
    sequence_number: int
    type: Literal["response.created", "response.in_progress",
                  "response.completed", "response.failed",
                  "response.incomplete"]
