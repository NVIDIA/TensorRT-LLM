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
"""Pydantic models for the Anthropic Messages API (``POST /v1/messages``).

Wire-format reference: https://platform.claude.com/docs/en/api/messages

These models cover the subset of the protocol required to serve Anthropic
SDK clients and Claude Code. Request models are permissive (``extra="allow"``)
because Anthropic clients attach evolving auxiliary fields (``metadata``,
``betas``, ``output_config``, ...) that must not fail validation; response
models emit only the fields this server populates.
"""

import time
import uuid
from typing import Annotated, Any, Dict, List, Literal, Optional, Union, get_args

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag


class AnthropicBaseModel(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


# ---------------------------------------------------------------------------
# Content blocks
# ---------------------------------------------------------------------------


class AnthropicTextBlock(AnthropicBaseModel):
    type: Literal["text"] = "text"
    text: str


class AnthropicImageSource(AnthropicBaseModel):
    type: Literal["base64", "url"]
    media_type: Optional[str] = None
    data: Optional[str] = None
    url: Optional[str] = None


class AnthropicImageBlock(AnthropicBaseModel):
    type: Literal["image"] = "image"
    source: AnthropicImageSource


class AnthropicToolUseBlock(AnthropicBaseModel):
    type: Literal["tool_use"] = "tool_use"
    id: str = Field(default_factory=lambda: f"toolu_{uuid.uuid4().hex}")
    name: str
    input: Dict[str, Any] = Field(default_factory=dict)


class AnthropicToolResultBlock(AnthropicBaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool_use_id: str
    content: Optional[Union[str, List["AnthropicContentBlock"]]] = None
    is_error: Optional[bool] = None


class AnthropicThinkingBlock(AnthropicBaseModel):
    type: Literal["thinking"] = "thinking"
    thinking: str
    signature: Optional[str] = None


class AnthropicRedactedThinkingBlock(AnthropicBaseModel):
    type: Literal["redacted_thinking"] = "redacted_thinking"
    data: Optional[str] = None


class AnthropicUnknownBlock(AnthropicBaseModel):
    """Catch-all so an unrecognised block reaches the adapter.

    Without a permissive last member the union is closed, and a block type this
    server does not model fails validation at the FastAPI boundary: the client
    gets an error listing every failed union member instead of a message naming
    the block, and the adapter's own handling for unsupported blocks is
    unreachable.
    """

    type: str


# Tag names must equal each model's ``type`` literal, so derive them from the
# models rather than restating the list here.
_CONTENT_BLOCK_MODELS = (
    AnthropicTextBlock,
    AnthropicImageBlock,
    AnthropicToolUseBlock,
    AnthropicToolResultBlock,
    AnthropicThinkingBlock,
    AnthropicRedactedThinkingBlock,
)
_KNOWN_CONTENT_BLOCK_TYPES = frozenset(
    get_args(model.model_fields["type"].annotation)[0] for model in _CONTENT_BLOCK_MODELS
)


def _content_block_tag(value: Any) -> str:
    """Pick the block model from ``type`` before any field is validated.

    An order-based union accepts the first member that validates, which sends
    a *malformed known* block to the catch-all: ``{"type": "text"}`` with no
    ``text`` fails AnthropicTextBlock, validates as AnthropicUnknownBlock, and
    then reaches the adapter, which dispatches on ``type`` and raises
    AttributeError on ``block.text`` -- a 500 for what is a client error.

    Selecting on ``type`` first makes that block fail as a text block and name
    the field it is missing, and leaves the catch-all for the case it exists
    for: a block type this server does not model at all.
    """
    block_type = value.get("type") if isinstance(value, dict) else getattr(value, "type", None)
    return block_type if block_type in _KNOWN_CONTENT_BLOCK_TYPES else "unknown"


AnthropicContentBlock = Annotated[
    Union[
        Annotated[AnthropicTextBlock, Tag("text")],
        Annotated[AnthropicImageBlock, Tag("image")],
        Annotated[AnthropicToolUseBlock, Tag("tool_use")],
        Annotated[AnthropicToolResultBlock, Tag("tool_result")],
        Annotated[AnthropicThinkingBlock, Tag("thinking")],
        Annotated[AnthropicRedactedThinkingBlock, Tag("redacted_thinking")],
        Annotated[AnthropicUnknownBlock, Tag("unknown")],
    ],
    Discriminator(_content_block_tag),
]

AnthropicToolResultBlock.model_rebuild()


class AnthropicMessage(AnthropicBaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[AnthropicContentBlock]]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

# Anthropic-provided tools use a versioned ``type``. Server tools are executed
# by Anthropic's API, while schema client tools are executed by the caller.
SERVER_TOOL_TYPE_PREFIXES = (
    "web_search",
    "web_fetch",
    "code_execution",
    "tool_search_tool_",
    "advisor_",
    "mcp_toolset",
)
SCHEMA_CLIENT_TOOL_TYPE_PREFIXES = (
    "bash_",
    "text_editor_",
    "computer_",
    "memory_",
)


class AnthropicTool(AnthropicBaseModel):
    name: str
    type: Optional[str] = None
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None

    def is_server_tool(self) -> bool:
        if self.type is None or self.type == "custom":
            return False
        return any(self.type.startswith(prefix) for prefix in SERVER_TOOL_TYPE_PREFIXES)

    def is_schema_client_tool(self) -> bool:
        if self.type is None or self.type == "custom":
            return False
        return any(self.type.startswith(prefix) for prefix in SCHEMA_CLIENT_TOOL_TYPE_PREFIXES)


class AnthropicToolChoice(AnthropicBaseModel):
    type: Literal["auto", "any", "tool", "none"]
    name: Optional[str] = None
    disable_parallel_tool_use: Optional[bool] = None


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class AnthropicThinkingEnabled(AnthropicBaseModel):
    """Extended thinking with an explicit budget.

    Anthropic's documented floor is 1024; below that the budget cannot hold a
    usable reasoning trace. Enforced here so the client gets a field-level 400
    rather than a message assembled by hand in the adapter.
    """

    type: Literal["enabled"] = "enabled"
    budget_tokens: int = Field(ge=1024)


class AnthropicThinkingAdaptive(AnthropicBaseModel):
    """Thinking on, budget left to the server.

    extra="forbid" overrides the permissive base: these variants take no
    budget, and accepting one silently would let a client believe it had set a
    budget that is in fact ignored.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: Literal["adaptive"] = "adaptive"


class AnthropicThinkingDisabled(AnthropicBaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    type: Literal["disabled"] = "disabled"


AnthropicThinkingConfig = Annotated[
    Union[AnthropicThinkingEnabled, AnthropicThinkingAdaptive, AnthropicThinkingDisabled],
    Field(discriminator="type"),
]


class AnthropicMessagesRequest(AnthropicBaseModel):
    model: str
    # min_length is the only guard that catches an empty conversation. The
    # check in convert_anthropic_request runs after _convert_messages, which
    # prepends a system message when `system` is set -- so the converted list
    # is non-empty and the request reaches the engine with nothing to answer.
    messages: List[AnthropicMessage] = Field(min_length=1)
    # Forwarded to max_completion_tokens, where 0 or a negative is meaningless.
    max_tokens: int = Field(ge=1)
    system: Optional[Union[str, List[AnthropicTextBlock]]] = None
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[AnthropicToolChoice] = None
    # Anthropic's documented ranges. Rejecting here turns a sampler-level
    # failure deep in the engine into a 400 naming the offending field.
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=0)
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    metadata: Optional[Dict[str, Any]] = None
    # Extended-thinking control. A discriminated union, so an unknown type or
    # a budget below the floor is rejected by the model and surfaces through
    # the Anthropic error envelope like any other field.
    thinking: Optional[AnthropicThinkingConfig] = None
    # Claude Code attaches output_config (effort, format) and betas.
    output_config: Optional[Dict[str, Any]] = None
    betas: Optional[List[str]] = None
    # Context editing directives, e.g.
    # {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}.
    # Clients send this alongside extended thinking to say whether reasoning
    # from earlier turns should survive into the next prompt.
    context_management: Optional[Dict[str, Any]] = None


class AnthropicCountTokensRequest(AnthropicBaseModel):
    """Body of ``POST /v1/messages/count_tokens``.

    Deliberately mirrors AnthropicMessagesRequest minus the generation
    controls: the count has to be taken over exactly the prompt the real
    request would build, so anything that changes rendering (system, tools,
    thinking, context_management) has to be accepted here too.
    """

    model: str
    messages: List[AnthropicMessage] = Field(min_length=1)
    system: Optional[Union[str, List[AnthropicTextBlock]]] = None
    tools: Optional[List[AnthropicTool]] = None
    tool_choice: Optional[AnthropicToolChoice] = None
    thinking: Optional[AnthropicThinkingConfig] = None
    output_config: Optional[Dict[str, Any]] = None
    betas: Optional[List[str]] = None
    context_management: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------

AnthropicStopReason = Literal["end_turn", "max_tokens", "stop_sequence", "tool_use", "refusal"]


class AnthropicUsage(AnthropicBaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None


class AnthropicMessagesResponse(AnthropicBaseModel):
    id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex}")
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    model: str
    content: List[AnthropicContentBlock] = Field(default_factory=list)
    stop_reason: Optional[AnthropicStopReason] = None
    stop_sequence: Optional[str] = None
    usage: AnthropicUsage = Field(default_factory=AnthropicUsage)


class AnthropicCountTokensResponse(AnthropicBaseModel):
    input_tokens: int


# ---------------------------------------------------------------------------
# Message Batches
# ---------------------------------------------------------------------------

# Only these three. The batch as a whole is in_progress until every request has
# reached a terminal state, at which point it is ended - "ended" covers success,
# failure and expiry alike, and the breakdown lives in request_counts. Some
# summaries of this API list processing/completed/failed/expired instead; those
# values are not what the official schema or the SDKs use, and a client
# deserializing this field would reject them.
AnthropicBatchProcessingStatus = Literal["in_progress", "canceling", "ended"]

# What one line of the results .jsonl says happened to one request.
AnthropicBatchResultType = Literal["succeeded", "errored", "canceled", "expired"]

# Anthropic's documented ceiling. Enforced because every request in a batch is
# held in memory here (see anthropic_batches), so an unbounded batch is a way
# to exhaust the server's memory with a single call.
ANTHROPIC_BATCH_MAX_REQUESTS = 100_000

_CUSTOM_ID_PATTERN = r"^[a-zA-Z0-9_-]{1,64}$"


class AnthropicBatchRequestCounts(AnthropicBaseModel):
    """Tally of requests by state; the sum always equals the batch size."""

    canceled: int = 0
    errored: int = 0
    expired: int = 0
    processing: int = 0
    succeeded: int = 0


class AnthropicBatchRequestItem(AnthropicBaseModel):
    # Results may come back in any order, so the client matches them up by
    # custom_id rather than by position.
    custom_id: str = Field(pattern=_CUSTOM_ID_PATTERN)
    # The params of a batched request are an ordinary Messages request, so it
    # reuses that model outright: validation, thinking rules and tool handling
    # then cannot drift between the batched and unbatched paths.
    params: AnthropicMessagesRequest


class AnthropicCreateBatchRequest(AnthropicBaseModel):
    requests: List[AnthropicBatchRequestItem] = Field(
        min_length=1, max_length=ANTHROPIC_BATCH_MAX_REQUESTS
    )


class AnthropicMessageBatch(AnthropicBaseModel):
    id: str = Field(default_factory=lambda: f"msgbatch_{uuid.uuid4().hex}")
    type: Literal["message_batch"] = "message_batch"
    processing_status: AnthropicBatchProcessingStatus = "in_progress"
    request_counts: AnthropicBatchRequestCounts = Field(default_factory=AnthropicBatchRequestCounts)
    created_at: str
    expires_at: str
    # All three stay null until the corresponding thing happens, which is how a
    # client distinguishes "still running" from "finished" from "archived".
    ended_at: Optional[str] = None
    archived_at: Optional[str] = None
    cancel_initiated_at: Optional[str] = None
    results_url: Optional[str] = None


class AnthropicBatchList(AnthropicBaseModel):
    data: List[AnthropicMessageBatch] = Field(default_factory=list)
    has_more: bool = False
    first_id: Optional[str] = None
    last_id: Optional[str] = None


class AnthropicBatchDeleteResponse(AnthropicBaseModel):
    id: str
    type: Literal["message_batch_deleted"] = "message_batch_deleted"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

AnthropicErrorType = Literal[
    "invalid_request_error",
    "authentication_error",
    "permission_error",
    "not_found_error",
    "request_too_large",
    "rate_limit_error",
    "api_error",
    "overloaded_error",
]


class AnthropicError(AnthropicBaseModel):
    type: AnthropicErrorType = "api_error"
    message: str


class AnthropicErrorResponse(AnthropicBaseModel):
    type: Literal["error"] = "error"
    error: AnthropicError
    request_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Streaming events (SSE)
#
# Wire framing is ``event: <type>\ndata: <json>\n\n``; the ``type`` field
# inside the payload must match the event name.
# ---------------------------------------------------------------------------


class AnthropicMessageStartEvent(AnthropicBaseModel):
    type: Literal["message_start"] = "message_start"
    message: AnthropicMessagesResponse


class AnthropicContentBlockStartEvent(AnthropicBaseModel):
    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: AnthropicContentBlock


class AnthropicTextDelta(AnthropicBaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class AnthropicInputJsonDelta(AnthropicBaseModel):
    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


class AnthropicThinkingDelta(AnthropicBaseModel):
    type: Literal["thinking_delta"] = "thinking_delta"
    thinking: str


class AnthropicSignatureDelta(AnthropicBaseModel):
    type: Literal["signature_delta"] = "signature_delta"
    signature: str


AnthropicContentDelta = Union[
    AnthropicTextDelta, AnthropicInputJsonDelta, AnthropicThinkingDelta, AnthropicSignatureDelta
]


class AnthropicContentBlockDeltaEvent(AnthropicBaseModel):
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: AnthropicContentDelta


class AnthropicContentBlockStopEvent(AnthropicBaseModel):
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class AnthropicMessageDelta(AnthropicBaseModel):
    stop_reason: Optional[AnthropicStopReason] = None
    stop_sequence: Optional[str] = None


class AnthropicMessageDeltaEvent(AnthropicBaseModel):
    type: Literal["message_delta"] = "message_delta"
    delta: AnthropicMessageDelta
    usage: Optional[AnthropicUsage] = None


class AnthropicMessageStopEvent(AnthropicBaseModel):
    type: Literal["message_stop"] = "message_stop"


class AnthropicErrorEvent(AnthropicBaseModel):
    type: Literal["error"] = "error"
    error: AnthropicError


def anthropic_sse(event: AnthropicBaseModel) -> str:
    """Serialize an event model into one Anthropic SSE frame."""
    return f"event: {event.type}\ndata: {event.model_dump_json(exclude_none=True)}\n\n"


def current_timestamp() -> int:
    return int(time.time())
