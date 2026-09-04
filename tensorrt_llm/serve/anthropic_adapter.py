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
"""Adapter between the Anthropic Messages API and the OpenAI chat pipeline.

Request direction: :class:`AnthropicMessagesRequest` is translated into a
:class:`ChatCompletionRequest` so the existing ``openai_chat`` path (chat
template, tool parser, reasoning parser, post-processing) is reused verbatim.
Response direction: the resulting :class:`ChatCompletionResponse` is
translated back into an :class:`AnthropicMessagesResponse`.

The adapter is a pure protocol layer: it never touches the tokenizer,
chat templates, or the engine.
"""

import json
import re
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from fastapi.responses import JSONResponse

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.anthropic_protocol import (
    AnthropicContentBlockDeltaEvent,
    AnthropicContentBlockStartEvent,
    AnthropicContentBlockStopEvent,
    AnthropicCountTokensRequest,
    AnthropicError,
    AnthropicErrorEvent,
    AnthropicErrorResponse,
    AnthropicErrorType,
    AnthropicInputJsonDelta,
    AnthropicMessageDelta,
    AnthropicMessageDeltaEvent,
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicMessageStartEvent,
    AnthropicMessageStopEvent,
    AnthropicStopReason,
    AnthropicTextBlock,
    AnthropicTextDelta,
    AnthropicThinkingBlock,
    AnthropicThinkingDelta,
    AnthropicToolUseBlock,
    AnthropicUsage,
    anthropic_sse,
)
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatCompletionToolsParam,
    FunctionDefinition,
    UsageInfo,
)

# OpenAI finish_reason -> Anthropic stop_reason. A string-valued OpenAI
# stop_reason identifies the matched stop sequence and is handled separately.
STOP_REASON_MAP: Dict[str, AnthropicStopReason] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
}


class AnthropicRequestError(ValueError):
    """Invalid Anthropic request; maps to a 400 with an Anthropic envelope."""


class AnthropicResponseError(ValueError):
    """Invalid upstream model response; maps to an Anthropic API error."""


def anthropic_error_response(
    message: str, error_type: AnthropicErrorType = "api_error", status_code: int = 500
) -> JSONResponse:
    envelope = AnthropicErrorResponse(error=AnthropicError(type=error_type, message=message))
    return JSONResponse(content=envelope.model_dump(exclude_none=True), status_code=status_code)


# ---------------------------------------------------------------------------
# Request conversion: Anthropic -> ChatCompletionRequest
# ---------------------------------------------------------------------------


# Claude Code prepends a per-request billing block to the system prompt. It is
# pure client telemetry - version, entrypoint, subagent flag - and carries no
# instruction for the model, so forwarding it just spends tokens on every turn
# and pollutes the system prompt.
#
# Matched by shape, not by an inventory of known fields: a literal marker,
# then any number of `key=value;` pairs in any order. The client's field list
# drifts (`cc_is_subagent`, and the `sdk-py` entrypoint the Claude Agent SDK
# sends), and a pattern pinned to a fixed inventory stops matching as soon as
# it does -- silently, leaving the telemetry in every prompt with nothing to
# indicate the strip has become a no-op.
_ANTHROPIC_BILLING_SYSTEM_BLOCK = re.compile(
    r"x-anthropic-billing-header:\s*"
    r"(?:[A-Za-z_][\w.-]*=[^;]*;\s*)+"
)


def _is_anthropic_billing_system_block(text: str) -> bool:
    """Recognize Claude Code's model-irrelevant per-request billing block."""
    return _ANTHROPIC_BILLING_SYSTEM_BLOCK.fullmatch(text) is not None


def _system_text_parts(system: Optional[Union[str, List[AnthropicTextBlock]]]) -> List[str]:
    if system is None:
        return []
    if isinstance(system, str):
        return [system] if system else []
    parts = []
    for index, block in enumerate(system):
        text = block.text
        # Only position 0, which is where the client puts it. Restricting by
        # position keeps a genuine user system block that happens to look like
        # telemetry from being silently dropped.
        if index == 0 and text and _is_anthropic_billing_system_block(text):
            continue
        if text:
            parts.append(text)
    return parts


def _image_part(block: Any) -> Dict[str, Any]:
    source = block.source
    if source.type == "url" and source.url:
        return {"type": "image_url", "image_url": {"url": source.url}}
    if source.type == "base64" and source.data:
        media_type = source.media_type or "image/png"
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{media_type};base64,{source.data}"},
        }
    # url and data are both Optional, so a source can validate yet carry no
    # image. Dropping it would send the model a prompt missing content the
    # client believes it attached.
    raise AnthropicRequestError(f"image block with source type {source.type!r} has no url or data")


def _tool_result_text(content: Any) -> str:
    """Flatten a tool_result content payload into plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        block_type = block.type
        if block_type == "text":
            parts.append(block.text)
        else:
            raise AnthropicRequestError(
                f"Content block type {block_type!r} inside tool_result is not supported"
            )
    return "\n".join(parts)


def _convert_messages(request: AnthropicMessagesRequest) -> List[Dict[str, Any]]:
    system_parts = _system_text_parts(request.system)
    converted: List[Dict[str, Any]] = []

    # Only system messages that precede the first real turn belong to the opening
    # prompt. Anthropic clients also append transient system messages (task
    # reminders, background-task notifications) at the *tail* of the history;
    # hoisting those into the opening block rewrites the very front of the prompt
    # and invalidates the whole KV prefix -- tens of thousands of unchanged tokens
    # get re-prefilled every time one arrives. They stay where the client put them.
    in_leading_system_run = True

    for message in request.messages:
        if message.role == "system" and in_leading_system_run:
            if isinstance(message.content, str):
                system_parts.append(message.content)
            else:
                system_parts.extend(block.text for block in message.content if block.type == "text")
            continue

        in_leading_system_run = False

        if isinstance(message.content, str):
            converted.append({"role": message.role, "content": message.content})
            continue

        # Content parts accumulated for the current role; flushed before any
        # role:"tool" message so ordering user(pre) -> tool -> user(post) is
        # preserved.
        parts: List[Dict[str, Any]] = []
        tool_calls: List[Dict[str, Any]] = []
        reasoning_parts: List[str] = []

        def flush_parts():
            if parts:
                converted_message: Dict[str, Any] = {
                    "role": message.role,
                    "content": list(parts),
                }
                if message.role == "assistant" and reasoning_parts:
                    converted_message["reasoning"] = "".join(reasoning_parts)
                    reasoning_parts.clear()
                converted.append(converted_message)
                parts.clear()

        for block in message.content:
            block_type = block.type
            if block_type == "text":
                parts.append({"type": "text", "text": block.text})
            elif block_type == "image":
                parts.append(_image_part(block))
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": json.dumps(block.input),
                        },
                    }
                )
            elif block_type == "tool_result":
                flush_parts()
                tool_result_text = _tool_result_text(block.content)
                if block.is_error:
                    tool_result_text = f"Tool execution failed: {tool_result_text}"
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.tool_use_id,
                        "content": tool_result_text,
                    }
                )
            elif block_type == "thinking":
                if message.role != "assistant":
                    raise AnthropicRequestError(
                        "thinking content blocks are only valid in assistant messages"
                    )
                reasoning_parts.append(block.thinking)
            elif block_type == "redacted_thinking":
                raise AnthropicRequestError(
                    "redacted_thinking history is not supported by this server"
                )
            else:
                # Reachable via AnthropicUnknownBlock. Refusing beats dropping:
                # a silently discarded block changes what the model is asked
                # without the client ever learning the content did not arrive.
                raise AnthropicRequestError(
                    f"Anthropic content block type {block_type!r} is not supported by this server"
                )

        if message.role == "assistant" and tool_calls:
            non_text = [p for p in parts if p.get("type") != "text"]
            if non_text:
                # The chat message being built here carries text plus
                # tool_calls; an image alongside them has nowhere to go and
                # would be dropped without a trace.
                kinds = sorted({p.get("type", "?") for p in non_text})
                raise AnthropicRequestError(
                    f"assistant messages combining tool_use with {kinds} content "
                    "are not supported by this server"
                )
            text_content = "".join(p["text"] for p in parts if p.get("type") == "text")
            assistant_message: Dict[str, Any] = {
                "role": "assistant",
                "content": text_content or None,
                "tool_calls": tool_calls,
            }
            if reasoning_parts:
                assistant_message["reasoning"] = "".join(reasoning_parts)
                reasoning_parts.clear()
            converted.append(assistant_message)
            parts.clear()
        else:
            if tool_calls:
                # Only an assistant turn can carry tool_use; anywhere else the
                # accumulated calls have no destination in the chat message.
                raise AnthropicRequestError(
                    f"tool_use content blocks are only valid in assistant messages, "
                    f"not {message.role!r}"
                )
            flush_parts()
            if message.role == "assistant" and reasoning_parts:
                converted.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "reasoning": "".join(reasoning_parts),
                    }
                )

    if system_parts:
        converted.insert(0, {"role": "system", "content": "\n\n".join(system_parts)})
    return converted


def _convert_tools(request: AnthropicMessagesRequest) -> Optional[List[ChatCompletionToolsParam]]:
    if not request.tools:
        return None
    # tool_choice "none" forbids *calling* a tool, not knowing about one: the
    # definitions stay in the rendered prompt. Dropping them would leave the
    # model unable to interpret a later tool_result, and would change the
    # rendered prefix on the turn a client switches auto -> none, discarding
    # the KV prefix _convert_messages preserves. _convert_tool_choice carries
    # the prohibition instead.
    # "none" forbids calling anything, so a server tool listed under it can
    # never be invoked and is not an error -- it is simply dropped. Client
    # tools are still converted, so the rendered prefix stays stable.
    forbids_calls = request.tool_choice is not None and request.tool_choice.type == "none"
    tools = []
    for tool in request.tools:
        if tool.is_server_tool():
            if forbids_calls:
                continue
            raise AnthropicRequestError(
                f"Anthropic server tool {tool.name!r} (type={tool.type!r}) "
                "is not supported by this server"
            )
        if tool.input_schema is None:
            if tool.is_schema_client_tool():
                raise AnthropicRequestError(
                    f"Anthropic schema client tool {tool.name!r} "
                    f"(type={tool.type!r}) is recognized, but its built-in "
                    "input schema is not implemented by this server"
                )
            raise AnthropicRequestError(f"Client tool {tool.name!r} requires input_schema")
        tools.append(
            ChatCompletionToolsParam(
                function=FunctionDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.input_schema,
                    strict=tool.strict,
                )
            )
        )
    return tools or None


def _convert_tool_choice(
    request: AnthropicMessagesRequest,
    tools: Optional[List[ChatCompletionToolsParam]],
) -> Optional[str]:
    choice = request.tool_choice
    if choice is None:
        # check_tool_choice validator defaults to "auto" when tools are set.
        return "auto" if tools else None
    if choice.disable_parallel_tool_use and choice.type != "none":
        raise AnthropicRequestError("tool_choice.disable_parallel_tool_use=true is not supported")
    if choice.type == "none":
        return "none"
    if tools is None:
        raise AnthropicRequestError(
            f"tool_choice type {choice.type!r} requires at least one "
            "client-executable tool; all provided tools were server tools "
            "or the tools list was empty"
        )
    if choice.type == "auto":
        return "auto"
    if choice.type == "any":
        raise AnthropicRequestError(
            "Anthropic tool_choice type 'any' is not supported because the "
            "chat pipeline cannot require an arbitrary tool call"
        )
    if choice.type == "tool":
        if not choice.name:
            raise AnthropicRequestError("tool_choice type 'tool' requires a 'name'")
        tool_names = {t.function.name for t in tools}
        if choice.name not in tool_names:
            raise AnthropicRequestError(f"tool_choice names unknown tool {choice.name!r}")
        # Same underlying limitation as 'any' above, and rejected for the same
        # reason. Forcing a named tool makes the chat pipeline emit the call
        # without running a tool parser, so the call arrives with an empty
        # arguments string. Converting that to a tool_use block would mean
        # inventing an input the model never produced - a call the client would
        # execute with missing arguments. Declining keeps the failure at request
        # time instead of surfacing as a 500 from the response converter.
        raise AnthropicRequestError(
            "Anthropic tool_choice type 'tool' is not supported because the "
            "chat pipeline emits a forced call without parsed arguments; use "
            "tool_choice 'auto' and describe the required tool in the prompt"
        )
    raise AnthropicRequestError(f"Unsupported tool_choice {choice.type!r}")


# Chat-template kwargs that turn off pruning of earlier-turn reasoning. The
# name differs per model family but the meaning is identical, and a template
# that does not know a key simply ignores it, so both are always sent
# together:
#   * `clear_thinking`  - GLM-family Jinja templates
#   * `drop_thinking`   - DeepSeek-V4 (`DeepseekV4Tokenizer.apply_chat_template`)
_KEEP_ALL_THINKING_KWARGS = {"clear_thinking": False, "drop_thinking": False}


def _thinking_retention_kwargs(
    context_management: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Translate Anthropic context-editing directives into chat-template kwargs.

    Clients that enable extended thinking also declare what should happen to
    reasoning from earlier turns, e.g.::

        {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]}

    Chat templates default to pruning that reasoning (everything before the
    last user message), so a ``keep: "all"`` directive has to be forwarded or
    the client's stated intent is silently dropped.

    Only ``keep: "all"`` is translated. Any other retention policy is left to
    the template default rather than guessed at, because dropping *more* than
    asked is the safer failure mode than keeping more than asked.
    """
    if not isinstance(context_management, dict):
        return {}

    edits = context_management.get("edits")
    if not isinstance(edits, list):
        return {}

    for edit in edits:
        if not isinstance(edit, dict):
            continue
        edit_type = edit.get("type")
        # The type carries a dated suffix (clear_thinking_20251015); match on
        # the family so a future revision keeps working.
        if not isinstance(edit_type, str) or not edit_type.startswith("clear_thinking"):
            continue
        keep = edit.get("keep")
        if keep == "all":
            return dict(_KEEP_ALL_THINKING_KWARGS)
        logger.warning(
            f"Unsupported {edit_type!r} retention policy {keep!r}; falling back "
            "to the chat template default (earlier-turn reasoning is pruned)."
        )
    return {}


def convert_anthropic_request(request: AnthropicMessagesRequest) -> ChatCompletionRequest:
    """Translate an Anthropic Messages request into a chat completion request."""
    messages = _convert_messages(request)
    if not messages:
        raise AnthropicRequestError("messages must not be empty")
    tools = _convert_tools(request)
    tool_choice = _convert_tool_choice(request, tools)

    chat_request: Dict[str, Any] = {
        "model": request.model,
        "messages": messages,
        "max_completion_tokens": request.max_tokens,
        "stream": bool(request.stream),
    }
    if tools is not None:
        chat_request["tools"] = [t.model_dump() for t in tools]
    if tool_choice is not None:
        chat_request["tool_choice"] = tool_choice
    if request.temperature is not None:
        chat_request["temperature"] = request.temperature
    if request.top_p is not None:
        chat_request["top_p"] = request.top_p
    if request.top_k is not None:
        chat_request["top_k"] = request.top_k
    if request.stop_sequences:
        chat_request["stop"] = list(request.stop_sequences)

    # Anthropic extended-thinking controls need to reach both the tokenizer
    # template and the reasoning postprocessor.  DeepSeek-V4 selects thinking
    # mode from these template kwargs; without them, configuring the V4
    # reasoning parser alone leaves it in identity mode.
    chat_template_kwargs: Dict[str, Any] = {}
    if request.thinking is not None:
        # The union already rejected an unknown type, a budget below the floor,
        # and budget_tokens on a variant that does not take one. What remains
        # is the cross-field rule, which spans two models and so cannot live
        # on either of them.
        chat_template_kwargs["enable_thinking"] = request.thinking.type != "disabled"
        if request.thinking.type == "enabled":
            if request.thinking.budget_tokens >= request.max_tokens:
                raise AnthropicRequestError("thinking budget_tokens must be less than max_tokens")
            chat_request["thinking_token_budget"] = request.thinking.budget_tokens

    if request.output_config:
        reasoning_effort = request.output_config.get("effort")
        if isinstance(reasoning_effort, str) and reasoning_effort:
            chat_template_kwargs["reasoning_effort"] = reasoning_effort
        output_format = request.output_config.get("format")
        if output_format is not None:
            if not isinstance(output_format, dict):
                raise AnthropicRequestError("output_config.format must be an object")
            if output_format.get("type") != "json_schema":
                raise AnthropicRequestError(
                    "Only output_config.format type 'json_schema' is supported"
                )
            schema = output_format.get("schema")
            if not isinstance(schema, dict):
                raise AnthropicRequestError("output_config.format.schema must be an object")
            chat_request["response_format"] = {
                "type": "json_schema",
                "json_schema": {"schema": schema},
            }

    chat_template_kwargs.update(_thinking_retention_kwargs(request.context_management))

    if chat_template_kwargs:
        chat_request["chat_template_kwargs"] = chat_template_kwargs
    if request.stream:
        chat_request["stream_options"] = {
            "include_usage": True,
            "continuous_usage_stats": True,
        }
    return ChatCompletionRequest(**chat_request)


def convert_anthropic_count_tokens_request(
    request: AnthropicCountTokensRequest,
) -> ChatCompletionRequest:
    """Translate count-tokens input through the Messages request converter.

    Reusing convert_anthropic_request is the point: a count taken by any other
    route would drift from the prompt the real request builds, and a client
    that sizes its context against a stale number will either overflow the
    window or compact when it did not need to.
    """
    # max_tokens only has to clear the thinking budget: convert_anthropic_request
    # rejects a budget that is not strictly smaller, and counting must not fail
    # on a request that would have been accepted.
    max_tokens = 1
    if request.thinking is not None and request.thinking.type == "enabled":
        max_tokens = request.thinking.budget_tokens + 1
    messages_request = AnthropicMessagesRequest(
        model=request.model,
        messages=request.messages,
        max_tokens=max_tokens,
        system=request.system,
        tools=request.tools,
        tool_choice=request.tool_choice,
        thinking=request.thinking,
        output_config=request.output_config,
        betas=request.betas,
        # Retention changes how much history is rendered, so the count has to
        # be taken under the same policy as the real request.
        context_management=request.context_management,
    )
    return convert_anthropic_request(messages_request)


# ---------------------------------------------------------------------------
# Response conversion: ChatCompletionResponse -> Anthropic
# ---------------------------------------------------------------------------


def map_stop_reason(finish_reason: Optional[str]) -> AnthropicStopReason:
    if finish_reason is None:
        return "end_turn"
    mapped = STOP_REASON_MAP.get(finish_reason)
    if mapped is None:
        logger.warning(f"Unmapped finish_reason {finish_reason!r} defaulted to 'end_turn'")
        return "end_turn"
    return mapped


def _map_stop_result(
    finish_reason: Optional[str], stop_reason: Optional[Union[int, str]]
) -> tuple[AnthropicStopReason, Optional[str]]:
    if finish_reason == "stop" and isinstance(stop_reason, str):
        return "stop_sequence", stop_reason
    return map_stop_reason(finish_reason), None


def convert_usage(usage: Optional[UsageInfo]) -> AnthropicUsage:
    if usage is None:
        return AnthropicUsage()
    cached = 0
    if usage.prompt_tokens_details is not None:
        cached = usage.prompt_tokens_details.cached_tokens or 0
    if cached > usage.prompt_tokens:
        # Only reachable through an upstream accounting error. Clamping keeps
        # the response well-formed, but silently reporting input_tokens=0 next
        # to a large cache_read_input_tokens would hide the inconsistency from
        # anyone reconciling token counts.
        logger.warning(
            f"cached_tokens ({cached}) exceeds prompt_tokens "
            f"({usage.prompt_tokens}); reporting input_tokens=0"
        )
    input_tokens = max(usage.prompt_tokens - cached, 0)
    anthropic_usage = AnthropicUsage(
        input_tokens=input_tokens,
        output_tokens=usage.completion_tokens or 0,
    )
    if cached > 0:
        anthropic_usage.cache_read_input_tokens = cached
    return anthropic_usage


def convert_chat_response(chat_response: ChatCompletionResponse) -> AnthropicMessagesResponse:
    """Translate a non-streaming chat completion into an Anthropic message."""
    content: List[Any] = []
    stop_reason: AnthropicStopReason = "end_turn"
    stop_sequence: Optional[str] = None

    if chat_response.choices:
        choice = chat_response.choices[0]
        message = choice.message
        reasoning = message.reasoning_content or message.reasoning
        if reasoning:
            content.append(AnthropicThinkingBlock(thinking=reasoning))
        if message.content:
            content.append(AnthropicTextBlock(text=message.content))
        for tool_call in message.tool_calls:
            try:
                tool_input = json.loads(tool_call.function.arguments)
                if not isinstance(tool_input, dict):
                    raise ValueError("arguments is not a JSON object")
            except (json.JSONDecodeError, ValueError) as e:
                raise AnthropicResponseError(
                    f"Tool call {tool_call.function.name!r} arguments are not a valid JSON object"
                ) from e
            content.append(
                AnthropicToolUseBlock(
                    id=tool_call.id, name=tool_call.function.name, input=tool_input
                )
            )
        stop_reason, stop_sequence = _map_stop_result(choice.finish_reason, choice.stop_reason)

    if any(block.type == "tool_use" for block in content):
        # A response carrying tool_use blocks must say stop_reason="tool_use";
        # the client's tool loop keys off exactly that and treats anything else
        # as the end of the turn, so the tools would never run. The upstream
        # finish_reason is only promoted to "tool_calls" when a tool parser
        # ran, which the forced tool_choice path skips, so it cannot be relied
        # on here. "max_tokens" still wins: the content was truncated.
        if stop_reason != "max_tokens":
            stop_reason, stop_sequence = "tool_use", None

    if not content:
        # Anthropic responses must carry at least one content block.
        content.append(AnthropicTextBlock(text=""))

    return AnthropicMessagesResponse(
        model=chat_response.model,
        content=content,
        stop_reason=stop_reason,
        stop_sequence=stop_sequence,
        usage=convert_usage(chat_response.usage),
    )


# ---------------------------------------------------------------------------
# Streaming: OpenAI SSE chunks -> Anthropic SSE events
# ---------------------------------------------------------------------------


@dataclass
class _ToolCallStream:
    """Everything known about one upstream tool call, keyed by its index.

    Upstream tells parallel calls apart with ``tool_call.index`` and may emit
    fragments of several of them in a single chunk (see the loop over parsed
    calls in ``postprocess_handlers``), while the Anthropic protocol only ever
    has one content block open. Each call therefore needs its own record, so
    fragments arriving while another call owns the open block can be buffered
    in ``pending`` instead of dropped.
    """

    key: Any
    id: str
    name: Optional[str] = None
    #: Argument fragments received while this call did not own the open block.
    pending: List[str] = field(default_factory=list)
    #: Whether a content_block_start was already emitted for this call.
    started: bool = False


class AnthropicStreamReframer:
    """Stateful reframer from OpenAI chat chunks to Anthropic SSE events.

    Consumes the ``data: <ChatCompletionStreamResponse json>`` lines produced
    by the ``openai_chat`` streaming path and emits the Anthropic event
    sequence::

        message_start
        (content_block_start (content_block_delta)* content_block_stop)*
        message_delta
        message_stop

    Invariants maintained: ``message_start`` is emitted exactly once and
    first; every content block is opened before any delta and closed before a
    block of another type (or another tool call) is opened; block indices are
    monotonically increasing; the delta type always matches the open block
    type.

    Because blocks are strictly sequential, upstream tool calls that interleave
    are *serialised* here rather than passed through: the call that owns the
    open block keeps streaming its arguments, and every other call accumulates
    in :class:`_ToolCallStream.pending` until its own block is opened. No
    argument fragment is ever discarded, so each call's concatenated
    ``partial_json`` is exactly what upstream sent for it.
    """

    def __init__(self, model: str):
        self.model = model
        self.message_id = f"msg_{uuid.uuid4().hex}"
        self.message_started = False
        self.block_index = -1
        self.open_block_type: Optional[str] = None
        # Upstream index of the tool call whose block is currently open.
        self.open_tool_key: Optional[Any] = None
        # Per-upstream-index tool call state. Insertion order is preserved and
        # is the order the tool_use blocks are emitted in.
        self.tool_calls: Dict[Any, _ToolCallStream] = {}
        # Counter for calls that arrive without an index; see _tool_call_state.
        self.unindexed_calls = 0
        # Whether this stream emitted any tool_use block, which decides the
        # terminating stop_reason; see finish().
        self.emitted_tool_use = False
        self.stop_reason: AnthropicStopReason = "end_turn"
        self.stop_sequence: Optional[str] = None
        self.final_usage: Optional[AnthropicUsage] = None

    # -- block state machine -------------------------------------------------

    def _close_block(self) -> List[str]:
        if self.open_block_type is None:
            return []
        event = AnthropicContentBlockStopEvent(index=self.block_index)
        self.open_block_type = None
        self.open_tool_key = None
        return [anthropic_sse(event)]

    def _open_block(self, block: Any, block_type: str) -> List[str]:
        frames = self._close_block()
        self.block_index += 1
        self.open_block_type = block_type
        if block_type == "tool_use":
            self.emitted_tool_use = True
        frames.append(
            anthropic_sse(
                AnthropicContentBlockStartEvent(index=self.block_index, content_block=block)
            )
        )
        return frames

    def _ensure_block(self, block_type: str) -> List[str]:
        if self.open_block_type == block_type:
            return []
        # A block cannot receive deltas once another block opens, so any tool
        # call still holding buffered arguments has to be emitted before text
        # or thinking takes the open slot.
        frames = self._flush_pending_tool_calls()
        if block_type == "text":
            frames.extend(self._open_block(AnthropicTextBlock(text=""), "text"))
        elif block_type == "thinking":
            frames.extend(self._open_block(AnthropicThinkingBlock(thinking=""), "thinking"))
        else:
            raise ValueError(f"unexpected block type {block_type}")
        return frames

    # -- tool call bookkeeping ------------------------------------------------

    def _tool_call_state(self, tool_call: Any) -> Optional[_ToolCallStream]:
        """Return the record for ``tool_call``, creating it on first sight.

        ``None`` means the fragment cannot be attributed to any call and the
        caller has to skip it.
        """
        function = tool_call.function
        key = tool_call.index
        if key is None:
            # Without an index, a name is the only signal that a new call
            # started; unnamed fragments belong to the call currently being
            # streamed, or failing that to the most recently seen one.
            if function.name:
                self.unindexed_calls += 1
                key = f"unindexed-{self.unindexed_calls}"
            elif self.open_tool_key is not None:
                key = self.open_tool_key
            elif self.tool_calls:
                key = next(reversed(self.tool_calls))
            else:
                logger.warning(
                    "Dropping tool argument fragment that belongs to no known "
                    "tool call (upstream sent neither an index nor a name)"
                )
                return None

        state = self.tool_calls.get(key)
        if state is None:
            state = _ToolCallStream(
                key=key,
                id=tool_call.id or f"toolu_{uuid.uuid4().hex}",
                name=function.name,
            )
            self.tool_calls[key] = state
            return state
        # Some producers repeat the id and name on every fragment of one call -
        # the forced tool_choice path in postprocess_handlers does - so a
        # repeat must not start a second call. Late values only fill in what
        # was missing, and never after the block start announced them.
        if not state.started:
            if function.name and not state.name:
                state.name = function.name
            if tool_call.id:
                state.id = tool_call.id
        return state

    def _tool_argument_deltas(self, fragments: List[str]) -> List[str]:
        return [
            anthropic_sse(
                AnthropicContentBlockDeltaEvent(
                    index=self.block_index,
                    delta=AnthropicInputJsonDelta(partial_json=fragment),
                )
            )
            for fragment in fragments
            if fragment
        ]

    def _open_tool_call_block(self, state: _ToolCallStream) -> List[str]:
        """Give ``state`` the open block and flush everything buffered for it."""
        block = AnthropicToolUseBlock(id=state.id, name=state.name, input={})
        frames = self._open_block(block, "tool_use")
        self.open_tool_key = state.key
        state.started = True
        frames.extend(self._tool_argument_deltas(state.pending))
        state.pending.clear()
        return frames

    def _flush_pending_tool_calls(self) -> List[str]:
        """Emit a complete block for every call still holding buffered arguments."""
        frames: List[str] = []
        for state in list(self.tool_calls.values()):
            if state.key == self.open_tool_key:
                # Fragments for the open call stream immediately, so this is
                # only defensive.
                frames.extend(self._tool_argument_deltas(state.pending))
                state.pending.clear()
                continue
            if state.started and not state.pending:
                continue
            if not state.name:
                # A tool_use block has no way to say which tool to run.
                logger.warning(
                    f"Dropping tool call arguments for index {state.key!r}: "
                    "upstream never sent a tool name"
                )
                state.pending.clear()
                continue
            if state.started:
                # Arguments arrived after this call's block was already closed
                # by non-tool content. No known producer resumes a call that
                # way; emitting the remainder under the same tool_use id at
                # least keeps it recoverable, whereas dropping it hands the
                # client silently truncated arguments.
                logger.warning(
                    f"Tool call {state.name!r} (index {state.key!r}) continued after its "
                    "content block closed; emitting the remaining arguments in a second block"
                )
            frames.extend(self._open_tool_call_block(state))
        return frames

    # -- chunk handling -------------------------------------------------------

    def adopt_model(self, model: Optional[str]) -> None:
        """Take the model name from the upstream response.

        The non-streaming path reports ChatCompletionResponse.model, so reading
        it from the chunk here keeps both paths reporting the same name for the
        same request. The constructor value is the fallback for a stream that
        fails before its first chunk.
        """
        if model and not self.message_started:
            self.model = model

    def _start_message(self, usage: Optional[AnthropicUsage]) -> List[str]:
        if self.message_started:
            return []
        self.message_started = True
        skeleton = AnthropicMessagesResponse(
            id=self.message_id,
            model=self.model,
            content=[],
            usage=usage or AnthropicUsage(),
        )
        # ``stop_reason``/``stop_sequence`` intentionally stay None here and
        # are delivered by the final message_delta.
        return [anthropic_sse(AnthropicMessageStartEvent(message=skeleton))]

    def process_chunk(self, chunk: ChatCompletionStreamResponse) -> List[str]:
        frames: List[str] = []

        self.adopt_model(getattr(chunk, "model", None))

        usage = None
        if chunk.usage is not None:
            usage = convert_usage(chunk.usage)
            self.final_usage = usage
        if not self.message_started:
            start_usage = None
            if usage is not None:
                start_usage = AnthropicUsage(
                    input_tokens=usage.input_tokens,
                    output_tokens=0,
                    cache_read_input_tokens=usage.cache_read_input_tokens,
                )
            frames.extend(self._start_message(start_usage))

        for choice in chunk.choices:
            delta = choice.delta
            reasoning_delta = delta.reasoning_content or delta.reasoning
            if reasoning_delta:
                frames.extend(self._ensure_block("thinking"))
                frames.append(
                    anthropic_sse(
                        AnthropicContentBlockDeltaEvent(
                            index=self.block_index,
                            delta=AnthropicThinkingDelta(thinking=reasoning_delta),
                        )
                    )
                )
            if delta.content:
                frames.extend(self._ensure_block("text"))
                frames.append(
                    anthropic_sse(
                        AnthropicContentBlockDeltaEvent(
                            index=self.block_index, delta=AnthropicTextDelta(text=delta.content)
                        )
                    )
                )
            for tool_call in delta.tool_calls or []:
                function = tool_call.function
                if function is None:
                    continue
                state = self._tool_call_state(tool_call)
                if state is None:
                    continue
                if not state.started and state.name and self.open_block_type != "tool_use":
                    # Nothing else is streaming, so this call can take the open
                    # block right away and have its arguments forwarded
                    # incrementally - the single-call case, and the first of a
                    # parallel batch.
                    frames.extend(self._open_tool_call_block(state))
                if function.arguments:
                    if state.key == self.open_tool_key:
                        frames.extend(self._tool_argument_deltas([function.arguments]))
                    else:
                        # Another call owns the open block. Buffer rather than
                        # drop: this call gets its own block once the open one
                        # closes, and the fragments are replayed there in
                        # order, so its partial_json still concatenates to
                        # exactly what upstream sent.
                        state.pending.append(function.arguments)
            if choice.finish_reason:
                self.stop_reason, self.stop_sequence = _map_stop_result(
                    choice.finish_reason, choice.stop_reason
                )

        return frames

    def finish(self) -> List[str]:
        frames = self._start_message(None)  # degenerate empty stream
        # Tool calls that never got the open block are emitted here as complete
        # blocks. This also has to run before the stop_reason is computed, so
        # that a stream whose only tool call was buffered still reports
        # stop_reason="tool_use".
        frames.extend(self._flush_pending_tool_calls())
        frames.extend(self._close_block())
        stop_reason = self.stop_reason
        stop_sequence = self.stop_sequence
        if self.emitted_tool_use and stop_reason != "max_tokens":
            # Same contract as the non-streaming path: a turn that emitted
            # tool_use blocks ends with stop_reason="tool_use", or the client
            # never runs the tools it was just handed.
            stop_reason, stop_sequence = "tool_use", None
        frames.append(
            anthropic_sse(
                AnthropicMessageDeltaEvent(
                    delta=AnthropicMessageDelta(
                        stop_reason=stop_reason,
                        stop_sequence=stop_sequence,
                    ),
                    usage=self.final_usage or AnthropicUsage(),
                )
            )
        )
        frames.append(anthropic_sse(AnthropicMessageStopEvent()))
        return frames

    def error(self, message: str) -> List[str]:
        """Close any open block, then surface an error event."""
        frames = self._close_block()
        frames.append(
            anthropic_sse(
                AnthropicErrorEvent(error=AnthropicError(type="api_error", message=message))
            )
        )
        return frames


async def _iter_openai_sse_lines(
    openai_sse: AsyncIterator[Union[str, bytes]],
) -> AsyncIterator[str]:
    """Yield complete lines from string frames or arbitrarily chunked bytes.

    The in-process server produces complete SSE strings, while the
    disaggregated frontend relays raw ``aiohttp`` byte chunks.  Network chunks
    need not align with either UTF-8 characters or SSE line boundaries, so
    retain incomplete input until the next chunk arrives.
    """
    import codecs

    decoder = codecs.getincrementaldecoder("utf-8")()
    buffer = ""
    input_type: Optional[type] = None

    async for payload in openai_sse:
        if not isinstance(payload, (str, bytes)):
            raise TypeError(f"OpenAI SSE payload must be str or bytes, got {type(payload)!r}")

        current_type = type(payload)
        if input_type is None:
            input_type = current_type
        elif current_type is not input_type:
            raise TypeError("OpenAI SSE stream cannot mix str and bytes payloads")

        if isinstance(payload, bytes):
            buffer += decoder.decode(payload)
        else:
            buffer += payload

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            yield line.rstrip("\r")

    if input_type is bytes:
        buffer += decoder.decode(b"", final=True)
    if buffer:
        yield buffer.rstrip("\r")


async def reframe_openai_stream(
    openai_sse: AsyncIterator[Union[str, bytes]], model: str
) -> AsyncIterator[str]:
    """Translate an OpenAI SSE stream into Anthropic SSE frames."""
    reframer = AnthropicStreamReframer(model=model)
    try:
        async for line in _iter_openai_sse_lines(openai_sse):
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if not data:
                continue
            if data == "[DONE]":
                for frame in reframer.finish():
                    yield frame
                return
            try:
                chunk = ChatCompletionStreamResponse(**json.loads(data))
            except (json.JSONDecodeError, ValueError) as e:
                raise AnthropicResponseError("Malformed upstream stream chunk") from e
            for frame in reframer.process_chunk(chunk):
                yield frame
        # Upstream ended without [DONE]. Both streaming producers emit it only
        # after the response is complete, so its absence means the stream was
        # cut short. Calling finish() here would close the message with
        # message_stop / end_turn and the client would accept truncated output
        # as a finished answer -- a silent wrong result rather than a failure.
        raise AnthropicResponseError(
            "Upstream stream ended before [DONE]; the response is incomplete"
        )
    except AnthropicResponseError as e:
        # Raised only on upstream framing this module can describe precisely,
        # so the text originates here and carries no internal detail: pass it
        # through rather than flattening it to a generic message.
        logger.error(f"Anthropic stream failed: {e}")
        for frame in reframer.error(str(e)):
            yield frame
    except Exception as e:  # noqa: BLE001 - stream must end with an event
        logger.error(f"Anthropic stream reframing failed: {e}\n{traceback.format_exc()}")
        for frame in reframer.error("Internal server error"):
            yield frame
