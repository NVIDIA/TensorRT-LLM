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
"""Serving extension for Kimi K3 checkpoints (``model_type == "kimi_k3"``).

Implements the Kimi/Moonshot API semantics layered on the OpenAI chat surface:
message-level (dynamic) tool validation, the optional pinned sampling-parameter
policy, default streaming usage, request-field to chat-template-kwarg
derivation, and the K3 XTML placement of guided-decoding constraints. On the
Responses surface it derives the same chat-template kwargs from the Responses
request fields and adapts a replayed conversation to the way the K3 renderer
binds tool results to their calls.
"""

import os
import re
from typing import Any, Dict, List, Optional, Union

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionMessageParam,
    ChatCompletionRequest,
    ResponsesRequest,
    StreamOptions,
)
from tensorrt_llm.serve.serving_extensions import ServingExtension, register_serving_extension

# Valid function-tool name for message-level tools: no leading digit, word
# chars/dash only, at most 256 chars.
_DYNAMIC_TOOL_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]{0,255}\Z")

_RESPONSE_CHANNEL_OPEN = "<|open|>response<|sep|>"
_RESPONSE_CHANNEL_CLOSE = "<|close|>response<|sep|>"


def enforce_kimi_param_policy(request: Union[ChatCompletionRequest, ResponsesRequest]) -> None:
    """Enforce Kimi's pinned sampling-parameter policy when opted in.

    Kimi's API pins top_p, the penalties, and n, and bounds temperature to
    [0, 1]; out-of-policy values fail fast with HTTP 400 rather than generate.
    top_p unset or the OpenAI-default 1.0 is coerced to the pinned 0.95 instead
    of rejected. Each rule applies to the sampling fields the request type
    declares: a chat request has them all, a Responses request only temperature
    and top_p. Off unless ``TRTLLM_KIMI_PARAM_POLICY=1``, so deployments keep
    accepting the requests they accept without the policy.
    """
    if os.getenv("TRTLLM_KIMI_PARAM_POLICY", "0") != "1":
        return
    declared = type(request).model_fields
    if request.top_p is None or request.top_p == 1.0:
        # None would fall back to 1.0 in to_sampling_params; an explicit 1.0
        # is the OpenAI SDK default many clients send unconditionally. Coerce
        # both to the pinned value rather than rejecting.
        request.top_p = 0.95
    if request.temperature is not None and not (0.0 <= request.temperature <= 1.0):
        raise ValueError(
            f"temperature must be within [0, 1] for this model; got {request.temperature}."
        )
    if request.top_p is not None and request.top_p != 0.95:
        raise ValueError(f"top_p is fixed at 0.95 for this model; got {request.top_p}.")
    if "presence_penalty" in declared and request.presence_penalty:
        raise ValueError(
            f"presence_penalty is fixed at 0 for this model; got {request.presence_penalty}."
        )
    if "frequency_penalty" in declared and request.frequency_penalty:
        raise ValueError(
            f"frequency_penalty is fixed at 0 for this model; got {request.frequency_penalty}."
        )
    if "n" in declared and request.n != 1:
        raise ValueError(f"n is fixed at 1 for this model; got {request.n}.")


# OpenAI's Responses `reasoning.effort` levels on the three the kimi_k3 template
# accepts for `thinking_effort` (low/high/max). Each level maps to the nearest K3
# level at or above it, so a request never gets less thinking than it asked for.
# "none" is not a level: it turns thinking off (see apply_responses_extensions).
# No effort at all leaves the template default (max).
_THINKING_EFFORT_BY_RESPONSES_EFFORT = {
    "minimal": "low",
    "low": "low",
    "medium": "high",
    "high": "high",
    "xhigh": "max",
    "max": "max",
}


def _joined_tool_content(first: Any, second: Any) -> Any:
    """Two segments of one call's result, in order, as one content value."""
    if isinstance(first, str) and isinstance(second, str):
        if not first or first.endswith("\n"):
            return first + second
        return first + "\n" + second

    def parts(content: Any) -> list:
        if isinstance(content, list):
            return list(content)
        return [{"type": "text", "text": content or ""}]

    return parts(first) + parts(second)


def _merge_split_tool_results(
    messages: List[ChatCompletionMessageParam],
) -> List[ChatCompletionMessageParam]:
    """Give each call at most one result message within a run of results.

    A client can return one call's result in more than one item, all with the
    same ``call_id``. The K3 renderer sorts a run of results by the position of
    the call each one names but numbers them by counting result messages, so a
    second segment of one call takes the next call's index and every later
    result in the turn names the wrong call. Within each run of consecutive
    tool messages, the segments of one call are joined into its first message
    in the order they came; calls keep the order of their first segment.
    Nothing is dropped and nothing moves out of its run, so a result that
    arrived in a later turn stays there.
    """
    merged: List[ChatCompletionMessageParam] = []
    first_index_by_call: Dict[str, int] = {}
    joined = 0
    for message in messages:
        if message.get("role") != "tool":
            first_index_by_call = {}
            merged.append(message)
            continue
        call_id = message.get("tool_call_id")
        key = None if call_id is None else str(call_id)
        if key is not None and key in first_index_by_call:
            first = merged[first_index_by_call[key]]
            merged[first_index_by_call[key]] = {
                **first,
                "content": _joined_tool_content(first.get("content"), message.get("content")),
            }
            joined += 1
            continue
        if key is not None:
            first_index_by_call[key] = len(merged)
        merged.append(message)
    if joined:
        logger.warning_once(
            "Responses API: a tool call came back with its result split across several "
            "items; kimi_k3 numbers results by count, so the segments are joined into one "
            "result per call.",
            key="kimi_k3_responses_split_tool_results",
        )
    return merged


def _orphan_tool_result_note(message: ChatCompletionMessageParam) -> ChatCompletionMessageParam:
    """A tool result no call claims, as a plain user note carrying its output."""
    header = (
        f"[Output of tool call {message.get('tool_call_id')!r}, which matches no tool call "
        "before it:]"
    )
    content = message.get("content")
    parts = (
        list(content) if isinstance(content, list) else [{"type": "text", "text": content or ""}]
    )
    return {"role": "user", "content": [{"type": "text", "text": header}, *parts]}


def _settle_orphan_tool_results(
    messages: List[ChatCompletionMessageParam],
) -> List[ChatCompletionMessageParam]:
    """Keep the K3 renderer from rejecting a tool result it cannot bind to a call.

    The renderer binds each run of tool results to the most recent assistant
    message's ``tool_calls`` by id, falls back to position when any result in
    the run is unmatched, and raises when a result then has no call at all. A
    client that replays its whole history every turn would fail every later
    turn of the session on one result without a call (its call compacted away,
    or a late result for a turn the client abandoned), and the positional
    fallback pins the wrong call's name on a result when it does not raise.

    A result whose ``tool_call_id`` matches no call of the assistant message in
    front of it, the renderer's own test, is therefore replayed as a user note
    carrying its output, placed after its run of results so the rest of the run
    still binds by id. Nothing the client sent is dropped, and the conversion is
    deterministic, so the replayed history stays a stable prompt prefix.
    """
    settled: List[ChatCompletionMessageParam] = []
    notes: List[ChatCompletionMessageParam] = []
    call_ids: set = set()
    orphans = 0
    for message in messages:
        if message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if call_id is not None and str(call_id) in call_ids:
                settled.append(message)
            else:
                notes.append(_orphan_tool_result_note(message))
                orphans += 1
            continue
        settled.extend(notes)
        notes = []
        if message.get("role") == "assistant":
            call_ids = {
                str(call["id"])
                for call in message.get("tool_calls") or []
                if isinstance(call, dict) and call.get("id") is not None
            }
        settled.append(message)
    settled.extend(notes)
    if orphans:
        logger.warning_once(
            "Responses API: a tool result matched no tool call in the conversation; "
            "kimi_k3 cannot render that, so it is replayed as a user note carrying the output.",
            key="kimi_k3_responses_orphan_tool_results",
        )
    return settled


def _derive_from_reasoning_effort(kwargs: dict[str, Any], effort: object) -> None:
    """Derive the K3 ``thinking`` / ``thinking_effort`` kwargs from a reasoning effort.

    A no-op when ``kwargs`` turns thinking off. Otherwise ``"none"`` turns
    thinking off unless ``kwargs`` sets it, ``"low"``, ``"high"`` and ``"max"``
    set the effort unless ``kwargs`` sets one, and other values (e.g. harmony's
    ``"medium"``) have no K3 equivalent and leave the template default.
    """
    if not isinstance(effort, str) or kwargs.get("thinking") is False:
        return
    effort = effort.lower()
    if effort == "none":
        kwargs.setdefault("thinking", False)
    elif effort in ("low", "high", "max"):
        kwargs.setdefault("thinking_effort", effort)


def dynamic_tool_dicts(messages: Optional[List[ChatCompletionMessageParam]]) -> list[dict]:
    """Collect message-level (dynamic) tool declarations from system messages."""
    tools: list[dict] = []
    for msg in messages or []:
        if isinstance(msg, dict) and msg.get("role") == "system" and msg.get("tools"):
            tools.extend(msg["tools"])
    return tools


def validate_kimi_dynamic_tools(request: ChatCompletionRequest) -> None:
    """Validate message-level (dynamic) tool declarations.

    Kimi-style dynamic tools ride on system messages: system-only carrier,
    empty content, function-typed tools with valid unique names (unique also
    against request-level tools).
    """
    seen_names = set()
    for tool in request.tools or []:
        seen_names.add(tool.function.name)
    for message in request.messages or []:
        # A null tools key is treated as absent (some SDKs serialize
        # optional fields as null); only declared tools are validated.
        if not isinstance(message, dict) or message.get("tools") is None:
            continue
        if message.get("role") != "system":
            raise ValueError("Message-level `tools` are only allowed on system messages.")
        if message.get("content"):
            raise ValueError("A system message carrying `tools` must have empty content.")
        message_tools = message["tools"]
        if not isinstance(message_tools, list):
            raise ValueError("Message-level `tools` must be an array.")
        for tool in message_tools:
            if not isinstance(tool, dict):
                raise ValueError("Each message-level tool must be an object.")
            if tool.get("type") != "function":
                raise ValueError(f"Unsupported message-level tool type: {tool.get('type')!r}.")
            function = tool.get("function")
            if not isinstance(function, dict):
                raise ValueError("Message-level tools must carry a `function` object.")
            name = function.get("name")
            if not isinstance(name, str) or not _DYNAMIC_TOOL_NAME_RE.match(name):
                raise ValueError(f"Invalid message-level tool name: {name!r}.")
            if name in seen_names:
                raise ValueError(f"Duplicate tool name: {name!r}.")
            seen_names.add(name)


@register_serving_extension(model_types=("kimi_k3",), reasoning_parsers=("kimi_k3",))
class KimiK3ServingExtension(ServingExtension):
    """Kimi/Moonshot API semantics for kimi_k3 chat and Responses requests."""

    render_version = 2

    # Trailing tokens of the native K3 generation prompt (the channel opener
    # ``<|open|>think|response<|sep|>``) that are not reported as prompt usage.
    _GENERATION_CHANNEL_OPENER_TOKENS = 3

    def allows_required_tool_choice(self) -> bool:
        """K3 enforces ``tool_choice="required"`` through its template."""
        return True

    def serialize_tool(self, tool) -> Dict[str, Any]:
        """Drop pydantic-injected null defaults (strict, description, parameters).

        They would otherwise leak into the rendered tool-declaration JSON and
        skew prompt-token parity with the reference renderer.
        """
        return tool.model_dump(exclude_none=True)

    def dynamic_tools(self, messages: Optional[List[ChatCompletionMessageParam]]) -> List[dict]:
        """Message-level (dynamic) tools are a Kimi API extension."""
        return dynamic_tool_dicts(messages)

    def prompt_tokens_excluded_from_usage(
        self,
        *,
        add_generation_prompt: bool,
        pre_tokenized: bool,
        request_chat_template: Optional[str],
    ) -> int:
        """Exclude the generation channel opener from prompt usage.

        Kimi's prompt-token accounting excludes the trailing generation channel
        opener; the model still sees the full rendered prompt. A pre-tokenized
        prompt is not rendered here, so there is no opener to exclude. A
        request-level template may end differently, so report unadjusted usage
        for it.
        """
        if add_generation_prompt and not pre_tokenized and request_chat_template is None:
            return self._GENERATION_CHANNEL_OPENER_TOKENS
        return 0

    def apply_chat_extensions(self, request: ChatCompletionRequest) -> None:
        """Derive chat-template kwargs and defaults from request-level fields.

        The kimi_k3 template natively renders control messages for thinking
        effort, tool_choice, and response_format, but only reads them from
        chat-template kwargs. Derive those kwargs from the request-level
        fields so the OpenAI-style API surface drives the template; explicit
        client-supplied ``chat_template_kwargs`` win over derived values.
        The template reads ``thinking`` / ``thinking_effort``, not the
        ``enable_thinking`` / ``reasoning_effort`` kwargs most other chat
        templates read, so those generic kwargs are translated as well. The
        merged kwargs also steer the kimi_k3 reasoning parser's initial
        channel, the guided-decoding structural tag, and the thinking-budget
        logits processor downstream.

        Kimi's API reports usage in the final streaming chunk without the
        client opting in, so streaming requests get default ``stream_options``.
        """
        validate_kimi_dynamic_tools(request)
        enforce_kimi_param_policy(request)
        if request.stream and request.stream_options is None:
            # StreamOptions defaults: include_usage=True, continuous off.
            request.stream_options = StreamOptions()
        # Sources of the ``thinking`` / ``thinking_effort`` kwargs, highest
        # precedence first; each one only sets what no source above it set:
        #   1. those two keys in the client's ``chat_template_kwargs``;
        #   2. the request-level ``thinking`` object;
        #   3. the request-level ``reasoning_effort``;
        #   4. the generic ``enable_thinking`` / ``reasoning_effort`` keys in
        #      ``chat_template_kwargs`` (the Anthropic adapter maps ``thinking``
        #      and ``output_config.effort`` onto them);
        #   5. the template default: thinking on, effort ``max``.
        # An effort of ``"none"`` turns thinking off. No effort is derived for a
        # request whose thinking is off; a ``thinking_effort`` the client sets is
        # passed through, and the template ignores it while thinking is off.
        derived: dict[str, Any] = {}
        if request.thinking is not None:
            enabled = request.thinking.type != "disabled"
            derived["thinking"] = enabled
            if enabled and request.thinking.effort is not None:
                derived["thinking_effort"] = request.thinking.effort
        if "reasoning_effort" in request.model_fields_set and request.reasoning_effort is not None:
            _derive_from_reasoning_effort(
                derived, getattr(request.reasoning_effort, "value", request.reasoning_effort)
            )
        if (
            "tool_choice" in request.model_fields_set
            and (request.tools or dynamic_tool_dicts(request.messages))
            and request.tool_choice in ("required", "none")
        ):
            derived["tool_choice"] = request.tool_choice
        response_format = request.response_format
        if response_format is not None and response_format.type in ("json_object", "json_schema"):
            derived["response_format"] = response_format.type
            if response_format.type == "json_schema":
                # Kimi requires the OpenAI wrapper shape: {name, schema[, strict]}.
                json_schema = response_format.json_schema
                if (
                    not isinstance(json_schema, dict)
                    or not isinstance(json_schema.get("name"), str)
                    or not json_schema["name"]
                ):
                    raise ValueError(
                        "response_format.json_schema requires a non-empty `name` string."
                    )
                if not isinstance(json_schema.get("schema"), dict):
                    raise ValueError("response_format.json_schema requires a `schema` object.")
                if "strict" in json_schema and not isinstance(json_schema["strict"], bool):
                    raise ValueError("response_format.json_schema.strict must be a boolean.")
                derived["response_schema"] = json_schema["schema"]
        chat_template_kwargs = {**derived, **(request.chat_template_kwargs or {})}
        unset = {"thinking", "thinking_effort"} - chat_template_kwargs.keys()
        enable_thinking = chat_template_kwargs.get("enable_thinking")
        if isinstance(enable_thinking, bool):
            chat_template_kwargs.setdefault("thinking", enable_thinking)
        _derive_from_reasoning_effort(
            chat_template_kwargs, chat_template_kwargs.get("reasoning_effort")
        )
        translated = unset & chat_template_kwargs.keys()
        exempted = set(translated)
        generic_effort = chat_template_kwargs.get("reasoning_effort")
        if isinstance(generic_effort, str) and generic_effort.lower() in (
            "none",
            "low",
            "high",
            "max",
        ):
            # The extension consumes this control even if a higher-precedence
            # K3 setting wins. Keep it available to custom templates as well.
            exempted.add("reasoning_effort")
        if exempted:
            # Derived and consumed controls need not be read again by a Jinja template.
            request.injected_chat_template_kwargs = sorted(
                {*(request.injected_chat_template_kwargs or ()), *exempted}
            )
        if derived or translated:
            request.chat_template_kwargs = chat_template_kwargs

    def apply_responses_extensions(self, request: ResponsesRequest) -> None:
        """Derive chat-template kwargs from a Responses request and apply the sampling policy.

        The kimi_k3 template reads thinking effort, tool_choice and
        response_format only from chat-template kwargs, so they are derived
        from the request's own fields: ``reasoning.effort`` becomes
        ``thinking_effort`` on K3's low/high/max scale (``"none"`` turns
        thinking off; no effort leaves the template default), ``tool_choice``
        ``"required"`` or ``"none"`` with tools declared becomes
        ``tool_choice``, and ``text.format`` of type json_object or json_schema
        becomes ``response_format`` (with ``response_schema`` for a schema).
        Explicit client-supplied ``chat_template_kwargs``, an accepted extra
        field on this endpoint, win over derived values, as on chat. The same
        kwargs steer the kimi_k3 reasoning parser's initial channel, so the
        parser starts in the channel the prompt opened. The sampling-parameter
        policy applies to the fields this endpoint has (temperature, top_p).
        """
        enforce_kimi_param_policy(request)
        derived: dict[str, Any] = {}
        effort = request.reasoning.effort if request.reasoning is not None else None
        if effort == "none":
            derived["thinking"] = False
        elif effort in _THINKING_EFFORT_BY_RESPONSES_EFFORT:
            derived["thinking_effort"] = _THINKING_EFFORT_BY_RESPONSES_EFFORT[effort]
        if request.tools and request.tool_choice in ("required", "none"):
            derived["tool_choice"] = request.tool_choice
        text_format = request.text.format if request.text is not None else None
        if text_format is not None and text_format.type in ("json_object", "json_schema"):
            derived["response_format"] = text_format.type
            if text_format.type == "json_schema":
                derived["response_schema"] = text_format.schema_
        if derived:
            explicit = getattr(request, "chat_template_kwargs", None)
            request.chat_template_kwargs = {
                **derived,
                **(explicit if isinstance(explicit, dict) else {}),
            }

    def prepare_responses_history(
        self, messages: List[ChatCompletionMessageParam]
    ) -> List[ChatCompletionMessageParam]:
        """Adapt a replayed conversation to how the K3 renderer binds tool results.

        The renderer numbers a run of tool results by counting result messages
        and rejects a result it cannot bind to a call, so each call gets at
        most one result message per run (``_merge_split_tool_results``) and a
        result no call claims becomes a user note carrying its output
        (``_settle_orphan_tool_results``).
        """
        return _settle_orphan_tool_results(_merge_split_tool_results(messages))

    def structured_output_format(
        self, content: Dict[str, Any], chat_template_kwargs: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Place the constraint on the K3 XTML response channel.

        The generation prompt already ends inside the channel the model
        starts in. In thinking mode (the default) the response channel opens
        mid-generation, so trigger the user constraint on it. In non-thinking
        mode the prompt ends inside ``<|open|>response<|sep|>``, the trigger
        would never be generated, and the raw grammar applies from the first
        generated token instead.
        """
        thinking = (chat_template_kwargs or {}).get("thinking", True) is not False
        if not thinking:
            return None
        return {
            "type": "triggered_tags",
            "triggers": [_RESPONSE_CHANNEL_OPEN],
            "tags": [
                {
                    "begin": _RESPONSE_CHANNEL_OPEN,
                    "content": content,
                    "end": _RESPONSE_CHANNEL_CLOSE,
                },
            ],
            "stop_after_first": True,
        }
