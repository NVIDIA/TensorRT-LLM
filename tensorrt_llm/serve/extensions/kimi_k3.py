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
derivation, and the K3 XTML placement of guided-decoding constraints.
"""

import os
import re
from typing import Any, Dict, List, Optional

from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionMessageParam,
    ChatCompletionRequest,
    StreamOptions,
)
from tensorrt_llm.serve.serving_extensions import ServingExtension, register_serving_extension

# Valid function-tool name for message-level tools: no leading digit, word
# chars/dash only, at most 256 chars.
_DYNAMIC_TOOL_NAME_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]{0,255}\Z")

_RESPONSE_CHANNEL_OPEN = "<|open|>response<|sep|>"
_RESPONSE_CHANNEL_CLOSE = "<|close|>response<|sep|>"


def enforce_kimi_param_policy(request: ChatCompletionRequest) -> None:
    """Enforce Kimi's pinned sampling-parameter policy when opted in.

    Kimi's API pins top_p, the penalties, and n, and bounds temperature to
    [0, 1]; out-of-policy values fail fast with HTTP 400 rather than generate.
    top_p unset or the OpenAI-default 1.0 is coerced to the pinned 0.95 instead
    of rejected. Off unless ``TRTLLM_KIMI_PARAM_POLICY=1``, so deployments keep
    accepting the requests they accept without the policy.
    """
    if os.getenv("TRTLLM_KIMI_PARAM_POLICY", "0") != "1":
        return
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
    if request.presence_penalty:
        raise ValueError(
            f"presence_penalty is fixed at 0 for this model; got {request.presence_penalty}."
        )
    if request.frequency_penalty:
        raise ValueError(
            f"frequency_penalty is fixed at 0 for this model; got {request.frequency_penalty}."
        )
    if request.n != 1:
        raise ValueError(f"n is fixed at 1 for this model; got {request.n}.")


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
    """Kimi/Moonshot API semantics for kimi_k3 chat requests."""

    def apply_chat_extensions(self, request: ChatCompletionRequest) -> None:
        """Derive chat-template kwargs and defaults from request-level fields.

        The kimi_k3 template natively renders control messages for thinking
        effort, tool_choice, and response_format, but only reads them from
        chat-template kwargs. Derive those kwargs from the request-level
        fields so the OpenAI-style API surface drives the template; explicit
        client-supplied ``chat_template_kwargs`` win over derived values. The
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
        derived: dict[str, Any] = {}
        if request.thinking is not None:
            enabled = request.thinking.type != "disabled"
            derived["thinking"] = enabled
            if enabled and request.thinking.effort is not None:
                derived["thinking_effort"] = request.thinking.effort
        if (
            "reasoning_effort" in request.model_fields_set
            and request.reasoning_effort is not None
            and "thinking_effort" not in derived
            and (request.thinking is None or request.thinking.type != "disabled")
        ):
            # An explicit thinking.effort wins, and an explicit thinking
            # object also wins the on/off axis: reasoning_effort only supplies
            # the effort when thinking.effort is absent, and
            # reasoning_effort="none" only disables thinking when no thinking
            # object was sent. No effort is ever derived for an explicitly
            # disabled request.
            effort = getattr(request.reasoning_effort, "value", request.reasoning_effort).lower()
            if effort == "none":
                if request.thinking is None:
                    derived["thinking"] = False
            elif effort in ("low", "high", "max"):
                derived["thinking_effort"] = effort
            # Other efforts (e.g. harmony's "medium") have no K3 equivalent;
            # leave the template default.
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
        if derived:
            request.chat_template_kwargs = {
                **derived,
                **(request.chat_template_kwargs or {}),
            }

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
