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
# ruff: noqa: E501
"""DeepSeek-V4.1 prompt rendering, ported from the checkpoint's text encoder.

Ported from the checkpoint's own ``encoding/encoding.py``, text path only.

This is a separate module from ``deepseek_v4`` rather than a subclass of it
because V4.1 changed the prompt contract in ways that touch nearly every helper.
Reusing the V4 renderer produces prompts that differ from what the checkpoint was
trained on, silently -- nothing raises, and the output stays fluent. The
differences, each verified against the checkpoint by
``tests/unittest/llmapi/test_deepseek_v41_tokenizer.py``:

1. **Reasoning effort is a numeric budget, and it is emitted by default.** V4
   emitted a prose paragraph only when ``reasoning_effort == "max"``. V4.1 emits
   ``Reasoning Effort: {budget} (range 1-100, ...)`` at index 0 of *every*
   thinking-mode prompt, defaulting to ``"high"`` -> 75. This is the largest of
   the deltas: it changes every thinking-mode prompt, which is the mode used for
   reasoning benchmarks.
2. **``<|System|>`` leads the conversation.** V4.1 emits it at index 0 whenever
   there is a reasoning-effort prefix or the first message is a system message,
   and again before a mid-conversation system message. V4 emitted it never.
3. **Mid-conversation system messages are generation boundaries.** They count as
   the "last user message" for the assistant header and for thinking retention.
   V4 recognised only ``user``/``developer``.
4. **The DSML tag names carry leading spaces** -- ``" calls"``, ``" invoke"``,
   ``" parameter"`` -- where V4 used ``tool_calls``/``invoke``/``parameter``.
   These render into different tokens, so a V4-rendered tool call is not the
   sequence V4.1 was trained to emit or to read back.
5. **Tool names are namespace-qualified** (``namespace::name``), and a namespace
   description is prepended to the tool description.
6. **There is no ``developer`` role**, and assistant reasoning is read only from
   ``reasoning_content`` (V4 also accepted ``reasoning``).

Renders text and image placeholders; the multimodal input pipeline decodes images
and inserts their placeholders.
"""

import copy
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from ..tokenizer import TransformersTokenizer

BOS_TOKEN = "<｜begin▁of▁sentence｜>"  # nosec B105
EOS_TOKEN = "<｜end▁of▁sentence｜>"  # nosec B105
SYSTEM_TOKEN = "<｜System｜>"  # nosec B105
USER_TOKEN = "<｜User｜>"  # nosec B105
ASSISTANT_TOKEN = "<｜Assistant｜>"  # nosec B105
LATEST_REMINDER_TOKEN = "<｜latest_reminder｜>"  # nosec B105
THINKING_START_TOKEN = "<think>"  # nosec B105
THINKING_END_TOKEN = "</think>"  # nosec B105
DSML_TOKEN = "｜DSML｜"  # nosec B105

# The leading spaces are part of the tag names, not formatting. See delta 4.
TOOL_CALLS_BLOCK_NAME = " calls"
TOOL_CALL_TAG_NAME = " invoke"
TOOL_PARAMETER_TAG_NAME = " parameter"

TASK_TOKENS = {
    "action": "<｜action｜>",
    "query": "<｜query｜>",
    "authority": "<｜authority｜>",
    "domain": "<｜domain｜>",
    "title": "<｜title｜>",
    "read_url": "<｜read_url｜>",
}
VALID_TASKS = frozenset(TASK_TOKENS)

REASONING_EFFORT_TEMPLATE = (
    "Reasoning Effort: {budget} "
    "(range 1-100, the higher the value, the more thorough the reasoning)\n\n"
)
REASONING_EFFORT_MAPPINGS = {"low": 50, "high": 75, "max": 100}
DEFAULT_REASONING_EFFORT = "high"

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}{tc_block_name}>" block like the following:

<{dsml_token}{tc_block_name}>
<{dsml_token}{tool_call_tag_name} name="$TOOL_NAME">
<{dsml_token}{tool_parameter_tag_name} name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}{tool_parameter_tag_name}>
...
</{dsml_token}{tool_call_tag_name}>
<{dsml_token}{tool_call_tag_name} name="$TOOL_NAME2">
...
</{dsml_token}{tool_call_tag_name}>
</{dsml_token}{tc_block_name}>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {thinking_start_token}), you MUST output your complete reasoning inside {thinking_start_token}...{thinking_end_token} BEFORE any tool calls or final response.

Otherwise, output directly after {thinking_end_token} with tool calls or final response.

### Available Tool Schemas

{tool_schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"""

RESPONSE_FORMAT_TEMPLATE = (
    "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}"
)
TOOL_CALL_TEMPLATE = (
    '<{dsml_token}{tool_call_tag_name} name="{name}">\n{arguments}\n'
    "</{dsml_token}{tool_call_tag_name}>"
)
TOOL_CALLS_TEMPLATE = "<{dsml_token}{tc_block_name}>\n{tool_calls}\n</{dsml_token}{tc_block_name}>"
TOOL_OUTPUT_TEMPLATE = "<tool_result>{content}</tool_result>"

_IMAGE_BLOCK_TYPES = ("image", "image_url", "input_image")


def _to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return json.dumps(value, ensure_ascii=True)


def _reject_image_content(content: Any) -> None:
    """Require image preprocessing instead of silently dropping raw image blocks."""
    if not isinstance(content, list):
        return
    for block in content:
        if isinstance(block, dict) and block.get("type") in _IMAGE_BLOCK_TYPES:
            raise NotImplementedError(
                "Raw image input is not supported directly by the V4.1 chat renderer; "
                f"got a {block.get('type')!r} block. Use the multimodal input pipeline "
                "to attach images and insert their placeholders."
            )


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        _reject_image_content(content)
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n\n".join(parts)
    return str(content)


def _split_tool_name(name: str, namespace: str | None = None) -> tuple[str | None, str]:
    prefix, separator, bare_name = name.partition("::")
    if separator:
        if namespace not in (None, prefix):
            raise ValueError(f"Conflicting tool namespaces: {namespace} != {prefix}")
        namespace, name = prefix, bare_name
    if "::" in name:
        raise ValueError(f"Tool name must not contain '::': {name}")
    if namespace is not None and "::" in namespace:
        raise ValueError(f"Tool namespace must not contain '::': {namespace}")
    return namespace, name


def _tool_name_for_encoding(tool: dict[str, Any]) -> str:
    namespace = tool.get("namespace")
    if isinstance(namespace, dict):
        namespace = namespace["name"]
    namespace, name = _split_tool_name(tool["name"], namespace)
    return name if namespace is None else f"{namespace}::{name}"


def _tools_from_openai_format(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract function definitions with namespace-qualified names."""
    functions = []
    for tool in tools:
        function = dict(tool["function"])
        if tool.get("namespace") is not None:
            function["namespace"] = tool["namespace"]
        function["name"] = _tool_name_for_encoding(function)
        namespace = function.pop("namespace", None)
        if isinstance(namespace, dict) and namespace.get("description"):
            function["description"] = (
                namespace["description"] + "\n" + (function.get("description") or "")
            )
        functions.append(function)
    return functions


def _tool_calls_from_openai_format(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool_call["function"]["name"],
            "arguments": tool_call["function"]["arguments"],
        }
        for tool_call in tool_calls
    ]


def _encode_arguments_to_dsml(tool_call: dict[str, Any]) -> str:
    arguments = tool_call["arguments"]
    if not isinstance(arguments, dict):
        # Tolerate JSON strings, including double-encoded ones, exactly as the
        # checkpoint's encoder does.
        for _ in range(2):
            if not isinstance(arguments, str):
                break
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                break
        if not isinstance(arguments, dict):
            arguments = {"arguments": tool_call["arguments"]}

    parameters = []
    for key, value in arguments.items():
        is_str = isinstance(value, str)
        parameters.append(
            f'<{DSML_TOKEN}{TOOL_PARAMETER_TAG_NAME} name="{key}" '
            f'string="{"true" if is_str else "false"}">'
            f"{value if is_str else _to_json(value)}"
            f"</{DSML_TOKEN}{TOOL_PARAMETER_TAG_NAME}>"
        )
    return "\n".join(parameters)


def _render_tools(tools: list[dict[str, Any]]) -> str:
    return TOOLS_TEMPLATE.format(
        tool_schemas="\n".join(_to_json(tool) for tool in tools),
        dsml_token=DSML_TOKEN,
        tc_block_name=TOOL_CALLS_BLOCK_NAME,
        tool_call_tag_name=TOOL_CALL_TAG_NAME,
        tool_parameter_tag_name=TOOL_PARAMETER_TAG_NAME,
        thinking_start_token=THINKING_START_TOKEN,
        thinking_end_token=THINKING_END_TOKEN,
    )


def render_reasoning_effort(index: int, thinking_mode: str, effort: str | int | None) -> str:
    """Render the V4.1 numeric reasoning-effort prefix (thinking mode, index 0).

    Note the default: ``None`` means ``"high"`` (75), *not* "omit". Omitting this
    prefix is itself a deviation from the trained prompt format.
    """
    if effort is None:
        effort = DEFAULT_REASONING_EFFORT
    if isinstance(effort, bool) or not (
        (type(effort) is int and 1 <= effort <= 100) or effort in REASONING_EFFORT_MAPPINGS
    ):
        raise ValueError(
            f"Invalid reasoning effort for deepseek_v41: {effort!r}, should be "
            f"an int within [1, 100] or one of "
            f"{sorted(REASONING_EFFORT_MAPPINGS)}"
        )
    if isinstance(effort, str):
        effort = REASONING_EFFORT_MAPPINGS[effort]
    if index == 0 and thinking_mode == "thinking":
        return REASONING_EFFORT_TEMPLATE.format(budget=effort)
    return ""


def find_last_user_index(messages: list[dict[str, Any]]) -> int:
    """Index of the last generation boundary.

    V4.1 supports mid-conversation system messages, which count as user messages
    for the purposes of the assistant generation header.
    """
    for index in range(len(messages) - 1, -1, -1):
        role = messages[index].get("role")
        if role == "user" or (role == "system" and index > 0):
            return index
    return -1


def merge_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fold ``tool`` messages into the preceding user message as content blocks.

    V4.1 has no standalone ``tool`` role; tool results are ``<tool_result>``
    blocks inside user messages.
    """
    merged: list[dict[str, Any]] = []
    for message in messages:
        message = copy.deepcopy(message)
        role = message.get("role")

        if role == "tool":
            tool_block = {
                "type": "tool_result",
                "tool_use_id": message.get("tool_call_id", ""),
                "content": message.get("content", ""),
            }
            if merged and merged[-1].get("role") == "user" and "content_blocks" in merged[-1]:
                merged[-1]["content_blocks"].append(tool_block)
            else:
                merged.append({"role": "user", "content_blocks": [tool_block]})
        elif role == "user":
            content_blocks = message.get("content_blocks")
            if content_blocks is None:
                _reject_image_content(message.get("content"))
                content_blocks = [
                    {
                        "type": "text",
                        "text": _message_content_to_text(message.get("content"))
                        if not isinstance(message.get("content"), str)
                        else message.get("content", ""),
                    }
                ]
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
                and merged[-1].get("task") is None
            ):
                merged[-1]["content_blocks"].extend(content_blocks)
            else:
                # Preserve structured content and all message-level metadata.
                message["content_blocks"] = content_blocks
                merged.append(message)
        else:
            merged.append(message)
    return merged


def sort_tool_results_by_call_order(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Order ``tool_result`` blocks by the preceding assistant's call order."""
    last_tool_call_order: dict[str, int] = {}

    for message in messages:
        role = message.get("role")
        if role == "assistant" and message.get("tool_calls"):
            last_tool_call_order = {}
            for index, tool_call in enumerate(message["tool_calls"]):
                tool_call_id = tool_call.get("id") or tool_call.get("function", {}).get("id", "")
                if tool_call_id:
                    last_tool_call_order[tool_call_id] = index
        elif role == "user" and message.get("content_blocks"):
            tool_blocks = [
                block for block in message["content_blocks"] if block.get("type") == "tool_result"
            ]
            if len(tool_blocks) > 1 and last_tool_call_order:
                sorted_blocks = sorted(
                    tool_blocks,
                    key=lambda block: last_tool_call_order.get(block.get("tool_use_id", ""), 0),
                )
                sorted_index = 0
                new_blocks = []
                for block in message["content_blocks"]:
                    if block.get("type") == "tool_result":
                        new_blocks.append(sorted_blocks[sorted_index])
                        sorted_index += 1
                    else:
                        new_blocks.append(block)
                message["content_blocks"] = new_blocks
    return messages


def _drop_thinking_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop reasoning before the last generation boundary."""
    last_user_index = find_last_user_index(messages)
    keep_roles = {"user", "system", "tool", "latest_reminder", "direct_search_results"}

    result = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role in keep_roles or index >= last_user_index:
            result.append(message)
        elif role == "assistant":
            message = copy.copy(message)
            message.pop("reasoning_content", None)
            result.append(message)
    return result


def _render_user_content(message: dict[str, Any]) -> str:
    content_blocks = message.get("content_blocks")
    if not content_blocks:
        return _message_content_to_text(message.get("content"))

    _reject_image_content(content_blocks)
    parts = []
    for block in content_blocks:
        block_type = block.get("type")
        if block_type == "text":
            parts.append(str(block.get("text", "")))
        elif block_type == "tool_result":
            tool_content = block.get("content", "")
            if isinstance(tool_content, list):
                text_parts = []
                for item in tool_content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
                    elif isinstance(item, dict):
                        text_parts.append(f"[Unsupported {item.get('type')}]")
                    else:
                        text_parts.append(str(item))
                tool_content = "\n\n".join(text_parts)
            parts.append(TOOL_OUTPUT_TEMPLATE.format(content=tool_content))
        else:
            parts.append(f"[Unsupported {block_type}]")
    return "\n\n".join(parts)


def render_message(
    index: int,
    messages: list[dict[str, Any]],
    thinking_mode: str,
    drop_thinking: bool = True,
    reasoning_effort: str | int | None = None,
    add_generation_prompt: bool = True,
) -> str:
    """Render one message into its V4.1 encoded form.

    ``add_generation_prompt`` has no counterpart in the checkpoint's encoder,
    which always appends the assistant header. It is honoured here because the
    HuggingFace API exposes it; ``False`` suppresses the header on the final
    message only, and is not a rendering the model was trained to continue from.
    """
    if not 0 <= index < len(messages):
        raise IndexError(f"index {index} out of range for {len(messages)} messages")
    if thinking_mode not in ("chat", "thinking"):
        raise ValueError(f"Invalid thinking_mode: {thinking_mode}")

    message = messages[index]
    last_user_index = find_last_user_index(messages)

    role = message.get("role")
    content = message.get("content")
    tools = message.get("tools")
    response_format = message.get("response_format")
    tool_calls = message.get("tool_calls")
    # V4.1 reads reasoning only from `reasoning_content`.
    reasoning_content = message.get("reasoning_content") or ""
    wo_eos = message.get("wo_eos", False)

    if tools:
        tools = _tools_from_openai_format(tools)
    if tool_calls:
        tool_calls = _tool_calls_from_openai_format(tool_calls)

    reasoning_effort_prompt = render_reasoning_effort(index, thinking_mode, reasoning_effort)
    # `<|System|>` leads the conversation when there is a reasoning-effort
    # prefix or the first message is a system message.
    prompt = SYSTEM_TOKEN if index == 0 and (reasoning_effort_prompt or role == "system") else ""
    prompt += reasoning_effort_prompt

    if role == "system":
        if index > 0:
            prompt += SYSTEM_TOKEN
        prompt += _message_content_to_text(content)
        if tools:
            prompt += "\n\n" + _render_tools(tools)
        if response_format:
            prompt += "\n\n" + RESPONSE_FORMAT_TEMPLATE.format(schema=_to_json(response_format))
    elif role == "user":
        prompt += USER_TOKEN + _render_user_content(message)
    elif role == "latest_reminder":
        prompt += LATEST_REMINDER_TOKEN + _message_content_to_text(content)
    elif role == "tool":
        raise NotImplementedError(
            "DeepSeek-V4.1 merges tool messages into user messages; "
            "preprocess with merge_tool_messages()."
        )
    elif role == "assistant":
        tool_calls_content = ""
        if tool_calls:
            rendered_tool_calls = [
                TOOL_CALL_TEMPLATE.format(
                    dsml_token=DSML_TOKEN,
                    tool_call_tag_name=TOOL_CALL_TAG_NAME,
                    name=tool_call.get("name"),
                    arguments=_encode_arguments_to_dsml(tool_call),
                )
                for tool_call in tool_calls
            ]
            tool_calls_content += "\n\n" + TOOL_CALLS_TEMPLATE.format(
                dsml_token=DSML_TOKEN,
                tc_block_name=TOOL_CALLS_BLOCK_NAME,
                tool_calls="\n".join(rendered_tool_calls),
            )

        thinking_part = ""
        prev_has_task = index - 1 >= 0 and messages[index - 1].get("task") is not None
        if thinking_mode == "thinking" and not prev_has_task:
            if not drop_thinking or index > last_user_index:
                thinking_part = reasoning_content + THINKING_END_TOKEN

        prompt += thinking_part + _message_content_to_text(content) + tool_calls_content
        if not wo_eos:
            prompt += EOS_TOKEN
    else:
        # V4.1 has no `developer` role, unlike V4.
        raise NotImplementedError(f"Unsupported DeepSeek-V4.1 message role: {role}")

    next_role = messages[index + 1].get("role") if index + 1 < len(messages) else None
    if next_role is not None and next_role not in ("assistant", "latest_reminder"):
        return prompt

    task = message.get("task")
    if task is not None:
        if task not in VALID_TASKS:
            raise ValueError(
                f"Invalid DeepSeek-V4.1 task: {task!r}. Valid tasks are {sorted(VALID_TASKS)}"
            )
        if task == "action":
            prompt += ASSISTANT_TOKEN
            prompt += THINKING_START_TOKEN if thinking_mode == "thinking" else THINKING_END_TOKEN
        prompt += TASK_TOKENS[task]
    elif role == "user" or (role == "system" and index > 0):
        is_last = index + 1 == len(messages)
        if add_generation_prompt or not is_last:
            prompt += ASSISTANT_TOKEN
            if thinking_mode == "thinking" and (not drop_thinking or index >= last_user_index):
                prompt += THINKING_START_TOKEN
            else:
                prompt += THINKING_END_TOKEN

    return prompt


def encode_messages(
    messages: list[dict[str, Any]],
    thinking_mode: str,
    context: list[dict[str, Any]] | None = None,
    drop_thinking: bool = True,
    add_default_bos_token: bool = True,
    reasoning_effort: str | int | None = None,
    add_generation_prompt: bool = True,
) -> str:
    """Encode a text conversation into the DeepSeek-V4.1 prompt format."""
    context = list(context) if context else []

    messages = merge_tool_messages(messages)
    messages = sort_tool_results_by_call_order(context + messages)[len(context) :]
    if context:
        context = merge_tool_messages(context)
        context = sort_tool_results_by_call_order(context)

    full_messages = context + messages
    prompt = BOS_TOKEN if add_default_bos_token and not context else ""

    effective_drop_thinking = drop_thinking
    if any(message.get("tools") for message in full_messages):
        effective_drop_thinking = False

    if thinking_mode == "thinking" and effective_drop_thinking:
        full_messages = _drop_thinking_messages(full_messages)
        num_to_render = len(full_messages) - len(_drop_thinking_messages(context))
        context_len = len(full_messages) - num_to_render
    else:
        num_to_render = len(messages)
        context_len = len(context)

    for index in range(num_to_render):
        prompt += render_message(
            index + context_len,
            full_messages,
            thinking_mode=thinking_mode,
            drop_thinking=effective_drop_thinking,
            reasoning_effort=reasoning_effort,
            add_generation_prompt=add_generation_prompt,
        )
    return prompt


class DeepseekV41Tokenizer(TransformersTokenizer):
    """DeepSeek-V4.1 tokenizer with the checkpoint reference chat format."""

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> "DeepseekV41Tokenizer":
        # AutoTokenizer resolves the checkpoint config first, and the
        # deepseek_v41 model_type is invisible to stock transformers; the
        # registration is an import side effect of tensorrt_llm._torch.configs.
        import tensorrt_llm._torch.configs  # noqa: F401

        tokenizer = AutoTokenizer.from_pretrained(
            path_or_repo_id,
            *args,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs,
        )
        return cls(tokenizer)

    def apply_chat_template(self, messages, tools=None, **kwargs):
        tokenize = kwargs.get("tokenize", False)
        thinking = kwargs.get("thinking", False) or kwargs.get("enable_thinking", False)
        thinking_mode = "thinking" if thinking else "chat"
        # Unlike V4, any value the checkpoint accepts is passed through: the
        # budget is numeric, so clamping to {"max", "high"} would silently
        # discard a caller's "low" or explicit integer.
        reasoning_effort = kwargs.get("reasoning_effort")

        conversation = kwargs.get("conversation", messages)
        messages = list(conversation)
        if tools:
            messages.insert(0, {"role": "system", "tools": tools})

        rendered = encode_messages(
            messages=messages,
            thinking_mode=thinking_mode,
            context=kwargs.get("context"),
            drop_thinking=kwargs.get("drop_thinking", True),
            add_default_bos_token=kwargs.get("add_default_bos_token", True),
            reasoning_effort=reasoning_effort,
            add_generation_prompt=kwargs.get("add_generation_prompt", True),
        )

        if tokenize:
            tokenizer_kwargs = {
                key: kwargs[key] for key in ("truncation", "max_length") if key in kwargs
            }
            return self.encode(rendered, add_special_tokens=False, **tokenizer_kwargs)
        return rendered
