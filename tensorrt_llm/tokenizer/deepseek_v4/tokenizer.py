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

import copy
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from ..tokenizer import TransformersTokenizer

BOS_TOKEN = "<｜begin▁of▁sentence｜>"  # nosec B105
EOS_TOKEN = "<｜end▁of▁sentence｜>"  # nosec B105
USER_TOKEN = "<｜User｜>"  # nosec B105
ASSISTANT_TOKEN = "<｜Assistant｜>"  # nosec B105
LATEST_REMINDER_TOKEN = "<｜latest_reminder｜>"  # nosec B105
THINKING_START_TOKEN = "<think>"  # nosec B105
THINKING_END_TOKEN = "</think>"  # nosec B105
DSML_TOKEN = "｜DSML｜"  # nosec B105

TOOL_CALLS_BLOCK_NAME = "tool_calls"
VALID_TASKS = {
    "action": "<｜action｜>",
    "query": "<｜query｜>",
    "authority": "<｜authority｜>",
    "domain": "<｜domain｜>",
    "title": "<｜title｜>",
    "read_url": "<｜read_url｜>",
}

REASONING_EFFORT_MAX = (
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose "
    "the problem to resolve the root cause, rigorously stress-testing your "
    "logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every "
    "intermediate step, considered alternative, and rejected hypothesis to "
    "ensure absolutely no assumption is left unchecked.\n\n"
)

TOOLS_TEMPLATE = """## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml_token}tool_calls>" block like the following:

<{dsml_token}tool_calls>
<{dsml_token}invoke name="$TOOL_NAME">
<{dsml_token}parameter name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml_token}parameter>
...
</{dsml_token}invoke>
<{dsml_token}invoke name="$TOOL_NAME2">
...
</{dsml_token}invoke>
</{dsml_token}tool_calls>

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
TOOL_CALL_TEMPLATE = '<{dsml_token}invoke name="{name}">\n{arguments}\n</{dsml_token}invoke>'
TOOL_CALLS_TEMPLATE = "<{dsml_token}{block_name}>\n{tool_calls}\n</{dsml_token}{block_name}>"
TOOL_OUTPUT_TEMPLATE = "<tool_result>{content}</tool_result>"


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "\n\n".join(parts)
    return str(content)


def _to_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return json.dumps(value, ensure_ascii=True)


def _tools_from_openai_format(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [tool["function"] for tool in tools]


def _tool_calls_from_openai_format(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": tool_call["function"]["name"],
            "arguments": tool_call["function"]["arguments"],
        }
        for tool_call in tool_calls
    ]


def _encode_arguments_to_dsml(tool_call: dict[str, Any]) -> str:
    raw_args = tool_call["arguments"]
    arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    if not isinstance(arguments, dict):
        raise ValueError("DeepSeek-V4 tool call arguments must be a JSON object.")

    parameters = []
    for key, value in arguments.items():
        parameters.append(
            f'<{DSML_TOKEN}parameter name="{key}" '
            f'string="{"true" if isinstance(value, str) else "false"}">'
            f"{value if isinstance(value, str) else _to_json(value)}</{DSML_TOKEN}parameter>"
        )
    return "\n".join(parameters)


def _render_tools(tools: list[dict[str, Any]]) -> str:
    return TOOLS_TEMPLATE.format(
        tool_schemas="\n".join(_to_json(tool) for tool in tools),
        dsml_token=DSML_TOKEN,
        thinking_start_token=THINKING_START_TOKEN,
        thinking_end_token=THINKING_END_TOKEN,
    )


def _find_last_user_index(messages: list[dict[str, Any]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") in ("user", "developer"):
            return index
    return -1


def _merge_tool_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
            text_block = {"type": "text", "text": _message_content_to_text(message.get("content"))}
            if (
                merged
                and merged[-1].get("role") == "user"
                and "content_blocks" in merged[-1]
                and merged[-1].get("task") is None
            ):
                merged[-1]["content_blocks"].append(text_block)
            else:
                new_message = {
                    "role": "user",
                    "content": message.get("content", ""),
                    "content_blocks": [text_block],
                }
                for key in ("task", "wo_eos", "mask"):
                    if key in message:
                        new_message[key] = message[key]
                merged.append(new_message)
        else:
            merged.append(message)

    return merged


def _sort_tool_results_by_call_order(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    last_tool_call_order: dict[str, int] = {}

    for message in messages:
        if message.get("role") == "assistant" and message.get("tool_calls"):
            last_tool_call_order = {}
            for index, tool_call in enumerate(message["tool_calls"]):
                tool_call_id = tool_call.get("id") or tool_call.get("function", {}).get("id", "")
                if tool_call_id:
                    last_tool_call_order[tool_call_id] = index
        elif message.get("role") == "user" and message.get("content_blocks"):
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
    last_user_index = _find_last_user_index(messages)
    result = []
    keep_roles = {"user", "system", "tool", "latest_reminder", "direct_search_results"}

    for index, message in enumerate(messages):
        role = message.get("role")
        if role in keep_roles or index >= last_user_index:
            result.append(message)
        elif role == "assistant":
            message_without_reasoning = copy.copy(message)
            message_without_reasoning.pop("reasoning", None)
            message_without_reasoning.pop("reasoning_content", None)
            result.append(message_without_reasoning)

    return result


def _render_user_content(message: dict[str, Any]) -> str:
    content_blocks = message.get("content_blocks")
    if not content_blocks:
        return _message_content_to_text(message.get("content"))

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


def _render_message(
    index: int,
    messages: list[dict[str, Any]],
    thinking_mode: str,
    drop_thinking: bool,
    add_generation_prompt: bool,
    reasoning_effort: str | None,
) -> str:
    if thinking_mode not in ("chat", "thinking"):
        raise ValueError(f"Invalid thinking_mode: {thinking_mode}")

    message = messages[index]
    last_user_index = _find_last_user_index(messages)
    role = message.get("role")
    content = _message_content_to_text(message.get("content"))
    tools = message.get("tools")
    response_format = message.get("response_format")
    tool_calls = message.get("tool_calls")
    reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
    prompt = ""

    if tools:
        tools = _tools_from_openai_format(tools)
    if tool_calls:
        tool_calls = _tool_calls_from_openai_format(tool_calls)

    if index == 0 and thinking_mode == "thinking" and reasoning_effort == "max":
        prompt += REASONING_EFFORT_MAX

    if role == "system":
        prompt += content
        if tools:
            prompt += "\n\n" + _render_tools(tools)
        if response_format:
            prompt += "\n\n" + RESPONSE_FORMAT_TEMPLATE.format(schema=_to_json(response_format))
    elif role == "developer":
        prompt += USER_TOKEN + content
        if tools:
            prompt += "\n\n" + _render_tools(tools)
        if response_format:
            prompt += "\n\n" + RESPONSE_FORMAT_TEMPLATE.format(schema=_to_json(response_format))
    elif role == "user":
        prompt += USER_TOKEN + _render_user_content(message)
    elif role == "latest_reminder":
        prompt += LATEST_REMINDER_TOKEN + content
    elif role == "tool":
        raise NotImplementedError(
            "DeepSeek-V4 merges tool messages into user messages; "
            "preprocess with _merge_tool_messages()."
        )
    elif role == "assistant":
        tool_calls_content = ""
        if tool_calls:
            rendered_tool_calls = [
                TOOL_CALL_TEMPLATE.format(
                    dsml_token=DSML_TOKEN,
                    name=tool_call.get("name"),
                    arguments=_encode_arguments_to_dsml(tool_call),
                )
                for tool_call in tool_calls
            ]
            tool_calls_content += "\n\n" + TOOL_CALLS_TEMPLATE.format(
                dsml_token=DSML_TOKEN,
                block_name=TOOL_CALLS_BLOCK_NAME,
                tool_calls="\n".join(rendered_tool_calls),
            )

        thinking_part = ""
        prev_has_task = index - 1 >= 0 and messages[index - 1].get("task") is not None
        if thinking_mode == "thinking" and not prev_has_task:
            if not drop_thinking or index > last_user_index:
                thinking_part = reasoning + THINKING_END_TOKEN

        if message.get("wo_eos", False):
            prompt += thinking_part + content + tool_calls_content
        else:
            prompt += thinking_part + content + tool_calls_content + EOS_TOKEN
    else:
        raise NotImplementedError(f"Unsupported DeepSeek-V4 message role: {role}")

    next_role = messages[index + 1].get("role") if index + 1 < len(messages) else None
    if next_role is not None and next_role not in ("assistant", "latest_reminder"):
        return prompt

    task = message.get("task")
    if task is not None:
        if task not in VALID_TASKS:
            raise ValueError(f"Invalid DeepSeek-V4 task: {task}")
        if task == "action":
            prompt += ASSISTANT_TOKEN
            prompt += THINKING_START_TOKEN if thinking_mode == "thinking" else THINKING_END_TOKEN
        prompt += VALID_TASKS[task]
    elif role in ("user", "developer") and (next_role == "assistant" or add_generation_prompt):
        prompt += ASSISTANT_TOKEN
        if thinking_mode == "thinking" and (not drop_thinking or index >= last_user_index):
            prompt += THINKING_START_TOKEN
        else:
            prompt += THINKING_END_TOKEN

    return prompt


def _encode_messages(
    messages: list[dict[str, Any]],
    thinking_mode: str,
    drop_thinking: bool,
    add_generation_prompt: bool,
    reasoning_effort: str | None,
) -> str:
    messages = _merge_tool_messages(messages)
    messages = _sort_tool_results_by_call_order(messages)

    effective_drop_thinking = drop_thinking
    if any(message.get("tools") for message in messages):
        effective_drop_thinking = False

    if thinking_mode == "thinking" and effective_drop_thinking:
        messages = _drop_thinking_messages(messages)

    prompt = BOS_TOKEN
    for index in range(len(messages)):
        prompt += _render_message(
            index,
            messages,
            thinking_mode=thinking_mode,
            drop_thinking=effective_drop_thinking,
            add_generation_prompt=add_generation_prompt,
            reasoning_effort=reasoning_effort,
        )
    return prompt


class DeepseekV4Tokenizer(TransformersTokenizer):
    """DeepSeek-V4 tokenizer with the checkpoint reference chat format."""

    @classmethod
    def from_pretrained(
        cls,
        path_or_repo_id: str | Path,
        *args,
        trust_remote_code: bool = False,
        revision: str | None = None,
        **kwargs,
    ) -> "DeepseekV4Tokenizer":
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
        reasoning_effort = kwargs.get("reasoning_effort")
        if reasoning_effort not in ("max", "high"):
            reasoning_effort = None

        conversation = kwargs.get("conversation", messages)
        messages = list(conversation)
        if tools:
            messages.insert(0, {"role": "system", "tools": tools})

        rendered = _encode_messages(
            messages=messages,
            thinking_mode=thinking_mode,
            drop_thinking=kwargs.get("drop_thinking", True),
            add_generation_prompt=True,
            reasoning_effort=reasoning_effort,
        )

        if tokenize:
            tokenizer_kwargs = {
                key: kwargs[key] for key in ("truncation", "max_length") if key in kwargs
            }
            return self.encode(rendered, add_special_tokens=False, **tokenizer_kwargs)
        return rendered
