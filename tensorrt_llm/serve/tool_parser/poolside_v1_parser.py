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
import json
import re
from typing import Any

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam as Tool
from tensorrt_llm.serve.tool_parser.base_tool_parser import BaseToolParser
from tensorrt_llm.serve.tool_parser.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from tensorrt_llm.serve.tool_parser.utils import infer_type_from_json_schema

TOOL_CALL_START = "<tool_call>"  # nosec B105
TOOL_CALL_END = "</tool_call>"  # nosec B105
ARG_KEY_START = "<arg_key>"  # nosec B105
ARG_KEY_END = "</arg_key>"  # nosec B105
ARG_VALUE_START = "<arg_value>"  # nosec B105
ARG_VALUE_END = "</arg_value>"  # nosec B105


def _get_argument_type(func_name: str, arg_key: str, tools: list[Tool]) -> str | None:
    """Return the JSON-schema type for a tool argument, if declared."""
    name_to_tool = {tool.function.name: tool for tool in tools}
    tool = name_to_tool.get(func_name)
    if tool is None:
        return None

    properties = (tool.function.parameters or {}).get("properties", {})
    if not isinstance(properties, dict):
        return None

    schema = properties.get(arg_key)
    if not isinstance(schema, dict):
        return None

    return infer_type_from_json_schema(schema)


def _convert_number(value: str) -> Any:
    """Convert a numeric string to int or float."""
    if "." in value or "e" in value.lower():
        return float(value)
    return int(value)


def _parse_argument_value(value: str, arg_type: str | None) -> Any:
    """Parse a raw argument value according to its schema type."""
    value = value.strip()

    if arg_type == "string":
        return value

    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        parsed = None
    else:
        if arg_type == "integer" and not isinstance(parsed, bool):
            try:
                return int(parsed)
            except (TypeError, ValueError):
                return parsed
        if arg_type == "number" and not isinstance(parsed, bool):
            try:
                return float(parsed)
            except (TypeError, ValueError):
                return parsed
        return parsed

    if arg_type == "integer":
        try:
            return int(value)
        except (TypeError, ValueError):
            return value

    if arg_type == "number":
        try:
            return _convert_number(value)
        except (TypeError, ValueError):
            return value

    if arg_type == "boolean":
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False

    return value


class PoolsideV1ToolParser(BaseToolParser):
    """Tool parser for Poolside Laguna's function call format.

    The Laguna chat template asks the model to emit tool calls as:

        <tool_call>function-name
        <arg_key>argument-key</arg_key>
        <arg_value>value-of-argument-key</arg_value>
        </tool_call>

    Multiple arguments are encoded as repeated ``arg_key`` / ``arg_value`` pairs,
    and multiple tool calls are encoded as repeated ``tool_call`` blocks.
    """

    def __init__(self):
        super().__init__()
        self.bot_token = TOOL_CALL_START
        self.eot_token = TOOL_CALL_END
        # Match each <arg_key>...</arg_key> / <arg_value>...</arg_value> pair,
        # allowing either real whitespace or a literal "\n" between the tags.
        self.func_arg_regex = re.compile(
            rf"{ARG_KEY_START}(.*?){ARG_KEY_END}(?:\\n|\s)*{ARG_VALUE_START}(.*?){ARG_VALUE_END}",
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        """Parse complete generated text.

        Returns a StreamingParseResult containing normal text with tool blocks
        removed and one ToolCallItem per complete Poolside v1 tool call.
        """
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=text, calls=[])

        normal_text_parts = []
        match_texts = []
        cursor = 0

        while cursor < len(text):
            bot_idx = text.find(self.bot_token, cursor)
            if bot_idx == -1:
                normal_text_parts.append(text[cursor:])
                break

            normal_text_parts.append(text[cursor:bot_idx])
            block_end = self._find_complete_tool_call_end(text, bot_idx)
            if block_end is None:
                # Non-streaming output cannot receive more chunks, so preserve
                # the malformed/incomplete tool block as normal assistant text.
                normal_text_parts.append(text[bot_idx:])
                break

            match_texts.append(text[bot_idx:block_end])
            cursor = block_end

        calls = []
        for match_text in match_texts:
            call = self._parse_tool_call_block(match_text, tools)
            if call is not None:
                calls.append(call)

        return StreamingParseResult(normal_text="".join(normal_text_parts).strip(), calls=calls)

    def parse_streaming_increment(self, new_text: str, tools: list[Tool]) -> StreamingParseResult:
        """Parse one streaming text increment.

        Returns normal text that can be emitted immediately plus tool-call
        deltas for any complete tool-call blocks. Incomplete tool calls remain
        buffered and produce no tool-call delta.
        """
        self._buffer += new_text
        normal_text_parts: list[str] = []
        calls: list[ToolCallItem] = []

        while self._buffer:
            bot_idx = self._buffer.find(self.bot_token)
            if bot_idx == -1:
                if self._ends_with_partial_token(self._buffer, self.bot_token):
                    break
                normal_text_parts.append(self._buffer.replace(self.eot_token, ""))
                self._buffer = ""
                break

            if bot_idx > 0:
                normal_text_parts.append(self._buffer[:bot_idx])
                self._buffer = self._buffer[bot_idx:]

            block_end = self._find_complete_tool_call_end(self._buffer, 0)
            if block_end is None:
                break

            block = self._buffer[:block_end]
            call = self._parse_tool_call_block(block, tools)
            if call is not None:
                if not self.current_tool_name_sent:
                    calls.append(self._start_streaming_tool_call(call.name or ""))
                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=None,
                        parameters=call.parameters,
                    )
                )
                self.current_tool_id += 1
                self.current_tool_name_sent = False

            self._buffer = self._buffer[block_end:]

        return StreamingParseResult(normal_text="".join(normal_text_parts), calls=calls)

    def supports_structural_tag(self) -> bool:
        """Return whether this parser supports strict structural tags."""
        return False

    def structure_info(self) -> _GetInfoFunc:
        """Return structural tag info for guided decoding."""
        raise NotImplementedError()

    def _parse_tool_call_block(self, text: str, tools: list[Tool]) -> ToolCallItem | None:
        """Parse one complete <tool_call>...</tool_call> block.

        Returns a ToolCallItem with JSON-encoded arguments, or None when the
        block is malformed or has no function name.
        """
        if not text.startswith(self.bot_token) or not text.endswith(self.eot_token):
            return None

        inner_text = text[len(self.bot_token) : -len(self.eot_token)]
        arg_start_idx = inner_text.find(ARG_KEY_START)
        if arg_start_idx == -1:
            func_name = inner_text.strip()
            func_args = ""
        else:
            func_name = inner_text[:arg_start_idx].strip()
            func_args = inner_text[arg_start_idx:]

        if not func_name:
            logger.warning("Empty Poolside v1 tool call name detected, skipping tool call")
            return None

        arguments = self._parse_argument_pairs(
            self.func_arg_regex.findall(func_args), func_name, tools
        )
        tool_indices = self._get_tool_indices(tools)
        tool_index = tool_indices.get(func_name, -1)
        if tool_index == -1:
            logger.warning(f"Model attempted to call undefined function: {func_name}")

        return ToolCallItem(
            tool_index=tool_index,
            name=func_name,
            parameters=json.dumps(arguments, ensure_ascii=False),
        )

    def _parse_argument_pairs(self, pairs, func_name: str, tools: list[Tool]) -> dict[str, Any]:
        """Convert regex argument key/value matches into a Python dict.

        Returns argument names mapped to schema-coerced Python values.
        """
        arguments = {}
        for arg_key, arg_value in pairs:
            arg_key = arg_key.strip()
            arg_type = _get_argument_type(func_name, arg_key, tools)
            arguments[arg_key] = _parse_argument_value(arg_value, arg_type)
        return arguments

    def _find_complete_tool_call_end(self, text: str, start_idx: int) -> int | None:
        """Find the end of a complete tool-call block.

        Returns the exclusive end index of the matching </tool_call>, or None
        if the block is incomplete. </tool_call> text inside <arg_value> is
        treated as argument content.
        """
        pos = start_idx + len(self.bot_token)
        in_arg_value = False
        while pos < len(text):
            if in_arg_value:
                arg_value_end_idx = text.find(ARG_VALUE_END, pos)
                if arg_value_end_idx == -1:
                    return None
                pos = arg_value_end_idx + len(ARG_VALUE_END)
                in_arg_value = False
                continue

            eot_idx = text.find(self.eot_token, pos)
            arg_value_start_idx = text.find(ARG_VALUE_START, pos)
            # Only treat </tool_call> as the block end when it appears before
            # the next <arg_value>; otherwise it is part of the argument text.
            if eot_idx != -1 and (arg_value_start_idx == -1 or eot_idx < arg_value_start_idx):
                return eot_idx + len(self.eot_token)
            if arg_value_start_idx == -1:
                return None

            pos = arg_value_start_idx + len(ARG_VALUE_START)
            in_arg_value = True

        return None

    def _start_streaming_tool_call(self, func_name: str) -> ToolCallItem:
        """Create the streaming delta that announces a new tool name."""
        if self.current_tool_id == -1:
            self.current_tool_id = 0

        self.current_tool_name_sent = True
        return ToolCallItem(
            tool_index=self.current_tool_id,
            name=func_name,
            parameters="",
        )
