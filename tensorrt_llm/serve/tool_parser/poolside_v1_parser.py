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
from typing import Any, List, Optional

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam as Tool
from tensorrt_llm.serve.tool_parser.base_tool_parser import BaseToolParser
from tensorrt_llm.serve.tool_parser.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from tensorrt_llm.serve.tool_parser.utils import infer_type_from_json_schema


def _get_argument_type(func_name: str, arg_key: str, tools: List[Tool]) -> Optional[str]:
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
    if "." in value or "e" in value.lower():
        return float(value)
    return int(value)


def _parse_argument_value(value: str, arg_type: Optional[str]) -> Any:
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
    r"""Tool parser for Poolside Laguna's v1 function call format.

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
        self.bot_token = "<tool_call>"  # nosec B105
        self.eot_token = "</tool_call>"  # nosec B105
        self.func_call_regex = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
        self.func_detail_regex = re.compile(
            r"<tool_call>(.*?)(<arg_key>.*?)?</tool_call>", re.DOTALL
        )
        self.func_arg_regex = re.compile(
            r"<arg_key>(.*?)</arg_key>(?:\\n|\s)*<arg_value>(.*?)</arg_value>",
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=text, calls=[])

        normal_text_parts = []
        match_texts = []
        last_end = 0

        for match in self.func_call_regex.finditer(text):
            if match.start() > last_end:
                normal_text_parts.append(text[last_end : match.start()])
            last_end = match.end()
            match_texts.append(match.group(0))

        if last_end < len(text):
            normal_text_parts.append(text[last_end:])

        calls = []
        for match_text in match_texts:
            call = self._parse_tool_call_block(match_text, tools)
            if call is not None:
                calls.append(call)

        return StreamingParseResult(normal_text="".join(normal_text_parts).strip(), calls=calls)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
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

            eot_idx = self._buffer.find(self.eot_token)
            if eot_idx == -1:
                name = self._extract_partial_function_name(self._buffer)
                if name and not self.current_tool_name_sent:
                    calls.append(self._start_streaming_tool_call(name))
                break

            block_end = eot_idx + len(self.eot_token)
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
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()

    def _parse_tool_call_block(self, text: str, tools: List[Tool]) -> Optional[ToolCallItem]:
        match = self.func_detail_regex.search(text)
        if match is None:
            return None

        func_name = match.group(1).strip() if match.group(1) else ""
        if not func_name:
            logger.warning("Empty Poolside v1 tool call name detected, skipping tool call")
            return None

        func_args = match.group(2) if match.group(2) else ""
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

    def _parse_argument_pairs(self, pairs, func_name: str, tools: List[Tool]) -> dict[str, Any]:
        arguments = {}
        for arg_key, arg_value in pairs:
            arg_key = arg_key.strip()
            arg_type = _get_argument_type(func_name, arg_key, tools)
            arguments[arg_key] = _parse_argument_value(arg_value, arg_type)
        return arguments

    def _extract_partial_function_name(self, text: str) -> str:
        inner_text = text[len(self.bot_token) :]
        arg_idx = inner_text.find("<arg_key>")
        newline_idx = inner_text.find("\n")

        candidates = [idx for idx in (arg_idx, newline_idx) if idx != -1]
        if not candidates:
            return ""

        return inner_text[: min(candidates)].strip()

    def _start_streaming_tool_call(self, func_name: str) -> ToolCallItem:
        if self.current_tool_id == -1:
            self.current_tool_id = 0

        self.current_tool_name_sent = True
        return ToolCallItem(
            tool_index=self.current_tool_id,
            name=func_name,
            parameters="",
        )
