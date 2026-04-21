# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/function_call/glm47_moe_detector.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import List, Optional

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam as Tool
from tensorrt_llm.serve.tool_parser.base_tool_parser import BaseToolParser
from tensorrt_llm.serve.tool_parser.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from tensorrt_llm.serve.tool_parser.glm4_parser import (
    StreamState,
    _convert_to_number,
    get_argument_type,
    parse_arguments,
)


class Glm47ToolParser(BaseToolParser):
    r"""Tool parser for GLM-4.7 and GLM-5 models.

    GLM-4.7 uses a slightly different tool call format compared to GLM-4.5:
      - The function name may appear on the same line as ``<tool_call>`` without
        a newline separator before the first ``<arg_key>``.
      - Tool calls may have zero arguments
        (e.g. ``<tool_call>func</tool_call>``).

    Example format::

        <tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value>
        <arg_key>date</arg_key><arg_value>2024-06-27</arg_value></tool_call>

    Or zero-argument::

        <tool_call>get_time</tool_call>
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
        self._partial_stream_regex = re.compile(
            r"<tool_call>(.*?)(?:(<arg_key.*?))?(?:(</tool_call>)|$)",
            re.DOTALL,
        )
        self._last_arguments = ""
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self._streamed_raw_length = 0
        self._reset_streaming_state()

    def _reset_streaming_state(self) -> None:
        self._stream_state = StreamState.INIT
        self._current_key = ""
        self._current_value = ""
        self._xml_tag_buffer = ""
        self._is_first_param = True
        self._value_started = False
        self._cached_value_type: Optional[str] = None

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=text, calls=[])

        normal_text_parts = []
        last_end = 0
        match_texts = []

        for match in self.func_call_regex.finditer(text):
            if match.start() > last_end:
                normal_text_parts.append(text[last_end : match.start()])
            last_end = match.end()
            match_texts.append(match.group(0))

        if last_end < len(text):
            normal_text_parts.append(text[last_end:])

        normal_text = "".join(normal_text_parts).strip()

        calls = []
        try:
            for match_result in match_texts:
                func_detail = self.func_detail_regex.search(match_result)
                if func_detail is None:
                    continue
                func_name = func_detail.group(1).strip() if func_detail.group(1) else ""
                func_args = func_detail.group(2) if func_detail.group(2) else ""
                arguments = {}
                if func_args:
                    pairs = self.func_arg_regex.findall(func_args)
                    arguments = self._parse_argument_pairs(pairs, func_name, tools)

                match_result = {"name": func_name, "parameters": arguments}
                calls.extend(self.parse_base_json(match_result, tools))
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            return StreamingParseResult(normal_text=text)

    def _get_value_type(self, func_name: str, key: str, tools: List[Tool]) -> str:
        """Get parameter type from tool definition, with fallback to auto-detection."""
        arg_type = get_argument_type(func_name, key, tools)
        if arg_type:
            return arg_type

        value_content = self._current_value.strip() if self._current_value else ""

        if not value_content:
            return "string"

        try:
            parsed = json.loads(value_content)
            if isinstance(parsed, dict):
                return "object"
            elif isinstance(parsed, list):
                return "array"
            elif isinstance(parsed, bool):
                return "boolean"
            elif isinstance(parsed, (int, float)):
                return "number"
            elif isinstance(parsed, str):
                if parsed.isdigit() or (parsed.startswith("-") and parsed[1:].isdigit()):
                    return "number"
                return "string"
        except json.JSONDecodeError:
            first_char = value_content[0] if value_content else ""
            if first_char.isdigit() or first_char in ["-", "."]:
                return "number"
            elif first_char in ["{", "["]:
                return "object"
            elif first_char in ['"', "'"]:
                return "string"

        return "string"

    def _format_value_complete(self, value: str, value_type: str) -> str:
        if value_type == "string":
            return json.dumps(value, ensure_ascii=False)
        elif value_type == "number":
            try:
                num = _convert_to_number(value.strip() if value else "")
                return str(num)
            except (ValueError, AttributeError):
                logger.warning(f"Failed to parse '{value}' as number, treating as string")
                return json.dumps(str(value) if value else "", ensure_ascii=False)
        else:
            return value

    def _process_xml_to_json_streaming(
        self, raw_increment: str, func_name: str, tools: List[Tool]
    ) -> str:
        """Convert XML increment to JSON streaming output using state machine."""
        json_output = ""

        for char in raw_increment:
            self._xml_tag_buffer += char

            if self._stream_state in [StreamState.INIT, StreamState.BETWEEN]:
                if self._xml_tag_buffer.endswith("<arg_key>"):
                    self._stream_state = StreamState.IN_KEY
                    self._current_key = ""
                    self._xml_tag_buffer = ""

            elif self._stream_state == StreamState.IN_KEY:
                if self._xml_tag_buffer.endswith("</arg_key>"):
                    self._current_key = self._xml_tag_buffer[:-10].strip()
                    self._xml_tag_buffer = ""
                    self._stream_state = StreamState.WAITING_VALUE
                    json_output += "{" if self._is_first_param else ", "
                    self._is_first_param = False
                    json_output += json.dumps(self._current_key, ensure_ascii=False) + ": "

            elif self._stream_state == StreamState.WAITING_VALUE:
                if self._xml_tag_buffer.endswith("<arg_value>"):
                    self._stream_state = StreamState.IN_VALUE
                    self._current_value = ""
                    self._xml_tag_buffer = ""
                    self._value_started = False
                    self._cached_value_type = self._get_value_type(
                        func_name, self._current_key, tools
                    )

            elif self._stream_state == StreamState.IN_VALUE:
                if self._xml_tag_buffer.endswith("</arg_value>"):
                    final_value = self._xml_tag_buffer[:-12]
                    self._current_value += final_value

                    value_type = self._cached_value_type or "string"

                    if self._value_started:
                        if final_value:
                            if value_type == "string":
                                json_output += json.dumps(final_value, ensure_ascii=False)[1:-1]
                            else:
                                json_output += final_value
                        if value_type == "string":
                            json_output += '"'
                    else:
                        json_output += self._format_value_complete(self._current_value, value_type)

                    self._xml_tag_buffer = ""
                    self._stream_state = StreamState.BETWEEN
                    self._current_value = ""
                    self._value_started = False
                    self._cached_value_type = None
                else:
                    closing_tag = "</arg_value>"
                    is_potential_closing = len(self._xml_tag_buffer) <= len(
                        closing_tag
                    ) and closing_tag.startswith(self._xml_tag_buffer)

                    if not is_potential_closing:
                        content = self._xml_tag_buffer
                        value_type = self._cached_value_type or "string"

                        if value_type == "string":
                            if not self._value_started:
                                json_output += '"'
                                self._value_started = True
                            if content:
                                json_output += json.dumps(content, ensure_ascii=False)[1:-1]
                                self._current_value += content
                                self._xml_tag_buffer = ""
                        elif value_type == "number":
                            if content:
                                if not self._value_started:
                                    self._value_started = True
                                json_output += content
                                self._current_value += content
                                self._xml_tag_buffer = ""
                        else:
                            if content:
                                if not self._value_started:
                                    self._value_started = True
                                json_output += content
                                self._current_value += content
                                self._xml_tag_buffer = ""

        return json_output

    def _extract_match_groups(self, match: re.Match) -> tuple[str, str, str]:
        func_name = match.group(1).strip()
        func_args_raw = match.group(2).strip() if match.group(2) else ""
        is_tool_end = match.group(3) or ""
        return func_name, func_args_raw, is_tool_end

    def _send_tool_name_if_needed(
        self, func_name: str, has_arg_key: bool, is_tool_end: str
    ) -> Optional[ToolCallItem]:
        if self.current_tool_name_sent:
            return None

        is_func_name_complete = has_arg_key or is_tool_end == self.eot_token
        if not is_func_name_complete:
            return None

        if not func_name:
            logger.warning("Empty function name detected, skipping tool call")
            return None

        self.current_tool_name_sent = True
        self._streamed_raw_length = 0
        self._reset_streaming_state()

        self.prev_tool_call_arr[self.current_tool_id] = {
            "name": func_name,
            "arguments": {},
        }

        return ToolCallItem(
            tool_index=self.current_tool_id,
            name=func_name,
            parameters="",
        )

    def _process_arguments_streaming(
        self, func_name: str, func_args_raw: str, tools: List[Tool]
    ) -> Optional[ToolCallItem]:
        current_raw_length = len(func_args_raw)

        if current_raw_length <= self._streamed_raw_length:
            return None

        raw_increment = func_args_raw[self._streamed_raw_length :]

        json_increment = self._process_xml_to_json_streaming(raw_increment, func_name, tools)

        self._streamed_raw_length = current_raw_length

        if not json_increment:
            return None

        self._last_arguments += json_increment
        self.streamed_args_for_tool[self.current_tool_id] += json_increment

        return ToolCallItem(
            tool_index=self.current_tool_id,
            name=None,
            parameters=json_increment,
        )

    def _finalize_tool_call(
        self,
        func_name: str,
        func_args_raw: str,
        tools: List[Tool],
        match_end_pos: int,
        current_text: str,
    ) -> List[ToolCallItem]:
        calls = []
        closing = (
            "{}"
            if self._is_first_param
            else ("}" if not self._last_arguments.endswith("}") else "")
        )
        if closing:
            calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id,
                    name=None,
                    parameters=closing,
                )
            )
            self._last_arguments += closing
            self.streamed_args_for_tool[self.current_tool_id] += closing

        if func_args_raw:
            try:
                pairs = self.func_arg_regex.findall(func_args_raw)
                if pairs:
                    arguments = self._parse_argument_pairs(pairs, func_name, tools)
                    self.prev_tool_call_arr[self.current_tool_id]["arguments"] = arguments
            except Exception as e:
                logger.debug(f"Failed to parse arguments: {e}")

        self._buffer = current_text[match_end_pos:]

        self.current_tool_id += 1
        self._last_arguments = ""
        self.current_tool_name_sent = False
        self._streamed_raw_length = 0
        self._reset_streaming_state()

        return calls

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """Streaming incremental parsing for GLM-4.7 format.

        Uses a state machine to convert XML to JSON incrementally for
        true character-by-character streaming.
        """
        self._buffer += new_text
        current_text = self._buffer

        bot_idx = current_text.find(self.bot_token)

        if bot_idx == -1:
            tail_len = min(len(current_text), len(self.bot_token) - 1)
            is_potential_start = tail_len > 0 and any(
                self.bot_token.startswith(current_text[-i:]) for i in range(1, tail_len + 1)
            )

            if not is_potential_start:
                output_text = current_text.replace(self.eot_token, "")
                self._buffer = ""
                return StreamingParseResult(normal_text=output_text)
            return StreamingParseResult(normal_text="", calls=[])

        normal_text = ""
        if bot_idx > 0:
            normal_text = current_text[:bot_idx]
            current_text = current_text[bot_idx:]
            self._buffer = current_text

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: list[ToolCallItem] = []
        try:
            partial_match = self._partial_stream_regex.search(current_text)

            if not partial_match:
                return StreamingParseResult(normal_text=normal_text, calls=[])

            func_name, func_args_raw, is_tool_end = self._extract_match_groups(partial_match)

            if self.current_tool_id == -1:
                self.current_tool_id = 0
                self.prev_tool_call_arr = []
                self.streamed_args_for_tool = [""]
                self._streamed_raw_length = 0
                self.current_tool_name_sent = False
                self._reset_streaming_state()

            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                self.prev_tool_call_arr.append({})
            while len(self.streamed_args_for_tool) <= self.current_tool_id:
                self.streamed_args_for_tool.append("")

            has_arg_key = partial_match.group(2) is not None

            tool_name_item = self._send_tool_name_if_needed(func_name, has_arg_key, is_tool_end)
            if tool_name_item:
                calls.append(tool_name_item)

            if self.current_tool_name_sent:
                arg_item = self._process_arguments_streaming(func_name, func_args_raw, tools)
                if arg_item:
                    calls.append(arg_item)

                if is_tool_end == self.eot_token:
                    finalize_calls = self._finalize_tool_call(
                        func_name,
                        func_args_raw,
                        tools,
                        partial_match.end(),
                        current_text,
                    )
                    calls.extend(finalize_calls)
                    return StreamingParseResult(normal_text=normal_text, calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _parse_argument_pairs(self, pairs, func_name: str, tools: List[Tool]) -> dict:
        """Parse argument key-value pairs with type coercion."""
        arguments = {}
        for arg_key, arg_value in pairs:
            arg_key = arg_key.strip()
            arg_type = get_argument_type(func_name, arg_key, tools)
            parsed_value, is_good_json = parse_arguments(arg_value, arg_type)

            if arg_type == "string":
                if isinstance(parsed_value, str):
                    arguments[arg_key] = parsed_value
                elif isinstance(parsed_value, (dict, list)):
                    arguments[arg_key] = json.dumps(parsed_value, ensure_ascii=False)
                else:
                    arguments[arg_key] = str(parsed_value)
            else:
                arguments[arg_key] = parsed_value if is_good_json else arg_value

        return arguments

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()
