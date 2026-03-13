# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from typing import List

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam as Tool
from tensorrt_llm.serve.tool_parser.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from tensorrt_llm.serve.tool_parser.glm4_parser import Glm4ToolParser


class Glm47ToolParser(Glm4ToolParser):
    r"""Tool parser for GLM-4.7 models.

    Extends the GLM-4 tool parser with support for:
    - Optional arguments (tool calls with no parameters)
    - Flexible whitespace handling between function name and arguments

    GLM-4.7 tool call format:
        <tool_call>get_weather
        <arg_key>city</arg_key>
        <arg_value>Beijing</arg_value>
        </tool_call>

    Or without arguments:
        <tool_call>get_current_time</tool_call>
    """

    def __init__(self):
        """Initialize the GLM-4.7 tool parser with optional-argument regex."""
        super().__init__()
        # Override regex to make arguments optional
        self.func_detail_regex = re.compile(
            r"<tool_call>(.*?)(?:(?:\\n|\n)(.*?))?</tool_call>", re.DOTALL
        )

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-time parsing with support for optional arguments."""
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        match_result_list = re.findall(self.func_call_regex, text, re.DOTALL)
        calls = []
        try:
            for match_result in match_result_list:
                func_detail = self.func_detail_regex.search(match_result)
                if func_detail is None:
                    continue
                func_name = func_detail.group(1).strip() if func_detail.group(1) else ""
                func_args = func_detail.group(2) if func_detail.group(2) else ""

                if not func_name:
                    continue

                if func_args:
                    func_args = func_args.strip()
                    pairs = self.func_arg_regex.findall(func_args)
                    arguments = self._parse_argument_pairs(pairs, func_name, tools) if pairs else {}
                else:
                    arguments = {}

                match_result_dict = {"name": func_name, "parameters": arguments}
                calls.extend(self.parse_base_json(match_result_dict, tools))
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in Glm47ToolParser detect_and_parse: {e}")
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """Streaming incremental parsing for GLM-4.7 format.

        Extends the GLM-4 streaming parser to handle tool calls with no arguments.
        """
        self._buffer += new_text
        current_text = self._buffer

        has_tool_call = self.bot_token in current_text

        if not has_tool_call:
            is_potential_start = any(
                self.bot_token.startswith(current_text[-i:])
                for i in range(1, min(len(current_text), len(self.bot_token)) + 1)
            )

            if not is_potential_start:
                output_text = current_text
                self._buffer = ""
                if self.eot_token in output_text:
                    output_text = output_text.replace(self.eot_token, "")
                return StreamingParseResult(normal_text=output_text)
            return StreamingParseResult(normal_text="", calls=[])

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: list[ToolCallItem] = []
        try:
            # Match partial tool calls (may or may not have closing tag)
            partial_match = re.search(
                pattern=r"<tool_call>(.*?)(?:(?:\\n|\n)(.*?))?(</tool_call>|$)",
                string=current_text,
                flags=re.DOTALL,
            )
            if partial_match:
                func_name_raw = partial_match.group(1)
                func_args_raw = partial_match.group(2)
                is_tool_end = partial_match.group(3)

                if func_name_raw is None or not func_name_raw.strip():
                    return StreamingParseResult(normal_text="", calls=[])

                func_name = func_name_raw.strip()
                func_args_raw = func_args_raw.strip() if func_args_raw else ""

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

                if not self.current_tool_name_sent:
                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )
                    self.current_tool_name_sent = True
                    self._streamed_raw_length = 0
                    self._reset_streaming_state()
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": {},
                    }
                else:
                    if func_args_raw:
                        current_raw_length = len(func_args_raw)

                        if current_raw_length > self._streamed_raw_length:
                            raw_increment = func_args_raw[self._streamed_raw_length :]

                            json_increment = self._process_xml_to_json_streaming(
                                raw_increment, func_name, tools
                            )

                            self._streamed_raw_length = current_raw_length

                            if json_increment:
                                calls.append(
                                    ToolCallItem(
                                        tool_index=self.current_tool_id,
                                        name=None,
                                        parameters=json_increment,
                                    )
                                )
                                self._last_arguments += json_increment
                                self.streamed_args_for_tool[self.current_tool_id] += json_increment

                    if is_tool_end == self.eot_token:
                        # Handle tool calls with no arguments
                        if not func_args_raw or (self._is_first_param and not self._last_arguments):
                            empty_object = "{}"
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=None,
                                    parameters=empty_object,
                                )
                            )
                            self._last_arguments += empty_object
                        elif not self._last_arguments.endswith("}"):
                            closing_brace = "}"
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=None,
                                    parameters=closing_brace,
                                )
                            )
                            self._last_arguments += closing_brace
                            self.streamed_args_for_tool[self.current_tool_id] += closing_brace

                        if func_args_raw:
                            try:
                                pairs = self.func_arg_regex.findall(func_args_raw)
                                if pairs:
                                    arguments = self._parse_argument_pairs(pairs, func_name, tools)
                                    self.prev_tool_call_arr[self.current_tool_id]["arguments"] = (
                                        arguments
                                    )
                            except Exception as e:
                                logger.debug(f"Failed to parse arguments: {e}")

                        self._buffer = current_text[partial_match.end(3) :]

                        result = StreamingParseResult(normal_text="", calls=calls)
                        self.current_tool_id += 1
                        self._last_arguments = ""
                        self.current_tool_name_sent = False
                        self._streamed_raw_length = 0
                        self._reset_streaming_state()
                        return result

            return StreamingParseResult(normal_text="", calls=calls)

        except Exception as e:
            logger.error(f"Error in Glm47ToolParser parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)

    def supports_structural_tag(self) -> bool:
        """Return whether this parser supports structural tag guided decoding."""
        return False

    def structure_info(self) -> _GetInfoFunc:
        """Return structure info for guided decoding (not supported)."""
        raise NotImplementedError()
