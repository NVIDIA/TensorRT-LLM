# Adapted from https://github.com/sgl-project/sglang/blob/94e1251131ca27260cb0e8938aeb7b4a4e630b19/python/sglang/srt/function_call/deepseekv3_detector.py
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import re
from typing import List

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam as Tool
from tensorrt_llm.serve.tool_parser.base_tool_parser import BaseToolParser
from tensorrt_llm.serve.tool_parser.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)

from .utils import is_complete_json


class DeepSeekV3Parser(BaseToolParser):
    (
        r"""Tool parser for DeepSeek V3 model function call format.

    The DeepSeek V3 format uses special Unicode tokens to delimit function calls
    with JSON code blocks for arguments.

    Format Structure:
    ```
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>{function_name}\n```json\n{json_arguments}\n```<｜tool▁calls▁end｜><｜end▁of▁sentence｜>
    ```
    Examples:
    ```
    """
        r"""<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather\n"""
        r"""```json\n{"location": "Tokyo"}\n```<｜tool▁call▁end｜>\n"""
        r"""<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_current_weather\n"""
        r"""```json\n{"location": "Paris"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜>"""
        r"""
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜tool▁calls▁begin｜>` and `<｜tool▁calls▁end｜>`
    - Individual Tool Call: Wrapped between `<｜tool▁call▁begin｜>` and `<｜tool▁call▁end｜>`
    - Function Declaration: `function<｜tool▁sep｜>{function_name}`
    - Arguments: JSON code block between ````json` and ````
    - Supports multiple tool calls

    Reference: https://huggingface.co/deepseek-ai/DeepSeek-V3-0324?chat_template=default
    """
    )

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜tool▁calls▁begin｜>"  # nosec B105
        self.eot_token = "<｜tool▁calls▁end｜>"  # nosec B105
        self.func_call_regex = r"<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>"
        self.func_detail_regex = (
            r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)\n```json\n(.*)\n```<｜tool▁call▁end｜>"
        )
        self._last_arguments = ""
        self.current_tool_id = -1

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])
        match_result_list = re.findall(self.func_call_regex, text, re.DOTALL)
        calls = []
        try:
            for match_result in match_result_list:
                # Get function name
                func_detail = re.search(self.func_detail_regex, match_result, re.DOTALL)
                func_name = func_detail.group(2)
                func_args = func_detail.group(3)
                func_args = json.loads(func_args)
                # construct match_result for parse_base_json
                match_result = {"name": func_name, "parameters": func_args}
                calls.extend(self.parse_base_json(match_result, tools))
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """Streaming incremental parsing tool calls for DeepSeekV3 format.

        Every call whose arguments are complete by the end of an increment is
        sent in full by that increment, since it may be the last one of the
        stream.
        """
        self._buffer += new_text
        normal_text = ""
        calls: list[ToolCallItem] = []
        # A pass over the buffer completes at most one call. When it does, what
        # is left may already hold the next call, so parse again.
        while True:
            result, completed_call = self._parse_buffered_tool_call(tools)
            normal_text += result.normal_text
            calls.extend(result.calls)
            if not completed_call:
                return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _parse_buffered_tool_call(self, tools: List[Tool]) -> tuple[StreamingParseResult, bool]:
        """Make one pass of ``parse_streaming_increment`` over the buffer.

        Returns what the pass parsed and whether it completed a call.
        """
        current_text = self._buffer

        # Check if we have a tool call (either the start token or individual tool call)
        start_tokens = [self.bot_token, "<｜tool▁call▁begin｜>"]
        end_tokens = [self.eot_token, "<｜tool▁call▁end｜>"]
        start_indices = [idx for idx in map(current_text.find, start_tokens) if idx != -1]
        has_tool_call = bool(start_indices)

        if not has_tool_call:
            # Only a tail that may still become a marker waits for the next
            # increment; the text before it is content.
            held_len = max(
                self._ends_with_partial_token(current_text, token)
                for token in start_tokens + end_tokens
            )
            normal_text = current_text[: len(current_text) - held_len]
            self._buffer = current_text[len(current_text) - held_len :]
            for e_token in end_tokens:
                normal_text = normal_text.replace(e_token, "")
            return StreamingParseResult(normal_text=normal_text), False

        # Text before the earliest start token is content, so stream it without
        # the end markers of earlier sections and keep the buffer from that
        # token onwards.
        prefix_len = min(start_indices, default=0)
        normal_text = current_text[:prefix_len]
        if normal_text:
            current_text = current_text[prefix_len:]
            self._buffer = current_text
        for e_token in end_tokens:
            normal_text = normal_text.replace(e_token, "")

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: list[ToolCallItem] = []
        try:
            # Parse the first call in the buffer: the pattern is greedy and
            # would otherwise read on into a later call.
            first_call = re.search(self.func_call_regex, current_text, re.DOTALL)
            partial_match = re.search(
                pattern=r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)\n```json\n(.*)\n```.*",
                string=current_text[: first_call.end()] if first_call else current_text,
                flags=re.DOTALL,
            )
            if partial_match:
                func_name = partial_match.group(2).strip()
                func_args_raw = partial_match.group(3).strip()
                is_complete = is_complete_json(func_args_raw)

                # Initialize state if this is the first tool call
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]

                # Ensure we have enough entries in our tracking arrays
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
                    # Store the tool call info for serving layer completions endpoint
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": {},
                    }
                    # A call that is already complete sends its arguments in
                    # the same item: left to a later increment, they would be
                    # lost when this increment is the last of the stream.
                    if is_complete:
                        calls[0].parameters = func_args_raw
                        self._last_arguments = func_args_raw
                        self.streamed_args_for_tool[self.current_tool_id] = func_args_raw
                else:
                    argument_diff = (
                        func_args_raw[len(self._last_arguments) :]
                        if func_args_raw.startswith(self._last_arguments)
                        else func_args_raw
                    )

                    if argument_diff:
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=None,
                                parameters=argument_diff,
                            )
                        )
                        self._last_arguments += argument_diff
                        self.streamed_args_for_tool[self.current_tool_id] += argument_diff

                # The call closes with its whole end marker, which may arrive
                # after the arguments are complete: closed before that, it
                # would leave the rest of the marker to be read as text.
                if is_complete and first_call is not None:
                    # Update the stored arguments
                    try:
                        parsed_args = json.loads(func_args_raw)
                        self.prev_tool_call_arr[self.current_tool_id]["arguments"] = parsed_args
                    except json.JSONDecodeError:
                        pass

                    # Remove the completed tool call from buffer, keep any remaining content
                    self._buffer = current_text[first_call.end() :]

                    result = StreamingParseResult(normal_text=normal_text, calls=calls)
                    self.current_tool_id += 1
                    self._last_arguments = ""
                    self.current_tool_name_sent = False
                    return result, True

            return StreamingParseResult(normal_text=normal_text, calls=calls), False

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=normal_text + current_text), False

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=">" + name + "\n```json\n",
            end="\n```<",
            trigger=">" + name + "\n```json\n",
        )
