# Adapted from https://github.com/sgl-project/sglang/blob/083629c23564e1a64deaa052f1df5c5d914358d8/python/sglang/srt/function_call/kimik2_detector.py
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import re
from typing import List

from tensorrt_llm.logger import logger

from ..openai_protocol import ChatCompletionToolsParam as Tool
from .base_tool_parser import BaseToolParser
from .core_types import StreamingParseResult, StructureInfo, ToolCallItem, _GetInfoFunc


class KimiK2ToolParser(BaseToolParser):
    """Detector for Kimi K2 model function call format.

    Format Structure:
    ```
    <|tool_calls_section_begin|>
    <|tool_call_begin|>functions.{func_name}:{index}<|tool_call_argument_begin|>{json_args}<|tool_call_end|>
    <|tool_calls_section_end|>
    ```

    Reference: https://huggingface.co/moonshotai/Kimi-K2-Instruct/blob/main/docs/tool_call_guidance.md
    """

    def __init__(self):
        super().__init__()

        self.bot_token: str = "<|tool_calls_section_begin|>"
        self.eot_token: str = "<|tool_calls_section_end|>"

        self.tool_call_start_token: str = "<|tool_call_begin|>"
        self.tool_call_end_token: str = "<|tool_call_end|>"

        self.tool_call_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w.-]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*?\})\s*<\|tool_call_end\|>"
        )

        self.stream_tool_call_portion_regex = re.compile(
            r"<\|tool_call_begin\|>\s*(?P<tool_call_id>[\w.-]+:\d+)\s*<\|tool_call_argument_begin\|>\s*(?P<function_arguments>\{.*)"
        )

        self._last_arguments = ""
        # Set once the stream reaches its first tool-call token. Like
        # detect_and_parse, streaming sends no text after that token.
        self._tool_calls_started = False

        # Robust parser for ids like "functions.search:0", "functions.get-weather:0"
        # or fallback "search:0"
        self.tool_call_id_regex = re.compile(r"^(?:functions\.)?(?P<name>[\w.-]+):(?P<index>\d+)$")

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a KimiK2 format tool call."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=text, calls=[])
        try:
            # there are two possible captures - between tags, or between a
            # tag and end-of-string so the result of
            # findall is an array of tuples where one is a function call and
            # the other is None
            function_call_tuples = self.tool_call_regex.findall(text)
            tool_indices = self._get_tool_indices(tools)

            logger.debug("function_call_tuples: %s", function_call_tuples)

            tool_calls = []
            for match in function_call_tuples:
                function_id, function_args = match
                m = self.tool_call_id_regex.match(function_id)
                if not m:
                    logger.warning("Unexpected tool_call_id format: %s", function_id)
                    continue
                function_name = m.group("name")
                function_idx = int(m.group("index"))

                if function_name not in tool_indices:
                    logger.warning(f"Model attempted to call undefined function: {function_name}")
                    continue

                logger.debug(f"function_name {function_name}")

                tool_calls.append(
                    ToolCallItem(
                        tool_index=function_idx,
                        name=function_name,
                        parameters=function_args,
                    )
                )

            content = text[: text.find(self.bot_token)]
            return StreamingParseResult(normal_text=content, calls=tool_calls)

        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """Streaming incremental parsing tool calls for KimiK2 format.

        Every call that is complete by the end of an increment is released in
        full by that increment, since it may be the last one of the stream.
        The streamed text is the text before the first tool-call token, as
        ``detect_and_parse`` returns it, however the output is split.
        """
        self._buffer += new_text
        current_text = self._buffer

        normal_text = ""
        if not self._tool_calls_started:
            start = self._tool_calls_start(current_text)
            if start == -1:
                # An end of the text that may be the start of a token is held
                # back until the next increment shows whether it is one.
                cut = len(current_text) - self._partial_token_length(current_text)
                return StreamingParseResult(normal_text=self._take_text(cut))
            # The text before the tool calls goes out with the calls that
            # follow it in this increment.
            self._tool_calls_started = True
            normal_text = self._take_text(start)

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: list[ToolCallItem] = []
        try:
            # A pass over the buffer reads one call. When it completes that
            # call, the rest of the buffer may already hold the next one.
            while True:
                buffered = self._buffer
                calls.extend(self._parse_buffered_tool_call())
                if self._buffer == buffered:
                    break

            if self._tool_calls_start(self._buffer) == -1:
                # No call is left. The calls have started, so the rest of the
                # buffer is not text: drop it, keeping only an end that may
                # be the start of a token.
                cut = len(self._buffer) - self._partial_token_length(self._buffer)
                self._buffer = self._buffer[cut:]
            return StreamingParseResult(normal_text=normal_text, calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)

    def _parse_buffered_tool_call(self) -> List[ToolCallItem]:
        """Make one pass of ``parse_streaming_increment`` over the buffer."""
        current_text = self._buffer
        match = self.stream_tool_call_portion_regex.search(current_text)
        if not match:
            return []
        function_id = match.group("tool_call_id")
        function_args = match.group("function_arguments")

        m = self.tool_call_id_regex.match(function_id)
        if not m:
            logger.warning("Unexpected tool_call_id format: %s", function_id)
            return []
        function_name = m.group("name")

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

        # The arguments end at the end token. While they are incomplete, an
        # end that may be the start of that token is not streamed yet.
        arguments = function_args.split(self.tool_call_end_token, 1)[0]
        try:
            parsed_args = json.loads(arguments)
        except ValueError:  # JSONDecodeError, or an integer too long for int()
            parsed_args = None
            held = self._ends_with_partial_token(arguments, self.tool_call_end_token)
            arguments = arguments[: len(arguments) - held]

        calls: list[ToolCallItem] = []
        if not self.current_tool_name_sent:
            calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id,
                    name=function_name,
                    parameters="",
                )
            )
            self.current_tool_name_sent = True
            # Store the tool call info for serving layer completions endpoint
            self.prev_tool_call_arr[self.current_tool_id] = {
                "name": function_name,
                "arguments": {},
            }

            # A call that is already complete sends its arguments in the same
            # item: left to a later increment, they would be lost when this
            # increment is the last of the stream.
            if parsed_args is not None:
                calls[0].parameters = arguments
                self.streamed_args_for_tool[self.current_tool_id] += arguments
                self._complete_current_tool(parsed_args, match.start("function_arguments"))
        else:
            argument_diff = (
                arguments[len(self._last_arguments) :]
                if arguments.startswith(self._last_arguments)
                else arguments
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

            if parsed_args is not None:
                self._complete_current_tool(parsed_args, match.start("function_arguments"))

        return calls

    def _complete_current_tool(self, parsed_args: dict, arguments_start: int) -> None:
        """Close the current tool call, whose arguments start at ``arguments_start``."""
        self.prev_tool_call_arr[self.current_tool_id]["arguments"] = parsed_args

        # Remove the call from the buffer, up to its end token if that has
        # arrived, and keep any remaining content
        call_end = self._buffer.find(self.tool_call_end_token, arguments_start)
        if call_end == -1:
            self._buffer = ""
        else:
            self._buffer = self._buffer[call_end + len(self.tool_call_end_token) :]

        self.current_tool_id += 1
        self._last_arguments = ""
        self.current_tool_name_sent = False

    def _tool_calls_start(self, text: str) -> int:
        """Index of the first token in ``text`` that starts tool calls, or -1."""
        starts = [
            index
            for index in (text.find(self.bot_token), text.find(self.tool_call_start_token))
            if index != -1
        ]
        return min(starts, default=-1)

    def _take_text(self, end: int) -> str:
        """Take the buffer up to ``end`` as text, dropping stray end tokens."""
        text, self._buffer = self._buffer[:end], self._buffer[end:]
        for e_token in (self.eot_token, self.tool_call_end_token):
            text = text.replace(e_token, "")
        return text

    def _partial_token_length(self, text: str) -> int:
        """Length of the end of ``text`` that may be the start of a token."""
        return max(
            self._ends_with_partial_token(text, token)
            for token in (
                self.bot_token,
                self.eot_token,
                self.tool_call_start_token,
                self.tool_call_end_token,
            )
        )

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """Release the text held back as the possible start of a token.

        Once the tool calls have started, nothing is released: neither a call
        the stream left unfinished nor the start of a token.
        """
        if self._tool_calls_started:
            return StreamingParseResult()
        normal_text, self._buffer = self._buffer, ""
        return StreamingParseResult(normal_text=normal_text)

    def structure_info(self) -> _GetInfoFunc:
        """Return function that creates StructureInfo for guided generation."""

        def get_info(name: str) -> StructureInfo:
            return StructureInfo(
                begin=f"<|tool_calls_section_begin|><|tool_call_begin|>functions.{name}:0<|tool_call_argument_begin|>",
                end="<|tool_call_end|><|tool_calls_section_end|>",
                trigger="<|tool_calls_section_begin|>",
            )

        return get_info
