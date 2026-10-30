# Adapted from https://github.com/sgl-project/sglang/blob/94e1251131ca27260cb0e8938aeb7b4a4e630b19/python/sglang/srt/function_call/deepseekv31_detector.py
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

_TOOL_SEP = "<｜tool▁sep｜>"
_TOOL_CALL_END = "<｜tool▁call▁end｜>"


class DeepSeekV31Parser(BaseToolParser):
    (
        """Tool parser for DeepSeek V3 model function call format.

        The DeepSeek V3 format uses special Unicode tokens to delimit function calls
        with JSON code blocks for arguments.

        Format Structure:
        ```
        <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>{function_name}<｜tool▁sep｜>{json_arguments}<｜tool▁calls▁end｜><｜end▁of▁sentence｜>
        ```
        Examples:
        ```
        """
        """<｜tool▁calls▁begin｜>"""
        """<｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "Tokyo"}<｜tool▁call▁end｜>"""
        """<｜tool▁call▁begin｜>get_current_weather<｜tool▁sep｜>{"location": "Paris"}<｜tool▁call▁end｜>"""
        """<｜tool▁calls▁end｜><｜end▁of▁sentence｜>
        ```

        Key Components:
        - Tool Calls Section: Wrapped between `<｜tool▁calls▁begin｜>` and `<｜tool▁calls▁end｜>`
        - Individual Tool Call: Wrapped between `<｜tool▁call▁begin｜>` and `<｜tool▁call▁end｜>`
        - Function Declaration: `<｜tool▁call▁begin｜>{function_name}<｜tool▁sep｜>`
        - Arguments: JSON code block between `<｜tool▁sep｜>` and `<｜tool▁call▁end｜>`
        - Supports multiple tool calls

        Reference: https://www.modelscope.cn/models/deepseek-ai/DeepSeek-V3.1
    """
    )

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜tool▁calls▁begin｜>"  # nosec B105
        self.eot_token = "<｜tool▁calls▁end｜>"  # nosec B105
        self.func_call_regex = r"<｜tool▁call▁begin｜>.*?<｜tool▁call▁end｜>"
        self.func_detail_regex = r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*)<｜tool▁call▁end｜>"
        self._last_arguments = ""
        self.current_tool_id = -1
        # Streaming state of the current call's arguments, see _stream_arguments.
        # An _args_start of -1 means the pattern match reads the arguments.
        self._start_arguments(-1)
        # Buffer index from which a tool separator could still appear. Text
        # before it was already searched.
        self._sep_search_start = 0
        self._pending_call_end = False

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek format tool call."""
        return self.bot_token in text

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """Send the arguments of a tool call whose end came in the last delta.

        An increment sends at most one tool name or the arguments of one call,
        and the arguments go out one increment after the name. So when the last
        delta holds the end of a call, its name has been sent and its arguments
        have not. They are sent here when the buffer holds only that call, with
        its end token, the same non-empty name and a JSON object that continues
        what was already sent. In any other case nothing is sent.
        """
        func_detail = None
        if self.current_tool_name_sent and self._buffer.count("<｜tool▁call▁begin｜>") == 1:
            match = re.search(self.func_call_regex, self._buffer, re.DOTALL)
            func_detail = match and re.search(self.func_detail_regex, match.group(), re.DOTALL)
        if not func_detail:
            return StreamingParseResult()
        func_name = func_detail.group(1)
        func_args = func_detail.group(2).strip()
        if (
            not func_name
            or func_name != self.prev_tool_call_arr[self.current_tool_id]["name"]
            or not (func_args.startswith("{") and is_complete_json(func_args))
            or not func_args.startswith(self._last_arguments)
            or func_args == self._last_arguments
        ):
            return StreamingParseResult()
        argument_diff = func_args[len(self._last_arguments) :]
        self._last_arguments = func_args
        self.streamed_args_for_tool[self.current_tool_id] += argument_diff
        self.prev_tool_call_arr[self.current_tool_id]["arguments"] = json.loads(func_args)
        return StreamingParseResult(
            calls=[ToolCallItem(tool_index=self.current_tool_id, parameters=argument_diff)]
        )

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
                func_name = func_detail.group(1)
                func_args = func_detail.group(2)
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
        """Streaming incremental parsing tool calls for DeepSeekV3 format."""
        self._buffer += new_text
        current_text = self._buffer

        if self._pending_call_end:
            pending = current_text.lstrip()
            if not pending or (len(pending) < len(_TOOL_CALL_END)
                               and _TOOL_CALL_END.startswith(pending)):
                self._buffer = pending
                return StreamingParseResult()
            if pending.startswith(_TOOL_CALL_END):
                self._buffer = pending[len(_TOOL_CALL_END) :]
                current_text = self._buffer
            self._pending_call_end = False

        # Check if we have a tool call (either the start token or individual tool call)
        start_tokens = [self.bot_token, "<｜tool▁call▁begin｜>"]
        start_indices = [idx for idx in map(current_text.find, start_tokens) if idx != -1]
        has_tool_call = bool(start_indices)

        if not has_tool_call:
            if any(
                self._ends_with_partial_token(current_text, token)
                for token in (*start_tokens, self.eot_token, _TOOL_CALL_END)
            ):
                return StreamingParseResult()
            normal_text = current_text
            self._buffer = ""
            self._sep_search_start = 0
            for e_token in [self.eot_token, "<｜tool▁call▁end｜>"]:
                normal_text = normal_text.replace(e_token, "")
            return StreamingParseResult(normal_text=normal_text)

        # Text before the earliest start token is content, so stream it and keep the
        # buffer from that token onwards.
        prefix_len = min(start_indices, default=0)
        normal_text = current_text[:prefix_len]
        if normal_text:
            current_text = current_text[prefix_len:]
            self._buffer = current_text
            self._sep_search_start = max(0, self._sep_search_start - prefix_len)

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: list[ToolCallItem] = []
        try:
            # The pattern below matches only once a separator arrives, and while
            # the arguments of a call stream, a new separator is what changes its
            # match. Without one in the new text, the arguments are read from
            # the new text alone, so a long tool call is not rescanned on every
            # increment.
            if current_text.find(_TOOL_SEP, self._sep_search_start) == -1:
                self._sep_search_start = max(0, len(current_text) - len(_TOOL_SEP) + 1)
                if self.current_tool_name_sent and self._args_start != -1:
                    return self._stream_arguments(current_text, normal_text)
                if not self.current_tool_name_sent:
                    return StreamingParseResult(normal_text=normal_text)
            elif self.current_tool_name_sent:
                self._args_start = -1

            partial_match = re.search(
                pattern=r"<｜tool▁call▁begin｜>(.*)<｜tool▁sep｜>(.*?)(<｜tool▁call▁end｜>|$)",
                string=current_text,
                flags=re.DOTALL,
            )
            if partial_match:
                func_name = partial_match.group(1).strip()
                func_args_raw = partial_match.group(2).strip()
                is_tool_end = partial_match.group(3)

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
                    self._start_arguments(partial_match.start(2))
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

                    if is_complete_json(func_args_raw):
                        # Update the stored arguments
                        try:
                            parsed_args = json.loads(func_args_raw)
                            self.prev_tool_call_arr[self.current_tool_id]["arguments"] = parsed_args
                        except json.JSONDecodeError:
                            pass

                        # Find the end of the current tool call and remove only that part from buffer
                        if is_tool_end:
                            # Remove the completed tool call from buffer, keep any remaining content
                            self._buffer = current_text[partial_match.end(3) :]
                        else:
                            self._buffer = ""
                        self._pending_call_end = not bool(is_tool_end)

                        result = StreamingParseResult(normal_text=normal_text, calls=calls)
                        self.current_tool_id += 1
                        self._last_arguments = ""
                        self.current_tool_name_sent = False
                        self._sep_search_start = 0
                        return result
            else:
                self._sep_search_start = max(0, len(current_text) - len(_TOOL_SEP) + 1)

            return StreamingParseResult(normal_text=normal_text, calls=calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=normal_text + current_text)

    def _start_arguments(self, args_start: int) -> None:
        """Start reading the arguments of a call at buffer index `args_start`."""
        self._args_start = args_start
        self._args_sent_end = args_start
        self._sep_search_start = args_start
        self._end_search_start = args_start
        self._args_depth = 0
        self._args_in_string = False
        self._args_escaped = False

    def _stream_arguments(self, current_text: str, normal_text: str) -> StreamingParseResult:
        """Stream the arguments of the current call from the text that arrived.

        Gives the same result as matching the whole buffer: the arguments run
        from the separator to the first call end token (or the buffer end,
        less a partial call end token) and are sent without surrounding
        whitespace.
        """
        calls: list[ToolCallItem] = []
        end_idx = current_text.find(_TOOL_CALL_END, self._end_search_start)
        if end_idx == -1:
            self._end_search_start = max(
                self._args_start, len(current_text) - len(_TOOL_CALL_END) + 1
            )
            # A partial call end token is held back until the rest of it
            # arrives. The token holds one "<", so only the last one can start it.
            partial_end = current_text.rfind("<", self._end_search_start)
            if partial_end == -1 or not _TOOL_CALL_END.startswith(current_text[partial_end:]):
                partial_end = len(current_text)
            args_end = partial_end
        else:
            args_end = end_idx
            self._end_search_start = end_idx

        if not self._last_arguments:
            while self._args_sent_end < args_end and current_text[self._args_sent_end].isspace():
                self._args_sent_end += 1
        while args_end > self._args_sent_end and current_text[args_end - 1].isspace():
            args_end -= 1

        argument_diff = current_text[self._args_sent_end : args_end]
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
            self._args_sent_end = args_end
            self._track_json_nesting(argument_diff)

        # The arguments can only be complete JSON while no string or bracket
        # is open, so the full parse runs only then.
        if (
            not self._args_in_string
            and self._args_depth <= 0
            and is_complete_json(self._last_arguments)
        ):
            # Update the stored arguments
            try:
                parsed_args = json.loads(self._last_arguments)
                self.prev_tool_call_arr[self.current_tool_id]["arguments"] = parsed_args
            except json.JSONDecodeError:
                pass

            # Remove the completed tool call from buffer, keep any remaining content
            if end_idx != -1:
                self._buffer = current_text[end_idx + len(_TOOL_CALL_END) :]
            else:
                self._buffer = current_text[partial_end:]
            self._pending_call_end = end_idx == -1
            self._sep_search_start = 0
            self._args_start = -1

            result = StreamingParseResult(normal_text=normal_text, calls=calls)
            self.current_tool_id += 1
            self._last_arguments = ""
            self.current_tool_name_sent = False
            return result

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def _track_json_nesting(self, text: str) -> None:
        """Update the open-string and bracket-depth state of the sent arguments."""
        for char in text:
            if self._args_in_string:
                if self._args_escaped:
                    self._args_escaped = False
                elif char == "\\":
                    self._args_escaped = True
                elif char == '"':
                    self._args_in_string = False
            elif char == '"':
                self._args_in_string = True
            elif char in "{[":
                self._args_depth += 1
            elif char in "}]":
                self._args_depth -= 1

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin="<｜tool▁call▁begin｜>" + name + "<｜tool▁sep｜>",
            end="<｜tool▁call▁end｜>",
            trigger="<｜tool▁call▁begin｜>" + name + "<｜tool▁sep｜>",
        )
