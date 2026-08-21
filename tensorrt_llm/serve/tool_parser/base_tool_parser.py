# Adapted from https://github.com/sgl-project/sglang/blob/083629c23564e1a64deaa052f1df5c5d914358d8/python/sglang/srt/function_call/base_format_detector.py
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from partial_json_parser.core.exceptions import MalformedJSON
from partial_json_parser.core.options import Allow

from tensorrt_llm.logger import logger

from ..openai_protocol import ChatCompletionToolsParam as Tool
from .core_types import StreamingParseResult, ToolCallItem, _GetInfoFunc
from .utils import find_common_prefix, is_complete_json, partial_json_loads


class BaseToolParser(ABC):
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    needs_raw_special_tokens: bool = False

    def __init__(self):
        # Streaming state management
        # Buffer for accumulating incomplete patterns that arrive across multiple streaming chunks
        self._buffer = ""
        # Stores complete tool call info (name and arguments) for each tool being parsed.
        # Used by serving layer for completion handling when streaming ends.
        # Format: [{"name": str, "arguments": dict}, ...]
        self.prev_tool_call_arr: List[Dict] = []
        # Index of currently streaming tool call. Starts at -1 (no active tool),
        # increments as each tool completes. Tracks which tool's arguments are streaming.
        self.current_tool_id: int = -1
        # Flag for whether current tool's name has been sent to client.
        # Tool names sent first with empty parameters, then arguments stream incrementally.
        self.current_tool_name_sent: bool = False
        # Tracks raw JSON string content streamed to client for each tool's arguments.
        # Critical for serving layer to calculate remaining content when streaming ends.
        # Each index corresponds to a tool_id. Example: ['{"location": "San Francisco"', '{"temp": 72']
        self.streamed_args_for_tool: List[str] = []

        # Token configuration (override in subclasses)
        self.bot_token = ""  # nosec B105
        self.eot_token = ""  # nosec B105
        self.tool_call_separator = ", "

    def _get_tool_indices(self, tools: List[Tool]) -> Dict[str, int]:
        """
        Get a mapping of tool names to their indices in the tools list.

        This utility method creates a dictionary mapping function names to their
        indices in the tools list, which is commonly needed for tool validation
        and ToolCallItem creation.

        Args:
            tools: List of available tools

        Returns:
            Dictionary mapping tool names to their indices
        """
        return {
            tool.function.name: i
            for i, tool in enumerate(tools) if tool.function.name
        }

    def parse_base_json(self, action: Any,
                        tools: List[Tool]) -> List[ToolCallItem]:
        tool_indices = self._get_tool_indices(tools)
        if not isinstance(action, list):
            action = [action]

        results = []
        for act in action:
            name = act.get("name")
            if name:
                if name not in tool_indices:
                    logger.warning(
                        f"Model attempted to call undefined function: {name}")
                results.append(
                    ToolCallItem(
                        tool_index=
                        -1,  # Caller should update this based on the actual tools array called
                        name=name,
                        parameters=json.dumps(
                            act.get("parameters") or act.get("arguments", {}),
                            ensure_ascii=False,
                        ),
                    ))

        return results

    @abstractmethod
    def detect_and_parse(self, text: str,
                         tools: List[Tool]) -> StreamingParseResult:
        """
        Parses the text in one go. Returns success=True if the format matches, otherwise False.
        Note that leftover_text here represents "content that this parser will not consume further".
        """
        action = json.loads(text)
        return StreamingParseResult(calls=self.parse_base_json(action, tools))

    def _ends_with_partial_token(self, buffer: str, bot_token: str) -> int:
        """
        Check if buffer ends with a partial bot_token.
        Return the length of the partial bot_token.

        For some format, the bot_token is not a token in model's vocabulary, such as
        `[TOOL_CALLS] [` in Mistral.
        """
        for i in range(1, min(len(buffer) + 1, len(bot_token))):
            if bot_token.startswith(buffer[-i:]):
                return i
        return 0

    def _starts_with_leftover_eot_token(self, buffer: str) -> bool:
        r"""Check if the buffer opens with the eot_token of a finished call.

        Completing a tool call leaves everything the parser did not consume in
        the buffer, starting with that call's eot_token. When the eot_token
        itself begins with the tool_call_separator, as in Qwen3 where calls are
        separated by "\n" and closed by "\n</tool_call>", that leftover looks
        exactly like the separator that introduces the next tool call.
        """
        return bool(self.eot_token) and buffer.startswith(self.eot_token)

    def _may_begin_tool_call(self, text: str) -> bool:
        """Check whether text could be the opening of a tool call.

        Tells a tool_call_separator that introduces another call apart from one
        that merely precedes ordinary prose. A call opens either with the
        bot_token or, in the formats this base class streams, with bare JSON.
        Text too short to judge counts as a maybe, so the buffer keeps growing
        until the answer is certain.
        """
        if not text:
            return True
        if self.bot_token and (text.startswith(self.bot_token)
                               or self.bot_token.startswith(text)):
            return True
        return text[0] in "{["

    def parse_streaming_increment(self, new_text: str,
                                  tools: List[Tool]) -> StreamingParseResult:
        """
        Streaming incremental parsing with tool validation.

        This base implementation works best with formats where:
        1. bot_token is followed immediately by JSON (e.g., bot_token + JSON_array)
        2. JSON can be parsed incrementally using partial_json_loads
        3. Multiple tool calls are separated by "; " or ", "

        Examples of incompatible formats (need custom implementation, may reuse some logic from this class):
        - Each tool call is wrapped in a separate block: See Qwen25Detector
        - Multiple separate blocks: [TOOL_CALLS] [...] \n [TOOL_CALLS] [...]
        - Tool call is Pythonic style

        For incompatible formats, detectors should override this method with custom logic.
        """
        pending = self._buffer + new_text
        name_sent = self.current_tool_name_sent
        result = self._parse_increment_once(new_text, tools)

        # A pass stops at the end of one tool call and leaves the rest of the
        # increment in the buffer for the increment after it. Nothing drains
        # the buffer once the stream ends, so whatever arrived in the same
        # chunk as the closing markup -- trailing content, a further tool call,
        # or the arguments of the call that just opened -- would never be
        # emitted. Keep parsing while a pass still moves the parser forward.
        #
        # Forward means the buffer shrank, or the pass sent a tool name and so
        # the next one will stream that call's arguments. A pass that does
        # neither has nothing left to give, and repeating it would re-parse the
        # same bytes on every token of the stream.
        while self._buffer and (self._buffer != pending or
                                (self.current_tool_name_sent
                                 and not name_sent)):
            pending = self._buffer
            name_sent = self.current_tool_name_sent
            step = self._parse_increment_once("", tools)
            result.normal_text += step.normal_text
            result.calls.extend(step.calls)

        return result

    def _parse_increment_once(self, new_text: str,
                              tools: List[Tool]) -> StreamingParseResult:
        """Run a single parsing pass over the buffer plus the new text."""
        # Append new text to buffer
        self._buffer += new_text

        # Parsing a tool call stops at its closing markup, leaving the
        # eot_token at the head of the buffer. Drop it before looking at what
        # follows: it is markup rather than content, and while it is still
        # there the checks below read it as part of the next tool call.
        if self.current_tool_id > 0 and self.eot_token:
            if self._starts_with_leftover_eot_token(self._buffer):
                self._buffer = self._buffer[len(self.eot_token):]
            elif self.eot_token.startswith(self._buffer):
                # The end token is still arriving one token at a time. Hold it
                # so its opening bytes are not mistaken for content.
                return StreamingParseResult()

        current_text = self._buffer

        # The current_text has tool_call if it is the start of a new tool call sequence
        # or it is the start of a new tool call after a tool call separator, when there is a previous tool call.
        # The separator only introduces a call when a call actually follows it;
        # a response that resumes with prose on a new line starts the same way,
        # and reading that as a call leaves it stuck in the tool call branch
        # below, where prose never parses as JSON.
        starts_next_tool_call = (
            self.current_tool_id > 0
            and current_text.startswith(self.tool_call_separator)
            and self._may_begin_tool_call(
                current_text[len(self.tool_call_separator):]))

        if not (self.has_tool_call(current_text) or starts_next_tool_call):
            # Only clear buffer if we're sure no tool call is starting
            if not self._ends_with_partial_token(self._buffer, self.bot_token):
                normal_text = self._buffer
                self._buffer = ""
                if self.eot_token in normal_text:
                    normal_text = normal_text.replace(self.eot_token, "")
                return StreamingParseResult(normal_text=normal_text)
            else:
                # Might be partial bot_token, keep buffering
                return StreamingParseResult()

        # Build tool indices if not already built
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        flags = Allow.ALL if self.current_tool_name_sent else Allow.ALL & ~Allow.STR

        try:
            try:
                tool_call_pos = current_text.find(self.bot_token)
                if tool_call_pos != -1:
                    start_idx = tool_call_pos + len(self.bot_token)
                elif starts_next_tool_call:
                    start_idx = len(self.tool_call_separator)
                else:
                    start_idx = 0

                if start_idx >= len(current_text):
                    return StreamingParseResult()

                (obj, end_idx) = partial_json_loads(current_text[start_idx:],
                                                    flags)

                is_current_complete = is_complete_json(
                    current_text[start_idx:start_idx + end_idx])

                # Handle parameters/arguments consistency
                # NOTE: we assume here that the obj is always partial of a single tool call
                if "parameters" in obj:
                    assert ("arguments" not in obj
                            ), "model generated both parameters and arguments"
                    obj["arguments"] = obj["parameters"]

                current_tool_call = obj

            except MalformedJSON:
                return StreamingParseResult()

            if not current_tool_call:
                return StreamingParseResult()

            # Case 1: Handle tool name streaming
            # This happens when we encounter a tool but haven't sent its name yet
            if not self.current_tool_name_sent:
                function_name = current_tool_call.get("name")

                if function_name:
                    # If this is a new tool (current_tool_id was -1), initialize it
                    if self.current_tool_id == -1:
                        self.current_tool_id = 0
                        self.streamed_args_for_tool.append("")
                    # If this is a subsequent tool, ensure streamed_args_for_tool is large enough
                    elif self.current_tool_id >= len(
                            self.streamed_args_for_tool):
                        while len(self.streamed_args_for_tool
                                  ) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")

                    # Send the tool name with empty parameters
                    res = StreamingParseResult(calls=[
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=function_name,
                            parameters="",
                        )
                    ], )
                    self.current_tool_name_sent = True
                else:
                    res = StreamingParseResult()

            # Case 2: Handle streaming arguments
            # This happens when we've already sent the tool name and now need to stream arguments incrementally
            else:
                cur_arguments = current_tool_call.get("arguments")
                res = StreamingParseResult()
                argument_diff = None
                # Save the ID of the tool that's completing
                completing_tool_id = self.current_tool_id

                if cur_arguments:
                    # Calculate how much of the arguments we've already streamed
                    sent = len(
                        self.streamed_args_for_tool[self.current_tool_id])
                    cur_args_json = json.dumps(cur_arguments)
                    prev_arguments = None
                    if self.current_tool_id < len(self.prev_tool_call_arr):
                        prev_arguments = self.prev_tool_call_arr[
                            self.current_tool_id].get("arguments")

                    # If the current tool's JSON is complete, send all remaining arguments
                    if is_current_complete:
                        argument_diff = cur_args_json[sent:]

                    # If the tool is still being parsed, send incremental changes
                    elif prev_arguments:
                        prev_args_json = json.dumps(prev_arguments)
                        if cur_args_json != prev_args_json:
                            prefix = find_common_prefix(prev_args_json,
                                                        cur_args_json)
                            argument_diff = prefix[sent:]

                # Close the call out whenever its JSON is complete, not only
                # when it carried arguments. A call invoked with none still
                # ends here, and leaving it open keeps its markup and
                # everything after it stuck in the buffer for the rest of the
                # stream.
                if is_current_complete:
                    # Only remove the processed portion, keep unprocessed content
                    self._buffer = current_text[start_idx + end_idx:]

                    if self.current_tool_id < len(self.prev_tool_call_arr):
                        self.prev_tool_call_arr[self.current_tool_id].clear()
                    self.current_tool_name_sent = False
                    self.streamed_args_for_tool[self.current_tool_id] = ""
                    self.current_tool_id += 1

                # Send the argument diff if there's something new
                if argument_diff is not None:
                    # Use the correct tool_index: completing_tool_id for completed tools, current_tool_id for ongoing
                    tool_index_to_use = (completing_tool_id
                                         if is_current_complete else
                                         self.current_tool_id)
                    res = StreamingParseResult(calls=[
                        ToolCallItem(
                            tool_index=tool_index_to_use,
                            parameters=argument_diff,
                        )
                    ], )
                    if not is_current_complete:
                        self.streamed_args_for_tool[
                            self.current_tool_id] += argument_diff

            # Update prev_tool_call_arr with current state
            if self.current_tool_id >= 0:
                # Ensure prev_tool_call_arr is large enough
                while len(self.prev_tool_call_arr) <= self.current_tool_id:
                    self.prev_tool_call_arr.append({})
                self.prev_tool_call_arr[
                    self.current_tool_id] = current_tool_call

            return res

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult()

    @abstractmethod
    def has_tool_call(self, text: str) -> bool:
        """
        Check if the given text contains function call markers specific to this format.
        """
        raise NotImplementedError()

    def supports_structural_tag(self) -> bool:
        """Return True if this detector supports structural tag format."""
        return True

    @abstractmethod
    def structure_info(self) -> _GetInfoFunc:
        """
        Return a function that creates StructureInfo for constrained generation.

        The returned function takes a tool name and returns a StructureInfo object
        containing the begin/end patterns and trigger tokens needed for constrained
        generation of function calls in this format.

        Returns:
            A function that takes a tool name (str) and returns StructureInfo
        """
        raise NotImplementedError()
