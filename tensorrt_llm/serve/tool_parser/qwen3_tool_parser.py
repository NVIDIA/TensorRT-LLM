# Adapted from https://github.com/sgl-project/sglang/blob/083629c23564e1a64deaa052f1df5c5d914358d8/python/sglang/srt/function_call/qwen25_detector.py
import json
import re
from typing import List

from tensorrt_llm.logger import logger

from ..openai_protocol import ChatCompletionToolsParam as Tool
from .base_tool_parser import BaseToolParser
from .core_types import (StreamingParseResult, StructureInfo, ToolCallItem,
                         _GetInfoFunc)


class Qwen3ToolParser(BaseToolParser):
    r"""
    Detector for Qwen 2.5 and Qwen 3 model function call format.

    Format Structure:
    ```
    <tool_call>\n{"name":"func1", "arguments":{...}}\n</tool_call>\n<tool_call>\n{"name":"func2", "arguments":{...}}\n</tool_call>
    ```

    Key Components:
    - Tool Call Tags: `<tool_call>` and `</tool_call>` wrap each individual call
    - Function Call Object: JSON object with "name" and "arguments" fields

    Some Qwen3 chat templates (notably Qwen3.6 FP8 served with
    `--reasoning_parser qwen3_5 --tool_parser qwen3`) emit tool calls as bare
    JSON objects without the `<tool_call>...</tool_call>` wrapper once the
    reasoning parser strips the `</think>` block. Both `detect_and_parse` and
    `parse_streaming_increment` fall back to a bare-JSON parse in that case
    (see NVBug 6240584).

    Reference: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct?chat_template=default
    """

    # Streaming decision state for the bare-JSON fallback path.
    _STREAM_MODE_UNDECIDED = "undecided"
    _STREAM_MODE_WRAPPED = "wrapped"
    _STREAM_MODE_BARE_JSON = "bare_json"

    def __init__(self):
        """
        Initializes the detector with necessary state variables.
        """
        super().__init__()
        self.bot_token = "<tool_call>\n"  # nosec B105
        self.eot_token = "\n</tool_call>"  # nosec B105
        self.tool_call_separator = "\n"
        self._normal_text_buffer = ""  # Buffer for handling partial end tokens
        # Bare-JSON streaming fallback state (see NVBug 6240584).
        self._bare_json_buffer = ""
        self._stream_mode = self._STREAM_MODE_UNDECIDED

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Qwen 3 format tool call.

        Note: intentionally strict — this only checks for the `<tool_call>`
        wrapper. Existing callers rely on it as a wrapped-form gate. The
        bare-JSON fallback lives inside `detect_and_parse` /
        `parse_streaming_increment` and does not go through this method.
        """
        return self.bot_token in text

    def detect_and_parse(self, text: str,
                         tools: List[Tool]) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            # Some Qwen3 chat templates (e.g. Qwen3.6 FP8 with thinking enabled)
            # emit tool calls as bare JSON without a <tool_call> wrapper. Try
            # to recover those before dropping the text into normal_text. Use
            # an explicit type guard to avoid relying on AttributeError to
            # catch scalar JSON like "42" or null.
            try:
                parsed = json.loads(text.strip())
            except json.JSONDecodeError:
                return StreamingParseResult(normal_text=normal_text, calls=[])
            if isinstance(parsed, (dict, list)):
                calls = self.parse_base_json(parsed, tools)
                if calls:
                    return StreamingParseResult(normal_text="", calls=calls)
            return StreamingParseResult(normal_text=normal_text, calls=[])

        # Find all <tool_call>\n...\n</tool_call> blocks
        pattern = rf"{re.escape(self.bot_token)}(.*?){re.escape(self.eot_token)}"
        match_result_list = re.findall(pattern, text, re.DOTALL)
        calls = []
        for match_result in match_result_list:
            try:
                parsed_call = json.loads(match_result.strip())
                calls.extend(self.parse_base_json(parsed_call, tools))
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse JSON part: {match_result}, JSON parse error: {str(e)}"
                )
                continue
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(self, new_text: str,
                                  tools: List[Tool]) -> StreamingParseResult:
        r"""
        Streaming incremental parsing for Qwen 3 tool calls.

        Handles two shapes:
        - Wrapped: `<tool_call>\n{...}\n</tool_call>` — delegates to the base
          class implementation, which streams the name first and then the
          arguments incrementally. Applies the same partial-end-token
          scrubbing as before.
        - Bare JSON (NVBug 6240584): when the stream never contains
          `<tool_call>` but the accumulated buffer is a complete JSON object
          matching `{"name": ..., "arguments"/"parameters": ...}`, parse it
          via `parse_base_json` and emit the calls in the same
          name-first-then-arguments pattern used by the base class.

        The parser picks a mode once and stays in it for the remainder of the
        stream. Modes:
          - undecided: buffer input; peek to decide.
          - wrapped:   route everything through the base implementation.
          - bare_json: bare-JSON tool call already emitted; drop trailing text.
        """
        if self._stream_mode == self._STREAM_MODE_WRAPPED:
            return self._wrapped_streaming(new_text, tools)
        if self._stream_mode == self._STREAM_MODE_BARE_JSON:
            # Tool call already emitted; ignore any trailing text.
            return StreamingParseResult()

        # Undecided: buffer and decide.
        self._bare_json_buffer += new_text
        accumulated = self._bare_json_buffer

        if self.bot_token in accumulated:
            # Wrapped form: replay the accumulated buffer through the base
            # streaming path.
            self._stream_mode = self._STREAM_MODE_WRAPPED
            replay = accumulated
            self._bare_json_buffer = ""
            return self._wrapped_streaming(replay, tools)

        if self._ends_with_partial_token(accumulated, self.bot_token):
            # Could still resolve into a wrapped form — keep buffering.
            return StreamingParseResult()

        stripped = accumulated.lstrip()
        if not stripped:
            # Only whitespace so far — keep buffering.
            return StreamingParseResult()

        if stripped[0] not in "{[":
            # Not JSON-like at all — flush as normal text via wrapped path.
            self._stream_mode = self._STREAM_MODE_WRAPPED
            replay = accumulated
            self._bare_json_buffer = ""
            return self._wrapped_streaming(replay, tools)

        # Leading `{` or `[` — could be a bare JSON tool call, or partial.
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            # Not yet complete — buffer more.
            return StreamingParseResult()

        if isinstance(parsed, (dict, list)):
            calls = self.parse_base_json(parsed, tools)
            if calls:
                streaming_calls: List[ToolCallItem] = []
                for i, call_item in enumerate(calls):
                    # Emit name first with empty parameters, then the full
                    # arguments — mirrors the base class streaming pattern.
                    streaming_calls.append(
                        ToolCallItem(
                            tool_index=i,
                            name=call_item.name,
                            parameters="",
                        ))
                    if call_item.parameters:
                        streaming_calls.append(
                            ToolCallItem(
                                tool_index=i,
                                parameters=call_item.parameters,
                            ))
                self._stream_mode = self._STREAM_MODE_BARE_JSON
                self._bare_json_buffer = ""
                # Best-effort bookkeeping for callers that inspect these.
                self.current_tool_id = max(0, len(calls) - 1)
                self.current_tool_name_sent = True
                return StreamingParseResult(normal_text="",
                                            calls=streaming_calls)

        # Parsed as JSON but not a tool call — flush as normal text via
        # wrapped path.
        self._stream_mode = self._STREAM_MODE_WRAPPED
        replay = accumulated
        self._bare_json_buffer = ""
        return self._wrapped_streaming(replay, tools)

    def _wrapped_streaming(self, new_text: str,
                           tools: List[Tool]) -> StreamingParseResult:
        """Wrapped-form streaming: base implementation + partial-end-token scrubbing."""
        result = super().parse_streaming_increment(new_text, tools)

        # Handle partial end tokens that are streamed character by character.
        if result.normal_text:
            self._normal_text_buffer += result.normal_text

            # Check if buffer contains complete end token (without leading newline)
            end_token_without_newline = self.eot_token[1:]  # "</tool_call>"
            if end_token_without_newline in self._normal_text_buffer:
                cleaned_text = self._normal_text_buffer.replace(
                    end_token_without_newline, "")
                self._normal_text_buffer = ""
                result.normal_text = cleaned_text
            else:
                # Check if buffer might contain partial end token at the end
                partial_match_len = self._ends_with_partial_token(
                    self._normal_text_buffer, end_token_without_newline)

                if partial_match_len:
                    # Keep potential partial match in buffer, return the rest
                    result.normal_text = self._normal_text_buffer[:
                                                                  -partial_match_len]
                    self._normal_text_buffer = self._normal_text_buffer[
                        -partial_match_len:]
                else:
                    # No partial match, return all buffered text
                    result.normal_text = self._normal_text_buffer
                    self._normal_text_buffer = ""

        return result

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin='<tool_call>\n{"name":"' + name + '", "arguments":',
            end="}\n</tool_call>",
            trigger="<tool_call>",
        )
