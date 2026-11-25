# Adapted from https://github.com/sgl-project/sglang/blob/0071fe9c407ad59f2803cc319e1bcaa3ac2021f1/python/sglang/srt/function_call/deepseekv32_detector.py
import json
import re
from typing import List, Optional, Tuple

from tensorrt_llm.logger import logger

from ..openai_protocol import ChatCompletionToolsParam as Tool
from .base_tool_parser import BaseToolParser
from .core_types import StreamingParseResult, StructureInfo, ToolCallItem, _GetInfoFunc


class DeepSeekV32Parser(BaseToolParser):
    """Tool parser for DeepSeek V3.2 model function call format.

    The DeepSeek V3.2 format uses XML-like DSML tags to delimit function calls.
    Supports two parameter formats:

    Format 1 - XML Parameter Tags:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="function_name">
        <｜DSML｜parameter name="param_name" string="true">value</｜DSML｜parameter>
        ...
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Format 2 - Direct JSON:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="function_name">
        {
            "param_name": "value"
        }
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Examples:
    ```
    <｜DSML｜function_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜function_calls>

    <｜DSML｜function_calls>
        <｜DSML｜invoke name="get_favorite_tourist_spot">
        { "city": "San Francisco" }
    </｜DSML｜invoke>
    </｜DSML｜function_calls>
    ```

    Key Components:
    - Tool Calls Section: Wrapped between `<｜DSML｜function_calls>` and `</｜DSML｜function_calls>`
    - Individual Tool Call: Wrapped between `<｜DSML｜invoke name="...">` and `</｜DSML｜invoke>`
    - Parameters: Either XML tags or direct JSON format
    - Supports multiple tool calls

    Reference: DeepSeek V3.2 format specification
    """

    needs_raw_special_tokens = True

    _eos_token = "<｜end▁of▁sentence｜>"  # nosec B105

    # The encoders write a blank line between a reply's content and its tool-call
    # block: it belongs to the format, not to the content.
    _CALLS_SEPARATOR = "\n\n"

    # Tag names that follow the ``｜DSML｜`` marker; subclasses override these
    # to describe their DSML dialect.
    _DSML = "｜DSML｜"
    _CALLS_TAG = "function_calls"
    _INVOKE_TAG = "invoke"
    _PARAMETER_TAG = "parameter"

    def __init__(self):
        super().__init__()
        calls = self._DSML + self._CALLS_TAG
        invoke = self._DSML + self._INVOKE_TAG
        parameter = self._DSML + self._PARAMETER_TAG
        self.bot_token = f"<{calls}>"  # nosec B105
        self.eot_token = f"</{calls}>"  # nosec B105
        self.invoke_begin_token = f"<{invoke}"  # nosec B105
        self.invoke_end_token = f"</{invoke}>"  # nosec B105
        self.parameter_begin_token = f"<{parameter}"  # nosec B105
        self.parameter_end_token = f"</{parameter}>"  # nosec B105
        # Invoke header up to the function name, which is arbitrary text.
        self._invoke_header_prefix = f'<{invoke} name="'  # nosec B105
        self._invoke_begin_template = f'<{invoke} name="{{name}}">'  # nosec B105
        self.invoke_begin_regex = rf'<{re.escape(invoke)}\s+name="([^"]+)"\s*>'
        self._invoke_header_re = re.compile(self.invoke_begin_regex)
        self._parameter_header_re = re.compile(
            rf'\s*<{re.escape(parameter)}\s+name="([^"]+)"\s+string="([^"]+)"\s*>'
        )
        self._last_arguments = ""
        self.current_tool_id = -1
        # Between a block's opening and closing tags, the newlines that separate
        # the invokes and precede the closing tag are the format's layout (the
        # checkpoint's encoder writes them), not content.
        self._in_calls_block = False
        # Set while an increment re-reads what its last invoke left behind.
        self._reading_rest = False
        # Set by finish(): the buffered text is all the stream will produce.
        self._stream_final = False

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek v32 format tool call."""
        return self.bot_token in text

    @staticmethod
    def _follower(
        text: str, pos: int, followers: Tuple[str, ...], at_end_ok: bool
    ) -> Optional[bool]:
        """Whether ``text[pos:]``, after whitespace, starts with one of ``followers``.

        Returns None when the remaining text is only a prefix of a follower, so
        a streaming caller has to wait for more output to decide.
        """
        rest = text[pos:].lstrip()
        if not rest:
            return True if at_end_ok else None
        if rest.startswith(followers):
            return True
        if any(f.startswith(rest) for f in followers):
            return None
        return False

    @staticmethod
    def _convert_parameter(param_type: str, value: str):
        if param_type == "true":  # string type
            # Taken as written: the encoder renders a string value verbatim
            # between the tags, so its whitespace (a file's last newline, a
            # line's indentation) is part of the value.
            return value
        try:
            return json.loads(value.strip())
        except (json.JSONDecodeError, ValueError):
            return value.strip()

    def _parse_parameters_from_xml(self, invoke_content: str) -> dict:
        """Parse parameters from either XML-like format or JSON format to dict.

        Supports two formats:
        1. XML parameter tags: <｜DSML｜parameter name="..." string="...">value</｜DSML｜parameter>
        2. Direct JSON: { "key": "value" }

        A parameter closing tag only ends the value when it is followed by the
        next parameter or by the end of the invoke body, so values may contain
        literal DSML tags. Without such a tag the first closing tag is used.
        """
        # First, try to parse as direct JSON (new format)
        invoke_content_stripped = invoke_content.strip()

        if invoke_content_stripped.startswith("{") and invoke_content_stripped.endswith("}"):
            try:
                parameters = json.loads(invoke_content_stripped)
                if isinstance(parameters, dict):
                    return parameters
            except (json.JSONDecodeError, ValueError):
                # If JSON parsing fails, fall through to XML parsing
                pass

        # Fall back to XML parameter tag parsing (original format)
        parameters = {}
        end_token = self.parameter_end_token
        followers = (self.parameter_begin_token,)
        pos = 0
        while True:
            header = self._parameter_header_re.match(invoke_content, pos)
            if not header:
                break
            param_name, param_type = header.group(1), header.group(2)
            value_start = header.end()
            value_end = first_end = -1
            search = value_start
            while True:
                idx = invoke_content.find(end_token, search)
                if idx == -1:
                    break
                if first_end == -1:
                    first_end = idx
                search = idx + len(end_token)
                if self._follower(invoke_content, search, followers, at_end_ok=True):
                    value_end = idx
                    break
            if value_end == -1:
                value_end = first_end
            if value_end == -1:
                break
            parameters[param_name] = self._convert_parameter(
                param_type, invoke_content[value_start:value_end]
            )
            pos = value_end + len(end_token)
        return parameters

    def _find_invoke(self, text: str, start: int, final: bool) -> Optional[tuple]:
        """Find the next invoke block at or after ``start``.

        An invoke closing tag ends the block only when it is followed by another
        invoke, the closing calls tag, the end of sentence token, a new calls
        block or, when ``final`` is set, the end of the text.

        Returns None when there is no invoke header, otherwise
        ``(name, content, end, complete)``. ``complete`` is False while the
        closing tag is missing or not yet confirmed by the text after it. When
        ``final`` is set and no closing tag is confirmed, the last one is used.
        """
        header = self._invoke_header_re.search(text, start)
        if not header:
            return None
        name = header.group(1).strip()
        end_token = self.invoke_end_token
        followers = (self.invoke_begin_token, self.eot_token, self._eos_token, self.bot_token)
        last_end = -1
        search = header.end()
        while True:
            idx = text.find(end_token, search)
            if idx == -1:
                break
            last_end = idx
            search = idx + len(end_token)
            verdict = self._follower(text, search, followers, at_end_ok=final)
            if verdict:
                return name, text[header.end() : idx], search, True
            if verdict is None and not final:
                return name, text[header.end() : idx], search, False
        if final and last_end != -1:
            return name, text[header.end() : last_end], last_end + len(end_token), True
        return name, text[header.end() :], len(text), False

    def _find_calls_end(self, text: str, start: int) -> int:
        """Index of the closing calls tag that ends the block, or len(text) if none."""
        followers = (self._eos_token, self.bot_token)
        search = start
        while True:
            idx = text.find(self.eot_token, search)
            if idx == -1:
                return len(text)
            search = idx + len(self.eot_token)
            if self._follower(text, search, followers, at_end_ok=True):
                return idx

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        return self._parse_text(text, tools, final=True)

    def parse_partial(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Inspect unfinished output without updating the streaming state."""
        return self._parse_text(text, tools, final=False)

    def _parse_text(self, text: str, tools: List[Tool], final: bool) -> StreamingParseResult:
        idx = text.find(self.bot_token)
        if idx == -1:
            return StreamingParseResult(normal_text=text.replace(self._eos_token, ""), calls=[])
        normal_text = text[:idx].replace(self._eos_token, "").removesuffix(self._CALLS_SEPARATOR)

        calls = []
        try:
            body_start = idx + len(self.bot_token)
            body = text[body_start : self._find_calls_end(text, body_start)]
            pos = 0
            while True:
                found = self._find_invoke(body, pos, final=final)
                if not found:
                    break
                func_name, invoke_content, pos, complete = found
                if not complete:
                    break
                # Parse parameters from XML format
                func_args = self._parse_parameters_from_xml(invoke_content)
                # construct match_result for parse_base_json
                match_result = {"name": func_name, "parameters": func_args}
                calls.extend(self.parse_base_json(match_result, tools))

            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text.replace(self._eos_token, ""))

    def _separator_prefix_at_end(self, text: str) -> str:
        """The tail of ``text`` that could still begin the blank line before a calls block."""
        for size in range(len(self._CALLS_SEPARATOR), 0, -1):
            if text.endswith(self._CALLS_SEPARATOR[:size]):
                return text[-size:]
        return ""

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """Flush the stream: the text buffered at the end can no longer grow.

        An invoke whose closing tag is the last generated text is emitted. Outside
        a calls block, what the stream ended while holding back (a trailing blank
        line no tool-call block followed, or a tail that never completed into a
        delimiter) is emitted as content. Inside an unfinished calls block no
        content is emitted.
        """
        if not self._buffer:
            return StreamingParseResult()
        self._stream_final = True
        try:
            return self.parse_streaming_increment("", tools)
        finally:
            self._stream_final = False

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """Streaming incremental parsing tool calls for DeepSeekV32 format.

        Supports multiple consecutive invoke blocks.
        """
        self._buffer += new_text
        current_text = self._buffer

        start_tokens = [self.bot_token, self._invoke_header_prefix]
        start_indices = [idx for idx in map(current_text.find, start_tokens) if idx != -1]
        has_tool_call = bool(start_indices)

        # Hold the buffer back only while its tail could still complete into one of
        # the DSML delimiters.
        partial_tokens = [
            self.bot_token,
            self._invoke_header_prefix,
            self.eot_token,
            self.invoke_end_token,
            self._eos_token,
        ]
        ends_with_partial_token = not self._stream_final and any(
            self._ends_with_partial_token(current_text, token) for token in partial_tokens
        )

        if self._in_calls_block and not has_tool_call:
            closing = current_text.find(self.eot_token)
            layout = current_text if closing == -1 else current_text[:closing]
            if not layout.strip():
                if closing == -1:
                    # Still layout until the next invoke or the closing tag.
                    return StreamingParseResult()
                self._in_calls_block = False
                current_text = current_text[closing + len(self.eot_token) :]
                self._buffer = current_text
                ends_with_partial_token = not self._stream_final and any(
                    self._ends_with_partial_token(current_text, token) for token in partial_tokens
                )

        if not has_tool_call and not ends_with_partial_token:
            normal_text = current_text
            # Trailing newlines may begin the blank line before a tool-call block,
            # which is not content: hold them until the next increment, or
            # finish(), shows what follows.
            held = (
                ""
                if self._in_calls_block or self._stream_final
                else self._separator_prefix_at_end(normal_text)
            )
            self._buffer = held
            normal_text = normal_text[: len(normal_text) - len(held)]
            for e_token in [self.eot_token, self.invoke_end_token, self._eos_token]:
                normal_text = normal_text.replace(e_token, "")
            return StreamingParseResult(normal_text=normal_text)

        # Text before the earliest start token is content, so stream it and keep the
        # buffer from that token onwards.
        prefix_len = min(start_indices, default=0)
        normal_text = current_text[:prefix_len]
        if normal_text:
            current_text = current_text[prefix_len:]
            self._buffer = current_text
        if has_tool_call:
            if self._in_calls_block and not normal_text.strip():
                # The separator before the next invoke.
                normal_text = ""
            elif not self._in_calls_block and current_text.startswith(self.bot_token):
                normal_text = normal_text.removesuffix(self._CALLS_SEPARATOR)
            self._in_calls_block = True

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        all_calls: list[ToolCallItem] = []
        invokes_exhausted = False
        try:
            # Loop to handle multiple consecutive invoke blocks
            while True:
                found = self._find_invoke(current_text, 0, final=self._stream_final)
                if not found:
                    invokes_exhausted = True
                    break
                func_name, invoke_content, invoke_end, is_tool_end = found

                # Initialize state if this is the first tool call
                if self.current_tool_id == -1:
                    self.current_tool_id = 0
                    self.prev_tool_call_arr = []
                    self.streamed_args_for_tool = [""]

                # Don't pre-allocate arrays until we actually complete a tool call
                # This prevents _check_for_unstreamed_tool_args from sending incomplete calls

                # Parse current parameters from XML/JSON
                current_params = self._parse_parameters_from_xml(invoke_content)
                current_args_json = json.dumps(current_params, ensure_ascii=False)

                # Check if tool call is complete (has closing tag)
                if is_tool_end:
                    # Only emit the tool call when it's complete (confirmed invoke end tag)
                    # This ensures each function returns at most once
                    calls_for_this_invoke: list[ToolCallItem] = []

                    # Note: invoke_content can be empty for functions with no parameters
                    # This is valid and should NOT be skipped

                    # Send tool name
                    calls_for_this_invoke.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=func_name,
                            parameters="",
                        )
                    )

                    # Send parameters as complete JSON
                    # Always send parameters, even if empty, to maintain consistency
                    calls_for_this_invoke.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters=current_args_json,
                        )
                    )

                    # Ensure arrays are large enough for current tool
                    while len(self.prev_tool_call_arr) <= self.current_tool_id:
                        self.prev_tool_call_arr.append({})
                    while len(self.streamed_args_for_tool) <= self.current_tool_id:
                        self.streamed_args_for_tool.append("")

                    # Update the stored arguments
                    self.prev_tool_call_arr[self.current_tool_id] = {
                        "name": func_name,
                        "arguments": current_params,
                    }
                    self.streamed_args_for_tool[self.current_tool_id] = current_args_json

                    # Remove the completed tool call from buffer
                    self._buffer = current_text[invoke_end:]
                    current_text = self._buffer  # Update for next iteration

                    # Add calls for this invoke to all_calls
                    all_calls.extend(calls_for_this_invoke)

                    # Move to next tool call
                    self.current_tool_id += 1
                    self._last_arguments = ""
                    self.current_tool_name_sent = False

                    # Don't pre-allocate arrays for the next tool
                    # Only allocate when we actually complete a tool call
                    # This prevents _check_for_unstreamed_tool_args from sending incomplete calls

                    # Continue loop to check for more invoke blocks
                    continue
                else:
                    # Tool call not complete yet, don't return anything
                    # Wait for more chunks until the invoke end tag is confirmed
                    break

            # The rest of this increment may close the block and carry text after
            # it. Read it now: a later increment may never come, and nothing
            # flushes this parser when the stream ends.
            if (
                invokes_exhausted
                and self._in_calls_block
                and not self._reading_rest
                and self.eot_token in self._buffer
                and not any(token in self._buffer for token in start_tokens)
            ):
                self._reading_rest = True
                try:
                    rest = self.parse_streaming_increment("", tools)
                finally:
                    self._reading_rest = False
                normal_text += rest.normal_text
                all_calls.extend(rest.calls)
            return StreamingParseResult(normal_text=normal_text, calls=all_calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=normal_text + current_text)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=self._invoke_begin_template.format(name=name),
            end=self.invoke_end_token,
            trigger=self._invoke_begin_template.format(name=name),
        )
