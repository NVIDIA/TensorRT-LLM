# Adapted from https://github.com/sgl-project/sglang/blob/0071fe9c407ad59f2803cc319e1bcaa3ac2021f1/python/sglang/srt/function_call/deepseekv32_detector.py
import json
import re
from typing import List

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

    # Invoke header up to the function name, which is arbitrary text.
    _INVOKE_HEADER_PREFIX = '<｜DSML｜invoke name="'  # nosec B105

    def __init__(self):
        super().__init__()
        self.bot_token = "<｜DSML｜function_calls>"  # nosec B105
        self.eot_token = "</｜DSML｜function_calls>"  # nosec B105
        self.invoke_begin_regex = r'<｜DSML｜invoke\s+name="([^"]+)"\s*>'
        self.invoke_end_token = "</｜DSML｜invoke>"  # nosec B105
        self.parameter_regex = (
            r'<｜DSML｜parameter\s+name="([^"]+)"\s+string="([^"]+)"\s*>(.*?)</｜DSML｜parameter>'
        )
        self._last_arguments = ""
        self.current_tool_id = -1

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a deepseek v32 format tool call."""
        return self.bot_token in text

    def _parse_parameters_from_xml(self, invoke_content: str) -> dict:
        """Parse parameters from either XML-like format or JSON format to dict.

        Supports two formats:
        1. XML parameter tags: <｜DSML｜parameter name="..." string="...">value</｜DSML｜parameter>
        2. Direct JSON: { "key": "value" }
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
        param_matches = re.findall(self.parameter_regex, invoke_content, re.DOTALL)
        for param_name, param_type, param_value in param_matches:
            # Convert value based on type
            if param_type == "true":  # string type
                parameters[param_name] = param_value.strip()
            else:
                # Try to parse as JSON for other types
                try:
                    parameters[param_name] = json.loads(param_value.strip())
                except (json.JSONDecodeError, ValueError):
                    parameters[param_name] = param_value.strip()
        return parameters

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-time parsing: Detects and parses tool calls in the provided text.

        :param text: The complete text to parse.
        :param tools: List of available tools.
        :return: ParseResult indicating success or failure, consumed text, leftover text, and parsed calls.
        """
        if self._eos_token in text:
            text = text.replace(self._eos_token, "")
        idx = text.find(self.bot_token)
        normal_text = text[:idx].strip() if idx != -1 else text
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=normal_text, calls=[])

        calls = []
        try:
            # Extract content between function_calls tags
            function_calls_match = re.search(
                re.escape(self.bot_token) + r"(.*?)" + re.escape(self.eot_token),
                text,
                re.DOTALL,
            )
            if not function_calls_match:
                return StreamingParseResult(normal_text=normal_text, calls=[])

            function_calls_content = function_calls_match.group(1)

            # Find all invoke blocks
            invoke_pattern = r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)</｜DSML｜invoke>'
            invoke_matches = re.findall(invoke_pattern, function_calls_content, re.DOTALL)

            for func_name, invoke_content in invoke_matches:
                # Parse parameters from XML format
                func_args = self._parse_parameters_from_xml(invoke_content)
                # construct match_result for parse_base_json
                match_result = {"name": func_name, "parameters": func_args}
                calls.extend(self.parse_base_json(match_result, tools))

            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:
            logger.error(f"Error in detect_and_parse: {e}")
            # return the normal text if parsing fails
            return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """Streaming incremental parsing tool calls for DeepSeekV32 format.

        Supports multiple consecutive invoke blocks.
        """
        self._buffer += new_text
        current_text = self._buffer

        start_tokens = [self.bot_token, self._INVOKE_HEADER_PREFIX]
        start_indices = [idx for idx in map(current_text.find, start_tokens) if idx != -1]
        has_tool_call = bool(start_indices)

        # Hold the buffer back only while its tail could still complete into one of
        # the DSML delimiters.
        partial_tokens = [
            self.bot_token,
            self._INVOKE_HEADER_PREFIX,
            self.eot_token,
            self.invoke_end_token,
            self._eos_token,
        ]
        ends_with_partial_token = any(
            self._ends_with_partial_token(current_text, token) for token in partial_tokens
        )

        if not has_tool_call and not ends_with_partial_token:
            normal_text = current_text
            self._buffer = ""
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

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        all_calls: list[ToolCallItem] = []
        try:
            # Loop to handle multiple consecutive invoke blocks
            while True:
                # Try to match an invoke block (may be partial)
                invoke_match = re.search(
                    pattern=r'<｜DSML｜invoke\s+name="([^"]+)"\s*>(.*?)(</｜DSML｜invoke>|$)',
                    string=current_text,
                    flags=re.DOTALL,
                )

                if not invoke_match:
                    break

                func_name = invoke_match.group(1).strip()
                invoke_content = invoke_match.group(2)
                # group(3) is either "</｜DSML｜invoke>" (complete) or "" (incomplete, matched with $)
                is_tool_end = bool(invoke_match.group(3))

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
                    # Only emit the tool call when it's complete (saw </｜DSML｜invoke>)
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
                    self._buffer = current_text[invoke_match.end() :]
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
                    # Wait for more chunks until we see </｜DSML｜invoke>
                    break

            # No more invoke blocks found
            return StreamingParseResult(normal_text=normal_text, calls=all_calls)

        except Exception as e:
            logger.error(f"Error in parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=normal_text + current_text)

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """Release text the stream ended on while it was still withheld.

        ``parse_streaming_increment`` holds the buffer back whenever its tail
        could still grow into a DSML delimiter, and releases it as soon as the
        next chunk resolves the ambiguity. When generation stops instead --
        max_tokens, a stop string, an abort -- nothing resolves it, and the
        base class's no-op ``finish`` drops whatever was held. A response
        ending on ``<`` loses every character buffered since the last release,
        silently: the request succeeds and the tail of the answer is missing.

        Only a buffer with no tool-call section is released. Text that precedes
        a section is already streamed by the increment path, so what remains
        there is partial DSML markup rather than content, and surfacing that
        raw would trade one defect for another.
        """
        buffer = self._buffer
        if not buffer:
            return StreamingParseResult()

        start_tokens = [self.bot_token, self._INVOKE_HEADER_PREFIX]
        if any(idx != -1 for idx in map(buffer.find, start_tokens)):
            return StreamingParseResult()

        self._buffer = ""
        # The same delimiters the increment path strips before emitting, so a
        # stream that ended just after one does not show it to the caller.
        for token in (self.eot_token, self.invoke_end_token, self._eos_token):
            buffer = buffer.replace(token, "")
        return StreamingParseResult(normal_text=buffer)

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f'<｜DSML｜invoke name="{name}">',
            end="</｜DSML｜invoke>",
            trigger=f'<｜DSML｜invoke name="{name}">',
        )
