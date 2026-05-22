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
from typing import List, Optional, Tuple

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam as Tool
from tensorrt_llm.serve.tool_parser.base_tool_parser import BaseToolParser
from tensorrt_llm.serve.tool_parser.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)

# Gemma4 special tokens for tool calling
BOT_TOKEN = "<|tool_call>"  # nosec B105
EOT_TOKEN = "<tool_call|>"  # nosec B105
STRING_DELIM = '<|"|>'
CALL_PREFIX = "call:"


def _find_matching_brace(text: str, start: int) -> int:
    """Find the closing brace matching the opening brace at `start`.

    Respects STRING_DELIM boundaries so braces inside strings are not counted.
    Returns the index of the closing brace, or -1 if not found.
    """
    depth = 0
    i = start
    in_string = False
    while i < len(text):
        if text[i:].startswith(STRING_DELIM):
            in_string = not in_string
            i += len(STRING_DELIM)
            continue
        if not in_string:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _parse_gemma4_value(text: str):
    """Parse a single Gemma4-format value.

    Handles: strings (<|"|>...<|"|>), booleans, numbers, nested objects, arrays.
    """
    text = text.strip()

    # String value
    if text.startswith(STRING_DELIM):
        end = text.find(STRING_DELIM, len(STRING_DELIM))
        if end == -1:
            return text[len(STRING_DELIM) :]
        return text[len(STRING_DELIM) : end]

    # Nested object
    if text.startswith("{"):
        close = _find_matching_brace(text, 0)
        if close == -1:
            return text
        return _parse_gemma4_args(text[1:close])

    # Array
    if text.startswith("["):
        return _parse_gemma4_array(text)

    # Boolean
    if text == "true":
        return True
    if text == "false":
        return False

    # Null
    if text == "null":
        return None

    # Number
    try:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)
    except ValueError:
        return text


def _parse_gemma4_array(text: str) -> list:
    """Parse a Gemma4-format array: [elem1,elem2,...].

    Handles nested objects, arrays, and string delimiters.
    """
    text = text.strip()
    if not text.startswith("["):
        return []

    # Find matching closing bracket
    depth = 0
    in_string = False
    close_idx = -1
    for i, ch in enumerate(text):
        if text[i:].startswith(STRING_DELIM):
            in_string = not in_string
            continue
        if not in_string:
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    close_idx = i
                    break
    if close_idx == -1:
        return []

    inner = text[1:close_idx].strip()
    if not inner:
        return []

    # Split elements respecting nesting and string delimiters
    elements = []
    elem_start = 0
    depth = 0
    in_str = False
    i = 0
    while i < len(inner):
        if inner[i:].startswith(STRING_DELIM):
            in_str = not in_str
            i += len(STRING_DELIM)
            continue
        if not in_str:
            if inner[i] in ("{", "["):
                depth += 1
            elif inner[i] in ("}", "]"):
                depth -= 1
            elif inner[i] == "," and depth == 0:
                elements.append(inner[elem_start:i].strip())
                elem_start = i + 1
        i += 1
    # Last element
    last = inner[elem_start:].strip()
    if last:
        elements.append(last)

    return [_parse_gemma4_value(elem) for elem in elements]


def _parse_gemma4_args(text: str) -> dict:
    """Parse Gemma4-format arguments: key1:<|"|>val1<|"|>,key2:42,...

    Keys are bare identifiers. Values can be strings, numbers, booleans,
    nested objects, or arrays.
    """
    result = {}
    text = text.strip()
    if not text:
        return result

    i = 0
    while i < len(text):
        # Skip whitespace and commas
        while i < len(text) and text[i] in (" ", "\t", "\n", "\r", ","):
            i += 1
        if i >= len(text):
            break

        # Parse key (bare identifier up to ':')
        colon_pos = text.find(":", i)
        if colon_pos == -1:
            break
        key = text[i:colon_pos].strip()
        i = colon_pos + 1

        # Parse value
        if i < len(text) and text[i:].startswith(STRING_DELIM):
            # String value: find closing delimiter
            after_open = i + len(STRING_DELIM)
            close_pos = text.find(STRING_DELIM, after_open)
            if close_pos == -1:
                result[key] = text[after_open:]
                break
            result[key] = text[after_open:close_pos]
            i = close_pos + len(STRING_DELIM)
        elif i < len(text) and text[i] == "{":
            # Nested object
            close = _find_matching_brace(text, i)
            if close == -1:
                break
            result[key] = _parse_gemma4_args(text[i + 1 : close])
            i = close + 1
        elif i < len(text) and text[i] == "[":
            # Array value: find matching bracket
            depth = 0
            in_string = False
            j = i
            while j < len(text):
                if text[j:].startswith(STRING_DELIM):
                    in_string = not in_string
                    j += len(STRING_DELIM)
                    continue
                if not in_string:
                    if text[j] == "[":
                        depth += 1
                    elif text[j] == "]":
                        depth -= 1
                        if depth == 0:
                            result[key] = _parse_gemma4_array(text[i : j + 1])
                            i = j + 1
                            break
                j += 1
            else:
                # Unmatched bracket
                result[key] = _parse_gemma4_array(text[i:])
                break
        else:
            # Bare value (number, boolean, null, or unquoted string)
            # Scan until comma, closing brace/bracket, or end
            j = i
            depth = 0
            while j < len(text):
                if text[j] in ("{", "["):
                    depth += 1
                elif text[j] in ("}", "]"):
                    if depth == 0:
                        break
                    depth -= 1
                elif text[j] == "," and depth == 0:
                    break
                j += 1
            raw_val = text[i:j].strip()
            result[key] = _parse_gemma4_value(raw_val)
            i = j

    return result


def _extract_tool_calls(text: str) -> List[Tuple[str, str]]:
    """Extract all tool call blocks from text.

    Returns list of (function_name, args_string) tuples.
    """
    calls = []
    search_from = 0
    while True:
        start = text.find(BOT_TOKEN, search_from)
        if start == -1:
            break
        end = text.find(EOT_TOKEN, start + len(BOT_TOKEN))
        if end == -1:
            break

        inner = text[start + len(BOT_TOKEN) : end].strip()
        if inner.startswith(CALL_PREFIX):
            inner = inner[len(CALL_PREFIX) :]
            # Find function name (everything before the first '{')
            brace_pos = inner.find("{")
            if brace_pos != -1:
                func_name = inner[:brace_pos].strip()
                # Find matching closing brace
                close = _find_matching_brace(inner, brace_pos)
                if close != -1:
                    args_str = inner[brace_pos + 1 : close]
                else:
                    args_str = inner[brace_pos + 1 :]
                calls.append((func_name, args_str))

        search_from = end + len(EOT_TOKEN)
    return calls


class Gemma4ToolParser(BaseToolParser):
    """Tool parser for Gemma4 model function call format.

    Gemma4 uses a custom non-JSON format with special string delimiters.

    Format Structure:
    ```
    <|tool_call>call:function_name{key1:<|"|>value1<|"|>,key2:42}<tool_call|>
    ```

    Key Components:
    - Tool Call Tags: `<|tool_call>` and `<tool_call|>` wrap each call
    - Call Prefix: `call:` precedes the function name
    - Arguments: `key:value` pairs (NOT JSON), comma-separated
    - String Delimiter: `<|"|>` wraps string values (instead of JSON quotes)
    - Supports: strings, numbers, booleans, null, nested objects, arrays

    Multiple tool calls are placed sequentially with no separator.
    """

    needs_raw_special_tokens: bool = True

    def __init__(self):
        super().__init__()
        self.bot_token = BOT_TOKEN  # nosec B105
        self.eot_token = EOT_TOKEN  # nosec B105

        # Streaming state
        self._is_inside_tool_call = False
        self._current_func_name: Optional[str] = None

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a Gemma4 format tool call."""
        return BOT_TOKEN in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-time parsing: detect and parse all tool calls in text."""
        if BOT_TOKEN not in text:
            return StreamingParseResult(normal_text=text, calls=[])

        tool_indices = self._get_tool_indices(tools)
        raw_calls = _extract_tool_calls(text)

        calls = []
        for func_name, args_str in raw_calls:
            try:
                parsed_args = _parse_gemma4_args(args_str)
            except (ValueError, IndexError, TypeError, KeyError) as e:
                logger.warning(f"Failed to parse Gemma4 tool call args: {args_str}, error: {e}")
                continue

            tool_index = tool_indices.get(func_name, -1)
            if tool_index == -1:
                logger.warning(f"Model attempted to call undefined function: {func_name}")

            calls.append(
                ToolCallItem(
                    tool_index=tool_index,
                    name=func_name,
                    parameters=json.dumps(parsed_args, ensure_ascii=False),
                )
            )

        # Normal text is everything before the first tool call
        first_tc = text.find(BOT_TOKEN)
        normal_text = text[:first_tc].strip() if first_tc > 0 else ""

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """Streaming incremental parsing for Gemma4 tool calls.

        Maintains state across chunks to handle tool calls that span
        multiple streaming increments.
        """
        self._buffer += new_text
        current_text = self._buffer

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        try:
            # If not inside a tool call, look for the start token
            if not self._is_inside_tool_call:
                bot_pos = current_text.find(BOT_TOKEN)
                if bot_pos == -1:
                    # Check for partial start token at the end
                    partial_len = self._ends_with_partial_token(current_text, BOT_TOKEN)
                    if partial_len:
                        # Keep the partial match, emit everything before it
                        normal = current_text[:-partial_len]
                        self._buffer = current_text[-partial_len:]
                        return (
                            StreamingParseResult(normal_text=normal)
                            if normal
                            else StreamingParseResult()
                        )
                    # No tool call, emit everything
                    self._buffer = ""
                    # Strip any stray end tokens
                    cleaned = current_text.replace(EOT_TOKEN, "")
                    return StreamingParseResult(normal_text=cleaned)

                # Found start of tool call
                normal_text = current_text[:bot_pos]
                self._buffer = current_text[bot_pos:]
                self._is_inside_tool_call = True
                self._current_func_name = None

                if normal_text:
                    return StreamingParseResult(normal_text=normal_text)
                # Fall through to process the tool call content

            # Inside a tool call - process the content
            current_text = self._buffer

            # Look for the end token
            eot_pos = current_text.find(EOT_TOKEN)

            if eot_pos == -1:
                # Tool call not complete yet
                # Try to extract function name if we haven't already
                if self._current_func_name is None:
                    prefix = BOT_TOKEN + CALL_PREFIX
                    if current_text.startswith(prefix):
                        rest = current_text[len(prefix) :]
                        brace_pos = rest.find("{")
                        if brace_pos != -1:
                            func_name = rest[:brace_pos].strip()
                            self._current_func_name = func_name

                            # Initialize tool tracking
                            if self.current_tool_id == -1:
                                self.current_tool_id = 0
                            self.streamed_args_for_tool.append("")
                            while len(self.prev_tool_call_arr) <= self.current_tool_id:
                                self.prev_tool_call_arr.append({})
                            self.prev_tool_call_arr[self.current_tool_id] = {
                                "name": func_name,
                                "arguments": {},
                            }
                            self.current_tool_name_sent = True

                            return StreamingParseResult(
                                calls=[
                                    ToolCallItem(
                                        tool_index=self.current_tool_id,
                                        name=func_name,
                                        parameters="",
                                    )
                                ]
                            )

                # Stream raw argument text incrementally
                if self._current_func_name is not None:
                    prefix = BOT_TOKEN + CALL_PREFIX + self._current_func_name + "{"
                    if current_text.startswith(prefix) and len(current_text) > len(prefix):
                        args_so_far = current_text[len(prefix) :]
                        already_sent = self.streamed_args_for_tool[self.current_tool_id]
                        if len(args_so_far) > len(already_sent):
                            diff = args_so_far[len(already_sent) :]
                            self.streamed_args_for_tool[self.current_tool_id] += diff
                            return StreamingParseResult(
                                calls=[
                                    ToolCallItem(
                                        tool_index=self.current_tool_id,
                                        parameters=diff,
                                    )
                                ]
                            )

                return StreamingParseResult()

            # Found end token - complete the tool call
            inner = current_text[len(BOT_TOKEN) : eot_pos].strip()
            remaining = current_text[eot_pos + len(EOT_TOKEN) :]

            if inner.startswith(CALL_PREFIX):
                inner = inner[len(CALL_PREFIX) :]
                brace_pos = inner.find("{")
                if brace_pos != -1:
                    func_name = inner[:brace_pos].strip()
                    close = _find_matching_brace(inner, brace_pos)
                    args_str = (
                        inner[brace_pos + 1 : close] if close != -1 else inner[brace_pos + 1 :]
                    )

                    try:
                        parsed_args = _parse_gemma4_args(args_str)
                        args_json = json.dumps(parsed_args, ensure_ascii=False)
                    except (ValueError, IndexError, TypeError, KeyError):
                        args_json = "{}"

                    calls = []

                    if self._current_func_name is None:
                        # We haven't sent the name yet (whole call in one chunk)
                        if self.current_tool_id == -1:
                            self.current_tool_id = 0
                        while len(self.prev_tool_call_arr) <= self.current_tool_id:
                            self.prev_tool_call_arr.append({})
                        while len(self.streamed_args_for_tool) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")

                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=func_name,
                                parameters=args_json,
                            )
                        )
                    else:
                        # We already sent the name, now send final parsed JSON
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                parameters=args_json,
                            )
                        )

                    # Store for serving layer
                    try:
                        self.prev_tool_call_arr[self.current_tool_id] = {
                            "name": func_name,
                            "arguments": json.loads(args_json),
                        }
                    except (json.JSONDecodeError, IndexError):
                        pass

                    # Reset state for next tool call
                    self._is_inside_tool_call = False
                    self._current_func_name = None
                    self._buffer = remaining
                    self.current_tool_id += 1
                    self.current_tool_name_sent = False

                    return StreamingParseResult(calls=calls)

            # Couldn't parse the tool call content
            self._is_inside_tool_call = False
            self._current_func_name = None
            self._buffer = remaining
            return StreamingParseResult()

        except (ValueError, IndexError, TypeError, KeyError) as e:
            logger.error(f"Error in Gemma4 streaming parse: {e}")
            self._is_inside_tool_call = False
            self._current_func_name = None
            self._buffer = ""
            return StreamingParseResult()

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=BOT_TOKEN + CALL_PREFIX + name + "{",
            end="}" + EOT_TOKEN,
            trigger=BOT_TOKEN,
        )
