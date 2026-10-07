# Adapted from https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/function_call/glm4_moe_detector.py
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
import ast
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam as Tool
from tensorrt_llm.serve.tool_parser.base_tool_parser import BaseToolParser
from tensorrt_llm.serve.tool_parser.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)

from .utils import infer_type_from_json_schema

_ARG_KEY = "<arg_key>"
_ARG_KEY_END = "</arg_key>"
_ARG_VALUE = "<arg_value>"
_ARG_VALUE_END = "</arg_value>"
# The filler GLM writes between tags: whitespace or a literal backslash-n.
_SEPARATORS = re.compile(r"(?:\s|\\n)*")
# The end of the GLM-4.5 name line: a newline or a literal backslash-n.
_NAME_LINE_END = re.compile(r"\\n|\n")


def parse_argument_text(text: str) -> Optional[List[Tuple[str, str]]]:
    """Read a call's argument text as ``(key, raw value)`` pairs.

    The text must be separators and complete
    ``<arg_key>K</arg_key><arg_value>V</arg_value>`` pairs, or None is returned.
    A value runs to the first ``</arg_value>`` that is followed, over separators,
    by the next ``<arg_key>`` or the end of the call, so a value may quote that tag.
    """
    pairs = []
    pos = _SEPARATORS.match(text).end()
    while pos < len(text):
        if not text.startswith(_ARG_KEY, pos):
            return None
        key_start = pos + len(_ARG_KEY)
        key_end = text.find("<", key_start)
        if not text.startswith(_ARG_KEY_END, key_end):
            return None
        value_start = _SEPARATORS.match(text, key_end + len(_ARG_KEY_END)).end()
        if not text.startswith(_ARG_VALUE, value_start):
            return None
        value_start += len(_ARG_VALUE)
        value_end = text.find(_ARG_VALUE_END, value_start)
        while value_end != -1:
            pos = _SEPARATORS.match(text, value_end + len(_ARG_VALUE_END)).end()
            if pos == len(text) or text.startswith(_ARG_KEY, pos):
                break
            value_end = text.find(_ARG_VALUE_END, value_end + 1)
        if value_end == -1:
            return None
        pairs.append((text[key_start:key_end].strip(), text[value_start:value_end]))
    return pairs


def get_argument_type(func_name: str, arg_key: str, defined_tools: List[Tool]) -> Optional[str]:
    """Get the expected type of a function argument from tool definitions."""
    name2tool = {tool.function.name: tool for tool in defined_tools}
    if func_name not in name2tool:
        return None
    tool = name2tool[func_name]
    properties = (tool.function.parameters or {}).get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    if arg_key not in properties:
        return None
    return infer_type_from_json_schema(properties[arg_key])


def _convert_to_number(value: str) -> Any:
    """Convert string to appropriate number type (int or float)."""
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        else:
            return int(value)
    except (ValueError, AttributeError):
        return value


def _is_strict_json(value: Any) -> bool:
    """Whether ``json.dumps`` spells `value` as strict JSON (no NaN, Infinity, sets...)."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        return False
    return True


def _loads_escaped_json(value: str) -> Any:
    r"""Decode JSON written with escaped quotes, such as ``[{\"a\": 1}]``."""
    return json.loads(json.loads('{"tmp": "' + value + '"}')["tmp"])


def parse_arguments(value: str, arg_type: Optional[str] = None) -> Any:
    """Decode an argument's raw text for its schema type, or return the text itself.

    With no resolvable type only strict JSON decodes; a declared non-string type
    also accepts JSON with escaped quotes and Python literals. A result that is
    not strict JSON (``NaN``, an overflowing ``1e309``, a set) keeps the text.
    """
    decoders = [json.loads]
    if arg_type is not None:
        decoders += [_loads_escaped_json, ast.literal_eval]
    for decode in decoders:
        try:
            parsed = decode(value)
        except (ValueError, TypeError, SyntaxError, RecursionError):
            continue
        if arg_type in ("number", "integer") and isinstance(parsed, str):
            parsed = _convert_to_number(parsed)
        if _is_strict_json(parsed):
            return parsed
    return value


def parse_argument_pairs(
    pairs: List[Tuple[str, str]], func_name: str, tools: List[Tool]
) -> Dict[str, Any]:
    """Type each raw value by the declared schema; a repeated key keeps its first value.

    A value declared a string is passed through verbatim; any other value goes
    through ``parse_arguments``.
    """
    arguments = {}
    for key, value in pairs:
        if key in arguments:
            logger.debug(f"Tool argument {key!r} repeats; keeping its first value")
            continue
        arg_type = get_argument_type(func_name, key, tools)
        arguments[key] = value if arg_type == "string" else parse_arguments(value, arg_type)
    return arguments


class Glm4ToolParser(BaseToolParser):
    r"""Tool parser for GLM-4.5 and GLM-4.6 models.

    Assumes function call format (with actual newlines):
        <tool_call>get_weather
        <arg_key>city</arg_key>
        <arg_value>北京</arg_value>
        <arg_key>date</arg_key>
        <arg_value>2024-06-27</arg_value>
        </tool_call>

    Or with literal \n characters (escaped as \\n in the output):
        <tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>北京</arg_value>\n</tool_call>

    Streaming reports each call once its ``</tool_call>`` arrives, as one item
    holding what ``detect_and_parse`` makes of the same markup.
    """

    streaming_matches_whole_parse = True

    def __init__(self):
        super().__init__()
        self.bot_token = "<tool_call>"  # nosec B105
        self.eot_token = "</tool_call>"  # nosec B105
        self.func_call_regex = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
        # Where the `</tool_call>` of the call open at the buffer start may begin.
        self._close_search_from = 0

    def has_tool_call(self, text: str) -> bool:
        """Check if the text contains a GLM-4 format tool call."""
        return self.bot_token in text

    def _split_call(self, body: str) -> Optional[Tuple[str, str, str]]:
        """Split a call's body into (name, markup after the name, argument text).

        The name line ends at the first newline and the name at its first ``<``;
        None when there is no name line.
        """
        line_end = _NAME_LINE_END.search(body)
        if line_end is None:
            return None
        name, tag, junk = body[: line_end.start()].partition("<")
        return name, tag + junk, body[line_end.end() :]

    def _call_name(self, name: str, junk: str, tools: List[Tool]) -> Optional[str]:
        """The name a call is delivered under; falsy when it is not a call.

        A declared tool the name maps onto wins over the name as written; markup
        after the name is only accepted when it resolves onto a declared tool.
        """
        tool_indices = self._get_tool_indices(tools)
        name = name.strip()
        if junk:
            return self.resolve_tool_name((name + junk).strip(), tool_indices)
        return name and (self.resolve_tool_name(name, tool_indices) or name)

    def _parse_call_segment(
        self, segment: str, tools: List[Tool]
    ) -> Tuple[List[ToolCallItem], str]:
        """Parse one ``<tool_call>...</tool_call>`` into (calls, text to release).

        A ``<tool_call>`` after the name restarts the call: the abandoned prefix
        is released and the rest parsed. A call is released whole as text when it
        is not one (see ``_call_name``) or its argument text is not all pairs (see
        ``parse_argument_text``).
        """
        parts = self._split_call(segment[len(self.bot_token) : -len(self.eot_token)])
        restart = -1 if parts is None else parts[1].rfind(self.bot_token)
        if restart != -1:
            cut = len(self.bot_token) + len(parts[0]) + restart
            calls, released = self._parse_call_segment(segment[cut:], tools)
            return calls, segment[:cut] + released
        name = parts and self._call_name(parts[0], parts[1], tools)
        pairs = parse_argument_text(parts[2]) if name else None
        if pairs is None:
            logger.warning(f"Releasing {len(segment)} characters of unparsable tool-call markup")
            return [], segment
        arguments = parse_argument_pairs(pairs, name, tools)
        return self.parse_base_json({"name": name, "parameters": arguments}, tools), ""

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-time parsing: Detects and parses tool calls in the provided text."""
        if self.bot_token not in text:
            return StreamingParseResult(normal_text=text, calls=[])
        normal_parts, calls, last_end = [], [], 0
        for match in self.func_call_regex.finditer(text):
            segment_calls, released = self._parse_call_segment(match.group(0), tools)
            normal_parts += [text[last_end : match.start()], released]
            calls += segment_calls
            last_end = match.end()
        normal_parts.append(text[last_end:])
        return StreamingParseResult(normal_text="".join(normal_parts).strip(), calls=calls)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """Release text that cannot start a call and report at most one complete call.

        Text after a reported call stays buffered, so the text returned with a
        call always precedes it.
        """
        self._buffer += new_text
        return self._consume(tools, at_end=False)

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """Report the calls the stream still holds and release the rest as text."""
        return self._consume(tools, at_end=True)

    def _release(self, length: int) -> str:
        """Remove and return the first `length` characters of the buffer."""
        released, self._buffer = self._buffer[:length], self._buffer[length:]
        self._close_search_from = 0
        return released

    def _consume(self, tools: List[Tool], at_end: bool) -> StreamingParseResult:
        """Turn the buffer into text and complete calls.

        Mid-stream this stops after the first call and keeps a call still open;
        at the end of the stream it takes everything, an unterminated call as text.
        """
        normal_parts, calls = [], []
        while self._buffer and (at_end or not calls):
            start = self._buffer.find(self.bot_token)
            if start == -1:
                held = 0 if at_end else self._ends_with_partial_token(self._buffer, self.bot_token)
                normal_parts.append(self._release(len(self._buffer) - held))
                break
            if start:
                normal_parts.append(self._release(start))
            end = self._buffer.find(
                self.eot_token, max(len(self.bot_token), self._close_search_from)
            )
            if end == -1:
                if at_end:
                    normal_parts.append(self._release(len(self._buffer)))
                else:
                    self._close_search_from = len(self._buffer) - len(self.eot_token) + 1
                break
            segment_calls, released = self._parse_call_segment(
                self._release(end + len(self.eot_token)), tools
            )
            normal_parts.append(released)
            for call in segment_calls:
                self.current_tool_id += 1
                call.tool_index = self.current_tool_id
            calls += segment_calls
        return StreamingParseResult(normal_text="".join(normal_parts), calls=calls)

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError()
