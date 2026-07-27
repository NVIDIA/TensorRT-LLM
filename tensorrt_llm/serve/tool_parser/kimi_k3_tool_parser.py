# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tool-call parser for the Kimi K3 XTML output format.

K3 emits tool calls as an XTML tag stream built from the special tokens
``<|open|>`` / ``<|close|>`` / ``<|sep|>`` with plain-text tag headers
(authoritative rendering: the checkpoint's ``encoding_k3.py``)::

    <|open|>tools<|sep|>
      <|open|>call tool="NAME" index="1"<|sep|>
        <|open|>argument key="K" type="string|number|boolean|null|object|array"<|sep|>
          VALUE
        <|close|>argument<|sep|>
        ...
      <|close|>call<|sep|>
      ...
    <|close|>tools<|sep|>

Alternatively a call body may carry one raw JSON block::

    <|open|>json type="object"<|sep|>{...}<|close|>json<|sep|>

Attribute values are escaped (``&`` -> ``&amp;``, ``"`` -> ``&quot;``).
``argument`` bodies are raw text for ``type="string"`` and JSON text for
every other type. The section arrives after the response body, so streaming
buffers the section and emits complete calls once ``<|close|>tools<|sep|>``
is seen.
"""

import json
import re
from typing import Any, Dict, List

from tensorrt_llm.logger import logger

from ..openai_protocol import ChatCompletionToolsParam as Tool
from .base_tool_parser import BaseToolParser
from .core_types import StreamingParseResult, ToolCallItem, _GetInfoFunc


def _unescape_attr(value: str) -> str:
    return value.replace("&quot;", '"').replace("&amp;", "&")


def _parse_attrs(header: str) -> Dict[str, str]:
    return {key: _unescape_attr(value) for key, value in re.findall(r'(\w+)="([^"]*)"', header)}


class KimiK3ToolParser(BaseToolParser):
    """Detector for the Kimi K3 XTML function-call format."""

    needs_raw_special_tokens = True

    def __init__(self):
        super().__init__()
        self.bot_token = "<|open|>tools<|sep|>"  # nosec B105
        self.eot_token = "<|close|>tools<|sep|>"  # nosec B105
        # Structural leftovers that may trail the tools section when the
        # reasoning parser is not in front of this parser.
        self._trailing_structural = re.compile(
            r"(?:<\|close\|>message<\|sep\|>|<\|end_of_msg\|>)+\s*$"
        )

        self._call_regex = re.compile(
            r"<\|open\|>call(?P<attrs>[^<]*?)<\|sep\|>"
            r"(?P<body>.*?)<\|close\|>call<\|sep\|>",
            re.DOTALL,
        )
        self._argument_regex = re.compile(
            r"<\|open\|>argument(?P<attrs>[^<]*?)<\|sep\|>"
            r"(?P<value>.*?)<\|close\|>argument<\|sep\|>",
            re.DOTALL,
        )
        self._json_regex = re.compile(
            r"<\|open\|>json(?P<attrs>[^<]*?)<\|sep\|>"
            r"(?P<value>.*?)<\|close\|>json<\|sep\|>",
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def supports_structural_tag(self) -> bool:
        # XTML argument bodies are tag-structured text, not JSON — the
        # JSON-schema-driven structural-tag constrained decoding used for
        # strict tools does not apply.
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError(
            "kimi_k3 XTML tool calls do not support structural-tag constrained decoding"
        )

    @staticmethod
    def _coerce_value(value: str, value_type: str) -> Any:
        if value_type == "string":
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.warning(
                "kimi_k3 tool parser: argument declared type=%s but body is "
                "not valid JSON; keeping raw text",
                value_type,
            )
            return value

    def _parse_call_arguments(self, body: str) -> str:
        """Reconstruct the OpenAI ``function.arguments`` JSON string from a call body."""
        json_match = self._json_regex.search(body)
        if json_match is not None:
            raw = json_match.group("value").strip()
            try:
                return json.dumps(json.loads(raw), ensure_ascii=False)
            except json.JSONDecodeError:
                logger.warning(
                    "kimi_k3 tool parser: json block is not valid JSON; passing raw text through"
                )
                return raw
        arguments: Dict[str, Any] = {}
        for match in self._argument_regex.finditer(body):
            attrs = _parse_attrs(match.group("attrs"))
            key = attrs.get("key")
            if key is None:
                continue
            arguments[key] = self._coerce_value(match.group("value"), attrs.get("type", "string"))
        return json.dumps(arguments, ensure_ascii=False)

    def _parse_tools_section(self, section: str, tools: List[Tool]) -> List[ToolCallItem]:
        tool_indices = self._get_tool_indices(tools)
        calls: List[ToolCallItem] = []
        for position, match in enumerate(self._call_regex.finditer(section)):
            attrs = _parse_attrs(match.group("attrs"))
            name = attrs.get("tool")
            if not name:
                logger.warning(
                    "kimi_k3 tool parser: call without tool attribute: %s", match.group("attrs")
                )
                continue
            if name not in tool_indices:
                logger.warning("Model attempted to call undefined function: %s", name)
            calls.append(
                ToolCallItem(
                    tool_index=position,
                    name=name,
                    parameters=self._parse_call_arguments(match.group("body")),
                )
            )
        return calls

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        bot_idx = text.find(self.bot_token)
        if bot_idx == -1:
            return StreamingParseResult(normal_text=self._trailing_structural.sub("", text))
        normal_text = text[:bot_idx]
        section = text[bot_idx + len(self.bot_token) :]
        eot_idx = section.find(self.eot_token)
        if eot_idx != -1:
            section = section[:eot_idx]
        calls = self._parse_tools_section(section, tools)
        self.prev_tool_call_arr = []
        for call in calls:
            try:
                arguments = json.loads(call.parameters)
            except json.JSONDecodeError:
                arguments = call.parameters
            self.prev_tool_call_arr.append(
                {
                    "name": call.name,
                    "arguments": arguments,
                }
            )
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        self._buffer += new_text
        bot_idx = self._buffer.find(self.bot_token)
        if bot_idx == -1:
            hold = self._ends_with_partial_token(self._buffer, self.bot_token)
            emit_len = len(self._buffer) - hold
            normal_text = self._buffer[:emit_len]
            self._buffer = self._buffer[emit_len:]
            return StreamingParseResult(normal_text=normal_text)

        # Flush any response text preceding the section, then buffer the
        # whole section until it completes: K3 tool calls terminate the
        # message, so latency cost is negligible and complete calls avoid
        # partial-argument reconstruction entirely.
        normal_text = self._buffer[:bot_idx]
        self._buffer = self._buffer[bot_idx:]
        eot_idx = self._buffer.find(self.eot_token)
        if eot_idx == -1:
            return StreamingParseResult(normal_text=normal_text)
        section_end = eot_idx + len(self.eot_token)
        result = self.detect_and_parse(self._buffer[:section_end], tools)
        # Anything after the section (normally empty) is re-examined on the
        # next increment rather than dropped.
        self._buffer = self._buffer[section_end:]
        return StreamingParseResult(
            normal_text=normal_text + result.normal_text, calls=result.calls
        )
