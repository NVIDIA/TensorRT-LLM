# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, List

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam as Tool
from tensorrt_llm.serve.tool_parser.base_tool_parser import BaseToolParser
from tensorrt_llm.serve.tool_parser.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)

TOOL_START_TOKEN = "<tool_call>"
TOOL_END_TOKEN = "</tool_call>"
PARAM_KEY_START_TOKEN = "<param_key>"

TOOL_CALL_REGEX = re.compile(
    rf"{re.escape(TOOL_START_TOKEN)}(.*?){re.escape(TOOL_END_TOKEN)}",
    re.DOTALL,
)
PARAM_REGEX = re.compile(
    r"<param_key>(.*?)</param_key>\s*<param_value>(.*?)</param_value>",
    re.DOTALL,
)


def _get_attr_or_item(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(exclude_none=True)
        if isinstance(dumped, Mapping):
            return dumped
    return {}


def _tool_function(tool: Any) -> Any:
    return _get_attr_or_item(tool, "function")


def _tool_name(tool: Any) -> str | None:
    function = _tool_function(tool)
    name = _get_attr_or_item(function, "name")
    return str(name) if name else None


def _tool_parameters(tool: Any) -> Mapping[str, Any]:
    function = _tool_function(tool)
    return _as_mapping(_get_attr_or_item(function, "parameters"))


def _iter_tool_names(tools: Sequence[Any] | None) -> list[str]:
    if tools is None:
        return []
    names = [_tool_name(tool) for tool in tools]
    return sorted((name for name in names if name), key=len, reverse=True)


def _is_string_type(
    tool_name: str,
    arg_name: str,
    tools: Sequence[Any] | None,
) -> bool:
    if tools is None:
        return False
    for tool in tools:
        if _tool_name(tool) != tool_name:
            continue
        parameters = _tool_parameters(tool)
        properties = _as_mapping(parameters.get("properties"))
        arg_schema = _as_mapping(properties.get(arg_name))
        arg_type = arg_schema.get("type")
        if isinstance(arg_type, str):
            return arg_type == "string"
        if isinstance(arg_type, Sequence) and not isinstance(arg_type, str):
            return "string" in arg_type
        return False
    logger.debug("No tool named '%s'.", tool_name)
    return False


def _deserialize(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        pass
    try:
        return ast.literal_eval(value)
    except Exception:
        pass
    return value


def _json_arguments(value: str) -> dict[str, Any]:
    parsed = _deserialize(value)
    if not isinstance(parsed, Mapping):
        return {}
    arguments = parsed.get("arguments", parsed.get("parameters", parsed))
    if isinstance(arguments, Mapping):
        return dict(arguments)
    return {}


def _split_payload(
    payload: str,
    tools: Sequence[Any] | None,
) -> tuple[str, str, str]:
    payload = payload.strip()
    param_pos = payload.find(PARAM_KEY_START_TOKEN)
    if param_pos != -1:
        return payload[:param_pos].strip(), payload[param_pos:], ""

    for tool_name in _iter_tool_names(tools):
        if payload == tool_name:
            return tool_name, "", ""
        if payload.startswith(tool_name):
            rest = payload[len(tool_name) :].strip()
            if rest.startswith("{"):
                return tool_name, "", rest
    return payload, "", ""


def _parse_payload(
    payload: str,
    tools: Sequence[Any] | None,
) -> tuple[str, dict[str, Any]]:
    tool_name, params_text, json_text = _split_payload(payload, tools)
    arguments = _json_arguments(json_text) if json_text else {}

    for key, value in PARAM_REGEX.findall(params_text):
        arg_key = key.strip()
        arg_val = value.strip()
        if not _is_string_type(tool_name, arg_key, tools):
            arg_val = _deserialize(arg_val)
        arguments[arg_key] = arg_val

    return tool_name, arguments


def _partial_suffix_len(text: str, token: str) -> int:
    max_len = min(len(text), len(token) - 1)
    for size in range(max_len, 0, -1):
        if token.startswith(text[-size:]):
            return size
    return 0


class XingChen4ToolParser(BaseToolParser):
    """Tool parser for the XingChen4 tool-call format.

    Supports both JSON payloads and tag-based key/value payloads inside a
    ``<tool_call>..</tool_call>`` block.
    """

    def __init__(self):
        super().__init__()
        self.bot_token = TOOL_START_TOKEN
        self.eot_token = TOOL_END_TOKEN
        self.tool_call_separator = ""

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=self.bot_token + '{"name":"' + name + '", "arguments":',
            end="}" + self.eot_token,
            trigger=self.bot_token,
        )

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        idx = text.find(self.bot_token)
        if idx == -1:
            return StreamingParseResult(normal_text=text, calls=[])

        normal_text = text[:idx]
        tool_indices = self._get_tool_indices(tools) if tools else {}
        calls: list[ToolCallItem] = []
        try:
            for match in TOOL_CALL_REGEX.finditer(text):
                tool_name, arguments = _parse_payload(match.group(1), tools)
                if not tool_name:
                    continue
                if tool_indices and tool_name not in tool_indices:
                    logger.warning(
                        "Model attempted to call undefined function: %s",
                        tool_name,
                    )
                calls.append(
                    ToolCallItem(
                        tool_index=tool_indices.get(tool_name, -1),
                        name=tool_name,
                        parameters=json.dumps(arguments, ensure_ascii=False),
                    )
                )
        except Exception:
            logger.exception("Failed to extract XingChen4 tool call spec")
            return StreamingParseResult(normal_text=text, calls=[])

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        self._buffer += new_text
        normal_text_parts: list[str] = []
        calls: list[ToolCallItem] = []

        tool_indices = self._get_tool_indices(tools) if tools else {}

        while True:
            start_idx = self._buffer.find(self.bot_token)
            if start_idx == -1:
                partial_len = _partial_suffix_len(self._buffer, self.bot_token)
                if partial_len:
                    normal_text_parts.append(self._buffer[:-partial_len])
                    self._buffer = self._buffer[-partial_len:]
                else:
                    normal_text_parts.append(self._buffer)
                    self._buffer = ""
                break

            if start_idx > 0:
                normal_text_parts.append(self._buffer[:start_idx])
                self._buffer = self._buffer[start_idx:]

            end_idx = self._buffer.find(self.eot_token)
            if end_idx == -1:
                break

            end_pos = end_idx + len(self.eot_token)
            block = self._buffer[:end_pos]
            self._buffer = self._buffer[end_pos:]

            inner_match = TOOL_CALL_REGEX.search(block)
            if inner_match is None:
                logger.warning(
                    "XingChen4 tool block did not match expected shape: %r",
                    block,
                )
                normal_text_parts.append(block)
                continue

            try:
                tool_name, arguments = _parse_payload(inner_match.group(1), tools)
            except Exception:
                logger.exception(
                    "Failed to parse XingChen4 tool call payload: %r",
                    inner_match.group(1),
                )
                tool_name, arguments = "", {}

            if not tool_name:
                logger.warning("Failed to extract any tool call from %r.", block)
                normal_text_parts.append(block)
                continue

            if tool_indices and tool_name not in tool_indices:
                logger.warning(
                    "Model attempted to call undefined function: %s",
                    tool_name,
                )

            self.current_tool_id += 1
            tool_index = tool_indices.get(tool_name, self.current_tool_id)
            calls.append(
                ToolCallItem(
                    tool_index=tool_index,
                    name=tool_name,
                    parameters=json.dumps(arguments, ensure_ascii=False),
                )
            )

        return StreamingParseResult(
            normal_text="".join(normal_text_parts),
            calls=calls,
        )
