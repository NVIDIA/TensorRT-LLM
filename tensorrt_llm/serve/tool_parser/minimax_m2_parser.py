# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from typing import Any, Dict, List, Optional

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam as Tool
from tensorrt_llm.serve.tool_parser.base_tool_parser import BaseToolParser
from tensorrt_llm.serve.tool_parser.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)


def _get_param_types(param_name: str, func_name: str, tools: List[Tool]) -> Optional[str]:
    """Get the expected type of a parameter from tool definitions."""
    for tool in tools:
        if tool.function.name == func_name:
            props = (tool.function.parameters or {}).get("properties", {})
            if param_name in props:
                type_val = props[param_name].get("type")
                if isinstance(type_val, str):
                    return type_val
                if isinstance(type_val, list):
                    non_null = [t for t in type_val if t != "null"]
                    return non_null[0] if non_null else "string"
    return None


def _parse_param_value(value_str: str, param_type: Optional[str]) -> Any:
    """Parse a parameter value string into the appropriate Python type."""
    value_str = value_str.strip()

    # Short-circuit: declared strings should never be coerced.
    if param_type == "string":
        return value_str

    # Try JSON parsing for structured types (object, array, null, etc.).
    try:
        parsed = json.loads(value_str)
        return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # For numeric types, try numeric conversion
    if param_type in ("number", "integer"):
        try:
            if "." in value_str or "e" in value_str.lower():
                return float(value_str)
            return int(value_str)
        except (ValueError, TypeError):
            pass

    # For boolean type
    if param_type == "boolean":
        if value_str.lower() == "true":
            return True
        if value_str.lower() == "false":
            return False

    # Default: return as string
    return value_str


class MiniMaxM2ToolParser(BaseToolParser):
    r"""Tool parser for MiniMax-M2 models.

    Parses the MiniMax-M2 tool call format:
        <minimax:tool_call>
          <invoke name="function_name">
            <parameter name="param1">value1</parameter>
            <parameter name="param2">value2</parameter>
          </invoke>
        </minimax:tool_call>

    Supports both single and parallel (multiple <invoke> blocks) tool calls.
    """

    def __init__(self):
        """Initialize the MiniMax-M2 tool parser with XML tag patterns."""
        super().__init__()
        self.bot_token = "<minimax:tool_call>"  # nosec B105
        self.eot_token = "</minimax:tool_call>"  # nosec B105

        # Regex patterns for parsing
        self.tool_call_block_regex = re.compile(
            r"<minimax:tool_call>(.*?)</minimax:tool_call>", re.DOTALL
        )
        self.invoke_regex = re.compile(
            r'<invoke\s+name=["\']?(.*?)["\']?\s*>(.*?)</invoke>', re.DOTALL
        )
        self.param_regex = re.compile(
            r'<parameter\s+name=["\']?(.*?)["\']?\s*>(.*?)</parameter>', re.DOTALL
        )

        # Streaming state
        self._streamed_raw_length = 0
        self._current_invoke_count = 0
        self._json_buffers: Dict[int, str] = {}

    def has_tool_call(self, text: str) -> bool:
        """Check whether the text contains a MiniMax-M2 tool call tag."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-time parsing: detect and parse all tool calls in the text."""
        idx = text.find(self.bot_token)
        if idx == -1:
            return StreamingParseResult(normal_text=text, calls=[])

        # Preserve prefix text before the tool call block.
        prefix = text[:idx].strip()

        # Preserve suffix text after the closing tag (if present).
        suffix = ""
        eot_idx = text.rfind(self.eot_token)
        if eot_idx != -1:
            suffix = text[eot_idx + len(self.eot_token) :].strip()

        normal_text = (prefix + " " + suffix).strip() if prefix and suffix else (prefix or suffix)

        calls: List[ToolCallItem] = []
        tool_call_blocks = self.tool_call_block_regex.findall(text)
        tool_index = 0

        for block in tool_call_blocks:
            invocations = self.invoke_regex.findall(block)
            for func_name, params_text in invocations:
                func_name = func_name.strip()
                params = self.param_regex.findall(params_text)

                arguments: Dict[str, Any] = {}
                for param_name, param_value in params:
                    param_name = param_name.strip()
                    param_type = _get_param_types(param_name, func_name, tools)
                    arguments[param_name] = _parse_param_value(param_value, param_type)

                calls.append(
                    ToolCallItem(
                        tool_index=tool_index,
                        name=func_name,
                        parameters=json.dumps(arguments, ensure_ascii=False),
                    )
                )
                tool_index += 1

        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """Streaming incremental parsing for MiniMax-M2 tool call format."""
        self._buffer += new_text
        current_text = self._buffer

        has_tool = self.bot_token in current_text

        if not has_tool:
            # Check for partial bot token
            is_potential_start = any(
                self.bot_token.startswith(current_text[-i:])
                for i in range(1, min(len(current_text), len(self.bot_token)) + 1)
            )

            if not is_potential_start:
                output_text = current_text
                self._buffer = ""
                # Clean up end tokens
                if self.eot_token in output_text:
                    output_text = output_text.replace(self.eot_token, "")
                return StreamingParseResult(normal_text=output_text)
            return StreamingParseResult(normal_text="", calls=[])

        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        calls: List[ToolCallItem] = []

        try:
            # Extract content after <minimax:tool_call>
            tc_start = current_text.find(self.bot_token)
            # Preserve any text before the tool call opening tag.
            prefix_text = current_text[:tc_start].strip() if tc_start > 0 else ""
            inner_text = current_text[tc_start + len(self.bot_token) :]

            # Find all complete invoke blocks
            complete_invocations = list(self.invoke_regex.finditer(inner_text))
            # Check for partial invoke (started but not closed)
            partial_invoke = re.search(
                r'<invoke\s+name=["\']?(.*?)["\']?\s*>((?:(?!</invoke>).)*?)$',
                inner_text,
                re.DOTALL,
            )

            # Process complete invocations that haven't been processed yet
            for i, match in enumerate(complete_invocations):
                if i < self._current_invoke_count:
                    continue
                func_name = match.group(1).strip()
                params_text = match.group(2)

                params = self.param_regex.findall(params_text)
                arguments: Dict[str, Any] = {}
                for param_name, param_value in params:
                    param_name = param_name.strip()
                    param_type = _get_param_types(param_name, func_name, tools)
                    arguments[param_name] = _parse_param_value(param_value, param_type)

                # Send function name
                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id + 1 if self.current_tool_id >= 0 else 0,
                        name=func_name,
                        parameters="",
                    )
                )
                # Send complete arguments
                args_json = json.dumps(arguments, ensure_ascii=False)
                self.current_tool_id = self.current_tool_id + 1 if self.current_tool_id >= 0 else 0
                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=None,
                        parameters=args_json,
                    )
                )
                self._current_invoke_count = i + 1

            # Handle partial invoke (streaming in progress)
            if partial_invoke and len(complete_invocations) == self._current_invoke_count:
                func_name = partial_invoke.group(1).strip()
                partial_params_text = partial_invoke.group(2)

                # If we haven't sent the name for this tool yet
                if (
                    not self.current_tool_name_sent
                    or self.current_tool_id < self._current_invoke_count
                ):
                    if func_name:
                        self.current_tool_id = self._current_invoke_count
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=func_name,
                                parameters="",
                            )
                        )
                        self.current_tool_name_sent = True
                        self._json_buffers[self.current_tool_id] = ""

                # Stream partial arguments
                if self.current_tool_name_sent and partial_params_text:
                    complete_params = self.param_regex.findall(partial_params_text)
                    if complete_params:
                        arguments = {}
                        for param_name, param_value in complete_params:
                            param_name = param_name.strip()
                            param_type = _get_param_types(param_name, func_name, tools)
                            arguments[param_name] = _parse_param_value(param_value, param_type)
                        new_args_json = json.dumps(arguments, ensure_ascii=False)
                        prev_json = self._json_buffers.get(self.current_tool_id, "")
                        if new_args_json != prev_json:
                            # Send incremental diff
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=None,
                                    parameters=new_args_json,
                                )
                            )
                            self._json_buffers[self.current_tool_id] = new_args_json

            # Check if tool call block is complete
            if self.eot_token in current_text:
                self._buffer = current_text[
                    current_text.find(self.eot_token) + len(self.eot_token) :
                ]
                self.current_tool_name_sent = False
                self._current_invoke_count = 0
                self._json_buffers.clear()

            return StreamingParseResult(normal_text=prefix_text, calls=calls)

        except Exception as e:
            logger.error(f"Error in MiniMaxM2 parse_streaming_increment: {e}")
            return StreamingParseResult(normal_text=current_text)

    def supports_structural_tag(self) -> bool:
        """Return whether this parser supports structural tag guided decoding."""
        return False

    def structure_info(self) -> _GetInfoFunc:
        """Return structure info for guided decoding (not supported)."""
        raise NotImplementedError()
