# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from: https://raw.githubusercontent.com/sgl-project/sglang/d8fcbaa38da95201914a1277971044ee66837b26/python/sglang/srt/function_call/qwen3_coder_detector.py

import ast
import html
import json
import math
import re
from typing import Any, Dict, List, Tuple

from jsonschema import exceptions, protocols, validators

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam as Tool
from tensorrt_llm.serve.tool_parser.base_tool_parser import BaseToolParser
from tensorrt_llm.serve.tool_parser.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)


def _normalize_schema_type(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = value.strip().lower()
    if value in ("str", "text", "varchar", "char", "enum"):
        return "string"
    if value.startswith(("int", "uint", "long", "short", "unsigned")):
        return "integer"
    if value.startswith(("num", "float")):
        return "number"
    if value in ("bool", "binary"):
        return "boolean"
    if value.startswith("dict"):
        return "object"
    if value == "arr" or value.startswith("list"):
        return "array"
    return value


def _normalize_parameter_schema(schema: Any) -> Any:
    """Normalize legacy type names only within schema nodes."""
    if not isinstance(schema, dict):
        return schema
    if any(key in schema for key in ("$ref", "$dynamicRef", "$recursiveRef")):
        raise ValueError("Parameter references require the enclosing tool schema.")
    normalized = dict(schema)
    if "type" in schema:
        value = schema["type"]
        normalized["type"] = (
            [_normalize_schema_type(item) for item in value]
            if isinstance(value, list)
            else _normalize_schema_type(value)
        )
    for key in (
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        "dependentSchemas",
        "dependencies",
    ):
        if isinstance(schema.get(key), dict):
            normalized[key] = {
                name: _normalize_parameter_schema(child) for name, child in schema[key].items()
            }
    for key in (
        "anyOf",
        "oneOf",
        "allOf",
        "prefixItems",
        "items",
        "additionalItems",
        "additionalProperties",
        "unevaluatedProperties",
        "unevaluatedItems",
        "contains",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
    ):
        if key in schema:
            value = schema[key]
            normalized[key] = (
                [_normalize_parameter_schema(child) for child in value]
                if isinstance(value, list)
                else _normalize_parameter_schema(value)
            )
    return normalized


def _is_json_value(value: Any) -> bool:
    """Reject Python literals whose JSON serialization would change their types."""
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _is_json_value(item) for key, item in value.items())
    return False


class Qwen3CoderToolParser(BaseToolParser):
    """Tool parser for Qwen 3 models.

    Assumes function call format:
        <tool_call>
        <function=execute_bash>
        <parameter=command>
        pwd && ls
        </parameter>
        </function>
        </tool_call>
    """

    def __init__(self):
        super().__init__()
        self.tool_call_start_token: str = "<tool_call>"
        self.tool_call_end_token: str = "</tool_call>"
        self.tool_call_prefix: str = "<function="
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
        )
        self.tool_call_function_regex = re.compile(
            r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)</parameter>|<parameter=(.*?)$", re.DOTALL
        )
        self._buf: str = ""
        self._parameter_validators: dict[tuple[str, str], protocols.Validator | None] = {}

        # Streaming state variables
        self._current_function_name: str = ""
        self._current_parameters: Dict[str, Any] = {}
        self._streamed_parameters: Dict[
            str, str
        ] = {}  # Track what parameter content we've streamed
        self._in_tool_call: bool = False
        self._function_name_sent: bool = False

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        normal, calls = self._extract(text, tools)
        return StreamingParseResult(normal_text=normal, calls=calls)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        self._buf += new_text
        normal = ""
        calls: List[ToolCallItem] = []

        # Build tool indices for validation
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        while True:
            # If we're not in a tool call and don't see a start token, return normal text
            if not self._in_tool_call and self.tool_call_start_token not in self._buf:
                normal += self._buf
                self._buf = ""
                break

            # Look for tool call start
            if not self._in_tool_call:
                s = self._buf.find(self.tool_call_start_token)
                if s == -1:
                    normal += self._buf
                    self._buf = ""
                    break

                normal += self._buf[:s]
                self._buf = self._buf[s:]

                self._in_tool_call = True
                self._function_name_sent = False
                self._current_function_name = ""
                self._current_parameters = {}
                self._streamed_parameters = {}

                # Remove the start token
                self._buf = self._buf[len(self.tool_call_start_token) :]
                continue

            # We're in a tool call, try to parse function name if not sent yet
            if not self._function_name_sent:
                # Look for function name pattern: <function=name>
                function_match = re.search(r"<function=([^>]+)>", self._buf)
                if function_match:
                    function_name = function_match.group(1).strip()

                    # Validate function name
                    if function_name:
                        self._current_function_name = function_name
                        self._function_name_sent = True

                        # Initialize tool call tracking
                        if self.current_tool_id == -1:
                            self.current_tool_id = 0

                        # Ensure tracking arrays are large enough
                        while len(self.prev_tool_call_arr) <= self.current_tool_id:
                            self.prev_tool_call_arr.append({})
                        while len(self.streamed_args_for_tool) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")

                        # Store tool call info
                        self.prev_tool_call_arr[self.current_tool_id] = {
                            "name": function_name,
                            "arguments": {},
                        }

                        # Send tool name with empty parameters
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=function_name,
                                parameters="",
                            )
                        )

                        # Remove the processed function declaration
                        self._buf = self._buf[function_match.end() :]
                        continue
                else:
                    # Function name not complete yet, wait for more text
                    break

            # Parse parameters incrementally
            if self._function_name_sent:
                # Process parameters and get any calls to emit
                param_config = self._get_arguments_config(self._current_function_name, tools)
                parameter_calls = self._parse_and_stream_parameters(self._buf, param_config)
                calls.extend(parameter_calls)

                # Check if tool call is complete
                if self.tool_call_end_token in self._buf:
                    end_pos = self._buf.find(self.tool_call_end_token)

                    # Add closing brace to complete the JSON object
                    current_streamed = self.streamed_args_for_tool[self.current_tool_id]
                    if current_streamed:
                        # Count opening and closing braces to check if JSON is complete
                        open_braces = current_streamed.count("{")
                        close_braces = current_streamed.count("}")
                        if open_braces > close_braces:
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=None,
                                    parameters="}",
                                )
                            )
                            self.streamed_args_for_tool[self.current_tool_id] = (
                                current_streamed + "}"
                            )

                    # Complete the tool call
                    self._buf = self._buf[end_pos + len(self.tool_call_end_token) :]
                    self._reset_streaming_state()
                    self.current_tool_id += 1
                    continue
                else:
                    # Tool call not complete yet, wait for more text
                    break

        return StreamingParseResult(normal_text=normal, calls=calls)

    def _parse_and_stream_parameters(
        self, text_to_parse: str, param_config: dict
    ) -> List[ToolCallItem]:
        """Parse complete parameter blocks from text and return any tool call items to emit.

        This method:
        1. Finds all complete <parameter> blocks
        2. Parses them into a dictionary
        3. Compares with current parameters and generates diff if needed
        4. Updates internal state

        Args:
            text_to_parse: The text to search for parameter blocks

        Returns:
            List of ToolCallItem objects to emit (may be empty)
        """
        calls: List[ToolCallItem] = []

        # Find all complete parameter patterns
        param_matches = list(
            re.finditer(r"<parameter=([^>]+)>(.*?)</parameter>", text_to_parse, re.DOTALL)
        )

        # Build new parameters dictionary
        new_params = {}
        for match in param_matches:
            param_name = match.group(1).strip()
            param_value = match.group(2)
            new_params[param_name] = self._convert_param_value(
                param_value, param_name, param_config, self._current_function_name
            )

        # Calculate parameter diff to stream with proper incremental JSON building
        if new_params != self._current_parameters:
            previous_args_json = self.streamed_args_for_tool[self.current_tool_id]

            # Build incremental JSON properly
            if not self._current_parameters:
                # First parameter(s) - start JSON object but don't close it yet
                items = []
                for key, value in new_params.items():
                    items.append(
                        f"{json.dumps(key, ensure_ascii=False)}: {json.dumps(value, ensure_ascii=False)}"
                    )
                json_fragment = "{" + ", ".join(items)

                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=None,
                        parameters=json_fragment,
                    )
                )
                self.streamed_args_for_tool[self.current_tool_id] = json_fragment

            else:
                # Additional parameters - add them incrementally
                new_keys = set(new_params.keys()) - set(self._current_parameters.keys())
                if new_keys:
                    # Build the continuation part (no closing brace yet)
                    continuation_parts = []
                    for key in new_keys:
                        value = new_params[key]
                        continuation_parts.append(
                            f"{json.dumps(key, ensure_ascii=False)}: {json.dumps(value, ensure_ascii=False)}"
                        )

                    json_fragment = ", " + ", ".join(continuation_parts)

                    calls.append(
                        ToolCallItem(
                            tool_index=self.current_tool_id,
                            name=None,
                            parameters=json_fragment,
                        )
                    )
                    self.streamed_args_for_tool[self.current_tool_id] = (
                        previous_args_json + json_fragment
                    )

            # Update current state
            self._current_parameters = new_params
            self.prev_tool_call_arr[self.current_tool_id]["arguments"] = new_params

        return calls

    def _reset_streaming_state(self):
        """Reset streaming state for the next tool call."""
        self._in_tool_call = False
        self._function_name_sent = False
        self._current_function_name = ""
        self._current_parameters = {}
        self._streamed_parameters = {}
        self.current_tool_name_sent = False

    def _extract(self, text: str, tools: List[Tool]) -> Tuple[str, List[ToolCallItem]]:
        normal_parts: List[str] = []
        calls: List[ToolCallItem] = []
        cursor = 0
        while True:
            s = text.find(self.tool_call_start_token, cursor)
            if s == -1:
                normal_parts.append(text[cursor:])
                break
            normal_parts.append(text[cursor:s])
            e = text.find(self.tool_call_end_token, s)
            if e == -1:
                normal_parts.append(text[s:])
                break
            block = text[s : e + len(self.tool_call_end_token)]
            cursor = e + len(self.tool_call_end_token)
            calls.extend(self._parse_block(block, tools))
        return "".join(normal_parts), calls

    def _parse_block(self, block: str, tools: List[Tool]) -> List[ToolCallItem]:
        res: List[ToolCallItem] = []
        for m in self.tool_call_function_regex.findall(block):
            txt = m[0] if m[0] else m[1]
            if ">" not in txt:
                continue
            idx = txt.index(">")
            fname = txt[:idx].strip()
            body = txt[idx + 1 :]
            params: Dict[str, Any] = {}
            for pm in self.tool_call_parameter_regex.findall(body):
                ptxt = pm[0] if pm[0] else pm[1]
                if ">" not in ptxt:
                    continue
                pidx = ptxt.index(">")
                pname = ptxt[:pidx].strip()
                pval = ptxt[pidx + 1 :].lstrip("\n").rstrip("\n")
                # Convert parameter value to the correct type.
                param_config = self._get_arguments_config(fname, tools)
                params[pname] = self._convert_param_value(pval, pname, param_config, fname)
            res.append(
                ToolCallItem(
                    tool_index=-1,
                    name=fname,
                    parameters=json.dumps(params, ensure_ascii=False),
                )
            )
        return res

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError

    def _get_arguments_config(self, func_name: str, tools: list[Tool] | None) -> dict:
        """Extract argument configuration for a function."""
        # Copy from `vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py`.
        if tools is None:
            return {}
        for config in tools:
            if not hasattr(config, "type") or not (
                hasattr(config, "function") and hasattr(config.function, "name")
            ):
                continue
            if config.type == "function" and config.function.name == func_name:
                if not hasattr(config.function, "parameters"):
                    return {}
                params = config.function.parameters
                if isinstance(params, dict) and "properties" in params:
                    return params["properties"]
                elif isinstance(params, dict):
                    return params
                else:
                    return {}
        logger.warning(f"Tool '{func_name}' is not defined in the tools list.")
        return {}

    def _convert_param_value(
        self, param_value: str, param_name: str, param_config: dict, func_name: str
    ) -> Any:
        """Preserve valid strings and accept only schema-valid conversions.

        The exact token ``null`` denotes null when allowed by the schema.
        ``None`` denotes null only when the schema rejects it as a string.
        """
        param_value = html.unescape(param_value.strip())
        if param_name not in param_config:
            if param_config != {}:
                logger.warning(
                    f"Parsed parameter '{param_name}' is not defined in the tool "
                    f"parameters for tool '{func_name}', directly returning the string value."
                )
            return param_value

        schema = param_config[param_name]
        if schema == {} or schema is True:
            return param_value
        key = (func_name, param_name)
        if key not in self._parameter_validators:
            try:
                normalized = _normalize_parameter_schema(schema)
                validator_type = validators.validator_for(normalized)
                validator_type.check_schema(normalized)
                self._parameter_validators[key] = validator_type(normalized)
            except (exceptions.SchemaError, TypeError, ValueError):
                self._parameter_validators[key] = None
                logger.warning(
                    f"Unsupported schema for parameter '{param_name}' in tool '{func_name}'; "
                    "preserving its text."
                )
        validator = self._parameter_validators[key]
        if validator is None:
            return param_value

        if param_value == "null" and validator.is_valid(None):
            return None
        if validator.is_valid(param_value):
            return param_value
        if param_value == "None" and validator.is_valid(None):
            return None

        for decode in (json.loads, int, float):
            try:
                candidate = decode(param_value)
            except (ValueError, TypeError, OverflowError):
                continue
            if _is_json_value(candidate) and validator.is_valid(candidate):
                return candidate
        if param_value.lower() in ("true", "false"):
            candidate = param_value.lower() == "true"
            if validator.is_valid(candidate):
                return candidate
        if param_value.startswith(("{", "[")):
            try:
                candidate = ast.literal_eval(param_value)
            except (ValueError, SyntaxError, TypeError):
                pass
            else:
                if (
                    isinstance(candidate, (dict, list))
                    and _is_json_value(candidate)
                    and validator.is_valid(candidate)
                ):
                    return candidate
        logger.warning(
            f"Cannot convert parameter '{param_name}' in tool '{func_name}' to its schema; "
            "preserving its text."
        )
        return param_value
