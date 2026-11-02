# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Adapted from: https://raw.githubusercontent.com/sgl-project/sglang/d8fcbaa38da95201914a1277971044ee66837b26/python/sglang/srt/function_call/qwen3_coder_detector.py

import json
import math
import re
from typing import Any

import jsonschema
import xgrammar
from xgrammar import structural_tag

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam as Tool
from tensorrt_llm.serve.tool_parser.base_tool_parser import BaseToolParser
from tensorrt_llm.serve.tool_parser.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)

# Parameter values are typed the way vLLM's Qwen3 parser types them
# (extract_types_from_schema and coerce_to_schema_type in
# vllm/tool_parsers/utils.py), so the same model output yields the same
# arguments from both servers.

_TYPE_ALIASES: dict[str, str] = {
    "str": "string",
    "text": "string",
    "varchar": "string",
    "char": "string",
    "enum": "string",
    "int": "integer",
    "int32": "integer",
    "int64": "integer",
    "uint": "integer",
    "uint32": "integer",
    "uint64": "integer",
    "long": "integer",
    "short": "integer",
    "unsigned": "integer",
    "float": "number",
    "float32": "number",
    "float64": "number",
    "double": "number",
    "bool": "boolean",
    "dict": "object",
    "arr": "array",
    "list": "array",
    "sequence": "array",
}

# The order in which the types a parameter accepts are tried.
_TYPE_PRIORITY = ("null", "integer", "number", "boolean", "object", "array", "string")

# A parameter's opening tag, and the start of one that may be cut short.
_PARAMETER_OPENING_TAG = re.compile(r"<\s*parameter\s*=\s*([^>]*)>")
_PARAMETER_TAG_START = re.compile(r"<\s*parameter\s*=")
# A parameter that ends only at its closing tag, as grammar-constrained output
# writes every parameter.
_CLOSED_PARAMETER = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)


def _trim_wrapping_newlines(value: str) -> str:
    """Strip the one newline the chat template puts on each side of a value."""
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def _extract_types_from_schema(schema: Any) -> list[str]:
    """Return every JSON Schema type a parameter accepts.

    Reads ``type`` (a string or a list), infers types from ``enum`` values and
    recurses into ``anyOf``/``oneOf``/``allOf``. A schema that names no type
    accepts ``["string"]``.
    """
    if not isinstance(schema, dict):
        return ["string"]

    types = set()
    type_value = schema.get("type")
    if isinstance(type_value, str):
        types.add(type_value)
    elif isinstance(type_value, list):
        types.update(t for t in type_value if isinstance(t, str))

    enum_values = schema.get("enum")
    if isinstance(enum_values, list):
        for value in enum_values:
            if value is None:
                types.add("null")
            elif isinstance(value, bool):
                types.add("boolean")
            elif isinstance(value, int):
                types.add("integer")
            elif isinstance(value, float):
                types.add("number")
            elif isinstance(value, str):
                types.add("string")
            elif isinstance(value, list):
                types.add("array")
            elif isinstance(value, dict):
                types.add("object")

    for keyword in ("anyOf", "oneOf", "allOf"):
        choices = schema.get(keyword)
        if isinstance(choices, list):
            for choice in choices:
                types.update(_extract_types_from_schema(choice))

    return list(types) if types else ["string"]


def _is_json_finite(value: Any) -> bool:
    """Whether ``value`` serializes to valid JSON, i.e. holds no inf or nan."""
    try:
        json.dumps(value, allow_nan=False)
        return True
    except (ValueError, TypeError):
        return False


def _coerce_to_schema_type(value: str, schema_type: str | list[str]) -> Any:
    """Convert a raw parameter value to the first accepted type it parses as.

    Types are tried in the order null, integer, number, boolean, object, array,
    string. A value that fits none of them is decoded as JSON when possible and
    otherwise kept as the original string, as is a non-finite number, which
    JSON cannot represent. Python's ``None`` reads as null only where a string
    is not accepted, since otherwise it may be the text "None".
    """
    if isinstance(schema_type, str):
        schema_type = [schema_type]
    accepted = {_TYPE_ALIASES.get(key, key) for key in (t.strip().lower() for t in schema_type)}

    for candidate in _TYPE_PRIORITY:
        if candidate not in accepted:
            continue
        if candidate == "null":
            if value.lower() == "null" or (value == "None" and "string" not in accepted):
                return None
        elif candidate == "string":
            return value
        elif candidate == "integer":
            try:
                return int(value)
            except (ValueError, TypeError):
                continue
        elif candidate == "number":
            try:
                number = float(value)
            except (ValueError, TypeError):
                continue
            if math.isfinite(number):
                return number if number != int(number) else int(number)
        elif candidate == "boolean":
            lowered = value.lower().strip()
            if lowered in ("true", "1"):
                return True
            if lowered in ("false", "0"):
                return False
        else:  # object or array
            try:
                parsed = json.loads(value)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
            if _is_json_finite(parsed):
                return parsed

    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value
    return parsed if _is_json_finite(parsed) else value


class Qwen3CoderToolParser(BaseToolParser):
    """Parse tool calls with ``<function=...>`` and ``<parameter=...>`` tags.

    Qwen3-Coder, Nemotron 3 Nano, Nemotron 3 Super, Nemotron 3 Ultra, and
    Nemotron 3.5 Super VL chat templates use this format and share the
    ``qwen3_coder`` parser. Parameter values are converted using the supplied
    tool schemas.

    Format:
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
        # A parameter ends at its closing tag or, when the model leaves that
        # out, at the next parameter tag.
        self.tool_call_parameter_regex = re.compile(
            r"<\s*parameter\s*=\s*([^>]*)>(.*?)(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))",
            re.DOTALL,
        )
        self._buf: str = ""

        # Streaming state variables
        self._current_function_name: str = ""
        self._current_parameters: dict[str, Any] = {}
        self._streamed_parameters: dict[
            str, str
        ] = {}  # Track what parameter content we've streamed
        self._in_tool_call: bool = False
        self._function_name_sent: bool = False

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def build_required_structural_tag(
        self, tools: list[Tool], *, thinking: bool
    ) -> structural_tag.StructuralTag:
        """Build a mandatory grammar for Qwen-Coder XML tool calls.

        The serving layer must validate model eligibility before calling this;
        selecting this parser alone does not enable required tool choice.
        ``thinking`` must be resolved from the rendered prompt and indicates
        whether generation starts inside an already-open ``<think>`` block.
        """
        names = [tool.function.name for tool in tools]
        if not names or len(set(names)) != len(names):
            raise ValueError("Required tool calls need nonempty tools with unique names.")
        if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", name) for name in names):
            raise ValueError("Tool names must contain only letters, digits, '_', '-', or '.'.")
        tool_dicts = [tool.model_dump(exclude_none=True) for tool in tools]
        for tool in tool_dicts:
            schema = tool["function"].get("parameters") or {"type": "object"}
            try:
                jsonschema.validators.validator_for(schema).check_schema(schema)
            except jsonschema.SchemaError as exc:
                raise ValueError("Invalid required-tool parameter schema.") from exc
            if schema.get("type") != "object":
                raise ValueError("Required tools need object parameter schemas.")
            if schema.get("additionalProperties", False) is not False:
                raise ValueError("Required XML tools do not support additionalProperties.")
            tool["function"]["parameters"] = {"properties": {}, **schema}
        try:
            # XGrammar's builtin Qwen3-Coder XML tool-call format.
            template = xgrammar.get_builtin_structural_tag(
                model="qwen_3_coder", tools=tool_dicts, reasoning=False
            )
            for tag, tool in zip(template.format.tags, tool_dicts):
                schema = tool["function"]["parameters"]
                if not schema["properties"] and schema.keys() <= {
                    "type",
                    "properties",
                    "additionalProperties",
                    "required",
                    "minProperties",
                    "maxProperties",
                    "title",
                    "description",
                    "$schema",
                }:
                    # Empty argument lists have no XML parameter blocks.
                    jsonschema.validate({}, schema)
                    tag.content = structural_tag.ConstStringFormat(value="")
                else:
                    parameters = str(
                        xgrammar.Grammar.from_structural_tag(
                            structural_tag.StructuralTag(format=tag.content)
                        )
                    )
                    # Allow spacing after XML parameter terminals while preserving
                    # the excluded delimiter in string-value dispatch rules.
                    parameters = "\n".join(
                        re.sub(
                            r'"(?:\\.|[^"\\])*"',
                            lambda match: match[0] + r" [ \t\r\n]*"
                            if match[0] == '"</parameter>"'
                            else match[0],
                            line,
                        )
                        if "::=" in line
                        else line
                        for line in parameters.splitlines()
                    )
                    tag.content = structural_tag.GrammarFormat(grammar=parameters)
                tag.end = "</function>\n</tool_call>"
            calls = template.format
            calls.at_least_one = True
            calls.triggers = [self.tool_call_start_token]
            elements = [structural_tag.RegexFormat(pattern=r"[ \t\r\n]*"), calls]
            if thinking:
                # The opening <think> is already in the prompt.
                elements.insert(
                    0,
                    structural_tag.TagFormat(
                        begin="",
                        content=structural_tag.AnyTextFormat(excludes=[self.tool_call_start_token]),
                        end="</think>",
                    ),
                )
            template.format = structural_tag.SequenceFormat(elements=elements)
            xgrammar.Grammar.from_structural_tag(template)
        except (RuntimeError, jsonschema.ValidationError) as exc:
            raise ValueError("Invalid or unsupported required-tool parameter schema.") from exc
        return template

    def detect_and_parse(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        normal, calls = self._extract(text, tools)
        return StreamingParseResult(normal_text=normal, calls=calls)

    def parse_partial(self, text: str, tools: list[Tool]) -> StreamingParseResult:
        normal, calls = self._extract(text, tools, complete=False)
        return StreamingParseResult(normal_text=normal, calls=calls)

    def parse_streaming_increment(self, new_text: str, tools: list[Tool]) -> StreamingParseResult:
        return self._parse_streaming(new_text, tools, complete=False)

    def _parse_streaming(
        self, new_text: str, tools: list[Tool], *, complete: bool
    ) -> StreamingParseResult:
        self._buf += new_text
        normal = ""
        calls: list[ToolCallItem] = []

        # Build tool indices for validation
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        while True:
            # If we're not in a tool call and don't see a start token, return normal text
            if not self._in_tool_call and self.tool_call_start_token not in self._buf:
                partial = self._ends_with_partial_token(self._buf, self.tool_call_start_token)
                if partial:
                    normal += self._buf[:-partial]
                    self._buf = self._buf[-partial:]
                else:
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
                function_match = re.search(r"<function=([^<>]*)>", self._buf)
                if function_match:
                    function_name = function_match.group(1).strip()

                    if not function_name:
                        end_pos = self._find_closing_tag(
                            self._buf,
                            self.tool_call_end_token,
                            function_match.end(),
                            complete=complete,
                        )
                        if end_pos == -1:
                            # Keep the call buffered until all its markup can be discarded.
                            break
                        self._buf = self._buf[end_pos + len(self.tool_call_end_token) :]
                        self._reset_streaming_state()
                        continue
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
                # A single increment can contain several calls. Parse only this call's body.
                end_pos = self._find_closing_tag(
                    self._buf, self.tool_call_end_token, complete=complete
                )
                body = self._buf if end_pos == -1 else self._buf[:end_pos]
                if end_pos != -1:
                    function_end = self._find_closing_tag(
                        self._buf[: end_pos + len(self.tool_call_end_token)],
                        "</function>",
                        complete=True,
                    )
                    if function_end != -1:
                        body = self._buf[:function_end]
                parameter_calls = self._parse_and_stream_parameters(
                    body, param_config, complete=end_pos != -1
                )
                calls.extend(parameter_calls)

                # Check if tool call is complete
                if end_pos != -1:
                    # Parameter deltas leave the outer object open until the call closes.
                    current_streamed = self.streamed_args_for_tool[self.current_tool_id]
                    closing = "}" if current_streamed else "{}"
                    calls.append(
                        ToolCallItem(tool_index=self.current_tool_id, name=None, parameters=closing)
                    )
                    self.streamed_args_for_tool[self.current_tool_id] = current_streamed + closing

                    # Complete the tool call
                    self._buf = self._buf[end_pos + len(self.tool_call_end_token) :]
                    self._reset_streaming_state()
                    self.current_tool_id += 1
                    continue
                else:
                    # Tool call not complete yet, wait for more text
                    break

        return StreamingParseResult(normal_text=normal, calls=calls)

    def finish(self, tools: list[Tool]) -> StreamingParseResult:
        result = self._parse_streaming("", tools, complete=True)
        if not self._in_tool_call:
            result.normal_text += self._buf
        self._buf = ""
        self._reset_streaming_state()
        return result

    def _parse_and_stream_parameters(
        self, text_to_parse: str, param_config: dict, *, complete: bool = False
    ) -> list[ToolCallItem]:
        """Emit argument deltas for the current function body.

        Once the body is complete, unconstrained output can recover its final
        parameter without a closing marker.
        """
        calls: list[ToolCallItem] = []

        new_params = {
            name: self._convert_param_value(value, name, param_config, self._current_function_name)
            for name, value in self._parameter_values(
                text_to_parse, self._current_function_name, complete
            ).items()
        }

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
                # Additional parameters - add them incrementally, in the order
                # the model wrote them, as the non-streaming parse does.
                new_keys = [key for key in new_params if key not in self._current_parameters]
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

    def _find_closing_tag(
        self, text: str, end_token: str, start: int = 0, *, complete: bool = False
    ) -> int:
        """Find an outer closer, recovering unclosed parameters in unconstrained output.

        A complete next-call header takes precedence over an unclosed parameter
        in unconstrained output. Constrained output keeps that markup as literal
        argument text. A final call can recover only when no parameter closer remains;
        when its function also lacks ``</function>``, the unclosed parameter ends
        at a tool-call closer that ends the text.
        """
        cursor = start
        while True:
            end = text.find(end_token, cursor)
            parameter = text.find("<parameter=", cursor)
            if end == -1 or parameter == -1 or end < parameter:
                return end
            parameter_end = text.find("</parameter>", parameter)
            if not self.grammar_constrained:
                limit = parameter_end if parameter_end != -1 else len(text)
                for boundary in re.finditer(r"</function>\s*</tool_call>", text[parameter:limit]):
                    boundary_start = parameter + boundary.start()
                    boundary_end = parameter + boundary.end()
                    next_call = re.match(
                        r"\s*<tool_call>\s*<function=([^<>]+)>", text[boundary_end:limit]
                    )
                    if (next_call and next_call.group(1).strip()) or (
                        complete and parameter_end == -1
                    ):
                        return text.find(end_token, boundary_start, boundary_end)
            if parameter_end == -1:
                last = text.rfind(end_token)
                if (
                    complete
                    and not self.grammar_constrained
                    and end_token == self.tool_call_end_token
                    and not text[last + len(end_token) :].strip()
                ):
                    return last
                return -1
            cursor = parameter_end + len("</parameter>")

    def _extract(
        self, text: str, tools: list[Tool], *, complete: bool = True
    ) -> tuple[str, list[ToolCallItem]]:
        normal_parts: list[str] = []
        calls: list[ToolCallItem] = []
        cursor = 0
        while True:
            s = text.find(self.tool_call_start_token, cursor)
            if s == -1:
                normal_parts.append(text[cursor:])
                break
            normal_parts.append(text[cursor:s])
            e = self._find_closing_tag(text, self.tool_call_end_token, s, complete=complete)
            if e == -1:
                normal_parts.append(text[s:])
                break
            block = text[s : e + len(self.tool_call_end_token)]
            cursor = e + len(self.tool_call_end_token)
            calls.extend(self._parse_block(block, tools))
        return "".join(normal_parts), calls

    def _parse_block(self, block: str, tools: list[Tool]) -> list[ToolCallItem]:
        res: list[ToolCallItem] = []
        cursor = 0
        while match := re.search(r"<function=([^<>]*)>", block[cursor:]):
            fname = match.group(1).strip()
            body_start = cursor + match.end()
            body_end = self._find_closing_tag(block, "</function>", body_start, complete=True)
            if body_end == -1:
                # Without </function>, the body ends where the tool call does.
                body_end = len(block)
                if block.endswith(self.tool_call_end_token):
                    body_end -= len(self.tool_call_end_token)
            body = block[body_start:body_end]
            cursor = body_end + len("</function>")
            if not fname:
                continue
            params: dict[str, Any] = {}
            param_config = self._get_arguments_config(fname, tools)
            for pname, pval in self._parameter_values(body, fname, final=True).items():
                params[pname] = self._convert_param_value(pval, pname, param_config, fname)
            res.append(
                ToolCallItem(
                    tool_index=-1,
                    name=fname,
                    parameters=json.dumps(params, ensure_ascii=False),
                )
            )
        return res

    def _parameter_values(self, body: str, func_name: str, final: bool) -> dict[str, str]:
        """Return the raw value of each parameter in ``body``, in the order written.

        A value ends at its closing tag or at the next parameter's opening tag.
        When ``final`` is set, ``body`` is the whole function body, so a last
        parameter whose closing tag is missing ends where the body ends.
        Grammar-constrained output closes every parameter, so there a value
        runs to its closing tag and may hold parameter markup as literal text.
        """
        if self.grammar_constrained:
            return {
                match.group(1).strip(): _trim_wrapping_newlines(match.group(2))
                for match in _CLOSED_PARAMETER.finditer(body)
            }
        values = {}
        end = 0
        for match in self.tool_call_parameter_regex.finditer(body):
            values[match.group(1)] = _trim_wrapping_newlines(match.group(2))
            end = match.end()
        if not final:
            return values

        tail = body[end:]
        opening = _PARAMETER_OPENING_TAG.search(tail)
        if opening:
            name = opening.group(1)
            values[name] = _trim_wrapping_newlines(tail[opening.end() :])
            logger.warning(
                f"Parameter '{name}' of tool '{func_name}' has no closing </parameter> tag; "
                "its value runs to the end of the function."
            )
        elif _PARAMETER_TAG_START.search(tail):
            logger.warning(
                f"Dropping a parameter of tool '{func_name}' whose opening tag is incomplete."
            )
        return values

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError

    def _get_arguments_config(self, func_name: str, tools: list[Tool] | None) -> dict:
        """Return the property schemas of a tool's parameters, or {} if it declares none."""
        if tools is None:
            return {}
        for config in tools:
            if not hasattr(config, "type") or not (
                hasattr(config, "function") and hasattr(config.function, "name")
            ):
                continue
            if config.type == "function" and config.function.name == func_name:
                params = getattr(config.function, "parameters", None)
                properties = params.get("properties") if isinstance(params, dict) else None
                return properties if isinstance(properties, dict) else {}
        logger.warning(f"Tool '{func_name}' is not defined in the tools list.")
        return {}

    def _convert_param_value(
        self, param_value: str, param_name: str, param_config: dict, func_name: str
    ) -> Any:
        """Convert a parameter's raw value to the type its schema declares.

        A parameter without a schema keeps its raw text, as does every
        parameter of a tool that declares no properties.
        """
        schema = param_config.get(param_name)
        if not isinstance(schema, dict):
            if param_config:
                logger.warning(
                    f"Parsed parameter '{param_name}' is not defined in the tool "
                    f"parameters for tool '{func_name}', directly returning the string value."
                )
            return param_value
        return _coerce_to_schema_type(param_value, _extract_types_from_schema(schema))
