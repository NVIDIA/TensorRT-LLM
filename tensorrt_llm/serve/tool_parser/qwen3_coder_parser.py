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

import ast
import html
import json
import re
from typing import Any, Dict, List, Tuple

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
        self.tool_call_parameter_regex = re.compile(
            r"<parameter=(.*?)</parameter>|<parameter=(.*?)$", re.DOTALL
        )
        self._buf: str = ""

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
            template = xgrammar.get_builtin_structural_tag(
                model="qwen_coder", tools=tool_dicts, reasoning=False
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

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        normal, calls = self._extract(text, tools)
        return StreamingParseResult(normal_text=normal, calls=calls)

    def parse_partial(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        normal, calls = self._extract(text, tools, complete=False)
        return StreamingParseResult(normal_text=normal, calls=calls)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        return self._parse_streaming(new_text, tools, complete=False)

    def _parse_streaming(
        self, new_text: str, tools: List[Tool], *, complete: bool
    ) -> StreamingParseResult:
        self._buf += new_text
        normal = ""
        calls: List[ToolCallItem] = []

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

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        result = self._parse_streaming("", tools, complete=True)
        if not self._in_tool_call:
            result.normal_text += self._buf
        self._buf = ""
        self._reset_streaming_state()
        return result

    def _parse_and_stream_parameters(
        self, text_to_parse: str, param_config: dict, *, complete: bool = False
    ) -> List[ToolCallItem]:
        """Emit argument deltas for the current function body.

        Once the body is complete, unconstrained output can recover its final
        parameter without a closing marker.
        """
        calls: List[ToolCallItem] = []

        new_params = {}
        pattern = (
            r"<parameter=([^>]+)>(.*?)(?:</parameter>|$)"
            if (complete and not self.grammar_constrained)
            else r"<parameter=([^>]+)>(.*?)</parameter>"
        )
        for match in re.finditer(pattern, text_to_parse, re.DOTALL):
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

    def _find_closing_tag(
        self, text: str, end_token: str, start: int = 0, *, complete: bool = False
    ) -> int:
        """Find an outer closer, recovering unclosed parameters in unconstrained output.

        A complete next-call header takes precedence over an unclosed parameter
        in unconstrained output. Constrained output keeps that markup as literal
        argument text. A final call can recover only when no parameter closer remains.
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
                return -1
            cursor = parameter_end + len("</parameter>")

    def _extract(
        self, text: str, tools: List[Tool], *, complete: bool = True
    ) -> Tuple[str, List[ToolCallItem]]:
        normal_parts: List[str] = []
        calls: List[ToolCallItem] = []
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

    def _parse_block(self, block: str, tools: List[Tool]) -> List[ToolCallItem]:
        res: List[ToolCallItem] = []
        cursor = 0
        while match := re.search(r"<function=([^<>]*)>", block[cursor:]):
            fname = match.group(1).strip()
            body_start = cursor + match.end()
            body_end = self._find_closing_tag(block, "</function>", body_start, complete=True)
            if body_end == -1:
                body_end = len(block)
            body = block[body_start:body_end]
            cursor = body_end + len("</function>")
            if not fname:
                continue
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
        """Convert parameter value based on its type in the schema."""
        # Copy and modify from `vllm/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py`.
        param_value = html.unescape(param_value.strip())
        if param_value.lower() == "null":
            return None

        if param_name not in param_config:
            if param_config != {}:
                logger.warning(
                    f"Parsed parameter '{param_name}' is not defined in the tool "
                    f"parameters for tool '{func_name}', directly returning the string value."
                )
            return param_value

        if isinstance(param_config[param_name], dict) and "type" in param_config[param_name]:
            param_type = str(param_config[param_name]["type"]).strip().lower()
        elif isinstance(param_config[param_name], dict) and "anyOf" in param_config[param_name]:
            # anyOf has no top-level "type"; treat as object to trigger json.loads.
            param_type = "object"
        else:
            param_type = "string"

        if param_type in ["string", "str", "text", "varchar", "char", "enum"]:
            return param_value
        elif (
            param_type.startswith("int")
            or param_type.startswith("uint")
            or param_type.startswith("long")
            or param_type.startswith("short")
            or param_type.startswith("unsigned")
        ):
            try:
                return int(param_value)
            except (ValueError, TypeError):
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not an "
                    f"integer in tool '{func_name}', degenerating to string."
                )
                return param_value
        elif param_type.startswith("num") or param_type.startswith("float"):
            try:
                float_param_value = float(param_value)
                return (
                    float_param_value
                    if float_param_value - int(float_param_value) != 0
                    else int(float_param_value)
                )
            except (ValueError, TypeError):
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a float "
                    f"in tool '{func_name}', degenerating to string."
                )
                return param_value
        elif param_type in ["boolean", "bool", "binary"]:
            param_value = param_value.lower()
            if param_value not in ["true", "false"]:
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' is not a boolean "
                    f"(`true` or `false`) in tool '{func_name}', degenerating to false."
                )
            return param_value == "true"
        else:
            if (
                param_type in ["object", "array", "arr"]
                or param_type.startswith("dict")
                or param_type.startswith("list")
            ):
                try:
                    param_value = json.loads(param_value)
                    return param_value
                except (json.JSONDecodeError, TypeError, ValueError):
                    logger.warning(
                        f"Parsed value '{param_value}' of parameter '{param_name}' cannot be "
                        f"parsed with json.loads in tool '{func_name}', will try other methods to parse it."
                    )
            try:
                param_value = ast.literal_eval(param_value)  # safer
            except (ValueError, SyntaxError, TypeError):
                logger.warning(
                    f"Parsed value '{param_value}' of parameter '{param_name}' cannot be "
                    f"converted via Python `ast.literal_eval()` in tool '{func_name}', degenerating to string."
                )
            return param_value
