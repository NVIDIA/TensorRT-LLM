# Adapted from: https://raw.githubusercontent.com/sgl-project/sglang/d8fcbaa38da95201914a1277971044ee66837b26/python/sglang/srt/function_call/qwen3_coder_detector.py

import json
import math
import re
from typing import Any, Dict, List, Tuple, Union

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

_TYPE_ALIASES: Dict[str, str] = {
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


def _trim_wrapping_newlines(value: str) -> str:
    """Strip the one newline the chat template puts on each side of a value."""
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


def _extract_types_from_schema(schema: Any) -> List[str]:
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


def _coerce_to_schema_type(value: str, schema_type: Union[str, List[str]]) -> Any:
    """Convert a raw parameter value to the first accepted type it parses as.

    Types are tried in the order null, integer, number, boolean, object, array,
    string. A value that fits none of them is decoded as JSON when possible and
    otherwise kept as the original string, as is a non-finite number, which
    JSON cannot represent.
    """
    if isinstance(schema_type, str):
        schema_type = [schema_type]
    accepted = {_TYPE_ALIASES.get(key, key) for key in (t.strip().lower() for t in schema_type)}

    for candidate in _TYPE_PRIORITY:
        if candidate not in accepted:
            continue
        if candidate == "null":
            if value.lower() == "null":
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
            param_value = _trim_wrapping_newlines(match.group(2))
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
            param_config = self._get_arguments_config(fname, tools)
            for pm in self.tool_call_parameter_regex.findall(body):
                ptxt = pm[0] if pm[0] else pm[1]
                if ">" not in ptxt:
                    continue
                pidx = ptxt.index(">")
                pname = ptxt[:pidx].strip()
                pval = _trim_wrapping_newlines(ptxt[pidx + 1 :])
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
