# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from tensorrt_llm.serve.tool_parser.utils import infer_type_from_json_schema

# M3's chat template prefixes every tool-call-related tag with this literal
# namespace string (the ``ns_token`` in chat_template.jinja). It is just
# an opaque sentinel chosen so the tag set cannot collide with anything
# the model might naturally produce in regular text.
_M3_NS = "]<]minimax[>["
_M3_NS_RE = re.escape(_M3_NS)


def _get_param_type(param_name: str, func_name: str, tools: List[Tool]) -> Optional[str]:
    """Look up the declared JSON-schema type of a parameter, if any."""
    for tool in tools:
        if tool.function.name != func_name:
            continue
        parameters = tool.function.parameters
        if not isinstance(parameters, dict):
            continue
        properties = parameters.get("properties")
        if not isinstance(properties, dict):
            continue
        if param_name not in properties:
            continue
        return infer_type_from_json_schema(properties[param_name])
    return None


def _coerce_value(value_str: str, param_type: Optional[str]) -> Any:
    """Coerce a string parameter value into the type declared by the tool schema.

    Mirrors :func:`tensorrt_llm.serve.tool_parser.minimax_m2_parser._parse_param_value`
    so that the two MiniMax variants share the same type-coercion
    behavior. Declared strings are kept verbatim; structured types
    (object / array / null) attempt JSON first; numerics and booleans
    fall back to their textual conversion. Anything else stays a string.
    """
    value_str = value_str.strip()

    if param_type == "string":
        return value_str

    try:
        return json.loads(value_str)
    except (json.JSONDecodeError, ValueError):
        pass

    if param_type in ("number", "integer"):
        try:
            if "." in value_str or "e" in value_str.lower():
                return float(value_str)
            return int(value_str)
        except (ValueError, TypeError):
            pass

    if param_type == "boolean":
        if value_str.lower() == "true":
            return True
        if value_str.lower() == "false":
            return False

    return value_str


class MiniMaxM3ToolParser(BaseToolParser):
    r"""Tool parser for MiniMax-M3 models.

    The M3 chat template renders tool calls with every tag prefixed by
    the literal namespace string ``]<]minimax[>[`` (chosen because it is
    extremely unlikely to appear in natural text). A complete invocation
    block looks like::

        ]<]minimax[>[<tool_call>
        ]<]minimax[>[<invoke name="function_name">
        ]<]minimax[>[<param1>value1]<]minimax[>[</param1>
        ]<]minimax[>[<param2>value2]<]minimax[>[</param2>
        ]<]minimax[>[</invoke>
        ]<]minimax[>[</tool_call>

    Compared with the M2 format (handled by
    :class:`MiniMaxM2ToolParser`) there are two differences this parser
    must absorb:

    1. The opening / closing block uses the namespaced ``<tool_call>``
       tag instead of ``<minimax:tool_call>``.
    2. Each parameter is rendered as ``<key>value</key>`` where the key
       *is* the tag name; M2's ``<parameter name="key">value</parameter>``
       form does not apply.

    The parser implements one-shot ``detect_and_parse`` (used by the
    non-streaming chat completion path); streaming is left to the
    fallback that emits the raw text — the M3 production smoke tests
    use non-streaming tool calls and a richer streaming implementation
    can be added once the non-streaming path is proven.
    """

    def __init__(self):
        super().__init__()
        self.bot_token = f"{_M3_NS}<tool_call>"  # nosec B105
        self.eot_token = f"{_M3_NS}</tool_call>"  # nosec B105

        self.tool_call_block_regex = re.compile(
            f"{_M3_NS_RE}<tool_call>(.*?){_M3_NS_RE}</tool_call>",
            re.DOTALL,
        )
        self.invoke_regex = re.compile(
            f"{_M3_NS_RE}<invoke\\s+name=[\"']?(.*?)[\"']?\\s*>(.*?){_M3_NS_RE}</invoke>",
            re.DOTALL,
        )
        # Match ``]<]minimax[>[<key>value]<]minimax[>[</key>`` where the
        # key is the tag name. Backreference ``\1`` forces the opening
        # and closing tags to use the same name so we do not accidentally
        # span across sibling parameters.
        self.param_regex = re.compile(
            f"{_M3_NS_RE}<([^/\\s>]+)>(.*?){_M3_NS_RE}</\\1>",
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        """Return True iff ``text`` contains an M3 tool-call opening tag."""
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """One-shot parse: extract every tool call, return remaining prose.

        The text before the first ``]<]minimax[>[<tool_call>`` opener and
        the text after the last ``]<]minimax[>[</tool_call>`` closer are
        joined into ``normal_text`` (matching the M2 parser's contract).
        Each ``<invoke name=...>`` block produces one :class:`ToolCallItem`
        with the parameter dict JSON-encoded.
        """
        idx = text.find(self.bot_token)
        if idx == -1:
            return StreamingParseResult(normal_text=text, calls=[])

        prefix = text[:idx].strip()
        suffix = ""
        eot_idx = text.rfind(self.eot_token)
        if eot_idx != -1:
            suffix = text[eot_idx + len(self.eot_token) :].strip()

        if prefix and suffix:
            normal_text = f"{prefix} {suffix}"
        else:
            normal_text = prefix or suffix

        calls: List[ToolCallItem] = []
        tool_index = 0
        for block in self.tool_call_block_regex.findall(text):
            for func_name, params_text in self.invoke_regex.findall(block):
                func_name = func_name.strip()
                arguments: Dict[str, Any] = {}
                for param_name, param_value in self.param_regex.findall(params_text):
                    param_name = param_name.strip()
                    arguments[param_name] = _coerce_value(
                        param_value, _get_param_type(param_name, func_name, tools)
                    )
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
        """Buffer text until a complete tool-call block is seen, then parse.

        M3's parameter values may be entire nested XML sub-trees, so a
        token-level incremental parser is significantly more involved
        than M2's flat ``<parameter>``-tag list. For now this method
        falls back to a simple buffered strategy: accumulate text until
        a ``</tool_call>`` is observed and then defer to
        :meth:`detect_and_parse`. Prose tokens that arrive before any
        ``<tool_call>`` opener are flushed as normal text so non-tool
        responses still stream incrementally.
        """
        self._buffer += new_text
        current_text = self._buffer

        if not self.has_tool_call(current_text):
            # Hold back any tail that could be the start of bot_token,
            # otherwise flush as normal text.
            partial = self._ends_with_partial_token(current_text, self.bot_token)
            if partial:
                return StreamingParseResult()
            self._buffer = ""
            return StreamingParseResult(normal_text=current_text)

        if self.eot_token not in current_text:
            # Still inside an open tool-call block; wait for the closer.
            return StreamingParseResult()

        try:
            result = self.detect_and_parse(current_text, tools)
        except Exception as exc:
            logger.error(f"Error in MiniMaxM3 parse_streaming_increment: {exc}")
            return StreamingParseResult(normal_text=current_text)

        self._buffer = ""
        return result

    def supports_structural_tag(self) -> bool:
        """Structural-tag-driven decoding is not implemented for M3 yet."""
        return False

    def structure_info(self) -> _GetInfoFunc:
        """Structural-tag-driven decoding is not implemented for M3 yet."""
        raise NotImplementedError()
