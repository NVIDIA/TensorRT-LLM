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
import os
import re
from typing import Any, Dict, List

from tensorrt_llm.logger import logger

from ..openai_protocol import ChatCompletionToolsParam as Tool
from .base_tool_parser import BaseToolParser
from .core_types import StreamingParseResult, ToolCallItem, _GetInfoFunc


def _unescape_attr(value: str) -> str:
    return value.replace("&quot;", '"').replace("&amp;", "&")


def _escape_attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;")


def _parse_attrs(header: str) -> Dict[str, str]:
    return {key: _unescape_attr(value) for key, value in re.findall(r'(\w+)="([^"]*)"', header)}


class KimiK3ToolParser(BaseToolParser):
    """Detector for the Kimi K3 XTML function-call format."""

    needs_raw_special_tokens = True
    # Forced/named tool_choice has no grammar for XTML (no structural-tag
    # support), so the model output still carries preamble + markup and the
    # serving layer must extract instead of passing raw text through.
    extracts_forced_tool_calls = True

    def __init__(self):
        super().__init__()
        self.bot_token = "<|open|>tools<|sep|>"  # nosec B105
        self.eot_token = "<|close|>tools<|sep|>"  # nosec B105
        # Set once a complete tools section has been emitted. A K3 tools
        # section terminates the message, so anything streamed afterwards is
        # structural framing, not content.
        self._section_done = False
        # Structural leftovers that may trail the tools section when the
        # reasoning parser is not in front of this parser.
        self._trailing_structural = re.compile(
            r"(?:<\|close\|>message<\|sep\|>|<\|end_of_msg\|>)+\s*$"
        )

        # Tag headers run to the next special token. The encoder escapes only
        # ``&`` and ``"`` in attribute values, so a literal ``<`` (or ``>``)
        # can appear inside one; only ``<|`` is impossible without ending the
        # header, so headers match any text that doesn't contain ``<|``.
        attrs_pattern = r"(?:(?!<\|).)*?"
        self._call_open_regex = re.compile(r"<\|open\|>call(?![a-zA-Z])")
        self._call_regex = re.compile(
            r"<\|open\|>call(?P<attrs>" + attrs_pattern + r")<\|sep\|>"
            r"(?P<body>.*?)<\|close\|>call<\|sep\|>",
            re.DOTALL,
        )
        self._argument_regex = re.compile(
            r"<\|open\|>argument(?P<attrs>" + attrs_pattern + r")<\|sep\|>"
            r"(?P<value>.*?)<\|close\|>argument<\|sep\|>",
            re.DOTALL,
        )
        self._json_regex = re.compile(
            r"<\|open\|>json(?P<attrs>" + attrs_pattern + r")<\|sep\|>"
            r"(?P<value>.*?)<\|close\|>json<\|sep\|>",
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def supports_structural_tag(self) -> bool:
        # XTML argument bodies are tag-structured text, not JSON — the
        # generic begin/end/trigger structural-tag path does not apply.
        # Strict tools are handled by build_strict_structural_tag_format.
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError(
            "kimi_k3 XTML tool calls do not support structural-tag constrained decoding"
        )

    def build_strict_structural_tag_format(self, tools: List[Tool]) -> Dict[str, Any] | None:
        """Xgrammar structural-tag format enforcing well-formed K3 tool calls.

        Any generated tools section is constrained to calls of the declared
        tools; a strict tool with a parameters schema additionally gets its
        arguments constrained to that JSON Schema via the K3 json-block body
        form (the per-argument XTML form has no xgrammar equivalent).
        Non-strict tools keep free-form bodies. The outer triggered_tags
        must keep ``at_least_one``/``stop_after_first`` False: True would
        forbid the think/response text before the section and the message
        close after it, deadlocking generation.
        """
        if os.getenv("TRTLLM_KIMI_K3_STRICT_TOOL_GRAMMAR", "0") != "1":
            # Experimental, opt-in: under concurrent guided load with
            # production tool schemas, sampling tripped a device-side assert
            # and hard-killed the deployment (KVV schema suite, job 3054205:
            # TensorCompare.cu _assert_async in sampler.update_requests).
            # Root-cause investigation pending; strict tools fall back to
            # the warn-and-continue path meanwhile.
            return None
        if not tools:
            return None
        for tool in tools:
            if "<" in tool.function.name:
                # A literal '<' in an attribute value has no escaped form in
                # the K3 wire format (the checkpoint renderer escapes only
                # '&' and '"'), and the parser's attribute regex stops at
                # '<' — a grammar-forced call with such a name would be
                # dropped. Skip constrained decoding rather than teach the
                # model a dialect the reference renderer never produces.
                logger.warning(
                    "Tool name %r contains '<'; skipping the kimi_k3 "
                    "strict-tool grammar for this request.",
                    tool.function.name,
                )
                return None
        call_tags: List[Dict[str, Any]] = []
        for tool in tools:
            begin = f'<|open|>call tool="{_escape_attr(tool.function.name)}"'
            if tool.function.strict and tool.function.parameters:
                call_tags.append(
                    {
                        "type": "tag",
                        "begin": begin,
                        "content": {
                            "type": "sequence",
                            "elements": [
                                {
                                    "type": "regex",
                                    "pattern": ' index="[1-9][0-9]{0,2}"',
                                },
                                {
                                    "type": "const_string",
                                    "value": '<|sep|><|open|>json type="object"<|sep|>',
                                },
                                {
                                    "type": "json_schema",
                                    "json_schema": tool.function.parameters,
                                },
                            ],
                        },
                        "end": "<|close|>json<|sep|><|close|>call<|sep|>",
                    }
                )
            else:
                call_tags.append(
                    {
                        "type": "tag",
                        "begin": begin,
                        "content": {"type": "any_text"},
                        "end": "<|close|>call<|sep|>",
                    }
                )
        return {
            "type": "triggered_tags",
            "triggers": [self.bot_token],
            "tags": [
                {
                    "type": "tag",
                    "begin": self.bot_token,
                    "content": {
                        "type": "tags_with_separator",
                        "separator": "",
                        "at_least_one": True,
                        "tags": call_tags,
                    },
                    "end": self.eot_token,
                }
            ],
            "at_least_one": False,
            "stop_after_first": False,
        }

    @staticmethod
    def _coerce_value(value: str, value_type: str) -> Any:
        if value_type == "string":
            return value
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            logger.warning(
                f"kimi_k3 tool parser: argument declared type={value_type} but "
                "body is not valid JSON; keeping raw text"
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
        opened_calls = len(self._call_open_regex.findall(section))
        matched_calls = 0
        for position, match in enumerate(self._call_regex.finditer(section)):
            matched_calls += 1
            attrs = _parse_attrs(match.group("attrs"))
            name = attrs.get("tool")
            if not name:
                logger.warning(
                    f"kimi_k3 tool parser: call without tool attribute: {match.group('attrs')}"
                )
                continue
            if name not in tool_indices:
                logger.warning(f"Model attempted to call undefined function: {name}")
            calls.append(
                ToolCallItem(
                    tool_index=position,
                    name=name,
                    parameters=self._parse_call_arguments(match.group("body")),
                )
            )
        if matched_calls < opened_calls:
            logger.warning(
                f"kimi_k3 tool parser: {opened_calls - matched_calls} of "
                f"{opened_calls} call blocks were malformed or truncated and "
                "could not be parsed"
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
        if self._section_done:
            # The completed tools section terminated the K3 message; any later
            # text is structural framing (``<|close|>message<|sep|>``,
            # ``<|end_of_msg|>``), never user content. Buffer it so ``finish``
            # strips it — matching ``detect_and_parse`` — instead of emitting
            # protocol tokens as content.
            return StreamingParseResult()
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
        # The section terminates the message; hold any trailing framing in the
        # buffer for ``finish`` to strip rather than emitting it as content.
        self._buffer = self._buffer[section_end:]
        self._section_done = True
        return StreamingParseResult(
            normal_text=normal_text + result.normal_text, calls=result.calls
        )

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """Emit whatever the buffer holds when the stream ends early.

        ``parse_streaming_increment`` buffers the whole tools section until
        ``<|close|>tools<|sep|>``; if generation stops first (length limit,
        cancellation), the buffered content would otherwise be dropped.
        Complete call blocks are salvaged; a call truncated mid-block is
        reported by the malformed-call warning in ``_parse_tools_section``.
        """
        buffer, self._buffer = self._buffer, ""
        if not buffer:
            return StreamingParseResult()
        if self.bot_token not in buffer:
            # The buffer holds either a partial bot_token prefix or the
            # structural residue left after a completed tools section; the
            # stream is over, so it is plain text after stripping any
            # trailing structural tokens (matching detect_and_parse).
            return StreamingParseResult(normal_text=self._trailing_structural.sub("", buffer))
        logger.warning(
            f"kimi_k3 tool parser: stream ended before {self.eot_token}; "
            "parsing the partial tools section"
        )
        return self.detect_and_parse(buffer, tools)
