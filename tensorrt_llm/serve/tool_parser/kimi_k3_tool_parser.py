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
every other type.

The grammar itself (where a section, a call and a value end) is
``KimiK3ReasoningParser``'s, shared so the reasoning parser in front of this
one reads the same sections. Streaming buffers a section until it ends and
then emits its calls at once; ``detect_and_parse`` is that same machine run
over the whole text, so the two views of a generation agree for every
chunking. Nothing the model generated is dropped: a section without a
well-formed call, a call cut off mid-block and prose between calls all come
out as text, and so does text after a section.
"""

import json
import os
from typing import Any, Dict, List

from tensorrt_llm.llmapi.reasoning_parser import KimiK3ReasoningParser
from tensorrt_llm.logger import logger

from ..openai_protocol import ChatCompletionToolsParam as Tool
from .base_tool_parser import BaseToolParser
from .core_types import StreamingParseResult, ToolCallItem, _GetInfoFunc

# The K3 XTML grammar, shared with the reasoning parser.
_XTML = KimiK3ReasoningParser


def _unescape_attr(value: str) -> str:
    return value.replace("&quot;", '"').replace("&amp;", "&")


def _escape_attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;")


def _parse_attrs(header: str) -> Dict[str, str]:
    return _XTML.parse_xtml_attrs(header)


def _reject_constant(token: str) -> Any:
    raise ValueError(f"{token} is not JSON")


def _is_json(text: str) -> bool:
    """Whether `text` is a JSON document (Python's json also takes NaN/Infinity; JSON does not)."""
    try:
        json.loads(text, parse_constant=_reject_constant)
    except (ValueError, RecursionError):
        return False
    return True


class KimiK3ToolParser(BaseToolParser):
    """Detector for the Kimi K3 XTML function-call format."""

    needs_raw_special_tokens = True
    # Forced/named tool_choice has no grammar for XTML (no structural-tag
    # support), so the model output still carries preamble + markup and the
    # serving layer must extract instead of passing raw text through.
    extracts_forced_tool_calls = True

    def __init__(self):
        super().__init__()
        self.bot_token = _XTML.TOOLS_OPEN  # nosec B105
        self.eot_token = _XTML.TOOLS_END  # nosec B105
        # _buffer holds an open tools section from its opening tag on.
        self._in_section = False
        # Characters of the open section already known not to end it.
        self._section_checked = 0
        # Calls are numbered across every section of the generation: the
        # streaming assembly keys call fragments by tool_index.
        self._next_tool_index = 0

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
        must keep `at_least_one`/`stop_after_first` False: True would
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
                    f"Tool name {tool.function.name!r} contains '<'; "
                    "skipping the kimi_k3 strict-tool grammar for this request."
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

    def _ends_with_partial_token(self, buffer: str, bot_token: str) -> int:
        """Length of the LONGEST suffix of `buffer` that is a proper prefix of `bot_token`.

        The base class returns the shortest one, which for ``Hi<|open|>tools<``
        is the final ``<`` - and releases ``<|open|>tools`` as text one
        character before the opening tag completes.
        """
        for length in range(min(len(buffer), len(bot_token) - 1), 0, -1):
            if bot_token.startswith(buffer[-length:]):
                return length
        return 0

    @staticmethod
    def _call_arguments(text: str, call: "KimiK3ReasoningParser.XtmlItem") -> str:
        """The OpenAI ``function.arguments`` JSON string of a call.

        Values keep the model's own JSON text wherever it is JSON: a
        re-serialization would rewrite ``1e3`` as ``1000.0`` and ``1.10`` as
        ``1.1``. String-typed values (and invalid JSON, with a warning) are
        JSON-quoted raw text.
        """
        elements = call.elements
        if elements and elements[0].kind == "json":
            raw = text[elements[0].value_start : elements[0].value_end]
            if not _is_json(raw):
                logger.warning(
                    "kimi_k3 tool parser: json block is not valid JSON; passing raw text through"
                )
            return raw
        members: Dict[str, str] = {}
        for element in elements:
            key = element.attrs["key"]
            raw = text[element.value_start : element.value_end]
            value_type = element.attrs.get("type", "string")
            if value_type == "string":
                encoded = json.dumps(raw, ensure_ascii=False)
            elif _is_json(raw):
                encoded = raw
            else:
                logger.warning(
                    f"kimi_k3 tool parser: argument declared type={value_type} but "
                    "body is not valid JSON; keeping raw text"
                )
                encoded = json.dumps(raw, ensure_ascii=False)
            if key in members:
                logger.warning(
                    f"kimi_k3 tool parser: argument {key!r} repeated in one call; "
                    "keeping the last value"
                )
            members[key] = encoded
        return (
            "{"
            + ", ".join(
                f"{json.dumps(key, ensure_ascii=False)}: {value}" for key, value in members.items()
            )
            + "}"
        )

    def _emit_section(
        self,
        text: str,
        section: "KimiK3ReasoningParser.XtmlSection",
        tools: List[Tool],
        out_text: List[str],
        out_calls: List[ToolCallItem],
    ) -> None:
        """Deliver a section's calls; everything else in it goes out as text."""
        if not any(item.deliverable for item in section.items):
            # Not a single call a client could run: prose about the format,
            # or a call cut off before it was complete. Its bytes are model
            # output and go out as the model wrote them.
            logger.warning(
                "kimi_k3 tool parser: tools section holds no well-formed call; "
                f"releasing its {section.end} characters as text"
            )
            out_text.append(text[: section.end])
            return
        if not section.terminated:
            logger.warning(
                f"kimi_k3 tool parser: tools section never closed with {self.eot_token}; "
                "delivering its complete calls"
            )
        tool_indices = self._get_tool_indices(tools)
        released = 0
        for item in section.items:
            if item.deliverable:
                name = item.attrs["tool"]
                if name not in tool_indices:
                    logger.warning(f"Model attempted to call undefined function: {name}")
                out_calls.append(
                    ToolCallItem(
                        tool_index=self._next_tool_index,
                        name=name,
                        parameters=self._call_arguments(text, item),
                    )
                )
                self._next_tool_index += 1
                continue
            piece = text[item.start : item.end]
            if item.kind == "gap" and piece.isspace():
                # Whitespace between two tags of the section is formatting.
                continue
            out_text.append(piece)
            released += len(piece)
        if released:
            logger.warning(
                f"kimi_k3 tool parser: releasing {released} characters of the tools "
                "section that are not a well-formed call as text"
            )

    def _step_text(self, final: bool, out_text: List[str]) -> bool:
        """Emit text up to the next tools section; True once one has opened."""
        buf = self._buffer
        search = 0
        while True:
            start = buf.find(self.bot_token, search)
            if start == -1:
                break
            opens = _XTML.tools_section_opens(buf, start + len(self.bot_token), final)
            if opens is False:
                # The tag is not followed by a call: the model is writing
                # about the format, and the tag is text.
                search = start + 1
                continue
            out_text.append(buf[:start])
            self._buffer = buf[start:]
            if opens is None:
                return False
            self._in_section = True
            self._section_checked = 0
            return True
        if final:
            out_text.append(_XTML.strip_trailing_framing(buf))
            self._buffer = ""
            return False
        # Hold back whatever could still become an opening tag, or the
        # framing that ends the generation (it is stripped, never emitted).
        hold = max(
            self._ends_with_partial_token(buf, self.bot_token),
            _XTML.trailing_framing_hold(buf),
        )
        out_text.append(buf[: len(buf) - hold])
        self._buffer = buf[len(buf) - hold :]
        return False

    def _step_section(
        self, final: bool, tools: List[Tool], out_text: List[str], out_calls: List[ToolCallItem]
    ) -> bool:
        """Emit the open section once it has ended; True if it did."""
        buf = self._buffer
        if final:
            buf = _XTML.strip_trailing_framing(buf)
        elif not _XTML.section_may_end(buf, self._section_checked):
            self._section_checked = len(buf)
            return False
        section = _XTML.scan_tools_section(buf, len(self.bot_token), final)
        if section is None:
            self._section_checked = len(buf)
            return False
        self._emit_section(buf, section, tools, out_text, out_calls)
        self._buffer = buf[section.end :]
        self._in_section = False
        return True

    def _drain(self, tools: List[Tool], final: bool) -> StreamingParseResult:
        out_text: List[str] = []
        out_calls: List[ToolCallItem] = []
        while True:
            if not self._in_section:
                if not self._step_text(final, out_text):
                    break
            elif not self._step_section(final, tools, out_text, out_calls):
                break
            elif out_calls and not final:
                # A result cannot say whether its text came before or after
                # its calls, and the serving layer places text first. What
                # follows this section waits for the next increment (or for
                # the end-of-stream drain), so each result reads in order.
                break
        return StreamingParseResult(normal_text="".join(out_text), calls=out_calls)

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Parse a whole generation exactly the way the stream reads it.

        The serving layer rebuilds the final response from this, and the
        client already holds what the stream sent; running the streaming
        machine over the full text makes the two readings one.
        """
        reader = type(self)()
        head = reader.parse_streaming_increment(text, tools)
        tail = reader.finish(tools)
        calls = head.calls + tail.calls
        self.prev_tool_call_arr = []
        for call in calls:
            try:
                arguments = json.loads(call.parameters)
            except json.JSONDecodeError:
                arguments = call.parameters
            self.prev_tool_call_arr.append({"name": call.name, "arguments": arguments})
        return StreamingParseResult(normal_text=head.normal_text + tail.normal_text, calls=calls)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """Emit text once it cannot start a tools section, calls once their section ends.

        A K3 section comes last in its message, so buffering it whole costs
        no latency on the calls and avoids partial-argument reconstruction.
        """
        self._buffer += new_text
        return self._drain(tools, final=False)

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """Decide everything still held: the generation has ended.

        A section the model never closed delivers its complete calls; a call
        cut off mid-block goes out as text, as does a held-back prefix of an
        opening tag. Framing that ends the generation is stripped.
        """
        result = self._drain(tools, final=True)
        self._buffer = ""
        return result
