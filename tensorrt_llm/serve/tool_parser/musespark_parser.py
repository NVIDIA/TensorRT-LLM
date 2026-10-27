# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tool parser for the MuseSpark ``<atem:function_calls>`` format.

Behavior follows the reference implementation's tool parser; each
deliberate deviation from it has a case in
``tests/unittest/llmapi/apps/test_musespark_tool_parser.py``.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam as Tool
from tensorrt_llm.serve.tool_parser.base_tool_parser import BaseToolParser
from tensorrt_llm.serve.tool_parser.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)
from tensorrt_llm.serve.tool_parser.utils import infer_type_from_json_schema

_NAME = r"[A-Za-z0-9_.\-]+"


def _declared_type(param_name: str, func_name: str, tools: List[Tool]) -> Optional[str]:
    """JSON-schema type declared for ``param_name`` of ``func_name``, if any."""
    for tool in tools or ():
        if tool.function.name != func_name:
            continue
        parameters = tool.function.parameters
        if not isinstance(parameters, dict):
            continue
        properties = parameters.get("properties")
        if isinstance(properties, dict) and param_name in properties:
            return infer_type_from_json_schema(properties[param_name])
    return None


def _parse_param_value(raw: str, declared: Optional[str]) -> Any:
    """Type a parameter value the way the reference does.

    ``json.loads`` on the raw text, else keep the raw string (no trimming).

    One addition: a parameter the tool schema declares as ``string`` is kept
    raw even when its text happens to be valid JSON (``"quoted"``, ``1e3``,
    ``true``), so the model's string is not re-typed.
    """
    if declared == "string":
        return raw
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


class MuseSparkToolParser(BaseToolParser):
    r"""Tool parser for the MuseSpark model.

    Each tool call is its own assistant message whose body is one block::

        <|start|>assistant to=FN<|message|><atem:function_calls>
        <atem:invoke name="FN">
        <atem:parameter name="k">v</atem:parameter>
        </atem:invoke>
        </atem:function_calls><|eom|>            (<|eot|> on the last one)

    The function name is read from ``<atem:invoke name=...>``, not from the
    header. A block may hold several ``<atem:invoke>`` elements and a
    response may hold several blocks (parallel calls); tool indices run
    consecutively across all of them.

    Text reaching this parser is either the ``musespark`` reasoning parser's
    ``content`` (headers of tool messages replayed, end markers removed) or,
    with no reasoning parser configured, the raw decoded stream. Both are
    handled: routing headers (``<|start|>assistant to=X<|message|>`` or
    `` to=X<|message|>``) and the structural tokens ``<|start|>``,
    ``<|message|>``, ``<|eom|>``, ``<|eot|>``, ``<|end_of_text|>`` are never
    emitted as content, and a `` to=self<|message|>`` span is suppressed up to
    its ``<|eom|>``/``<|eot|>``.

    Streaming: the name of each invoke is emitted as soon as its opening tag
    is seen (``ToolCallItem(name=..., parameters="")``), then argument deltas
    follow as the parameters close. Intermediate JSON strings drop the
    trailing ``}`` so every emitted string is a prefix of the next; the final
    delta closes the object (``"{}"`` for a closed invoke with no params).
    """

    needs_raw_special_tokens = True

    START = "<|start|>"
    MESSAGE = "<|message|>"
    EOM = "<|eom|>"
    EOT = "<|eot|>"
    EOS = "<|end_of_text|>"
    STRUCTURAL_TOKENS = (START, MESSAGE, EOM, EOT, EOS)

    _HEADER_RE = re.compile(r"(?:<\|start\|>assistant)?(\s+to=(" + _NAME + r"))<\|message\|>")
    # Header still growing at the end of the buffer (no <|message|> yet).
    _HEADER_TAIL_RE = re.compile(r"(?:<\|start\|>assistant\s+|\s+)to=" + _NAME + r"\s*$")
    # Both paths anchor on the opening tag; the closer is found by search, so a
    # truncated invoke is still seen. See _iter_invokes.
    _INVOKE_START_RE = re.compile(r"<atem:invoke\s+name=(?:\"|')?(" + _NAME + r")(?:\"|')?[^>]*>")
    _PARAM_RE = re.compile(
        r"<atem:parameter\s+name=(?:\"|')?(" + _NAME + r")(?:\"|')?[^>]*>(.*?)</atem:parameter>",
        re.DOTALL,
    )
    _NON_TOOL_RECIPIENTS = frozenset({"self", "user", "commentary"})

    def __init__(self):
        super().__init__()
        self.bot_token = "<atem:function_calls>"  # nosec B105
        self.eot_token = "</atem:function_calls>"  # nosec B105
        self.invoke_end_token = "</atem:invoke>"  # nosec B105
        self._block_re = re.compile(
            re.escape(self.bot_token)
            + r"(.*?)"
            + re.escape(self.eot_token)
            + r"|"
            + re.escape(self.bot_token)
            + r"(.*)",
            re.DOTALL,
        )
        self._reset_streaming_state()

    def _reset_streaming_state(self) -> None:
        self._buffer = ""
        self._in_tool_block = False
        self._in_reasoning_block = False
        self._block_text = ""
        self._block_base_index = 0
        self._block_invoke_count = 0
        self.prev_tool_call_arr = []
        self.streamed_args_for_tool = []
        self.current_tool_id = -1
        self.current_tool_name_sent = False

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _find_first(text: str, markers: Tuple[str, ...]) -> Tuple[int, str]:
        best_pos, best = -1, ""
        for m in markers:
            pos = text.find(m)
            if pos >= 0 and (best_pos < 0 or pos < best_pos):
                best_pos, best = pos, m
        return best_pos, best

    @staticmethod
    def _partial_suffix_len(text: str, marker: str) -> int:
        for n in range(min(len(text), len(marker) - 1), 0, -1):
            if marker.startswith(text[-n:]):
                return n
        return 0

    @classmethod
    def _clean(cls, text: str) -> str:
        for tok in cls.STRUCTURAL_TOKENS:
            text = text.replace(tok, "")
        return text

    def _parse_params(
        self, invoke_content: str, func_name: str, tools: List[Tool]
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        for m in self._PARAM_RE.finditer(invoke_content):
            name, raw = m.group(1), m.group(2)
            params[name] = _parse_param_value(raw, _declared_type(name, func_name, tools))
        return params

    def _parse_invoke_content(
        self, invoke_content: str, func_name: str, tools: List[Tool]
    ) -> Dict[str, Any]:
        """Parse XML parameters, or a bare JSON object.

        The bare-JSON form is what structural-tag guided decoding produces
        between ``structure_info`` begin/end.
        """
        stripped = invoke_content.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                obj = json.loads(stripped)
                if isinstance(obj, dict):
                    return obj
            except (json.JSONDecodeError, ValueError):
                pass
        return self._parse_params(invoke_content, func_name, tools)

    def _iter_invokes(self, body: str) -> List[Tuple[str, str]]:
        """Every ``<atem:invoke>`` in a block body, terminated or not.

        Anchoring on the opening tag rather than on a matched open/close pair
        is what lets a truncated generation still yield the call it had begun.
        A generation cut off mid-invoke never emits ``</atem:invoke>``, and a
        pattern that requires the closer simply finds nothing -- the model
        asked for a tool and the caller is told no tool was requested.

        The streaming path already behaves this way: ``finish()`` passes
        ``block_complete=True`` and ``_update_tool_deltas`` parses the
        unterminated tail. Matching that here is what makes the two paths
        agree, which is the contract the serving smoke checks.

        An unterminated invoke runs to the start of the next invoke, or to the
        end of the body. ``_parse_invoke_content`` keeps only the parameters
        that did close, so a half-written one is dropped rather than guessed.
        """
        out: List[Tuple[str, str]] = []
        starts = list(self._INVOKE_START_RE.finditer(body))
        for i, start in enumerate(starts):
            content_from = start.end()
            content_to = starts[i + 1].start() if i + 1 < len(starts) else len(body)
            segment = body[content_from:content_to]
            closer = segment.find(self.invoke_end_token)
            if closer != -1:
                segment = segment[:closer]
            out.append((start.group(1), segment))
        return out

    # ------------------------------------------------------------ interface
    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        """Parse a complete message in one shot.

        ``normal_text`` is everything outside the tool blocks, with headers,
        ``to=self`` spans and structural tokens removed, so commentary prose
        before a call is kept.
        """
        expect_tool_calls = any(
            m.group(2) not in self._NON_TOOL_RECIPIENTS for m in self._HEADER_RE.finditer(text)
        )

        calls: List[ToolCallItem] = []
        segments: List[str] = []
        pos = 0
        for block in self._block_re.finditer(text):
            segments.append(text[pos : block.start()])
            pos = block.end()
            body = block.group(1) if block.group(1) is not None else block.group(2)
            for func_name, content in self._iter_invokes(body or ""):
                params = self._parse_invoke_content(content, func_name, tools)
                calls.append(
                    ToolCallItem(
                        tool_index=len(calls),
                        name=func_name,
                        parameters=json.dumps(params, ensure_ascii=False),
                    )
                )
        segments.append(text[pos:])

        if expect_tool_calls and not calls:
            # Reference raises ToolCallParsingException("Function tool call
            # is malformed!"); TRT-LLM parsers have no exception path, so
            # the text is returned as content and the event is logged.
            logger.warning(
                "MuseSpark tool parser: routing header addressed a tool but "
                "no <atem:invoke> block could be parsed."
            )

        normal_text = self._strip_headers_and_reasoning("".join(segments))
        return StreamingParseResult(normal_text=normal_text.strip(), calls=calls)

    def _strip_headers_and_reasoning(self, text: str) -> str:
        """Non-streaming cleanup of text outside tool blocks."""
        out: List[str] = []
        pos = 0
        while True:
            m = self._HEADER_RE.search(text, pos)
            if m is None:
                out.append(text[pos:])
                break
            out.append(text[pos : m.start()])
            pos = m.end()
            if m.group(2) == "self":
                end_pos, end_tok = self._find_first(text[pos:], (self.EOM, self.EOT))
                pos = len(text) if end_pos < 0 else pos + end_pos + len(end_tok)
        return self._clean("".join(out))

    # ------------------------------------------------------------- streaming
    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        self._buffer += new_text
        content_parts: List[str] = []
        calls: List[ToolCallItem] = []
        try:
            while True:
                if not self._in_tool_block:
                    if self._in_reasoning_block:
                        end_pos, end_tok = self._find_first(self._buffer, (self.EOM, self.EOT))
                        if end_pos >= 0:
                            self._buffer = self._buffer[end_pos + len(end_tok) :]
                            self._in_reasoning_block = False
                            continue
                        hold = max(
                            self._partial_suffix_len(self._buffer, self.EOM),
                            self._partial_suffix_len(self._buffer, self.EOT),
                        )
                        self._buffer = self._buffer[len(self._buffer) - hold :]
                        break

                    start_pos = self._buffer.find(self.bot_token)
                    if start_pos >= 0:
                        leading = self._buffer[:start_pos]
                        tail = self._HEADER_TAIL_RE.search(leading)
                        if tail is not None:
                            leading = leading[: tail.start()]
                        leading = self._strip_headers_and_reasoning(leading)
                        if leading:
                            content_parts.append(leading)
                        self._buffer = self._buffer[start_pos + len(self.bot_token) :]
                        self._in_tool_block = True
                        self._block_text = ""
                        self._block_base_index = len(self.streamed_args_for_tool)
                        self._block_invoke_count = 0
                        continue

                    hold = self._partial_suffix_len(self._buffer, self.bot_token)
                    safe = self._buffer[: len(self._buffer) - hold]
                    if safe:
                        parts, consumed, entered_reasoning = self._strip_header(safe)
                        content_parts.extend(parts)
                        if entered_reasoning:
                            self._in_reasoning_block = True
                        if consumed > 0:
                            # One delta can hold several messages; rescan the rest.
                            self._buffer = self._buffer[consumed:]
                            continue
                    break

                # inside a tool block
                end_pos = self._buffer.find(self.eot_token)
                if end_pos >= 0:
                    self._block_text += self._buffer[:end_pos]
                    self._buffer = self._buffer[end_pos + len(self.eot_token) :]
                    self._update_tool_deltas(calls, tools, block_complete=True)
                    self._in_tool_block = False
                    self._block_text = ""
                    self._block_invoke_count = 0
                    continue
                hold = self._partial_suffix_len(self._buffer, self.eot_token)
                safe_len = len(self._buffer) - hold
                if safe_len > 0:
                    self._block_text += self._buffer[:safe_len]
                    self._buffer = self._buffer[safe_len:]
                    self._update_tool_deltas(calls, tools)
                break

            normal_text = "".join(self._clean(p) for p in content_parts)
            return StreamingParseResult(normal_text=normal_text, calls=calls)
        except Exception as e:  # pragma: no cover - defensive, mirrors upstream
            logger.error(f"Error in MuseSpark parse_streaming_increment: {e}")
            return StreamingParseResult()

    def _strip_header(self, safe: str) -> Tuple[List[str], int, bool]:
        """Split the front of ``safe`` into content, through its first routing header.

        Returns ``(parts, consumed, entered_reasoning)``; ``consumed`` is how
        much of the buffer front is settled (0 = hold everything). At most one
        header is consumed per call and the caller rescans what follows it: a
        delta can hold several messages, and every header in it, including a
        later `` to=self``, must be stripped.
        """
        parts: List[str] = []
        start_pos = safe.find(self.START)
        m = self._HEADER_RE.search(safe)
        if m is not None and (start_pos < 0 or m.start() <= start_pos):
            # The first message after the generation prompt has no <|start|>
            # (`` to=X<|message|>``); a later one includes it in the match.
            if m.start() > 0:
                parts.append(safe[: m.start()])
            return parts, m.end(), m.group(2) == "self"
        if start_pos >= 0:
            # A <|start|> whose header is not a recipient header: drop the
            # header once its <|message|> arrives, hold it until then.
            if start_pos > 0:
                parts.append(safe[:start_pos])
            msg_pos = safe.find(self.MESSAGE, start_pos)
            if msg_pos < 0:
                return parts, start_pos, False
            return parts, msg_pos + len(self.MESSAGE), False

        tail = self._HEADER_TAIL_RE.search(safe)
        if tail is not None:
            if tail.start() > 0:
                parts.append(safe[: tail.start()])
            return parts, tail.start(), False

        hold = self._header_hold_len(safe)
        emit_len = len(safe) - hold
        if emit_len > 0:
            parts.append(safe[:emit_len])
        return parts, emit_len, False

    # Header still growing: ``[<|start|>assistant][ to=NAME][<|mess``.
    _HEADER_PREFIX_RE = re.compile(
        r"^(?:<\|start\|>assistant)?(?:\s*(?:t|to|to=[A-Za-z0-9_.\-]*)?)$"
    )

    @classmethod
    def _header_hold_len(cls, safe: str) -> int:
        """Number of trailing characters to hold back rather than emit.

        The longest suffix of ``safe`` that could still grow into a routing
        header or a structural token. Unlike the reference, which holds
        `` to=`` partials only from 3 chars, every prefix is held; the cost
        is a one-chunk delay on text ending in whitespace or `` t``.
        """
        hold = max(cls._partial_suffix_len(safe, tok) for tok in cls.STRUCTURAL_TOKENS)
        for n in range(len(safe), hold, -1):
            suffix = safe[-n:]
            core = suffix[: len(suffix) - cls._partial_suffix_len(suffix, cls.MESSAGE)]
            if "<|start|>assistant".startswith(core) or cls._HEADER_PREFIX_RE.match(core):
                return n
        return hold

    def _update_tool_deltas(
        self, calls: List[ToolCallItem], tools: List[Tool], block_complete: bool = False
    ) -> None:
        invokes = list(self._INVOKE_START_RE.finditer(self._block_text))

        while self._block_invoke_count < len(invokes):
            local = self._block_invoke_count
            idx = self._block_base_index + local
            func_name = invokes[local].group(1)
            self.current_tool_id = idx
            self.streamed_args_for_tool.append("")
            self.prev_tool_call_arr.append({"name": func_name, "arguments": {}})
            self.current_tool_name_sent = True
            self._block_invoke_count += 1
            calls.append(ToolCallItem(tool_index=idx, name=func_name, parameters=""))

        for local, inv in enumerate(invokes[: self._block_invoke_count]):
            idx = self._block_base_index + local
            func_name = inv.group(1)
            content_start = inv.end()
            content_end = (
                invokes[local + 1].start() if local + 1 < len(invokes) else len(self._block_text)
            )
            invoke_content = self._block_text[content_start:content_end]
            close_pos = invoke_content.find(self.invoke_end_token)
            closed = close_pos >= 0
            parsable = invoke_content[:close_pos] if closed else invoke_content

            params = (
                self._parse_invoke_content(parsable, func_name, tools)
                if (closed or block_complete)
                else self._parse_params(parsable, func_name, tools)
            )
            if idx < len(self.prev_tool_call_arr):
                self.prev_tool_call_arr[idx]["arguments"] = params

            prev = self.streamed_args_for_tool[idx]
            if params:
                full = json.dumps(params, ensure_ascii=False)
                cur = full if (closed or block_complete) else full[:-1]
            elif closed or block_complete:
                cur = "{}"
            else:
                continue
            if len(cur) > len(prev) and cur.startswith(prev):
                self.streamed_args_for_tool[idx] = cur
                calls.append(ToolCallItem(tool_index=idx, parameters=cur[len(prev) :]))

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        """Flush at end of stream.

        Closes an unterminated block (a truncated ``</atem:function_calls>``)
        and releases held-back plain text.
        """
        calls: List[ToolCallItem] = []
        normal_text = ""
        if self._in_tool_block:
            self._block_text += self._buffer
            self._buffer = ""
            self._update_tool_deltas(calls, tools, block_complete=True)
            self._in_tool_block = False
        elif not self._in_reasoning_block and self._buffer:
            leftover = self._buffer
            self._buffer = ""
            tail = self._HEADER_TAIL_RE.search(leftover)
            if tail is not None:
                leftover = leftover[: tail.start()]
            normal_text = self._clean(leftover)
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    # ------------------------------------------------------- structural tags
    def supports_structural_tag(self) -> bool:
        return True

    def structure_info(self) -> _GetInfoFunc:
        # Guided decoding fills a JSON object between begin and end;
        # ``_parse_invoke_content`` accepts that bare-JSON form.
        return lambda name: StructureInfo(
            begin=f'<atem:function_calls>\n<atem:invoke name="{name}">\n',
            end="\n</atem:invoke>\n</atem:function_calls>",
            trigger="<atem:function_calls>",
        )
