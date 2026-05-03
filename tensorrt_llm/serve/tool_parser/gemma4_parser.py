# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import re
from typing import Any, Dict, List

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import ChatCompletionToolsParam as Tool
from tensorrt_llm.serve.tool_parser.base_tool_parser import BaseToolParser
from tensorrt_llm.serve.tool_parser.core_types import (
    StreamingParseResult,
    StructureInfo,
    ToolCallItem,
    _GetInfoFunc,
)

# Gemma 4 wraps string values in this five-character delimiter instead of `"`.
_STRING_DELIM = '<|"|>'


def _gemma4_args_to_json(body: str) -> str:
    """Convert a Gemma 4 tool-call argument body to standard JSON.

    Input uses:
        - ``<|"|>`` delimiters around string literals (no regular ``"``).
        - Bare identifier keys in objects (``{key:value,...}``).
        - Standard JSON for numbers, booleans, arrays, and nested objects.

    The conversion replaces the custom string delimiter with ``"`` and then
    scans character-by-character to quote bare keys while skipping over
    string literals.
    """
    body = body.replace(_STRING_DELIM, '"')
    out: List[str] = []
    i = 0
    n = len(body)
    while i < n:
        ch = body[i]
        if ch == '"':
            out.append(ch)
            i += 1
            while i < n:
                if body[i] == "\\" and i + 1 < n:
                    out.append(body[i])
                    out.append(body[i + 1])
                    i += 2
                    continue
                out.append(body[i])
                if body[i] == '"':
                    i += 1
                    break
                i += 1
            continue
        if ch in "{,":
            out.append(ch)
            i += 1
            while i < n and body[i].isspace():
                out.append(body[i])
                i += 1
            j = i
            if j < n and (body[j].isalpha() or body[j] == "_"):
                while j < n and (body[j].isalnum() or body[j] == "_"):
                    j += 1
                k = j
                while k < n and body[k].isspace():
                    k += 1
                if k < n and body[k] == ":":
                    out.append('"')
                    out.append(body[i:j])
                    out.append('"')
                    i = j
                    continue
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _parse_gemma4_args(body: str) -> Dict[str, Any]:
    """Parse a Gemma 4 argument body (``{k:v,...}``) into a dict."""
    return json.loads(_gemma4_args_to_json(body))


class Gemma4ToolParser(BaseToolParser):
    """Tool parser for Gemma 4 models.

    Format (as emitted by the model, per the chat template):

    .. code-block::

        <|tool_call>call:FUNC_NAME{key1:value1,key2:value2}<tool_call|>

    Where:

    - ``<|tool_call>`` / ``<tool_call|>`` are registered special tokens that
      wrap each tool call.
    - The function name follows the literal prefix ``call:`` up to the first
      ``{``.
    - Arguments form a custom JSON-like object with bare identifier keys and
      ``<|"|>``-delimited string literals.

    Because the delimiter tokens are registered as special tokens,
    ``needs_raw_special_tokens = True`` is set so the server disables
    ``skip_special_tokens`` on the sampling params.
    """

    needs_raw_special_tokens = True

    def __init__(self):
        super().__init__()
        self.bot_token = "<|tool_call>"  # nosec B105
        self.eot_token = "<tool_call|>"  # nosec B105
        self.tool_call_separator = ""
        # Regex matches a single complete tool call block. The non-greedy
        # ``.*?`` backtracks past inner braces because the trailing literal
        # ``<tool_call|>`` anchors the end of the block.
        self._tool_call_regex = re.compile(
            rf"{re.escape(self.bot_token)}call:([^{{]+?)(\{{.*?\}}){re.escape(self.eot_token)}",
            re.DOTALL,
        )

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        idx = text.find(self.bot_token)
        if idx == -1:
            return StreamingParseResult(normal_text=text, calls=[])

        normal_text = text[:idx].rstrip()
        calls: List[ToolCallItem] = []
        for match in self._tool_call_regex.finditer(text):
            name = match.group(1).strip()
            args_body = match.group(2)
            try:
                args_dict = _parse_gemma4_args(args_body)
            except (ValueError, json.JSONDecodeError) as e:
                logger.warning(f"Failed to parse Gemma 4 tool call arguments {args_body!r}: {e}")
                continue
            calls.extend(self.parse_base_json({"name": name, "arguments": args_dict}, tools))
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        """Streaming incremental parser for Gemma 4 tool calls.

        Emits a tool call only after its entire ``<|tool_call>...<tool_call|>``
        block has arrived: the function name first (with empty parameters),
        followed by the full arguments JSON in a second call item. This
        mirrors the behavior of the MiniMax-M2 parser and avoids the need to
        parse Gemma 4's custom argument format incrementally.
        """
        self._buffer += new_text
        calls: List[ToolCallItem] = []
        normal_text_parts: List[str] = []

        if not hasattr(self, "_in_tool_call"):
            self._in_tool_call = False

        while True:
            if not self._in_tool_call:
                idx = self._buffer.find(self.bot_token)
                if idx == -1:
                    hold = self._ends_with_partial_token(self._buffer, self.bot_token)
                    emit_len = len(self._buffer) - hold
                    normal_text_parts.append(self._buffer[:emit_len])
                    self._buffer = self._buffer[emit_len:]
                    break
                normal_text_parts.append(self._buffer[:idx])
                self._buffer = self._buffer[idx + len(self.bot_token) :]
                self._in_tool_call = True
            else:
                end_idx = self._buffer.find(self.eot_token)
                if end_idx == -1:
                    break
                block = self._buffer[:end_idx]
                self._buffer = self._buffer[end_idx + len(self.eot_token) :]
                self._in_tool_call = False
                self._emit_tool_call_block(block, calls)

        return StreamingParseResult(
            normal_text="".join(normal_text_parts),
            calls=calls,
        )

    def _emit_tool_call_block(self, block: str, calls: List[ToolCallItem]) -> None:
        brace_idx = block.find("{")
        prefix = block[:brace_idx].strip() if brace_idx != -1 else block.strip()
        if brace_idx == -1 or not prefix.startswith("call:"):
            logger.warning(f"Malformed Gemma 4 tool call block: {block!r}")
            return
        name = prefix[len("call:") :].strip()
        args_body = block[brace_idx:]
        try:
            args_dict = _parse_gemma4_args(args_body)
            args_json = json.dumps(args_dict, ensure_ascii=False)
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to parse Gemma 4 tool call arguments {args_body!r}: {e}")
            args_json = "{}"

        self.current_tool_id += 1
        while len(self.streamed_args_for_tool) <= self.current_tool_id:
            self.streamed_args_for_tool.append("")

        calls.append(ToolCallItem(tool_index=self.current_tool_id, name=name, parameters=""))
        calls.append(ToolCallItem(tool_index=self.current_tool_id, name=None, parameters=args_json))
        self.streamed_args_for_tool[self.current_tool_id] = args_json

    def supports_structural_tag(self) -> bool:
        # Gemma 4's argument format is not standard JSON; structural-tag
        # guided decoding (which expects JSON after the begin pattern) would
        # not produce valid output.
        return False

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f"{self.bot_token}call:{name}{{",
            end=f"}}{self.eot_token}",
            trigger=self.bot_token,
        )
