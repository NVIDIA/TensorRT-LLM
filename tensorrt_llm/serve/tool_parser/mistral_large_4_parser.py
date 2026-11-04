# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tool-call parser for Mistral-Large-4 output.

Mistral-Large-4 (mistral-common Tekken v15) writes a call as the control tokens
``[TOOL_CALLS]`` and ``[ARGS]`` around the function name, followed by the JSON arguments.
Parallel calls repeat the block, and nothing closes the last one but the end of sequence::

    optional content[TOOL_CALLS]get_weather[ARGS]{"city": "Paris"}[TOOL_CALLS]...</s>

The rules follow vLLM's Mistral parser for v11+ tokenizers
(``vllm/parser/mistral.py``):

- content is the text before the first ``[TOOL_CALLS]``, stripped once a call follows;
- a name runs to ``[ARGS]``, or to the first ``{`` when ``[ARGS]`` is missing, and is
  returned even when it is empty or not a declared tool;
- the arguments are the first complete JSON value after the name, verbatim, and the text
  after that value, up to the next ``[TOOL_CALLS]``, is dropped; arguments that never
  complete are returned as their stripped raw text, and missing ones as ``{}``.

The control tokens reach the parser only while the request keeps special tokens in its
decoded text (``needs_raw_special_tokens``), which also keeps the end-of-sequence
``</s>``; the parser drops it.

Streaming returns each name once it is complete and then the arguments as they arrive,
and yields exactly the calls the one-shot parse finds, however the text is split.
"""

import json
from typing import List

from tensorrt_llm.logger import logger

from ..openai_protocol import ChatCompletionToolsParam as Tool
from .base_tool_parser import BaseToolParser
from .core_types import StreamingParseResult, StructureInfo, ToolCallItem, _GetInfoFunc

TOOL_CALLS = "[TOOL_CALLS]"
ARGS = "[ARGS]"
EOS = "</s>"

_CONTENT, _NAME, _ARGUMENTS, _END = "content", "name", "arguments", "end"
_DECODER = json.JSONDecoder()


def _partial_marker_length(text: str, markers) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of a marker."""
    best = 0
    for marker in markers:
        for length in range(min(len(text), len(marker) - 1), best, -1):
            if marker.startswith(text[-length:]):
                best = length
                break
    return best


class _Arguments:
    """The arguments of one call: raw text up to the end of the first JSON value."""

    def __init__(self):
        self.text = ""
        self.sent = 0
        self.structured = False
        self.complete = False
        self._depth = 0
        self._in_string = False
        self._escaped = False

    def feed(self, chunk: str) -> None:
        if self.complete:
            return
        if not self.text:
            chunk = chunk.lstrip()
            if not chunk:
                return
            self.structured = chunk[0] in "{["
        if not self.structured:
            self.text += chunk
            return
        for index, char in enumerate(chunk):
            if self._in_string:
                if self._escaped:
                    self._escaped = False
                elif char == "\\":
                    self._escaped = True
                elif char == '"':
                    self._in_string = False
            elif char == '"':
                self._in_string = True
            elif char in "{[":
                self._depth += 1
            elif char in "}]":
                self._depth -= 1
                if self._depth == 0:
                    self.text += chunk[: index + 1]
                    self.complete = True
                    return
        self.text += chunk

    def ready(self) -> str:
        """The part that may be streamed while more text can still arrive.

        Trailing whitespace of an open value is held back, since a value that never
        completes loses it. Unstructured arguments are held until the call ends.
        """
        if not self.structured:
            return ""
        return self.text if self.complete else self.text.rstrip()

    def value(self) -> str:
        """The arguments once the call has ended."""
        if not self.text:
            return "{}"
        if self.structured:
            return self.ready()
        stripped = self.text.strip()
        try:
            _, end = _DECODER.raw_decode(stripped)
        except ValueError:
            return stripped
        return stripped[:end]


class MistralLarge4ToolParser(BaseToolParser):
    """Parses ``[TOOL_CALLS]name[ARGS]{...}`` calls of Mistral-Large-4."""

    needs_raw_special_tokens = True
    streaming_matches_whole_parse = True
    extracts_forced_tool_calls = True

    def __init__(self):
        super().__init__()
        self.bot_token = TOOL_CALLS
        self._state = _CONTENT
        self._held_space = ""
        self._arguments = _Arguments()

    def has_tool_call(self, text: str) -> bool:
        return TOOL_CALLS in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        result = MistralLarge4ToolParser()._feed(text, final=True)
        calls: List[ToolCallItem] = []
        for item in result.calls:
            if item.name is None:
                calls[-1].parameters += item.parameters
            else:
                calls.append(item.model_copy())
        declared = self._get_tool_indices(tools)
        for call in calls:
            if call.name not in declared:
                logger.warning(f"Model attempted to call undefined function: {call.name!r}")
        normal_text = result.normal_text.strip() if calls else result.normal_text
        return StreamingParseResult(normal_text=normal_text, calls=calls)

    def parse_streaming_increment(self, new_text: str, tools: List[Tool]) -> StreamingParseResult:
        return self._feed(new_text, final=False)

    def finish(self, tools: List[Tool]) -> StreamingParseResult:
        return self._feed("", final=True)

    def supports_structural_tag(self) -> bool:
        # The xgrammar vocabulary of Mistral-Large-4 cannot produce control tokens as
        # text, so a grammar could not open a call (vLLM builds Mistral tool
        # grammars with mistral-common on llguidance instead).
        return False

    def structure_info(self) -> _GetInfoFunc:
        return lambda name: StructureInfo(
            begin=f"{TOOL_CALLS}{name}{ARGS}", end="", trigger=TOOL_CALLS
        )

    def _feed(self, text: str, final: bool) -> StreamingParseResult:
        self._buffer += text
        content: List[str] = []
        calls: List[ToolCallItem] = []
        while self._step(content, calls, final):
            pass
        return StreamingParseResult(normal_text="".join(content), calls=calls)

    def _step(self, content: List[str], calls: List[ToolCallItem], final: bool) -> bool:
        """Consume as much of the buffer as the state allows; False when it needs more."""
        buffer = self._buffer
        if self._state == _END:
            self._buffer = ""
            return False

        if self._state == _CONTENT:
            index = buffer.find(TOOL_CALLS)
            if index == -1:
                keep = 0 if final else _partial_marker_length(buffer, (TOOL_CALLS, EOS))
                self._content(buffer[: len(buffer) - keep], content)
                self._buffer = buffer[len(buffer) - keep :]
                if final:
                    content.append(self._held_space)
                    self._held_space = ""
                return False
            self._content(buffer[:index], content)
            # Content before a call is stripped, so its trailing whitespace is dropped.
            self._held_space = ""
            self._buffer = buffer[index + len(TOOL_CALLS) :]
            self._state = _NAME
            return True

        if self._state == _NAME:
            found = [(buffer.find(marker), marker) for marker in (ARGS, "{", TOOL_CALLS, EOS)]
            found = [(index, marker) for index, marker in found if index != -1]
            if not found:
                if final:
                    self._open_call(buffer, calls)
                    self._close_call(calls)
                    self._state = _END
                return False
            index, marker = min(found)
            self._open_call(buffer[:index], calls)
            if marker == "{":
                self._buffer = buffer[index:]
                self._state = _ARGUMENTS
            elif marker == ARGS:
                self._buffer = buffer[index + len(ARGS) :]
                self._state = _ARGUMENTS
            else:
                self._close_call(calls)
                self._buffer = buffer[index + len(marker) :]
                self._state = _NAME if marker == TOOL_CALLS else _END
            return True

        found = [(buffer.find(marker), marker) for marker in (TOOL_CALLS, EOS)]
        found = [(index, marker) for index, marker in found if index != -1]
        if not found:
            keep = 0 if final else _partial_marker_length(buffer, (TOOL_CALLS, EOS))
            self._arguments.feed(buffer[: len(buffer) - keep])
            self._buffer = buffer[len(buffer) - keep :]
            if final:
                self._close_call(calls)
                self._state = _END
            else:
                self._send_arguments(self._arguments.ready(), calls)
            return False
        index, marker = min(found)
        self._arguments.feed(buffer[:index])
        self._close_call(calls)
        self._buffer = buffer[index + len(marker) :]
        self._state = _NAME if marker == TOOL_CALLS else _END
        return True

    def _content(self, text: str, content: List[str]) -> None:
        """Pass content on, holding back trailing whitespace a call would strip."""
        combined = self._held_space + text.replace(EOS, "")
        body = combined.rstrip()
        self._held_space = combined[len(body) :]
        if body:
            content.append(body)

    def _open_call(self, name: str, calls: List[ToolCallItem]) -> None:
        self.current_tool_id += 1
        self._arguments = _Arguments()
        calls.append(
            ToolCallItem(tool_index=self.current_tool_id, name=name.strip(), parameters="")
        )

    def _close_call(self, calls: List[ToolCallItem]) -> None:
        self._send_arguments(self._arguments.value(), calls)

    def _send_arguments(self, ready: str, calls: List[ToolCallItem]) -> None:
        delta = ready[self._arguments.sent :]
        if not delta:
            return
        self._arguments.sent = len(ready)
        last = calls[-1] if calls else None
        if last is not None and last.tool_index == self.current_tool_id and last.name is None:
            last.parameters += delta
        else:
            calls.append(ToolCallItem(tool_index=self.current_tool_id, parameters=delta))
