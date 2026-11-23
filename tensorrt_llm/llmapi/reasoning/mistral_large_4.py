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
"""Reasoning parser for Mistral-Large-4's ``[THINK]`` sections."""

from typing import Any, Optional

from tensorrt_llm.llmapi.reasoning_parser import (
    BaseReasoningParser,
    ReasoningParserResult,
    _trailing_partial_marker,
    register_reasoning_parser,
)


@register_reasoning_parser("mistral_large_4")
class MistralLarge4ReasoningParser(BaseReasoningParser):
    """Reasoning parser for Mistral-Large-4.

    Mistral-Large-4 (mistral-common Tekken v15) writes its reasoning between
    the control tokens ``[THINK]`` and ``[/THINK]``, and starts its tool calls
    with ``[TOOL_CALLS]``. The rules follow vLLM's Mistral parser
    (``vllm/parser/mistral.py``):

    * the output starts as content; ``[THINK]`` opens reasoning and
      ``[/THINK]`` closes it, so text around a block stays content and
      several blocks are joined;
    * a ``[THINK]`` inside reasoning and a ``[/THINK]`` outside it are
      dropped, and a block that never closes is reasoning to the end;
    * ``[TOOL_CALLS]`` also ends reasoning; it and everything after it are
      content, for the tool parser;
    * reasoning loses its trailing whitespace.

    The delimiters reach the parser only with special tokens kept in the
    decoded text, which also keeps the end-of-sequence ``</s>``; it is
    dropped.
    """

    needs_raw_special_tokens = True

    THINK_START = "[THINK]"
    THINK_END = "[/THINK]"
    TOOL_CALLS = "[TOOL_CALLS]"
    EOS = "</s>"

    def __init__(self, *, chat_template_kwargs: Optional[dict[str, Any]] = None) -> None:
        super().__init__(chat_template_kwargs=chat_template_kwargs)
        self._buffer = ""
        self._in_reasoning = False
        self._in_tool_calls = False
        # Trailing reasoning whitespace, sent only if more reasoning follows.
        self._held_space = ""

    def parse(self, text: str) -> ReasoningParserResult:
        parser = MistralLarge4ReasoningParser()
        result = parser._feed(text, final=True)
        return ReasoningParserResult(
            content=result.content, reasoning_content=result.reasoning_content
        )

    def parse_delta(self, delta_text: str) -> ReasoningParserResult:
        return self._feed(delta_text, final=False)

    def finish(self) -> ReasoningParserResult:
        return self._feed("", final=True)

    def _feed(self, text: str, final: bool) -> ReasoningParserResult:
        self._buffer += text
        content: list[str] = []
        reasoning: list[str] = []
        while self._step(content, reasoning, final):
            pass
        return ReasoningParserResult(content="".join(content), reasoning_content="".join(reasoning))

    def _step(self, content: list[str], reasoning: list[str], final: bool) -> bool:
        """Consume the buffer up to the next marker; False when it needs more."""
        buffer = self._buffer
        markers = (
            (self.EOS,)
            if self._in_tool_calls
            else (self.THINK_START, self.THINK_END, self.TOOL_CALLS, self.EOS)
        )
        found = [(buffer.find(marker), marker) for marker in markers]
        found = [(index, marker) for index, marker in found if index != -1]
        if not found:
            keep = 0 if final else _trailing_partial_marker(buffer, *markers)
            self._emit(buffer[: len(buffer) - keep], content, reasoning)
            self._buffer = buffer[len(buffer) - keep :]
            return False
        index, marker = min(found)
        self._emit(buffer[:index], content, reasoning)
        if marker == self.TOOL_CALLS:
            # The tool parser reads the calls from the content.
            self._in_reasoning = False
            self._in_tool_calls = True
            self._buffer = buffer[index:]
            return True
        self._buffer = buffer[index + len(marker) :]
        if marker == self.THINK_START:
            self._in_reasoning = True
        elif marker == self.THINK_END:
            self._in_reasoning = False
        return True

    def _emit(self, text: str, content: list[str], reasoning: list[str]) -> None:
        if not self._in_reasoning:
            content.append(text)
            return
        combined = self._held_space + text
        body = combined.rstrip()
        self._held_space = combined[len(body) :]
        if body:
            reasoning.append(body)
