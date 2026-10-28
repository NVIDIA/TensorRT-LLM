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
"""Reasoning parser for the MuseSpark header/message chat format."""

import re
from typing import Any, Optional

from tensorrt_llm import logger
from tensorrt_llm.llmapi.reasoning_parser import (
    BaseReasoningParser,
    ReasoningParserResult,
    register_reasoning_parser,
)


@register_reasoning_parser("musespark")
class MuseSparkReasoningParser(BaseReasoningParser):
    """Reasoning parser for the MuseSpark header/message stream.

    Every assistant message is a plain-text header between special tokens,
    a body, and an end marker::

        <|start|>assistant to=RECIPIENT<|message|>BODY<|eom|>   (more follows)
        <|start|>assistant to=RECIPIENT<|message|>BODY<|eot|>   (end of turn)

    The generation prompt ends with ``<|start|>assistant``, so the first
    message starts at `` to=RECIPIENT<|message|>`` (leading space included).

    Recipient routing:

    ==================  ==================================================
    ``to=self``         ``reasoning_content`` (private CoT)
    bare ``<|message|>`` ``reasoning_content`` (public CoT, no recipient)
    ``to=user``         ``content`` (final answer)
    ``to=commentary``   ``content`` (interim user-visible text)
    ``to=developer``,
    ``to=assistant``    dropped
    anything else       tool call: header + ``<|message|>`` + body are passed
                        through verbatim into ``content`` for the
                        ``musespark`` tool parser; end markers are dropped
    ==================  ==================================================

    Streaming: header text is buffered until ``<|message|>`` resolves the
    recipient, so no fragment of `` to=self`` ever reaches ``content``. Text
    that cannot be a header (does not start with `` to=``/``<|start|>``/
    ``<|message|>``) is streamed as content, matching the reference default.

    ``needs_raw_special_tokens`` makes the serving layer decode with
    ``skip_special_tokens=False``. If the parser is nevertheless fed text
    with the special tokens stripped (`` to=selfTHOUGHTassistant to=userANSWER``),
    no ``<|message|>`` ever arrives; the parser buffers until ``finish()``
    and splits on the bare recipient headers (``_parse_stripped``), never
    returning a ``to=self`` span as content.
    """

    needs_raw_special_tokens = True

    # Structured output with reasoning enabled builds a structural tag that
    # lets the model think freely and then constrains the answer, and it does
    # that from these two markers (openai_protocol.py, the "force thinking and
    # then trigger user constraint" branch). This model does not wrap its
    # reasoning in a symmetric tag pair like ``<think>``/``</think>``: it
    # emits a private-recipient header, the thought, an end-of-message marker,
    # and then a *new* header addressed to the user before the answer. So the
    # opening marker is the private-recipient header and the closing marker
    # carries the end of the thought together with the answer's header, which
    # leaves the grammar applying to exactly the answer body and nothing else.
    #
    # Without these, a json_schema request with a reasoning parser configured
    # fails at the server with AttributeError and the client sees HTTP 400.
    # With reasoning off the model writes no thought, but the generation prompt
    # still stops at ``<|start|>assistant``, so it writes this header before the
    # answer body. Structured output has to trigger on it rather than applying
    # the schema from the first generated token, which would forbid the very
    # header the model has to write next.
    answer_start = " to=user<|message|>"

    reasoning_start = " to=self<|message|>"
    reasoning_end = "<|eom|><|start|>assistant" + answer_start

    START = "<|start|>"
    MESSAGE = "<|message|>"
    EOM = "<|eom|>"
    EOT = "<|eot|>"
    EOS = "<|end_of_text|>"

    _REASONING = "reasoning"
    _CONTENT = "content"
    _TOOL = "tool"
    _DROP = "drop"

    _CONTENT_RECIPIENTS = frozenset({"user", "commentary"})
    _DROP_RECIPIENTS = frozenset({"developer", "assistant"})
    _NAME_CHARS = r"[A-Za-z0-9_.\-]"

    # Header text between the message boundary and ``<|message|>``.
    _RECIPIENT_RE = re.compile(r"(?:^|\s)to=(" + _NAME_CHARS + r"+)\s*$")
    # A buffer that may still grow into a header (kept-token case).
    _HEADER_PARTIAL_RE = re.compile(
        r"^(?:<\|start\|>assistant)?(?:\s*(?:t|to|to=" + _NAME_CHARS + r"*)?)$"
    )
    # A buffer that looks like a header whose ``<|message|>`` was stripped.
    _HEADER_STRIPPED_RE = re.compile(
        r"^(?:<\|start\|>assistant|assistant)?\s+to=" + _NAME_CHARS + r"+"
    )

    # Stripped-token fallback: the plain header text that survives detokenization.
    _STRIPPED_HEADER_RE = re.compile(
        r"(?:^|(?<=assistant)) to=(self|user|commentary|"
        + _NAME_CHARS
        + r"+(?=<atem:function_calls>))"
    )
    _STRIPPED_NEXT_RE = re.compile(r"assistant to=")

    def __init__(self, *, chat_template_kwargs: Optional[dict[str, Any]] = None) -> None:
        super().__init__(chat_template_kwargs=chat_template_kwargs)
        self._reset()

    def _reset(self) -> None:
        self._buffer = ""
        # 'header' | 'body' | 'stripped' | 'done'
        self._state = "header"
        self._channel: Optional[str] = None
        self._saw_message_marker = False
        # One warning per response, not per delta: the condition it reports
        # repeats on every delta that follows it.
        self._warned_no_progress = False

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _partial_suffix_len(text: str, markers: tuple[str, ...]) -> int:
        """Longest suffix of ``text`` that is a proper prefix of any marker."""
        best = 0
        for marker in markers:
            for length in range(min(len(text), len(marker) - 1), best, -1):
                if marker.startswith(text[-length:]):
                    best = length
                    break
        return best

    @classmethod
    def _channel_for_recipient(cls, recipient: Optional[str]) -> str:
        if recipient is None or recipient == "self":
            return cls._REASONING
        if recipient in cls._CONTENT_RECIPIENTS:
            return cls._CONTENT
        if recipient in cls._DROP_RECIPIENTS:
            return cls._DROP
        return cls._TOOL

    @classmethod
    def _recipient_of(cls, header: str) -> Optional[str]:
        m = cls._RECIPIENT_RE.search(header)
        return m.group(1) if m else None

    @classmethod
    def _could_be_header(cls, buf: str) -> bool:
        """Whether ``buf`` is still a proper prefix of some header form.

        The forms are ``[<|start|>assistant] to=NAME<|message|>`` and a bare
        ``<|message|>``.
        """
        hold = cls._partial_suffix_len(buf, (cls.MESSAGE,))
        core = buf[: len(buf) - hold]
        if "<|start|>assistant".startswith(core):
            return True
        return cls._HEADER_PARTIAL_RE.match(core) is not None

    def _emit(self, text: str, content: list, reasoning: list) -> None:
        if not text:
            return
        if self._channel == self._REASONING:
            reasoning.append(text)
        elif self._channel in (self._CONTENT, self._TOOL):
            content.append(text)

    def _drop_ruled_out_start(self) -> None:
        """Consume a leading ``<|start|>`` the header state has just ruled out.

        The body state deliberately leaves ``<|start|>`` in the buffer when it
        hands control back to the header state, so the header regexes see the
        canonical ``<|start|>assistant to=`` form. If the header state then
        decides the text is not a header, leaving the marker in place sends the
        identical buffer back to the body state, which finds the marker at
        index 0, emits nothing and hands it straight back. The two states pass
        the same buffer to each other with nothing consumed, so ``_feed``'s
        loop never ends.

        That is not a theoretical case: it is what wedged the server under
        concurrent load. The spin runs in the frontend's event loop and holds
        the GIL, so ``/health`` stops answering and no new request reaches the
        engine; rank 0 then blocks on an empty request queue while the other
        ranks sit in the per-iteration request-count broadcast, with the GPUs
        idle. Consuming the marker here is what makes every step consume
        something. It is a special token, so it is dropped rather than emitted:
        the alternative leaks ``<|start|>`` into user-visible content.
        """
        if self._buffer.startswith(self.START):
            self._buffer = self._buffer[len(self.START) :]

    # ------------------------------------------------------------ state step
    def _step(self, content: list, reasoning: list) -> bool:
        """Consume as much of the buffer as possible; False = need more input."""
        buf = self._buffer
        if self._state in ("done", "stripped"):
            return False

        if self._state == "header":
            terminators = (self.MESSAGE, self.EOT, self.EOS)
            hits = [(buf.find(m), m) for m in terminators]
            hits = [(i, m) for i, m in hits if i != -1]
            if not hits:
                if self._could_be_header(buf):
                    return False  # still a plausible header prefix
                if not self._saw_message_marker and self._HEADER_STRIPPED_RE.match(buf):
                    # `` to=self`` followed by body text with no <|message|>:
                    # special tokens were stripped. Hold everything for the
                    # finish() fallback rather than leak the private channel.
                    self._state = "stripped"
                    return False
                # Not a header at all: the reference streams this as content.
                self._drop_ruled_out_start()
                self._channel = self._CONTENT
                self._state = "body"
                return True
            idx, marker = min(hits)
            if marker != self.MESSAGE:
                head = buf[:idx]
                if (
                    head
                    and not self._could_be_header(head)
                    and not (not self._saw_message_marker and self._HEADER_STRIPPED_RE.match(head))
                ):
                    # A body that never had a header. Guided decoding pins the
                    # whole generation to one value -- a forced tool call's
                    # arguments, a constrained answer -- so the model writes no
                    # `` to=RECIPIENT<|message|>``, and this state waits for one
                    # that cannot arrive. Clearing the buffer here discards the
                    # whole response: a forced tool_choice decoded
                    # ``{"a": 2, "b": 3}<|end_of_text|>`` and the caller got an
                    # empty message and no tool call. Route it the way the
                    # no-terminator branch above routes unheadered text and let
                    # the body branch consume the end marker.
                    self._drop_ruled_out_start()
                    self._channel = self._CONTENT
                    self._state = "body"
                    return True
                # <|eot|>/<|end_of_text|> right after <|eom|>: turn is over.
                self._state = "done"
                self._buffer = ""
                return False
            self._saw_message_marker = True
            header = buf[:idx]
            self._channel = self._channel_for_recipient(self._recipient_of(header))
            if self._channel == self._TOOL:
                # Replay the routing header for the tool parser.
                content.append(header + self.MESSAGE)
            self._buffer = buf[idx + len(self.MESSAGE) :]
            self._state = "body"
            return True

        # body
        markers = (self.EOM, self.EOT, self.EOS, self.START)
        hits = [(buf.find(m), m) for m in markers]
        hits = [(i, m) for i, m in hits if i != -1]
        if not hits:
            hold = self._partial_suffix_len(buf, markers)
            emit_len = len(buf) - hold
            self._emit(buf[:emit_len], content, reasoning)
            self._buffer = buf[emit_len:]
            return False
        idx, marker = min(hits)
        self._emit(buf[:idx], content, reasoning)
        if marker in (self.EOT, self.EOS):
            self._state = "done"
            self._buffer = ""
            return False
        # <|eom|>: the next header follows as ``<|start|>assistant to=...``.
        # A bare <|start|> (model skipped <|eom|>) is kept so the header
        # regexes see the canonical form.
        self._buffer = buf[idx + (len(marker) if marker == self.EOM else 0) :]
        self._channel = None
        self._state = "header"
        return True

    def _feed(self, text: str) -> ReasoningParserResult:
        self._buffer += text
        content: list = []
        reasoning: list = []
        # A step that returns True has consumed at least one character or has
        # moved to a state that will, so the number of steps is bounded by the
        # buffer length. The bound is enforced rather than assumed: this loop
        # runs in the server's event loop, so a step sequence that stops
        # consuming does not spoil one response, it takes the whole server down
        # (see _drop_ruled_out_start for the occurrence that did). Breaking out
        # leaves the remainder to the next delta or to finish(), which is a bad
        # parse; spinning here is a dead server.
        budget = len(self._buffer) + 8
        while self._step(content, reasoning):
            budget -= 1
            if budget <= 0:
                if not self._warned_no_progress:
                    self._warned_no_progress = True
                    logger.warning(
                        "MuseSpark reasoning parser stopped making progress in "
                        f"state {self._state!r} with {len(self._buffer)} "
                        "characters buffered; the rest of this response is left"
                        f" to finish(). Buffer head: {self._buffer[:64]!r}"
                    )
                break
        return ReasoningParserResult(content="".join(content), reasoning_content="".join(reasoning))

    # ------------------------------------------------- stripped-token fallback
    @classmethod
    def _parse_stripped(cls, text: str) -> ReasoningParserResult:
        """Split on bare recipient headers when special tokens were removed.

        As in the reference implementation, a `` to=self`` span is never
        returned as content; with no recipient header at all the text is
        ordinary content. Bodies end at the next literal
        ``assistant to=`` (what ``<|start|>assistant to=`` reduces to).
        """
        first = cls._STRIPPED_HEADER_RE.search(text)
        if first is None:
            return ReasoningParserResult(content=text)
        content: list = []
        reasoning: list = []
        pre = text[: first.start()]
        if pre.endswith("assistant"):
            pre = pre[: -len("assistant")]
        if pre:
            content.append(pre)
        m = first
        while m is not None:
            channel = cls._channel_for_recipient(m.group(1))
            body_start = m.end()
            nxt = cls._STRIPPED_NEXT_RE.search(text, body_start)
            if nxt is None:
                body, pos = text[body_start:], len(text)
            else:
                body = text[body_start : nxt.start()]
                pos = nxt.start() + len("assistant")
            if channel == cls._REASONING:
                reasoning.append(body)
            elif channel in (cls._CONTENT, cls._TOOL):
                content.append(body)
            m = cls._STRIPPED_HEADER_RE.search(text, pos) if pos < len(text) else None
            if m is None and pos < len(text):
                content.append(text[pos:])
        return ReasoningParserResult(content="".join(content), reasoning_content="".join(reasoning))

    # ------------------------------------------------------------- interface
    def parse(self, text: str) -> ReasoningParserResult:
        self._reset()
        result = self._feed(text)
        tail = self.finish()
        return ReasoningParserResult(
            content=result.content + tail.content,
            reasoning_content=result.reasoning_content + tail.reasoning_content,
        )

    def parse_delta(self, delta_text: str) -> ReasoningParserResult:
        return self._feed(delta_text)

    def finish(self) -> ReasoningParserResult:
        remaining = self._buffer
        self._buffer = ""
        state, self._state = self._state, "done"
        if state == "done":
            return ReasoningParserResult()
        if state == "body":
            # Truncated mid-body (max_tokens): flush to the current channel.
            if self._channel == self._REASONING:
                return ReasoningParserResult(reasoning_content=remaining)
            if self._channel in (self._CONTENT, self._TOOL):
                return ReasoningParserResult(content=remaining)
            return ReasoningParserResult()
        if state == "stripped" or (not self._saw_message_marker and remaining):
            return self._parse_stripped(remaining)
        # A header fragment after a completed message: dropped.
        return ReasoningParserResult()
