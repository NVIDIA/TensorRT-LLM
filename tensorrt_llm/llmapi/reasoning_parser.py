# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
import re
from abc import ABC, abstractmethod
from collections.abc import KeysView
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Optional, Type

from tensorrt_llm import logger


@dataclass
class ReasoningParserResult:
    content: str = ""
    reasoning_content: str = ""


# Enough of the rendered prompt's tail to hold a prefilled marker and any
# trailing whitespace, without copying a prompt that may be very long.
_PROMPT_TAIL_CHARS = 64


def register_reasoning_parser(*keys: str, **default_kwargs):
    """Decorator that registers a BaseReasoningParser under one or more keys.

    Any extra keyword arguments are stored as defaults and forwarded to
    the parser constructor at creation time.

    Usage::

        @register_reasoning_parser("my-model", reasoning_at_start=True)
        class MyParser(BaseReasoningParser):
            ...
    """

    def decorator(parser_cls: Type["BaseReasoningParser"]):
        if parser_cls.resolves_thinking_from_prompt:
            # Fail at import rather than per request: `resolve_prefilled_thinking`
            # reads these off the class, and subclasses of parsers that only set
            # them in `__init__` would otherwise raise inside the request path.
            for attr in ("reasoning_start", "reasoning_end"):
                if not isinstance(getattr(parser_cls, attr, None), str):
                    raise TypeError(
                        f"{parser_cls.__name__} sets "
                        f"resolves_thinking_from_prompt but does not define "
                        f"{attr} as a class attribute")
        for key in keys:
            ReasoningParserFactory._parsers[key] = (parser_cls, default_kwargs)
        return parser_cls

    return decorator


class ReasoningParserFactory:
    _parsers: dict[str, tuple[Type["BaseReasoningParser"], dict[str, Any]]] = {}

    @classmethod
    def create_reasoning_parser(
        cls,
        reasoning_parser: str,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
    ) -> "BaseReasoningParser":
        key = reasoning_parser.lower()
        try:
            parser_cls, default_kwargs = cls._parsers[key]
        except KeyError as e:
            raise ValueError(
                f"Invalid reasoning parser: {reasoning_parser}\n"
                f"Supported parsers: {list(cls._parsers.keys())}") from e
        return parser_cls(chat_template_kwargs=chat_template_kwargs,
                          **default_kwargs)

    @classmethod
    def resolves_thinking_from_prompt(cls, reasoning_parser: str) -> bool:
        """Whether this parser selects its mode from the rendered prompt."""
        entry = cls._parsers.get(reasoning_parser.lower())
        return bool(entry) and entry[0].resolves_thinking_from_prompt

    @classmethod
    def resolve_prefilled_thinking(cls, reasoning_parser: str,
                                   prompt: str) -> Optional[bool]:
        """Read the reasoning mode off the tail of a rendered prompt.

        These templates append the marker last, after the assistant header:
        `...<|assistant|><think>` with thinking on, `...<|assistant|></think>`
        with it off. The two are mutually exclusive and both land at the very
        end, so whichever marker ends the prompt *is* the mode.

        Testing the suffix rather than the whole prompt is what makes this
        correct, not merely cheap: prior assistant turns render with their own
        `<think>...</think>` pairs, so a containment check would misfire on
        every multi-turn request. `_PROMPT_TAIL_CHARS` is a copy bound, not a
        semantic window; whitespace before the marker is unbounded, but more
        trailing whitespace than that slice would push the marker out of view
        and read as unresolved.

        Returns:
            True  - the template prefilled `<think>`: reasoning is open.
            False - the template prefilled `</think>`: reasoning is already
                    closed, so all model output is content.
            None  - the mode cannot be determined from this prompt (unknown
                    parser, parser has not opted in, or neither marker is at
                    the tail). Callers must treat this as "ask elsewhere"
                    (e.g. the relayed disagg value), not as thinking-off.
            A prompt ending with neither marker is referred to the parser's
            `resolve_unmarked_prompt`, which returns None unless the parser
            recognises the tail.
        """
        entry = cls._parsers.get(reasoning_parser.lower())
        if entry is None:
            return None
        parser_cls = entry[0]
        if not parser_cls.resolves_thinking_from_prompt:
            return None
        # Only the tail matters, so avoid copying the whole prompt.
        tail = prompt[-_PROMPT_TAIL_CHARS:].rstrip()
        if tail.endswith(parser_cls.reasoning_end):
            return False
        if tail.endswith(parser_cls.reasoning_start):
            return True
        return parser_cls.resolve_unmarked_prompt(tail)

    @classmethod
    def keys(cls) -> KeysView[str]:
        return cls._parsers.keys()

    @classmethod
    def needs_raw_special_tokens(cls, reasoning_parser: str) -> bool:
        """Whether the registered parser must see special tokens.

        See ``BaseReasoningParser.needs_raw_special_tokens``.
        """
        entry = cls._parsers.get(reasoning_parser.lower())
        return bool(entry
                    and getattr(entry[0], "needs_raw_special_tokens", False))


class BaseReasoningParser(ABC):

    # Parsers whose delimiters are registered special tokens must see the
    # raw decoded stream; the serving layer checks this flag and disables
    # ``skip_special_tokens`` for the request (mirrors
    # ``BaseToolParser.needs_raw_special_tokens``, which only takes effect
    # when the request carries tools).
    needs_raw_special_tokens: bool = False

    # Opt in on parsers whose template prefills the reasoning marker into the
    # prompt and that select their mode from `enable_thinking`. Only those can
    # have the mode resolved from the rendered prompt, and they must define
    # both markers below.
    resolves_thinking_from_prompt: ClassVar[bool] = False
    reasoning_start: ClassVar[str]
    reasoning_end: ClassVar[str]

    @classmethod
    def resolve_unmarked_prompt(cls, tail: str) -> Optional[bool]:
        """Mode of a rendered prompt whose stripped tail ends with neither marker.

        None leaves the mode unresolved, which is right for a template that
        prefills a marker whenever it prefills anything at all.
        """
        return None

    def __init__(self,
                 *,
                 chat_template_kwargs: Optional[dict[str, Any]] = None) -> None:
        pass

    @abstractmethod
    def parse(self, text: str) -> ReasoningParserResult:
        raise NotImplementedError

    @abstractmethod
    def parse_delta(self, delta_text: str) -> ReasoningParserResult:
        raise NotImplementedError

    def finish(self) -> ReasoningParserResult:
        """Called when the stream ends. Subclasses may override to flush
        buffered state or reclassify accumulated content. The default
        implementation returns an empty result."""
        return ReasoningParserResult()


class IdentityReasoningParser(BaseReasoningParser):
    """Reasoning parser that treats all model output as visible content."""

    reasoning_start = "<think>"
    reasoning_end = "</think>"

    def parse(self, text: str) -> ReasoningParserResult:
        return ReasoningParserResult(content=text)

    def parse_delta(self, delta_text: str) -> ReasoningParserResult:
        return ReasoningParserResult(content=delta_text)


@register_reasoning_parser("deepseek-r1", reasoning_at_start=True)
@register_reasoning_parser("qwen3")
# Qwen3.5 (and forced-thinking Qwen3 variants) use a chat template that
# pre-injects `<think>\n` into the assistant prompt prefix, so the model
# output begins inside the reasoning block with no opening tag to search
# for. That requires `reasoning_at_start=True`. The existing `qwen3` key
# keeps `reasoning_at_start=False` for back-compat, and `parse()` is
# binary on this flag (it either requires `<think>` to be present in the
# output, or assumes the output begins at the start of reasoning) - so
# the two behaviors must be registered under separate keys.
@register_reasoning_parser("qwen3_5", reasoning_at_start=True)
@register_reasoning_parser("minimax_m2", reasoning_at_start=True)
@register_reasoning_parser("minimax_m2_append_think", reasoning_at_start=True)
class DeepSeekR1Parser(BaseReasoningParser):
    """
    Reasoning parser for DeepSeek-R1. Reasoning format: <think>(.*)</think>.
    Since the latest official tokenizer_config.json initially adds "<think>\\n" at the end of the prompt
    (https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/tokenizer_config.json),
    treat all the text before the </think> tag as `reasoning_content` and the text after as `content`.
    """

    def __init__(self,
                 *,
                 reasoning_at_start: bool = False,
                 chat_template_kwargs: Optional[dict[str, Any]] = None) -> None:
        super().__init__(chat_template_kwargs=chat_template_kwargs)
        self.reasoning_start = "<think>"
        self.reasoning_end = "</think>"
        self.reasoning_at_start = reasoning_at_start
        self.in_reasoning = self.reasoning_at_start
        self._buffer = ""

    def _create_reasoning_end_result(self, content: str,
                                     reasoning_content: str):
        if len(content) == 0:
            reasoning_parser_result = ReasoningParserResult(
                reasoning_content=reasoning_content)
        elif len(reasoning_content) == 0:
            reasoning_parser_result = ReasoningParserResult(content=content)
        else:
            reasoning_parser_result = ReasoningParserResult(
                content=content, reasoning_content=reasoning_content)
        return reasoning_parser_result

    def parse(self, text: str) -> ReasoningParserResult:
        if not self.reasoning_at_start:
            splits = text.partition(self.reasoning_start)
            if splits[1] == "":
                # no reasoning start tag found
                return ReasoningParserResult(content=text)
            # reasoning start tag found
            # text before reasoning start tag is dropped
            text = splits[2]
        splits = text.partition(self.reasoning_end)
        reasoning_content, content = splits[0], splits[2]
        return ReasoningParserResult(content=content,
                                     reasoning_content=reasoning_content)

    def parse_delta(self, delta_text: str) -> ReasoningParserResult:
        self._buffer += delta_text
        delta_text = self._buffer
        reasoning_content = None
        content = None
        if (self.reasoning_start.startswith(delta_text)
                or self.reasoning_end.startswith(delta_text)):
            # waiting for more text to determine if it's a reasoning start or end tag
            return ReasoningParserResult()

        if not self.in_reasoning:
            begin_idx = delta_text.find(self.reasoning_start)
            if begin_idx == -1:
                self._buffer = ""
                return ReasoningParserResult(content=delta_text)
            self.in_reasoning = True
            # set reasoning_content, will be processed by the next block
            reasoning_content = delta_text[begin_idx +
                                           len(self.reasoning_start):]

        if self.in_reasoning:
            delta_text = reasoning_content if reasoning_content is not None else delta_text
            end_idx = delta_text.find(self.reasoning_end)
            if end_idx == -1:
                last_idx = delta_text.rfind(self.reasoning_end[0])
                if last_idx != -1 and self.reasoning_end.startswith(
                        delta_text[last_idx:]):
                    self._buffer = delta_text[last_idx:]
                    reasoning_content = delta_text[:last_idx]
                else:
                    self._buffer = ""
                    reasoning_content = delta_text
                return ReasoningParserResult(
                    reasoning_content=reasoning_content)
            reasoning_content = delta_text[:end_idx]
            content = delta_text[end_idx + len(self.reasoning_end):]
            self.in_reasoning = False
            self._buffer = ""
            return ReasoningParserResult(content=content,
                                         reasoning_content=reasoning_content)
        raise RuntimeError(
            "Unreachable code reached in `DeepSeekR1Parser.parse_delta`")

    def finish(self) -> ReasoningParserResult:
        """Flush text withheld by `parse_delta` when the stream ends.

        `parse_delta` holds back a trailing fragment that could still grow
        into a `<think>` / `</think>` tag. If the stream ends while such a
        fragment is buffered - because the response was truncated mid-tag, or
        simply ends with a literal `<` - the fragment is ordinary model output
        and must still be emitted, otherwise it is silently dropped. The
        buffered text is attributed to the block it was withheld in: inside a
        reasoning block it is reasoning content, otherwise it is visible
        content.

        A buffer holding exactly a complete tag is a delimiter rather than
        model output, so it is discarded as it would have been had more text
        followed.
        """
        remaining = self._buffer
        self._buffer = ""
        if not remaining or remaining in (self.reasoning_start,
                                          self.reasoning_end):
            return ReasoningParserResult()
        if self.in_reasoning:
            return ReasoningParserResult(reasoning_content=remaining)
        return ReasoningParserResult(content=remaining)


def _trailing_partial_marker(text: str, *markers: str) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of a marker.

    A streaming parser withholds that many characters: they may still grow
    into a marker split across deltas.
    """
    best = 0
    for marker in markers:
        for length in range(min(len(text), len(marker) - 1), best, -1):
            if marker.startswith(text[-length:]):
                best = length
                break
    return best


@register_reasoning_parser("glm", reasoning_at_start=True)
@register_reasoning_parser("glm45")
@register_reasoning_parser("glm47", reasoning_at_start=True)
@register_reasoning_parser("glm_moe_dsa", reasoning_at_start=True)
class GlmReasoningParser(BaseReasoningParser):
    """Reasoning parser for the GLM family.

    GLM-4.7 and GLM-5 templates prefill `<think>` (`reasoning_at_start`), or
    `</think>` when `enable_thinking` is false. GLM-4.5/4.6 (`glm45`) do not
    prefill `<think>`; the model opens the block at the start of its output.

    The block ends at the first `</think>`, or at a `<tool_call>` opened
    before it; a later `<think>` is text. A `</think>` before any content, or
    directly before the first `<tool_call>`, closes the block again and is
    dropped. Text from the first `<tool_call>` on is kept verbatim.
    """

    reasoning_start = "<think>"
    reasoning_end = "</think>"
    _tool_call_start = "<tool_call>"

    def __init__(self,
                 *,
                 reasoning_at_start: bool = False,
                 chat_template_kwargs: Optional[dict[str, Any]] = None) -> None:
        super().__init__(chat_template_kwargs=chat_template_kwargs)
        # The templates close the block in the prompt when `enable_thinking`
        # is passed and false.
        if not (chat_template_kwargs or {}).get("enable_thinking", True):
            reasoning_at_start = False
        self.reasoning_at_start = reasoning_at_start
        # "start": the block may still open after leading whitespace;
        # "closed": the block ended and no content followed yet.
        self._state = "reasoning" if reasoning_at_start else "start"
        self._buffer = ""

    def parse(self, text: str) -> ReasoningParserResult:
        # The whole text as one delta to a fresh parser, so both paths agree.
        parser = GlmReasoningParser(reasoning_at_start=self.reasoning_at_start)
        result, tail = parser.parse_delta(text), parser.finish()
        result.content += tail.content
        result.reasoning_content += tail.reasoning_content
        return result

    def parse_delta(self, delta_text: str) -> ReasoningParserResult:
        data, self._buffer = self._buffer + delta_text, ""
        content, reasoning = [], []
        while data:
            state = self._state
            if state == "tool_calls":
                content.append(data)
                break
            if state == "start":
                stripped = data.lstrip()
                if stripped.startswith(self.reasoning_start):
                    data = stripped[len(self.reasoning_start):]
                    self._state = "reasoning"
                elif self.reasoning_start.startswith(stripped):
                    self._buffer = data
                    break
                else:
                    self._state = "closed"
                continue
            out = reasoning if state == "reasoning" else content
            end = data.find(self.reasoning_end)
            call = data.find(self._tool_call_start)
            if call != -1 and (end == -1 or call < end):
                # A tool call ends the block; everything after it is content.
                out.append(data[:call])
                content.append(data[call:])
                self._state = "tool_calls"
                break
            if end == -1:
                cut = len(data) - _trailing_partial_marker(
                    data, self.reasoning_end, self._tool_call_start)
                out.append(data[:cut])
                self._buffer = data[cut:]
                if cut and state == "closed":
                    self._state = "content"
                break
            out.append(data[:end])
            rest = data[end + len(self.reasoning_end):]
            if state == "reasoning":
                self._state = "closed"
            elif end:
                self._state = "content"
            if self._state == "closed" or rest.startswith(
                    self._tool_call_start):
                # The tag ends the block or closes it again: drop it.
                data = rest
            elif self._tool_call_start.startswith(rest):
                # Hold the tag until it is known whether a tool call follows.
                self._buffer = data[end:]
                break
            else:
                content.append(self.reasoning_end)
                data = rest
        return ReasoningParserResult(content="".join(content),
                                     reasoning_content="".join(reasoning))

    def finish(self) -> ReasoningParserResult:
        remaining, self._buffer = self._buffer, ""
        if self._state == "reasoning":
            return ReasoningParserResult(reasoning_content=remaining)
        return ReasoningParserResult(content=remaining)


@register_reasoning_parser("deepseek_v4")
class DeepSeekV4ReasoningParser(BaseReasoningParser):
    """DeepSeek-V4 parser selected by thinking-mode chat template kwargs."""

    reasoning_start = "<think>"
    reasoning_end = "</think>"

    def __init__(
        self,
        *,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(chat_template_kwargs=chat_template_kwargs)
        chat_template_kwargs = chat_template_kwargs or {}
        thinking = bool(
            chat_template_kwargs.get("thinking", False)
            or chat_template_kwargs.get("enable_thinking", False))
        if thinking:
            self._parser = DeepSeekR1Parser(
                reasoning_at_start=True,
                chat_template_kwargs=chat_template_kwargs,
            )
        else:
            self._parser = IdentityReasoningParser(
                chat_template_kwargs=chat_template_kwargs)

    def parse(self, text: str) -> ReasoningParserResult:
        return self._parser.parse(text)

    def parse_delta(self, delta_text: str) -> ReasoningParserResult:
        return self._parser.parse_delta(delta_text)

    def finish(self) -> ReasoningParserResult:
        return self._parser.finish()


@register_reasoning_parser("deepseek_v41")
class DeepSeekV41ReasoningParser(DeepSeekV4ReasoningParser):
    """DeepSeek-V4.1 reasoning adapter.

    V4.1 keeps V4's ``<think>...</think>`` reasoning protocol and thinking
    controls.  The dedicated adapter makes the model-version pairing explicit
    while intentionally inheriting the shared wire-format implementation.
    """


@register_reasoning_parser("poolside_v1", "laguna")
class PoolsideV1ReasoningParser(DeepSeekV4ReasoningParser):
    """Poolside Laguna models, which prefill the marker the same way.

    The family's templates disagree on the `enable_thinking` default, so the
    mode is resolved from the rendered prompt rather than from a constant.
    `laguna` stays as an alias of `poolside_v1` for existing deployments.
    """

    resolves_thinking_from_prompt = True

    def __init__(
        self,
        *,
        chat_template_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(chat_template_kwargs=chat_template_kwargs)
        kwargs = chat_template_kwargs or {}
        if kwargs.get("thinking") is None and kwargs.get(
                "enable_thinking") is None:
            # Mode unresolved (offline LLM API, disagg generation server,
            # add_generation_prompt=false). Keep splitting on a `<think>` the
            # model emits itself, as these models do in multi-turn and tools.
            self._parser = DeepSeekR1Parser(
                reasoning_at_start=False,
                chat_template_kwargs=chat_template_kwargs)


@register_reasoning_parser("minimax_m3")
class MiniMaxM3ReasoningParser(DeepSeekR1Parser):
    """Reasoning parser for MiniMax-M3.

    The M3 chat template (``]<]minimax[>[`` family) renders the assistant
    turn in one of two shapes:

    * With reasoning::

          <mm:think>{reasoning}</mm:think>{content}

    * Without reasoning (a bare ``</mm:think>`` sentinel marks where the
      visible content starts; the template prefills it when thinking is
      disabled, and in adaptive mode the model generates it)::

          </mm:think>{content}

    Which shape the model produces depends on how the prompt was rendered:
    with thinking enabled the template prefills ``<mm:think>`` into the
    prompt, so the output opens inside the reasoning block with no opening
    tag; with thinking disabled it prefills ``</mm:think>``, so the output is
    all content; in adaptive mode (the template default) it prefills nothing
    and the model emits ``<mm:think>`` or the bare sentinel first. The parser
    takes its mode from the one the server resolved off the rendered prompt
    (relayed as ``enable_thinking``), falls back to ``thinking_mode`` when
    nothing was resolved, and otherwise starts outside reasoning. Both tags
    are delimiters and never appear in the result; text before an opening tag
    is content; an opening tag without a closing tag is reasoning cut off by
    the token limit. ``parse`` and ``parse_delta`` agree for every chunking.
    """

    resolves_thinking_from_prompt = True
    reasoning_start = "<mm:think>"
    reasoning_end = "</mm:think>"
    # The template's generation prompt is ``bos_token ~ 'ai' ~ '\\n'``.
    _assistant_header = "]~b]ai"

    @classmethod
    def resolve_unmarked_prompt(cls, tail: str) -> Optional[bool]:
        # Adaptive mode prefills nothing after the header, so the output
        # starts outside reasoning. Any other unmarked tail, such as a custom
        # template that prefills text after ``<mm:think>``, carries no mode the
        # tail can read and is left to ``thinking_mode``.
        if tail.endswith(cls._assistant_header):
            return False
        return None

    def __init__(self,
                 *,
                 chat_template_kwargs: Optional[dict[str, Any]] = None) -> None:
        kwargs = chat_template_kwargs or {}
        # The server overwrites `enable_thinking` with the mode resolved from
        # the rendered prompt, so it wins over `thinking_mode`, which decides
        # only when nothing was resolved. The Responses path's bare `thinking`
        # is ignored because the template never reads it.
        resolved = kwargs.get("enable_thinking")
        thinking_mode = kwargs.get("thinking_mode")
        if resolved is not None:
            prefilled_open = resolved is True
        elif thinking_mode is not None:
            prefilled_open = thinking_mode == "enabled"
        else:
            prefilled_open = False
        super().__init__(reasoning_at_start=prefilled_open,
                         chat_template_kwargs=chat_template_kwargs)
        self.reasoning_start = "<mm:think>"
        self.reasoning_end = "</mm:think>"

    def parse(self, text: str) -> ReasoningParserResult:
        if self.reasoning_at_start:
            # The prompt ends with the opening tag, so the output begins
            # inside the reasoning block.
            text = self.reasoning_start + text
        start_idx = text.find(self.reasoning_start)
        end_idx = text.find(self.reasoning_end)
        if start_idx != -1 and (end_idx == -1 or start_idx < end_idx):
            prefix = text[:start_idx]
            body = text[start_idx + len(self.reasoning_start):]
            if end_idx == -1:
                # An opening tag with no closing tag is reasoning cut off
                # (e.g. by max_tokens); streaming reports it as reasoning.
                return self._create_reasoning_end_result(prefix, body)
            reasoning_content, _, content = body.partition(self.reasoning_end)
            return self._create_reasoning_end_result(prefix + content,
                                                     reasoning_content)
        if end_idx == -1:
            # No tags at all: no reasoning block in this response.
            return ReasoningParserResult(content=text)
        # A closing tag with no opening tag is the empty-reasoning sentinel:
        # a delimiter, dropped; what follows is parsed by the same rules.
        rest = self.parse(text[end_idx + len(self.reasoning_end):])
        return self._create_reasoning_end_result(text[:end_idx] + rest.content,
                                                 rest.reasoning_content)

    def parse_delta(self, delta_text: str) -> ReasoningParserResult:
        # Outside a reasoning block the inherited parser only looks for the
        # opening tag, so a bare ``</mm:think>`` sentinel would leak into the
        # content; it drops text that precedes an opening tag in the same
        # delta although it reports that text as content when it arrives on
        # its own; and it only withholds a partial tag when the whole delta
        # is one. Resolve the three here so streaming matches ``parse`` for
        # every chunking; inside a reasoning block the inherited behaviour
        # already does.
        if self.in_reasoning:
            return super().parse_delta(delta_text)
        pending = self._buffer + delta_text
        self._buffer = ""
        start_idx = pending.find(self.reasoning_start)
        end_idx = pending.find(self.reasoning_end)
        if end_idx != -1 and (start_idx == -1 or end_idx < start_idx):
            # Sentinel: a delimiter, dropped; the rest is parsed as usual.
            before = pending[:end_idx]
            after = pending[end_idx + len(self.reasoning_end):]
        elif start_idx != -1:
            # Text before the opening tag is content; the inherited parser
            # takes over at the tag.
            rest = super().parse_delta(pending[start_idx:])
            return ReasoningParserResult(
                content=pending[:start_idx] + rest.content,
                reasoning_content=rest.reasoning_content)
        else:
            # Withhold a trailing fragment that could still grow into a tag.
            tag_start = pending.rfind("<")
            if tag_start != -1 and (
                    self.reasoning_start.startswith(pending[tag_start:])
                    or self.reasoning_end.startswith(pending[tag_start:])):
                self._buffer = pending[tag_start:]
                pending = pending[:tag_start]
            return ReasoningParserResult(content=pending)
        rest = (self.parse_delta(after) if after else ReasoningParserResult())
        return ReasoningParserResult(content=before + rest.content,
                                     reasoning_content=rest.reasoning_content)


MODEL_TYPE_TO_REASONING_PARSER: dict[str, str] = {
    "qwen3": "qwen3",
    "qwen3_moe": "qwen3",
    "qwen3_5": "qwen3",
    "qwen3_5_moe": "qwen3",
    "qwen3_next": "qwen3",
    "deepseek_v3": "deepseek-r1",
    "deepseek_v32": "deepseek-r1",
    "laguna": "poolside_v1",
    "deepseek_v4": "deepseek_v4",
    "deepseek_v41": "deepseek_v41",
    "deepseek_v41_text": "deepseek_v41",
    "nemotron_h": "nemotron-v3",
    "nemotron_h_puzzle": "nemotron-v3",
    "nemotron_h_omni": "nemotron-v3",
    "exaone_moe": "k-exaone",
    "gemma4": "gemma4",
    "glm_moe_dsa": "glm_moe_dsa",
    "kimi_k2": "kimi_k2",
    "kimi_k25": "kimi_k25",
    "kimi_k3": "kimi_k3",
    "minimax_m3": "minimax_m3",
    "minimax_m3_vl": "minimax_m3",
}

_QWEN3_MODEL_TYPES = frozenset({
    "qwen3",
    "qwen3_moe",
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_next",
})


def _resolve_qwen3_reasoning_parser(model: str) -> Optional[str]:
    """Distinguish Qwen3 hybrid / forced-thinking / forced-non-thinking models.

    The Qwen3 family has three reasoning variants with different chat templates:
    - **Hybrid** (e.g. Qwen3-235B-A22B): the template contains an
      ``enable_thinking`` flag that lets users toggle ``<think>`` on/off.
      → use the ``"qwen3"`` reasoning parser.
    - **Forced-thinking** (e.g. Qwen3-235B-A22B-Thinking-2507): the template
      always injects ``<think>`` in the generation prompt without any toggle.
      → use the ``"deepseek-r1"`` parser (``reasoning_at_start=True``).
    - **Forced-non-thinking** (e.g. Qwen3-235B-A22B-Instruct-2507): the
      template never injects ``<think>``.
      → no reasoning parser needed (returns ``None``).
    """
    tokenizer_config_path = Path(model) / "tokenizer_config.json"
    if not tokenizer_config_path.exists():
        logger.warning(
            "Cannot read tokenizer_config.json for Qwen3 model at '%s'. "
            "Defaulting to 'qwen3' reasoning parser. If this is a "
            "forced-thinking model (*-Thinking-*), use "
            "'--reasoning_parser deepseek-r1' instead.",
            model,
        )
        return "qwen3"

    with open(tokenizer_config_path) as f:
        tokenizer_config = json.load(f)

    chat_template = tokenizer_config.get("chat_template", "")

    if "enable_thinking" in chat_template:
        # Hybrid model: has enable_thinking toggle.
        return "qwen3"

    if "<think>" in chat_template:
        # Forced-thinking model: always injects <think> tag.
        logger.info(
            "Detected forced-thinking Qwen3 model (no enable_thinking "
            "toggle, but <think> tag present in chat template). "
            "Using 'deepseek-r1' reasoning parser.", )
        return "deepseek-r1"

    # Forced-non-thinking model: no <think> tag at all.
    logger.info(
        "Detected forced-non-thinking Qwen3 model (no <think> tag in "
        "chat template). No reasoning parser needed.", )
    return None


def resolve_auto_reasoning_parser(model: str) -> Optional[str]:
    """Resolve 'auto' reasoning parser by reading the model's HF config.

    For DeepSeek models, only maps to deepseek-r1 if the model path
    suggests it is a reasoning model (contains 'R1' in the name).

    For Qwen3 models, inspects the chat template to distinguish hybrid,
    forced-thinking, and forced-non-thinking variants.
    """
    config_path = Path(model) / "config.json"
    if not config_path.exists():
        return None

    with open(config_path) as f:
        config = json.load(f)

    model_type = config.get("model_type", "")

    if model_type in ("deepseek_v3", "deepseek_v32"):
        model_name = Path(model).name.lower()
        if "r1" not in model_name:
            return None

    if model_type in _QWEN3_MODEL_TYPES:
        return _resolve_qwen3_reasoning_parser(model)

    return MODEL_TYPE_TO_REASONING_PARSER.get(model_type)


@register_reasoning_parser("nemotron-v3")
@register_reasoning_parser("nano-v3")
@register_reasoning_parser("k-exaone")
class NemotronV3ReasoningParser(DeepSeekR1Parser):
    """Reasoning parser for Nemotron Nano v3.

    If the model is with reasoning (default behavior), `reasoning_at_start` is `True` and the
    starting response is parsed into `reasoning_content`.
    When the model is without reasoning, `reasoning_at_start` is `False` so the response is parsed
    into `content` fields.

    The `enable_thinking` flag is read from `chat_template_kwargs`.
    """

    resolves_thinking_from_prompt = True
    reasoning_start = "<think>"
    reasoning_end = "</think>"

    def __init__(self,
                 *,
                 reasoning_at_start: bool = True,
                 chat_template_kwargs: Optional[dict[str, Any]] = None) -> None:
        self._force_nonempty_content = False
        if isinstance(chat_template_kwargs, dict):
            reasoning_at_start = chat_template_kwargs.get(
                "enable_thinking", reasoning_at_start)
            self._force_nonempty_content = chat_template_kwargs.get(
                "force_nonempty_content", False) is True
        super().__init__(reasoning_at_start=reasoning_at_start,
                         chat_template_kwargs=chat_template_kwargs)
        self._tool_call_start = "<tool_call>"
        # Workaround: the model sometimes does not send closing think tags
        # which affects downstream applications. This is addressed by
        # optionally accumulating reasoning tokens and returning them as
        # content at the end of streaming.
        self._accumulated_reasoning = ""
        self._found_closing_tag = False

    def _maybe_swap_content(
            self, result: ReasoningParserResult) -> ReasoningParserResult:
        """When force_nonempty_content is set and content is empty, move
        reasoning_content into content so the response always has content.

        Whitespace-only content (e.g. a newline after the closing think tag) is
        treated as empty so the swap still runs (NVBug 6060281)."""
        content = result.content or ""
        if self._force_nonempty_content and not content.strip(
        ) and result.reasoning_content:
            return ReasoningParserResult(content=result.reasoning_content,
                                         reasoning_content="")
        return result

    def parse_delta(self, delta_text: str) -> ReasoningParserResult:
        """End reasoning at the first closing marker or tool-call opener."""
        buffered = self._buffer + delta_text
        tool_idx = buffered.find(self._tool_call_start)
        end_idx = buffered.find(self.reasoning_end)
        if self.in_reasoning and tool_idx != -1 and (end_idx == -1
                                                     or tool_idx < end_idx):
            self._buffer = ""
            self.in_reasoning = False
            reasoning = buffered[:tool_idx]
            content = buffered[tool_idx:]
            if self._force_nonempty_content:
                self._found_closing_tag = True
                self._accumulated_reasoning = ""
            return ReasoningParserResult(content=content,
                                         reasoning_content=reasoning)

        pending = ""
        if self.in_reasoning and end_idx == -1 and tool_idx == -1:
            for size in range(
                    min(len(buffered),
                        len(self._tool_call_start) - 1), 0, -1):
                if buffered.endswith(self._tool_call_start[:size]):
                    pending = buffered[-size:]
                    self._buffer = ""
                    delta_text = buffered[:-size]
                    break
        was_in_reasoning = self.in_reasoning
        result = super().parse_delta(delta_text)
        self._buffer += pending
        if self._force_nonempty_content:
            if result.reasoning_content:
                self._accumulated_reasoning += result.reasoning_content
            if was_in_reasoning and not self.in_reasoning:
                self._found_closing_tag = True
                self._accumulated_reasoning = ""
        return result

    def finish(self) -> ReasoningParserResult:
        """Called when the stream ends.

        If no closing think tag was found and force_nonempty_content is
        set, returns the full accumulated reasoning as content so the
        response is never empty. If no closing tag was found and
        force_nonempty_content is not set, returns any remaining buffer
        as reasoning_content since we are still in reasoning mode.

        If the closing tag was already found (or reasoning was never
        entered), flushes any remaining buffer as content."""
        if self.in_reasoning and not self._found_closing_tag:
            remaining = self._buffer
            self._buffer = ""
            if self._force_nonempty_content:
                all_content = self._accumulated_reasoning + remaining
                self._accumulated_reasoning = ""
                self.in_reasoning = False
                return ReasoningParserResult(content=all_content)
            self._accumulated_reasoning = ""
            self.in_reasoning = False
            if remaining:
                return ReasoningParserResult(reasoning_content=remaining)
            return ReasoningParserResult()
        remaining = self._buffer
        self._buffer = ""
        if remaining:
            return ReasoningParserResult(content=remaining)
        return ReasoningParserResult()

    def parse(self, text: str) -> ReasoningParserResult:
        start = 0 if self.reasoning_at_start else text.find(
            self.reasoning_start)
        if start != -1:
            if not self.reasoning_at_start:
                start += len(self.reasoning_start)
            tool_idx = text.find(self._tool_call_start, start)
            end_idx = text.find(self.reasoning_end, start)
            if tool_idx != -1 and (end_idx == -1 or tool_idx < end_idx):
                return self._maybe_swap_content(
                    ReasoningParserResult(
                        content=text[tool_idx:],
                        reasoning_content=text[start:tool_idx]))
        return self._maybe_swap_content(super().parse(text))


@register_reasoning_parser("gemma4")
class Gemma4ReasoningParser(BaseReasoningParser):
    r"""Reasoning parser for Gemma 4.

    Gemma 4 emits reasoning inside a channel block delimited by the
    ``<|channel>`` and ``<channel|>`` special tokens, e.g.::

        <|channel>thought
        REASONING_CONTENT<channel|>VISIBLE_CONTENT

    When the chat template is rendered with ``enable_thinking=False``, the
    server prefills ``<|channel>thought\n<channel|>`` so the model emits
    content directly without a reasoning block. When ``enable_thinking=True``,
    the model decides when to open/close the channel and may emit multiple
    channel blocks interleaved with content.

    Because ``<|channel>`` / ``<channel|>`` are registered special tokens in
    the Gemma 4 tokenizer, callers must set ``skip_special_tokens=False`` (or
    use a tool parser with ``needs_raw_special_tokens=True``) to ensure the
    delimiters appear in the decoded text stream.
    """

    CHANNEL_OPEN = "<|channel>"
    CHANNEL_CLOSE = "<channel|>"

    def __init__(self,
                 *,
                 chat_template_kwargs: Optional[dict[str, Any]] = None) -> None:
        super().__init__(chat_template_kwargs=chat_template_kwargs)
        self.in_reasoning = False
        self._buffer = ""

    def parse(self, text: str) -> ReasoningParserResult:
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        i = 0
        n = len(text)
        while i < n:
            open_idx = text.find(self.CHANNEL_OPEN, i)
            if open_idx == -1:
                content_parts.append(text[i:])
                break
            content_parts.append(text[i:open_idx])
            body_start = open_idx + len(self.CHANNEL_OPEN)
            close_idx = text.find(self.CHANNEL_CLOSE, body_start)
            if close_idx == -1:
                # Unterminated channel: remainder is reasoning.
                reasoning_parts.append(text[body_start:])
                i = n
                break
            reasoning_parts.append(text[body_start:close_idx])
            i = close_idx + len(self.CHANNEL_CLOSE)
        return ReasoningParserResult(
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts),
        )

    def parse_delta(self, delta_text: str) -> ReasoningParserResult:
        self._buffer += delta_text
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        while True:
            if not self.in_reasoning:
                idx = self._buffer.find(self.CHANNEL_OPEN)
                if idx == -1:
                    hold = _trailing_partial_marker(self._buffer,
                                                    self.CHANNEL_OPEN)
                    emit_len = len(self._buffer) - hold
                    content_parts.append(self._buffer[:emit_len])
                    self._buffer = self._buffer[emit_len:]
                    break
                content_parts.append(self._buffer[:idx])
                self._buffer = self._buffer[idx + len(self.CHANNEL_OPEN):]
                self.in_reasoning = True
            else:
                idx = self._buffer.find(self.CHANNEL_CLOSE)
                if idx == -1:
                    hold = _trailing_partial_marker(self._buffer,
                                                    self.CHANNEL_CLOSE)
                    emit_len = len(self._buffer) - hold
                    reasoning_parts.append(self._buffer[:emit_len])
                    self._buffer = self._buffer[emit_len:]
                    break
                reasoning_parts.append(self._buffer[:idx])
                self._buffer = self._buffer[idx + len(self.CHANNEL_CLOSE):]
                self.in_reasoning = False
        return ReasoningParserResult(
            content="".join(content_parts),
            reasoning_content="".join(reasoning_parts),
        )

    def finish(self) -> ReasoningParserResult:
        remaining = self._buffer
        self._buffer = ""
        if not remaining:
            return ReasoningParserResult()
        if self.in_reasoning:
            return ReasoningParserResult(reasoning_content=remaining)
        return ReasoningParserResult(content=remaining)


@register_reasoning_parser("kimi_k2")
@register_reasoning_parser("kimi_k25", reasoning_at_start=True)
class KimiK2ReasoningParser(DeepSeekR1Parser):
    """Reasoning parser for Kimi-K2 and Kimi-K2.5 models.

    Extends DeepSeekR1Parser to support interleaved thinking where reasoning
    content may be implicitly ended by a tool call section. The model uses
    ``<think>...</think>`` tokens and may also start tool calls via
    ``<|tool_calls_section_begin|>`` without an explicit ``</think>`` tag.

    Supported patterns:

    * ``<think>reasoning</think>content`` – standard thinking
    * ``<think>reasoning<|tool_calls_section_begin|>...`` – interleaved
      thinking (reasoning interrupted by tool call)
    * ``content`` (no ``<think>``) – no reasoning

    For Kimi-K2.5, the chat template defaults to thinking mode (appends
    ``<think>`` to prompt). When ``thinking=False`` is passed via
    ``chat_template_kwargs``, the template appends ``<think></think>``
    instead, and the model output has no thinking tags — this parser
    dynamically adjusts ``reasoning_at_start`` accordingly.

    Adapted from:
    * vLLM ``vllm/reasoning/kimi_k2_reasoning_parser.py``
    * sglang ``sglang/srt/parser/reasoning_parser.py``
    """

    def __init__(self,
                 *,
                 reasoning_at_start: bool = False,
                 chat_template_kwargs: Optional[dict[str, Any]] = None) -> None:
        # For Kimi-K2.5: chat template defaults to thinking mode unless
        # thinking=False is explicitly passed. Override reasoning_at_start
        # based on the actual thinking state.
        if chat_template_kwargs is not None:
            thinking = chat_template_kwargs.get("thinking")
            if thinking is False:
                reasoning_at_start = False
        super().__init__(reasoning_at_start=reasoning_at_start,
                         chat_template_kwargs=chat_template_kwargs)
        self.tool_section_start = "<|tool_calls_section_begin|>"

    def parse(self, text: str) -> ReasoningParserResult:
        # Strip <think> tag if reasoning_at_start is False.
        if not self.reasoning_at_start:
            splits = text.partition(self.reasoning_start)
            if splits[1] == "":
                # No <think> tag found – entire text is content.
                return ReasoningParserResult(content=text)
            text = splits[2]

        # Find the earliest end marker: </think> or <|tool_calls_section_begin|>.
        end_idx = text.find(self.reasoning_end)
        tool_idx = text.find(self.tool_section_start)

        if end_idx != -1 and (tool_idx == -1 or end_idx <= tool_idx):
            # Standard </think> end.
            reasoning_content = text[:end_idx]
            content = text[end_idx + len(self.reasoning_end):]
        elif tool_idx != -1:
            # Implicit end: tool call section starts before any </think>.
            reasoning_content = text[:tool_idx]
            content = text[tool_idx:]
        else:
            # No end marker found.
            if self.reasoning_at_start:
                # reasoning_at_start=True but no </think>: this is
                # instant mode (thinking=False) where the model output
                # has no thinking tags — treat everything as content.
                reasoning_content = ""
                content = text
            else:
                # reasoning_at_start=False and we already stripped
                # <think>: text is incomplete reasoning (e.g. truncated
                # output) — treat everything as reasoning.
                reasoning_content = text
                content = ""

        return ReasoningParserResult(content=content,
                                     reasoning_content=reasoning_content)

    def _find_partial_tag_suffix(self, text: str) -> int:
        """Find trailing partial prefix of a special token at the end of text.

        Returns the index where the partial suffix starts, or -1 if none found.
        """
        last_lt = text.rfind("<")
        if last_lt != -1:
            suffix = text[last_lt:]
            if (self.reasoning_start.startswith(suffix)
                    or self.reasoning_end.startswith(suffix)
                    or self.tool_section_start.startswith(suffix)):
                return last_lt
        return -1

    def parse_delta(self, delta_text: str) -> ReasoningParserResult:
        self._buffer += delta_text
        delta_text = self._buffer
        reasoning_content = None

        # Wait if the buffer is a prefix of any special token.
        if (self.reasoning_start.startswith(delta_text)
                or self.reasoning_end.startswith(delta_text)
                or self.tool_section_start.startswith(delta_text)):
            return ReasoningParserResult()

        if not self.in_reasoning:
            begin_idx = delta_text.find(self.reasoning_start)
            if begin_idx == -1:
                # No <think> found -- check for partial start-tag at end.
                partial_idx = self._find_partial_tag_suffix(delta_text)
                if partial_idx != -1:
                    self._buffer = delta_text[partial_idx:]
                    return ReasoningParserResult(
                        content=delta_text[:partial_idx])
                self._buffer = ""
                return ReasoningParserResult(content=delta_text)
            self.in_reasoning = True
            reasoning_content = delta_text[begin_idx +
                                           len(self.reasoning_start):]

        if self.in_reasoning:
            delta_text = (reasoning_content
                          if reasoning_content is not None else delta_text)

            # Find the earliest end marker.
            end_idx = delta_text.find(self.reasoning_end)
            tool_idx = delta_text.find(self.tool_section_start)

            if end_idx != -1 and (tool_idx == -1 or end_idx <= tool_idx):
                # Standard </think> end.
                reasoning_content = delta_text[:end_idx]
                content = delta_text[end_idx + len(self.reasoning_end):]
                self.in_reasoning = False
                # Check for partial special tag at end of content.
                partial_idx = self._find_partial_tag_suffix(content)
                if partial_idx != -1:
                    self._buffer = content[partial_idx:]
                    content = content[:partial_idx]
                else:
                    self._buffer = ""
                return ReasoningParserResult(
                    content=content, reasoning_content=reasoning_content)
            elif tool_idx != -1:
                # Implicit end via tool-call section start.
                reasoning_content = delta_text[:tool_idx]
                content = delta_text[tool_idx:]
                self.in_reasoning = False
                self._buffer = ""
                return ReasoningParserResult(
                    content=content, reasoning_content=reasoning_content)

            # No complete end marker - check for partial tag at end of buffer
            # (could be a prefix of </think> or <|tool_calls_section_begin|>).
            last_lt = delta_text.rfind("<")
            if last_lt != -1:
                suffix = delta_text[last_lt:]
                if (self.reasoning_end.startswith(suffix)
                        or self.tool_section_start.startswith(suffix)):
                    self._buffer = suffix
                    reasoning_content = delta_text[:last_lt]
                    return ReasoningParserResult(
                        reasoning_content=reasoning_content)

            self._buffer = ""
            reasoning_content = delta_text
            return ReasoningParserResult(reasoning_content=reasoning_content)

        raise RuntimeError(
            "Unreachable code reached in `KimiK2ReasoningParser.parse_delta`")


# A `key="value"` attribute of a Kimi-K3 XTML tag header.
_XTML_ATTRIBUTE = re.compile(r'(\w+)="([^"]*)"')


@register_reasoning_parser("kimi_k3")
class KimiK3ReasoningParser(BaseReasoningParser):
    """Reasoning parser for Kimi-K3 XTML output.

    K3 renders assistant messages as an XTML tag stream built from the
    special tokens ``<|open|>`` / ``<|close|>`` / ``<|sep|>`` /
    ``<|end_of_msg|>`` with plain-text tag headers (see the checkpoint's
    ``encoding_k3.py``)::

        <|open|>think<|sep|>REASONING<|close|>think<|sep|>
        <|open|>response<|sep|>CONTENT<|close|>response<|sep|>
        [<|open|>tools<|sep|>...calls...<|close|>tools<|sep|>]
        <|close|>message<|sep|><|end_of_msg|>

    The generation prompt already ends inside ``<|open|>think<|sep|>``
    (thinking mode, the default) or ``<|open|>response<|sep|>``
    (``chat_template_kwargs={"thinking": False}``), so the model output
    begins mid-channel with no opening tag.

    Every generated byte lands in exactly one place:

    * the think channel is ``reasoning_content``, the response channel is
      ``content``;
    * text outside any channel - after ``<|close|>think<|sep|>`` with no
      ``<|open|>response<|sep|>``, after a tools section - is ``content``.
      A gap of pure whitespace between two structural tags is formatting
      and is consumed with them;
    * a tools section holding at least one well-formed call goes to
      ``content`` verbatim for the ``kimi_k3`` tool parser, except the
      ``think`` blocks between its calls, whose bodies are
      ``reasoning_content``. A "section" without a single well-formed call
      is the model writing about the format and stays, verbatim, in the
      channel it appeared in;
    * a tag is structural only where the grammar allows it: opening a
      channel other than the current one, closing the current one, or the
      ``<|close|>message<|sep|>`` / ``<|end_of_msg|>`` framing that ends the
      generation. Anywhere else it is literal text of the current channel.

    ``parse`` runs the state machine behind ``parse_delta`` + ``finish``
    over the whole text, so the one-shot reading equals the streamed one
    for every chunking.

    The grammar of a tools section (``scan_tools_section`` and its helpers)
    lives here and is shared with ``KimiK3ToolParser``, so the two parsers
    cannot disagree about where a section, a call or a value ends.

    Limitation: the parsers see decoded text only. Marker text the model
    spelled out of ordinary tokens decodes to the same string as the marker
    tokens (ids 163586-163589), so markup that is valid where it appears is
    read as structure - a complete call quoted inside reasoning is
    delivered as a call, a value that literally contains
    ``<|close|>argument<|sep|>`` followed by more structure is cut there,
    and an ``<|open|>response<|sep|>`` written inside the think channel
    switches channels. Such text is never dropped, but it can be routed to
    the wrong place.
    """

    needs_raw_special_tokens = True

    OPEN = "<|open|>"
    CLOSE = "<|close|>"
    SEP = "<|sep|>"
    EOM = "<|end_of_msg|>"
    TOOLS_OPEN = OPEN + "tools" + SEP
    TOOLS_END = CLOSE + "tools" + SEP
    MESSAGE_END = CLOSE + "message" + SEP

    _THINK_OPEN = OPEN + "think" + SEP
    _THINK_CLOSE = CLOSE + "think" + SEP
    _CALL_CLOSE = CLOSE + "call" + SEP
    # Channel tags this parser acts on; any other tag is text.
    _OPEN_TAGS = ("think", "response", "tools")
    _CLOSE_TAGS = ("think", "response", "tools", "message")
    # Structural only where it ends the generation.
    _FRAMING = (MESSAGE_END, EOM)
    # Message-level structure a tools section cannot contain: it ends a
    # section the model never closed (the tag itself is not consumed).
    _SECTION_BREAKS = (TOOLS_OPEN, _THINK_CLOSE, OPEN + "response" + SEP,
                       CLOSE + "response" + SEP, MESSAGE_END, EOM)
    _SECTION_ENDS = (TOOLS_END, ) + _SECTION_BREAKS
    _LONGEST_SECTION_END = max(len(tag) for tag in _SECTION_ENDS)
    # What may follow the close tag of an argument value: the next element,
    # the end of the call, or an enclosing tag.
    _VALUE_FOLLOWER_OPENS = ("argument", "json", "call", "tools", "think",
                             "response")
    _VALUE_FOLLOWER_TAGS = (_CALL_CLOSE, TOOLS_END, _THINK_CLOSE,
                            CLOSE + "response" + SEP, MESSAGE_END, EOM)
    # What may follow the close tag of a call: the next call, a think block,
    # or the end of the section.
    _CALL_FOLLOWER_OPENS = ("call", "think")
    # Close tags that can only end structure: a run of them (with framing
    # and whitespace) up to the end of the generation is the model closing
    # its message, whatever it left open.
    _CLOSE_RUN_TAGS = (CLOSE + "argument" + SEP, CLOSE + "json" + SEP,
                       _CALL_CLOSE, TOOLS_END, CLOSE + "response" + SEP,
                       MESSAGE_END, EOM)

    @dataclass(frozen=True)
    class XtmlElement:
        """An ``argument`` or ``json`` block of a call.

        ``kind`` is the tag name, ``attrs`` its unescaped attributes and
        ``value_start:value_end`` the span of its raw value text.
        """
        kind: str
        attrs: dict
        value_start: int
        value_end: int

    @dataclass(frozen=True)
    class XtmlItem:
        """One piece of a tools section body; the items tile the body.

        ``kind`` is ``"call"`` (a complete call block: ``attrs`` and
        ``elements`` filled in), ``"think"`` (a think block between calls,
        body at ``body_start:body_end``) or ``"gap"`` (anything else:
        whitespace, prose, a malformed or cut-off call). ``recovered`` names
        the close tags a call was read without (see ``_scan_call``); empty
        for a call written in full.
        """
        kind: str
        start: int
        end: int
        attrs: Optional[dict] = None
        elements: tuple = ()
        body_start: int = 0
        body_end: int = 0
        recovered: str = ""

        @property
        def deliverable(self) -> bool:
            """A complete call naming its tool, with every argument keyed."""
            return (self.kind == "call" and bool(self.attrs.get("tool"))
                    and all(element.kind != "argument" or "key" in element.attrs
                            for element in self.elements))

    @dataclass(frozen=True)
    class XtmlSection:
        """A scanned tools section.

        ``end`` is where the section stops: just past ``<|close|>tools<|sep|>``
        (``terminated``), or at the message-level tag or end of text that cut
        an unclosed section short.
        """
        end: int
        terminated: bool
        items: tuple

    def __init__(self,
                 *,
                 chat_template_kwargs: Optional[dict[str, Any]] = None) -> None:
        super().__init__(chat_template_kwargs=chat_template_kwargs)
        thinking = True
        if chat_template_kwargs is not None:
            thinking = chat_template_kwargs.get("thinking", True) is not False
        # Channel the model starts generating in (its opening tag is part
        # of the prompt).
        self._initial_channel = "think" if thinking else "response"
        self._reset()

    def _reset(self) -> None:
        self._buffer = ""
        # 'think' | 'response' | None (between channels).
        self._channel: Optional[str] = self._initial_channel
        # Inside a tools section: _buffer holds it from its opening tag on.
        self._in_section = False
        self._section_channel: Optional[str] = None
        self._section_checked = 0
        # Between channels: whitespace held until the gap shows whether it
        # is text (non-whitespace follows) or formatting (a tag follows).
        self._gap_ws = ""
        self._gap_is_text = False

    # ------------------------------------------------------------------
    # XTML grammar, shared with KimiK3ToolParser. Each predicate answers
    # True / False, or None while the text seen so far cannot tell; with
    # final=True the text is complete and the answer is never None.
    # ------------------------------------------------------------------

    @staticmethod
    def parse_xtml_attrs(header: str) -> dict[str, str]:
        """The ``key="value"`` attributes of a tag header, unescaped.

        The renderer escapes only ``&`` and ``"`` in attribute values.
        """
        return {
            key: value.replace("&quot;", '"').replace("&amp;", "&")
            for key, value in _XTML_ATTRIBUTE.findall(header)
        }

    @staticmethod
    def _match_prefix(text: str, pos: int, token: str,
                      final: bool) -> Optional[bool]:
        """Whether `token` starts at `pos` (None: the text ends inside it)."""
        piece = text[pos:pos + len(token)]
        if piece == token:
            return True
        if not final and len(piece) < len(token) and token.startswith(piece):
            return None
        return False

    @classmethod
    def _open_name_at(cls, text: str, pos: int, names: tuple,
                      final: bool) -> Optional[Any]:
        """Match ``<|open|>NAME`` at `pos`, its header going on after it.

        Returns the name when whitespace or ``<|sep|>`` follows it, None if
        undecided, else False.
        """
        undecided = False
        for name in names:
            head = cls.OPEN + name
            found = cls._match_prefix(text, pos, head, final)
            if found is None:
                undecided = True
            elif found:
                end = pos + len(head)
                if end == len(text):
                    undecided = undecided or not final
                elif text[end].isspace() or text[end] == "<":
                    return name
        return None if undecided else False

    @classmethod
    def _open_tag_at(cls, text: str, pos: int, names: tuple,
                     final: bool) -> Optional[Any]:
        """Match a complete ``<|open|>NAME ...<|sep|>`` header at `pos`.

        Returns ``(name, attrs, end)`` on a match, None if undecided, else
        False.
        """
        name = cls._open_name_at(text, pos, names, final)
        if not name:
            return name
        header = pos + len(cls.OPEN) + len(name)
        # The header runs to <|sep|> and cannot hold any other marker.
        marker = text.find("<|", header)
        if marker == -1:
            return False if final else None
        sep = cls._match_prefix(text, marker, cls.SEP, final)
        if not sep:
            return sep
        return (name, cls.parse_xtml_attrs(text[header:marker]),
                marker + len(cls.SEP))

    @classmethod
    def _followed_by_structure(cls, text: str, pos: int,
                               final: bool) -> Optional[bool]:
        """Whether the value close tag ending at `pos` really closes the value.

        In decoded text a close tag the value contains is indistinguishable
        from the real one, but only the real one is followed - after
        optional whitespace - by more structure (the next argument, the end
        of the call, an enclosing tag) or by the end of the generation.
        """
        n = len(text)
        m = pos
        while m < n and text[m].isspace():
            m += 1
        if m == n:
            return True if final else None
        found = cls._open_name_at(text, m, cls._VALUE_FOLLOWER_OPENS, final)
        if found:
            return True
        undecided = found is None
        for tag in cls._VALUE_FOLLOWER_TAGS:
            found = cls._match_prefix(text, m, tag, final)
            if found:
                return True
            undecided = undecided or found is None
        return None if undecided else False

    @classmethod
    def _close_run(cls, text: str, pos: int, final: bool) -> Optional[bool]:
        """Whether ``text[pos:]`` is close tags and whitespace up to the end.

        Such a run ending the generation is the model closing its message,
        whatever it left open. It must hold at least one tag.
        """
        n = len(text)
        m = pos
        seen = False
        while True:
            while m < n and text[m].isspace():
                m += 1
            if m == n:
                return seen if final else None
            undecided = False
            for tag in cls._CLOSE_RUN_TAGS:
                found = cls._match_prefix(text, m, tag, final)
                if found:
                    m += len(tag)
                    seen = True
                    break
                undecided = undecided or found is None
            else:
                return None if undecided else False

    @classmethod
    def _call_close_ends_call(cls, text: str, pos: int,
                              final: bool) -> Optional[bool]:
        """Whether the call close tag ending at `pos` ends an unclosed value.

        Only where a call can end: before the next call or a think block, at
        the end of the section, or before a run of close tags that ends the
        generation.
        """
        n = len(text)
        m = pos
        while m < n and text[m].isspace():
            m += 1
        if m == n:
            return True if final else None
        found = cls._open_name_at(text, m, cls._CALL_FOLLOWER_OPENS, final)
        if found:
            return True
        undecided = found is None
        for tag in (cls.TOOLS_END, ) + cls._SECTION_BREAKS:
            match = cls._match_prefix(text, m, tag, final)
            if match:
                return True
            undecided = undecided or match is None
        return None if undecided else False

    @classmethod
    def _value_end(cls, text: str, start: int, close: str,
                   final: bool) -> Optional[Any]:
        """Find where the value starting at `start` ends, and how.

        Returns ``(index, how)``. ``how`` is ``"close"`` when the value's own
        close tag ends it, ``"call"`` when the model went straight on to the
        call's close tag, and ``"run"`` when it closed its message instead (a
        run of close tags that ends the generation). Returns -1 if the text
        ends inside the value, or None if undecided.

        A close tag ends the value only where structure follows it. The value's
        own tag must be followed by the next element, the end of the call or
        an enclosing tag; the call's tag must be where a call can end. Any
        other close tag in the value is part of it.
        """
        search = start
        while True:
            at = text.find(cls.CLOSE, search)
            if at == -1:
                return -1 if final else None
            if text.startswith(close, at):
                ends = cls._followed_by_structure(text, at + len(close), final)
                how = "close"
            elif text.startswith(cls._CALL_CLOSE, at):
                ends = cls._call_close_ends_call(text,
                                                 at + len(cls._CALL_CLOSE),
                                                 final)
                how = "call"
            else:
                ends = cls._close_run(text, at, final)
                how = "run"
            if ends is None:
                return None
            if ends:
                return at, how
            search = at + 1

    @classmethod
    def _past_stray_closes(cls, text: str, pos: int,
                           final: bool) -> Optional[int]:
        """Where a call read without its close tags ends: past stray close tags.

        A model that left a value or a call open often closes its message
        with close tags of things it never opened, ``<|close|>response<|sep|>``
        after a call. When those tags run to the end of the generation they
        belong to the malformed call, not to the content after the section.
        The section's own close tag and the message framing are left to the
        section and the framing.
        """
        run = cls._close_run(text, pos, final)
        if run is None:
            return None
        if not run:
            return pos
        stray = (cls.CLOSE + "argument" + cls.SEP, cls.CLOSE + "json" + cls.SEP,
                 cls._CALL_CLOSE, cls.CLOSE + "response" + cls.SEP)
        n = len(text)
        end = pos
        while True:
            m = end
            while m < n and text[m].isspace():
                m += 1
            tag = next((tag for tag in stray if text.startswith(tag, m)), None)
            if tag is None:
                return end
            end = m + len(tag)

    @classmethod
    def _unclosed_call_end(cls, text: str, pos: int,
                           final: bool) -> Optional[Any]:
        """Where a call whose last element is complete ends without its close tag.

        Before a run of close tags that ends the generation it ends past the
        stray ones. Before the next call, a think block, the end of the
        section or message-level structure, it ends where that structure
        starts. Returns the end, False where the call grammar is broken
        instead, or None if undecided.
        """
        found = cls._open_name_at(text, pos, cls._CALL_FOLLOWER_OPENS, final)
        if found:
            return pos
        undecided = found is None
        # Whether stray close tags are absorbed depends on whether they run to
        # the end of the generation, so that is settled first.
        run = cls._close_run(text, pos, final)
        if run is None:
            return None
        if run:
            return cls._past_stray_closes(text, pos, final)
        for tag in (cls.TOOLS_END, ) + cls._SECTION_BREAKS:
            match = cls._match_prefix(text, pos, tag, final)
            if match:
                return pos
            undecided = undecided or match is None
        return None if undecided else False

    @classmethod
    def _scan_call(cls, text: str, pos: int, final: bool) -> Optional[Any]:
        """Read the call block whose header starts at `pos`.

        A call is also read when the model left out close tags. The outcome is
        not in doubt in three cases, recorded in ``XtmlItem.recovered``:
        - a value followed directly by the call's close tag;
        - a value or a call left open by a run of close tags that ends the
          generation;
        - a call whose last element is complete, followed by the next call,
          a think block, the end of the section or message-level structure.
        A generation that simply stops inside a call is still not a call:
        the cut-off value may be incomplete.

        Returns:
            An ``XtmlItem`` for a complete call; an int, the position where
            the call grammar broke (``len(text)`` when the text ends inside
            the call) - everything from `pos` up to it is not a call; None
            if undecided; False if no call header starts at `pos`.
        """
        header = cls._open_tag_at(text, pos, ("call", ), final)
        if not header:
            return header
        _, attrs, cursor = header
        n = len(text)
        elements = []

        def call(end: int,
                 recovered: str = "") -> "KimiK3ReasoningParser.XtmlItem":
            return cls.XtmlItem("call",
                                pos,
                                end,
                                attrs=attrs,
                                elements=tuple(elements),
                                recovered=recovered)

        # A call body is one json block or any number of arguments.
        allowed = ("argument", "json")
        while True:
            m = cursor
            while m < n and text[m].isspace():
                m += 1
            if m == n:
                return n if final else None
            closed = cls._match_prefix(text, m, cls._CALL_CLOSE, final)
            if closed:
                return call(m + len(cls._CALL_CLOSE))
            tag = cls._open_tag_at(text, m, allowed,
                                   final) if allowed else False
            if tag is None or (closed is None and not tag):
                return None
            if not tag:
                if not elements:
                    return m
                end = cls._unclosed_call_end(text, m, final)
                if end is None:
                    return None
                if end is False:
                    return m
                return call(end, f"no {cls._CALL_CLOSE}")
            kind, element_attrs, value_start = tag
            close = cls.CLOSE + kind + cls.SEP
            found = cls._value_end(text, value_start, close, final)
            if found is None:
                return None
            if found == -1:
                return n
            value_end, how = found
            elements.append(
                cls.XtmlElement(kind, element_attrs, value_start, value_end))
            if how == "call":
                end = cls._past_stray_closes(text,
                                             value_end + len(cls._CALL_CLOSE),
                                             final)
                if end is None:
                    return None
                return call(end, f"no {close}")
            if how == "run":
                end = cls._past_stray_closes(text, value_end, final)
                if end is None:
                    return None
                return call(end, f"no {close} or {cls._CALL_CLOSE}")
            cursor = value_end + len(close)
            allowed = ("argument", ) if kind == "argument" else ()

    @classmethod
    def _scan_think(cls, text: str, pos: int, final: bool) -> Optional[Any]:
        """An ``XtmlItem`` for the think block at `pos`; None/False."""
        opened = cls._match_prefix(text, pos, cls._THINK_OPEN, final)
        if not opened:
            return opened
        body_start = pos + len(cls._THINK_OPEN)
        body_end = text.find(cls._THINK_CLOSE, body_start)
        if body_end == -1:
            if not final:
                return None
            return cls.XtmlItem("think",
                                pos,
                                len(text),
                                body_start=body_start,
                                body_end=len(text))
        return cls.XtmlItem("think",
                            pos,
                            body_end + len(cls._THINK_CLOSE),
                            body_start=body_start,
                            body_end=body_end)

    @classmethod
    def tools_section_opens(cls, text: str, pos: int,
                            final: bool) -> Optional[bool]:
        """Whether the ``<|open|>tools<|sep|>`` ending at `pos` opens a section.

        A section starts with a call; the tag followed by anything else is
        the model writing about the format, and is text.
        """
        n = len(text)
        m = pos
        while m < n and text[m].isspace():
            m += 1
        if m == n:
            return False if final else None
        name = cls._open_name_at(text, m, ("call", ), final)
        return None if name is None else bool(name)

    @classmethod
    def section_may_end(cls, text: str, checked: int) -> bool:
        """Cheap gate for `scan_tools_section` on a growing section.

        A section can only end at one of `_SECTION_ENDS`; a scan that came
        back undecided over the first `checked` characters stays undecided
        until a new one of those tags arrives.
        """
        start = max(len(cls.TOOLS_OPEN), checked - cls._LONGEST_SECTION_END + 1)
        return any(text.find(tag, start) != -1 for tag in cls._SECTION_ENDS)

    @classmethod
    def scan_tools_section(cls, text: str, start: int,
                           final: bool) -> Optional["XtmlSection"]:
        """Read the tools section whose body starts at `start`.

        `start` is just past an opening ``<|open|>tools<|sep|>``. Returns
        None while the text seen so far does not determine where the
        section ends.
        """
        n = len(text)
        items = []
        gap_start = None

        def end_gap(at: int) -> None:
            nonlocal gap_start
            if gap_start is not None and at > gap_start:
                items.append(cls.XtmlItem("gap", gap_start, at))
            gap_start = None

        pos = start
        while True:
            if pos >= n:
                if not final:
                    return None
                end_gap(n)
                return cls.XtmlSection(n, False, tuple(items))
            tag_start = text.find("<", pos)
            if tag_start != pos:
                if gap_start is None:
                    gap_start = pos
                pos = n if tag_start == -1 else tag_start
                continue
            undecided = False
            closed = cls._match_prefix(text, pos, cls.TOOLS_END, final)
            if closed:
                end_gap(pos)
                return cls.XtmlSection(pos + len(cls.TOOLS_END), True,
                                       tuple(items))
            undecided = closed is None
            for tag in cls._SECTION_BREAKS:
                broken = cls._match_prefix(text, pos, tag, final)
                if broken:
                    end_gap(pos)
                    return cls.XtmlSection(pos, False, tuple(items))
                undecided = undecided or broken is None
            call = cls._scan_call(text, pos, final)
            if call is None:
                return None
            if isinstance(call, cls.XtmlItem):
                end_gap(pos)
                items.append(call)
                pos = call.end
                continue
            if call is not False:
                # Not a call after all: its bytes are part of the gap.
                if gap_start is None:
                    gap_start = pos
                pos = call
                continue
            think = cls._scan_think(text, pos, final)
            if think is None or (undecided and not think):
                return None
            if think:
                end_gap(pos)
                items.append(think)
                pos = think.end
                continue
            if gap_start is None:
                gap_start = pos
            pos += 1

    @classmethod
    def framing_ends_text(cls, text: str, pos: int,
                          final: bool) -> Optional[bool]:
        """Whether ``text[pos:]`` is only framing (and whitespace after it)."""
        n = len(text)
        while True:
            undecided = False
            for tag in cls._FRAMING:
                found = cls._match_prefix(text, pos, tag, final)
                if found:
                    pos += len(tag)
                    break
                undecided = undecided or found is None
            else:
                return None if undecided else False
            while pos < n and text[pos].isspace():
                pos += 1
            if pos == n:
                return True if final else None

    @classmethod
    def _framing_suffix_start(cls, text: str, end: int) -> Optional[int]:
        """Start of the framing run (with its whitespace) that ends at `end`.

        None if no framing tag ends there.
        """
        start = None
        while True:
            cut = end
            while cut > 0 and text[cut - 1].isspace():
                cut -= 1
            for tag in cls._FRAMING:
                if text.endswith(tag, 0, cut):
                    end = start = cut - len(tag)
                    break
            else:
                return start

    @classmethod
    def strip_trailing_framing(cls, text: str) -> str:
        """`text` without the framing that ends the generation."""
        start = cls._framing_suffix_start(text, len(text))
        return text if start is None else text[:start]

    @classmethod
    def trailing_framing_hold(cls, text: str) -> int:
        """Length of the longest suffix of `text` that may still be framing.

        A stream holds it back; it is stripped if the generation ends there.
        """
        n = len(text)
        hold = 0
        longest = max(len(tag) for tag in cls._FRAMING)
        for partial in range(min(n, longest - 1) + 1):
            tail = text[n - partial:]
            if partial and not any(
                    tag.startswith(tail) for tag in cls._FRAMING):
                continue
            start = cls._framing_suffix_start(text, n - partial)
            if start is not None:
                hold = max(hold, n - start)
            elif partial:
                hold = max(hold, partial)
        return hold

    # ------------------------------------------------------------------
    # Channel routing
    # ------------------------------------------------------------------

    def _end_gap(self) -> None:
        """End the gap between channels at a structural tag.

        Whitespace still held in the gap was formatting.
        """
        self._gap_ws = ""
        self._gap_is_text = False

    def _emit(self, text: str, content: list, reasoning: list) -> None:
        if not text:
            return
        if self._channel == "think":
            reasoning.append(text)
        elif self._channel == "response" or self._gap_is_text:
            content.append(text)
        elif text.isspace():
            self._gap_ws += text
        else:
            # Text between channels is still the model's answer.
            content.append(self._gap_ws + text)
            self._gap_ws = ""
            self._gap_is_text = True

    def _consume(self, length: int) -> bool:
        self._buffer = self._buffer[length:]
        return True

    def _on_open(self, final: bool, content: list, reasoning: list) -> bool:
        buf = self._buffer
        for name in self._OPEN_TAGS:
            tag = self.OPEN + name + self.SEP
            found = self._match_prefix(buf, 0, tag, final)
            if found is None:
                return False
            if not found:
                continue
            if name == "tools":
                opens = self.tools_section_opens(buf, len(tag), final)
                if opens is None:
                    return False
                if opens:
                    self._in_section = True
                    self._section_channel = self._channel
                    self._section_checked = 0
                    return True
            elif name != self._channel:
                self._channel = name
                self._end_gap()
                return self._consume(len(tag))
            break
        self._emit(self.OPEN, content, reasoning)
        return self._consume(len(self.OPEN))

    def _on_close(self, final: bool, content: list, reasoning: list) -> bool:
        buf = self._buffer
        for name in self._CLOSE_TAGS:
            tag = self.CLOSE + name + self.SEP
            found = self._match_prefix(buf, 0, tag, final)
            if found is None:
                return False
            if not found:
                continue
            if name == "message":
                return self._on_framing(final, content, reasoning)
            if name == self._channel:
                self._channel = None
                self._end_gap()
                return self._consume(len(tag))
            break
        self._emit(self.CLOSE, content, reasoning)
        return self._consume(len(self.CLOSE))

    def _on_framing(self, final: bool, content: list, reasoning: list) -> bool:
        ends = self.framing_ends_text(self._buffer, 0, final)
        if ends is None:
            return False
        if ends:
            self._buffer = ""
            self._end_gap()
            return False
        marker = self.EOM if self._buffer.startswith(self.EOM) else self.CLOSE
        self._emit(marker, content, reasoning)
        return self._consume(len(marker))

    def _step_section(self, final: bool, content: list,
                      reasoning: list) -> bool:
        buf = self._buffer
        if final:
            buf = self.strip_trailing_framing(buf)
        elif not self.section_may_end(buf, self._section_checked):
            self._section_checked = len(buf)
            return False
        section = self.scan_tools_section(buf, len(self.TOOLS_OPEN), final)
        if section is None:
            self._section_checked = len(buf)
            return False
        self._in_section = False
        self._channel = self._section_channel
        self._buffer = buf[section.end:]
        if not any(item.deliverable for item in section.items):
            # Not a single call a client could run: prose about the format,
            # or a call cut off before it was complete. It stays where it
            # was written.
            self._emit(buf[:section.end], content, reasoning)
            return True
        self._end_gap()
        kept = [self.TOOLS_OPEN]
        for item in section.items:
            if item.kind == "think":
                reasoning.append(buf[item.body_start:item.body_end])
            else:
                kept.append(buf[item.start:item.end])
        if section.terminated:
            kept.append(self.TOOLS_END)
        content.append("".join(kept))
        return True

    def _step(self, final: bool, content: list, reasoning: list) -> bool:
        """Consume as much of the buffer as can be decided.

        Returns False when more input is needed (or, with `final`, when the
        buffer is used up).
        """
        if self._in_section:
            return self._step_section(final, content, reasoning)
        buf = self._buffer
        found = [
            index for index in (buf.find(self.OPEN), buf.find(self.CLOSE),
                                buf.find(self.EOM)) if index != -1
        ]
        if not found:
            hold = 0 if final else _trailing_partial_marker(
                buf, self.OPEN, self.CLOSE, self.EOM)
            self._emit(buf[:len(buf) - hold], content, reasoning)
            self._buffer = buf[len(buf) - hold:]
            return False
        start = min(found)
        self._emit(buf[:start], content, reasoning)
        self._buffer = buf[start:]
        if self._buffer.startswith(self.OPEN):
            return self._on_open(final, content, reasoning)
        if self._buffer.startswith(self.CLOSE):
            return self._on_close(final, content, reasoning)
        return self._on_framing(final, content, reasoning)

    def _state(self) -> tuple:
        return (self._buffer, self._channel, self._in_section,
                self._section_channel, self._section_checked, self._gap_ws,
                self._gap_is_text)

    def _set_state(self, state: tuple) -> None:
        (self._buffer, self._channel, self._in_section, self._section_channel,
         self._section_checked, self._gap_ws, self._gap_is_text) = state

    def _run(self, final: bool) -> ReasoningParserResult:
        content: list = []
        reasoning: list = []
        while True:
            state = self._state()
            content_before, reasoning_before = len(content), len(reasoning)
            progressed = self._step(final, content, reasoning)
            if (not final and content_before
                    and len(reasoning) > reasoning_before):
                # A result cannot say whether its reasoning came before or
                # after its content, and the serving layer places reasoning
                # first. Reasoning that follows content (think after a tools
                # section, say) waits for the next delta, so each result
                # reads in order.
                self._set_state(state)
                del content[content_before:]
                del reasoning[reasoning_before:]
                break
            if not progressed:
                break
        return ReasoningParserResult(content="".join(content),
                                     reasoning_content="".join(reasoning))

    def _feed(self, text: str) -> ReasoningParserResult:
        self._buffer += text
        return self._run(final=False)

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
        """Decide everything still held: the generation has ended."""
        result = self._run(final=True)
        self._buffer = ""
        self._end_gap()
        return result
