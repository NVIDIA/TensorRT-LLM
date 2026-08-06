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
        return None

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
        self._entered_reasoning = self.reasoning_at_start
        thinking_disabled = (isinstance(chat_template_kwargs, dict)
                             and chat_template_kwargs.get("enable_thinking")
                             is False)
        # Hold pre-<think> text for qwen3-style parsers so stream matches
        # parse() drop of the preamble (#17296). Thinking-disabled Nemotron
        # paths still stream plain content live.
        self._hold_preamble = (not reasoning_at_start) and (
            not thinking_disabled)
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
        # When ``_hold_preamble`` is set (qwen3-style), text before the first
        # ``<think>`` is withheld and dropped when the tag arrives (#17296),
        # matching ``parse()``. After a block ends, later text / ``<think>``
        # blocks stream normally (interleaved thinking). Thinking-disabled
        # paths keep live content emission without a start tag.
        self._buffer += delta_text
        delta_text = self._buffer
        reasoning_content = None
        if (self.reasoning_start.startswith(delta_text)
                or self.reasoning_end.startswith(delta_text)):
            # waiting for more text to determine if it's a reasoning start or end tag
            return ReasoningParserResult()

        if not self.in_reasoning:
            begin_idx = delta_text.find(self.reasoning_start)
            if begin_idx == -1:
                if self._hold_preamble and not self._entered_reasoning:
                    self._buffer = delta_text
                    return ReasoningParserResult()
                # After the first reasoning block, a trailing suffix may be a
                # partial ``<think>`` split across deltas (``mid<th`` + ``ink>``).
                partial_start_idx = delta_text.rfind(self.reasoning_start[0])
                if (partial_start_idx != -1 and self.reasoning_start.startswith(
                        delta_text[partial_start_idx:])):
                    self._buffer = delta_text[partial_start_idx:]
                    return ReasoningParserResult(
                        content=delta_text[:partial_start_idx])
                self._buffer = ""
                return ReasoningParserResult(content=delta_text)
            # Start tag found.
            # First open: drop text before the tag (same as ``parse()``).
            # Later opens (interleaved): keep text before the tag as content.
            content_prefix = ""
            if begin_idx > 0:
                if self._entered_reasoning:
                    content_prefix = delta_text[:begin_idx]
                # else: initial preamble — drop
            self.in_reasoning = True
            self._entered_reasoning = True
            reasoning_content = delta_text[begin_idx +
                                           len(self.reasoning_start):]
        else:
            content_prefix = ""

        if self.in_reasoning:
            delta_text = (reasoning_content
                          if reasoning_content is not None else delta_text)
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
                    content=content_prefix, reasoning_content=reasoning_content)
            reasoning_content = delta_text[:end_idx]
            content_after = delta_text[end_idx + len(self.reasoning_end):]
            self.in_reasoning = False
            self._buffer = ""
            content = content_prefix + content_after
            # Remainder may open another reasoning block in the same delta
            # (e.g. buffered ``</think>`` + ``y<think>z``).
            if content_after:
                more = self.parse_delta(content_after)
                # parse_delta re-appended content_after to the buffer; the
                # recursive call already consumed it — avoid double count:
                # we passed content_after both as emitted content_prefix path
                # and recursively. Fix: only take recursive result, do not
                # pre-include content_after in content.
                content = content_prefix + more.content
                reasoning_content = reasoning_content + (more.reasoning_content
                                                         or "")
            return ReasoningParserResult(content=content,
                                         reasoning_content=reasoning_content)
        raise RuntimeError(
            "Unreachable code reached in `DeepSeekR1Parser.parse_delta`")

    def finish(self) -> ReasoningParserResult:
        """Flush text withheld by ``parse_delta`` when the stream ends.

        Trailing tag prefixes are ordinary model output once the stream ends.
        A complete end tag while inside reasoning is a delimiter. A complete
        start tag while not inside reasoning opens an empty block. A complete
        start tag while already inside reasoning is kept as literal text
        (``parse()`` parity for deepseek-r1 on ``"a<think>"``).
        """
        remaining = self._buffer
        self._buffer = ""
        if not remaining:
            return ReasoningParserResult()
        if remaining == self.reasoning_end and self.in_reasoning:
            self.in_reasoning = False
            return ReasoningParserResult()
        if remaining == self.reasoning_start and not self.in_reasoning:
            self.in_reasoning = True
            self._entered_reasoning = True
            return ReasoningParserResult()
        if remaining == self.reasoning_start and self.in_reasoning:
            return ReasoningParserResult(reasoning_content=remaining)
        if remaining == self.reasoning_end and not self.in_reasoning:
            return ReasoningParserResult(content=remaining)
        if self.in_reasoning:
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

    * Without reasoning (the template still emits a bare ``</mm:think>``
      as a sentinel so the model knows where the visible content starts)::

          </mm:think>{content}

    M3 is therefore not strictly ``reasoning_at_start`` — the leading
    ``<mm:think>`` may or may not be present — so we partition on the
    closing tag first and then strip an optional leading opening tag
    from the reasoning portion. This keeps streaming behavior identical
    to :class:`DeepSeekR1Parser` for the common (``<mm:think>...``) case
    while also handling the bare-sentinel form.
    """

    def __init__(self,
                 *,
                 chat_template_kwargs: Optional[dict[str, Any]] = None) -> None:
        super().__init__(reasoning_at_start=False,
                         chat_template_kwargs=chat_template_kwargs)
        self.reasoning_start = "<mm:think>"
        self.reasoning_end = "</mm:think>"

    def parse(self, text: str) -> ReasoningParserResult:
        end_idx = text.find(self.reasoning_end)
        if end_idx == -1:
            # No closing tag → no reasoning block in this response.
            return ReasoningParserResult(content=text)
        reasoning_content = text[:end_idx]
        content = text[end_idx + len(self.reasoning_end):]
        # Strip an optional leading <mm:think> from the reasoning portion
        # so the sentinel-only shape (no opening tag) reduces to
        # reasoning_content="".
        if reasoning_content.startswith(self.reasoning_start):
            reasoning_content = reasoning_content[len(self.reasoning_start):]
        return self._create_reasoning_end_result(content, reasoning_content)


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
    "nemotron_h": "nemotron-v3",
    "nemotron_h_puzzle": "nemotron-v3",
    "gemma4": "gemma4",
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
class NemotronV3ReasoningParser(DeepSeekR1Parser):
    """Reasoning parser for Nemotron Nano v3.

    If the model is with reasoning (default behavior), `reasoning_at_start` is `True` and the
    starting response is parsed into `reasoning_content`.
    When the model is without reasoning, `reasoning_at_start` is `False` so the response is parsed
    into `content` fields.

    The `enable_thinking` flag is read from `chat_template_kwargs`.
    """

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
        """Wraps the parent parse_delta to also treat `<tool_call>` as an
        implicit end-of-reasoning marker.  When the model omits `</think>`
        before generating a tool call, the tag would otherwise be absorbed
        into reasoning_content and the downstream tool parser would never
        see it (NVBug 6082303).

        `<tool_call>` is a special token that always arrives as a single
        atomic delta, so we only need to check `delta_text` (not the
        parent's internal buffer)."""
        if (self.in_reasoning and self._tool_call_start in delta_text
                and self.reasoning_end not in self._buffer):
            remaining = self._buffer
            self._buffer = ""
            self.in_reasoning = False
            # Guaranteed non-negative: guarded by `in delta_text` above.
            tool_idx = delta_text.find(self._tool_call_start)
            reasoning = remaining + delta_text[:tool_idx]
            content = delta_text[tool_idx:]
            if self._force_nonempty_content:
                self._found_closing_tag = True
                self._accumulated_reasoning = ""
            return ReasoningParserResult(content=content,
                                         reasoning_content=reasoning)

        was_in_reasoning = self.in_reasoning
        result = super().parse_delta(delta_text)
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
        result = super().parse(text)
        tc = (result.reasoning_content.find(self._tool_call_start)
              if result.reasoning_content else -1)
        if tc != -1:
            result = ReasoningParserResult(
                content=result.reasoning_content[tc:] + result.content,
                reasoning_content=result.reasoning_content[:tc])
        return self._maybe_swap_content(result)


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

    @staticmethod
    def _partial_suffix_len(buf: str, tag: str) -> int:
        """Return length of the longest suffix of ``buf`` that is a prefix of ``tag``.

        Used to hold back potential partial delimiters during streaming.
        """
        max_len = min(len(buf), len(tag) - 1)
        for k in range(max_len, 0, -1):
            if tag.startswith(buf[-k:]):
                return k
        return 0

    def parse_delta(self, delta_text: str) -> ReasoningParserResult:
        self._buffer += delta_text
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        while True:
            if not self.in_reasoning:
                idx = self._buffer.find(self.CHANNEL_OPEN)
                if idx == -1:
                    hold = self._partial_suffix_len(self._buffer,
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
                    hold = self._partial_suffix_len(self._buffer,
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

    This parser emits the think body as ``reasoning_content`` and the
    response body as ``content``. A ``tools`` section is passed through
    into ``content`` verbatim (special tokens included) so the ``kimi_k3``
    tool parser can consume it; all other structural markup is dropped.
    """

    needs_raw_special_tokens = True

    OPEN = "<|open|>"
    CLOSE = "<|close|>"
    SEP = "<|sep|>"
    EOM = "<|end_of_msg|>"
    TOOLS_END = CLOSE + "tools" + SEP

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
        # 'body' | 'open_header' | 'close_header' | 'tools_pass' | 'done'
        self._state = "body"
        self._channel: Optional[str] = self._initial_channel

    @staticmethod
    def _partial_suffix_len(text: str, markers: tuple[str, ...]) -> int:
        """Length of the longest text suffix that is a proper prefix of any marker.

        Markers may contain internal ``<`` (e.g. ``<|close|>tools<|sep|>``),
        so every suffix length up to ``len(marker) - 1`` must be checked, not
        just the one starting at the last ``<``.
        """
        best = 0
        for marker in markers:
            for length in range(min(len(text), len(marker) - 1), best, -1):
                if marker.startswith(text[-length:]):
                    best = length
                    break
        return best

    def _emit(self, text: str, content: list, reasoning: list) -> None:
        if not text:
            return
        if self._channel == "think":
            reasoning.append(text)
        elif self._channel == "response":
            content.append(text)
        # channel None: structural gap – dropped

    def _step(self, content: list, reasoning: list) -> bool:
        """Consume as much of the buffer as possible.

        Returns False when more input is needed.
        """
        buf = self._buffer
        if self._state == "done":
            self._buffer = ""
            return False
        if self._state == "body":
            markers = (self.OPEN, self.CLOSE, self.EOM)
            indices = [(buf.find(m), m) for m in markers]
            indices = [(i, m) for i, m in indices if i != -1]
            if not indices:
                hold = self._partial_suffix_len(buf, markers)
                emit_len = len(buf) - hold
                self._emit(buf[:emit_len], content, reasoning)
                self._buffer = buf[emit_len:]
                return False
            idx, marker = min(indices)
            self._emit(buf[:idx], content, reasoning)
            self._buffer = buf[idx + len(marker):]
            if marker == self.OPEN:
                self._state = "open_header"
            elif marker == self.CLOSE:
                self._state = "close_header"
            else:
                self._state = "done"
                self._buffer = ""
            return True
        if self._state in ("open_header", "close_header"):
            idx = buf.find(self.SEP)
            if idx == -1:
                return False
            header = buf[:idx]
            self._buffer = buf[idx + len(self.SEP):]
            tag = header.split(None, 1)[0] if header.split() else ""
            if self._state == "open_header":
                if tag == "think":
                    self._channel = "think"
                elif tag == "response":
                    self._channel = "response"
                elif tag == "tools":
                    # Replay the section opener for the tool parser.
                    self._channel = "response"
                    self._emit(self.OPEN + header + self.SEP, content,
                               reasoning)
                    self._state = "tools_pass"
                    return True
                else:
                    self._channel = None
            else:
                if tag == "message":
                    self._state = "done"
                    self._buffer = ""
                    return False
                self._channel = None
            self._state = "body"
            return True
        if self._state == "tools_pass":
            idx = buf.find(self.TOOLS_END)
            if idx == -1:
                hold = self._partial_suffix_len(buf, (self.TOOLS_END, ))
                emit_len = len(buf) - hold
                self._emit(buf[:emit_len], content, reasoning)
                self._buffer = buf[emit_len:]
                return False
            end = idx + len(self.TOOLS_END)
            self._emit(buf[:end], content, reasoning)
            self._buffer = buf[end:]
            self._channel = None
            self._state = "body"
            return True
        raise RuntimeError("Unreachable state in `KimiK3ReasoningParser._step`")

    def _feed(self, text: str) -> ReasoningParserResult:
        self._buffer += text
        content: list = []
        reasoning: list = []
        while self._step(content, reasoning):
            pass
        return ReasoningParserResult(content="".join(content),
                                     reasoning_content="".join(reasoning))

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
        if not remaining or self._state == "done":
            return ReasoningParserResult()
        if self._state in ("body", "tools_pass"):
            if self._channel == "think":
                return ReasoningParserResult(reasoning_content=remaining)
            if self._channel == "response":
                return ReasoningParserResult(content=remaining)
        # Header fragments and structural-gap text are dropped.
        return ReasoningParserResult()
