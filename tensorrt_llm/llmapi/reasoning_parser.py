import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Type

from tensorrt_llm import logger


@dataclass
class ReasoningParserResult:
    content: str = ""
    reasoning_content: str = ""


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
    def keys(cls):
        return cls._parsers.keys()


class BaseReasoningParser(ABC):

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


@register_reasoning_parser("deepseek-r1", reasoning_at_start=True)
@register_reasoning_parser("qwen3")
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


MODEL_TYPE_TO_REASONING_PARSER: dict[str, str] = {
    "qwen3": "qwen3",
    "qwen3_moe": "qwen3",
    "qwen3_5": "qwen3",
    "qwen3_5_moe": "qwen3",
    "qwen3_next": "qwen3",
    "deepseek_v3": "deepseek-r1",
    "deepseek_v32": "deepseek-r1",
    "nemotron_h": "nano-v3",
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
        """Wraps the parent parse_delta to track accumulated reasoning when
        force_nonempty_content is set. When the closing tag is found
        (in_reasoning transitions from True to False), the accumulation
        is cleared to free memory."""
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
        return self._maybe_swap_content(super().parse(text))


@register_reasoning_parser("kimi_k2")
class KimiK2ReasoningParser(DeepSeekR1Parser):
    """Reasoning parser for Kimi-K2-Thinking model.

    Extends DeepSeekR1Parser to support interleaved thinking where reasoning
    content may be implicitly ended by a tool call section. The model uses
    ``<think>...</think>`` tokens and may also start tool calls via
    ``<|tool_calls_section_begin|>`` without an explicit ``</think>`` tag.

    Supported patterns:

    * ``<think>reasoning</think>content`` – standard thinking
    * ``<think>reasoning<|tool_calls_section_begin|>...`` – interleaved
      thinking (reasoning interrupted by tool call)
    * ``content`` (no ``<think>``) – no reasoning

    Adapted from:
    * vLLM ``vllm/reasoning/kimi_k2_reasoning_parser.py``
    * sglang ``sglang/srt/parser/reasoning_parser.py``
    """

    def __init__(self,
                 *,
                 reasoning_at_start: bool = False,
                 chat_template_kwargs: Optional[dict[str, Any]] = None) -> None:
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
            # No end marker found – everything is reasoning.
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
