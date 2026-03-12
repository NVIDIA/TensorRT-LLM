from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Type


@dataclass
class ContentBlock:
    """Typed content block for interleaved thinking responses."""
    type: Literal["thinking", "text"]
    content: str


@dataclass
class ReasoningParserResult:
    content: str = ""
    reasoning_content: str = ""
    # For interleaved thinking: the ordered list of content blocks.
    content_blocks: List[ContentBlock] = field(default_factory=list)
    # For streaming: indicates what type of content the current delta belongs to.
    current_block_type: Optional[Literal["thinking", "text"]] = None


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

    def parse_to_blocks(self, text: str) -> List[ContentBlock]:
        """Parse text into a list of interleaved thinking/text content blocks.

        Default implementation delegates to parse() and converts to blocks.
        Subclasses may override for more accurate multi-block parsing.
        """
        result = self.parse(text)
        blocks: List[ContentBlock] = []
        if result.reasoning_content:
            blocks.append(
                ContentBlock(type="thinking", content=result.reasoning_content))
        if result.content:
            blocks.append(ContentBlock(type="text", content=result.content))
        return blocks


@register_reasoning_parser("deepseek-r1", reasoning_at_start=True)
@register_reasoning_parser("qwen3")
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

    def parse_to_blocks(self, text: str) -> List[ContentBlock]:
        """Parse text into interleaved thinking/text content blocks.

        Handles multiple <think>...</think> blocks, producing alternating
        thinking and text blocks while preserving ordering.
        """
        blocks: List[ContentBlock] = []
        remaining = text
        in_reasoning = self.reasoning_at_start

        if in_reasoning:
            # Text starts in reasoning mode (e.g., deepseek-r1)
            end_idx = remaining.find(self.reasoning_end)
            if end_idx == -1:
                # Entire text is reasoning
                if remaining:
                    blocks.append(
                        ContentBlock(type="thinking", content=remaining))
                return blocks
            thinking_text = remaining[:end_idx]
            if thinking_text:
                blocks.append(
                    ContentBlock(type="thinking", content=thinking_text))
            remaining = remaining[end_idx + len(self.reasoning_end):]
        else:
            # For models like qwen3 where reasoning doesn't start immediately,
            # find the first <think> tag
            start_idx = remaining.find(self.reasoning_start)
            if start_idx == -1:
                # No thinking blocks at all
                if remaining:
                    blocks.append(ContentBlock(type="text", content=remaining))
                return blocks
            # Text before first <think> is dropped (consistent with parse())
            remaining = remaining[start_idx + len(self.reasoning_start):]
            end_idx = remaining.find(self.reasoning_end)
            if end_idx == -1:
                if remaining:
                    blocks.append(
                        ContentBlock(type="thinking", content=remaining))
                return blocks
            thinking_text = remaining[:end_idx]
            if thinking_text:
                blocks.append(
                    ContentBlock(type="thinking", content=thinking_text))
            remaining = remaining[end_idx + len(self.reasoning_end):]

        # Process remaining text for interleaved blocks
        while remaining:
            start_idx = remaining.find(self.reasoning_start)
            if start_idx == -1:
                # No more thinking blocks, rest is text
                if remaining:
                    blocks.append(ContentBlock(type="text", content=remaining))
                break

            # Text before the next <think> tag
            text_before = remaining[:start_idx]
            if text_before:
                blocks.append(ContentBlock(type="text", content=text_before))

            remaining = remaining[start_idx + len(self.reasoning_start):]
            end_idx = remaining.find(self.reasoning_end)
            if end_idx == -1:
                # Unclosed thinking block
                if remaining:
                    blocks.append(
                        ContentBlock(type="thinking", content=remaining))
                break

            thinking_text = remaining[:end_idx]
            if thinking_text:
                blocks.append(
                    ContentBlock(type="thinking", content=thinking_text))
            remaining = remaining[end_idx + len(self.reasoning_end):]

        return blocks

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
                return ReasoningParserResult(content=delta_text,
                                             current_block_type="text")
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
                    reasoning_content=reasoning_content,
                    current_block_type="thinking")
            reasoning_content = delta_text[:end_idx]
            content = delta_text[end_idx + len(self.reasoning_end):]
            self.in_reasoning = False
            self._buffer = ""
            # Determine block type: if there's content after </think>, it's text;
            # if only reasoning, it's thinking
            block_type = "text" if content else "thinking"
            return ReasoningParserResult(content=content,
                                         reasoning_content=reasoning_content,
                                         current_block_type=block_type)
        raise RuntimeError(
            "Unreachable code reached in `DeepSeekR1Parser.parse_delta`")


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

    def _maybe_swap_content(
            self, result: ReasoningParserResult) -> ReasoningParserResult:
        """When force_nonempty_content is set and content is empty, move
        reasoning_content into content so the response always has content."""
        if self._force_nonempty_content and not result.content and result.reasoning_content:
            return ReasoningParserResult(content=result.reasoning_content,
                                         reasoning_content="")
        return result

    def parse(self, text: str) -> ReasoningParserResult:
        return self._maybe_swap_content(super().parse(text))
