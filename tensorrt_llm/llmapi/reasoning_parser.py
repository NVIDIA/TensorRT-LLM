from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type


@dataclass
class ReasoningParserResult:
    content: str = ""
    reasoning_content: str = ""


class BaseReasoningParser(ABC):

    @abstractmethod
    def parse(self, text: str) -> ReasoningParserResult:
        raise NotImplementedError

    @abstractmethod
    def parse_delta(self, delta_text: str) -> ReasoningParserResult:
        raise NotImplementedError


class DeepSeekR1Parser(BaseReasoningParser):
    """
    Reasoning parser for DeepSeek-R1. Reasoning format: <think>(.*)</think>.
    Since the latest official tokenizer_config.json initially adds "<think>\\n" at the end of the prompt
    (https://huggingface.co/deepseek-ai/DeepSeek-R1/blob/main/tokenizer_config.json),
    treat all the text before the </think> tag as `reasoning_content` and the text after as `content`.
    """

    def __init__(self, reasoning_at_start: bool = False) -> None:
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


class ReasoningParserFactory:
    parsers: dict[str, Type[BaseReasoningParser]] = {
        "deepseek-r1": DeepSeekR1Parser,
        "qwen3": DeepSeekR1Parser,
    }

    @staticmethod
    def create_reasoning_parser(reasoning_parser: str) -> BaseReasoningParser:
        try:
            reasoning_parser_class = ReasoningParserFactory.parsers[
                reasoning_parser.lower()]
            if reasoning_parser == "deepseek-r1":
                return reasoning_parser_class(reasoning_at_start=True)
            return reasoning_parser_class()
        except KeyError as e:
            raise ValueError(
                f"Invalid reasoning parser: {reasoning_parser}\n"
                f"Supported parsers: {list(ReasoningParserFactory.parsers.keys())}"
            ) from e
