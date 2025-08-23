# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ReasoningParserResult:

    def __init__(self,
                 in_reasoning: bool,
                 content: Optional[str] = None,
                 reasoning_content: Optional[str] = None):
        self.in_reasoning = in_reasoning
        self.content = content
        self.reasoning_content = reasoning_content


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

    def __init__(self):
        self.reasoning_end = "</think>"
        self.in_reasoning = True

    def _create_reasoning_end_result(self, content: str,
                                     reasoning_content: str):
        if len(content) == 0:
            reasoning_parser_result = ReasoningParserResult(
                True, reasoning_content=reasoning_content)
        elif len(reasoning_content) == 0:
            reasoning_parser_result = ReasoningParserResult(False,
                                                            content=content)
        else:
            reasoning_parser_result = ReasoningParserResult(
                False, content=content, reasoning_content=reasoning_content)
        return reasoning_parser_result

    def parse(self, text: str) -> ReasoningParserResult:
        if self.reasoning_end not in text:
            return ReasoningParserResult(True, reasoning_content=text)

        splits = text.split(self.reasoning_end, maxsplit=1)
        reasoning_content = splits[0]
        content = splits[1]

        reasoning_parser_result = self._create_reasoning_end_result(
            content, reasoning_content)
        return reasoning_parser_result

    def parse_delta(self, delta_text: str) -> ReasoningParserResult:
        if self.in_reasoning and self.reasoning_end in delta_text:
            end_idx = delta_text.find(self.reasoning_end)
            reasoning_content = delta_text[:end_idx]
            content = delta_text[end_idx + len(self.reasoning_end):]
            reasoning_parser_result = self._create_reasoning_end_result(
                content, reasoning_content)
            self.in_reasoning = False
            return reasoning_parser_result

        if self.in_reasoning:
            return ReasoningParserResult(self.in_reasoning,
                                         reasoning_content=delta_text)

        # not self.in_reasoning:
        return ReasoningParserResult(self.in_reasoning, content=delta_text)


class GptOssParser(BaseReasoningParser):
    """
    Reasoning parser for GPT-OSS model using Harmony response format.
    The GPT-OSS model uses channels: 'analysis' for reasoning and 'final' for the response.
    The pattern appears to be: analysis content followed by 'assistantfinal' then the actual response.
    """

    def __init__(self):
        # Based on the actual response format observed
        self.reasoning_end = "assistantfinal"
        self.in_reasoning = True
        # State for handling multi-token transitions in streaming
        self.transition_buffer = ""
        self.looking_for_final = False

    def _create_reasoning_end_result(self, content: str,
                                     reasoning_content: str):
        if len(content) == 0:
            reasoning_parser_result = ReasoningParserResult(
                True, reasoning_content=reasoning_content)
        elif len(reasoning_content) == 0:
            reasoning_parser_result = ReasoningParserResult(False,
                                                            content=content)
        else:
            reasoning_parser_result = ReasoningParserResult(
                False, content=content, reasoning_content=reasoning_content)
        return reasoning_parser_result

    def parse(self, text: str) -> ReasoningParserResult:
        # Look for the pattern where analysis ends and final begins
        if self.reasoning_end not in text:
            # If we find "analysis" at the start, treat as reasoning
            if text.startswith("analysis"):
                return ReasoningParserResult(True, reasoning_content=text)
            return ReasoningParserResult(True, reasoning_content=text)

        splits = text.split(self.reasoning_end, 1)
        reasoning_content = splits[0]
        content = splits[1] if len(splits) > 1 else ""

        # Clean up reasoning content - remove "analysis" prefix if present
        if reasoning_content.startswith("analysis"):
            reasoning_content = reasoning_content[8:]  # Remove "analysis"

        reasoning_parser_result = self._create_reasoning_end_result(
            content, reasoning_content)
        return reasoning_parser_result

    def parse_delta(self, delta_text: str) -> ReasoningParserResult:
        # Add to transition buffer for multi-token transition handling
        self.transition_buffer += delta_text

        # Simple approach: if we see "final" token and we're in reasoning mode,
        # switch to content mode for the next tokens
        if self.in_reasoning and delta_text == "final":
            self.in_reasoning = False
            # Return empty to signal transition
            return ReasoningParserResult(False, content="")

        # If we just saw "assistant" token, don't output it
        if self.in_reasoning and delta_text == "assistant":
            self.looking_for_final = True
            return ReasoningParserResult(True, reasoning_content="")

        if self.in_reasoning:
            # Normal reasoning content
            clean_delta = delta_text
            if clean_delta.startswith("analysis"):
                clean_delta = clean_delta[8:]

            return ReasoningParserResult(True, reasoning_content=clean_delta)

        # not self.in_reasoning - we're in final content mode
        return ReasoningParserResult(False, content=delta_text)


class ReasoningParserFactory:
    parsers: Dict[str, BaseReasoningParser] = {
        "deepseek-r1": DeepSeekR1Parser,
        "gpt-oss": GptOssParser,
    }

    @staticmethod
    def create_reasoning_parser(reasoning_parser: str) -> BaseReasoningParser:
        if reasoning_parser not in ReasoningParserFactory.parsers:
            raise ValueError(f"Invalid reasoning_parser: {reasoning_parser}")
        reasoning_parser_class = ReasoningParserFactory.parsers.get(
            reasoning_parser.lower())
        return reasoning_parser_class()
