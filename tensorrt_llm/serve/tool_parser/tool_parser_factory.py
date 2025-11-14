from typing import Type

from .base_tool_parser import BaseToolParser
from .qwen3_coder_parser import Qwen3CoderToolParser
from .qwen3_tool_parser import Qwen3ToolParser


class ToolParserFactory:
    parsers: dict[str, Type[BaseToolParser]] = {
        "qwen3": Qwen3ToolParser,
        "qwen3_coder": Qwen3CoderToolParser,
    }

    @staticmethod
    def create_tool_parser(tool_parser: str) -> BaseToolParser:
        try:
            tool_parser_class = ToolParserFactory.parsers[tool_parser.lower()]
            return tool_parser_class()
        except KeyError as e:
            raise ValueError(
                f"Invalid tool_parser: {tool_parser}\n"
                f"Supported parsers: {list(ToolParserFactory.parsers.keys())}"
            ) from e
