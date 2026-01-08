from typing import Type

from .base_tool_parser import BaseToolParser
from .deepseekv3_parser import DeepSeekV3Parser
from .deepseekv31_parser import DeepSeekV31Parser
from .deepseekv32_parser import DeepSeekV32Parser
from .kimi_k2_tool_parser import KimiK2ToolParser
from .qwen3_coder_parser import Qwen3CoderToolParser
from .qwen3_tool_parser import Qwen3ToolParser


class ToolParserFactory:
    parsers: dict[str, Type[BaseToolParser]] = {
        "qwen3": Qwen3ToolParser,
        "qwen3_coder": Qwen3CoderToolParser,
        "kimi_k2": KimiK2ToolParser,
        "deepseek_v3": DeepSeekV3Parser,
        "deepseek_v31": DeepSeekV31Parser,
        "deepseek_v32": DeepSeekV32Parser,
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
