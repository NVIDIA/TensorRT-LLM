import json
from pathlib import Path
from typing import Optional, Type

from .base_tool_parser import BaseToolParser
from .deepseekv3_parser import DeepSeekV3Parser
from .deepseekv31_parser import DeepSeekV31Parser
from .deepseekv32_parser import DeepSeekV32Parser
from .gemma4_parser import Gemma4ToolParser
from .glm4_parser import Glm4ToolParser
from .glm47_parser import Glm47ToolParser
from .kimi_k2_tool_parser import KimiK2ToolParser
from .minimax_m2_parser import MiniMaxM2ToolParser
from .qwen3_coder_parser import Qwen3CoderToolParser
from .qwen3_tool_parser import Qwen3ToolParser

MODEL_TYPE_TO_TOOL_PARSER: dict[str, str] = {
    "qwen2": "qwen3",
    "qwen3": "qwen3",
    "qwen3_moe": "qwen3",
    "qwen3_5": "qwen3",
    "qwen3_5_moe": "qwen3",
    "qwen3_next": "qwen3",
    "deepseek_v3": "deepseek_v3",
    "deepseek_v32": "deepseek_v32",
    "kimi_k2": "kimi_k2",
    "kimi_k25": "kimi_k2",
    "glm4": "glm4",
    "glm4_moe": "glm47",
    "glm4_moe_lite": "glm47",
    "glm_moe_dsa": "glm47",
    "gemma4": "gemma4",
    "gemma4_text": "gemma4",
}


def resolve_auto_tool_parser(model: str) -> Optional[str]:
    """Resolve 'auto' tool parser by reading the model's HF config."""
    config_path = Path(model) / "config.json"
    if not config_path.exists():
        return None

    with open(config_path) as f:
        config = json.load(f)

    model_type = config.get("model_type", "")
    return MODEL_TYPE_TO_TOOL_PARSER.get(model_type)


class ToolParserFactory:
    parsers: dict[str, Type[BaseToolParser]] = {
        "qwen3": Qwen3ToolParser,
        "qwen3_coder": Qwen3CoderToolParser,
        "kimi_k2": KimiK2ToolParser,
        "deepseek_v3": DeepSeekV3Parser,
        "deepseek_v31": DeepSeekV31Parser,
        "deepseek_v32": DeepSeekV32Parser,
        "gemma4": Gemma4ToolParser,
        "glm4": Glm4ToolParser,
        "glm47": Glm47ToolParser,
        "minimax_m2": MiniMaxM2ToolParser,
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
