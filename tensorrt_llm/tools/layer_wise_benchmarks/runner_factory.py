from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config

from .deepseekv3_runner import DeepSeekV3Runner
from .qwen3_next_runner import Qwen3NextRunner


def get_runner_cls(pretrained_model_name_or_path: str) -> type:
    pretrained_config = load_pretrained_config(pretrained_model_name_or_path)
    return {
        "deepseek_v3": DeepSeekV3Runner,
        "deepseek_v32": DeepSeekV3Runner,
        "qwen3_next": Qwen3NextRunner,
    }[pretrained_config.model_type]
