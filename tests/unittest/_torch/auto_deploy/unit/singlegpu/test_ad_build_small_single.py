"""Testing build_and_run_ad end2end."""

from typing import Dict

import pytest
from _model_test_utils import get_small_model_config
from build_and_run_ad import main
from simple_config import SimpleConfig


@pytest.mark.parametrize(
    "config",
    [
        get_small_model_config(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            attn_backend="FlashInfer",
            compile_backend="torch-opt",
        ),
        get_small_model_config(
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            attn_backend="TritonWithFlattenedInputs",
            compile_backend="torch-simple",
        ),
        get_small_model_config(
            "microsoft/Phi-3-mini-4k-instruct",
            attn_backend="TritonWithFlattenedInputs",
            compile_backend="torch-simple",
        ),
        get_small_model_config(
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            attn_backend="FlashInfer",
            compile_backend="torch-opt",
        ),
        get_small_model_config(
            "deepseek-ai/DeepSeek-V3",
            attn_backend="TritonWithFlattenedInputs",
            compile_backend="torch-simple",
        ),
    ],
)
def test_build_ad(config: Dict):
    config["runtime"] = "demollm"  # Default runtime set to demollm
    config["world_size"] = 0  # Default world_size set to 0
    simple_config = SimpleConfig(**config)
    print(f"Simple Config: {simple_config}")
    main(simple_config)
