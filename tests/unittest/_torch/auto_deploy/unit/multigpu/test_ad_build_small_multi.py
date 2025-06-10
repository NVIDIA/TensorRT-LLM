"""Testing build_and_run_ad end2end."""

from typing import Dict

import pytest
from _model_test_utils import get_small_model_config
from build_and_run_ad import main
from simple_config import SimpleConfig


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize(
    "config",
    [
        get_small_model_config(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            attn_backend="FlashInfer",
            compile_backend="torch-opt",
        ),
    ],
)
def test_build_ad(world_size: int, config: Dict):
    if world_size > 1:
        pytest.skip("https://nvbugspro.nvidia.com/bug/5331013")
    config["world_size"] = world_size
    config["runtime"] = "trtllm"  # Default runtime set to trtllm
    simple_config = SimpleConfig(**config)
    print(f"Simple Config: {simple_config}")
    main(simple_config)
