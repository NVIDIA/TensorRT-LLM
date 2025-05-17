"""Testing build_and_run_ad end2end."""

from typing import Dict, Optional

import pytest
from _model_test_utils import _hf_model_dir_or_hub_id
from build_and_run_ad import main
from simple_config import SimpleConfig
from utils.llm_data import llm_models_root


@pytest.mark.parametrize(
    "world_size, config",
    [
        # small llama3.1-8B model with world_size 1 (processes are spawned)
        (
            1,
            {
                "model": _hf_model_dir_or_hub_id(
                    f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct",
                    "meta-llama/Meta-Llama-3.1-8B-Instruct",
                ),
                "runtime": "demollm",
                "attn_backend": "TritonWithFlattenedInputs",
                "compile_backend": "torch-simple",
                "model_kwargs": {"num_hidden_layers": 2},
            },
        ),
        # small llama3.1-8B model with world_size 1 + trtllm runtime
        (
            1,
            {
                "model": _hf_model_dir_or_hub_id(
                    f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct",
                    "meta-llama/Meta-Llama-3.1-8B-Instruct",
                ),
                "runtime": "trtllm",
                "attn_backend": "TritonWithFlattenedInputs",
                "compile_backend": "torch-simple",
                "model_kwargs": {"num_hidden_layers": 2},
            },
        ),
        # small Mixtral-8x7B model with world_size 1
        (
            1,
            {
                "model": _hf_model_dir_or_hub_id(
                    f"{llm_models_root()}/Mixtral-8x7B-Instruct-v0.1",
                    "mistralai/Mixtral-8x7B-Instruct-v0.1",
                ),
                "runtime": "demollm",
                "attn_backend": "TritonWithFlattenedInputs",
                "compile_backend": "torch-simple",
                "model_kwargs": {"num_hidden_layers": 2},
            },
        ),
        # small llama3.1-8B model with world_size 0 (no processes are spawned) + torch-opt
        (
            0,
            {
                "model": _hf_model_dir_or_hub_id(
                    f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct",
                    "meta-llama/Meta-Llama-3.1-8B-Instruct",
                ),
                "runtime": "demollm",
                "attn_backend": "TritonWithFlattenedInputs",
                "compile_backend": "torch-opt",
                "model_kwargs": {"num_hidden_layers": 2},
            },
        ),
    ],
)
def test_build_ad(world_size: Optional[int], config: Dict):
    simple_config = SimpleConfig(**config)
    simple_config.world_size = world_size
    simple_config.skip_loading_weights = True
    simple_config.free_mem_ratio = 0.01  # we don't need the cache and it may cause OOM issues
    print(f"Simple Config: {simple_config}")
    main(simple_config)
