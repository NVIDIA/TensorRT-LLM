"""Testing build_and_run_ad end2end.

NOTE (lucaslie): this test is for local testing only. It is not registered to run as part of CI.
"""

from typing import Dict, Optional

import pytest
from _dist_test_utils import param_with_device_count
from _model_test_utils import _hf_model_dir_or_hub_id
from build_and_run_ad import main
from simple_config import SimpleConfig
from utils.llm_data import llm_models_root


@pytest.mark.parametrize(
    "world_size, config",
    [
        # full llama3.1-8B model
        (
            2,
            {
                "model": _hf_model_dir_or_hub_id(
                    f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct",
                    "meta-llama/Meta-Llama-3.1-8B-Instruct",
                ),
                "runtime": "demollm",
                "attn_backend": "TritonWithFlattenedInputs",
                "compile_backend": "torch-simple",
            },
        ),
        # full llama3.1-8B model with demollm runtime + torch-opt
        (
            2,
            {
                "model": _hf_model_dir_or_hub_id(
                    f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct",
                    "meta-llama/Meta-Llama-3.1-8B-Instruct",
                ),
                "runtime": "demollm",
                "attn_backend": "TritonWithFlattenedInputs",
                "compile_backend": "torch-opt",
            },
        ),
        # full llama3.1-8B model with demollm runtime + torch-opt + FlashInfer attn backend
        (
            2,
            {
                "model": _hf_model_dir_or_hub_id(
                    f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct",
                    "meta-llama/Meta-Llama-3.1-8B-Instruct",
                ),
                "runtime": "demollm",
                "attn_backend": "FlashInfer",
                "compile_backend": "torch-opt",
            },
        ),
        # full llama3.1-8B model with trtllm runtime
        param_with_device_count(
            2,
            {
                "model": _hf_model_dir_or_hub_id(
                    f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct",
                    "meta-llama/Meta-Llama-3.1-8B-Instruct",
                ),
                "runtime": "trtllm",
                "attn_backend": "FlashInfer",
                "compile_backend": "torch-opt",
            },
        ),
        # 2-layer llama3.1-8B model on 4 GPUs
        param_with_device_count(
            4,
            {
                "model": _hf_model_dir_or_hub_id(
                    f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct",
                    "meta-llama/Meta-Llama-3.1-8B-Instruct",
                ),
                "benchmark": True,
                "model_kwargs": {"num_hidden_layers": 2},
            },
        ),
        # full llama3.1-8B model in fp8 with torch backend + simplellm runtime
        # TODO: FP8 cache support is lacking right now: https://nvbugspro.nvidia.com/bug/5152021
        param_with_device_count(
            2,
            {
                "model": _hf_model_dir_or_hub_id(
                    f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct-FP8",
                    "nvidia/Llama-3.1-8B-Instruct-FP8",
                ),
                "benchmark": True,
                "attn_backend": "FlashInfer",
            },
        ),
        # full NemotronNAS (Llama-3.1-Nemotron-51B) with torch-opt backend + simple runtime
        param_with_device_count(
            4,
            {
                "model": _hf_model_dir_or_hub_id(
                    f"{llm_models_root()}/nemotron-nas/Llama-3_1-Nemotron-51B-Instruct",
                    "nvidia/Llama-3_1-Nemotron-51B-Instruct",
                )
            },
        ),
        # Mixtral 8x7B with torch-simple backend + simple runtime
        param_with_device_count(
            4,
            {
                "model": _hf_model_dir_or_hub_id(
                    f"{llm_models_root()}/Mixtral-8x7B-Instruct-v0.1",
                    "mistralai/Mixtral-8x7B-Instruct-v0.1",
                ),
                "compile_backend": "torch-simple",
            },
        ),
        # Phi3-mini-4k with torch-opt backend + simple runtime
        param_with_device_count(
            2,
            {
                "model": _hf_model_dir_or_hub_id(
                    f"{llm_models_root()}/Phi-3/Phi-3-mini-4k-instruct",
                    "microsoft/Phi-3-mini-4k-instruct",
                ),
                "compile_backend": "torch-opt",
                "attn_backend": "TritonWithFlattenedInputs",
            },
        ),
        # Llama4 Scout Instruct
        param_with_device_count(
            4,
            {
                "model": _hf_model_dir_or_hub_id(
                    f"{llm_models_root()}/Llama-4-Scout-17B-16E-Instruct",
                    "meta-llama/Llama-4-Scout-17B-16E-Instruct",
                ),
                "model_factory": "AutoModelForImageTextToText",
                "compile_backend": "torch-simple",
                "attn_backend": "FlashInfer",
            },
        ),
    ],
)
def test_build_ad(world_size: Optional[int], config: Dict):
    simple_config = SimpleConfig(**config)
    simple_config.world_size = world_size
    simple_config.free_mem_ratio = 0.01  # we don't need the cache and it may cause OOM issues
    main(simple_config)
