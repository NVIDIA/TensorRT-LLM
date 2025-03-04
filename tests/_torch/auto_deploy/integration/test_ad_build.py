"""Testing build_and_run_ad end2end."""

from typing import Dict, Optional

import pytest
from _dist_test_utils import param_with_device_count
from _model_test_utils import _hf_model_dir_or_hub_id
from _torch_test_utils import fp8_compatible
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
            marks_extra=[
                pytest.mark.skip(reason="https://nvbugspro.nvidia.com/bug/5095416"),
            ],
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
        param_with_device_count(
            2,
            {
                "model": _hf_model_dir_or_hub_id(
                    f"{llm_models_root()}/llama-3.1-model/Llama-3.1-8B-Instruct-FP8",
                    "nvidia/Llama-3.1-8B-Instruct-FP8",
                ),
                "benchmark": True,
            },
            marks_extra=[
                pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support"),
            ],
        ),
        # full NVSmall (Llama-3.1-Nemotron-51B) with torch-opt backend + simple runtime
        param_with_device_count(
            4,
            {
                "model": _hf_model_dir_or_hub_id(
                    f"{llm_models_root()}/nemotron-nas/Llama-3_1-Nemotron-51B-Instruct",
                    "nvidia/Llama-3_1-Nemotron-51B-Instruct",
                )
            },
            marks_extra=[
                pytest.mark.skip(reason="https://nvbugspro.nvidia.com/bug/5121522"),
            ],
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
            marks_extra=[
                pytest.mark.skip(reason="Pending support for custom MoE Op sharding"),
            ],
        ),
    ],
)
def test_build_ad(world_size: Optional[int], config: Dict):
    simple_config = SimpleConfig(**config)
    simple_config.world_size = world_size
    main(simple_config)
