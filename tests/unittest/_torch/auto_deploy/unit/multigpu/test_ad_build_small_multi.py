"""Testing build_and_run_ad end2end."""

import pytest
from _model_test_utils import get_small_model_config, get_transforms_config
from build_and_run_ad import ExperimentConfig, main


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("mode", ["graph", "transformers"])
@pytest.mark.parametrize(
    "modle_hub_id, attn_backend, compile_backend",
    [
        (
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "flashinfer",
            "torch-opt",
        ),
    ],
)
def test_build_ad(
    world_size: int, modle_hub_id: str, attn_backend: str, compile_backend: str, mode: str
):
    transforms_config = get_transforms_config(mode, attn_backend, compile_backend)
    experiment_config = get_small_model_config(
        modle_hub_id, mode=mode, transforms=transforms_config
    )

    experiment_config["args"]["world_size"] = world_size
    experiment_config["args"]["runtime"] = "trtllm"  # Default runtime set to trtllm

    experiment_config = ExperimentConfig(**experiment_config)
    print(f"Experiment Config: {experiment_config}")
    main(experiment_config)
