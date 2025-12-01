"""Testing build_and_run_ad end2end."""

import pytest
from _model_test_utils import get_small_model_config
from build_and_run_ad import ExperimentConfig, main


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize(
    "model_hub_id, llm_extra_args",
    [
        (
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            {
                "transforms": {
                    "insert_cached_attention": {"backend": "flashinfer"},
                    "compile_model": {"backend": "torch-opt"},
                },
            },
        ),
        (
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            {
                "transforms": {
                    "transformers_replace_cached_attn": {"backend": "flashinfer"},
                },
                "mode": "transformers",
            },
        ),
    ],
)
def test_build_ad(world_size: int, model_hub_id: str, llm_extra_args: dict):
    experiment_config = get_small_model_config(model_hub_id, **llm_extra_args)

    experiment_config["args"]["world_size"] = world_size
    experiment_config["args"]["runtime"] = "trtllm"  # Default runtime set to trtllm

    experiment_config = ExperimentConfig(**experiment_config)
    print(f"Experiment Config: {experiment_config}")
    main(experiment_config)
