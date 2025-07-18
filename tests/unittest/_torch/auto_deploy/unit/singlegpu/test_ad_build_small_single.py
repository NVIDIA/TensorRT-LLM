"""Testing build_and_run_ad end2end."""

from typing import Dict

import pytest
from _model_test_utils import get_small_model_config
from build_and_run_ad import ExperimentConfig, main

from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs, _ParallelConfig
from tensorrt_llm._torch.auto_deploy.transformations.transform import InferenceOptimizer


def _check_ad_config(experiment_config: ExperimentConfig, ad_config: LlmArgs):
    # Verify that ad_config was captured
    assert ad_config is not None, "ad_config should have been captured"

    # Check that ad_config is an instance of LlmArgs
    assert isinstance(ad_config, LlmArgs), f"Expected AutoDeploy LlmArgs, got {type(ad_config)}"

    # check that ad_config and experiment_config have the same args
    assert experiment_config.args == ad_config, (
        f"Expected experiment_config.args {experiment_config.args}, got {ad_config}"
    )

    # check expected parallel config
    world_size = experiment_config.args.world_size
    expected_parallel_config = _ParallelConfig(
        auto_parallel=True, gpus_per_node=experiment_config.args.gpus_per_node
    )
    expected_parallel_config.world_size = world_size
    assert ad_config._parallel_config == expected_parallel_config, (
        f"Expected parallel_config {expected_parallel_config}, got {ad_config._parallel_config}"
    )

    # backend should always be "_autodeploy"
    assert ad_config.backend == "_autodeploy", (
        f"Expected backend '_autodeploy', got {ad_config.backend}"
    )


@pytest.mark.parametrize(
    "experiment_config",
    [
        get_small_model_config(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            attn_backend="flashinfer",
            compile_backend="torch-opt",
        ),
        get_small_model_config(
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            attn_backend="triton",
            compile_backend="torch-simple",
        ),
        get_small_model_config(
            "Qwen/Qwen3-30B-A3B",
            attn_backend="triton",
            compile_backend="torch-simple",
        ),
        get_small_model_config(
            "microsoft/Phi-3-mini-4k-instruct",
            attn_backend="triton",
            compile_backend="torch-simple",
        ),
        get_small_model_config(
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            attn_backend="flashinfer",
            compile_backend="torch-opt",
        ),
        get_small_model_config(
            "deepseek-ai/DeepSeek-V3",
            attn_backend="triton",
            compile_backend="torch-simple",
        ),
    ],
)
def test_build_ad(experiment_config: Dict):
    experiment_config["args"]["runtime"] = "demollm"  # Default runtime set to demollm
    experiment_config["args"]["world_size"] = 0  # Default world_size set to 0
    experiment_config = ExperimentConfig(**experiment_config)
    print(f"Experiment Config: {experiment_config}")
    original_init = InferenceOptimizer.__init__

    def check_and_original_init(self, factory, ad_config):
        _check_ad_config(experiment_config, ad_config)
        return original_init(self, factory, ad_config=ad_config)

    # Temporarily replace the __init__ method
    InferenceOptimizer.__init__ = check_and_original_init

    try:
        main(experiment_config)
    finally:
        # Restore original __init__
        InferenceOptimizer.__init__ = original_init
