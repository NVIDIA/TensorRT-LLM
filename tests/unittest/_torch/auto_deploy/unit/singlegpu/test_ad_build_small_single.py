"""Testing build_and_run_ad end2end."""

from typing import Dict

import pytest
from _model_test_utils import get_small_model_config_pytest_param
from build_and_run_ad import ExperimentConfig, main

from tensorrt_llm._torch.auto_deploy.llm_args import AutoDeployConfig, LlmArgs, _ParallelConfig
from tensorrt_llm._torch.auto_deploy.transformations.transform import InferenceOptimizer


def _check_ad_config(experiment_config: ExperimentConfig, llm_args: LlmArgs):
    # Verify that llm_args was captured
    assert llm_args is not None, "llm_args should have been captured"

    # Check that llm_args is an instance of LlmArgs and also an instance of AutoDeployConfig
    assert isinstance(llm_args, LlmArgs), f"Expected LlmArgs, got {type(llm_args)}"
    assert isinstance(llm_args, AutoDeployConfig), (
        f"Expected AutoDeployConfig, got {type(llm_args)}"
    )

    # check that llm_args and experiment_config have the same args
    expected_ad_config: AutoDeployConfig = experiment_config.args
    expected_llm_args: LlmArgs = expected_ad_config.to_llm_args()
    assert expected_llm_args == llm_args, f"Expected llm args {expected_llm_args}, got {llm_args}"

    # check expected parallel config
    world_size = expected_ad_config.world_size
    expected_parallel_config = _ParallelConfig(
        auto_parallel=True, gpus_per_node=expected_llm_args.gpus_per_node
    )
    expected_parallel_config.world_size = world_size
    assert llm_args._parallel_config == expected_parallel_config, (
        f"Expected parallel_config {expected_parallel_config}, got {llm_args._parallel_config}"
    )

    # backend should always be "_autodeploy"
    assert llm_args.backend == "_autodeploy", (
        f"Expected backend '_autodeploy', got {llm_args.backend}"
    )


@pytest.mark.parametrize(
    "experiment_config",
    [
        get_small_model_config_pytest_param(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            attn_backend="flashinfer",
            compile_backend="torch-opt",
        ),
        get_small_model_config_pytest_param(
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            attn_backend="triton",
            compile_backend="torch-simple",
        ),
        get_small_model_config_pytest_param(
            "Qwen/Qwen3-30B-A3B",
            attn_backend="triton",
            compile_backend="torch-simple",
        ),
        get_small_model_config_pytest_param(
            "microsoft/Phi-3-mini-4k-instruct",
            attn_backend="triton",
            compile_backend="torch-simple",
        ),
        get_small_model_config_pytest_param(
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            attn_backend="flashinfer",
            compile_backend="torch-simple",
        ),
        get_small_model_config_pytest_param(
            "deepseek-ai/DeepSeek-V3",
            attn_backend="triton",
            compile_backend="torch-simple",
        ),
        get_small_model_config_pytest_param(
            "microsoft/Phi-3-mini-4k-instruct",
            attn_backend="torch",
            compile_backend="torch-simple",
        ),
        get_small_model_config_pytest_param(
            "Qwen/Qwen2.5-3B-Instruct",
            attn_backend="triton",
            compile_backend="torch-compile",
        ),
        get_small_model_config_pytest_param(
            "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            attn_backend="flashinfer",
            compile_backend="torch-simple",
        ),
    ],
)
def test_build_ad(experiment_config: Dict):
    experiment_config["args"]["runtime"] = "demollm"  # Default runtime set to demollm
    experiment_config["args"]["world_size"] = 0  # Default world_size set to 0
    experiment_config = ExperimentConfig(**experiment_config)
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
