"""Testing build_and_run_ad end2end."""

import pytest
from _model_test_utils import get_small_model_config, get_transforms_config
from build_and_run_ad import ExperimentConfig, main

from tensorrt_llm._torch.auto_deploy.llm_args import AutoDeployConfig, LlmArgs, _ParallelConfig
from tensorrt_llm._torch.auto_deploy.shim.ad_executor import ADEngine


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
    expected_llm_args: LlmArgs = LlmArgs(**expected_ad_config.to_llm_kwargs())
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


@pytest.mark.parametrize("mode", ["graph", "transformers"])
@pytest.mark.parametrize(
    "modle_hub_id, transform_args",
    [
        (
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            {
                "attn_backend": "flashinfer",
                "compile_backend": "torch-opt",
                "free_mem_ratio": 0.0001,
            },
        ),
        (
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            {
                "attn_backend": "triton",
                "compile_backend": "torch-simple",
            },
        ),
        (
            "Qwen/Qwen3-30B-A3B",
            {
                "attn_backend": "triton",
                "compile_backend": "torch-simple",
            },
        ),
        (
            "microsoft/Phi-3-mini-4k-instruct",
            {
                "attn_backend": "triton",
                "compile_backend": "torch-simple",
            },
        ),
        (
            "microsoft/Phi-3-mini-4k-instruct",
            {
                "attn_backend": "torch",
                "compile_backend": "torch-simple",
            },
        ),
        (
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            {
                "attn_backend": "flashinfer",
                "compile_backend": "torch-opt",
            },
        ),
        (
            "deepseek-ai/DeepSeek-V3",
            {
                "attn_backend": "triton",
                "compile_backend": "torch-simple",
            },
        ),
        (
            "Qwen/Qwen2.5-3B-Instruct",
            {
                "attn_backend": "triton",
                "compile_backend": "torch-compile",
            },
        ),
        (
            "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            {
                "attn_backend": "flashinfer",
                "compile_backend": "torch-cudagraph",
            },
        ),
        (
            "nvidia/NVIDIA-Nemotron-Nano-12B-v2",
            {
                "attn_backend": "flashinfer",
                "compile_backend": "torch-simple",
            },
        ),
    ],
)
def test_build_ad(modle_hub_id: str, transform_args: dict, mode: str):
    transforms_config = get_transforms_config(mode, **transform_args)
    experiment_config = get_small_model_config(
        modle_hub_id, mode=mode, transforms=transforms_config
    )
    if (
        "DeepSeek-V3" in modle_hub_id
        or "Phi-3-mini-4k-instruct" in modle_hub_id
        or "NVIDIA-Nemotron-Nano-12B-v2" in modle_hub_id
        and mode == "transformers"
    ):
        pytest.skip(f"{experiment_config['args']['model']} is not supported in transformers mode")

    experiment_config["args"]["runtime"] = "demollm"  # Default runtime set to demollm
    experiment_config["args"]["world_size"] = 0  # Default world_size set to 0

    print(f"Experiment Config: {experiment_config}")
    experiment_config = ExperimentConfig(**experiment_config)
    original_build_from_config = ADEngine.build_from_config

    @classmethod
    def check_and_original_build(cls, ad_config):
        _check_ad_config(experiment_config, ad_config)
        return original_build_from_config.__func__(cls, ad_config)

    # Temporarily replace the build_from_config classmethod
    ADEngine.build_from_config = check_and_original_build

    try:
        main(experiment_config)
    finally:
        # Restore original build_from_config
        ADEngine.build_from_config = original_build_from_config
