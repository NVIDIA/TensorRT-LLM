"""Testing build_and_run_ad end2end."""

import pytest
from _model_test_utils import get_small_model_config
from build_and_run_ad import ExperimentConfig, main

from tensorrt_llm._torch.auto_deploy.llm_args import LlmArgs, _ParallelConfig
from tensorrt_llm._torch.auto_deploy.shim.ad_executor import ADEngine


def _check_ad_config(experiment_config: ExperimentConfig, llm_args: LlmArgs):
    # Verify that llm_args was captured
    assert llm_args is not None, "llm_args should have been captured"

    # Check that llm_args is an instance of LlmArgs.
    assert isinstance(llm_args, LlmArgs), f"Expected LlmArgs, got {type(llm_args)}"

    # check that llm_args and experiment_config have the same args
    expected_ad_config: LlmArgs = experiment_config.args
    expected_llm_args: LlmArgs = LlmArgs(**expected_ad_config.model_dump())
    assert expected_llm_args == llm_args, f"Expected llm args {expected_llm_args}, got {llm_args}"

    # check expected parallel config
    world_size = expected_ad_config.world_size
    expected_parallel_config = _ParallelConfig(
        tp_size=world_size, gpus_per_node=expected_llm_args.gpus_per_node
    )
    assert llm_args._parallel_config == expected_parallel_config, (
        f"Expected parallel_config {expected_parallel_config}, got {llm_args._parallel_config}"
    )

    # backend should always be "_autodeploy"
    assert llm_args.backend == "_autodeploy", (
        f"Expected backend '_autodeploy', got {llm_args.backend}"
    )


@pytest.mark.parametrize(
    "model_hub_id, llm_extra_args",
    [
        (
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            {
                "kv_cache_config": {
                    "free_gpu_memory_fraction": 0.0001,
                },
                "transforms": {
                    "insert_cached_attention": {"backend": "flashinfer"},
                    # TODO: https://github.com/NVIDIA/TensorRT-LLM/issues/9878
                    # "compile_model": {"backend": "torch-opt"},
                    "compile_model": {"backend": "torch-cudagraph"},
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
        (
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            {
                "transforms": {
                    "insert_cached_attention": {"backend": "triton"},
                    "compile_model": {"backend": "torch-simple"},
                },
            },
        ),
        (
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            {
                "transforms": {
                    "transformers_replace_cached_attn": {"backend": "triton"},
                },
                "mode": "transformers",
            },
        ),
        (
            "Qwen/Qwen3-30B-A3B",
            {
                "transforms": {
                    "insert_cached_attention": {"backend": "triton"},
                    "compile_model": {"backend": "torch-simple"},
                },
            },
        ),
        (
            "Qwen/Qwen3-30B-A3B",
            {
                "transforms": {
                    "transformers_replace_cached_attn": {"backend": "triton"},
                },
                "mode": "transformers",
            },
        ),
        (
            "microsoft/Phi-3-mini-4k-instruct",
            {
                "transforms": {
                    "insert_cached_attention": {"backend": "triton"},
                    "compile_model": {"backend": "torch-simple"},
                },
            },
        ),
        (
            "microsoft/Phi-3-mini-4k-instruct",
            {
                "transforms": {
                    "insert_cached_attention": {"backend": "torch"},
                    "compile_model": {"backend": "torch-simple"},
                },
            },
        ),
        (
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            {
                "transforms": {
                    "insert_cached_attention": {"backend": "flashinfer"},
                    "compile_model": {"backend": "torch-opt"},
                },
            },
        ),
        (
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            {
                "transforms": {
                    "transformers_replace_cached_attn": {"backend": "flashinfer"},
                },
                "mode": "transformers",
            },
        ),
        (
            "deepseek-ai/DeepSeek-V3",
            {
                "transforms": {
                    "insert_cached_attention": {"backend": "triton"},
                    "compile_model": {"backend": "torch-simple"},
                },
            },
        ),
        (
            "Qwen/Qwen2.5-3B-Instruct",
            {
                "transforms": {
                    "insert_cached_attention": {"backend": "triton"},
                    "compile_model": {"backend": "torch-compile"},
                },
            },
        ),
        (
            "Qwen/Qwen2.5-3B-Instruct",
            {
                "transforms": {
                    "transformers_replace_cached_attn": {"backend": "triton"},
                },
                "mode": "transformers",
            },
        ),
        (
            "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            {
                "transforms": {
                    "insert_cached_attention": {"backend": "flashinfer"},
                    "compile_model": {"backend": "torch-cudagraph"},
                },
            },
        ),
        (
            "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            {
                "transforms": {
                    "transformers_replace_cached_attn": {"backend": "flashinfer"},
                },
                "mode": "transformers",
            },
        ),
        (
            "nvidia/NVIDIA-Nemotron-Nano-12B-v2",
            {
                "transforms": {
                    "insert_cached_attention": {"backend": "flashinfer"},
                    "compile_model": {"backend": "torch-simple"},
                    "insert_cached_ssm_attention": {"backend": "triton_ssm"},
                },
            },
        ),
        (
            "nvidia/Nemotron-Nano-3-30B-A3.5B-dev-1024",
            {
                "transforms": {
                    "multi_stream_moe": {"stage": "compile", "enabled": True},
                    "insert_cached_ssm_attention": {"backend": "triton_ssm"},
                    # TODO: https://github.com/NVIDIA/TensorRT-LLM/issues/9878
                    "compile_model": {"backend": "torch-cudagraph"},
                },
            },
        ),
    ],
)
def test_build_ad(model_hub_id: str, llm_extra_args: dict):
    experiment_config = get_small_model_config(model_hub_id, **llm_extra_args)
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
