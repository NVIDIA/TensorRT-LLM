"""Testing build_and_run_ad end2end."""

from typing import Dict

import pytest
from _model_test_utils import get_small_model_config
from build_and_run_ad import main
from simple_config import SimpleConfig

from tensorrt_llm._torch.auto_deploy.models import ModelFactoryRegistry
from tensorrt_llm._torch.auto_deploy.transformations.transform import InferenceOptimizer
from tensorrt_llm.llmapi.llm_args import _AutoDeployLlmArgs


def _check_ad_config(simple_config: SimpleConfig, ad_config: _AutoDeployLlmArgs):
    # Verify that ad_config was captured
    assert ad_config is not None, "ad_config should have been captured"

    # Check that ad_config is an instance of _AutoDeployLlmArgs
    assert isinstance(ad_config, _AutoDeployLlmArgs), (
        f"Expected _AutoDeployLlmArgs, got {type(ad_config)}"
    )

    # Fields that map directly from simple_config to ad_config
    direct_mapping_fields = {
        "max_batch_size",
        "attn_backend",
        "mla_backend",
        "skip_loading_weights",
        "free_mem_ratio",
        "simple_shard_only",
        "attn_page_size",
        "model_factory",
        "model_kwargs",
    }

    # Check direct mappings
    for field_name in direct_mapping_fields:
        if hasattr(simple_config, field_name):
            config_value = getattr(simple_config, field_name)
            ad_config_value = getattr(ad_config, field_name)
            assert ad_config_value == config_value, (
                f"Field {field_name}: expected {config_value}, got {ad_config_value}"
            )

    # for model we need to the snapshot check with the factory
    factory = ModelFactoryRegistry.get(simple_config.model_factory)(
        model=simple_config.model,
        model_kwargs=simple_config.model_kwargs,
        skip_loading_weights=True,
    )
    assert ad_config.model == factory.model, (
        f"Expected model {factory.model}, got {ad_config.model}"
    )

    # world_size -> tensor_parallel_size
    assert ad_config.tensor_parallel_size == simple_config.world_size, (
        f"Expected tensor_parallel_size {simple_config.world_size}, got {ad_config.tensor_parallel_size}"
    )

    # compile_backend -> use_cuda_graph
    expected_cuda_graph = simple_config.compile_backend in ["torch-opt", "torch-cudagraph"]
    assert ad_config.use_cuda_graph == expected_cuda_graph, (
        f"Expected use_cuda_graph {expected_cuda_graph} for {simple_config.compile_backend}, "
        f"got {ad_config.use_cuda_graph}"
    )

    # compile_backend -> torch_compile_config
    expected_torch_compile = simple_config.compile_backend in ["torch-opt", "torch-compile"]
    assert bool(ad_config.torch_compile_config) == expected_torch_compile, (
        f"Expected torch_compile_config to be {expected_torch_compile} for "
        f"{simple_config.compile_backend}, got {ad_config.torch_compile_config}"
    )

    # backend should always be "_autodeploy"
    assert ad_config.backend == "_autodeploy", (
        f"Expected backend '_autodeploy', got {ad_config.backend}"
    )


@pytest.mark.parametrize(
    "config",
    [
        get_small_model_config(
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            attn_backend="FlashInfer",
            compile_backend="torch-opt",
        ),
        get_small_model_config(
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            attn_backend="TritonWithFlattenedInputs",
            compile_backend="torch-simple",
        ),
        get_small_model_config(
            "microsoft/Phi-3-mini-4k-instruct",
            attn_backend="TritonWithFlattenedInputs",
            compile_backend="torch-simple",
        ),
        get_small_model_config(
            "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            attn_backend="FlashInfer",
            compile_backend="torch-opt",
        ),
        get_small_model_config(
            "deepseek-ai/DeepSeek-V3",
            attn_backend="TritonWithFlattenedInputs",
            compile_backend="torch-simple",
        ),
    ],
)
def test_build_ad(config: Dict):
    config["runtime"] = "demollm"  # Default runtime set to demollm
    config["world_size"] = 0  # Default world_size set to 0
    simple_config = SimpleConfig(**config)
    print(f"Simple Config: {simple_config}")
    original_init = InferenceOptimizer.__init__

    def check_and_original_init(self, factory, *, ad_config, **kwargs):
        _check_ad_config(simple_config, ad_config)
        return original_init(self, factory, ad_config=ad_config, **kwargs)

    # Temporarily replace the __init__ method
    InferenceOptimizer.__init__ = check_and_original_init

    try:
        main(simple_config)
    finally:
        # Restore original __init__
        InferenceOptimizer.__init__ = original_init
