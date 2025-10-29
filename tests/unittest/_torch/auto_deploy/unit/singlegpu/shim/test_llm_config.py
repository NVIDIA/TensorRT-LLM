from unittest.mock import MagicMock, patch

import pydantic
import pytest

from tensorrt_llm._torch.auto_deploy import LLM, DemoLLM, LlmArgs
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer


def test_custom_values():
    """Test that AutoDeploy LlmArgs correctly accepts custom values."""
    custom_kwargs = {
        "model": "test-model",
        "model_factory": "AutoModelForImageTextToText",
        "model_kwargs": {"custom_param": True},
        "skip_loading_weights": True,
        "attn_page_size": 128,
        "max_seq_len": 2048,
        "transforms": {
            "detect_sharding": {
                "stage": "sharding",
                "simple_shard_only": True,
            },
            "insert_cached_attention": {
                "stage": "cache_init",
                "backend": "flashinfer",
            },
            "resize_kv_cache": {
                "stage": "cache_init",
                "free_mem_ratio": 0.9,
            },
        },
    }

    args = LlmArgs(**custom_kwargs)

    assert args.model_factory == "AutoModelForImageTextToText"
    assert args.model_kwargs == {
        "custom_param": True,
    }
    assert args.skip_loading_weights
    assert args.transforms["resize_kv_cache"]["free_mem_ratio"] == 0.9
    assert args.transforms["detect_sharding"]["simple_shard_only"]
    assert args.attn_page_size == 128
    assert args.max_seq_len == 2048
    # backend should be overridden if it was 'TRTLLM'
    assert args.transforms["insert_cached_attention"]["backend"] == "flashinfer"


def test_free_mem_ratio_validation():
    """Test free_mem_ratio validation."""

    def get_transform_config(free_mem_ratio):
        return {"resize_kv_cache": {"stage": "cache_init", "free_mem_ratio": free_mem_ratio}}

    # Valid values
    InferenceOptimizer(None, get_transform_config(0.0))
    InferenceOptimizer(None, get_transform_config(1.0))
    InferenceOptimizer(None, get_transform_config(0.5))

    # Invalid values
    with pytest.raises(ValueError):
        InferenceOptimizer(None, get_transform_config(-0.1))
    with pytest.raises(ValueError):
        InferenceOptimizer(None, get_transform_config(1.1))


def test_get_pytorch_backend_config():
    """Test that get_pytorch_backend_config returns self."""
    args = LlmArgs(model="test-model")
    assert args.get_pytorch_backend_config() == args


# ================================
# Config Flow Tests
# ================================


@pytest.fixture
def test_config_params():
    """Common test configuration parameters."""
    return {
        "model": "test-model",
        "model_factory": "AutoModelForImageTextToText",
        "skip_loading_weights": True,
        "attn_page_size": 17,
        "max_seq_len": 19,
        "max_batch_size": 5,
        "world_size": 3,
        "transforms": {
            "detect_sharding": {
                "stage": "sharding",
                "simple_shard_only": True,
            },
            "insert_cached_attention": {
                "stage": "cache_init",
                "backend": "flashinfer",
            },
            "resize_kv_cache": {
                "stage": "cache_init",
                "free_mem_ratio": 0.7,
            },
        },
    }


@pytest.mark.parametrize(
    "api_class,backend,extra_kwargs,expected_executor_call",
    [
        (DemoLLM, None, {}, True),  # DemoLLM doesn't use backend param, should call executor
        (
            LLM,
            "_autodeploy",
            {"backend": "_autodeploy"},
            False,
        ),  # LLM with _autodeploy backend, no executor call
    ],
)
@patch("tensorrt_llm._torch.auto_deploy.llm.DemoGenerationExecutor")
@patch("tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface.SequenceInfo")
@patch("tensorrt_llm._torch.auto_deploy.shim.demollm.dist_ad.initialize_or_skip")
@patch("tensorrt_llm._torch.auto_deploy.llm.LLM._create_input_processor")
@patch("tensorrt_llm._torch.auto_deploy.llm.LLM._build_model")
def test_config_flow(
    mock_build_model,
    mock_input_processor,
    mock_dist_init,
    mock_seq_info,
    mock_executor,
    api_class,
    backend,
    extra_kwargs,
    expected_executor_call,
    test_config_params,
):
    """Test that config flows correctly through both DemoLLM and LLM initialization."""
    # Mock the executor and its methods for DemoLLM
    mock_executor_instance = MagicMock()
    mock_executor.return_value = mock_executor_instance

    # Mock sequence info for DemoLLM
    mock_seq_info_instance = MagicMock()
    mock_seq_info.return_value = mock_seq_info_instance

    # Merge extra kwargs for the specific API
    config_params = {**test_config_params, **extra_kwargs}

    # Create instance with appropriate mocking
    with patch.object(api_class, "_try_load_tokenizer", return_value=MagicMock()):
        with patch.object(api_class, "_prefetch_model", return_value=MagicMock()):
            with patch.object(api_class, "_build_model", return_value=MagicMock()):
                instance = api_class(**config_params)

    # Verify args were created correctly
    assert hasattr(instance, "args")
    assert isinstance(instance.args, LlmArgs)

    # Common assertions for both APIs
    assert instance.args.model_factory == test_config_params["model_factory"]
    assert (
        instance.args.transforms["resize_kv_cache"]["free_mem_ratio"]
        == test_config_params["transforms"]["resize_kv_cache"]["free_mem_ratio"]
    )
    assert (
        instance.args.transforms["detect_sharding"]["simple_shard_only"]
        == test_config_params["transforms"]["detect_sharding"]["simple_shard_only"]
    )
    assert instance.args.skip_loading_weights == test_config_params["skip_loading_weights"]
    assert instance.args.attn_page_size == test_config_params["attn_page_size"]
    assert instance.args.max_seq_len == test_config_params["max_seq_len"]
    assert instance.args.max_batch_size == test_config_params["max_batch_size"]

    # Verify executor behavior for DemoLLM
    if expected_executor_call:
        mock_executor.assert_called_once()
        call_kwargs = mock_executor.call_args[1]
        assert call_kwargs["world_size"] == test_config_params["world_size"]
    else:
        # For LLM with _autodeploy backend, executor should not be called directly
        pass


@pytest.mark.parametrize(
    "model_factory",
    [
        "Foo",
        # typo.
        "AutomodelForCausalLMFactory",
    ],
)
def test_non_registered_model_factory(model_factory: str):
    with pytest.raises(
        pydantic.ValidationError, match="does not exist in the model factory registry"
    ):
        LlmArgs(model="test-model", model_factory=model_factory)


@pytest.mark.parametrize(
    "parallel_field,invalid_value",
    [
        ("tensor_parallel_size", 2),
        ("pipeline_parallel_size", 2),
        ("context_parallel_size", 2),
        ("moe_cluster_parallel_size", 2),
        ("moe_tensor_parallel_size", 2),
        ("moe_expert_parallel_size", 2),
        ("enable_attention_dp", True),
        ("cp_config", {"some_key": "some_value"}),
    ],
)
def test_parallel_config_validation(parallel_field, invalid_value):
    """Test that parallel config fields raise ValueError when set to non-default values."""
    kwargs = {
        "model": "test-model",
        parallel_field: invalid_value,
    }

    with pytest.raises(
        ValueError, match="AutoDeploy only supports parallelization via the `world_size` argument."
    ):
        LlmArgs(**kwargs)


@pytest.mark.parametrize(
    "backend,expected_attn_page_size",
    [
        ("flashinfer", 64),  # Default attn_page_size
        ("triton", 1024),  # Should equal max_seq_len
    ],
)
def test_attention_backend_page_size_logic(backend, expected_attn_page_size):
    """Test attn_page_size logic for different attention backends."""
    args = LlmArgs(
        model="test-model",
        max_seq_len=1024,
        transforms={"insert_cached_attention": {"stage": "cache_init", "backend": backend}},
    )
    assert args.attn_page_size == expected_attn_page_size
