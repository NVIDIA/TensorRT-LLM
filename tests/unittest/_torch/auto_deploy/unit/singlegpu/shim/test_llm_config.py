from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm._torch.auto_deploy.shim.demollm import DemoLLM
from tensorrt_llm._torch.auto_deploy.transformations.transform import InferenceOptimizer
from tensorrt_llm.llmapi.llm import LLM
from tensorrt_llm.llmapi.llm_args import _AutoDeployLlmArgs

# ================================
# _AutoDeployLlmArgs Direct Tests
# ================================


def test_custom_values():
    """Test that _AutoDeployLlmArgs correctly accepts custom values."""
    custom_kwargs = {
        "model": "test-model",
        "model_factory": "AutoModelForImageTextToText",
        "model_kwargs": {"custom_param": True},
        "mla_backend": "MultiHeadLatentAttention",
        "skip_loading_weights": True,
        "free_mem_ratio": 0.9,
        "simple_shard_only": True,
        "attn_page_size": 128,
        "attn_backend": "CustomBackend",
        "max_seq_len": 2048,
    }

    args = _AutoDeployLlmArgs(**custom_kwargs)

    assert args.model_factory == "AutoModelForImageTextToText"
    assert args.model_kwargs == {
        "custom_param": True,
        "max_position_embeddings": args.max_seq_len,
    }
    assert args.skip_loading_weights
    assert args.free_mem_ratio == 0.9
    assert args.simple_shard_only
    assert args.attn_page_size == 128
    assert args.max_seq_len == 2048
    # attn_backend should be overridden if it was 'TRTLLM'
    assert args.attn_backend == "CustomBackend"


def test_free_mem_ratio_validation():
    """Test free_mem_ratio validation."""
    # Valid values
    _AutoDeployLlmArgs(model="test-model", free_mem_ratio=0.0)
    _AutoDeployLlmArgs(model="test-model", free_mem_ratio=1.0)
    _AutoDeployLlmArgs(model="test-model", free_mem_ratio=0.5)

    # Invalid values
    with pytest.raises(ValueError):
        _AutoDeployLlmArgs(model="test-model", free_mem_ratio=-0.1)
    with pytest.raises(ValueError):
        _AutoDeployLlmArgs(model="test-model", free_mem_ratio=1.1)


def test_get_pytorch_backend_config():
    """Test that get_pytorch_backend_config returns self."""
    args = _AutoDeployLlmArgs(model="test-model")
    assert args.get_pytorch_backend_config() is args


# ================================
# Config Flow Tests
# ================================


@pytest.fixture
def test_config_params():
    """Common test configuration parameters."""
    return {
        "model": "test-model",
        "model_factory": "AutoModelForImageTextToText",
        "free_mem_ratio": 0.7,
        "simple_shard_only": True,
        "skip_loading_weights": True,
        "attn_page_size": 17,
        "attn_backend": "CustomBackend",
        "max_seq_len": 19,
        "max_batch_size": 5,
        "tensor_parallel_size": 3,
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
@patch("tensorrt_llm._torch.auto_deploy.shim.demollm.DemoGenerationExecutor")
@patch("tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface.SequenceInfo")
@patch("tensorrt_llm._torch.auto_deploy.shim.demollm.dist_ad.initialize_or_skip")
@patch("tensorrt_llm._torch.auto_deploy.shim.demollm.create_input_processor")
@patch("tensorrt_llm.llmapi.llm.LLM._build_model")
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
        instance = api_class(**config_params)

    # Verify args were created correctly
    assert hasattr(instance, "args")
    assert isinstance(instance.args, _AutoDeployLlmArgs)

    # Common assertions for both APIs
    assert instance.args.model_factory == test_config_params["model_factory"]
    assert instance.args.free_mem_ratio == test_config_params["free_mem_ratio"]
    assert instance.args.simple_shard_only == test_config_params["simple_shard_only"]
    assert instance.args.skip_loading_weights == test_config_params["skip_loading_weights"]
    assert instance.args.attn_page_size == test_config_params["attn_page_size"]
    assert instance.args.max_seq_len == test_config_params["max_seq_len"]
    assert instance.args.max_batch_size == test_config_params["max_batch_size"]

    # Verify executor behavior for DemoLLM
    if expected_executor_call:
        mock_executor.assert_called_once()
        call_kwargs = mock_executor.call_args[1]
        assert call_kwargs["world_size"] == test_config_params["tensor_parallel_size"]
        assert call_kwargs["model"] == test_config_params["model"]
    else:
        # For LLM with _autodeploy backend, executor should not be called directly
        pass


@pytest.mark.parametrize(
    "use_cuda_graph,torch_compile_enabled,expected_backend",
    [
        (True, True, "torch-opt"),
        (True, False, "torch-cudagraph"),
        (False, True, "torch-compile"),
        (False, False, "torch-simple"),
    ],
)
@patch("tensorrt_llm._torch.auto_deploy.models.factory.ModelFactoryRegistry.get")
def test_compile_backend_mapping(
    mock_factory_registry, use_cuda_graph, torch_compile_enabled, expected_backend
):
    """Test that compile backend is correctly determined from config."""
    ad_config = _AutoDeployLlmArgs(
        model="test-model",
        use_cuda_graph=use_cuda_graph,
        torch_compile_enabled=torch_compile_enabled,
    )
    optimizer = InferenceOptimizer(factory=MagicMock(), ad_config=ad_config)
    assert optimizer.compile_backend == expected_backend


# ================================
# Additional Edge Case Tests
# ================================


def test_invalid_model_factory():
    """Test behavior with invalid model factory."""
    # Pydantic validates Literal types at runtime, so this should raise ValidationError
    with pytest.raises(Exception):  # Could be ValidationError or ValueError
        _AutoDeployLlmArgs(model="test-model", model_factory="InvalidFactory")


@pytest.mark.parametrize(
    "attn_backend,expected_attn_page_size",
    [
        ("FlashInfer", 64),  # Default attn_page_size
        ("TritonWithFlattenedInputs", 1024),  # Should equal max_seq_len
        ("CustomBackend", 64),  # Default attn_page_size
    ],
)
def test_attention_backend_page_size_logic(attn_backend, expected_attn_page_size):
    """Test attn_page_size logic for different attention backends."""
    args = _AutoDeployLlmArgs(model="test-model", attn_backend=attn_backend, max_seq_len=1024)
    assert args.attn_page_size == expected_attn_page_size
