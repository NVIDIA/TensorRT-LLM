from unittest.mock import MagicMock, patch

import pydantic
import pytest

from tensorrt_llm._torch.auto_deploy import LLM, DemoLLM, LlmArgs


def test_custom_values():
    """Test that AutoDeploy LlmArgs correctly accepts custom values."""
    custom_kwargs = {
        "model": "test-model",
        "model_factory": "AutoModelForImageTextToText",
        "model_kwargs": {"custom_param": True},
        "skip_loading_weights": True,
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
        },
    }

    args = LlmArgs(**custom_kwargs)

    assert args.model_factory == "AutoModelForImageTextToText"
    assert args.model_kwargs == {
        "custom_param": True,
    }
    assert args.skip_loading_weights
    assert args.transforms["detect_sharding"]["simple_shard_only"]
    assert args.max_seq_len == 2048
    # backend should be overridden if it was 'TRTLLM'
    assert args.transforms["insert_cached_attention"]["backend"] == "flashinfer"


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
        instance.args.transforms["detect_sharding"]["simple_shard_only"]
        == test_config_params["transforms"]["detect_sharding"]["simple_shard_only"]
    )
    assert instance.args.skip_loading_weights == test_config_params["skip_loading_weights"]
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


# ================================
# CUDA Graph Batch Sizes Tests
# ================================


class TestCudaGraphBatchSizesHeuristic:
    """Test that cuda_graph_batch_sizes heuristic respects max_batch_size."""

    def test_small_max_batch_size_caps_heuristic(self):
        """Test that heuristic batch sizes are capped at small max_batch_size.

        When max_batch_size is small (e.g., 4), the heuristic should NOT include
        batch sizes like 17, 33, 49, 65, 81, 97, 113 which exceed max_batch_size.
        """
        args = LlmArgs(
            model="test-model",
            max_batch_size=4,
        )

        # All batch sizes should be <= max_batch_size
        assert all(bs <= 4 for bs in args.cuda_graph_batch_sizes), (
            f"Expected all batch sizes <= 4, got {args.cuda_graph_batch_sizes}"
        )
        # Should include 1 and max_batch_size
        assert 1 in args.cuda_graph_batch_sizes
        assert 4 in args.cuda_graph_batch_sizes
        # Should NOT include heuristic values that exceed max_batch_size
        assert 17 not in args.cuda_graph_batch_sizes
        assert 113 not in args.cuda_graph_batch_sizes

    def test_medium_max_batch_size_caps_heuristic(self):
        """Test heuristic with medium max_batch_size (e.g., 64)."""
        args = LlmArgs(
            model="test-model",
            max_batch_size=64,
        )

        # All batch sizes should be <= max_batch_size
        assert all(bs <= 64 for bs in args.cuda_graph_batch_sizes), (
            f"Expected all batch sizes <= 64, got {args.cuda_graph_batch_sizes}"
        )
        # Should include some heuristic values up to 64
        assert 1 in args.cuda_graph_batch_sizes
        assert 17 in args.cuda_graph_batch_sizes
        assert 33 in args.cuda_graph_batch_sizes
        assert 49 in args.cuda_graph_batch_sizes
        assert 64 in args.cuda_graph_batch_sizes
        # Should NOT include values > 64
        assert 65 not in args.cuda_graph_batch_sizes
        assert 81 not in args.cuda_graph_batch_sizes

    def test_large_max_batch_size_includes_all_heuristic_values(self):
        """Test heuristic with large max_batch_size (e.g., 256)."""
        args = LlmArgs(
            model="test-model",
            max_batch_size=256,
        )

        # All batch sizes should be <= max_batch_size
        assert all(bs <= 256 for bs in args.cuda_graph_batch_sizes), (
            f"Expected all batch sizes <= 256, got {args.cuda_graph_batch_sizes}"
        )
        # Should include heuristic values from range(1, 129, 16)
        for bs in [1, 17, 33, 49, 65, 81, 97, 113]:
            assert bs in args.cuda_graph_batch_sizes, f"Expected {bs} in batch sizes"
        # Should include 128 from range(128, max_batch_size+1, 128)
        assert 128 in args.cuda_graph_batch_sizes
        assert 256 in args.cuda_graph_batch_sizes

    def test_explicit_cuda_graph_batch_sizes_filtered(self):
        """Test that explicitly provided batch sizes are filtered to max_batch_size."""
        args = LlmArgs(
            model="test-model",
            max_batch_size=16,
            cuda_graph_batch_sizes=[1, 4, 8, 16, 32, 64, 128],
        )

        # Should only include values <= max_batch_size
        assert all(bs <= 16 for bs in args.cuda_graph_batch_sizes), (
            f"Expected all batch sizes <= 16, got {args.cuda_graph_batch_sizes}"
        )
        # Values <= 16 should be present
        assert 1 in args.cuda_graph_batch_sizes
        assert 4 in args.cuda_graph_batch_sizes
        assert 8 in args.cuda_graph_batch_sizes
        assert 16 in args.cuda_graph_batch_sizes
        # Values > 16 should be filtered out
        assert 32 not in args.cuda_graph_batch_sizes
        assert 64 not in args.cuda_graph_batch_sizes
        assert 128 not in args.cuda_graph_batch_sizes

    def test_batch_sizes_sorted_descending(self):
        """Test that cuda_graph_batch_sizes are sorted in descending order."""
        args = LlmArgs(
            model="test-model",
            max_batch_size=64,
        )

        # Should be sorted in descending order
        assert args.cuda_graph_batch_sizes == sorted(args.cuda_graph_batch_sizes, reverse=True), (
            f"Expected descending order, got {args.cuda_graph_batch_sizes}"
        )


class TestSequenceInfoExampleBatchSize:
    """Test that SequenceInfo generates proper example batch sizes for export."""

    def test_example_batch_size_at_least_2_when_max_batch_size_1(self):
        """Test that example batch size is at least 2 even when max_batch_size=1.

        This is critical because torch.export specializes dimensions when the
        example input has a dimension value of 1, breaking dynamic batching.
        """
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import SequenceInfo

        seq_info = SequenceInfo(
            max_batch_size=1,
            max_seq_len=128,
            max_num_tokens=128,
            tokens_per_block=64,
        )

        # Set example sequence (this is what's used during export)
        seq_info.set_example_sequence()

        # The example batch size should be at least 2 to prevent torch.export
        # from specializing the batch dimension
        assert len(seq_info.named_args["input_ids"]) >= 2, (
            f"Example batch size should be >= 2 for export, got {len(seq_info.named_args['input_ids'])}"
        )

    def test_example_batch_size_normal_max_batch_size(self):
        """Test example batch size with normal max_batch_size."""
        from tensorrt_llm._torch.auto_deploy.custom_ops.attention_interface import SequenceInfo

        seq_info = SequenceInfo(
            max_batch_size=32,
            max_seq_len=128,
            max_num_tokens=128,
            tokens_per_block=64,
        )

        seq_info.set_example_sequence()

        # With larger max_batch_size, example should still be 2
        assert len(seq_info.named_args["input_ids"]) == 2, (
            f"Expected example batch size of 2, got {len(seq_info.named_args['input_ids'])}"
        )
