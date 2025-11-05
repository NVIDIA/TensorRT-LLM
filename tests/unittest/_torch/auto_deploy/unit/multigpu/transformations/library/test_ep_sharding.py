"""Tests for EP sharding transformation."""

from functools import partial
from unittest.mock import MagicMock

import pytest
import torch
from _dist_test_utils import get_device_counts
from _graph_test_helpers import run_sharding_pattern_detection_test, run_test_transformed_gm
from _model_test_utils import MoEOpModel

import tensorrt_llm._torch.auto_deploy.distributed.common as dist_common
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.models.hf import AutoModelForCausalLMFactory
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op
from tensorrt_llm._torch.auto_deploy.utils.sharding_utils import (
    EPShardingInfo,
    FP8EPShardingInfo,
    NVFP4EPShardingInfo,
)


def _run_ep_shard_job(num_experts: int, rank: int, world_size: int) -> None:
    device = "cuda"
    hidden_size = 32
    intermediate_size = 16
    model = MoEOpModel(
        hidden_size=hidden_size, num_experts=num_experts, intermediate_size=intermediate_size
    ).to(device=device, dtype=torch.bfloat16)
    x = model.get_input(device=device, dtype=torch.bfloat16)

    if world_size > num_experts:
        print(f"world_size {world_size} > num_experts {num_experts}, skipping test")
        return

    def _get_expected_num_params(rank: int, world_size: int, num_p_og: int) -> int:
        if world_size <= 1:
            return num_p_og
        # the gate's weight and bias node
        # NOTE:gate layer is also distributed using simple_shard during tp_transform
        n_gate = num_experts * (hidden_size + 1)  # // world_size
        num_experts_per_rank = num_experts // world_size
        if rank == world_size - 1:
            num_experts_per_rank += num_experts % world_size
        expected_expert = num_experts_per_rank * hidden_size * intermediate_size * 3
        return n_gate + expected_expert

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        None,
        {
            "detect_sharding": {
                "stage": "sharding",
                "use_sharding_from_factory": False,
                "sharding_dims": ["ep"],
            },
            "sharding_transform_executor": {
                "stage": "sharding",
            },
        },
    )(None, gm)

    op_expected = torch.ops.auto_deploy.torch_dist_all_reduce

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        check_transformed_graph=lambda gm: any(is_op(n, op_expected) for n in gm.graph.nodes)
        == (world_size > 1),
        _get_expected_num_params=partial(_get_expected_num_params, rank, world_size),
        test_load_hook=False,
    )


def _run_pattern_detection_job(num_experts: int, rank: int, world_size: int) -> None:
    device = "cuda"
    hidden_size = 32
    intermediate_size = 16
    model = MoEOpModel(
        hidden_size=hidden_size, num_experts=num_experts, intermediate_size=intermediate_size
    ).to(device=device, dtype=torch.bfloat16)
    x = model.get_input(device=device, dtype=torch.bfloat16)

    # Test pattern detection - create expected transformations for validation
    gm = torch_export_to_gm(model, args=(x,), clone=True)
    expected_transformations = []
    # if world_size == 1, no sharding transformations should be detected
    if world_size > 1:
        for node in gm.graph.nodes:
            if is_op(node, torch.ops.auto_deploy.torch_moe):
                expected_transformations.append(
                    EPShardingInfo(
                        target_node=node.name,
                        rank=rank,
                        world_size=world_size,
                    )
                )
            elif is_op(node, torch.ops.auto_deploy.torch_quant_fp8_moe):
                expected_transformations.append(
                    FP8EPShardingInfo(
                        target_node=node.name,
                        rank=rank,
                        world_size=world_size,
                    )
                )
            elif is_op(node, torch.ops.auto_deploy.torch_quant_nvfp4_moe):
                expected_transformations.append(
                    NVFP4EPShardingInfo(
                        target_node=node.name,
                        rank=rank,
                        world_size=world_size,
                    )
                )

    # get detected transformations
    optimizer = InferenceOptimizer(
        None,
        {
            "detect_sharding": {
                "stage": "sharding",
                "use_sharding_from_factory": False,
            },
        },
    )
    optimizer.shared_config.local_rank = rank
    optimizer.shared_config.world_size = world_size
    _ = optimizer(None, gm)
    detected_transformations = optimizer.shared_config.sharding_config.ep_transforms

    # Run pattern detection test
    run_sharding_pattern_detection_test(detected_transformations, expected_transformations)


@pytest.mark.parametrize("device_count", get_device_counts())
@pytest.mark.parametrize("num_experts", [3, 8])
def test_ep_shard(device_count: int, num_experts: int):
    dist_common.spawn_multiprocess_job(
        job=partial(_run_ep_shard_job, num_experts),
        size=device_count,
    )


@pytest.mark.parametrize("world_size", [1, 8])
@pytest.mark.parametrize("num_experts", [3, 8])
def test_sharding_pattern_detection(world_size: int, num_experts: int):
    """Test pattern detection logic without distributed execution.

    This test verifies only the pattern detection logic with provided world_size.
    No need to run distributed job, can be run on single process.
    """
    _run_pattern_detection_job(
        num_experts=num_experts,
        rank=0,
        world_size=world_size,
    )


@pytest.mark.parametrize(
    "model_type,expected_strategy,should_set",
    [
        ("mixtral", "tp", True),
        ("qwen", None, False),
    ],
    ids=["mixtral_sets_tp", "non_mixtral_no_override"],
)
def test_moe_sharding_strategy(model_type, expected_strategy, should_set):
    """Test that HF factory sets MOE sharding strategy based on model type.

    Parametrized tests verify:
    1. mixtral_sets_tp: Mixtral models set TP strategy in factory config
    2. non_mixtral_no_override: Non-Mixtral models don't override (use system default)
    """
    # Create a mock PretrainedConfig for the model type
    mock_config = MagicMock()
    mock_config.model_type = model_type
    mock_config.hidden_size = 32
    mock_config.num_attention_heads = 2
    mock_config.num_hidden_layers = 4

    # Create factory instance (just need a valid model path)
    factory = AutoModelForCausalLMFactory(model="dummy_model")

    # Call _set_sharding_config with the mocked config
    factory._set_sharding_config(mock_config)

    # Verify the factory's sharding config
    sharding_config = factory.get_sharding_config()

    if should_set:
        # Model should set moe_sharding_strategy in factory config
        assert "moe_sharding_strategy" in sharding_config, (
            f"{model_type} should set moe_sharding_strategy"
        )
        assert sharding_config["moe_sharding_strategy"] == expected_strategy, (
            f"{model_type} should use {expected_strategy} strategy, "
            f"got {sharding_config.get('moe_sharding_strategy')}"
        )
    else:
        # Model should NOT set moe_sharding_strategy (will use system default)
        assert "moe_sharding_strategy" not in sharding_config, (
            f"{model_type} should not override moe_sharding_strategy"
        )
