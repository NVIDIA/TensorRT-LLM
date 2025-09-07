"""Tests for basic graph sharding."""

from functools import partial

import pytest
import torch
import torch.nn as nn
from _dist_test_utils import get_device_counts
from _graph_test_helpers import run_sharding_pattern_detection_test, run_test_transformed_gm

import tensorrt_llm._torch.auto_deploy.distributed.common as dist_common
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.library.sharding import BMMShardingInfo
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


class BMM(nn.Module):
    def __init__(self, num_experts, num_features):
        super().__init__()
        self.num_experts = num_experts
        self.num_features = num_features
        self.gate_up_proj = nn.Parameter(
            torch.randn(self.num_experts, self.num_features, 2 * self.num_features)
        )
        self.down_proj = nn.Parameter(
            torch.randn((self.num_experts, self.num_features, self.num_features))
        )
        self.act_fn = torch.nn.functional.relu

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        This should really not be run on a single machine, as we are reaching compute bound:
        - the inputs are expected to be "sorted" per expert already.
        - the weights are viewed with another dim, to match num_expert, 1, shape * num_tokens, shape

        Args:
            hidden_states (torch.Tensor): (batch_size * token_num, hidden_size)
            selected_experts (torch.Tensor): (batch_size * token_num, top_k)
            routing_weights (torch.Tensor): (batch_size * token_num, top_k)
        Returns:
            torch.Tensor
        """
        hidden_states = hidden_states.view(self.num_experts, -1, self.num_features)
        gate_up = torch.bmm(hidden_states, self.gate_up_proj)
        gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
        next_states = torch.bmm((up * self.act_fn(gate)), self.down_proj)
        next_states = next_states.view(-1, self.num_features)
        return next_states


def _run_job(
    num_experts_multiplier: int,
    rank: int,
    world_size: int,
) -> None:
    # init model and input
    batch_size = 4
    num_features = 10
    num_experts = num_experts_multiplier * world_size
    model = BMM(num_experts, num_features).to(device="cuda", dtype=torch.float16)
    x = torch.randn(batch_size * num_experts, num_features, device="cuda", dtype=torch.float16)
    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        None,
        {
            "detect_sharding": {
                "stage": "sharding",
                "use_sharding_from_factory": False,
                "sharding_dims": ["bmm"],
            },
            "sharding_transform_executor": {
                "stage": "sharding",
            },
        },
    )(None, gm)

    def _get_expected_num_params(num_p_og: int) -> int:
        num_params = num_p_og // world_size
        return num_params

    # now run the test
    op_expected = getattr(torch.ops.auto_deploy, "torch_dist_all_gather")
    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        check_transformed_graph=lambda gm: any(is_op(n, op_expected) for n in gm.graph.nodes)
        == (world_size > 1),
        _get_expected_num_params=_get_expected_num_params,
    )


def _run_pattern_detection_job(
    rank: int,
    world_size: int,
    num_experts_multiplier: int,
) -> None:
    # init model and input
    batch_size = 4
    num_features = 10
    num_experts = num_experts_multiplier * world_size
    start_idx = rank * num_experts_multiplier
    end_idx = start_idx + num_experts_multiplier
    model = BMM(num_experts, num_features).to(device="cuda", dtype=torch.float16)
    x = torch.randn(batch_size * num_experts, num_features, device="cuda", dtype=torch.float16)

    # Test pattern detection - create expected transformations for validation
    gm = torch_export_to_gm(model, args=(x,), clone=True)
    expected_transformations = []
    # if world_size == 1, no sharding transformations should be detected
    if world_size > 1:
        for node in gm.graph.nodes:
            if is_op(node, torch.ops.aten.bmm):
                expected_transformations.append(
                    BMMShardingInfo(
                        target_node=node.name,
                        rank=rank,
                        world_size=world_size,
                        start_idx=start_idx,
                        end_idx=end_idx,
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
    detected_transformations = optimizer.shared_config.sharding_config.bmm_transforms

    # Run pattern detection test
    run_sharding_pattern_detection_test(detected_transformations, expected_transformations)


@pytest.mark.parametrize("num_experts_multiplier", [1, 2])
@pytest.mark.parametrize("device_count", get_device_counts())
def test_sharding(device_count: int, num_experts_multiplier: int):
    dist_common.spawn_multiprocess_job(
        job=partial(_run_job, num_experts_multiplier),
        size=device_count,
    )


@pytest.mark.parametrize("world_size", [1, 8])
@pytest.mark.parametrize("num_experts_multiplier", [1, 2])
def test_sharding_pattern_detection(world_size: int, num_experts_multiplier: int):
    """Test pattern detection logic without distributed execution.

    This test verifies only the pattern detection logic with provided world_size.
    No need to run distributed job, can be run on single process.
    """
    _run_pattern_detection_job(
        num_experts_multiplier=num_experts_multiplier,
        rank=0,
        world_size=world_size,
    )
