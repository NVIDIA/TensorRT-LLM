"""Tests for basic graph sharding."""

from functools import partial

import pytest
import torch
import torch.nn as nn
from _dist_test_utils import get_device_counts
from _graph_test_helpers import run_test

import tensorrt_llm._torch.auto_deploy.distributed.common as dist_common
from tensorrt_llm._torch.auto_deploy.transformations.library.sharding import dp_bmm_shard
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
    rank: int,
    world_size: int,
    num_experts_multiplier: int,
) -> None:
    # init model and input
    batch_size = 4
    num_features = 10
    num_experts = num_experts_multiplier * world_size
    model = BMM(num_experts, num_features).to(device="cuda", dtype=torch.float16)
    x = torch.randn(batch_size * num_experts, num_features, device="cuda", dtype=torch.float16)

    def _get_expected_num_params(num_p_og: int) -> int:
        num_params = num_p_og // world_size
        return num_params

    # now run the test
    op_expected = getattr(torch.ops.auto_deploy, "torch_dist_all_gather")
    run_test(
        model,
        x,
        transform=partial(dp_bmm_shard, rank=rank, world_size=world_size),
        check_transformed_graph=lambda gm: any(is_op(n, op_expected) for n in gm.graph.nodes)
        == (world_size > 1),
        _get_expected_num_params=_get_expected_num_params,
    )


@pytest.mark.parametrize("num_experts_multiplier", [1, 2])
@pytest.mark.parametrize("device_count", get_device_counts())
def test_sharding(device_count: int, num_experts_multiplier: int):
    dist_common.spawn_multiprocess_job(
        job=partial(_run_job, num_experts_multiplier=num_experts_multiplier),
        size=device_count,
    )
