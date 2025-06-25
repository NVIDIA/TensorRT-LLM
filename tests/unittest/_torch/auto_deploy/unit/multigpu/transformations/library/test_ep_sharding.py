"""Tests for EP sharding transformation."""

from functools import partial

import pytest
import torch
from _dist_test_utils import get_device_counts
from _graph_test_helpers import run_test
from _model_test_utils import MoEOpModel

import tensorrt_llm._torch.auto_deploy.distributed.common as dist_common
from tensorrt_llm._torch.auto_deploy.transformations.library import ep_shard
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


def _run_ep_shard_job(num_experts: int, rank: int, world_size: int) -> None:
    device = "cuda"
    hidden_size = 32
    intermediate_size = 16
    model = MoEOpModel(
        hidden_size=hidden_size, num_experts=num_experts, intermediate_size=intermediate_size
    ).to(device=device, dtype=torch.bfloat16)
    x = model.get_input(device=device, dtype=torch.bfloat16)

    def _get_expected_num_params(rank: int, world_size: int, num_p_og: int) -> int:
        if world_size <= 1:
            return num_p_og
        # the gate's weight and bias node
        n_gate = num_experts * (hidden_size + 1)
        num_experts_per_rank = num_experts // world_size
        if rank == world_size - 1:
            num_experts_per_rank += num_experts % world_size
        expected_expert = num_experts_per_rank * hidden_size * intermediate_size * 3
        return n_gate + expected_expert

    op_expected = torch.ops.auto_deploy.torch_dist_all_reduce

    run_test(
        model,
        x,
        transform=partial(ep_shard, rank=rank, world_size=world_size),
        check_transformed_graph=lambda gm: any(is_op(n, op_expected) for n in gm.graph.nodes)
        == (world_size > 1),
        _get_expected_num_params=partial(_get_expected_num_params, rank, world_size),
        test_load_hook=False,
    )


@pytest.mark.parametrize("device_count", get_device_counts())
@pytest.mark.parametrize("num_experts", [3, 8])
def test_ep_shard(device_count: int, num_experts: int):
    dist_common.spawn_multiprocess_job(
        job=partial(_run_ep_shard_job, num_experts),
        size=device_count,
    )
