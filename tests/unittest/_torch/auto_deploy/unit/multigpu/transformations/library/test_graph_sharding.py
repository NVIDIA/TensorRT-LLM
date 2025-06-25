"""Tests for basic graph sharding."""

from functools import partial
from typing import Type

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _dist_test_utils import get_device_counts
from _graph_test_helpers import run_test

import tensorrt_llm._torch.auto_deploy.distributed.common as dist_common
from tensorrt_llm._torch.auto_deploy.transformations.library import column_row_shard
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


class GQA_Block(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        hidden_size: int,
        num_key_value_heads: int,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.is_gqa = num_key_value_heads < num_attention_heads
        assert self.hidden_size == self.num_attention_heads * self.head_dim

        # key, query, value, out projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_key_value_heads,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_key_value_heads,
            bias=False,
        )

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape

        q = self.q_proj(x).view(b, s, -1, self.head_dim)
        k = self.k_proj(x).view(b, s, -1, self.head_dim)
        v = self.v_proj(x).view(b, s, -1, self.head_dim)

        y = torch.ops.auto_deploy.torch_attention_bsnd_grouped_sdpa(q, k, v, is_causal=True)
        y = y.contiguous().view(b, s, -1)

        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear1 = nn.Linear(in_features, 4 * in_features, bias=bias)
        self.linear2 = nn.Linear(4 * in_features, out_features, bias=bias)

    def forward(self, x):
        y = F.relu(self.linear1(x))
        return self.linear2(y)


def _run_job(
    model_cls: nn.Module,
    dist_op_expected: str,
    bias: bool,
    rank: int,
    world_size: int,
) -> None:
    # init model and input
    batch_size = 4
    sequence_len = 8
    num_features = 32

    # GQA specific parameters
    num_heads = 4
    num_key_value_heads = 1

    if model_cls == GQA_Block:
        model = model_cls(
            num_attention_heads=num_heads,
            hidden_size=num_features,
            num_key_value_heads=num_key_value_heads,
        ).to(device="cuda", dtype=torch.float16)
    else:
        model = model_cls(num_features, num_features, bias=bias).to(
            device="cuda", dtype=torch.float16
        )
    x = torch.randn(batch_size, sequence_len, num_features, device="cuda", dtype=torch.float16)

    if model_cls == GQA_Block:
        head_dim = num_features // num_heads
        min_local_size = head_dim
    else:
        min_local_size = 1

    def _get_expected_num_params(num_p_og: int) -> int:
        num_update = 0
        if bias and dist_op_expected == "torch_dist_all_reduce":
            num_p_og -= num_features
            num_update = num_features * (rank == world_size - 1)

        if min_local_size > 1:
            # it means we are in the GQA. W_kv are partially replicated, we need to count
            # the number of parameters manually.
            W_q_local_size = num_features * num_features // world_size
            W_o_local_size = W_q_local_size
            W_k_local_size = num_features * head_dim * max(num_key_value_heads // world_size, 1)
            W_v_local_size = W_k_local_size
            num_params = W_q_local_size + W_k_local_size + W_v_local_size + W_o_local_size
        else:
            num_params = num_p_og // world_size + num_update
        return num_params

    def verify_local_weight_sizes(gm) -> bool:
        """Verify that all weight tensors have first dimension >= min_local_size after sharding."""
        for name, param in gm.named_parameters():
            # Only check parameters that have at least 1 dimension and are weight matrices
            if param.dim() >= 1 and "weight" in name:
                if param.shape[0] < min_local_size:
                    print(
                        f"Weight {name} has shape {param.shape}, dim {param.shape[0]} < min_local_size {min_local_size}"
                    )
                    return False
        return True

    # now run the test
    op_expected = getattr(torch.ops.auto_deploy, dist_op_expected)

    transform_func = partial(column_row_shard, rank=rank, world_size=world_size)

    def combined_graph_check(gm) -> bool:
        # Check for expected distributed operations
        has_expected_dist_ops = any(is_op(n, op_expected) for n in gm.graph.nodes) == (
            world_size > 1
        )
        # Check weight size constraints
        weight_sizes_valid = verify_local_weight_sizes(gm)
        return has_expected_dist_ops and weight_sizes_valid

    run_test(
        model,
        x,
        transform=transform_func,
        check_transformed_graph=combined_graph_check,
        _get_expected_num_params=_get_expected_num_params,
    )


@pytest.mark.parametrize("device_count", get_device_counts())
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    "model_cls, dist_op_expected",
    (
        (MLP, "torch_dist_all_reduce"),
        (nn.Linear, "torch_dist_all_gather"),
        (GQA_Block, "torch_dist_all_reduce"),
    ),
)
def test_sharding(model_cls: Type[nn.Module], dist_op_expected: str, bias: bool, device_count: int):
    dist_common.spawn_multiprocess_job(
        job=partial(_run_job, model_cls, dist_op_expected, bias),
        size=device_count,
    )
