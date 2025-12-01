from functools import partial

import pytest
import torch
import torch.distributed as dist
from _dist_test_utils import get_device_counts

from tensorrt_llm._torch.auto_deploy.custom_ops.fused_moe.mxfp4_moe import (
    IS_TRITON_KERNELS_AVAILABLE,
)
from tensorrt_llm._torch.auto_deploy.distributed.common import spawn_multiprocess_job


def _split_range_last_remainder(n: int, world_size: int, rank: int):
    """[lo, hi) split along dim0; last rank gets remainder."""
    base = n // world_size
    lo = base * rank
    hi = n if rank == world_size - 1 else base * (rank + 1)
    return lo, hi


def _run_mxfp4_mlp_ep_dtype_test(num_experts: int, topk: int, rank: int, world_size: int):
    torch.cuda.set_device(rank)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    B, S = 2, 16
    H = 32  # hidden size
    In = 8  # intermediate size
    TWO_I = 2 * In
    KB, BS = 4, 4  # K-tiling such that KB * BS * 2 == H (FP4 packs 2 per byte)
    assert KB * BS * 2 == H

    dtype_x = torch.bfloat16
    dtype_bf16 = torch.bfloat16
    dtype_u8 = torch.uint8

    x = torch.randn((B, S, H), dtype=dtype_x, device="cuda") * 0.5
    router_weight = torch.randn((num_experts, H), device="cuda").to(dtype_bf16) * 0.2
    router_bias = torch.randn((num_experts,), device="cuda").to(dtype_bf16) * 0.1

    # Using 1..15 keeps scales away from 0 while staying small.
    def _rand_scales(shape):
        return torch.randint(1, 16, shape, dtype=dtype_u8, device="cuda")

    gate_up_blocks_full = torch.randint(
        0, 256, (num_experts, TWO_I, KB, BS), dtype=dtype_u8, device="cuda"
    )
    gate_up_scales_full = _rand_scales((num_experts, TWO_I, KB))
    gate_up_bias_full = (0.1 * torch.randn((num_experts, TWO_I), device="cuda")).to(dtype_bf16)
    down_blocks_full = torch.randint(
        0, 256, (num_experts, H, KB, BS), dtype=dtype_u8, device="cuda"
    )
    down_scales_full = _rand_scales((num_experts, H, KB))
    down_bias_full = (0.1 * torch.randn((num_experts, H), device="cuda")).to(dtype_bf16)

    assert (gate_up_scales_full != 0).all() and (down_scales_full != 0).all(), (
        "Zero scales would cause NaNs/inf."
    )

    alpha = 1.0
    limit = 1.0

    ref_out = torch.ops.auto_deploy.triton_mxfp4_moe(
        x,
        router_weight,
        router_bias,
        topk,
        gate_up_blocks_full,
        gate_up_bias_full,
        gate_up_scales_full,
        alpha,
        limit,
        down_blocks_full,
        down_bias_full,
        down_scales_full,
    )

    lo, hi = _split_range_last_remainder(num_experts, world_size, rank)

    gate_up_blocks = gate_up_blocks_full[lo:hi]
    gate_up_bias = gate_up_bias_full[lo:hi]
    gate_up_scales = gate_up_scales_full[lo:hi]
    down_blocks = down_blocks_full[lo:hi]
    down_bias = down_bias_full[lo:hi]
    down_scales = down_scales_full[lo:hi]

    assert (gate_up_scales != 0).all() and (down_scales != 0).all(), "Zero scales on shard."

    part_out = torch.ops.auto_deploy.triton_mxfp4_moe_ep(
        x,
        router_weight,
        router_bias,
        topk,
        gate_up_blocks,
        gate_up_bias,
        gate_up_scales,
        alpha,
        limit,
        down_blocks,
        down_bias,
        down_scales,
        world_size,
        rank,
    )

    dist.all_reduce(part_out, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    torch.testing.assert_close(part_out, ref_out, rtol=5e-2, atol=5e-2, equal_nan=True)


@pytest.mark.skipif(
    not IS_TRITON_KERNELS_AVAILABLE,
    reason="triton_kernels unavailable",
)
@pytest.mark.parametrize("num_experts", [6, 8])
@pytest.mark.parametrize("topk", [4])  # must be <= num_experts
@pytest.mark.parametrize("device_count", get_device_counts())
def test_mxfp4_mlp_ep_dtypes(device_count, num_experts, topk):
    if topk > num_experts:
        pytest.skip("topk must be <= num_experts")
    spawn_multiprocess_job(
        job=partial(_run_mxfp4_mlp_ep_dtype_test, num_experts, topk), size=device_count
    )
