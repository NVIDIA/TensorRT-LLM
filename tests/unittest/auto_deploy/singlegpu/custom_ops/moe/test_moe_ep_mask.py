# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for auto_deploy::moe_ep_mask.

The Triton kernel replaces the aten chain (sub, floordiv, eq/ge, cast, mul) used
by the MoE EP-sharding transform. Correctness is verified against that chain.
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401


def _reference_moe_ep_mask(
    selected_experts: torch.Tensor,
    top_k_weights: torch.Tensor,
    ep_rank: int,
    ep_size: int,
    experts_per_rank: int,
):
    """Matches the aten-op chain in sharding.py (AllReduce paradigm)."""
    lower = ep_rank * experts_per_rank
    local_indices = selected_experts - lower
    group = selected_experts // experts_per_rank
    if ep_rank == ep_size - 1:
        rank_mask = group >= ep_rank
    else:
        rank_mask = group == ep_rank
    masked_weights = top_k_weights * rank_mask.to(top_k_weights.dtype)
    return local_indices, masked_weights


@pytest.mark.parametrize("weights_dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("indices_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "ep_rank,ep_size,num_experts",
    [
        (0, 2, 256),  # even split, first rank (uses ==)
        (1, 2, 256),  # even split, last rank (uses >=)
        (0, 4, 256),  # 4-way split, first rank
        (3, 4, 256),  # 4-way split, last rank
        (1, 4, 256),  # 4-way split, middle rank
    ],
)
@pytest.mark.parametrize("M,top_k", [(1, 8), (20, 8), (256, 8), (1024, 8)])
def test_moe_ep_mask_matches_reference(
    M, top_k, ep_rank, ep_size, num_experts, indices_dtype, weights_dtype
):
    device = "cuda"
    experts_per_rank = num_experts // ep_size
    torch.manual_seed(42 + ep_rank * 100 + M)

    indices = torch.randint(0, num_experts, (M, top_k), dtype=indices_dtype, device=device)
    weights = torch.randn(M, top_k, dtype=weights_dtype, device=device)

    ref_idx, ref_w = _reference_moe_ep_mask(indices, weights, ep_rank, ep_size, experts_per_rank)
    out_idx, out_w = torch.ops.auto_deploy.moe_ep_mask(
        indices, weights, ep_rank, ep_size, experts_per_rank
    )

    assert out_idx.dtype == indices.dtype
    assert out_w.dtype == weights.dtype
    assert out_idx.shape == indices.shape
    assert out_w.shape == weights.shape

    torch.testing.assert_close(out_idx, ref_idx)
    torch.testing.assert_close(out_w, ref_w)


def test_moe_ep_mask_3d_shape():
    """Confirm leading-dim flatten works for [B, S, top_k] inputs."""
    device = "cuda"
    ep_rank, ep_size, experts_per_rank = 0, 2, 128
    num_experts = ep_size * experts_per_rank

    indices = torch.randint(0, num_experts, (4, 32, 8), dtype=torch.int32, device=device)
    weights = torch.randn(4, 32, 8, dtype=torch.bfloat16, device=device)

    ref_idx, ref_w = _reference_moe_ep_mask(indices, weights, ep_rank, ep_size, experts_per_rank)
    out_idx, out_w = torch.ops.auto_deploy.moe_ep_mask(
        indices, weights, ep_rank, ep_size, experts_per_rank
    )

    torch.testing.assert_close(out_idx, ref_idx)
    torch.testing.assert_close(out_w, ref_w)
