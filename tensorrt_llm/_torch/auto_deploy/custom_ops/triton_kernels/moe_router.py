# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Triton kernel for fused MoE routing: softmax + top-k selection + optional normalization."""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _moe_router_kernel(
    logits_ptr,
    topk_weights_ptr,
    topk_indices_ptr,
    logits_row_stride: tl.constexpr,
    N_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    NORMALIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused softmax + top-k selection kernel for MoE routing.

    Grid: (num_tokens,) -- one program per token row.

    For each token, computes:
    1. Softmax over expert logits (in float32 for numerical stability)
    2. Top-k selection of expert indices and weights
    3. Optional normalization of top-k weights to sum to 1
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < N_EXPERTS

    # Load logits row and upcast to float32
    logits_row_ptr = logits_ptr + row_idx * logits_row_stride
    logits = tl.load(logits_row_ptr + col_offsets, mask=mask, other=-float("inf"))
    logits_f32 = logits.to(tl.float32)

    # Online softmax: max subtraction for numerical stability
    max_val = tl.max(logits_f32, axis=0)
    exp_vals = tl.exp(logits_f32 - max_val)
    sum_exp = tl.sum(exp_vals, axis=0)
    softmax_vals = exp_vals / sum_exp

    # Iterative top-k selection: find the top TOP_K values and their indices
    # We iteratively find the max, record it, and mask it out
    remaining = softmax_vals
    out_weights_base = topk_weights_ptr + row_idx * TOP_K
    out_indices_base = topk_indices_ptr + row_idx * TOP_K

    for k in tl.static_range(TOP_K):
        # Find max value and its index among remaining (unmasked) values
        cur_max = tl.max(remaining, axis=0)
        # Create a mask for elements equal to the max (pick first via cumsum trick)
        is_max = (remaining == cur_max) & mask
        # Use argmax-like approach: pick the smallest index where is_max is true
        # We compute index by finding the minimum col_offset where is_max is true
        max_idx = tl.min(tl.where(is_max, col_offsets, BLOCK_N), axis=0)

        # Store the k-th top value and index (cast index to int64 for output)
        tl.store(out_weights_base + k, cur_max)
        tl.store(out_indices_base + k, max_idx.to(tl.int64))

        # Mask out the selected element for subsequent iterations
        remaining = tl.where(col_offsets == max_idx, 0.0, remaining)

    # Optional normalization: make top-k weights sum to 1
    if NORMALIZE:
        # Reload the stored top-k weights using power-of-2 block size
        topk_offsets = tl.arange(0, BLOCK_K)
        topk_mask = topk_offsets < TOP_K
        topk_vals = tl.load(out_weights_base + topk_offsets, mask=topk_mask, other=0.0)
        topk_sum = tl.sum(topk_vals, axis=0)
        normalized = topk_vals / topk_sum
        tl.store(out_weights_base + topk_offsets, normalized, mask=topk_mask)


def moe_router(
    logits: Tensor,
    top_k: int,
    normalize: bool = True,
) -> tuple[Tensor, Tensor]:
    """Python launcher for the Triton MoE router kernel.

    Args:
        logits: Router logits of shape (M, E) where M is the number of tokens
            and E is the number of experts. Can be any float dtype.
        top_k: Number of top experts to select per token.
        normalize: Whether to normalize the top-k weights to sum to 1.

    Returns:
        A tuple (topk_weights, topk_indices) where:
        - topk_weights: float32 tensor of shape (M, top_k) with the selected routing weights
        - topk_indices: int64 tensor of shape (M, top_k) with the selected expert indices
    """
    assert logits.ndim == 2, f"Expected 2D logits, got shape {logits.shape}"
    M, N_EXPERTS = logits.shape
    assert top_k <= N_EXPERTS, f"top_k ({top_k}) must be <= N_EXPERTS ({N_EXPERTS})"

    logits = logits.contiguous()

    BLOCK_N = triton.next_power_of_2(N_EXPERTS)
    BLOCK_K = triton.next_power_of_2(top_k)

    # Output tensors: weights in float32, indices in int64
    topk_weights = torch.empty((M, top_k), device=logits.device, dtype=torch.float32)
    topk_indices = torch.empty((M, top_k), device=logits.device, dtype=torch.int64)

    grid = (M,)
    _moe_router_kernel[grid](
        logits,
        topk_weights,
        topk_indices,
        logits_row_stride=logits.stride(0),
        N_EXPERTS=N_EXPERTS,
        TOP_K=top_k,
        NORMALIZE=normalize,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )

    return topk_weights, topk_indices
