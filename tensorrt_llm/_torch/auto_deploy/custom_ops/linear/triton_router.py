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

"""Triton kernel for fused MoE routing: softmax + top-k selection + scatter."""

import torch
import triton
import triton.language as tl


@triton.jit
def _moe_router_softmax_topk_kernel(
    logits_ptr,
    output_ptr,
    logits_row_stride: tl.constexpr,
    output_row_stride: tl.constexpr,
    N_EXPERTS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Fused top-k + softmax + scatter kernel for MoE routing.

    For each token row, computes:
    1. Top-k selection on raw logits
    2. Softmax over the selected top-k values
    3. Scatter the softmax values back into a full [E]-sized output row
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_N)
    mask = col_offsets < N_EXPERTS

    # Load logits row and upcast to float32
    logits_row_ptr = logits_ptr + row_idx * logits_row_stride
    logits = tl.load(logits_row_ptr + col_offsets, mask=mask, other=-float("inf"))
    logits_f32 = logits.to(tl.float32)

    # Iterative top-k selection on raw logits
    remaining = logits_f32
    # We'll store topk values and indices in registers
    # Use output row to scatter into
    out_row_ptr = output_ptr + row_idx * output_row_stride

    # Zero out the output row first
    tl.store(out_row_ptr + col_offsets, tl.zeros([BLOCK_N], dtype=tl.float32), mask=mask)

    # Collect top-k values for softmax normalization
    # First pass: find top-k values
    topk_offsets = tl.arange(0, BLOCK_K)
    topk_vals = tl.zeros([BLOCK_K], dtype=tl.float32)
    topk_idxs = tl.zeros([BLOCK_K], dtype=tl.int32)

    for k in tl.static_range(TOP_K):
        cur_max = tl.max(remaining, axis=0)
        is_max = (remaining == cur_max) & mask
        max_idx = tl.min(tl.where(is_max, col_offsets, BLOCK_N), axis=0)
        topk_vals = tl.where(topk_offsets == k, cur_max, topk_vals)
        topk_idxs = tl.where(topk_offsets == k, max_idx, topk_idxs)
        remaining = tl.where(col_offsets == max_idx, -float("inf"), remaining)

    # Softmax over the top-k values
    topk_mask = topk_offsets < TOP_K
    max_val = tl.max(topk_vals, axis=0)
    exp_vals = tl.exp(topk_vals - max_val)
    exp_vals = tl.where(topk_mask, exp_vals, 0.0)
    sum_exp = tl.sum(exp_vals, axis=0)
    softmax_vals = exp_vals / sum_exp

    # Scatter softmax values back to their expert positions
    for k in tl.static_range(TOP_K):
        idx = tl.sum(tl.where(topk_offsets == k, topk_idxs, 0), axis=0)
        val = tl.sum(tl.where(topk_offsets == k, softmax_vals, 0.0), axis=0)
        tl.store(out_row_ptr + idx, val)


def triton_moe_router_impl(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    top_k: int = 2,
) -> torch.Tensor:
    """Triton-accelerated MoE routing matching torch_moe_router signature.

    Performs linear projection then fused top-k + softmax + scatter using Triton.

    Args:
        hidden_states: Input tensor [B, S, H] or [B*S, H].
        weight: Router weight [E, H].
        bias: Router bias [E].
        top_k: Number of top experts to select.

    Returns:
        router_scores: [T, E] with softmax scores at selected expert positions.
    """
    hidden_dim = hidden_states.shape[-1]
    hidden_states_2d = hidden_states.reshape(-1, hidden_dim)

    # Linear projection (cuBLAS is optimal for GEMM)
    router_logits = torch.nn.functional.linear(hidden_states_2d, weight, bias)

    T, N_EXPERTS = router_logits.shape
    router_logits = router_logits.contiguous()

    BLOCK_N = triton.next_power_of_2(N_EXPERTS)
    BLOCK_K = triton.next_power_of_2(top_k)

    output = torch.zeros((T, N_EXPERTS), device=router_logits.device, dtype=router_logits.dtype)

    grid = (T,)
    _moe_router_softmax_topk_kernel[grid](
        router_logits,
        output,
        logits_row_stride=router_logits.stride(0),
        output_row_stride=output.stride(0),
        N_EXPERTS=N_EXPERTS,
        TOP_K=top_k,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=3,
    )

    return output
