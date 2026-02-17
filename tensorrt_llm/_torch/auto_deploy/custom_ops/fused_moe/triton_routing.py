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

"""Fused Triton kernel for MoE top-k routing with softmax.

Leverages the mathematical equivalence:
    topk(softmax(x)); x /= x.sum()  ≡  softmax(topk(x))

Instead of computing softmax over ALL experts (e.g. 256), then selecting top-k,
then renormalizing, this kernel:
  1. Finds top-k from raw logits (softmax is monotonic, preserves ordering)
  2. Computes softmax only over the k selected logits (e.g. k=8)

This fuses three separate kernel launches into one and avoids intermediate
global memory traffic.
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_topk_softmax_kernel(
    logits_ptr,  # Input: (T, E) router logits
    weights_ptr,  # Output: (T, K) routing weights (float32)
    indices_ptr,  # Output: (T, K) expert indices (int32)
    num_tokens,  # number of tokens (T)
    num_experts,  # number of experts (E), e.g. 256
    stride_lt,  # logits stride along token dim
    stride_le,  # logits stride along expert dim
    stride_wt,  # weights stride along token dim
    stride_wk,  # weights stride along topk dim
    stride_it,  # indices stride along token dim
    stride_ik,  # indices stride along topk dim
    BLOCK_E: tl.constexpr,  # >= num_experts, must be power of 2
    TOP_K: tl.constexpr,  # number of top experts to select (any positive int)
    BLOCK_K: tl.constexpr,  # >= TOP_K, must be power of 2 (for Triton tensor ops)
):
    """Fused top-k selection + softmax routing kernel.

    Each Triton program processes one token (row). It loads all expert logits,
    iteratively finds the top-k values/indices via repeated argmax, then
    computes a numerically-stable softmax over only the k selected logits.
    """
    # Each program handles one token
    token_id = tl.program_id(0)
    if token_id >= num_tokens:
        return

    # Load all expert logits for this token into registers
    offs_e = tl.arange(0, BLOCK_E)
    mask_e = offs_e < num_experts
    logits = tl.load(
        logits_ptr + token_id * stride_lt + offs_e * stride_le,
        mask=mask_e,
        other=float("-inf"),
    ).to(tl.float32)

    # --- Iterative top-k: find k largest values and their indices ---
    # Allocate with BLOCK_K (power of 2) so Triton tensor ops work for any TOP_K.
    # Unused padding slots stay at -inf → exp(-inf) = 0, so softmax is unaffected.
    topk_vals = tl.full([BLOCK_K], float("-inf"), dtype=tl.float32)
    topk_idxs = tl.zeros([BLOCK_K], dtype=tl.int32)
    offs_k = tl.arange(0, BLOCK_K)

    for k_i in tl.static_range(TOP_K):
        # Find the current maximum value across all experts
        max_val = tl.max(logits, axis=0)

        # Find the index of the maximum (pick smallest index on ties)
        is_max = logits == max_val
        # For non-max positions, substitute a large index so tl.min ignores them
        candidate = tl.where(is_max, offs_e, BLOCK_E)
        max_idx = tl.min(candidate, axis=0)

        # Store into the k-th slot of our top-k arrays
        ki_mask = offs_k == k_i
        topk_vals = tl.where(ki_mask, max_val, topk_vals)
        topk_idxs = tl.where(ki_mask, max_idx.to(tl.int32), topk_idxs)

        # Mask out the found maximum so it is not selected again
        logits = tl.where(offs_e == max_idx, float("-inf"), logits)

    # --- Numerically-stable softmax over only the top-k values ---
    max_topk = tl.max(topk_vals, axis=0)
    exp_vals = tl.exp(topk_vals - max_topk)
    sum_exp = tl.sum(exp_vals, axis=0)
    softmax_vals = exp_vals / sum_exp

    # --- Store results (only the valid TOP_K entries, not the BLOCK_K padding) ---
    mask_k = offs_k < TOP_K
    tl.store(
        weights_ptr + token_id * stride_wt + offs_k * stride_wk,
        softmax_vals,
        mask=mask_k,
    )
    tl.store(
        indices_ptr + token_id * stride_it + offs_k * stride_ik,
        topk_idxs,
        mask=mask_k,
    )


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    return 1 << math.ceil(math.log2(max(n, 1)))


def triton_fused_topk_softmax_fn(
    router_logits: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused top-k + softmax routing using a single Triton kernel.

    Args:
        router_logits: (T, E) float tensor of router logits.
        top_k: Number of experts to select per token.

    Returns:
        routing_weights: (T, top_k) float32 tensor of softmax routing weights.
        selected_experts: (T, top_k) int32 tensor of expert indices.
    """
    assert router_logits.ndim == 2, "router_logits must be 2-D (T, E)"
    num_tokens, num_experts = router_logits.shape

    # Allocate outputs
    routing_weights = torch.empty(
        (num_tokens, top_k), dtype=torch.float32, device=router_logits.device
    )
    selected_experts = torch.empty(
        (num_tokens, top_k), dtype=torch.int32, device=router_logits.device
    )

    # Determine compile-time constants
    BLOCK_E = _next_power_of_2(num_experts)
    BLOCK_K = _next_power_of_2(top_k)

    # Launch grid: one program per token
    grid = (num_tokens,)

    _fused_topk_softmax_kernel[grid](
        router_logits,
        routing_weights,
        selected_experts,
        num_tokens,
        num_experts,
        router_logits.stride(0),
        router_logits.stride(1),
        routing_weights.stride(0),
        routing_weights.stride(1),
        selected_experts.stride(0),
        selected_experts.stride(1),
        BLOCK_E=BLOCK_E,
        TOP_K=top_k,
        BLOCK_K=BLOCK_K,
    )

    return routing_weights, selected_experts


# ---------------------------------------------------------------------------
# Register as a torch custom op for graph tracing / export compatibility
# ---------------------------------------------------------------------------


@torch.library.custom_op("auto_deploy::triton_fused_topk_softmax", mutates_args=())
def triton_fused_topk_softmax(
    router_logits: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused top-k + softmax routing custom op.

    Computes ``softmax(topk(router_logits))`` in a single fused Triton kernel.
    This is mathematically equivalent to the 3-step sequence
    ``softmax → topk → renormalize`` used in standard MoE routers (e.g. Qwen3.5).

    Args:
        router_logits: (T, E) tensor of raw router logits.
        top_k: Number of top experts to select per token.

    Returns:
        A tuple of:
        - routing_weights: (T, top_k) float32 tensor.
        - selected_experts: (T, top_k) int32 tensor.
    """
    return triton_fused_topk_softmax_fn(router_logits, top_k)


@triton_fused_topk_softmax.register_fake
def _triton_fused_topk_softmax_fake(
    router_logits: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake (meta) implementation for tracing / export."""
    num_tokens = router_logits.shape[0]
    routing_weights = router_logits.new_empty((num_tokens, top_k), dtype=torch.float32)
    selected_experts = router_logits.new_empty((num_tokens, top_k), dtype=torch.int32)
    return routing_weights, selected_experts
