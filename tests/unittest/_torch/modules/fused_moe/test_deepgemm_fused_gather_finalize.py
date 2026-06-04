# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Differential correctness test for ``triton_fused_gather_finalize``.

The DeepGemm post-GEMM pipeline used to be a pair of ops::

    triton_masked_index_gather(permuted_data, h3, ...)  # gather into buffer
    torch.ops.trtllm.moe_finalize_scale_op(permuted_data, ...)  # weight + reduce

``triton_fused_gather_finalize`` fuses both: it reads the expert GEMM output
``h3`` directly via the reverse-permute map, applies the routing weights, and
accumulates to the final output, eliminating the intermediate ``permuted_data``
buffer.

This test feeds **the same, genuinely self-consistent permutation maps** (built
by the real ``moe_permute_op``) to both the old pair and the new fused kernel,
then asserts the two outputs match. The C++ ``finalizeMoeRoutingKernel`` and the
fused Triton kernel both accumulate ``k = 0..topk-1`` in order in an fp32
accumulator with the same round-to-nearest fp32->bf16 final cast, so the result
should be bit-identical; the only residual difference is whether the
``acc += scale * value`` multiply-add contracts to an FMA identically across the
Triton and nvcc backends. The committed gate is therefore a tight tolerance, and
the test additionally reports whether the result happened to be bitwise equal.

Run as::

    pytest tests/unittest/_torch/modules/fused_moe/test_deepgemm_fused_gather_finalize.py -v
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm.*)
from tensorrt_llm._torch.modules.fused_moe.fused_moe_deepgemm import (
    preprocess_after_permute,
    triton_fused_gather_finalize,
    triton_masked_index_gather,
)
from tensorrt_llm._utils import get_sm_version

skip_unsupported = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() < 90,
    reason="Requires CUDA SM90+ (DeepGemm MoE post-GEMM kernels)",
)


@dataclass(frozen=True)
class GatherFinalizeShape:
    name: str
    num_rows: int
    hidden: int
    num_experts: int
    top_k: int


# Anchors mirror the PR-3 microbench sweep (study-plan note): the model anchor
# (32 x 8 x 4096 x 128), the single-token decode corner, and a medium batch.
SHAPES = [
    GatherFinalizeShape("anchor", num_rows=32, hidden=4096, num_experts=128, top_k=8),
    GatherFinalizeShape("anchor_h7168", num_rows=32, hidden=7168, num_experts=128, top_k=8),
    GatherFinalizeShape("single_token", num_rows=1, hidden=512, num_experts=64, top_k=4),
    GatherFinalizeShape("medium_batch", num_rows=64, hidden=7168, num_experts=128, top_k=8),
    GatherFinalizeShape("topk4", num_rows=16, hidden=4096, num_experts=64, top_k=4),
]


def _make_routing(
    num_rows: int, num_experts: int, top_k: int, *, device: str, generator: torch.Generator
):
    """Synthesize a valid (token_selected_experts, token_final_scales) pair.

    Each row selects ``top_k`` *distinct* experts (top-k never repeats an
    expert), with positive routing weights that sum to one per row.
    """
    logits = torch.randn(
        (num_rows, num_experts), device=device, dtype=torch.float32, generator=generator
    )
    topk_vals, topk_ids = logits.topk(top_k, dim=-1)
    token_selected_experts = topk_ids.to(torch.int32)
    token_final_scales = torch.softmax(topk_vals, dim=-1).to(torch.float32)
    return token_selected_experts, token_final_scales


@skip_unsupported
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.name)
def test_fused_gather_finalize_matches_unfused(shape: GatherFinalizeShape) -> None:
    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(1234)

    num_rows = shape.num_rows
    hidden = shape.hidden
    num_experts = shape.num_experts
    top_k = shape.top_k

    # Single-rank configuration: every expert is local to this node.
    tp_size, tp_rank, ep_size, ep_rank = 1, 0, 1, 0
    cluster_size, cluster_rank = 1, 0
    num_experts_per_node = num_experts

    token_selected_experts, token_final_scales = _make_routing(
        num_rows, num_experts, top_k, device=device, generator=gen
    )

    # Driver activations for the permutation. Only the maps are consumed below;
    # the expert GEMM output ``h3`` is synthesized independently.
    x = torch.randn((num_rows, hidden), device=device, dtype=torch.bfloat16, generator=gen)

    # Real permutation maps (self-consistent by construction).
    (
        permuted_row_to_unpermuted_row_tensor,
        _permuted_token_selected_experts_tensor,
        permuted_data_tensor,
        expert_first_token_offset_tensor,
        _permuted_token_final_scales_tensor,
        unpermuted_row_to_permuted_row_tensor,
    ) = torch.ops.trtllm.moe_permute_op(
        x,
        token_selected_experts,
        token_final_scales,
        None,  # fc1_expert_weights
        None,  # fc2_expert_weights
        None,  # quant_scales
        input_sf=None,
        num_experts_on_rank=num_experts_per_node,
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        cluster_size=cluster_size,
        cluster_rank=cluster_rank,
        min_latency_mode=False,
        use_fp8_block_scaling=True,
    )

    num_expanded = num_rows * top_k
    assert permuted_data_tensor.shape[0] == num_expanded

    _masked_m, token_to_expert_map = preprocess_after_permute(
        expert_first_token_offset_tensor, permuted_data_tensor
    )

    # Synthesize the expert GEMM output h3: [num_experts, max_tokens_per_expert,
    # hidden]. At most ``num_rows`` tokens can route to any single expert (top-k
    # picks distinct experts per token), so an m_max of align(num_rows, 128)
    # safely bounds the per-expert column dimension, matching forward().
    m_max = ((num_rows + 127) // 128) * 128
    h3 = torch.randn(
        (num_experts, m_max, hidden), device=device, dtype=torch.bfloat16, generator=gen
    )

    # ---- Old path: gather into permuted_data, then finalize-scale ----
    gather_out = torch.empty((num_expanded, hidden), device=device, dtype=h3.dtype)
    triton_masked_index_gather(
        gather_out, h3, expert_first_token_offset_tensor, token_to_expert_map
    )
    out_unfused = torch.ops.trtllm.moe_finalize_scale_op(
        gather_out,
        None,  # biases
        token_final_scales,
        unpermuted_row_to_permuted_row_tensor,
        permuted_row_to_unpermuted_row_tensor,
        token_selected_experts,
        expert_first_token_offset_tensor,
        False,  # enable_alltoall
        num_rows,
        hidden,  # (possibly padded) hidden_size
        hidden,  # unpadded hidden size (no padding in this test)
        top_k,
        num_experts_per_node,
        tp_size,
        tp_rank,
        ep_size,
        ep_rank,
    )

    # ---- New path: single fused kernel reading h3 directly ----
    out_fused = triton_fused_gather_finalize(
        h3=h3,
        token_final_scales=token_final_scales,
        unpermuted_row_to_permuted_row=unpermuted_row_to_permuted_row_tensor,
        token_to_expert_map=token_to_expert_map,
        expert_first_token_offset=expert_first_token_offset_tensor,
        token_selected_experts=token_selected_experts,
        num_rows=num_rows,
        hidden_size=hidden,
        unpadded_hidden_size=hidden,
        experts_per_token=top_k,
        num_experts_per_node=num_experts_per_node,
        ep_rank=ep_rank,
    )

    assert out_fused.shape == out_unfused.shape == (num_rows, hidden)
    assert out_fused.dtype == out_unfused.dtype == torch.bfloat16

    # Report bit-exactness (documents the PR claim) without gating on it.
    bitwise_equal = torch.equal(out_fused, out_unfused)
    max_abs_diff = (out_fused.float() - out_unfused.float()).abs().max().item()
    print(f"[{shape.name}] bitwise_equal={bitwise_equal} max_abs_diff={max_abs_diff:.3e}")

    # Committed gate: tight tolerance (robust across FMA-contraction differences
    # between the Triton and nvcc backends).
    torch.testing.assert_close(out_fused, out_unfused, rtol=1e-2, atol=1e-2)
