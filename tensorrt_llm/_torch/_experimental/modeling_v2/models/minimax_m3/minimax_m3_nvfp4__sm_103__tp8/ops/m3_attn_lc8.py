# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 long-context sparse decode attention at 8 query heads in one launch.

Replaces the generic chain's split-K producer (``_gqa_sparse_decode_kernel``, one block per chunk) and the LSE
merge (``_merge_topk_attn_out_kernel``, with its MXFP8 copy for the o_proj) once the top-k block table exists.

Grid: one cluster of 16 CTAs per query token; CTA ``r`` attends the token's ``r``-th selected block for all 8 heads,
keeps the partial in shared memory, and the cluster merges the 16 partials over DSMEM into the BF16 row and,
optionally, its MXFP8 copy and swizzled UE8M0 scales exactly as ``_store_mxfp8_head`` lays them out.

Per-block math follows the Triton producer: NVFP4 K/V dequantized to FP16 (exact), FP8 q widened to FP16, QK and PV
on FP16 tensor cores with FP32 accumulation, log2-domain softmax with either the linear denominator or the
log-domain recurrence. Accumulation order inside the MMAs and the reductions differs, so results match to rounding,
not bit for bit.

Schedule: the kernel releases its PDL dependents right after its grid wait, so the o_proj launches and streams its
weights while attention runs. Up to ``LC8_PREWAIT_MAX_ROWS`` (modeling.py) query rows it also stages the request
length, the query rows and the request's block-table row before that wait, overlapping them with the select ahead
when it releases this grid early (``SELECT_EARLY_MAX_ROWS`` in modeling.py). Neither changes a value.

Geometry: 8 query heads per KV head (TP8), one KV head, head_dim 128, page 128, top-k 16. The kernel is
``torch.ops.trtllm.minimax_m3_tp8_sparse_decode_lc8``, called through this target's ``m3_sparse_decode_lc8`` entry.
"""

import torch

from ..ops.attention.m3_sparse_decode_lc8 import NUM_HEADS, WIDTH, m3_sparse_decode_lc8


def _prewait_max_rows() -> int:
    """``LC8_PREWAIT_MAX_ROWS`` of the target's dispatch (imported at call time: modeling imports this module)."""
    from .. import modeling

    return modeling.LC8_PREWAIT_MAX_ROWS


def lc8_applies(q, k_paged, v_paged, topk_idx, k_block_scale, v_block_scale, output) -> bool:
    """Whether the step's operands fit the kernel: TP8 geometry, packed NVFP4 pages, FP8 q, BF16 [T, 8, 128] out."""
    if q.dim() != 3 or tuple(q.shape[1:]) != (NUM_HEADS, 128) or q.dtype != torch.float8_e4m3fn:
        return False
    if q.stride(2) != 1 or q.stride(1) % 16 or q.stride(0) % 16 or q.data_ptr() % 16:
        return False
    if int(k_paged.shape[1]) != 1 or int(topk_idx.shape[0]) != 1 or int(topk_idx.shape[-1]) != 16:
        return False
    for t in (k_paged, v_paged):
        if (
            t.stride(3) != 1
            or t.stride(2) != 64
            or int(t.shape[3]) != 64
            or t.data_ptr() % 16
            or t.stride(0) % 16
        ):
            return False
    for t in (k_block_scale, v_block_scale):
        if t is None or t.stride(3) != 1 or t.stride(2) != 8 or t.data_ptr() % 8 or t.stride(0) % 8:
            return False
    return (
        output.dtype == torch.bfloat16
        and output.dim() == 3
        and output.is_contiguous()
        and tuple(output.shape) == tuple(q.shape)
    )


def m3_lc8_attn(
    q,
    k_paged,
    v_paged,
    topk_idx,
    block_table,
    seq_lens,
    *,
    sm_scale,
    output,
    decode_query_len,
    k_block_scale,
    v_block_scale,
    k_global_scale,
    v_global_scale,
    linear_softmax=True,
    mxfp8=None,
    use_pdl=True,
):
    """Drop-in for ``minimax_m3_sparse_attn_decode`` on the TP8 NVFP4 shape; callers check ``lc8_applies`` first.

    ``topk_idx`` is the head-major [1, total_q, 16] view the Triton producer takes. ``mxfp8`` is ``(q8, sf)`` from
    ``MERGE_MXFP8.buffers(1024, dev)`` to also emit the o_proj hand-off.
    """
    total_q = int(q.shape[0])
    q8, sf = (
        (mxfp8[0].view(torch.uint8), mxfp8[1].view(torch.uint8))
        if mxfp8 is not None
        else (None, None)
    )
    m3_sparse_decode_lc8(
        q.view(torch.uint8),
        k_paged.view(torch.uint8),
        v_paged.view(torch.uint8),
        k_block_scale.view(torch.uint8),
        v_block_scale.view(torch.uint8),
        k_global_scale,
        v_global_scale,
        topk_idx[0],
        block_table,
        seq_lens,
        output.view(total_q, WIDTH),
        q8,
        sf,
        WIDTH // 128,
        int(decode_query_len),
        float(sm_scale),
        bool(linear_softmax),
        bool(use_pdl),
        total_q <= _prewait_max_rows(),
    )
