# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 long-context sparse decode attention in one launch.

Replaces the generic chain's split-K producer (``_gqa_sparse_decode_kernel``, one block per chunk) and the LSE
merge (``_merge_topk_attn_out_kernel``, with its MXFP8 copy for the o_proj) once the top-k block table exists.

Grid: one cluster of 16 CTAs per (query token, KV head); CTA ``r`` attends the token's ``r``-th selected block of
that head (the Triton producer's chunk ``r`` at 16 chunks), keeps the partial in shared memory, and after a cluster
barrier merges head ``r`` of the KV head's group from all 16 partials over DSMEM, writing the BF16 row and,
optionally, its MXFP8 copy and swizzled UE8M0 scales, bitwise ``torch.ops.trtllm.mxfp8_quantize(out, True)``.

Per-block math follows the Triton producer: NVFP4 K/V dequantized to FP16 (exact), FP8 q widened to FP16, QK and
PV on FP16 tensor cores with FP32 accumulation, log2-domain softmax with either the linear denominator or the
log-domain recurrence. Accumulation order inside the MMAs and the reductions differs, so results match to rounding,
not bit for bit.

Schedule: the request length and the query rows (FP8 widened to FP16 in shared memory) are read before the PDL grid
wait, which overlaps them with the select ahead when it releases this grid early (modeling's ``SELECT_EARLY_MAX_ROWS``);
the split-K partials travel by ``st.async`` into each receiving CTA's own mbarrier instead of a cluster-wide barrier.
Neither changes a value or the merge order.

Geometry: 16 query heads per KV head, two KV heads per rank (TP2), head_dim 128, page 128, top-k 16. The kernel
is ``torch.ops.trtllm.minimax_m3_sm107_tp2_sparse_decode_lc``, called through this target's ``m3_sparse_decode_lc``
entry.
"""

import torch

from tensorrt_llm.logger import logger

from ..ops.attention.m3_sparse_decode_lc import m3_sparse_decode_lc


class N1Slot:
    """The sparse layer's N1 id buffer (modeling's ``SPARSE_DECODE_LC_PREFETCH_SELF_KV``), set by the layer around its
    decode call and taken by the long-context decode; empty means no prefetch."""

    def __init__(self) -> None:
        self.ids = None

    def take(self):
        ids, self.ids = self.ids, None
        return ids


# int32 [2, N1_MAX_ROWS, 16]: the layer's selected block ids per (KV head, query row, slot) at its previous decode step.
N1_IDS = N1Slot()


def lc_applies(q, k_paged, v_paged, topk_idx, k_block_scale, v_block_scale, output) -> bool:
    """Whether the step's operands fit the kernel: TP2 geometry, packed NVFP4 pages, FP8 q, BF16 [T, 32, 128] out."""
    if q.dim() != 3 or tuple(q.shape[1:]) != (32, 128) or q.dtype != torch.float8_e4m3fn:
        return False
    if q.stride(2) != 1 or q.stride(1) % 16 or q.stride(0) % 16 or q.data_ptr() % 16:
        return False
    if (
        int(k_paged.shape[1]) != 2
        or topk_idx.dim() != 3
        or int(topk_idx.shape[0]) != 2
        or int(topk_idx.shape[1]) != int(q.shape[0])
        or int(topk_idx.shape[-1]) != 16
        or topk_idx.dtype != torch.int32
        or topk_idx.stride(2) != 1
    ):
        return False
    if v_paged.stride() != k_paged.stride():
        return False
    for t in (k_paged, v_paged):
        if (
            t.stride(3) != 1
            or t.stride(2) != 64
            or int(t.shape[3]) != 64
            or t.data_ptr() % 16
            or t.stride(0) % 16
            or t.stride(1) % 16
        ):
            return False
    for t in (k_block_scale, v_block_scale):
        if (
            t is None
            or int(t.shape[1]) != 2
            or t.stride(3) != 1
            or t.stride(2) != 8
            or t.data_ptr() % 8
            or t.stride(0) % 8
            or t.stride(1) % 8
        ):
            return False
    if v_block_scale.stride() != k_block_scale.stride():
        return False
    return (
        output.dtype == torch.bfloat16
        and output.dim() == 3
        and output.is_contiguous()
        and tuple(output.shape) == tuple(q.shape)
    )


def m3_lc_attn(
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
    linear_softmax=False,
    mxfp8=None,
    use_pdl=True,
    prewait=False,
    n1_ids=None,
):
    """Drop-in for ``minimax_m3_sparse_attn_decode`` on the TP2 NVFP4 shape; callers check ``lc_applies`` first.

    ``topk_idx`` is the head-major [2, total_q, 16] table the Triton producer takes. ``mxfp8`` is ``(q8, sf)`` from
    ``MERGE_MXFP8.buffers(4096, dev)`` to also emit the o_proj hand-off. ``n1_ids`` is the layer's N1 id buffer
    (``N1_IDS``): the decode prefetches last step's blocks into L2 and records this step's.
    """
    total_q = int(q.shape[0])
    q8, sf = (
        (mxfp8[0].view(torch.uint8), mxfp8[1].view(torch.uint8))
        if mxfp8 is not None
        else (None, None)
    )
    m3_sparse_decode_lc(
        q.view(torch.uint8),
        k_paged.view(torch.uint8),
        v_paged.view(torch.uint8),
        k_block_scale.view(torch.uint8),
        v_block_scale.view(torch.uint8),
        k_global_scale,
        v_global_scale,
        topk_idx,
        block_table,
        seq_lens,
        output.view(total_q, 4096),
        q8,
        sf,
        4096 // 128,
        int(decode_query_len),
        float(sm_scale),
        bool(linear_softmax),
        bool(use_pdl),
        bool(prewait),
        n1_ids,
    )
    key = (total_q, q8 is not None, n1_ids is not None)
    if key not in _engaged:
        _engaged.add(key)
        logger.info(
            f"[M3 sm_107 tp2] sparse decode Lc engaged: T={total_q}, kv_heads=2, mxfp8_handoff={q8 is not None}, "
            f"n1_self_kv={n1_ids is not None}"
        )


_engaged = set()
