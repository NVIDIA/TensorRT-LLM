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
weights while attention runs. Up to ``PREWAIT_MAX_ROWS`` query rows it also stages the request length,
the query rows and the request's block-table row before that wait, overlapping them with the select ahead when it
releases this grid early (modeling's ``SELECT_EARLY_MAX_ROWS``). Neither changes a value.

Geometry: 8 query heads per KV head (TP8), one KV head, head_dim 128, page 128, top-k 16. The kernel is
``torch.ops.trtllm.minimax_m3_tp8_sparse_decode_lc8``, called through this target's ``m3_sparse_decode_lc8`` entry.
"""

import torch

from ..m31_switches import m31_disabled, m31_enabled
from ..ops.attention.m3_sparse_decode_lc8 import NUM_HEADS, WIDTH, m3_sparse_decode_lc8

# Up to this many query rows the kernel stages the request length, the query rows and the block-table row before its
# grid wait.
PREWAIT_MAX_ROWS = 8  # capacity: fixed: measured occupancy crossover
# The split-K merge forms its weights once per CTA (MERGE_ONCE: also enables the identity route and the L2 prefetches)
# and divides once per output (ONE_DIV_MERGE: sum_c w_c acc_c / sum_c w_c l_c in chunk order; deterministic, in the
# float64 reference's band). ONE_DIV_MERGE is default OFF (its VR TP8 natural-AL gate failed the frozen rule):
# TRTLLM_M31_ENABLE_LC8_ONEDIV=1 opts in; TRTLLM_M31_DISABLE_LC8_ONEDIV=1 still wins.
MERGE_ONCE = True
ONE_DIV_MERGE = m31_enabled("LC8_ONEDIV") and not m31_disabled("LC8_ONEDIV")
# Up to this many query rows the merge divides once (VR TP8: ONE_DIV wins at 4 / 8 rows and turns
# to a loss from 16 rows at 128K). TRTLLM_M31_DISABLE_ONEDIV_CAP=1 keeps ONE_DIV at every row count.
ONE_DIV_MAX_ROWS = (
    1 << 30 if m31_disabled("ONEDIV_CAP") else 8
)  # capacity: fixed: measured crossover


def one_div_applies(total_q: int) -> bool:
    """Whether the merge divides once at ``total_q`` query rows."""
    return ONE_DIV_MERGE and MERGE_ONCE and total_q <= ONE_DIV_MAX_ROWS


class PrefetchSlot:
    """The read-only tensors the next long-context sparse decode prefetches into L2 (the attention layer's o_proj weight
    and scales), set by the layer around its decode call and taken by the decode; empty means no prefetch."""

    def __init__(self) -> None:
        self.tensors = None

    def take(self):
        tensors, self.tensors = self.tensors, None
        return tensors


PREFETCH = PrefetchSlot()
# (this layer's id buffer, the next sparse layer's id buffer, its K / V pools, its K / V block scales)
NEXT_KV = PrefetchSlot()


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
    identity=False,
    identity_step=False,
    prefetch=None,
    next_kv=None,
):
    """Drop-in for ``minimax_m3_sparse_attn_decode`` on the TP8 NVFP4 shape; callers check ``lc8_applies`` first.

    ``topk_idx`` is the head-major [1, total_q, 16] view the Triton producer takes. ``mxfp8`` is ``(q8, sf)`` from
    ``MERGE_MXFP8.buffers(1024, dev)`` to also emit the o_proj hand-off. With ``MERGE_ONCE``,
    ``prefetch`` (up to two contiguous tensors) is pulled into L2 by the op's CTAs before their grid dependency, and
    ``next_kv`` drives the next-layer KV prefetch.
    """
    merge_once = MERGE_ONCE
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
        total_q <= PREWAIT_MAX_ROWS,
        merge_once=merge_once,
        identity=bool(identity) and merge_once,
        identity_step=bool(identity_step) and merge_once,
        one_div=merge_once and one_div_applies(total_q),
        prefetch=prefetch if merge_once else None,
        next_kv=(
            tuple(next_kv[:2]) + tuple(t.view(torch.uint8) for t in next_kv[2:])
            if merge_once and next_kv is not None
            else None
        ),
    )
