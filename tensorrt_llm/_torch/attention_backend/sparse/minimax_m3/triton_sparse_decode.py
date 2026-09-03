# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: Apache-2.0
# Vendored from vLLM (Apache-2.0):
# https://github.com/vllm-project/vllm/blob/6f91edf96d3f3272945809c04702380053bff4de/vllm/models/minimax_m3/common/ops/sparse_attn.py
"""Triton block-sparse GQA decode attention for MiniMax-M3.

Flash-decoding over the blocks the indexer selected: one CTA per (query token,
top-k chunk, KV head) accumulates a partial output plus its log-sum-exp, and a
second kernel merges the chunks by LSE weight. That beats running the
context-schedule FMHA kernel at decode, where a single query token leaves most
of a 128-row Q tile idle.

Vendored from the vLLM source linked in the file header (v0.26.1rc0-77-g6f91edf96).
Differences from upstream:

* K and V are separate HND paged views ([num_pages, num_kv_heads, page_size,
  head_dim]) with independent strides rather than one cache fused along the
  last dim, because that is how the M3 KV pool is laid out here.
* Only the scalar KV-scale mode is kept; M3 stores unscaled E4M3 K/V.
* Partial-output and LSE scratch come from the persistent buffer arena so their
  addresses survive CUDA graph replay.
* The merge kernel zeroes rows whose chunks are all empty instead of letting
  them produce NaN, since CUDA-graph padding rows flow on into the rest of the
  network here.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.memory_buffer_utils import get_memory_buffers

# One sparse block is exactly one KV page.
SPARSE_BLOCK_SIZE = 128

_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)

_TL_DTYPES = {
    torch.bfloat16: tl.bfloat16,
    torch.float16: tl.float16,
    torch.float32: tl.float32,
}

# Total CTAs the split-K partitioning aims for before it stops splitting.
_TARGET_GRID = 256


def _pdl_enabled() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(16, triton.next_power_of_2(args["gqa_group_size"])),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _gqa_sparse_decode_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    k_ptr,  # paged K: [num_pages, num_kv_heads, page_size, head_dim]
    v_ptr,  # paged V: same layout as K
    kv_scale_ptr,  # scalar dequant scale, or a dummy when USE_SCALE is False
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # partial out: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partial lse (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    block_table_ptr,  # [num_reqs, max_blocks]
    seq_lens,  # [num_reqs]
    total_q,
    gqa_group_size,
    head_dim,
    max_topk,
    sm_scale,
    decode_query_len,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_k_blk,
    stride_k_h,
    stride_k_pos,
    stride_k_d,
    stride_v_blk,
    stride_v_h,
    stride_v_pos,
    stride_v_d,
    stride_th,
    stride_tn,
    stride_tk,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_bt_b,
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE (128)
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    USE_SCALE: tl.constexpr,  # apply the scalar dequant scale to an fp8 cache
    WIDEN_Q: tl.constexpr,  # q arrives FP8 and is widened in-register
    COMPUTE_DTYPE: tl.constexpr,  # dtype q is widened to; drives the QK/PV math
    USE_PDL: tl.constexpr,
):
    sm_scale_log2e = sm_scale * 1.4426950409
    # Split-K over the topk dimension: pid(0) folds (query token, chunk).
    pid_bc, pid_kh = tl.program_id(0), tl.program_id(1)
    pid_b = pid_bc % total_q
    pid_c = pid_bc // total_q
    req_id = pid_b // decode_query_len
    q_offset = pid_b - req_id * decode_query_len
    pid_h = pid_kh * gqa_group_size
    chunk_size_topk = (max_topk + NUM_TOPK_CHUNKS - 1) // NUM_TOPK_CHUNKS
    chunk_start = pid_c * chunk_size_topk

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    seq_len = tl.load(seq_lens + req_id)
    query_pos = seq_len - decode_query_len + q_offset
    # Full-CG padding uses zero-length request rows. Clamp to an empty
    # attention range instead of letting padded rows produce negative lengths.
    kv_len = tl.maximum(query_pos + 1, 0)

    # Bound the walk by the token's own valid block count rather than a
    # sentinel: the selector emits ascending ids and only pads with -1 past
    # min(topk, cdiv(kv_len, blk)) entries, so those are never dereferenced.
    num_blocks = (kv_len + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
    real_topk = tl.minimum(max_topk, num_blocks)
    chunk_end = tl.minimum(chunk_start + chunk_size_topk, real_topk)

    off_n = tl.arange(0, BLOCK_SIZE_K)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_h = tl.arange(0, BLOCK_SIZE_H)
    d_mask = off_d < head_dim
    h_mask = off_h < gqa_group_size
    hd_mask = h_mask[:, None] & d_mask[None, :]
    bt_row = block_table_ptr + req_id * stride_bt_b

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    acc_o = tl.zeros((BLOCK_SIZE_H, BLOCK_SIZE_D), dtype=tl.float32)
    q = tl.load(
        q_ptr
        + pid_b * stride_qn
        + (pid_h + off_h[:, None]) * stride_qh
        + off_d[None, :] * stride_qd,
        mask=hd_mask,
        other=0.0,
    )
    # Widen before anything reads q.dtype: K, V and p are all cast to it below,
    # so leaving q narrow would silently run the whole attention in FP8 and
    # quantize the softmax probabilities. E4M3 -> BF16 is exact, so the result
    # matches a caller that widened q itself before the call.
    if WIDEN_Q:
        q = q.to(COMPUTE_DTYPE)
    kv_scale = tl.load(kv_scale_ptr) if USE_SCALE else 1.0

    cur_idx_ptr = t_ptr + pid_kh * stride_th + pid_b * stride_tn + chunk_start * stride_tk
    for _ in tl.range(chunk_start, chunk_end):
        blk = tl.load(cur_idx_ptr)
        cur_idx_ptr = cur_idx_ptr + stride_tk
        # int64 page offsets: a large cache overflows int32 well before the
        # per-page block offsets do.
        page = tl.load(bt_row + blk).to(tl.int64)
        pos_mask = blk * BLOCK_SIZE_K + off_n < kv_len
        k = tl.load(
            k_ptr
            + page * stride_k_blk
            + pid_kh * stride_k_h
            + off_n[None, :] * stride_k_pos
            + off_d[:, None] * stride_k_d,
            mask=d_mask[:, None] & pos_mask[None, :],
            other=0.0,
        ).to(q.dtype)
        if USE_SCALE:
            k = (k * kv_scale).to(q.dtype)
        qk = tl.where(pos_mask[None, :], 0.0, float("-inf"))
        qk += tl.dot(q, k) * sm_scale_log2e
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
        v = tl.load(
            v_ptr
            + page * stride_v_blk
            + pid_kh * stride_v_h
            + off_n[:, None] * stride_v_pos
            + off_d[None, :] * stride_v_d,
            mask=pos_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(q.dtype)
        if USE_SCALE:
            v = (v * kv_scale).to(q.dtype)
        acc_o += tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)

    # An empty chunk of an active row must store zero, or the merge hits 0 * NaN.
    scale = tl.where(lse_i > float("-inf"), tl.exp2(m_i - lse_i), 0.0)
    acc_o = acc_o * scale[:, None]
    o_base = o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h
    tl.store(
        o_base + off_h[:, None] * stride_o_h + off_d[None, :] * stride_o_d,
        acc_o.to(o_ptr.dtype.element_ty),
        mask=hd_mask,
    )
    lse_base = lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h
    tl.store(lse_base + off_h * stride_l_h, lse_i, mask=h_mask)

    # After the stores, never before: the merge grid's gdc_wait() releases on
    # this trigger, and it reads exactly the partials written above.
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@triton.heuristics({"BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"])})
@triton.jit
def _merge_topk_attn_out_kernel(
    o_ptr,  # partials: [NUM_TOPK_CHUNKS, total_q, num_heads, head_dim]
    lse_ptr,  # partials (log2): [NUM_TOPK_CHUNKS, total_q, num_heads]
    out_ptr,  # merged out: [total_q, num_heads, head_dim]
    head_dim,
    stride_o_c,
    stride_o_b,
    stride_o_h,
    stride_o_d,
    stride_l_c,
    stride_l_b,
    stride_l_h,
    stride_out_n,
    stride_out_h,
    stride_out_d,
    NUM_TOPK_CHUNKS: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    pid_b, pid_h = tl.program_id(0), tl.program_id(1)

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    off_c = tl.arange(0, NUM_TOPK_CHUNKS)
    off_d = tl.arange(0, BLOCK_SIZE_D)
    o = tl.load(
        o_ptr
        + pid_b * stride_o_b
        + pid_h * stride_o_h
        + off_c[:, None] * stride_o_c
        + off_d[None, :] * stride_o_d,
        mask=off_d[None, :] < head_dim,
        other=0.0,
    ).to(tl.float32)
    # Empty chunks contribute -inf, hence weight 0.
    lse = tl.load(lse_ptr + pid_b * stride_l_b + pid_h * stride_l_h + off_c * stride_l_c)
    lse_max = tl.max(lse, axis=0)
    # A row whose every chunk is empty (a CUDA-graph padding row) would give
    # -inf - -inf here. Zero it instead: the padded output is discarded, but a
    # NaN would survive into the residual stream and the all-reduce.
    lse_max = tl.where(lse_max == float("-inf"), 0.0, lse_max)
    weights = tl.exp2(lse - lse_max)
    denom = tl.sum(weights, axis=0)
    weights = weights / tl.where(denom > 0, denom, 1.0)
    o_merged = tl.sum(o * weights[:, None], axis=0)
    out_ptrs = out_ptr + pid_b * stride_out_n + pid_h * stride_out_h + off_d * stride_out_d
    tl.store(out_ptrs, o_merged.to(out_ptr.dtype.element_ty), mask=off_d < head_dim)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def resolve_num_topk_chunks(total_q: int, num_kv_heads: int, max_topk: int) -> int:
    """Split-K factor over the top-k blocks, as a power of two.

    Depends only on shapes that are fixed for a captured batch size, so the
    launch geometry is frozen inside a CUDA graph.
    """
    target = max(1, min(max_topk, _TARGET_GRID // max(1, total_q * num_kv_heads)))
    return 1 << (target.bit_length() - 1)


@torch.no_grad()
def minimax_m3_sparse_attn_decode(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    k_paged: torch.Tensor,  # [num_pages, num_kv_heads, page_size, head_dim]
    v_paged: torch.Tensor,  # same layout as k_paged
    topk_idx: torch.Tensor,  # [num_kv_heads, total_q, topk]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    *,
    sm_scale: float,
    output: torch.Tensor,  # [total_q, num_heads, head_dim]
    decode_query_len: int,
    kv_scale: Optional[torch.Tensor] = None,
    num_topk_chunks: Optional[int] = None,
) -> None:
    """Block-sparse GQA decode attention, written into output in place.

    q may be FP8, as a fused QK-norm/RoPE producer emits it. The kernel widens
    it to output.dtype in-register rather than making the caller materialize a
    widened copy, which would cost a standalone kernel on every sparse layer.
    The attention math is unchanged either way, since E4M3 widens exactly.

    kv_scale is an optional scalar dequantization factor for an FP8 cache;
    MiniMax-M3 stores unscaled E4M3, so it is normally None. num_topk_chunks
    overrides the split-K factor and exists for tests: the merged result must
    not depend on it.
    """
    total_q, num_heads, head_dim = q.shape
    num_kv_heads = int(k_paged.shape[1])
    if total_q != int(seq_lens.shape[0]) * decode_query_len:
        raise ValueError(
            f"total_q ({total_q}) must be batch ({int(seq_lens.shape[0])}) * "
            f"decode_query_len ({decode_query_len})."
        )
    if int(k_paged.shape[2]) != SPARSE_BLOCK_SIZE:
        raise ValueError(
            f"MiniMax-M3 sparse decode requires page_size={SPARSE_BLOCK_SIZE}; "
            f"got {int(k_paged.shape[2])}."
        )
    max_topk = int(topk_idx.shape[-1])
    gqa_group_size = num_heads // num_kv_heads
    use_scale = k_paged.dtype in _FP8_DTYPES and kv_scale is not None
    # Triton needs a real pointer even for the unused argument.
    scale_arg = kv_scale if use_scale else output

    widen_q = q.dtype in _FP8_DTYPES
    if widen_q and output.dtype not in _TL_DTYPES:
        raise ValueError(
            f"MiniMax-M3 sparse decode cannot widen FP8 q into {output.dtype}; "
            f"supported compute dtypes are {sorted(d.__str__() for d in _TL_DTYPES)}."
        )
    compute_dtype = _TL_DTYPES[output.dtype] if widen_q else tl.float32

    if num_topk_chunks is None:
        num_topk_chunks = resolve_num_topk_chunks(total_q, num_kv_heads, max_topk)
    elif num_topk_chunks & (num_topk_chunks - 1):
        raise ValueError(f"num_topk_chunks must be a power of two; got {num_topk_chunks}.")

    # Persistent arena rather than torch.empty, so the partials keep one address
    # across CUDA graph replays.
    reserve = torch.cuda.is_current_stream_capturing()
    # fp32 partials: at decode these are a couple of MB, and keeping them wide
    # means the split-K factor cannot perturb the merged result.
    o_partial = get_memory_buffers().get_buffer(
        [num_topk_chunks, total_q, num_heads, head_dim],
        torch.float32,
        buffer_name="m3_sparse_decode_o_partial",
        reserve_buffer=reserve,
    )
    lse_partial = get_memory_buffers().get_buffer(
        [num_topk_chunks, total_q, num_heads],
        torch.float32,
        buffer_name="m3_sparse_decode_lse_partial",
        reserve_buffer=reserve,
    )

    use_pdl = _pdl_enabled()
    pdl_launch = {"launch_pdl": True} if use_pdl else {}

    _gqa_sparse_decode_kernel[(total_q * num_topk_chunks, num_kv_heads)](
        q,
        k_paged,
        v_paged,
        scale_arg,
        topk_idx,
        o_partial,
        lse_partial,
        block_table,
        seq_lens,
        total_q,
        gqa_group_size,
        head_dim,
        max_topk,
        sm_scale,
        decode_query_len,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_paged.stride(0),
        k_paged.stride(1),
        k_paged.stride(2),
        k_paged.stride(3),
        v_paged.stride(0),
        v_paged.stride(1),
        v_paged.stride(2),
        v_paged.stride(3),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_SCALE=use_scale,
        WIDEN_Q=widen_q,
        COMPUTE_DTYPE=compute_dtype,
        USE_PDL=use_pdl,
        **pdl_launch,
    )
    _merge_topk_attn_out_kernel[(total_q, num_heads)](
        o_partial,
        lse_partial,
        output,
        head_dim,
        o_partial.stride(0),
        o_partial.stride(1),
        o_partial.stride(2),
        o_partial.stride(3),
        lse_partial.stride(0),
        lse_partial.stride(1),
        lse_partial.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_PDL=use_pdl,
        **pdl_launch,
    )


__all__ = [
    "SPARSE_BLOCK_SIZE",
    "minimax_m3_sparse_attn_decode",
    "resolve_num_topk_chunks",
]
