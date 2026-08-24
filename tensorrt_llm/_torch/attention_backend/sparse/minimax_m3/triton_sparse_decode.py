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
* A packed NVFP4 cache is read in place, dequantized in-register from its E2M1
  nibbles and E4M3 block scales.
* Partial-output and LSE scratch come from the persistent buffer arena so their
  addresses survive CUDA graph replay.
* A single top-k chunk writes the final output directly, avoiding scratch and
  the otherwise redundant LSE merge.
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

from .msa_utils import check_decode_span_shape

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

# NVFP4 quantizes in groups of 16 cache elements along head_dim, one E4M3 scale
# byte each; see the fp4_quantize call in msa_scatter.fused_write_layer_caches_nvfp4.
NVFP4_SF_VEC_SIZE = 16


def _pdl_enabled() -> bool:
    return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


@triton.jit
def _e2m1x2_to_f16x2(packed):
    """Convert one packed E2M1 byte into its low/high FP16 values."""
    return tl.inline_asm_elementwise(
        asm="""{
        .reg .b8 src;
        .reg .b32 pair;
        mov.b32 {src, _, _, _}, $2;
        cvt.rn.f16x2.e2m1x2 pair, src;
        mov.b32 {$0, $1}, pair;
        }""",
        constraints="=h,=h,r",
        args=(packed,),
        dtype=(tl.float16, tl.float16),
        is_pure=True,
        pack=1,
    )


@triton.jit
def _dequant_nvfp4_rows(
    packed,  # [BLOCK_N, D // 2] uint8, two E2M1 nibbles per byte
    scale_bytes,  # [BLOCK_N, SCALE_COLS] uint8, one E4M3 scale per 16 elements
    BLOCK_N: tl.constexpr,
    D: tl.constexpr,
    SCALE_COLS: tl.constexpr,
    SF_VEC: tl.constexpr,
):
    """Dequantize one packed NVFP4 [BLOCK_N, D] tile to fp16.

    The caller gathers the block scales in whichever layout its role stores
    them, so this handles only the value math and the scale broadcast. The
    per-tensor scale is applied by the caller.
    """
    # Blackwell converts both E2M1 nibbles with one instruction. Joining the
    # low/high results along a new innermost axis and flattening it rebuilds
    # the original element order without scalar exponent/sign reconstruction.
    value_lo, value_hi = _e2m1x2_to_f16x2(packed)
    values = tl.reshape(tl.join(value_lo, value_hi), (BLOCK_N, D))

    scales = scale_bytes.to(tl.float8e4nv, bitcast=True).to(tl.float16)
    scales = tl.reshape(
        tl.broadcast_to(scales[:, :, None], (BLOCK_N, SCALE_COLS, SF_VEC)), (BLOCK_N, D)
    )
    return values * scales


@triton.heuristics(
    {
        "BLOCK_SIZE_H": lambda args: max(16, triton.next_power_of_2(args["gqa_group_size"])),
        "BLOCK_SIZE_D": lambda args: triton.next_power_of_2(args["head_dim"]),
    }
)
@triton.jit(do_not_specialize=["decode_query_len"])
def _gqa_sparse_decode_kernel(
    q_ptr,  # [total_q, num_heads, head_dim]
    k_ptr,  # paged K: [num_pages, num_kv_heads, page_size, head_dim (or /2 packed)]
    v_ptr,  # paged V: same layout as K
    kv_scale_ptr,  # scalar dequant scale, or a dummy when USE_SCALE is False
    k_sc_ptr,  # paged K block scales: [num_pages, num_kv_heads, page_size, D/16]
    v_sc_ptr,  # paged V block scales: same shape, 4x4 token-quad order
    k_gs_ptr,  # NVFP4 per-tensor K dequant scale, one fp32
    v_gs_ptr,  # NVFP4 per-tensor V dequant scale, one fp32
    t_ptr,  # topk_idx: [num_kv_heads, total_q, topk]
    o_ptr,  # partial out, or final [total_q, num_heads, head_dim] when DIRECT_OUTPUT
    lse_ptr,  # partial lse (log2), unused when DIRECT_OUTPUT
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
    stride_ksc_blk,
    stride_ksc_h,
    stride_vsc_blk,
    stride_vsc_h,
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
    KV_NVFP4: tl.constexpr,  # cache holds packed E2M1 plus E4M3 block scales
    USE_LINEAR_SOFTMAX: tl.constexpr,  # measured NVFP4 online-softmax recurrence
    SCALE_COLS: tl.constexpr,  # head_dim // NVFP4_SF_VEC_SIZE, NVFP4 only
    DIRECT_OUTPUT: tl.constexpr,  # one chunk needs neither FP32 scratch nor a merge
):
    sm_scale_log2e = sm_scale * 1.4426950409
    if KV_NVFP4:
        # Both per-tensor dequant scales factor out of the inner loop: QK is
        # linear in K, and V's scale commutes with the softmax weights. Reading
        # them from device memory keeps the launch free of a host sync, so
        # CUDA-graph capture stays unconstrained.
        sm_scale_log2e = sm_scale_log2e * tl.load(k_gs_ptr)
        v_global_scale = tl.load(v_gs_ptr)
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
    if KV_NVFP4:
        # Packed byte column and block-scale column. The NVFP4 path runs only at
        # head_dim == BLOCK_SIZE_D (128), so neither needs a tail mask.
        off_dp = tl.arange(0, BLOCK_SIZE_D // 2)
        off_s = tl.arange(0, SCALE_COLS)
    d_mask = off_d < head_dim
    h_mask = off_h < gqa_group_size
    hd_mask = h_mask[:, None] & d_mask[None, :]
    bt_row = block_table_ptr + req_id * stride_bt_b

    m_i = tl.full((BLOCK_SIZE_H,), float("-inf"), dtype=tl.float32)
    if USE_LINEAR_SOFTMAX:
        # Keep the linear softmax denominator and reuse the producer's alpha.
        # This removes one exp2 plus one log2 from every visited KV page; only
        # the final LSE written for split-K merging still needs a logarithm.
        l_i = tl.zeros((BLOCK_SIZE_H,), dtype=tl.float32)
    else:
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
    # quantize the softmax probabilities. E4M3 -> BF16 is exact, so this
    # reproduces the values the caller's standalone widening kernel produced.
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
        if KV_NVFP4:
            # K block scales are token-major linear within the page-head region;
            # see _fused_nvfp4_paged_scatter_kernel in msa_scatter.
            k_deq = _dequant_nvfp4_rows(
                tl.load(
                    k_ptr
                    + page * stride_k_blk
                    + pid_kh * stride_k_h
                    + off_n[:, None] * stride_k_pos
                    + off_dp[None, :] * stride_k_d,
                    mask=pos_mask[:, None],
                    other=0,
                ),
                tl.load(
                    k_sc_ptr
                    + page * stride_ksc_blk
                    + pid_kh * stride_ksc_h
                    + off_n[:, None] * SCALE_COLS
                    + off_s[None, :],
                    mask=pos_mask[:, None],
                    other=0,
                ),
                BLOCK_N=BLOCK_SIZE_K,
                D=BLOCK_SIZE_D,
                SCALE_COLS=SCALE_COLS,
                SF_VEC=BLOCK_SIZE_D // SCALE_COLS,
            )
            k = tl.trans(k_deq).to(q.dtype)
        else:
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
        if USE_LINEAR_SOFTMAX:
            alpha = tl.exp2(m_i - m_ij)
            acc_o = acc_o * alpha[:, None]
        else:
            acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]
        if KV_NVFP4:
            # V block scales use vLLM's 4x4 token-quad order rather than K's
            # linear one: [token // 4, scale_col, token % 4].
            v = _dequant_nvfp4_rows(
                tl.load(
                    v_ptr
                    + page * stride_v_blk
                    + pid_kh * stride_v_h
                    + off_n[:, None] * stride_v_pos
                    + off_dp[None, :] * stride_v_d,
                    mask=pos_mask[:, None],
                    other=0,
                ),
                tl.load(
                    v_sc_ptr
                    + page * stride_vsc_blk
                    + pid_kh * stride_vsc_h
                    + (off_n[:, None] // 4) * (4 * SCALE_COLS)
                    + 4 * off_s[None, :]
                    + (off_n[:, None] % 4),
                    mask=pos_mask[:, None],
                    other=0,
                ),
                BLOCK_N=BLOCK_SIZE_K,
                D=BLOCK_SIZE_D,
                SCALE_COLS=SCALE_COLS,
                SF_VEC=BLOCK_SIZE_D // SCALE_COLS,
            ).to(q.dtype)
        else:
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
        if USE_LINEAR_SOFTMAX:
            l_i = l_i * alpha + l_ij
        else:
            lse_i = m_ij + tl.log2(tl.exp2(lse_i - m_ij) + l_ij)
        m_i = m_ij

    # An empty chunk of an active row must store zero, or the merge hits 0 * NaN.
    if USE_LINEAR_SOFTMAX:
        scale = tl.where(l_i > 0.0, 1.0 / l_i, 0.0)
        lse_out = tl.where(l_i > 0.0, m_i + tl.log2(l_i), float("-inf"))
    else:
        scale = tl.where(lse_i > float("-inf"), tl.exp2(m_i - lse_i), 0.0)
        lse_out = lse_i
    if KV_NVFP4:
        # Applying V's scale per partial is exact: the merge is a weighted
        # average whose weights sum to one, so a constant survives it.
        scale = scale * v_global_scale
    acc_o = acc_o * scale[:, None]
    o_base = o_ptr + pid_c * stride_o_c + pid_b * stride_o_b + pid_h * stride_o_h
    tl.store(
        o_base + off_h[:, None] * stride_o_h + off_d[None, :] * stride_o_d,
        acc_o.to(o_ptr.dtype.element_ty),
        mask=hd_mask,
    )
    if not DIRECT_OUTPUT:
        lse_base = lse_ptr + pid_c * stride_l_c + pid_b * stride_l_b + pid_h * stride_l_h
        tl.store(lse_base + off_h * stride_l_h, lse_out, mask=h_mask)

    # After the stores, never before: this releases either the merge grid or,
    # for direct output, the next PDL-enabled consumer of the final tensor.
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


def _sm103_nvfp4_num_topk_chunks(
    *,
    total_q: int,
    num_kv_heads: int,
    gqa_group_size: int,
    max_topk: int,
    decode_query_len: int,
    capability: Optional[tuple[int, int]] = None,
) -> Optional[int]:
    """Select measured split-K factors for M3's SM103 DEP8 decode shape."""
    if capability is None:
        capability = torch.cuda.get_device_capability()
    if (
        capability != (10, 3)
        or num_kv_heads != 4
        or gqa_group_size != 16
        or max_topk != 16
        or decode_query_len != 4
        or total_q % decode_query_len != 0
    ):
        return None
    local_batch = total_q // decode_query_len
    return {
        2: 8,
        3: 8,
        4: 8,
        5: 16,
        6: 4,
        7: 4,
        8: 2,
        9: 2,
        10: 8,
        11: 4,
        12: 2,
        13: 2,
        14: 2,
    }.get(local_batch)


def _sm103_nvfp4_use_linear_softmax(
    *,
    total_q: int,
    num_kv_heads: int,
    gqa_group_size: int,
    max_topk: int,
    decode_query_len: int,
    capability: Optional[tuple[int, int]] = None,
) -> bool:
    """Use the measured lower-register recurrence on validated M3 shapes."""
    if capability is None:
        capability = torch.cuda.get_device_capability()
    if (
        capability != (10, 3)
        or num_kv_heads not in (1, 2, 4)
        or gqa_group_size != 16
        or max_topk != 16
        or decode_query_len != 4
        or total_q % decode_query_len != 0
    ):
        return False
    local_batch = total_q // decode_query_len
    # The TP4/DEP32 single-request graph is 8.9% faster with the original
    # log-domain recurrence; every other measured DEP8/16/32 bucket is neutral
    # or faster with the linear denominator.
    return not (num_kv_heads == 1 and local_batch == 1)


def _sm103_nvfp4_launch_options(
    *,
    total_q: int,
    num_kv_heads: int,
    gqa_group_size: int,
    max_topk: int,
    decode_query_len: int,
    capability: Optional[tuple[int, int]] = None,
) -> dict[str, int]:
    """Select measured SM103 native-NVFP4 launch geometry.

    The FP16 QK/PV path is register-sensitive. GB300 dense replay shows four
    warps/two stages wins through local batch 28, while two warps/one stage
    avoids the next residency cliff at local batches 29--32. Leave every
    unmeasured architecture, model geometry, and larger batch on Triton's
    established defaults.
    """
    if capability is None:
        capability = torch.cuda.get_device_capability()
    if (
        capability != (10, 3)
        or num_kv_heads != 4
        or gqa_group_size != 16
        or max_topk != 16
        or decode_query_len != 4
    ):
        return {}
    local_batch = total_q // decode_query_len
    if local_batch == 1 or local_batch in (2, 3, 5, 6, 7, 10, 11, 12):
        return {"num_warps": 4, "num_stages": 1}
    if local_batch == 4:
        return {"num_warps": 2, "num_stages": 1}
    if 8 <= local_batch <= 28:
        return {"num_warps": 4, "num_stages": 2}
    if 29 <= local_batch <= 32:
        return {"num_warps": 2, "num_stages": 1}
    return {}


def _sm103_nvfp4_merge_launch_options(
    *,
    total_q: int,
    num_kv_heads: int,
    gqa_group_size: int,
    max_topk: int,
    decode_query_len: int,
    capability: Optional[tuple[int, int]] = None,
) -> dict[str, int]:
    """Select measured one-warp merge launches for M3's DEP8 graphs."""
    if capability is None:
        capability = torch.cuda.get_device_capability()
    if (
        capability == (10, 3)
        and num_kv_heads == 4
        and gqa_group_size == 16
        and max_topk == 16
        and decode_query_len == 4
        and total_q % decode_query_len == 0
        and total_q // decode_query_len in (2, 4, 5, 6, 7, 10, 11)
    ):
        return {"num_warps": 1}
    return {}


def _check_nvfp4_inputs(
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    nvfp4_args: tuple[Optional[torch.Tensor], ...],
    *,
    head_dim: int,
    scale_cols: int,
) -> None:
    """Validate the packed-NVFP4 cache against what the kernel assumes.

    The kernel indexes block scales by a flat within-page offset and reshapes a
    [., SCALE_COLS] scale tile up to head_dim, so it needs contiguous scale rows
    and head_dim exactly BLOCK_SIZE_D. Both hold for MiniMax-M3; check them
    rather than let a mismatch read neighboring pages.
    """
    k_block_scale, v_block_scale, k_global_scale, v_global_scale = nvfp4_args
    if any(arg is None for arg in nvfp4_args):
        raise ValueError(
            "MiniMax-M3 NVFP4 sparse decode needs all four of k_block_scale, "
            "v_block_scale, k_global_scale and v_global_scale."
        )
    if head_dim != triton.next_power_of_2(head_dim):
        raise ValueError(f"NVFP4 sparse decode requires a power-of-two head_dim; got {head_dim}.")
    for name, data in (("k_paged", k_paged), ("v_paged", v_paged)):
        if int(data.shape[3]) * 2 != head_dim:
            raise ValueError(
                f"NVFP4 {name} must pack two elements per byte: expected "
                f"head_dim/2 ({head_dim // 2}) columns, got {int(data.shape[3])}."
            )
    for name, scale in (("k_block_scale", k_block_scale), ("v_block_scale", v_block_scale)):
        if tuple(scale.shape) != tuple(k_paged.shape[:3]) + (scale_cols,):
            raise ValueError(
                f"NVFP4 {name} must be [num_pages, num_kv_heads, page_size, "
                f"{scale_cols}] to match the cache; got {tuple(scale.shape)}."
            )
        if scale.stride(3) != 1 or scale.stride(2) != scale_cols:
            raise ValueError(
                f"NVFP4 {name} needs contiguous per-page scale bytes; got strides "
                f"{tuple(scale.stride())}."
            )
    for name, gs in (("k_global_scale", k_global_scale), ("v_global_scale", v_global_scale)):
        if gs.dtype != torch.float32 or gs.numel() != 1:
            raise ValueError(
                f"NVFP4 {name} must be a single-element FP32 tensor; got "
                f"{gs.numel()} element(s) of {gs.dtype}."
            )


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
    k_block_scale: Optional[torch.Tensor] = None,
    v_block_scale: Optional[torch.Tensor] = None,
    k_global_scale: Optional[torch.Tensor] = None,
    v_global_scale: Optional[torch.Tensor] = None,
    num_topk_chunks: Optional[int] = None,
) -> None:
    """Block-sparse GQA decode attention, written into output in place.

    q may be FP8, as the fused MiniMax-M3 producer emits it; the kernel widens
    it to output.dtype in-register rather than making the caller materialize a
    widened copy. The attention math is unchanged either way, since E4M3 widens
    exactly.

    kv_scale is an optional scalar dequantization factor for an FP8 cache;
    MiniMax-M3 stores unscaled E4M3, so it is normally None.

    Passing the four NVFP4 scale tensors selects the packed-NVFP4 cache instead:
    k_paged/v_paged then hold two E2M1 elements per byte, the block scales are
    paged E4M3 bytes in the layouts msa_scatter writes, and the two per-tensor
    dequant scales are single-element FP32 device tensors.

    num_topk_chunks overrides the split-K factor and exists for tests: the
    merged result must not depend on it.
    """
    total_q, num_heads, head_dim = q.shape
    num_kv_heads = int(k_paged.shape[1])
    check_decode_span_shape(
        "MiniMax-M3 Triton sparse decode", total_q, int(seq_lens.shape[0]), decode_query_len
    )
    if int(k_paged.shape[2]) != SPARSE_BLOCK_SIZE:
        raise ValueError(
            f"MiniMax-M3 sparse decode requires page_size={SPARSE_BLOCK_SIZE}; "
            f"got {int(k_paged.shape[2])}."
        )
    nvfp4_args = (k_block_scale, v_block_scale, k_global_scale, v_global_scale)
    kv_nvfp4 = any(arg is not None for arg in nvfp4_args)
    scale_cols = head_dim // NVFP4_SF_VEC_SIZE
    if kv_nvfp4:
        _check_nvfp4_inputs(k_paged, v_paged, nvfp4_args, head_dim=head_dim, scale_cols=scale_cols)
        # Two E2M1 elements per byte; Triton must see them unsigned so that the
        # high-nibble shift does not sign-extend.
        k_paged, v_paged = k_paged.view(torch.uint8), v_paged.view(torch.uint8)
        k_block_scale, v_block_scale = (
            k_block_scale.view(torch.uint8),
            v_block_scale.view(torch.uint8),
        )
    max_topk = int(topk_idx.shape[-1])
    gqa_group_size = num_heads // num_kv_heads
    use_scale = k_paged.dtype in _FP8_DTYPES and kv_scale is not None
    # Triton needs a real pointer even for the unused arguments.
    scale_arg = kv_scale if use_scale else output
    k_sc_arg, v_sc_arg = (k_block_scale, v_block_scale) if kv_nvfp4 else (output, output)
    k_gs_arg, v_gs_arg = (k_global_scale, v_global_scale) if kv_nvfp4 else (output, output)

    widen_q = q.dtype in _FP8_DTYPES
    if widen_q and output.dtype not in _TL_DTYPES:
        raise ValueError(
            f"MiniMax-M3 sparse decode cannot widen FP8 q into {output.dtype}; "
            f"supported compute dtypes are {sorted(d.__str__() for d in _TL_DTYPES)}."
        )
    # Packed NVFP4 dequantizes to FP16. The production producer supplies FP8
    # q, which widens exactly to FP16, so keep both dot operands there instead
    # of converting every K/V tile to BF16. Accumulation remains FP32.
    if widen_q:
        compute_dtype = tl.float16 if kv_nvfp4 else _TL_DTYPES[output.dtype]
    else:
        compute_dtype = tl.float32

    if num_topk_chunks is None:
        measured_chunks = (
            _sm103_nvfp4_num_topk_chunks(
                total_q=total_q,
                num_kv_heads=num_kv_heads,
                gqa_group_size=gqa_group_size,
                max_topk=max_topk,
                decode_query_len=decode_query_len,
            )
            if kv_nvfp4
            else None
        )
        num_topk_chunks = (
            measured_chunks
            if measured_chunks is not None
            else resolve_num_topk_chunks(total_q, num_kv_heads, max_topk)
        )
    elif num_topk_chunks & (num_topk_chunks - 1):
        raise ValueError(f"num_topk_chunks must be a power of two; got {num_topk_chunks}.")

    direct_output = num_topk_chunks == 1
    if direct_output:
        # With one chunk, acc_o is already normalized and final. Point the
        # producer at the caller's output and compile away the LSE store.
        # Keeping a real dummy pointer preserves Triton's pointer contract.
        o_arg = output
        lse_arg = output
        o_strides = (0, output.stride(0), output.stride(1), output.stride(2))
        lse_strides = (0, 0, 0)
    else:
        # Persistent arena rather than torch.empty, so the partials keep one
        # address across CUDA graph replays. FP32 keeps the split-K factor from
        # perturbing the merged result.
        reserve = torch.cuda.is_current_stream_capturing()
        o_arg = get_memory_buffers().get_buffer(
            [num_topk_chunks, total_q, num_heads, head_dim],
            torch.float32,
            buffer_name="m3_sparse_decode_o_partial",
            reserve_buffer=reserve,
        )
        lse_arg = get_memory_buffers().get_buffer(
            [num_topk_chunks, total_q, num_heads],
            torch.float32,
            buffer_name="m3_sparse_decode_lse_partial",
            reserve_buffer=reserve,
        )
        o_strides = tuple(o_arg.stride())
        lse_strides = tuple(lse_arg.stride())

    use_pdl = _pdl_enabled()
    pdl_launch = {"launch_pdl": True} if use_pdl else {}
    use_linear_softmax = kv_nvfp4 and _sm103_nvfp4_use_linear_softmax(
        total_q=total_q,
        num_kv_heads=num_kv_heads,
        gqa_group_size=gqa_group_size,
        max_topk=max_topk,
        decode_query_len=decode_query_len,
    )
    launch_options = (
        _sm103_nvfp4_launch_options(
            total_q=total_q,
            num_kv_heads=num_kv_heads,
            gqa_group_size=gqa_group_size,
            max_topk=max_topk,
            decode_query_len=decode_query_len,
        )
        if kv_nvfp4
        else {}
    )

    _gqa_sparse_decode_kernel[(total_q * num_topk_chunks, num_kv_heads)](
        q,
        k_paged,
        v_paged,
        scale_arg,
        k_sc_arg,
        v_sc_arg,
        k_gs_arg,
        v_gs_arg,
        topk_idx,
        o_arg,
        lse_arg,
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
        k_sc_arg.stride(0),
        k_sc_arg.stride(1),
        v_sc_arg.stride(0),
        v_sc_arg.stride(1),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        *o_strides,
        *lse_strides,
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        NUM_TOPK_CHUNKS=num_topk_chunks,
        USE_SCALE=use_scale,
        WIDEN_Q=widen_q,
        COMPUTE_DTYPE=compute_dtype,
        USE_PDL=use_pdl,
        KV_NVFP4=kv_nvfp4,
        USE_LINEAR_SOFTMAX=use_linear_softmax,
        SCALE_COLS=scale_cols,
        DIRECT_OUTPUT=direct_output,
        **pdl_launch,
        **launch_options,
    )
    if not direct_output:
        merge_launch_options = (
            _sm103_nvfp4_merge_launch_options(
                total_q=total_q,
                num_kv_heads=num_kv_heads,
                gqa_group_size=gqa_group_size,
                max_topk=max_topk,
                decode_query_len=decode_query_len,
            )
            if kv_nvfp4
            else {}
        )
        _merge_topk_attn_out_kernel[(total_q, num_heads)](
            o_arg,
            lse_arg,
            output,
            head_dim,
            *o_strides,
            *lse_strides,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            NUM_TOPK_CHUNKS=num_topk_chunks,
            USE_PDL=use_pdl,
            **pdl_launch,
            **merge_launch_options,
        )


__all__ = [
    "NVFP4_SF_VEC_SIZE",
    "SPARSE_BLOCK_SIZE",
    "_sm103_nvfp4_merge_launch_options",
    "_sm103_nvfp4_num_topk_chunks",
    "_sm103_nvfp4_use_linear_softmax",
    "minimax_m3_sparse_attn_decode",
    "resolve_num_topk_chunks",
]
