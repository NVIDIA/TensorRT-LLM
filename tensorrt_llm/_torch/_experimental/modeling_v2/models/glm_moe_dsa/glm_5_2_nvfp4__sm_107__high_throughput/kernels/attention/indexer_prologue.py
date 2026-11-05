# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused FP8 indexer preparation and optional direct indexer-cache write."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

FUSED_PROLOGUE_ENABLED = os.environ.get("TRTLLM_DSA_INDEXER_FUSED_PROLOGUE", "1").strip() != "0"
_PDL_ENABLED = os.environ.get("TRTLLM_ENABLE_PDL", "0").strip() == "1"


@triton.jit
def _rcp_approx_ftz(x):
    # fused_cat_fp8 uses the MUFU reciprocal (rcp.approx.ftz.f32), not IEEE 1/x.
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        "=r,r",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp8_row_scale(amax, UE8M0: tl.constexpr):
    """fused_cat_fp8 per-row scale: amax clamped at 1e-12, / 448, optional UE8M0 round-up."""
    amax = tl.maximum(amax, 1.0e-12)
    ratio = amax * 0.002232142857142857  # 1.0f / 448.0f
    if UE8M0:
        bits = ratio.to(tl.uint32, bitcast=True)
        mant = bits & 0x007FFFFF
        exp_bits = bits & 0x7F800000
        exp_bits = tl.where(mant != 0, exp_bits + 0x00800000, exp_bits)
        scale = exp_bits.to(tl.float32, bitcast=True)
    else:
        scale = ratio
    return scale


@triton.jit
def _rope_terms(d, ROPE_DIM: tl.constexpr, INTERLEAVE: tl.constexpr):
    """Partner column, cos/sin column and sign for column ``d`` (flashinfer layout)."""
    HALF: tl.constexpr = ROPE_DIM // 2
    is_rope = d < ROPE_DIM
    if INTERLEAVE:
        # GPT-J style: pairs (2i, 2i + 1) share cos/sin column i; even lanes subtract.
        partner = tl.where(is_rope, d ^ 1, d)
        cs_col = d // 2
        neg = (d % 2) == 0
    else:
        # NeoX style: halves [0, HALF) and [HALF, ROPE_DIM); the first half subtracts.
        partner = tl.where(d < HALF, d + HALF, tl.where(is_rope, d - HALF, d))
        cs_col = d % HALF
        neg = d < HALF
    cs_col = tl.where(is_rope, cs_col, 0)
    return is_rope, partner, cs_col, neg


@triton.jit
def _apply_rope(x, xp, c, s, neg):
    # flashinfer: vec[i] = vec[i] * cos[i] + (neg ? -perm[i] : perm[i]) * sin[i];
    # nvcc contracts this to fma(vec, cos, (+-perm) * sin).
    t = tl.where(neg, -xp, xp) * s
    return tl.fma(x, c, t)


@triton.jit
def _unravel_offset(flat, d1, d2, d3, s0, s1, s2, s3):
    """indexerKCacheScatter's flatIndexToMemoryOffset over a [d0, d1, d2, d3] strided view."""
    i3 = flat % d3
    flat = flat // d3
    i2 = flat % d2
    flat = flat // d2
    i1 = flat % d1
    i0 = flat // d1
    return i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3


@triton.jit
def _dsa_indexer_prologue_fused_kernel(
    # inputs
    k_ptr,  # [N, D] pre-norm indexer K (bf16 or fp32; rounded to bf16 first)
    k_stride,
    ln_w_ptr,  # [D] LayerNorm weight
    ln_b_ptr,  # [D] LayerNorm bias
    q_ptr,  # [N, H, D] bf16 (wq_b output view, unit inner stride)
    q_stride_tok,
    q_stride_head,
    w_ptr,  # [N, H] weights_proj output (bf16 or fp32)
    w_stride,
    pos_ptr,  # [N] positions
    cos_sin_ptr,  # [max_pos, ROPE_DIM] fp32: [cos(0..HALF) | sin(0..HALF)]
    cos_sin_stride,
    # outputs
    q_fp8_ptr,  # [N, H, D] fp8 e4m3 (contiguous)
    q_scale_ptr,  # [N, H] fp32
    k_fp8_ptr,  # [N, D] fp8 e4m3 (contiguous)
    k_scale_ptr,  # [N] fp32
    w_out_ptr,  # [N, H] fp32
    # indexer K cache (uint8 [d0, d1, d2, d3] strided view) + slot mappings
    k_cache_ptr,
    slot_fp8_ptr,
    slot_scale_ptr,
    n_write,
    cache_d1,
    cache_d2,
    cache_d3,
    cache_s0,
    cache_s1,
    cache_s2,
    cache_s3,
    # scalars
    eps,
    weight_scale,
    H: tl.constexpr,
    D: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HEADS_PER_PROG: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    UE8M0: tl.constexpr,
    WRITE_CACHE: tl.constexpr,
    USE_PDL: tl.constexpr,
    LAUNCH_DEPENDENTS: tl.constexpr,
    PREWAIT: tl.constexpr,
    WRITE_Q: tl.constexpr,
):
    tok = tl.program_id(0)
    hblk = tl.program_id(1)
    HALF: tl.constexpr = ROPE_DIM // 2
    if PREWAIT:
        # Positions, cos/sin, the K LayerNorm and the slot mappings: written before the
        # kernels this one depends on run.
        tok64 = tok.to(tl.int64)
        d = tl.arange(0, D)
        is_rope, partner, cs_col, neg = _rope_terms(d, ROPE_DIM, INTERLEAVE)
        k_prog = hblk == 0
        pos = tl.load(pos_ptr + tok).to(tl.int64)
        cs_row = cos_sin_ptr + pos * cos_sin_stride
        cos_v = tl.load(cs_row + cs_col, mask=is_rope, other=1.0)
        sin_v = tl.load(cs_row + HALF + cs_col, mask=is_rope, other=0.0)
        lw = tl.load(ln_w_ptr + d, mask=k_prog, other=0.0).to(tl.float32)
        lb = tl.load(ln_b_ptr + d, mask=k_prog, other=0.0).to(tl.float32)
        lwp = tl.load(ln_w_ptr + partner, mask=k_prog, other=0.0).to(tl.float32)
        lbp = tl.load(ln_b_ptr + partner, mask=k_prog, other=0.0).to(tl.float32)
        if WRITE_CACHE:
            slot = tl.load(slot_fp8_ptr + tok, mask=k_prog & (tok < n_write), other=-1)
            slot_s = tl.load(slot_scale_ptr + tok, mask=k_prog & (tok < n_write), other=-1)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        if LAUNCH_DEPENDENTS:
            tl.extra.cuda.gdc_launch_dependents()

    if not PREWAIT:
        tok64 = tok.to(tl.int64)
        d = tl.arange(0, D)
        is_rope, partner, cs_col, neg = _rope_terms(d, ROPE_DIM, INTERLEAVE)

        pos = tl.load(pos_ptr + tok).to(tl.int64)
        cs_row = cos_sin_ptr + pos * cos_sin_stride
        cos_v = tl.load(cs_row + cs_col, mask=is_rope, other=1.0)
        sin_v = tl.load(cs_row + HALF + cs_col, mask=is_rope, other=0.0)

    if WRITE_Q:
        # ---------------- Q: HEADS_PER_PROG heads of this token ----------------
        h = hblk * HEADS_PER_PROG + tl.arange(0, HEADS_PER_PROG)
        q_row = q_ptr + tok64 * q_stride_tok + h[:, None].to(tl.int64) * q_stride_head
        qx = tl.load(q_row + d[None, :]).to(tl.float32)
        qp = tl.load(q_row + partner[None, :]).to(tl.float32)
        if PREWAIT:
            # Every load from the kernels this one waits for, before the first store.
            w = tl.load(w_ptr + tok64 * w_stride + h).to(tl.float32)
            k_row = k_ptr + tok64 * k_stride
            kx = tl.load(k_row + d, mask=k_prog, other=0.0).to(tl.bfloat16).to(tl.float32)
            kxp = tl.load(k_row + partner, mask=k_prog, other=0.0).to(tl.bfloat16).to(tl.float32)
        q_rot = _apply_rope(qx, qp, cos_v[None, :], sin_v[None, :], neg[None, :])
        qv = tl.where(is_rope[None, :], q_rot, qx).to(tl.bfloat16).to(tl.float32)
        q_amax = tl.max(tl.abs(qv), axis=1)
        q_scale = _fp8_row_scale(q_amax, UE8M0)
        q_inv = _rcp_approx_ftz(q_scale)
        q8 = (qv * q_inv[:, None]).to(tl.float8e4nv)
        qo = (tok64 * H + h[:, None]) * D + d[None, :]
        tl.store(q_fp8_ptr + qo, q8)
        tl.store(q_scale_ptr + tok64 * H + h, q_scale)

        # weights * q_scale * (softmax_scale * n_heads^-0.5), same op order as _scale().
        if not PREWAIT:
            w = tl.load(w_ptr + tok64 * w_stride + h).to(tl.float32)
        tl.store(w_out_ptr + tok64 * H + h, (w * q_scale) * weight_scale)
    elif PREWAIT:
        # K only: the K row, right after the wait.
        k_row = k_ptr + tok64 * k_stride
        kx = tl.load(k_row + d, mask=k_prog, other=0.0).to(tl.bfloat16).to(tl.float32)
        kxp = tl.load(k_row + partner, mask=k_prog, other=0.0).to(tl.bfloat16).to(tl.float32)

    # ---------------- K: once per token ----------------
    if hblk == 0:
        if not PREWAIT:
            k_row = k_ptr + tok64 * k_stride
            kx = tl.load(k_row + d).to(tl.bfloat16).to(tl.float32)
            kxp = tl.load(k_row + partner).to(tl.bfloat16).to(tl.float32)
        mean = tl.sum(kx, axis=0) / D
        kc = kx - mean
        var = tl.sum(kc * kc, axis=0) / D
        rstd = tl.math.rsqrt(var + eps)
        if not PREWAIT:
            lw = tl.load(ln_w_ptr + d).to(tl.float32)
            lb = tl.load(ln_b_ptr + d).to(tl.float32)
            lwp = tl.load(ln_w_ptr + partner).to(tl.float32)
            lbp = tl.load(ln_b_ptr + partner).to(tl.float32)
        kn = (kc * rstd * lw + lb).to(tl.bfloat16).to(tl.float32)
        knp = ((kxp - mean) * rstd * lwp + lbp).to(tl.bfloat16).to(tl.float32)
        k_rot = _apply_rope(kn, knp, cos_v, sin_v, neg)
        kv = tl.where(is_rope, k_rot, kn).to(tl.bfloat16).to(tl.float32)
        k_amax = tl.max(tl.abs(kv), axis=0)
        k_scale = _fp8_row_scale(k_amax, UE8M0)
        # inline asm needs a block operand: broadcast the row scale over D.
        k_inv = _rcp_approx_ftz(k_scale + tl.zeros([D], dtype=tl.float32))
        k8 = (kv * k_inv).to(tl.float8e4nv)
        tl.store(k_fp8_ptr + tok64 * D + d, k8)
        tl.store(k_scale_ptr + tok64, k_scale)

        if WRITE_CACHE:
            if tok < n_write:
                if not PREWAIT:
                    slot = tl.load(slot_fp8_ptr + tok)
                    slot_s = tl.load(slot_scale_ptr + tok)
                if (slot >= 0) & (slot_s >= 0):
                    # 4-byte chunks, each chunk base unravelled (as the scatter kernel).
                    chunk = slot + (d // 4) * 4
                    dst = _unravel_offset(
                        chunk, cache_d1, cache_d2, cache_d3, cache_s0, cache_s1, cache_s2, cache_s3
                    ) + (d % 4)
                    tl.store(k_cache_ptr + dst, k8.to(tl.uint8, bitcast=True))
                    b = tl.arange(0, 4)
                    sbits = k_scale.to(tl.int32, bitcast=True)
                    sbytes = ((sbits >> (b * 8)) & 0xFF).to(tl.uint8)
                    dst_s = _unravel_offset(
                        slot_s, cache_d1, cache_d2, cache_d3, cache_s0, cache_s1, cache_s2, cache_s3
                    )
                    tl.store(k_cache_ptr + dst_s + b, sbytes)


def _heads_per_prog(n_heads: int) -> int:
    for hp in (8, 4, 2, 1):
        if n_heads % hp == 0:
            return hp
    return 1


def dsa_indexer_prologue_fused(
    indexer_k: torch.Tensor,
    weights: torch.Tensor,
    q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    eps: float,
    weight_scale: float,
    rope_dim: int,
    interleave: bool,
    use_ue8m0: bool,
    k_cache: Optional[torch.Tensor] = None,
    slot_mapping_fp8: Optional[torch.Tensor] = None,
    slot_mapping_scale: Optional[torch.Tensor] = None,
    num_cache_tokens: int = 0,
    launch_dependents: bool = False,
    prewait: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-launch indexer prologue.

    Args:
        indexer_k: [N, D] pre-norm indexer K (bf16, or fp32 which is rounded to bf16).
        weights: [N, H] weights_proj output (bf16 or fp32), unit inner stride.
        q: [N, H, D] bf16 indexer Q (wq_b output), unit inner stride.
        positions: [N] (any int dtype) rope positions.
        cos_sin: [max_pos, rope_dim] fp32 flashinfer cos/sin cache ([cos | sin]).
        k_cache / slot_mapping_*: when given, the first ``num_cache_tokens`` K rows
            are also written into the indexer K cache (same layout and slot rule
            as ``indexer_k_cache_scatter_op``).
        launch_dependents: with PDL, let the dependent kernel launch right after
            this kernel's griddepcontrol.wait. Only for a dependent that reads
            this kernel's outputs after its own wait.
        prewait: with PDL, load the positions, the cos/sin rows, the LayerNorm
            weight and bias and the slot mappings before this kernel's
            griddepcontrol.wait. Only when the positions and slot mappings were
            written before the kernels this one depends on started.

    Returns (q_fp8 [N, H, D], k_fp8 [N, D], k_scale [N, 1], weights [N, H] fp32,
    q_scale [N, H, 1]) -- the ``pre_indexer_proj`` FP8 contract.
    """
    n_tok, n_heads, head_dim = q.shape
    assert head_dim == 128 and indexer_k.shape == (n_tok, head_dim)
    assert q.stride(-1) == 1 and indexer_k.stride(-1) == 1 and weights.stride(-1) == 1
    assert cos_sin.dtype == torch.float32 and cos_sin.stride(-1) == 1
    dev = q.device
    q_fp8 = torch.empty((n_tok, n_heads, head_dim), dtype=torch.float8_e4m3fn, device=dev)
    q_scale = torch.empty((n_tok, n_heads, 1), dtype=torch.float32, device=dev)
    k_fp8 = torch.empty((n_tok, head_dim), dtype=torch.float8_e4m3fn, device=dev)
    k_scale = torch.empty((n_tok, 1), dtype=torch.float32, device=dev)
    w_out = torch.empty((n_tok, n_heads), dtype=torch.float32, device=dev)
    if n_tok == 0:
        return q_fp8, k_fp8, k_scale, w_out, q_scale

    write_cache = k_cache is not None and num_cache_tokens > 0
    if write_cache:
        assert k_cache.dim() == 4 and k_cache.element_size() == 1
        cache_u8 = k_cache.view(torch.uint8)
        cd = cache_u8.shape
        cs = cache_u8.stride()
        cache_args = (
            cache_u8,
            slot_mapping_fp8,
            slot_mapping_scale,
            int(num_cache_tokens),
            cd[1],
            cd[2],
            cd[3],
            cs[0],
            cs[1],
            cs[2],
            cs[3],
        )
    else:
        cache_args = (k_fp8, k_scale, k_scale, 0, 1, 1, 1, 0, 0, 0, 0)

    hp = _heads_per_prog(n_heads)
    grid = (n_tok, n_heads // hp)
    _dsa_indexer_prologue_fused_kernel[grid](
        indexer_k,
        indexer_k.stride(0),
        ln_weight,
        ln_bias,
        q,
        q.stride(0),
        q.stride(1),
        weights,
        weights.stride(0),
        positions,
        cos_sin,
        cos_sin.stride(0),
        q_fp8,
        q_scale,
        k_fp8,
        k_scale,
        w_out,
        *cache_args,
        float(eps),
        float(weight_scale),
        H=n_heads,
        D=head_dim,
        ROPE_DIM=rope_dim,
        HEADS_PER_PROG=hp,
        INTERLEAVE=bool(interleave),
        UE8M0=bool(use_ue8m0),
        WRITE_CACHE=write_cache,
        USE_PDL=_PDL_ENABLED,
        LAUNCH_DEPENDENTS=_PDL_ENABLED and launch_dependents,
        PREWAIT=_PDL_ENABLED and prewait,
        WRITE_Q=True,
        num_warps=4,
        launch_pdl=_PDL_ENABLED,
    )
    return q_fp8, k_fp8, k_scale, w_out, q_scale
