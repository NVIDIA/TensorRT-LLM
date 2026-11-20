# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned H3 grouped producer with exact duplicate-half RoPE load reuse."""

import triton
import triton.language as tl

from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4.kernels.fused_fp8_qk import (
    _div_rn,
    _load_amax,
    _packed_out_offs,
    _rotate_pair,
    _row_valid,
    _scale_from_amax,
)


@triton.jit
def half_group_vectors(cos_ptr, sin_ptr, w_ptr, t, c0, ROT: tl.constexpr, HALF: tl.constexpr):
    """Load BF16-rounded rotary values and norm weights for both column groups.

    Group ``c0`` (first half, ``sign = -1``) and its partner group ``c0 + HALF`` (``sign = +1``).
    """
    c1 = c0 + HALF
    cos_a = tl.load(cos_ptr + t * ROT + c0).to(tl.float32).to(tl.bfloat16).to(tl.float32)
    sin_a = tl.load(sin_ptr + t * ROT + c0).to(tl.float32).to(tl.bfloat16).to(tl.float32)
    cos_b = cos_a
    sin_b = sin_a
    # (sign * sin) of the shipping kernel: -1.0 * sin for the first half, 1.0 * sin for the second
    ss_a = -1.0 * sin_a
    ss_b = 1.0 * sin_b
    w_a = tl.load(w_ptr + c0).to(tl.float32)
    w_b = tl.load(w_ptr + c1).to(tl.float32)
    return cos_a, ss_a, w_a, cos_b, ss_b, w_b


@triton.jit
def half_rope_tensor_cast(
    x_ptr,
    w_ptr,
    cos_ptr,
    sin_ptr,
    amax_ptr,
    descale_ptr,
    out_ptr,
    sat_ptr,
    S,
    eps,
    STATIC: tl.constexpr,
    SAT_COUNT: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    ROT: tl.constexpr,
    BLOCK_H: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    PACKED: tl.constexpr,
    HL: tl.constexpr,
    WHICH: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    TAIL: tl.constexpr,
    AMAX_N: tl.constexpr = 1,
    AMAX_STRIDE: tl.constexpr = 1,
    PLANES: tl.constexpr = 3,
    v_ptr=None,
    vamax_ptr=None,
    len_ptr=None,
    VAMAX: tl.constexpr = False,
    HAS_LEN: tl.constexpr = False,
):
    HALF: tl.constexpr = ROT // 2
    r = tl.program_id(0)
    t = r % S
    d = tl.arange(0, D)
    hh = tl.arange(0, BLOCK_H)
    row = x_ptr + r.to(tl.int64) * ROW_STRIDE
    if VAMAX:
        # §692 F.1 (a): V's per-tensor amax folded into this pass (H3_FP8_VAMAX_FOLD=1): the same
        # loads, |x| and max as _tensor_amax_kernel's identity path over V's columns of this row, so
        # the max (exact, order-free) is bit-identical; one launch fewer per block.
        vrow = v_ptr + r.to(tl.int64) * ROW_STRIDE
        vm = tl.zeros((BLOCK_H, D), dtype=tl.float32)
    out_row = out_ptr + r.to(tl.int64) * (H * D)
    inv_d = 1.0 / D
    if STATIC:
        s = tl.load(descale_ptr)
    else:
        s = _scale_from_amax(_load_amax(amax_ptr, AMAX_N, AMAX_STRIDE))
        if r == 0:
            tl.store(descale_ptr, s)
    f = _div_rn(1.0, s)
    g0 = tl.arange(0, C0)
    cos_a0, ss_a0, w_a0, cos_b0, ss_b0, w_b0 = half_group_vectors(
        cos_ptr, sin_ptr, w_ptr, t, g0, ROT, HALF
    )
    if C1 > 0:
        g1 = C0 + tl.arange(0, C1)
        cos_a1, ss_a1, w_a1, cos_b1, ss_b1, w_b1 = half_group_vectors(
            cos_ptr, sin_ptr, w_ptr, t, g1, ROT, HALF
        )
    if TAIL > 0:
        gt = ROT + tl.arange(0, TAIL)
        w_t = tl.load(w_ptr + gt).to(tl.float32)
    if SAT_COUNT:
        nsat = tl.zeros((BLOCK_H,), dtype=tl.int32)
    for h0 in range(0, H, BLOCK_H):
        h = h0 + hh
        mask = (h < H)[:, None] & (d < D)[None, :]
        offs = h[:, None] * D + d[None, :]
        x = tl.load(row + offs, mask=mask, other=0.0).to(tl.float32)
        if VAMAX:
            vm = tl.maximum(vm, tl.abs(tl.load(vrow + offs, mask=mask, other=0.0).to(tl.float32)))
        ss = tl.sum(x * x, axis=1)
        inv = tl.rsqrt(ss * inv_d + eps)
        hm = (h < H)[:, None]
        base = row + h[:, None] * D
        # group 0 and its partner
        ma = hm & (g0 < C0)[None, :]
        x_a = tl.load(base + g0[None, :], mask=ma, other=0.0).to(tl.float32)
        x_b = tl.load(base + (HALF + g0)[None, :], mask=ma, other=0.0).to(tl.float32)
        r_a, r_b = _rotate_pair(x_a, x_b, inv, w_a0, w_b0, cos_a0, ss_a0, cos_b0, ss_b0)
        sc_a = r_a * f
        sc_b = r_b * f
        if SAT_COUNT:
            nsat += tl.sum(
                (tl.abs(sc_a) > 448.0).to(tl.int32) + (tl.abs(sc_b) > 448.0).to(tl.int32), axis=1
            )
        if PACKED:
            tl.store(
                out_ptr + _packed_out_offs(h, g0, t, S, WHICH, HL, D, PLANES),
                sc_a.to(tl.float8e4nv),
                mask=ma,
            )
            tl.store(
                out_ptr + _packed_out_offs(h, HALF + g0, t, S, WHICH, HL, D, PLANES),
                sc_b.to(tl.float8e4nv),
                mask=ma,
            )
        else:
            tl.store(out_row + h[:, None] * D + g0[None, :], sc_a.to(tl.float8e4nv), mask=ma)
            tl.store(
                out_row + h[:, None] * D + (HALF + g0)[None, :], sc_b.to(tl.float8e4nv), mask=ma
            )
        if C1 > 0:
            mb = hm & (g1 < D)[None, :]
            x_a = tl.load(base + g1[None, :], mask=mb, other=0.0).to(tl.float32)
            x_b = tl.load(base + (HALF + g1)[None, :], mask=mb, other=0.0).to(tl.float32)
            r_a, r_b = _rotate_pair(x_a, x_b, inv, w_a1, w_b1, cos_a1, ss_a1, cos_b1, ss_b1)
            sc_a = r_a * f
            sc_b = r_b * f
            if SAT_COUNT:
                nsat += tl.sum(
                    (tl.abs(sc_a) > 448.0).to(tl.int32) + (tl.abs(sc_b) > 448.0).to(tl.int32),
                    axis=1,
                )
            if PACKED:
                tl.store(
                    out_ptr + _packed_out_offs(h, g1, t, S, WHICH, HL, D, PLANES),
                    sc_a.to(tl.float8e4nv),
                    mask=mb,
                )
                tl.store(
                    out_ptr + _packed_out_offs(h, HALF + g1, t, S, WHICH, HL, D, PLANES),
                    sc_b.to(tl.float8e4nv),
                    mask=mb,
                )
            else:
                tl.store(out_row + h[:, None] * D + g1[None, :], sc_a.to(tl.float8e4nv), mask=mb)
                tl.store(
                    out_row + h[:, None] * D + (HALF + g1)[None, :], sc_b.to(tl.float8e4nv), mask=mb
                )
        if TAIL > 0:
            mc = hm & (gt < D)[None, :]
            x_t = tl.load(base + gt[None, :], mask=mc, other=0.0).to(tl.float32)
            n_t = (x_t * inv[:, None]).to(tl.bfloat16).to(tl.float32) * w_t[None, :]
            sc_t = n_t.to(tl.bfloat16).to(tl.float32) * f
            if SAT_COUNT:
                nsat += tl.sum((tl.abs(sc_t) > 448.0).to(tl.int32), axis=1)
            if PACKED:
                tl.store(
                    out_ptr + _packed_out_offs(h, gt, t, S, WHICH, HL, D, PLANES),
                    sc_t.to(tl.float8e4nv),
                    mask=mc,
                )
            else:
                tl.store(out_row + h[:, None] * D + gt[None, :], sc_t.to(tl.float8e4nv), mask=mc)
    if SAT_COUNT:
        tl.atomic_add(sat_ptr, tl.sum(nsat, axis=0))
    if VAMAX:
        vvalid = _row_valid(len_ptr, r // S, t, HAS_LEN)
        tl.atomic_max(vamax_ptr, tl.max(tl.max(vm, axis=1), axis=0) * vvalid)
