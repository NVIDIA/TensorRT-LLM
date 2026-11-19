# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 MoE routing top-16 (SM100/SM103/SM107).

:func:`k31_route_topk` (the launcher at the end of this file) computes the top-16 experts and their
BF16 weights of ``1 <= M <= 2048`` rows of FP32 router logits ``[M, 896]`` on a grid of independent
CTAs (:class:`RouteWideKernel`), bit-identical to ``noaux_tc_op`` followed by the cast of its FP32
weights to BF16. :class:`RouteTailKernel` holds the selection code that kernel shares with the MoE
prologue kernels (``moe_prologue``).

* Top-k, bit-identical to ``noaux_tc_op`` on the same logits (selection, order, weights). One warp
  routes one token, lane ``l`` owning experts ``32 j + l``. Each lane forms the order-preserving
  bits of ``sigmoid(logit) + bias`` (the high word of ``noaux_tc_op``'s 64-bit key, whose low word
  ``65535 - expert`` breaks ties toward smaller experts). The 16th largest lane maximum is a lower
  bound of the token's 16th largest score. Every expert at or above it (usually about 20 per
  token) is a candidate: the warp compacts them into shared memory and ranks them by their full
  keys. If the warp has more than 64 candidates (heavy ties), it instead runs 16 exact arg-max
  rounds over all 896 keys. The weights are the ``sigmoid`` scores renormalized like
  ``noaux_tc_op``.

The kernels keep no state between launches, so a captured CUDA graph replays them as is.
"""

from __future__ import annotations

import functools
import struct

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .route_ptx import (
    _EPS_F64,
    NUM_EXPERTS,
    TOP_K,
    _asm,
    _div_rn_f32,
    _fv,
    _iv,
    _lds_b32,
    _lds_f32,
    _lv,
    _renormalize,
    _st_global_b32,
    _sts_b32,
    _sts_v4_f32,
    _unit_scale_fast_path,
    _unpack,
    _warp_inclusive_sum,
    _warp_sum_f32,
)

#: Largest token count of the routing (:class:`RouteWideKernel`).
WIDE_MAX_M = 2048
#: Expert slots per lane: 7 blocks of 128 experts x 4 consecutive experts.
_BLOCKS = NUM_EXPERTS // 128
_SLOTS = _BLOCKS * 4
#: Bytes of one compacted candidate: ``{65535 - expert, key high word, sigmoid bits, 0}``.
_CAND_BYTES = 16

assert NUM_EXPERTS % 128 == 0


# ---------------------------------------------------------------------------
# Inline PTX helpers
# ---------------------------------------------------------------------------


@dsl_user_op
def _ld_v4_f32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


@dsl_user_op
def _ld_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(), [_lv(addr, loc=loc, ip=ip)], "ld.global.f32 $0, [$1];", "=f,l", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _lds_v4_f32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_iv(addr, loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


@dsl_user_op
def _bar_sync_named(barrier_id, num_threads: int, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(barrier_id, loc=loc, ip=ip)],
        f"bar.sync $0, {num_threads};",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_nc_v4_f32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [_lv(addr, loc=loc, ip=ip)],
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


@dsl_user_op
def _score_bits(score, *, loc=None, ip=None):
    """Order-preserving bits of an FP32 score (the high word of ``noaux_tc``'s key)."""
    return Int32(
        _asm(
            T.i32(),
            [_fv(score, loc=loc, ip=ip)],
            "{\n.reg .b32 b, m;\nmov.b32 b, $1;\nshr.s32 m, b, 31;\n"
            "or.b32 m, m, 0x80000000;\nxor.b32 $0, b, m;\n}",
            "=r,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _max_u32(a, b, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(a, loc=loc, ip=ip), _iv(b, loc=loc, ip=ip)],
            "max.u32 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _warp_16th_largest_u32(value, lane, *, loc=None, ip=None):
    """16th largest of the 32 lane values (unsigned), in every lane.

    Bitonic-sorts the lower 16 lanes descending and the upper 16 ascending; the pairwise maxima
    of lanes ``i`` and ``i + 16`` are then the 16 largest values, whose minimum is the answer.
    """
    body = "{\n.reg .b32 x, o, mx, mn, t, u;\n.reg .pred pj, pk, km;\nmov.b32 x, $1;\n"
    k = 2
    while k <= 16:
        j = k // 2
        while j >= 1:
            body += f"shfl.sync.bfly.b32 o, x, {j}, 0x1f, 0xffffffff;\n"
            body += "max.u32 mx, x, o;\nmin.u32 mn, x, o;\n"
            # Keep the maximum iff the lane is the lower one of its pair in a descending block.
            body += f"and.b32 t, $2, {j};\nsetp.eq.u32 pj, t, 0;\n"
            body += f"and.b32 u, $2, {k};\nsetp.eq.u32 pk, u, 0;\nxor.pred km, pj, pk;\n"
            body += "selp.b32 x, mn, mx, km;\n"
            j //= 2
        k *= 2
    body += "shfl.sync.bfly.b32 o, x, 16, 0x1f, 0xffffffff;\nmax.u32 x, x, o;\n"
    body += "redux.sync.min.u32 $0, x, 0xffffffff;\n}"
    return Int32(
        _asm(
            T.i32(),
            [_iv(value, loc=loc, ip=ip), _iv(lane, loc=loc, ip=ip)],
            body,
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _count_ge_u32(values, threshold, *, loc=None, ip=None):
    """Number of ``values`` (unsigned) at or above ``threshold``."""
    n = len(values)
    body = "{\n.reg .pred p;\nmov.u32 $0, 0;\n"
    for i in range(n):
        body += f"setp.ge.u32 p, ${2 + i}, $1;\n@p add.u32 $0, $0, 1;\n"
    body += "}"
    return Int32(
        _asm(
            T.i32(),
            [_iv(threshold, loc=loc, ip=ip)] + [_iv(v, loc=loc, ip=ip) for v in values],
            body,
            "=&r,r" + ",r" * n,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _store_candidates(addr, his, los, sigs, threshold, *, loc=None, ip=None):
    """Append the slots with ``hi >= threshold`` as 16-byte records at ``addr``; returns the end.

    Record: ``{65535 - expert, hi, sigmoid bits, 0}`` (``ld.shared.b64`` of its first eight
    bytes is ``noaux_tc``'s key).
    """
    n = len(his)
    body = "{\n.reg .pred p;\n.reg .b32 z;\nmov.b32 z, 0;\nmov.u32 $0, $1;\n"
    for i in range(n):
        h, lo, s = 3 + i, 3 + n + i, 3 + 2 * n + i
        body += (
            f"setp.ge.u32 p, ${h}, $2;\n"
            f"@p st.shared.v4.b32 [$0], {{${lo}, ${h}, ${s}, z}};\n"
            "@p add.u32 $0, $0, 16;\n"
        )
    body += "}"
    return Int32(
        _asm(
            T.i32(),
            [_iv(addr, loc=loc, ip=ip), _iv(threshold, loc=loc, ip=ip)]
            + [_iv(v, loc=loc, ip=ip) for v in his]
            + [_iv(v, loc=loc, ip=ip) for v in los]
            + [_fv(v, loc=loc, ip=ip) for v in sigs],
            body,
            "=&r,r,r" + ",r" * (2 * n) + ",f" * n,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _count_greater4(acc, addr, key, *, loc=None, ip=None):
    """``acc`` plus how many of the four candidate keys at ``addr`` (16-byte records) exceed
    ``key``."""
    body = "{\n.reg .b64 a, b, c, d;\n.reg .pred p, q, r, s;\n"
    body += "ld.shared.b64 a, [$2];\nld.shared.b64 b, [$2+16];\n"
    body += "ld.shared.b64 c, [$2+32];\nld.shared.b64 d, [$2+48];\n"
    body += "setp.gt.u64 p, a, $3;\nsetp.gt.u64 q, b, $3;\nsetp.gt.u64 r, c, $3;\nsetp.gt.u64 s, d, $3;\n"
    body += "mov.u32 $0, $1;\n@p add.u32 $0, $0, 1;\n@q add.u32 $0, $0, 1;\n"
    body += "@r add.u32 $0, $0, 1;\n@s add.u32 $0, $0, 1;\n}"
    return Int32(
        _asm(
            T.i32(),
            [_iv(acc, loc=loc, ip=ip), _iv(addr, loc=loc, ip=ip), _lv(key, loc=loc, ip=ip)],
            body,
            "=&r,r,r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sts_zero16(addr, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(addr, loc=loc, ip=ip)],
        "{\n.reg .b32 z;\nmov.b32 z, 0;\nst.shared.v4.b32 [$0], {z, z, z, z};\n}",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_b32_or_zero(valid, addr, *, loc=None, ip=None):
    """Shared ``*addr`` if ``valid != 0`` else 0."""
    return Int32(
        _asm(
            T.i32(),
            [_iv(valid, loc=loc, ip=ip), _iv(addr, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.u32 p, $1, 0;\nmov.b32 $0, 0;\n@p ld.shared.b32 $0, [$2];\n}",
            "=&r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _sts_b32_if(valid, addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(valid, loc=loc, ip=ip), _iv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $0, 0;\n@p st.shared.b32 [$1], $2;\n}",
        "r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _lds_b64_or_zero(valid, addr, *, loc=None, ip=None):
    """``*addr`` if ``valid != 0`` else 0."""
    return Int64(
        _asm(
            T.i64(),
            [_iv(valid, loc=loc, ip=ip), _iv(addr, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.u32 p, $1, 0;\nmov.b64 $0, 0;\n@p ld.shared.b64 $0, [$2];\n}",
            "=&l,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _syncwarp(*, loc=None, ip=None):
    _asm(None, [], "bar.warp.sync 0xffffffff;", "", loc=loc, ip=ip)


@dsl_user_op
def _shfl_idx_b32(value, src_lane, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(value, loc=loc, ip=ip), _iv(src_lane, loc=loc, ip=ip)],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_idx_f32(value, src_lane, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(value, loc=loc, ip=ip), _iv(src_lane, loc=loc, ip=ip)],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _redux_max_u32(value, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(value, loc=loc, ip=ip)],
            "redux.sync.max.u32 $0, $1, 0xffffffff;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _redux_min_u32(value, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_iv(value, loc=loc, ip=ip)],
            "redux.sync.min.u32 $0, $1, 0xffffffff;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _argmax_scan4(
    best_hi, best_slot, best_sig, removed, his, sigs, slot0: int, *, loc=None, ip=None
):
    """Fold four slots (``slot0..slot0+3``) into a lane's running best ``(hi, slot, sigmoid)``:
    the largest ``hi`` whose bit in ``removed`` is clear, the first slot on ties (slots are in
    expert order, so the first is the smallest expert)."""
    body = "{\n.reg .pred p;\n.reg .b32 h, b;\nmov.u32 $0, $3;\nmov.u32 $1, $4;\nmov.b32 $2, $5;\n"
    for i in range(4):
        body += (
            f"bfe.u32 b, $6, {slot0 + i}, 1;\n"
            "setp.eq.u32 p, b, 0;\n"
            f"selp.b32 h, ${7 + i}, 0, p;\n"
            "setp.gt.u32 p, h, $0;\n"
            "@p mov.u32 $0, h;\n"
            f"@p mov.u32 $1, {slot0 + i};\n"
            f"@p mov.b32 $2, ${11 + i};\n"
        )
    body += "}"
    res = _asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.f32()]),
        [
            _iv(best_hi, loc=loc, ip=ip),
            _iv(best_slot, loc=loc, ip=ip),
            _fv(best_sig, loc=loc, ip=ip),
            _iv(removed, loc=loc, ip=ip),
        ]
        + [_iv(v, loc=loc, ip=ip) for v in his]
        + [_fv(v, loc=loc, ip=ip) for v in sigs],
        body,
        "=&r,=&r,=&f,r,r,f,r,r,r,r,r,f,f,f,f",
        loc=loc,
        ip=ip,
    )
    return (
        Int32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Int32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [2], loc=loc, ip=ip)),
    )


@dsl_user_op
def _cluster_arrive_relaxed(*, loc=None, ip=None):
    _asm(None, [], "barrier.cluster.arrive.relaxed.aligned;", "", loc=loc, ip=ip)


@dsl_user_op
def _cluster_wait_acquire(*, loc=None, ip=None):
    _asm(None, [], "barrier.cluster.wait.acquire.aligned;", "", loc=loc, ip=ip)


@dsl_user_op
def _griddepcontrol_wait(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.wait;", "", loc=loc, ip=ip)


@dsl_user_op
def _griddepcontrol_launch_dependents(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.launch_dependents;", "", loc=loc, ip=ip)


@dsl_user_op
def _bit(n, *, loc=None, ip=None):
    """``1 << n`` for ``0 <= n < 32``."""
    return Int32(
        _asm(T.i32(), [_iv(n, loc=loc, ip=ip)], "shl.b32 $0, 1, $1;", "=r,r", loc=loc, ip=ip)
    )


@dsl_user_op
def _renormalize_bf16_f64(score, total, scale_bits, *, loc=None, ip=None):
    """BF16 bits of ``double(score) * scale / (double(total) + 1e-20)`` rounded once to nearest: the
    weight ``kimi_k3_noaux_tc_mxfp8_quant`` stores (a BF16 cast of the FP64 expression)."""
    return Int32(
        _asm(
            T.i32(),
            [
                _fv(score, loc=loc, ip=ip),
                _fv(total, loc=loc, ip=ip),
                _lv(scale_bits, loc=loc, ip=ip),
            ],
            "{\n"
            ".reg .f64 a, b, s;\n"
            ".reg .b16 h;\n"
            "mov.b64 s, $3;\n"
            "cvt.f64.f32 a, $1;\n"
            "mul.rn.f64 a, a, s;\n"
            "cvt.f64.f32 b, $2;\n"
            f"add.rn.f64 b, b, {_EPS_F64};\n"
            "div.rn.f64 a, a, b;\n"
            "cvt.rn.bf16.f64 h, a;\n"
            "cvt.u32.u16 $0, h;\n"
            "}",
            "=r,f,f,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16_bits_of_f32(value, *, loc=None, ip=None):
    """BF16 bits of an FP32 value rounded to nearest even (``torch`` FP32 -> BF16 cast)."""
    return Int32(
        _asm(
            T.i32(),
            [_fv(value, loc=loc, ip=ip)],
            "{\n.reg .b16 h;\ncvt.rn.bf16.f32 h, $1;\ncvt.u32.u16 $0, h;\n}",
            "=r,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f32_bits(value, *, loc=None, ip=None):
    """The bits of an FP32 value."""
    return Int32(
        _asm(T.i32(), [_fv(value, loc=loc, ip=ip)], "mov.b32 $0, $1;", "=r,f", loc=loc, ip=ip)
    )


@dsl_user_op
def _biased_roundtrip(score, bias, *, loc=None, ip=None):
    """``(score + bias) - bias`` in FP32 round to nearest: the TRTLLM-Gen routing kernel's recovered sigmoid (it
    keeps only the selection score ``sigmoid + bias`` through the top-k and subtracts the bias back)."""
    return Float32(
        _asm(
            T.f32(),
            [_fv(score, loc=loc, ip=ip), _fv(bias, loc=loc, ip=ip)],
            "{\n.reg .f32 k;\nadd.rn.f32 k, $1, $2;\nsub.rn.f32 $0, k, $2;\n}",
            "=f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_global_b16(addr, value, *, loc=None, ip=None):
    """Store the low 16 bits of ``value``."""
    _asm(
        None,
        [_lv(addr, loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "{\n.reg .b16 h;\ncvt.u16.u32 h, $1;\nst.global.b16 [$0], h;\n}",
        "l,r",
        loc=loc,
        ip=ip,
    )


def _mxfp8_block_ptx(from_regs: bool = False) -> str:
    """PTX of :func:`_mxfp8_quant_block`, built once (32 elements unrolled); ``from_regs``: of
    :func:`_mxfp8_quant_regs`, the 16 words coming in as operands ``$4..$19`` instead of from ``[$0]``."""
    lines = [
        ".reg .b32 w<16>, o<8>, r, eb, sb;",
        ".reg .f32 f<32>, t, amax, sf, sc;",
        ".reg .b16 e, h<16>;",
        ".reg .pred pz, pe0, pe254;",
    ]
    for q in range(4):
        if from_regs:
            lines += [f"mov.b32 w{4 * q + e}, ${4 + 4 * q + e};" for e in range(4)]
        else:
            lines.append(
                f"ld.global.v4.b32 {{w{4 * q}, w{4 * q + 1}, w{4 * q + 2}, w{4 * q + 3}}}, [$0+{16 * q}];"
            )
    for i in range(16):
        # Element 2i is the low half of word i (little endian), 2i + 1 the high half.
        lines += [
            f"shl.b32 r, w{i}, 16;",
            f"mov.b32 f{2 * i}, r;",
            f"and.b32 r, w{i}, 0xFFFF0000;",
            f"mov.b32 f{2 * i + 1}, r;",
        ]
    lines.append("abs.f32 amax, f0;")
    for i in range(1, 32):
        lines += [f"abs.f32 t, f{i};", "max.f32 amax, amax, t;"]
    lines += [
        # SF = amax / 448 as amax * rcp.approx(448), rounded up to a power of two (UE8M0, saturating).
        "mul.rn.f32 sf, amax, $3;",
        "cvt.rp.satfinite.ue8m0x2.f32 e, sf, sf;",
        "cvt.u32.u16 eb, e;",
        "and.b32 eb, eb, 255;",
        # Output scale 1 / SF = 2^(127 - eb) (eb 0: 1.0, eb 254: the subnormal 2^-127), 0 for an all-zero block.
        "setp.eq.f32 pz, amax, 0f00000000;",
        "setp.eq.u32 pe0, eb, 0;",
        "setp.eq.u32 pe254, eb, 254;",
        "sub.u32 sb, 254, eb;",
        "shl.b32 sb, sb, 23;",
        "selp.b32 sb, 0x00400000, sb, pe254;",
        "selp.b32 sb, 0x3F800000, sb, pe0;",
        "selp.b32 sb, 0, sb, pz;",
        "mov.b32 sc, sb;",
        "st.global.u8 [$2], eb;",
    ]
    for i in range(32):
        lines.append(f"mul.rn.f32 f{i}, f{i}, sc;")
    for k in range(16):
        # Upper byte = first source: element 2k + 1 above element 2k.
        lines.append(f"cvt.rn.satfinite.e4m3x2.f32 h{k}, f{2 * k + 1}, f{2 * k};")
    for j in range(8):
        lines.append(f"mov.b32 o{j}, {{h{2 * j}, h{2 * j + 1}}};")
    lines += [
        "st.global.v4.b32 [$1], {o0, o1, o2, o3};",
        "st.global.v4.b32 [$1+16], {o4, o5, o6, o7};",
    ]
    return "{\n" + "\n".join(lines) + "\n}"


_MXFP8_BLOCK_PTX = _mxfp8_block_ptx()
_MXFP8_REGS_PTX = _mxfp8_block_ptx(from_regs=True)


@dsl_user_op
def _mxfp8_quant_block(src, dst, sf_dst, rcp448, *, loc=None, ip=None):
    """MXFP8 (E4M3 x UE8M0 per 32) quantization of the 32 BF16 values at ``src`` into 32 bytes at ``dst`` and one
    scale byte at ``sf_dst``, with ``mxfp8_quantize``'s arithmetic (quantization.cuh cvt_warp_fp16_to_mxfp8):
    ``rcp448`` is ``rcp.approx.ftz(448)``. ``src`` and ``dst`` 16-byte aligned."""
    _asm(
        None,
        [
            _lv(src, loc=loc, ip=ip),
            _lv(dst, loc=loc, ip=ip),
            _lv(sf_dst, loc=loc, ip=ip),
            _fv(rcp448, loc=loc, ip=ip),
        ],
        _MXFP8_BLOCK_PTX,
        "l,l,l,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _mxfp8_quant_regs(words, dst, sf_dst, rcp448, *, loc=None, ip=None):
    """:func:`_mxfp8_quant_block` of 32 BF16 values held in registers: ``words`` are their 16 words (element ``2 i`` in
    the low half of word ``i``), the same arithmetic and stores."""
    _asm(
        None,
        [
            _lv(Int64(0), loc=loc, ip=ip),
            _lv(dst, loc=loc, ip=ip),
            _lv(sf_dst, loc=loc, ip=ip),
            _fv(rcp448, loc=loc, ip=ip),
        ]
        + [_iv(w, loc=loc, ip=ip) for w in words],
        _MXFP8_REGS_PTX,
        "l,l,l,f" + ",r" * 16,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _rcp_approx_ftz(value, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [_fv(value, loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


#: ``RouteTailKernel.weights`` values: the BF16 cast of ``noaux_tc_op``'s FP32 weight.
WEIGHTS_BF16 = "bf16"
#: BF16 rounded once from the FP64 expression (``kimi_k3_noaux_tc_mxfp8_quant``'s), without FP64
#: arithmetic on the common path: at unit scale (and a total of at least 2^-12)
#: ``div.rn.f32(score, total)`` is the FP64 expression rounded to FP32, and rounding that to BF16
#: equals rounding the FP64 value once unless the FP32 value is a BF16 rounding midpoint (low 16
#: bits 0x8000); those lanes, and other scales, take the FP64 path.
WEIGHTS_BF16_F64_FAST = "bf16_f64_fast"
#: The TRTLLM-Gen MoE runners' in-kernel routing weights (runner.cu, one-group DeepSeek-V3: ``routingCustom``
#: SigmoidBias + ScaledSumNormalize): the sigmoid recovered as ``(sigmoid + bias) - bias`` in FP32 (the kernel keeps
#: only the selection score through the top-k), summed over the top-16 by the butterfly warp sum, divided in FP32
#: (round to nearest), then BF16. Unit routed scale only (the kernel multiplies by it before the division).
WEIGHTS_RUNNER = "runner"


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------


class RouteTailKernel:
    """Top-16 selection of the routing kernels: one warp routes one token at a time, lane ``l``
    owning experts ``32 j + l``. Subclasses launch it (:class:`RouteWideKernel`, the MoE prologue
    kernels).

    Args:
        num_warps: Warps per CTA.
        use_pdl: Launch with programmatic dependent launch (the logits come from the
            previous kernel).
        weights: :data:`WEIGHTS_BF16`, :data:`WEIGHTS_BF16_F64_FAST` or :data:`WEIGHTS_RUNNER`.
    """

    def __init__(self, num_warps: int, use_pdl: bool, weights: str):
        if weights not in (WEIGHTS_BF16, WEIGHTS_BF16_F64_FAST, WEIGHTS_RUNNER):
            raise ValueError(f"weights={weights!r} is not a weights mode of this kernel")
        self.weights = weights
        self.threads = 32 * num_warps
        # Tokens routed at a time, one per warp.
        self.teams = num_warps
        self.slots = NUM_EXPERTS // 32
        # Candidate records per warp (plus zero padding to a multiple of 4); more fall back to
        # the exact arg-max rounds.
        self.region_cap = 64
        self.region_slots = self.region_cap + 4
        self.use_pdl = use_pdl

    def load_logits(self, row: Int64, g: Int32, stride: int, n: int):
        """Logit ``s * stride + g`` (``s < n``) of the token whose logits start at ``row``. A
        subclass may read them differently (the sharded MoE front polls them)."""
        return [_ld_f32(row + Int64((s * stride + g) * 4)) for s in range(n)]

    def load_logits4(self, row: Int64, lane: Int32, n: int):
        """Logits ``128 j + 4 lane .. + 3`` (``j < n``) of the token whose logits start at
        ``row``, concatenated. Not ``cute.jit`` (the DSL preprocessor does not
        rewrite this method), so the loop is a plain Python loop, unrolled at trace time."""
        logit = []
        for blk in range(n):
            logit.extend(_ld_v4_f32(row + Int64((blk * 128 + lane * 4) * 4)))
        return logit

    @cute.jit
    def select_token(
        self,
        token: Int32,
        lane: Int32,
        logits_addr: Int64,
        bias_smem: Int32,
        scale_bits: Int64,
        cand_smem: Int32,
        top_smem: Int32,
    ):
        """Top-16 of one token by one warp: returns ``(expert, weight)``, the ``lane``-th
        selection in lanes ``0..15``."""
        n_slots = self.slots
        row = logits_addr + Int64(token) * Int64(NUM_EXPERTS * 4)
        logit = self.load_logits(row, lane, 32, n_slots)
        bias = [_lds_f32(bias_smem + (s * 32 + lane) * 4) for s in range(n_slots)]
        sig = [
            cute.math.tanh(logit[s] * Float32(0.5)) * Float32(0.5) + Float32(0.5)
            for s in range(n_slots)
        ]
        his = [_score_bits(sig[s] + bias[s]) for s in range(n_slots)]
        lane_max = his[0]
        for s in cutlass.range_constexpr(1, n_slots):
            lane_max = _max_u32(lane_max, his[s])
        # A lower bound of the 16th largest score: the 16th largest lane maximum (16 of the
        # warp's experts score at least that much).
        threshold = _warp_16th_largest_u32(lane_max, lane)
        count = Int32(0)
        for c4 in cutlass.range_constexpr(0, n_slots, 4):
            count = count + _count_ge_u32(his[c4 : min(c4 + 4, n_slots)], threshold)
        inclusive = _warp_inclusive_sum(count)
        num_mine = _shfl_idx_b32(inclusive, Int32(31))
        cap = self.region_cap
        if num_mine <= cap:
            los = [Int32(65535 - s * 32) - lane for s in range(n_slots)]
            dst = cand_smem + (inclusive - count) * _CAND_BYTES
            for c4 in cutlass.range_constexpr(0, n_slots, 4):
                q = slice(c4, min(c4 + 4, n_slots))
                dst = _store_candidates(dst, his[q], los[q], sig[q], threshold)
            # Zero keys after the last candidate, so the ranking reads groups of four.
            if lane < 4:
                _sts_zero16(cand_smem + (num_mine + lane) * _CAND_BYTES)
        _syncwarp()
        overflow = Int32(0)
        if num_mine > cap:
            overflow = Int32(1)

        expert = Int32(0)
        score = Float32(0.0)
        if overflow == 0:
            # Rank the warp's candidates against each other.
            for c0 in cutlass.range_constexpr(cap // 32):
                c = lane + c0 * 32
                valid = Int32(0)
                if c < num_mine:
                    valid = Int32(1)
                key = _lds_b64_or_zero(valid, cand_smem + c * _CAND_BYTES)
                r = Int32(0)
                limit = num_mine
                if cutlass.const_expr(c0 > 0):
                    # No candidate of this warp in this round: skip the count.
                    if num_mine <= c0 * 32:
                        limit = Int32(0)
                for o in cutlass.range(0, limit, 4, unroll=1):
                    r = _count_greater4(r, cand_smem + o * _CAND_BYTES, key)
                if valid != 0 and r < TOP_K:
                    _sts_b32(top_smem + r * 8, 65535 - (Int32(key) & 0xFFFF))
                    _sts_b32(top_smem + r * 8 + 4, _lds_b32(cand_smem + c * _CAND_BYTES + 8))
            _syncwarp()
            if lane < TOP_K:
                expert = _lds_b32(top_smem + lane * 8)
                score = _lds_f32(top_smem + lane * 8 + 4)
        else:
            expert, score = self.topk_rounds(row, bias_smem, lane)

        if cutlass.const_expr(self.weights == WEIGHTS_RUNNER):
            recovered = Float32(0.0)
            if lane < TOP_K:
                recovered = _biased_roundtrip(score, _lds_f32(bias_smem + expert * 4))
            total = _warp_sum_f32(recovered)
            return expert, _bf16_bits_of_f32(_div_rn_f32(recovered, total))
        if cutlass.const_expr(self.weights == WEIGHTS_BF16_F64_FAST):
            total = _warp_sum_f32(score)
            w32 = _div_rn_f32(score, total)
            bits = _bf16_bits_of_f32(w32)
            if _unit_scale_fast_path(scale_bits, total) == 0 or (_f32_bits(w32) & 0xFFFF) == 0x8000:
                bits = _renormalize_bf16_f64(score, total, scale_bits)
            return expert, bits
        total = _warp_sum_f32(score)
        # Exact FP32 fast path of noaux_tc's FP64 expression (see route_ptx).
        weight = Float32(0.0)
        if _unit_scale_fast_path(scale_bits, total) != 0:
            weight = _div_rn_f32(score, total)
        else:
            weight = _renormalize(score, total, scale_bits)
        # BF16 bits of noaux_tc's FP32 weight (round to nearest even, like a torch cast).
        return expert, _bf16_bits_of_f32(weight)

    @cute.jit
    def topk_rounds(self, row: Int64, bias_smem: Int32, lane: Int32):
        """Exact fallback by one warp: 16 arg-max rounds over all 896 keys of the token (lane
        ``l`` owns experts ``128 j + 4 l + i``); lane ``k`` gets the ``k``-th expert."""
        logit = self.load_logits4(row, lane, _BLOCKS)
        sig = [
            cute.math.tanh(logit[s] * Float32(0.5)) * Float32(0.5) + Float32(0.5)
            for s in range(_SLOTS)
        ]
        his = []
        for blk in cutlass.range_constexpr(_BLOCKS):
            b = _lds_v4_f32(bias_smem + (blk * 128 + lane * 4) * 4)
            for i in cutlass.range_constexpr(4):
                his.append(_score_bits(sig[blk * 4 + i] + b[i]))
        removed = Int32(0)
        expert = Int32(0)
        score = Float32(0.0)
        for r in cutlass.range_constexpr(TOP_K):
            best_hi = Int32(0)
            best_slot = Int32(0)
            best_sig = Float32(0.0)
            for blk in cutlass.range_constexpr(_BLOCKS):
                q = slice(blk * 4, blk * 4 + 4)
                best_hi, best_slot, best_sig = _argmax_scan4(
                    best_hi, best_slot, best_sig, removed, his[q], sig[q], blk * 4
                )
            top_hi = _redux_max_u32(best_hi)
            # Slot s of this lane is expert 128 (s / 4) + 4 lane + s % 4.
            cand = Int32(0xFFFF)
            if best_hi == top_hi:
                cand = (best_slot >> 2) * 128 + lane * 4 + (best_slot & 3)
            winner = _redux_min_u32(cand)
            owner = (winner & 127) >> 2
            if lane == owner:
                removed = removed | _bit(best_slot)
            win_sig = _shfl_idx_f32(best_sig, owner)
            if lane == r:
                expert = winner
                score = win_sig
        return expert, score


# ---------------------------------------------------------------------------
# Multi-CTA routing
# ---------------------------------------------------------------------------

#: Shared-memory words of the staged bias (the start of the top-k scratch).
_WIDE_BIAS_WORDS = NUM_EXPERTS
#: Budget (bytes) of :func:`wide_table_fits`: ``CTAs x L4`` mask bytes (``L4``: experts rounded up
#: to 4), then two bytes per block of 8 CTAs and expert.
WIDE_COMPACT_TABLE_BYTES = 13824


class RouteWideKernel(RouteTailKernel):
    """Top-16 of up to :data:`WIDE_MAX_M` tokens on a grid of independent CTAs.

    CTA ``c`` routes tokens ``[T c, T c + T)`` (``T = tokens_per_cta``) with the warps of
    :class:`RouteTailKernel`. No CTA waits on another, so the grid needs no co-residency.

    Args:
        tokens_per_cta: Tokens per CTA: 8, 16 or 32.
        num_warps: Warps per CTA: 4, 8 or 16.
        use_pdl: Launch with programmatic dependent launch.
        weights: The weights mode (see :class:`RouteTailKernel`).
    """

    def __init__(self, tokens_per_cta: int, num_warps: int, use_pdl: bool, weights: str):
        super().__init__(num_warps, use_pdl, weights)
        if tokens_per_cta not in (8, 16, 32):
            raise ValueError(f"tokens_per_cta={tokens_per_cta} must be 8, 16 or 32")
        if num_warps not in (4, 8, 16):
            raise ValueError(f"num_warps={num_warps} must be 4, 8 or 16")
        self.tokens = tokens_per_cta
        self.cand_words = num_warps * self.region_slots * 4
        self.topk_words = _WIDE_BIAS_WORDS + self.cand_words + self.teams * TOP_K * 2

    @cute.jit
    def __call__(
        self,
        m: Int32,
        logits_addr: Int64,
        bias_addr: Int64,
        ids_addr: Int64,
        vals_addr: Int64,
        scale_bits: Int64,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            # Bias, candidate regions and top-16 slots.
            scratch: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, self.topk_words], 16]

        self.shared_storage = SharedStorage

        num_ctas = (m + (self.tokens - 1)) // self.tokens
        self.kernel(
            m,
            logits_addr,
            bias_addr,
            ids_addr,
            vals_addr,
            scale_bits,
        ).launch(
            grid=(num_ctas, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        m: Int32,
        logits_addr: Int64,
        bias_addr: Int64,
        ids_addr: Int64,
        vals_addr: Int64,
        scale_bits: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        bias_smem = storage.scratch.data_ptr().toint()
        cand_all = bias_smem + _WIDE_BIAS_WORDS * 4
        top_all = cand_all + self.cand_words * 4
        cand_smem = cand_all + warp * (self.region_slots * _CAND_BYTES)
        top_smem = top_all + warp * (TOP_K * 8)

        t_begin = cta * self.tokens
        t_end = t_begin + self.tokens
        if t_end > m:
            t_end = m

        # The bias is a weight: stage it in shared memory before waiting on the previous grid.
        for e4 in cutlass.range(tidx, NUM_EXPERTS // 4, self.threads, unroll=1):
            v = _ld_nc_v4_f32(bias_addr + Int64(e4 * 16))
            _sts_v4_f32(bias_smem + e4 * 16, v[0], v[1], v[2], v[3])
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
        cute.arch.barrier()

        for token in cutlass.range(t_begin + warp, t_end, self.teams, unroll=1):
            expert, weight = self.select_token(
                token,
                lane,
                logits_addr,
                bias_smem,
                scale_bits,
                cand_smem,
                top_smem,
            )
            if lane < TOP_K:
                out = token * TOP_K + lane
                _st_global_b32(ids_addr + Int64(out * 4), expert)
                _st_global_b16(vals_addr + Int64(out * 2), weight)
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_launch_dependents()


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


def _check_inputs(logits: torch.Tensor, e_score_correction_bias: torch.Tensor) -> None:
    if logits.dtype != torch.float32 or logits.dim() != 2:
        raise ValueError("logits must be a 2D [M, 896] float32 tensor")
    if logits.shape[-1] != NUM_EXPERTS or not logits.is_contiguous():
        raise ValueError(f"logits must be contiguous with {NUM_EXPERTS} columns")
    if (
        e_score_correction_bias.dtype != torch.float32
        or e_score_correction_bias.numel() != NUM_EXPERTS
        or not e_score_correction_bias.is_contiguous()
    ):
        raise ValueError(f"e_score_correction_bias must be a contiguous float32 [{NUM_EXPERTS}]")
    for t in (logits, e_score_correction_bias):
        if not t.is_cuda or t.device != logits.device or t.data_ptr() % 16:
            raise ValueError("inputs must be 16-byte aligned tensors on the same CUDA device")


#: Largest token count routed with 8-token CTAs (see :func:`default_wide_config`).
_WIDE_COMPACT_MAX_M = 1024


def wide_table_fits(m: int, local_num_experts: int) -> bool:
    """Whether ``m`` tokens over ``local_num_experts`` experts in CTAs of 8 tokens stay within
    :data:`WIDE_COMPACT_TABLE_BYTES`."""
    ctas = (m + 7) // 8
    row_bytes = 4 * ((local_num_experts + 3) // 4)
    table = (ctas * row_bytes + 15) // 16 * 16
    return table + (ctas + 7) // 8 * local_num_experts * 2 <= WIDE_COMPACT_TABLE_BYTES


def default_wide_config(m: int, local_num_experts: int) -> tuple[int, int]:
    """``(tokens_per_cta, num_warps)`` of the routing of ``m`` tokens (``1 <= m <= WIDE_MAX_M``):
    8-warp CTAs of 8 tokens up to :data:`_WIDE_COMPACT_MAX_M` tokens while :func:`wide_table_fits`,
    else 16-warp CTAs of 16 tokens (one token per warp).
    """
    if m <= _WIDE_COMPACT_MAX_M and wide_table_fits(m, local_num_experts):
        return 8, 8
    return 16, 16


@functools.cache
def _compiled_wide(tokens_per_cta: int, num_warps: int):
    kernel = RouteWideKernel(tokens_per_cta, num_warps, use_pdl=True, weights=WEIGHTS_BF16)
    dummy = Int64(1 << 20)
    return cute.compile(
        kernel,
        Int32(WIDE_MAX_M),
        *([dummy] * 4),
        Int64(0),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )


def run_route_wide(
    logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> None:
    """Launch the routing on the current stream (inputs are not re-validated): the top-16
    ``topk_ids`` and BF16 ``topk_weights`` of the ``1 <= M <= WIDE_MAX_M`` rows of ``logits``, in
    the configuration :func:`default_wide_config` picks.
    """
    m = logits.shape[0]
    if not 1 <= m <= WIDE_MAX_M:
        raise ValueError(f"num_tokens={m} must be in [1, {WIDE_MAX_M}]")
    compiled = _compiled_wide(*default_wide_config(m, NUM_EXPERTS))
    scale_bits = struct.unpack("<q", struct.pack("<d", float(routed_scaling_factor)))[0]
    compiled(
        Int32(m),
        Int64(logits.data_ptr()),
        Int64(e_score_correction_bias.data_ptr()),
        Int64(topk_ids.data_ptr()),
        Int64(topk_weights.data_ptr()),
        Int64(scale_bits),
        cuda.CUstream(torch.cuda.current_stream(logits.device).cuda_stream),
    )


def _check_topk_outputs(m: int) -> None:
    if not 1 <= m <= WIDE_MAX_M:
        raise ValueError(f"num_tokens={m} must be in [1, {WIDE_MAX_M}]")


def k31_route_topk(
    logits: torch.Tensor,
    e_score_correction_bias: torch.Tensor,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``noaux_tc`` top-16 of FP32 router logits with BF16 weights, by the multi-CTA kernel.

    Args:
        logits: ``fp32 [M, 896]`` router logits, contiguous, 16-byte aligned, ``1 <= M <= 2048``.
        e_score_correction_bias: ``fp32 [896]``.
        routed_scaling_factor: Multiplier of the renormalized weights.

    Returns:
        ``(topk_ids int32 [M, 16], topk_weights bf16 [M, 16])``: the experts, in
        ``noaux_tc_op``'s order, and ``bf16(noaux_tc_op's fp32 weights)``, bit for bit.
    """
    _check_inputs(logits, e_score_correction_bias)
    m = logits.shape[0]
    _check_topk_outputs(m)
    topk_ids = torch.empty((m, TOP_K), dtype=torch.int32, device=logits.device)
    topk_weights = torch.empty((m, TOP_K), dtype=torch.bfloat16, device=logits.device)
    run_route_wide(logits, e_score_correction_bias, routed_scaling_factor, topk_ids, topk_weights)
    return topk_ids, topk_weights


__all__ = [
    "NUM_EXPERTS",
    "TOP_K",
    "WIDE_MAX_M",
    "RouteTailKernel",
    "RouteWideKernel",
    "default_wide_config",
    "k31_route_topk",
    "run_route_wide",
]
