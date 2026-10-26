# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4.1 mHC boundary (lagged mode) for decode-size batches.

The hidden dimension is split into 16 slices. For every token, a slice's CTA updates its part of the
residual stream, collapses it with the lagged pre-mix, and multiplies the updated residual with its
slice of the mixer weight on BF16 tensor cores. The weight is rounded to TF32 and split into BF16
high and low parts that sum to it exactly (``prepare_boundary_weight``), so every product equals the
TF32 product of ``mhc_fused_hc``. Up to one wave of tokens, a 16-CTA cluster owns a tile of tokens and
its CTAs push their partial mixes and sums of squares to the CTA that owns the token through
distributed shared memory (``MhcBoundaryKernel``). Larger batches stream token groups through
persistent CTAs that keep their weight slice in shared memory and write the partials to global
memory for a second kernel (``MhcStreamKernel``, ``MhcFinalizeKernel``). Every reduction runs in a
fixed order, so results are deterministic.
"""

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import SmemAllocator

from tensorrt_llm.math_utils import ceil_div, pad_up

from ..utils import TRTLLM_ENABLE_PDL

Int32, Int64, Uint32, Float32 = cutlass.Int32, cutlass.Int64, cutlass.Uint32, cutlass.Float32

CLUSTER = 16
THREADS = 640
MMA_WARPS = 10
MULT = 4
MIXES = 24
RECV = 28  # floats per pushed token partial: 24 mixes, sum of squares of residual_cur, padding
TILE = 28
# Seven 16-CTA clusters with this kernel's shared memory are co-resident on GB300: one wave of tiles.
MAX_TOKENS = 7 * TILE
STREAM_WARPS = 12
PART = 32  # floats per token and slice partial: 24 mixes, sums of squares of residual_cur and li
FINAL_TOKENS = 8

_ATT = llvm.AsmDialect.AD_ATT


def _asm(result, args, body, constraints, side_effects=True, loc=None, ip=None):
    return llvm.inline_asm(
        result,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        body,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=_ATT,
        loc=loc,
        ip=ip,
    )


def _fields(out, n, ty, wrap, loc, ip):
    return tuple(wrap(llvm.extractvalue(ty, out, [i], loc=loc, ip=ip)) for i in range(n))


@dsl_user_op
def lds_v4_u32(addr, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(addr)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _fields(out, 4, T.i32(), Uint32, loc, ip)


@dsl_user_op
def lds_f32(addr, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [Int32(addr)], "ld.shared.f32 $0, [$1];", "=f,r", loc=loc, ip=ip))


@dsl_user_op
def ldsm_x4(addr, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(addr)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _fields(out, 4, T.i32(), Uint32, loc, ip)


@dsl_user_op
def sts_f32(addr, value, *, loc=None, ip=None):
    _asm(None, [Int32(addr), Float32(value)], "st.shared.f32 [$0], $1;", "r,f", loc=loc, ip=ip)


@dsl_user_op
def sts_v2_f32(addr, a, b, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(addr), Float32(a), Float32(b)],
        "st.shared.v2.f32 [$0], {$1, $2};",
        "r,f,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def sts_v4_u32(addr, w, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(addr)] + [Uint32(v) for v in w],
        "st.shared.v4.u32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def sts_u32(addr, w, *, loc=None, ip=None):
    _asm(None, [Int32(addr), Uint32(w)], "st.shared.u32 [$0], $1;", "r,r", loc=loc, ip=ip)


@dsl_user_op
def ldg_v4_u32(addr, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(addr)],
        "ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return _fields(out, 4, T.i32(), Uint32, loc, ip)


@dsl_user_op
def ldg_v4_u32_ordered(addr, *, loc=None, ip=None):
    """``ldg_v4_u32`` kept in program order, so a load issued ahead stays ahead."""
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(addr)],
        "ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return _fields(out, 4, T.i32(), Uint32, loc, ip)


@dsl_user_op
def ldg_v4_f32(addr, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int64(addr)],
        "ld.global.nc.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return _fields(out, 4, T.f32(), Float32, loc, ip)


@dsl_user_op
def ldg_f32(addr, *, loc=None, ip=None):
    return Float32(
        _asm(T.f32(), [Int64(addr)], "ld.global.nc.f32 $0, [$1];", "=f,l", loc=loc, ip=ip)
    )


@dsl_user_op
def stg_v4_u32(addr, w, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr)] + [Uint32(v) for v in w],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_f32(addr, value, *, loc=None, ip=None):
    _asm(None, [Int64(addr), Float32(value)], "st.global.f32 [$0], $1;", "l,f", loc=loc, ip=ip)


@dsl_user_op
def stg_v2_u32(addr, a, b, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr), Uint32(a), Uint32(b)],
        "st.global.v2.u32 [$0], {$1, $2};",
        "l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stg_u8(addr, value, *, loc=None, ip=None):
    _asm(None, [Int64(addr), Uint32(value)], "st.global.u8 [$0], $1;", "l,r", loc=loc, ip=ip)


@dsl_user_op
def mxfp8_block(amax, vals, *, loc=None, ip=None):
    """E4M3 bytes and UE8M0 scale of 8 values of a 32-element MXFP8 block with absolute maximum
    ``amax``, as ``cvt_warp_fp16_to_mxfp8`` computes them (approximate reciprocals, IEEE products
    that keep denormals): two words of E4M3 bytes and the scale in the low byte of a third."""
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 3),
        [Float32(amax)] + [Float32(v) for v in vals],
        "{\n.reg .f32 r, sf, os, f<8>;\n.reg .b16 e, h<4>;\n.reg .b32 sb;\n.reg .pred z;\n"
        "rcp.approx.ftz.f32 r, 0f43E00000;\nmul.rn.f32 sf, $3, r;\n"
        "cvt.rp.satfinite.ue8m0x2.f32 e, 0f00000000, sf;\ncvt.rn.bf16x2.ue8m0x2 sb, e;\n"
        "shl.b32 sb, sb, 16;\nmov.b32 sf, sb;\nrcp.approx.ftz.f32 os, sf;\n"
        "setp.neu.f32 z, $3, 0f00000000;\nselp.f32 os, os, 0f00000000, z;\n"
        "mul.rn.f32 f0, $4, os;\nmul.rn.f32 f1, $5, os;\nmul.rn.f32 f2, $6, os;\n"
        "mul.rn.f32 f3, $7, os;\nmul.rn.f32 f4, $8, os;\nmul.rn.f32 f5, $9, os;\n"
        "mul.rn.f32 f6, $10, os;\nmul.rn.f32 f7, $11, os;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h0, f1, f0;\ncvt.rn.satfinite.e4m3x2.f32 h1, f3, f2;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h2, f5, f4;\ncvt.rn.satfinite.e4m3x2.f32 h3, f7, f6;\n"
        "mov.b32 $0, {h0, h1};\nmov.b32 $1, {h2, h3};\ncvt.u32.u16 $2, e;\n}",
        "=r,=r,=r,f,f,f,f,f,f,f,f,f",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return _fields(out, 3, T.i32(), Uint32, loc, ip)


@dsl_user_op
def bulk_g2s(dst, src, size, bar, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(dst), Int64(src), Int32(size), Int32(bar)],
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_async_v4(addr, w, bar, rank, *, loc=None, ip=None):
    """Store four words at the same shared-memory offset of CTA ``rank``, completing on its ``bar``."""
    _asm(
        None,
        [Int32(addr)] + [Uint32(v) for v in w] + [Int32(bar), Int32(rank)],
        "{\n.reg .b32 ra, rb;\nmapa.shared::cluster.u32 ra, $0, $6;\nmapa.shared::cluster.u32 rb, $5, $6;\n"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [ra], {$1, $2, $3, $4}, [rb];\n}",
        "r,r,r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbar_init(bar, count, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(bar), Int32(count)],
        "mbarrier.init.shared::cta.b64 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbar_fence_init(*, loc=None, ip=None):
    _asm(None, [], "fence.mbarrier_init.release.cluster;", "", loc=loc, ip=ip)


@dsl_user_op
def mbar_expect_tx(bar, size, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(bar), Int32(size)],
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbar_wait(bar, phase, *, scope="cta", loc=None, ip=None):
    """Wait for ``phase``; ``scope="cluster"`` acquires data that peers delivered with st.async."""
    sem = ".acquire.cluster" if scope == "cluster" else ""
    _asm(
        None,
        [Int32(bar), Int32(phase)],
        "{\n.reg .pred p;\nWAIT:\n"
        f"mbarrier.try_wait.parity{sem}.shared::cta.b64 p, [$0], $1;\n"
        "@!p bra WAIT;\n}",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mma_bf16(a, b, c, *, loc=None, ip=None):
    """mma.sync m16n8k16 BF16 with FP32 accumulation."""
    out = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Uint32(v) for v in a] + [Uint32(v) for v in b] + [Float32(v) for v in c],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {$0, $1, $2, $3}, {$4, $5, $6, $7}, "
        "{$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return _fields(out, 4, T.f32(), Float32, loc, ip)


@dsl_user_op
def shfl_xor_f32(value, mask, *, loc=None, ip=None):
    return Float32(
        _asm(
            T.f32(),
            [Float32(value), Int32(mask)],
            "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def rcp(x, *, loc=None, ip=None):
    """IEEE reciprocal of a normal input: the fast path of rcp.rn.f32 without its out-of-range branch."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(x)],
            "{\n.reg .f32 r, e;\nrcp.approx.ftz.f32 r, $1;\nfma.rn.f32 e, $1, r, 0fBF800000;\n"
            "neg.f32 e, e;\nfma.rn.f32 $0, r, e, r;\n}",
            "=f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bf16x2_lo(word, *, loc=None, ip=None):
    return Float32(
        llvm.bitcast(T.f32(), (Uint32(word) << 16).ir_value(loc=loc, ip=ip), loc=loc, ip=ip)
    )


@dsl_user_op
def bf16x2_hi(word, *, loc=None, ip=None):
    high = Uint32(word) & Uint32(0xFFFF0000)
    return Float32(llvm.bitcast(T.f32(), high.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def f32_bits(value, *, loc=None, ip=None):
    return Uint32(llvm.bitcast(T.i32(), Float32(value).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def pack_bf16x2(lo, hi, *, loc=None, ip=None):
    """Round two floats to BF16 (nearest even); ``lo`` goes to the low half."""
    return Uint32(
        _asm(
            T.i32(),
            [Float32(hi), Float32(lo)],
            "cvt.rn.bf16x2.f32 $0, $1, $2;",
            "=r,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bf16x2_absmax4(w, *, loc=None, ip=None):
    """``max(|w0|, |w1|, |w2|, |w3|)`` per BF16 half, as ``__hmax2(__habs2(...))`` folds them."""
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(v) for v in w],
            "{\n.reg .b32 a, t;\nabs.bf16x2 a, $1;\nabs.bf16x2 t, $2;\nmax.bf16x2 a, a, t;\n"
            "abs.bf16x2 t, $3;\nmax.bf16x2 a, a, t;\nabs.bf16x2 t, $4;\nmax.bf16x2 $0, a, t;\n}",
            "=r,r,r,r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def shfl_xor_u32(value, mask, *, loc=None, ip=None):
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(value), Int32(mask)],
            "shfl.sync.bfly.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def nvfp4_block(absmax_pair, other_pair, global_scale, values, *, loc=None, ip=None):
    """NVFP4 E2M1 word of 8 floats and the E4M3 scale byte of their 16-element block, with the
    arithmetic of ``fp4_quantize`` (``cvt_warp_fp16_to_fp4``: approximate reciprocals, products and
    the zero test that keep denormals): ``absmax_pair``/``other_pair`` are the BF16x2 absolute maxima
    of this lane's and the paired lane's 8 values."""
    out = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [Uint32(absmax_pair), Uint32(other_pair), Float32(global_scale)]
        + [Float32(v) for v in values],
        "{\n.reg .b32 a, m;\n.reg .b16 lo, hi, mh, e, sfh;\n"
        ".reg .f32 r6, vmax, sf, rg, os, zero, f0, f1, f2, f3, f4, f5, f6, f7;\n"
        ".reg .pred z;\n.reg .b8 b0, b1, b2, b3;\n"
        "max.bf16x2 a, $3, $2;\nmov.b32 {lo, hi}, a;\nmax.bf16 mh, lo, hi;\n"
        "cvt.u32.u16 m, mh;\nshl.b32 m, m, 16;\nmov.b32 vmax, m;\n"
        "rcp.approx.ftz.f32 r6, 0f40C00000;\nmul.rn.f32 sf, vmax, r6;\nmul.rn.f32 sf, $4, sf;\n"
        "mov.f32 zero, 0f00000000;\ncvt.rn.satfinite.e4m3x2.f32 e, zero, sf;\n"
        "cvt.u32.u16 $1, e;\nand.b32 $1, $1, 255;\n"
        "cvt.rn.f16x2.e4m3x2 a, e;\nmov.b32 {sfh, lo}, a;\ncvt.f32.f16 sf, sfh;\n"
        "rcp.approx.ftz.f32 rg, $4;\nmul.rn.f32 os, sf, rg;\nrcp.approx.ftz.f32 os, os;\n"
        "setp.eq.f32 z, vmax, zero;\nselp.f32 os, zero, os, z;\n"
        "mul.rn.f32 f0, $5, os;\nmul.rn.f32 f1, $6, os;\nmul.rn.f32 f2, $7, os;\n"
        "mul.rn.f32 f3, $8, os;\nmul.rn.f32 f4, $9, os;\nmul.rn.f32 f5, $10, os;\n"
        "mul.rn.f32 f6, $11, os;\nmul.rn.f32 f7, $12, os;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b0, f1, f0;\ncvt.rn.satfinite.e2m1x2.f32 b1, f3, f2;\n"
        "cvt.rn.satfinite.e2m1x2.f32 b2, f5, f4;\ncvt.rn.satfinite.e2m1x2.f32 b3, f7, f6;\n"
        "mov.b32 $0, {b0, b1, b2, b3};\n}",
        "=r,=r,r,r,f,f,f,f,f,f,f,f,f",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return _fields(out, 2, T.i32(), Uint32, loc, ip)


@dsl_user_op
def stg_u32(addr, w, *, loc=None, ip=None):
    _asm(None, [Int64(addr), Uint32(w)], "st.global.u32 [$0], $1;", "l,r", loc=loc, ip=ip)


def sigmoid(v):
    """``1 / (1 + exp(-v))``; the denominator is clamped below 2^126 so the reciprocal is exact."""
    return rcp(cute.arch.fmin(Float32(1.0) + cute.math.exp(-v), Float32(2.0**126)))


def residual_update(pw, cm, xw, rw):
    """BF16 pairs of ``bf16(post[s] * x + sum_j comb[j][s] * residual[j])`` for 8 hidden elements per
    stream ``s``, ``mhc_fused_hc``'s flush-to-zero FMA chain."""
    xf = [(bf16x2_lo(xw[p]), bf16x2_hi(xw[p])) for p in range(4)]
    rf = [[(bf16x2_lo(rw[j][p]), bf16x2_hi(rw[j][p])) for p in range(4)] for j in range(MULT)]
    cur = []
    for s in range(MULT):
        words = []
        for p in range(4):
            nu = cute.arch.mul_packed_f32x2((pw[s], pw[s]), xf[p], ftz=True)
            for j in range(MULT):
                nu = cute.arch.fma_packed_f32x2((cm[j][s], cm[j][s]), rf[j][p], nu, ftz=True)
            words.append(pack_bf16x2(nu[0], nu[1]))
        cur.append(words)
    return cur


def lagged_collapse(cur, pre, ssq_rc, ssq_li):
    """BF16 pairs of ``li = bf16(sum_j pre[j] * residual_cur[j])`` for 8 hidden elements, with the running
    sums of squares of residual_cur and li."""
    li = []
    for p in range(4):
        acc = (Float32(0.0), Float32(0.0))
        for j in range(MULT):
            v = (bf16x2_lo(cur[j][p]), bf16x2_hi(cur[j][p]))
            ssq_rc = cute.arch.fma_packed_f32x2(v, v, ssq_rc)
            acc = cute.arch.fma_packed_f32x2((pre[j], pre[j]), v, acc)
        word = pack_bf16x2(acc[0], acc[1])
        li.append(word)
        v = (bf16x2_lo(word), bf16x2_hi(word))
        ssq_li = cute.arch.fma_packed_f32x2(v, v, ssq_li)
    return li, ssq_rc, ssq_li


class MhcBoundaryKernel:
    """``TILE`` tokens per 16-CTA cluster; CTA ``r`` owns hidden ``[r * slice, (r + 1) * slice)``.

    Mixer GEMM per CTA: ``[48, K slice] x [K slice, tokens]`` with rows 0-23 the high and rows 24-47
    the low BF16 part of the weight. Warp ``w < MMA_WARPS`` takes the k16 steps
    ``[w * steps_per_warp, (w + 1) * steps_per_warp)``.
    """

    def __init__(self, hidden: int, with_post: bool, sinkhorn_iters: int, mxfp8: bool = False):
        self.hidden = hidden
        self.with_post = with_post
        self.sinkhorn_iters = sinkhorn_iters
        # Also write layer_input as torch.ops.trtllm.mxfp8_quantize(layer_input, True) does: E4M3
        # [B, hidden] and UE8M0 scales in the 128x4 swizzled layout (padding rows' not written).
        self.mxfp8 = mxfp8
        self.slice = hidden // CLUSTER
        self.chunks = self.slice // 8
        self.parts = self.chunks // 2  # elementwise threads per token, two 8-wide chunks each
        self.k_steps = MULT * self.slice // 16
        self.steps_per_warp = self.k_steps // MMA_WARPS
        self.n_tiles = (TILE + 7) // 8
        self.cols = 40  # padded token columns of the per-warp partials
        self.owned = (
            TILE + CLUSTER - 1
        ) // CLUSTER  # at most 2: warp 0 runs 16 lanes per owned token
        self.li_words = (TILE + 3) // 4 * 4
        self.w_bytes = self.k_steps * 32 * 48
        # residual_cur rows in k16 block order; 16 bytes of padding per 32-wide chunk and per row keep
        # the elementwise stores and the ldmatrix rows on distinct banks.
        self.r_chunk = 8 * 32 + 16
        self.r_row = self.slice // 32 * self.r_chunk + 16
        self.x_row = self.slice * 2
        off = 0

        def take(size, align=16):
            nonlocal off
            off = (off + align - 1) // align * align
            start = off
            off += size
            return start

        # The weight fragments are dead after the mixer GEMM; its per-warp partials reuse the space.
        self.o_w = take(max(self.w_bytes, MMA_WARPS * MIXES * self.cols * 4), 128)
        self.o_r = take(TILE * self.r_row)
        self.o_x = take(TILE * self.x_row)
        self.o_prc = take(TILE * self.parts * 4)
        self.o_pli = take(TILE * self.parts * 4)
        self.o_cta = take(TILE * RECV * 4)
        self.o_cli = take(self.li_words * 4)
        self.o_rmix = take(self.owned * CLUSTER * RECV * 4)
        self.o_rli = take(CLUSTER * self.li_words * 4)
        self.o_tot = take(2 * RECV * 4)
        self.o_rows = take(2 * MULT * MULT * 4)
        self.o_rstd = take(self.li_words * 4)
        self.o_hcw = take(32 * 4)
        self.o_bar = take(16, 8)
        self.smem_bytes = off

    @staticmethod
    def supports(hidden: int) -> bool:
        slice_ = hidden // CLUSTER
        return (
            hidden % 512 == 0
            and slice_ % 32 == 0
            and (MULT * slice_ // 16) % MMA_WARPS == 0
            and TILE * (slice_ // 16) <= THREADS
        )

    @cute.jit
    def __call__(
        self,
        x,
        residual,
        post,
        comb,
        pre_lag,
        weight,
        scale,
        base,
        norm_w,
        residual_out,
        layer_input,
        pre_out,
        post_out,
        comb_out,
        mxfp8_out: Int64,
        mxfp8_scale: Int64,
        rms_eps: Float32,
        hc_eps: Float32,
        sinkhorn_eps: Float32,
        post_mult: Float32,
        norm_eps: Float32,
        stream,
    ):
        tiles = (residual.shape[0] + TILE - 1) // TILE
        self.kernel(
            x,
            residual,
            post,
            comb,
            pre_lag,
            weight,
            scale,
            base,
            norm_w,
            residual_out,
            layer_input,
            pre_out,
            post_out,
            comb_out,
            mxfp8_out,
            mxfp8_scale,
            rms_eps,
            hc_eps,
            sinkhorn_eps,
            post_mult,
            norm_eps,
        ).launch(
            grid=[tiles * CLUSTER, 1, 1],
            block=[THREADS, 1, 1],
            cluster=[CLUSTER, 1, 1],
            stream=stream,
            use_pdl=TRTLLM_ENABLE_PDL,
        )

    @cute.kernel
    def kernel(
        self,
        x,
        residual,
        post,
        comb,
        pre_lag,
        weight,
        scale,
        base,
        norm_w,
        residual_out,
        layer_input,
        pre_out,
        post_out,
        comb_out,
        mxfp8_out: Int64,
        mxfp8_scale: Int64,
        rms_eps: Float32,
        hc_eps: Float32,
        sinkhorn_eps: Float32,
        post_mult: Float32,
        norm_eps: Float32,
    ):
        H = self.hidden
        S = self.slice
        PARTS = self.parts
        tid = cute.arch.thread_idx()[0]
        bid = cute.arch.block_idx()[0]
        rank = bid % CLUSTER
        t0 = (bid // CLUSTER) * TILE
        valid = cutlass.min(TILE, residual.shape[0] - t0)

        smem = SmemAllocator()
        sbase = smem.allocate(self.smem_bytes, 128).toint()
        s_w = sbase + self.o_w
        s_r = sbase + self.o_r
        s_x = sbase + self.o_x
        s_hcw = sbase + self.o_hcw
        bar_w = sbase + self.o_bar
        bar_x = bar_w + 8
        hslice = Int64(rank * S)
        owned_here = (TILE - rank + CLUSTER - 1) // CLUSTER

        w_chunks = 10
        if tid == 0:
            mbar_init(bar_w, 1)
            mbar_init(bar_x, 1)
            mbar_fence_init()
            mbar_expect_tx(bar_w, self.w_bytes)
            mbar_expect_tx(bar_x, owned_here * CLUSTER * RECV * 4 + CLUSTER * self.li_words * 4)
        cute.arch.sync_threads()
        # Peers push into bar_x: every CTA's barriers must be initialized before the first push.
        cute.arch.cluster_arrive_relaxed()

        # ---- the weight slice in a few bulk copies; hc scale and base to shared memory
        jw = tid - (THREADS - w_chunks)
        if jw >= 0:
            size = self.w_bytes // w_chunks
            bulk_g2s(
                s_w + jw * size,
                weight.iterator.toint() + Int64(rank) * self.w_bytes + jw * size,
                size,
                bar_w,
            )
        elif (tid >= 256) & (tid < 256 + 3 + MIXES):
            i = tid - 256
            if i < 3:
                sts_f32(s_hcw + i * 4, ldg_f32(scale.iterator.toint() + i * 4))
            else:
                sts_f32(s_hcw + i * 4, ldg_f32(base.iterator.toint() + (i - 3) * 4))

        # Everything above reads only weights; the activations come from the previous kernel.
        cute.arch.griddepcontrol_wait()

        # ---- elementwise pass: residual update, lagged collapse li, sums of squares. Thread (t, part)
        # handles the 8-wide chunks part and part + PARTS of token t, loaded straight to registers.
        t = tid // PARTS
        part = tid % PARTS
        if t < valid:
            gt = Int64(t0 + t)
            pre = ldg_v4_f32(pre_lag.iterator.toint() + gt * 16)
            pw = pre
            cm = [pre] * MULT
            if cutlass.const_expr(self.with_post):
                pw = ldg_v4_f32(post.iterator.toint() + gt * 16)
                cm = [ldg_v4_f32(comb.iterator.toint() + gt * 64 + j * 16) for j in range(MULT)]
            g_row = residual.iterator.toint() + (gt * MULT * H + hslice) * 2
            g_xrow = x.iterator.toint() + (gt * H + hslice) * 2
            rws = []
            xws = []
            for half in cutlass.range_constexpr(2):
                c = part + half * PARTS
                rws.append([ldg_v4_u32(g_row + j * H * 2 + c * 16) for j in range(MULT)])
                if cutlass.const_expr(self.with_post):
                    xws.append(ldg_v4_u32(g_xrow + c * 16))
            ssq_rc = (Float32(0.0), Float32(0.0))
            ssq_li = (Float32(0.0), Float32(0.0))
            for half in cutlass.range_constexpr(2):
                c = part + half * PARTS
                r_addr = s_r + t * self.r_row + (c // 4) * self.r_chunk + (c % 4) * 4
                cur = rws[half]
                if cutlass.const_expr(self.with_post):
                    cur = residual_update(pw, cm, xws[half], rws[half])
                    for s in cutlass.range_constexpr(MULT):
                        stg_v4_u32(
                            residual_out.iterator.toint()
                            + ((gt * MULT + s) * H + hslice + c * 8) * 2,
                            cur[s],
                        )
                # Word w of stream s lands in k16 block 2 * s + w // 2, fragment column half w % 2.
                for s in cutlass.range_constexpr(MULT):
                    for w in cutlass.range_constexpr(4):
                        sts_u32(r_addr + (2 * s + w // 2) * 32 + (w % 2) * 16, cur[s][w])
                li, ssq_rc, ssq_li = lagged_collapse(cur, pre, ssq_rc, ssq_li)
                sts_v4_u32(s_x + t * self.x_row + c * 16, li)
            sts_f32(sbase + self.o_prc + (part * TILE + t) * 4, ssq_rc[0] + ssq_rc[1])
            sts_f32(sbase + self.o_pli + (part * TILE + t) * 4, ssq_li[0] + ssq_li[1])
        cute.arch.sync_threads()

        # ---- mixer GEMM on m16n8k16 BF16: A = weight fragments, B = residual_cur tokens
        mbar_wait(bar_w, 0)
        warp = tid // 32
        lane = tid % 32
        g = lane // 4
        q = lane % 4
        # ldmatrix rows: lanes 0-7 / 8-15 / 16-23 / 24-31 = tokens 0-7 at k, 0-7 at k + 8,
        # 8-15 at k, 8-15 at k + 8
        mrow = (lane % 8) + (lane // 16) * 8
        mk = ((lane // 8) % 2) * 8
        if warp < MMA_WARPS:
            acc = [[[Float32(0.0)] * 4 for _ in range(self.n_tiles)] for _ in range(3)]
            for step in cutlass.range_constexpr(self.steps_per_warp):
                kk = warp * self.steps_per_warp + step
                frag = s_w + (kk * 32 + lane) * 48
                a = [lds_v4_u32(frag + 16 * mt) for mt in range(3)]
                b = []
                for pair in cutlass.range_constexpr((self.n_tiles + 1) // 2):
                    tok = cutlass.min(pair * 16 + mrow, TILE - 1)
                    m4 = ldsm_x4(
                        s_r + tok * self.r_row + kk // 8 * self.r_chunk + (kk % 8) * 32 + mk * 2
                    )
                    b.append((m4[0], m4[1]))
                    b.append((m4[2], m4[3]))
                for nt in cutlass.range_constexpr(self.n_tiles):
                    for mt in cutlass.range_constexpr(3):
                        acc[mt][nt] = list(mma_bf16(a[mt], b[nt], acc[mt][nt]))
            # Rows g / g + 8 of the three m tiles are high g, high g + 8 | high g + 16, low g |
            # low g + 8, low g + 16, so each thread holds both parts of mixes g, g + 8 and g + 16.
            cute.arch.barrier(barrier_id=1, number_of_threads=MMA_WARPS * 32)
            for nt in cutlass.range_constexpr(self.n_tiles):
                col = nt * 8 + 2 * q
                for mi in cutlass.range_constexpr(3):
                    hi_t, hi_r = ((0, 0), (0, 2), (1, 0))[mi]
                    lo_t, lo_r = ((1, 2), (2, 0), (2, 2))[mi]
                    sts_v2_f32(
                        s_w + ((warp * MIXES + mi * 8 + g) * self.cols + col) * 4,
                        acc[hi_t][nt][hi_r] + acc[lo_t][nt][lo_r],
                        acc[hi_t][nt][hi_r + 1] + acc[lo_t][nt][lo_r + 1],
                    )
        cute.arch.sync_threads()

        # ---- CTA partial per token: 24 mixes (warps in order), sums of squares of residual_cur and li
        s_cta = sbase + self.o_cta
        s_cli = sbase + self.o_cli
        for k in cutlass.range_constexpr((TILE * 26 + THREADS - 1) // THREADS):
            i = tid + k * THREADS
            if i < TILE * MIXES:
                n = i // TILE
                tt = i % TILE
                parts = [
                    lds_f32(s_w + ((w * MIXES + n) * self.cols + tt) * 4) for w in range(MMA_WARPS)
                ]
                v = Float32(0.0)
                for w in cutlass.range_constexpr(MMA_WARPS):
                    v = v + parts[w]
                sts_f32(s_cta + (tt * RECV + n) * 4, v)
            elif i < TILE * 26:
                ii = i - TILE * MIXES
                kind = ii // TILE
                tt = ii % TILE
                src = sbase + self.o_prc + kind * (self.o_pli - self.o_prc) + tt * 4
                parts = [lds_f32(src + j * TILE * 4) for j in range(PARTS)]
                v = Float32(0.0)
                for j in cutlass.range_constexpr(PARTS):
                    v = v + parts[j]
                if kind == 0:
                    sts_f32(s_cta + (tt * RECV + MIXES) * 4, v)
                else:
                    sts_f32(s_cli + tt * 4, v)
        cute.arch.sync_threads()

        # ---- push token t's partial to its owner (rank t % 16, slot t // 16) and the li sums to all
        cute.arch.cluster_wait()
        quads = RECV // 4
        li_quads = self.li_words // 4
        if tid < TILE * quads:
            tt = tid // quads
            j = tid % quads
            dst = sbase + self.o_rmix + (((tt // CLUSTER) * CLUSTER + rank) * RECV + j * 4) * 4
            st_async_v4(dst, lds_v4_u32(s_cta + (tt * RECV + j * 4) * 4), bar_x, tt % CLUSTER)
        elif tid < TILE * quads + CLUSTER * li_quads:
            i = tid - TILE * quads
            j = i % li_quads
            dst = sbase + self.o_rli + (rank * self.li_words + j * 4) * 4
            st_async_v4(dst, lds_v4_u32(s_cli + j * 16), bar_x, i // li_quads)
        mbar_wait(bar_x, 0, scope="cluster")
        cute.arch.cluster_arrive_relaxed()
        cute.arch.griddepcontrol_launch_dependents()

        # ---- totals in rank order: li rstd of every token, the 24 mixes and sum of squares of owned ones
        s_tot = sbase + self.o_tot
        s_rstd = sbase + self.o_rstd
        if tid < TILE:
            parts = [
                lds_f32(sbase + self.o_rli + (p * self.li_words + tid) * 4) for p in range(CLUSTER)
            ]
            v = Float32(0.0)
            for p in cutlass.range_constexpr(CLUSTER):
                v = v + parts[p]
            sts_f32(s_rstd + tid * 4, cute.math.rsqrt(v / Float32(H) + norm_eps))
        item = tid - 32
        if (item >= 0) & (item < self.owned * 25):
            o = item // 25
            vi = item % 25
            parts = [
                lds_f32(sbase + self.o_rmix + ((o * CLUSTER + p) * RECV + vi) * 4)
                for p in range(CLUSTER)
            ]
            v = Float32(0.0)
            for p in cutlass.range_constexpr(CLUSTER):
                v = v + parts[p]
            sts_f32(s_tot + (o * RECV + vi) * 4, v)
        cute.arch.sync_threads()

        if tid < 32:
            # 16 lanes per owned token
            self.coefficients(
                s_tot + (lane // 16) * RECV * 4,
                s_hcw,
                sbase + self.o_rows + (lane // 16) * 64,
                rank + (lane // 16) * CLUSTER,
                lane % 16,
                t0,
                valid,
                pre_out,
                post_out,
                comb_out,
                rms_eps,
                hc_eps,
                sinkhorn_eps,
                post_mult,
            )
        else:
            # layer_input = bf16(li * rstd * norm_w) for this CTA's hidden slice
            for k in cutlass.range_constexpr((TILE * self.chunks + THREADS - 33) // (THREADS - 32)):
                u = tid - 32 + k * (THREADS - 32)
                tt = u // self.chunks
                c = u % self.chunks
                if cutlass.const_expr(self.mxfp8):
                    # The 4 lanes of a 32-wide MXFP8 block share its maximum by shuffles, so every
                    # lane computes (past the last token on the tile's last row) and valid ones store.
                    ts = cutlass.min(tt, TILE - 1)
                    out = self.layer_input_words(s_x, s_rstd, norm_w, hslice, ts, c)
                    vals = []
                    for p in cutlass.range_constexpr(4):
                        vals.append(bf16x2_lo(out[p]))
                        vals.append(bf16x2_hi(out[p]))
                    amax = Float32(0.0)
                    for v in vals:
                        amax = cute.arch.fmax(amax, cute.arch.fmax(v, -v))
                    amax = cute.arch.fmax(amax, shfl_xor_f32(amax, 1))
                    amax = cute.arch.fmax(amax, shfl_xor_f32(amax, 2))
                    if tt < valid:
                        row = Int64(t0 + tt)
                        stg_v4_u32(
                            layer_input.iterator.toint() + (row * H + hslice + c * 8) * 2, out
                        )
                        w0, w1, sf = mxfp8_block(amax, vals)
                        stg_v2_u32(mxfp8_out + row * H + hslice + c * 8, w0, w1)
                        if c % 4 == 0:
                            kb = (hslice + c * 8) // 32
                            stg_u8(
                                mxfp8_scale
                                + (row // 128) * (H // 128 * 512)
                                + (kb // 4) * 512
                                + (row % 32) * 16
                                + (row % 128) // 32 * 4
                                + kb % 4,
                                sf,
                            )
                elif tt < valid:
                    out = self.layer_input_words(s_x, s_rstd, norm_w, hslice, tt, c)
                    stg_v4_u32(
                        layer_input.iterator.toint() + (Int64(t0 + tt) * H + hslice + c * 8) * 2,
                        out,
                    )
        # No CTA leaves while its peers may still read its barriers.
        cute.arch.cluster_wait()

    @cute.jit
    def layer_input_words(self, s_x, s_rstd, norm_w, hslice, tt, c):
        """BF16 pairs of ``li * rstd * norm_w`` for 8-wide chunk ``c`` of tile token ``tt``."""
        li = lds_v4_u32(s_x + tt * self.x_row + c * 16)
        nw = ldg_v4_u32(norm_w.iterator.toint() + hslice * 2 + c * 16)
        rstd = lds_f32(s_rstd + tt * 4)
        out = []
        for p in cutlass.range_constexpr(4):
            lo = (bf16x2_lo(li[p]) * rstd) * bf16x2_lo(nw[p])
            hi = (bf16x2_hi(li[p]) * rstd) * bf16x2_hi(nw[p])
            out.append(pack_bf16x2(lo, hi))
        return out

    @cute.jit
    def coefficients(
        self,
        tot,
        s_hcw,
        s_rows,
        tt,
        r,
        t0,
        valid,
        pre_out,
        post_out,
        comb_out,
        rms_eps: Float32,
        hc_eps: Float32,
        eps: Float32,
        post_mult: Float32,
    ):
        """pre / post / Sinkhorn comb of token ``tt`` from its mixes, in ``mhc_fused_hc``'s order.

        Lane ``r`` of the token's 16: 0-3 pre, 4-7 post, 8-11 the softmax of comb row ``r - 8``;
        lane 12 then runs the Sinkhorn iterations on the four rows. With ``sinkhorn_iters == 0``
        lanes 8-11 store the softmax rows to ``comb_out`` for ``MhcSinkhornKernel``.
        """
        rstd = cute.math.rsqrt(lds_f32(tot + MIXES * 4) / Float32(MULT * self.hidden) + rms_eps)
        gt = Int64(t0 + tt)
        if r < 2 * MULT:
            v = cute.math.fma(
                lds_f32(tot + r * 4) * rstd,
                lds_f32(s_hcw + (r // MULT) * 4),
                lds_f32(s_hcw + (3 + r) * 4),
            )
            if r < MULT:
                if tt < valid:
                    stg_f32(pre_out.iterator.toint() + (gt * MULT + r) * 4, sigmoid(v) + hc_eps)
            elif tt < valid:
                stg_f32(
                    post_out.iterator.toint() + (gt * MULT + r - MULT) * 4, sigmoid(v) * post_mult
                )
        elif r < 3 * MULT:
            i = r - 2 * MULT
            s2 = lds_f32(s_hcw + 2 * 4)
            row = [
                cute.math.fma(
                    lds_f32(tot + (8 + 4 * i + k) * 4) * rstd,
                    s2,
                    lds_f32(s_hcw + (11 + 4 * i + k) * 4),
                )
                for k in range(MULT)
            ]
            row_max = cute.arch.fmax(cute.arch.fmax(row[0], row[1]), cute.arch.fmax(row[2], row[3]))
            row = [cute.math.exp(v - row_max) for v in row]
            inv = rcp(row[0] + row[1] + row[2] + row[3])
            row = [cute.math.fma(v, inv, eps) for v in row]
            if cutlass.const_expr(self.sinkhorn_iters == 0):
                if tt < valid:
                    stg_v4_u32(
                        comb_out.iterator.toint() + (gt * 16 + i * 4) * 4,
                        [f32_bits(v) for v in row],
                    )
            else:
                for k in cutlass.range_constexpr(MULT):
                    sts_f32(s_rows + (4 * i + k) * 4, row[k])
        if cutlass.const_expr(self.sinkhorn_iters > 0):
            cute.arch.sync_warp()
            if r == 3 * MULT:
                out = self.sinkhorn([lds_f32(s_rows + k * 4) for k in range(MULT * MULT)], eps)
                if tt < valid:
                    for i in cutlass.range_constexpr(MULT):
                        stg_v4_u32(
                            comb_out.iterator.toint() + (gt * 16 + i * 4) * 4,
                            [f32_bits(v) for v in out[4 * i : 4 * i + 4]],
                        )

    @cute.jit
    def sinkhorn(self, c, eps: Float32):
        """A column normalization, then ``sinkhorn_iters - 1`` row and column normalizations."""
        c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15 = self.col_normalize(
            c, eps
        )
        for _ in cutlass.range(self.sinkhorn_iters - 1, unroll=1):
            cc = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15]
            rr = []
            for i in cutlass.range_constexpr(MULT):
                rinv = rcp(cc[4 * i] + cc[4 * i + 1] + cc[4 * i + 2] + cc[4 * i + 3] + eps)
                rr.extend([cc[4 * i + k] * rinv for k in range(MULT)])
            c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15 = (
                self.col_normalize(rr, eps)
            )
        return [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15]

    @staticmethod
    def col_normalize(c, eps):
        """Column sums (r0 + r1) + (r2 + r3), ``mhc_fused_hc``'s shuffle order."""
        out = list(c)
        for k in range(MULT):
            inv = rcp((c[k] + c[4 + k]) + (c[8 + k] + c[12 + k]) + eps)
            for i in range(MULT):
                out[4 * i + k] = c[4 * i + k] * inv
        return out


class MhcStreamKernel:
    """Batches above one wave: persistent CTAs per hidden slice stream groups of 8 tokens.

    CTA ``(r, p)`` keeps slice ``r`` of the prepared weight in shared memory and takes token groups
    ``p, p + ctas, ...``. A warp runs a whole group over the slice's 32-wide chunks: lane ``(n, q)``
    owns hidden ``[32 c + 8 q, 32 c + 8 q + 8)`` of token ``n`` in chunk ``c``, which is its B fragment
    of the chunk's eight k16 blocks, while the next chunk's activations are in flight. It writes li
    unnormalized to ``layer_input`` and one partial per token and slice (24 mixes, sums of squares of
    residual_cur and li) that ``MhcFinalizeKernel`` reduces in slice order.
    """

    def __init__(self, hidden: int, with_post: bool, ctas: int):
        self.hidden = hidden
        self.with_post = with_post
        self.ctas = ctas
        self.slice = hidden // CLUSTER
        self.chunks = self.slice // 32
        self.w_bytes = MULT * self.slice // 16 * 32 * 48

    @cute.jit
    def __call__(
        self, x, residual, post, comb, pre_lag, weight, residual_out, layer_input, partials, stream
    ):
        self.kernel(
            x, residual, post, comb, pre_lag, weight, residual_out, layer_input, partials
        ).launch(
            grid=[CLUSTER * self.ctas, 1, 1],
            block=[STREAM_WARPS * 32, 1, 1],
            stream=stream,
            use_pdl=TRTLLM_ENABLE_PDL,
        )

    @cute.kernel
    def kernel(self, x, residual, post, comb, pre_lag, weight, residual_out, layer_input, partials):
        H = self.hidden
        tid = cute.arch.thread_idx()[0]
        bid = cute.arch.block_idx()[0]
        r = bid % CLUSTER
        p = bid // CLUSTER
        M = residual.shape[0]
        groups = ((M + 7) // 8 - p + self.ctas - 1) // self.ctas

        smem = SmemAllocator()
        s_w = smem.allocate(self.w_bytes + 8, 128).toint()
        bar_w = s_w + self.w_bytes
        if tid == 0:
            mbar_init(bar_w, 1)
            mbar_fence_init()
            mbar_expect_tx(bar_w, self.w_bytes)
        cute.arch.sync_threads()
        copies = 10
        jw = tid - (STREAM_WARPS * 32 - copies)
        if jw >= 0:
            size = self.w_bytes // copies
            bulk_g2s(
                s_w + jw * size,
                weight.iterator.toint() + Int64(r) * self.w_bytes + jw * size,
                size,
                bar_w,
            )
        # The weight is prepared at load time; the activations come from the previous kernel.
        cute.arch.griddepcontrol_wait()

        warp = tid // 32
        lane = tid % 32
        n = lane // 4
        q = lane % 4
        waited = Int32(0)
        act = self.load_chunk(
            x, residual, p + cutlass.min(warp, groups - 1) * self.ctas, 0, r, n, q, M
        )
        for u in cutlass.range(warp, groups, STREAM_WARPS, unroll=1):
            grp = p + u * self.ctas
            t = grp * 8 + n
            tc = Int64(cutlass.min(t, M - 1))
            pre = ldg_v4_f32(pre_lag.iterator.toint() + tc * 16)
            pw = pre
            cm = [pre] * MULT
            if cutlass.const_expr(self.with_post):
                pw = ldg_v4_f32(post.iterator.toint() + tc * 16)
                cm = [ldg_v4_f32(comb.iterator.toint() + tc * 64 + j * 16) for j in range(MULT)]
            acc = [[Float32(0.0)] * 4 for _ in range(3)]
            ssq_rc = (Float32(0.0), Float32(0.0))
            ssq_li = (Float32(0.0), Float32(0.0))
            for c in cutlass.range_constexpr(self.chunks):
                h0 = Int64(r * self.slice + c * 32 + q * 8)
                if cutlass.const_expr(c + 1 < self.chunks):
                    nxt = self.load_chunk(x, residual, grp, c + 1, r, n, q, M)
                else:
                    nxt_u = cutlass.min(u + STREAM_WARPS, groups - 1)
                    nxt = self.load_chunk(x, residual, p + nxt_u * self.ctas, 0, r, n, q, M)
                cur = act[:MULT]
                if cutlass.const_expr(self.with_post):
                    cur = residual_update(pw, cm, act[MULT], act[:MULT])
                    if t < M:
                        for s in cutlass.range_constexpr(MULT):
                            stg_v4_u32(
                                residual_out.iterator.toint()
                                + ((Int64(t) * MULT + s) * H + h0) * 2,
                                cur[s],
                            )
                li, ssq_rc, ssq_li = lagged_collapse(cur, pre, ssq_rc, ssq_li)
                if t < M:
                    stg_v4_u32(layer_input.iterator.toint() + (Int64(t) * H + h0) * 2, li)
                if cutlass.const_expr(c == 0):
                    if waited == 0:
                        mbar_wait(bar_w, 0)
                        waited = Int32(1)
                # k16 block kb of the chunk: stream kb // 2, hidden half kb % 2 of every lane's 8.
                frag = s_w + (c * 8 * 32 + lane) * 48
                for kb in cutlass.range_constexpr(8):
                    a = [lds_v4_u32(frag + kb * 32 * 48 + 16 * mt) for mt in range(3)]
                    b = (cur[kb // 2][2 * (kb % 2)], cur[kb // 2][2 * (kb % 2) + 1])
                    for mt in cutlass.range_constexpr(3):
                        acc[mt] = list(mma_bf16(a[mt], b, acc[mt]))
                act = nxt
            src = ssq_rc[0] + ssq_rc[1]
            sli = ssq_li[0] + ssq_li[1]
            for m in cutlass.range_constexpr(2):
                src = src + shfl_xor_f32(src, 1 << m)
                sli = sli + shfl_xor_f32(sli, 1 << m)
            # The accumulator columns are tokens 2q and 2q + 1 of the group (see MhcBoundaryKernel).
            g = lane // 4
            for e in cutlass.range_constexpr(2):
                tok = grp * 8 + 2 * q + e
                if tok < M:
                    dst = partials.iterator.toint() + (Int64(tok) * CLUSTER + r) * PART * 4
                    for mi in cutlass.range_constexpr(3):
                        hi_t, hi_r = ((0, 0), (0, 2), (1, 0))[mi]
                        lo_t, lo_r = ((1, 2), (2, 0), (2, 2))[mi]
                        stg_f32(dst + (mi * 8 + g) * 4, acc[hi_t][hi_r + e] + acc[lo_t][lo_r + e])
            if (q == 0) & (t < M):
                dst = partials.iterator.toint() + (Int64(t) * CLUSTER + r) * PART * 4
                stg_f32(dst + MIXES * 4, src)
                stg_f32(dst + (MIXES + 1) * 4, sli)
        # Triggered last: dependents launched earlier would hold SMs while they wait.
        cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def load_chunk(self, x, residual, grp, c, r, n, q, M):
        """This lane's residual (and x) words of chunk ``c`` of token group ``grp``."""
        H = self.hidden
        tc = Int64(cutlass.min(grp * 8 + n, M - 1))
        h0 = Int64(r * self.slice + c * 32 + q * 8)
        row = residual.iterator.toint() + (tc * MULT * H + h0) * 2
        out = [ldg_v4_u32_ordered(row + j * H * 2) for j in range(MULT)]
        if cutlass.const_expr(self.with_post):
            out.append(ldg_v4_u32_ordered(x.iterator.toint() + (tc * H + h0) * 2))
        return out


class MhcFinalizeKernel(MhcBoundaryKernel):
    """``MhcStreamKernel``'s second launch, ``FINAL_TOKENS`` tokens per CTA.

    Warps ``2j`` and ``2j + 1`` normalize token ``j``'s li in place with all their loads in flight; the
    last four warps sum the slice partials in slice order and run the coefficients, 16 lanes a token.
    """

    @cute.jit
    def __call__(
        self,
        partials,
        scale,
        base,
        norm_w,
        layer_input,
        pre_out,
        post_out,
        comb_out,
        rms_eps: Float32,
        hc_eps: Float32,
        sinkhorn_eps: Float32,
        post_mult: Float32,
        norm_eps: Float32,
        stream,
    ):
        self.kernel(
            partials,
            scale,
            base,
            norm_w,
            layer_input,
            pre_out,
            post_out,
            comb_out,
            rms_eps,
            hc_eps,
            sinkhorn_eps,
            post_mult,
            norm_eps,
        ).launch(
            grid=[(partials.shape[0] + FINAL_TOKENS - 1) // FINAL_TOKENS, 1, 1],
            block=[(2 * FINAL_TOKENS + 4) * 32, 1, 1],
            stream=stream,
            use_pdl=TRTLLM_ENABLE_PDL,
        )

    @cute.kernel
    def kernel(
        self,
        partials,
        scale,
        base,
        norm_w,
        layer_input,
        pre_out,
        post_out,
        comb_out,
        rms_eps: Float32,
        hc_eps: Float32,
        sinkhorn_eps: Float32,
        post_mult: Float32,
        norm_eps: Float32,
    ):
        H = self.hidden
        threads = (2 * FINAL_TOKENS + 4) * 32
        tid = cute.arch.thread_idx()[0]
        t0 = cute.arch.block_idx()[0] * FINAL_TOKENS
        valid = cutlass.min(FINAL_TOKENS, partials.shape[0] - t0)
        warp = tid // 32
        lane = tid % 32

        smem = SmemAllocator()
        s_tot = smem.allocate(FINAL_TOKENS * (PART + 16) * 4 + 32 * 4 + H * 2, 128).toint()
        s_rows = s_tot + FINAL_TOKENS * PART * 4
        s_hcw = s_rows + FINAL_TOKENS * 64
        s_nw = s_hcw + 32 * 4
        if tid < 3 + MIXES:
            if tid < 3:
                sts_f32(s_hcw + tid * 4, ldg_f32(scale.iterator.toint() + tid * 4))
            else:
                sts_f32(s_hcw + tid * 4, ldg_f32(base.iterator.toint() + (tid - 3) * 4))
        for k in cutlass.range_constexpr((H // 8 + threads - 1) // threads):
            c = tid + k * threads
            if c < H // 8:
                sts_v4_u32(s_nw + c * 16, ldg_v4_u32(norm_w.iterator.toint() + c * 16))
        cute.arch.griddepcontrol_wait()

        if warp < 2 * FINAL_TOKENS:
            j = warp // 2
            row = layer_input.iterator.toint() + Int64(t0 + cutlass.min(j, valid - 1)) * H * 2
            cs = [lane + (warp % 2 + 2 * i) * 32 for i in range(H // 8 // 64)]
            lis = [ldg_v4_u32_ordered(row + c * 16) for c in cs]
            cute.arch.sync_threads()
            if j < valid:
                rstd = cute.math.rsqrt(
                    lds_f32(s_tot + (j * PART + MIXES + 1) * 4) / Float32(H) + norm_eps
                )
                for i in cutlass.range_constexpr(len(cs)):
                    nw = lds_v4_u32(s_nw + cs[i] * 16)
                    out = []
                    for p in cutlass.range_constexpr(4):
                        lo = (bf16x2_lo(lis[i][p]) * rstd) * bf16x2_lo(nw[p])
                        hi = (bf16x2_hi(lis[i][p]) * rstd) * bf16x2_hi(nw[p])
                        out.append(pack_bf16x2(lo, hi))
                    stg_v4_u32(row + cs[i] * 16, out)
        else:
            i = tid - 2 * FINAL_TOKENS * 32
            if i < FINAL_TOKENS * PART // 4:
                tt = i // (PART // 4)
                src = (
                    partials.iterator.toint()
                    + (
                        (Int64(t0 + cutlass.min(tt, valid - 1)) * CLUSTER) * PART
                        + i % (PART // 4) * 4
                    )
                    * 4
                )
                parts = [ldg_v4_f32(src + r * PART * 4) for r in range(CLUSTER)]
                tot = [Float32(0.0)] * 4
                for r in cutlass.range_constexpr(CLUSTER):
                    tot = [tot[k] + parts[r][k] for k in range(4)]
                for k in cutlass.range_constexpr(4):
                    sts_f32(s_tot + (tt * PART + i % (PART // 4) * 4 + k) * 4, tot[k])
            cute.arch.sync_threads()
            self.coefficients(
                s_tot + (i // 16) * PART * 4,
                s_hcw,
                s_rows + (i // 16) * 64,
                i // 16,
                i % 16,
                t0,
                valid,
                pre_out,
                post_out,
                comb_out,
                rms_eps,
                hc_eps,
                sinkhorn_eps,
                post_mult,
            )


class MhcSinkhornKernel(MhcBoundaryKernel):
    """The Sinkhorn iterations of the comb rows that a boundary run with ``sinkhorn_iters = 0`` wrote,
    in place, one token per thread."""

    def __init__(self, sinkhorn_iters: int):
        self.sinkhorn_iters = sinkhorn_iters

    @cute.jit
    def __call__(self, comb, sinkhorn_eps: Float32, stream):
        self.kernel(comb, sinkhorn_eps).launch(
            grid=[(comb.shape[0] + 127) // 128, 1, 1], block=[128, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, comb, sinkhorn_eps: Float32):
        t = cute.arch.block_idx()[0] * 128 + cute.arch.thread_idx()[0]
        if t < comb.shape[0]:
            row = comb.iterator.toint() + Int64(t) * 64
            c = [v for i in range(MULT) for v in ldg_v4_f32(row + i * 16)]
            out = self.sinkhorn(c, sinkhorn_eps)
            for i in cutlass.range_constexpr(MULT):
                stg_v4_u32(row + i * 16, [f32_bits(v) for v in out[4 * i : 4 * i + 4]])


class Nvfp4QuantizeKernel:
    """``fp4_quantize(x, global_scale, 16, False, False)`` of contiguous BF16 rows: E2M1 bytes
    ``[rows, hidden / 2]`` and one E4M3 scale per 16 elements ``[rows, hidden / 16]`` (row-major).

    A thread converts ``CHUNKS`` chunks of 8 elements, ``THREADS`` chunks apart, and issues all their
    loads first; lanes ``2m`` and ``2m + 1`` hold the two halves of a 16-element block.
    """

    THREADS = 256
    CHUNKS = 4

    @cute.jit
    def __call__(self, x, fp4_out, sf_out, global_scale, stream):
        chunks = x.shape[0] * (x.shape[1] // 8)
        per_cta = self.THREADS * self.CHUNKS
        self.kernel(x, fp4_out, sf_out, global_scale, chunks).launch(
            grid=[ceil_div(chunks, per_cta), 1, 1],
            block=[self.THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(self, x, fp4_out, sf_out, global_scale, chunks: Int32):
        tid = cute.arch.thread_idx()[0]
        first = cute.arch.block_idx()[0] * (self.THREADS * self.CHUNKS) + tid
        g = ldg_f32(global_scale.iterator.toint())
        words = []
        for i in cutlass.range_constexpr(self.CHUNKS):
            # Chunks past the end load the last one; their lane pairs are past the end too.
            c = cutlass.min(first + i * self.THREADS, chunks - 1)
            words.append(ldg_v4_u32_ordered(x.iterator.toint() + Int64(c) * 16))
        for i in cutlass.range_constexpr(self.CHUNKS):
            c = first + i * self.THREADS
            amax = bf16x2_absmax4(words[i])
            values = []
            for p in cutlass.range_constexpr(4):
                values += [bf16x2_lo(words[i][p]), bf16x2_hi(words[i][p])]
            word, sf = nvfp4_block(amax, shfl_xor_u32(amax, 1), g, values)
            if c < chunks:
                stg_u32(fp4_out.iterator.toint() + Int64(c) * 4, word)
                if tid % 2 == 0:
                    stg_u8(sf_out.iterator.toint() + Int64(c // 2), sf)


def prepare_boundary_weight(fn: torch.Tensor) -> torch.Tensor:
    """``fn [24, 4 * hidden]`` -> int32 mma A fragments ``[16 slices, k16 steps, 32 lanes, 12]``.

    The weight is rounded to TF32 (nearest, ties away, as ``cvt.rna.tf32.f32``) and split into
    BF16 high and low parts whose sum is exact; the 12 words of a lane are its A fragments of the
    three m16 tiles of the stacked ``[high; low]`` rows. k16 step ``8 c + kb`` of a slice is chunk
    ``c`` (hidden ``[32 c, 32 c + 32)``), stream ``kb // 2``; fragment columns ``2q, 2q + 1`` and
    ``2q + 8, 2q + 9`` hold hidden ``8 q + 4 (kb % 2) + {0, 1}`` and ``+ {2, 3}`` of the chunk.
    """
    hidden = fn.shape[1] // MULT
    S = hidden // CLUSTER
    bits = fn.contiguous().view(torch.int32)
    rounded = ((bits + 0x1000) & -0x2000).view(torch.float32)
    hi = rounded.to(torch.bfloat16)
    lo = (rounded - hi.float()).to(torch.bfloat16)
    pairs = torch.cat([hi, lo]).view(torch.int32)
    dev = fn.device
    rank = torch.arange(CLUSTER, device=dev).view(-1, 1, 1, 1)
    kk = torch.arange(MULT * S // 16, device=dev).view(1, -1, 1, 1)
    lane = torch.arange(32, device=dev).view(1, 1, -1, 1)
    g, q = lane // 4, lane % 4
    c, kb = kk // 8, kk % 8
    base = (kb // 2) * hidden + rank * S + c * 32 + q * 8 + (kb % 2) * 4
    rows = torch.tensor([0, 8, 0, 8, 16, 24, 16, 24, 32, 40, 32, 40], device=dev).view(1, 1, 1, -1)
    upper = torch.tensor([0, 0, 2, 2, 0, 0, 2, 2, 0, 0, 2, 2], device=dev).view(1, 1, 1, -1)
    return pairs[rows + g, (base + upper) // 2].contiguous()


_COMPILED = set()


def compiled(
    hidden: int,
    with_post: bool,
    sinkhorn_iters: int,
    tokens: int,
    deferred: bool = False,
    mxfp8: bool = False,
) -> bool:
    """Whether the kernels for this variant and batch are compiled (not allowed inside graph capture);
    ``deferred`` asks for the boundary without Sinkhorn iterations and ``mhc_sinkhorn``, ``mxfp8``
    for the single-pass kernel's MXFP8 ``layer_input``."""
    if deferred:
        return ("sinkhorn", sinkhorn_iters) in _COMPILED and compiled(
            hidden, with_post, 0, tokens, mxfp8=mxfp8
        )
    if tokens <= MAX_TOKENS:
        return (hidden, with_post, sinkhorn_iters, mxfp8) in _COMPILED
    return ("stream", hidden, with_post) in _COMPILED and (
        "finalize",
        hidden,
        sinkhorn_iters,
    ) in _COMPILED


def _mat(dtype, *shape):
    return cute.runtime.make_fake_compact_tensor(
        dtype, shape, stride_order=tuple(reversed(range(len(shape)))), assumed_align=16
    )


@functools.cache
def _compile(hidden: int, with_post: bool, sinkhorn_iters: int, mxfp8: bool = False):
    tokens = cute.sym_int()
    bf16 = cutlass.BFloat16
    vec = (tokens, MULT)
    fn = cute.compile(
        MhcBoundaryKernel(hidden, with_post, sinkhorn_iters, mxfp8),
        _mat(bf16, tokens, hidden),
        _mat(bf16, tokens, MULT, hidden),
        _mat(Float32, *vec),
        _mat(Float32, tokens, MULT * MULT),
        _mat(Float32, *vec),
        _mat(Int32, CLUSTER, MULT * (hidden // CLUSTER) // 16, 32, 12),
        _mat(Float32, 3),
        _mat(Float32, MIXES),
        _mat(bf16, hidden),
        _mat(bf16, tokens, MULT, hidden),
        _mat(bf16, tokens, hidden),
        _mat(Float32, *vec),
        _mat(Float32, *vec),
        _mat(Float32, tokens, MULT * MULT),
        Int64(0),
        Int64(0),
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )
    _COMPILED.add((hidden, with_post, sinkhorn_iters, mxfp8))
    return fn


@functools.cache
def _compile_stream(hidden: int, with_post: bool):
    tokens = cute.sym_int()
    bf16 = cutlass.BFloat16
    ctas = torch.cuda.get_device_properties().multi_processor_count // CLUSTER
    fn = cute.compile(
        MhcStreamKernel(hidden, with_post, ctas),
        _mat(bf16, tokens, hidden),
        _mat(bf16, tokens, MULT, hidden),
        _mat(Float32, tokens, MULT),
        _mat(Float32, tokens, MULT * MULT),
        _mat(Float32, tokens, MULT),
        _mat(Int32, CLUSTER, MULT * (hidden // CLUSTER) // 16, 32, 12),
        _mat(bf16, tokens, MULT, hidden),
        _mat(bf16, tokens, hidden),
        _mat(Float32, tokens, CLUSTER, PART),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )
    _COMPILED.add(("stream", hidden, with_post))
    return fn


@functools.cache
def _compile_finalize(hidden: int, sinkhorn_iters: int):
    tokens = cute.sym_int()
    vec = (tokens, MULT)
    fn = cute.compile(
        MhcFinalizeKernel(hidden, True, sinkhorn_iters),
        _mat(Float32, tokens, CLUSTER, PART),
        _mat(Float32, 3),
        _mat(Float32, MIXES),
        _mat(cutlass.BFloat16, hidden),
        _mat(cutlass.BFloat16, tokens, hidden),
        _mat(Float32, *vec),
        _mat(Float32, *vec),
        _mat(Float32, tokens, MULT * MULT),
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )
    _COMPILED.add(("finalize", hidden, sinkhorn_iters))
    return fn


@functools.cache
def _compile_sinkhorn(sinkhorn_iters: int):
    fn = cute.compile(
        MhcSinkhornKernel(sinkhorn_iters),
        _mat(Float32, cute.sym_int(), MULT * MULT),
        Float32(0.0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )
    _COMPILED.add(("sinkhorn", sinkhorn_iters))
    return fn


@functools.cache
def _compile_nvfp4_quantize(hidden: int):
    rows = cute.sym_int()
    fn = cute.compile(
        Nvfp4QuantizeKernel(),
        _mat(cutlass.BFloat16, rows, hidden),
        _mat(cutlass.Uint8, rows, hidden // 2),
        _mat(cutlass.Uint8, rows, hidden // 16),
        _mat(Float32, 1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )
    _COMPILED.add(("nvfp4_quantize", hidden))
    return fn


def mhc_boundary(
    x_prev,
    residual_prev,
    post_prev,
    comb_prev,
    pre_lag,
    weight,
    scale,
    base,
    norm_weight,
    rms_eps,
    hc_eps,
    sinkhorn_eps,
    post_mult,
    norm_eps,
    sinkhorn_iters,
    mxfp8=False,
):
    """One lagged boundary; ``x_prev is None`` collapses ``residual_prev`` without a post update.

    Takes contiguous ``x_prev [B, hidden]``, ``residual_prev [B, 4, hidden]`` (BF16), ``post_prev
    [B, 4]``, ``comb_prev [B, 16]``, ``pre_lag [B, 4]`` (FP32), ``weight =
    prepare_boundary_weight(fn)`` and the BF16 ``norm_weight`` folded into ``layer_input``.
    Up to ``MAX_TOKENS`` tokens run the single-pass kernel, more the stream and finalize kernels.
    With ``sinkhorn_iters = 0``, ``comb_cur`` holds the row softmax for ``mhc_sinkhorn``.
    Returns ``(residual_cur, pre_cur, post_cur, comb_cur, layer_input, layer_input_mxfp8)``; with
    ``mxfp8`` the single-pass kernel also returns ``torch.ops.trtllm.mxfp8_quantize(layer_input,
    True)`` as ``layer_input_mxfp8`` (E4M3 values, swizzled UE8M0 scales), which is None otherwise.
    """
    tokens, mult, hidden = residual_prev.shape
    with_post = x_prev is not None
    dev = residual_prev.device
    layer_input = torch.empty(tokens, hidden, dtype=torch.bfloat16, device=dev)
    pre_cur, post_cur = torch.empty(2, tokens, mult, dtype=torch.float32, device=dev)
    comb_cur = torch.empty(tokens, mult * mult, dtype=torch.float32, device=dev)
    residual_cur = torch.empty_like(residual_prev) if with_post else residual_prev
    activations = (
        x_prev if with_post else layer_input,
        residual_prev,
        post_prev if with_post else pre_cur,
        comb_prev if with_post else comb_cur,
        pre_lag,
        weight,
    )
    layer_input_mxfp8 = None
    if tokens <= MAX_TOKENS:
        if mxfp8:
            layer_input_mxfp8 = (
                torch.empty(tokens, hidden, dtype=torch.float8_e4m3fn, device=dev),
                torch.empty(pad_up(tokens, 128) * hidden // 32, dtype=torch.uint8, device=dev),
            )
        _compile(hidden, with_post, sinkhorn_iters, mxfp8)(
            *activations,
            scale,
            base,
            norm_weight,
            residual_cur,
            layer_input,
            pre_cur,
            post_cur,
            comb_cur,
            layer_input_mxfp8[0].data_ptr() if mxfp8 else 0,
            layer_input_mxfp8[1].data_ptr() if mxfp8 else 0,
            rms_eps,
            hc_eps,
            sinkhorn_eps,
            post_mult,
            norm_eps,
        )
    else:
        partials = torch.empty(tokens, CLUSTER, PART, dtype=torch.float32, device=dev)
        _compile_stream(hidden, with_post)(*activations, residual_cur, layer_input, partials)
        _compile_finalize(hidden, sinkhorn_iters)(
            partials,
            scale,
            base,
            norm_weight,
            layer_input,
            pre_cur,
            post_cur,
            comb_cur,
            rms_eps,
            hc_eps,
            sinkhorn_eps,
            post_mult,
            norm_eps,
        )
    return residual_cur, pre_cur, post_cur, comb_cur, layer_input, layer_input_mxfp8


def nvfp4_quantize_compiled(hidden: int) -> bool:
    """Whether ``nvfp4_quantize`` is compiled for ``hidden`` (not allowed inside graph capture)."""
    return ("nvfp4_quantize", hidden) in _COMPILED


def nvfp4_quantize(
    x: torch.Tensor, global_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``fp4_quantize(x, global_scale, 16, False, False)`` of contiguous BF16 ``x [rows, hidden]`` with
    an FP32 ``global_scale [1]``: E2M1 ``[rows, hidden / 2]`` and E4M3 scales ``[rows, hidden / 16]``
    (uint8)."""
    rows, hidden = x.shape
    fp4 = torch.empty(rows, hidden // 2, dtype=torch.uint8, device=x.device)
    sf = torch.empty(rows, hidden // 16, dtype=torch.uint8, device=x.device)
    if rows:
        _compile_nvfp4_quantize(hidden)(x, fp4, sf, global_scale)
    return fp4, sf


def mhc_sinkhorn(comb, sinkhorn_eps, sinkhorn_iters):
    """Runs the Sinkhorn iterations of ``comb [B, 16]`` from ``mhc_boundary(..., sinkhorn_iters=0)`` in
    place."""
    _compile_sinkhorn(sinkhorn_iters)(comb, sinkhorn_eps)
