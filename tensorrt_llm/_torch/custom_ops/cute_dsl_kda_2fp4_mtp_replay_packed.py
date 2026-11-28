# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed-channel native 2FP4 KDA MTP replay for grids above the warp-specialized range."""

# Fused Conv4/SiLU + QK norm + native-NVFP4 KDA MTP replay + RMSNorm/sigmoid gate.
# One CTA (128 threads) per (head, batch-row).  Both checkpoint projections (state x K-operand
# and state x Q-operand, all T tokens at once) run as a single tcgen05.mma kind::mxf4nvf4
# block16 GEMM: A = primary checkpoint (E2M1, UE4M3 block-16 scales from the pool),
# B = per-token quantized [k_t; q_t] rows (E2M1 + UE4M3 block-16 scales), D = FP32 in TMEM.
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Boolean, Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir import ir as mlir_ir
from cutlass._mlir.dialects import nvvm as nvvm_raw
from cutlass.cutlass_dsl import T as MT
from cutlass.experimental import primitives as prims
from cutlass.experimental.cuda.tensor_map import (
    TensorMap,
    TensorMapSwizzle,
    create_tensor_map_tiled,
)

from ..flashinfer_utils import get_env_enable_pdl
from .cute_dsl_kda_2fp4_commit import STATE_BYTES

KD = 128  # K = V = 128
HROWS = 16  # history window W
NT = 128  # threads per CTA
ASTR = 136  # padded bf16 row stride for dot-product operands (272 B, conflict-free)
CTSTR = 24  # padded bf16 row stride of the transposed history coefficients (48 B, conflict-free)
LOG2E = 1.4426950408889634
INV2688 = 1.0 / (6 * 448)
GMEM = cutlass.AddressSpace.gmem
# packed launches with more CTAs copy only the live history rows (length-dependent copies)
LIVE_MIN_GRID = 0
TMEM_COLS = 32  # [0,16) accumulator, [16,24) SFA (2 k-blocks x 4 cols), [24,32) SFB


# Row flags, slots, replay slots and history lengths are a few bytes read by every
# KDA layer; keep them in L2 across the weight traffic between layers.
_KEEP_S32_PTX = """{
.reg .b64 pol;
createpolicy.fractional.L2::evict_last.b64 pol, 1.0;
ld.global.L2::cache_hint.b32 {$w0}, [{$r0}], pol;
}"""
_KEEP_U8_PTX = """{
.reg .b64 pol;
.reg .b16 t;
createpolicy.fractional.L2::evict_last.b64 pol, 1.0;
ld.global.L2::cache_hint.b8 t, [{$r0}], pol;
cvt.u32.u16 {$w0}, t;
}"""


def _evict_first_policy():
    # L2 policy for streaming data (checkpoint / history rows read once): keep the producer's
    # projections (read by later CTAs) resident instead
    return cute.arch.inline_ptx(
        "createpolicy.fractional.L2::evict_first.b64 {$w0}, 1.0;", write_only_types=[Int64]
    )


def _cp_async16_ef(dst, src, nbytes, pol):
    # 16 B cp.async.cg with an L2 cache policy (zero-fills when nbytes < 16)
    cute.arch.inline_ptx(
        "cp.async.cg.shared.global.L2::cache_hint [{$r0}], [{$r1}], 16, {$r2}, {$r3};",
        read_only_args=[Int32(dst.toint()), Int64(src.toint()), Int32(nbytes), Int64(pol)],
    )


def _ld_keep_s32(addr):
    return cute.arch.inline_ptx(
        _KEEP_S32_PTX, write_only_types=[Int32], read_only_args=[Int64(addr)]
    )


def _ld_keep_u8(addr):
    return cute.arch.inline_ptx(
        _KEEP_U8_PTX, write_only_types=[Int32], read_only_args=[Int64(addr)]
    )


def _exp_tl(x):
    # Triton tl.exp on fp32: ex2.approx(x * log2(e))
    return cute.exp2(x * Float32(LOG2E), approx=True)


def _bf(x):
    return Float32(BFloat16(x))


def _expf(x):
    # approximate exp for the non-pinned paths: ex2.approx.ftz(x * log2(e))
    return cute.arch.inline_ptx(
        "ex2.approx.ftz.f32 {$w0}, {$r0};",
        write_only_types=[Float32],
        read_only_args=[Float32(x) * Float32(LOG2E)],
    )


def _rcpf(x):
    return cute.arch.rcp_approx(Float32(x))


def _rcp_ge1_n(ds):
    # correctly rounded 1/d (== div.rn(1, d)) for every d >= 1 of the list: the hardware fast path
    # (MUFU.RCP + one fused Newton step) for all of them; only when some lane of the warp holds a
    # d >= 2^126 (subnormal/zero quotient, +inf) the IEEE divisions run, behind a warp-uniform
    # branch (a predicated div.rn is expanded unconditionally: ~13 extra instructions per gate).
    n = len(ds)
    regs = ", ".join(f"gr{i}, gq{i}, gn{i}" for i in range(n))
    lines = ["{", f".reg .b32 gm, {regs};", ".reg .pred gp, ga;"]
    for i in range(n):
        lines += [
            f"rcp.approx.ftz.f32 gr{i}, {{$r{i}}};",
            f"neg.f32 gn{i}, {{$r{i}}};",
            f"fma.rn.ftz.f32 gq{i}, gn{i}, gr{i}, 0f3F800000;",
            f"fma.rn.ftz.f32 {{$w{i}}}, gr{i}, gq{i}, gr{i};",
        ]
    lines.append("mov.b32 gm, {$r0};")
    for i in range(1, n):
        lines.append(f"max.f32 gm, gm, {{$r{i}}};")
    lines += [
        "setp.ge.f32 gp, gm, 0f7E800000;",
        "vote.sync.any.pred ga, gp, 0xffffffff;",
        "@!ga bra.uni GATE_RCP_DONE;",
    ]
    for i in range(n):
        lines += [
            f"setp.ge.f32 gp, {{$r{i}}}, 0f7E800000;",
            f"@gp div.rn.f32 {{$w{i}}}, 0f3F800000, {{$r{i}}};",
        ]
    lines += ["GATE_RCP_DONE:", "}"]
    res = cute.arch.inline_ptx(
        "\n".join(lines), write_only_types=[Float32] * n, read_only_args=[Float32(d) for d in ds]
    )
    if n == 1:
        return [res]
    return list(res)


# a / b == div.rn(a, b) (Markstein: q = a * RN(1/b), q' = q + RN(1/b) * (a - q b), residual exact
# via FMA) for 2^-100 <= b < 2^100; other b (and only then) take the IEEE division behind a branch
# that is uniform over the active lanes.
def _div_scale_ptx(label):
    return "\n".join(
        [
            "{",
            ".reg .b32 dr, dq, de, dn, dt, dm;",
            ".reg .pred dp, da;",
            "rcp.approx.ftz.f32 dr, {$r1};",
            "neg.f32 dn, {$r1};",
            "fma.rn.ftz.f32 dt, dn, dr, 0f3F800000;",
            "fma.rn.ftz.f32 dr, dr, dt, dr;",
            "mul.rn.f32 dq, {$r0}, dr;",
            "fma.rn.f32 de, dn, dq, {$r0};",
            "fma.rn.f32 {$w0}, de, dr, dq;",
            "setp.lt.f32 dp, {$r1}, 0f0D800000;",
            "setp.ge.or.f32 dp, {$r1}, 0f71800000, dp;",
            "activemask.b32 dm;",
            "vote.sync.any.pred da, dp, dm;",
            f"@!da bra.uni {label};",
            "@dp div.rn.f32 {$w0}, {$r0}, {$r1};",
            f"{label}:",
            "}",
        ]
    )


def _div_scale(a, b, label):
    return cute.arch.inline_ptx(
        _div_scale_ptx(label), write_only_types=[Float32], read_only_args=[Float32(a), Float32(b)]
    )


# RN(1/x) for 2^-126 < x < 2^126 (fast path of rcp.rn: MUFU.RCP + one fused Newton step)
_RCP_FAST_PTX = """{
.reg .b32 fr, fn, ft;
rcp.approx.ftz.f32 fr, {$r0};
neg.f32 fn, {$r0};
fma.rn.ftz.f32 ft, fn, fr, 0f3F800000;
fma.rn.ftz.f32 {$w0}, fr, ft, fr;
}"""


def _rcp_fast(x):
    return cute.arch.inline_ptx(
        _RCP_FAST_PTX, write_only_types=[Float32], read_only_args=[Float32(x)]
    )


# UE4M3 scale byte (low byte of the cvt result) -> exact f32 value (e4m3 -> f16 -> f32)
_E4M3_F32_PTX = """{
.reg .b32 eh;
.reg .b16 el, eu;
cvt.rn.f16x2.e4m3x2 eh, {$r0};
mov.b32 {el, eu}, eh;
cvt.f32.f16 {$w0}, el;
}"""


def _e4m3_f32(w16):
    return cute.arch.inline_ptx(
        _E4M3_F32_PTX, write_only_types=[Float32], read_only_args=[cutlass.Int16(w16)]
    )


def _mul_rn(a, b):
    return cute.arch.inline_ptx(
        "mul.rn.f32 {$w0}, {$r0}, {$r1};",
        write_only_types=[Float32],
        read_only_args=[Float32(a), Float32(b)],
    )


def _add_rn(a, b):
    return cute.arch.inline_ptx(
        "add.rn.f32 {$w0}, {$r0}, {$r1};",
        write_only_types=[Float32],
        read_only_args=[Float32(a), Float32(b)],
    )


def _mul_qscale_rn(a):
    # a * fp32(128**-0.5) with explicit .rn (immediates must live in the asm string)
    return cute.arch.inline_ptx(
        "mul.rn.f32 {$w0}, {$r0}, 0f3DB504F3;",
        write_only_types=[Float32],
        read_only_args=[Float32(a)],
    )


def _add1_rn(a):
    return cute.arch.inline_ptx(
        "add.rn.f32 {$w0}, {$r0}, 0f3F800000;",
        write_only_types=[Float32],
        read_only_args=[Float32(a)],
    )


# ---- packed f32x2 forms (sm_100+): every lane rounds exactly like the scalar .rn instruction,
# so pairing two independent values (typically the k/q pair of one token) is bit-preserving
# while halving the issued FP instructions.
_FMA2_PTX = """{
.reg .b64 A, B, C;
mov.b64 A, {{$r0}, {$r1}};
mov.b64 B, {{$r2}, {$r3}};
mov.b64 C, {{$r4}, {$r5}};
fma.rn.f32x2 C, A, B, C;
mov.b64 {{$w0}, {$w1}}, C;
}"""


def _fma2(a0, a1, b0, b1, c0, c1):
    # (a0 * b0 + c0, a1 * b1 + c1), each a single-rounding fma.rn
    return cute.arch.inline_ptx(
        _FMA2_PTX,
        write_only_types=[Float32, Float32],
        read_only_args=[
            Float32(a0),
            Float32(a1),
            Float32(b0),
            Float32(b1),
            Float32(c0),
            Float32(c1),
        ],
    )


_MUL2_PTX = """{
.reg .b64 A, B;
mov.b64 A, {{$r0}, {$r1}};
mov.b64 B, {{$r2}, {$r3}};
mul.rn.f32x2 A, A, B;
mov.b64 {{$w0}, {$w1}}, A;
}"""


def _mul2(a0, a1, b0, b1):
    return cute.arch.inline_ptx(
        _MUL2_PTX,
        write_only_types=[Float32, Float32],
        read_only_args=[Float32(a0), Float32(a1), Float32(b0), Float32(b1)],
    )


# Two Markstein-corrected quotients x_i / d_i given nd_i = -d_i and r_i = RN(1 / d_i); per lane
# identical to _div_mk (fma(q, -d, x) == fma(-q, d, x) exactly).
_DIV2_PTX = """{
.reg .b64 X, D, R, Q, E;
mov.b64 X, {{$r0}, {$r1}};
mov.b64 D, {{$r2}, {$r3}};
mov.b64 R, {{$r4}, {$r5}};
mul.rn.f32x2 Q, X, R;
fma.rn.f32x2 E, Q, D, X;
fma.rn.f32x2 X, E, R, Q;
mov.b64 {{$w0}, {$w1}}, X;
}"""


def _div2_mk(x0, x1, nd0, nd1, r0, r1):
    return cute.arch.inline_ptx(
        _DIV2_PTX,
        write_only_types=[Float32, Float32],
        read_only_args=[
            Float32(x0),
            Float32(x1),
            Float32(nd0),
            Float32(nd1),
            Float32(r0),
            Float32(r1),
        ],
    )


# Eight Markstein quotients v_i / e (nd = -e, r = RN(1/e)) packed to E2M1 (RNE, satfinite) in
# one 32-bit word; element 2j is the low nibble of byte j (same as the PTX e2m1x2 operand order).
_Q8_PTX = """{
.reg .b64 V, R, D, Q, E;
.reg .b32 n0, n1, n2, n3, n4, n5, n6, n7;
.reg .b8 c0, c1, c2, c3;
mov.b64 R, {{$r9}, {$r9}};
mov.b64 D, {{$r8}, {$r8}};
mov.b64 V, {{$r0}, {$r1}};
mul.rn.f32x2 Q, V, R;
fma.rn.f32x2 E, Q, D, V;
fma.rn.f32x2 V, E, R, Q;
mov.b64 {n0, n1}, V;
mov.b64 V, {{$r2}, {$r3}};
mul.rn.f32x2 Q, V, R;
fma.rn.f32x2 E, Q, D, V;
fma.rn.f32x2 V, E, R, Q;
mov.b64 {n2, n3}, V;
mov.b64 V, {{$r4}, {$r5}};
mul.rn.f32x2 Q, V, R;
fma.rn.f32x2 E, Q, D, V;
fma.rn.f32x2 V, E, R, Q;
mov.b64 {n4, n5}, V;
mov.b64 V, {{$r6}, {$r7}};
mul.rn.f32x2 Q, V, R;
fma.rn.f32x2 E, Q, D, V;
fma.rn.f32x2 V, E, R, Q;
mov.b64 {n6, n7}, V;
cvt.rn.satfinite.e2m1x2.f32 c0, n1, n0;
cvt.rn.satfinite.e2m1x2.f32 c1, n3, n2;
cvt.rn.satfinite.e2m1x2.f32 c2, n5, n4;
cvt.rn.satfinite.e2m1x2.f32 c3, n7, n6;
mov.b32 {$w0}, {c0, c1, c2, c3};
}"""


def _quant8(v, nd, r):
    return cute.arch.inline_ptx(
        _Q8_PTX,
        write_only_types=[Int32],
        read_only_args=[Float32(x) for x in v] + [Float32(nd), Float32(r)],
    )


# max |v_i| of 16 values with three-input max.abs (sm_100+): exact, 8 instructions
_AMAX16_PTX = """{
.reg .b32 a, b, c, d, e, f, g;
max.abs.f32 a, {$r0}, {$r1}, {$r2};
max.abs.f32 b, {$r3}, {$r4}, {$r5};
max.abs.f32 c, {$r6}, {$r7}, {$r8};
max.abs.f32 d, {$r9}, {$r10}, {$r11};
max.abs.f32 e, {$r12}, {$r13}, {$r14};
max.abs.f32 f, a, b, {$r15};
max.f32 g, c, d, e;
max.f32 {$w0}, f, g;
}"""


def _amax16(v):
    return cute.arch.inline_ptx(
        _AMAX16_PTX, write_only_types=[Float32], read_only_args=[Float32(x) for x in v]
    )


# Two-lane form of _SILU_PTX on packed f32x2 FMA/ADD/MUL (each lane rounds exactly like the
# scalar instruction).  -x is folded into negated constants: fma(-x, c, -s) == -fma(x, c, s).
_SILU2_PTX = """{
.reg .b64 X, T, J, S, G, E, D, R, O, K0, K1, K2, K3, K4, K5, K6, K7;
.reg .b32 t0, t1, j0, j1, g0, g1, e0, e1, d0, d1, r0, r1, q0, q1, m0, m1;
.reg .pred p0, p1;
mov.b64 K0, 0xBBBB989DBBBB989D;
mov.b64 K1, 0x3F0000003F000000;
mov.b64 K2, 0x437C0000437C0000;
mov.b64 K3, 0x4B4000014B400001;
mov.b64 K4, 0xCB40007FCB40007F;
mov.b64 K5, 0x3FB8AA3B3FB8AA3B;
mov.b64 K6, 0x32A5706032A57060;
mov.b64 K7, 0x3F8000003F800000;
mov.b64 X, {{$r0}, {$r1}};
fma.rn.ftz.f32x2 T, X, K0, K1;
mov.b64 {t0, t1}, T;
cvt.ftz.sat.f32.f32 t0, t0;
cvt.ftz.sat.f32.f32 t1, t1;
mov.b64 T, {t0, t1};
fma.rm.ftz.f32x2 J, T, K2, K3;
add.rn.f32x2 S, J, K4;
fma.rn.ftz.f32x2 G, X, K5, S;
fma.rn.ftz.f32x2 G, X, K6, G;
mov.b64 {g0, g1}, G;
neg.f32 g0, g0;
neg.f32 g1, g1;
mov.b64 {j0, j1}, J;
shl.b32 j0, j0, 23;
shl.b32 j1, j1, 23;
ex2.approx.ftz.f32 e0, g0;
ex2.approx.ftz.f32 e1, g1;
mov.b64 E, {e0, e1};
mov.b64 J, {j0, j1};
mul.rn.f32x2 E, E, J;
add.rn.f32x2 D, E, K7;
mov.b64 {d0, d1}, D;
rcp.approx.ftz.f32 r0, d0;
rcp.approx.ftz.f32 r1, d1;
neg.f32 m0, d0;
neg.f32 m1, d1;
fma.rn.ftz.f32 q0, m0, r0, 0f3F800000;
fma.rn.ftz.f32 q1, m1, r1, 0f3F800000;
fma.rn.ftz.f32 r0, r0, q0, r0;
fma.rn.ftz.f32 r1, r1, q1, r1;
setp.eq.f32 p0, d0, 0f7F800000;
setp.eq.f32 p1, d1, 0f7F800000;
selp.f32 r0, 0f00000000, r0, p0;
selp.f32 r1, 0f00000000, r1, p1;
mov.b64 R, {r0, r1};
mul.rn.f32x2 O, R, X;
mov.b64 {{$w0}, {$w1}}, O;
}"""


def _silu2_ref(a, b):
    return cute.arch.inline_ptx(
        _SILU2_PTX, write_only_types=[Float32, Float32], read_only_args=[Float32(a), Float32(b)]
    )


def _tree_reduce(vals, offs, lane):
    """Transposed xor-butterfly over the lane bits ``offs`` (in order) for an arbitrary number of
    values.  Every value's partial sums are combined over exactly the same lane pairs, in the same
    order, as a plain xor butterfly (bit-identical sums); pairs are only formed inside groups of
    values that share their lane-index structure, so no padding values are shuffled.
    Returns [(value, idx_const, {offset: coef}, plain_mask)]: the lane holds the full sum of value
    index idx_const + sum(coef * lane_bit(offset)); lanes with any plain_mask bit set hold a
    duplicate (store only from (lane & plain_mask) == 0)."""
    cur = [(v, j, (), 0) for j, v in enumerate(vals)]
    for o in offs:
        up = (lane & o) != 0
        groups = {}
        order = []
        for e in cur:
            key = (e[3], e[2])
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(e)
        nxt = []
        for key in order:
            g = groups[key]
            p = len(g) // 2
            for i in range(p):
                a = g[i]
                bb = g[i + p]
                snd = Float32(cutlass.select_(up, a[0], bb[0]))
                kep = Float32(cutlass.select_(up, bb[0], a[0]))
                v = kep + cute.arch.shuffle_sync_bfly(snd, offset=o)
                nxt.append((v, a[1], a[2] + ((o, bb[1] - a[1]),), a[3]))
            if len(g) % 2 == 1:
                e = g[-1]
                v = e[0] + cute.arch.shuffle_sync_bfly(e[0], offset=o)
                nxt.append((v, e[1], e[2], e[3] | o))
        cur = nxt
    return cur


def _tree_idx(e, lane):
    idx = Int32(e[1])
    for o, cf in e[2]:
        idx = idx + ((lane // o) % 2) * cf
    return idx


# Two adjusted history keys k_j * exp2((anchor - g_j) * log2e) (ex2.approx.ftz), lane-paired.
_ADJ2_PTX = """{
.reg .b64 A, G, L, X, K;
.reg .b32 x0, x1;
mov.b64 A, {{$r0}, {$r0}};
mov.b64 G, {{$r1}, {$r2}};
mov.b64 L, 0x3FB8AA3B3FB8AA3B;
sub.rn.f32x2 X, A, G;
mul.rn.f32x2 X, X, L;
mov.b64 {x0, x1}, X;
ex2.approx.ftz.f32 x0, x0;
ex2.approx.ftz.f32 x1, x1;
mov.b64 X, {x0, x1};
mov.b64 K, {{$r3}, {$r4}};
mul.rn.f32x2 X, K, X;
mov.b64 {{$w0}, {$w1}}, X;
}"""


def _adj2(anchor, g0, g1, k0, k1):
    return cute.arch.inline_ptx(
        _ADJ2_PTX,
        write_only_types=[Float32, Float32],
        read_only_args=[Float32(anchor), Float32(g0), Float32(g1), Float32(k0), Float32(k1)],
    )


# Two output-gate sigmoids 1 / (1 + exp2(-z * log2e)) (approx, tolerance-checked path).
_SIG2_PTX = """{
.reg .b64 Z, C;
.reg .b32 e0, e1;
mov.b64 Z, {{$r0}, {$r1}};
mov.b64 C, 0xBFB8AA3BBFB8AA3B;
mul.rn.f32x2 Z, Z, C;
mov.b64 {e0, e1}, Z;
ex2.approx.ftz.f32 e0, e0;
ex2.approx.ftz.f32 e1, e1;
mov.b64 Z, {e0, e1};
mov.b64 C, 0x3F8000003F800000;
add.rn.f32x2 Z, Z, C;
mov.b64 {e0, e1}, Z;
rcp.approx.ftz.f32 {$w0}, e0;
rcp.approx.ftz.f32 {$w1}, e1;
}"""


def _sig2(z0, z1):
    return cute.arch.inline_ptx(
        _SIG2_PTX, write_only_types=[Float32, Float32], read_only_args=[Float32(z0), Float32(z1)]
    )


def _history_row_s(replay, p_hist, slot_stride, dtype, row):
    """Pointer to ``row`` of the replay slot in one history plane (replay slot read from smem)."""
    return cutlass.inttoptr(p_hist + Int64(replay) * slot_stride, GMEM, dtype) + row


@cute.jit
def _coef_tiles_t(
    lane,
    length,
    s_A,
    s_R,
    s_coefT,
    s_coefn,
    T: cutlass.Constexpr,
    N2: cutlass.Constexpr,
    NROWS: cutlass.Constexpr,
    MMA_N: cutlass.Constexpr,
):
    """C^T[16 x 24] = R[16 x 128] A[24 x 128]^T on one warp: m16 = relative pair rows n (clamped to
    N2 - 1), three n8 tiles over the A rows r (0..15 history, 16.. new tokens; clamped to NROWS - 1).
    The products / k16 grouping / even-odd accumulator chains are those of the C = A R^T tiles."""
    gid = lane // 4
    tig = lane % 4
    lmj = lane // 8
    arow = (lmj % 2) * 8 + lane % 8
    if cutlass.const_expr(N2 < 16):
        arow = arow if arow < N2 else Int32(N2 - 1)
    pla = s_R.data_ptr() + (arow * ASTR + (lmj // 2) * 8)
    plbs = []
    for t in cutlass.range_constexpr(3):
        brow = 8 * t + lane % 8
        if cutlass.const_expr(8 * t + 8 > NROWS):
            brow = brow if brow < NROWS else Int32(NROWS - 1)
        plbs.append(s_A.data_ptr() + (brow * ASTR + (lmj // 2) * 16 + (lmj % 2) * 8))
    ces = [None, None, None]
    cos = [None, None, None]
    bfr = [None, None, None]
    for ks in cutlass.range_constexpr(KD // 16):
        afrag = prims.ldmatrix(pla + 16 * ks, 4, prims.MMALayout.ROW)
        if cutlass.const_expr(ks % 2 == 0):
            for t in cutlass.range_constexpr(3):
                bfr[t] = prims.ldmatrix(plbs[t] + 16 * ks, 4, prims.MMALayout.ROW)
        for t in cutlass.range_constexpr(3):
            fb0 = bfr[t][2 * (ks % 2)]
            fb1 = bfr[t][2 * (ks % 2) + 1]
            if cutlass.const_expr(ks < 2):
                acc4 = cute.arch.inline_ptx(
                    "{\n.reg .f32 zz;\nmov.f32 zz, 0f00000000;\n"
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, {{$r4}, {$r5}}, "
                    "{zz, zz, zz, zz};\n}",
                    write_only_types=[Float32, Float32, Float32, Float32],
                    read_only_args=[afrag[0], afrag[1], afrag[2], afrag[3], fb0, fb1],
                )
            else:
                prev4 = ces[t] if cutlass.const_expr(ks % 2 == 0) else cos[t]
                acc4 = cute.arch.inline_ptx(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, {{$r4}, {$r5}}, "
                    "{{$r6}, {$r7}, {$r8}, {$r9}};",
                    write_only_types=[Float32, Float32, Float32, Float32],
                    read_only_args=[
                        afrag[0],
                        afrag[1],
                        afrag[2],
                        afrag[3],
                        fb0,
                        fb1,
                        prev4[0],
                        prev4[1],
                        prev4[2],
                        prev4[3],
                    ],
                )
            if cutlass.const_expr(ks % 2 == 0):
                ces[t] = acc4
            else:
                cos[t] = acc4
    for t in cutlass.range_constexpr(3):
        d0 = ces[t][0] + cos[t][0]
        d1 = ces[t][1] + cos[t][1]
        d2 = ces[t][2] + cos[t][2]
        d3 = ces[t][3] + cos[t][3]
        # fragment: (n = gid, r = 8 t + 2 tig + {0, 1}) and (n = gid + 8, same r)
        r0 = 8 * t + 2 * tig
        if cutlass.const_expr(t < 2):
            ok0 = r0 < length
            ok1 = r0 + 1 < length
            v01 = cutlass.Vector.from_elements(
                (BFloat16(d0 if ok0 else Float32(0.0)), BFloat16(d1 if ok1 else Float32(0.0))),
                BFloat16,
            )
            v23 = cutlass.Vector.from_elements(
                (BFloat16(d2 if ok0 else Float32(0.0)), BFloat16(d3 if ok1 else Float32(0.0))),
                BFloat16,
            )
            (s_coefT.data_ptr() + (gid * CTSTR + r0)).store(v01, alignment=4)
            (s_coefT.data_ptr() + ((gid + 8) * CTSTR + r0)).store(v23, alignment=4)
        else:
            s0 = 2 * tig
            if s0 < T:
                if gid < N2:
                    s_coefn[s0 * MMA_N + gid] = d0
                if gid + 8 < N2:
                    s_coefn[s0 * MMA_N + gid + 8] = d2
            if s0 + 1 < T:
                if gid < N2:
                    s_coefn[(s0 + 1) * MMA_N + gid] = d1
                if gid + 8 < N2:
                    s_coefn[(s0 + 1) * MMA_N + gid + 8] = d3


@cute.kernel
def _replay_kernel(
    p_raw: Int64,
    p_conv: Int64,
    p_cw: Int64,
    p_g: Int64,
    p_beta: Int64,
    p_z: Int64,
    p_nw: Int64,
    p_pool: Int64,
    p_slots: Int64,
    p_dummy: Int64,
    p_ridx: Int64,
    p_hk: Int64,
    p_hu: Int64,
    p_hg: Int64,
    p_hlen: Int64,
    p_a_log: Int64,
    p_dtb: Int64,
    p_out: Int64,
    p_cand: Int64,
    p_oldlen: Int64,
    ck_desc: cutlass.GridConstant[TensorMap],
    candidate_bs: Int64,
    lower: Float32,
    eps: Float32,
    H: cutlass.Constexpr[int],
    raw_bs: cutlass.Constexpr[int],
    raw_rs: cutlass.Constexpr[int],
    conv_ss: cutlass.Constexpr[int],
    g_bs: cutlass.Constexpr[int],
    g_rs: cutlass.Constexpr[int],
    beta_bs: cutlass.Constexpr[int],
    beta_rs: cutlass.Constexpr[int],
    z_bs: cutlass.Constexpr[int],
    z_rs: cutlass.Constexpr[int],
    pool_ss: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    BOUNDED: cutlass.Constexpr[int],
    POOL_VEC: cutlass.Constexpr[int],
    CAP: cutlass.Constexpr[int],
    Z_VEC: cutlass.Constexpr[int],
    PDL: cutlass.Constexpr[int],
    hk_ss: cutlass.Constexpr[int],
    hu_ss: cutlass.Constexpr[int],
    hg_ss: cutlass.Constexpr[int],
    hl_ss: cutlass.Constexpr[int],
    EPS_FTZ: cutlass.Constexpr[int],
):
    N2 = 2 * T
    MMA_N: cutlass.Constexpr[int] = 8 if cutlass.const_expr(T <= 4) else 16
    NROWS = HROWS + T
    # padded (power of two) value counts of the transposed warp reductions
    RV1: cutlass.Constexpr[int] = 1 << (N2 - 1).bit_length()
    tid, _, _ = cute.arch.thread_idx()
    h, b, _ = cute.arch.block_idx()
    warp = tid // 32
    lane = tid % 32
    HK = H * KD
    C3 = HK * 3

    # ---------------- shared memory ----------------
    sA = cutlass.Array(Uint8, (8192,), space=cutlass.AddressSpace.smem, alignment=1024)
    sB = cutlass.Array(Uint32, (MMA_N * 16,), space=cutlass.AddressSpace.smem, alignment=1024)
    # staged history K rows (stride ASTR); turned in place into the adjusted-key rows of the
    # coefficient products, followed by T adjusted new-key rows
    s_uh = cutlass.Array(BFloat16, (HROWS * ASTR,), space=cutlass.AddressSpace.smem, alignment=128)
    s_gh = cutlass.Array(Float32, (HROWS * KD,), space=cutlass.AddressSpace.smem, alignment=128)
    s_A = cutlass.Array(BFloat16, (NROWS * ASTR,), space=cutlass.AddressSpace.smem, alignment=128)
    s_R = cutlass.Array(BFloat16, (N2 * ASTR,), space=cutlass.AddressSpace.smem, alignment=128)
    s_coefT = cutlass.Array(
        BFloat16, (MMA_N * CTSTR,), space=cutlass.AddressSpace.smem, alignment=16
    )
    s_coefn = cutlass.Array(Float32, (T * MMA_N,), space=cutlass.AddressSpace.smem, alignment=16)
    # SFB staging in the UTCCP 32x128b atom image: [k-block][row L][4 words], word 0 = 4 scales
    s_sfb = cutlass.Array(Uint32, (2 * 128,), space=cutlass.AddressSpace.smem, alignment=128)
    # row pitch 18 floats: the transposed-butterfly partial stores (8 rows x 4 groups per warp) are
    # bank-conflict free (pitch 16 put a warp's 32 stores into 4 banks)
    s_red1 = cutlass.Array(Float32, (RV1 * 18,), space=cutlass.AddressSpace.smem, alignment=16)
    s_red2 = cutlass.Array(Float32, (4 * 16,), space=cutlass.AddressSpace.smem, alignment=16)
    s_dr = cutlass.Array(Float32, (32,), space=cutlass.AddressSpace.smem, alignment=16)
    # per-token product of the operand outer scale and the checkpoint outer scale, duplicated
    # so that each (k, q) pair is one aligned f32x2 operand
    s_osc = cutlass.Array(Float32, (16,), space=cutlass.AddressSpace.smem, alignment=16)
    s_red3 = cutlass.Array(Float32, (4 * 8,), space=cutlass.AddressSpace.smem, alignment=16)
    s_bsig = cutlass.Array(Float32, (8,), space=cutlass.AddressSpace.smem, alignment=16)
    s_z = cutlass.Array(BFloat16, (T * KD,), space=cutlass.AddressSpace.smem, alignment=16)
    s_tmem = cutlass.Array(Int32, (1,), space=cutlass.AddressSpace.smem, alignment=16)
    s_rep = cutlass.Array(Int32, (4,), space=cutlass.AddressSpace.smem, alignment=16)
    s_mbar = cutlass.Array(Int64, (1,), space=cutlass.AddressSpace.smem, alignment=8)
    # completion barrier of the G history bulk copy (large launches)
    s_mbar_g = cutlass.Array(Int64, (1,), space=cutlass.AddressSpace.smem, alignment=8)
    # completion barrier of the checkpoint TMA tile copy (16 B aligned pool rows)
    s_mbar_a = cutlass.Array(Int64, (1,), space=cutlass.AddressSpace.smem, alignment=8)
    # native-MMA descriptors, built once at CTA start (rebuilding them after the post-quantization
    # barrier costs the issuing warp ~240 cycles of dependent uniform-datapath ops)
    s_desc = cutlass.Array(Int64, (4,), space=cutlass.AddressSpace.smem, alignment=16)

    # TMEM allocation + permit release first: the next CTA on this SM is only launched once this
    # one has relinquished its allocation permit
    if warp == 0:
        prims.tcgen05_alloc(s_tmem, TMEM_COLS, group=prims.CTAGroup.CTA_1)
        prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_1)
    out_ptr = cutlass.inttoptr(p_out, GMEM, BFloat16)
    cand_ptr = cutlass.inttoptr(p_cand, GMEM, BFloat16)
    oldlen_ptr = cutlass.inttoptr(p_oldlen, GMEM, Int32)
    b64 = Int64(b)
    h64 = Int64(h)
    hoff = h * KD + tid

    # the row's slot is fetched together with its dummy flag (one dependent round trip less);
    # it is only dereferenced on live rows
    is_dummy = _ld_keep_u8(p_dummy + b64)
    slot = _ld_keep_s32(p_slots + b64 * 4)
    # the replay slot index is row metadata (not a slot dereference): fetch it in the same round
    # trip so that the length -> history chain starts one round trip earlier
    replay_slot = _ld_keep_s32(p_ridx + b64 * 4)

    raw_ptr = cutlass.inttoptr(p_raw, GMEM, BFloat16)
    z_ptr = cutlass.inttoptr(p_z, GMEM, BFloat16)
    g_ptr = cutlass.inttoptr(p_g, GMEM, BFloat16)
    beta_ptr = cutlass.inttoptr(p_beta, GMEM, BFloat16)
    cw_ptr = cutlass.inttoptr(p_cw, GMEM, BFloat16)
    cws = []
    for sec in cutlass.range_constexpr(3):
        cws.append((cw_ptr + Int64(sec * HK + hoff) * 4).load(alignment=8, count=4))
    dtb = (cutlass.inttoptr(p_dtb, GMEM, Float32) + hoff).load()
    a_log = (cutlass.inttoptr(p_a_log, GMEM, Float32) + h).load()
    nw = (cutlass.inttoptr(p_nw, GMEM, BFloat16) + tid).load()

    if cutlass.const_expr(MMA_N != 8 and N2 <= 12):
        # warps 0..2 own every async-proxy operand write (SFB, B codes, padding rows), so only they
        # need the proxy fence before the MMA (warp 3's MEMBAR is off the coefficient path)
        if tid < 96:
            s_sfb[tid] = Uint32(0)
            s_sfb[tid + 96] = Uint32(0)
            if tid < 64:
                s_sfb[tid + 192] = Uint32(0)
    else:
        s_sfb[tid] = Uint32(0)
        s_sfb[tid + NT] = Uint32(0)
    if cutlass.const_expr(MMA_N != 8 and N2 <= 12):
        # descriptor-visible padding operand rows N2..15 are exact zeros (the quantization phase
        # then only writes live rows)
        if tid < (MMA_N - N2) * 16:
            sB[N2 * 16 + tid] = Uint32(0)
    if tid == 0:
        prims.mbarrier_init(s_mbar, 1)
        prims.mbarrier_init(s_mbar_g, 1)
        prims.mbarrier_init(s_mbar_a, 1)
        prims.fence_mbarrier_init()
        s_desc[0] = Int64(
            prims.Tcgen05SmemDesc.build(
                sA,
                leading_byte_offset=16,
                stride_byte_offset=512,
                base_offset=0,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
            )
        )
        s_desc[1] = Int64(
            prims.Tcgen05SmemDesc.build(
                sB,
                leading_byte_offset=16,
                stride_byte_offset=512,
                base_offset=0,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
            )
        )
        s_desc[2] = Int64(
            prims.Tcgen05SmemDesc.build(
                s_sfb,
                leading_byte_offset=16,
                stride_byte_offset=128,
                base_offset=0,
                layout=prims.Tcgen05SmemSwizzle.NONE,
            )
        )
        s_desc[3] = Int64(
            Int32(
                prims.Tcgen05MxOmmaInstrDesc.build(
                    a_dtype=cutlass.Float4E2M1FN,
                    b_dtype=cutlass.Float4E2M1FN,
                    scale_format=0,
                    n_dim=MMA_N,
                    m_dim=128,
                    sparsity_version=0,
                )
            )
        )

    if is_dummy != Int32(0):
        if cutlass.const_expr(PDL != 0):
            prims.griddepcontrol("wait")
        # Dummy row: exact zeros for every destination element (16-byte stores), nothing else
        # touched.  Thread chunk c = tid + NT*j covers segment c // 16 = 4 t + part (part 0 =
        # output, 1..3 = candidate q/k/v sections).
        ZCH: cutlass.Constexpr[int] = (64 * T + NT - 1) // NT
        zv = cutlass.Vector.from_elements((Int32(0), Int32(0), Int32(0), Int32(0)), Int32)
        zpart = (tid // 16) % 4
        zoff = (tid % 16) * 8
        for j in cutlass.range_constexpr(ZCH):
            zt = tid // 64 + 2 * j
            if zt < T:
                o_addr = p_out + (((b64 * T + zt) * H + h64) * KD + zoff) * 2
                c_addr = (
                    p_cand + (b64 * candidate_bs + zt * C3 + (zpart - 1) * HK + h * KD + zoff) * 2
                )
                z_addr = o_addr if zpart == 0 else c_addr
                cutlass.inttoptr(z_addr, GMEM, Int32).store(zv, alignment=16)
        if h == 0:
            if tid == 0:
                (oldlen_ptr + b64).store(Int32(0))
        prims.barrier_cta_sync(0)
        if warp == 0:
            prims.tcgen05_dealloc(
                cutlass.inttoptr(s_tmem[0], 6, Float32),
                TMEM_COLS,
                group=prims.CTAGroup.CTA_1,
            )
    else:
        # ---- slot-dependent loads ----
        slot64 = Int64(slot)
        pool_row = p_pool + slot64 * pool_ss + h64 * STATE_BYTES
        pool_u8 = cutlass.inttoptr(pool_row, GMEM, Uint8)
        # state primary values -> SW64 K-major smem tile (8 rows x 64B atoms, chunk ^= (row>>1)&3)
        sfa = []
        if cutlass.const_expr(POOL_VEC != 0):
            # 16B-aligned slot stride: one TMA tile copy (64 B x 128 rows, SWIZZLE_64B = the same
            # chunk ^= (row >> 1) & 3 image) + vector scale loads.  The TMA writes bypass the LSU
            # pipe (4 cp.async per thread cost ~170 shared-memory wavefronts per CTA).
            # (the TMA copy itself is issued after the history copies: the launch start is
            # DRAM-bandwidth bound and the checkpoint is only needed at the MMA)
            # (keep the history-staging loop variables type-stable across branches)
            q = tid
            r = q // 16
            for j in cutlass.range_constexpr(4):
                m = lane + 32 * j
                v = cutlass.inttoptr(pool_row + 16384 + m * 8, GMEM, Uint32).load(
                    alignment=8, count=2
                )
                sfa.append(v)
            outer_s = cutlass.inttoptr(pool_row + 18432, GMEM, Float32).load()
        else:
            # arbitrary (unaligned) slot stride: byte-granular loads, same smem/TMEM images
            for k in cutlass.range_constexpr(4):
                q = tid + NT * k
                r = q // 4
                c = q % 4
                dst = r * 64 + ((c ^ ((r >> 1) & 3)) * 16)
                bv = (pool_u8 + (r * 64 + c * 16)).load(alignment=1, count=16)
                (sA.data_ptr() + dst).store(bv, alignment=16)
            for j in cutlass.range_constexpr(4):
                m = lane + 32 * j
                bv = (pool_u8 + (16384 + m * 8)).load(alignment=1, count=8)
                sfa.append(bv.bitcast(Uint32))
            outer_s = (pool_u8 + 18432).load(alignment=1, count=4).bitcast(Float32)[0]
        conv_ptr = cutlass.inttoptr(p_conv, GMEM, BFloat16)
        cs = []
        for sec in cutlass.range_constexpr(3):
            cbase = slot64 * conv_ss + Int64(sec * HK + hoff) * 3
            for p in cutlass.range_constexpr(3):
                cs.append((conv_ptr + (cbase + p)).load())

        # ---- live-dependent loads: history rows (zero-filled past length) ----
        # Histories and their length are addressed by replay slot (64-bit
        # byte strides: pool pages); each slot's payload is dense.
        replay64 = Int64(replay_slot)
        length = _ld_keep_s32(p_hlen + replay64 * hl_ss)
        if tid == 0:
            s_rep[0] = replay_slot
        hrow = h64 * CAP  # row index of (h, 0) within the replay slot
        hk_ptr = cutlass.inttoptr(p_hk + replay64 * hk_ss, GMEM, BFloat16)
        hu_ptr = cutlass.inttoptr(p_hu + replay64 * hu_ss, GMEM, BFloat16)
        hg_ptr = cutlass.inttoptr(p_hg + replay64 * hg_ss, GMEM, Float32)
        hk_base = hk_ptr + hrow * KD
        hu_base = hu_ptr + hrow * KD
        hg_base = hg_ptr + hrow * KD
        # ---- current-step projections (behind the PDL wait; everything above only
        # depends on earlier steps) ----
        if cutlass.const_expr(PDL != 0):
            prims.griddepcontrol("wait")
        xs = []
        x_row0 = raw_ptr + (b64 * raw_bs + hoff)
        for sec in cutlass.range_constexpr(3):
            x_row = x_row0 + sec * HK
            for t in cutlass.range_constexpr(T):
                xs.append(x_row.load())
                x_row = x_row + raw_rs
        # z is only consumed by the output gate: with 16 B aligned rows it is staged into shared
        # memory by cp.async instead of occupying T registers for the whole kernel.  beta is a
        # per-(row, token) scalar: thread t < T loads it once and publishes sigmoid(beta_t)
        # through shared memory.
        zs = []
        gs = []
        z_row = z_ptr + (b64 * z_bs + hoff)
        g_row = g_ptr + (b64 * g_bs + hoff)
        for t in cutlass.range_constexpr(T):
            if cutlass.const_expr(Z_VEC == 0):
                zs.append(z_row.load())
            gs.append(g_row.load())
            z_row = z_row + z_rs
            g_row = g_row + g_rs
        tb = tid if tid < T else Int32(0)
        beta_t = (beta_ptr + (b64 * beta_bs + h64 + tb * beta_rs)).load()
        # history staging (state of earlier launches) is issued after the projection loads: a CTA
        # that starts after the producer has finished does not delay them behind the length
        # round trip, and the history is only needed after the norms
        gdx, gdy, _ = cute.arch.grid_dim()
        if gdx * gdy <= LIVE_MIN_GRID:
            # K/G coefficients are masked later. U must be zero outside the
            # accepted prefix: zero coefficients do not suppress NaN operands.
            for k in cutlass.range_constexpr(2):
                q = tid + NT * k
                r = q // 16
                prims.cp_async_shared_global(
                    s_A.data_ptr() + (q * 8 + r * 8), hk_base + q * 8, 16, "cg"
                )
                prims.cp_async_shared_global(
                    s_uh.data_ptr() + (q * 8 + r * 8),
                    hu_base + q * 8,
                    16,
                    "cg",
                    cp_size=Int32(16) if r < length else Int32(0),
                )
            for k in cutlass.range_constexpr(4):
                q = tid + NT * k
                prims.cp_async_shared_global(s_gh.data_ptr() + q * 4, hg_base + q * 4, 16, "cg")
        else:
            # large launch: only the live rows (bandwidth bound); streaming L2 policy
            hpol = _evict_first_policy()
            for k in cutlass.range_constexpr(2):
                q = tid + NT * k
                r = q // 16
                nbytes = Int32(16) if r < length else Int32(0)
                _cp_async16_ef(s_A.data_ptr() + (q * 8 + r * 8), hk_base + q * 8, nbytes, hpol)
                _cp_async16_ef(s_uh.data_ptr() + (q * 8 + r * 8), hu_base + q * 8, nbytes, hpol)
            # G rows (dense 512 B rows in global and shared): one bulk copy of the live rows; rows
            # past the length stay stale (the Adj loop masks them by selects)
            if tid == 0:
                prims.mbarrier_arrive_expect_tx(s_mbar_g, length * 512)
                if length > 0:
                    prims.cp_async_bulk_shared_cluster_global(
                        s_gh.data_ptr(),
                        hg_base,
                        s_mbar_g,
                        length * 512,
                        l2_cache_hint=_evict_first_policy(),
                    )
        prims.cp_async_commit_group()

        # ---- gates (Triton-pinned elementary ops) ----
        rate = _exp_tl(a_log)
        gates = []
        if cutlass.const_expr(BOUNDED != 0):
            dgates = []
            for t in cutlass.range_constexpr(T):
                shifted = Float32(gs[t]) + dtb
                dgates.append(_add1_rn(_expf(_mul_rn(shifted, -rate))))
            rgates = _rcp_ge1_n(dgates)
            for t in cutlass.range_constexpr(T):
                gates.append(_mul_rn(lower, rgates[t]))
        else:
            for t in cutlass.range_constexpr(T):
                shifted = Float32(gs[t]) + dtb
                spv = cute.log(Float32(1.0) + _exp_tl(shifted))
                sp = spv if shifted <= Float32(20.0) else shifted
                gates.append(-rate * sp)
        if tid < T:
            s_bsig[tid] = _rcpf(Float32(1.0) + _expf(-Float32(beta_t)))

        # ---- Conv4 + SiLU (thread = key/value channel) ----
        # q and k of one token run as one packed lane pair (bit-identical per lane); the first
        # tap stays a scalar fma with +0 so the accumulator start matches the reference exactly.
        qv = []
        kv = []
        vv = []
        wq = cws[0]
        wk = cws[1]
        wv2 = cws[2]
        for t in cutlass.range_constexpr(T):
            aq = Float32(0.0)
            ak = Float32(0.0)
            av = Float32(0.0)
            for tap in cutlass.range_constexpr(4):
                pos = t + tap
                if cutlass.const_expr(pos < 3):
                    xq = Float32(cs[pos])
                    xk = Float32(cs[3 + pos])
                    xvv = Float32(cs[6 + pos])
                else:
                    xq = Float32(xs[pos - 3])
                    xk = Float32(xs[T + pos - 3])
                    xvv = Float32(xs[2 * T + pos - 3])
                if cutlass.const_expr(tap == 0):
                    aq = cute.fma(xq, Float32(wq[0]), aq)
                    ak = cute.fma(xk, Float32(wk[0]), ak)
                else:
                    aq, ak = _fma2(xq, xk, Float32(wq[tap]), Float32(wk[tap]), aq, ak)
                av = cute.fma(xvv, Float32(wv2[tap]), av)
            sq_, sk_ = _silu2_ref(aq, ak)
            qv.append(_bf(sq_))
            kv.append(_bf(sk_))
            vv.append(_bf(av * _rcpf(Float32(1.0) + _expf(-av))))

        cw_row = cand_ptr + (b64 * candidate_bs + hoff)
        for t in cutlass.range_constexpr(T):
            for sec in cutlass.range_constexpr(3):
                (cw_row + sec * HK).store(xs[sec * T + t])
            cw_row = cw_row + C3

        # ---- reduction 1: sum of squares of q and k per token ----
        sqq = []
        sqk = []
        for t in cutlass.range_constexpr(T):
            a2, b2 = _mul2(qv[t], kv[t], qv[t], kv[t])
            sqq.append(a2)
            sqk.append(b2)
        sq = sqq + sqk
        # Bit-exact replica of the reference tl.sum order (Triton layout: 16 lanes x 8 consecutive
        # keys per row): in-thread ((e0+e2)+(e4+e6))+((e1+e3)+(e5+e7)), then lane butterfly
        # xor 8,4,2,1 over the 16 key-groups.  Here thread = key, so the in-thread part is a
        # xor 2,4,1 butterfly inside 8-lane groups and the group butterfly runs from smem.
        # Transposed butterfly: every level pairs exactly the same lanes as the plain xor 2,4,1
        # butterfly (so each sum is bit-identical) but halves the values a lane carries.
        red1 = _tree_reduce(sq, (2, 4, 1), lane)
        for e1 in red1:
            if (lane & e1[3]) == 0:
                s_red1[_tree_idx(e1, lane) * 18 + warp * 4 + lane // 8] = e1[0]
        prims.barrier_cta_sync(0)
        if tid < N2:
            tgv = (s_red1.data_ptr() + tid * 18).load(alignment=8, count=16)
            ug = []
            for m in cutlass.range_constexpr(8):
                ug.append(tgv[m] + tgv[m + 8])
            vg = []
            for m in cutlass.range_constexpr(4):
                vg.append(ug[m] + ug[m + 4])
            tot = (vg[0] + vg[2]) + (vg[1] + vg[3])
            dden = cute.sqrt(tot + Float32(1e-6))
            # [token][-den_q, -den_k, 1/den_q, 1/den_k]: one 16 B read per token below
            ktok = tid >= T
            dslot = (tid - T) * 4 + 1 if ktok else tid * 4
            s_dr[dslot] = -dden
            s_dr[dslot + 2] = Float32(1.0) / dden
        prims.barrier_cta_sync(0)
        # SFA (checkpoint block-16 scales) -> TMEM cols 16..23; each warp replicates into its
        # sub-partition: col 16 + 4*kb + j holds rows 32*j + lane, k-blocks 4kb..4kb+3
        tmem_base = s_tmem[0]
        sfa_vals = []
        for kb in cutlass.range_constexpr(2):
            for j in cutlass.range_constexpr(4):
                sfa_vals.append(Int32(sfa[j][kb]))
        sfa_vec = cutlass.Vector.from_elements(tuple(sfa_vals), Int32)
        prims.tcgen05_st(
            "32x32b", cutlass.inttoptr(tmem_base + ((warp * 32) << 16) + 16, 6, Int32), sfa_vec
        )
        # (its completion is awaited before the pre-quantization barrier)
        qn = []
        kn = []
        for t in cutlass.range_constexpr(T):
            # denominators sqrt(sum+eps) >= 1e-3 and conv outputs are normal-range: the
            # Markstein form equals div.rn bit-for-bit here
            drv = (s_dr.data_ptr() + t * 4).load(alignment=16, count=4)
            qd, kd = _div2_mk(_mul_qscale_rn(qv[t]), kv[t], drv[0], drv[1], drv[2], drv[3])
            qn.append(_bf(qd))
            kn.append(_bf(kd))

        # ---- history staged ----
        prims.cp_async_wait_group(0)
        prims.barrier_cta_sync(0)
        if cutlass.const_expr(POOL_VEC != 0):
            if tid == 0:
                prims.mbarrier_arrive_expect_tx(s_mbar_a, 8192)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    sA.data_ptr(),
                    ck_desc.get_ptr(),
                    [Int32(0), Int32(0), h, slot],
                    s_mbar_a,
                    l2_cache_hint=_evict_first_policy(),
                )
        if cutlass.const_expr(Z_VEC != 0):
            # output-gate rows (needed in the epilogue) only now: the early bandwidth goes to the
            # critical x / history loads
            zt = tid // 16
            if zt < T:
                zc = (tid % 16) * 8
                prims.cp_async_shared_global(
                    s_z.data_ptr() + (zt * KD + zc),
                    z_ptr + (b64 * z_bs + h64 * KD + zt * z_rs + zc),
                    16,
                    "cg",
                )
            prims.cp_async_commit_group()
        if gdx * gdy > LIVE_MIN_GRID:
            while not prims.mbarrier_try_wait_parity(s_mbar_g, 0):
                pass
        aidx = (length - 1) if length > 0 else Int32(0)
        anchor_ld = s_gh[aidx * KD + tid]
        anchor = anchor_ld if length > 0 else Float32(0.0)

        # ---- cumulative G (Triton-pinned elementary ops) ----
        cum = []
        prev = anchor
        for t in cutlass.range_constexpr(T):
            cur = _add_rn(prev, gates[t])
            cum.append(cur)
            prev = cur

        # ---- per-token native operand values ----
        # history adjusted keys first: they read s_gh, whose columns are then reused for pv
        # groups of 4 rows keep four independent smem round trips in flight; rows of the last
        # group at or past the accepted length are written as exact zeros
        a_hist = s_A.data_ptr() + tid
        g_hist = s_gh.data_ptr() + tid
        for gi in cutlass.range((length + 3) // 4, unroll=1):
            khs = []
            ghs = []
            for j in cutlass.range_constexpr(4):
                khs.append(Float32((a_hist + j * ASTR).load()))
                ghs.append((g_hist + j * KD).load())
            for jp in cutlass.range_constexpr(2):
                adjp = _adj2(anchor, ghs[2 * jp], ghs[2 * jp + 1], khs[2 * jp], khs[2 * jp + 1])
                for jj in cutlass.range_constexpr(2):
                    j = 2 * jp + jj
                    adj = adjp[jj] if gi * 4 + j < length else Float32(0.0)
                    (a_hist + j * ASTR).store(BFloat16(adj))
            a_hist = a_hist + 4 * ASTR
            g_hist = g_hist + 4 * KD
        # qoff is a permutation within each 32-column warp partition, so this
        # shared-region lifetime dependency is warp-local.
        cute.arch.sync_warp()
        qoff = (((tid // 4) ^ (tid // 16)) * 4) + tid % 4
        for t in cutlass.range_constexpr(T):
            ecum = _exp_tl(cum[t])
            s_gh[(2 * t) * KD + qoff] = _mul_rn(kn[t], ecum)
            s_gh[(2 * t + 1) * KD + qoff] = _mul_rn(qn[t], ecum)
        kqp = []
        for t in cutlass.range_constexpr(T):
            kqp.append(kn[t] * qn[t])
        red2 = _tree_reduce(kqp, (16, 8, 4, 2, 1), lane)
        for e2 in red2:
            if (lane & e2[3]) == 0:
                s_red2[_tree_idx(e2, lane) * 4 + warp] = e2[0]

        # relative pair (B operand of history/new-token coefficient products) and
        # adjusted new keys; adjusted history keys.
        for t in cutlass.range_constexpr(T):
            er = _expf(cum[t] - anchor)
            s_R[(2 * t) * ASTR + tid] = BFloat16(kn[t] * er)
            s_R[(2 * t + 1) * ASTR + tid] = BFloat16(qn[t] * er)
            ea = _expf(anchor - cum[t])
            s_A[(HROWS + t) * ASTR + tid] = BFloat16(kn[t] * ea)

        # ---- persistent writes that need no further math ----
        if cutlass.const_expr(T > 4):
            hk_w = _history_row_s(s_rep[0], p_hk, hk_ss, BFloat16, (hrow + length) * KD + tid)
        else:
            hk_w = hk_ptr + ((hrow + length) * KD + tid)
        if cutlass.const_expr(T > 4):
            hg_w = _history_row_s(s_rep[0], p_hg, hg_ss, Float32, (hrow + length) * KD + tid)
        else:
            hg_w = hg_ptr + ((hrow + length) * KD + tid)
        for t in cutlass.range_constexpr(T):
            (hk_w + t * KD).store(BFloat16(kn[t]))
            (hg_w + t * KD).store(cum[t])

        prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        prims.barrier_cta_sync(0)
        # ---- T <= 4 (N8): warps 2..3 form the coefficient products while warps 0..1 quantize;
        # the post-quantization barrier then also publishes s_coefT / s_coefn ----
        if cutlass.const_expr(MMA_N == 8):
            if warp >= 2:
                mt = warp - 2
                nt = 0
                # For N8, nt==1 would duplicate/clamp B rows and discard every result.  Keep every
                # fragment/address/accumulator local to this dynamic branch so its inactive warps do
                # not create a type-unstable SSA join.
                gid = lane // 4
                tig = lane % 4
                ra0 = 16 * mt + gid
                ra1 = ra0 + 8
                if cutlass.const_expr(NROWS < 32):
                    ra0 = ra0 if ra0 < NROWS else Int32(NROWS - 1)
                    ra1 = ra1 if ra1 < NROWS else Int32(NROWS - 1)
                rb = 8 * nt + gid
                if cutlass.const_expr(N2 < 16):
                    rb = rb if rb < N2 else Int32(N2 - 1)
                # operand fragments via ldmatrix: A tiles (rows +0/+8) x (k +0/+8); B tiles cover
                # two k16 steps per x4 load
                lmj = lane // 8
                arow = 16 * mt + (lmj % 2) * 8 + lane % 8
                if cutlass.const_expr(NROWS < 32):
                    arow = arow if arow < NROWS else Int32(NROWS - 1)
                brow = 8 * nt + lane % 8
                if cutlass.const_expr(N2 < 16):
                    brow = brow if brow < N2 else Int32(N2 - 1)
                pla = s_A.data_ptr() + (arow * ASTR + (lmj // 2) * 8)
                plb = s_R.data_ptr() + (brow * ASTR + (lmj // 2) * 16 + (lmj % 2) * 8)
                d0 = Float32(0.0)
                d1 = Float32(0.0)
                d2 = Float32(0.0)
                d3 = Float32(0.0)
                for ks in cutlass.range_constexpr(KD // 16):
                    afrag = prims.ldmatrix(pla + 16 * ks, 4, prims.MMALayout.ROW)
                    if cutlass.const_expr(ks % 2 == 0):
                        bfrag = prims.ldmatrix(plb + 16 * ks, 4, prims.MMALayout.ROW)
                    fa0 = afrag[0]
                    fa1 = afrag[1]
                    fa2 = afrag[2]
                    fa3 = afrag[3]
                    fb0 = bfrag[2 * (ks % 2)]
                    fb1 = bfrag[2 * (ks % 2) + 1]
                    # two independent accumulator chains (even / odd k16 steps) halve the
                    # dependent mma.sync latency; they are summed once below
                    if cutlass.const_expr(ks < 2):
                        # zero accumulator supplied inside the asm (immediates cannot be asm operands)
                        acc4 = cute.arch.inline_ptx(
                            "{\n.reg .f32 zz;\nmov.f32 zz, 0f00000000;\n"
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, {{$r4}, {$r5}}, "
                            "{zz, zz, zz, zz};\n}",
                            write_only_types=[Float32, Float32, Float32, Float32],
                            read_only_args=[fa0, fa1, fa2, fa3, fb0, fb1],
                        )
                        if cutlass.const_expr(ks == 0):
                            ce = acc4
                        else:
                            co = acc4
                    else:
                        prev4 = ce if cutlass.const_expr(ks % 2 == 0) else co
                        acc4 = cute.arch.inline_ptx(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, {{$r4}, {$r5}}, "
                            "{{$r6}, {$r7}, {$r8}, {$r9}};",
                            write_only_types=[Float32, Float32, Float32, Float32],
                            read_only_args=[
                                fa0,
                                fa1,
                                fa2,
                                fa3,
                                fb0,
                                fb1,
                                prev4[0],
                                prev4[1],
                                prev4[2],
                                prev4[3],
                            ],
                        )
                        if cutlass.const_expr(ks % 2 == 0):
                            ce = acc4
                        else:
                            co = acc4
                d0 = ce[0] + co[0]
                d1 = ce[1] + co[1]
                d2 = ce[2] + co[2]
                d3 = ce[3] + co[3]
                cn = 8 * nt + 2 * tig
                if cutlass.const_expr(HROWS == 16):
                    if mt == 0:
                        # transposed bf16 coefficients [n][r]; rows past the accepted length are
                        # exact zeros so the window-wide contraction only sees accepted history
                        lo_ok = gid < length
                        hi_ok = gid + 8 < length
                        pct = s_coefT.data_ptr() + (cn * CTSTR + gid)
                        pct.store(BFloat16(d0 if lo_ok else Float32(0.0)))
                        (pct + CTSTR).store(BFloat16(d1 if lo_ok else Float32(0.0)))
                        (pct + 8).store(BFloat16(d2 if hi_ok else Float32(0.0)))
                        (pct + (CTSTR + 8)).store(BFloat16(d3 if hi_ok else Float32(0.0)))
                    else:
                        if cn < N2:
                            if gid < T:
                                s_coefn[gid * MMA_N + cn] = d0
                                s_coefn[gid * MMA_N + cn + 1] = d1
                            if gid + 8 < T:
                                s_coefn[(gid + 8) * MMA_N + cn] = d2
                                s_coefn[(gid + 8) * MMA_N + cn + 1] = d3
        # ---- quantize operand rows to E2M1 + UE4M3 block-16 scales ----
        e4m3 = mlir_ir.TypeAttr.get(cutlass.Float8E4M3FN.mlir_type)
        # one thread per (operand row n = 2t + p, 16-key block): no shuffles, one scale per block
        qrow = tid // 8
        qblk = tid % 8
        qphys = qrow * 64 + (((qblk // 2) ^ ((qrow >> 1) & 3)) * 16) + (qblk % 2) * 8
        if cutlass.const_expr(MMA_N != 8 and N2 <= 12):
            # N16 with T <= 6: every operand row lives in warps 0..2; warp 3 writes the padding
            # rows 12..15 and forms the coefficient products (below)
            if warp < 3:
                # one 16-lane half-warp per token (two operand rows x eight blocks); padding rows
                # read row 0 and contribute 0 to the token maxima
                qrow_c = qrow if qrow < N2 else Int32(0)
                qbase = s_gh.data_ptr() + qrow_c * KD
                qv4 = []
                for i in cutlass.range_constexpr(4):
                    qg = (qblk * 4 + i) ^ qblk
                    qv4.append((qbase + 4 * qg).load(alignment=16, count=4))
                vals = []
                for i in cutlass.range_constexpr(16):
                    vals.append(qv4[i // 4][i % 4])
                bmx = _amax16(vals)
                bmx = bmx if qrow < N2 else Float32(0.0)
                # per-half-warp max as two full-warp reductions (exact; values are >= 0)
                lo_in = bmx if lane < 16 else Float32(0.0)
                hi_in = Float32(0.0) if lane < 16 else bmx
                tmx_lo = cute.arch.inline_ptx(
                    "redux.sync.max.f32 {$w0}, {$r0}, 0xffffffff;",
                    write_only_types=[Float32],
                    read_only_args=[lo_in],
                )
                tmx_hi = cute.arch.inline_ptx(
                    "redux.sync.max.f32 {$w0}, {$r0}, 0xffffffff;",
                    write_only_types=[Float32],
                    read_only_args=[hi_in],
                )
                tmx = tmx_lo if lane < 16 else tmx_hi
                if qrow < N2:
                    outer = tmx * Float32(INV2688)
                    safe = Float32(1.0) if outer == Float32(0.0) else outer
                    den = Float32(6.0) * safe
                    sc = cute.arch.fmin(Float32(448.0), _div_scale(bmx, den, "QSC_DIV_N16"))
                    w16 = prims.dialect.convert_f32x2_to_f8x2(
                        MT.i16(),
                        Float32(0.0),
                        sc,
                        e4m3,
                        rnd=nvvm_raw.FPRoundingMode.RN,
                        sat=nvvm_raw.SaturationMode.SATFINITE,
                    )
                    sbyte = Int32(cutlass.Int16(w16)) & Int32(0xFF)
                    # SFB staging image: one byte store per block
                    sfb8 = cutlass.inttoptr(
                        s_sfb.data_ptr().toint(), cutlass.AddressSpace.smem, Uint8
                    )
                    (sfb8 + (((qblk // 4) * 128 + qrow * 4) * 4 + (qblk % 4))).store(Uint8(sbyte))
                    if tid % 16 == 0:
                        oscv = outer * outer_s
                        (s_osc.data_ptr() + qrow).store(
                            cutlass.Vector.from_elements((oscv, oscv), Float32), alignment=8
                        )
                    # decoded scale: hardware e4m3 -> f16 -> f32 (exact)
                    expd = _e4m3_f32(w16) * outer
                    if expd >= Float32(7.888609052210118e-31) and expd < Float32(
                        8.507059173023462e37
                    ):
                        rexp = _rcp_fast(expd)
                        nexp = -expd
                        qw0 = _quant8(vals[0:8], nexp, rexp)
                        qw1 = _quant8(vals[8:16], nexp, rexp)
                        (sB.data_ptr() + qphys // 4).store(
                            cutlass.Vector.from_elements((Uint32(qw0), Uint32(qw1)), Uint32),
                            alignment=8,
                        )
                    elif expd == Float32(0.0):
                        (sB.data_ptr() + qphys // 4).store(
                            cutlass.Vector.from_elements((Uint32(0), Uint32(0)), Uint32),
                            alignment=8,
                        )
                    elif expd < Float32(7.888609052210118e-31):
                        ed = expd * Float32(18446744073709551616.0)
                        rexs = Float32(1.0) / ed
                        vsc = [v * Float32(18446744073709551616.0) for v in vals]
                        qws0 = _quant8(vsc[0:8], -ed, rexs)
                        qws1 = _quant8(vsc[8:16], -ed, rexs)
                        (sB.data_ptr() + qphys // 4).store(
                            cutlass.Vector.from_elements((Uint32(qws0), Uint32(qws1)), Uint32),
                            alignment=8,
                        )
                    else:
                        # expd >= 2^126 or NaN: plain IEEE reciprocal (as before)
                        rexh = Float32(1.0) / expd
                        qwh0 = _quant8(vals[0:8], -expd, rexh)
                        qwh1 = _quant8(vals[8:16], -expd, rexh)
                        (sB.data_ptr() + qphys // 4).store(
                            cutlass.Vector.from_elements((Uint32(qwh0), Uint32(qwh1)), Uint32),
                            alignment=8,
                        )
        else:
            if qrow < N2:
                # Each token occupies one 16-lane half-warp (two operand rows x eight blocks).
                # On odd-T tail warps, exclude the inactive upper half from synchronizing shuffles.
                if cutlass.const_expr(T % 2 == 1):
                    qmask = Int32(0x0000FFFF) if warp == T // 2 else Int32(-1)
                else:
                    qmask = Int32(-1)
                qbase = s_gh.data_ptr() + qrow * KD
                qv4 = []
                for i in cutlass.range_constexpr(4):
                    qg = (qblk * 4 + i) ^ qblk
                    qv4.append((qbase + 4 * qg).load(alignment=16, count=4))
                vals = []
                for i in cutlass.range_constexpr(16):
                    vals.append(qv4[i // 4][i % 4])
                bmx = _amax16(vals)
                tmx = bmx
                for o in cutlass.range_constexpr(4):
                    tmx = cute.arch.fmax(
                        tmx, cute.arch.shuffle_sync_bfly(tmx, offset=8 >> o, mask=qmask)
                    )
                outer = tmx * Float32(INV2688)
                safe = Float32(1.0) if outer == Float32(0.0) else outer
                den = Float32(6.0) * safe
                sc = cute.arch.fmin(Float32(448.0), bmx / den)
                w16 = prims.dialect.convert_f32x2_to_f8x2(
                    MT.i16(),
                    Float32(0.0),
                    sc,
                    e4m3,
                    rnd=nvvm_raw.FPRoundingMode.RN,
                    sat=nvvm_raw.SaturationMode.SATFINITE,
                )
                sbyte = Int32(cutlass.Int16(w16)) & Int32(0xFF)
                sfw = sbyte | (cute.arch.shuffle_sync_down(sbyte, offset=1, mask=qmask) << 8)
                sfw = sfw | (cute.arch.shuffle_sync_down(sfw, offset=2, mask=qmask) << 16)
                if tid % 16 == 0:
                    oscv = outer * outer_s
                    (s_osc.data_ptr() + qrow).store(
                        cutlass.Vector.from_elements((oscv, oscv), Float32), alignment=8
                    )
                if qblk % 4 == 0:
                    s_sfb[(qblk // 4) * 128 + qrow * 4] = Uint32(sfw)
                ex = sbyte >> 3
                mn = sbyte & 7
                nrm_bits = ((ex + 120) << 23) | (mn << 20)
                scf_n = nrm_bits.bitcast(Float32)
                scf_s = Float32(mn) * Float32(0.001953125)
                scf = scf_n if ex > 0 else scf_s
                expd = scf * outer
                if expd >= Float32(7.888609052210118e-31):
                    # normal-range scale (>= 2^-100): one correctly rounded reciprocal shared by the
                    # 16 elements, Markstein-corrected quotients (bit-identical to div.rn)
                    rexp = Float32(1.0) / expd
                    nexp = -expd
                    qw0 = _quant8(vals[0:8], nexp, rexp)
                    qw1 = _quant8(vals[8:16], nexp, rexp)
                    (sB.data_ptr() + qphys // 4).store(
                        cutlass.Vector.from_elements((Uint32(qw0), Uint32(qw1)), Uint32),
                        alignment=8,
                    )
                elif expd == Float32(0.0):
                    # all-zero block scale (underflowed operand block): every code is +0, no division
                    (sB.data_ptr() + qphys // 4).store(
                        cutlass.Vector.from_elements((Uint32(0), Uint32(0)), Uint32), alignment=8
                    )
                else:
                    # tiny nonzero scale: rescale numerator and denominator by 2^64 (exact), so the
                    # Markstein quotients stay bit-identical to div.rn without the IEEE slow path
                    ed = expd * Float32(18446744073709551616.0)
                    rexs = Float32(1.0) / ed
                    vsc = [v * Float32(18446744073709551616.0) for v in vals]
                    qws0 = _quant8(vsc[0:8], -ed, rexs)
                    qws1 = _quant8(vsc[8:16], -ed, rexs)
                    (sB.data_ptr() + qphys // 4).store(
                        cutlass.Vector.from_elements((Uint32(qws0), Uint32(qws1)), Uint32),
                        alignment=8,
                    )
            else:
                # Only descriptor-visible padding rows need packed zeros.  SFB was cleared in full
                # before this phase, so its padding scales remain exact zero without extra stores.
                if qrow < MMA_N:
                    sB[qphys // 4] = Uint32(0)
                    sB[qphys // 4 + 1] = Uint32(0)
        if cutlass.const_expr(MMA_N != 8 and N2 <= 12):
            # N16 with T <= 6: warp 3 has no operand row to quantize -> it forms all four
            # coefficient tiles meanwhile; the post-quantization barrier publishes them
            if warp == 3:
                _coef_tiles_t(lane, length, s_A, s_R, s_coefT, s_coefn, T, N2, NROWS, MMA_N)
        if cutlass.const_expr(MMA_N != 8 and N2 <= 12):
            if warp < 3:
                prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
        else:
            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
        prims.barrier_cta_sync(0)

        # ---- native NVFP4 MMA: D[128 x 16] = S(E2M1,UE4M3) x [k_t;q_t](E2M1,UE4M3)^T ----
        if warp == 0:
            if cutlass.const_expr(POOL_VEC != 0):
                while not prims.mbarrier_try_wait_parity(s_mbar_a, 0):
                    pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            dv01 = (s_desc.data_ptr()).load(alignment=16, count=2)
            dv23 = (s_desc.data_ptr() + 2).load(alignment=16, count=2)
            desc_a = prims.Tcgen05SmemDesc(dv01[0])
            desc_b = prims.Tcgen05SmemDesc(dv01[1])
            desc_sf = prims.Tcgen05SmemDesc(dv23[0])
            idesc = prims.Tcgen05MxOmmaInstrDesc(Int32(dv23[1]))
            d_ptr = cutlass.inttoptr(tmem_base, 6, Float32)
            sf_shape, sf_mc = prims.S2TCopyMode.S2T_32x128b_WARPX4
            if prims.elect_sync():
                for kb in cutlass.range_constexpr(2):
                    prims.tcgen05_cp(
                        sf_shape,
                        cutlass.inttoptr(tmem_base + 24 + 4 * kb, 6, Int32),
                        desc_sf + 32 * kb,
                        group=prims.CTAGroup.CTA_1,
                        multicast=sf_mc,
                    )
                for kb in cutlass.range_constexpr(2):
                    sa_ptr = cutlass.inttoptr(tmem_base + 16 + 4 * kb, 6, Int32)
                    sb_ptr = cutlass.inttoptr(tmem_base + 24 + 4 * kb, 6, Int32)
                    prims.tcgen05_mma_block_scale(
                        prims.MMABlockScaleKind.MXF4NVF4,
                        prims.CTAGroup.CTA_1,
                        d_ptr,
                        desc_a + 2 * kb,
                        desc_b + 2 * kb,
                        idesc,
                        Boolean(kb != 0),
                        scale_a=sa_ptr,
                        scale_b=sb_ptr,
                        scale_vec_size=prims.Tcgen05MMABlockScale.BLOCK16,
                    )
                prims.tcgen05_commit(s_mbar, group=prims.CTAGroup.CTA_1)

        if cutlass.const_expr(MMA_N != 8 and N2 > 12):
            # ---- coefficient products on warp tensor cores (bf16 x bf16 -> fp32) ----
            # C[32 x 16] = A[32 x 128] R[16 x 128]^T; rows 0..15 history (bf16-rounded result),
            # rows 16..16+T-1 new tokens (fp32).  Warp w: m-tile w // 2, n-tile w % 2, 8 k16 steps.
            mt = warp // 2
            nt = warp % 2
            if cutlass.const_expr(MMA_N == 8):
                coefficient_active = nt == 0
            else:
                coefficient_active = True
            if coefficient_active:
                # For N8, nt==1 would duplicate/clamp B rows and discard every result.  Keep every
                # fragment/address/accumulator local to this dynamic branch so its inactive warps do
                # not create a type-unstable SSA join.
                gid = lane // 4
                tig = lane % 4
                ra0 = 16 * mt + gid
                ra1 = ra0 + 8
                if cutlass.const_expr(NROWS < 32):
                    ra0 = ra0 if ra0 < NROWS else Int32(NROWS - 1)
                    ra1 = ra1 if ra1 < NROWS else Int32(NROWS - 1)
                rb = 8 * nt + gid
                if cutlass.const_expr(N2 < 16):
                    rb = rb if rb < N2 else Int32(N2 - 1)
                # operand fragments via ldmatrix: A tiles (rows +0/+8) x (k +0/+8); B tiles cover
                # two k16 steps per x4 load
                lmj = lane // 8
                arow = 16 * mt + (lmj % 2) * 8 + lane % 8
                if cutlass.const_expr(NROWS < 32):
                    arow = arow if arow < NROWS else Int32(NROWS - 1)
                brow = 8 * nt + lane % 8
                if cutlass.const_expr(N2 < 16):
                    brow = brow if brow < N2 else Int32(N2 - 1)
                pla = s_A.data_ptr() + (arow * ASTR + (lmj // 2) * 8)
                plb = s_R.data_ptr() + (brow * ASTR + (lmj // 2) * 16 + (lmj % 2) * 8)
                d0 = Float32(0.0)
                d1 = Float32(0.0)
                d2 = Float32(0.0)
                d3 = Float32(0.0)
                for ks in cutlass.range_constexpr(KD // 16):
                    afrag = prims.ldmatrix(pla + 16 * ks, 4, prims.MMALayout.ROW)
                    if cutlass.const_expr(ks % 2 == 0):
                        bfrag = prims.ldmatrix(plb + 16 * ks, 4, prims.MMALayout.ROW)
                    fa0 = afrag[0]
                    fa1 = afrag[1]
                    fa2 = afrag[2]
                    fa3 = afrag[3]
                    fb0 = bfrag[2 * (ks % 2)]
                    fb1 = bfrag[2 * (ks % 2) + 1]
                    # two independent accumulator chains (even / odd k16 steps) halve the
                    # dependent mma.sync latency; they are summed once below
                    if cutlass.const_expr(ks < 2):
                        # zero accumulator supplied inside the asm (immediates cannot be asm operands)
                        acc4 = cute.arch.inline_ptx(
                            "{\n.reg .f32 zz;\nmov.f32 zz, 0f00000000;\n"
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, {{$r4}, {$r5}}, "
                            "{zz, zz, zz, zz};\n}",
                            write_only_types=[Float32, Float32, Float32, Float32],
                            read_only_args=[fa0, fa1, fa2, fa3, fb0, fb1],
                        )
                        if cutlass.const_expr(ks == 0):
                            ce = acc4
                        else:
                            co = acc4
                    else:
                        prev4 = ce if cutlass.const_expr(ks % 2 == 0) else co
                        acc4 = cute.arch.inline_ptx(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, {{$r4}, {$r5}}, "
                            "{{$r6}, {$r7}, {$r8}, {$r9}};",
                            write_only_types=[Float32, Float32, Float32, Float32],
                            read_only_args=[
                                fa0,
                                fa1,
                                fa2,
                                fa3,
                                fb0,
                                fb1,
                                prev4[0],
                                prev4[1],
                                prev4[2],
                                prev4[3],
                            ],
                        )
                        if cutlass.const_expr(ks % 2 == 0):
                            ce = acc4
                        else:
                            co = acc4
                d0 = ce[0] + co[0]
                d1 = ce[1] + co[1]
                d2 = ce[2] + co[2]
                d3 = ce[3] + co[3]
                cn = 8 * nt + 2 * tig
                if cutlass.const_expr(HROWS == 16):
                    if mt == 0:
                        # transposed bf16 coefficients [n][r]; rows past the accepted length are
                        # exact zeros so the window-wide contraction only sees accepted history
                        lo_ok = gid < length
                        hi_ok = gid + 8 < length
                        pct = s_coefT.data_ptr() + (cn * CTSTR + gid)
                        pct.store(BFloat16(d0 if lo_ok else Float32(0.0)))
                        (pct + CTSTR).store(BFloat16(d1 if lo_ok else Float32(0.0)))
                        (pct + 8).store(BFloat16(d2 if hi_ok else Float32(0.0)))
                        (pct + (CTSTR + 8)).store(BFloat16(d3 if hi_ok else Float32(0.0)))
                    else:
                        if cn < N2:
                            if gid < T:
                                s_coefn[gid * MMA_N + cn] = d0
                                s_coefn[gid * MMA_N + cn + 1] = d1
                            if gid + 8 < T:
                                s_coefn[(gid + 8) * MMA_N + cn] = d2
                                s_coefn[(gid + 8) * MMA_N + cn + 1] = d3
            prims.barrier_cta_sync(0)

        # ---- history-U contraction on warp tensor cores (overlaps the native MMA) ----
        # H[i, n] = sum_r U[r, i] coef[r, n]: A = U^T via ldmatrix.trans of the [r][i] staging,
        # B = transposed bf16 coefficients; fp32 fragments are transposed through the now free
        # s_gh region (XOR-swizzled 16 B chunks) so that thread = value row again.
        gq = lane // 4
        tq = lane % 4
        lj = lane // 8
        lrow = (lj // 2) * 8 + lane % 8
        hsc = s_gh.data_ptr() + warp * 512
        for mq in cutlass.range_constexpr(2):
            afr = prims.ldmatrix(
                s_uh.data_ptr() + (lrow * ASTR + warp * 32 + 16 * mq + (lj % 2) * 8),
                4,
                prims.MMALayout.COL,
            )
            for nq in cutlass.range_constexpr(MMA_N // 8):
                pbt = s_coefT.data_ptr() + ((8 * nq + gq) * CTSTR + 2 * tq)
                hb0 = pbt.load(alignment=4, count=2).bitcast(Int32)[0]
                hb1 = (pbt + 8).load(alignment=4, count=2).bitcast(Int32)[0]
                h0, h1, h2, h3 = cute.arch.inline_ptx(
                    "{\n.reg .f32 zz;\nmov.f32 zz, 0f00000000;\n"
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, {{$r4}, {$r5}}, "
                    "{zz, zz, zz, zz};\n}",
                    write_only_types=[Float32, Float32, Float32, Float32],
                    read_only_args=[afr[0], afr[1], afr[2], afr[3], hb0, hb1],
                )
                c4 = 2 * nq + tq // 2
                hr0 = 16 * mq + gq
                hr1 = hr0 + 8
                (
                    hsc
                    + (
                        hr0 * 16
                        + (c4 ^ ((((hr0 >> 1) & 1) << 1) | ((hr0 >> 2) & 1))) * 4
                        + 2 * (tq % 2)
                    )
                ).store(cutlass.Vector.from_elements((h0, h1), Float32), alignment=8)
                (
                    hsc
                    + (
                        hr1 * 16
                        + (c4 ^ ((((hr1 >> 1) & 1) << 1) | ((hr1 >> 2) & 1))) * 4
                        + 2 * (tq % 2)
                    )
                ).store(cutlass.Vector.from_elements((h2, h3), Float32), alignment=8)
        cute.arch.sync_warp()
        hist = []
        for c in cutlass.range_constexpr(MMA_N // 4):
            # chunk swizzle x(r) = bit-swapped (r >> 1) & 3: conflict-free for both the half-warp STS.64
            # fragment stores (rows of equal parity in distinct chunk pairs) and these quarter-warp
            # LDS.128 row reads
            hv = (
                hsc + (lane * 16 + (c ^ ((((lane >> 1) & 1) << 1) | ((lane >> 2) & 1))) * 4)
            ).load(alignment=16, count=4)
            for e in cutlass.range_constexpr(4):
                hist.append(hv[e])

        # ---- epilogue: thread = value row i ----
        while not prims.mbarrier_try_wait_parity(s_mbar, 0):
            pass
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        acc_ptr = cutlass.inttoptr(tmem_base + ((warp * 32) << 16), 6, Float32)
        accv = prims.tcgen05_ld("32x32b", acc_ptr, num=MMA_N)
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        oscs = []
        for c in cutlass.range_constexpr((N2 + 3) // 4):
            ov4 = (s_osc.data_ptr() + 4 * c).load(alignment=16, count=4)
            for e in cutlass.range_constexpr(4):
                oscs.append(ov4[e])
        pred = []
        for t in cutlass.range_constexpr(T):
            pk0, pq0 = _fma2(
                accv[2 * t],
                accv[2 * t + 1],
                oscs[2 * t],
                oscs[2 * t + 1],
                hist[2 * t],
                hist[2 * t + 1],
            )
            pred.append(pk0)
            pred.append(pq0)
        kqs = []
        for t in cutlass.range_constexpr(T):
            kv4 = (s_red2.data_ptr() + t * 4).load(alignment=16, count=4)
            kqs.append(kv4[0] + kv4[1] + kv4[2] + kv4[3])
        bsigs = []
        for c in cutlass.range_constexpr((T + 3) // 4):
            bv4 = (s_bsig.data_ptr() + 4 * c).load(alignment=16, count=4)
            for e in cutlass.range_constexpr(4):
                bsigs.append(bv4[e])
        us = []
        cores = []
        for t in cutlass.range_constexpr(T):
            pk = pred[2 * t]
            pq = pred[2 * t + 1]
            for s in cutlass.range_constexpr(t):
                # (k, q) coefficient pair of token t against new token s: one 8 B read
                cf2 = (s_coefn.data_ptr() + (s * MMA_N + 2 * t)).load(alignment=8, count=2)
                pk, pq = _fma2(cf2[0], cf2[1], us[s], us[s], pk, pq)
            u = bsigs[t] * (vv[t] - pk)
            us.append(u)
            cores.append(_bf(pq + u * kqs[t]))
        csq = []
        for t in cutlass.range_constexpr(T):
            csq.append(cores[t] * cores[t])
        red3 = _tree_reduce(csq, (16, 8, 4, 2, 1), lane)
        for e3 in red3:
            if (lane & e3[3]) == 0:
                s_red3[_tree_idx(e3, lane) * 4 + warp] = e3[0]
        if cutlass.const_expr(Z_VEC != 0):
            prims.cp_async_wait_group(0)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        prims.barrier_cta_sync(0)
        if warp == 0:
            prims.tcgen05_dealloc(
                cutlass.inttoptr(tmem_base, 6, Float32), TMEM_COLS, group=prims.CTAGroup.CTA_1
            )
        if cutlass.const_expr(PDL != 0):
            prims.griddepcontrol("launch_dependents")
        nwf = Float32(nw)
        if cutlass.const_expr(T > 4):
            hu_w = _history_row_s(s_rep[0], p_hu, hu_ss, BFloat16, (hrow + length) * KD + tid)
        else:
            hu_w = hu_ptr + ((hrow + length) * KD + tid)
        o_row2 = out_ptr + ((b64 * T * H + h64) * KD + tid)
        # tokens are processed in lane pairs (t, t + 1) on the f32x2 pipe
        rss = []
        zfs = []
        for t in cutlass.range_constexpr(T):
            rv4 = (s_red3.data_ptr() + t * 4).load(alignment=16, count=4)
            tot = rv4[0] + rv4[1] + rv4[2] + rv4[3]
            if cutlass.const_expr(EPS_FTZ != 0):
                # eps is a normal float -> the operand is never subnormal: the .ftz form is identical
                rss.append(
                    cute.arch.inline_ptx(
                        "rsqrt.approx.ftz.f32 {$w0}, {$r0};",
                        write_only_types=[Float32],
                        read_only_args=[tot * Float32(1.0 / 128.0) + eps],
                    )
                )
            else:
                rss.append(cute.rsqrt(tot * Float32(1.0 / 128.0) + eps))
            if cutlass.const_expr(Z_VEC != 0):
                zfs.append(Float32(s_z[t * KD + tid]))
            else:
                zfs.append(Float32(zs[t]))
        outs = []
        for tp in cutlass.range_constexpr(T // 2):
            t0 = 2 * tp
            sg0, sg1 = _sig2(zfs[t0], zfs[t0 + 1])
            o0, o1 = _mul2(cores[t0], cores[t0 + 1], rss[t0], rss[t0 + 1])
            o0, o1 = _mul2(o0, o1, nwf, nwf)
            o0, o1 = _mul2(o0, o1, sg0, sg1)
            outs.append(o0)
            outs.append(o1)
        if cutlass.const_expr(T % 2 == 1):
            sgl = _rcpf(Float32(1.0) + _expf(-zfs[T - 1]))
            outs.append(cores[T - 1] * rss[T - 1] * nwf * sgl)
        for t in cutlass.range_constexpr(T):
            (o_row2 + t * HK).store(BFloat16(outs[t]))
            (hu_w + t * KD).store(BFloat16(us[t]))
        if h == 0:
            if tid == 0:
                (oldlen_ptr + b64).store(length)


@cute.jit
def _replay_launch(
    p_raw: Int64,
    p_conv: Int64,
    p_cw: Int64,
    p_g: Int64,
    p_beta: Int64,
    p_z: Int64,
    p_nw: Int64,
    p_pool: Int64,
    p_slots: Int64,
    p_dummy: Int64,
    p_ridx: Int64,
    p_hk: Int64,
    p_hu: Int64,
    p_hg: Int64,
    p_hlen: Int64,
    p_a_log: Int64,
    p_dtb: Int64,
    p_out: Int64,
    p_cand: Int64,
    p_oldlen: Int64,
    B: Int32,
    n_slots: Int32,
    candidate_bs: Int64,
    lower: Float32,
    eps: Float32,
    stream: cuda.CUstream,
    H: cutlass.Constexpr[int],
    raw_bs: cutlass.Constexpr[int],
    raw_rs: cutlass.Constexpr[int],
    conv_ss: cutlass.Constexpr[int],
    g_bs: cutlass.Constexpr[int],
    g_rs: cutlass.Constexpr[int],
    beta_bs: cutlass.Constexpr[int],
    beta_rs: cutlass.Constexpr[int],
    z_bs: cutlass.Constexpr[int],
    z_rs: cutlass.Constexpr[int],
    pool_ss: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    BOUNDED: cutlass.Constexpr[int],
    POOL_VEC: cutlass.Constexpr[int],
    CAP: cutlass.Constexpr[int],
    Z_VEC: cutlass.Constexpr[int],
    PDL: cutlass.Constexpr[int],
    hk_ss: cutlass.Constexpr[int],
    hu_ss: cutlass.Constexpr[int],
    hg_ss: cutlass.Constexpr[int],
    hl_ss: cutlass.Constexpr[int],
    EPS_FTZ: cutlass.Constexpr[int],
):
    # checkpoint primary values: [slot][head][row 0..127][64 B] (pool_ss / STATE_BYTES byte strides)
    if cutlass.const_expr(POOL_VEC != 0):
        ck_desc = create_tensor_map_tiled(
            global_address=p_pool,
            dtype=cutlass.Uint8,
            global_dims=[64, 128, H, n_slots],
            global_strides=[64 // 16, STATE_BYTES // 16, pool_ss // 16],
            box_dims=[64, 128, 1, 1],
            swizzle=TensorMapSwizzle.s64b,
        )
    else:
        # unused (byte-granular checkpoint path): any 16 B aligned address
        ck_desc = create_tensor_map_tiled(
            global_address=(p_pool // 16) * 16,
            dtype=cutlass.Uint8,
            global_dims=[64, 128, 1, 1],
            global_strides=[64 // 16, 8192 // 16, 8192 // 16],
            box_dims=[64, 128, 1, 1],
            swizzle=TensorMapSwizzle.s64b,
        )
    _replay_kernel(
        p_raw,
        p_conv,
        p_cw,
        p_g,
        p_beta,
        p_z,
        p_nw,
        p_pool,
        p_slots,
        p_dummy,
        p_ridx,
        p_hk,
        p_hu,
        p_hg,
        p_hlen,
        p_a_log,
        p_dtb,
        p_out,
        p_cand,
        p_oldlen,
        ck_desc,
        candidate_bs,
        lower,
        eps,
        H,
        raw_bs,
        raw_rs,
        conv_ss,
        g_bs,
        g_rs,
        beta_bs,
        beta_rs,
        z_bs,
        z_rs,
        pool_ss,
        T,
        BOUNDED,
        POOL_VEC,
        CAP,
        Z_VEC,
        PDL,
        hk_ss,
        hu_ss,
        hg_ss,
        hl_ss,
        EPS_FTZ,
    ).launch(
        grid=(H, B, 1),
        block=(NT, 1, 1),
        stream=stream,
        use_pdl=PDL != 0,
        min_blocks_per_mp=6 if cutlass.const_expr(T > 4) else 5,
        preferred_smem_carveout=100,
    )


def _check_histories(
    hk: torch.Tensor,
    hu: torch.Tensor,
    hg: torch.Tensor,
    lengths: torch.Tensor,
    heads: int,
    capacity: int,
) -> None:
    """Validate one layer's K/U/G histories and lengths, addressed by replay slot.

    Replay slots may be strided (pool pages); each slot's [H, capacity, 128]
    payload must be dense and 16-byte aligned. ``lengths`` is an int32 vector
    with one entry per replay slot and any stride.
    """
    slots = hk.shape[0]
    for history, dtype in ((hk, torch.bfloat16), (hu, torch.bfloat16), (hg, torch.float32)):
        if history.shape != (slots, heads, capacity, KD) or history.dtype != dtype:
            raise ValueError("2FP4 histories must be [replay slots, H, capacity, 128]")
        if history.stride()[1:] != (capacity * KD, KD, 1):
            raise ValueError("2FP4 history slot payloads must be dense")
        if history.data_ptr() % 16 or (history.stride(0) * history.element_size()) % 16:
            raise ValueError("2FP4 history slots must be 16-byte aligned")
    if lengths.ndim != 1 or lengths.shape[0] != slots or lengths.dtype != torch.int32:
        raise ValueError("2FP4 history lengths must be int32 with one entry per replay slot")


_compiled = {}


def _get_compiled(
    T,
    bounded,
    pool_vec,
    z_vec,
    H,
    CAP,
    raw_bs,
    raw_rs,
    conv_ss,
    g_bs,
    g_rs,
    beta_bs,
    beta_rs,
    z_bs,
    z_rs,
    pool_ss,
    pdl,
    history_strides,
    eps_ftz,
):
    # Specialize only stable model topology and layout metadata.  Request shape B,
    # accepted lengths, slots, masks and every tensor value remain runtime inputs.
    key = (
        torch.cuda.current_device(),
        T,
        bounded,
        pool_vec,
        z_vec,
        H,
        CAP,
        raw_bs,
        raw_rs,
        conv_ss,
        g_bs,
        g_rs,
        beta_bs,
        beta_rs,
        z_bs,
        z_rs,
        pool_ss,
        pdl,
        history_strides,
        eps_ftz,
    )
    fn = _compiled.get(key)
    if fn is None:
        z64 = Int64(0)
        # Runtime ABI: 20 pointers, B, candidate stride, lower, eps and stream. Every argument
        # following stream is a compile-time topology/layout constant.
        args = [z64] * 20 + [Int32(1), Int32(1), Int64(0), Float32(0.0), Float32(0.0)]
        fn = cute.compile(
            _replay_launch,
            *args,
            cuda.CUstream(0),
            H,
            raw_bs,
            raw_rs,
            conv_ss,
            g_bs,
            g_rs,
            beta_bs,
            beta_rs,
            z_bs,
            z_rs,
            pool_ss,
            T,
            bounded,
            pool_vec,
            CAP,
            z_vec,
            pdl,
            *history_strides,
            eps_ftz,
        )
        _compiled[key] = fn
    return fn


@torch.no_grad()
def fused_replay(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    pool: torch.Tensor,
    slots: torch.Tensor,
    dummy: torch.Tensor,
    replay_indices: torch.Tensor,
    hk: torch.Tensor,
    hu: torch.Tensor,
    hg: torch.Tensor,
    lengths: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    norm_eps: float,
    out: torch.Tensor,
    old_lengths: torch.Tensor,
    *,
    candidate_x: torch.Tensor,
    tokens_per_step: int = 1,
) -> None:
    """Write [B*T,H,128] BF16 output and candidate histories in one launch.

    Checkpoint/Conv slot strides and projection row strides are preserved.
    K/U/G histories and their lengths are addressed by ``replay_indices`` (one
    replay slot per batch row); replay slots may be strided, and each slot keeps
    its configured capacity when T shrinks.
    Candidate Conv rows use the configured batch stride, not the current T.
    Neither the checkpoints nor the history lengths are advanced here.
    """
    batch, heads = slots.numel(), a_log.numel()
    tokens, capacity = tokens_per_step, hk.shape[2]
    if not batch:
        return
    if not 1 <= tokens <= 8 or not HROWS + tokens <= capacity <= HROWS + 8:
        raise ValueError("2FP4 MTP replay requires 1..8 tokens and sufficient history capacity")
    _check_histories(hk, hu, hg, lengths, heads, capacity)
    if candidate_x.stride(1) != 3 * heads * KD or candidate_x.stride(2) != 1:
        raise ValueError("Candidate Conv tokens require dense channel rows")
    if conv_state.stride(1) != 3 or conv_state.stride(2) != 1:
        raise ValueError("Conv4 state requires dense inner dimensions")
    raw_x = x.view(batch, tokens, 3 * heads * KD)
    raw_g = gate.view(batch, tokens, heads, KD)
    raw_beta = beta.view(batch, tokens, heads)
    output_gate = z.view(batch, tokens, heads, KD)
    pool_vec = int(pool.stride(0) % 16 == 0 and pool.data_ptr() % 16 == 0)
    # output-gate rows can be staged with 16 B cp.async when every (row, token, head) row start
    # is 16 B aligned (layout metadata only)
    z_vec = int(
        output_gate.stride(0) % 8 == 0
        and output_gate.stride(1) % 8 == 0
        and output_gate.data_ptr() % 16 == 0
    )
    strides = (
        raw_x.stride(0),
        raw_x.stride(1),
        conv_state.stride(0),
        raw_g.stride(0),
        raw_g.stride(1),
        raw_beta.stride(0),
        raw_beta.stride(1),
        output_gate.stride(0),
        output_gate.stride(1),
        pool.stride(0),
    )
    fn = _get_compiled(
        tokens,
        int(lower_bound is not None),
        pool_vec,
        z_vec,
        heads,
        capacity,
        *strides,
        int(get_env_enable_pdl()),
        # Replay-slot strides in bytes: fixed for a cache allocation.
        (
            hk.stride(0) * hk.element_size(),
            hu.stride(0) * hu.element_size(),
            hg.stride(0) * hg.element_size(),
            lengths.stride(0) * lengths.element_size(),
        ),
        int(float(norm_eps) >= 1.1754943508222875e-38),
    )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn(
        raw_x.data_ptr(),
        conv_state.data_ptr(),
        conv_weight.data_ptr(),
        raw_g.data_ptr(),
        raw_beta.data_ptr(),
        output_gate.data_ptr(),
        norm_weight.data_ptr(),
        pool.data_ptr(),
        slots.data_ptr(),
        dummy.data_ptr(),
        replay_indices.data_ptr(),
        hk.data_ptr(),
        hu.data_ptr(),
        hg.data_ptr(),
        lengths.data_ptr(),
        a_log.data_ptr(),
        dt_bias.data_ptr(),
        out.data_ptr(),
        candidate_x.data_ptr(),
        old_lengths.data_ptr(),
        batch,
        pool.shape[0],
        candidate_x.stride(0),
        float(lower_bound) if lower_bound is not None else 0.0,
        float(norm_eps),
        stream,
    )
