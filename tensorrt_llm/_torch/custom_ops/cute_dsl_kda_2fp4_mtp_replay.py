# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused native 2FP4 KDA MTP replay with occupancy-based kernel dispatch."""

# Fused Conv4/SiLU + QK norm + native-NVFP4 KDA MTP replay + RMSNorm/sigmoid gate.
# One warp-specialized CTA per (head, batch-row). Both checkpoint projections (state x K-operand
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

from ..flashinfer_utils import get_env_enable_pdl
from . import cute_dsl_kda_2fp4_mtp_replay_packed as _packed
from .cute_dsl_kda_2fp4_commit import STATE_BYTES
from .cute_dsl_kda_2fp4_mtp_replay_packed import _check_histories, _ld_keep_s32, _ld_keep_u8

KD = 128  # K = V = 128
HROWS = 16  # history window W
NT = 128  # threads in each main/coefficient warp group
ASTR = 136  # padded bf16 row stride for dot-product operands (272 B, conflict-free)
CTSTR = 24  # padded bf16 row stride of the transposed history coefficients (48 B, conflict-free)
LOG2E = 1.4426950408889634
INV2688 = 1.0 / (6 * 448)
GMEM = cutlass.AddressSpace.gmem
TMEM_COLS = 32  # [0,16) accumulator, [16,24) SFA (2 k-blocks x 4 cols), [24,32) SFB
WS_BLOCKS = 3  # CTAs per SM targeted by the 256-thread warp-specialized variant
# launches with more than 1.5 items per SM go to the packed one-item-per-CTA kernel (6 CTAs/SM)
PACKED_MIN_2X = 3
LIVE_ROWS_MIN_GRID = 160  # above this many CTAs, stage only the accepted history rows


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


# rcp.rn of xd >= 1 is the hardware fast path above (MUFU.RCP + one fused Newton step, exactly
# what rcp.rn compiles to for in-range operands) without the slow-path range check; xd = inf -> 0.
_RCP_GE1_PTX = """{
.reg .b32 rr, rm, rq;
rcp.approx.ftz.f32 rr, {$r0};
neg.f32 rm, {$r0};
fma.rn.ftz.f32 rq, rm, rr, 0f3F800000;
fma.rn.ftz.f32 {$w0}, rr, rq, rr;
}"""


def _rcp_ge1(d):
    # correctly rounded 1/d for 1 <= d < 2^126 (callers route larger d to the IEEE division)
    return cute.arch.inline_ptx(_RCP_GE1_PTX, write_only_types=[Float32], read_only_args=[d])


def _rcp_ge1_n(ds):
    # correctly rounded 1/d (== div.rn(1, d)) for every d >= 1 of the list: the hardware fast path
    # (MUFU.RCP + one fused Newton step) for all of them; only when some lane of the warp holds a
    # d >= 2^126 (subnormal/zero quotient, +inf) the IEEE divisions run, behind a warp-uniform
    # branch (a predicated / if-converted div.rn costs ~13 extra instructions per gate).
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


# Packed f32x2 operations preserve each scalar lane's .rn result while cutting
# issued FP instructions for the natural K/Q pairs.
_FMA2_PTX = """{
.reg .b64 A, B, C;
mov.b64 A, {{$r0}, {$r1}};
mov.b64 B, {{$r2}, {$r3}};
mov.b64 C, {{$r4}, {$r5}};
fma.rn.f32x2 C, A, B, C;
mov.b64 {{$w0}, {$w1}}, C;
}"""


def _fma2(a0, a1, b0, b1, c0, c1):
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


# Eight normal-range Markstein quotients packed directly to E2M1.  f32x2 keeps
# each lane's rounding and E2M1 encoding identical to the scalar sequence.
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


def _conv_acc(t, T, sec, cs, xs, cws):
    # Conv4 accumulator of token t, section sec, in the reference tap order
    wv = cws[sec]
    acc = Float32(0.0)
    for tap in range(4):
        pos = t + tap
        if pos < 3:
            xv = Float32(cs[sec * 3 + pos])
        else:
            xv = Float32(xs[sec * T + pos - 3])
        acc = cute.fma(xv, Float32(wv[tap]), acc)
    return acc


def _conv_acc_qk(t, T, cs, xs, cws):
    # Pair q/k taps after the first FMA so both lanes retain the reference's
    # independent scalar rounding sequence.
    aq = Float32(0.0)
    ak = Float32(0.0)
    for tap in range(4):
        pos = t + tap
        if pos < 3:
            xq = Float32(cs[pos])
            xk = Float32(cs[3 + pos])
        else:
            xq = Float32(xs[pos - 3])
            xk = Float32(xs[T + pos - 3])
        if tap == 0:
            aq = cute.fma(xq, Float32(cws[0][0]), aq)
            ak = cute.fma(xk, Float32(cws[1][0]), ak)
        else:
            aq, ak = _fma2(xq, xk, Float32(cws[0][tap]), Float32(cws[1][tap]), aq, ak)
    return aq, ak


def _silu_v(av):
    return _bf(av * _rcpf(Float32(1.0) + _expf(-av)))


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
    WS: cutlass.Constexpr[int],
    PDL: cutlass.Constexpr[int],
    hk_ss: cutlass.Constexpr[int],
    hu_ss: cutlass.Constexpr[int],
    hg_ss: cutlass.Constexpr[int],
    hl_ss: cutlass.Constexpr[int],
):
    # Warp-specialized verify: warps 0..3 ("main") own the q/k Conv, norms, the FP4 operand
    # quantization, the native MMA and the epilogue (thread = channel, then thread = value row);
    # warps 4..7 ("coef") own the history staging and adjusted keys (before the PDL wait), the
    # gates (published first), v Conv, sigmoids, and the BF16 coefficient products.
    # Named barriers: 1 main (128), 4 coef (128), 2 gates coef->main, 6 k/q main->coef,
    # 3 coefficient results coef->main, 5 accumulator TMEM loads done main->coef, 7 TMEM address.
    N2 = 2 * T
    MMA_N: cutlass.Constexpr[int] = 8 if cutlass.const_expr(T <= 4) else 16
    NROWS = HROWS + T
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
    s_uh = cutlass.Array(BFloat16, (HROWS * ASTR,), space=cutlass.AddressSpace.smem, alignment=128)
    s_gh = cutlass.Array(Float32, (HROWS * KD,), space=cutlass.AddressSpace.smem, alignment=128)
    s_A = cutlass.Array(BFloat16, (NROWS * ASTR,), space=cutlass.AddressSpace.smem, alignment=128)
    s_R = cutlass.Array(BFloat16, (N2 * ASTR,), space=cutlass.AddressSpace.smem, alignment=128)
    s_coefT = cutlass.Array(
        BFloat16, (MMA_N * CTSTR,), space=cutlass.AddressSpace.smem, alignment=16
    )
    s_coefn = cutlass.Array(Float32, (T * MMA_N,), space=cutlass.AddressSpace.smem, alignment=16)
    s_sfb = cutlass.Array(Uint32, (2 * 128,), space=cutlass.AddressSpace.smem, alignment=128)
    s_red1 = cutlass.Array(Float32, (RV1 * 16,), space=cutlass.AddressSpace.smem, alignment=16)
    s_red2 = cutlass.Array(Float32, (4 * 16,), space=cutlass.AddressSpace.smem, alignment=16)
    s_den = cutlass.Array(Float32, (16,), space=cutlass.AddressSpace.smem, alignment=16)
    s_rden = cutlass.Array(Float32, (16,), space=cutlass.AddressSpace.smem, alignment=16)
    s_outer = cutlass.Array(Float32, (16,), space=cutlass.AddressSpace.smem, alignment=16)
    s_red3 = cutlass.Array(Float32, (4 * 8,), space=cutlass.AddressSpace.smem, alignment=16)
    s_tmem = cutlass.Array(Int32, (1,), space=cutlass.AddressSpace.smem, alignment=16)
    s_mbar = cutlass.Array(Int64, (1,), space=cutlass.AddressSpace.smem, alignment=8)
    s_qv = cutlass.Array(Float32, (N2 * KD + 4,), space=cutlass.AddressSpace.smem, alignment=128)
    s_hist = cutlass.Array(Float32, (4 * 512 + 4,), space=cutlass.AddressSpace.smem, alignment=128)
    s_kqn = cutlass.Array(BFloat16, (N2 * KD + 8,), space=cutlass.AddressSpace.smem, alignment=128)
    s_gate = cutlass.Array(Float32, (T * KD + 4,), space=cutlass.AddressSpace.smem, alignment=128)
    s_sig = cutlass.Array(Float32, (N2 * KD + 4,), space=cutlass.AddressSpace.smem, alignment=128)
    s_vv = cutlass.Array(Float32, (T * KD + 4,), space=cutlass.AddressSpace.smem, alignment=128)
    s_anc = cutlass.Array(Float32, (KD,), space=cutlass.AddressSpace.smem, alignment=128)
    # Verify CTAs cannot start on an SM until its producer CTAs have drained (shared-memory carveout),
    # and the SMs that drain first are then filled to capacity.  Cap the CTAs per SM at what the grid
    # needs (one-wave grids: 1, grids up to two waves: 2) so that the grid spreads over the SMs.
    if cutlass.const_expr(WS == 2):
        s_pad1 = cutlass.Array(Uint8, (65536,), space=cutlass.AddressSpace.smem, alignment=128)
        s_pad1[tid] = Uint8(0)
    if cutlass.const_expr(WS == 3):
        s_pad1 = cutlass.Array(Uint8, (20480,), space=cutlass.AddressSpace.smem, alignment=128)
        s_pad1[tid] = Uint8(0)

    b64 = Int64(b)
    h64 = Int64(h)
    # the row's flag and slot are fetched together (one dependent round trip less); the slot
    # is only dereferenced on live rows
    is_dummy = _ld_keep_u8(p_dummy + b64)
    slot = _ld_keep_s32(p_slots + b64 * 4)

    if tid < NT:
        out_ptr = cutlass.inttoptr(p_out, GMEM, BFloat16)
        cand_ptr = cutlass.inttoptr(p_cand, GMEM, BFloat16)
        oldlen_ptr = cutlass.inttoptr(p_oldlen, GMEM, Int32)
        hoff = h * KD + tid
        raw_ptr = cutlass.inttoptr(p_raw, GMEM, BFloat16)
        cw_ptr = cutlass.inttoptr(p_cw, GMEM, BFloat16)
        cws = []
        for sec in cutlass.range_constexpr(2):
            cws.append((cw_ptr + Int64(sec * HK + hoff) * 4).load(alignment=8, count=4))
        nw = (cutlass.inttoptr(p_nw, GMEM, BFloat16) + tid).load()

        s_sfb[tid] = Uint32(0)
        s_sfb[tid + NT] = Uint32(0)
        if warp == 0:
            prims.tcgen05_alloc(s_tmem, TMEM_COLS, group=prims.CTAGroup.CTA_1)
            prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_1)
        if tid == 0:
            prims.mbarrier_init(s_mbar, 1)
            prims.fence_mbarrier_init()

        if is_dummy != Int32(0):
            if cutlass.const_expr(PDL != 0):
                prims.griddepcontrol("wait")
            # Dummy row: exact zeros for every destination element, nothing else touched.
            o_row = out_ptr + ((b64 * T * H + h64) * KD + tid)
            c_row = cand_ptr + (b64 * candidate_bs + hoff)
            for t in cutlass.range_constexpr(T):
                o_row.store(BFloat16(0.0))
                for sec in cutlass.range_constexpr(3):
                    (c_row + sec * HK).store(BFloat16(0.0))
                o_row = o_row + HK
                c_row = c_row + C3
            if h == 0:
                if tid == 0:
                    (oldlen_ptr + b64).store(Int32(0))
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            prims.barrier_cta_sync(1, thread_count=NT)
            if warp == 0:
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                prims.tcgen05_dealloc(
                    cutlass.inttoptr(s_tmem[0], 6, Float32),
                    TMEM_COLS,
                    group=prims.CTAGroup.CTA_1,
                )
        else:
            # ---- slot-dependent loads (state of earlier launches only) ----
            slot64 = Int64(slot)
            replay_slot = _ld_keep_s32(p_ridx + b64 * 4)
            pool_row = p_pool + slot64 * pool_ss + h64 * STATE_BYTES
            pool_u8 = cutlass.inttoptr(pool_row, GMEM, Uint8)
            sfa = []
            if cutlass.const_expr(POOL_VEC != 0):
                for k in cutlass.range_constexpr(4):
                    q = tid + NT * k
                    r = q // 4
                    c = q % 4
                    dst = r * 64 + ((c ^ ((r >> 1) & 3)) * 16)
                    prims.cp_async_shared_global(
                        sA.data_ptr() + dst, pool_u8 + (r * 64 + c * 16), 16, "cg"
                    )
                for j in cutlass.range_constexpr(4):
                    m = lane + 32 * j
                    v = cutlass.inttoptr(pool_row + 16384 + m * 8, GMEM, Uint32).load(
                        alignment=8, count=2
                    )
                    sfa.append(v)
                outer_s = cutlass.inttoptr(pool_row + 18432, GMEM, Float32).load()
            else:
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
            prims.cp_async_commit_group()
            conv_ptr = cutlass.inttoptr(p_conv, GMEM, BFloat16)
            cs = []
            for sec in cutlass.range_constexpr(2):
                cbase = slot64 * conv_ss + Int64(sec * HK + hoff) * 3
                for p in cutlass.range_constexpr(3):
                    cs.append((conv_ptr + (cbase + p)).load())
            replay64 = Int64(replay_slot)
            length = _ld_keep_s32(p_hlen + replay64 * hl_ss)
            hrow = h64 * CAP
            hk_ptr = cutlass.inttoptr(p_hk + replay64 * hk_ss, GMEM, BFloat16)
            hu_ptr = cutlass.inttoptr(p_hu + replay64 * hu_ss, GMEM, BFloat16)
            hg_ptr = cutlass.inttoptr(p_hg + replay64 * hg_ss, GMEM, Float32)

            # TMEM address visible to the main warps; checkpoint block scales -> TMEM cols 16..23
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            prims.barrier_cta_sync(1, thread_count=NT)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            tmem_base = s_tmem[0]
            sfa_vals = []
            for kb in cutlass.range_constexpr(2):
                for j in cutlass.range_constexpr(4):
                    sfa_vals.append(Int32(sfa[j][kb]))
            sfa_vec = cutlass.Vector.from_elements(tuple(sfa_vals), Int32)
            prims.tcgen05_st(
                "32x32b", cutlass.inttoptr(tmem_base + ((warp * 32) << 16) + 16, 6, Int32), sfa_vec
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.STORE)

            # ---- current-step projections, behind the PDL wait ----
            if cutlass.const_expr(PDL != 0):
                prims.griddepcontrol("wait")
            xs = []
            x_row0 = raw_ptr + (b64 * raw_bs + hoff)
            for sec in cutlass.range_constexpr(2):
                x_row = x_row0 + sec * HK
                for t in cutlass.range_constexpr(T):
                    xs.append(x_row.load())
                    x_row = x_row + raw_rs
            qv = []
            kv = []
            for t in cutlass.range_constexpr(T):
                aq, ak = _conv_acc_qk(t, T, cs, xs, cws)
                sq_, sk_ = _silu2_ref(aq, ak)
                qv.append(_bf(sq_))
                kv.append(_bf(sk_))

            # ---- reduction 1: sum of squares of q and k per token (bit-exact order) ----
            sqq = []
            sqk = []
            for t in cutlass.range_constexpr(T):
                q2, k2 = _mul2(qv[t], kv[t], qv[t], kv[t])
                sqq.append(q2)
                sqk.append(k2)
            sq = sqq + sqk
            red1 = _tree_reduce(sq, (2, 4, 1), lane)
            for e1 in red1:
                if (lane & e1[3]) == 0:
                    s_red1[_tree_idx(e1, lane) * 16 + warp * 4 + lane // 8] = e1[0]
            prims.barrier_cta_sync(1, thread_count=NT)
            if tid < N2:
                tgv = (s_red1.data_ptr() + tid * 16).load(alignment=16, count=16)
                ug = []
                for m in cutlass.range_constexpr(8):
                    ug.append(tgv[m] + tgv[m + 8])
                vg = []
                for m in cutlass.range_constexpr(4):
                    vg.append(ug[m] + ug[m + 4])
                tot = (vg[0] + vg[2]) + (vg[1] + vg[3])
                dden = cute.sqrt(tot + Float32(1e-6))
                s_den[tid] = dden
                s_rden[tid] = Float32(1.0) / dden
            prims.barrier_cta_sync(1, thread_count=NT)
            qn = []
            kn = []
            for t in cutlass.range_constexpr(T):
                qd, kd = _div2_mk(
                    _mul_qscale_rn(qv[t]),
                    kv[t],
                    -s_den[t],
                    -s_den[T + t],
                    s_rden[t],
                    s_rden[T + t],
                )
                qn.append(_bf(qd))
                kn.append(_bf(kd))
            # hand k/q to the coefficient warps
            for t in cutlass.range_constexpr(T):
                s_kqn[(2 * t) * KD + tid] = BFloat16(kn[t])
                s_kqn[(2 * t + 1) * KD + tid] = BFloat16(qn[t])
            prims.barrier_cta_arrive(6, 2 * NT)

            # k.q dot products of each token (bit-exact butterfly; one [token][warp] partial)
            kqp = []
            for t in cutlass.range_constexpr(T):
                kqp.append(kn[t] * qn[t])
            red2 = _tree_reduce(kqp, (16, 8, 4, 2, 1), lane)
            for e2 in red2:
                if (lane & e2[3]) == 0:
                    s_red2[_tree_idx(e2, lane) * 4 + warp] = e2[0]

            # ---- cumulative G (gates published by the coefficient warps) ----
            prims.barrier_cta_sync(2, thread_count=2 * NT)
            cum = []
            prev = s_anc[tid]
            for t in cutlass.range_constexpr(T):
                cur = _add_rn(prev, s_gate[t * KD + tid])
                cum.append(cur)
                prev = cur
            qoff = (((tid // 4) ^ (tid // 16)) * 4) + tid % 4
            for t in cutlass.range_constexpr(T):
                ecum = _exp_tl(cum[t])
                s_qv[(2 * t) * KD + qoff] = _mul_rn(kn[t], ecum)
                s_qv[(2 * t + 1) * KD + qoff] = _mul_rn(qn[t], ecum)
            prims.barrier_cta_sync(1, thread_count=NT)

            # ---- quantize operand rows to E2M1 + UE4M3 block-16 scales ----
            e4m3 = mlir_ir.TypeAttr.get(cutlass.Float8E4M3FN.mlir_type)
            qrow = tid // 8
            qblk = tid % 8
            qphys = qrow * 64 + (((qblk // 2) ^ ((qrow >> 1) & 3)) * 16) + (qblk % 2) * 8
            # one 16-lane half-warp per token (two operand rows x eight blocks); padding rows read
            # row 0 and contribute 0 to the row maxima
            qrow_c = qrow if qrow < N2 else Int32(0)
            qbase = s_qv.data_ptr() + qrow_c * KD
            qv4 = []
            for i in cutlass.range_constexpr(4):
                qg = (qblk * 4 + i) ^ qblk
                qv4.append((qbase + 4 * qg).load(alignment=16, count=4))
            vals = []
            for i in cutlass.range_constexpr(16):
                vals.append(qv4[i // 4][i % 4])
            bmx = _amax16(vals)
            bmx = bmx if qrow < N2 else Float32(0.0)
            # per-half-warp max as two full-warp reductions (a half-warp member mask serializes the
            # halves through the uniform datapath)
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
                # SFB staging in the UTCCP 32x128b atom image: [k-block][row][4 words], word 0 =
                # the 4 block scales of that k-block (one byte store per block)
                sfb8 = cutlass.inttoptr(s_sfb.data_ptr().toint(), cutlass.AddressSpace.smem, Uint8)
                (sfb8 + (((qblk // 4) * 128 + qrow * 4) * 4 + (qblk % 4))).store(Uint8(sbyte))
                if tid % 16 == 0:
                    s_outer[qrow >> 1] = outer
                # decoded scale: hardware e4m3 -> f16 -> f32 (exact)
                expd = _e4m3_f32(w16) * outer
                if expd >= Float32(7.888609052210118e-31) and expd < Float32(8.507059173023462e37):
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
                        cutlass.Vector.from_elements((Uint32(0), Uint32(0)), Uint32), alignment=8
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
                if qrow < MMA_N:
                    sB[qphys // 4] = Uint32(0)
                    sB[qphys // 4 + 1] = Uint32(0)
            prims.cp_async_wait_group(0)
            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
            prims.barrier_cta_sync(1, thread_count=NT)

            # ---- native NVFP4 MMA: D[128 x N] = S(E2M1,UE4M3) x [k_t;q_t](E2M1,UE4M3)^T ----
            if warp == 0:
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                idesc = prims.Tcgen05MxOmmaInstrDesc.build(
                    a_dtype=cutlass.Float4E2M1FN,
                    b_dtype=cutlass.Float4E2M1FN,
                    scale_format=0,
                    n_dim=MMA_N,
                    m_dim=128,
                    sparsity_version=0,
                )
                desc_a = prims.Tcgen05SmemDesc.build(
                    sA,
                    leading_byte_offset=16,
                    stride_byte_offset=512,
                    base_offset=0,
                    layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
                )
                desc_b = prims.Tcgen05SmemDesc.build(
                    sB,
                    leading_byte_offset=16,
                    stride_byte_offset=512,
                    base_offset=0,
                    layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
                )
                d_ptr = cutlass.inttoptr(tmem_base, 6, Float32)
                desc_sf = prims.Tcgen05SmemDesc.build(
                    s_sfb,
                    leading_byte_offset=16,
                    stride_byte_offset=128,
                    base_offset=0,
                    layout=prims.Tcgen05SmemSwizzle.NONE,
                )
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

            # ---- persistent history / candidate writes (off the MMA critical path) ----
            hk_w = hk_ptr + ((hrow + length) * KD + tid)
            hg_w = hg_ptr + ((hrow + length) * KD + tid)
            for t in cutlass.range_constexpr(T):
                (hk_w + t * KD).store(BFloat16(kn[t]))
                (hg_w + t * KD).store(cum[t])
            cw_row = cand_ptr + (b64 * candidate_bs + hoff)
            for t in cutlass.range_constexpr(T):
                for sec in cutlass.range_constexpr(2):
                    (cw_row + sec * HK).store(xs[sec * T + t])
                cw_row = cw_row + C3
            kqs = []
            for t in cutlass.range_constexpr(T):
                kv4 = (s_red2.data_ptr() + 4 * t).load(alignment=16, count=4)
                kqs.append(kv4[0] + kv4[1] + kv4[2] + kv4[3])

            # coefficient warps publish s_coefn, the history-U rows, v and the sigmoids
            prims.barrier_cta_sync(3, thread_count=2 * NT)
            hsc = s_hist.data_ptr() + warp * 512
            hist = []
            for c in cutlass.range_constexpr(MMA_N // 4):
                hv = (hsc + (lane * 16 + (c ^ ((lane >> 1) & 3)) * 4)).load(alignment=16, count=4)
                for e in cutlass.range_constexpr(4):
                    hist.append(hv[e])

            # ---- epilogue: thread = value row i ----
            while not prims.mbarrier_try_wait_parity(s_mbar, 0):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            acc_ptr = cutlass.inttoptr(tmem_base + ((warp * 32) << 16), 6, Float32)
            accv = prims.tcgen05_ld("32x32b", acc_ptr, num=MMA_N)
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            prims.barrier_cta_arrive(5, 2 * NT)
            osc = []
            for c in cutlass.range_constexpr((T + 3) // 4):
                ov4 = (s_outer.data_ptr() + 4 * c).load(alignment=16, count=4)
                for e in cutlass.range_constexpr(4):
                    if cutlass.const_expr(4 * c + e < T):
                        osc.append(outer_s * ov4[e])
            pks = []
            pqs = []
            for t in cutlass.range_constexpr(T):
                pk, pq = _fma2(
                    accv[2 * t],
                    accv[2 * t + 1],
                    osc[t],
                    osc[t],
                    hist[2 * t],
                    hist[2 * t + 1],
                )
                pks.append(pk)
                pqs.append(pq)
            us = []
            cores = []
            for s in cutlass.range_constexpr(T):
                u = s_sig[s * KD + tid] * (s_vv[s * KD + tid] - pks[s])
                us.append(u)
                cores.append(_bf(pqs[s] + u * kqs[s]))
                if cutlass.const_expr(s + 1 < T):
                    crow = {}
                    for c in cutlass.range_constexpr((2 * s + 2) // 4, (2 * T + 3) // 4):
                        cv4 = (s_coefn.data_ptr() + (s * MMA_N + 4 * c)).load(alignment=16, count=4)
                        for e in cutlass.range_constexpr(4):
                            crow[4 * c + e] = cv4[e]
                    for t in cutlass.range_constexpr(s + 1, T):
                        pks[t], pqs[t] = _fma2(u, u, crow[2 * t], crow[2 * t + 1], pks[t], pqs[t])
            csq = []
            for t in cutlass.range_constexpr(T):
                csq.append(cores[t] * cores[t])
            red3 = _tree_reduce(csq, (16, 8, 4, 2, 1), lane)
            for e3 in red3:
                if (lane & e3[3]) == 0:
                    s_red3[_tree_idx(e3, lane) * 4 + warp] = e3[0]
            prims.barrier_cta_sync(1, thread_count=NT)
            if cutlass.const_expr(PDL != 0):
                prims.griddepcontrol("launch_dependents")
            nwf = Float32(nw)
            hu_w = hu_ptr + ((hrow + length) * KD + tid)
            o_row2 = out_ptr + ((b64 * T * H + h64) * KD + tid)
            for t in cutlass.range_constexpr(T):
                tv4 = (s_red3.data_ptr() + 4 * t).load(alignment=16, count=4)
                tot = tv4[0] + tv4[1] + tv4[2] + tv4[3]
                rs = cute.rsqrt(tot * Float32(1.0 / 128.0) + eps)
                o = cores[t] * rs * nwf * s_sig[(T + t) * KD + tid]
                (o_row2 + t * HK).store(BFloat16(o))
                (hu_w + t * KD).store(BFloat16(us[t]))
            if h == 0:
                if tid == 0:
                    (oldlen_ptr + b64).store(length)
    else:
        if is_dummy == Int32(0):
            ctid = tid - NT
            cwarp = warp - 4
            choff = h * KD + ctid
            dtb = (cutlass.inttoptr(p_dtb, GMEM, Float32) + choff).load()
            a_log = (cutlass.inttoptr(p_a_log, GMEM, Float32) + h).load()
            cwv = (cutlass.inttoptr(p_cw, GMEM, BFloat16) + Int64(2 * HK + choff) * 4).load(
                alignment=8, count=4
            )
            slot64 = Int64(slot)
            vcbase = slot64 * conv_ss + Int64(2 * HK + choff) * 3
            csv = []
            for p in cutlass.range_constexpr(3):
                csv.append((cutlass.inttoptr(p_conv, GMEM, BFloat16) + (vcbase + p)).load())
            # ---- history staging (earlier launches' state): K/G rows 0..15, U zero past L ----
            replay64 = Int64(_ld_keep_s32(p_ridx + b64 * 4))
            hrow = h64 * CAP
            hk_base = cutlass.inttoptr(p_hk + replay64 * hk_ss, GMEM, BFloat16) + hrow * KD
            hu_base = cutlass.inttoptr(p_hu + replay64 * hu_ss, GMEM, BFloat16) + hrow * KD
            hg_base = cutlass.inttoptr(p_hg + replay64 * hg_ss, GMEM, Float32) + hrow * KD
            gdx, gdy, _ = cute.arch.grid_dim()
            if gdx * gdy > LIVE_ROWS_MIN_GRID:
                # rows 0..7 do not wait for the length (their copies overlap its round trip); rows
                # past the length among them are cleared after landing (U) or masked later (K, G)
                q0 = ctid
                prims.cp_async_shared_global(
                    s_A.data_ptr() + (q0 * 8 + (q0 // 16) * 8), hk_base + q0 * 8, 16, "cg"
                )
                prims.cp_async_shared_global(
                    s_uh.data_ptr() + (q0 * 8 + (q0 // 16) * 8), hu_base + q0 * 8, 16, "cg"
                )
                for k in cutlass.range_constexpr(2):
                    q1 = ctid + NT * k
                    prims.cp_async_shared_global(
                        s_gh.data_ptr() + q1 * 4, hg_base + q1 * 4, 16, "cg"
                    )
            length = _ld_keep_s32(p_hlen + replay64 * hl_ss)
            if gdx * gdy <= LIVE_ROWS_MIN_GRID:
                # small launch (latency bound): stage all rows without waiting for the length
                for k in cutlass.range_constexpr(2):
                    q = ctid + NT * k
                    r = q // 16
                    prims.cp_async_shared_global(
                        s_A.data_ptr() + (q * 8 + r * 8), hk_base + q * 8, 16, "cg"
                    )
                for k in cutlass.range_constexpr(4):
                    q = ctid + NT * k
                    prims.cp_async_shared_global(s_gh.data_ptr() + q * 4, hg_base + q * 4, 16, "cg")
                for k in cutlass.range_constexpr(2):
                    q = ctid + NT * k
                    r = q // 16
                    prims.cp_async_shared_global(
                        s_uh.data_ptr() + (q * 8 + r * 8), hu_base + q * 8, 16, "cg"
                    )
                prims.cp_async_commit_group()
                prims.cp_async_wait_group(0)
                # U must be exact zeros outside the accepted prefix (zero coefficients do not cancel NaN)
                zv8 = cutlass.Vector.from_elements(tuple([BFloat16(0.0)] * 8), BFloat16)
                for k in cutlass.range_constexpr(2):
                    q = ctid + NT * k
                    r = q // 16
                    if r >= length:
                        (s_uh.data_ptr() + (q * 8 + r * 8)).store(zv8, alignment=16)
            else:
                # larger launch (bandwidth bound): rows 8..15 only when accepted (zero-filled past L)
                for k in cutlass.range_constexpr(1, 2):
                    q = ctid + NT * k
                    r = q // 16
                    nbytes = Int32(16) if r < length else Int32(0)
                    prims.cp_async_shared_global(
                        s_A.data_ptr() + (q * 8 + r * 8), hk_base + q * 8, 16, "cg", cp_size=nbytes
                    )
                    prims.cp_async_shared_global(
                        s_uh.data_ptr() + (q * 8 + r * 8), hu_base + q * 8, 16, "cg", cp_size=nbytes
                    )
                for k in cutlass.range_constexpr(2, 4):
                    q = ctid + NT * k
                    r = q // 32
                    nbytes = Int32(16) if r < length else Int32(0)
                    prims.cp_async_shared_global(
                        s_gh.data_ptr() + q * 4, hg_base + q * 4, 16, "cg", cp_size=nbytes
                    )
                prims.cp_async_commit_group()
                prims.cp_async_wait_group(0)
                if ctid // 16 >= length:
                    zu8 = cutlass.Vector.from_elements(tuple([BFloat16(0.0)] * 8), BFloat16)
                    (s_uh.data_ptr() + (ctid * 8 + (ctid // 16) * 8)).store(zu8, alignment=16)
            prims.barrier_cta_sync(4, thread_count=NT)
            aidx = (length - 1) if length > 0 else Int32(0)
            anchor_ld = s_gh[aidx * KD + ctid]
            anchor = anchor_ld if length > 0 else Float32(0.0)
            s_anc[ctid] = anchor
            # history adjusted keys (in place), before the wait: rows of the last group at or past
            # the accepted length are exact zeros; later rows are masked in the coefficients
            a_hist = s_A.data_ptr() + ctid
            g_hist = s_gh.data_ptr() + ctid
            for gi in cutlass.range((length + 7) // 8, unroll=1):
                khs = []
                ghs = []
                for j in cutlass.range_constexpr(8):
                    khs.append(Float32((a_hist + j * ASTR).load()))
                    ghs.append((g_hist + j * KD).load())
                for j in cutlass.range_constexpr(8):
                    adj = khs[j] * _expf(anchor - ghs[j])
                    adj = adj if gi * 8 + j < length else Float32(0.0)
                    (a_hist + j * ASTR).store(BFloat16(adj))
                a_hist = a_hist + 8 * ASTR
                g_hist = g_hist + 8 * KD

            # ---- current-step projections, behind the PDL wait ----
            g_row = cutlass.inttoptr(p_g, GMEM, BFloat16) + (b64 * g_bs + choff)
            z_row = cutlass.inttoptr(p_z, GMEM, BFloat16) + (b64 * z_bs + choff)
            b_row = cutlass.inttoptr(p_beta, GMEM, BFloat16) + (b64 * beta_bs + h64)
            xv_row = cutlass.inttoptr(p_raw, GMEM, BFloat16) + (b64 * raw_bs + 2 * HK + choff)
            if cutlass.const_expr(PDL != 0):
                prims.griddepcontrol("wait")
            gs = []
            zs = []
            betas = []
            xs = [None] * (2 * T)
            for t in cutlass.range_constexpr(T):
                gs.append(g_row.load())
                g_row = g_row + g_rs
            for t in cutlass.range_constexpr(T):
                xs.append(xv_row.load())
                xv_row = xv_row + raw_rs
            for t in cutlass.range_constexpr(T):
                betas.append(b_row.load())
                zs.append(z_row.load())
                z_row = z_row + z_rs
                b_row = b_row + beta_rs
            # ---- gates (Triton-pinned elementary ops), published first ----
            rate = _exp_tl(a_log)
            gates = []
            for t in cutlass.range_constexpr(T):
                shifted = Float32(gs[t]) + dtb
                if cutlass.const_expr(BOUNDED != 0):
                    dgate = _add1_rn(_expf(_mul_rn(shifted, -rate)))
                    rgate = _rcp_ge1(dgate)
                    if dgate >= Float32(8.507059173023462e37):
                        rgate = Float32(1.0) / dgate
                    gates.append(_mul_rn(lower, rgate))
                else:
                    spv = cute.log(Float32(1.0) + _exp_tl(shifted))
                    sp = spv if shifted <= Float32(20.0) else shifted
                    gates.append(-rate * sp)
            for t in cutlass.range_constexpr(T):
                s_gate[t * KD + ctid] = gates[t]
            prims.barrier_cta_arrive(2, 2 * NT)
            # exact candidate copy of the v section (q/k sections: main warps)
            cw_row = cutlass.inttoptr(p_cand, GMEM, BFloat16) + (
                b64 * candidate_bs + 2 * HK + choff
            )
            for t in cutlass.range_constexpr(T):
                cw_row.store(xs[2 * T + t])
                cw_row = cw_row + C3
            # v-section Conv4 + SiLU and the beta / output-gate sigmoids (epilogue operands)
            for t in cutlass.range_constexpr(T):
                s_vv[t * KD + ctid] = _silu_v(
                    _conv_acc(t, T, 2, [None] * 6 + csv, xs, [None, None, cwv])
                )
            for t in cutlass.range_constexpr(T):
                s_sig[t * KD + ctid] = _rcpf(Float32(1.0) + _expf(-Float32(betas[t])))
                s_sig[(T + t) * KD + ctid] = _rcpf(Float32(1.0) + _expf(-Float32(zs[t])))
            cum = []
            prev = anchor
            for t in cutlass.range_constexpr(T):
                cur = _add_rn(prev, gates[t])
                cum.append(cur)
                prev = cur
            ers = []
            eas = []
            for t in cutlass.range_constexpr(T):
                ers.append(_expf(cum[t] - anchor))
                eas.append(_expf(anchor - cum[t]))
            # ---- k/q from the main warps: relative pairs and adjusted new keys ----
            prims.barrier_cta_sync(6, thread_count=2 * NT)
            for t in cutlass.range_constexpr(T):
                knt = Float32(s_kqn[(2 * t) * KD + ctid])
                qnt = Float32(s_kqn[(2 * t + 1) * KD + ctid])
                s_R[(2 * t) * ASTR + ctid] = BFloat16(knt * ers[t])
                s_R[(2 * t + 1) * ASTR + ctid] = BFloat16(qnt * ers[t])
                s_A[(HROWS + t) * ASTR + ctid] = BFloat16(knt * eas[t])
            prims.barrier_cta_sync(4, thread_count=NT)
            # ---- coefficient products on warp tensor cores (bf16 x bf16 -> fp32) ----
            mt = cwarp // 2
            nt = cwarp % 2
            if cutlass.const_expr(MMA_N == 8):
                coefficient_active = nt == 0
            else:
                coefficient_active = True
            if coefficient_active:
                gid = lane // 4
                tig = lane % 4
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
                    if cutlass.const_expr(ks == 0):
                        d0, d1, d2, d3 = cute.arch.inline_ptx(
                            "{\n.reg .f32 zz;\nmov.f32 zz, 0f00000000;\n"
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, {{$r4}, {$r5}}, "
                            "{zz, zz, zz, zz};\n}",
                            write_only_types=[Float32, Float32, Float32, Float32],
                            read_only_args=[fa0, fa1, fa2, fa3, fb0, fb1],
                        )
                    elif cutlass.const_expr(ks == 1):
                        e0, e1, e2, e3 = cute.arch.inline_ptx(
                            "{\n.reg .f32 zz;\nmov.f32 zz, 0f00000000;\n"
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, {{$r4}, {$r5}}, "
                            "{zz, zz, zz, zz};\n}",
                            write_only_types=[Float32, Float32, Float32, Float32],
                            read_only_args=[fa0, fa1, fa2, fa3, fb0, fb1],
                        )
                    elif cutlass.const_expr(ks % 2 == 0):
                        d0, d1, d2, d3 = cute.arch.inline_ptx(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, {{$r4}, {$r5}}, "
                            "{{$r6}, {$r7}, {$r8}, {$r9}};",
                            write_only_types=[Float32, Float32, Float32, Float32],
                            read_only_args=[fa0, fa1, fa2, fa3, fb0, fb1, d0, d1, d2, d3],
                        )
                    else:
                        e0, e1, e2, e3 = cute.arch.inline_ptx(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, {{$r4}, {$r5}}, "
                            "{{$r6}, {$r7}, {$r8}, {$r9}};",
                            write_only_types=[Float32, Float32, Float32, Float32],
                            read_only_args=[fa0, fa1, fa2, fa3, fb0, fb1, e0, e1, e2, e3],
                        )
                d0 = d0 + e0
                d1 = d1 + e1
                d2 = d2 + e2
                d3 = d3 + e3
                cn = 8 * nt + 2 * tig
                if cutlass.const_expr(HROWS == 16):
                    if mt == 0:
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
            prims.barrier_cta_sync(4, thread_count=NT)
            # ---- history-U contraction on warp tensor cores ----
            gq = lane // 4
            tq = lane % 4
            lj = lane // 8
            lrow = (lj // 2) * 8 + lane % 8
            hsc = s_hist.data_ptr() + cwarp * 512
            for mq in cutlass.range_constexpr(2):
                afr = prims.ldmatrix(
                    s_uh.data_ptr() + (lrow * ASTR + cwarp * 32 + 16 * mq + (lj % 2) * 8),
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
                    (hsc + (hr0 * 16 + (c4 ^ ((hr0 >> 1) & 3)) * 4 + 2 * (tq % 2))).store(
                        cutlass.Vector.from_elements((h0, h1), Float32), alignment=8
                    )
                    (hsc + (hr1 * 16 + (c4 ^ ((hr1 >> 1) & 3)) * 4 + 2 * (tq % 2))).store(
                        cutlass.Vector.from_elements((h2, h3), Float32), alignment=8
                    )
            prims.barrier_cta_arrive(3, 2 * NT)
            # the main warps signal barrier 5 once their accumulator TMEM loads completed; an
            # otherwise idle coefficient warp releases TMEM off the epilogue critical path
            prims.barrier_cta_sync(5, thread_count=2 * NT)
            if cwarp == 0:
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                prims.tcgen05_dealloc(
                    cutlass.inttoptr(s_tmem[0], 6, Float32),
                    TMEM_COLS,
                    group=prims.CTAGroup.CTA_1,
                )


@cute.kernel
def _replay_kernel_shadow(
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
    WS: cutlass.Constexpr[int],
    PDL: cutlass.Constexpr[int],
    hk_ss: cutlass.Constexpr[int],
    hu_ss: cutlass.Constexpr[int],
    hg_ss: cutlass.Constexpr[int],
    hl_ss: cutlass.Constexpr[int],
):
    # Warp-specialized verify: warps 0..3 ("main") own the q/k Conv, norms, the FP4 operand
    # quantization, the native MMA and the epilogue (thread = channel, then thread = value row);
    # warps 4..7 ("coef") own the history staging and adjusted keys (before the PDL wait), the
    # gates (published first), v Conv, sigmoids, and the BF16 coefficient products.
    # Named barriers: 1 main (128), 4 coef (128), 2 gates coef->main, 6 k/q main->coef,
    # 3 coefficient results coef->main, 5 accumulator TMEM loads done main->coef, 7 TMEM address.
    N2 = 2 * T
    MMA_N: cutlass.Constexpr[int] = 8 if cutlass.const_expr(T <= 4) else 16
    NROWS = HROWS + T
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
    s_uh = cutlass.Array(BFloat16, (HROWS * ASTR,), space=cutlass.AddressSpace.smem, alignment=128)
    s_gh = cutlass.Array(Float32, (HROWS * KD,), space=cutlass.AddressSpace.smem, alignment=128)
    s_A = cutlass.Array(BFloat16, (NROWS * ASTR,), space=cutlass.AddressSpace.smem, alignment=128)
    s_R = cutlass.Array(BFloat16, (N2 * ASTR,), space=cutlass.AddressSpace.smem, alignment=128)
    s_coefT = cutlass.Array(
        BFloat16, (MMA_N * CTSTR,), space=cutlass.AddressSpace.smem, alignment=16
    )
    s_coefn = cutlass.Array(Float32, (T * MMA_N,), space=cutlass.AddressSpace.smem, alignment=16)
    s_sfb = cutlass.Array(Uint32, (2 * 128,), space=cutlass.AddressSpace.smem, alignment=128)
    s_red1 = cutlass.Array(Float32, (RV1 * 16,), space=cutlass.AddressSpace.smem, alignment=16)
    s_red2 = cutlass.Array(Float32, (4 * 16,), space=cutlass.AddressSpace.smem, alignment=16)
    s_den = cutlass.Array(Float32, (16,), space=cutlass.AddressSpace.smem, alignment=16)
    s_rden = cutlass.Array(Float32, (16,), space=cutlass.AddressSpace.smem, alignment=16)
    s_outer = cutlass.Array(Float32, (16,), space=cutlass.AddressSpace.smem, alignment=16)
    s_red3 = cutlass.Array(Float32, (4 * 8,), space=cutlass.AddressSpace.smem, alignment=16)
    s_tmem = cutlass.Array(Int32, (1,), space=cutlass.AddressSpace.smem, alignment=16)
    s_mbar = cutlass.Array(Int64, (1,), space=cutlass.AddressSpace.smem, alignment=8)
    s_qv = cutlass.Array(Float32, (N2 * KD + 4,), space=cutlass.AddressSpace.smem, alignment=128)
    s_hist = cutlass.Array(Float32, (4 * 512 + 4,), space=cutlass.AddressSpace.smem, alignment=128)
    s_kqn = cutlass.Array(BFloat16, (N2 * KD + 8,), space=cutlass.AddressSpace.smem, alignment=128)
    s_gate = cutlass.Array(Float32, (T * KD + 4,), space=cutlass.AddressSpace.smem, alignment=128)
    s_sig = cutlass.Array(Float32, (N2 * KD + 4,), space=cutlass.AddressSpace.smem, alignment=128)
    s_vv = cutlass.Array(Float32, (T * KD + 4,), space=cutlass.AddressSpace.smem, alignment=128)
    s_anc = cutlass.Array(Float32, (KD,), space=cutlass.AddressSpace.smem, alignment=128)
    # Verify CTAs cannot start on an SM until its producer CTAs have drained (shared-memory carveout),
    # and the SMs that drain first are then filled to capacity.  Cap the CTAs per SM at what the grid
    # needs (one-wave grids: 1, grids up to two waves: 2) so that the grid spreads over the SMs.
    if cutlass.const_expr(WS == 2):
        s_pad1 = cutlass.Array(Uint8, (65536,), space=cutlass.AddressSpace.smem, alignment=128)
        s_pad1[tid] = Uint8(0)
    if cutlass.const_expr(WS == 3):
        s_pad1 = cutlass.Array(Uint8, (20480,), space=cutlass.AddressSpace.smem, alignment=128)
        s_pad1[tid] = Uint8(0)

    b64 = Int64(b)
    h64 = Int64(h)
    # the row's flag and slot are fetched together (one dependent round trip less); the slot
    # is only dereferenced on live rows
    is_dummy = _ld_keep_u8(p_dummy + b64)
    slot = _ld_keep_s32(p_slots + b64 * 4)

    if tid < NT:
        out_ptr = cutlass.inttoptr(p_out, GMEM, BFloat16)
        cand_ptr = cutlass.inttoptr(p_cand, GMEM, BFloat16)
        oldlen_ptr = cutlass.inttoptr(p_oldlen, GMEM, Int32)
        hoff = h * KD + tid
        raw_ptr = cutlass.inttoptr(p_raw, GMEM, BFloat16)
        cw_ptr = cutlass.inttoptr(p_cw, GMEM, BFloat16)
        cws = []
        for sec in cutlass.range_constexpr(2):
            cws.append((cw_ptr + Int64(sec * HK + hoff) * 4).load(alignment=8, count=4))
        nw = (cutlass.inttoptr(p_nw, GMEM, BFloat16) + tid).load()

        s_sfb[tid] = Uint32(0)
        s_sfb[tid + NT] = Uint32(0)
        if warp == 0:
            prims.tcgen05_alloc(s_tmem, TMEM_COLS, group=prims.CTAGroup.CTA_1)
            prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_1)
        if tid == 0:
            prims.mbarrier_init(s_mbar, 1)
            prims.fence_mbarrier_init()

        if is_dummy != Int32(0):
            if cutlass.const_expr(PDL != 0):
                prims.griddepcontrol("wait")
            # Dummy row: exact zeros for every destination element, nothing else touched.
            o_row = out_ptr + ((b64 * T * H + h64) * KD + tid)
            c_row = cand_ptr + (b64 * candidate_bs + hoff)
            for t in cutlass.range_constexpr(T):
                o_row.store(BFloat16(0.0))
                for sec in cutlass.range_constexpr(3):
                    (c_row + sec * HK).store(BFloat16(0.0))
                o_row = o_row + HK
                c_row = c_row + C3
            if h == 0:
                if tid == 0:
                    (oldlen_ptr + b64).store(Int32(0))
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            prims.barrier_cta_sync(1, thread_count=NT)
            if warp == 0:
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                prims.tcgen05_dealloc(
                    cutlass.inttoptr(s_tmem[0], 6, Float32),
                    TMEM_COLS,
                    group=prims.CTAGroup.CTA_1,
                )
        else:
            # ---- slot-dependent loads (state of earlier launches only) ----
            slot64 = Int64(slot)
            replay_slot = _ld_keep_s32(p_ridx + b64 * 4)
            pool_row = p_pool + slot64 * pool_ss + h64 * STATE_BYTES
            pool_u8 = cutlass.inttoptr(pool_row, GMEM, Uint8)
            sfa = []
            if cutlass.const_expr(POOL_VEC != 0):
                for k in cutlass.range_constexpr(4):
                    q = tid + NT * k
                    r = q // 4
                    c = q % 4
                    dst = r * 64 + ((c ^ ((r >> 1) & 3)) * 16)
                    prims.cp_async_shared_global(
                        sA.data_ptr() + dst, pool_u8 + (r * 64 + c * 16), 16, "cg"
                    )
                for j in cutlass.range_constexpr(4):
                    m = lane + 32 * j
                    v = cutlass.inttoptr(pool_row + 16384 + m * 8, GMEM, Uint32).load(
                        alignment=8, count=2
                    )
                    sfa.append(v)
                outer_s = cutlass.inttoptr(pool_row + 18432, GMEM, Float32).load()
            else:
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
            prims.cp_async_commit_group()
            conv_ptr = cutlass.inttoptr(p_conv, GMEM, BFloat16)
            cs = []
            for sec in cutlass.range_constexpr(2):
                cbase = slot64 * conv_ss + Int64(sec * HK + hoff) * 3
                for p in cutlass.range_constexpr(3):
                    cs.append((conv_ptr + (cbase + p)).load())
            replay64 = Int64(replay_slot)
            length = _ld_keep_s32(p_hlen + replay64 * hl_ss)
            hrow = h64 * CAP
            hk_ptr = cutlass.inttoptr(p_hk + replay64 * hk_ss, GMEM, BFloat16)
            hu_ptr = cutlass.inttoptr(p_hu + replay64 * hu_ss, GMEM, BFloat16)
            hg_ptr = cutlass.inttoptr(p_hg + replay64 * hg_ss, GMEM, Float32)

            # ---- current-step projections, behind the PDL wait ----
            if cutlass.const_expr(PDL != 0):
                prims.griddepcontrol("wait")
            xs = []
            x_row0 = raw_ptr + (b64 * raw_bs + hoff)
            for sec in cutlass.range_constexpr(2):
                x_row = x_row0 + sec * HK
                for t in cutlass.range_constexpr(T):
                    xs.append(x_row.load())
                    x_row = x_row + raw_rs
            # TMEM address + checkpoint block scales -> TMEM cols 16..23, in the shadow of the
            # projection loads
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            prims.barrier_cta_sync(1, thread_count=NT)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            tmem_base = s_tmem[0]
            sfa_vals = []
            for kb in cutlass.range_constexpr(2):
                for j in cutlass.range_constexpr(4):
                    sfa_vals.append(Int32(sfa[j][kb]))
            sfa_vec = cutlass.Vector.from_elements(tuple(sfa_vals), Int32)
            prims.tcgen05_st(
                "32x32b", cutlass.inttoptr(tmem_base + ((warp * 32) << 16) + 16, 6, Int32), sfa_vec
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
            qv = []
            kv = []
            for t in cutlass.range_constexpr(T):
                aq, ak = _conv_acc_qk(t, T, cs, xs, cws)
                sq_, sk_ = _silu2_ref(aq, ak)
                qv.append(_bf(sq_))
                kv.append(_bf(sk_))

            # ---- reduction 1: sum of squares of q and k per token (bit-exact order) ----
            sqq = []
            sqk = []
            for t in cutlass.range_constexpr(T):
                q2, k2 = _mul2(qv[t], kv[t], qv[t], kv[t])
                sqq.append(q2)
                sqk.append(k2)
            sq = sqq + sqk
            red1 = _tree_reduce(sq, (2, 4, 1), lane)
            for e1 in red1:
                if (lane & e1[3]) == 0:
                    s_red1[_tree_idx(e1, lane) * 16 + warp * 4 + lane // 8] = e1[0]
            prims.barrier_cta_sync(1, thread_count=NT)
            if tid < N2:
                tgv = (s_red1.data_ptr() + tid * 16).load(alignment=16, count=16)
                ug = []
                for m in cutlass.range_constexpr(8):
                    ug.append(tgv[m] + tgv[m + 8])
                vg = []
                for m in cutlass.range_constexpr(4):
                    vg.append(ug[m] + ug[m + 4])
                tot = (vg[0] + vg[2]) + (vg[1] + vg[3])
                dden = cute.sqrt(tot + Float32(1e-6))
                s_den[tid] = dden
                s_rden[tid] = Float32(1.0) / dden
            prims.barrier_cta_sync(1, thread_count=NT)
            qn = []
            kn = []
            for t in cutlass.range_constexpr(T):
                qd, kd = _div2_mk(
                    _mul_qscale_rn(qv[t]),
                    kv[t],
                    -s_den[t],
                    -s_den[T + t],
                    s_rden[t],
                    s_rden[T + t],
                )
                qn.append(_bf(qd))
                kn.append(_bf(kd))
            # hand k/q to the coefficient warps
            for t in cutlass.range_constexpr(T):
                s_kqn[(2 * t) * KD + tid] = BFloat16(kn[t])
                s_kqn[(2 * t + 1) * KD + tid] = BFloat16(qn[t])
            prims.barrier_cta_arrive(6, 2 * NT)

            # k.q dot products of each token (bit-exact butterfly; one [token][warp] partial)
            kqp = []
            for t in cutlass.range_constexpr(T):
                kqp.append(kn[t] * qn[t])
            red2 = _tree_reduce(kqp, (16, 8, 4, 2, 1), lane)
            for e2 in red2:
                if (lane & e2[3]) == 0:
                    s_red2[_tree_idx(e2, lane) * 4 + warp] = e2[0]

            # ---- cumulative G (gates published by the coefficient warps) ----
            prims.barrier_cta_sync(2, thread_count=2 * NT)
            cum = []
            prev = s_anc[tid]
            for t in cutlass.range_constexpr(T):
                cur = _add_rn(prev, s_gate[t * KD + tid])
                cum.append(cur)
                prev = cur
            qoff = (((tid // 4) ^ (tid // 16)) * 4) + tid % 4
            for t in cutlass.range_constexpr(T):
                ecum = _exp_tl(cum[t])
                s_qv[(2 * t) * KD + qoff] = _mul_rn(kn[t], ecum)
                s_qv[(2 * t + 1) * KD + qoff] = _mul_rn(qn[t], ecum)
            prims.barrier_cta_sync(1, thread_count=NT)

            # ---- quantize operand rows to E2M1 + UE4M3 block-16 scales ----
            e4m3 = mlir_ir.TypeAttr.get(cutlass.Float8E4M3FN.mlir_type)
            qrow = tid // 8
            qblk = tid % 8
            qphys = qrow * 64 + (((qblk // 2) ^ ((qrow >> 1) & 3)) * 16) + (qblk % 2) * 8
            # one 16-lane half-warp per token (two operand rows x eight blocks); padding rows read
            # row 0 and contribute 0 to the row maxima
            qrow_c = qrow if qrow < N2 else Int32(0)
            qbase = s_qv.data_ptr() + qrow_c * KD
            qv4 = []
            for i in cutlass.range_constexpr(4):
                qg = (qblk * 4 + i) ^ qblk
                qv4.append((qbase + 4 * qg).load(alignment=16, count=4))
            vals = []
            for i in cutlass.range_constexpr(16):
                vals.append(qv4[i // 4][i % 4])
            bmx = _amax16(vals)
            bmx = bmx if qrow < N2 else Float32(0.0)
            # per-half-warp max as two full-warp reductions (a half-warp member mask serializes the
            # halves through the uniform datapath)
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
                # SFB staging in the UTCCP 32x128b atom image: [k-block][row][4 words], word 0 =
                # the 4 block scales of that k-block (one byte store per block)
                sfb8 = cutlass.inttoptr(s_sfb.data_ptr().toint(), cutlass.AddressSpace.smem, Uint8)
                (sfb8 + (((qblk // 4) * 128 + qrow * 4) * 4 + (qblk % 4))).store(Uint8(sbyte))
                if tid % 16 == 0:
                    s_outer[qrow >> 1] = outer
                # decoded scale: hardware e4m3 -> f16 -> f32 (exact)
                expd = _e4m3_f32(w16) * outer
                if expd >= Float32(7.888609052210118e-31) and expd < Float32(8.507059173023462e37):
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
                        cutlass.Vector.from_elements((Uint32(0), Uint32(0)), Uint32), alignment=8
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
                if qrow < MMA_N:
                    sB[qphys // 4] = Uint32(0)
                    sB[qphys // 4 + 1] = Uint32(0)
            prims.cp_async_wait_group(0)
            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
            prims.barrier_cta_sync(1, thread_count=NT)

            # ---- native NVFP4 MMA: D[128 x N] = S(E2M1,UE4M3) x [k_t;q_t](E2M1,UE4M3)^T ----
            if warp == 0:
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                idesc = prims.Tcgen05MxOmmaInstrDesc.build(
                    a_dtype=cutlass.Float4E2M1FN,
                    b_dtype=cutlass.Float4E2M1FN,
                    scale_format=0,
                    n_dim=MMA_N,
                    m_dim=128,
                    sparsity_version=0,
                )
                desc_a = prims.Tcgen05SmemDesc.build(
                    sA,
                    leading_byte_offset=16,
                    stride_byte_offset=512,
                    base_offset=0,
                    layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
                )
                desc_b = prims.Tcgen05SmemDesc.build(
                    sB,
                    leading_byte_offset=16,
                    stride_byte_offset=512,
                    base_offset=0,
                    layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
                )
                d_ptr = cutlass.inttoptr(tmem_base, 6, Float32)
                desc_sf = prims.Tcgen05SmemDesc.build(
                    s_sfb,
                    leading_byte_offset=16,
                    stride_byte_offset=128,
                    base_offset=0,
                    layout=prims.Tcgen05SmemSwizzle.NONE,
                )
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

            # ---- persistent history / candidate writes (off the MMA critical path) ----
            hk_w = hk_ptr + ((hrow + length) * KD + tid)
            hg_w = hg_ptr + ((hrow + length) * KD + tid)
            for t in cutlass.range_constexpr(T):
                (hk_w + t * KD).store(BFloat16(kn[t]))
                (hg_w + t * KD).store(cum[t])
            cw_row = cand_ptr + (b64 * candidate_bs + hoff)
            for t in cutlass.range_constexpr(T):
                for sec in cutlass.range_constexpr(2):
                    (cw_row + sec * HK).store(xs[sec * T + t])
                cw_row = cw_row + C3
            kqs = []
            for t in cutlass.range_constexpr(T):
                kv4 = (s_red2.data_ptr() + 4 * t).load(alignment=16, count=4)
                kqs.append(kv4[0] + kv4[1] + kv4[2] + kv4[3])

            # coefficient warps publish s_coefn, the history-U rows, v and the sigmoids
            prims.barrier_cta_sync(3, thread_count=2 * NT)
            hsc = s_hist.data_ptr() + warp * 512
            hist = []
            for c in cutlass.range_constexpr(MMA_N // 4):
                hv = (hsc + (lane * 16 + (c ^ ((lane >> 1) & 3)) * 4)).load(alignment=16, count=4)
                for e in cutlass.range_constexpr(4):
                    hist.append(hv[e])

            # ---- epilogue: thread = value row i ----
            while not prims.mbarrier_try_wait_parity(s_mbar, 0):
                pass
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            acc_ptr = cutlass.inttoptr(tmem_base + ((warp * 32) << 16), 6, Float32)
            accv = prims.tcgen05_ld("32x32b", acc_ptr, num=MMA_N)
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            prims.barrier_cta_arrive(5, 2 * NT)
            osc = []
            for c in cutlass.range_constexpr((T + 3) // 4):
                ov4 = (s_outer.data_ptr() + 4 * c).load(alignment=16, count=4)
                for e in cutlass.range_constexpr(4):
                    if cutlass.const_expr(4 * c + e < T):
                        osc.append(outer_s * ov4[e])
            pks = []
            pqs = []
            for t in cutlass.range_constexpr(T):
                pk, pq = _fma2(
                    accv[2 * t],
                    accv[2 * t + 1],
                    osc[t],
                    osc[t],
                    hist[2 * t],
                    hist[2 * t + 1],
                )
                pks.append(pk)
                pqs.append(pq)
            us = []
            cores = []
            for s in cutlass.range_constexpr(T):
                u = s_sig[s * KD + tid] * (s_vv[s * KD + tid] - pks[s])
                us.append(u)
                cores.append(_bf(pqs[s] + u * kqs[s]))
                if cutlass.const_expr(s + 1 < T):
                    crow = {}
                    for c in cutlass.range_constexpr((2 * s + 2) // 4, (2 * T + 3) // 4):
                        cv4 = (s_coefn.data_ptr() + (s * MMA_N + 4 * c)).load(alignment=16, count=4)
                        for e in cutlass.range_constexpr(4):
                            crow[4 * c + e] = cv4[e]
                    for t in cutlass.range_constexpr(s + 1, T):
                        pks[t], pqs[t] = _fma2(u, u, crow[2 * t], crow[2 * t + 1], pks[t], pqs[t])
            csq = []
            for t in cutlass.range_constexpr(T):
                csq.append(cores[t] * cores[t])
            red3 = _tree_reduce(csq, (16, 8, 4, 2, 1), lane)
            for e3 in red3:
                if (lane & e3[3]) == 0:
                    s_red3[_tree_idx(e3, lane) * 4 + warp] = e3[0]
            prims.barrier_cta_sync(1, thread_count=NT)
            if cutlass.const_expr(PDL != 0):
                prims.griddepcontrol("launch_dependents")
            nwf = Float32(nw)
            hu_w = hu_ptr + ((hrow + length) * KD + tid)
            o_row2 = out_ptr + ((b64 * T * H + h64) * KD + tid)
            for t in cutlass.range_constexpr(T):
                tv4 = (s_red3.data_ptr() + 4 * t).load(alignment=16, count=4)
                tot = tv4[0] + tv4[1] + tv4[2] + tv4[3]
                rs = cute.rsqrt(tot * Float32(1.0 / 128.0) + eps)
                o = cores[t] * rs * nwf * s_sig[(T + t) * KD + tid]
                (o_row2 + t * HK).store(BFloat16(o))
                (hu_w + t * KD).store(BFloat16(us[t]))
            if h == 0:
                if tid == 0:
                    (oldlen_ptr + b64).store(length)
    else:
        if is_dummy == Int32(0):
            ctid = tid - NT
            cwarp = warp - 4
            choff = h * KD + ctid
            dtb = (cutlass.inttoptr(p_dtb, GMEM, Float32) + choff).load()
            a_log = (cutlass.inttoptr(p_a_log, GMEM, Float32) + h).load()
            cwv = (cutlass.inttoptr(p_cw, GMEM, BFloat16) + Int64(2 * HK + choff) * 4).load(
                alignment=8, count=4
            )
            slot64 = Int64(slot)
            vcbase = slot64 * conv_ss + Int64(2 * HK + choff) * 3
            csv = []
            for p in cutlass.range_constexpr(3):
                csv.append((cutlass.inttoptr(p_conv, GMEM, BFloat16) + (vcbase + p)).load())
            # ---- history staging (earlier launches' state): K/G rows 0..15, U zero past L ----
            replay64 = Int64(_ld_keep_s32(p_ridx + b64 * 4))
            hrow = h64 * CAP
            hk_base = cutlass.inttoptr(p_hk + replay64 * hk_ss, GMEM, BFloat16) + hrow * KD
            hu_base = cutlass.inttoptr(p_hu + replay64 * hu_ss, GMEM, BFloat16) + hrow * KD
            hg_base = cutlass.inttoptr(p_hg + replay64 * hg_ss, GMEM, Float32) + hrow * KD
            gdx, gdy, _ = cute.arch.grid_dim()
            length = _ld_keep_s32(p_hlen + replay64 * hl_ss)
            if gdx * gdy <= LIVE_ROWS_MIN_GRID:
                # small launch (latency bound): stage all rows without waiting for the length
                for k in cutlass.range_constexpr(2):
                    q = ctid + NT * k
                    r = q // 16
                    prims.cp_async_shared_global(
                        s_A.data_ptr() + (q * 8 + r * 8), hk_base + q * 8, 16, "cg"
                    )
                for k in cutlass.range_constexpr(4):
                    q = ctid + NT * k
                    prims.cp_async_shared_global(s_gh.data_ptr() + q * 4, hg_base + q * 4, 16, "cg")
                for k in cutlass.range_constexpr(2):
                    q = ctid + NT * k
                    r = q // 16
                    prims.cp_async_shared_global(
                        s_uh.data_ptr() + (q * 8 + r * 8), hu_base + q * 8, 16, "cg"
                    )
                prims.cp_async_commit_group()
            else:
                # larger launch (bandwidth bound): only the accepted rows (zero-filled past L)
                for k in cutlass.range_constexpr(2):
                    q = ctid + NT * k
                    r = q // 16
                    nbytes = Int32(16) if r < length else Int32(0)
                    prims.cp_async_shared_global(
                        s_A.data_ptr() + (q * 8 + r * 8), hk_base + q * 8, 16, "cg", cp_size=nbytes
                    )
                    prims.cp_async_shared_global(
                        s_uh.data_ptr() + (q * 8 + r * 8), hu_base + q * 8, 16, "cg", cp_size=nbytes
                    )
                for k in cutlass.range_constexpr(4):
                    q = ctid + NT * k
                    r = q // 32
                    nbytes = Int32(16) if r < length else Int32(0)
                    prims.cp_async_shared_global(
                        s_gh.data_ptr() + q * 4, hg_base + q * 4, 16, "cg", cp_size=nbytes
                    )
                prims.cp_async_commit_group()
            # ---- current-step projections, behind the PDL wait ----
            g_row = cutlass.inttoptr(p_g, GMEM, BFloat16) + (b64 * g_bs + choff)
            z_row = cutlass.inttoptr(p_z, GMEM, BFloat16) + (b64 * z_bs + choff)
            b_row = cutlass.inttoptr(p_beta, GMEM, BFloat16) + (b64 * beta_bs + h64)
            xv_row = cutlass.inttoptr(p_raw, GMEM, BFloat16) + (b64 * raw_bs + 2 * HK + choff)
            if cutlass.const_expr(PDL != 0):
                prims.griddepcontrol("wait")
            gs = []
            zs = []
            betas = []
            xs = [None] * (2 * T)
            for t in cutlass.range_constexpr(T):
                gs.append(g_row.load())
                g_row = g_row + g_rs
            for t in cutlass.range_constexpr(T):
                xs.append(xv_row.load())
                xv_row = xv_row + raw_rs
            for t in cutlass.range_constexpr(T):
                betas.append(b_row.load())
                zs.append(z_row.load())
                z_row = z_row + z_rs
                b_row = b_row + beta_rs
            # history staged before the wait (normally already landed)
            prims.cp_async_wait_group(0)
            if gdx * gdy <= LIVE_ROWS_MIN_GRID:
                # U must be exact zeros outside the accepted prefix (zero coefficients do not cancel NaN)
                zv8 = cutlass.Vector.from_elements(tuple([BFloat16(0.0)] * 8), BFloat16)
                for k in cutlass.range_constexpr(2):
                    q = ctid + NT * k
                    r = q // 16
                    if r >= length:
                        (s_uh.data_ptr() + (q * 8 + r * 8)).store(zv8, alignment=16)
            prims.barrier_cta_sync(4, thread_count=NT)
            aidx = (length - 1) if length > 0 else Int32(0)
            anchor_ld = s_gh[aidx * KD + ctid]
            anchor = anchor_ld if length > 0 else Float32(0.0)
            s_anc[ctid] = anchor
            # history adjusted keys (in place), in the shadow of the projection loads: rows of the last group at or past
            # the accepted length are exact zeros; later rows are masked in the coefficients
            a_hist = s_A.data_ptr() + ctid
            g_hist = s_gh.data_ptr() + ctid
            for gi in cutlass.range((length + 7) // 8, unroll=1):
                khs = []
                ghs = []
                for j in cutlass.range_constexpr(8):
                    khs.append(Float32((a_hist + j * ASTR).load()))
                    ghs.append((g_hist + j * KD).load())
                for j in cutlass.range_constexpr(8):
                    adj = khs[j] * _expf(anchor - ghs[j])
                    adj = adj if gi * 8 + j < length else Float32(0.0)
                    (a_hist + j * ASTR).store(BFloat16(adj))
                a_hist = a_hist + 8 * ASTR
                g_hist = g_hist + 8 * KD

            # ---- gates (Triton-pinned elementary ops), published first ----
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
            for t in cutlass.range_constexpr(T):
                s_gate[t * KD + ctid] = gates[t]
            prims.barrier_cta_arrive(2, 2 * NT)
            # exact candidate copy of the v section (q/k sections: main warps)
            cw_row = cutlass.inttoptr(p_cand, GMEM, BFloat16) + (
                b64 * candidate_bs + 2 * HK + choff
            )
            for t in cutlass.range_constexpr(T):
                cw_row.store(xs[2 * T + t])
                cw_row = cw_row + C3
            # v-section Conv4 + SiLU and the beta / output-gate sigmoids (epilogue operands)
            for t in cutlass.range_constexpr(T):
                s_vv[t * KD + ctid] = _silu_v(
                    _conv_acc(t, T, 2, [None] * 6 + csv, xs, [None, None, cwv])
                )
            for t in cutlass.range_constexpr(T):
                s_sig[t * KD + ctid] = _rcpf(Float32(1.0) + _expf(-Float32(betas[t])))
                s_sig[(T + t) * KD + ctid] = _rcpf(Float32(1.0) + _expf(-Float32(zs[t])))
            cum = []
            prev = anchor
            for t in cutlass.range_constexpr(T):
                cur = _add_rn(prev, gates[t])
                cum.append(cur)
                prev = cur
            ers = []
            eas = []
            for t in cutlass.range_constexpr(T):
                ers.append(_expf(cum[t] - anchor))
                eas.append(_expf(anchor - cum[t]))
            # ---- k/q from the main warps: relative pairs and adjusted new keys ----
            prims.barrier_cta_sync(6, thread_count=2 * NT)
            for t in cutlass.range_constexpr(T):
                knt = Float32(s_kqn[(2 * t) * KD + ctid])
                qnt = Float32(s_kqn[(2 * t + 1) * KD + ctid])
                s_R[(2 * t) * ASTR + ctid] = BFloat16(knt * ers[t])
                s_R[(2 * t + 1) * ASTR + ctid] = BFloat16(qnt * ers[t])
                s_A[(HROWS + t) * ASTR + ctid] = BFloat16(knt * eas[t])
            prims.barrier_cta_sync(4, thread_count=NT)
            # ---- coefficient products on warp tensor cores (bf16 x bf16 -> fp32) ----
            mt = cwarp // 2
            nt = cwarp % 2
            if cutlass.const_expr(MMA_N == 8):
                coefficient_active = nt == 0
            else:
                coefficient_active = True
            if coefficient_active:
                gid = lane // 4
                tig = lane % 4
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
                    if cutlass.const_expr(ks == 0):
                        d0, d1, d2, d3 = cute.arch.inline_ptx(
                            "{\n.reg .f32 zz;\nmov.f32 zz, 0f00000000;\n"
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, {{$r4}, {$r5}}, "
                            "{zz, zz, zz, zz};\n}",
                            write_only_types=[Float32, Float32, Float32, Float32],
                            read_only_args=[fa0, fa1, fa2, fa3, fb0, fb1],
                        )
                    elif cutlass.const_expr(ks == 1):
                        e0, e1, e2, e3 = cute.arch.inline_ptx(
                            "{\n.reg .f32 zz;\nmov.f32 zz, 0f00000000;\n"
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, {{$r4}, {$r5}}, "
                            "{zz, zz, zz, zz};\n}",
                            write_only_types=[Float32, Float32, Float32, Float32],
                            read_only_args=[fa0, fa1, fa2, fa3, fb0, fb1],
                        )
                    elif cutlass.const_expr(ks % 2 == 0):
                        d0, d1, d2, d3 = cute.arch.inline_ptx(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, {{$r4}, {$r5}}, "
                            "{{$r6}, {$r7}, {$r8}, {$r9}};",
                            write_only_types=[Float32, Float32, Float32, Float32],
                            read_only_args=[fa0, fa1, fa2, fa3, fb0, fb1, d0, d1, d2, d3],
                        )
                    else:
                        e0, e1, e2, e3 = cute.arch.inline_ptx(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, {{$r4}, {$r5}}, "
                            "{{$r6}, {$r7}, {$r8}, {$r9}};",
                            write_only_types=[Float32, Float32, Float32, Float32],
                            read_only_args=[fa0, fa1, fa2, fa3, fb0, fb1, e0, e1, e2, e3],
                        )
                d0 = d0 + e0
                d1 = d1 + e1
                d2 = d2 + e2
                d3 = d3 + e3
                cn = 8 * nt + 2 * tig
                if cutlass.const_expr(HROWS == 16):
                    if mt == 0:
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
            prims.barrier_cta_sync(4, thread_count=NT)
            # ---- history-U contraction on warp tensor cores ----
            gq = lane // 4
            tq = lane % 4
            lj = lane // 8
            lrow = (lj // 2) * 8 + lane % 8
            hsc = s_hist.data_ptr() + cwarp * 512
            for mq in cutlass.range_constexpr(2):
                afr = prims.ldmatrix(
                    s_uh.data_ptr() + (lrow * ASTR + cwarp * 32 + 16 * mq + (lj % 2) * 8),
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
                    (hsc + (hr0 * 16 + (c4 ^ ((hr0 >> 1) & 3)) * 4 + 2 * (tq % 2))).store(
                        cutlass.Vector.from_elements((h0, h1), Float32), alignment=8
                    )
                    (hsc + (hr1 * 16 + (c4 ^ ((hr1 >> 1) & 3)) * 4 + 2 * (tq % 2))).store(
                        cutlass.Vector.from_elements((h2, h3), Float32), alignment=8
                    )
            prims.barrier_cta_arrive(3, 2 * NT)
            # the main warps signal barrier 5 once their accumulator TMEM loads completed; an
            # otherwise idle coefficient warp releases TMEM off the epilogue critical path
            prims.barrier_cta_sync(5, thread_count=2 * NT)
            if cwarp == 0:
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                prims.tcgen05_dealloc(
                    cutlass.inttoptr(s_tmem[0], 6, Float32),
                    TMEM_COLS,
                    group=prims.CTAGroup.CTA_1,
                )


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
    WS: cutlass.Constexpr[int],
    PDL: cutlass.Constexpr[int],
    hk_ss: cutlass.Constexpr[int],
    hu_ss: cutlass.Constexpr[int],
    hg_ss: cutlass.Constexpr[int],
    hl_ss: cutlass.Constexpr[int],
):
    # CTAs that start late (T > 5 with >= 6 heads, whose SMs must first drain of producer CTAs, and
    # grids needing three CTAs per SM) move prologue work into the post-wait load shadow; CTAs that
    # start early run the plain prologue.
    kern = (
        _replay_kernel_shadow
        if cutlass.const_expr((T > 5 and H >= 6) or WS == 1)
        else _replay_kernel
    )
    kern(
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
        WS,
        PDL,
        hk_ss,
        hu_ss,
        hg_ss,
        hl_ss,
    ).launch(
        grid=(H, B, 1),
        block=(NT * 2 if cutlass.const_expr(WS != 0) else NT, 1, 1),
        stream=stream,
        use_pdl=PDL != 0,
        min_blocks_per_mp=(
            1 if cutlass.const_expr(WS == 2) else (2 if cutlass.const_expr(WS == 3) else WS_BLOCKS)
        )
        if cutlass.const_expr(WS != 0)
        else 5,
        preferred_smem_carveout=100,
    )


_compiled = {}
_sm_count = {}


def _num_sms():
    dev = torch.cuda.current_device()
    n = _sm_count.get(dev)
    if n is None:
        n = torch.cuda.get_device_properties(dev).multi_processor_count
        _sm_count[dev] = n
    return n


def _get_compiled(
    T,
    bounded,
    pool_vec,
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
    ws,
    pdl,
    history_strides,
):
    # Specialize only stable model topology and layout metadata.  Request shape B,
    # accepted lengths, slots, masks and every tensor value remain runtime inputs.
    key = (
        torch.cuda.current_device(),
        T,
        bounded,
        pool_vec,
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
        ws,
        pdl,
        history_strides,
    )
    fn = _compiled.get(key)
    if fn is None:
        z64 = Int64(0)
        # Runtime ABI: 20 pointers, B, candidate stride, lower, eps and stream. Every argument
        # following stream is a compile-time topology/layout constant.
        args = [z64] * 20 + [Int32(1), Int64(0), Float32(0.0), Float32(0.0)]
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
            ws,
            pdl,
            *history_strides,
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
    n_items = batch * heads
    sms = _num_sms()
    if 2 * n_items > PACKED_MIN_2X * sms:
        _packed.fused_replay(
            x,
            conv_state,
            conv_weight,
            gate,
            beta,
            z,
            norm_weight,
            pool,
            slots,
            dummy,
            replay_indices,
            hk,
            hu,
            hg,
            lengths,
            a_log,
            dt_bias,
            lower_bound,
            norm_eps,
            out,
            old_lengths,
            candidate_x=candidate_x,
            tokens_per_step=tokens,
        )
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
    # Select one of two resident-warp budgets. Exact request batch size is runtime.
    # 2: one wave (<= 1 CTA per SM, large register budget); 3: <= 2 CTAs per SM; 1: up to WS_BLOCKS per SM
    ws = 2 if n_items <= sms and tokens > 4 else (3 if n_items <= 2 * sms else 1)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    fn = _get_compiled(
        tokens,
        int(lower_bound is not None),
        pool_vec,
        heads,
        capacity,
        *strides,
        ws,
        int(get_env_enable_pdl()),
        # Replay-slot strides in bytes: fixed for a cache allocation.
        (
            hk.stride(0) * hk.element_size(),
            hu.stride(0) * hu.element_size(),
            hg.stride(0) * hg.element_size(),
            lengths.stride(0) * lengths.element_size(),
        ),
    )
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
        candidate_x.stride(0),
        float(lower_bound) if lower_bound is not None else 0.0,
        float(norm_eps),
        stream,
    )
