# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inline-PTX primitives of the persistent expert kernel.

Every helper is one PTX statement (or a short fixed sequence) whose rounding
or memory semantics the kernel relies on by name: the epilogue arithmetic has
to match the TRTLLM-gen expert kernels instruction for instruction, and the
cross-CTA hand-off has to use exactly the scopes its ordering proof states.
"""

from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Float32, Int32, Int64, T, Uint32, dsl_user_op


def _asm(res, operands, text, constraints, *, pure=False, loc=None, ip=None):
    """One inline-PTX statement. ``pure`` marks register-only arithmetic the
    compiler may schedule freely (memory, barrier and TMEM statements keep
    their program order)."""
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=not pure,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# Epilogue arithmetic (TRTLLM-gen's rounding, instruction for instruction)
# ---------------------------------------------------------------------------
@dsl_user_op
def cvt_e4m3_rn_sat(x: Float32, *, loc=None, ip=None) -> Uint32:
    """FP32 -> E4M3 (RN, satfinite), returned in the low byte."""
    return Uint32(
        _asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b16 h;\n.reg .f32 z;\n"
            "mov.f32 z, 0f00000000;\n"
            "cvt.rn.satfinite.e4m3x2.f32 h, z, $1;\n"
            "cvt.u32.u16 $0, h;\n}",
            "=r,f",
            pure=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def e4m3_to_f32(byte: Uint32, *, loc=None, ip=None) -> Float32:
    """E4M3 in the low byte -> FP32 (exact)."""
    return Float32(
        _asm(
            T.f32(),
            [Uint32(byte).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b16 lo;\n.reg .b32 hx2;\n.reg .b16 h0, h1;\n"
            "cvt.u16.u32 lo, $1;\n"
            "cvt.rn.f16x2.e4m3x2 hx2, lo;\n"
            "mov.b32 {h0, h1}, hx2;\n"
            "cvt.f32.f16 $0, h0;\n}",
            "=f,r",
            pure=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_e2m1x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """Two FP32 -> one byte of E2M1 (RN, satfinite); ``lo`` in bits 0..3."""
    return Uint32(
        _asm(
            T.i32(),
            [Float32(hi).ir_value(loc=loc, ip=ip), Float32(lo).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b8 b;\n.reg .b32 t;\n"
            "cvt.rn.satfinite.e2m1x2.f32 b, $1, $2;\n"
            "mov.b32 t, {b, b, b, b};\n"
            "and.b32 $0, t, 255;\n}",
            "=r,f,f",
            pure=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def rcp_approx_ftz(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            pure=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ex2_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            pure=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def mul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """IEEE FP32 multiply, round to nearest (never contracted into an FMA)."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            pure=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def mul_ftz(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """FP32 multiply, round to nearest, subnormal inputs and result flushed (``FMUL.FTZ``)."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "mul.rn.ftz.f32 $0, $1, $2;",
            "=f,f,f",
            pure=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def add_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """IEEE FP32 add, round to nearest (never contracted into an FMA)."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            pure=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fma_rn_zero(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """IEEE ``a * b + (+0)`` in one rounding (``FFMA a, b, RZ``): the product,
    except that an exact ``-0`` product becomes ``+0``."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "fma.rn.f32 $0, $1, $2, 0f00000000;",
            "=f,f,f",
            pure=True,
            loc=loc,
            ip=ip,
        )
    )


def _f32x2(text: str, operands, loc, ip):
    """One packed two-lane FP32 statement: ``(lo, hi)`` results, IEEE per lane."""
    res = _asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Float32(x).ir_value(loc=loc, ip=ip) for x in operands],
        text,
        "=f,=f," + ",".join("f" * len(operands)),
        pure=True,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def mul_rn_x2(a0: Float32, a1: Float32, b0: Float32, b1: Float32, *, loc=None, ip=None):
    """``(a0 * b0, a1 * b1)`` as one ``mul.rn.f32x2`` (``FMUL2``; IEEE, no flush)."""
    return _f32x2(
        "{\n.reg .b64 a, b, d;\nmov.b64 a, {$2, $3};\nmov.b64 b, {$4, $5};\n"
        "mul.rn.f32x2 d, a, b;\nmov.b64 {$0, $1}, d;\n}",
        (a0, a1, b0, b1),
        loc,
        ip,
    )


@dsl_user_op
def add_rn_x2(a0: Float32, a1: Float32, b0: Float32, b1: Float32, *, loc=None, ip=None):
    """``(a0 + b0, a1 + b1)`` as one ``add.rn.f32x2`` (``FADD2``; IEEE, no flush)."""
    return _f32x2(
        "{\n.reg .b64 a, b, d;\nmov.b64 a, {$2, $3};\nmov.b64 b, {$4, $5};\n"
        "add.rn.f32x2 d, a, b;\nmov.b64 {$0, $1}, d;\n}",
        (a0, a1, b0, b1),
        loc,
        ip,
    )


@dsl_user_op
def fma_rn_zero_x2(a0: Float32, a1: Float32, b0: Float32, b1: Float32, *, loc=None, ip=None):
    """``(a0 * b0 + 0, a1 * b1 + 0)`` as one ``fma.rn.f32x2`` with a ``+0`` addend (``FFMA2``)."""
    return _f32x2(
        "{\n.reg .b64 a, b, z, d;\nmov.b64 a, {$2, $3};\nmov.b64 b, {$4, $5};\n"
        "mov.b64 z, 0;\nfma.rn.f32x2 d, a, b, z;\nmov.b64 {$0, $1}, d;\n}",
        (a0, a1, b0, b1),
        loc,
        ip,
    )


@dsl_user_op
def bf16_bits(x: Float32, *, loc=None, ip=None) -> Uint32:
    """FP32 -> BF16 (RN, satfinite) bits in the low half."""
    return Uint32(
        _asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b16 h;\ncvt.rn.satfinite.bf16.f32 h, $1;\ncvt.u32.u16 $0, h;\n}",
            "=r,f",
            pure=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def tmem_ld_32x32b_x8(taddr: Int32, *, loc=None, ip=None):
    """tcgen05.ld 32x32b.x8: this thread's TMEM lane, 8 consecutive columns."""
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * 8),
        [Int32(taddr).ir_value(loc=loc, ip=ip)],
        "tcgen05.ld.sync.aligned.32x32b.x8.b32 "
        "{$0, $1, $2, $3, $4, $5, $6, $7}, [$8];\n"
        "tcgen05.wait::ld.sync.aligned;",
        "=f,=f,=f,=f,=f,=f,=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(8)]


@dsl_user_op
def tmem_ld_32x32b_xn(taddr: Int32, n: int, *, loc=None, ip=None):
    """tcgen05.ld 32x32b.x``n`` (``n`` 16 or 32): this thread's TMEM lane, ``n`` columns."""
    assert n in (16, 32), n
    outs = ", ".join(f"${i}" for i in range(n))
    res = _asm(
        llvm.StructType.get_literal([T.f32()] * n),
        [Int32(taddr).ir_value(loc=loc, ip=ip)],
        f"tcgen05.ld.sync.aligned.32x32b.x{n}.b32 {{{outs}}}, [${n}];\n"
        "tcgen05.wait::ld.sync.aligned;",
        ",".join(["=f"] * n) + ",r",
        loc=loc,
        ip=ip,
    )
    return [Float32(llvm.extractvalue(T.f32(), res, [i], loc=loc, ip=ip)) for i in range(n)]


# ---------------------------------------------------------------------------
# Global memory
# ---------------------------------------------------------------------------
@dsl_user_op
def ld_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    """Plain (coherent) 4-byte global load: for tensors an earlier kernel wrote."""
    return Uint32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.u32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_v4_u32(addr: Int64, *, loc=None, ip=None):
    """Plain (coherent) 16-byte global load."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return [Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def ld_nc_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
    """Read-only-path load: only for parameters no kernel writes."""
    return Float32(
        _asm(
            T.f32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.f32 $0, [$1];",
            "=f,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_u32(addr: Int64, value: Uint32, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
        "st.global.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_u16(addr: Int64, value: Uint32, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b16 h;\ncvt.u16.u32 h, $1;\nst.global.u16 [$0], h;\n}",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_u64(addr: Int64, value: Int64, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int64(value).ir_value(loc=loc, ip=ip)],
        "st.global.u64 [$0], $1;",
        "l,l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_relaxed_gpu_u64_pair(addr: Int64, lo: Uint32, hi: Uint32, *, loc=None, ip=None):
    """One single-copy-atomic 64-bit relaxed GPU-scope store of ``(lo, hi)``."""
    _asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(lo).ir_value(loc=loc, ip=ip),
            Uint32(hi).ir_value(loc=loc, ip=ip),
        ],
        "{\n.reg .b64 t;\nmov.b64 t, {$1, $2};\nst.relaxed.gpu.global.u64 [$0], t;\n}",
        "l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_relaxed_gpu_u64_pair(addr: Int64, *, loc=None, ip=None):
    """One single-copy-atomic 64-bit relaxed GPU-scope load, as ``(lo, hi)``."""
    res = _asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b64 t;\nld.relaxed.gpu.global.u64 t, [$2];\nmov.b64 {$0, $1}, t;\n}",
        "=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def st_v4_u32(addr: Int64, a: Uint32, b: Uint32, c: Uint32, d: Uint32, *, loc=None, ip=None):
    _asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(a).ir_value(loc=loc, ip=ip),
            Uint32(b).ir_value(loc=loc, ip=ip),
            Uint32(c).ir_value(loc=loc, ip=ip),
            Uint32(d).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_relaxed_gpu_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.relaxed.gpu.global.u32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_relaxed_gpu_u32(addr: Int64, value: Uint32, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
        "st.relaxed.gpu.global.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_release_gpu_u32(addr: Int64, value: Uint32, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
        "st.release.gpu.global.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_acquire_gpu_u32(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.acquire.gpu.global.u32 $0, [$1];",
            "=r,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fence_acq_rel_gpu(*, loc=None, ip=None):
    _asm(None, [], "fence.acq_rel.gpu;", "", loc=loc, ip=ip)


@dsl_user_op
def fence_proxy_async_global(*, loc=None, ip=None):
    """Order this thread's generic-proxy global accesses with its async-proxy
    (bulk copy) accesses."""
    _asm(None, [], "fence.proxy.async.global;", "", loc=loc, ip=ip)


@dsl_user_op
def bulk_g2s(smem_dst: Int32, gmem_src: Int64, nbytes: Int32, mbar: Int32, *, loc=None, ip=None):
    """cp.async.bulk global -> shared, completing on ``mbar`` (tx bytes)."""
    _asm(
        None,
        [
            Int32(smem_dst).ir_value(loc=loc, ip=ip),
            Int64(gmem_src).ir_value(loc=loc, ip=ip),
            Int32(nbytes).ir_value(loc=loc, ip=ip),
            Int32(mbar).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def globaltimer_u64(*, loc=None, ip=None) -> Int64:
    return Int64(_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", loc=loc, ip=ip))


@dsl_user_op
def nanosleep(ns: int, *, loc=None, ip=None):
    """``nanosleep.u32`` of a compile-time duration (a poll's back-off)."""
    _asm(None, [], f"nanosleep.u32 {int(ns)};", "", loc=loc, ip=ip)


@dsl_user_op
def red_add_relaxed_gpu_u32(addr: Int64, value: Uint32, *, loc=None, ip=None):
    """``red.relaxed.gpu.global.add.u32``: a counter increment nobody waits on."""
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
        "red.relaxed.gpu.global.add.u32 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# Shared memory and barriers
# ---------------------------------------------------------------------------
@dsl_user_op
def mbar_arrive_expect_tx(mbar: Int32, nbytes: Int32, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(mbar).ir_value(loc=loc, ip=ip), Int32(nbytes).ir_value(loc=loc, ip=ip)],
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def sts_u8(smem_addr: Int32, value: Uint32, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(smem_addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b16 h;\ncvt.u16.u32 h, $1;\nst.shared.u8 [$0], h;\n}",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def sts_u32(smem_addr: Int32, value: Uint32, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(smem_addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
        "st.shared.u32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def sts_v4_u32(smem_addr: Int32, a: Uint32, b: Uint32, c: Uint32, d: Uint32, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)]
        + [Uint32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c, d)],
        "st.shared.v4.u32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def lds_u32(smem_addr: Int32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.u32 $0, [$1];",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def lds_v4_u32(smem_addr: Int32, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return [Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def atom_or_shared_u32(smem_addr: Int32, value: Uint32, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(smem_addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
        "red.shared.or.b32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atom_min_shared_s32(smem_addr: Int32, value: Int32, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(smem_addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
        "red.shared.min.s32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ballot_u32(pred: Int32, *, loc=None, ip=None) -> Uint32:
    """``vote.sync.ballot`` over the full warp of ``pred != 0`` (all 32 lanes must execute it)."""
    return Uint32(
        _asm(
            T.i32(),
            [Int32(pred).ir_value(loc=loc, ip=ip)],
            "{\n.reg .pred p;\nsetp.ne.s32 p, $1, 0;\nvote.sync.ballot.b32 $0, p, 0xffffffff;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def shfl_bfly_u32(x: Uint32, lane_mask: int, *, loc=None, ip=None) -> Uint32:
    """``shfl.sync.bfly`` of a 32-bit word over the full warp (``lane_mask`` a constant)."""
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            f"shfl.sync.bfly.b32 $0, $1, {int(lane_mask)}, 31, 0xffffffff;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def redux_or_u32(x: Uint32, *, loc=None, ip=None) -> Uint32:
    """``redux.sync.or`` of a 32-bit word over the full warp."""
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            "redux.sync.or.b32 $0, $1, 0xffffffff;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fns_u32(mask: Uint32, n: Int32, *, loc=None, ip=None) -> Int32:
    """Position of the ``n``-th (1-based) set bit of ``mask`` counting from bit 0."""
    return Int32(
        _asm(
            T.i32(),
            [Uint32(mask).ir_value(loc=loc, ip=ip), Int32(n).ir_value(loc=loc, ip=ip)],
            "fns.b32 $0, $1, 0, $2;",
            "=r,r,r",
            pure=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def popc_u32(x: Uint32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            "popc.b32 $0, $1;",
            "=r,r",
            pure=True,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fence_proxy_async_shared(*, loc=None, ip=None):
    """Order this thread's generic-proxy shared stores before later async-proxy
    (UMMA, tcgen05.cp) reads of the same bytes."""
    _asm(None, [], "fence.proxy.async.shared::cta;", "", loc=loc, ip=ip)


@dsl_user_op
def bar_sync(bar_id: int, num_threads: int, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(bar_id).ir_value(loc=loc, ip=ip), Int32(num_threads).ir_value(loc=loc, ip=ip)],
        "bar.sync $0, $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )
