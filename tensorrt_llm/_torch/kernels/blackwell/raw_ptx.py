# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inline-PTX helpers on raw addresses for CuTe DSL kernels.

Kernels that address memory through ``Int32`` shared-memory offsets and ``Int64`` global
addresses, and issue ``mma.sync`` / mbarrier / cluster instructions directly, share these
helpers. Every helper except :func:`asm`, :func:`asm_pure` and the operand converters is a
``dsl_user_op``; values are ``Int32`` / ``Int64`` / ``Float32``. ``asm`` marks the instruction
as having side effects (kept in program order); ``asm_pure`` lets the compiler schedule it.
The PDL instructions are in :mod:`.utils` (``griddepcontrol_wait``,
``griddepcontrol_launch_dependents``).
"""

from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

# ---------------------------------------------------------------------------
# Inline assembly and operands
# ---------------------------------------------------------------------------


def asm(res, operands, text, constraints, *, loc=None, ip=None):
    """``llvm.inline_asm`` with side effects (AT&T dialect)."""
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def asm_pure(res, operands, text, constraints, *, loc=None, ip=None):
    """``llvm.inline_asm`` without side effects (arithmetic the compiler may schedule freely)."""
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def iv(value, *, loc=None, ip=None):
    """``value`` as an ``i32`` operand."""
    return Int32(value).ir_value(loc=loc, ip=ip)


def lv(value, *, loc=None, ip=None):
    """``value`` as an ``i64`` operand."""
    return Int64(value).ir_value(loc=loc, ip=ip)


def fv(value, *, loc=None, ip=None):
    """``value`` as an ``f32`` operand."""
    return Float32(value).ir_value(loc=loc, ip=ip)


def unpack(res, ty, cls, n, *, loc=None, ip=None):
    """The ``n`` fields of a struct result as ``cls`` values."""
    return tuple(cls(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(n))


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


@dsl_user_op
def add_f32(a, b, *, loc=None, ip=None):
    """``a + b`` (``add.rn.f32``, not contracted into an FMA)."""
    return Float32(
        asm(T.f32(), [fv(a, loc=loc, ip=ip), fv(b, loc=loc, ip=ip)], "add.rn.f32 $0, $1, $2;",
            "=f,f,f", loc=loc, ip=ip)
    )  # fmt: skip


@dsl_user_op
def mul_f32(a, b, *, loc=None, ip=None):
    """``a * b`` (``mul.rn.f32``, not contracted into an FMA)."""
    return Float32(
        asm(T.f32(), [fv(a, loc=loc, ip=ip), fv(b, loc=loc, ip=ip)], "mul.rn.f32 $0, $1, $2;",
            "=f,f,f", loc=loc, ip=ip)
    )  # fmt: skip


@dsl_user_op
def rcp_approx_ftz(x, *, loc=None, ip=None):
    return Float32(
        asm_pure(T.f32(), [fv(x, loc=loc, ip=ip)], "rcp.approx.ftz.f32 $0, $1;", "=f,f", loc=loc,
                 ip=ip)
    )  # fmt: skip


@dsl_user_op
def min_s32(a, b, *, loc=None, ip=None):
    return Int32(
        asm(T.i32(), [iv(a, loc=loc, ip=ip), iv(b, loc=loc, ip=ip)], "min.s32 $0, $1, $2;",
            "=r,r,r", loc=loc, ip=ip)
    )  # fmt: skip


@dsl_user_op
def pack_bf16x2(lo, hi, *, loc=None, ip=None):
    """Round two FP32 values to BF16 (nearest even), ``lo`` in the low half."""
    return Int32(
        asm(T.i32(), [fv(lo, loc=loc, ip=ip), fv(hi, loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f", loc=loc, ip=ip)
    )  # fmt: skip


@dsl_user_op
def bf16x2_to_f32x2(word, *, loc=None, ip=None):
    """The two BF16 halves of ``word`` (low half first) widened to FP32 (exact)."""
    res = asm_pure(
        llvm.StructType.get_literal([T.f32()] * 2),
        [iv(word, loc=loc, ip=ip)],
        "{\n.reg .b32 lo, hi;\nshl.b32 lo, $2, 16;\nand.b32 hi, $2, 0xffff0000;\n"
        "mov.b32 $0, lo;\nmov.b32 $1, hi;\n}",
        "=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return unpack(res, T.f32(), Float32, 2, loc=loc, ip=ip)


@dsl_user_op
def acc_bf16x2(word, acc_lo, acc_hi, *, loc=None, ip=None):
    """``(acc_lo + lo, acc_hi + hi)`` in FP32 (round to nearest) for the BF16 pair ``word``."""
    res = asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [iv(word, loc=loc, ip=ip), fv(acc_lo, loc=loc, ip=ip), fv(acc_hi, loc=loc, ip=ip)],
        "{\n.reg .b32 l, h;\n.reg .f32 fl, fh;\n"
        "shl.b32 l, $2, 16;\nand.b32 h, $2, 0xffff0000;\n"
        "mov.b32 fl, l;\nmov.b32 fh, h;\n"
        "add.rn.f32 $0, $3, fl;\nadd.rn.f32 $1, $4, fh;\n}",
        "=f,=f,r,f,f",
        loc=loc,
        ip=ip,
    )
    return unpack(res, T.f32(), Float32, 2, loc=loc, ip=ip)


@dsl_user_op
def sanitize_bf16x2(word, *, loc=None, ip=None):
    """Replace each ``-0`` half (0x8000) of a BF16 pair by ``+0``."""
    return Int32(
        asm(
            T.i32(),
            [iv(word, loc=loc, ip=ip)],
            "{\n.reg .pred p, q;\n.reg .b32 l, h;\n"
            "and.b32 l, $1, 0xffff;\nsetp.eq.b32 p, l, 0x8000;\nselp.b32 l, 0, l, p;\n"
            "and.b32 h, $1, 0xffff0000;\nsetp.eq.b32 q, h, 0x80000000;\nselp.b32 h, 0, h, q;\n"
            "or.b32 $0, l, h;\n}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def shfl_bfly_f32(value, offset: int, *, loc=None, ip=None):
    return Float32(
        asm(T.f32(), [fv(value, loc=loc, ip=ip)], f"shfl.sync.bfly.b32 $0, $1, {offset}, 31, -1;",
            "=f,f", loc=loc, ip=ip)
    )  # fmt: skip


@dsl_user_op
def shfl_bfly_b32(value, offset: int, *, loc=None, ip=None):
    return Int32(
        asm(T.i32(), [iv(value, loc=loc, ip=ip)], f"shfl.sync.bfly.b32 $0, $1, {offset}, 31, -1;",
            "=r,r", loc=loc, ip=ip)
    )  # fmt: skip


@dsl_user_op
def mma_bf16(a, b0, b1, c, *, loc=None, ip=None):
    """``d[16x8] = a[16x16] @ b[16x8] + c`` (``mma.sync`` m16n8k16, BF16 inputs, FP32
    accumulation): ``a`` four words, ``b0`` / ``b1`` two, ``c`` four FP32 values."""
    res = asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [iv(x, loc=loc, ip=ip) for x in (*a, b0, b1)] + [fv(x, loc=loc, ip=ip) for x in c],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        loc=loc,
        ip=ip,
    )
    return unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


# ---------------------------------------------------------------------------
# Global memory
# ---------------------------------------------------------------------------


@dsl_user_op
def ld_v4(addr, *, loc=None, ip=None):
    """16 bytes of a tensor an earlier grid wrote."""
    res = asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [lv(addr, loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def ld_nc_u8(addr, *, loc=None, ip=None):
    return Int32(
        asm(T.i32(), [lv(addr, loc=loc, ip=ip)], "ld.global.nc.u8 $0, [$1];", "=r,l", loc=loc,
            ip=ip)
    )  # fmt: skip


@dsl_user_op
def ld_nc_u64(addr, *, loc=None, ip=None):
    """A 64-bit word of a tensor no kernel writes."""
    return Int64(
        asm(T.i64(), [lv(addr, loc=loc, ip=ip)], "ld.global.nc.u64 $0, [$1];", "=l,l", loc=loc,
            ip=ip)
    )  # fmt: skip


@dsl_user_op
def ld_nc_v2(addr, *, loc=None, ip=None):
    res = asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [lv(addr, loc=loc, ip=ip)],
        "ld.global.nc.v2.b32 {$0, $1}, [$2];",
        "=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return unpack(res, T.i32(), Int32, 2, loc=loc, ip=ip)


@dsl_user_op
def ld_nc_v4(addr, *, loc=None, ip=None):
    """16 bytes of a tensor the kernel does not write."""
    res = asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [lv(addr, loc=loc, ip=ip)],
        "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def ld_volatile_v4(addr, *, loc=None, ip=None):
    """16 bytes read past L1 (peers write them over NVLink)."""
    res = asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [lv(addr, loc=loc, ip=ip)],
        "ld.volatile.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def ld_acquire_u32(addr, *, loc=None, ip=None):
    return Int32(
        asm(T.i32(), [lv(addr, loc=loc, ip=ip)], "ld.acquire.gpu.global.u32 $0, [$1];", "=r,l",
            loc=loc, ip=ip)
    )  # fmt: skip


@dsl_user_op
def st_u8(addr, value, *, loc=None, ip=None):
    asm(None, [lv(addr, loc=loc, ip=ip), iv(value, loc=loc, ip=ip)], "st.global.u8 [$0], $1;",
        "l,r", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def st_u16(addr, value, *, loc=None, ip=None):
    asm(None, [lv(addr, loc=loc, ip=ip), iv(value, loc=loc, ip=ip)], "st.global.u16 [$0], $1;",
        "l,r", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def st_u32(addr, value, *, loc=None, ip=None):
    asm(None, [lv(addr, loc=loc, ip=ip), iv(value, loc=loc, ip=ip)], "st.global.u32 [$0], $1;",
        "l,r", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def st_v2(addr, v0, v1, *, loc=None, ip=None):
    asm(
        None,
        [lv(addr, loc=loc, ip=ip), iv(v0, loc=loc, ip=ip), iv(v1, loc=loc, ip=ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_v4(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    asm(
        None,
        [lv(addr, loc=loc, ip=ip)] + [iv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_sentinel_v4(addr, *, loc=None, ip=None):
    """16 bytes of ``0xff`` (the empty-word sentinel of a receive buffer)."""
    asm(
        None,
        [lv(addr, loc=loc, ip=ip)],
        "{\n.reg .b32 t;\nmov.b32 t, -1;\nst.global.v4.b32 [$0], {t, t, t, t};\n}",
        "l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def multimem_st_v4(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    """16-byte store to the address in every rank's buffer (multicast mapping)."""
    asm(
        None,
        [lv(addr, loc=loc, ip=ip)] + [iv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "multimem.st.relaxed.sys.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atom_add_u64(addr, value, *, loc=None, ip=None):
    """Relaxed GPU-scope ``atomicAdd``; returns the old value."""
    return Int64(
        asm(T.i64(), [lv(addr, loc=loc, ip=ip), lv(value, loc=loc, ip=ip)],
            "atom.relaxed.gpu.global.add.u64 $0, [$1], $2;", "=l,l,l", loc=loc, ip=ip)
    )  # fmt: skip


@dsl_user_op
def atom_add_acq_rel_u32(addr, value, *, loc=None, ip=None):
    """Acquire-release GPU-scope ``atomicAdd``; returns the old value."""
    return Int32(
        asm(T.i32(), [lv(addr, loc=loc, ip=ip), iv(value, loc=loc, ip=ip)],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], $2;", "=r,l,r", loc=loc, ip=ip)
    )  # fmt: skip


@dsl_user_op
def atom_add_relaxed_u32(addr, value, *, loc=None, ip=None):
    """Relaxed GPU-scope ``atomicAdd``; returns the old value (the thread stalls only where it
    reads it)."""
    return Int32(
        asm(T.i32(), [lv(addr, loc=loc, ip=ip), iv(value, loc=loc, ip=ip)],
            "atom.relaxed.gpu.global.add.u32 $0, [$1], $2;", "=r,l,r", loc=loc, ip=ip)
    )  # fmt: skip


@dsl_user_op
def red_release_add(addr, value, *, loc=None, ip=None):
    """Release GPU-scope reduction add (no return value)."""
    asm(None, [lv(addr, loc=loc, ip=ip), iv(value, loc=loc, ip=ip)],
        "red.release.gpu.global.add.u32 [$0], $1;", "l,r", loc=loc, ip=ip)  # fmt: skip


# ---------------------------------------------------------------------------
# Shared memory
# ---------------------------------------------------------------------------


@dsl_user_op
def lds_b32(addr, *, loc=None, ip=None):
    return Int32(
        asm(T.i32(), [iv(addr, loc=loc, ip=ip)], "ld.shared.b32 $0, [$1];", "=r,r", loc=loc, ip=ip)
    )


@dsl_user_op
def lds_b64(addr, *, loc=None, ip=None):
    return Int64(
        asm(T.i64(), [iv(addr, loc=loc, ip=ip)], "ld.shared.b64 $0, [$1];", "=l,r", loc=loc, ip=ip)
    )


@dsl_user_op
def lds_f32(addr, *, loc=None, ip=None):
    return Float32(
        asm(T.f32(), [iv(addr, loc=loc, ip=ip)], "ld.shared.f32 $0, [$1];", "=f,r", loc=loc, ip=ip)
    )


@dsl_user_op
def lds_v4(addr, *, loc=None, ip=None):
    """16 bytes of shared memory as four 32-bit words."""
    res = asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [iv(addr, loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def lds_v4_f32(addr, *, loc=None, ip=None):
    res = asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [iv(addr, loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        loc=loc,
        ip=ip,
    )
    return unpack(res, T.f32(), Float32, 4, loc=loc, ip=ip)


@dsl_user_op
def ldsm_x4(addr, *, loc=None, ip=None):
    """Four 8x8 b16 matrices from shared memory (``ldmatrix.sync.aligned.m8n8.x4``)."""
    res = asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [iv(addr, loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


@dsl_user_op
def sts_b32(addr, value, *, loc=None, ip=None):
    asm(None, [iv(addr, loc=loc, ip=ip), iv(value, loc=loc, ip=ip)], "st.shared.b32 [$0], $1;",
        "r,r", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def sts_b64(addr, value, *, loc=None, ip=None):
    asm(None, [iv(addr, loc=loc, ip=ip), lv(value, loc=loc, ip=ip)], "st.shared.b64 [$0], $1;",
        "r,l", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def sts_f32(addr, value, *, loc=None, ip=None):
    asm(None, [iv(addr, loc=loc, ip=ip), fv(value, loc=loc, ip=ip)], "st.shared.f32 [$0], $1;",
        "r,f", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def sts_bf16(addr, value, *, loc=None, ip=None):
    """Store ``value`` rounded to BF16 (nearest even) at shared ``addr``."""
    asm(
        None,
        [iv(addr, loc=loc, ip=ip), fv(value, loc=loc, ip=ip)],
        "{\n.reg .b16 h;\ncvt.rn.bf16.f32 h, $1;\nst.shared.b16 [$0], h;\n}",
        "r,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def sts_v2(addr, v0, v1, *, loc=None, ip=None):
    asm(
        None,
        [iv(addr, loc=loc, ip=ip), iv(v0, loc=loc, ip=ip), iv(v1, loc=loc, ip=ip)],
        "st.shared.v2.b32 [$0], {$1, $2};",
        "r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def sts_v4(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    asm(
        None,
        [iv(x, loc=loc, ip=ip) for x in (addr, v0, v1, v2, v3)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def sts_v4_f32(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    asm(
        None,
        [iv(addr, loc=loc, ip=ip)] + [fv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.shared.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f",
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# Barriers, mbarriers and bulk copies
# ---------------------------------------------------------------------------


@dsl_user_op
def bar_sync(barrier_id: int, count: int, *, loc=None, ip=None):
    asm(None, [], f"bar.sync {barrier_id}, {count};", "", loc=loc, ip=ip)


@dsl_user_op
def cluster_arrive_relaxed(*, loc=None, ip=None):
    asm(None, [], "barrier.cluster.arrive.relaxed.aligned;", "", loc=loc, ip=ip)


@dsl_user_op
def cluster_sync(*, loc=None, ip=None):
    """Cluster barrier with release / acquire semantics."""
    asm(None, [], "barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;",
        "", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def fence_proxy_async(*, loc=None, ip=None):
    asm(None, [], "fence.proxy.async.shared::cta;", "", loc=loc, ip=ip)


@dsl_user_op
def fence_mbar_init(*, loc=None, ip=None):
    asm(None, [], "fence.mbarrier_init.release.cluster;", "", loc=loc, ip=ip)


@dsl_user_op
def mbar_init(addr, count: int, *, loc=None, ip=None):
    asm(None, [iv(addr, loc=loc, ip=ip)], f"mbarrier.init.shared::cta.b64 [$0], {count};", "r",
        loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def mbar_init_fenced(addr, count: int, *, loc=None, ip=None):
    """:func:`mbar_init` followed by :func:`fence_mbar_init`."""
    asm(
        None,
        [iv(addr, loc=loc, ip=ip)],
        f"mbarrier.init.shared::cta.b64 [$0], {count};\nfence.mbarrier_init.release.cluster;",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbar_arrive(addr, *, loc=None, ip=None):
    asm(None, [iv(addr, loc=loc, ip=ip)], "mbarrier.arrive.shared::cta.b64 _, [$0];", "r",
        loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def mbar_expect_tx(addr, nbytes: int, *, loc=None, ip=None):
    """Arrive on the mbarrier at ``addr`` and expect ``nbytes`` of transactions."""
    asm(None, [iv(addr, loc=loc, ip=ip)],
        f"mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], {nbytes};", "r", loc=loc,
        ip=ip)  # fmt: skip


@dsl_user_op
def mbar_try_wait(addr, phase, *, loc=None, ip=None):
    """1 once phase ``phase`` of the shared mbarrier at ``addr`` has completed, else 0."""
    return Int32(
        asm(
            T.i32(),
            [iv(addr, loc=loc, ip=ip), iv(phase, loc=loc, ip=ip)],
            "{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2;\n"
            "selp.u32 $0, 1, 0, p;\n}",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bulk_g2s(dst, src, nbytes: int, mbar, *, loc=None, ip=None):
    """Bulk copy of ``nbytes`` from global ``src`` to shared ``dst``, completing on ``mbar``."""
    asm(
        None,
        [iv(dst, loc=loc, ip=ip), lv(src, loc=loc, ip=ip), iv(mbar, loc=loc, ip=ip)],
        f"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], {nbytes}, [$2];",
        "r,l,r",
        loc=loc,
        ip=ip,
    )
