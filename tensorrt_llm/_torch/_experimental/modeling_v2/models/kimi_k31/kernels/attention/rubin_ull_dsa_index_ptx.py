# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inline-PTX helpers of the DSA indexer CuTe DSL kernels.

Every memory access of these kernels goes through an explicit PTX instruction on a raw address
(64-bit for global, 32-bit for shared), so the cache operator, the predicate and the memory scope
of each access are visible here. Conditions are 0/1 ``Uint32`` values.
"""

from cutlass import Uint32, Uint64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op


def _asm(code, constraints, operands, result=None, *, effects=True, loc=None, ip=None):
    """One inline-asm op; ``result`` is a DSL type, a tuple of them, or None."""
    if result is None:
        ir_type = None
    elif isinstance(result, tuple):
        ir_type = llvm.StructType.get_literal([t.mlir_type for t in result])
    else:
        ir_type = result.mlir_type
    value = llvm.inline_asm(
        ir_type,
        [o.ir_value(loc=loc, ip=ip) for o in operands],
        code,
        constraints,
        has_side_effects=effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    if result is None:
        return None
    if isinstance(result, tuple):
        return tuple(
            t(llvm.extractvalue(t.mlir_type, value, [i], loc=loc, ip=ip))
            for i, t in enumerate(result)
        )
    return result(value)


# ---- integer helpers ---------------------------------------------------------------------------


@dsl_user_op
def sel(cond, a, b, *, loc=None, ip=None) -> Uint32:
    """``cond ? a : b`` (u32)."""
    return _asm(
        "{ .reg .pred p; setp.ne.u32 p, $1, 0; selp.b32 $0, $2, $3, p; }",
        "=r,r,r,r",
        (Uint32(cond), Uint32(a), Uint32(b)),
        Uint32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def lt_u32(a, b, *, loc=None, ip=None) -> Uint32:
    return _asm(
        "{ .reg .pred p; setp.lt.u32 p, $1, $2; selp.u32 $0, 1, 0, p; }",
        "=r,r,r",
        (Uint32(a), Uint32(b)),
        Uint32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def eq_u32(a, b, *, loc=None, ip=None) -> Uint32:
    return _asm(
        "{ .reg .pred p; setp.eq.u32 p, $1, $2; selp.u32 $0, 1, 0, p; }",
        "=r,r,r",
        (Uint32(a), Uint32(b)),
        Uint32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def min_u32(a, b, *, loc=None, ip=None) -> Uint32:
    return _asm(
        "min.u32 $0, $1, $2;",
        "=r,r,r",
        (Uint32(a), Uint32(b)),
        Uint32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ordered_bits(bits, *, loc=None, ip=None) -> Uint32:
    """Order-preserving unsigned image of FP32 ``bits`` (``-0.0`` just below ``+0.0``): the low
    word of the signed order key ``bits >= 0 ? bits : bits ^ 0x7FFFFFFF``, sign-flipped."""
    return _asm(
        "{ .reg .b32 m; shr.s32 m, $1, 31; or.b32 m, m, 0x80000000; xor.b32 $0, $1, m; }",
        "=r,r",
        (Uint32(bits),),
        Uint32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def sel64(cond, a, b, *, loc=None, ip=None) -> Uint64:
    """``cond ? a : b`` (u64)."""
    return _asm(
        "{ .reg .pred p; setp.ne.u32 p, $1, 0; selp.b64 $0, $2, $3, p; }",
        "=l,r,l,l",
        (Uint32(cond), Uint64(a), Uint64(b)),
        Uint64,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def pack64(lo, hi, *, loc=None, ip=None) -> Uint64:
    return _asm(
        "mov.b64 $0, {$1, $2};",
        "=l,r,r",
        (Uint32(lo), Uint32(hi)),
        Uint64,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def scan_step(value, offset: int, *, loc=None, ip=None) -> Uint32:
    """One step of a warp inclusive scan: ``value + (lane >= offset ? value[lane - offset] : 0)``."""
    return _asm(
        "{ .reg .b32 t; .reg .pred p; "
        f"shfl.sync.up.b32 t|p, $1, {offset}, 0, -1; "
        "@!p mov.b32 t, 0; add.u32 $0, $1, t; }",
        "=r,r",
        (Uint32(value),),
        Uint32,
        effects=True,
        loc=loc,
        ip=ip,
    )


# ---- shared memory -----------------------------------------------------------------------------


@dsl_user_op
def ld_shared_u32(addr, *, loc=None, ip=None) -> Uint32:
    return _asm("ld.shared.b32 $0, [$1];", "=r,r", (Uint32(addr),), Uint32, loc=loc, ip=ip)


@dsl_user_op
def ld_shared_u64(addr, *, loc=None, ip=None) -> Uint64:
    return _asm("ld.shared.b64 $0, [$1];", "=l,r", (Uint32(addr),), Uint64, loc=loc, ip=ip)


@dsl_user_op
def ld_shared_u32_if(cond, addr, *, loc=None, ip=None) -> Uint32:
    """``cond ? *addr : 0``."""
    return _asm(
        "{ .reg .pred p; setp.ne.u32 p, $1, 0; mov.b32 $0, 0; @p ld.shared.b32 $0, [$2]; }",
        "=r,r,r",
        (Uint32(cond), Uint32(addr)),
        Uint32,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_shared_v4_if(cond, addr, *, loc=None, ip=None):
    """Four words at 16-byte aligned ``addr`` if ``cond`` (else zeros)."""
    return _asm(
        "{ .reg .pred p; setp.ne.u32 p, $4, 0; mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; "
        "mov.b32 $3, 0; @p ld.shared.v4.b32 {$0, $1, $2, $3}, [$5]; }",
        "=r,=r,=r,=r,r,r",
        (Uint32(cond), Uint32(addr)),
        (Uint32, Uint32, Uint32, Uint32),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def zero_shared_v4_if(cond, addr, *, loc=None, ip=None) -> None:
    """Four zero words at 16-byte aligned ``addr`` if ``cond``."""
    _asm(
        "{ .reg .pred p; .reg .b32 z; setp.ne.u32 p, $0, 0; mov.b32 z, 0; "
        "@p st.shared.v4.b32 [$1], {z, z, z, z}; }",
        "r,r",
        (Uint32(cond), Uint32(addr)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_shared_u32_if(cond, addr, value, *, loc=None, ip=None) -> None:
    _asm(
        "{ .reg .pred p; setp.ne.u32 p, $0, 0; @p st.shared.b32 [$1], $2; }",
        "r,r,r",
        (Uint32(cond), Uint32(addr), Uint32(value)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_shared_u64_if(cond, addr, value, *, loc=None, ip=None) -> None:
    _asm(
        "{ .reg .pred p; setp.ne.u32 p, $0, 0; @p st.shared.b64 [$1], $2; }",
        "r,r,l",
        (Uint32(cond), Uint32(addr), Uint64(value)),
        loc=loc,
        ip=ip,
    )


# ---- global memory -----------------------------------------------------------------------------


@dsl_user_op
def ld_global_u32(addr, *, loc=None, ip=None) -> Uint32:
    """Data written by an earlier grid (read after ``griddepcontrol.wait``): L2, not L1."""
    return _asm("ld.global.cg.b32 $0, [$1];", "=r,l", (Uint64(addr),), Uint32, loc=loc, ip=ip)


@dsl_user_op
def ld_global_u64_nc(addr, *, loc=None, ip=None) -> Uint64:
    """Read-only for the kernel's lifetime (set up at workspace creation)."""
    return _asm("ld.global.nc.b64 $0, [$1];", "=l,l", (Uint64(addr),), Uint64, loc=loc, ip=ip)


@dsl_user_op
def ld_global_v4_if(cond, addr, *, loc=None, ip=None):
    """Four words at 16-byte aligned ``addr`` if ``cond`` (else zeros), through L2."""
    return _asm(
        "{ .reg .pred p; setp.ne.u32 p, $4, 0; mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; "
        "mov.b32 $3, 0; @p ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$5]; }",
        "=r,=r,=r,=r,r,l",
        (Uint32(cond), Uint64(addr)),
        (Uint32, Uint32, Uint32, Uint32),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_global_u32_if(cond, addr, value, *, loc=None, ip=None) -> None:
    _asm(
        "{ .reg .pred p; setp.ne.u32 p, $0, 0; @p st.global.b32 [$1], $2; }",
        "r,l,r",
        (Uint32(cond), Uint64(addr), Uint32(value)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_global_u64_if(cond, addr, value, *, loc=None, ip=None) -> None:
    _asm(
        "{ .reg .pred p; setp.ne.u32 p, $0, 0; @p st.global.b64 [$1], $2; }",
        "r,l,l",
        (Uint32(cond), Uint64(addr), Uint64(value)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def publish_sys_if(cond, addr, value, *, loc=None, ip=None) -> None:
    """Release (system scope) exchange of a flag word if ``cond``: every write this CTA ordered
    before it (by a CTA barrier) is visible to a peer that acquires the flag."""
    _asm(
        "{ .reg .b32 old; .reg .pred p; setp.ne.u32 p, $0, 0; "
        "@p atom.release.sys.global.exch.b32 old, [$1], $2; }",
        "r,l,r",
        (Uint32(cond), Uint64(addr), Uint32(value)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fence_acq_rel_gpu_if(cond, *, loc=None, ip=None) -> None:
    """If ``cond``: ``fence.acq_rel.gpu`` (after a CTA barrier: releases the CTA's earlier writes
    to a later store of this thread; before reads: acquires what an earlier load observed)."""
    _asm(
        "{ .reg .pred p; setp.ne.u32 p, $0, 0; @p fence.acq_rel.gpu; }",
        "r",
        (Uint32(cond),),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def count_shared_if(cond, addr, *, loc=None, ip=None) -> None:
    """``*addr += 1`` (shared u32) if ``cond``."""
    _asm(
        "{ .reg .pred p; setp.ne.u32 p, $0, 0; @p red.shared.add.u32 [$1], 1; }",
        "r,r",
        (Uint32(cond), Uint32(addr)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def warp_min_u64(value, *, loc=None, ip=None) -> Uint64:
    """Unsigned minimum over the warp (every lane gets it)."""
    steps = "".join(
        f"mov.b64 {{lo, hi}}, m; shfl.sync.bfly.b32 lo, lo, {o}, 31, -1; "
        f"shfl.sync.bfly.b32 hi, hi, {o}, 31, -1; mov.b64 t, {{lo, hi}}; min.u64 m, m, t; "
        for o in (16, 8, 4, 2, 1)
    )
    return _asm(
        "{ .reg .b64 m, t; .reg .b32 lo, hi; mov.b64 m, $1; " + steps + "mov.b64 $0, m; }",
        "=l,l",
        (Uint64(value),),
        Uint64,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def min_u64(a, b, *, loc=None, ip=None) -> Uint64:
    return _asm(
        "min.u64 $0, $1, $2;",
        "=l,l,l",
        (Uint64(a), Uint64(b)),
        Uint64,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def shfl_idx(value, lane, *, loc=None, ip=None) -> Uint32:
    """``value`` of lane ``lane`` (every lane of the warp calls)."""
    return _asm(
        f"shfl.sync.idx.b32 $0, $1, {lane}, 31, -1;",
        "=r,r",
        (Uint32(value),),
        Uint32,
        loc=loc,
        ip=ip,
    )


# ---- key-shard merge ---------------------------------------------------------------------------


@dsl_user_op
def shr_u32(value, shift, *, loc=None, ip=None) -> Uint32:
    """``value >> shift`` with PTX clamping (a shift of 32 or more gives 0)."""
    return _asm(
        "shr.b32 $0, $1, $2;",
        "=r,r,r",
        (Uint32(value), Uint32(shift)),
        Uint32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def shl_u64(value, shift, *, loc=None, ip=None) -> Uint64:
    """``value << shift`` with PTX clamping (a shift of 64 or more gives 0)."""
    return _asm(
        "shl.b64 $0, $1, $2;",
        "=l,l,r",
        (Uint64(value), Uint32(shift)),
        Uint64,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def unpack64(value, *, loc=None, ip=None):
    """(low, high) words of ``value``."""
    return _asm(
        "mov.b64 {$0, $1}, $2;",
        "=r,=r,l",
        (Uint64(value),),
        (Uint32, Uint32),
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ge_u64(a, b, *, loc=None, ip=None) -> Uint32:
    return _asm(
        "{ .reg .pred p; setp.ge.u64 p, $1, $2; selp.u32 $0, 1, 0, p; }",
        "=r,l,l",
        (Uint64(a), Uint64(b)),
        Uint32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ne_u64(a, b, *, loc=None, ip=None) -> Uint32:
    return _asm(
        "{ .reg .pred p; setp.ne.u64 p, $1, $2; selp.u32 $0, 1, 0, p; }",
        "=r,l,l",
        (Uint64(a), Uint64(b)),
        Uint32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_global_cv_u32_if(cond, addr, *, loc=None, ip=None) -> Uint32:
    """``cond ? *addr : 0``, fetched again (written by peers; read after acquiring their flag)."""
    return _asm(
        "{ .reg .pred p; setp.ne.u32 p, $1, 0; mov.b32 $0, 0; @p ld.global.cv.b32 $0, [$2]; }",
        "=r,r,l",
        (Uint32(cond), Uint64(addr)),
        Uint32,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_global_cv_u64_if(cond, addr, *, loc=None, ip=None) -> Uint64:
    return _asm(
        "{ .reg .pred p; setp.ne.u32 p, $1, 0; mov.b64 $0, 0; @p ld.global.cv.b64 $0, [$2]; }",
        "=l,r,l",
        (Uint32(cond), Uint64(addr)),
        Uint64,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_global_cv_v2_u64(addr, *, loc=None, ip=None):
    """Two words at 16-byte aligned ``addr``, fetched again."""
    return _asm(
        "ld.global.cv.v2.b64 {$0, $1}, [$2];",
        "=l,=l,l",
        (Uint64(addr),),
        (Uint64, Uint64),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_global_cv_v4_if(cond, addr, *, loc=None, ip=None):
    """Four words at 16-byte aligned ``addr`` if ``cond`` (else zeros), fetched again."""
    return _asm(
        "{ .reg .pred p; setp.ne.u32 p, $4, 0; mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; "
        "mov.b32 $3, 0; @p ld.global.cv.v4.b32 {$0, $1, $2, $3}, [$5]; }",
        "=r,=r,=r,=r,r,l",
        (Uint32(cond), Uint64(addr)),
        (Uint32, Uint32, Uint32, Uint32),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_global_u64_if(cond, addr, *, loc=None, ip=None) -> Uint64:
    """``cond ? *addr : 0`` through L2 (data of an earlier grid)."""
    return _asm(
        "{ .reg .pred p; setp.ne.u32 p, $1, 0; mov.b64 $0, 0; @p ld.global.cg.b64 $0, [$2]; }",
        "=l,r,l",
        (Uint32(cond), Uint64(addr)),
        Uint64,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_global_u32_if(cond, addr, *, loc=None, ip=None) -> Uint32:
    return _asm(
        "{ .reg .pred p; setp.ne.u32 p, $1, 0; mov.b32 $0, 0; @p ld.global.cg.b32 $0, [$2]; }",
        "=r,r,l",
        (Uint32(cond), Uint64(addr)),
        Uint32,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def prefetch_l2_if(cond, addr, *, loc=None, ip=None) -> None:
    """``prefetch.global.L2`` of the line at ``addr`` if ``cond``."""
    _asm(
        "{ .reg .pred p; setp.ne.u32 p, $0, 0; @p prefetch.global.L2 [$1]; }",
        "r,l",
        (Uint32(cond), Uint64(addr)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def kv_row_if(cond, table, cols, local, stride_rows, row_offset, *, loc=None, ip=None) -> Uint32:
    """The KV pool row of request-local token ``local`` (64-token pages) if ``cond``, as
    ``convertReqIndexToGlobal``: ``table[local / 64] * stride_rows + row_offset + local % 64``
    in 32-bit arithmetic, ``-1`` where the page is past ``cols`` or the entry is negative.
    ``table``: the request's block-table row (int32 page indices; step metadata that an earlier
    grid of the graph may write, so read after ``griddepcontrol.wait`` through L2, not the
    non-coherent path); without ``cond`` no load and ``-1``."""
    return _asm(
        "{ .reg .pred p, q; .reg .b32 blk, page, off; .reg .b64 a; "
        "shr.u32 blk, $2, 6; setp.ne.u32 p, $1, 0; setp.lt.and.u32 p, blk, $4, p; "
        "mov.b32 page, -1; mul.wide.u32 a, blk, 4; add.s64 a, a, $3; "
        "@p ld.global.cg.b32 page, [a]; "
        "and.b32 off, $2, 63; add.s32 off, off, $6; mad.lo.s32 $0, page, $5, off; "
        "setp.lt.s32 q, page, 0; @q mov.b32 $0, -1; }",
        "=r,r,r,l,r,r,r",
        (
            Uint32(cond),
            Uint32(local),
            Uint64(table),
            Uint32(cols),
            Uint32(stride_rows),
            Uint32(row_offset),
        ),
        Uint32,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_global_u8_if(cond, addr, value, *, loc=None, ip=None) -> None:
    _asm(
        "{ .reg .pred p; .reg .b16 h; setp.ne.u32 p, $0, 0; cvt.u16.u32 h, $2; "
        "@p st.global.b8 [$1], h; }",
        "r,l,r",
        (Uint32(cond), Uint64(addr), Uint32(value)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_global_if(cond, addr, value, *, loc=None, ip=None) -> None:
    """``*addr += value`` (global u32, relaxed, GPU scope) if ``cond``."""
    _asm(
        "{ .reg .pred p; setp.ne.u32 p, $0, 0; @p red.relaxed.gpu.global.add.u32 [$1], $2; }",
        "r,l,r",
        (Uint32(cond), Uint64(addr), Uint32(value)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def add_shared_if(cond, addr, value, *, loc=None, ip=None) -> None:
    """``*addr += value`` (shared u32) if ``cond``."""
    _asm(
        "{ .reg .pred p; setp.ne.u32 p, $0, 0; @p red.shared.add.u32 [$1], $2; }",
        "r,r,r",
        (Uint32(cond), Uint32(addr), Uint32(value)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def wait_flag_sys_if(cond, addr, value, max_spin, *, loc=None, ip=None) -> Uint32:
    """If ``cond``: spin (system-scope acquire loads) until ``*addr == value`` or ``max_spin``
    further polls; returns 1 on a timeout, else 0."""
    return _asm(
        "{ .reg .pred p; .reg .b32 v, n; mov.b32 $0, 0; mov.b32 n, 0; "
        "setp.eq.u32 p, $1, 0; @p bra WAIT_DONE_${:uid}; "
        "WAIT_LOOP_${:uid}: ld.acquire.sys.global.b32 v, [$2]; "
        "setp.eq.b32 p, v, $3; @p bra WAIT_DONE_${:uid}; "
        "add.u32 n, n, 1; setp.lt.u32 p, n, $4; @p bra WAIT_LOOP_${:uid}; "
        "mov.b32 $0, 1; "
        "WAIT_DONE_${:uid}: }",
        "=r,r,l,r,r",
        (Uint32(cond), Uint64(addr), Uint32(value), Uint32(max_spin)),
        Uint32,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def advance_epoch_if(cond, state, epoch, ctas, *, loc=None, ip=None) -> None:
    """If ``cond``: take a ticket (``state[1]``); the last of ``ctas`` CTAs resets it and
    publishes ``epoch`` into ``state[0]`` (release, GPU scope)."""
    _asm(
        "{ .reg .pred q, p; .reg .b32 t, old; setp.ne.u32 q, $0, 0; mov.b32 t, 0; "
        "@q atom.acq_rel.gpu.global.add.u32 t, [$1+4], 1; "
        "setp.eq.and.u32 p, t, $3, q; "
        "@p st.relaxed.gpu.global.b32 [$1+4], 0; "
        "@p atom.release.gpu.global.exch.b32 old, [$1], $2; }",
        "r,l,r,r",
        (Uint32(cond), Uint64(state), Uint32(epoch), Uint32(ctas) - Uint32(1)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def warp_min_u32(value, *, loc=None, ip=None) -> Uint32:
    """Unsigned minimum over the warp (every lane gets it)."""
    return _asm("redux.sync.min.u32 $0, $1, -1;", "=r,r", (Uint32(value),), Uint32, loc=loc, ip=ip)


@dsl_user_op
def min_shared_if(cond, addr, value, *, loc=None, ip=None) -> None:
    """``*addr = min(*addr, value)`` (shared u32) if ``cond``."""
    _asm(
        "{ .reg .pred p; setp.ne.u32 p, $0, 0; @p red.shared.min.u32 [$1], $2; }",
        "r,r,r",
        (Uint32(cond), Uint32(addr), Uint32(value)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def stamp_if(cond, addr, *, loc=None, ip=None) -> None:
    """If ``cond``: ``addr[0] = %globaltimer`` (ns), ``addr[1] = %clock64`` (SM cycles)."""
    _asm(
        "{ .reg .pred p; .reg .b64 g, c; setp.ne.u32 p, $0, 0; mov.u64 g, %globaltimer; "
        "mov.u64 c, %clock64; @p st.global.v2.b64 [$1], {g, c}; }",
        "r,l",
        (Uint32(cond), Uint64(addr)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def bar_sync(threads, *, loc=None, ip=None) -> None:
    """Named barrier 1 over ``threads`` threads (a multiple of 32): the first warps of the CTA
    that work on a row, the others never reach it."""
    _asm("bar.sync 1, $0;", "r", (Uint32(threads),), loc=loc, ip=ip)


@dsl_user_op
def st_global_v2_u64_if(cond, addr, a, b, *, loc=None, ip=None) -> None:
    _asm(
        "{ .reg .pred p; setp.ne.u32 p, $0, 0; @p st.global.v2.b64 [$1], {$2, $3}; }",
        "r,l,l,l",
        (Uint32(cond), Uint64(addr), Uint64(a), Uint64(b)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def max_u32(a, b, *, loc=None, ip=None) -> Uint32:
    return _asm(
        "max.u32 $0, $1, $2;",
        "=r,r,r",
        (Uint32(a), Uint32(b)),
        Uint32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def popc(value, *, loc=None, ip=None) -> Uint32:
    return _asm(
        "popc.b32 $0, $1;",
        "=r,r",
        (Uint32(value),),
        Uint32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def or_shared_if(cond, addr, value, *, loc=None, ip=None) -> None:
    """``*addr |= value`` (shared u32) if ``cond``."""
    _asm(
        "{ .reg .pred p; setp.ne.u32 p, $0, 0; @p red.shared.or.b32 [$1], $2; }",
        "r,r,r",
        (Uint32(cond), Uint32(addr), Uint32(value)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def globaltimer(*, loc=None, ip=None) -> Uint64:
    """``%globaltimer`` (ns)."""
    return _asm("mov.u64 $0, %globaltimer;", "=l", (), Uint64, loc=loc, ip=ip)


# ---- LL exchange -------------------------------------------------------------------------------
# An LL pair is 16 bytes: two 64-bit halves (word, epoch), each with the word in its low 32 bits.
# Both are stored by one ``st.relaxed.sys.v2.b64`` and loaded by one ``ld.relaxed.sys.v2.b64``.
# The PTX memory model treats a vector access as one access per element, and two morally strong
# (relaxed, system scope) accesses to the same aligned 8 bytes are single-copy atomic: a half whose
# epoch matches holds the word stored with that epoch. The data is its own flag (no fence).


@dsl_user_op
def st_ll_pair_if(cond, addr, w0, w1, epoch, *, loc=None, ip=None) -> None:
    """If ``cond``: the LL pair ``(w0, epoch), (w1, epoch)`` at 16-byte aligned ``addr``."""
    _asm(
        "{ .reg .pred p; .reg .b64 a, b; setp.ne.u32 p, $0, 0; mov.b64 a, {$2, $4}; "
        "mov.b64 b, {$3, $4}; @p st.relaxed.sys.global.v2.b64 [$1], {a, b}; }",
        "r,l,r,r,r",
        (Uint32(cond), Uint64(addr), Uint32(w0), Uint32(w1), Uint32(epoch)),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_ll_pair_if(cond, addr, *, loc=None, ip=None):
    """``(w0, e0, w1, e1)`` of the LL pair at ``addr`` if ``cond`` (else zeros)."""
    return _asm(
        "{ .reg .pred p; .reg .b64 a, b; setp.ne.u32 p, $4, 0; mov.b64 a, 0; mov.b64 b, 0; "
        "@p ld.relaxed.sys.global.v2.b64 {a, b}, [$5]; mov.b64 {$0, $1}, a; mov.b64 {$2, $3}, b; }",
        "=r,=r,=r,=r,r,l",
        (Uint32(cond), Uint64(addr)),
        (Uint32, Uint32, Uint32, Uint32),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def settle_ll_if(cond, addr, w0, e0, w1, e1, epoch, deadline, *, loc=None, ip=None):
    """If ``cond``: the words of the LL pair at ``addr`` once both halves carry ``epoch``, from a
    first load ``(w0, e0, w1, e1)``; reloads until then or until ``%globaltimer`` passes
    ``deadline`` (ns). Returns ``(w0, w1, timed_out)``; without ``cond`` the first load's words."""
    return _asm(
        "{ .reg .pred p, q; .reg .b64 a, b, t; .reg .b32 x0, x1, ea, eb, to; "
        "mov.b32 x0, $4; mov.b32 ea, $5; mov.b32 x1, $6; mov.b32 eb, $7; mov.b32 to, 0; "
        "setp.eq.u32 p, $3, 0; @p bra LL_DONE_${:uid}; "
        "LL_LOOP_${:uid}: setp.eq.u32 p, ea, $8; setp.eq.and.u32 p, eb, $8, p; "
        "@p bra LL_DONE_${:uid}; "
        "mov.u64 t, %globaltimer; setp.gt.u64 q, t, $9; @q bra LL_TIMEOUT_${:uid}; "
        "ld.relaxed.sys.global.v2.b64 {a, b}, [$10]; mov.b64 {x0, ea}, a; mov.b64 {x1, eb}, b; "
        "bra LL_LOOP_${:uid}; "
        "LL_TIMEOUT_${:uid}: mov.b32 to, 1; "
        "LL_DONE_${:uid}: mov.b32 $0, x0; mov.b32 $1, x1; mov.b32 $2, to; }",
        "=r,=r,=r,r,r,r,r,r,r,l,l",
        (
            Uint32(cond),
            Uint32(w0),
            Uint32(e0),
            Uint32(w1),
            Uint32(e1),
            Uint32(epoch),
            Uint64(deadline),
            Uint64(addr),
        ),
        (Uint32, Uint32, Uint32),
        loc=loc,
        ip=ip,
    )


# ---- thread-block cluster (distributed shared memory) ------------------------------------------
# Shared addresses below are 32-bit; ``mapa`` turns one of this CTA's into the shared::cluster
# address of the same offset in cluster CTA ``rank`` (each CTA's window is contiguous, so an offset
# may be added to a mapped base).


@dsl_user_op
def cluster_ctarank(*, loc=None, ip=None) -> Uint32:
    """This CTA's rank in its cluster."""
    return _asm("mov.u32 $0, %cluster_ctarank;", "=r", (), Uint32, effects=False, loc=loc, ip=ip)


@dsl_user_op
def cluster_sync(*, loc=None, ip=None) -> None:
    """Cluster barrier: every thread of the cluster arrives (release: its earlier shared and
    distributed shared accesses) and waits for all (acquire)."""
    _asm(
        "barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;",
        "",
        (),
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mapa_shared(addr, rank, *, loc=None, ip=None) -> Uint32:
    """The shared::cluster address of this CTA's shared ``addr`` in cluster CTA ``rank``."""
    return _asm(
        "mapa.shared::cluster.u32 $0, $1, $2;",
        "=r,r,r",
        (Uint32(addr), Uint32(rank)),
        Uint32,
        effects=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_cluster_u32(addr, *, loc=None, ip=None) -> Uint32:
    """The u32 at shared::cluster ``addr``."""
    return _asm("ld.shared::cluster.b32 $0, [$1];", "=r,r", (Uint32(addr),), Uint32, loc=loc, ip=ip)


@dsl_user_op
def st_cluster_u32_if(cond, addr, value, *, loc=None, ip=None) -> None:
    """If ``cond``: the u32 ``value`` at shared::cluster ``addr``."""
    _asm(
        "{ .reg .pred p; setp.ne.u32 p, $0, 0; @p st.shared::cluster.b32 [$1], $2; }",
        "r,r,r",
        (Uint32(cond), Uint32(addr), Uint32(value)),
        loc=loc,
        ip=ip,
    )
