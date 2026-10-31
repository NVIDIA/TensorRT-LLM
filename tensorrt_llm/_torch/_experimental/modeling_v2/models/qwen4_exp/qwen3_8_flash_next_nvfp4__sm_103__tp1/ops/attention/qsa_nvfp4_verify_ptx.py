# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inline-PTX pieces of `qsa_paged_sparse_gqa_nvfp4_verify`'s kernel beyond the ones the
per-row kernel (`qsa_paged_sparse_gqa_nvfp4_decode`) already has: the union scan's bit
operations, vector global loads and stores, the tcgen05 thread-sync fences, and tcgen05.mma issue with
precomputed descriptors.

`gemm_ts` issues every K step of a GEMM whose A operand sits in tensor memory from one
inline-PTX block, the shared-memory descriptor of B advanced by immediates; the encodings are
those of CUTLASS's `cute/arch/mma_sm100_desc.hpp`. `cute.gemm` rebuilds the descriptors per K
step on the uniform datapath (about 18 instructions a UMMA), which the verify kernel's single
MMA warp cannot hide.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass._mlir.dialects import nvvm as _nvvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_paged_sparse_gqa_nvfp4_decode import (
    _asm,
    _ir,
)

# ---------------------------------------------------------------------------------------
# scalar and memory helpers
# ---------------------------------------------------------------------------------------


@dsl_user_op
def popc(x, *, loc=None, ip=None):
    return cutlass.Int32(
        _asm(T.i32(), [_ir(x, cutlass.Int32, loc, ip)], "popc.b32 $0, $1;", "=r,r", loc, ip)
    )


@dsl_user_op
def ctz(x, *, loc=None, ip=None):
    """Trailing zeros of a nonzero word."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [_ir(x, cutlass.Int32, loc, ip)],
            "{ .reg .b32 t; brev.b32 t, $1; clz.b32 $0, t; }",
            "=r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def red_or_shared(saddr, v, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(saddr, cutlass.Int32, loc, ip), _ir(v, cutlass.Int32, loc, ip)],
        "{ red.shared.or.b32 [$1], $2; mov.u32 $0, 0; }",
        "=r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def stg4(gaddr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(gaddr, cutlass.Int64, loc, ip)]
        + [_ir(v, cutlass.Int32, loc, ip) for v in (a, b, c, d)],
        "{ st.global.v4.b32 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0; }",
        "=r,l,r,r,r,r",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def stg4f(gaddr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [_ir(gaddr, cutlass.Int64, loc, ip)]
        + [_ir(v, cutlass.Float32, loc, ip) for v in (a, b, c, d)],
        "{ st.global.v4.f32 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0; }",
        "=r,l,f,f,f,f",
        loc,
        ip,
        side=True,
    )


@dsl_user_op
def ldg2(gaddr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_ir(gaddr, cutlass.Int64, loc, ip)],
        "ld.global.v2.b32 {$0, $1}, [$2];",
        "=r,=r,l",
        loc,
        ip,
        side=True,
    )
    return [cutlass.Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(2)]


@dsl_user_op
def stg2(gaddr, a, b, *, loc=None, ip=None):
    _asm(
        T.i32(),
        [
            _ir(gaddr, cutlass.Int64, loc, ip),
            _ir(a, cutlass.Int32, loc, ip),
            _ir(b, cutlass.Int32, loc, ip),
        ],
        "{ st.global.v2.b32 [$1], {$2, $3}; mov.u32 $0, 0; }",
        "=r,l,r,r",
        loc,
        ip,
        side=True,
    )


def groups8(x):
    """Bit i of the result: any of bits 4i .. 4i + 3 of `x` (the eight four-token groups of a
    32-token bitmap word)."""
    y = x | (x >> 1)
    y = y | (y >> 2)
    z = y & 1
    for i in range(1, 8):
        z = z | (((y >> (4 * i)) & 1) << i)
    return z


def spread4(nib):
    """Bits 0..3 of `nib` to bits 0, 8, 16, 24."""
    return (nib & 1) | ((nib & 2) << 7) | ((nib & 4) << 14) | ((nib & 8) << 21)


@dsl_user_op
def tcgen05_fence_before(*, loc=None, ip=None):
    _nvvm.tcgen05_fence(_nvvm.Tcgen05FenceKind.BEFORE_THREAD_SYNC, loc=loc, ip=ip)


@dsl_user_op
def tcgen05_fence_after(*, loc=None, ip=None):
    _nvvm.tcgen05_fence(_nvvm.Tcgen05FenceKind.AFTER_THREAD_SYNC, loc=loc, ip=ip)


# ---------------------------------------------------------------------------------------
# tcgen05.mma issue with precomputed descriptors
# ---------------------------------------------------------------------------------------

#: Swizzle<B, 4, 3>: B -> the descriptor's layout type (none, 32B, 64B, 128B).
_SW_TYPES = {0: 0, 1: 6, 2: 4, 3: 2}
#: Layout type -> swizzle atom size along MN (16-byte units).
_SW_ATOM_MN = {0: 1, 6: 2, 4: 4, 2: 8}


def instr_desc(op) -> int:
    """The 32-bit instruction descriptor of an f16/bf16 x f16/bf16 -> f32 tcgen05 MMA op."""
    fmt = {cutlass.Float16: 0, cutlass.BFloat16: 1}
    a_major = 0 if op.a_major_mode == tcgen05.OperandMajorMode.K else 1
    b_major = 0 if op.b_major_mode == tcgen05.OperandMajorMode.K else 1
    m, n = op.shape_mnk[0], op.shape_mnk[1]
    assert op.acc_dtype is cutlass.Float32 and m in (64, 128) and n % 8 == 0 and 8 <= n <= 256
    d = 0
    d |= 1 << 4  # c_format f32
    d |= fmt[op.a_dtype] << 7
    d |= fmt[op.b_dtype] << 10
    d |= a_major << 15
    d |= b_major << 16
    d |= (n >> 3) << 17
    d |= (m >> 4) << 24
    return d


def smem_desc_base(layout_u128, swizzle, k_major: bool) -> int:
    """The 64-bit shared-memory descriptor without its start address, for an operand whose
    (MN, K) layout `layout_u128` is in 16-byte units."""
    assert swizzle.num_base == 4 and swizzle.num_shift == 3
    ltype = _SW_TYPES[swizzle.num_bits]
    atom_mn = _SW_ATOM_MN[ltype]
    if not k_major:
        canon = cute.logical_divide(layout_u128, (atom_mn, 8))
        s00 = canon.stride[0][0]
        s10 = canon.stride[1][0]
        assert (ltype == 0 or s00 == 1) and s10 == atom_mn, "not a canonical UMMA MN layout"
        s01, s11 = canon.stride[0][1], canon.stride[1][1]
        sbo, lbo = (s01, s11) if ltype == 0 else (s11, s01)
    else:
        canon = cute.logical_divide(layout_u128, (8, 2))
        s00 = canon.stride[0][0]
        s10 = canon.stride[1][0]
        assert s00 == atom_mn and (ltype == 0 or s10 == 1), "not a canonical UMMA K layout"
        sbo, lbo = canon.stride[0][1], s10
    d = 0
    d |= (lbo & 0x3FFF) << 16
    d |= (sbo & 0x3FFF) << 32
    d |= 1 << 46  # descriptor version
    d |= (ltype & 0x7) << 61
    return d


def gemm_ts(op, acc_tmem, a_tmem, tCrA, tCrB, sB, zero_init: bool):
    """acc (+)= A B over every K step: A in tensor memory at `a_tmem` (fragment `tCrA`), B the
    shared-memory operand `sB` (fragment `tCrB`, one stage). Issued by one elected thread of
    the calling warp. Plain Python run while the kernel is traced: the K loop unrolls into one
    PTX block."""
    idesc = instr_desc(op)
    k_major_b = op.b_major_mode == tcgen05.OperandMajorMode.K
    base_b = smem_desc_base(
        cute.recast_layout(128, op.b_dtype.width, sB.layout[0]),
        sB.iterator.type.swizzle_type,
        k_major_b,
    )
    b_lo = base_b & 0xFFFFFFFF
    b_hi = (base_b >> 32) & 0xFFFFFFFF
    a_layout = cute.recast_layout(32, tCrA.element_type.width, tCrA.layout)
    n_k = cute.size(tCrA.shape[2])
    off_a = [cute.crd2idx((0, 0, k), a_layout) for k in range(n_k)]
    off_b = [cute.crd2idx((0, 0, k), tCrB.layout) for k in range(n_k)]
    start_b = Int32(b_lo) | Int32((sB[None, None, 0].iterator.toint() & 0x3FFFF) >> 4)
    body = "{\n\t.reg .pred leader;\n\t.reg .b32 db_lo;\n\t.reg .b64 db;\n\telect.sync _|leader, -1;\n\t"
    for k in range(n_k):
        acc = "0" if (zero_init and k == 0) else "1"
        body += (
            f"add.u32 db_lo, $2, {hex(off_b[k])};\n\t"
            f"mov.b64 db, {{db_lo, {hex(b_hi)}}};\n\t"
            f"@leader tcgen05.mma.cta_group::1.kind::f16 [$0], [$1 + {hex(off_a[k])}], db, "
            f"{hex(idesc)}, {acc};\n\t"
        )
    body += "}\n"
    llvm.inline_asm(
        None,
        [
            Int32(cute.arch.make_warp_uniform(acc_tmem)).ir_value(),
            Int32(cute.arch.make_warp_uniform(a_tmem)).ir_value(),
            Int32(cute.arch.make_warp_uniform(start_b)).ir_value(),
        ],
        body,
        "r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
