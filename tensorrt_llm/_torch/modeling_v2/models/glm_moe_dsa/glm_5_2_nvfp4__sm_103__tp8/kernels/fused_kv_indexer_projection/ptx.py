# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inline-PTX helpers of the fused_kv_indexer_projection kernel (CuTe DSL).

fused_kv_indexer_projection owns every device helper its kernel uses. The arithmetic helpers spell
out the rounding of each operation (``add.rn``, ``cvt.rn``) so that the
kernel's result is defined by the instruction sequence alone.
"""

import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


def _asm(res, args, text, cons, side_effects=True, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        args,
        text,
        cons,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _struct(n, ty):
    return llvm.StructType.get_literal([ty] * n)


def _unpack(res, n, cls, ty, loc, ip):
    return tuple(cls(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(n))


# --------------------------------------------------------------------------
# Shared memory, mbarriers and bulk copies.
# --------------------------------------------------------------------------


@dsl_user_op
def saddr(p: cute.Pointer, *, loc=None, ip=None) -> Int32:
    """32-bit shared-window address of a shared-memory pointer."""
    return Int32(p.toint(loc=loc, ip=ip))


@dsl_user_op
def bulk_g2s(smem: Int32, gaddr: Int64, nbytes: Int32, mbar: Int32, *, loc=None, ip=None):
    """1-D bulk copy global -> shared that completes ``nbytes`` on ``mbar``."""
    _asm(
        None,
        [smem.ir_value(), gaddr.ir_value(), nbytes.ir_value(), mbar.ir_value()],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def tensor_g2s_3d(
    smem: Int32, tmap: Int64, c0: Int32, c1: Int32, c2: Int32, mbar: Int32, *, loc=None, ip=None
):
    """3-D tensor (TMA) copy global -> shared of the box at ``(c0, c1, c2)`` of
    the tensor map at global address ``tmap``; completes the box bytes on ``mbar``."""
    _asm(
        None,
        [
            smem.ir_value(),
            tmap.ir_value(),
            c0.ir_value(),
            c1.ir_value(),
            c2.ir_value(),
            mbar.ir_value(),
        ],
        "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[$0], [$1, {$2, $3, $4}], [$5];",
        "r,l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fence_tensormap_acquire(tmap: Int64, *, loc=None, ip=None):
    """Acquire a tensor map that the host wrote into global memory (the
    tensormap proxy is not acquired at kernel start)."""
    _asm(
        None,
        [tmap.ir_value()],
        "fence.proxy.tensormap::generic.acquire.sys [$0], 128;",
        "l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def prefetch_l2(gaddr: Int64, nbytes: Int32, *, loc=None, ip=None):
    _asm(
        None,
        [gaddr.ir_value(), nbytes.ir_value()],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbar_init(mbar: Int32, count: Int32, *, loc=None, ip=None):
    _asm(
        None,
        [mbar.ir_value(), count.ir_value()],
        "mbarrier.init.shared::cta.b64 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbar_fence_init(*, loc=None, ip=None):
    """Make the mbarrier initializations visible to the async proxy."""
    _asm(
        None,
        [],
        "fence.mbarrier_init.release.cluster;\n\tfence.proxy.async.shared::cta;",
        "",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbar_arrive_expect_tx(mbar: Int32, nbytes: Int32, *, loc=None, ip=None):
    _asm(
        None,
        [mbar.ir_value(), nbytes.ir_value()],
        "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbar_wait(mbar: Int32, phase: Int32, *, loc=None, ip=None):
    """Spin until the phase with parity ``phase`` of ``mbar`` has completed."""
    _asm(
        None,
        [mbar.ir_value(), phase.ir_value()],
        "{\n\t.reg .pred p;\n\tFUSED_KV_MBAR_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1;\n\t"
        "@!p bra FUSED_KV_MBAR_WAIT;\n\t}",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async16(smem: Int32, gaddr: Int64, *, loc=None, ip=None):
    """16-byte ``cp.async`` global -> shared (L2 only)."""
    _asm(
        None,
        [smem.ir_value(), gaddr.ir_value()],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_mbar_arrive(mbar: Int32, *, loc=None, ip=None):
    """Arrive on ``mbar`` once this thread's prior ``cp.async`` copies have
    completed (``.noinc``: the arrival counts against the initial count)."""
    _asm(
        None,
        [mbar.ir_value()],
        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [$0];",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def sync_warp(*, loc=None, ip=None):
    _asm(None, [], "bar.warp.sync 0xffffffff;", "", loc=loc, ip=ip)


@dsl_user_op
def st_shared_v4(smem: Int32, a: Int32, b: Int32, c: Int32, d: Int32, *, loc=None, ip=None):
    _asm(
        None,
        [smem.ir_value(), a.ir_value(), b.ir_value(), c.ir_value(), d.ir_value()],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_shared_f32(smem: Int32, v: Float32, *, loc=None, ip=None):
    _asm(None, [smem.ir_value(), v.ir_value()], "st.shared.f32 [$0], $1;", "r,f", loc=loc, ip=ip)


@dsl_user_op
def ld_shared_f32(smem: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(T.f32(), [smem.ir_value()], "ld.shared.f32 $0, [$1];", "=f,r", loc=loc, ip=ip)
    )


@dsl_user_op
def ldsm_x4(smem: Int32, *, loc=None, ip=None):
    out = _asm(
        _struct(4, T.i32()),
        [smem.ir_value()],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(out, 4, Int32, T.i32(), loc, ip)


@dsl_user_op
def ldsm_x2(smem: Int32, *, loc=None, ip=None):
    out = _asm(
        _struct(2, T.i32()),
        [smem.ir_value()],
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {$0, $1}, [$2];",
        "=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(out, 2, Int32, T.i32(), loc, ip)


# --------------------------------------------------------------------------
# Global memory.
# --------------------------------------------------------------------------


@dsl_user_op
def ld_cg_b32(gaddr: Int64, *, loc=None, ip=None) -> Int32:
    """4-byte global load cached in L2 only (no L1 line can be stale)."""
    return Int32(
        _asm(T.i32(), [gaddr.ir_value()], "ld.global.cg.b32 $0, [$1];", "=r,l", loc=loc, ip=ip)
    )


@dsl_user_op
def ld_ca_b32(gaddr: Int64, *, loc=None, ip=None) -> Int32:
    """4-byte global load cached in L1 and L2 (the default ``ld.global``)."""
    return Int32(
        _asm(T.i32(), [gaddr.ir_value()], "ld.global.ca.b32 $0, [$1];", "=r,l", loc=loc, ip=ip)
    )


@dsl_user_op
def ld_nc_b32(gaddr: Int64, *, loc=None, ip=None) -> Int32:
    """4-byte global load through the non-coherent read-only path (benchmarks only)."""
    return Int32(
        _asm(T.i32(), [gaddr.ir_value()], "ld.global.nc.b32 $0, [$1];", "=r,l", loc=loc, ip=ip)
    )


@dsl_user_op
def ld_cg_b32_if(gaddr: Int64, pred: Int32, *, loc=None, ip=None) -> Int32:
    """``ld_cg_b32`` if ``pred`` else 0 (no access when ``pred`` is 0)."""
    return Int32(
        _asm(
            T.i32(),
            [gaddr.ir_value(), pred.ir_value()],
            "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\tmov.b32 $0, 0;\n\t"
            "@p ld.global.cg.b32 $0, [$1];\n\t}",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_ca_b32_if(gaddr: Int64, pred: Int32, *, loc=None, ip=None) -> Int32:
    """``ld_ca_b32`` if ``pred`` else 0 (no access when ``pred`` is 0)."""
    return Int32(
        _asm(
            T.i32(),
            [gaddr.ir_value(), pred.ir_value()],
            "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $2, 0;\n\tmov.b32 $0, 0;\n\t"
            "@p ld.global.ca.b32 $0, [$1];\n\t}",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_global_b16(gaddr: Int64, v: Int32, *, loc=None, ip=None):
    _asm(
        None,
        [gaddr.ir_value(), v.ir_value()],
        "{\n\t.reg .b16 lo;\n\tcvt.u16.u32 lo, $1;\n\tst.global.b16 [$0], lo;\n\t}",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_global_b64(gaddr: Int64, v: Int64, *, loc=None, ip=None):
    _asm(None, [gaddr.ir_value(), v.ir_value()], "st.global.b64 [$0], $1;", "l,l", loc=loc, ip=ip)


@dsl_user_op
def globaltimer(*, loc=None, ip=None) -> Int64:
    return Int64(_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", loc=loc, ip=ip))


# --------------------------------------------------------------------------
# Integer selects.
# --------------------------------------------------------------------------


@dsl_user_op
def sel_i32(pred: Int32, a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [pred.ir_value(), a.ir_value(), b.ir_value()],
            "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $1, 0;\n\tselp.b32 $0, $2, $3, p;\n\t}",
            "=r,r,r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def sel_i64(pred: Int32, a: Int64, b: Int64, *, loc=None, ip=None) -> Int64:
    return Int64(
        _asm(
            T.i64(),
            [pred.ir_value(), a.ir_value(), b.ir_value()],
            "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $1, 0;\n\tselp.b64 $0, $2, $3, p;\n\t}",
            "=l,r,l,l",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def xor_i32(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(T.i32(), [a.ir_value(), b.ir_value()], "xor.b32 $0, $1, $2;", "=r,r,r", loc=loc, ip=ip)
    )


@dsl_user_op
def min_i32(a: Int32, b: Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [a.ir_value(), b.ir_value()],
            "min.s32 $0, $1, $2;",
            "=r,r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


# --------------------------------------------------------------------------
# Arithmetic.
# --------------------------------------------------------------------------


@dsl_user_op
def mma_bf16(
    a0: Int32,
    a1: Int32,
    a2: Int32,
    a3: Int32,
    b0: Int32,
    b1: Int32,
    c0: Float32,
    c1: Float32,
    c2: Float32,
    c3: Float32,
    *,
    loc=None,
    ip=None,
):
    """``mma.sync.m16n8k16`` BF16 x BF16 + FP32 (row.col), the instruction of
    the native fused-A GEMM."""
    out = _asm(
        _struct(4, T.f32()),
        [
            a0.ir_value(),
            a1.ir_value(),
            a2.ir_value(),
            a3.ir_value(),
            b0.ir_value(),
            b1.ir_value(),
            c0.ir_value(),
            c1.ir_value(),
            c2.ir_value(),
            c3.ir_value(),
        ],
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return _unpack(out, 4, Float32, T.f32(), loc, ip)


@dsl_user_op
def add_f32(a: Float32, b: Float32, ftz: bool = False, *, loc=None, ip=None) -> Float32:
    """IEEE round-to-nearest fp32 add (``.ftz`` flushes subnormal operands and results)."""
    op = "add.rn.ftz.f32" if ftz else "add.rn.f32"
    return Float32(
        _asm(
            T.f32(),
            [a.ir_value(), b.ir_value()],
            f"{op} $0, $1, $2;",
            "=f,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bf16_bits(x: Float32, *, loc=None, ip=None) -> Int32:
    """Round-to-nearest-even of one fp32 value to bf16 (in the low half)."""
    return Int32(
        _asm(
            T.i32(),
            [x.ir_value()],
            "{\n\t.reg .b16 h;\n\tcvt.rn.bf16.f32 h, $1;\n\tcvt.u32.u16 $0, h;\n\t}",
            "=r,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )
