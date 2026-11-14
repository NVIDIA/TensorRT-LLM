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
"""Inline-PTX primitives used by the routed-MoE kernel."""

from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Float32, Int32, Int64, T, Uint32, dsl_user_op


def _asm(res, operands, text, constraints, *, loc=None, ip=None):
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


@dsl_user_op
def f32_bits(x: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(llvm.bitcast(T.i32(), Float32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def bits_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.bitcast(T.f32(), Uint32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def redux_max_u32(x: Uint32, *, loc=None, ip=None) -> Uint32:
    """Warp-wide unsigned max (all 32 lanes must participate)."""
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            "redux.sync.max.u32 $0, $1, 0xffffffff;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


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
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bf16_round(x: Float32, *, loc=None, ip=None) -> Float32:
    """Round FP32 to BF16 (RN) and back to FP32."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b16 h;\ncvt.rn.bf16.f32 h, $1;\ncvt.f32.bf16 $0, h;\n}",
            "=f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bf16_bits(x: Float32, *, loc=None, ip=None) -> Uint32:
    """FP32 -> BF16 (RN) bits in the low half."""
    return Uint32(
        _asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b16 h;\ncvt.rn.bf16.f32 h, $1;\ncvt.u32.u16 $0, h;\n}",
            "=r,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def bf16x2_to_f32(packed: Uint32, upper: bool, *, loc=None, ip=None) -> Float32:
    """One BF16 lane of a packed pair -> FP32 (exact)."""
    text = (
        "{\n.reg .b32 t;\nand.b32 t, $1, 0xFFFF0000;\nmov.b32 $0, t;\n}"
        if upper
        else "{\n.reg .b32 t;\nshl.b32 t, $1, 16;\nmov.b32 $0, t;\n}"
    )
    return Float32(
        _asm(T.f32(), [Uint32(packed).ir_value(loc=loc, ip=ip)], text, "=f,r", loc=loc, ip=ip)
    )


@dsl_user_op
def tmem_ld_32x32b_x8(taddr: Int32, *, loc=None, ip=None):
    """tcgen05.ld 32x32b.x8: this thread's lane, 8 consecutive columns."""
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
def red_add_f32(addr: Int64, value: Float32, *, loc=None, ip=None):
    """Relaxed GPU-scope FP32 atomic add without a return value."""
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Float32(value).ir_value(loc=loc, ip=ip)],
        "red.relaxed.gpu.global.add.f32 [$0], $1;",
        "l,f",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_f32(addr: Int64, value: Float32, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Float32(value).ir_value(loc=loc, ip=ip)],
        "st.global.f32 [$0], $1;",
        "l,f",
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
def popc_u32(x: Uint32, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            "popc.b32 $0, $1;",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_nc_f32(addr: Int64, *, loc=None, ip=None) -> Float32:
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
def ld_nc_u16(addr: Int64, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "{\n.reg .b16 h;\nld.global.nc.u16 h, [$1];\ncvt.u32.u16 $0, h;\n}",
            "=r,l",
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
def fence_acq_rel_gpu(*, loc=None, ip=None):
    _asm(None, [], "fence.acq_rel.gpu;", "", loc=loc, ip=ip)


@dsl_user_op
def fence_proxy_async_shared(*, loc=None, ip=None):
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


@dsl_user_op
def globaltimer_u64(*, loc=None, ip=None) -> Int64:
    return Int64(_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", loc=loc, ip=ip))


@dsl_user_op
def clock64(*, loc=None, ip=None) -> Int64:
    return Int64(_asm(T.i64(), [], "mov.u64 $0, %clock64;", "=l", loc=loc, ip=ip))


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
def sts_u64(smem_addr: Int32, value: Int64, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(smem_addr).ir_value(loc=loc, ip=ip), Int64(value).ir_value(loc=loc, ip=ip)],
        "st.shared.u64 [$0], $1;",
        "r,l",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def lds_u64(smem_addr: Int32, *, loc=None, ip=None) -> Int64:
    return Int64(
        _asm(
            T.i64(),
            [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.u64 $0, [$1];",
            "=l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_e2m1x8(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Eight FP32 -> eight E2M1 (RN, satfinite); ``v0`` in bits 0..3."""
    ops = [Float32(v).ir_value(loc=loc, ip=ip) for v in (v1, v0, v3, v2, v5, v4, v7, v6)]
    return Uint32(
        _asm(
            T.i32(),
            ops,
            "{\n.reg .b8 b0, b1, b2, b3;\n"
            "cvt.rn.satfinite.e2m1x2.f32 b0, $1, $2;\n"
            "cvt.rn.satfinite.e2m1x2.f32 b1, $3, $4;\n"
            "cvt.rn.satfinite.e2m1x2.f32 b2, $5, $6;\n"
            "cvt.rn.satfinite.e2m1x2.f32 b3, $7, $8;\n"
            "mov.b32 $0, {b0, b1, b2, b3};\n}",
            "=r,f,f,f,f,f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_ll2(addr: Int64, d0: Uint32, d1: Uint32, flag: Uint32, *, loc=None, ip=None):
    """Two LL words ``{d0, flag}, {d1, flag}`` in one 16-byte store.

    LL protocol (as NCCL's LL): every 4-byte payload travels with the epoch
    of the launch that wrote it, so a reader that sees the expected epoch in
    an 8-byte half also sees that half's payload; no fence or flag is needed.
    GPU-scope relaxed (not volatile, which is system scope) so the stores
    and polls stay within the GPU's L2.
    """
    _asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(d0).ir_value(loc=loc, ip=ip),
            Uint32(flag).ir_value(loc=loc, ip=ip),
            Uint32(d1).ir_value(loc=loc, ip=ip),
        ],
        "st.relaxed.gpu.global.v4.u32 [$0], {$1, $2, $3, $2};",
        "l,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_ll2(addr: Int64, *, loc=None, ip=None):
    """16-byte poll of two LL words: ``[d0, flag0, d1, flag1]``."""
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.relaxed.gpu.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return [Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def prefetch_l2_bulk(addr: Int64, nbytes: Int32, *, loc=None, ip=None):
    """Bulk L2 prefetch of ``nbytes`` (multiple of 16) contiguous bytes."""
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int32(nbytes).ir_value(loc=loc, ip=ip)],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fence_proxy_async_global(*, loc=None, ip=None):
    """Order generic-proxy global accesses with async-proxy (bulk copy)
    accesses of this thread."""
    _asm(None, [], "fence.proxy.async.global;", "", loc=loc, ip=ip)


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
def st_ll1(addr: Int64, d0: Uint32, flag: Uint32, *, loc=None, ip=None):
    """One LL word ``{d0, flag}`` (8-byte GPU-scope relaxed store)."""
    _asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(d0).ir_value(loc=loc, ip=ip),
            Uint32(flag).ir_value(loc=loc, ip=ip),
        ],
        "st.relaxed.gpu.global.v2.u32 [$0], {$1, $2};",
        "l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def lds_v2_u32(smem_addr: Int32, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.u32 {$0, $1}, [$2];",
        "=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return [Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(2)]


@dsl_user_op
def sts_v2_u32(smem_addr: Int32, a: Uint32, b: Uint32, *, loc=None, ip=None):
    _asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Uint32(a).ir_value(loc=loc, ip=ip),
            Uint32(b).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v2.u32 [$0], {$1, $2};",
        "r,r,r",
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------
# Thread block cluster primitives (expert-cluster schedule)
# ---------------------------------------------------------------------------
@dsl_user_op
def cluster_ctarank(*, loc=None, ip=None) -> Int32:
    return Int32(_asm(T.i32(), [], "mov.u32 $0, %cluster_ctarank;", "=r", loc=loc, ip=ip))


@dsl_user_op
def mapa_u32(smem_addr: Int32, cta_rank: Int32, *, loc=None, ip=None) -> Int32:
    """Shared::cluster address of ``smem_addr`` in cluster CTA ``cta_rank``."""
    return Int32(
        _asm(
            T.i32(),
            [Int32(smem_addr).ir_value(loc=loc, ip=ip), Int32(cta_rank).ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_cluster_v4_u32(
    cluster_addr: Int32, a: Uint32, b: Uint32, c: Uint32, d: Uint32, *, loc=None, ip=None
):
    _asm(
        None,
        [Int32(cluster_addr).ir_value(loc=loc, ip=ip)]
        + [Uint32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c, d)],
        "st.shared::cluster.v4.u32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_cluster_u32(cluster_addr: Int32, value: Uint32, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(cluster_addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
        "st.shared::cluster.u32 [$0], $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fence_proxy_async_shared_cluster(*, loc=None, ip=None):
    """Order this thread's generic-proxy shared::cluster stores before later
    async-proxy (UMMA, tcgen05.cp) reads of the same bytes."""
    _asm(None, [], "fence.proxy.async.shared::cluster;", "", loc=loc, ip=ip)


@dsl_user_op
def mbar_arrive_remote(cluster_addr: Int32, *, loc=None, ip=None):
    """Release-arrive (cluster scope) on an mbarrier of a cluster CTA."""
    _asm(
        None,
        [Int32(cluster_addr).ir_value(loc=loc, ip=ip)],
        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [$0];",
        "r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mbar_try_wait_cluster(mbar: Int32, phase: Int32, *, loc=None, ip=None) -> Int32:
    """One acquire (cluster scope) try-wait on a local mbarrier phase; 1 when done."""
    return Int32(
        _asm(
            T.i32(),
            [Int32(mbar).ir_value(loc=loc, ip=ip), Int32(phase).ir_value(loc=loc, ip=ip)],
            "{\n.reg .pred p;\n"
            "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [$1], $2;\n"
            "selp.u32 $0, 1, 0, p;\n}",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_nc_v4_u32(addr: Int64, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return [Uint32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4)]


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
def fma_rn(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def pack_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """Two FP32 values -> one word of two BF16 (RN), ``lo`` in the low half."""
    return Uint32(
        _asm(
            T.i32(),
            [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def atom_add_relaxed_gpu_u32(addr: Int64, value: Uint32, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(value).ir_value(loc=loc, ip=ip)],
            "atom.relaxed.gpu.global.add.u32 $0, [$1], $2;",
            "=r,l,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def smid_u32(*, loc=None, ip=None) -> Int32:
    return Int32(_asm(T.i32(), [], "mov.u32 $0, %smid;", "=r", loc=loc, ip=ip))


@dsl_user_op
def bar_arrive(bar_id: int, num_threads: int, *, loc=None, ip=None):
    _asm(
        None,
        [Int32(bar_id).ir_value(loc=loc, ip=ip), Int32(num_threads).ir_value(loc=loc, ip=ip)],
        "bar.arrive $0, $1;",
        "r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def bulk_s2cluster(
    dst_cluster: Int32, src_cta: Int32, nbytes: Int32, mbar_cluster: Int32, *, loc=None, ip=None
):
    """cp.async.bulk shared::cta -> shared::cluster (a peer CTA), completing
    on the peer's mbarrier (tx bytes)."""
    _asm(
        None,
        [
            Int32(dst_cluster).ir_value(loc=loc, ip=ip),
            Int32(src_cta).ir_value(loc=loc, ip=ip),
            Int32(nbytes).ir_value(loc=loc, ip=ip),
            Int32(mbar_cluster).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,r,r,r",
        loc=loc,
        ip=ip,
    )
