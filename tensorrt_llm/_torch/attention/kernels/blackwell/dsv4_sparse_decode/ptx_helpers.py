# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inline PTX helpers: bulk and TMA gather copies, mbarriers, cluster and DSMEM operations, timers."""

import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op


def _asm(res_t, args, s, cons, loc, ip):
    return llvm.inline_asm(
        res_t,
        args,
        s,
        cons,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _v(x, loc, ip):
    return x.ir_value(loc=loc, ip=ip)


@dsl_user_op
def gtimer(*, loc=None, ip=None):
    return Int64(_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", loc, ip))


@dsl_user_op
def clk64(*, loc=None, ip=None):
    return Int64(_asm(T.i64(), [], "mov.u64 $0, %clock64;", "=l", loc, ip))


@dsl_user_op
def smid(*, loc=None, ip=None):
    return Int32(_asm(T.i32(), [], "mov.u32 $0, %smid;", "=r", loc, ip))


@dsl_user_op
def cluster_ctarank(*, loc=None, ip=None):
    return Int32(_asm(T.i32(), [], "mov.u32 $0, %cluster_ctarank;", "=r", loc, ip))


@dsl_user_op
def nanosleep(ns, *, loc=None, ip=None):
    _asm(None, [_v(ns, loc, ip)], "nanosleep.u32 $0;", "r", loc, ip)


@dsl_user_op
def fence_acq_rel_gpu(*, loc=None, ip=None):
    _asm(None, [], "fence.acq_rel.gpu;", "", loc, ip)


@dsl_user_op
def fence_proxy_async_smem(*, loc=None, ip=None):
    _asm(None, [], "fence.proxy.async.shared::cta;", "", loc, ip)


@dsl_user_op
def fence_proxy_async_global(*, loc=None, ip=None):
    _asm(None, [], "fence.proxy.async.global;", "", loc, ip)


@dsl_user_op
def fence_mbar_init_cluster(*, loc=None, ip=None):
    _asm(None, [], "fence.mbarrier_init.release.cluster;", "", loc, ip)


@dsl_user_op
def ld_acq_gpu_u32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(T.i32(), [_v(addr, loc, ip)], "ld.acquire.gpu.global.u32 $0, [$1];", "=r,l", loc, ip)
    )


@dsl_user_op
def ld_rlx_gpu_u32(addr, *, loc=None, ip=None):
    return Int32(
        _asm(T.i32(), [_v(addr, loc, ip)], "ld.relaxed.gpu.global.u32 $0, [$1];", "=r,l", loc, ip)
    )


@dsl_user_op
def st_rel_gpu_u32(addr, val, *, loc=None, ip=None):
    _asm(
        None,
        [_v(addr, loc, ip), _v(val, loc, ip)],
        "st.release.gpu.global.u32 [$0], $1;",
        "l,r",
        loc,
        ip,
    )


@dsl_user_op
def red_add_rel_gpu_u32(addr, val, *, loc=None, ip=None):
    _asm(
        None,
        [_v(addr, loc, ip), _v(val, loc, ip)],
        "red.release.gpu.global.add.u32 [$0], $1;",
        "l,r",
        loc,
        ip,
    )


@dsl_user_op
def st_g_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        None,
        [_v(addr, loc, ip), _v(a, loc, ip), _v(b, loc, ip), _v(c, loc, ip), _v(d, loc, ip)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc,
        ip,
    )


@dsl_user_op
def st_g_cg_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        None,
        [_v(addr, loc, ip), _v(a, loc, ip), _v(b, loc, ip), _v(c, loc, ip), _v(d, loc, ip)],
        "st.global.cg.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc,
        ip,
    )


@dsl_user_op
def ldg_cg_v4(gaddr, *, loc=None, ip=None):
    ret = _asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [_v(gaddr, loc, ip)],
        "ld.global.cg.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
    )
    return tuple((Int32(llvm.extractvalue(T.i32(), ret, [i], loc=loc, ip=ip)) for i in range(4)))


@dsl_user_op
def ldg_nc_v4(gaddr, *, loc=None, ip=None):
    ret = _asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [_v(gaddr, loc, ip)],
        "ld.global.nc.L1::no_allocate.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc,
        ip,
    )
    return tuple((Int32(llvm.extractvalue(T.i32(), ret, [i], loc=loc, ip=ip)) for i in range(4)))


@dsl_user_op
def prefetch_l2_line(gaddr, *, loc=None, ip=None):
    _asm(None, [_v(gaddr, loc, ip)], "prefetch.global.L2 [$0];", "l", loc, ip)


@dsl_user_op
def bulk_prefetch_l2(gaddr, nbytes, *, loc=None, ip=None):
    _asm(
        None,
        [_v(gaddr, loc, ip), _v(nbytes, loc, ip)],
        "cp.async.bulk.prefetch.L2.global [$0], $1;",
        "l,r",
        loc,
        ip,
    )


@dsl_user_op
def lds_u32(saddr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [_v(saddr, loc, ip)], "ld.shared.u32 $0, [$1];", "=r,r", loc, ip))


@dsl_user_op
def lds_v4(saddr, *, loc=None, ip=None):
    ret = _asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [_v(saddr, loc, ip)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc,
        ip,
    )
    return tuple((Int32(llvm.extractvalue(T.i32(), ret, [i], loc=loc, ip=ip)) for i in range(4)))


@dsl_user_op
def sts_u32(saddr, val, *, loc=None, ip=None):
    _asm(None, [_v(saddr, loc, ip), _v(val, loc, ip)], "st.shared.u32 [$0], $1;", "r,r", loc, ip)


@dsl_user_op
def sts_v4(saddr, a, b, c, d, *, loc=None, ip=None):
    _asm(
        None,
        [_v(saddr, loc, ip), _v(a, loc, ip), _v(b, loc, ip), _v(c, loc, ip), _v(d, loc, ip)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        loc,
        ip,
    )


@dsl_user_op
def mapa_u32(saddr, rank, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_v(saddr, loc, ip), _v(rank, loc, ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def st_async_v4(caddr, cmbar, a, b, c, d, *, loc=None, ip=None):
    _asm(
        None,
        [
            _v(caddr, loc, ip),
            _v(cmbar, loc, ip),
            _v(a, loc, ip),
            _v(b, loc, ip),
            _v(c, loc, ip),
            _v(d, loc, ip),
        ],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$2, $3, $4, $5}, [$1];",
        "r,r,r,r,r,r",
        loc,
        ip,
    )


@dsl_user_op
def st_async_v2(caddr, cmbar, a, b, *, loc=None, ip=None):
    _asm(
        None,
        [_v(caddr, loc, ip), _v(cmbar, loc, ip), _v(a, loc, ip), _v(b, loc, ip)],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [$0], {$2, $3}, [$1];",
        "r,r,r,r",
        loc,
        ip,
    )


@dsl_user_op
def mbar_init(saddr, count, *, loc=None, ip=None):
    _asm(
        None,
        [_v(saddr, loc, ip), _v(count, loc, ip)],
        "mbarrier.init.shared::cta.b64 [$0], $1;",
        "r,r",
        loc,
        ip,
    )


@dsl_user_op
def mbar_arrive(saddr, *, loc=None, ip=None):
    _asm(
        None,
        [_v(saddr, loc, ip)],
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [$0];",
        "r",
        loc,
        ip,
    )


@dsl_user_op
def mbar_arrive_expect_tx(saddr, tx, *, loc=None, ip=None):
    _asm(
        None,
        [_v(saddr, loc, ip), _v(tx, loc, ip)],
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [$0], $1;",
        "r,r",
        loc,
        ip,
    )


@dsl_user_op
def mbar_expect_tx(saddr, tx, *, loc=None, ip=None):
    _asm(
        None,
        [_v(saddr, loc, ip), _v(tx, loc, ip)],
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [$0], $1;",
        "r,r",
        loc,
        ip,
    )


@dsl_user_op
def mbar_arrive_remote(caddr, *, loc=None, ip=None):
    _asm(
        None,
        [_v(caddr, loc, ip)],
        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [$0];",
        "r",
        loc,
        ip,
    )


@dsl_user_op
def mbar_try_wait(saddr, parity, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_v(saddr, loc, ip), _v(parity, loc, ip)],
            "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2;\n\tselp.u32 $0, 1, 0, p;\n}",
            "=r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def mbar_try_wait_cluster(saddr, parity, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_v(saddr, loc, ip), _v(parity, loc, ip)],
            "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [$1], $2;\n\tselp.u32 "
            "$0, 1, 0, p;\n}",
            "=r,r,r",
            loc,
            ip,
        )
    )


@dsl_user_op
def mbar_test_wait(saddr, parity, *, loc=None, ip=None):
    return Int32(
        _asm(
            T.i32(),
            [_v(saddr, loc, ip), _v(parity, loc, ip)],
            "{\n\t.reg .pred p;\n\tmbarrier.test_wait.parity.acquire.cluster.shared::cta.b64 p, [$1], $2;\n\tselp.u32 "
            "$0, 1, 0, p;\n}",
            "=r,r,r",
            loc,
            ip,
        )
    )


@cute.jit
def spin(saddr, parity):
    par = Int32(parity)
    ok = mbar_test_wait(Int32(saddr), par)
    while ok == Int32(0):
        ok = mbar_test_wait(Int32(saddr), par)


@cute.jit
def wait(saddr, parity):
    par = Int32(parity)
    ok = mbar_try_wait(Int32(saddr), par)
    while ok == Int32(0):
        ok = mbar_try_wait(Int32(saddr), par)


@cute.jit
def wait_cl(saddr, parity):
    par = Int32(parity)
    ok = mbar_try_wait_cluster(Int32(saddr), par)
    while ok == Int32(0):
        ok = mbar_try_wait_cluster(Int32(saddr), par)


@dsl_user_op
def cluster_arrive(*, loc=None, ip=None):
    _asm(None, [], "barrier.cluster.arrive.release.aligned;", "", loc, ip)


@dsl_user_op
def cluster_arrive_relaxed(*, loc=None, ip=None):
    _asm(None, [], "barrier.cluster.arrive.relaxed.aligned;", "", loc, ip)


@dsl_user_op
def cluster_wait(*, loc=None, ip=None):
    _asm(None, [], "barrier.cluster.wait.acquire.aligned;", "", loc, ip)


@cute.jit
def cluster_sync():
    cluster_arrive()
    cluster_wait()


@dsl_user_op
def cp_async16(saddr, gaddr, *, loc=None, ip=None):
    _asm(
        None,
        [_v(saddr, loc, ip), _v(gaddr, loc, ip)],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        loc,
        ip,
    )


@dsl_user_op
def cp_async_mbar_arrive_noinc(saddr, *, loc=None, ip=None):
    _asm(
        None,
        [_v(saddr, loc, ip)],
        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [$0];",
        "r",
        loc,
        ip,
    )


@dsl_user_op
def bulk_g2s(saddr, gaddr, nbytes, smbar, *, loc=None, ip=None):
    _asm(
        None,
        [_v(saddr, loc, ip), _v(gaddr, loc, ip), _v(nbytes, loc, ip), _v(smbar, loc, ip)],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r",
        loc,
        ip,
    )


@dsl_user_op
def bulk_g2s_mc(saddr, gaddr, nbytes, smbar, mask, *, loc=None, ip=None):
    _asm(
        None,
        [
            _v(saddr, loc, ip),
            _v(gaddr, loc, ip),
            _v(nbytes, loc, ip),
            _v(smbar, loc, ip),
            _v(mask, loc, ip),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [$0], [$1], $2, [$3], "
        "$4;",
        "r,l,r,r,h",
        loc,
        ip,
    )


@dsl_user_op
def bulk_s2c(caddr, saddr, nbytes, cmbar, *, loc=None, ip=None):
    _asm(
        None,
        [_v(caddr, loc, ip), _v(saddr, loc, ip), _v(nbytes, loc, ip), _v(cmbar, loc, ip)],
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,r,r,r",
        loc,
        ip,
    )


@dsl_user_op
def bulk_s2g(gaddr, saddr, nbytes, *, loc=None, ip=None):
    _asm(
        None,
        [_v(gaddr, loc, ip), _v(saddr, loc, ip), _v(nbytes, loc, ip)],
        "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;",
        "l,r,r",
        loc,
        ip,
    )


@dsl_user_op
def bulk_commit(*, loc=None, ip=None):
    _asm(None, [], "cp.async.bulk.commit_group;", "", loc, ip)


@dsl_user_op
def bulk_wait0(*, loc=None, ip=None):
    _asm(None, [], "cp.async.bulk.wait_group 0;", "", loc, ip)


@dsl_user_op
def prefetch_tensormap(tmap_gaddr, *, loc=None, ip=None):
    _asm(None, [_v(tmap_gaddr, loc, ip)], "prefetch.tensormap [$0];", "l", loc, ip)


@dsl_user_op
def tma_g4(saddr, tmap, col, r0, r1, r2, r3, smbar, *, loc=None, ip=None):
    _asm(
        None,
        [
            _v(saddr, loc, ip),
            _v(tmap, loc, ip),
            _v(col, loc, ip),
            _v(r0, loc, ip),
            _v(r1, loc, ip),
            _v(r2, loc, ip),
            _v(r3, loc, ip),
            _v(smbar, loc, ip),
        ],
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes [$0], [$1, {$2, $3, "
        "$4, $5, $6}], [$7];",
        "r,l,r,r,r,r,r,r",
        loc,
        ip,
    )


@dsl_user_op
def tma_g4_mc(saddr, tmap, col, r0, r1, r2, r3, smbar, mask, *, loc=None, ip=None):
    _asm(
        None,
        [
            _v(saddr, loc, ip),
            _v(tmap, loc, ip),
            _v(col, loc, ip),
            _v(r0, loc, ip),
            _v(r1, loc, ip),
            _v(r2, loc, ip),
            _v(r3, loc, ip),
            _v(smbar, loc, ip),
            _v(mask, loc, ip),
        ],
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes.multicast::cluster "
        "[$0], [$1, {$2, $3, $4, $5, $6}], [$7], $8;",
        "r,l,r,r,r,r,r,r,h",
        loc,
        ip,
    )


@dsl_user_op
def tma_2d(saddr, tmap, c0, c1, smbar, *, loc=None, ip=None):
    _asm(
        None,
        [
            _v(saddr, loc, ip),
            _v(tmap, loc, ip),
            _v(c0, loc, ip),
            _v(c1, loc, ip),
            _v(smbar, loc, ip),
        ],
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [$0], [$1, {$2, $3}], [$4];",
        "r,l,r,r,r",
        loc,
        ip,
    )


@dsl_user_op
def tma_2d_mc(saddr, tmap, c0, c1, smbar, mask, *, loc=None, ip=None):
    _asm(
        None,
        [
            _v(saddr, loc, ip),
            _v(tmap, loc, ip),
            _v(c0, loc, ip),
            _v(c1, loc, ip),
            _v(smbar, loc, ip),
            _v(mask, loc, ip),
        ],
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster [$0], "
        "[$1, {$2, $3}], [$4], $5;",
        "r,l,r,r,r,h",
        loc,
        ip,
    )


@dsl_user_op
def tma_prefetch_g4(tmap, col, r0, r1, r2, r3, *, loc=None, ip=None):
    _asm(
        None,
        [
            _v(tmap, loc, ip),
            _v(col, loc, ip),
            _v(r0, loc, ip),
            _v(r1, loc, ip),
            _v(r2, loc, ip),
            _v(r3, loc, ip),
        ],
        "cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4 [$0, {$1, $2, $3, $4, $5}];",
        "l,r,r,r,r,r",
        loc,
        ip,
    )


@dsl_user_op
def tma_store_2d(tmap, c0, c1, saddr, *, loc=None, ip=None):
    _asm(
        None,
        [
            _v(Int64(tmap), loc, ip),
            _v(Int32(c0), loc, ip),
            _v(Int32(c1), loc, ip),
            _v(Int32(saddr), loc, ip),
        ],
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group [$0, {$1, $2}], [$3];",
        "l,r,r,r",
        loc,
        ip,
    )


@dsl_user_op
def ld_acq_gpu_u64(addr, *, loc=None, ip=None):
    return Int64(
        _asm(T.i64(), [_v(addr, loc, ip)], "ld.acquire.gpu.global.u64 $0, [$1];", "=l,l", loc, ip)
    )


@dsl_user_op
def red_add_rel_gpu_u64(addr, val, *, loc=None, ip=None):
    _asm(
        None,
        [_v(addr, loc, ip), _v(val, loc, ip)],
        "red.release.gpu.global.add.u64 [$0], $1;",
        "l,l",
        loc,
        ip,
    )


@dsl_user_op
def red_add_gpu_u32(addr, val, *, loc=None, ip=None):
    _asm(
        None,
        [_v(addr, loc, ip), _v(val, loc, ip)],
        "red.relaxed.gpu.global.add.u32 [$0], $1;",
        "l,r",
        loc,
        ip,
    )
