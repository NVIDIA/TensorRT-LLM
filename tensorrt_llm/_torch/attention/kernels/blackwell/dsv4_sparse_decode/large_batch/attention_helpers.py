# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA tile loads, MMA and softmax helpers shared by the two kernels."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cutlass_dsl import T as _T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

HEADS = 64
LATENT = 512
WINDOW = 128
NROWS = 640
TILE_N = 128
NTILES = NROWS // TILE_N
SLICE = 128
NSLICE = 4
Q_SLICE_BYTES = HEADS * SLICE
SLOT_BYTES = TILE_N * SLICE
LOG2_E = 1.4426950408889634
LOG2_448 = 8.8073549
NEG_FLT_MAX = -3.4028235e38
PANEL1 = 16 << 16
TM_S = 128
TM_O = 256
E4M3 = cutlass.Float8E4M3FN
BF16 = cutlass.BFloat16


@cute.jit
def _sleep_us(us: cutlass.Constexpr):
    t0 = Int64(cute.arch.globaltimer())
    t1 = Int64(cute.arch.globaltimer())
    while t1 - t0 < Int64(us * 1000):
        _nsleep(Int32(1000))
        t1 = Int64(cute.arch.globaltimer())


@cute.jit
def _pace_us(us: cutlass.Constexpr, pace_ptr, pace_lim):
    if pace_lim >= Int32(0):
        t0 = Int64(cute.arch.globaltimer())
        t1 = t0
        pend = cute.arch.load(pace_ptr, Int32, sem="acquire", scope="cta")
        while (t1 - t0 < Int64(us * 1000)) & (pend <= pace_lim):
            _nsleep(Int32(500))
            t1 = Int64(cute.arch.globaltimer())
            pend = cute.arch.load(pace_ptr, Int32, sem="acquire", scope="cta")


@dsl_user_op
def _nsleep(ns, *, loc=None, ip=None):
    _llvm.inline_asm(
        None,
        [Int32(ns).ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _wait(mbar, phase):
    while not prims.mbarrier_try_wait_parity(mbar, phase):
        pass


@cute.jit
def _wait_mma(bmma, ph, wl, bar_id: cutlass.Constexpr):
    if wl == Int32(0):
        _wait(bmma, ph >> Int32(8) & Int32(1))
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.barrier(barrier_id=bar_id, number_of_threads=128)
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    return ph ^ Int32(1 << 8)


@cute.jit
def _tma_gather4(smem_dst, tma_desc, mbar, col, r0, r1, r2, r3, pol=None) -> None:
    bar = prims.cvta_to(mbar, prims.CvtaSpace.SHARED)
    if cutlass.const_expr(pol is None):
        prims.inline_ptx_hl(
            "{\n\t.reg .pred leader;\n\telect.sync _|leader, -1;\n\t"
            "@leader cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes "
            "[{$r0}], [{$r1}, {{$r2}, {$r3}, {$r4}, {$r5}, {$r6}}], [{$r7}];\n\t}",
            read_only_args=[smem_dst, tma_desc, col, r0, r1, r2, r3, bar],
        )
    else:
        prims.inline_ptx_hl(
            "{\n\t.reg .pred leader;\n\telect.sync _|leader, -1;\n\t"
            "@leader cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes"
            ".L2::cache_hint "
            "[{$r0}], [{$r1}, {{$r2}, {$r3}, {$r4}, {$r5}, {$r6}}], [{$r7}], {$r8};\n\t}",
            read_only_args=[smem_dst, tma_desc, col, r0, r1, r2, r3, bar, pol],
        )


@dsl_user_op
def l2_policy(kind: int, *, loc=None, ip=None):
    asm = {
        1: "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;",
        2: "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;",
        3: "createpolicy.fractional.L2::evict_normal.b64 $0, 1.0;",
    }[kind]
    return Int64(
        _llvm.inline_asm(
            _T.i64(),
            [],
            asm,
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _pack4_e4m3(v0: Float32, v1: Float32, v2: Float32, v3: Float32):
    return cute.arch.inline_ptx(
        "{\n  .reg .b16 lo;\n  .reg .b16 hi;\n  "
        "cvt.rn.satfinite.e4m3x2.f32 lo, {$r1}, {$r0};\n  cvt.rn.satfinite.e4m3x2.f32 hi, {$r3}, {$r2};\n  "
        "mov.b32 {$w0}, {lo, hi};\n}",
        write_only_types=[Int32],
        read_only_args=[v0, v1, v2, v3],
    )


@cute.jit
def _tmem_st_p8(tmem_addr, regs) -> None:
    body = ", ".join(("{$r%d}" % (i + 1) for i in range(8)))
    cute.arch.inline_ptx(
        f"tcgen05.st.sync.aligned.16x32bx2.x8.b32 [{{$r0}}], 16, {{ {body} }};",
        read_only_args=[tmem_addr] + [regs[i] for i in range(8)],
    )


@cute.jit
def _shfl_xor_f32(x: Float32, dist: cutlass.Constexpr[int]) -> Float32:
    return Float32(
        prims.shfl_sync(
            thread_mask=4294967295, val=x, offset=dist, mask_and_clamp=31, kind=prims.Shfl.BFLY
        )
    )


@cute.jit
def _shfl_idx_i32(x: Int32, src) -> Int32:
    return Int32(
        prims.shfl_sync(
            thread_mask=4294967295, val=x, offset=src, mask_and_clamp=31, kind=prims.Shfl.IDX
        )
    )


@cute.jit
def _pack_bf16x2(v0: Float32, v1: Float32):
    return cutlass.Vector.from_elements((v0, v1), Float32).to(BF16).bitcast(Int32)[0]


@dsl_user_op
def _mul_rn(a, b, *, loc=None, ip=None):
    return Float32(
        _llvm.inline_asm(
            _T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fma_rn(a, b, c, *, loc=None, ip=None):
    return Float32(
        _llvm.inline_asm(
            _T.f32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                Float32(c).ir_value(loc=loc, ip=ip),
            ],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _final_norm(
    final_max: Float32, reduced_sum: Float32, sink: Float32, scale_log2: Float32
) -> Float32:
    t = _mul_rn(final_max, scale_log2)
    delta = _fma_rn(sink, Float32(LOG2_E), -t)
    attention_rescale = cute.math.exp2(-cute.math.max(delta, Float32(0)), approx=True)
    sink_mass = cute.math.exp2(cute.math.min(delta, Float32(0)), approx=True)
    sink_mass *= Float32(448.0)
    return (
        Float32(1.0)
        * attention_rescale
        * cute.math.rcp(reduced_sum * attention_rescale + sink_mass, approx=True)
    )


@cute.jit
def _rescale_o(tmem, lane_base, scale: Float32):
    for v in cutlass.range_constexpr(NSLICE):
        o_addr = tmem + lane_base + Int32(TM_O + v % 2 * 128 + v // 2 * PANEL1)
        for h in cutlass.range_constexpr(2):
            ptr = prims.make_tmem_ptr(o_addr + Int32(32 * h), Float32)
            o = prims.tcgen05_ld(
                prims.Tcgen05LdStShape.SHAPE_16X32BX2, ptr, num=32, offset=Int64(64)
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            vals = []
            for r in cutlass.range_constexpr(0, 32, 2):
                pr = cute.arch.mul_packed_f32x2(
                    (Float32(o[r]), Float32(o[r + 1])), (scale, scale), ftz=False, rnd="rn"
                )
                vals.append(pr[0])
                vals.append(pr[1])
            prims.tcgen05_st(
                prims.Tcgen05LdStShape.SHAPE_16X32BX2,
                ptr,
                cutlass.Vector.from_elements(tuple(vals), Float32),
                offset=Int64(64),
            )


@cute.jit
def _issue_kv_slice(slot_ptr, mbar, tma_swa, tma_cmp, use_swa, gidx_t, c, wl, lane, pol=None):
    if wl == Int32(0):
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(mbar, SLOT_BYTES)
    dst0 = slot_ptr + wl * Int32(32 * SLICE)
    for q in cutlass.range(8, unroll=1):
        r0 = _shfl_idx_i32(gidx_t, Int32(4 * q))
        r1 = _shfl_idx_i32(gidx_t, Int32(4 * q + 1))
        r2 = _shfl_idx_i32(gidx_t, Int32(4 * q + 2))
        r3 = _shfl_idx_i32(gidx_t, Int32(4 * q + 3))
        if use_swa:
            _tma_gather4(
                dst0 + Int32(4 * q * SLICE),
                tma_swa.get_ptr(),
                mbar,
                c * Int32(SLICE),
                r0,
                r1,
                r2,
                r3,
                pol,
            )
        else:
            _tma_gather4(
                dst0 + Int32(4 * q * SLICE),
                tma_cmp.get_ptr(),
                mbar,
                c * Int32(SLICE),
                r0,
                r1,
                r2,
                r3,
                pol,
            )


@cute.jit
def issue_row_loads(
    tid,
    b,
    s_gv,
    deep_base,
    kv_slot0,
    kv_slot7,
    deep_q,
    q_slot0,
    bq,
    bkv,
    gfree,
    tma_q,
    tma_swa,
    tma_cmp,
    bar_id: cutlass.Constexpr,
    jb=None,
    je=None,
):
    wl = tid >> Int32(5)
    lane = tid & Int32(31)
    j_b = Int32(0)
    j_e = Int32(NTILES)
    if cutlass.const_expr(jb is not None):
        j_b = jb
        j_e = je
    if j_b == Int32(0):
        if tid == Int32(0):
            for c in cutlass.range_constexpr(NSLICE):
                prims.mbarrier_arrive_expect_tx(bq + c, Q_SLICE_BYTES)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    _q_slot(deep_q, q_slot0, Int32(c)),
                    tma_q.get_ptr(),
                    [Int32(c * SLICE), Int32(0), b],
                    bq + c,
                )
    for j in cutlass.range(j_b, j_e, unroll=1):
        g4 = (j & Int32(1)) * Int32(4)
        if j >= Int32(2):
            _wait(gfree + (j & Int32(1)), j - Int32(2) >> Int32(1) & Int32(1))
        grow = _grow(Int32(s_gv[j * Int32(TILE_N) + tid]))
        for c in cutlass.range_constexpr(NSLICE):
            _issue_kv_slice(
                _kv_slot(deep_base, kv_slot0, g4 + Int32(c), kv_slot7),
                bkv + g4 + Int32(c),
                tma_swa,
                tma_cmp,
                j == Int32(0),
                grow,
                Int32(c),
                wl,
                lane,
            )


OOB_ROW = 2147483647


@cute.jit
def _grow(g):
    r = g
    if g < Int32(0):
        r = Int32(OOB_ROW)
    return r


@cute.jit
def _pick(j, g0, g1, g2, g3, g4):
    g = g0
    if j == Int32(1):
        g = g1
    elif j == Int32(2):
        g = g2
    elif j == Int32(3):
        g = g3
    elif j == Int32(4):
        g = g4
    return g


@cute.jit
def _kv_slot(deep_base, kv_slot0, k, kv_slot7):
    a = deep_base + k * Int32(SLOT_BYTES)
    if k >= Int32(6):
        a = kv_slot0
    if k >= Int32(7):
        a = kv_slot7
    return a


@cute.jit
def deep_ok(hi_ptr):
    a = cute.arch.load(hi_ptr + Int32(11), Int32, sem="acquire", scope="cta")
    b = cute.arch.load(hi_ptr + Int32(3), Int32, sem="acquire", scope="cta")
    cute.arch.fence_acq_rel_cta()
    r = Int32(0)
    if (a != Int32(0)) & (b != Int32(0)):
        r = Int32(1)
    return r


@cute.jit
def _q_slot(deep_q, q_slot0, c):
    a = deep_q + c * Int32(Q_SLICE_BYTES)
    if c == Int32(3):
        a = q_slot0
    return a


@cute.jit
def att_row(
    rtid,
    b,
    gv0,
    gv1,
    gv2,
    gv3,
    gv4,
    tmem,
    deep0,
    q_slot0,
    kv_slot0,
    deep_base,
    deep_q,
    bq,
    bkv,
    bmma,
    ph,
    tma_q,
    tma_swa,
    tma_cmp,
    sink_ptr,
    out_addr: Int64,
    scale_log2: Float32,
    s_msk,
    dflag_ptr,
    bar_id: cutlass.Constexpr,
    skip: cutlass.Constexpr = 0,
    allow_deep: cutlass.Constexpr = True,
    prof: cutlass.Constexpr = False,
    pst=None,
    pbase=None,
    kv_slot7=None,
    ext=None,
    gfree=None,
    ahint: cutlass.Constexpr = 0,
    phased: cutlass.Constexpr = False,
    j0=None,
    j1=None,
    st_in=None,
    apace: cutlass.Constexpr = 0,
    kv_slotB=None,
    two=None,
    pace_lim=None,
    tbase=None,
    qfree=None,
):
    TW = two is not None
    tw = Int32(0)
    tb = Int32(0)
    if cutlass.const_expr(tbase is not None):
        tb = tbase
    if cutlass.const_expr(TW):
        tw = two
    wl = rtid >> Int32(5)
    lane = rtid & Int32(31)
    head = wl * Int32(16) + (lane & Int32(15))
    half = lane >> Int32(4)
    lane_base = wl * Int32(32) << Int32(16)
    s_addr = tmem + Int32(TM_S)
    sink = Float32((sink_ptr + head).load())
    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=Float32, a_dtype=E4M3, b_dtype=E4M3, n_dim=TILE_N, m_dim=HEADS
    )
    idesc_pv = prims.Tcgen05InstrDesc.build(
        c_dtype=Float32, a_dtype=E4M3, b_dtype=E4M3, a_major=0, b_major=1, n_dim=SLICE, m_dim=HEADS
    )
    pol_keep = None
    pol_drop = None
    if cutlass.const_expr(ahint == 1):
        pol_keep = l2_policy(1)
        pol_drop = l2_policy(2)
    m_run = Float32(NEG_FLT_MAX)
    l_run = Float32(0.0)
    dp = deep0
    qres = Int32(0)
    pre = Int32(0)
    jb = Int32(0)
    je = Int32(NTILES)
    if cutlass.const_expr(phased):
        m_run = Float32(st_in[0])
        l_run = Float32(st_in[1])
        qres = Int32(st_in[2])
        pre = Int32(st_in[3])
        dp = Int32(st_in[4])
        jb = j0
        je = j1
    for j in cutlass.range(jb, je, unroll=1):
        gsel = _pick(j, gv0, gv1, gv2, gv3, gv4)
        valid = gsel >= Int32(0)
        grow = _grow(gsel)
        vm = Int32(cute.arch.vote_ballot_sync(valid))
        if lane == Int32(0):
            s_msk[wl] = vm
        if rtid == Int32(0):
            dnow = dp
            if cutlass.const_expr(allow_deep):
                if dp == Int32(0):
                    dnow = deep_ok(dflag_ptr)
            if cutlass.const_expr(ext is not None):
                if ext != Int32(0):
                    dnow = Int32(1)
            s_msk[4] = dnow
        cute.arch.barrier(barrier_id=bar_id, number_of_threads=128)
        if cutlass.const_expr(apace > 0):
            if j > jb:
                if s_msk[4] == Int32(0):
                    if rtid == Int32(0):
                        if cutlass.const_expr(pace_lim is None):
                            _sleep_us(apace)
                        else:
                            _pace_us(apace, dflag_ptr, pace_lim)
                    cute.arch.barrier(barrier_id=bar_id, number_of_threads=128)
        if cutlass.const_expr(prof):
            if rtid == Int32(0):
                pst[pbase + Int32(4) * j] = Int64(cute.arch.globaltimer())
        dp = Int32(0)
        if s_msk[4] != Int32(0):
            dp = Int32(1)
        use_swa = j == Int32(0)
        grp4 = (tb + j & Int32(1)) * Int32(4)
        if dp != Int32(0):
            ldq = qres == Int32(0)
            iss = Boolean(True)
            if cutlass.const_expr(ext is not None):
                if ext != Int32(0):
                    iss = Boolean(False)
            if cutlass.const_expr(skip & 16 == 0):
                if ldq & iss:
                    if rtid == Int32(0):
                        for c in cutlass.range_constexpr(NSLICE):
                            prims.mbarrier_arrive_expect_tx(bq + c, Q_SLICE_BYTES)
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                _q_slot(deep_q, q_slot0, Int32(c)),
                                tma_q.get_ptr(),
                                [Int32(c * SLICE), Int32(0), b],
                                bq + c,
                            )
                if (pre == Int32(0)) & iss:
                    for c in cutlass.range_constexpr(NSLICE):
                        _issue_kv_slice(
                            _kv_slot(deep_base, kv_slot0, grp4 + Int32(c), kv_slot7),
                            bkv + grp4 + Int32(c),
                            tma_swa,
                            tma_cmp,
                            use_swa,
                            grow,
                            Int32(c),
                            wl,
                            lane,
                            pol_drop,
                        )
                if (j + Int32(1) < Int32(NTILES)) & iss:
                    gnx = _grow(_pick(j + Int32(1), gv0, gv1, gv2, gv3, gv4))
                    ng4 = Int32(4) - grp4
                    for c in cutlass.range_constexpr(NSLICE):
                        _issue_kv_slice(
                            _kv_slot(deep_base, kv_slot0, ng4 + Int32(c), kv_slot7),
                            bkv + ng4 + Int32(c),
                            tma_swa,
                            tma_cmp,
                            Boolean(False),
                            gnx,
                            Int32(c),
                            wl,
                            lane,
                            pol_drop,
                        )
            if wl == Int32(0):
                for c in cutlass.range(NSLICE, unroll=1):
                    if cutlass.const_expr(skip & 16 == 0):
                        if ldq:
                            _wait(bq + c, ph >> c & Int32(1))
                        _wait(bkv + grp4 + c, ph >> Int32(16) + grp4 + c & Int32(1))
                    qdesc = prims.Tcgen05SmemDesc.build(
                        _q_slot(deep_q, q_slot0, c),
                        leading_byte_offset=Q_SLICE_BYTES,
                        stride_byte_offset=1024,
                        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
                    )
                    kdesc = prims.Tcgen05SmemDesc.build(
                        _kv_slot(deep_base, kv_slot0, grp4 + c, kv_slot7),
                        leading_byte_offset=SLOT_BYTES,
                        stride_byte_offset=1024,
                        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
                    )
                    if prims.elect_sync():
                        if cutlass.const_expr(skip & 8 == 0):
                            for kb in cutlass.range_constexpr(SLICE // 32):
                                en = Boolean(True)
                                if cutlass.const_expr(kb == 0):
                                    en = Boolean(c > Int32(0))
                                prims.tcgen05_mma(
                                    prims.Tcgen05MMAKind.F8F6F4,
                                    prims.CTAGroup.CTA_1,
                                    prims.make_tmem_ptr(s_addr, Float32),
                                    qdesc.advance_start_address(kb * 32),
                                    kdesc.advance_start_address(kb * 32),
                                    idesc_qk,
                                    en,
                                )
                if prims.elect_sync():
                    prims.tcgen05_commit(bmma, group=prims.CTAGroup.CTA_1)
                    if cutlass.const_expr(qfree is not None):
                        if j == Int32(NTILES - 1):
                            prims.tcgen05_commit(qfree, group=prims.CTAGroup.CTA_1)
            if cutlass.const_expr(skip & 16 == 0):
                if ldq:
                    ph = ph ^ Int32(15)
                ph = ph ^ Int32(15) << Int32(16) + grp4
            qres = Int32(1)
            pre = Int32(0)
            if j + Int32(1) < Int32(NTILES):
                pre = Int32(1)
            ph = _wait_mma(bmma, ph, wl, bar_id)
        else:
            if cutlass.const_expr(TW and skip & 16 == 0):
                if tw != Int32(0):
                    _issue_kv_slice(
                        kv_slot0, bkv, tma_swa, tma_cmp, use_swa, grow, Int32(0), wl, lane, pol_keep
                    )
            for c in cutlass.range(NSLICE, unroll=1):
                sb = tw & c & Int32(1)
                if cutlass.const_expr(skip & 16 == 0):
                    ldq_s = Boolean(True)
                    if cutlass.const_expr(skip & 128 != 0):
                        ldq_s = j == jb
                    if ldq_s:
                        if rtid == Int32(0):
                            prims.mbarrier_arrive_expect_tx(bq, Q_SLICE_BYTES)
                            prims.cp_async_bulk_tensor_shared_cta_global(
                                q_slot0, tma_q.get_ptr(), [c * Int32(SLICE), Int32(0), b], bq
                            )
                    if cutlass.const_expr(TW):
                        if tw != Int32(0):
                            if c + Int32(1) < Int32(NSLICE):
                                if c & Int32(1) == Int32(0):
                                    _issue_kv_slice(
                                        kv_slotB,
                                        bkv + Int32(1),
                                        tma_swa,
                                        tma_cmp,
                                        use_swa,
                                        grow,
                                        c + Int32(1),
                                        wl,
                                        lane,
                                        pol_keep,
                                    )
                                else:
                                    _issue_kv_slice(
                                        kv_slot0,
                                        bkv,
                                        tma_swa,
                                        tma_cmp,
                                        use_swa,
                                        grow,
                                        c + Int32(1),
                                        wl,
                                        lane,
                                        pol_keep,
                                    )
                        else:
                            _issue_kv_slice(
                                kv_slot0,
                                bkv,
                                tma_swa,
                                tma_cmp,
                                use_swa,
                                grow,
                                c,
                                wl,
                                lane,
                                pol_keep,
                            )
                    else:
                        _issue_kv_slice(
                            kv_slot0, bkv, tma_swa, tma_cmp, use_swa, grow, c, wl, lane, pol_keep
                        )
                    if ldq_s:
                        _wait(bq, ph & Int32(1))
                        ph = ph ^ Int32(1)
                    _wait(bkv + sb, ph >> Int32(16) + sb & Int32(1))
                    ph = ph ^ Int32(1) << Int32(16) + sb
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                kslot = kv_slot0
                if cutlass.const_expr(TW):
                    if sb != Int32(0):
                        kslot = kv_slotB
                qdesc = prims.Tcgen05SmemDesc.build(
                    q_slot0,
                    leading_byte_offset=Q_SLICE_BYTES,
                    stride_byte_offset=1024,
                    layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
                )
                kdesc = prims.Tcgen05SmemDesc.build(
                    kslot,
                    leading_byte_offset=SLOT_BYTES,
                    stride_byte_offset=1024,
                    layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
                )
                if wl == Int32(0):
                    if prims.elect_sync():
                        if cutlass.const_expr(skip & 8 == 0):
                            for kb in cutlass.range_constexpr(SLICE // 32):
                                en = Boolean(True)
                                if cutlass.const_expr(kb == 0):
                                    en = Boolean(c > Int32(0))
                                prims.tcgen05_mma(
                                    prims.Tcgen05MMAKind.F8F6F4,
                                    prims.CTAGroup.CTA_1,
                                    prims.make_tmem_ptr(s_addr, Float32),
                                    qdesc.advance_start_address(kb * 32),
                                    kdesc.advance_start_address(kb * 32),
                                    idesc_qk,
                                    en,
                                )
                        prims.tcgen05_commit(bmma, group=prims.CTAGroup.CTA_1)
                ph = _wait_mma(bmma, ph, wl, bar_id)
            pre = Int32(0)
        if cutlass.const_expr(prof):
            if rtid == Int32(0):
                pst[pbase + Int32(4) * j + Int32(1)] = Int64(cute.arch.globaltimer())
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        L = scale_log2
        mk = Int32(s_msk[half * Int32(2)])
        mk2 = Int32(s_msk[half * Int32(2) + Int32(1)])
        old_max = m_run
        mx0 = old_max
        mx1 = Float32(NEG_FLT_MAX)
        mx2 = Float32(NEG_FLT_MAX)
        mx3 = Float32(NEG_FLT_MAX)
        tsa = s_addr + lane_base
        for ch in cutlass.range_constexpr(2):
            s = prims.tcgen05_ld(
                prims.Tcgen05LdStShape.SHAPE_16X32BX2,
                prims.make_tmem_ptr(tsa + Int32(32 * ch), Float32),
                num=32,
                offset=Int64(64),
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            mword = mk
            if cutlass.const_expr(ch == 1):
                mword = mk2
            for i in cutlass.range_constexpr(0, 32, 4):
                x0 = Float32(s[i])
                x1 = Float32(s[i + 1])
                x2 = Float32(s[i + 2])
                x3 = Float32(s[i + 3])
                if mword >> Int32(i) & Int32(1) == Int32(0):
                    x0 = Float32(-float("inf"))
                if mword >> Int32(i + 1) & Int32(1) == Int32(0):
                    x1 = Float32(-float("inf"))
                if mword >> Int32(i + 2) & Int32(1) == Int32(0):
                    x2 = Float32(-float("inf"))
                if mword >> Int32(i + 3) & Int32(1) == Int32(0):
                    x3 = Float32(-float("inf"))
                mx0 = cute.math.max(mx0, x0, ftz=True)
                mx1 = cute.math.max(mx1, x1, ftz=True)
                mx2 = cute.math.max(mx2, x2, ftz=True)
                mx3 = cute.math.max(mx3, x3, ftz=True)
        lmax = cute.math.max(
            cute.math.max(mx0, mx1, ftz=True), cute.math.max(mx2, mx3, ftz=True), ftz=True
        )
        lmax = cute.math.max(lmax, _shfl_xor_f32(lmax, 16), ftz=True)
        new_max = lmax
        safe_max = new_max
        if safe_max == Float32(NEG_FLT_MAX):
            safe_max = Float32(0.0)
        neg_scaled = -L * safe_max
        bias = neg_scaled + Float32(LOG2_448)
        sm0 = Float32(0.0)
        sm1 = Float32(0.0)
        sm2 = Float32(0.0)
        sm3 = Float32(0.0)
        regs = []
        for ch in cutlass.range_constexpr(2):
            s = prims.tcgen05_ld(
                prims.Tcgen05LdStShape.SHAPE_16X32BX2,
                prims.make_tmem_ptr(tsa + Int32(32 * ch), Float32),
                num=32,
                offset=Int64(64),
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            mword = mk
            if cutlass.const_expr(ch == 1):
                mword = mk2
            for pk in cutlass.range_constexpr(8):
                xs = [Float32(s[4 * pk + u]) for u in range(4)]
                for u in cutlass.range_constexpr(4):
                    if mword >> Int32(4 * pk + u) & Int32(1) == Int32(0):
                        xs[u] = Float32(-float("inf"))
                x01 = cute.arch.fma_packed_f32x2(
                    (xs[0], xs[1]), (L, L), (bias, bias), ftz=False, rnd="rn"
                )
                x23 = cute.arch.fma_packed_f32x2(
                    (xs[2], xs[3]), (L, L), (bias, bias), ftz=False, rnd="rn"
                )
                p0 = cute.math.exp2(x01[0], fastmath=True)
                p1 = cute.math.exp2(x01[1], fastmath=True)
                p2 = cute.math.exp2(x23[0], fastmath=True)
                p3 = cute.math.exp2(x23[1], fastmath=True)
                sm0, sm1 = cute.arch.add_packed_f32x2((sm0, sm1), (p0, p1), ftz=False, rnd="rn")
                sm2, sm3 = cute.arch.add_packed_f32x2((sm2, sm3), (p2, p3), ftz=False, rnd="rn")
                regs.append(_pack4_e4m3(p0, p1, p2, p3))
        local_sum = sm0 + sm1 + (sm2 + sm3)
        p_addr = tsa + Int32(TILE_N // 4)
        if cutlass.const_expr(skip & 1 == 0):
            _tmem_st_p8(p_addr, regs[0:8])
            _tmem_st_p8(p_addr + Int32(8), regs[8:16])
            _tmem_st_p8(p_addr + Int32(PANEL1), regs[0:8])
            _tmem_st_p8(p_addr + Int32(PANEL1 + 8), regs[8:16])
        exp_scale = cute.math.exp2(L * (old_max - new_max), fastmath=True)
        l_run = exp_scale * l_run + local_sum
        m_run = new_max
        if j > Int32(0):
            ident = old_max == new_max
            scale = Float32(1.0)
            if not ident:
                scale = exp_scale
            skip_c = prims.vote_sync(4294967295, ident, prims.VoteSync.ALL)
            if not skip_c:
                if cutlass.const_expr(skip & 2 == 0):
                    _rescale_o(tmem, lane_base, scale)
        prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.barrier(barrier_id=bar_id, number_of_threads=128)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        if cutlass.const_expr(prof):
            if rtid == Int32(0):
                pst[pbase + Int32(4) * j + Int32(2)] = Int64(cute.arch.globaltimer())
        if dp != Int32(0):
            if wl == Int32(0):
                if prims.elect_sync():
                    if cutlass.const_expr(skip & 4 == 0):
                        for v in cutlass.range_constexpr(NSLICE):
                            vdesc = prims.Tcgen05SmemDesc.build(
                                _kv_slot(deep_base, kv_slot0, grp4 + Int32(v), kv_slot7),
                                leading_byte_offset=0,
                                stride_byte_offset=1024,
                                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
                            )
                            d_addr = tmem + Int32(TM_O + (v & 1) * 128 + (v >> 1) * PANEL1)
                            a_addr = s_addr + Int32(TILE_N // 4 + (v >> 1) * PANEL1)
                            for kb in cutlass.range_constexpr(TILE_N // 32):
                                en = Boolean(True)
                                if cutlass.const_expr(kb == 0):
                                    en = Boolean(j > Int32(0))
                                prims.tcgen05_mma(
                                    prims.Tcgen05MMAKind.F8F6F4,
                                    prims.CTAGroup.CTA_1,
                                    prims.make_tmem_ptr(d_addr, Float32),
                                    prims.make_tmem_ptr(a_addr + Int32(8 * kb), E4M3),
                                    vdesc.advance_start_address(kb * 32 * SLICE),
                                    idesc_pv,
                                    en,
                                )
                    prims.tcgen05_commit(bmma, group=prims.CTAGroup.CTA_1)
                    if cutlass.const_expr(ext is not None):
                        if ext != Int32(0):
                            prims.tcgen05_commit(
                                gfree + (tb + j & Int32(1)), group=prims.CTAGroup.CTA_1
                            )
            ph = _wait_mma(bmma, ph, wl, bar_id)
        else:
            n1 = Int32(NSLICE)
            if cutlass.const_expr(TW):
                if tw != Int32(0):
                    n1 = Int32(0)
                    for hh in cutlass.range_constexpr(2):
                        if cutlass.const_expr(hh == 1):
                            if cutlass.const_expr(skip & (16 | 64) == 0):
                                _issue_kv_slice(
                                    kv_slot0,
                                    bkv,
                                    tma_swa,
                                    tma_cmp,
                                    use_swa,
                                    grow,
                                    Int32(0),
                                    wl,
                                    lane,
                                    pol_drop,
                                )
                                _issue_kv_slice(
                                    kv_slotB,
                                    bkv + Int32(1),
                                    tma_swa,
                                    tma_cmp,
                                    use_swa,
                                    grow,
                                    Int32(1),
                                    wl,
                                    lane,
                                    pol_drop,
                                )
                                _wait(bkv, ph >> Int32(16) & Int32(1))
                                ph = ph ^ Int32(1 << 16)
                                _wait(bkv + Int32(1), ph >> Int32(17) & Int32(1))
                                ph = ph ^ Int32(1 << 17)
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        if wl == Int32(0):
                            if prims.elect_sync():
                                if cutlass.const_expr(skip & 4 == 0):
                                    for vv in cutlass.range_constexpr(2):
                                        v2 = 2 - 2 * hh + vv
                                        slot2 = kv_slot0
                                        if cutlass.const_expr(vv == 1):
                                            slot2 = kv_slotB
                                        vdesc2 = prims.Tcgen05SmemDesc.build(
                                            slot2,
                                            leading_byte_offset=0,
                                            stride_byte_offset=1024,
                                            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
                                        )
                                        d2 = tmem + Int32(
                                            TM_O + (v2 & 1) * 128 + (v2 >> 1) * PANEL1
                                        )
                                        a2 = s_addr + Int32(TILE_N // 4 + (v2 >> 1) * PANEL1)
                                        for kb in cutlass.range_constexpr(TILE_N // 32):
                                            en2 = Boolean(True)
                                            if cutlass.const_expr(kb == 0):
                                                en2 = Boolean(j > Int32(0))
                                            prims.tcgen05_mma(
                                                prims.Tcgen05MMAKind.F8F6F4,
                                                prims.CTAGroup.CTA_1,
                                                prims.make_tmem_ptr(d2, Float32),
                                                prims.make_tmem_ptr(a2 + Int32(8 * kb), E4M3),
                                                vdesc2.advance_start_address(kb * 32 * SLICE),
                                                idesc_pv,
                                                en2,
                                            )
                                prims.tcgen05_commit(bmma, group=prims.CTAGroup.CTA_1)
                        ph = _wait_mma(bmma, ph, wl, bar_id)
            for vi in cutlass.range(n1, unroll=1):
                v = vi + Int32(3) & Int32(3)
                if v != Int32(3):
                    if cutlass.const_expr(skip & (16 | 64) == 0):
                        _issue_kv_slice(
                            kv_slot0, bkv, tma_swa, tma_cmp, use_swa, grow, v, wl, lane, pol_drop
                        )
                        _wait(bkv, ph >> Int32(16) & Int32(1))
                        ph = ph ^ Int32(1 << 16)
                    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                vdesc = prims.Tcgen05SmemDesc.build(
                    kv_slot0,
                    leading_byte_offset=0,
                    stride_byte_offset=1024,
                    layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
                )
                d_addr = (
                    tmem
                    + Int32(TM_O)
                    + (v & Int32(1)) * Int32(128)
                    + (v >> Int32(1)) * Int32(PANEL1)
                )
                a_addr = s_addr + Int32(TILE_N // 4) + (v >> Int32(1)) * Int32(PANEL1)
                if wl == Int32(0):
                    if prims.elect_sync():
                        if cutlass.const_expr(skip & 4 == 0):
                            for kb in cutlass.range_constexpr(TILE_N // 32):
                                en = Boolean(True)
                                if cutlass.const_expr(kb == 0):
                                    en = Boolean(j > Int32(0))
                                prims.tcgen05_mma(
                                    prims.Tcgen05MMAKind.F8F6F4,
                                    prims.CTAGroup.CTA_1,
                                    prims.make_tmem_ptr(d_addr, Float32),
                                    prims.make_tmem_ptr(a_addr + Int32(8 * kb), E4M3),
                                    vdesc.advance_start_address(kb * 32 * SLICE),
                                    idesc_pv,
                                    en,
                                )
                        prims.tcgen05_commit(bmma, group=prims.CTAGroup.CTA_1)
                ph = _wait_mma(bmma, ph, wl, bar_id)
        if cutlass.const_expr(prof):
            if rtid == Int32(0):
                pst[pbase + Int32(4) * j + Int32(3)] = Int64(cute.arch.globaltimer())
    do_epi = Int32(1)
    if cutlass.const_expr(phased):
        if je != Int32(NTILES):
            do_epi = Int32(0)
    if do_epi != Int32(0):
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        l_tot = l_run + _shfl_xor_f32(l_run, 16)
        norm = _final_norm(m_run, l_tot, sink, scale_log2)
        out_base = (
            out_addr
            + Int64(b) * Int64(HEADS * LATENT * 2)
            + Int64((head * Int32(LATENT) + half * Int32(64)) * Int32(2))
        )
        for v in cutlass.range_constexpr(NSLICE):
            for hc in cutlass.range_constexpr(2):
                o_addr = tmem + lane_base + Int32(TM_O + v % 2 * 128 + v // 2 * PANEL1 + 32 * hc)
                o = prims.tcgen05_ld(
                    prims.Tcgen05LdStShape.SHAPE_16X32BX2,
                    prims.make_tmem_ptr(o_addr, Float32),
                    num=32,
                    offset=Int64(64),
                )
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                ob = []
                for pi in cutlass.range_constexpr(16):
                    fp = cute.arch.mul_packed_f32x2(
                        (norm, norm),
                        (Float32(o[2 * pi]), Float32(o[2 * pi + 1])),
                        ftz=False,
                        rnd="rn",
                    )
                    ob.append(_pack_bf16x2(fp[0], fp[1]))
                for k in cutlass.range_constexpr(4):
                    dst = cutlass.inttoptr(
                        out_base + Int64((v * SLICE + hc * 32 + k * 8) * 2),
                        mem_space=1,
                        dtype=Int32,
                    )
                    dst.store(
                        cutlass.Vector.from_elements(
                            tuple((ob[4 * k + i] for i in range(4))), Int32
                        ),
                        alignment=16,
                    )
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.barrier(barrier_id=bar_id, number_of_threads=128)
    if cutlass.const_expr(phased):
        return (ph, m_run, l_run, qres, pre, dp)
    return ph
