# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp roles of the team CTAs' sparse MLA: KV gather, MMA, softmax, correction and epilogue, and the two-row-group
split with the DSMEM pair merge.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, Int64, Uint32
from cutlass.cutlass_dsl import dsl_user_op as _dsl_user_op
from cutlass.experimental import primitives as prims

from . import attention_engine as AE
from . import ptx_helpers as B
from . import topk_select_lib as LS

HEADS = AE.HEADS
LATENT = AE.LATENT
WINDOW = AE.WINDOW
SLICE = AE.SLICE
NSLICE = AE.NSLICE
Q_SLICE_BYTES = AE.Q_SLICE_BYTES
Q_BYTES = AE.Q_BYTES
LOG2_E = AE.LOG2_E
LOG2_448 = AE.LOG2_448
NEG_FLT_MAX = AE.NEG_FLT_MAX
PANEL1 = AE.PANEL1
TMEM_COLS = AE.TMEM_COLS
TMEM_O = AE.TMEM_O
E4M3 = AE.E4M3
BF16 = AE.BF16
TILE_N = 128
# S buffers: lane panel 0, columns 128k (k = tile % 4); P_k: panel 1, column 128k + 32; O: panel 1, column 448
O_COL4 = 448
OUT_SLICE = 64
OUT_REGS = OUT_SLICE // 2
SLOT_BYTES = TILE_N * SLICE
SLOT_BY_POS = True
W_SM0 = 0
W_CTL0 = 4
W_MMA = 8
W_G0 = 9
NGW = 7
REG_SOFTMAX = 184
REG_CTL = 232
REG_OTHER = 48
_wait = AE._wait
_pack4_e4m3 = AE._pack4_e4m3
_tmem_st_p_fp8 = AE._tmem_st_p_fp8
_shfl_xor_f32 = AE._shfl_xor_f32
_pack_bf16x2 = AE._pack_bf16x2
_stamp = AE._stamp
TS_Q = 24
TS_G = 25
TS_QK = 30
TS_SM = 35
TS_PV = 40
TS_COR = 45
TS_FIN = 50
TS_STORED = 51


@cute.jit
def _slot_ptr(s_kv, s_ext, slot):
    return s_kv + slot * Int32(SLOT_BYTES)


@cute.jit
def role_q_load(tma_q, s_q, q_full, qrow=0):
    if prims.elect_sync():
        prims.mbarrier_arrive_expect_tx(q_full, Q_BYTES)
        for c in cutlass.range_constexpr(NSLICE):
            prims.cp_async_bulk_tensor_shared_cta_global(
                s_q + c * Q_SLICE_BYTES, tma_q.get_ptr(), [c * SLICE, 0, qrow], q_full
            )


@cute.jit
def role_gather_team(
    tma_swa,
    tma_cmp,
    s_kv,
    s_ext,
    s_idx,
    kv_full,
    kv_empty,
    swa_ready,
    sel_ready,
    gw,
    v_lo,
    ntiles: cutlass.Constexpr[int],
    nslots: cutlass.Constexpr[int],
    tr=None,
    j_begin=0,
    ngw_late: cutlass.Constexpr[int] = NGW,
    sel_early=None,
    j_join: cutlass.Constexpr[int] = 1,
):
    lane = cute.arch.lane_idx()
    # Adjacent team CTAs retain the same full K/V slice for QK, then
    # consume separate 64-column halves for PV and the final output.
    v_slice = v_lo >> Int32(1)
    for j in cutlass.range(j_begin, ntiles, unroll=1):
        nw = Int32(NGW)
        if j >= Int32(j_join):
            nw = Int32(ngw_late)
        if j == Int32(0):
            _wait(swa_ready, Int32(0))
        if cutlass.const_expr(sel_early is None):
            if j == Int32(1):
                _wait(sel_ready, Int32(0))
        else:
            if j == Int32(1):
                _wait(sel_early, Int32(0))
            if j == Int32(ntiles - 1):
                _wait(sel_ready, Int32(0))
        tm = tma_cmp.get_ptr().toint()
        if j == Int32(0):
            tm = tma_swa.get_ptr().toint()
        for c in cutlass.range_constexpr(NSLICE):
            sl = v_slice + Int32(1 + c) & Int32(3)
            k = j * Int32(NSLICE) + sl
            if cutlass.const_expr(SLOT_BY_POS):
                k = j * Int32(NSLICE) + Int32(c)
            slot = k % Int32(nslots)
            _wait(kv_empty + slot, k // Int32(nslots) & Int32(1) ^ Int32(1))
            if gw == Int32(0):
                if lane == Int32(0):
                    prims.mbarrier_arrive_expect_tx(kv_full + slot, SLOT_BYTES)
            g = lane * nw + gw
            if g < Int32(TILE_N // 4):
                r4 = B.lds_v4(Int32(s_idx.toint()) + (j * Int32(TILE_N) + g * Int32(4)) * Int32(4))
                dst = Int32(_slot_ptr(s_kv, s_ext, slot).toint()) + g * Int32(4 * SLICE)
                B.tma_g4(
                    dst,
                    tm,
                    sl * Int32(SLICE),
                    r4[0],
                    r4[1],
                    r4[2],
                    r4[3],
                    Int32(kv_full.toint()) + slot * Int32(8),
                )
        if cutlass.const_expr(tr is not None):
            _stamp(tr, Int32(TS_G) + j, gw == Int32(0) and lane == Int32(0))


@cute.jit
def _qk(
    tmem,
    s_q,
    s_kv,
    s_ext,
    j,
    kv_full,
    kv_empty,
    s_full,
    s_empty,
    qdesc,
    idesc_qk,
    v_lo,
    nslots: cutlass.Constexpr[int],
):
    v_slice = v_lo >> Int32(1)
    st = j & Int32(3)
    if j >= Int32(4):
        _wait(s_empty + st, (j >> Int32(2)) - Int32(1) & Int32(1))
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    s_addr = tmem + st * Int32(TILE_N)
    # Consume in the gather order. The first three slices can issue and
    # release immediately; the CTA's retained V slice is consumed last.
    for step in cutlass.range_constexpr(NSLICE):
        c = (v_slice + Int32(1 + step)) & Int32(3)
        k = j * Int32(NSLICE) + c
        if cutlass.const_expr(SLOT_BY_POS):
            k = j * Int32(NSLICE) + Int32(step)
        slot = k % Int32(nslots)
        _wait(kv_full + slot, k // Int32(nslots) & Int32(1))
        kdesc = prims.Tcgen05SmemDesc.build(
            _slot_ptr(s_kv, s_ext, slot),
            leading_byte_offset=SLOT_BYTES,
            stride_byte_offset=1024,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        if prims.elect_sync():
            for kb in cutlass.range_constexpr(SLICE // 32):
                prims.tcgen05_mma(
                    prims.Tcgen05MMAKind.F8F6F4,
                    prims.CTAGroup.CTA_1,
                    prims.make_tmem_ptr(s_addr, Float32),
                    qdesc.advance_start_address(c * Int32(Q_SLICE_BYTES) + Int32(kb * 32)),
                    kdesc.advance_start_address(kb * 32),
                    idesc_qk,
                    Boolean(step > 0 or kb > 0),
                )
            if cutlass.const_expr(step < NSLICE - 1):
                prims.tcgen05_commit(kv_empty + slot, group=prims.CTAGroup.CTA_1)
    if prims.elect_sync():
        prims.tcgen05_commit(s_full + st, group=prims.CTAGroup.CTA_1)


@cute.jit
def _pv(
    tmem,
    s_kv,
    s_ext,
    j,
    p_full,
    o_corr,
    kv_empty,
    s_empty,
    o_full,
    idesc_pv,
    v_lo,
    nslots: cutlass.Constexpr[int],
    tr=None,
):
    st = j & Int32(3)
    _wait(p_full + st, j >> Int32(2) & Int32(1))
    if cutlass.const_expr(tr is not None):
        _stamp(
            tr,
            Int32(56) + (j & Int32(1)) * Int32(7),
            (j == Int32(1) or j == Int32(2)) and cute.arch.lane_idx() == Int32(0),
        )
    if cutlass.const_expr(o_corr is not None):
        if j > Int32(0):
            _wait(o_corr, j - Int32(1) & Int32(1))
    if cutlass.const_expr(tr is not None):
        _stamp(
            tr,
            Int32(57) + (j & Int32(1)) * Int32(5),
            (j == Int32(1) or j == Int32(2)) and cute.arch.lane_idx() == Int32(0),
        )
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    v = v_lo >> Int32(1)
    slot = (j * Int32(NSLICE) + v) % Int32(nslots)
    if cutlass.const_expr(SLOT_BY_POS):
        slot = (j * Int32(NSLICE) + Int32(3)) % Int32(nslots)
    d_addr = tmem + Int32(O_COL4) + Int32(PANEL1)
    a_addr = tmem + st * Int32(TILE_N) + Int32(TILE_N // 4) + Int32(PANEL1)
    # Select a half of the existing SW128 tile. The physical row pitch
    # remains 128 bytes and the 1024-byte swizzle origin is unchanged.
    v_ptr = _slot_ptr(s_kv, s_ext, slot) + (v_lo & Int32(1)) * Int32(OUT_SLICE)
    vdesc = prims.Tcgen05SmemDesc.build(
        v_ptr,
        leading_byte_offset=0,
        stride_byte_offset=1024,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    acc0 = Boolean(j > Int32(0))
    if prims.elect_sync():
        for kb in cutlass.range_constexpr(TILE_N // 32):
            en = acc0 if cutlass.const_expr(kb == 0) else Boolean(True)
            prims.tcgen05_mma(
                prims.Tcgen05MMAKind.F8F6F4,
                prims.CTAGroup.CTA_1,
                prims.make_tmem_ptr(d_addr, Float32),
                prims.make_tmem_ptr(a_addr + Int32(8 * kb), E4M3),
                vdesc.advance_start_address(kb * 32 * SLICE),
                idesc_pv,
                en,
            )
        prims.tcgen05_commit(kv_empty + slot, group=prims.CTAGroup.CTA_1)
        prims.tcgen05_commit(s_empty + st, group=prims.CTAGroup.CTA_1)
        prims.tcgen05_commit(o_full, group=prims.CTAGroup.CTA_1)


@cute.jit
def _landed(kv_full, j, nslots: cutlass.Constexpr[int]):
    ok = Int32(1)
    for c in cutlass.range_constexpr(NSLICE):
        k = j * Int32(NSLICE) + Int32(c)
        if B.mbar_test_wait(
            Int32(kv_full.toint()) + k % Int32(nslots) * Int32(8), k // Int32(nslots) & Int32(1)
        ) == Int32(0):
            ok = Int32(0)
    return ok


@cute.jit
def role_mma_team(
    tmem,
    s_q,
    s_kv,
    s_ext,
    q_full,
    kv_full,
    kv_empty,
    s_full,
    s_empty,
    p_full,
    o_corr,
    o_full,
    v_lo,
    ntiles: cutlass.Constexpr[int],
    nslots: cutlass.Constexpr[int],
    tr=None,
    qk_ahead: cutlass.Constexpr[bool] = False,
    pv_last=None,
):
    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=Float32, a_dtype=E4M3, b_dtype=E4M3, n_dim=TILE_N, m_dim=HEADS
    )
    idesc_pv = prims.Tcgen05InstrDesc.build(
        c_dtype=Float32,
        a_dtype=E4M3,
        b_dtype=E4M3,
        a_major=0,
        b_major=1,
        n_dim=OUT_SLICE,
        m_dim=HEADS,
    )
    qdesc = prims.Tcgen05SmemDesc.build(
        s_q,
        leading_byte_offset=Q_SLICE_BYTES,
        stride_byte_offset=1024,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    lane = cute.arch.lane_idx()
    _wait(q_full, Int32(0))
    _qk(
        tmem,
        s_q,
        s_kv,
        s_ext,
        Int32(0),
        kv_full,
        kv_empty,
        s_full,
        s_empty,
        qdesc,
        idesc_qk,
        v_lo,
        nslots,
    )
    if cutlass.const_expr(tr is not None):
        _stamp(tr, Int32(TS_QK), lane == Int32(0))
    for j in cutlass.range(ntiles, unroll=1):
        if j + Int32(1) < Int32(ntiles):
            ahead = Int32(0)
            if cutlass.const_expr(qk_ahead):
                if j >= Int32(1):
                    _wait(p_full + (j & Int32(1)), j >> Int32(1) & Int32(1))
                    ahead = _landed(kv_full, j + Int32(1), nslots)
            if ahead != Int32(0):
                _qk(
                    tmem,
                    s_q,
                    s_kv,
                    s_ext,
                    j + Int32(1),
                    kv_full,
                    kv_empty,
                    s_full,
                    s_empty,
                    qdesc,
                    idesc_qk,
                    v_lo,
                    nslots,
                )
            _pv(
                tmem,
                s_kv,
                s_ext,
                j,
                p_full,
                o_corr,
                kv_empty,
                s_empty,
                o_full,
                idesc_pv,
                v_lo,
                nslots,
                tr,
            )
            if cutlass.const_expr(tr is not None):
                _stamp(tr, Int32(TS_PV) + j, lane == Int32(0))
            if ahead == Int32(0):
                _qk(
                    tmem,
                    s_q,
                    s_kv,
                    s_ext,
                    j + Int32(1),
                    kv_full,
                    kv_empty,
                    s_full,
                    s_empty,
                    qdesc,
                    idesc_qk,
                    v_lo,
                    nslots,
                )
            if cutlass.const_expr(tr is not None):
                _stamp(tr, Int32(TS_QK + 1) + j, lane == Int32(0))
        else:
            _pv(
                tmem,
                s_kv,
                s_ext,
                j,
                p_full,
                o_corr,
                kv_empty,
                s_empty,
                o_full,
                idesc_pv,
                v_lo,
                nslots,
                tr,
            )
            if cutlass.const_expr(pv_last is not None):
                if prims.elect_sync():
                    prims.tcgen05_commit(pv_last, group=prims.CTAGroup.CTA_1)
            if cutlass.const_expr(tr is not None):
                _stamp(tr, Int32(TS_PV) + j, lane == Int32(0))


@cute.jit
def role_mma_ordered(
    tmem,
    s_q,
    s_kv,
    s_ext,
    q_full,
    kv_full,
    kv_empty,
    s_full,
    s_empty,
    p_full,
    o_full,
    v_lo,
    ntiles: cutlass.Constexpr[int],
    nslots: cutlass.Constexpr[int],
    pv_last,
    tr=None,
    sel_ready=None,
    idle_ns: cutlass.Constexpr[int] = 0,
):
    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=Float32, a_dtype=E4M3, b_dtype=E4M3, n_dim=TILE_N, m_dim=HEADS
    )
    idesc_pv = prims.Tcgen05InstrDesc.build(
        c_dtype=Float32,
        a_dtype=E4M3,
        b_dtype=E4M3,
        a_major=0,
        b_major=1,
        n_dim=OUT_SLICE,
        m_dim=HEADS,
    )
    qdesc = prims.Tcgen05SmemDesc.build(
        s_q,
        leading_byte_offset=Q_SLICE_BYTES,
        stride_byte_offset=1024,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    lane = cute.arch.lane_idx()
    _wait(q_full, Int32(0))
    _qk(
        tmem,
        s_q,
        s_kv,
        s_ext,
        Int32(0),
        kv_full,
        kv_empty,
        s_full,
        s_empty,
        qdesc,
        idesc_qk,
        v_lo,
        nslots,
    )
    if cutlass.const_expr(tr is not None):
        _stamp(tr, Int32(TS_QK), lane == Int32(0))
    _pv(
        tmem,
        s_kv,
        s_ext,
        Int32(0),
        p_full,
        None,
        kv_empty,
        s_empty,
        o_full,
        idesc_pv,
        v_lo,
        nslots,
        tr,
    )
    if cutlass.const_expr(tr is not None):
        _stamp(tr, Int32(TS_PV), lane == Int32(0))
    if cutlass.const_expr(sel_ready is not None):
        _wait(sel_ready, Int32(0))
    for j in cutlass.range(1, ntiles + 1, unroll=1):
        if j < Int32(ntiles):
            _qk(
                tmem,
                s_q,
                s_kv,
                s_ext,
                j,
                kv_full,
                kv_empty,
                s_full,
                s_empty,
                qdesc,
                idesc_qk,
                v_lo,
                nslots,
            )
            if cutlass.const_expr(tr is not None):
                _stamp(tr, Int32(TS_QK) + j, lane == Int32(0))
        if j > Int32(1):
            _pv(
                tmem,
                s_kv,
                s_ext,
                j - Int32(1),
                p_full,
                None,
                kv_empty,
                s_empty,
                o_full,
                idesc_pv,
                v_lo,
                nslots,
                tr,
            )
            if cutlass.const_expr(tr is not None):
                _stamp(tr, Int32(TS_PV) + j - Int32(1), lane == Int32(0))
    if prims.elect_sync():
        prims.tcgen05_commit(pv_last, group=prims.CTAGroup.CTA_1)


@cute.jit
def role_softmax_team(
    tmem,
    s_full,
    p_full,
    s_stat,
    s_fin,
    fin_bar,
    scale_log2: Float32,
    swa_len,
    n_sel,
    j0,
    j1,
    m_run,
    l_run,
    final: cutlass.Constexpr[bool],
    tr=None,
    o_full=None,
    v_lo=None,
    full_tiles: cutlass.Constexpr[bool] = False,
):
    half_n: cutlass.Constexpr[int] = TILE_N // 2
    npk: cutlass.Constexpr[int] = TILE_N // 8
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % Int32(4)
    lane = cute.arch.lane_idx()
    head = warp * Int32(16) + (lane & Int32(15))
    lane_base = warp * Int32(32) << Int32(16)
    col0 = (lane >> Int32(4)) * Int32(half_n)
    for j in cutlass.range(j0, j1, unroll=1):
        st = j & Int32(3)
        _wait(s_full + st, j >> Int32(2) & Int32(1))
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        s_addr = tmem + lane_base + st * Int32(TILE_N)
        s = prims.tcgen05_ld(
            prims.Tcgen05LdStShape.SHAPE_16X32BX2,
            prims.make_tmem_ptr(s_addr, Float32),
            num=half_n,
            offset=Int64(half_n),
        )
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        limit = Int32(WINDOW) + n_sel
        if j == Int32(0):
            limit = swa_len
        nv_here = limit - (j * Int32(TILE_N) + col0)
        old_max = m_run
        mx0 = old_max
        mx1 = Float32(NEG_FLT_MAX)
        mx2 = Float32(NEG_FLT_MAX)
        mx3 = Float32(NEG_FLT_MAX)
        sv = []
        for i in cutlass.range_constexpr(half_n):
            if cutlass.const_expr(full_tiles):
                sv.append(Float32(s[i]))
            else:
                sv.append(cutlass.select_(nv_here > Int32(i), Float32(s[i]), Float32(NEG_FLT_MAX)))
        for i in cutlass.range_constexpr(0, half_n, 4):
            mx0 = cute.math.max(mx0, sv[i], ftz=True)
            mx1 = cute.math.max(mx1, sv[i + 1], ftz=True)
            mx2 = cute.math.max(mx2, sv[i + 2], ftz=True)
            mx3 = cute.math.max(mx3, sv[i + 3], ftz=True)
        lmax = cute.math.max(
            cute.math.max(mx0, mx1, ftz=True), cute.math.max(mx2, mx3, ftz=True), ftz=True
        )
        lmax = cute.math.max(lmax, _shfl_xor_f32(lmax, 16), ftz=True)
        new_max = lmax
        safe_max = new_max
        if safe_max == Float32(NEG_FLT_MAX):
            safe_max = Float32(0.0)
        bias = -scale_log2 * safe_max + Float32(LOG2_448)
        sm0 = Float32(0.0)
        sm1 = Float32(0.0)
        sm2 = Float32(0.0)
        sm3 = Float32(0.0)
        regs = []
        for pk in cutlass.range_constexpr(npk):
            x01 = cute.arch.fma_packed_f32x2(
                (sv[4 * pk], sv[4 * pk + 1]),
                (scale_log2, scale_log2),
                (bias, bias),
                ftz=False,
                rnd="rn",
            )
            x23 = cute.arch.fma_packed_f32x2(
                (sv[4 * pk + 2], sv[4 * pk + 3]),
                (scale_log2, scale_log2),
                (bias, bias),
                ftz=False,
                rnd="rn",
            )
            p0 = cute.math.exp2(x01[0], fastmath=True)
            p1 = cute.math.exp2(x01[1], fastmath=True)
            p2 = cute.math.exp2(x23[0], fastmath=True)
            p3 = cute.math.exp2(x23[1], fastmath=True)
            sm0, sm1 = cute.arch.add_packed_f32x2((sm0, sm1), (p0, p1), ftz=False, rnd="rn")
            sm2, sm3 = cute.arch.add_packed_f32x2((sm2, sm3), (p2, p3), ftz=False, rnd="rn")
            regs.append(_pack4_e4m3(p0, p1, p2, p3))
        local_sum = sm0 + sm1 + (sm2 + sm3)
        p_addr = s_addr + Int32(TILE_N // 4)
        if cutlass.const_expr(v_lo is not None):
            # P lives in lane panel 1 next to O (the PV reads A from the same panel as D).
            _tmem_st_p_fp8(p_addr + Int32(PANEL1), regs, npk)
        else:
            _tmem_st_p_fp8(p_addr, regs, npk)
            _tmem_st_p_fp8(p_addr + Int32(PANEL1), regs, npk)
        prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        exp_scale = cute.math.exp2(scale_log2 * (old_max - new_max), fastmath=True)
        l_run = exp_scale * l_run + local_sum
        m_run = new_max
        if cutlass.const_expr(o_full is None):
            if lane < Int32(16):
                (s_stat + st * Int32(128) + head).store(old_max)
                (s_stat + st * Int32(128) + Int32(64) + head).store(new_max)
        if cutlass.const_expr(o_full is not None):
            if j > Int32(0):
                _wait(o_full, j - Int32(1) & Int32(1))
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                ident = old_max == new_max
                cscale = Float32(1.0)
                if not ident:
                    cscale = cute.math.exp2(scale_log2 * (old_max - new_max), fastmath=True)
                skip = prims.vote_sync(4294967295, ident, prims.VoteSync.ALL)
                if not skip:
                    o_addr0 = tmem + lane_base + Int32(O_COL4) + Int32(PANEL1)
                    ptr = prims.make_tmem_ptr(o_addr0, Float32)
                    o = prims.tcgen05_ld(
                        prims.Tcgen05LdStShape.SHAPE_16X32BX2,
                        ptr,
                        num=OUT_REGS,
                        offset=Int64(OUT_REGS),
                    )
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    vals = []
                    for rr in cutlass.range_constexpr(0, OUT_REGS, 2):
                        pr = cute.arch.mul_packed_f32x2(
                            (Float32(o[rr]), Float32(o[rr + 1])),
                            (cscale, cscale),
                            ftz=False,
                            rnd="rn",
                        )
                        vals.append(pr[0])
                        vals.append(pr[1])
                    prims.tcgen05_st(
                        prims.Tcgen05LdStShape.SHAPE_16X32BX2,
                        ptr,
                        cutlass.Vector.from_elements(tuple(vals), Float32),
                        offset=Int64(OUT_REGS),
                    )
                    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.sync_warp()
        if prims.elect_sync():
            prims.mbarrier_arrive(p_full + st)
        if cutlass.const_expr(tr is not None):
            _stamp(tr, Int32(TS_SM) + j, warp == Int32(0) and lane == Int32(0))
    if cutlass.const_expr(final):
        l_tot = l_run + _shfl_xor_f32(l_run, 16)
        if lane < Int32(16):
            (s_fin + head).store(m_run)
            (s_fin + Int32(64) + head).store(l_tot)
        cute.arch.sync_warp()
        if prims.elect_sync():
            prims.mbarrier_arrive(fin_bar)
    return (m_run, l_run)


@cute.jit
def _final_norm(
    final_max: Float32, reduced_sum: Float32, sink: Float32, scale_log2: Float32
) -> Float32:
    t = cute.arch.mul_packed_f32x2(
        (final_max, final_max), (scale_log2, scale_log2), ftz=False, rnd="rn"
    )[0]
    delta = cute.arch.fma_packed_f32x2(
        (sink, sink), (Float32(LOG2_E), Float32(LOG2_E)), (-t, -t), ftz=False, rnd="rn"
    )[0]
    attention_rescale = cute.math.exp2(-cute.math.max(delta, Float32(0)), approx=True)
    sink_mass = cute.math.exp2(cute.math.min(delta, Float32(0)), approx=True)
    sink_mass *= Float32(448.0)
    return (
        Float32(1.0)
        * attention_rescale
        * cute.math.rcp(reduced_sum * attention_rescale + sink_mass, approx=True)
    )


@cute.jit
def role_corr_epi_team(
    tmem,
    p_full,
    o_full,
    o_corr,
    s_stat,
    s_fin,
    fin_bar,
    sink_ptr,
    out_addr: Int64,
    scale_log2: Float32,
    v_lo,
    stage,
    ntiles: cutlass.Constexpr[int],
    bar_id: cutlass.Constexpr[int],
    tr=None,
    pv_last=None,
    dry=None,
):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % Int32(4)
    lane = cute.arch.lane_idx()
    head = warp * Int32(16) + (lane & Int32(15))
    half = lane >> Int32(4)
    lane_base = warp * Int32(32) << Int32(16)
    sink = Float32((sink_ptr + head).load())
    o_addr0 = tmem + lane_base + Int32(O_COL4) + Int32(PANEL1)
    if cutlass.const_expr(pv_last is None):
        for j in cutlass.range(1, ntiles, unroll=1):
            st = j & Int32(1)
            _wait(p_full + st, j >> Int32(1) & Int32(1))
            old_max = (s_stat + st * Int32(128) + head).load()
            new_max = (s_stat + st * Int32(128) + Int32(64) + head).load()
            _wait(o_full, j - Int32(1) & Int32(1))
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            ident = old_max == new_max
            scale = Float32(1.0)
            if not ident:
                scale = cute.math.exp2(scale_log2 * (old_max - new_max), fastmath=True)
            skip = prims.vote_sync(4294967295, ident, prims.VoteSync.ALL)
            if not skip:
                for hc in cutlass.range_constexpr(OUT_REGS // 32):
                    ptr = prims.make_tmem_ptr(o_addr0 + Int32(32 * hc), Float32)
                    o = prims.tcgen05_ld(
                        prims.Tcgen05LdStShape.SHAPE_16X32BX2, ptr, num=32, offset=Int64(OUT_REGS)
                    )
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    vals = []
                    for rr in cutlass.range_constexpr(0, 32, 2):
                        pr = cute.arch.mul_packed_f32x2(
                            (Float32(o[rr]), Float32(o[rr + 1])),
                            (scale, scale),
                            ftz=False,
                            rnd="rn",
                        )
                        vals.append(pr[0])
                        vals.append(pr[1])
                    prims.tcgen05_st(
                        prims.Tcgen05LdStShape.SHAPE_16X32BX2,
                        ptr,
                        cutlass.Vector.from_elements(tuple(vals), Float32),
                        offset=Int64(OUT_REGS),
                    )
            prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            cute.arch.sync_warp()
            if prims.elect_sync():
                prims.mbarrier_arrive(o_corr)
            if cutlass.const_expr(tr is not None):
                _stamp(tr, Int32(TS_COR) + j, warp == Int32(0) and lane == Int32(0))
    # dry != 0: code warm-up during the warm-up pass (no waits, no output stores; values discarded)
    wet = Int32(1)
    if cutlass.const_expr(dry is not None):
        wet = Int32(1) - dry
    if wet != Int32(0):
        if cutlass.const_expr(pv_last is None):
            _wait(o_full, Int32(ntiles - 1 & 1))
        else:
            _wait(pv_last, Int32(0))
        _wait(fin_bar, Int32(0))
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    if cutlass.const_expr(tr is not None):
        _stamp(tr, Int32(TS_FIN), warp == Int32(0) and lane == Int32(0))
    final_max = (s_fin + head).load()
    reduced_sum = (s_fin + Int32(64) + head).load()
    norm = _final_norm(final_max, reduced_sum, sink, scale_log2)
    ob = []
    o = prims.tcgen05_ld(
        prims.Tcgen05LdStShape.SHAPE_16X32BX2,
        prims.make_tmem_ptr(o_addr0, Float32),
        num=OUT_REGS,
        offset=Int64(OUT_REGS),
    )
    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
    for pi in cutlass.range_constexpr(OUT_REGS // 2):
        fp = cute.arch.mul_packed_f32x2(
            (norm, norm), (Float32(o[2 * pi]), Float32(o[2 * pi + 1])), ftz=False, rnd="rn"
        )
        ob.append(_pack_bf16x2(fp[0], fp[1]))
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    # Each lane owns a contiguous 32-element output segment. Vector stores
    # avoid the shared staging, CTA rendezvous, and bulk-store completion.
    rowb = out_addr + Int64(
        (head * Int32(LATENT) + v_lo * Int32(OUT_SLICE) + half * Int32(OUT_REGS)) * Int32(2)
    )
    if wet != Int32(0):
        for ch in cutlass.range_constexpr(OUT_REGS // 8):
            B.st_g_v4(
                rowb + Int64(ch * 16), ob[ch * 4], ob[ch * 4 + 1], ob[ch * 4 + 2], ob[ch * 4 + 3]
            )
    if cutlass.const_expr(tr is not None):
        _stamp(tr, Int32(TS_STORED), warp == Int32(0) and lane == Int32(0))


# ===================================================================== 2 row groups x 4 V groups, DSMEM merge
RS24_O_COL = 384  # O accumulator: lane panel 1, columns 384-511 (N = 128); P_t at 128t + 32, t <= 2
RS24_DRY_MASK = 2047  # dry pass: every merge address wraps inside this CTA's 2 KB stat/fin scratch


def rs24_rx_bytes(f32x):
    # partner push: nchunk x [128 threads][16 B] partial O, then [128 threads][8 B] (max, sum)
    return (8 if f32x else 4) * 2048 + 1024


@_dsl_user_op
def _lds_vol(saddr, *, loc=None, ip=None):
    return Int32(
        B._asm(
            B.T.i32(), [B._v(saddr, loc, ip)], "ld.volatile.shared.u32 $0, [$1];", "=r,r", loc, ip
        )
    )


@cute.jit
def rs24_tiles(d):
    # row group g = d & 1: g == 0 -> tiles {0 (SWA), 1, 2}; g == 1 -> tiles {3, 4}
    j0 = Int32(0)
    cnt = Int32(3)
    if (d & Int32(1)) != Int32(0):
        j0 = Int32(3)
        cnt = Int32(2)
    return (j0, cnt)


LSU_ARR = (
    128  # control WG threads arriving (noinc) on kv_full[4..7] of g = 1 CTAs when rs24_lsu > 0
)


@cute.jit
def lsu_tile4(
    tq, d, lsu_g: cutlass.Constexpr[int], s_kv_a, s_idx_a, kvf_a, kv_empty, pool: Int64, tr=None
):
    # LSU-assisted gather (g = 1 CTAs, tile 4 = local tile 1, slots 4-7): groups [32 - lsu_g, 32) of every slot via
    # cp.async (128 threads, 8 per 128 B row slice, SW128 swizzle by hand), then a noinc arrive per thread and slot
    k = d >> Int32(1)
    t = Int32(1)
    j = Int32(4)
    sbase = (t & Int32(1)) * Int32(4)
    epar = ((t >> Int32(1)) & Int32(1)) ^ Int32(1)
    if cutlass.const_expr(tr is not None):
        _stamp(tr, Int32(53), tq == Int32(0))
    for step in cutlass.range(NSLICE, unroll=1):
        sl = (k + Int32(1) + step) & Int32(3)
        slot = sbase + step
        _wait(kv_empty + slot, epar)
        for p in cutlass.range_constexpr(lsu_g // 4):
            q = tq + Int32(p * LSU_ARR)
            r = Int32((32 - lsu_g) * 4) + (q >> Int32(3))
            c = q & Int32(7)
            row = cutlass.max(B.lds_u32(s_idx_a + (j * Int32(TILE_N) + r) * Int32(4)), Int32(0))
            src = pool + Int64(row) * Int64(LATENT) + Int64(sl * Int32(SLICE) + c * Int32(16))
            dst = (
                s_kv_a
                + slot * Int32(SLOT_BYTES)
                + r * Int32(SLICE)
                + ((c ^ (r & Int32(7))) << Int32(4))
            )
            B.cp_async16(dst, src)
        B.cp_async_mbar_arrive_noinc(kvf_a + slot * Int32(8))
        if cutlass.const_expr(tr is not None):
            _stamp(tr, Int32(54) + step, tq == Int32(0))


@cute.jit
def role_gather_24(
    tma_swa,
    tma_cmp,
    s_kv,
    s_ext,
    s_idx,
    kv_full,
    kv_empty,
    swa_ready,
    sel_early,
    sel_ready,
    gw,
    d,
    ntiles: cutlass.Constexpr[int],
    tr=None,
    above_ptr=None,
    lsu_g: cutlass.Constexpr[int] = 0,
):
    # above_ptr (t4split): the last tile's rows below `above` (selection-certain) are gathered at the early release
    lane = cute.arch.lane_idx()
    k = d >> Int32(1)
    j0, cnt = rs24_tiles(d)
    g = lane * Int32(NGW) + gw
    for t in cutlass.range(cnt, unroll=1):
        j = j0 + t
        lo_g = Int32(0)
        if j == Int32(0):
            _wait(swa_ready, Int32(0))
        elif j == Int32(ntiles - 1):
            if cutlass.const_expr(above_ptr is not None):
                _wait(sel_early, Int32(0))
                lo_g = cutlass.max(
                    Int32(0),
                    cutlass.min(
                        Int32(TILE_N // 4),
                        (above_ptr.load() - j * Int32(TILE_N) + Int32(WINDOW)) // Int32(4),
                    ),
                )
            else:
                _wait(sel_ready, Int32(0))
        else:
            _wait(sel_early, Int32(0))
        tm = tma_cmp.get_ptr().toint()
        if j == Int32(0):
            tm = tma_swa.get_ptr().toint()
        g_t = Int32(TILE_N // 4)
        if cutlass.const_expr(lsu_g > 0):
            if j == Int32(ntiles - 1):
                g_t = Int32(TILE_N // 4 - lsu_g)
        hi_g = g_t
        if cutlass.const_expr(above_ptr is not None):
            if j == Int32(ntiles - 1):
                hi_g = cutlass.min(lo_g, g_t)
        sbase = (t & Int32(1)) * Int32(4)
        epar = ((t >> Int32(1)) & Int32(1)) ^ Int32(1)
        for step in cutlass.range_constexpr(NSLICE):
            sl = (k + Int32(1 + step)) & Int32(3)
            slot = sbase + Int32(step)
            _wait(kv_empty + slot, epar)
            if gw == Int32(0):
                if lane == Int32(0):
                    if cutlass.const_expr(lsu_g > 0):
                        prims.mbarrier_arrive_expect_tx(kv_full + slot, g_t * Int32(4 * SLICE))
                    else:
                        prims.mbarrier_arrive_expect_tx(kv_full + slot, SLOT_BYTES)
            if g < hi_g:
                r4 = B.lds_v4(Int32(s_idx.toint()) + (j * Int32(TILE_N) + g * Int32(4)) * Int32(4))
                dst = Int32(_slot_ptr(s_kv, s_ext, slot).toint()) + g * Int32(4 * SLICE)
                B.tma_g4(
                    dst,
                    tm,
                    sl * Int32(SLICE),
                    r4[0],
                    r4[1],
                    r4[2],
                    r4[3],
                    Int32(kv_full.toint()) + slot * Int32(8),
                )
        if cutlass.const_expr(above_ptr is not None):
            if j == Int32(ntiles - 1):
                _wait(sel_ready, Int32(0))
                for step in cutlass.range_constexpr(NSLICE):
                    sl = (k + Int32(1 + step)) & Int32(3)
                    slot = sbase + Int32(step)
                    if g >= hi_g and g < g_t:
                        r4 = B.lds_v4(
                            Int32(s_idx.toint()) + (j * Int32(TILE_N) + g * Int32(4)) * Int32(4)
                        )
                        dst = Int32(_slot_ptr(s_kv, s_ext, slot).toint()) + g * Int32(4 * SLICE)
                        B.tma_g4(
                            dst,
                            tm,
                            sl * Int32(SLICE),
                            r4[0],
                            r4[1],
                            r4[2],
                            r4[3],
                            Int32(kv_full.toint()) + slot * Int32(8),
                        )
        if cutlass.const_expr(tr is not None):
            _stamp(tr, Int32(TS_G) + j, gw == Int32(0) and lane == Int32(0))


@cute.jit
def _qk_24(
    tmem,
    s_q,
    s_kv,
    s_ext,
    t,
    k,
    kv_full,
    kv_empty,
    s_full,
    qdesc,
    idesc_qk,
    pfence: cutlass.Constexpr[bool] = False,
):
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    s_addr = tmem + t * Int32(TILE_N)
    sbase = (t & Int32(1)) * Int32(4)
    fpar = (t >> Int32(1)) & Int32(1)
    for step in cutlass.range_constexpr(NSLICE):
        c = (k + Int32(1 + step)) & Int32(3)
        slot = sbase + Int32(step)
        _wait(kv_full + slot, fpar)
        if cutlass.const_expr(pfence):
            # the slot may hold cp.async (generic proxy) bytes: order them before the tensor core's async-proxy reads
            B.fence_proxy_async_smem()
        kdesc = prims.Tcgen05SmemDesc.build(
            _slot_ptr(s_kv, s_ext, slot),
            leading_byte_offset=SLOT_BYTES,
            stride_byte_offset=1024,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        if prims.elect_sync():
            for kb in cutlass.range_constexpr(SLICE // 32):
                prims.tcgen05_mma(
                    prims.Tcgen05MMAKind.F8F6F4,
                    prims.CTAGroup.CTA_1,
                    prims.make_tmem_ptr(s_addr, Float32),
                    qdesc.advance_start_address(c * Int32(Q_SLICE_BYTES) + Int32(kb * 32)),
                    kdesc.advance_start_address(kb * 32),
                    idesc_qk,
                    Boolean(step > 0 or kb > 0),
                )
            if cutlass.const_expr(step < NSLICE - 1):
                prims.tcgen05_commit(kv_empty + slot, group=prims.CTAGroup.CTA_1)
    if prims.elect_sync():
        prims.tcgen05_commit(s_full + t, group=prims.CTAGroup.CTA_1)


@cute.jit
def _pv_24(tmem, s_kv, s_ext, t, p_full, kv_empty, o_full, idesc_pv):
    _wait(p_full + t, Int32(0))
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    a_addr = tmem + t * Int32(TILE_N) + Int32(TILE_N // 4) + Int32(PANEL1)
    slot = (t & Int32(1)) * Int32(4) + Int32(NSLICE - 1)
    d_addr = tmem + Int32(RS24_O_COL) + Int32(PANEL1)
    vdesc = prims.Tcgen05SmemDesc.build(
        _slot_ptr(s_kv, s_ext, slot),
        leading_byte_offset=0,
        stride_byte_offset=1024,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    acc0 = Boolean(t > Int32(0))
    if prims.elect_sync():
        for kb in cutlass.range_constexpr(TILE_N // 32):
            en = acc0 if cutlass.const_expr(kb == 0) else Boolean(True)
            prims.tcgen05_mma(
                prims.Tcgen05MMAKind.F8F6F4,
                prims.CTAGroup.CTA_1,
                prims.make_tmem_ptr(d_addr, Float32),
                prims.make_tmem_ptr(a_addr + Int32(8 * kb), E4M3),
                vdesc.advance_start_address(kb * 32 * SLICE),
                idesc_pv,
                en,
            )
        prims.tcgen05_commit(kv_empty + slot, group=prims.CTAGroup.CTA_1)
        prims.tcgen05_commit(o_full, group=prims.CTAGroup.CTA_1)


@cute.jit
def role_mma_24(
    tmem,
    s_q,
    s_kv,
    s_ext,
    q_full,
    kv_full,
    kv_empty,
    s_full,
    p_full,
    o_full,
    d,
    pv_last,
    tr=None,
    pfence: cutlass.Constexpr[bool] = False,
):
    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=Float32, a_dtype=E4M3, b_dtype=E4M3, n_dim=TILE_N, m_dim=HEADS
    )
    idesc_pv = prims.Tcgen05InstrDesc.build(
        c_dtype=Float32, a_dtype=E4M3, b_dtype=E4M3, a_major=0, b_major=1, n_dim=128, m_dim=HEADS
    )
    qdesc = prims.Tcgen05SmemDesc.build(
        s_q,
        leading_byte_offset=Q_SLICE_BYTES,
        stride_byte_offset=1024,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    lane = cute.arch.lane_idx()
    k = d >> Int32(1)
    j0, cnt = rs24_tiles(d)
    _wait(q_full, Int32(0))
    # E1d order: QK of the next tile is issued before PV of the current one (its softmax overlaps the next QK); the SWA
    # tile's PV goes right after its QK (its P is ready long before tile 1 lands)
    pvn = Int32(0)
    for t in cutlass.range(cnt, unroll=1):
        _qk_24(tmem, s_q, s_kv, s_ext, t, k, kv_full, kv_empty, s_full, qdesc, idesc_qk, pfence)
        if cutlass.const_expr(tr is not None):
            _stamp(tr, Int32(TS_QK) + j0 + t, lane == Int32(0))
        if t == Int32(0) and j0 == Int32(0) or t > pvn:
            _pv_24(tmem, s_kv, s_ext, pvn, p_full, kv_empty, o_full, idesc_pv)
            if cutlass.const_expr(tr is not None):
                _stamp(tr, Int32(TS_PV) + j0 + pvn, lane == Int32(0))
            pvn = pvn + Int32(1)
    _pv_24(tmem, s_kv, s_ext, pvn, p_full, kv_empty, o_full, idesc_pv)
    if cutlass.const_expr(tr is not None):
        _stamp(tr, Int32(TS_PV) + j0 + pvn, lane == Int32(0))
    if prims.elect_sync():
        prims.tcgen05_commit(pv_last, group=prims.CTAGroup.CTA_1)


@cute.jit
def _merge_24(
    tmem,
    tq,
    d,
    m_me,
    l_me,
    sink_ptr,
    out_addr: Int64,
    rx_s,
    stat_s,
    pvl_a,
    rx_a,
    pr_a,
    dry_a,
    scale_log2: Float32,
    dry,
    tr=None,
    f32x: cutlass.Constexpr[bool] = False,
):
    # Pair merge on the softmax warpgroup (128 threads, tq = loop-variant thread index; thread (head, half) holds O
    # columns 64c + 32 half + [0, 32) of block c). Push block g ^ 1 to the partner, combine block g with its push.
    nchunk: cutlass.Constexpr[int] = 8 if f32x else 4
    data_b: cutlass.Constexpr[int] = nchunk * 2048
    warp = tq >> Int32(5)
    lane = tq & Int32(31)
    head = warp * Int32(16) + (lane & Int32(15))
    half = lane >> Int32(4)
    lane_base = warp * Int32(32) << Int32(16)
    g = d & Int32(1)
    k = d >> Int32(1)
    wet = dry == Int32(0)
    sink = Float32((sink_ptr + head).load())
    # dry: a parity-1 wait on the private barrier returns at once
    B.wait_cl(cutlass.select_(wet, pvl_a, dry_a), cutlass.select_(wet, Int32(0), Int32(1)))
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    if cutlass.const_expr(tr is not None):
        _stamp(tr, Int32(TS_FIN), tq == Int32(0))
    o_base = tmem + lane_base + Int32(RS24_O_COL) + Int32(PANEL1)
    po = prims.tcgen05_ld(
        prims.Tcgen05LdStShape.SHAPE_16X32BX2,
        prims.make_tmem_ptr(o_base + Int32(64) * (g ^ Int32(1)), Float32),
        num=32,
        offset=Int64(32),
    )
    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
    if wet:
        cute.arch.cluster_wait()
    # the partner's selection scratch is free (its control WG arrived on our B_PR)
    B.wait_cl(cutlass.select_(wet, pr_a, dry_a), cutlass.select_(wet, Int32(0), Int32(1)))
    me = B.cluster_ctarank()
    trank = cutlass.select_(wet, me ^ Int32(1), me)
    amask = cutlass.select_(wet, Int32(-1), Int32(RS24_DRY_MASK))
    rc = B.mapa_u32(cutlass.select_(wet, rx_s, stat_s), trank)
    mc = B.mapa_u32(cutlass.select_(wet, rx_a, dry_a), trank)
    if cutlass.const_expr(f32x):
        for i in cutlass.range_constexpr(8):
            B.st_async_v4(
                rc + (((Int32(i * 128) + tq) * Int32(16)) & amask),
                mc,
                LS.u32_of_f32(Float32(po[4 * i])),
                LS.u32_of_f32(Float32(po[4 * i + 1])),
                LS.u32_of_f32(Float32(po[4 * i + 2])),
                LS.u32_of_f32(Float32(po[4 * i + 3])),
            )
    else:
        for i in cutlass.range_constexpr(4):
            B.st_async_v4(
                rc + (((Int32(i * 128) + tq) * Int32(16)) & amask),
                mc,
                _pack_bf16x2(Float32(po[8 * i]), Float32(po[8 * i + 1])),
                _pack_bf16x2(Float32(po[8 * i + 2]), Float32(po[8 * i + 3])),
                _pack_bf16x2(Float32(po[8 * i + 4]), Float32(po[8 * i + 5])),
                _pack_bf16x2(Float32(po[8 * i + 6]), Float32(po[8 * i + 7])),
            )
    B.st_async_v2(
        rc + ((Int32(data_b) + tq * Int32(8)) & amask), mc, LS.u32_of_f32(m_me), LS.u32_of_f32(l_me)
    )
    if cutlass.const_expr(tr is not None):
        _stamp(tr, Int32(121), tq == Int32(0))
    own = prims.tcgen05_ld(
        prims.Tcgen05LdStShape.SHAPE_16X32BX2,
        prims.make_tmem_ptr(o_base + Int32(64) * g, Float32),
        num=32,
        offset=Int64(32),
    )
    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    # the partner's push landed (B_RX expects its bytes from kernel start); dry: our own dry stores landed
    B.wait_cl(cutlass.select_(wet, rx_a, dry_a), Int32(0))
    if cutlass.const_expr(tr is not None):
        _stamp(tr, Int32(123), tq == Int32(0))
    lb = cutlass.select_(wet, rx_s, stat_s)
    xs = []
    for i in cutlass.range_constexpr(nchunk):
        xs.append(B.lds_v4(lb + (((Int32(i * 128) + tq) * Int32(16)) & amask)))
    sa = lb + ((Int32(data_b) + tq * Int32(8)) & amask)
    m_p = LS.f32_of_u32(Uint32(B.lds_u32(sa)))
    l_p = LS.f32_of_u32(Uint32(B.lds_u32(sa + Int32(4))))
    m = cute.math.max(m_me, m_p, ftz=True)
    w_me = cute.math.exp2(scale_log2 * (m_me - m), fastmath=True)
    w_p = cute.math.exp2(scale_log2 * (m_p - m), fastmath=True)
    l_sum = w_me * l_me + w_p * l_p
    acc = []
    for i in cutlass.range_constexpr(32):
        acc.append(w_me * Float32(own[i]))
    if cutlass.const_expr(f32x):
        for i in cutlass.range_constexpr(8):
            for e in cutlass.range_constexpr(4):
                acc[4 * i + e] = acc[4 * i + e] + w_p * LS.f32_of_u32(Uint32(xs[i][e]))
    else:
        for i in cutlass.range_constexpr(4):
            for e in cutlass.range_constexpr(4):
                wd = Uint32(xs[i][e])
                acc[8 * i + 2 * e] = acc[8 * i + 2 * e] + w_p * LS.f32_of_u32(wd << Uint32(16))
                acc[8 * i + 2 * e + 1] = acc[8 * i + 2 * e + 1] + w_p * LS.f32_of_u32(
                    wd & Uint32(4294901760)
                )
    norm = _final_norm(m, l_sum, sink, scale_log2)
    ob = []
    for pi in cutlass.range_constexpr(16):
        fp = cute.arch.mul_packed_f32x2(
            (norm, norm), (acc[2 * pi], acc[2 * pi + 1]), ftz=False, rnd="rn"
        )
        ob.append(_pack_bf16x2(fp[0], fp[1]))
    if cutlass.const_expr(tr is not None):
        _stamp(tr, Int32(124), tq == Int32(0))
    # dry: the same stores with throw-away values (this CTA's own output block; the real pass overwrites it)
    rowb = out_addr + Int64(
        (head * Int32(LATENT) + k * Int32(128) + g * Int32(64) + half * Int32(32)) * Int32(2)
    )
    for ch in cutlass.range_constexpr(4):
        B.st_g_v4(rowb + Int64(ch * 16), ob[ch * 4], ob[ch * 4 + 1], ob[ch * 4 + 2], ob[ch * 4 + 3])
    if cutlass.const_expr(tr is not None):
        _stamp(tr, Int32(TS_STORED), tq == Int32(0))


@cute.jit
def _merge_24a(
    tmem,
    tq,
    d,
    m_me,
    l_me,
    sink_ptr,
    out_addr: Int64,
    rx_s,
    stat_s,
    pvl_a,
    rx_a,
    pr_a,
    dry_a,
    scale_log2: Float32,
    dry,
    tr=None,
):
    # Asymmetric pair merge (rs24_asym): row group 0 (tiles 0-2, no boundary-rank dependency; it finishes first) pushes
    # its whole 128-column partial O (bf16) + (max, sum) to row group 1, which combines and stores all 128 columns of
    # the V group. No DSMEM round trip after the last PV. Thread (head, half) holds columns 64c + 32 half + [0, 32).
    warp = tq >> Int32(5)
    lane = tq & Int32(31)
    head = warp * Int32(16) + (lane & Int32(15))
    half = lane >> Int32(4)
    lane_base = warp * Int32(32) << Int32(16)
    g = d & Int32(1)
    k = d >> Int32(1)
    wet = dry == Int32(0)
    B.wait_cl(cutlass.select_(wet, pvl_a, dry_a), cutlass.select_(wet, Int32(0), Int32(1)))
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    if cutlass.const_expr(tr is not None):
        _stamp(tr, Int32(TS_FIN), tq == Int32(0))
    o_base = tmem + lane_base + Int32(RS24_O_COL) + Int32(PANEL1)
    o0 = prims.tcgen05_ld(
        prims.Tcgen05LdStShape.SHAPE_16X32BX2,
        prims.make_tmem_ptr(o_base, Float32),
        num=32,
        offset=Int64(32),
    )
    o1 = prims.tcgen05_ld(
        prims.Tcgen05LdStShape.SHAPE_16X32BX2,
        prims.make_tmem_ptr(o_base + Int32(64), Float32),
        num=32,
        offset=Int64(32),
    )
    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    amask = cutlass.select_(wet, Int32(-1), Int32(RS24_DRY_MASK))
    if g == Int32(0):
        if wet:
            cute.arch.cluster_wait()
        B.wait_cl(cutlass.select_(wet, pr_a, dry_a), cutlass.select_(wet, Int32(0), Int32(1)))
        me = B.cluster_ctarank()
        trank = cutlass.select_(wet, me ^ Int32(1), me)
        rc = B.mapa_u32(cutlass.select_(wet, rx_s, stat_s), trank)
        mc = B.mapa_u32(cutlass.select_(wet, rx_a, dry_a), trank)
        for i in cutlass.range_constexpr(4):
            B.st_async_v4(
                rc + (((Int32(i * 128) + tq) * Int32(16)) & amask),
                mc,
                _pack_bf16x2(Float32(o0[8 * i]), Float32(o0[8 * i + 1])),
                _pack_bf16x2(Float32(o0[8 * i + 2]), Float32(o0[8 * i + 3])),
                _pack_bf16x2(Float32(o0[8 * i + 4]), Float32(o0[8 * i + 5])),
                _pack_bf16x2(Float32(o0[8 * i + 6]), Float32(o0[8 * i + 7])),
            )
        for i in cutlass.range_constexpr(4):
            B.st_async_v4(
                rc + (((Int32((i + 4) * 128) + tq) * Int32(16)) & amask),
                mc,
                _pack_bf16x2(Float32(o1[8 * i]), Float32(o1[8 * i + 1])),
                _pack_bf16x2(Float32(o1[8 * i + 2]), Float32(o1[8 * i + 3])),
                _pack_bf16x2(Float32(o1[8 * i + 4]), Float32(o1[8 * i + 5])),
                _pack_bf16x2(Float32(o1[8 * i + 6]), Float32(o1[8 * i + 7])),
            )
        B.st_async_v2(
            rc + ((Int32(8 * 2048) + tq * Int32(8)) & amask),
            mc,
            LS.u32_of_f32(m_me),
            LS.u32_of_f32(l_me),
        )
        if cutlass.const_expr(tr is not None):
            _stamp(tr, Int32(121), tq == Int32(0))
            _stamp(tr, Int32(TS_STORED), tq == Int32(0))
    else:
        sink = Float32((sink_ptr + head).load())
        # the partner's push landed (B_RX expects it from kernel start); dry: a parity-1 wait returns at once
        B.wait_cl(cutlass.select_(wet, rx_a, dry_a), cutlass.select_(wet, Int32(0), Int32(1)))
        if cutlass.const_expr(tr is not None):
            _stamp(tr, Int32(123), tq == Int32(0))
        lb = cutlass.select_(wet, rx_s, stat_s)
        xs = []
        for i in cutlass.range_constexpr(8):
            xs.append(B.lds_v4(lb + (((Int32(i * 128) + tq) * Int32(16)) & amask)))
        sa = lb + ((Int32(8 * 2048) + tq * Int32(8)) & amask)
        m_p = LS.f32_of_u32(Uint32(B.lds_u32(sa)))
        l_p = LS.f32_of_u32(Uint32(B.lds_u32(sa + Int32(4))))
        m = cute.math.max(m_me, m_p, ftz=True)
        w_me = cute.math.exp2(scale_log2 * (m_me - m), fastmath=True)
        w_p = cute.math.exp2(scale_log2 * (m_p - m), fastmath=True)
        norm = _final_norm(m, w_me * l_me + w_p * l_p, sink, scale_log2)
        wn_me = w_me * norm
        wn_p = w_p * norm
        for c in cutlass.range_constexpr(2):
            oc = o0 if cutlass.const_expr(c == 0) else o1
            ob = []
            for i in cutlass.range_constexpr(4):
                for e in cutlass.range_constexpr(4):
                    wd = Uint32(xs[4 * c + i][e])
                    a0 = wn_me * Float32(oc[8 * i + 2 * e]) + wn_p * LS.f32_of_u32(wd << Uint32(16))
                    a1 = wn_me * Float32(oc[8 * i + 2 * e + 1]) + wn_p * LS.f32_of_u32(
                        wd & Uint32(4294901760)
                    )
                    ob.append(_pack_bf16x2(a0, a1))
            # dry: the same stores with throw-away values (this CTA's own output block; the real pass overwrites it)
            rowb = out_addr + Int64(
                (head * Int32(LATENT) + k * Int32(128) + Int32(64 * c) + half * Int32(32))
                * Int32(2)
            )
            for ch in cutlass.range_constexpr(4):
                B.st_g_v4(
                    rowb + Int64(ch * 16),
                    ob[ch * 4],
                    ob[ch * 4 + 1],
                    ob[ch * 4 + 2],
                    ob[ch * 4 + 3],
                )
        if cutlass.const_expr(tr is not None):
            _stamp(tr, Int32(TS_STORED), tq == Int32(0))


@cute.jit
def role_softmax_24(
    tmem,
    s_full,
    p_full,
    scale_log2: Float32,
    swa_len,
    n_sel,
    d,
    o_full,
    sink_ptr,
    out_addr: Int64,
    rx_s,
    stat_s,
    pvl_a,
    rx_a,
    pr_a,
    dry_a,
    zaddr,
    tr=None,
    full_tiles: cutlass.Constexpr[bool] = False,
    dry_merge: cutlass.Constexpr[bool] = True,
    f32x: cutlass.Constexpr[bool] = False,
    asym: cutlass.Constexpr[bool] = False,
):
    # Softmax of this CTA's tiles, then the pair merge. With dry_merge the merge first runs dry at the same call site
    # in the idle window (g == 0: after the SWA tile; g == 1: before its first tile) so the real merge runs warm.
    half_n: cutlass.Constexpr[int] = TILE_N // 2
    npk: cutlass.Constexpr[int] = TILE_N // 8
    tid0 = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % Int32(4)) * Int32(
        32
    ) + cute.arch.lane_idx()
    B.sts_u32(zaddr, Int32(0))
    j0, cnt = rs24_tiles(d)
    t_pre = Int32(1) - (d & Int32(1))
    ps0 = Int32(1)
    if cutlass.const_expr(dry_merge):
        ps0 = Int32(0)
    m_run = Float32(NEG_FLT_MAX)
    l_run = Float32(0.0)
    for ps in cutlass.range(ps0, 2, unroll=1):
        # loop-variant thread index: keeps LLVM from hoisting the per-thread math in front of this loop
        tq = tid0 + _lds_vol(zaddr)
        warp = tq >> Int32(5)
        lane = tq & Int32(31)
        lane_base = warp * Int32(32) << Int32(16)
        col0 = (lane >> Int32(4)) * Int32(half_n)
        tb = Int32(0)
        te = cnt
        if cutlass.const_expr(dry_merge):
            te = t_pre
            if ps == Int32(1):
                tb = t_pre
                te = cnt
        for t in cutlass.range(tb, te, unroll=1):
            j = j0 + t
            _wait(s_full + t, Int32(0))
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            s_addr = tmem + lane_base + t * Int32(TILE_N)
            s = prims.tcgen05_ld(
                prims.Tcgen05LdStShape.SHAPE_16X32BX2,
                prims.make_tmem_ptr(s_addr, Float32),
                num=half_n,
                offset=Int64(half_n),
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            limit = Int32(WINDOW) + n_sel
            if j == Int32(0):
                limit = swa_len
            nv_here = limit - (j * Int32(TILE_N) + col0)
            old_max = m_run
            mx0 = old_max
            mx1 = Float32(NEG_FLT_MAX)
            mx2 = Float32(NEG_FLT_MAX)
            mx3 = Float32(NEG_FLT_MAX)
            sv = []
            for i in cutlass.range_constexpr(half_n):
                if cutlass.const_expr(full_tiles):
                    sv.append(Float32(s[i]))
                else:
                    sv.append(
                        cutlass.select_(nv_here > Int32(i), Float32(s[i]), Float32(NEG_FLT_MAX))
                    )
            for i in cutlass.range_constexpr(0, half_n, 4):
                mx0 = cute.math.max(mx0, sv[i], ftz=True)
                mx1 = cute.math.max(mx1, sv[i + 1], ftz=True)
                mx2 = cute.math.max(mx2, sv[i + 2], ftz=True)
                mx3 = cute.math.max(mx3, sv[i + 3], ftz=True)
            lmax = cute.math.max(
                cute.math.max(mx0, mx1, ftz=True), cute.math.max(mx2, mx3, ftz=True), ftz=True
            )
            lmax = cute.math.max(lmax, _shfl_xor_f32(lmax, 16), ftz=True)
            new_max = lmax
            safe_max = new_max
            if safe_max == Float32(NEG_FLT_MAX):
                safe_max = Float32(0.0)
            bias = -scale_log2 * safe_max + Float32(LOG2_448)
            sm0 = Float32(0.0)
            sm1 = Float32(0.0)
            sm2 = Float32(0.0)
            sm3 = Float32(0.0)
            regs = []
            for pk in cutlass.range_constexpr(npk):
                x01 = cute.arch.fma_packed_f32x2(
                    (sv[4 * pk], sv[4 * pk + 1]),
                    (scale_log2, scale_log2),
                    (bias, bias),
                    ftz=False,
                    rnd="rn",
                )
                x23 = cute.arch.fma_packed_f32x2(
                    (sv[4 * pk + 2], sv[4 * pk + 3]),
                    (scale_log2, scale_log2),
                    (bias, bias),
                    ftz=False,
                    rnd="rn",
                )
                p0 = cute.math.exp2(x01[0], fastmath=True)
                p1 = cute.math.exp2(x01[1], fastmath=True)
                p2 = cute.math.exp2(x23[0], fastmath=True)
                p3 = cute.math.exp2(x23[1], fastmath=True)
                sm0, sm1 = cute.arch.add_packed_f32x2((sm0, sm1), (p0, p1), ftz=False, rnd="rn")
                sm2, sm3 = cute.arch.add_packed_f32x2((sm2, sm3), (p2, p3), ftz=False, rnd="rn")
                regs.append(_pack4_e4m3(p0, p1, p2, p3))
            local_sum = sm0 + sm1 + (sm2 + sm3)
            p_addr = s_addr + Int32(TILE_N // 4)
            _tmem_st_p_fp8(p_addr + Int32(PANEL1), regs, npk)
            prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            exp_scale = cute.math.exp2(scale_log2 * (old_max - new_max), fastmath=True)
            l_run = exp_scale * l_run + local_sum
            m_run = new_max
            if t > Int32(0):
                # O holds PV of the previous tile: rescale its 128 columns once
                _wait(o_full, t - Int32(1) & Int32(1))
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                ident = old_max == new_max
                cscale = Float32(1.0)
                if not ident:
                    cscale = cute.math.exp2(scale_log2 * (old_max - new_max), fastmath=True)
                skip = prims.vote_sync(4294967295, ident, prims.VoteSync.ALL)
                if not skip:
                    for ch in cutlass.range_constexpr(2):
                        ptr = prims.make_tmem_ptr(
                            tmem + lane_base + Int32(RS24_O_COL + 64 * ch) + Int32(PANEL1), Float32
                        )
                        o = prims.tcgen05_ld(
                            prims.Tcgen05LdStShape.SHAPE_16X32BX2, ptr, num=32, offset=Int64(32)
                        )
                        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                        vals = []
                        for rr in cutlass.range_constexpr(0, 32, 2):
                            pr = cute.arch.mul_packed_f32x2(
                                (Float32(o[rr]), Float32(o[rr + 1])),
                                (cscale, cscale),
                                ftz=False,
                                rnd="rn",
                            )
                            vals.append(pr[0])
                            vals.append(pr[1])
                        prims.tcgen05_st(
                            prims.Tcgen05LdStShape.SHAPE_16X32BX2,
                            ptr,
                            cutlass.Vector.from_elements(tuple(vals), Float32),
                            offset=Int64(32),
                        )
                        prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            cute.arch.sync_warp()
            if prims.elect_sync():
                prims.mbarrier_arrive(p_full + t)
            if cutlass.const_expr(tr is not None):
                _stamp(tr, Int32(TS_SM) + j, warp == Int32(0) and lane == Int32(0))
        l_tot = l_run + _shfl_xor_f32(l_run, 16)
        if cutlass.const_expr(asym):
            _merge_24a(
                tmem,
                tq,
                d,
                m_run,
                l_tot,
                sink_ptr,
                out_addr,
                rx_s,
                stat_s,
                pvl_a,
                rx_a,
                pr_a,
                dry_a,
                scale_log2,
                Int32(1) - ps,
                tr,
            )
        else:
            _merge_24(
                tmem,
                tq,
                d,
                m_run,
                l_tot,
                sink_ptr,
                out_addr,
                rx_s,
                stat_s,
                pvl_a,
                rx_a,
                pr_a,
                dry_a,
                scale_log2,
                Int32(1) - ps,
                tr,
                f32x,
            )
