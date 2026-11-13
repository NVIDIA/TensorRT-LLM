# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA building blocks shared by the attention roles: Q / KV tensor maps, tcgen05 MMA and softmax helpers."""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import torch
from cutlass import Boolean, Float32, Int32, Int64
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cutlass_dsl import T as _T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

HEADS = 64
LATENT = 512
WINDOW = 128
TOPK = 512
NROWS = WINDOW + TOPK
SLICE = 128
NSLICE = LATENT // SLICE
Q_SLICE_BYTES = HEADS * SLICE
Q_BYTES = HEADS * LATENT
LOG2_E = 1.4426950408889634
LOG2_448 = 8.8073549
NEG_FLT_MAX = -3.4028235e38
PANEL1 = 16 << 16
TMEM_COLS = 512
TMEM_O = 256
REG_SOFTMAX = 184
REG_CORR = 232
REG_OTHER = 48
W_CORR0 = 4
W_MMA = 8
W_QLOAD = 9
W_GATHER0 = 10
NGW = 4
THREADS = 512
E4M3 = cutlass.Float8E4M3FN
BF16 = cutlass.BFloat16


@cute.jit
def _wait(mbar, phase):
    while not prims.mbarrier_try_wait_parity(mbar, phase):
        pass


@cute.jit
def _tma_gather4(smem_dst, tma_desc, mbar, col, r0, r1, r2, r3) -> None:
    bar = prims.cvta_to(mbar, prims.CvtaSpace.SHARED)
    prims.inline_ptx_hl(
        "{\n\t.reg .pred leader;\n\telect.sync _|leader, -1;\n\t@leader "
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4.mbarrier::complete_tx::bytes [{$r0}], [{$r1}, "
        "{{$r2}, {$r3}, {$r4}, {$r5}, {$r6}}], [{$r7}];\n\t}",
        read_only_args=[smem_dst, tma_desc, col, r0, r1, r2, r3, bar],
    )


@cute.jit
def _pack4_e4m3(v0: Float32, v1: Float32, v2: Float32, v3: Float32):
    return cute.arch.inline_ptx(
        "{\n  .reg .b16 lo;\n  .reg .b16 hi;\n  cvt.rn.satfinite.e4m3x2.f32 lo, {$r1}, {$r0};\n  "
        "cvt.rn.satfinite.e4m3x2.f32 hi, {$r3}, {$r2};\n  mov.b32 {$w0}, {lo, hi};\n}",
        write_only_types=[Int32],
        read_only_args=[v0, v1, v2, v3],
    )


@cute.jit
def _tmem_st_p_fp8(tmem_addr, regs, n: cutlass.Constexpr[int]) -> None:
    body = ", ".join(("{$r%d}" % (i + 1) for i in range(n)))
    cute.arch.inline_ptx(
        f"tcgen05.st.sync.aligned.16x32bx2.x{n}.b32 [{{$r0}}], {n}, {{ {body} }};",
        read_only_args=[tmem_addr] + [regs[i] for i in range(n)],
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


@dsl_user_op
def _gtimer(*, loc=None, ip=None):
    return Int64(
        _llvm.inline_asm(
            _T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _stamp(tr, slot, cond) -> None:
    if cond:
        t = _gtimer()
        cutlass.inttoptr(tr + Int64(slot) * Int64(8), mem_space=1, dtype=Int64).store(t)


@cute.jit
def _pack_bf16x2(v0: Float32, v1: Float32):
    return cutlass.Vector.from_elements((v0, v1), Float32).to(BF16).bitcast(Int32)[0]


@cute.jit
def role_q_load(tma_q, s_q, q_full, q_empty, row0, row_step, n_items):
    for it in cutlass.range(n_items, unroll=1):
        b = row0 + it * row_step
        if it > Int32(0):
            _wait(q_empty, it - Int32(1) & Int32(1))
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(q_full, Q_BYTES)
            for c in cutlass.range_constexpr(NSLICE):
                prims.cp_async_bulk_tensor_shared_cta_global(
                    s_q + c * Q_SLICE_BYTES, tma_q.get_ptr(), [c * SLICE, 0, b], q_full
                )


@cute.jit
def role_gather(
    tma_swa,
    tma_cmp,
    s_kv,
    idx_ptr,
    kv_full,
    kv_empty,
    gwarp,
    row0,
    row_step,
    n_items,
    lb,
    tile_n: cutlass.Constexpr[int],
    ntiles: cutlass.Constexpr[int],
    nslots: cutlass.Constexpr[int],
    tr=None,
):
    R: cutlass.Constexpr[int] = tile_n // NGW
    per_warp: cutlass.Constexpr[int] = R // 4
    slot_bytes: cutlass.Constexpr[int] = tile_n * SLICE
    lane = cute.arch.lane_idx()
    total = n_items * Int32(ntiles)
    nxt = Int32(0)
    b0 = row0
    pos0 = lb + gwarp * Int32(R) + lane
    if lane < Int32(R):
        nxt = (idx_ptr + b0 * Int32(NROWS) + pos0).load()
    for g in cutlass.range(total, unroll=1):
        it = g // Int32(ntiles)
        j = g - it * Int32(ntiles)
        cur = nxt
        list_start = lb + j * Int32(tile_n)
        g1 = g + Int32(1)
        if g1 < total:
            it1 = g1 // Int32(ntiles)
            j1 = g1 - it1 * Int32(ntiles)
            b1 = row0 + it1 * row_step
            if lane < Int32(R):
                nxt = (
                    idx_ptr + b1 * Int32(NROWS) + lb + j1 * Int32(tile_n) + gwarp * Int32(R) + lane
                ).load()
        use_swa = list_start < Int32(WINDOW)
        for c in cutlass.range_constexpr(NSLICE):
            k = g * Int32(NSLICE) + Int32(c)
            slot = k % Int32(nslots)
            ph = k // Int32(nslots) & Int32(1)
            _wait(kv_empty + slot, ph ^ Int32(1))
            if gwarp == Int32(0):
                if prims.elect_sync():
                    prims.mbarrier_arrive_expect_tx(kv_full + slot, slot_bytes)
            dst0 = s_kv + slot * Int32(slot_bytes) + gwarp * Int32(R * SLICE)
            for q in cutlass.range_constexpr(per_warp):
                r0 = _shfl_idx_i32(cur, Int32(4 * q))
                r1 = _shfl_idx_i32(cur, Int32(4 * q + 1))
                r2 = _shfl_idx_i32(cur, Int32(4 * q + 2))
                r3 = _shfl_idx_i32(cur, Int32(4 * q + 3))
                if use_swa:
                    _tma_gather4(
                        dst0 + Int32(4 * q * SLICE),
                        tma_swa.get_ptr(),
                        kv_full + slot,
                        Int32(c * SLICE),
                        r0,
                        r1,
                        r2,
                        r3,
                    )
                else:
                    _tma_gather4(
                        dst0 + Int32(4 * q * SLICE),
                        tma_cmp.get_ptr(),
                        kv_full + slot,
                        Int32(c * SLICE),
                        r0,
                        r1,
                        r2,
                        r3,
                    )
        if cutlass.const_expr(tr is not None):
            _stamp(
                tr, Int32(10) + j, g < Int32(ntiles) and gwarp == Int32(0) and (lane == Int32(0))
            )
            _stamp(
                tr,
                Int32(100) + it * Int32(8) + j,
                it < Int32(12) and gwarp == Int32(0) and (lane == Int32(0)),
            )


@cute.jit
def _pv(
    tmem,
    s_kv,
    g,
    it,
    j,
    p_full,
    o_corr,
    o_free,
    kv_empty,
    s_empty,
    o_full,
    idesc_pv,
    tile_n: cutlass.Constexpr[int],
    ntiles: cutlass.Constexpr[int],
    nslots: cutlass.Constexpr[int],
    v_lo=0,
    nv: cutlass.Constexpr[int] = NSLICE,
):
    slot_bytes: cutlass.Constexpr[int] = tile_n * SLICE
    st = g & Int32(1)
    _wait(p_full + st, g >> Int32(1) & Int32(1))
    if j > Int32(0):
        _wait(o_corr, it * Int32(ntiles - 1) + j - Int32(1) & Int32(1))
    elif it > Int32(0):
        _wait(o_free, it - Int32(1) & Int32(1))
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    acc = j > Int32(0)
    for vv in cutlass.range_constexpr(nv):
        v = Int32(v_lo) + Int32(vv)
        k = g * Int32(NSLICE) + v
        slot = k % Int32(nslots)
        d_addr = tmem + Int32(TMEM_O) + v % Int32(2) * Int32(128) + (v >> Int32(1)) * Int32(PANEL1)
        a_addr = tmem + st * Int32(tile_n) + Int32(tile_n // 4) + (v >> Int32(1)) * Int32(PANEL1)
        vdesc = prims.Tcgen05SmemDesc.build(
            s_kv + slot * Int32(slot_bytes),
            leading_byte_offset=0,
            stride_byte_offset=1024,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        if prims.elect_sync():
            for kb in cutlass.range_constexpr(tile_n // 32):
                en = acc if cutlass.const_expr(kb == 0) else Boolean(True)
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
    if prims.elect_sync():
        prims.tcgen05_commit(s_empty + st, group=prims.CTAGroup.CTA_1)
        prims.tcgen05_commit(o_full, group=prims.CTAGroup.CTA_1)


@cute.jit
def role_mma(
    tmem,
    s_q,
    s_kv,
    q_full,
    q_empty,
    kv_full,
    kv_empty,
    s_full,
    s_empty,
    p_full,
    o_corr,
    o_free,
    o_full,
    n_items,
    tile_n: cutlass.Constexpr[int],
    ntiles: cutlass.Constexpr[int],
    nslots: cutlass.Constexpr[int],
    tr=None,
    v_lo=0,
    nv: cutlass.Constexpr[int] = NSLICE,
):
    slot_bytes: cutlass.Constexpr[int] = tile_n * SLICE
    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=Float32, a_dtype=E4M3, b_dtype=E4M3, n_dim=tile_n, m_dim=HEADS
    )
    idesc_pv = prims.Tcgen05InstrDesc.build(
        c_dtype=Float32, a_dtype=E4M3, b_dtype=E4M3, a_major=0, b_major=1, n_dim=SLICE, m_dim=HEADS
    )
    qdesc = prims.Tcgen05SmemDesc.build(
        s_q,
        leading_byte_offset=Q_SLICE_BYTES,
        stride_byte_offset=1024,
        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
    )
    lane = cute.arch.lane_idx()
    for it in cutlass.range(n_items, unroll=1):
        _wait(q_full, it & Int32(1))
        if cutlass.const_expr(tr is not None):
            _stamp(tr, Int32(90), it == Int32(0) and lane == Int32(0))
            _stamp(tr, Int32(180) + it, it < Int32(12) and lane == Int32(0))
        for j in cutlass.range_constexpr(ntiles):
            g = it * Int32(ntiles) + Int32(j)
            st = g & Int32(1)
            if g >= Int32(2):
                _wait(s_empty + st, (g >> Int32(1)) - Int32(1) & Int32(1))
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            s_addr = tmem + st * Int32(tile_n)
            for c in cutlass.range_constexpr(NSLICE):
                k = g * Int32(NSLICE) + Int32(c)
                slot = k % Int32(nslots)
                _wait(kv_full + slot, k // Int32(nslots) & Int32(1))
                kdesc = prims.Tcgen05SmemDesc.build(
                    s_kv + slot * Int32(slot_bytes),
                    leading_byte_offset=slot_bytes,
                    stride_byte_offset=1024,
                    layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
                )
                if prims.elect_sync():
                    for kb in cutlass.range_constexpr(SLICE // 32):
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F8F6F4,
                            prims.CTAGroup.CTA_1,
                            prims.make_tmem_ptr(s_addr, Float32),
                            qdesc.advance_start_address(c * Q_SLICE_BYTES + kb * 32),
                            kdesc.advance_start_address(kb * 32),
                            idesc_qk,
                            Boolean(c > 0 or kb > 0),
                        )
            if cutlass.const_expr(tr is not None):
                _stamp(tr, Int32(20 + j), it == Int32(0) and lane == Int32(0))
            if prims.elect_sync():
                prims.tcgen05_commit(s_full + st, group=prims.CTAGroup.CTA_1)
                if cutlass.const_expr(j == ntiles - 1):
                    prims.tcgen05_commit(q_empty, group=prims.CTAGroup.CTA_1)
            if cutlass.const_expr(j >= 1):
                _pv(
                    tmem,
                    s_kv,
                    g - Int32(1),
                    it,
                    Int32(j - 1),
                    p_full,
                    o_corr,
                    o_free,
                    kv_empty,
                    s_empty,
                    o_full,
                    idesc_pv,
                    tile_n,
                    ntiles,
                    nslots,
                    v_lo,
                    nv,
                )
                if cutlass.const_expr(tr is not None):
                    _stamp(tr, Int32(70 + j - 1), it == Int32(0) and lane == Int32(0))
        _pv(
            tmem,
            s_kv,
            it * Int32(ntiles) + Int32(ntiles - 1),
            it,
            Int32(ntiles - 1),
            p_full,
            o_corr,
            o_free,
            kv_empty,
            s_empty,
            o_full,
            idesc_pv,
            tile_n,
            ntiles,
            nslots,
            v_lo,
            nv,
        )
        if cutlass.const_expr(tr is not None):
            _stamp(tr, Int32(70 + ntiles - 1), it == Int32(0) and lane == Int32(0))
            _stamp(tr, Int32(220) + it, it < Int32(12) and lane == Int32(0))


@cute.jit
def role_softmax(
    tmem,
    s_full,
    p_full,
    s_stat,
    s_fin,
    fin_bar,
    scale_log2: Float32,
    n_items,
    tile_n: cutlass.Constexpr[int],
    ntiles: cutlass.Constexpr[int],
    tr=None,
):
    half_n: cutlass.Constexpr[int] = tile_n // 2
    npk: cutlass.Constexpr[int] = tile_n // 8
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % Int32(4)
    lane = cute.arch.lane_idx()
    head = warp * Int32(16) + (lane & Int32(15))
    lane_base = warp * Int32(32) << Int32(16)
    for it in cutlass.range(n_items, unroll=1):
        m_run = Float32(NEG_FLT_MAX)
        l_run = Float32(0.0)
        for j in cutlass.range(ntiles, unroll=1):
            g = it * Int32(ntiles) + j
            st = g & Int32(1)
            _wait(s_full + st, g >> Int32(1) & Int32(1))
            if cutlass.const_expr(tr is not None):
                _stamp(
                    tr, Int32(40) + j, it == Int32(0) and warp == Int32(0) and (lane == Int32(0))
                )
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            s_addr = tmem + lane_base + st * Int32(tile_n)
            s = prims.tcgen05_ld(
                prims.Tcgen05LdStShape.SHAPE_16X32BX2,
                prims.make_tmem_ptr(s_addr, Float32),
                num=half_n,
                offset=Int64(half_n),
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            old_max = m_run
            mx0 = old_max
            mx1 = Float32(NEG_FLT_MAX)
            mx2 = Float32(NEG_FLT_MAX)
            mx3 = Float32(NEG_FLT_MAX)
            for i in cutlass.range_constexpr(0, half_n, 4):
                mx0 = cute.math.max(mx0, Float32(s[i]), ftz=True)
                mx1 = cute.math.max(mx1, Float32(s[i + 1]), ftz=True)
                mx2 = cute.math.max(mx2, Float32(s[i + 2]), ftz=True)
                mx3 = cute.math.max(mx3, Float32(s[i + 3]), ftz=True)
            lmax = cute.math.max(
                cute.math.max(mx0, mx1, ftz=True), cute.math.max(mx2, mx3, ftz=True), ftz=True
            )
            lmax = cute.math.max(lmax, _shfl_xor_f32(lmax, 16), ftz=True)
            new_max = lmax
            safe_max = new_max
            if safe_max == Float32(NEG_FLT_MAX):
                safe_max = Float32(0.0)
            neg_scaled = -scale_log2 * safe_max
            bias = neg_scaled + Float32(LOG2_448)
            sm0 = Float32(0.0)
            sm1 = Float32(0.0)
            sm2 = Float32(0.0)
            sm3 = Float32(0.0)
            regs = []
            for pk in cutlass.range_constexpr(npk):
                x01 = cute.arch.fma_packed_f32x2(
                    (Float32(s[4 * pk]), Float32(s[4 * pk + 1])),
                    (scale_log2, scale_log2),
                    (bias, bias),
                    ftz=False,
                    rnd="rn",
                )
                x23 = cute.arch.fma_packed_f32x2(
                    (Float32(s[4 * pk + 2]), Float32(s[4 * pk + 3])),
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
            p_addr = s_addr + Int32(tile_n // 4)
            _tmem_st_p_fp8(p_addr, regs, npk)
            _tmem_st_p_fp8(p_addr + Int32(PANEL1), regs, npk)
            prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            exp_scale = cute.math.exp2(scale_log2 * (old_max - new_max), fastmath=True)
            l_run = exp_scale * l_run + local_sum
            m_run = new_max
            if lane < Int32(16):
                (s_stat + st * Int32(128) + head).store(old_max)
                (s_stat + st * Int32(128) + Int32(64) + head).store(new_max)
            cute.arch.sync_warp()
            if prims.elect_sync():
                prims.mbarrier_arrive(p_full + st)
            if cutlass.const_expr(tr is not None):
                _stamp(
                    tr, Int32(50) + j, it == Int32(0) and warp == Int32(0) and (lane == Int32(0))
                )
        l_tot = l_run + _shfl_xor_f32(l_run, 16)
        fb = (it & Int32(1)) * Int32(128)
        if lane < Int32(16):
            (s_fin + fb + head).store(m_run)
            (s_fin + fb + Int32(64) + head).store(l_tot)
        cute.arch.sync_warp()
        if prims.elect_sync():
            prims.mbarrier_arrive(fin_bar)


@cute.jit
def _rescale_o(tmem, lane_base, scale: Float32):
    for v in cutlass.range_constexpr(NSLICE):
        o_addr = tmem + lane_base + Int32(TMEM_O + v % 2 * 128 + v // 2 * PANEL1)
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
def _correct_tile(
    tmem,
    lane_base,
    head,
    p_full,
    o_full,
    o_corr,
    s_stat,
    g,
    it,
    j,
    scale_log2: Float32,
    ntiles: cutlass.Constexpr[int],
):
    st = g & Int32(1)
    _wait(p_full + st, g >> Int32(1) & Int32(1))
    old_max = (s_stat + st * Int32(128) + head).load()
    new_max = (s_stat + st * Int32(128) + Int32(64) + head).load()
    _wait(o_full, g - Int32(1) & Int32(1))
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    ident = old_max == new_max
    scale = Float32(1.0)
    if not ident:
        scale = cute.math.exp2(scale_log2 * (old_max - new_max), fastmath=True)
    skip = prims.vote_sync(4294967295, ident, prims.VoteSync.ALL)
    if not skip:
        _rescale_o(tmem, lane_base, scale)
    prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.sync_warp()
    if prims.elect_sync():
        prims.mbarrier_arrive(o_corr)


@cute.jit
def _final_norm(
    final_max: Float32, reduced_sum: Float32, sink: Float32, scale_log2: Float32
) -> Float32:
    delta = sink * Float32(LOG2_E) - scale_log2 * final_max
    attention_rescale = cute.math.exp2(-cute.math.max(delta, Float32(0)), approx=True)
    sink_mass = cute.math.exp2(cute.math.min(delta, Float32(0)), approx=True)
    sink_mass *= Float32(448.0)
    return (
        Float32(1.0)
        * attention_rescale
        * cute.math.rcp(reduced_sum * attention_rescale + sink_mass, approx=True)
    )


@cute.jit
def role_correction_final(
    tmem,
    p_full,
    o_full,
    o_corr,
    o_free,
    s_stat,
    s_fin,
    fin_bar,
    sink_ptr,
    out_addr: Int64,
    scale_log2: Float32,
    row0,
    row_step,
    n_items,
    ntiles: cutlass.Constexpr[int],
    tr=None,
):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % Int32(4)
    lane = cute.arch.lane_idx()
    head = warp * Int32(16) + (lane & Int32(15))
    half = lane >> Int32(4)
    lane_base = warp * Int32(32) << Int32(16)
    sink = Float32((sink_ptr + head).load())
    for it in cutlass.range(n_items, unroll=1):
        b = row0 + it * row_step
        for j in cutlass.range_constexpr(1, ntiles):
            g = it * Int32(ntiles) + Int32(j)
            _correct_tile(
                tmem,
                lane_base,
                head,
                p_full,
                o_full,
                o_corr,
                s_stat,
                g,
                it,
                Int32(j),
                scale_log2,
                ntiles,
            )
            if cutlass.const_expr(tr is not None):
                _stamp(
                    tr, Int32(60 + j), it == Int32(0) and warp == Int32(0) and (lane == Int32(0))
                )
        g_last = it * Int32(ntiles) + Int32(ntiles - 1)
        _wait(o_full, g_last & Int32(1))
        if cutlass.const_expr(tr is not None):
            _stamp(tr, Int32(80), it == Int32(0) and warp == Int32(0) and (lane == Int32(0)))
            _stamp(tr, Int32(160) + it, it < Int32(12) and warp == Int32(0) and (lane == Int32(0)))
        _wait(fin_bar, it & Int32(1))
        if cutlass.const_expr(tr is not None):
            _stamp(tr, Int32(240) + it, it < Int32(12) and warp == Int32(0) and (lane == Int32(0)))
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        fb = (it & Int32(1)) * Int32(128)
        final_max = (s_fin + fb + head).load()
        reduced_sum = (s_fin + fb + Int32(64) + head).load()
        norm = _final_norm(final_max, reduced_sum, sink, scale_log2)
        out_base = (
            out_addr
            + Int64(b) * Int64(HEADS * LATENT * 2)
            + Int64((head * Int32(LATENT) + half * Int32(64)) * Int32(2))
        )
        ob = []
        for v in cutlass.range_constexpr(NSLICE):
            for hc in cutlass.range_constexpr(2):
                o_addr = tmem + lane_base + Int32(TMEM_O + v % 2 * 128 + v // 2 * PANEL1 + 32 * hc)
                o = prims.tcgen05_ld(
                    prims.Tcgen05LdStShape.SHAPE_16X32BX2,
                    prims.make_tmem_ptr(o_addr, Float32),
                    num=32,
                    offset=Int64(64),
                )
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                for pi in cutlass.range_constexpr(16):
                    fp = cute.arch.mul_packed_f32x2(
                        (norm, norm),
                        (Float32(o[2 * pi]), Float32(o[2 * pi + 1])),
                        ftz=False,
                        rnd="rn",
                    )
                    ob.append(_pack_bf16x2(fp[0], fp[1]))
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.sync_warp()
        if prims.elect_sync():
            prims.mbarrier_arrive(o_free)
        for v in cutlass.range_constexpr(NSLICE):
            for ch in cutlass.range_constexpr(8):
                dst = cutlass.inttoptr(
                    out_base + Int64((v * SLICE + ch * 8) * 2), mem_space=1, dtype=Int32
                )
                dst.store(
                    cutlass.Vector.from_elements(
                        tuple((ob[v * 32 + ch * 4 + i] for i in range(4))), Int32
                    ),
                    alignment=16,
                )
        if cutlass.const_expr(tr is not None):
            _stamp(tr, Int32(81), it == Int32(0) and warp == Int32(0) and (lane == Int32(0)))
            _stamp(tr, Int32(200) + it, it < Int32(12) and warp == Int32(0) and (lane == Int32(0)))


@cute.jit
def _alloc_tmem(tmem_hold, warp):
    if warp == Int32(W_MMA):
        prims.tcgen05_alloc(tmem_hold.data_ptr(), TMEM_COLS, group=prims.CTAGroup.CTA_1)
        prims.tcgen05_relinquish_alloc_permit(group=prims.CTAGroup.CTA_1)


@cute.kernel
def _vl_kernel(
    tma_q: cutlass.GridConstant[cuda.TensorMap],
    tma_swa: cutlass.GridConstant[cuda.TensorMap],
    tma_cmp: cutlass.GridConstant[cuda.TensorMap],
    idx: cute.Tensor,
    sink: cute.Tensor,
    out: cute.Tensor,
    trace: cute.Tensor,
    softmax_scale: Float32,
    nslots: cutlass.Constexpr[int],
    use_trace: cutlass.Constexpr[bool],
    regs: cutlass.Constexpr[bool] = True,
):
    tile_n: cutlass.Constexpr[int] = 128
    ntiles: cutlass.Constexpr[int] = NROWS // tile_n
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    cta, _, _ = cute.arch.block_idx()
    grid, _, _ = cute.arch.grid_dim()
    if cutlass.const_expr(use_trace):
        _stamp(
            trace.iterator.toint() + Int64(cta) * Int64(2048),
            Int32(1),
            warp == Int32(0) and cute.arch.lane_idx() == Int32(0),
        )
    nrows_total = idx.shape[0]
    n_items = (Int32(nrows_total) - cta + grid - Int32(1)) // grid
    s_q = cutlass.Array(E4M3, Q_BYTES, space=cutlass.AddressSpace.smem, alignment=1024)
    s_kv = cutlass.Array(
        E4M3, nslots * tile_n * SLICE, space=cutlass.AddressSpace.smem, alignment=1024
    )
    s_stat = cutlass.Array(Float32, 2 * 128, space=cutlass.AddressSpace.smem, alignment=16)
    s_fin = cutlass.Array(Float32, 2 * 128, space=cutlass.AddressSpace.smem, alignment=16)
    q_full = cutlass.Array(Int64, 1, space=cutlass.AddressSpace.smem)
    q_empty = cutlass.Array(Int64, 1, space=cutlass.AddressSpace.smem)
    kv_full = cutlass.Array(Int64, nslots, space=cutlass.AddressSpace.smem)
    kv_empty = cutlass.Array(Int64, nslots, space=cutlass.AddressSpace.smem)
    s_full = cutlass.Array(Int64, 2, space=cutlass.AddressSpace.smem)
    s_empty = cutlass.Array(Int64, 2, space=cutlass.AddressSpace.smem)
    p_full = cutlass.Array(Int64, 2, space=cutlass.AddressSpace.smem)
    o_full = cutlass.Array(Int64, 1, space=cutlass.AddressSpace.smem)
    o_corr = cutlass.Array(Int64, 1, space=cutlass.AddressSpace.smem)
    o_free = cutlass.Array(Int64, 1, space=cutlass.AddressSpace.smem)
    fin_bar = cutlass.Array(Int64, 1, space=cutlass.AddressSpace.smem)
    tmem_hold = cutlass.Array(Int32, 1, space=cutlass.AddressSpace.smem, alignment=16)
    if warp == Int32(0):
        if prims.elect_sync():
            prims.mbarrier_init(q_full.data_ptr(), 1)
            prims.mbarrier_init(q_empty.data_ptr(), 1)
            for i in cutlass.range_constexpr(nslots):
                prims.mbarrier_init(kv_full.data_ptr() + i, 1)
                prims.mbarrier_init(kv_empty.data_ptr() + i, 1)
            for i in cutlass.range_constexpr(2):
                prims.mbarrier_init(s_full.data_ptr() + i, 1)
                prims.mbarrier_init(s_empty.data_ptr() + i, 1)
                prims.mbarrier_init(p_full.data_ptr() + i, 4)
            prims.mbarrier_init(o_full.data_ptr(), 1)
            prims.mbarrier_init(o_corr.data_ptr(), 4)
            prims.mbarrier_init(o_free.data_ptr(), 4)
            prims.mbarrier_init(fin_bar.data_ptr(), 4)
    prims.fence_mbarrier_init()
    _alloc_tmem(tmem_hold, warp)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    tmem = tmem_hold[0]
    scale_log2 = softmax_scale * Float32(LOG2_E)
    tr = None
    if cutlass.const_expr(use_trace):
        tr = trace.iterator.toint() + Int64(cta) * Int64(2048)
        _stamp(tr, Int32(0), warp == Int32(0) and cute.arch.lane_idx() == Int32(0))
    if warp < Int32(W_CORR0):
        if cutlass.const_expr(regs):
            prims.setmaxregister(REG_SOFTMAX, prims.SetMaxRegisterAction.INCREASE)
        role_softmax(
            tmem,
            s_full.data_ptr(),
            p_full.data_ptr(),
            s_stat.data_ptr(),
            s_fin.data_ptr(),
            fin_bar.data_ptr(),
            scale_log2,
            n_items,
            tile_n,
            ntiles,
            tr,
        )
    elif warp < Int32(W_MMA):
        if cutlass.const_expr(regs):
            prims.setmaxregister(REG_CORR, prims.SetMaxRegisterAction.INCREASE)
        role_correction_final(
            tmem,
            p_full.data_ptr(),
            o_full.data_ptr(),
            o_corr.data_ptr(),
            o_free.data_ptr(),
            s_stat.data_ptr(),
            s_fin.data_ptr(),
            fin_bar.data_ptr(),
            sink.iterator.raw_ptr(),
            out.iterator.toint(),
            scale_log2,
            cta,
            grid,
            n_items,
            ntiles,
            tr,
        )
    elif warp == Int32(W_MMA):
        if cutlass.const_expr(regs):
            prims.setmaxregister(REG_OTHER, prims.SetMaxRegisterAction.DECREASE)
        role_mma(
            tmem,
            s_q.data_ptr(),
            s_kv.data_ptr(),
            q_full.data_ptr(),
            q_empty.data_ptr(),
            kv_full.data_ptr(),
            kv_empty.data_ptr(),
            s_full.data_ptr(),
            s_empty.data_ptr(),
            p_full.data_ptr(),
            o_corr.data_ptr(),
            o_free.data_ptr(),
            o_full.data_ptr(),
            n_items,
            tile_n,
            ntiles,
            nslots,
            tr,
        )
    elif warp == Int32(W_QLOAD):
        if cutlass.const_expr(regs):
            prims.setmaxregister(REG_OTHER, prims.SetMaxRegisterAction.DECREASE)
        role_q_load(
            tma_q, s_q.data_ptr(), q_full.data_ptr(), q_empty.data_ptr(), cta, grid, n_items
        )
    elif warp < Int32(W_GATHER0 + NGW):
        if cutlass.const_expr(regs):
            prims.setmaxregister(REG_OTHER, prims.SetMaxRegisterAction.DECREASE)
        role_gather(
            tma_swa,
            tma_cmp,
            s_kv.data_ptr(),
            idx.iterator.raw_ptr(),
            kv_full.data_ptr(),
            kv_empty.data_ptr(),
            warp - Int32(W_GATHER0),
            cta,
            grid,
            n_items,
            Int32(0),
            tile_n,
            ntiles,
            nslots,
            tr,
        )
    elif cutlass.const_expr(regs):
        prims.setmaxregister(REG_OTHER, prims.SetMaxRegisterAction.DECREASE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    if warp == Int32(W_MMA):
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        prims.tcgen05_dealloc(
            prims.make_tmem_ptr(tmem, Float32), TMEM_COLS, group=prims.CTAGroup.CTA_1
        )
        if cutlass.const_expr(use_trace):
            _stamp(tr, Int32(2), cute.arch.lane_idx() == Int32(0))


def _q_map(q):
    return cuda.create_tensor_map_tiled(
        global_address=q.iterator.toint(),
        dtype=E4M3,
        global_dims=[LATENT, HEADS, q.shape[0]],
        global_strides=[LATENT // 16, HEADS * LATENT // 16],
        box_dims=[SLICE, HEADS, 1],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )


def _pool_map(p):
    return cuda.create_tensor_map_tiled(
        global_address=p.iterator.toint(),
        dtype=E4M3,
        global_dims=[LATENT, p.shape[0]],
        global_strides=[LATENT // 16],
        box_dims=[SLICE, 1],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )


@cute.jit
def _launch_vl(
    q: cute.Tensor,
    swa: cute.Tensor,
    cmp: cute.Tensor,
    idx: cute.Tensor,
    sink: cute.Tensor,
    out: cute.Tensor,
    trace: cute.Tensor,
    softmax_scale: Float32,
    grid: Int32,
    stream: cuda_driver.CUstream,
    nslots: cutlass.Constexpr[int],
    use_trace: cutlass.Constexpr[bool],
):
    tma_q = _q_map(q)
    tma_swa = _pool_map(swa)
    tma_cmp = _pool_map(cmp)
    _vl_kernel(
        tma_q, tma_swa, tma_cmp, idx, sink, out, trace, softmax_scale, nslots, use_trace
    ).launch(grid=(grid, 1, 1), block=(THREADS, 1, 1), stream=stream, min_blocks_per_mp=1)


_COMPILED: dict = {}
_NUM_SMS: dict = {}


def _num_sms(dev) -> int:
    i = dev.index if dev.index is not None else torch.cuda.current_device()
    if i not in _NUM_SMS:
        _NUM_SMS[i] = torch.cuda.get_device_properties(i).multi_processor_count
    return _NUM_SMS[i]


def _fake_inputs():
    mk = cute.runtime.make_fake_compact_tensor
    b = cute.sym_int64()
    rs = cute.sym_int64()
    rc = cute.sym_int64()
    return (
        mk(cutlass.Uint8, (b, HEADS, LATENT), stride_order=(2, 1, 0), assumed_align=16),
        mk(cutlass.Uint8, (rs, LATENT), stride_order=(1, 0), assumed_align=16),
        mk(cutlass.Uint8, (rc, LATENT), stride_order=(1, 0), assumed_align=16),
        mk(Int32, (b, NROWS), stride_order=(1, 0), assumed_align=16),
        mk(Float32, (HEADS,), stride_order=(0,), assumed_align=16),
        mk(BF16, (b, HEADS, LATENT), stride_order=(2, 1, 0), assumed_align=16),
        mk(Int64, (cute.sym_int64(),), stride_order=(0,), assumed_align=16),
    )


def compile_vl(nslots: int = 12, trace: bool = False):
    key = ("vl", nslots, trace)
    if key not in _COMPILED:
        _COMPILED[key] = cute.compile(
            _launch_vl,
            *_fake_inputs(),
            Float32(1.0),
            Int32(1),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            nslots,
            trace,
            options="--enable-tvm-ffi",
        )
    return _COMPILED[key]


_DUMMY: dict = {}


def _dummy_trace(dev):
    if dev not in _DUMMY:
        _DUMMY[dev] = torch.zeros(16, dtype=torch.int64, device=dev)
    return _DUMMY[dev]


def _u8(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8) if t.dtype != torch.uint8 else t


def run_vl(
    q8: torch.Tensor,
    swa_pool: torch.Tensor,
    cmp_pool: torch.Tensor,
    idx: torch.Tensor,
    sink: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float = 1.0 / math.sqrt(512.0),
    nslots: int = 12,
    grid: int | None = None,
    stream=None,
    trace: torch.Tensor | None = None,
):
    fn = compile_vl(nslots, trace is not None)
    if stream is None:
        stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    if grid is None:
        grid = min(q8.shape[0], _num_sms(q8.device))
    tr = trace if trace is not None else _dummy_trace(q8.device)
    fn(
        _u8(q8),
        _u8(swa_pool),
        _u8(cmp_pool),
        idx,
        sink,
        out,
        tr,
        Float32(softmax_scale),
        Int32(grid),
        stream,
    )
    return out


@dsl_user_op
def _mapa_u32(addr_i32, rank, *, loc=None, ip=None):
    return Int32(
        _llvm.inline_asm(
            _T.i32(),
            [Int32(addr_i32).ir_value(loc=loc, ip=ip), Int32(rank).ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_async_v4(raddr, rbar, a, b, c, d, *, loc=None, ip=None):
    _llvm.inline_asm(
        None,
        [
            Int32(raddr).ir_value(loc=loc, ip=ip),
            Int32(rbar).ir_value(loc=loc, ip=ip),
            Int32(a).ir_value(loc=loc, ip=ip),
            Int32(b).ir_value(loc=loc, ip=ip),
            Int32(c).ir_value(loc=loc, ip=ip),
            Int32(d).ir_value(loc=loc, ip=ip),
        ],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [$0], {$2, $3, $4, $5}, [$1];",
        "r,r,r,r,r,r",
        has_side_effects=True,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_async_v2(raddr, rbar, a, b, *, loc=None, ip=None):
    _llvm.inline_asm(
        None,
        [
            Int32(raddr).ir_value(loc=loc, ip=ip),
            Int32(rbar).ir_value(loc=loc, ip=ip),
            Int32(a).ir_value(loc=loc, ip=ip),
            Int32(b).ir_value(loc=loc, ip=ip),
        ],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [$0], {$2, $3}, [$1];",
        "r,r,r,r",
        has_side_effects=True,
        asm_dialect=_llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _f2i(x: Float32) -> Int32:
    return prims.mov_b32(x, target_type=Int32)


@cute.jit
def role_correction_export(
    tmem, o_full, o_free, s_fin, fin_bar, ml_addr: Int64, o_addr: Int64, row0, row_step, n_items
):
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % Int32(4)
    lane = cute.arch.lane_idx()
    head = warp * Int32(16) + (lane & Int32(15))
    half = lane >> Int32(4)
    lane_base = warp * Int32(32) << Int32(16)
    for it in cutlass.range(n_items, unroll=1):
        b = row0 + it * row_step
        _wait(o_full, it & Int32(1))
        _wait(fin_bar, it & Int32(1))
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        fb = (it & Int32(1)) * Int32(128)
        if half == Int32(0):
            mrow = cutlass.inttoptr(
                ml_addr + Int64(b) * Int64(512) + Int64(head * Int32(4)), mem_space=1, dtype=Float32
            )
            mrow.store((s_fin + fb + head).load())
            (mrow + Int32(64)).store((s_fin + fb + Int32(64) + head).load())
        obase = (
            o_addr
            + Int64(b) * Int64(HEADS * LATENT * 4)
            + Int64((head * Int32(LATENT) + half * Int32(64)) * Int32(4))
        )
        for v in cutlass.range_constexpr(NSLICE):
            oa = tmem + lane_base + Int32(TMEM_O + v % 2 * 128 + v // 2 * PANEL1)
            o = prims.tcgen05_ld(
                prims.Tcgen05LdStShape.SHAPE_16X32BX2,
                prims.make_tmem_ptr(oa, Float32),
                num=64,
                offset=Int64(64),
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            for ch in cutlass.range_constexpr(0, 64, 4):
                dst = cutlass.inttoptr(
                    obase + Int64((v * SLICE + ch) * 4), mem_space=1, dtype=Float32
                )
                dst.store(
                    cutlass.Vector.from_elements(
                        tuple((Float32(o[ch + i]) for i in range(4))), Float32
                    ),
                    alignment=16,
                )
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.sync_warp()
        if prims.elect_sync():
            prims.mbarrier_arrive(o_free)


@cute.jit
def role_vs_merge(
    tmem,
    o_full,
    fin_bar,
    s_fin,
    s_ml,
    s_recv,
    rx_bar,
    sink_ptr,
    swa_ml_addr: Int64,
    swa_o_addr: Int64,
    out_addr: Int64,
    scale_log2: Float32,
    b,
    r,
    d,
    cr: cutlass.Constexpr[int],
    cd: cutlass.Constexpr[int],
    has_swa: cutlass.Constexpr[bool],
    tr=None,
):
    nv: cutlass.Constexpr[int] = NSLICE // cd
    ddims: cutlass.Constexpr[int] = LATENT // cd
    hmax: cutlass.Constexpr[int] = (HEADS + cr - 1) // cr
    blk: cutlass.Constexpr[int] = 2 * hmax * 16
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % Int32(4)
    lane = cute.arch.lane_idx()
    tid = warp * Int32(32) + lane
    head = warp * Int32(16) + (lane & Int32(15))
    half = lane >> Int32(4)
    lane_base = warp * Int32(32) << Int32(16)
    h_lo = (r * Int32(HEADS) + Int32(cr - 1)) // Int32(cr)
    h_hi = ((r + Int32(1)) * Int32(HEADS) + Int32(cr - 1)) // Int32(cr)
    nel: cutlass.Constexpr[int] = nv * 8 * 2 * hmax
    nch: cutlass.Constexpr[int] = (nel + 127) // 128
    swa_pre = []
    swa_m = []
    swa_l = []
    sink_v = []
    for i in cutlass.range_constexpr(nch):
        t = tid + Int32(i * 128)
        e = t % Int32(2 * hmax)
        hh = e % Int32(hmax)
        h = h_lo + hh
        v8 = cutlass.Vector.from_elements(tuple((Float32(0.0) for _ in range(8))), Float32)
        ms = Float32(NEG_FLT_MAX)
        ls = Float32(0.0)
        if cutlass.const_expr(has_swa):
            if h < h_hi and t < Int32(nel):
                vc = t // Int32(2 * hmax)
                gdim = (
                    d * Int32(ddims)
                    + vc // Int32(8) * Int32(128)
                    + e // Int32(hmax) * Int32(64)
                    + vc % Int32(8) * Int32(8)
                )
                sp = cutlass.inttoptr(
                    swa_o_addr
                    + Int64(b) * Int64(HEADS * LATENT * 4)
                    + Int64((h * Int32(LATENT) + gdim) * Int32(4)),
                    mem_space=1,
                    dtype=Float32,
                )
                v8 = sp.load(count=8, alignment=16)
                mp = cutlass.inttoptr(
                    swa_ml_addr + Int64(b) * Int64(512) + Int64(h * Int32(4)),
                    mem_space=1,
                    dtype=Float32,
                )
                ms = mp.load()
                ls = (mp + Int32(64)).load()
        hs = h
        if hs >= Int32(HEADS):
            hs = Int32(HEADS - 1)
        swa_pre.append(v8)
        swa_m.append(ms)
        swa_l.append(ls)
        sink_v.append(Float32((sink_ptr + hs).load()))
    _wait(o_full, 0)
    _wait(fin_bar, 0)
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    m_self = (s_fin + head).load()
    l_self = (s_fin + Int32(64) + head).load()
    if cutlass.const_expr(tr is not None):
        _stamp(tr, Int32(80), tid == Int32(0))
    inv_l = Float32(1.0) / l_self
    o_idx = head * Int32(cr) // Int32(HEADS)
    hl = head - (o_idx * Int32(HEADS) + Int32(cr - 1)) // Int32(cr)
    owner = d * Int32(cr) + o_idx
    rbar = _mapa_u32(Int32(rx_bar.toint()), owner)
    if half == Int32(0):
        _st_async_v2(
            _mapa_u32(Int32(s_ml.toint()) + (r * Int32(hmax) + hl) * Int32(8), owner),
            rbar,
            _f2i(m_self),
            _f2i(l_self),
        )
    rbase = Int32(s_recv.toint()) + r * Int32(nv * 8 * blk) + (half * Int32(hmax) + hl) * Int32(16)
    for vv in cutlass.range_constexpr(nv):
        v = d * Int32(nv) + Int32(vv)
        oa = (
            tmem
            + lane_base
            + Int32(TMEM_O)
            + v % Int32(2) * Int32(128)
            + (v >> Int32(1)) * Int32(PANEL1)
        )
        o = prims.tcgen05_ld(
            prims.Tcgen05LdStShape.SHAPE_16X32BX2,
            prims.make_tmem_ptr(oa, Float32),
            num=64,
            offset=Int64(64),
        )
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        for c in cutlass.range_constexpr(8):
            pk = []
            for pi in cutlass.range_constexpr(4):
                fp = cute.arch.mul_packed_f32x2(
                    (inv_l, inv_l),
                    (Float32(o[8 * c + 2 * pi]), Float32(o[8 * c + 2 * pi + 1])),
                    ftz=False,
                    rnd="rn",
                )
                pk.append(_pack_bf16x2(fp[0], fp[1]))
            _st_async_v4(
                _mapa_u32(rbase + Int32((vv * 8 + c) * blk), owner),
                rbar,
                pk[0],
                pk[1],
                pk[2],
                pk[3],
            )
    _wait(rx_bar, 0)
    if cutlass.const_expr(tr is not None):
        _stamp(tr, Int32(84), tid == Int32(0))
    for i in cutlass.range_constexpr(nch):
        t = tid + Int32(i * 128)
        e = t % Int32(2 * hmax)
        vc = t // Int32(2 * hmax)
        hh = e % Int32(hmax)
        h = h_lo + hh
        if h < h_hi and t < Int32(nel):
            big_m = swa_m[i]
            for p in cutlass.range_constexpr(cr):
                big_m = cute.math.max(
                    big_m, (s_ml + (Int32(p * hmax) + hh) * Int32(2)).load(), ftz=True
                )
            l_tot = Float32(0.0)
            acc = cutlass.Vector.from_elements(tuple((Float32(0.0) for _ in range(8))), Float32)
            if cutlass.const_expr(has_swa):
                w_swa = cute.math.exp2(scale_log2 * (swa_m[i] - big_m), fastmath=True)
                l_tot = w_swa * swa_l[i]
                acc = swa_pre[i] * w_swa
            eoff = vc * Int32(blk // 2) + e * Int32(8)
            for p in cutlass.range_constexpr(cr):
                mpp = (s_ml + (Int32(p * hmax) + hh) * Int32(2)).load()
                lpp = (s_ml + (Int32(p * hmax) + hh) * Int32(2) + Int32(1)).load()
                a_p = cute.math.exp2(scale_log2 * (mpp - big_m), fastmath=True) * lpp
                l_tot = l_tot + a_p
                pv = (s_recv + (Int32(p * nv * 8 * blk // 2) + eoff)).load(count=8, alignment=16)
                acc = acc + pv.to(Float32) * a_p
            norm = _final_norm(big_m, l_tot, sink_v[i], scale_log2)
            gdim = (
                d * Int32(ddims)
                + vc // Int32(8) * Int32(128)
                + e // Int32(hmax) * Int32(64)
                + vc % Int32(8) * Int32(8)
            )
            dst = cutlass.inttoptr(
                out_addr
                + Int64(b) * Int64(HEADS * LATENT * 2)
                + Int64((h * Int32(LATENT) + gdim) * Int32(2)),
                mem_space=1,
                dtype=BF16,
            )
            dst.store((acc * norm).to(BF16), alignment=16)
    if cutlass.const_expr(tr is not None):
        _stamp(tr, Int32(81), tid == Int32(0))


@cute.kernel
def _vs_kernel(
    tma_q: cutlass.GridConstant[cuda.TensorMap],
    tma_swa: cutlass.GridConstant[cuda.TensorMap],
    tma_cmp: cutlass.GridConstant[cuda.TensorMap],
    idx: cute.Tensor,
    sink: cute.Tensor,
    out: cute.Tensor,
    swa_ml: cute.Tensor,
    swa_o: cute.Tensor,
    trace: cute.Tensor,
    softmax_scale: Float32,
    cr: cutlass.Constexpr[int],
    cd: cutlass.Constexpr[int],
    has_swa: cutlass.Constexpr[bool],
    use_trace: cutlass.Constexpr[bool],
):
    list_lo: cutlass.Constexpr[int] = WINDOW if has_swa else 0
    tile_n: cutlass.Constexpr[int] = (NROWS - list_lo) // cr
    nslots: cutlass.Constexpr[int] = NSLICE
    ddims: cutlass.Constexpr[int] = LATENT // cd
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    rank, b, _ = cute.arch.block_idx()
    r = rank % Int32(cr)
    d = rank // Int32(cr)
    lb = Int32(list_lo) + r * Int32(tile_n)
    tr = None
    if cutlass.const_expr(use_trace):
        tr = trace.iterator.toint() + (Int64(b) * Int64(cr * cd) + Int64(rank)) * Int64(2048)
        _stamp(tr, Int32(1), warp == Int32(0) and cute.arch.lane_idx() == Int32(0))
    s_q = cutlass.Array(E4M3, Q_BYTES, space=cutlass.AddressSpace.smem, alignment=1024)
    s_kv = cutlass.Array(
        E4M3, nslots * tile_n * SLICE, space=cutlass.AddressSpace.smem, alignment=1024
    )
    hmax: cutlass.Constexpr[int] = (HEADS + cr - 1) // cr
    s_recv = cutlass.Array(BF16, cr * hmax * ddims, space=cutlass.AddressSpace.smem, alignment=128)
    s_ml = cutlass.Array(Float32, cr * hmax * 2, space=cutlass.AddressSpace.smem, alignment=16)
    s_stat = cutlass.Array(Float32, 2 * 128, space=cutlass.AddressSpace.smem, alignment=16)
    s_fin = cutlass.Array(Float32, 2 * 128, space=cutlass.AddressSpace.smem, alignment=16)
    q_full = cutlass.Array(Int64, 1, space=cutlass.AddressSpace.smem)
    q_empty = cutlass.Array(Int64, 1, space=cutlass.AddressSpace.smem)
    kv_full = cutlass.Array(Int64, nslots, space=cutlass.AddressSpace.smem)
    kv_empty = cutlass.Array(Int64, nslots, space=cutlass.AddressSpace.smem)
    s_full = cutlass.Array(Int64, 2, space=cutlass.AddressSpace.smem)
    s_empty = cutlass.Array(Int64, 2, space=cutlass.AddressSpace.smem)
    p_full = cutlass.Array(Int64, 2, space=cutlass.AddressSpace.smem)
    o_full = cutlass.Array(Int64, 1, space=cutlass.AddressSpace.smem)
    o_corr = cutlass.Array(Int64, 1, space=cutlass.AddressSpace.smem)
    o_free = cutlass.Array(Int64, 1, space=cutlass.AddressSpace.smem)
    fin_bar = cutlass.Array(Int64, 1, space=cutlass.AddressSpace.smem)
    rx_bar = cutlass.Array(Int64, 1, space=cutlass.AddressSpace.smem)
    tmem_hold = cutlass.Array(Int32, 1, space=cutlass.AddressSpace.smem, alignment=16)
    if warp == Int32(0):
        if prims.elect_sync():
            prims.mbarrier_init(q_full.data_ptr(), 1)
            prims.mbarrier_init(q_empty.data_ptr(), 1)
            for i in cutlass.range_constexpr(nslots):
                prims.mbarrier_init(kv_full.data_ptr() + i, 1)
                prims.mbarrier_init(kv_empty.data_ptr() + i, 1)
            for i in cutlass.range_constexpr(2):
                prims.mbarrier_init(s_full.data_ptr() + i, 1)
                prims.mbarrier_init(s_empty.data_ptr() + i, 1)
                prims.mbarrier_init(p_full.data_ptr() + i, 4)
            prims.mbarrier_init(o_full.data_ptr(), 1)
            prims.mbarrier_init(o_corr.data_ptr(), 4)
            prims.mbarrier_init(o_free.data_ptr(), 4)
            prims.mbarrier_init(fin_bar.data_ptr(), 4)
            prims.mbarrier_init(rx_bar.data_ptr(), 1)
            n_own = ((r + Int32(1)) * Int32(HEADS) + Int32(cr - 1)) // Int32(cr) - (
                r * Int32(HEADS) + Int32(cr - 1)
            ) // Int32(cr)
            prims.mbarrier_arrive_expect_tx(rx_bar.data_ptr(), Int32(cr * (ddims * 2 + 8)) * n_own)
    prims.fence_mbarrier_init()
    _alloc_tmem(tmem_hold, warp)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cluster_arrive()
    prims.barrier_cluster_wait()
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    tmem = tmem_hold[0]
    scale_log2 = softmax_scale * Float32(LOG2_E)
    if cutlass.const_expr(use_trace):
        _stamp(tr, Int32(0), warp == Int32(0) and cute.arch.lane_idx() == Int32(0))
    if warp < Int32(W_CORR0):
        prims.setmaxregister(REG_SOFTMAX, prims.SetMaxRegisterAction.INCREASE)
        role_softmax(
            tmem,
            s_full.data_ptr(),
            p_full.data_ptr(),
            s_stat.data_ptr(),
            s_fin.data_ptr(),
            fin_bar.data_ptr(),
            scale_log2,
            Int32(1),
            tile_n,
            1,
            tr,
        )
    elif warp < Int32(W_MMA):
        prims.setmaxregister(REG_CORR, prims.SetMaxRegisterAction.INCREASE)
        role_vs_merge(
            tmem,
            o_full.data_ptr(),
            fin_bar.data_ptr(),
            s_fin.data_ptr(),
            s_ml.data_ptr(),
            s_recv.data_ptr(),
            rx_bar.data_ptr(),
            sink.iterator.raw_ptr(),
            swa_ml.iterator.toint(),
            swa_o.iterator.toint(),
            out.iterator.toint(),
            scale_log2,
            b,
            r,
            d,
            cr,
            cd,
            has_swa,
            tr,
        )
    elif warp == Int32(W_MMA):
        prims.setmaxregister(REG_OTHER, prims.SetMaxRegisterAction.DECREASE)
        role_mma(
            tmem,
            s_q.data_ptr(),
            s_kv.data_ptr(),
            q_full.data_ptr(),
            q_empty.data_ptr(),
            kv_full.data_ptr(),
            kv_empty.data_ptr(),
            s_full.data_ptr(),
            s_empty.data_ptr(),
            p_full.data_ptr(),
            o_corr.data_ptr(),
            o_free.data_ptr(),
            o_full.data_ptr(),
            Int32(1),
            tile_n,
            1,
            nslots,
            tr,
            d * Int32(NSLICE // cd),
            NSLICE // cd,
        )
    elif warp == Int32(W_QLOAD):
        prims.setmaxregister(REG_OTHER, prims.SetMaxRegisterAction.DECREASE)
        role_q_load(
            tma_q, s_q.data_ptr(), q_full.data_ptr(), q_empty.data_ptr(), b, Int32(0), Int32(1)
        )
    elif warp < Int32(W_GATHER0 + NGW):
        prims.setmaxregister(REG_OTHER, prims.SetMaxRegisterAction.DECREASE)
        role_gather(
            tma_swa,
            tma_cmp,
            s_kv.data_ptr(),
            idx.iterator.raw_ptr(),
            kv_full.data_ptr(),
            kv_empty.data_ptr(),
            warp - Int32(W_GATHER0),
            b,
            Int32(0),
            Int32(1),
            lb,
            tile_n,
            1,
            nslots,
            tr,
        )
    else:
        prims.setmaxregister(REG_OTHER, prims.SetMaxRegisterAction.DECREASE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    if warp == Int32(W_MMA):
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        prims.tcgen05_dealloc(
            prims.make_tmem_ptr(tmem, Float32), TMEM_COLS, group=prims.CTAGroup.CTA_1
        )
    prims.barrier_cluster_arrive()
    prims.barrier_cluster_wait()
    if cutlass.const_expr(use_trace):
        _stamp(tr, Int32(2), warp == Int32(0) and cute.arch.lane_idx() == Int32(0))


@cute.kernel
def _swa_kernel(
    tma_q: cutlass.GridConstant[cuda.TensorMap],
    tma_swa: cutlass.GridConstant[cuda.TensorMap],
    idx: cute.Tensor,
    swa_ml: cute.Tensor,
    swa_o: cute.Tensor,
    softmax_scale: Float32,
):
    tile_n: cutlass.Constexpr[int] = WINDOW
    nslots: cutlass.Constexpr[int] = 8
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    cta, _, _ = cute.arch.block_idx()
    grid, _, _ = cute.arch.grid_dim()
    n_items = (Int32(idx.shape[0]) - cta + grid - Int32(1)) // grid
    s_q = cutlass.Array(E4M3, Q_BYTES, space=cutlass.AddressSpace.smem, alignment=1024)
    s_kv = cutlass.Array(
        E4M3, nslots * tile_n * SLICE, space=cutlass.AddressSpace.smem, alignment=1024
    )
    s_stat = cutlass.Array(Float32, 2 * 128, space=cutlass.AddressSpace.smem, alignment=16)
    s_fin = cutlass.Array(Float32, 2 * 128, space=cutlass.AddressSpace.smem, alignment=16)
    q_full = cutlass.Array(Int64, 1, space=cutlass.AddressSpace.smem)
    q_empty = cutlass.Array(Int64, 1, space=cutlass.AddressSpace.smem)
    kv_full = cutlass.Array(Int64, nslots, space=cutlass.AddressSpace.smem)
    kv_empty = cutlass.Array(Int64, nslots, space=cutlass.AddressSpace.smem)
    s_full = cutlass.Array(Int64, 2, space=cutlass.AddressSpace.smem)
    s_empty = cutlass.Array(Int64, 2, space=cutlass.AddressSpace.smem)
    p_full = cutlass.Array(Int64, 2, space=cutlass.AddressSpace.smem)
    o_full = cutlass.Array(Int64, 1, space=cutlass.AddressSpace.smem)
    o_corr = cutlass.Array(Int64, 1, space=cutlass.AddressSpace.smem)
    o_free = cutlass.Array(Int64, 1, space=cutlass.AddressSpace.smem)
    fin_bar = cutlass.Array(Int64, 1, space=cutlass.AddressSpace.smem)
    tmem_hold = cutlass.Array(Int32, 1, space=cutlass.AddressSpace.smem, alignment=16)
    if warp == Int32(0):
        if prims.elect_sync():
            prims.mbarrier_init(q_full.data_ptr(), 1)
            prims.mbarrier_init(q_empty.data_ptr(), 1)
            for i in cutlass.range_constexpr(nslots):
                prims.mbarrier_init(kv_full.data_ptr() + i, 1)
                prims.mbarrier_init(kv_empty.data_ptr() + i, 1)
            for i in cutlass.range_constexpr(2):
                prims.mbarrier_init(s_full.data_ptr() + i, 1)
                prims.mbarrier_init(s_empty.data_ptr() + i, 1)
                prims.mbarrier_init(p_full.data_ptr() + i, 4)
            prims.mbarrier_init(o_full.data_ptr(), 1)
            prims.mbarrier_init(o_corr.data_ptr(), 4)
            prims.mbarrier_init(o_free.data_ptr(), 4)
            prims.mbarrier_init(fin_bar.data_ptr(), 4)
    prims.fence_mbarrier_init()
    _alloc_tmem(tmem_hold, warp)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    tmem = tmem_hold[0]
    scale_log2 = softmax_scale * Float32(LOG2_E)
    if warp < Int32(W_CORR0):
        prims.setmaxregister(REG_SOFTMAX, prims.SetMaxRegisterAction.INCREASE)
        role_softmax(
            tmem,
            s_full.data_ptr(),
            p_full.data_ptr(),
            s_stat.data_ptr(),
            s_fin.data_ptr(),
            fin_bar.data_ptr(),
            scale_log2,
            n_items,
            tile_n,
            1,
        )
    elif warp < Int32(W_MMA):
        prims.setmaxregister(REG_CORR, prims.SetMaxRegisterAction.INCREASE)
        role_correction_export(
            tmem,
            o_full.data_ptr(),
            o_free.data_ptr(),
            s_fin.data_ptr(),
            fin_bar.data_ptr(),
            swa_ml.iterator.toint(),
            swa_o.iterator.toint(),
            cta,
            grid,
            n_items,
        )
    elif warp == Int32(W_MMA):
        prims.setmaxregister(REG_OTHER, prims.SetMaxRegisterAction.DECREASE)
        role_mma(
            tmem,
            s_q.data_ptr(),
            s_kv.data_ptr(),
            q_full.data_ptr(),
            q_empty.data_ptr(),
            kv_full.data_ptr(),
            kv_empty.data_ptr(),
            s_full.data_ptr(),
            s_empty.data_ptr(),
            p_full.data_ptr(),
            o_corr.data_ptr(),
            o_free.data_ptr(),
            o_full.data_ptr(),
            n_items,
            tile_n,
            1,
            nslots,
        )
    elif warp == Int32(W_QLOAD):
        prims.setmaxregister(REG_OTHER, prims.SetMaxRegisterAction.DECREASE)
        role_q_load(
            tma_q, s_q.data_ptr(), q_full.data_ptr(), q_empty.data_ptr(), cta, grid, n_items
        )
    elif warp < Int32(W_GATHER0 + NGW):
        prims.setmaxregister(REG_OTHER, prims.SetMaxRegisterAction.DECREASE)
        role_gather(
            tma_swa,
            tma_swa,
            s_kv.data_ptr(),
            idx.iterator.raw_ptr(),
            kv_full.data_ptr(),
            kv_empty.data_ptr(),
            warp - Int32(W_GATHER0),
            cta,
            grid,
            n_items,
            Int32(0),
            tile_n,
            1,
            nslots,
        )
    else:
        prims.setmaxregister(REG_OTHER, prims.SetMaxRegisterAction.DECREASE)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    if warp == Int32(W_MMA):
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        prims.tcgen05_dealloc(
            prims.make_tmem_ptr(tmem, Float32), TMEM_COLS, group=prims.CTAGroup.CTA_1
        )


@cute.jit
def _launch_vs(
    q: cute.Tensor,
    swa: cute.Tensor,
    cmp: cute.Tensor,
    idx: cute.Tensor,
    sink: cute.Tensor,
    out: cute.Tensor,
    swa_ml: cute.Tensor,
    swa_o: cute.Tensor,
    trace: cute.Tensor,
    softmax_scale: Float32,
    stream: cuda_driver.CUstream,
    cr: cutlass.Constexpr[int],
    cd: cutlass.Constexpr[int],
    has_swa: cutlass.Constexpr[bool],
    use_trace: cutlass.Constexpr[bool],
):
    tma_q = _q_map(q)
    tma_swa = _pool_map(swa)
    tma_cmp = _pool_map(cmp)
    _vs_kernel(
        tma_q,
        tma_swa,
        tma_cmp,
        idx,
        sink,
        out,
        swa_ml,
        swa_o,
        trace,
        softmax_scale,
        cr,
        cd,
        has_swa,
        use_trace,
    ).launch(
        grid=(cr * cd, q.shape[0], 1),
        block=(THREADS, 1, 1),
        cluster=(cr * cd, 1, 1),
        stream=stream,
        min_blocks_per_mp=1,
    )


@cute.jit
def _launch_swa(
    q: cute.Tensor,
    swa: cute.Tensor,
    idx: cute.Tensor,
    swa_ml: cute.Tensor,
    swa_o: cute.Tensor,
    softmax_scale: Float32,
    grid: Int32,
    stream: cuda_driver.CUstream,
):
    tma_q = _q_map(q)
    tma_swa = _pool_map(swa)
    _swa_kernel(tma_q, tma_swa, idx, swa_ml, swa_o, softmax_scale).launch(
        grid=(grid, 1, 1), block=(THREADS, 1, 1), stream=stream, min_blocks_per_mp=1
    )


def _fake_vs():
    mk = cute.runtime.make_fake_compact_tensor
    b = cute.sym_int64()
    return (
        mk(cutlass.Uint8, (b, HEADS, LATENT), stride_order=(2, 1, 0), assumed_align=16),
        mk(cutlass.Uint8, (cute.sym_int64(), LATENT), stride_order=(1, 0), assumed_align=16),
        mk(cutlass.Uint8, (cute.sym_int64(), LATENT), stride_order=(1, 0), assumed_align=16),
        mk(Int32, (b, NROWS), stride_order=(1, 0), assumed_align=16),
        mk(Float32, (HEADS,), stride_order=(0,), assumed_align=16),
        mk(BF16, (b, HEADS, LATENT), stride_order=(2, 1, 0), assumed_align=16),
        mk(Float32, (cute.sym_int64(), 2, HEADS), stride_order=(2, 1, 0), assumed_align=16),
        mk(Float32, (cute.sym_int64(), HEADS, LATENT), stride_order=(2, 1, 0), assumed_align=16),
        mk(Int64, (cute.sym_int64(),), stride_order=(0,), assumed_align=16),
    )


def compile_vs(cr: int, cd: int, has_swa: bool, trace: bool = False):
    key = ("vs", cr, cd, has_swa, trace)
    if key not in _COMPILED:
        _COMPILED[key] = cute.compile(
            _launch_vs,
            *_fake_vs(),
            Float32(1.0),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            cr,
            cd,
            has_swa,
            trace,
            options="--enable-tvm-ffi",
        )
    return _COMPILED[key]


def compile_swa():
    key = ("swa",)
    if key not in _COMPILED:
        mk = cute.runtime.make_fake_compact_tensor
        b = cute.sym_int64()
        _COMPILED[key] = cute.compile(
            _launch_swa,
            mk(cutlass.Uint8, (b, HEADS, LATENT), stride_order=(2, 1, 0), assumed_align=16),
            mk(cutlass.Uint8, (cute.sym_int64(), LATENT), stride_order=(1, 0), assumed_align=16),
            mk(Int32, (b, NROWS), stride_order=(1, 0), assumed_align=16),
            mk(Float32, (cute.sym_int64(), 2, HEADS), stride_order=(2, 1, 0), assumed_align=16),
            mk(
                Float32, (cute.sym_int64(), HEADS, LATENT), stride_order=(2, 1, 0), assumed_align=16
            ),
            Float32(1.0),
            Int32(1),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
            options="--enable-tvm-ffi",
        )
    return _COMPILED[key]


class SwaWorkspace:
    def __init__(self, bs: int, device):
        self.ml = torch.empty((bs, 2, HEADS), dtype=torch.float32, device=device)
        self.o = torch.empty((bs, HEADS, LATENT), dtype=torch.float32, device=device)


def run_swa_partial(
    q8,
    swa_pool,
    idx,
    ws: SwaWorkspace,
    softmax_scale: float = 1.0 / math.sqrt(512.0),
    grid: int | None = None,
    stream=None,
):
    fn = compile_swa()
    if stream is None:
        stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    if grid is None:
        grid = min(q8.shape[0], _num_sms(q8.device))
    fn(_u8(q8), _u8(swa_pool), idx, ws.ml, ws.o, Float32(softmax_scale), Int32(grid), stream)


def run_vs(
    q8,
    swa_pool,
    cmp_pool,
    idx,
    sink,
    out,
    team: int = 16,
    cd: int | None = None,
    swa_ws: SwaWorkspace | None = None,
    softmax_scale: float = 1.0 / math.sqrt(512.0),
    stream=None,
    trace: torch.Tensor | None = None,
):
    if cd is None:
        cd = {16: 4, 8: 2, 4: 1, 2: 1, 5: 1, 10: 2}.get(team, 1)
    cr = team // cd
    has_swa = swa_ws is not None
    fn = compile_vs(cr, cd, has_swa, trace is not None)
    if stream is None:
        stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    dev = q8.device
    if swa_ws is None:
        swa_ws = _dummy_ws(dev)
    tr = trace if trace is not None else _dummy_trace(dev)
    fn(
        _u8(q8),
        _u8(swa_pool),
        _u8(cmp_pool),
        idx,
        sink,
        out,
        swa_ws.ml,
        swa_ws.o,
        tr,
        Float32(softmax_scale),
        stream,
    )
    return out


_DUMMY_WS: dict = {}


def _dummy_ws(dev):
    if dev not in _DUMMY_WS:
        _DUMMY_WS[dev] = SwaWorkspace(1, dev)
    return _DUMMY_WS[dev]
