# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""specDSA megakernel, attention: the sparse MLA over the selected compressed-KV rows, the sliding-window rows and
the attention sink (TMA gathers, tcgen05 MMA, softmax and correction warps).

The first half is the stand-alone attention engine, the second the roles that the team CTAs of the megakernel run. Where
both define a name differently the roles' name is kept and the engine's carries an ``_engine`` (``_ENGINE``) suffix.

The constants that follow from the model (heads, KV row size, window, top-K) are defined from the ``DS_V4_FLASH`` spec
of ``config.py``.
"""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import torch
from cutlass import Boolean, Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cutlass_dsl import T as _T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.cutlass_dsl import dsl_user_op as _dsl_user_op
from cutlass.experimental import primitives as prims

from . import megakernel_lib as L
from .config import DS_V4_FLASH as SPEC

# ====================================================================================================================
# Sparse MLA building blocks shared by the attention roles: Q / KV tensor maps, tcgen05 MMA and softmax helpers.
# ====================================================================================================================

HEADS = SPEC.attn_heads
LATENT = SPEC.kv_dim
WINDOW = SPEC.window
TOPK = SPEC.topk
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
NGW_ENGINE = 4
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
def role_q_load_engine(tma_q, s_q, q_full, q_empty, row0, row_step, n_items):
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
    R: cutlass.Constexpr[int] = tile_n // NGW_ENGINE
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
def _pv_engine(
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
                _pv_engine(
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
        _pv_engine(
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
def _final_norm_engine(
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
        norm = _final_norm_engine(final_max, reduced_sum, sink, scale_log2)
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
        role_q_load_engine(
            tma_q, s_q.data_ptr(), q_full.data_ptr(), q_empty.data_ptr(), cta, grid, n_items
        )
    elif warp < Int32(W_GATHER0 + NGW_ENGINE):
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
            norm = _final_norm_engine(big_m, l_tot, sink_v[i], scale_log2)
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
        role_q_load_engine(
            tma_q, s_q.data_ptr(), q_full.data_ptr(), q_empty.data_ptr(), b, Int32(0), Int32(1)
        )
    elif warp < Int32(W_GATHER0 + NGW_ENGINE):
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
        role_q_load_engine(
            tma_q, s_q.data_ptr(), q_full.data_ptr(), q_empty.data_ptr(), cta, grid, n_items
        )
    elif warp < Int32(W_GATHER0 + NGW_ENGINE):
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


# ====================================================================================================================
# Warp roles of the team CTAs' sparse MLA: KV gather, MMA, softmax, correction and epilogue, and the two-row-group
# split with the DSMEM pair merge.
# ====================================================================================================================

TILE_N = 128
# S buffers: lane panel 0, columns 128k (k = tile % 4); P_k: panel 1, column 128k + 32; O: panel 1, column 448
O_COL4 = 448
OUT_SLICE = 64
OUT_REGS = OUT_SLICE // 2
SLOT_BYTES = TILE_N * SLICE
SLOT_BY_POS = True
W_SM0 = 0
W_CTL0 = 4
W_G0 = 9
NGW = 7
REG_CTL = 232
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
                r4 = L.lds_v4(Int32(s_idx.toint()) + (j * Int32(TILE_N) + g * Int32(4)) * Int32(4))
                dst = Int32(_slot_ptr(s_kv, s_ext, slot).toint()) + g * Int32(4 * SLICE)
                L.tma_g4(
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
        if L.mbar_test_wait(
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
            L.st_g_v4(
                rowb + Int64(ch * 16), ob[ch * 4], ob[ch * 4 + 1], ob[ch * 4 + 2], ob[ch * 4 + 3]
            )
    if cutlass.const_expr(tr is not None):
        _stamp(tr, Int32(TS_STORED), warp == Int32(0) and lane == Int32(0))


# ===================================================================== 2 row groups x 4 V groups, DSMEM merge
RS24_O_COL = 384  # O accumulator: lane panel 1, columns 384-511 (N = 128); P_t at 128t + 32, t <= 2
# team4 (wide_v): a CTA owns a 256-column V slice (vslices = 2 of the 4 KV slices of a tile) instead of 128: the PV MMA
# is N = 256 and the O accumulator sits at columns 256-511; the pair merge (_merge_22) hands the partner the 128
# columns it owns
RS22_O_COL = 256
RS24_DRY_MASK = 2047  # dry pass: every merge address wraps inside this CTA's 2 KB stat/fin scratch
assert 3 * TILE_N // 4 <= RS22_O_COL and RS22_O_COL + 2 * SLICE <= TMEM_COLS


def rs24_rx_bytes(f32x, wide_v=False):
    # partner push: nchunk x [128 threads][16 B] partial O, then [128 threads][8 B] (max, sum); wide_v: twice the chunks
    return (8 if f32x else 4) * (2 if wide_v else 1) * 2048 + 1024


@_dsl_user_op
def _lds_vol(saddr, *, loc=None, ip=None):
    return Int32(
        L._asm(
            L.T.i32(), [L._v(saddr, loc, ip)], "ld.volatile.shared.u32 $0, [$1];", "=r,r", loc, ip
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
    tq,
    d,
    lsu_g: cutlass.Constexpr[int],
    s_kv_a,
    s_idx_a,
    kvf_a,
    kv_empty,
    pool: Int64,
    tr=None,
    wide_v: cutlass.Constexpr[bool] = False,
):
    # LSU-assisted gather (g = 1 CTAs, tile 4 = local tile 1, slots 4-7): groups [32 - lsu_g, 32) of every slot via
    # cp.async (128 threads, 8 per 128 B row slice, SW128 swizzle by hand), then a noinc arrive per thread and slot
    vslices: cutlass.Constexpr[int] = 2 if wide_v else 1
    k = d >> Int32(1)
    t = Int32(1)
    j = Int32(4)
    sbase = (t & Int32(1)) * Int32(4)
    epar = ((t >> Int32(1)) & Int32(1)) ^ Int32(1)
    if cutlass.const_expr(tr is not None):
        _stamp(tr, Int32(53), tq == Int32(0))
    for step in cutlass.range(NSLICE, unroll=1):
        sl = k * Int32(vslices) + Int32(vslices) + step & Int32(3)
        slot = sbase + step
        _wait(kv_empty + slot, epar)
        for p in cutlass.range_constexpr(lsu_g // 4):
            q = tq + Int32(p * LSU_ARR)
            r = Int32((32 - lsu_g) * 4) + (q >> Int32(3))
            c = q & Int32(7)
            row = cutlass.max(L.lds_u32(s_idx_a + (j * Int32(TILE_N) + r) * Int32(4)), Int32(0))
            src = pool + Int64(row) * Int64(LATENT) + Int64(sl * Int32(SLICE) + c * Int32(16))
            dst = (
                s_kv_a
                + slot * Int32(SLOT_BYTES)
                + r * Int32(SLICE)
                + ((c ^ (r & Int32(7))) << Int32(4))
            )
            L.cp_async16(dst, src)
        L.cp_async_mbar_arrive_noinc(kvf_a + slot * Int32(8))
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
    wide_v: cutlass.Constexpr[bool] = False,
):
    # above_ptr (t4split): the last tile's rows below `above` (selection-certain) are gathered at the early release
    vslices: cutlass.Constexpr[int] = 2 if wide_v else 1
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
            sl = k * Int32(vslices) + Int32(vslices + step) & Int32(3)
            slot = sbase + Int32(step)
            _wait(kv_empty + slot, epar)
            if gw == Int32(0):
                if lane == Int32(0):
                    if cutlass.const_expr(lsu_g > 0):
                        prims.mbarrier_arrive_expect_tx(kv_full + slot, g_t * Int32(4 * SLICE))
                    else:
                        prims.mbarrier_arrive_expect_tx(kv_full + slot, SLOT_BYTES)
            if g < hi_g:
                r4 = L.lds_v4(Int32(s_idx.toint()) + (j * Int32(TILE_N) + g * Int32(4)) * Int32(4))
                dst = Int32(_slot_ptr(s_kv, s_ext, slot).toint()) + g * Int32(4 * SLICE)
                L.tma_g4(
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
                    sl = k * Int32(vslices) + Int32(vslices + step) & Int32(3)
                    slot = sbase + Int32(step)
                    if g >= hi_g and g < g_t:
                        r4 = L.lds_v4(
                            Int32(s_idx.toint()) + (j * Int32(TILE_N) + g * Int32(4)) * Int32(4)
                        )
                        dst = Int32(_slot_ptr(s_kv, s_ext, slot).toint()) + g * Int32(4 * SLICE)
                        L.tma_g4(
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
    wide_v: cutlass.Constexpr[bool] = False,
):
    vslices: cutlass.Constexpr[int] = 2 if wide_v else 1
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    s_addr = tmem + t * Int32(TILE_N)
    sbase = (t & Int32(1)) * Int32(4)
    fpar = (t >> Int32(1)) & Int32(1)
    for step in cutlass.range_constexpr(NSLICE):
        c = k * Int32(vslices) + Int32(vslices + step) & Int32(3)
        slot = sbase + Int32(step)
        _wait(kv_full + slot, fpar)
        if cutlass.const_expr(pfence):
            # the slot may hold cp.async (generic proxy) bytes: order them before the tensor core's async-proxy reads
            L.fence_proxy_async_smem()
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
            if cutlass.const_expr(step < NSLICE - vslices):
                prims.tcgen05_commit(kv_empty + slot, group=prims.CTAGroup.CTA_1)
    if prims.elect_sync():
        prims.tcgen05_commit(s_full + t, group=prims.CTAGroup.CTA_1)


@cute.jit
def _pv_24(
    tmem,
    s_kv,
    s_ext,
    t,
    p_full,
    kv_empty,
    o_full,
    idesc_pv,
    wide_v: cutlass.Constexpr[bool] = False,
):
    vslices: cutlass.Constexpr[int] = 2 if wide_v else 1
    o_col: cutlass.Constexpr[int] = RS22_O_COL if wide_v else RS24_O_COL
    _wait(p_full + t, Int32(0))
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    a_addr = tmem + t * Int32(TILE_N) + Int32(TILE_N // 4) + Int32(PANEL1)
    if cutlass.const_expr(wide_v):
        a_addr = tmem + t * Int32(TILE_N // 4) + Int32(PANEL1)
    slot = (t & Int32(1)) * Int32(4) + Int32(NSLICE - vslices)
    d_addr = tmem + Int32(o_col) + Int32(PANEL1)
    vdesc = prims.Tcgen05SmemDesc.build(
        _slot_ptr(s_kv, s_ext, slot),
        leading_byte_offset=SLOT_BYTES if wide_v else 0,
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
        for v in cutlass.range_constexpr(vslices):
            prims.tcgen05_commit(kv_empty + slot + Int32(v), group=prims.CTAGroup.CTA_1)
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
    wide_v: cutlass.Constexpr[bool] = False,
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
        n_dim=256 if wide_v else 128,
        m_dim=HEADS,
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
        _qk_24(
            tmem, s_q, s_kv, s_ext, t, k, kv_full, kv_empty, s_full, qdesc, idesc_qk, pfence, wide_v
        )
        if cutlass.const_expr(tr is not None):
            _stamp(tr, Int32(TS_QK) + j0 + t, lane == Int32(0))
        if t == Int32(0) and j0 == Int32(0) or t > pvn:
            _pv_24(tmem, s_kv, s_ext, pvn, p_full, kv_empty, o_full, idesc_pv, wide_v)
            if cutlass.const_expr(tr is not None):
                _stamp(tr, Int32(TS_PV) + j0 + pvn, lane == Int32(0))
            pvn = pvn + Int32(1)
    _pv_24(tmem, s_kv, s_ext, pvn, p_full, kv_empty, o_full, idesc_pv, wide_v)
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
    L.wait_cl(cutlass.select_(wet, pvl_a, dry_a), cutlass.select_(wet, Int32(0), Int32(1)))
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
    L.wait_cl(cutlass.select_(wet, pr_a, dry_a), cutlass.select_(wet, Int32(0), Int32(1)))
    me = L.cluster_ctarank()
    trank = cutlass.select_(wet, me ^ Int32(1), me)
    amask = cutlass.select_(wet, Int32(-1), Int32(RS24_DRY_MASK))
    rc = L.mapa_u32(cutlass.select_(wet, rx_s, stat_s), trank)
    mc = L.mapa_u32(cutlass.select_(wet, rx_a, dry_a), trank)
    if cutlass.const_expr(f32x):
        for i in cutlass.range_constexpr(8):
            L.st_async_v4(
                rc + (((Int32(i * 128) + tq) * Int32(16)) & amask),
                mc,
                L.u32_of_f32(Float32(po[4 * i])),
                L.u32_of_f32(Float32(po[4 * i + 1])),
                L.u32_of_f32(Float32(po[4 * i + 2])),
                L.u32_of_f32(Float32(po[4 * i + 3])),
            )
    else:
        for i in cutlass.range_constexpr(4):
            L.st_async_v4(
                rc + (((Int32(i * 128) + tq) * Int32(16)) & amask),
                mc,
                _pack_bf16x2(Float32(po[8 * i]), Float32(po[8 * i + 1])),
                _pack_bf16x2(Float32(po[8 * i + 2]), Float32(po[8 * i + 3])),
                _pack_bf16x2(Float32(po[8 * i + 4]), Float32(po[8 * i + 5])),
                _pack_bf16x2(Float32(po[8 * i + 6]), Float32(po[8 * i + 7])),
            )
    L.st_async_v2(
        rc + ((Int32(data_b) + tq * Int32(8)) & amask), mc, L.u32_of_f32(m_me), L.u32_of_f32(l_me)
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
    L.wait_cl(cutlass.select_(wet, rx_a, dry_a), Int32(0))
    if cutlass.const_expr(tr is not None):
        _stamp(tr, Int32(123), tq == Int32(0))
    lb = cutlass.select_(wet, rx_s, stat_s)
    xs = []
    for i in cutlass.range_constexpr(nchunk):
        xs.append(L.lds_v4(lb + (((Int32(i * 128) + tq) * Int32(16)) & amask)))
    sa = lb + ((Int32(data_b) + tq * Int32(8)) & amask)
    m_p = L.f32_of_u32(Uint32(L.lds_u32(sa)))
    l_p = L.f32_of_u32(Uint32(L.lds_u32(sa + Int32(4))))
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
                acc[4 * i + e] = acc[4 * i + e] + w_p * L.f32_of_u32(Uint32(xs[i][e]))
    else:
        for i in cutlass.range_constexpr(4):
            for e in cutlass.range_constexpr(4):
                wd = Uint32(xs[i][e])
                acc[8 * i + 2 * e] = acc[8 * i + 2 * e] + w_p * L.f32_of_u32(wd << Uint32(16))
                acc[8 * i + 2 * e + 1] = acc[8 * i + 2 * e + 1] + w_p * L.f32_of_u32(
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
        L.st_g_v4(rowb + Int64(ch * 16), ob[ch * 4], ob[ch * 4 + 1], ob[ch * 4 + 2], ob[ch * 4 + 3])
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
    L.wait_cl(cutlass.select_(wet, pvl_a, dry_a), cutlass.select_(wet, Int32(0), Int32(1)))
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
        L.wait_cl(cutlass.select_(wet, pr_a, dry_a), cutlass.select_(wet, Int32(0), Int32(1)))
        me = L.cluster_ctarank()
        trank = cutlass.select_(wet, me ^ Int32(1), me)
        rc = L.mapa_u32(cutlass.select_(wet, rx_s, stat_s), trank)
        mc = L.mapa_u32(cutlass.select_(wet, rx_a, dry_a), trank)
        for i in cutlass.range_constexpr(4):
            L.st_async_v4(
                rc + (((Int32(i * 128) + tq) * Int32(16)) & amask),
                mc,
                _pack_bf16x2(Float32(o0[8 * i]), Float32(o0[8 * i + 1])),
                _pack_bf16x2(Float32(o0[8 * i + 2]), Float32(o0[8 * i + 3])),
                _pack_bf16x2(Float32(o0[8 * i + 4]), Float32(o0[8 * i + 5])),
                _pack_bf16x2(Float32(o0[8 * i + 6]), Float32(o0[8 * i + 7])),
            )
        for i in cutlass.range_constexpr(4):
            L.st_async_v4(
                rc + (((Int32((i + 4) * 128) + tq) * Int32(16)) & amask),
                mc,
                _pack_bf16x2(Float32(o1[8 * i]), Float32(o1[8 * i + 1])),
                _pack_bf16x2(Float32(o1[8 * i + 2]), Float32(o1[8 * i + 3])),
                _pack_bf16x2(Float32(o1[8 * i + 4]), Float32(o1[8 * i + 5])),
                _pack_bf16x2(Float32(o1[8 * i + 6]), Float32(o1[8 * i + 7])),
            )
        L.st_async_v2(
            rc + ((Int32(8 * 2048) + tq * Int32(8)) & amask),
            mc,
            L.u32_of_f32(m_me),
            L.u32_of_f32(l_me),
        )
        if cutlass.const_expr(tr is not None):
            _stamp(tr, Int32(121), tq == Int32(0))
            _stamp(tr, Int32(TS_STORED), tq == Int32(0))
    else:
        sink = Float32((sink_ptr + head).load())
        # the partner's push landed (B_RX expects it from kernel start); dry: a parity-1 wait returns at once
        L.wait_cl(cutlass.select_(wet, rx_a, dry_a), cutlass.select_(wet, Int32(0), Int32(1)))
        if cutlass.const_expr(tr is not None):
            _stamp(tr, Int32(123), tq == Int32(0))
        lb = cutlass.select_(wet, rx_s, stat_s)
        xs = []
        for i in cutlass.range_constexpr(8):
            xs.append(L.lds_v4(lb + (((Int32(i * 128) + tq) * Int32(16)) & amask)))
        sa = lb + ((Int32(8 * 2048) + tq * Int32(8)) & amask)
        m_p = L.f32_of_u32(Uint32(L.lds_u32(sa)))
        l_p = L.f32_of_u32(Uint32(L.lds_u32(sa + Int32(4))))
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
                    a0 = wn_me * Float32(oc[8 * i + 2 * e]) + wn_p * L.f32_of_u32(wd << Uint32(16))
                    a1 = wn_me * Float32(oc[8 * i + 2 * e + 1]) + wn_p * L.f32_of_u32(
                        wd & Uint32(4294901760)
                    )
                    ob.append(_pack_bf16x2(a0, a1))
            # dry: the same stores with throw-away values (this CTA's own output block; the real pass overwrites it)
            rowb = out_addr + Int64(
                (head * Int32(LATENT) + k * Int32(128) + Int32(64 * c) + half * Int32(32))
                * Int32(2)
            )
            for ch in cutlass.range_constexpr(4):
                L.st_g_v4(
                    rowb + Int64(ch * 16),
                    ob[ch * 4],
                    ob[ch * 4 + 1],
                    ob[ch * 4 + 2],
                    ob[ch * 4 + 3],
                )
        if cutlass.const_expr(tr is not None):
            _stamp(tr, Int32(TS_STORED), tq == Int32(0))


@cute.jit
def _merge_22(
    tmem, tq, d, m_me, l_me, sink_ptr, out_addr: Int64, rx_s, pvl_a, rx_a, pr_a, scale_log2: Float32
):
    # team4 pair merge (CTAs of the two row groups g = 0, 1 that share a V slice k): each CTA pushes the 128 columns of
    # its 256-column O accumulator that the partner owns (BF16, st.async into the partner's rx buffer, completing its
    # rx mbarrier) with its (max, sum), then combines its own columns with the partner's, applies the sink and the
    # final normalisation, and writes its 128 columns of the output.
    warp = tq >> Int32(5)
    lane = tq & Int32(31)
    head = warp * Int32(16) + (lane & Int32(15))
    half = lane >> Int32(4)
    lane_base = warp * Int32(32) << Int32(16)
    g = d & Int32(1)
    k = d >> Int32(1)
    sink = Float32((sink_ptr + head).load())
    L.wait_cl(pvl_a, Int32(0))
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    cute.arch.cluster_wait()
    L.wait_cl(pr_a, Int32(0))
    peer = L.cluster_ctarank() ^ Int32(1)
    rc = L.mapa_u32(rx_s, peer)
    mc = L.mapa_u32(rx_a, peer)
    o_base = tmem + lane_base + Int32(RS22_O_COL + PANEL1)
    for ch in cutlass.range(2, unroll=1):
        src = o_base + Int32(128) * (g ^ Int32(1)) + ch * Int32(64)
        po = prims.tcgen05_ld(
            prims.Tcgen05LdStShape.SHAPE_16X32BX2,
            prims.make_tmem_ptr(src, Float32),
            num=32,
            offset=Int64(32),
        )
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        for i in cutlass.range_constexpr(4):
            dst = rc + (ch * Int32(512) + Int32(i * 128) + tq) * Int32(16)
            L.st_async_v4(
                dst,
                mc,
                _pack_bf16x2(Float32(po[8 * i]), Float32(po[8 * i + 1])),
                _pack_bf16x2(Float32(po[8 * i + 2]), Float32(po[8 * i + 3])),
                _pack_bf16x2(Float32(po[8 * i + 4]), Float32(po[8 * i + 5])),
                _pack_bf16x2(Float32(po[8 * i + 6]), Float32(po[8 * i + 7])),
            )
    L.st_async_v2(rc + Int32(16384) + tq * Int32(8), mc, L.u32_of_f32(m_me), L.u32_of_f32(l_me))
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    L.wait_cl(rx_a, Int32(0))
    sa = rx_s + Int32(16384) + tq * Int32(8)
    m_p = L.f32_of_u32(Uint32(L.lds_u32(sa)))
    l_p = L.f32_of_u32(Uint32(L.lds_u32(sa + Int32(4))))
    m = cute.math.max(m_me, m_p, ftz=True)
    w_me = cute.math.exp2(scale_log2 * (m_me - m), fastmath=True)
    w_p = cute.math.exp2(scale_log2 * (m_p - m), fastmath=True)
    l_sum = w_me * l_me + w_p * l_p
    norm = _final_norm(m, l_sum, sink, scale_log2)
    for ch in cutlass.range(2, unroll=1):
        src = o_base + g * Int32(128) + ch * Int32(64)
        own = prims.tcgen05_ld(
            prims.Tcgen05LdStShape.SHAPE_16X32BX2,
            prims.make_tmem_ptr(src, Float32),
            num=32,
            offset=Int64(32),
        )
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        xs = []
        for i in cutlass.range_constexpr(4):
            xs.append(L.lds_v4(rx_s + (ch * Int32(512) + Int32(i * 128) + tq) * Int32(16)))
        acc = []
        for i in cutlass.range_constexpr(32):
            acc.append(w_me * Float32(own[i]))
        for i in cutlass.range_constexpr(4):
            for e in cutlass.range_constexpr(4):
                wd = Uint32(xs[i][e])
                acc[8 * i + 2 * e] = acc[8 * i + 2 * e] + w_p * L.f32_of_u32(wd << Uint32(16))
                acc[8 * i + 2 * e + 1] = acc[8 * i + 2 * e + 1] + w_p * L.f32_of_u32(
                    wd & Uint32(4294901760)
                )
        ob = []
        for pi in cutlass.range_constexpr(16):
            fp = cute.arch.mul_packed_f32x2(
                (norm, norm), (acc[2 * pi], acc[2 * pi + 1]), ftz=False, rnd="rn"
            )
            ob.append(_pack_bf16x2(fp[0], fp[1]))
        rowb = out_addr + Int64(
            (
                head * Int32(LATENT)
                + k * Int32(256)
                + g * Int32(128)
                + ch * Int32(64)
                + half * Int32(32)
            )
            * Int32(2)
        )
        for i in cutlass.range_constexpr(4):
            L.st_g_v4(rowb + Int64(i * 16), ob[i * 4], ob[i * 4 + 1], ob[i * 4 + 2], ob[i * 4 + 3])


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
    wide_v: cutlass.Constexpr[bool] = False,
):
    # Softmax of this CTA's tiles, then the pair merge. With dry_merge the merge first runs dry at the same call site
    # in the idle window (g == 0: after the SWA tile; g == 1: before its first tile) so the real merge runs warm.
    half_n: cutlass.Constexpr[int] = TILE_N // 2
    npk: cutlass.Constexpr[int] = TILE_N // 8
    o_col: cutlass.Constexpr[int] = RS22_O_COL if wide_v else RS24_O_COL
    ochunks: cutlass.Constexpr[int] = 4 if wide_v else 2
    tid0 = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % Int32(4)) * Int32(
        32
    ) + cute.arch.lane_idx()
    L.sts_u32(zaddr, Int32(0))
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
            if cutlass.const_expr(wide_v):
                p_addr = tmem + lane_base + t * Int32(TILE_N // 4)
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
                    for ch in cutlass.range_constexpr(ochunks):
                        ptr = prims.make_tmem_ptr(
                            tmem + lane_base + Int32(o_col + 64 * ch) + Int32(PANEL1), Float32
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
        if cutlass.const_expr(wide_v):
            _merge_22(
                tmem, tq, d, m_run, l_tot, sink_ptr, out_addr, rx_s, pvl_a, rx_a, pr_a, scale_log2
            )
        elif cutlass.const_expr(asym):
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
