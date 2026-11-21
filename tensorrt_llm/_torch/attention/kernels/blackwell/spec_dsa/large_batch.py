# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""specDSA large-batch kernels (CuTe DSL, SM100): the persistent indexer kernel (one CTA per SM: FP8 logits,
sampled-threshold top-512, mapping to compressed-KV rows, ready queue) and the attention kernel launched behind it with
programmatic dependent launch.

The kernels are specialized for the ``DS_V4_FLASH`` spec of ``config.py``: their structural constants are defined from
it and the kernel classes reject a config with another ``kernel_key``.
"""

from __future__ import annotations

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as xcuda
import cutlass.pipeline as pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Boolean, Float4E2M1FN, Float8E8M0FNU, Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm as _llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05 import find_tmem_tensor_col_offset
from cutlass.cutlass_dsl import T as _T
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from . import large_batch_lib as L
from .config import DS_V4_FLASH as SPEC
from .config import DS_V4_FLASH_FP4 as SPEC_FP4
from .config import NUM_SMS, SpecDsaConfig, require_built_for

# ====================================================================================================================
# Sparse MLA tile loads, MMA and softmax helpers shared by the two kernels.
# ====================================================================================================================

HEADS = SPEC.attn_heads
LATENT = SPEC.kv_dim
WINDOW = SPEC.window
NROWS = SPEC.selected_rows
TILE_N = 128
NTILES = NROWS // TILE_N
SLICE = 128
NSLICE = LATENT // SLICE
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


# ====================================================================================================================
# Persistent sparse-attention tail, consuming exact indexer-selected rows as the indexer publishes them.
# ====================================================================================================================

# mbarrier slots (all single-arrival)
NKVG = 3  # KV slot groups (one tile each): tile t uses group t % NKVG
B_Q = 0  # four Q slices                                  loader -> MMA warp
B_KV = 4  # 4 * NKVG KV slots                              loader -> MMA warp
B_SFULL = 16  # two S buffers: QK done                         MMA warp -> softmax
B_PV = 18  # PV done                                        MMA warp -> softmax
B_GFREE = 19  # NKVG KV groups, released by the PV that used them  MMA warp -> loader
B_QFREE = 22  # Q slices, released after the last QK of a row  MMA warp -> loader
B_PRDY = 23  # P stored and O rescaled                        softmax -> MMA warp
B_ROW = 24  # two row hand-off slots                         loader -> MMA warp, softmax
B_DRAIN = 26
NBAR = 27
TM_SB = 128  # S buffer t & 1 at column 128 * (t & 1); O at A.TM_O (columns 256-511, both 16-lane panels)


@cute.jit
def _issue_pv(abar, tp, jp, tmem, kv_base, idesc_pv):
    # O += P(tp) V(tp) once the softmax warps have stored P and rescaled O; releases the tile's KV group.
    _wait(abar + Int32(B_PRDY), tp & Int32(1))
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
    if prims.elect_sync():
        gp = tp - tp // Int32(NKVG) * Int32(NKVG)
        g4 = gp * Int32(4)
        s_addr = tmem + (tp & Int32(1)) * Int32(TM_SB)
        for v in cutlass.range_constexpr(NSLICE):
            vdesc = prims.Tcgen05SmemDesc.build(
                kv_base + (g4 + Int32(v)) * Int32(SLOT_BYTES),
                leading_byte_offset=0,
                stride_byte_offset=1024,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            )
            d_addr = tmem + Int32(TM_O + (v & 1) * 128 + (v >> 1) * PANEL1)
            a_addr = s_addr + Int32(TILE_N // 4 + (v >> 1) * PANEL1)
            for kb in cutlass.range_constexpr(TILE_N // 32):
                en = Boolean(True)
                if cutlass.const_expr(kb == 0):
                    en = Boolean(jp > Int32(0))
                prims.tcgen05_mma(
                    prims.Tcgen05MMAKind.F8F6F4,
                    prims.CTAGroup.CTA_1,
                    prims.make_tmem_ptr(d_addr, Float32),
                    prims.make_tmem_ptr(a_addr + Int32(8 * kb), E4M3),
                    vdesc.advance_start_address(kb * 32 * SLICE),
                    idesc_pv,
                    en,
                )
        prims.tcgen05_commit(abar + Int32(B_PV), group=prims.CTAGroup.CTA_1)
        prims.tcgen05_commit(abar + Int32(B_GFREE) + gp, group=prims.CTAGroup.CTA_1)


@cute.jit
def _mma_warp(abar, sRow, q_base, kv_base, tmem):
    # Tensor-pipe order per row: QK0, QK1, PV0, QK2, PV1, ..., QK4, PV3, PV4. QK(t) lands in S buffer t & 1 while
    # the softmax warps work on S buffer (t - 1) & 1; the pipe executes in issue order, so QK(t) never overwrites
    # P(t - 2) before PV(t - 2) has read it.
    idesc_qk = prims.Tcgen05InstrDesc.build(
        c_dtype=Float32, a_dtype=E4M3, b_dtype=E4M3, n_dim=TILE_N, m_dim=HEADS
    )
    idesc_pv = prims.Tcgen05InstrDesc.build(
        c_dtype=Float32,
        a_dtype=E4M3,
        b_dtype=E4M3,
        a_major=0,
        b_major=1,
        n_dim=SLICE,
        m_dim=HEADS,
    )
    n = Int32(0)
    t = Int32(0)
    ph = Int32(0)
    _wait(abar + Int32(B_ROW) + (n & Int32(1)), n >> Int32(1) & Int32(1))
    b = Int32(sRow[n & Int32(1)])
    while b >= Int32(0):
        for j in cutlass.range(NTILES, unroll=1):
            tt = t + j
            g4 = (tt - tt // Int32(NKVG) * Int32(NKVG)) * Int32(4)
            s_addr = tmem + (tt & Int32(1)) * Int32(TM_SB)
            if j == Int32(0):
                for c in cutlass.range_constexpr(NSLICE):
                    _wait(abar + Int32(B_Q + c), ph >> Int32(c) & Int32(1))
                ph = ph ^ Int32(15)
            for c in cutlass.range_constexpr(NSLICE):
                _wait(
                    abar + Int32(B_KV) + g4 + Int32(c), ph >> Int32(16) + g4 + Int32(c) & Int32(1)
                )
            ph = ph ^ Int32(15) << Int32(16) + g4
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            if prims.elect_sync():
                for c in cutlass.range_constexpr(NSLICE):
                    qdesc = prims.Tcgen05SmemDesc.build(
                        q_base + Int32(c * Q_SLICE_BYTES),
                        leading_byte_offset=Q_SLICE_BYTES,
                        stride_byte_offset=1024,
                        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
                    )
                    kdesc = prims.Tcgen05SmemDesc.build(
                        kv_base + (g4 + Int32(c)) * Int32(SLOT_BYTES),
                        leading_byte_offset=SLOT_BYTES,
                        stride_byte_offset=1024,
                        layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
                    )
                    for kb in cutlass.range_constexpr(SLICE // 32):
                        en = Boolean(True)
                        if cutlass.const_expr(kb == 0):
                            en = Boolean(c > 0)
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F8F6F4,
                            prims.CTAGroup.CTA_1,
                            prims.make_tmem_ptr(s_addr, Float32),
                            qdesc.advance_start_address(kb * 32),
                            kdesc.advance_start_address(kb * 32),
                            idesc_qk,
                            en,
                        )
                prims.tcgen05_commit(
                    abar + Int32(B_SFULL) + (tt & Int32(1)), group=prims.CTAGroup.CTA_1
                )
                if j == Int32(NTILES - 1):
                    prims.tcgen05_commit(abar + Int32(B_QFREE), group=prims.CTAGroup.CTA_1)
            if j >= Int32(1):
                _issue_pv(abar, tt - Int32(1), j - Int32(1), tmem, kv_base, idesc_pv)
        _issue_pv(abar, t + Int32(NTILES - 1), Int32(NTILES - 1), tmem, kv_base, idesc_pv)
        t = t + Int32(NTILES)
        n = n + Int32(1)
        _wait(abar + Int32(B_ROW) + (n & Int32(1)), n >> Int32(1) & Int32(1))
        b = Int32(sRow[n & Int32(1)])
    if prims.elect_sync():
        prims.tcgen05_commit(abar + Int32(B_DRAIN), group=prims.CTAGroup.CTA_1)
    _wait(abar + Int32(B_DRAIN), Int32(0))


@cute.jit
def _softmax_row(
    rtid,
    b,
    t,
    gv0,
    gv1,
    gv2,
    gv3,
    gv4,
    tmem,
    abar,
    s_msk,
    sink_ptr,
    out_addr: Int64,
    scale_log2: Float32,
):
    # Same arithmetic as att_row (online softmax with sinks, P in E4M3 scaled by 448), with S read
    # from TMEM once per tile and the PV of the previous tile running underneath.
    wl = rtid >> Int32(5)
    lane = rtid & Int32(31)
    head = wl * Int32(16) + (lane & Int32(15))
    half = lane >> Int32(4)
    lane_base = wl * Int32(32) << Int32(16)
    sink = Float32((sink_ptr + head).load())
    Ls = scale_log2
    m_run = Float32(NEG_FLT_MAX)
    l_run = Float32(0.0)
    for j in cutlass.range(NTILES, unroll=1):
        tt = t + j
        sb = tt & Int32(1)
        gsel = _pick(j, gv0, gv1, gv2, gv3, gv4)
        vm = Int32(cute.arch.vote_ballot_sync(gsel >= Int32(0)))
        if lane == Int32(0):
            s_msk[sb * Int32(4) + wl] = vm
        cute.arch.barrier(barrier_id=4, number_of_threads=128)
        mk = Int32(s_msk[sb * Int32(4) + half * Int32(2)])
        mk2 = Int32(s_msk[sb * Int32(4) + half * Int32(2) + Int32(1)])
        _wait(abar + Int32(B_SFULL) + sb, tt >> Int32(1) & Int32(1))
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        tsa = tmem + sb * Int32(TM_SB) + lane_base
        s0 = prims.tcgen05_ld(
            prims.Tcgen05LdStShape.SHAPE_16X32BX2,
            prims.make_tmem_ptr(tsa, Float32),
            num=32,
            offset=Int64(64),
        )
        s1 = prims.tcgen05_ld(
            prims.Tcgen05LdStShape.SHAPE_16X32BX2,
            prims.make_tmem_ptr(tsa + Int32(32), Float32),
            num=32,
            offset=Int64(64),
        )
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        xs = []
        for ch in cutlass.range_constexpr(2):
            for i in cutlass.range_constexpr(32):
                x = Float32(s0[i])
                mword = mk
                if cutlass.const_expr(ch == 1):
                    x = Float32(s1[i])
                    mword = mk2
                if mword >> Int32(i) & Int32(1) == Int32(0):
                    x = Float32(-float("inf"))
                xs.append(x)
        old_max = m_run
        mx0 = old_max
        mx1 = Float32(NEG_FLT_MAX)
        mx2 = Float32(NEG_FLT_MAX)
        mx3 = Float32(NEG_FLT_MAX)
        for i in cutlass.range_constexpr(0, 64, 4):
            mx0 = cute.math.max(mx0, xs[i], ftz=True)
            mx1 = cute.math.max(mx1, xs[i + 1], ftz=True)
            mx2 = cute.math.max(mx2, xs[i + 2], ftz=True)
            mx3 = cute.math.max(mx3, xs[i + 3], ftz=True)
        lmax = cute.math.max(
            cute.math.max(mx0, mx1, ftz=True), cute.math.max(mx2, mx3, ftz=True), ftz=True
        )
        lmax = cute.math.max(lmax, _shfl_xor_f32(lmax, 16), ftz=True)
        new_max = lmax
        safe_max = new_max
        if safe_max == Float32(NEG_FLT_MAX):
            safe_max = Float32(0.0)
        neg_scaled = -Ls * safe_max
        bias = neg_scaled + Float32(LOG2_448)
        sm0 = Float32(0.0)
        sm1 = Float32(0.0)
        sm2 = Float32(0.0)
        sm3 = Float32(0.0)
        regs = []
        for pk in cutlass.range_constexpr(16):
            x01 = cute.arch.fma_packed_f32x2(
                (xs[4 * pk], xs[4 * pk + 1]), (Ls, Ls), (bias, bias), ftz=False, rnd="rn"
            )
            x23 = cute.arch.fma_packed_f32x2(
                (xs[4 * pk + 2], xs[4 * pk + 3]), (Ls, Ls), (bias, bias), ftz=False, rnd="rn"
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
        _tmem_st_p8(p_addr, regs[0:8])
        _tmem_st_p8(p_addr + Int32(8), regs[8:16])
        _tmem_st_p8(p_addr + Int32(PANEL1), regs[0:8])
        _tmem_st_p8(p_addr + Int32(PANEL1 + 8), regs[8:16])
        exp_scale = cute.math.exp2(Ls * (old_max - new_max), fastmath=True)
        l_run = exp_scale * l_run + local_sum
        m_run = new_max
        if j > Int32(0):
            # O is still being accumulated by PV(tt - 1): wait before rescaling it.
            _wait(abar + Int32(B_PV), tt - Int32(1) & Int32(1))
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            ident = old_max == new_max
            scale = Float32(1.0)
            if not ident:
                scale = exp_scale
            skip_c = prims.vote_sync(4294967295, ident, prims.VoteSync.ALL)
            if not skip_c:
                _rescale_o(tmem, lane_base, scale)
        prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.barrier(barrier_id=4, number_of_threads=128)
        if rtid == Int32(0):
            prims.mbarrier_arrive(abar + Int32(B_PRDY))
    tl = t + Int32(NTILES - 1)
    _wait(abar + Int32(B_PV), tl & Int32(1))
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
                    (norm, norm), (Float32(o[2 * pi]), Float32(o[2 * pi + 1])), ftz=False, rnd="rn"
                )
                ob.append(_pack_bf16x2(fp[0], fp[1]))
            for k in cutlass.range_constexpr(4):
                dst = cutlass.inttoptr(
                    out_base + Int64((v * SLICE + hc * 32 + k * 8) * 2), mem_space=1, dtype=Int32
                )
                dst.store(
                    cutlass.Vector.from_elements(tuple((ob[4 * k + i] for i in range(4))), Int32),
                    alignment=16,
                )
    # O has been read: the next row's first PV (after its P-ready barrier) may overwrite it.
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)


@cute.jit
def _issue_kv_tile(slot_base, mbar_base, tma_swa, tma_cmp, use_swa, gidx_t, wl, lane):
    # All four 128-B slices of each gathered 512-B row are requested back to back, so the row's DRAM page is
    # opened once instead of once per slice.
    if wl == Int32(0):
        if prims.elect_sync():
            for c in cutlass.range_constexpr(NSLICE):
                prims.mbarrier_arrive_expect_tx(mbar_base + Int32(c), SLOT_BYTES)
    dst0 = slot_base + wl * Int32(32 * SLICE)
    for q in cutlass.range(8, unroll=1):
        r0 = _shfl_idx_i32(gidx_t, Int32(4 * q))
        r1 = _shfl_idx_i32(gidx_t, Int32(4 * q + 1))
        r2 = _shfl_idx_i32(gidx_t, Int32(4 * q + 2))
        r3 = _shfl_idx_i32(gidx_t, Int32(4 * q + 3))
        for c in cutlass.range_constexpr(NSLICE):
            dst = dst0 + Int32(c * SLOT_BYTES) + Int32(4 * q * SLICE)
            if use_swa:
                _tma_gather4(
                    dst, tma_swa.get_ptr(), mbar_base + Int32(c), Int32(c * SLICE), r0, r1, r2, r3
                )
            else:
                _tma_gather4(
                    dst, tma_cmp.get_ptr(), mbar_base + Int32(c), Int32(c * SLICE), r0, r1, r2, r3
                )


class AttentionTail:
    def __init__(self, config: SpecDsaConfig | None = None, exit_early: bool = False):
        require_built_for(
            config, SPEC_FP4 if config is not None and config.index_k_dtype == "fp4" else SPEC
        )
        self.threads = 288  # warps 0-3 loader, 4-7 softmax/epilogue, 8 MMA issue
        # exit_early: the "last CTA to leave resets the ready queue" accounting runs on the loader thread right
        # after the CTA's final claim, not behind the last row's epilogue (where its fence waits for the output stores)
        self.exit_early = bool(exit_early)

    @cute.jit
    def __call__(
        self,
        q_att: cute.Tensor,
        swa_pool: cute.Tensor,
        c_pool: cute.Tensor,
        gidx: cute.Tensor,
        sinks: cute.Tensor,
        out: cute.Tensor,
        aq: cute.Tensor,
        softmax_scale: Float32,
        nctas: Int32,
        stream: cuda.CUstream,
    ):
        tma_q = xcuda.create_tensor_map_tiled(
            global_address=q_att.iterator.toint(),
            dtype=cutlass.Float8E4M3FN,
            global_dims=[LATENT, HEADS, q_att.shape[0]],
            global_strides=[LATENT // 16, HEADS * LATENT // 16],
            box_dims=[SLICE, HEADS, 1],
            swizzle=xcuda.TensorMapSwizzle.s128b,
        )
        tma_swa = xcuda.create_tensor_map_tiled(
            global_address=swa_pool.iterator.toint(),
            dtype=cutlass.Float8E4M3FN,
            global_dims=[LATENT, swa_pool.shape[0]],
            global_strides=[LATENT // 16],
            box_dims=[SLICE, 1],
            swizzle=xcuda.TensorMapSwizzle.s128b,
        )
        tma_cmp = xcuda.create_tensor_map_tiled(
            global_address=c_pool.iterator.toint(),
            dtype=cutlass.Float8E4M3FN,
            global_dims=[LATENT, c_pool.shape[0]],
            global_strides=[LATENT // 16],
            box_dims=[SLICE, 1],
            swizzle=xcuda.TensorMapSwizzle.s128b,
        )
        self.kernel(tma_q, tma_swa, tma_cmp, gidx, sinks, out, aq, softmax_scale).launch(
            grid=(nctas, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=True,
        )

    @cute.jit
    def _claim(self, tid, aq_base: Int64, nrows, sRow, sLd, abar, n):
        # Loader thread 0 takes the next completed row (-1 once all rows are taken) and hands it to both groups.
        if tid == Int32(0):
            k = L.atom_add_i32(aq_base + Int64(L.AQ_CLAIM * 4), Int32(1))
            b = Int32(-1)
            if k < nrows:
                ea = aq_base + Int64(L.AQ_ENTRY * 4) + Int64(k) * Int64(4)
                b = L.ld_acquire_i32(ea)
                while b < Int32(0):
                    L.nanosleep(Int32(256))
                    b = L.ld_acquire_i32(ea)
                L.st_relaxed_i32(ea, Int32(-1))
            sRow[n & Int32(1)] = b
            sLd[n & Int32(1)] = b
            prims.mbarrier_arrive(abar + Int32(B_ROW) + (n & Int32(1)))
        cute.arch.barrier(barrier_id=2, number_of_threads=128)
        return Int32(sLd[n & Int32(1)])

    @cute.kernel
    def kernel(
        self,
        tma_q: cutlass.GridConstant[xcuda.TensorMap],
        tma_swa: cutlass.GridConstant[xcuda.TensorMap],
        tma_cmp: cutlass.GridConstant[xcuda.TensorMap],
        gidx: cute.Tensor,
        sinks: cute.Tensor,
        out: cute.Tensor,
        aq: cute.Tensor,
        softmax_scale: Float32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        gdim, _, _ = cute.arch.grid_dim()
        smem = cutlass.utils.SmemAllocator()
        sQ = smem.allocate_tensor(
            cutlass.Float8E4M3FN, cute.make_layout(NSLICE * Q_SLICE_BYTES), byte_alignment=1024
        )
        sKV = smem.allocate_tensor(
            cutlass.Float8E4M3FN, cute.make_layout(4 * NKVG * SLOT_BYTES), byte_alignment=1024
        )
        sBar = smem.allocate_tensor(Int64, cute.make_layout(32), byte_alignment=16)
        sMask = smem.allocate_tensor(Int32, cute.make_layout(8), byte_alignment=16)
        sRow = smem.allocate_tensor(Int32, cute.make_layout(4), byte_alignment=16)
        sLd = smem.allocate_tensor(Int32, cute.make_layout(4), byte_alignment=16)
        sTmem = smem.allocate_tensor(Int32, cute.make_layout(1), byte_alignment=16)
        if tid < Int32(NBAR):
            prims.mbarrier_init(sBar.iterator.raw_ptr() + tid, 1)
        prims.fence_mbarrier_init()
        tmem_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads)
        tmem = cutlass.utils.TmemAllocator(
            sTmem.iterator, barrier_for_retrieve=tmem_barrier, allocator_warp_id=0, is_two_cta=False
        )
        tmem.allocate(512)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(Float32)
        tmem_addr = cute.arch.load(sTmem.iterator, Int32, sem="relaxed", scope="cta")
        q_base = sQ.iterator.raw_ptr()
        kv_base = sKV.iterator.raw_ptr()
        abar = sBar.iterator.raw_ptr()
        aq_base = aq.iterator.toint()
        gidx_base = gidx.iterator.toint()
        nrows = Int32(gidx.shape[0])
        if tid < Int32(128):
            wl = tid >> Int32(5)
            lane = tid & Int32(31)
            n = Int32(0)
            t = Int32(0)
            b = self._claim(tid, aq_base, nrows, sRow, sLd, abar, n)
            while b >= Int32(0):
                if n >= Int32(1):
                    _wait(abar + Int32(B_QFREE), n - Int32(1) & Int32(1))
                if tid == Int32(0):
                    for c in cutlass.range_constexpr(NSLICE):
                        prims.mbarrier_arrive_expect_tx(abar + Int32(B_Q + c), Q_SLICE_BYTES)
                        prims.cp_async_bulk_tensor_shared_cta_global(
                            q_base + Int32(c * Q_SLICE_BYTES),
                            tma_q.get_ptr(),
                            [Int32(c * SLICE), Int32(0), b],
                            abar + Int32(B_Q + c),
                        )
                grow_base = gidx_base + Int64(b) * Int64(NROWS * 4) + Int64(tid * Int32(4))
                for j in cutlass.range(NTILES, unroll=1):
                    tq = t // Int32(NKVG)
                    g = t - tq * Int32(NKVG)
                    if t >= Int32(NKVG):
                        _wait(abar + Int32(B_GFREE) + g, tq - Int32(1) & Int32(1))
                    grow = _grow(L.ld_cg_i32(grow_base + Int64(j * Int32(TILE_N * 4))))
                    _issue_kv_tile(
                        kv_base + g * Int32(4 * SLOT_BYTES),
                        abar + Int32(B_KV) + g * Int32(4),
                        tma_swa,
                        tma_cmp,
                        j == Int32(0),
                        grow,
                        wl,
                        lane,
                    )
                    t = t + Int32(1)
                n = n + Int32(1)
                b = self._claim(tid, aq_base, nrows, sRow, sLd, abar, n)
            if cutlass.const_expr(self.exit_early):
                if tid == Int32(0):
                    L.fence_acq_rel_gpu()
                    d_e = L.atom_add_i32(aq_base + Int64(L.AQ_DONE * 4), Int32(1))
                    if d_e == Int32(gdim) - Int32(1):
                        L.fence_acq_rel_gpu()
                        L.st_relaxed_i32(aq_base + Int64(L.AQ_PUB * 4), Int32(0))
                        L.st_relaxed_i32(aq_base + Int64(L.AQ_CLAIM * 4), Int32(0))
                        L.st_relaxed_i32(aq_base + Int64(L.AQ_DONE * 4), Int32(0))
        elif tid < Int32(256):
            rtid = tid - Int32(128)
            scale_log2 = softmax_scale * Float32(LOG2_E)
            out_addr = out.iterator.toint()
            n = Int32(0)
            t = Int32(0)
            _wait(abar + Int32(B_ROW) + (n & Int32(1)), n >> Int32(1) & Int32(1))
            b = Int32(sRow[n & Int32(1)])
            while b >= Int32(0):
                gb = gidx_base + Int64(b) * Int64(NROWS * 4) + Int64(rtid * Int32(4))
                gv0 = L.ld_cg_i32(gb)
                gv1 = L.ld_cg_i32(gb + Int64(TILE_N * 4))
                gv2 = L.ld_cg_i32(gb + Int64(2 * TILE_N * 4))
                gv3 = L.ld_cg_i32(gb + Int64(3 * TILE_N * 4))
                gv4 = L.ld_cg_i32(gb + Int64(4 * TILE_N * 4))
                _softmax_row(
                    rtid,
                    b,
                    t,
                    gv0,
                    gv1,
                    gv2,
                    gv3,
                    gv4,
                    tmem_addr,
                    abar,
                    sMask,
                    sinks.iterator.raw_ptr() + b * Int32(HEADS),
                    out_addr,
                    scale_log2,
                )
                t = t + Int32(NTILES)
                n = n + Int32(1)
                _wait(abar + Int32(B_ROW) + (n & Int32(1)), n >> Int32(1) & Int32(1))
                b = Int32(sRow[n & Int32(1)])
        else:
            _mma_warp(abar, sRow, q_base, kv_base, tmem_addr)
        cute.arch.fence_view_async_tmem_load()
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.barrier(barrier_id=3, number_of_threads=self.threads)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)
        if tid == Int32(0):
            # Every CTA has made its final claim; the last one to leave resets the counters for the next call.
            # All rows were published before they could be claimed, so the indexer no longer touches the queue.
            if cutlass.const_expr(not self.exit_early):
                L.fence_acq_rel_gpu()
                d = L.atom_add_i32(aq_base + Int64(L.AQ_DONE * 4), Int32(1))
                if d == Int32(gdim) - Int32(1):
                    L.fence_acq_rel_gpu()
                    L.st_relaxed_i32(aq_base + Int64(L.AQ_PUB * 4), Int32(0))
                    L.st_relaxed_i32(aq_base + Int64(L.AQ_CLAIM * 4), Int32(0))
                    L.st_relaxed_i32(aq_base + Int64(L.AQ_DONE * 4), Int32(0))
            # Do not complete before the indexer grid: the next call's indexer reuses its workspaces.
            cute.arch.griddepcontrol_wait()


def compile_tail(tail: AttentionTail):
    rt = cute.runtime
    qa_fake = rt.make_fake_compact_tensor(
        cutlass.Uint8, (cute.sym_int64(), 64, 512), stride_order=(2, 1, 0), assumed_align=16
    )
    sp_fake = rt.make_fake_compact_tensor(
        cutlass.Uint8, (cute.sym_int64(), 512), stride_order=(1, 0), assumed_align=16
    )
    cp_fake = rt.make_fake_compact_tensor(
        cutlass.Uint8, (cute.sym_int64(), 512), stride_order=(1, 0), assumed_align=16
    )
    sb = cute.sym_int()
    gi_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb, NROWS), stride_order=(1, 0))
    sk_fake = rt.make_fake_compact_tensor(
        cutlass.Float32, (sb, 64), stride_order=(1, 0), assumed_align=16
    )
    ou_fake = rt.make_fake_compact_tensor(
        cutlass.BFloat16, (sb, 64, 512), stride_order=(2, 1, 0), assumed_align=16
    )
    aq_fake = rt.make_fake_compact_tensor(
        cutlass.Int32, (sb, 16), stride_order=(1, 0), assumed_align=16
    )
    stream = rt.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        tail,
        qa_fake,
        sp_fake,
        cp_fake,
        gi_fake,
        sk_fake,
        ou_fake,
        aq_fake,
        Float32(1.0),
        Int32(1),
        stream,
        options="--enable-tvm-ffi --generate-line-info",
    )


# ====================================================================================================================
# Persistent indexer kernel: FP8 indexer logits, exact top-512 selection and KV-row mapping.
# ====================================================================================================================

_RND_RN = "rn"
NEG_INF = -math.inf
K_TOP = SPEC.topk
RING = 256
RING_CHUNK = 64
MAIN_CAP = 2048
TRIG = 1280
PMAX_SMEM = 4
PMAX = 8
SEGW = 520
MARK_HI = 4294967295
SEL_BAR = 2
ROW_BAR = 4
END_BAR = 5
SEL_NTHR = 128
ATTQ = 8
DQ_CAP = 128
DEFER_LONG = 512
ST_SCHED = 1
ST_ROW = 2
ST_DQFULL = 4
COV_CAP = 32  # capacity of the list of the other pieces of a row (merge1)


@dsl_user_op
def _consume_f32(x, *, loc=None, ip=None):
    return Float32(
        _llvm.inline_asm(
            _T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=_llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _warp_atomic_i32(ptr, lane):
    # One atomic poll per warp; propagate the acquire edge to every reader.
    value = Int32(0)
    if lane == Int32(0):
        value = cute.arch.load(ptr, Int32, sem="acquire", scope="cta")
    value = cute.arch.shuffle_sync(value, 0)
    cute.arch.sync_warp()
    return value


@cute.jit
def _warp_atomic_u32(ptr, lane):
    # The packed tag and conservative floor are the complete atomic payload.
    value = Uint32(0)
    if lane == Int32(0):
        value = cute.arch.load(ptr, Uint32, sem="relaxed", scope="cta")
    return cute.arch.shuffle_sync(value, 0)


@cute.jit
def _pack_floor_key(key, seg):
    # At most 1024 segments per CTA in scope, so the epoch fits in ten bits.
    # Rounding an ordered key down can only admit additional candidates.
    return (Uint32(key) & Uint32(0xFFFFFC00)) | Uint32(seg)


@cute.jit
def _wait_chunk_empty(empty_ptr, stage, phase, stall, prof: cutlass.Constexpr = False):
    ready = prims.mbarrier_wait_parity(
        empty_ptr + stage, phase, prims.MBarrierWait.TEST, scope="cta", order="acquire"
    )
    ready = cute.arch.vote_all_sync(ready)
    if not ready:
        t_s = Int64(0)
        if cutlass.const_expr(prof):
            t_s = cutlass.Int64(cute.arch.globaltimer())
        ready = cutlass.Boolean(False)
        while not ready:
            ready = prims.mbarrier_try_wait_parity(
                empty_ptr + stage, phase, scope="cta", order="acquire"
            )
            ready = cute.arch.vote_all_sync(ready)
        if cutlass.const_expr(prof):
            stall = stall + (cutlass.Int64(cute.arch.globaltimer()) - t_s)
    return stall


@cute.jit
def _push_ring(
    hit,
    entry,
    head,
    s_ring,
    ring_off,
    empty_ptr,
    full_ptr,
    lane,
    lt_mask,
    stall,
    prof: cutlass.Constexpr = False,
):
    mask = Int32(cute.arch.vote_ballot_sync(hit))
    if mask != Int32(0):
        n = Int32(cute.arch.popc(mask))
        offset = head & Int32(RING_CHUNK - 1)
        stage = (head >> Int32(6)) & Int32(RING // RING_CHUNK - 1)
        if offset == Int32(0):
            phase = ((head >> Int32(8)) & Int32(1)) ^ Int32(1)
            stall = _wait_chunk_empty(empty_ptr, stage, phase, stall, prof)
        first = Int32(RING_CHUNK) - offset
        if first > n:
            first = n
        rank = Int32(cute.arch.popc(mask & lt_mask))
        if hit & (rank < first):
            slot = (head + rank) & Int32(RING - 1)
            s_ring[ring_off + slot] = entry
        head = head + first
        if offset + first == Int32(RING_CHUNK):
            # All 32 writers publish their own stores, including earlier pushes.
            prims.mbarrier_arrive(full_ptr + stage, scope="cta", relaxed=False)
        if first < n:
            # Publish the completed chunk before waiting on the spill chunk.
            stage = (head >> Int32(6)) & Int32(RING // RING_CHUNK - 1)
            phase = ((head >> Int32(8)) & Int32(1)) ^ Int32(1)
            stall = _wait_chunk_empty(empty_ptr, stage, phase, stall, prof)
            if hit & (rank >= first):
                slot = (head + rank - first) & Int32(RING - 1)
                s_ring[ring_off + slot] = entry
            head = head + n - first
    return (head, stall)


@cute.jit
def _push_marker(seg_ord, head, s_ring, ring_off, empty_ptr, full_ptr, lane):
    offset = head & Int32(RING_CHUNK - 1)
    stage = (head >> Int32(6)) & Int32(RING // RING_CHUNK - 1)
    if offset == Int32(0):
        phase = ((head >> Int32(8)) & Int32(1)) ^ Int32(1)
        _wait_chunk_empty(empty_ptr, stage, phase, Int64(0))
    chunk_base = head - offset
    # The consumer reads all 64 slots; initialize the complete marker suffix.
    if lane >= offset:
        s_ring[ring_off + ((chunk_base + lane) & Int32(RING - 1))] = Uint64(MARK_HI) << Uint64(
            32
        ) | Uint64(Uint32(seg_ord))
    if lane + Int32(32) >= offset:
        s_ring[ring_off + ((chunk_base + lane + Int32(32)) & Int32(RING - 1))] = Uint64(
            MARK_HI
        ) << Uint64(32) | Uint64(Uint32(seg_ord))
    prims.mbarrier_arrive(full_ptr + stage, scope="cta", relaxed=False)
    return chunk_base + Int32(RING_CHUNK)


@cute.jit
def _floor_of(tw_ptr, seg_ord, lane):
    tw = _warp_atomic_u32(tw_ptr, lane)
    t = Float32(NEG_INF)
    if (tw & Uint32(1023)) == Uint32(seg_ord):
        key = tw & Uint32(0xFFFFFC00)
        # Zero initialization and rounded -inf lie in the negative-NaN range.
        if key >= Uint32(0x007FFFFF):
            t = L.invkey(key)
    return t


@cute.jit
def _drain_ring(
    r,
    tail_r,
    done_r,
    kth_key,
    s_ring,
    full_ptr,
    empty_ptr,
    sMain,
    cnt_ptr,
    lane,
    lt_mask,
    kmin_ptr,
    kmax_ptr,
):
    used = Int32(0)
    prog = Int32(0)
    if done_r == Int32(0):
        stage = (tail_r >> Int32(6)) & Int32(RING // RING_CHUNK - 1)
        phase = (tail_r >> Int32(8)) & Int32(1)
        # A selector warp services two rings, so probe rather than park on one.
        ready = prims.mbarrier_wait_parity(
            full_ptr + stage, phase, prims.MBarrierWait.TEST, scope="cta", order="acquire"
        )
        ready = cute.arch.vote_all_sync(ready)
        if ready:
            e0 = Uint64(s_ring[r * Int32(RING) + ((tail_r + lane) & Int32(RING - 1))])
            e1 = Uint64(s_ring[r * Int32(RING) + ((tail_r + lane + Int32(32)) & Int32(RING - 1))])
            ism0 = Uint32(e0 >> Uint64(32)) == Uint32(MARK_HI)
            ism1 = Uint32(e1 >> Uint64(32)) == Uint32(MARK_HI)
            mm0 = Int32(cute.arch.vote_ballot_sync(ism0))
            mm1 = Int32(cute.arch.vote_ballot_sync(ism1))
            # Markers occupy a suffix and every slot in that suffix is initialized.
            nuse = Int32(RING_CHUNK) - Int32(cute.arch.popc(mm0)) - Int32(cute.arch.popc(mm1))
            if (mm0 | mm1) != Int32(0):
                done_r = Int32(1)
            fk0 = L.fkey_bits(L.entry_bits(e0))
            fk1 = L.fkey_bits(L.entry_bits(e1))
            # _select_role always passes a score-only floor (low 32 bits zero).
            floor = Uint32(kth_key >> Uint64(32))
            keep0 = (not ism0) & (fk0 >= floor)
            keep1 = (not ism1) & (fk1 >= floor)
            km0 = Int32(cute.arch.vote_ballot_sync(keep0))
            km1 = Int32(cute.arch.vote_ballot_sync(keep1))
            nk0 = Int32(cute.arch.popc(km0))
            nk = nk0 + Int32(cute.arch.popc(km1))
            fmn = Uint32(4294967295)
            fmx = Uint32(0)
            if keep0:
                fmn = fk0
                fmx = fk0
            if keep1:
                if fk1 < fmn:
                    fmn = fk1
                if fk1 > fmx:
                    fmx = fk1
            fmn = L.warp_min_u32(fmn)
            fmx = L.warp_max_u32(fmx)
            base = Int32(0)
            if lane == Int32(0):
                if nk > Int32(0):
                    base = Int32(cute.arch.atomic_add(cnt_ptr, nk, sem="relaxed", scope="cta"))
                    cute.arch.atomic_min(kmin_ptr, fmn, sem="relaxed", scope="cta")
                    cute.arch.atomic_max(kmax_ptr, fmx, sem="relaxed", scope="cta")
            base = cute.arch.shuffle_sync(base, 0)
            if keep0:
                sMain[base + Int32(cute.arch.popc(km0 & lt_mask))] = e0
            if keep1:
                sMain[base + nk0 + Int32(cute.arch.popc(km1 & lt_mask))] = e1
            used = nuse
            tail_r = tail_r + Int32(RING_CHUNK)
            # Every reader releases its own load before this chunk can be reused.
            prims.mbarrier_arrive(empty_ptr + stage, scope="cta", relaxed=False)
            prog = Int32(1)
    return (tail_r, done_r, used, prog)


EPT = MAIN_CAP // SEL_NTHR
KEEP_CAP = 1280


@cute.jit
def _compact_approx(
    sMain,
    C,
    sHist,
    sRes,
    cnt_ptr,
    kmin_ptr,
    kmax_ptr,
    kmin2_ptr,
    sScal,
    tid,
    nthr: cutlass.Constexpr,
    bar_id: cutlass.Constexpr,
    prof=None,
    keep_cap: cutlass.Constexpr = KEEP_CAP,
    nb: cutlass.Constexpr = 1024,
    tw_ptr=None,
    seg=None,
    tiebin: cutlass.Constexpr = False,
    sTb=None,
    fast_cmp: cutlass.Constexpr = False,
):
    lane = tid & Int32(31)
    tp0 = Int64(0)
    if cutlass.const_expr(prof is not None):
        tp0 = cutlass.Int64(cute.arch.globaltimer())
    lo = L.invkey(Uint32(cute.arch.load(kmin_ptr, Uint32, sem="relaxed", scope="cta")))
    hi = L.invkey(Uint32(cute.arch.load(kmax_ptr, Uint32, sem="relaxed", scope="cta")))
    wv = (hi - lo) * Float32(1.0 / nb)
    sc = Float32(0.0)
    if wv > Float32(0.0):
        if wv < Float32(3e38):
            sc = cute.arch.rcp_approx(wv)
    for z_ in cutlass.range_constexpr(max(1, nb // nthr)):
        if tid + Int32(z_ * nthr) < Int32(nb):
            sHist[tid + Int32(z_ * nthr)] = Int32(0)
    if tid == Int32(0):
        sScal[24] = Int32(0)
        cute.arch.store(kmin2_ptr, Uint32(4294967295), sem="relaxed", scope="cta")
    L.bar(bar_id, nthr)
    if cutlass.const_expr(prof is not None):
        if tid == Int32(0):
            tp1 = cutlass.Int64(cute.arch.globaltimer())
            prof[0] = prof[0] + (tp1 - tp0)
            tp0 = tp1
    ev = [Uint64(0)] * EPT
    i = tid
    if cutlass.const_expr(fast_cmp):
        # all loads first, then the bins, then the histogram atomics
        for k in cutlass.range_constexpr(EPT):
            ev[k] = Uint64(sMain[cutlass.min(tid + Int32(k * nthr), C - Int32(1))])
        hb = []
        for k in cutlass.range_constexpr(EPT):
            hb.append(L._vbin(ev[k], lo, sc, nb))
        for k in cutlass.range_constexpr(EPT):
            if tid + Int32(k * nthr) < C:
                cute.arch.atomic_add(sHist.iterator + hb[k], Int32(1), sem="relaxed", scope="cta")
    else:
        for k in cutlass.range_constexpr(EPT):
            i = tid + Int32(k * nthr)
            if i < C:
                ev[k] = Uint64(sMain[i])
                cute.arch.atomic_add(
                    sHist.iterator + L._vbin(ev[k], lo, sc, nb),
                    Int32(1),
                    sem="relaxed",
                    scope="cta",
                )
    L.bar(bar_id, nthr)
    if cutlass.const_expr(prof is not None):
        if tid == Int32(0):
            tp1 = cutlass.Int64(cute.arch.globaltimer())
            prof[1] = prof[1] + (tp1 - tp0)
            tp0 = tp1
    L.find_cross(sHist, Int32(K_TOP), tid, sRes, nb)
    L.bar(bar_id, nthr)
    if cutlass.const_expr(prof is not None):
        if tid == Int32(0):
            tp1 = cutlass.Int64(cute.arch.globaltimer())
            prof[2] = prof[2] + (tp1 - tp0)
            tp0 = tp1
    B = sRes[L.RES_B]
    kept = sRes[L.RES_ABOVE] + sRes[L.RES_M]
    mode2 = Int32(0)
    idx_thr = Int32(2147483647)
    if cutlass.const_expr(tiebin):
        if kept > Int32(keep_cap):
            above = sRes[L.RES_ABOVE]
            if tid == Int32(0):
                sTb[0] = Uint32(4294967295)
                sTb[1] = Uint32(0)
                sTb[2] = Uint32(4294967295)
                sTb[3] = Uint32(0)
            for z_ in cutlass.range_constexpr(max(1, nb // nthr)):
                if tid + Int32(z_ * nthr) < Int32(nb):
                    sHist[tid + Int32(z_ * nthr)] = Int32(0)
            L.bar(bar_id, nthr)
            fmn_b = Uint32(4294967295)
            fmx_b = Uint32(0)
            imn_b = Uint32(4294967295)
            imx_b = Uint32(0)
            for k in cutlass.range(EPT, unroll=1):
                i = tid + k * Int32(nthr)
                if i < C:
                    e = Uint64(sMain[i])
                    if L._vbin(e, lo, sc, nb) == B:
                        fk = L.fkey_bits(L.entry_bits(e))
                        ix = Uint32(e >> Uint64(32))
                        if fk < fmn_b:
                            fmn_b = fk
                        if fk > fmx_b:
                            fmx_b = fk
                        if ix < imn_b:
                            imn_b = ix
                        if ix > imx_b:
                            imx_b = ix
            fmn_b = L.warp_min_u32(fmn_b)
            fmx_b = L.warp_max_u32(fmx_b)
            imn_b = L.warp_min_u32(imn_b)
            imx_b = L.warp_max_u32(imx_b)
            if lane == Int32(0):
                cute.arch.atomic_min(sTb.iterator, fmn_b, sem="relaxed", scope="cta")
                cute.arch.atomic_max(sTb.iterator + 1, fmx_b, sem="relaxed", scope="cta")
                cute.arch.atomic_min(sTb.iterator + 2, imn_b, sem="relaxed", scope="cta")
                cute.arch.atomic_max(sTb.iterator + 3, imx_b, sem="relaxed", scope="cta")
            L.bar(bar_id, nthr)
            fmn_b = Uint32(sTb[0])
            fmx_b = Uint32(sTb[1])
            imn_b = Uint32(sTb[2])
            imx_b = Uint32(sTb[3])
            if fmn_b == fmx_b:
                mode2 = Int32(1)
                span = imx_b - imn_b
                sh = Uint32(0)
                while span >> sh >= Uint32(nb):
                    sh = sh + Uint32(1)
                for k in cutlass.range(EPT, unroll=1):
                    i = tid + k * Int32(nthr)
                    if i < C:
                        e = Uint64(sMain[i])
                        if L._vbin(e, lo, sc, nb) == B:
                            b2 = Int32(imx_b - Uint32(e >> Uint64(32)) >> sh)
                            cute.arch.atomic_add(
                                sHist.iterator + b2, Int32(1), sem="relaxed", scope="cta"
                            )
                L.bar(bar_id, nthr)
                L.find_cross(sHist, Int32(K_TOP) - above, tid, sRes, nb)
                L.bar(bar_id, nthr)
                B2 = sRes[L.RES_B]
                kept = above + sRes[L.RES_ABOVE] + sRes[L.RES_M]
                idx_thr = Int32(imx_b - (Uint32(B2) << sh))
    if cutlass.const_expr(tw_ptr is not None):
        if tid < Int32(8):
            if (kept <= Int32(keep_cap)) & (mode2 == Int32(0)):
                if B >= Int32(2):
                    te = lo + Float32(B - Int32(2)) * wv
                    mag = cute.math.max(abs(lo), abs(hi))
                    if wv > mag * Float32(1e-05):
                        if te > lo:
                            cute.arch.atomic_exch(
                                tw_ptr + tid,
                                _pack_floor_key(L.fkey(te), seg),
                                sem="relaxed",
                                scope="cta",
                            )
    tf = Uint32(0)
    if kept <= Int32(keep_cap):
        lm = Int32(cute.arch.lanemask_lt())
        fmn = Uint32(4294967295)
        kmask = [Int32(0)] * EPT
        wtot = Int32(0)
        if cutlass.const_expr(fast_cmp):
            kpl = []
            ceq = []
            for k in cutlass.range_constexpr(EPT):
                bbk = L._vbin(ev[k], lo, sc, nb)
                kpk = (tid + Int32(k * nthr) < C) & (bbk >= B)
                if cutlass.const_expr(tiebin):
                    kpk = kpk & ((bbk != B) | (Int32(Uint32(ev[k] >> Uint64(32))) <= idx_thr))
                kpl.append(kpk)
                ceq.append(kpk & (bbk == B))
            for k in cutlass.range_constexpr(EPT):
                fkk = L.fkey_bits(L.entry_bits(ev[k]))
                fcand = Uint32(cutlass.select_(ceq[k], fkk, Uint32(4294967295)))
                fmn = Uint32(cutlass.select_(fcand < fmn, fcand, fmn))
                kmask[k] = Int32(cute.arch.vote_ballot_sync(kpl[k]))
                wtot = wtot + Int32(cute.arch.popc(kmask[k]))
        else:
            for k in cutlass.range_constexpr(EPT):
                i = tid + Int32(k * nthr)
                kp = False
                bb = Int32(-1)
                if i < C:
                    bb = L._vbin(ev[k], lo, sc, nb)
                    if bb >= B:
                        kp = True
                    if cutlass.const_expr(tiebin):
                        if bb == B:
                            if Int32(Uint32(ev[k] >> Uint64(32))) > idx_thr:
                                kp = False
                if kp & (bb == B):
                    fk = L.fkey_bits(L.entry_bits(ev[k]))
                    if fk < fmn:
                        fmn = fk
                kmask[k] = Int32(cute.arch.vote_ballot_sync(kp))
                wtot = wtot + Int32(cute.arch.popc(kmask[k]))
        base = Int32(0)
        if lane == Int32(0):
            if wtot > Int32(0):
                base = Int32(
                    cute.arch.atomic_add(
                        sScal.iterator + Int32(24), wtot, sem="relaxed", scope="cta"
                    )
                )
        base = cute.arch.shuffle_sync(base, 0)
        for k in cutlass.range_constexpr(EPT):
            if kmask[k] >> lane & Int32(1) != Int32(0):
                sMain[base + Int32(cute.arch.popc(kmask[k] & lm))] = ev[k]
            base = base + Int32(cute.arch.popc(kmask[k]))
        fmn = L.warp_min_u32(fmn)
        if lane == Int32(0):
            cute.arch.atomic_min(kmin2_ptr, fmn, sem="relaxed", scope="cta")
        L.bar(bar_id, nthr)
        tf = Uint32(cute.arch.load(kmin2_ptr, Uint32, sem="relaxed", scope="cta"))
        if tid == Int32(0):
            cute.arch.store(cnt_ptr, kept, sem="relaxed", scope="cta")
            cute.arch.store(kmin_ptr, tf, sem="relaxed", scope="cta")
        L.bar(bar_id, nthr)
        if cutlass.const_expr(prof is not None):
            if tid == Int32(0):
                tp1 = cutlass.Int64(cute.arch.globaltimer())
                prof[3] = prof[3] + (tp1 - tp0)
    else:
        kept = Int32(-1)
    return (kept, tf)


@cute.jit
def _parts_of_row(q, cta, sched, num_sms, lane):
    P = Int32(0)
    p = Int32(0)
    nit = (num_sms + Int32(31)) // Int32(32)
    it = Int32(0)
    while it < nit:
        c = lane + it * Int32(32)
        v = False
        if c < num_sms:
            sq0 = sched[c, 0]
            skv0 = sched[c, 1]
            sq1 = sched[c + 1, 0]
            skv1 = sched[c + 1, 1]
            hi = sq1
            if skv1 == Int32(0):
                hi = sq1 - Int32(1)
            if (sq0 != sq1) | (skv0 != skv1):
                if sq0 <= q:
                    if q <= hi:
                        v = True
        bv = Int32(cute.arch.vote_ballot_sync(v))
        bl = Int32(cute.arch.vote_ballot_sync(v & (c < cta)))
        P = P + Int32(cute.arch.popc(bv))
        p = p + Int32(cute.arch.popc(bl))
        it = it + Int32(1)
    return (P, p)


@cute.jit
def _parts_cov(q, cta, sched, num_sms, lane, s_cov):
    # _parts_of_row that also lists the other pieces of the row: s_cov[j] = slot (cta * 2 + side)
    # of the j-th other covering CTA in CTA order, side 0 when the row is the first row of that
    # CTA's range, else 1 (the slot its piece of the row was written to).
    P = Int32(0)
    p = Int32(0)
    ncov = Int32(0)
    lm = Int32(cute.arch.lanemask_lt())
    nit = (num_sms + Int32(31)) // Int32(32)
    it = Int32(0)
    while it < nit:
        c = lane + it * Int32(32)
        v = False
        side = Int32(1)
        if c < num_sms:
            sq0 = sched[c, 0]
            skv0 = sched[c, 1]
            sq1 = sched[c + 1, 0]
            skv1 = sched[c + 1, 1]
            hi = sq1
            if skv1 == Int32(0):
                hi = sq1 - Int32(1)
            if (sq0 != sq1) | (skv0 != skv1):
                if sq0 <= q:
                    if q <= hi:
                        v = True
                        if sq0 == q:
                            side = Int32(0)
        bv = Int32(cute.arch.vote_ballot_sync(v))
        bl = Int32(cute.arch.vote_ballot_sync(v & (c < cta)))
        vx = v & (c != cta)
        bx = Int32(cute.arch.vote_ballot_sync(vx))
        if vx:
            pos = ncov + Int32(cute.arch.popc(bx & lm))
            if pos < Int32(COV_CAP):
                s_cov[pos] = c * Int32(2) + side
        ncov = ncov + Int32(cute.arch.popc(bx))
        P = P + Int32(cute.arch.popc(bv))
        p = p + Int32(cute.arch.popc(bl))
        it = it + Int32(1)
    return (P, p)


@cute.jit
def _cover_ballot(q, cta, sched, num_sms, lane, it):
    c = lane + it * Int32(32)
    v = False
    if c < num_sms:
        if c != cta:
            sq0 = sched[c, 0]
            skv0 = sched[c, 1]
            sq1 = sched[c + 1, 0]
            skv1 = sched[c + 1, 1]
            hi = sq1
            if skv1 == Int32(0):
                hi = sq1 - Int32(1)
            if (sq0 != sq1) | (skv0 != skv1):
                if sq0 <= q:
                    if q <= hi:
                        v = True
    return Int32(cute.arch.vote_ballot_sync(v))


@cute.jit
def _write_row_outputs_regs(
    row, n, s_out, nout, short, topk_out, gidx, bt_swa, bt_c, pos, req, mbs, mbc, tid
):
    gv = []
    for q in cutlass.range_constexpr(5):
        col = tid + Int32(q * 128)
        g = Int32(-1)
        if cutlass.const_expr(q == 0):
            g = L.l2g_swa(col, pos, req, bt_swa, mbs, Int32(0))
        else:
            i = col - Int32(L.SWA)
            idx = Int32(-1)
            if short != Int32(0):
                if i < n:
                    idx = i
            elif i < nout:
                idx = Int32(Uint32(4294967295) - Uint32(Uint64(s_out[i]) & Uint64(4294967295)))
            topk_out[row, i] = idx
            g = L.l2g_compressed(idx, req, bt_c, mbc, Int32(0))
        gv.append(g)
    for q in cutlass.range_constexpr(5):
        gidx[row, tid + Int32(q * 128)] = gv[q]
    return (gv[0], gv[1], gv[2], gv[3], gv[4])


@cute.jit
def _final512_token32(sHand, nout, tid, bar_id: cutlass.Constexpr):
    # topk is a set and attention does not depend on row order: convert the completed row's entries to the
    # output writer's inverted-token low word in place instead of sorting them.
    base = tid * Int32(4)
    for i in cutlass.range_constexpr(4):
        if base + Int32(i) < nout:
            sHand[base + Int32(i)] = Uint64(
                Uint32(4294967295) - Uint32(Uint64(sHand[base + Int32(i)]) >> Uint64(32))
            )
    L.bar(bar_id, 128)


@cute.jit
def _pick_final32(a, b, first):
    r = cutlass.min(a, b)
    if not first:
        r = cutlass.max(a, b)
    return r


@cute.jit
def _sort_final512_token32(sHand, nout, sScratch, tid, bar_id: cutlass.Constexpr):
    # Sort only the completed row. Four keys per thread keep the small
    # exchanges in registers; three large exchanges use dead merge scratch.
    base = tid * Int32(4)
    e0 = Uint32(4294967295)
    e1 = Uint32(4294967295)
    e2 = Uint32(4294967295)
    e3 = Uint32(4294967295)
    if base < nout:
        e0 = Uint32(Uint64(sHand[base]) >> Uint64(32))
    if base + Int32(1) < nout:
        e1 = Uint32(Uint64(sHand[base + Int32(1)]) >> Uint64(32))
    if base + Int32(2) < nout:
        e2 = Uint32(Uint64(sHand[base + Int32(2)]) >> Uint64(32))
    if base + Int32(3) < nout:
        e3 = Uint32(Uint64(sHand[base + Int32(3)]) >> Uint64(32))
    sx = cute.make_tensor(cute.recast_ptr(sScratch.iterator, dtype=Uint32), cute.make_layout(512))
    kk = Int32(1)
    while kk < Int32(10):
        jj = kk - Int32(1)
        while jj >= Int32(0):
            if jj >= Int32(7):
                # Transposed exchange gives contiguous words to each warp.
                # Scratch is separate from sHand: initial raw loads cannot
                # race with the first exchange's stores from another warp.
                sx[tid] = e0
                sx[tid + Int32(128)] = e1
                sx[tid + Int32(256)] = e2
                sx[tid + Int32(384)] = e3
                L.bar(bar_id, 128)
                peer = tid ^ (Int32(1) << (jj - Int32(2)))
                p0 = Uint32(sx[peer])
                p1 = Uint32(sx[peer + Int32(128)])
                p2 = Uint32(sx[peer + Int32(256)])
                p3 = Uint32(sx[peer + Int32(384)])
                L.bar(bar_id, 128)
                first = (((base >> jj) ^ (base >> kk)) & Int32(1)) == Int32(0)
                e0 = _pick_final32(e0, p0, first)
                e1 = _pick_final32(e1, p1, first)
                e2 = _pick_final32(e2, p2, first)
                e3 = _pick_final32(e3, p3, first)
            elif jj >= Int32(2):
                delta = Int32(1) << (jj - Int32(2))
                p0 = cute.arch.shuffle_sync_bfly(e0, delta)
                p1 = cute.arch.shuffle_sync_bfly(e1, delta)
                p2 = cute.arch.shuffle_sync_bfly(e2, delta)
                p3 = cute.arch.shuffle_sync_bfly(e3, delta)
                first = (((base >> jj) ^ (base >> kk)) & Int32(1)) == Int32(0)
                e0 = _pick_final32(e0, p0, first)
                e1 = _pick_final32(e1, p1, first)
                e2 = _pick_final32(e2, p2, first)
                e3 = _pick_final32(e3, p3, first)
            elif jj == Int32(1):
                first = ((base >> kk) & Int32(1)) == Int32(0)
                a0 = e0
                b0 = e2
                a1 = e1
                b1 = e3
                e0 = _pick_final32(a0, b0, first)
                e2 = _pick_final32(a0, b0, not first)
                e1 = _pick_final32(a1, b1, first)
                e3 = _pick_final32(a1, b1, not first)
            else:
                first0 = ((base >> kk) & Int32(1)) == Int32(0)
                first2 = (((base + Int32(2)) >> kk) & Int32(1)) == Int32(0)
                a0 = e0
                b0 = e1
                a2 = e2
                b2 = e3
                e0 = _pick_final32(a0, b0, first0)
                e1 = _pick_final32(a0, b0, not first0)
                e2 = _pick_final32(a2, b2, first2)
                e3 = _pick_final32(a2, b2, not first2)
            jj = jj - Int32(1)
        kk = kk + Int32(1)
    # The existing output writer consumes only the inverted token index in
    # the low word. No score data is needed after the completed-row sort.
    sHand[base] = Uint64(Uint32(4294967295) - e0)
    sHand[base + Int32(1)] = Uint64(Uint32(4294967295) - e1)
    sHand[base + Int32(2)] = Uint64(Uint32(4294967295) - e2)
    sHand[base + Int32(3)] = Uint64(Uint32(4294967295) - e3)
    L.bar(bar_id, 128)


@cute.jit
def _finalize_seg(
    tid,
    bar_id: cutlass.Constexpr,
    cta,
    start_q,
    q,
    n,
    P,
    p,
    sMb,
    cbuf,
    rel_ptr,
    sHand,
    sAttB,
    sAttO,
    sHistS,
    sCkS,
    sResS,
    sRedS,
    sWcS,
    s_flag,
    mKvLens,
    mReq,
    mBtSwa,
    mBtC,
    mTopk,
    mGidx,
    mRowStat,
    mSched,
    ws_seg_base,
    ws_ctr_base,
    mbs,
    mbc,
    num_sms: cutlass.Constexpr,
    skew_ns: cutlass.Constexpr,
    skew_seed,
    salt,
    stamps: cutlass.Constexpr,
    mStamps,
    sbase,
    dly_t=None,
    dly_us: cutlass.Constexpr = 0,
    rel_keep=None,
    rsort: cutlass.Constexpr = False,
    rel_bar=None,
    merge1: cutlass.Constexpr = 0,
    sBig=None,
    sCov=None,
    big_cap: cutlass.Constexpr = 0,
    sel_fast: cutlass.Constexpr = False,
):
    NT: cutlass.Constexpr[int] = SEL_NTHR
    if cutlass.const_expr(stamps):
        if tid == Int32(0):
            mStamps[sbase] = cutlass.Int64(cute.arch.globaltimer())
    fast = Int32(0)
    if (n <= Int32(K_TOP)) & (P == Int32(1)):
        fast = Int32(1)
    nout = Int32(0)
    if fast == Int32(0):
        nout, kth_, m_, inf_ = L.select_topk_list(
            sMb,
            cbuf,
            Int32(K_TOP),
            sHand,
            sHistS,
            sCkS,
            sResS,
            sRedS,
            sWcS,
            tid,
            False,
            NT,
            bar_id,
            fast_loops=sel_fast,
        )
    # Keep this parity slot until all raw segment unions have been selected.
    # The 2048-entry source buffer is also our bounded <=1024-entry merge list.
    if cutlass.const_expr(dly_us > 0):
        if dly_t >= Int32(0):
            if tid == Int32(0):
                t_n = Int32(Uint32(cutlass.Int64(cute.arch.globaltimer()) >> Int64(10)))
                while t_n - dly_t < Int32(dly_us):
                    L.nanosleep(Int32(1000))
                    t_n = Int32(Uint32(cutlass.Int64(cute.arch.globaltimer()) >> Int64(10)))
    L.bar(bar_id, NT)
    if cutlass.const_expr(stamps):
        if tid == Int32(0):
            mStamps[sbase + Int32(1)] = cutlass.Int64(cute.arch.globaltimer())
    if cutlass.const_expr(stamps):
        if tid == Int32(0):
            mStamps[sbase + Int32(2)] = cutlass.Int64(cute.arch.globaltimer())
    pos = Int32(mKvLens[q]) - Int32(1)
    req = Int32(mReq[q])
    short = Int32(0)
    if n <= Int32(K_TOP):
        short = Int32(1)
    done_row = Int32(0)
    gv0 = Int32(-1)
    gv1 = Int32(-1)
    gv2 = Int32(-1)
    gv3 = Int32(-1)
    gv4 = Int32(-1)
    if P == Int32(1):
        if short == Int32(0):
            _final512_token32(sHand, nout, tid, bar_id)
        gv0, gv1, gv2, gv3, gv4 = _write_row_outputs_regs(
            q, n, sHand, nout, short, mTopk, mGidx, mBtSwa, mBtC, pos, req, mbs, mbc, tid
        )
        done_row = Int32(1)
    else:
        pside = Int64(1)
        if q == start_q:
            pside = Int64(0)
        so = ws_seg_base + (Int64(cta) * Int64(2) + pside) * Int64(SEGW * 8)
        out_i = tid
        while out_i < nout:
            L.st_g_u64(so + Int64(out_i) * Int64(8), Uint64(sHand[out_i]))
            out_i = out_i + Int32(NT)
        if tid == Int32(0):
            L.st_g_u32(so + Int64(512 * 8), nout)
        L.bar(bar_id, NT)
        if tid == Int32(0):
            if cutlass.const_expr(skew_ns > 0):
                L.skew_sleep(skew_seed, salt, Int32(skew_ns))
            ca = ws_ctr_base + Int64(q) * Int64(128)
            old = L.gc_arrive(ca)
            lst = Int32(0)
            if Int32(Uint32(old & Uint64(65535))) == P - Int32(1):
                lst = Int32(1)
                L.gc_pad(ca, P)
                L.fence_acq_rel_gpu()
            s_flag[0] = lst
        L.bar(bar_id, NT)
        if cutlass.const_expr(stamps):
            if tid == Int32(0):
                mStamps[sbase + Int32(3)] = cutlass.Int64(cute.arch.globaltimer())
        if s_flag[0] != Int32(0):
            na = nout
            rlane = tid & Int32(31)
            nit = (Int32(num_sms) + Int32(31)) // Int32(32)
            if cutlass.const_expr(merge1 > 0):
                # One-shot merge (last segment of the CTA, K stages idle): the lists of the other
                # pieces, merge1 sources in flight at a time, go densely into sBig and one selection
                # runs over the union (its top-512 set is unique: entries are totally ordered by key
                # and position). More pieces than sBig holds: the serial merge below.
                ncov = P - Int32(1)
                if P * Int32(K_TOP) <= Int32(big_cap):
                    i0 = tid
                    while i0 < na:
                        sBig[i0] = Uint64(sHand[i0])
                        i0 = i0 + Int32(NT)
                    off = na
                    j0 = Int32(0)
                    while j0 < ncov:
                        es = []
                        nbs = []
                        for g in cutlass.range_constexpr(merge1):
                            jj = cutlass.min(j0 + Int32(g), ncov - Int32(1))
                            slot = Int32(sCov[jj])
                            src = ws_seg_base + Int64(slot) * Int64(SEGW * 8)
                            nbs.append(L.ld_cg_i32(src + Int64(512 * 8)))
                            for k in cutlass.range_constexpr(K_TOP // NT):
                                es.append(L.ld_cg_u64(src + Int64(tid + k * NT) * Int64(8)))
                        for g in cutlass.range_constexpr(merge1):
                            nb = cutlass.select_(j0 + Int32(g) < ncov, nbs[g], Int32(0))
                            for k in cutlass.range_constexpr(K_TOP // NT):
                                if tid + Int32(k * NT) < nb:
                                    sBig[off + tid + Int32(k * NT)] = es[g * (K_TOP // NT) + k]
                            off = off + nb
                        j0 = j0 + Int32(merge1)
                    L.bar(bar_id, NT)
                    nm1, kth_m1, m_m1, inf_m1 = L.select_topk_list(
                        sBig,
                        off,
                        Int32(K_TOP),
                        sHand,
                        sHistS,
                        sCkS,
                        sResS,
                        sRedS,
                        sWcS,
                        tid,
                        False,
                        NT,
                        bar_id,
                        fast_loops=sel_fast,
                    )
                    na = nm1
                    nit = Int32(0)
            it = Int32(0)
            while it < nit:
                bv = _cover_ballot(q, cta, mSched, Int32(num_sms), rlane, it)
                while bv != Int32(0):
                    low = bv & Int32(0) - bv
                    cs = it * Int32(32) + Int32(cute.arch.popc(low - Int32(1)))
                    sside = Int64(1)
                    if mSched[cs, 0] == q:
                        sside = Int64(0)
                    src = ws_seg_base + (Int64(cs) * Int64(2) + sside) * Int64(SEGW * 8)
                    nb = L.ld_cg_i32(src + Int64(512 * 8))
                    ia = tid
                    while ia < na:
                        sMb[ia] = Uint64(sHand[ia])
                        ia = ia + Int32(NT)
                    ib = tid
                    while ib < nb:
                        sMb[na + ib] = L.ld_cg_u64(src + Int64(ib) * Int64(8))
                        ib = ib + Int32(NT)
                    L.bar(bar_id, NT)
                    nm, kth_m, m_m, inf_m = L.select_topk_list(
                        sMb,
                        na + nb,
                        Int32(K_TOP),
                        sHand,
                        sHistS,
                        sCkS,
                        sResS,
                        sRedS,
                        sWcS,
                        tid,
                        False,
                        NT,
                        bar_id,
                        fast_loops=sel_fast,
                    )
                    na = nm
                    bv = bv ^ low
                it = it + Int32(1)
            if cutlass.const_expr(stamps):
                if tid == Int32(0):
                    mStamps[sbase + Int32(4)] = cutlass.Int64(cute.arch.globaltimer())
            if short == Int32(0):
                _final512_token32(sHand, na, tid, bar_id)
            gv0, gv1, gv2, gv3, gv4 = _write_row_outputs_regs(
                q, n, sHand, na, short, mTopk, mGidx, mBtSwa, mBtC, pos, req, mbs, mbc, tid
            )
            done_row = Int32(1)
    L.bar(bar_id, NT)
    if done_row != Int32(0):
        if tid == Int32(0):
            L.publish_row(mRowStat.iterator.toint(), q)
    if cutlass.const_expr(rel_bar is not None):
        rk = Int32(0)
        if cutlass.const_expr(rel_keep is not None):
            rk = rel_keep
        if rk == Int32(0):
            # Release the parity slot only after every union selection read.
            prims.mbarrier_arrive(rel_bar)
    return (done_row, gv0, gv1, gv2, gv3, gv4)


def _shared_storage(kv_stages: int, q_stages: int, umma_stages: int, fp4: bool):
    """The shared-storage struct of the indexer kernel; the FP4 variant adds the SFB write-after-read mbarrier."""
    # real annotation objects: this module postpones annotation evaluation, which cute.struct cannot resolve for
    # local names
    fields = {
        "kv_mbar_0": cute.struct.MemRange[cutlass.Int64, kv_stages * 2],
        "kv_mbar_1": cute.struct.MemRange[cutlass.Int64, kv_stages * 2],
        "q_mbar": cute.struct.MemRange[cutlass.Int64, q_stages * 2],
        "umma_mbar_0": cute.struct.MemRange[cutlass.Int64, umma_stages * 2],
        "umma_mbar_1": cute.struct.MemRange[cutlass.Int64, umma_stages * 2],
    }
    if fp4:
        fields["sfb_war_mbar"] = cute.struct.MemRange[cutlass.Int64, 2]
    fields["tmem_holding_buf"] = cutlass.Int32
    return cute.struct(type("SharedStorage", (), {"__annotations__": fields}))


class IndexerKernel:
    def __init__(
        self,
        num_sms: int = NUM_SMS,
        block_kv: int = 128,
        phys_block_kv: int = SPEC.index_page_tokens,
        num_heads: int = SPEC.index_heads,
        head_dim: int = SPEC.index_head_dim,
        policy: int = L.POL_NONE,
        debug_dump: bool = False,
        admit: bool = True,
        kv_stages: int = 3,
        q_stages: int = 3,
        math_regs: int = 136,
        sel_regs: int = 88,
        row_regs: int = 96,
        trig: int = TRIG,
        stamps: bool = False,
        floor_mask: int = 6,
        nolock: bool = False,
        lim_loop2: bool = False,
        lim_loop: bool = False,
        num_w_in_reg: int = 64,
        attn: bool = False,
        dbg_dup: bool = False,
        row_noop: bool = False,
        dbg_hand: bool = False,
        skew_ns: int = 0,
        no_deep: bool = False,
        dbg_addr: bool = False,
        att_skip: int = 0,
        late_release: bool = True,
        tmem_fences: bool = False,
        keep_cap: int = KEEP_CAP,
        desc: bool = False,
        rot: bool = False,
        sel_fin: bool = False,
        gath: bool = False,
        cbins: int = 1024,
        efl: bool = False,
        ahint: int = 0,
        etile: bool = False,
        spin_ns: int = 0,
        pfence: bool = True,
        sel_idle_ns: int = 32,
        rdelay: int = 0,
        apace: int = 0,
        ring2: bool = False,
        rsort: bool = False,
        trig0: int = 0,
        kfloor: bool = False,
        pace_adapt: bool = True,
        defer: bool = True,
        dq_cap: int = DQ_CAP,
        validate: bool = True,
        tiebin: bool = True,
        merge1: int = 0,
        fast_verdict: bool = False,
        sel_fast: bool = False,
        cmp_fast: bool = False,
        sel_hand2: bool = False,
        config: SpecDsaConfig | None = None,
    ):
        # merge1 = G > 0: the last piece of a split row merges the other pieces in one selection, G
        # sources gathered at a time (see _finalize_seg); fast_verdict: the schedule verdict issues
        # its global loads before the first branch
        self.merge1 = int(merge1)
        if not 0 <= self.merge1 <= 8:
            raise ValueError("merge1")
        self.fast_verdict = bool(fast_verdict)
        self.sel_fast = bool(sel_fast)
        self.cmp_fast = bool(cmp_fast)
        # sel_hand2: the selector warps finalize the last segment into their own 512-entry result list and do not
        # wait for the row warps to finish the segment before it (a two-segment CTA made the call 5 - 12 us slower at
        # 40 - 92 rows)
        self.sel_hand2 = bool(sel_hand2)
        self.fp4 = config is not None and config.index_k_dtype == "fp4"
        require_built_for(config, SPEC_FP4 if self.fp4 else SPEC)
        self.block_kv = block_kv
        self.phys_block_kv = phys_block_kv
        self.num_blocks_per_mma = block_kv // phys_block_kv
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.next_n_atom = 1
        self.N = num_heads
        self.num_sms = num_sms
        self.epi_dtype = cutlass.Float32
        self.acc_dtype = cutlass.Float32
        self.output_dtype = cutlass.Float32
        self.epi_bytes = 4
        w_stage_bytes = self.N * self.epi_bytes
        self.w_stage_stride = (w_stage_bytes + 127) // 128 * 128 // self.epi_bytes
        self.num_math_threads = 256
        self.num_math_warps = 8
        self.threads_per_cta = 640
        self.num_q_stages = q_stages
        self.num_umma_stages = 2
        self.num_kv_stages = kv_stages
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.mma_tiler_mn = (block_kv, self.N)
        if policy not in L.POL_NAMES:
            raise ValueError(f"unknown cache policy {policy}")
        self.policy = policy
        self.debug_dump = bool(debug_dump)
        self.admit = bool(admit)
        if attn and q_stages < 3:
            raise ValueError(
                "the deep attention ring keeps Q slices 0-2 in the 3 indexer Q stages (q_stages >= 3)"
            )
        if (
            math_regs * 256 + (24 + sel_regs + row_regs) * 128 > 96 * 640
            or (math_regs | sel_regs | row_regs) % 8
        ):
            raise ValueError("register budget (CTA pool 61,440)")
        self.math_regs = math_regs
        self.sel_regs = sel_regs
        self.row_regs = row_regs
        if not K_TOP < trig <= MAIN_CAP - 512:
            raise ValueError("trig must be in (512, MAIN_CAP - 512]")
        self.trig = trig
        self.stamps = bool(stamps)
        self.floor_mask = int(floor_mask)
        self.nolock = bool(nolock)
        self.lim_loop2 = bool(lim_loop2)
        # lim_loop: the math warps' tile loop ends on kv_idx < kv_lim (limit computed once per segment)
        self.lim_loop = bool(lim_loop)
        if num_heads != 64 or num_w_in_reg != num_heads:
            raise ValueError("split epilogue requires 64 heads and all head weights in registers")
        self.num_w_in_reg = num_w_in_reg
        self.attn = bool(attn)
        self.dbg_dup = bool(dbg_dup)
        self.row_noop = bool(row_noop)
        self.dbg_hand = bool(dbg_hand)
        self.skew_ns = int(skew_ns)
        self.no_deep = bool(no_deep)
        self.dbg_addr = bool(dbg_addr)
        self.att_skip = int(att_skip)
        self.late_release = bool(late_release)
        self.keep_cap = int(keep_cap)
        self.tmem_fences = bool(tmem_fences)
        self.desc = bool(desc)
        self.rot = bool(rot)
        self.sel_fin = bool(sel_fin)
        self.gath = bool(gath) and self.sel_fin and self.attn and (int(att_skip) == 0)
        if cbins not in (128, 256, 512, 1024):
            raise ValueError("cbins")
        self.cbins = int(cbins)
        self.etile = bool(etile) and self.gath
        self.pfence = bool(pfence)
        self.sel_idle_ns = int(sel_idle_ns)
        self.spin_ns = int(spin_ns)
        self.rdelay = int(rdelay)
        self.apace = int(apace)
        self.ahint = int(ahint)
        self.efl = bool(efl)
        self.ring2 = bool(ring2) and self.attn
        self.rsort = bool(rsort)
        if trig0 and trig0 <= K_TOP:
            raise ValueError("trig0 must be > 512")
        self.trig0 = min(int(trig0), int(trig)) if trig0 else 0
        self.kfloor = bool(kfloor)
        self.pace_adapt = bool(pace_adapt)
        if defer and (etile or ring2 or rdelay):
            raise ValueError(
                "defer is built for the shipped tail (gath without etile / ring2 / rdelay)"
            )
        self.defer = bool(defer) and self.attn
        if not 1 <= int(dq_cap) <= 1024:
            raise ValueError("dq_cap")
        self.dq_cap = int(dq_cap)
        self.validate = bool(validate)
        self.tiebin = bool(tiebin)
        if self.fp4:
            # MXFP4 block-scaled MMA (E2M1 data, a UE8M0 scale per 32 elements): the SF TMEM regions are not
            # staged, so there is one accumulator stage
            self.num_umma_stages = 1
            self.num_groups = 2
            self.sf_dtype = Float8E8M0FNU
            self.sf_vec_size = 32
            self.umma_inst_k = 64
        # entries of the idle K-stage smem of the first group (merge1 gathers the pieces' lists)
        self.merge_cap = (
            self.num_kv_stages * self.block_kv * self.head_dim * (4 if self.fp4 else 8) // 64
        )

    def _epi_half(self, tTR_rAcc, w_cache, head_start, s0x, s0y, s1x, s1y):
        # The same four FMA chains continue across the two 32-head TMEM loads.
        acc_vec = tTR_rAcc.load()
        for h in range(0, 32, 4):
            x0 = acc_vec[h]
            x1 = acc_vec[h + 1]
            x2 = acc_vec[h + 2]
            x3 = acc_vec[h + 3]
            a0, a1 = cute.arch.add_packed_f32x2((x0, x1), (abs(x0), abs(x1)), rnd=_RND_RN)
            a2, a3 = cute.arch.add_packed_f32x2((x2, x3), (abs(x2), abs(x3)), rnd=_RND_RN)
            w0 = w_cache[head_start + h]
            w1 = w_cache[head_start + h + 1]
            w2 = w_cache[head_start + h + 2]
            w3 = w_cache[head_start + h + 3]
            s0x, s0y = cute.arch.fma_packed_f32x2((a0, a1), (w0, w1), (s0x, s0y), rnd=_RND_RN)
            s1x, s1y = cute.arch.fma_packed_f32x2((a2, a3), (w2, w3), (s1x, s1y), rnd=_RND_RN)
        return (s0x, s0y, s1x, s1y)

    @cute.jit
    def _nseg(self, start_q, start_kv_half, end_q_idx, end_kv_half):
        qb = end_q_idx
        if end_kv_half == Int32(0):
            qb = end_q_idx - Int32(1)
        ns = qb - start_q + Int32(1)
        if (start_q == end_q_idx) & (start_kv_half == end_kv_half):
            ns = Int32(0)
        return ns

    @cute.jit
    def _tiles(self, q, mContextLens, batch_size):
        return (
            self._ctx_of(cutlass.min(q, batch_size - 1), mContextLens, batch_size)
            + Int32(self.block_kv - 1)
        ) // Int32(self.block_kv)

    @cute.jit
    def _walk_init(self, start_q, start_kv_half, end_q_idx, end_kv_half, mContextLens, batch_size):
        end_kv_idx = end_kv_half * Int32(2)
        nq = start_q
        nk = start_kv_half * Int32(2)
        ph = Int32(2)
        stop_kv = end_kv_idx
        rot_on = Int32(0)
        if cutlass.const_expr(self.rot):
            if (end_kv_half > Int32(0)) & (end_q_idx > start_q):
                rot_on = Int32(1)
                nq = end_q_idx
                nk = Int32(0)
                ph = Int32(1)
                stop_kv = Int32(0)
        nn = self._tiles(nq, mContextLens, batch_size)
        hw = (ph == Int32(1)) | ((nq != end_q_idx) | (nk != stop_kv))
        return (nq, nk, nn, ph, stop_kv, hw, rot_on)

    @cute.jit
    def _walk_next(
        self,
        q,
        nk,
        num_kv,
        nn,
        ph,
        stop_kv,
        start_q,
        start_kv_half,
        end_q_idx,
        end_kv_idx,
        mContextLens,
        batch_size,
        seg_end: cutlass.Constexpr = False,
    ):
        nq = q
        nk2 = nk
        nn2 = nn
        ph2 = ph
        if ph == Int32(1):
            if nk >= end_kv_idx:
                nq = start_q
                nk2 = start_kv_half * Int32(2)
                ph2 = Int32(2)
                nn2 = self._tiles(start_q, mContextLens, batch_size)
        elif nk >= num_kv:
            nq = q + Int32(1)
            nk2 = Int32(0)
            if nq < batch_size:
                nn2 = self._tiles(nq, mContextLens, batch_size)
        hw = (ph2 == Int32(1)) | ((nq != end_q_idx) | (nk2 != stop_kv))
        return (nq, nk2, nn2, ph2, hw)

    @cute.jit
    def _walk_qnext(self, q, ph, rot_on, start_q, end_q_idx, end_kv_idx):
        pq = q + Int32(1)
        ok = Int32(0)
        if ph == Int32(1):
            pq = start_q
            ok = Int32(1)
        elif pq < end_q_idx:
            ok = Int32(1)
        elif pq == end_q_idx:
            if end_kv_idx > Int32(0):
                if rot_on == Int32(0):
                    ok = Int32(1)
        return (pq, ok)

    @cute.jit
    def _kv_limit(self, kv_idx, num_kv, q_idx, end_q_idx, end_kv_idx):
        # the loop bound of a segment (see _seg_s): the row length rounded up to an even tile, or the end tile of the
        # CTA in its last row, at least one pair beyond the first tile
        b2 = num_kv + Int32(1) & Int32(-2)
        if q_idx == end_q_idx:
            b2 = end_kv_idx
        if b2 < kv_idx + Int32(2):
            b2 = kv_idx + Int32(2)
        return b2

    @cute.jit
    def _seg_s(self, kv_idx, num_kv, q_idx, end_q_idx, end_kv_idx):
        b2 = num_kv + Int32(1) & Int32(-2)
        if q_idx == end_q_idx:
            b2 = end_kv_idx
        if b2 < kv_idx + Int32(2):
            b2 = kv_idx + Int32(2)
        return kv_idx + b2 - Int32(2)

    @cute.jit
    def _sched_verdict(self, mSched, mContextLens, batch_size, lane):
        G = self.num_sms
        bad = Int32(0)
        for it in cutlass.range_constexpr((G + 1 + 31) // 32):
            s = lane + Int32(it * 32)
            if s <= Int32(G):
                q = mSched[s, 0]
                k = mSched[s, 1]
                if (q < Int32(0)) | (q > batch_size) | (k < Int32(0)):
                    bad = Int32(1)
                elif q == batch_size:
                    if k != Int32(0):
                        bad = Int32(1)
                else:
                    u = (mContextLens[q] + Int32(255)) // Int32(256)
                    if (k > Int32(0)) & (k >= u):
                        bad = Int32(1)
                if s < Int32(G):
                    q1 = mSched[s + Int32(1), 0]
                    k1 = mSched[s + Int32(1), 1]
                    if (q1 < q) | (q1 == q) & (k1 < k):
                        bad = Int32(1)
                elif q != batch_size:
                    bad = Int32(1)
        q0 = cutlass.min(cutlass.max(mSched[0, 0], Int32(0)), batch_size)
        r = lane
        while r < q0:
            if mContextLens[r] != Int32(0):
                bad = Int32(1)
            r = r + Int32(32)
        if mSched[0, 1] != Int32(0):
            bad = Int32(1)
        v = Int32(0)
        if Int32(cute.arch.vote_ballot_sync(bad != Int32(0))) != Int32(0):
            v = Int32(ST_SCHED)
        return v

    @cute.jit
    def _sched_verdict_fast(self, mSched, mContextLens, batch_size, lane):
        # The verdict of _sched_verdict with every global load issued before the first branch.
        G = self.num_sms
        n_it = (G + 1 + 31) // 32
        qs = []
        ks = []
        q1s = []
        k1s = []
        for it in cutlass.range_constexpr(n_it):
            s = lane + Int32(it * 32)
            sc = cutlass.min(s, Int32(G))
            s1 = cutlass.min(s + Int32(1), Int32(G))
            qs.append(mSched[sc, 0])
            ks.append(mSched[sc, 1])
            q1s.append(mSched[s1, 0])
            k1s.append(mSched[s1, 1])
        us = []
        for it in cutlass.range_constexpr(n_it):
            qc = cutlass.min(cutlass.max(qs[it], Int32(0)), batch_size - Int32(1))
            us.append((mContextLens[qc] + Int32(255)) // Int32(256))
        q00 = mSched[0, 0]
        k00 = mSched[0, 1]
        bad = Int32(0)
        for it in cutlass.range_constexpr(n_it):
            s = lane + Int32(it * 32)
            q = qs[it]
            k = ks[it]
            q1 = q1s[it]
            k1 = k1s[it]
            b1 = (q < Int32(0)) | (q > batch_size) | (k < Int32(0))
            b2 = (q == batch_size) & (k != Int32(0))
            b3 = (q >= Int32(0)) & (q < batch_size) & (k > Int32(0)) & (k >= us[it])
            b4 = (s < Int32(G)) & ((q1 < q) | ((q1 == q) & (k1 < k)))
            b5 = (s == Int32(G)) & (q != batch_size)
            bad = cutlass.select_((s <= Int32(G)) & (b1 | b2 | b3 | b4 | b5), Int32(1), bad)
        q0 = cutlass.min(cutlass.max(q00, Int32(0)), batch_size)
        r = lane
        while r < q0:
            if mContextLens[r] != Int32(0):
                bad = Int32(1)
            r = r + Int32(32)
        if k00 != Int32(0):
            bad = Int32(1)
        v = Int32(0)
        if Int32(cute.arch.vote_ballot_sync(bad != Int32(0))) != Int32(0):
            v = Int32(ST_SCHED)
        return v

    @cute.jit
    def _ctx_of(self, q, mContextLens, batch_size):
        ctx = cutlass.Int32(0)
        if q < batch_size:
            ctx = mContextLens[q]
        return ctx

    def _setup_mma(self, a_dtype, b_dtype, a_major, b_major):
        if self.fp4:
            return self._setup_mma_fp4(a_dtype, b_dtype, a_major, b_major)
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.mma_tiler = (*self.mma_tiler_mn, 1)
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            a_dtype, b_dtype, a_major, b_major, self.acc_dtype, self.cta_group, self.mma_tiler_mn
        )
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = self.head_dim // mma_inst_shape_k
        full_k = mma_inst_shape_k * mma_inst_tile_k
        self.mma_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], full_k)
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.epi_tile = self.cta_tile_shape_mnk[:2]
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, a_dtype, self.num_kv_stages
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, b_dtype, self.num_q_stages
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        self.num_tmem_alloc_cols = cutlass.utils.get_num_tmem_alloc_cols(tCtAcc_fake)
        self.num_tmem_alloc_cols_total = self.num_tmem_alloc_cols * 2 * self.num_umma_stages
        return tiled_mma

    def _setup_mma_fp4(self, a_dtype, b_dtype, a_major, b_major):
        """The block-scaled MXF4 MMA (K 64 per instruction, two per head dim) and its scale-factor layouts."""
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.mma_tiler = (*self.mma_tiler_mn, 1)
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            a_dtype,
            b_dtype,
            a_major,
            b_major,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_tiler_mn,
        )
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = self.head_dim // mma_inst_shape_k
        full_k = mma_inst_shape_k * mma_inst_tile_k
        self.mma_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], full_k)
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.epi_tile = self.cta_tile_shape_mnk[:2]
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, a_dtype, self.num_kv_stages
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, b_dtype, self.num_q_stages
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        self.num_tmem_alloc_cols = find_tmem_tensor_col_offset(tCtAcc_fake)
        sfa_smem = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, 1
        )
        sfb_smem = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, self.sf_vec_size, 1
        )
        tmem_sfa = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(sfa_smem, (None, None, None, 0)),
        )
        tmem_sfb = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            cute.slice_(sfb_smem, (None, None, None, 0)),
        )
        # a degenerate stage mode keeps the rank-4 slices of the MMA loop valid on both DSL versions
        self.tmem_sfa_layout = cute.append(tmem_sfa, cute.make_layout(1, stride=0))
        self.tmem_sfb_layout = cute.append(tmem_sfb, cute.make_layout(1, stride=0))
        sf_k_modes = cute.size(tmem_sfa, mode=[2])
        sf_tile_k = self.head_dim // self.umma_inst_k
        assert sf_k_modes % sf_tile_k == 0
        self.sf_k_step = sf_k_modes // sf_tile_k
        self.N_padded = ((self.N + 127) // 128) * 128
        self.num_sfa_tmem_cols = (self.cta_tile_shape_mnk[0] // 32) * sf_tile_k
        self.num_sfb_tmem_cols = (self.N_padded // 32) * sf_tile_k
        raw_cols = (
            self.num_tmem_alloc_cols * self.num_groups * self.num_umma_stages
            + self.num_sfa_tmem_cols * self.num_groups
            + self.num_sfb_tmem_cols
        )
        self.num_tmem_alloc_cols_total = max(1 << math.ceil(math.log2(raw_cols)), 32)
        assert self.num_tmem_alloc_cols_total <= 512, "FP4 TMEM columns exceed the 512 of SM100"
        self.sfa_chunk_smem_layout = sfa_smem
        self.sfb_chunk_smem_layout = sfb_smem
        self.copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype
        )
        return tiled_mma

    def _fp4_views(self, kv_fused, b, num_phys_blocks):
        """FP4 views of a K page pool [pages, 32 * 68 bytes] and of the queries.

        A page: the E2M1 data of 32 positions, then 32 int32 of four UE8M0 scales each.
        """
        phys_block_kv = self.phys_block_kv
        kv_fp4 = cute.recast_tensor(kv_fused, Float4E2M1FN)
        b = cute.recast_tensor(b, Float4E2M1FN)
        block_stride_bytes = kv_fused.layout.stride[0]
        a = cute.make_tensor(
            kv_fp4.iterator,
            cute.make_layout(
                (phys_block_kv, self.head_dim, num_phys_blocks),
                stride=(self.head_dim, 1, block_stride_bytes * 2),
            ),
        )
        sf_bytes = cute.make_tensor(
            kv_fused.iterator + phys_block_kv * (self.head_dim // 2),
            cute.make_layout((phys_block_kv * 4, num_phys_blocks), stride=(1, block_stride_bytes)),
        )
        return a, b, cute.recast_tensor(sf_bytes, cutlass.Int32)

    def _fp4_sf_q_atom(self, sf_q, tma_load_op):
        """TMA atom of the query scales [64, B] int32 (the SMEM stage is padded to the 128-token UTCCP atom)."""
        stage_stride = (self.N_padded * 4 + 127) // 128 * 128 // 4
        self.sf_q_smem_layout_staged = cute.make_layout(
            (self.N_padded, self.num_q_stages), stride=(1, stage_stride)
        )
        self.sf_q_tma_smem_layout_staged = cute.make_layout(
            (self.N, self.num_q_stages), stride=(1, stage_stride)
        )
        return cpasync.make_tiled_tma_atom(
            tma_load_op,
            sf_q,
            cute.select(self.sf_q_tma_smem_layout_staged, mode=[0]),
            (self.N,),
        )

    @cute.jit
    def __call__(
        self,
        kv_fused: cute.Tensor,
        b: cute.Tensor,
        weights: cute.Tensor,
        logits: cute.Tensor,
        block_table: cute.Tensor,
        context_lens: cute.Tensor,
        schedule_meta: cute.Tensor,
        kv_lens: cute.Tensor,
        req_id: cute.Tensor,
        bt_swa: cute.Tensor,
        bt_c: cute.Tensor,
        topk_out: cute.Tensor,
        gidx: cute.Tensor,
        ws_seg: cute.Tensor,
        ws_ctr: cute.Tensor,
        rowstat: cute.Tensor,
        stamps: cute.Tensor,
        q_att: cute.Tensor,
        swa_pool: cute.Tensor,
        c_pool: cute.Tensor,
        sinks: cute.Tensor,
        out: cute.Tensor,
        status: cute.Tensor,
        softmax_scale: cutlass.Float32,
        num_phys_blocks: cutlass.Int32,
        batch_size: cutlass.Int32,
        stream: cuda.CUstream,
        sf_q: cute.Tensor = None,
    ):
        phys_block_kv = self.phys_block_kv
        scale_offset_elems = phys_block_kv * self.head_dim
        if cutlass.const_expr(self.fp4):
            a, b, scales = self._fp4_views(kv_fused, b, num_phys_blocks)
        else:
            kv_fp8 = cute.recast_tensor(kv_fused, cutlass.Float8E4M3FN)
            b = cute.recast_tensor(b, cutlass.Float8E4M3FN)
            kv_block_stride = kv_fused.layout.stride[0]
            kv_layout = cute.make_layout(
                (phys_block_kv, self.head_dim, num_phys_blocks),
                stride=(self.head_dim, 1, kv_block_stride),
            )
            a = cute.make_tensor(kv_fp8.iterator, kv_layout)
            scale_fp8_layout = cute.make_layout(
                (phys_block_kv * 4, num_phys_blocks), stride=(1, kv_block_stride)
            )
            scale_fp8 = cute.make_tensor(kv_fp8.iterator + scale_offset_elems, scale_fp8_layout)
            scales = cute.recast_tensor(scale_fp8, cutlass.Float32)
        a_dtype = a.element_type
        b_dtype = b.element_type
        a_major = cutlass.utils.LayoutEnum.from_tensor(a).mma_major_mode()
        b_major = cutlass.utils.LayoutEnum.ROW_MAJOR.mma_major_mode()
        tiled_mma = self._setup_mma(a_dtype, b_dtype, a_major, b_major)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp()
        self.a_tma_view_layout = sm100_utils.make_smem_layout(
            cute.nvgpu.OperandMajorMode.K,
            (self.block_kv, self.head_dim),
            a_dtype,
            self.num_kv_stages,
        )
        self.a_tma_view_layout = cute.tiled_divide(
            self.a_tma_view_layout, (self.phys_block_kv, self.head_dim)
        )
        self.a_tma_view_layout = cute.select(self.a_tma_view_layout, mode=[0, 1, 3])
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            tma_load_op, a, self.a_tma_view_layout[0], (self.phys_block_kv, self.head_dim)
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        self.w_smem_layout_staged = cute.make_layout(
            (self.N, self.num_q_stages), stride=(1, self.w_stage_stride)
        )
        w_smem_per_stage = cute.select(self.w_smem_layout_staged, mode=[0])
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_load_op, weights, w_smem_per_stage, self.w_smem_layout_staged.shape[:1]
        )
        self.s_smem_layout_staged = cute.make_layout((self.block_kv, self.num_kv_stages))
        s_smem_per_subblock = cute.make_layout((phys_block_kv,))
        tma_atom_s, tma_tensor_s = cpasync.make_tiled_tma_atom(
            tma_load_op, scales, s_smem_per_subblock, (phys_block_kv,)
        )
        b_copy_size = cute.size_in_bytes(b_dtype, b_smem_layout)
        w_copy_size = self.N * self.epi_bytes
        kv_tma_bytes_per_subblock = phys_block_kv * self.head_dim // (2 if self.fp4 else 1)
        scale_tma_bytes_per_subblock = phys_block_kv * 4
        self.num_kv_scale_tma_bytes = self.num_blocks_per_mma * (
            kv_tma_bytes_per_subblock + scale_tma_bytes_per_subblock
        )
        self.num_q_tma_bytes = b_copy_size * atom_thr_size + w_copy_size
        tma_atom_sf_q = None
        tma_tensor_sf_q = None
        sf_q_smem_layout_staged = None
        sf_q_tma_smem_layout_staged = None
        tmem_sfa_layout = None
        tmem_sfb_layout = None
        sfa_chunk_smem_layout = None
        sfb_chunk_smem_layout = None
        copy_atom_s2t = None
        if cutlass.const_expr(self.fp4):
            tma_atom_sf_q, tma_tensor_sf_q = self._fp4_sf_q_atom(sf_q, tma_load_op)
            self.num_q_tma_bytes = self.num_q_tma_bytes + self.N * 4
            sf_q_smem_layout_staged = self.sf_q_smem_layout_staged
            sf_q_tma_smem_layout_staged = self.sf_q_tma_smem_layout_staged
            tmem_sfa_layout = self.tmem_sfa_layout
            tmem_sfb_layout = self.tmem_sfb_layout
            sfa_chunk_smem_layout = self.sfa_chunk_smem_layout
            sfb_chunk_smem_layout = self.sfb_chunk_smem_layout
            copy_atom_s2t = self.copy_atom_s2t

        SharedStorage = _shared_storage(
            self.num_kv_stages, self.num_q_stages, self.num_umma_stages, self.fp4
        )

        tma_q = xcuda.create_tensor_map_tiled(
            global_address=q_att.iterator.toint(),
            dtype=cutlass.Float8E4M3FN,
            global_dims=[LATENT, HEADS, q_att.shape[0]],
            global_strides=[LATENT // 16, HEADS * LATENT // 16],
            box_dims=[SLICE, HEADS, 1],
            swizzle=xcuda.TensorMapSwizzle.s128b,
        )
        tma_swa = xcuda.create_tensor_map_tiled(
            global_address=swa_pool.iterator.toint(),
            dtype=cutlass.Float8E4M3FN,
            global_dims=[LATENT, swa_pool.shape[0]],
            global_strides=[LATENT // 16],
            box_dims=[SLICE, 1],
            swizzle=xcuda.TensorMapSwizzle.s128b,
        )
        tma_cmp = xcuda.create_tensor_map_tiled(
            global_address=c_pool.iterator.toint(),
            dtype=cutlass.Float8E4M3FN,
            global_dims=[LATENT, c_pool.shape[0]],
            global_strides=[LATENT // 16],
            box_dims=[SLICE, 1],
            swizzle=xcuda.TensorMapSwizzle.s128b,
        )
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_w,
            tma_tensor_w,
            tma_atom_s,
            tma_tensor_s,
            tma_q,
            tma_swa,
            tma_cmp,
            sinks,
            out,
            status,
            softmax_scale,
            logits,
            block_table,
            context_lens,
            schedule_meta,
            kv_lens,
            req_id,
            bt_swa,
            bt_c,
            topk_out,
            gidx,
            ws_seg,
            ws_ctr,
            rowstat,
            stamps,
            batch_size,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.w_smem_layout_staged,
            self.s_smem_layout_staged,
            self.a_tma_view_layout,
            self.epi_tile,
            SharedStorage,
            tma_atom_sf_q,
            tma_tensor_sf_q,
            sf_q_smem_layout_staged,
            sf_q_tma_smem_layout_staged,
            tmem_sfa_layout,
            tmem_sfb_layout,
            sfa_chunk_smem_layout,
            sfb_chunk_smem_layout,
            copy_atom_s2t,
        ).launch(
            grid=(1, 1, self.num_sms),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_w: cute.CopyAtom,
        mW_tma: cute.Tensor,
        tma_atom_s: cute.CopyAtom,
        mS_tma: cute.Tensor,
        tma_q: cutlass.GridConstant[xcuda.TensorMap],
        tma_swa: cutlass.GridConstant[xcuda.TensorMap],
        tma_cmp: cutlass.GridConstant[xcuda.TensorMap],
        mSinks: cute.Tensor,
        mOut: cute.Tensor,
        mStatus: cute.Tensor,
        softmax_scale: cutlass.Float32,
        mLogits: cute.Tensor,
        mBlockTable: cute.Tensor,
        mContextLens: cute.Tensor,
        mScheduleMeta: cute.Tensor,
        mKvLens: cute.Tensor,
        mReq: cute.Tensor,
        mBtSwa: cute.Tensor,
        mBtC: cute.Tensor,
        mTopk: cute.Tensor,
        mGidx: cute.Tensor,
        mWsSeg: cute.Tensor,
        mWsCtr: cute.Tensor,
        mRowStat: cute.Tensor,
        mStamps: cute.Tensor,
        batch_size: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        w_smem_layout_staged: cute.Layout,
        s_smem_layout_staged: cute.Layout,
        a_tma_view_layout: cute.ComposedLayout,
        epi_tile: cute.Tile,
        SharedStorage: cutlass.Constexpr,
        tma_atom_sf_q: cute.CopyAtom = None,
        mSF_Q_tma: cute.Tensor = None,
        sf_q_smem_layout_staged: cute.Layout = None,
        sf_q_tma_smem_layout_staged: cute.Layout = None,
        tmem_sfa_layout: cute.Layout = None,
        tmem_sfb_layout: cute.Layout = None,
        sfa_chunk_smem_layout: cute.Layout = None,
        sfb_chunk_smem_layout: cute.Layout = None,
        copy_atom_s2t: cute.CopyAtom = None,
    ):
        # Let the attention grid start on the SMs this grid leaves free; it consumes rows as they are published.
        cute.arch.griddepcontrol_launch_dependents()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()
        warpgroup_idx = warp_idx // 4
        is_math_warp = warp_idx < 8
        is_tma_warp_0 = warp_idx == 8
        is_tma_warp_1 = warp_idx == 9
        is_tma_warp = is_tma_warp_0 | is_tma_warp_1
        is_umma_warp_0 = warp_idx == 10
        is_umma_warp_1 = warp_idx == 11
        is_sel_warp = (warp_idx >= 12) & (warp_idx < 16)
        is_row_warp = warp_idx >= 16
        if cutlass.const_expr(self.stamps):
            if tidx == 0:
                mStamps[bidz * 256 + 0] = cutlass.Int64(cute.arch.globaltimer())
                mStamps[bidz * 256 + 15] = cutlass.Int64(cute.arch.smid())
                mStamps[bidz * 256 + 13] = L.clk64()
        NUM_MATH_WG = 2
        NUM_BLOCKS_PER_MMA = self.num_blocks_per_mma
        sm_idx = bidz
        start_q = mScheduleMeta[sm_idx, 0]
        start_kv_half = mScheduleMeta[sm_idx, 1]
        end_q_idx = mScheduleMeta[sm_idx + 1, 0]
        end_kv_half = mScheduleMeta[sm_idx + 1, 1]
        refused = Int32(0)
        if cutlass.const_expr(self.validate):
            if cutlass.const_expr(self.fast_verdict):
                refused = self._sched_verdict_fast(
                    mScheduleMeta, mContextLens, batch_size, tidx % 32
                )
            else:
                refused = self._sched_verdict(mScheduleMeta, mContextLens, batch_size, tidx % 32)
            if refused != Int32(0):
                start_q = batch_size
                start_kv_half = Int32(0)
                end_q_idx = batch_size
                end_kv_half = Int32(0)
        if is_tma_warp:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_w)
            cpasync.prefetch_descriptor(tma_atom_s)
            if cutlass.const_expr(self.fp4):
                cpasync.prefetch_descriptor(tma_atom_sf_q)
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        block_kv_val = self.block_kv
        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        q_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 8)
        q_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_mbar.data_ptr(),
            num_stages=self.num_q_stages,
            producer_group=prod_group,
            consumer_group=q_cons_group,
            tx_count=self.num_q_tma_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=tidx,
            defer_sync=True,
        )
        q_prod_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_q_stages
        )
        q_cons_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        q_cons_state_umma_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        q_cons_state_umma_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_q_stages
        )
        math_warps_per_group = self.num_math_warps // 2
        if cutlass.const_expr(self.fp4):
            # The tensor core consumes the K tile and its scale factors: the UMMA warp releases the stage
            # (tcgen05.commit).
            kv_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
            kv_pipeline_0 = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.kv_mbar_0.data_ptr(),
                num_stages=self.num_kv_stages,
                producer_group=prod_group,
                consumer_group=kv_cons_group,
                tx_count=self.num_kv_scale_tma_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            kv_pipeline_1 = pipeline.PipelineTmaUmma.create(
                barrier_storage=storage.kv_mbar_1.data_ptr(),
                num_stages=self.num_kv_stages,
                producer_group=prod_group,
                consumer_group=kv_cons_group,
                tx_count=self.num_kv_scale_tma_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
        else:
            kv_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, math_warps_per_group)
            kv_pipeline_0 = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.kv_mbar_0.data_ptr(),
                num_stages=self.num_kv_stages,
                producer_group=prod_group,
                consumer_group=kv_cons_group,
                tx_count=self.num_kv_scale_tma_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                tidx=tidx,
                defer_sync=True,
            )
            kv_pipeline_1 = pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.kv_mbar_1.data_ptr(),
                num_stages=self.num_kv_stages,
                producer_group=prod_group,
                consumer_group=kv_cons_group,
                tx_count=self.num_kv_scale_tma_bytes,
                cta_layout_vmnk=cluster_layout_vmnk,
                tidx=tidx,
                defer_sync=True,
            )
        kv_prod_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )
        kv_prod_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_kv_stages
        )
        kv_cons_state_umma_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        kv_cons_state_umma_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        kv_cons_state_math_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        kv_cons_state_math_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_kv_stages
        )
        math_threads_per_group = self.num_math_threads // 2
        umma_pipeline_0 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar_0.data_ptr(),
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, math_threads_per_group),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        umma_pipeline_1 = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.umma_mbar_1.data_ptr(),
            num_stages=self.num_umma_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, math_threads_per_group),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        umma_prod_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_umma_stages
        )
        umma_prod_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_umma_stages
        )
        umma_cons_state_0 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_umma_stages
        )
        umma_cons_state_1 = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_umma_stages
        )
        if cutlass.const_expr(self.fp4):
            # At a q transition UMMA warp 0 overwrites the SFB TMEM region that both UMMA warps' previous MMAs
            # read; each warp's tcgen05.commit lands on this mbarrier and warp 0 waits for both before the copy.
            # The named barrier locksteps the two warps.
            sfb_war_pipeline = pipeline.PipelineUmmaAsync.create(
                barrier_storage=storage.sfb_war_mbar.data_ptr(),
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 2),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                cta_layout_vmnk=cluster_layout_vmnk,
                defer_sync=True,
            )
            sfb_war_prod_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            sfb_war_cons_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            sfb_sync_barrier = pipeline.NamedBarrier(barrier_id=6, num_threads=64)
        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=640)
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=False,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
        sKV_0 = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=a_smem_layout_staged.inner,
        )
        sKV_1 = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=a_smem_layout_staged.inner,
        )
        sQ = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=b_smem_layout_staged.inner,
        )
        sW = smem.allocate_tensor(
            element_type=self.epi_dtype, layout=w_smem_layout_staged, byte_alignment=128
        )
        scale_dtype = cutlass.Int32 if self.fp4 else cutlass.Float32
        sScales_0 = smem.allocate_tensor(
            element_type=scale_dtype, layout=s_smem_layout_staged, byte_alignment=128
        )
        sScales_1 = smem.allocate_tensor(
            element_type=scale_dtype, layout=s_smem_layout_staged, byte_alignment=128
        )
        if cutlass.const_expr(self.fp4):
            sSF_Q = smem.allocate_tensor(
                element_type=cutlass.Int32,
                layout=sf_q_smem_layout_staged,
                byte_alignment=128,
            )
        s_ring = smem.allocate_tensor(
            element_type=Uint64, layout=cute.make_layout(RING * 8), byte_alignment=1024
        )
        sMain = smem.allocate_tensor(
            element_type=Uint64, layout=cute.make_layout(2 * MAIN_CAP), byte_alignment=1024
        )
        sOut = smem.allocate_tensor(
            element_type=Uint64, layout=cute.make_layout(K_TOP), byte_alignment=128
        )
        sHand = smem.allocate_tensor(
            element_type=Uint64, layout=cute.make_layout(K_TOP), byte_alignment=128
        )
        sHand2 = sHand
        if cutlass.const_expr(self.sel_hand2):  # the selector warps' own result list (4 KB)
            sHand2 = smem.allocate_tensor(
                element_type=Uint64, layout=cute.make_layout(K_TOP), byte_alignment=128
            )
        sHB = smem.allocate_tensor(
            element_type=Uint64, layout=cute.make_layout(K_TOP), byte_alignment=128
        )
        sHist = cute.make_tensor(
            cute.recast_ptr(sHB.iterator, dtype=Int32), cute.make_layout(L.NB_SEL)
        )
        sAttB = smem.allocate_tensor(
            element_type=Uint64, layout=cute.make_layout(K_TOP), byte_alignment=128
        )
        sAttO = smem.allocate_tensor(
            element_type=Uint64, layout=cute.make_layout(K_TOP), byte_alignment=128
        )
        sHI = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(16), byte_alignment=16
        )
        # In the split path these arguments are unused: no attention role runs.
        # Alias existing storage instead of reserving the 24 KiB shallow ring.
        if cutlass.const_expr(self.attn):
            sQatt = smem.allocate_tensor(
                element_type=cutlass.Float8E4M3FN,
                layout=cute.make_layout(Q_SLICE_BYTES),
                byte_alignment=1024,
            )
            sKVatt = smem.allocate_tensor(
                element_type=cutlass.Float8E4M3FN,
                layout=cute.make_layout(SLOT_BYTES),
                byte_alignment=1024,
            )
        else:
            sQatt = cute.make_tensor(
                cute.recast_ptr(sQ.iterator, dtype=cutlass.Float8E4M3FN),
                cute.make_layout(Q_SLICE_BYTES),
            )
            sKVatt = cute.make_tensor(
                cute.recast_ptr(sKV_0.iterator, dtype=cutlass.Float8E4M3FN),
                cute.make_layout(SLOT_BYTES),
            )
        sAbar = smem.allocate_tensor(
            element_type=Int64, layout=cute.make_layout(16), byte_alignment=16
        )
        sMsk = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(8), byte_alignment=16
        )
        sRS = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(16), byte_alignment=16
        )
        smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(ATTQ + 4), byte_alignment=16
        )
        sKr = smem.allocate_tensor(
            element_type=Uint32, layout=cute.make_layout(4), byte_alignment=16
        )
        sTb = smem.allocate_tensor(
            element_type=Uint32, layout=cute.make_layout(8), byte_alignment=16
        )
        sCk = smem.allocate_tensor(
            element_type=Uint64, layout=cute.make_layout(L.QUADC), byte_alignment=16
        )
        sHead = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(8), byte_alignment=128
        )
        sTail = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(8), byte_alignment=128
        )
        sTw = smem.allocate_tensor(
            element_type=Uint32, layout=cute.make_layout(8), byte_alignment=128
        )
        sCnt = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(4), byte_alignment=16
        )
        sRes = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(8), byte_alignment=16
        )
        sRed = smem.allocate_tensor(
            element_type=Uint32, layout=cute.make_layout(8), byte_alignment=16
        )
        sWc = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(8), byte_alignment=16
        )
        sScal = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(32), byte_alignment=16
        )
        sHQ = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(16), byte_alignment=16
        )
        sRres = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(8), byte_alignment=16
        )
        sRred = smem.allocate_tensor(
            element_type=Uint32, layout=cute.make_layout(8), byte_alignment=16
        )
        sRwc = smem.allocate_tensor(
            element_type=Int32, layout=cute.make_layout(8), byte_alignment=16
        )
        sGv = cute.make_tensor(
            cute.recast_ptr(sOut.iterator, dtype=Int32), cute.make_layout(2 * K_TOP)
        )
        sDQ = smem.allocate_tensor(
            element_type=Int32,
            layout=cute.make_layout(self.dq_cap if self.defer else 4),
            byte_alignment=16,
        )
        sRingBar = smem.allocate_tensor(
            element_type=Int64,
            layout=cute.make_layout(2 * 8 * (RING // RING_CHUNK)),
            byte_alignment=16,
        )
        if tidx < 2 * 8 * (RING // RING_CHUNK):
            prims.mbarrier_init(sRingBar.iterator.raw_ptr() + tidx, 32)
        # full[2], empty[2], final row scratch ownership, TMEM-address ready.
        sRowBar = smem.allocate_tensor(
            element_type=Int64, layout=cute.make_layout(6), byte_alignment=16
        )
        sCov = None
        if cutlass.const_expr(self.merge1 > 0):
            sCov = smem.allocate_tensor(
                element_type=Int32, layout=cute.make_layout(COV_CAP + 8), byte_alignment=16
            )
        if tidx == 0:
            for i in cutlass.range_constexpr(5):
                prims.mbarrier_init(sRowBar.iterator.raw_ptr() + i, 128)
            prims.mbarrier_init(sRowBar.iterator.raw_ptr() + 5, 1)
        prims.fence_mbarrier_init()
        if tidx < 8:
            sHead[tidx] = Int32(0)
            sTail[tidx] = Int32(0)
        if cutlass.const_expr(self.dbg_addr):
            if tidx == 0:
                bufs = [
                    sKV_0,
                    sKV_1,
                    sQ,
                    sW,
                    sScales_0,
                    sScales_1,
                    s_ring,
                    sMain,
                    sOut,
                    sHand,
                    sHB,
                    sAttB,
                    sAttO,
                    sHI,
                    sQatt,
                    sKVatt,
                    sAbar,
                    sMsk,
                ]
                for k in cutlass.range_constexpr(len(bufs)):
                    mStamps[bidz * 256 + 64 + k] = cutlass.Int64(bufs[k].iterator.toint())
        if tidx < 8:
            sTw[tidx] = Uint32(0)
        if tidx == 0:
            sCnt[0] = Int32(0)
        if tidx < 16:
            sHI[tidx] = Int32(0)
        if cutlass.const_expr(self.attn):
            if warp_idx == 16:
                if prims.elect_sync():
                    for i in cutlass.range_constexpr(15):
                        prims.mbarrier_init(sAbar.iterator.raw_ptr() + i, 1)
                prims.fence_mbarrier_init()
        a_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        sKV_0_for_tma = cute.make_tensor(sKV_0.iterator, a_tma_view_layout.outer)
        sKV_1_for_tma = cute.make_tensor(sKV_1.iterator, a_tma_view_layout.outer)
        gA = cute.local_tile(mA_mkl, (self.phys_block_kv, self.head_dim), coord=(None, None, None))
        tAsA_0, tAgA_0 = cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1), sKV_0_for_tma, cute.group_modes(gA, 0, 2)
        )
        tAsA_1, tAgA_1 = cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1), sKV_1_for_tma, cute.group_modes(gA, 0, 2)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        tCgB = thr_mma.partition_B(gB_nkl)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tBgB = tBgB[None, 0, None, None]
        w_cta_layout = cute.make_layout((1,))
        tWsW, tWgW = cpasync.tma_partition(
            tma_atom_w, 0, w_cta_layout, cute.group_modes(sW, 0, 1), cute.group_modes(mW_tma, 0, 1)
        )
        s_tma_view_layout = cute.make_layout(
            (self.phys_block_kv, self.num_blocks_per_mma, self.num_kv_stages),
            stride=(1, self.phys_block_kv, self.block_kv),
        )
        sScales_0_for_tma = cute.make_tensor(sScales_0.iterator, s_tma_view_layout)
        sScales_1_for_tma = cute.make_tensor(sScales_1.iterator, s_tma_view_layout)
        gS = cute.local_tile(mS_tma, (self.phys_block_kv,), coord=(None, None))
        tSsS_0, tSgS_0 = cpasync.tma_partition(
            tma_atom_s, 0, cute.make_layout(1), sScales_0_for_tma, gS
        )
        tSsS_1, tSgS_1 = cpasync.tma_partition(
            tma_atom_s, 0, cute.make_layout(1), sScales_1_for_tma, gS
        )
        if cutlass.const_expr(self.fp4):
            sSF_Q_for_tma = cute.make_tensor(sSF_Q.iterator, sf_q_tma_smem_layout_staged)
            tSF_Q_sSF_Q, tSF_Q_gSF_Q = cpasync.tma_partition(
                tma_atom_sf_q,
                0,
                cute.make_layout((1,)),
                cute.group_modes(sSF_Q_for_tma, 0, 1),
                cute.group_modes(mSF_Q_tma, 0, 1),
            )
        tCrA_0 = tiled_mma.make_fragment_A(sKV_0)
        tCrA_1 = tiled_mma.make_fragment_A(sKV_1)
        tCrB = tiled_mma.make_fragment_B(sQ)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        us = self.num_umma_stages
        cols = self.num_tmem_alloc_cols
        acc_shape_staged = cute.append(acc_shape, us)
        tCtAcc_fake_staged = tiled_mma.make_fragment_C(acc_shape_staged)
        cols_per_group = cols * us * (32 // self.acc_dtype.width)
        num_tmem_alloc_cols_total = self.num_tmem_alloc_cols_total
        c_layout = cutlass.utils.LayoutEnum.ROW_MAJOR
        epi_sub_mn = (epi_tile[0], 32)
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            c_layout,
            self.acc_dtype,
            self.acc_dtype,
            epi_sub_mn,
            use_2cta_instrs,
        )
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        # All participants retrieve TMEM at the same static barrier site.
        # Separate waits inside the role branches violate aligned-barrier rules.
        if cutlass.const_expr(self.attn):
            tmem.allocate(512)
        else:
            tmem.allocate(num_tmem_alloc_cols_total)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        if warpgroup_idx == 2:
            cute.arch.warpgroup_reg_dealloc(24)
        if is_tma_warp_0:
            pol = None
            if cutlass.const_expr(self.policy != L.POL_NONE):
                pol = L.make_l2_policy(self.policy)
            end_kv_idx = end_kv_half * NUM_MATH_WG
            next_q_idx, next_kv_idx, next_num_kv, phase, stop_kv, has_work, rot_on = (
                self._walk_init(
                    start_q, start_kv_half, end_q_idx, end_kv_half, mContextLens, batch_size
                )
            )
            q_idx = batch_size
            lane_idx = tidx % 32
            cached_blks = [cutlass.Int32(0) for _ in range(NUM_BLOCKS_PER_MMA)]
            kv_blk_ptr = cutlass.Int32(32)
            bt_cols = cute.size(mBlockTable, mode=[1])
            if has_work:
                q_pipeline.producer_acquire(q_prod_state)
                q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                cute.copy(
                    tma_atom_b,
                    tBgB[None, 0, next_q_idx],
                    tBsB[None, q_prod_state.index],
                    tma_bar_ptr=q_bar,
                    mcast_mask=b_mcast_mask,
                )
                cute.copy(
                    tma_atom_w,
                    tWgW[None, next_q_idx],
                    tWsW[None, q_prod_state.index],
                    tma_bar_ptr=q_bar,
                )
                if cutlass.const_expr(self.fp4):
                    cute.copy(
                        tma_atom_sf_q,
                        tSF_Q_gSF_Q[None, next_q_idx],
                        tSF_Q_sSF_Q[None, q_prod_state.index],
                        tma_bar_ptr=q_bar,
                    )
                q_prod_state.advance()
            seg_s = Int32(0)
            while has_work:
                q_idx_old = q_idx
                q_idx = next_q_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv
                if cutlass.const_expr(self.lim_loop2):
                    kv_lim = self._kv_limit(kv_idx, num_kv, q_idx, end_q_idx, end_kv_idx)
                if q_idx != q_idx_old:
                    if cutlass.const_expr(self.desc):
                        seg_s = self._seg_s(kv_idx, num_kv, q_idx, end_q_idx, end_kv_idx)
                    kv_blk_ptr = cutlass.Int32(32)
                    prefetch_next, pf_ok = self._walk_qnext(
                        q_idx, phase, rot_on, start_q, end_q_idx, end_kv_idx
                    )
                    if pf_ok != Int32(0):
                        q_pipeline.producer_acquire(q_prod_state)
                        q_bar = q_pipeline.producer_get_barrier(q_prod_state)
                        cute.copy(
                            tma_atom_b,
                            tBgB[None, 0, prefetch_next],
                            tBsB[None, q_prod_state.index],
                            tma_bar_ptr=q_bar,
                            mcast_mask=b_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_w,
                            tWgW[None, prefetch_next],
                            tWsW[None, q_prod_state.index],
                            tma_bar_ptr=q_bar,
                        )
                        if cutlass.const_expr(self.fp4):
                            cute.copy(
                                tma_atom_sf_q,
                                tSF_Q_gSF_Q[None, prefetch_next],
                                tSF_Q_sSF_Q[None, q_prod_state.index],
                                tma_bar_ptr=q_bar,
                            )
                        q_prod_state.advance()
                # Keep row metadata and Q handoffs outside the per-tile loop.
                # One dummy pair is still issued for a zero-length row.
                kv_go = cutlass.Boolean(True)
                while kv_go:
                    kv_d = kv_idx
                    if cutlass.const_expr(self.desc):
                        kv_d = seg_s - kv_idx
                    if kv_blk_ptr == 32:
                        kv_blk_ptr = cutlass.Int32(0)
                        prefetch_kv = kv_idx + lane_idx * NUM_MATH_WG
                        if cutlass.const_expr(self.desc):
                            prefetch_kv = kv_d - lane_idx * NUM_MATH_WG
                        base_phys = prefetch_kv * NUM_BLOCKS_PER_MMA
                        for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                            cached_blks[i] = mBlockTable[
                                q_idx, cutlass.min(cutlass.max(base_phys + i, 0), bt_cols - 1)
                            ]
                    tile_ok = kv_d + 0 < num_kv
                    phys_blks = [cutlass.Int32(0)] * NUM_BLOCKS_PER_MMA
                    for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                        pv = cute.arch.shuffle_sync(cached_blks[i], kv_blk_ptr)
                        phys_blks[i] = cutlass.select_(tile_ok, pv, cutlass.Int32(0))
                    kv_blk_ptr = kv_blk_ptr + 1
                    kv_pipeline_0.producer_acquire(kv_prod_state_0)
                    bar = kv_pipeline_0.producer_get_barrier(kv_prod_state_0)
                    stage = kv_prod_state_0.index
                    for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                        cute.copy(
                            tma_atom_a,
                            tAgA_0[None, 0, 0, phys_blks[i]],
                            tAsA_0[None, i, stage],
                            tma_bar_ptr=bar,
                            mcast_mask=a_mcast_mask,
                            cache_policy=pol,
                        )
                        cute.copy(
                            tma_atom_s,
                            tSgS_0[None, 0, phys_blks[i]],
                            tSsS_0[None, i, stage],
                            tma_bar_ptr=bar,
                            cache_policy=pol,
                        )
                    kv_prod_state_0.advance()
                    kv_idx = kv_idx + NUM_MATH_WG
                    if cutlass.const_expr(self.lim_loop2):
                        kv_go = kv_idx < kv_lim
                    else:
                        kv_go = (kv_idx < num_kv) & ((q_idx != end_q_idx) | (kv_idx != end_kv_idx))
                next_q_idx, next_kv_idx, next_num_kv, phase, has_work = self._walk_next(
                    q_idx,
                    kv_idx,
                    num_kv,
                    next_num_kv,
                    phase,
                    stop_kv,
                    start_q,
                    start_kv_half,
                    end_q_idx,
                    end_kv_idx,
                    mContextLens,
                    batch_size,
                )
        elif is_tma_warp_1:
            pol = None
            if cutlass.const_expr(self.policy != L.POL_NONE):
                pol = L.make_l2_policy(self.policy)
            end_kv_idx = end_kv_half * NUM_MATH_WG
            next_q_idx, next_kv_idx, next_num_kv, phase, stop_kv, has_work, rot_on = (
                self._walk_init(
                    start_q, start_kv_half, end_q_idx, end_kv_half, mContextLens, batch_size
                )
            )
            q_idx = batch_size
            lane_idx = tidx % 32
            cached_blks = [cutlass.Int32(0) for _ in range(NUM_BLOCKS_PER_MMA)]
            kv_blk_ptr = cutlass.Int32(32)
            bt_cols = cute.size(mBlockTable, mode=[1])
            seg_s = Int32(0)
            while has_work:
                q_idx_old = q_idx
                q_idx = next_q_idx
                kv_idx = next_kv_idx
                num_kv = next_num_kv
                if cutlass.const_expr(self.lim_loop2):
                    kv_lim = self._kv_limit(kv_idx, num_kv, q_idx, end_q_idx, end_kv_idx)
                if q_idx != q_idx_old:
                    if cutlass.const_expr(self.desc):
                        seg_s = self._seg_s(kv_idx, num_kv, q_idx, end_q_idx, end_kv_idx)
                    kv_blk_ptr = cutlass.Int32(32)
                # Keep row metadata and Q handoffs outside the per-tile loop.
                # One dummy pair is still issued for a zero-length row.
                kv_go = cutlass.Boolean(True)
                while kv_go:
                    kv_d = kv_idx
                    if cutlass.const_expr(self.desc):
                        kv_d = seg_s - kv_idx
                    if kv_blk_ptr == 32:
                        kv_blk_ptr = cutlass.Int32(0)
                        prefetch_kv = kv_idx + 1 + lane_idx * NUM_MATH_WG
                        if cutlass.const_expr(self.desc):
                            prefetch_kv = kv_d + 1 - lane_idx * NUM_MATH_WG
                        base_phys = prefetch_kv * NUM_BLOCKS_PER_MMA
                        for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                            cached_blks[i] = mBlockTable[
                                q_idx, cutlass.min(cutlass.max(base_phys + i, 0), bt_cols - 1)
                            ]
                    tile_ok = kv_d + 1 < num_kv
                    phys_blks = [cutlass.Int32(0)] * NUM_BLOCKS_PER_MMA
                    for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                        pv = cute.arch.shuffle_sync(cached_blks[i], kv_blk_ptr)
                        phys_blks[i] = cutlass.select_(tile_ok, pv, cutlass.Int32(0))
                    kv_blk_ptr = kv_blk_ptr + 1
                    kv_pipeline_1.producer_acquire(kv_prod_state_1)
                    bar = kv_pipeline_1.producer_get_barrier(kv_prod_state_1)
                    stage = kv_prod_state_1.index
                    for i in cutlass.range_constexpr(NUM_BLOCKS_PER_MMA):
                        cute.copy(
                            tma_atom_a,
                            tAgA_1[None, 0, 0, phys_blks[i]],
                            tAsA_1[None, i, stage],
                            tma_bar_ptr=bar,
                            mcast_mask=a_mcast_mask,
                            cache_policy=pol,
                        )
                        cute.copy(
                            tma_atom_s,
                            tSgS_1[None, 0, phys_blks[i]],
                            tSsS_1[None, i, stage],
                            tma_bar_ptr=bar,
                            cache_policy=pol,
                        )
                    kv_prod_state_1.advance()
                    kv_idx = kv_idx + NUM_MATH_WG
                    if cutlass.const_expr(self.lim_loop2):
                        kv_go = kv_idx < kv_lim
                    else:
                        kv_go = (kv_idx < num_kv) & ((q_idx != end_q_idx) | (kv_idx != end_kv_idx))
                next_q_idx, next_kv_idx, next_num_kv, phase, has_work = self._walk_next(
                    q_idx,
                    kv_idx,
                    num_kv,
                    next_num_kv,
                    phase,
                    stop_kv,
                    start_q,
                    start_kv_half,
                    end_q_idx,
                    end_kv_idx,
                    mContextLens,
                    batch_size,
                )
        elif is_umma_warp_0:
            end_kv_idx = end_kv_half * NUM_MATH_WG
            next_q_idx, next_kv_idx, next_num_kv, phase, stop_kv, has_work, rot_on = (
                self._walk_init(
                    start_q, start_kv_half, end_q_idx, end_kv_half, mContextLens, batch_size
                )
            )
            q_idx = batch_size
            tCtAcc_base_0 = cute.make_tensor(tmem_ptr, tCtAcc_fake_staged.layout)
            if cutlass.const_expr(self.fp4):
                sf_base = self.num_tmem_alloc_cols * self.num_groups * self.num_umma_stages
                tCtSFA = cute.make_tensor(
                    cute.recast_ptr(tmem_ptr + sf_base, dtype=self.sf_dtype), tmem_sfa_layout
                )
                tCtSFB = cute.make_tensor(
                    cute.recast_ptr(
                        tmem_ptr + sf_base + self.num_sfa_tmem_cols * self.num_groups,
                        dtype=self.sf_dtype,
                    ),
                    tmem_sfb_layout,
                )
            if is_leader_cta:
                num_k_blocks = cute.size(tCrA_0.shape[2])
                q_stage_0 = cutlass.Int32(0)
                while has_work:
                    q_idx_old = q_idx
                    q_idx = next_q_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv
                    if cutlass.const_expr(self.lim_loop2):
                        kv_lim = self._kv_limit(kv_idx, num_kv, q_idx, end_q_idx, end_kv_idx)
                    if q_idx != q_idx_old:
                        if q_idx_old < batch_size:
                            if cutlass.const_expr(self.fp4):
                                sfb_war_pipeline.producer_commit(sfb_war_prod_state)
                            q_cons_state_umma_0.advance()
                        q_pipeline.consumer_wait(q_cons_state_umma_0)
                        q_stage_0 = q_cons_state_umma_0.index
                        if cutlass.const_expr(self.fp4):
                            for atom_idx in cutlass.range_constexpr(self.N_padded // 128):
                                L.sf_atom_transpose(
                                    sSF_Q.iterator
                                    + q_stage_0 * sSF_Q.layout.stride[1]
                                    + atom_idx * 128
                                )
                            cute.arch.fence_view_async_shared()
                            if q_idx_old < batch_size:
                                sfb_war_pipeline.consumer_wait(sfb_war_cons_state)
                                sfb_war_cons_state.advance()
                            sSF_Q_ue8m0 = cute.recast_tensor(sSF_Q, Float8E8M0FNU)
                            sSF_Q_chunk = cute.make_tensor(
                                sSF_Q_ue8m0.iterator + q_stage_0 * sSF_Q_ue8m0.layout.stride[1],
                                sfb_chunk_smem_layout,
                            )
                            tCsSFB_compact = cute.filter_zeros(sSF_Q_chunk)
                            tCtSFB_compact = cute.filter_zeros(tCtSFB)
                            tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(
                                copy_atom_s2t, tCtSFB_compact
                            )
                            thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
                            tCsSFB_s2t_ = thr_copy_s2t_sfb.partition_S(tCsSFB_compact)
                            tCsSFB_s2t = tcgen05.get_s2t_smem_desc_tensor(
                                tiled_copy_s2t_sfb, tCsSFB_s2t_
                            )
                            tCtSFB_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)
                            cute.copy(tiled_copy_s2t_sfb, tCsSFB_s2t, tCtSFB_s2t)
                            cute.arch.fence_view_async_tmem_store()
                            sfb_sync_barrier.arrive_and_wait()
                    # Keep row metadata and Q handoffs outside the per-tile loop.
                    # One dummy pair is still issued for a zero-length row.
                    kv_go = cutlass.Boolean(True)
                    while kv_go:
                        kv_pipeline_0.consumer_wait(kv_cons_state_umma_0)
                        umma_pipeline_0.producer_acquire(umma_prod_state_0)
                        if cutlass.const_expr(self.tmem_fences):
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        kv_stage = kv_cons_state_umma_0.index
                        if cutlass.const_expr(self.fp4):
                            for atom_idx in cutlass.range_constexpr(self.block_kv // 128):
                                L.sf_atom_transpose(
                                    sScales_0.iterator
                                    + kv_stage * sScales_0.layout.stride[1]
                                    + atom_idx * 128
                                )
                            cute.arch.fence_view_async_shared()
                            sSF_KV_0_ue8m0 = cute.recast_tensor(sScales_0, Float8E8M0FNU)
                            sSF_KV_0_chunk = cute.make_tensor(
                                sSF_KV_0_ue8m0.iterator
                                + kv_stage * sSF_KV_0_ue8m0.layout.stride[1],
                                sfa_chunk_smem_layout,
                            )
                            tCsSFA_compact_0 = cute.filter_zeros(sSF_KV_0_chunk)
                            tCtSFA_compact_0 = cute.filter_zeros(tCtSFA)
                            tiled_copy_s2t_sfa_0 = tcgen05.make_s2t_copy(
                                copy_atom_s2t, tCtSFA_compact_0
                            )
                            thr_copy_s2t_sfa_0 = tiled_copy_s2t_sfa_0.get_slice(0)
                            tCsSFA_s2t_0_ = thr_copy_s2t_sfa_0.partition_S(tCsSFA_compact_0)
                            tCsSFA_s2t_0 = tcgen05.get_s2t_smem_desc_tensor(
                                tiled_copy_s2t_sfa_0, tCsSFA_s2t_0_
                            )
                            tCtSFA_s2t_0 = thr_copy_s2t_sfa_0.partition_D(tCtSFA_compact_0)
                            cute.copy(tiled_copy_s2t_sfa_0, tCsSFA_s2t_0, tCtSFA_s2t_0)
                        tCtAcc_0 = tCtAcc_base_0[None, None, None, umma_prod_state_0.index]
                        for k_block in cutlass.range_constexpr(num_k_blocks):
                            if cutlass.const_expr(self.fp4):
                                tCtSFA_k = cute.slice_(
                                    tCtSFA, (None, None, k_block * self.sf_k_step, None)
                                )
                                tCtSFB_k = cute.slice_(
                                    tCtSFB, (None, None, k_block * self.sf_k_step, None)
                                )
                                tiled_mma.set(tcgen05.Field.SFA, tCtSFA_k.iterator)
                                tiled_mma.set(tcgen05.Field.SFB, tCtSFB_k.iterator)
                            cute.gemm(
                                tiled_mma,
                                tCtAcc_0,
                                tCrA_0[None, None, k_block, kv_stage],
                                tCrB[None, None, k_block, q_stage_0],
                                tCtAcc_0,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        if cutlass.const_expr(self.fp4):
                            kv_pipeline_0.consumer_release(kv_cons_state_umma_0)
                        kv_cons_state_umma_0.advance()
                        umma_pipeline_0.producer_commit(umma_prod_state_0)
                        umma_prod_state_0.advance()
                        if cutlass.const_expr(self.fp4 and not self.nolock):
                            sfb_sync_barrier.arrive_and_wait()
                        kv_idx = kv_idx + NUM_MATH_WG
                        if cutlass.const_expr(self.lim_loop2):
                            kv_go = kv_idx < kv_lim
                        else:
                            kv_go = (kv_idx < num_kv) & (
                                (q_idx != end_q_idx) | (kv_idx != end_kv_idx)
                            )
                    next_q_idx, next_kv_idx, next_num_kv, phase, has_work = self._walk_next(
                        q_idx,
                        kv_idx,
                        num_kv,
                        next_num_kv,
                        phase,
                        stop_kv,
                        start_q,
                        start_kv_half,
                        end_q_idx,
                        end_kv_idx,
                        mContextLens,
                        batch_size,
                    )
        elif is_umma_warp_1:
            end_kv_idx = end_kv_half * NUM_MATH_WG
            next_q_idx, next_kv_idx, next_num_kv, phase, stop_kv, has_work, rot_on = (
                self._walk_init(
                    start_q, start_kv_half, end_q_idx, end_kv_half, mContextLens, batch_size
                )
            )
            q_idx = batch_size
            tCtAcc_base_1 = cute.make_tensor(tmem_ptr + cols_per_group, tCtAcc_fake_staged.layout)
            if cutlass.const_expr(self.fp4):
                sf_base = self.num_tmem_alloc_cols * self.num_groups * self.num_umma_stages
                tCtSFA = cute.make_tensor(
                    cute.recast_ptr(
                        tmem_ptr + sf_base + self.num_sfa_tmem_cols, dtype=self.sf_dtype
                    ),
                    tmem_sfa_layout,
                )
                tCtSFB = cute.make_tensor(
                    cute.recast_ptr(
                        tmem_ptr + sf_base + self.num_sfa_tmem_cols * self.num_groups,
                        dtype=self.sf_dtype,
                    ),
                    tmem_sfb_layout,
                )
            if is_leader_cta:
                num_k_blocks_1 = cute.size(tCrA_1.shape[2])
                q_stage_1 = cutlass.Int32(0)
                while has_work:
                    q_idx_old = q_idx
                    q_idx = next_q_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv
                    if cutlass.const_expr(self.lim_loop2):
                        kv_lim = self._kv_limit(kv_idx, num_kv, q_idx, end_q_idx, end_kv_idx)
                    if q_idx != q_idx_old:
                        if q_idx_old < batch_size:
                            if cutlass.const_expr(self.fp4):
                                sfb_war_pipeline.producer_commit(sfb_war_prod_state)
                            q_cons_state_umma_1.advance()
                        q_pipeline.consumer_wait(q_cons_state_umma_1)
                        q_stage_1 = q_cons_state_umma_1.index
                        if cutlass.const_expr(self.fp4):
                            sfb_sync_barrier.arrive_and_wait()
                    # Keep row metadata and Q handoffs outside the per-tile loop.
                    # One dummy pair is still issued for a zero-length row.
                    kv_go = cutlass.Boolean(True)
                    while kv_go:
                        kv_pipeline_1.consumer_wait(kv_cons_state_umma_1)
                        umma_pipeline_1.producer_acquire(umma_prod_state_1)
                        if cutlass.const_expr(self.tmem_fences):
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        kv_stage_1 = kv_cons_state_umma_1.index
                        if cutlass.const_expr(self.fp4):
                            for atom_idx in cutlass.range_constexpr(self.block_kv // 128):
                                L.sf_atom_transpose(
                                    sScales_1.iterator
                                    + kv_stage_1 * sScales_1.layout.stride[1]
                                    + atom_idx * 128
                                )
                            cute.arch.fence_view_async_shared()
                            sSF_KV_1_ue8m0 = cute.recast_tensor(sScales_1, Float8E8M0FNU)
                            sSF_KV_1_chunk = cute.make_tensor(
                                sSF_KV_1_ue8m0.iterator
                                + kv_stage_1 * sSF_KV_1_ue8m0.layout.stride[1],
                                sfa_chunk_smem_layout,
                            )
                            tCsSFA_compact_1 = cute.filter_zeros(sSF_KV_1_chunk)
                            tCtSFA_compact_1 = cute.filter_zeros(tCtSFA)
                            tiled_copy_s2t_sfa_1 = tcgen05.make_s2t_copy(
                                copy_atom_s2t, tCtSFA_compact_1
                            )
                            thr_copy_s2t_sfa_1 = tiled_copy_s2t_sfa_1.get_slice(0)
                            tCsSFA_s2t_1_ = thr_copy_s2t_sfa_1.partition_S(tCsSFA_compact_1)
                            tCsSFA_s2t_1 = tcgen05.get_s2t_smem_desc_tensor(
                                tiled_copy_s2t_sfa_1, tCsSFA_s2t_1_
                            )
                            tCtSFA_s2t_1 = thr_copy_s2t_sfa_1.partition_D(tCtSFA_compact_1)
                            cute.copy(tiled_copy_s2t_sfa_1, tCsSFA_s2t_1, tCtSFA_s2t_1)
                        tCtAcc_1 = tCtAcc_base_1[None, None, None, umma_prod_state_1.index]
                        for k_block in cutlass.range_constexpr(num_k_blocks_1):
                            if cutlass.const_expr(self.fp4):
                                tCtSFA_k = cute.slice_(
                                    tCtSFA, (None, None, k_block * self.sf_k_step, None)
                                )
                                tCtSFB_k = cute.slice_(
                                    tCtSFB, (None, None, k_block * self.sf_k_step, None)
                                )
                                tiled_mma.set(tcgen05.Field.SFA, tCtSFA_k.iterator)
                                tiled_mma.set(tcgen05.Field.SFB, tCtSFB_k.iterator)
                            cute.gemm(
                                tiled_mma,
                                tCtAcc_1,
                                tCrA_1[None, None, k_block, kv_stage_1],
                                tCrB[None, None, k_block, q_stage_1],
                                tCtAcc_1,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        if cutlass.const_expr(self.fp4):
                            kv_pipeline_1.consumer_release(kv_cons_state_umma_1)
                        kv_cons_state_umma_1.advance()
                        umma_pipeline_1.producer_commit(umma_prod_state_1)
                        umma_prod_state_1.advance()
                        if cutlass.const_expr(self.fp4 and not self.nolock):
                            sfb_sync_barrier.arrive_and_wait()
                        kv_idx = kv_idx + NUM_MATH_WG
                        if cutlass.const_expr(self.lim_loop2):
                            kv_go = kv_idx < kv_lim
                        else:
                            kv_go = (kv_idx < num_kv) & (
                                (q_idx != end_q_idx) | (kv_idx != end_kv_idx)
                            )
                    next_q_idx, next_kv_idx, next_num_kv, phase, has_work = self._walk_next(
                        q_idx,
                        kv_idx,
                        num_kv,
                        next_num_kv,
                        phase,
                        stop_kv,
                        start_q,
                        start_kv_half,
                        end_q_idx,
                        end_kv_idx,
                        mContextLens,
                        batch_size,
                    )
        elif is_math_warp:
            cute.arch.warpgroup_reg_alloc(self.math_regs)
            end_kv_idx = end_kv_half * NUM_MATH_WG
            next_q_idx, next_kv_idx, next_num_kv, phase, stop_kv, has_work, rot_on = (
                self._walk_init(
                    start_q, start_kv_half, end_q_idx, end_kv_half, mContextLens, batch_size
                )
            )
            q_idx = batch_size
            if cutlass.const_expr(self.attn):
                if tidx == 0:
                    sHI[9] = cute.arch.load(
                        storage.tmem_holding_buf.ptr, Int32, sem="relaxed", scope="cta"
                    )
                    prims.mbarrier_arrive(sRowBar.iterator.raw_ptr() + Int32(5))
            local_tidx = tidx % 128
            m_lane = tidx % 32
            cC = cute.make_identity_tensor(epi_sub_mn)
            NUM_W_IN_REG = self.num_w_in_reg
            if cutlass.const_expr(self.debug_dump):
                logits_stride0 = cutlass.Int64(mLogits.layout.stride[0])
                mLogits_flat = cute.make_tensor(
                    mLogits.iterator,
                    cute.make_layout(cute.size(mLogits, mode=[0]) * logits_stride0),
                )
                out_row_off = cutlass.Int64(0)
            grp = warpgroup_idx
            tCtAcc_base_g = cute.make_tensor(
                tmem_ptr + grp * cols_per_group, tCtAcc_fake_staged.layout
            )
            tAcc_ref = tCtAcc_base_g[None, None, None, 0][(None, None), 0, 0]
            tAcc_ref_epi = cute.flat_divide(tAcc_ref, epi_sub_mn)
            tiled_copy_ref = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_ref_epi[None, None, 0, 0])
            thr_copy_ref = tiled_copy_ref.get_slice(local_tidx)
            tTR_cC = thr_copy_ref.partition_D(cC)
            m_coord = tTR_cC[0][0]
            tTR_rAcc = cute.make_fragment_like(tTR_cC, self.acc_dtype)
            w_cache = cute.make_rmem_tensor((max(NUM_W_IN_REG, 1),), self.epi_dtype)
            q_stage_local = cutlass.Int32(0)
            ctx_q = cutlass.Int32(0)
            lt_mask = cutlass.Int32(cute.arch.lanemask_lt())
            ring_off = warp_idx * Int32(RING)
            head = Int32(0)
            seg_ord = Int32(0)
            tw_ptr = sTw.iterator + warp_idx
            full_ptr = sRingBar.iterator.raw_ptr() + warp_idx * Int32(RING // RING_CHUNK)
            empty_ptr = full_ptr + Int32(8 * (RING // RING_CHUNK))
            T = Float32(NEG_INF)
            stall_ns = Int64(0)
            skew_seed = Int32(0)
            if cutlass.const_expr(self.skew_ns > 0):
                skew_seed = Int32(
                    Uint32(cutlass.Int64(cute.arch.globaltimer()) >> Int64(5))
                ) + bidz * Int32(7919)
            seg_s = Int32(0)
            tcount = Int32(0)
            if warpgroup_idx == 0:
                while has_work:
                    q_idx_old = q_idx
                    q_idx = next_q_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv
                    if cutlass.const_expr(self.desc):
                        seg_s = self._seg_s(kv_idx, num_kv, q_idx, end_q_idx, end_kv_idx)
                    if cutlass.const_expr(self.lim_loop):
                        kv_lim = self._kv_limit(kv_idx, num_kv, q_idx, end_q_idx, end_kv_idx)
                    if q_idx != q_idx_old:
                        if q_idx_old < batch_size:
                            if cutlass.const_expr(self.pfence):
                                cute.arch.fence_view_async_shared()
                            q_pipeline.consumer_release(q_cons_state)
                            q_cons_state.advance()
                        q_pipeline.consumer_wait(q_cons_state)
                        q_stage_local = q_cons_state.index
                        if cutlass.const_expr(self.debug_dump):
                            out_row_off = cutlass.Int64(q_idx) * logits_stride0
                        for w_j in cutlass.range_constexpr(NUM_W_IN_REG):
                            w_cache[w_j] = sW[w_j, q_stage_local]
                        ctx_q = self._ctx_of(q_idx, mContextLens, batch_size)
                        # A saved floor is valid only for this row segment.
                        T = Float32(NEG_INF)
                    kv_go = cutlass.Boolean(True)
                    while kv_go:
                        kv_pos = kv_idx * block_kv_val + m_coord
                        if cutlass.const_expr(self.desc):
                            kv_pos = (seg_s - kv_idx) * block_kv_val + m_coord
                        if cutlass.const_expr(not self.fp4):
                            kv_ready_0 = kv_pipeline_0.consumer_try_wait(kv_cons_state_math_0)
                        umma_ready_0 = umma_pipeline_0.consumer_try_wait(umma_cons_state_0)
                        if cutlass.const_expr(not self.fp4):
                            kv_pipeline_0.consumer_wait(kv_cons_state_math_0, kv_ready_0)
                            scale_val = sScales_0[m_coord, kv_cons_state_math_0.index]
                        umma_pipeline_0.consumer_wait(umma_cons_state_0, umma_ready_0)
                        if cutlass.const_expr(self.tmem_fences):
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        tCtAcc_c0 = tCtAcc_base_g[None, None, None, umma_cons_state_0.index]
                        tAcc_c0 = tCtAcc_c0[(None, None), 0, 0]
                        tAcc_c0_epi = cute.flat_divide(tAcc_c0, epi_sub_mn)
                        tTR_0 = thr_copy_ref.partition_S(tAcc_c0_epi)
                        if cutlass.const_expr(not self.late_release and not self.fp4):
                            scale_val = _consume_f32(scale_val)
                            if cutlass.const_expr(self.pfence):
                                cute.arch.fence_view_async_shared()
                            cute.arch.sync_warp()
                            kv_pipeline_0.consumer_release(kv_cons_state_math_0)
                            kv_cons_state_math_0.advance()
                        cute.copy(tiled_copy_ref, tTR_0[None, None, None, 0, 0], tTR_rAcc)
                        cute.arch.fence_view_async_tmem_load()
                        s0x, s0y, s1x, s1y = self._epi_half(
                            tTR_rAcc,
                            w_cache,
                            0,
                            Float32(0.0),
                            Float32(0.0),
                            Float32(0.0),
                            Float32(0.0),
                        )
                        cute.copy(tiled_copy_ref, tTR_0[None, None, None, 0, 1], tTR_rAcc)
                        cute.arch.fence_view_async_tmem_load()
                        if cutlass.const_expr(self.tmem_fences):
                            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        umma_pipeline_0.consumer_release(umma_cons_state_0)
                        umma_cons_state_0.advance()
                        s0x, s0y, s1x, s1y = self._epi_half(
                            tTR_rAcc, w_cache, 32, s0x, s0y, s1x, s1y
                        )
                        result_t = s0x + s0y + s1x + s1y
                        if cutlass.const_expr(self.fp4):
                            # the block-scaled MMA applies the scales; result_t is twice the sum (relu as x + |x|)
                            val = self.output_dtype(result_t * 0.5)
                        else:
                            scale_mul = scale_val * 0.5
                            val = self.output_dtype(result_t * scale_mul)
                        if cutlass.const_expr(self.late_release and not self.fp4):
                            if cutlass.const_expr(self.pfence):
                                cute.arch.fence_view_async_shared()
                            cute.arch.sync_warp()
                            kv_pipeline_0.consumer_release(kv_cons_state_math_0)
                            kv_cons_state_math_0.advance()
                        if cutlass.const_expr(self.debug_dump):
                            mLogits_flat[out_row_off + kv_pos] = val
                        if cutlass.const_expr(self.admit):
                            # Stale same-segment floors remain conservative.
                            if (kv_idx & Int32(self.floor_mask)) == Int32(0):
                                T = _floor_of(tw_ptr, seg_ord, m_lane)
                            hit = (kv_pos < ctx_q) & (val >= T)
                            ent = L.pack_entry(L.u32_of_f32(val), kv_pos)
                            head, stall_ns = _push_ring(
                                hit,
                                ent,
                                head,
                                s_ring,
                                ring_off,
                                empty_ptr,
                                full_ptr,
                                m_lane,
                                lt_mask,
                                stall_ns,
                                self.stamps,
                            )
                        kv_idx = kv_idx + NUM_MATH_WG
                        if cutlass.const_expr(self.stamps):
                            tcount = tcount + Int32(1)
                            if tcount & Int32(63) == Int32(0):
                                if warp_idx == Int32(0):
                                    if m_lane == Int32(0):
                                        tslot = tcount >> Int32(6)
                                        if tslot < Int32(94):
                                            mStamps[bidz * 256 + 162 + tslot] = cutlass.Int64(
                                                cute.arch.globaltimer()
                                            )
                        if cutlass.const_expr(self.lim_loop):
                            kv_go = kv_idx < kv_lim
                        else:
                            kv_go = (kv_idx < num_kv) & (
                                (q_idx != end_q_idx) | (kv_idx != end_kv_idx)
                            )
                    if cutlass.const_expr(self.admit):
                        if cutlass.const_expr(self.skew_ns > 0):
                            L.skew_sleep(
                                skew_seed + warp_idx,
                                seg_ord * Int32(8) + Int32(1),
                                Int32(self.skew_ns),
                            )
                        head = _push_marker(
                            seg_ord, head, s_ring, ring_off, empty_ptr, full_ptr, m_lane
                        )
                    if cutlass.const_expr(self.stamps):
                        if m_lane == Int32(0):
                            if seg_ord < Int32(8):
                                mStamps[bidz * 256 + 32 + warp_idx * 8 + seg_ord] = stall_ns
                                mStamps[bidz * 256 + 96 + seg_ord] = cutlass.Int64(
                                    cute.arch.globaltimer()
                                )
                    seg_ord = seg_ord + 1
                    next_q_idx, next_kv_idx, next_num_kv, phase, has_work = self._walk_next(
                        q_idx,
                        kv_idx,
                        num_kv,
                        next_num_kv,
                        phase,
                        stop_kv,
                        start_q,
                        start_kv_half,
                        end_q_idx,
                        end_kv_idx,
                        mContextLens,
                        batch_size,
                    )
                if q_idx < batch_size:
                    if cutlass.const_expr(self.pfence):
                        cute.arch.fence_view_async_shared()
                    q_pipeline.consumer_release(q_cons_state)
                    q_cons_state.advance()
            else:
                while has_work:
                    q_idx_old = q_idx
                    q_idx = next_q_idx
                    kv_idx = next_kv_idx
                    num_kv = next_num_kv
                    if cutlass.const_expr(self.desc):
                        seg_s = self._seg_s(kv_idx, num_kv, q_idx, end_q_idx, end_kv_idx)
                    if cutlass.const_expr(self.lim_loop):
                        kv_lim = self._kv_limit(kv_idx, num_kv, q_idx, end_q_idx, end_kv_idx)
                    if q_idx != q_idx_old:
                        if q_idx_old < batch_size:
                            if cutlass.const_expr(self.pfence):
                                cute.arch.fence_view_async_shared()
                            q_pipeline.consumer_release(q_cons_state)
                            q_cons_state.advance()
                        q_pipeline.consumer_wait(q_cons_state)
                        q_stage_local = q_cons_state.index
                        if cutlass.const_expr(self.debug_dump):
                            out_row_off = cutlass.Int64(q_idx) * logits_stride0
                        for w_j in cutlass.range_constexpr(NUM_W_IN_REG):
                            w_cache[w_j] = sW[w_j, q_stage_local]
                        ctx_q = self._ctx_of(q_idx, mContextLens, batch_size)
                        # A saved floor is valid only for this row segment.
                        T = Float32(NEG_INF)
                    kv_go = cutlass.Boolean(True)
                    while kv_go:
                        kv_idx_1 = kv_idx + 1
                        kv_pos = kv_idx_1 * block_kv_val + m_coord
                        if cutlass.const_expr(self.desc):
                            kv_pos = (seg_s - kv_idx + 1) * block_kv_val + m_coord
                        if cutlass.const_expr(not self.fp4):
                            kv_ready_1 = kv_pipeline_1.consumer_try_wait(kv_cons_state_math_1)
                        umma_ready_1 = umma_pipeline_1.consumer_try_wait(umma_cons_state_1)
                        if cutlass.const_expr(not self.fp4):
                            kv_pipeline_1.consumer_wait(kv_cons_state_math_1, kv_ready_1)
                            scale_val = sScales_1[m_coord, kv_cons_state_math_1.index]
                        umma_pipeline_1.consumer_wait(umma_cons_state_1, umma_ready_1)
                        if cutlass.const_expr(self.tmem_fences):
                            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
                        tCtAcc_c1 = tCtAcc_base_g[None, None, None, umma_cons_state_1.index]
                        tAcc_c1 = tCtAcc_c1[(None, None), 0, 0]
                        tAcc_c1_epi = cute.flat_divide(tAcc_c1, epi_sub_mn)
                        tTR_1 = thr_copy_ref.partition_S(tAcc_c1_epi)
                        if cutlass.const_expr(not self.late_release and not self.fp4):
                            scale_val = _consume_f32(scale_val)
                            if cutlass.const_expr(self.pfence):
                                cute.arch.fence_view_async_shared()
                            cute.arch.sync_warp()
                            kv_pipeline_1.consumer_release(kv_cons_state_math_1)
                            kv_cons_state_math_1.advance()
                        cute.copy(tiled_copy_ref, tTR_1[None, None, None, 0, 0], tTR_rAcc)
                        cute.arch.fence_view_async_tmem_load()
                        s0x, s0y, s1x, s1y = self._epi_half(
                            tTR_rAcc,
                            w_cache,
                            0,
                            Float32(0.0),
                            Float32(0.0),
                            Float32(0.0),
                            Float32(0.0),
                        )
                        cute.copy(tiled_copy_ref, tTR_1[None, None, None, 0, 1], tTR_rAcc)
                        cute.arch.fence_view_async_tmem_load()
                        if cutlass.const_expr(self.tmem_fences):
                            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                        umma_pipeline_1.consumer_release(umma_cons_state_1)
                        umma_cons_state_1.advance()
                        s0x, s0y, s1x, s1y = self._epi_half(
                            tTR_rAcc, w_cache, 32, s0x, s0y, s1x, s1y
                        )
                        result_t = s0x + s0y + s1x + s1y
                        if cutlass.const_expr(self.fp4):
                            # the block-scaled MMA applies the scales; result_t is twice the sum (relu as x + |x|)
                            val = self.output_dtype(result_t * 0.5)
                        else:
                            scale_mul = scale_val * 0.5
                            val = self.output_dtype(result_t * scale_mul)
                        if cutlass.const_expr(self.late_release and not self.fp4):
                            if cutlass.const_expr(self.pfence):
                                cute.arch.fence_view_async_shared()
                            cute.arch.sync_warp()
                            kv_pipeline_1.consumer_release(kv_cons_state_math_1)
                            kv_cons_state_math_1.advance()
                        if cutlass.const_expr(self.debug_dump):
                            mLogits_flat[out_row_off + kv_pos] = val
                        if cutlass.const_expr(self.admit):
                            # Stale same-segment floors remain conservative.
                            if (kv_idx & Int32(self.floor_mask)) == Int32(0):
                                T = _floor_of(tw_ptr, seg_ord, m_lane)
                            hit = (kv_pos < ctx_q) & (val >= T)
                            ent = L.pack_entry(L.u32_of_f32(val), kv_pos)
                            head, stall_ns = _push_ring(
                                hit,
                                ent,
                                head,
                                s_ring,
                                ring_off,
                                empty_ptr,
                                full_ptr,
                                m_lane,
                                lt_mask,
                                stall_ns,
                                self.stamps,
                            )
                        kv_idx = kv_idx + NUM_MATH_WG
                        if cutlass.const_expr(self.stamps):
                            tcount = tcount + Int32(1)
                            if tcount & Int32(63) == Int32(0):
                                if warp_idx == Int32(0):
                                    if m_lane == Int32(0):
                                        tslot = tcount >> Int32(6)
                                        if tslot < Int32(94):
                                            mStamps[bidz * 256 + 162 + tslot] = cutlass.Int64(
                                                cute.arch.globaltimer()
                                            )
                        if cutlass.const_expr(self.lim_loop):
                            kv_go = kv_idx < kv_lim
                        else:
                            kv_go = (kv_idx < num_kv) & (
                                (q_idx != end_q_idx) | (kv_idx != end_kv_idx)
                            )
                    if cutlass.const_expr(self.admit):
                        if cutlass.const_expr(self.skew_ns > 0):
                            L.skew_sleep(
                                skew_seed + warp_idx,
                                seg_ord * Int32(8) + Int32(1),
                                Int32(self.skew_ns),
                            )
                        head = _push_marker(
                            seg_ord, head, s_ring, ring_off, empty_ptr, full_ptr, m_lane
                        )
                    if cutlass.const_expr(self.stamps):
                        if m_lane == Int32(0):
                            if seg_ord < Int32(8):
                                mStamps[bidz * 256 + 32 + warp_idx * 8 + seg_ord] = stall_ns
                                mStamps[bidz * 256 + 96 + seg_ord] = cutlass.Int64(
                                    cute.arch.globaltimer()
                                )
                    seg_ord = seg_ord + 1
                    next_q_idx, next_kv_idx, next_num_kv, phase, has_work = self._walk_next(
                        q_idx,
                        kv_idx,
                        num_kv,
                        next_num_kv,
                        phase,
                        stop_kv,
                        start_q,
                        start_kv_half,
                        end_q_idx,
                        end_kv_idx,
                        mContextLens,
                        batch_size,
                    )
                if q_idx < batch_size:
                    if cutlass.const_expr(self.pfence):
                        cute.arch.fence_view_async_shared()
                    q_pipeline.consumer_release(q_cons_state)
                    q_cons_state.advance()
            if cutlass.const_expr(self.stamps):
                if tidx == 0:
                    mStamps[bidz * 256 + 1] = cutlass.Int64(cute.arch.globaltimer())
                if m_lane == Int32(0):
                    mStamps[bidz * 256 + 4 + warp_idx] = stall_ns
            cute.arch.fence_view_async_shared()
            cute.arch.fence_view_async_tmem_load()
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            cute.arch.barrier(barrier_id=3, number_of_threads=self.num_math_threads)
            if cutlass.const_expr(self.attn):
                if tidx == 0:
                    cute.arch.fence_acq_rel_cta()
                    cute.arch.atomic_exch(
                        sHI.iterator + Int32(11), Int32(1), sem="release", scope="cta"
                    )
                cute.arch.barrier(barrier_id=END_BAR, number_of_threads=self.num_math_threads + 128)
                prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)
        elif is_sel_warp:
            cute.arch.warpgroup_reg_dealloc(self.sel_regs)
            if cutlass.const_expr(self.admit):
                self._select_role(
                    tidx - 384,
                    bidz,
                    start_q,
                    start_kv_half,
                    end_q_idx,
                    end_kv_half,
                    batch_size,
                    mContextLens,
                    mScheduleMeta,
                    mRowStat,
                    mStamps,
                    s_ring,
                    sMain,
                    sOut,
                    sHist,
                    sCk,
                    sHead,
                    sTail,
                    sTw,
                    sCnt,
                    sRes,
                    sRed,
                    sWc,
                    sScal,
                    sKr,
                    sHI,
                    sHQ,
                    sHand,
                    sAttB,
                    sAttO,
                    sGv,
                    mKvLens,
                    mReq,
                    mBtSwa,
                    mBtC,
                    mTopk,
                    mGidx,
                    mWsSeg,
                    mWsCtr,
                    sKV_0,
                    sQ,
                    sQatt,
                    sKVatt,
                    sAbar,
                    tma_q,
                    tma_swa,
                    tma_cmp,
                    sTb,
                    sRowBar,
                    sRingBar,
                    sCov,
                    sHand2,
                )
        elif is_row_warp:
            if cutlass.const_expr(self.row_regs >= 96):
                cute.arch.warpgroup_reg_alloc(self.row_regs)
            else:
                cute.arch.warpgroup_reg_dealloc(self.row_regs)
            if cutlass.const_expr(self.admit):
                self._row_role(
                    tidx - 512,
                    bidz,
                    mKvLens,
                    mReq,
                    mBtSwa,
                    mBtC,
                    mTopk,
                    mGidx,
                    mWsSeg,
                    mWsCtr,
                    mRowStat,
                    mStamps,
                    sHand,
                    sAttB,
                    sAttO,
                    sHI,
                    sRS,
                    tma_q,
                    tma_swa,
                    tma_cmp,
                    mSinks,
                    mOut,
                    softmax_scale,
                    sQatt,
                    sKVatt,
                    sAbar,
                    sMsk,
                    sKV_0,
                    sQ,
                    start_q,
                    batch_size,
                    mScheduleMeta,
                    sMain,
                    sHQ,
                    sRres,
                    sRred,
                    sRwc,
                    s_ring,
                    sGv,
                    self._nseg(start_q, start_kv_half, end_q_idx, end_kv_half),
                    sDQ,
                    refused,
                    mStatus,
                    sRowBar,
                )
            if cutlass.const_expr(self.attn):
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                cute.arch.barrier_arrive(
                    barrier_id=END_BAR, number_of_threads=self.num_math_threads + 128, aligned=False
                )

    @cute.jit
    def _select_role(
        self,
        sel_tid,
        cta,
        start_q,
        start_kv_half,
        end_q_idx,
        end_kv_half,
        batch_size,
        mContextLens,
        mSched,
        mRowStat,
        mStamps,
        s_ring,
        sMain,
        sOut,
        sHist,
        sCk,
        sHead,
        sTail,
        sTw,
        sCnt,
        sRes,
        sRed,
        sWc,
        sScal,
        sKr,
        sHI,
        sHQ,
        sHand,
        sAttB,
        sAttO,
        sGv,
        mKvLens,
        mReq,
        mBtSwa,
        mBtC,
        mTopk,
        mGidx,
        mWsSeg,
        mWsCtr,
        sKV_0,
        sQ,
        sQatt,
        sKVatt,
        sAbar,
        tma_q,
        tma_swa,
        tma_cmp,
        sTb,
        sRowBar,
        sRingBar,
        sCov=None,
        sHand2=None,
    ):
        NT: cutlass.Constexpr[int] = SEL_NTHR
        lane = sel_tid & Int32(31)
        swarp = sel_tid >> Int32(5)
        lt_mask = Int32(cute.arch.lanemask_lt())
        r0 = swarp * Int32(2)
        r1 = r0 + Int32(1)
        empty = (start_q == end_q_idx) & (start_kv_half == end_kv_half)
        qb = end_q_idx
        if end_kv_half == Int32(0):
            qb = end_q_idx - Int32(1)
        nseg = qb - start_q + Int32(1)
        if empty:
            nseg = Int32(0)
        tail0 = Int32(0)
        tail1 = Int32(0)
        skew_seed = Int32(0)
        if cutlass.const_expr(self.skew_ns > 0):
            skew_seed = (
                Int32(Uint32(cutlass.Int64(cute.arch.globaltimer()) >> Int64(5)))
                + cta * Int32(7919)
                + Int32(13)
            )
        t_comp = Int64(0)
        prof_t = None
        if cutlass.const_expr(self.stamps):
            prof_t = cute.make_tensor(
                mStamps.iterator + cta * Int32(256) + Int32(16), cute.make_layout(4)
            )
        rot_on = Int32(0)
        if cutlass.const_expr(self.rot):
            if (end_kv_half > Int32(0)) & (end_q_idx > start_q):
                rot_on = Int32(1)
        seg = Int32(0)
        while seg < nseg:
            q = start_q + seg
            if rot_on != Int32(0):
                q = start_q + seg - Int32(1)
                if seg == Int32(0):
                    q = qb
            n = Int32(mContextLens[q])
            bsel = seg & Int32(1)
            sMb = cute.make_tensor(
                sMain.iterator + bsel * Int32(MAIN_CAP), cute.make_layout(MAIN_CAP)
            )
            if swarp == Int32(0):
                Pq = Int32(1)
                pq_ = Int32(0)
                if (q == start_q) | (q == qb):
                    if cutlass.const_expr(self.merge1 > 0):
                        Pq, pq_ = _parts_cov(q, cta, mSched, Int32(self.num_sms), lane, sCov)
                    else:
                        Pq, pq_ = _parts_of_row(q, cta, mSched, Int32(self.num_sms), lane)
                if lane == Int32(0):
                    sScal[0] = Pq
                    sScal[1] = pq_
            # Empty barriers start at phase 0; generation 0 waits against 1.
            _wait(
                sRowBar.iterator.raw_ptr() + Int32(2) + bsel,
                ((seg >> Int32(1)) & Int32(1)) ^ Int32(1),
            )
            if sel_tid == Int32(0):
                sCnt[0] = Int32(0)
                sKr[0] = Uint32(4294967295)
                sKr[1] = Uint32(0)
                if cutlass.const_expr(self.defer):
                    t_lo = Int32(0)
                    if q == start_q:
                        t_lo = start_kv_half * Int32(2)
                    t_hi = (n + Int32(self.block_kv - 1)) // Int32(self.block_kv) + Int32(
                        1
                    ) & Int32(-2)
                    if q == end_q_idx:
                        t_hi = end_kv_half * Int32(2)
                    cute.arch.atomic_exch(
                        sHI.iterator + Int32(5), t_hi - t_lo, sem="release", scope="cta"
                    )
            if sel_tid < Int32(8):
                cute.arch.atomic_exch(
                    sTw.iterator + sel_tid, Uint32(seg), sem="relaxed", scope="cta"
                )
            L.bar(SEL_BAR, NT)
            P = sScal[0]
            p = sScal[1]
            kth_key = Uint64(0)
            done0 = Int32(0)
            done1 = Int32(0)
            adm = Int32(0)
            ncomp = Int32(0)
            alldone = Int32(0)
            kf_done = Int32(0)
            while alldone == Int32(0):
                cnow = cute.arch.load(sCnt.iterator, Int32, sem="relaxed", scope="cta")
                mydone = done0 & done1
                trg = Int32(self.trig)
                if cutlass.const_expr(self.trig0 > 0):
                    if ncomp == Int32(0):
                        trg = Int32(self.trig0)
                lite = Int32(0)
                if cutlass.const_expr(self.kfloor):
                    if (kf_done == Int32(0)) & (ncomp == Int32(0)) & (cnow >= Int32(K_TOP)):
                        lite = Int32(1)
                if (cnow >= trg) | (mydone != Int32(0)) | (lite != Int32(0)):
                    if lane == Int32(0):
                        sScal[4 + swarp] = done0 + done1
                    L.bar(SEL_BAR, NT)
                    c = sCnt[0]
                    nd = sScal[4] + sScal[5] + sScal[6] + sScal[7]
                    L.bar(SEL_BAR, NT)
                    if c >= trg:
                        if cutlass.const_expr(self.skew_ns > 0):
                            if sel_tid == Int32(0):
                                L.skew_sleep(
                                    skew_seed, ncomp * Int32(16) + seg, Int32(self.skew_ns)
                                )
                        t_c0 = Int64(0)
                        if cutlass.const_expr(self.stamps):
                            t_c0 = cutlass.Int64(cute.arch.globaltimer())
                        tw_e = None
                        if cutlass.const_expr(self.efl):
                            tw_e = sTw.iterator
                        kept, tf = _compact_approx(
                            sMb,
                            c,
                            sHist,
                            sRes,
                            sCnt.iterator,
                            sKr.iterator,
                            sKr.iterator + 1,
                            sKr.iterator + 2,
                            sScal,
                            sel_tid,
                            NT,
                            SEL_BAR,
                            prof_t,
                            min(self.keep_cap, self.trig - 1),
                            self.cbins,
                            tw_e,
                            seg,
                            self.tiebin,
                            sTb,
                            fast_cmp=self.cmp_fast,
                        )
                        if kept < Int32(0):
                            no_, kth, m_, inf_ = L.select_topk_list(
                                sMb,
                                c,
                                Int32(K_TOP),
                                sOut,
                                sHist,
                                sCk,
                                sRes,
                                sRed,
                                sWc,
                                sel_tid,
                                False,
                                NT,
                                SEL_BAR,
                                True,
                            )
                            ic = sel_tid
                            while ic < Int32(K_TOP):
                                sMb[ic] = Uint64(sOut[ic])
                                ic = ic + Int32(NT)
                            tf = Uint32(kth >> Uint64(32))
                            L.bar(SEL_BAR, NT)
                            if sel_tid == Int32(0):
                                sCnt[0] = Int32(K_TOP)
                                sKr[0] = tf
                            L.bar(SEL_BAR, NT)
                        kth_key = Uint64(tf) << Uint64(32)
                        if sel_tid < Int32(8):
                            cute.arch.atomic_exch(
                                sTw.iterator + sel_tid,
                                _pack_floor_key(tf, seg),
                                sem="relaxed",
                                scope="cta",
                            )
                        ncomp = ncomp + Int32(1)
                        if cutlass.const_expr(self.stamps):
                            t_comp = t_comp + (cutlass.Int64(cute.arch.globaltimer()) - t_c0)
                    elif nd == Int32(8):
                        alldone = Int32(1)
                    if cutlass.const_expr(self.kfloor):
                        if (
                            (kf_done == Int32(0))
                            & (ncomp == Int32(0))
                            & (alldone == Int32(0))
                            & (c >= Int32(K_TOP))
                        ):
                            kf_done = Int32(1)
                            tfk = Uint32(sKr[0])
                            kth_key = Uint64(tfk) << Uint64(32)
                            if sel_tid < Int32(8):
                                cute.arch.atomic_exch(
                                    sTw.iterator + sel_tid,
                                    _pack_floor_key(tfk, seg),
                                    sem="relaxed",
                                    scope="cta",
                                )
                            L.bar(SEL_BAR, NT)
                else:
                    tail0, done0, u0, g0 = _drain_ring(
                        r0,
                        tail0,
                        done0,
                        kth_key,
                        s_ring,
                        sRingBar.iterator.raw_ptr() + r0 * Int32(RING // RING_CHUNK),
                        sRingBar.iterator.raw_ptr()
                        + Int32(8 * (RING // RING_CHUNK))
                        + r0 * Int32(RING // RING_CHUNK),
                        sMb,
                        sCnt.iterator,
                        lane,
                        lt_mask,
                        sKr.iterator,
                        sKr.iterator + 1,
                    )
                    tail1, done1, u1, g1 = _drain_ring(
                        r1,
                        tail1,
                        done1,
                        kth_key,
                        s_ring,
                        sRingBar.iterator.raw_ptr() + r1 * Int32(RING // RING_CHUNK),
                        sRingBar.iterator.raw_ptr()
                        + Int32(8 * (RING // RING_CHUNK))
                        + r1 * Int32(RING // RING_CHUNK),
                        sMb,
                        sCnt.iterator,
                        lane,
                        lt_mask,
                        sKr.iterator,
                        sKr.iterator + 1,
                    )
                    adm = adm + u0 + u1
                    if g0 | g1 == Int32(0):
                        L.nanosleep(Int32(self.sel_idle_ns))
            if cutlass.const_expr(self.stamps):
                if sel_tid == Int32(0):
                    if seg == nseg - Int32(1):
                        mStamps[cta * 256 + 2] = cutlass.Int64(cute.arch.globaltimer())
                    if seg < Int32(8):
                        mStamps[cta * 256 + 104 + seg] = cutlass.Int64(cute.arch.globaltimer())
            cfin = sCnt[0]
            if lane == Int32(0):
                sScal[12 + swarp] = adm
            if seg == nseg - Int32(1):
                cute.arch.fence_view_async_shared()
            L.bar(SEL_BAR, NT)
            pref = Int32(0)
            sdone = Int32(1)
            if seg == nseg - Int32(1):
                if sel_tid == Int32(0):
                    cute.arch.fence_acq_rel_cta()
                    cute.arch.atomic_exch(
                        sHI.iterator + Int32(3), Int32(1), sem="release", scope="cta"
                    )
                if cutlass.const_expr(self.etile):
                    if sel_tid == Int32(0):
                        az = cute.arch.load(
                            sHI.iterator + Int32(15), Int32, sem="acquire", scope="cta"
                        )
                        og = cute.arch.load(
                            sHI.iterator + Int32(4), Int32, sem="acquire", scope="cta"
                        )
                        dk = deep_ok(sHI.iterator)
                        ear = Int32(0)
                        if (az >= seg) & (dk != Int32(0)) & (og != Int32(0)):
                            ear = Int32(1)
                        sScal[9] = ear
                    L.bar(SEL_BAR, NT)
                    if sScal[9] != Int32(0):
                        sGv[sel_tid] = L.l2g_swa(
                            sel_tid,
                            Int32(mKvLens[q]) - Int32(1),
                            Int32(mReq[q]),
                            mBtSwa,
                            Int32(cute.size(mBtSwa, mode=[1])),
                            Int32(0),
                        )
                        L.bar(SEL_BAR, NT)
                        abar_e = sAbar.iterator.raw_ptr()
                        issue_row_loads(
                            sel_tid,
                            q,
                            sGv,
                            cute.recast_ptr(sKV_0.iterator, dtype=cutlass.Float8E4M3FN).raw_ptr(),
                            sKVatt.iterator.raw_ptr(),
                            cute.recast_ptr(s_ring.iterator, dtype=cutlass.Float8E4M3FN).raw_ptr(),
                            cute.recast_ptr(sQ.iterator, dtype=cutlass.Float8E4M3FN).raw_ptr(),
                            sQatt.iterator.raw_ptr(),
                            abar_e,
                            abar_e + Int32(4),
                            abar_e + Int32(13),
                            tma_q,
                            tma_swa,
                            tma_cmp,
                            SEL_BAR,
                            Int32(0),
                            Int32(1),
                        )
                        if sel_tid == Int32(0):
                            sHQ[14] = q
                            cute.arch.fence_acq_rel_cta()
                            cute.arch.atomic_exch(
                                sHI.iterator + Int32(1), Int32(1), sem="release", scope="cta"
                            )
                if cutlass.const_expr(self.sel_fin):
                    # sHand is the only buffer this finalize shares with the row warps' finalize of the earlier segment
                    # (sAttB / sAttO are not used by it), so with sHand2 there is nothing to wait for.
                    if cutlass.const_expr(not self.sel_hand2):
                        if seg > Int32(0):
                            # One-shot transfer after every earlier row finalizer.
                            _wait(sRowBar.iterator.raw_ptr() + Int32(4), Int32(0))
                    L.bar(SEL_BAR, NT)
                    pref = Int32(1)
                    sHand_f = sHand
                    if cutlass.const_expr(self.sel_hand2):
                        sHand_f = sHand2
                    sdone, s0, s1, s2, s3, s4 = _finalize_seg(
                        sel_tid,
                        SEL_BAR,
                        cta,
                        start_q,
                        q,
                        n,
                        P,
                        p,
                        sMb,
                        cfin,
                        None,
                        sHand_f,
                        sAttB,
                        sAttO,
                        sHist,
                        sCk,
                        sRes,
                        sRed,
                        sWc,
                        cute.make_tensor(sScal.iterator + Int32(8), cute.make_layout(1)),
                        mKvLens,
                        mReq,
                        mBtSwa,
                        mBtC,
                        mTopk,
                        mGidx,
                        mRowStat,
                        mSched,
                        mWsSeg.iterator.toint(),
                        mWsCtr.iterator.toint(),
                        Int32(cute.size(mBtSwa, mode=[1])),
                        Int32(cute.size(mBtC, mode=[1])),
                        self.num_sms,
                        self.skew_ns,
                        skew_seed,
                        seg * Int32(16) + Int32(7),
                        self.stamps,
                        mStamps,
                        cta * Int32(256) + Int32(153),
                        None,
                        0,
                        None,
                        self.rsort,
                        merge1=self.merge1,
                        sBig=cute.make_tensor(
                            cute.recast_ptr(sKV_0.iterator, dtype=Uint64),
                            cute.make_layout(self.merge_cap),
                        )
                        if self.merge1 > 0
                        else None,
                        sCov=sCov,
                        big_cap=self.merge_cap,
                        sel_fast=self.sel_fast,
                    )
                    if sdone != Int32(0):
                        sGv[sel_tid] = s0
                        sGv[sel_tid + Int32(128)] = s1
                        sGv[sel_tid + Int32(256)] = s2
                        sGv[sel_tid + Int32(384)] = s3
                        sGv[sel_tid + Int32(512)] = s4
                    L.bar(SEL_BAR, NT)
            if sel_tid == Int32(0):
                hs = bsel * Int32(8)
                sHQ[hs + Int32(0)] = q
                sHQ[hs + Int32(1)] = n
                sHQ[hs + Int32(2)] = P
                sHQ[hs + Int32(3)] = p
                sHQ[hs + Int32(4)] = cfin
                sHQ[hs + Int32(5)] = bsel
                sHQ[hs + Int32(6)] = pref
                sHQ[hs + Int32(7)] = Int32(
                    Uint32(cutlass.Int64(cute.arch.globaltimer()) >> Int64(10))
                )
                if cutlass.const_expr(self.skew_ns > 0):
                    L.skew_sleep(skew_seed, seg * Int32(16) + Int32(5), Int32(self.skew_ns))
                if sdone != Int32(0):
                    cute.arch.fence_acq_rel_cta()
                    cute.arch.atomic_exch(sHI.iterator, seg + Int32(1), sem="release", scope="cta")
                if cutlass.const_expr(self.stamps):
                    if seg < Int32(8):
                        mStamps[cta * 256 + 112 + seg] = cutlass.Int64(cute.arch.globaltimer())
            if sdone != Int32(0):
                # Every selector releases its own sMain/sGv writes; thread 0
                # also publishes sHQ before its arrival completes this phase.
                prims.mbarrier_arrive(sRowBar.iterator.raw_ptr() + bsel)
            if cutlass.const_expr(self.gath):
                if (pref != Int32(0)) & (sdone != Int32(0)):
                    if sel_tid == Int32(0):
                        az = cute.arch.load(
                            sHI.iterator + Int32(15), Int32, sem="acquire", scope="cta"
                        )
                        og = cute.arch.load(
                            sHI.iterator + Int32(4), Int32, sem="acquire", scope="cta"
                        )
                        dk = deep_ok(sHI.iterator)
                        while (az < seg) | (dk == Int32(0)) | (og == Int32(0)):
                            L.nanosleep(Int32(64))
                            az = cute.arch.load(
                                sHI.iterator + Int32(15), Int32, sem="acquire", scope="cta"
                            )
                            og = cute.arch.load(
                                sHI.iterator + Int32(4), Int32, sem="acquire", scope="cta"
                            )
                            dk = deep_ok(sHI.iterator)
                        cute.arch.fence_acq_rel_cta()
                        if cutlass.const_expr(self.stamps):
                            mStamps[cta * 256 + 158] = cutlass.Int64(cute.arch.globaltimer())
                    L.bar(SEL_BAR, NT)
                    abar_s = sAbar.iterator.raw_ptr()
                    jfirst = Int32(0)
                    if cutlass.const_expr(self.etile):
                        jfirst = sScal[9]
                    issue_row_loads(
                        sel_tid,
                        q,
                        sGv,
                        cute.recast_ptr(sKV_0.iterator, dtype=cutlass.Float8E4M3FN).raw_ptr(),
                        sKVatt.iterator.raw_ptr(),
                        cute.recast_ptr(s_ring.iterator, dtype=cutlass.Float8E4M3FN).raw_ptr(),
                        cute.recast_ptr(sQ.iterator, dtype=cutlass.Float8E4M3FN).raw_ptr(),
                        sQatt.iterator.raw_ptr(),
                        abar_s,
                        abar_s + Int32(4),
                        abar_s + Int32(13),
                        tma_q,
                        tma_swa,
                        tma_cmp,
                        SEL_BAR,
                        jfirst,
                        Int32(NTILES),
                    )
                    if cutlass.const_expr(self.stamps):
                        if sel_tid == Int32(0):
                            mStamps[cta * 256 + 159] = cutlass.Int64(cute.arch.globaltimer())
            seg = seg + Int32(1)
        if sel_tid == Int32(0):
            cute.arch.fence_acq_rel_cta()
            cute.arch.atomic_exch(sHI.iterator + Int32(3), Int32(1), sem="release", scope="cta")
            cute.arch.atomic_exch(sHI.iterator + Int32(2), Int32(1), sem="release", scope="cta")
        if cutlass.const_expr(self.stamps):
            if sel_tid == Int32(0):
                mStamps[cta * 256 + 3] = cutlass.Int64(cute.arch.globaltimer())
                mStamps[cta * 256 + 12] = t_comp

    @cute.jit
    def _row_role(
        self,
        rtid,
        cta,
        mKvLens,
        mReq,
        mBtSwa,
        mBtC,
        mTopk,
        mGidx,
        mWsSeg,
        mWsCtr,
        mRowStat,
        mStamps,
        sHand,
        sAttB,
        sAttO,
        sHI,
        sRS,
        tma_q,
        tma_swa,
        tma_cmp,
        mSinks,
        mOut,
        softmax_scale,
        sQatt,
        sKVatt,
        sAbar,
        sMsk,
        sKV_0,
        sQ,
        start_q,
        batch_size,
        mSched,
        sMain,
        sHQ,
        sRres,
        sRred,
        sRwc,
        s_ring,
        sGv,
        nseg_r,
        sDQ,
        refused,
        mStatus,
        sRowBar,
    ):
        NT: cutlass.Constexpr[int] = SEL_NTHR
        mbs = Int32(cute.size(mBtSwa, mode=[1]))
        mbc = Int32(cute.size(mBtC, mode=[1]))
        ws_seg_base = mWsSeg.iterator.toint()
        ws_ctr_base = mWsCtr.iterator.toint()
        nh = Int32(0)
        live = Int32(1)
        aph = Int32(0)
        skew_seed = Int32(0)
        if cutlass.const_expr(self.skew_ns > 0):
            skew_seed = (
                Int32(Uint32(cutlass.Int64(cute.arch.globaltimer()) >> Int64(5)))
                + cta * Int32(7919)
                + Int32(29)
            )
        scale_log2 = softmax_scale * Float32(LOG2_E)
        tmem_att = Int32(0)
        q_slot0 = sQatt.iterator.raw_ptr()
        kv_slot0 = sKVatt.iterator.raw_ptr()
        deep_base = cute.recast_ptr(sKV_0.iterator, dtype=cutlass.Float8E4M3FN).raw_ptr()
        deep_q = cute.recast_ptr(sQ.iterator, dtype=cutlass.Float8E4M3FN).raw_ptr()
        abar = sAbar.iterator.raw_ptr()
        out_addr = mOut.iterator.toint()
        sink_base = mSinks.iterator.raw_ptr()
        kv_slot7 = cute.recast_ptr(s_ring.iterator, dtype=cutlass.Float8E4M3FN).raw_ptr()
        main_b = cute.recast_ptr(sMain.iterator, dtype=cutlass.Float8E4M3FN).raw_ptr()
        sHistR = cute.make_tensor(
            cute.recast_ptr(sAttB.iterator, dtype=Int32), cute.make_layout(L.NB_SEL)
        )
        sCkR = cute.make_tensor(sAttO.iterator, cute.make_layout(L.QUADC))
        oend = cutlass.min(mSched[0, 0], batch_size)
        if refused != Int32(0):
            oend = Int32(0)
            rb = cta
            while rb < batch_size:
                ic = rtid
                while ic < Int32(K_TOP):
                    mTopk[rb, ic] = Int32(-1)
                    ic = ic + Int32(NT)
                ic = rtid
                while ic < Int32(640):
                    mGidx[rb, ic] = Int32(-1)
                    ic = ic + Int32(NT)
                if cutlass.const_expr(self.attn):
                    ob = mOut.iterator.toint() + Int64(rb) * Int64(HEADS * LATENT * 2)
                    iw = rtid
                    while iw < Int32(HEADS * LATENT // 2):
                        cutlass.inttoptr(ob + Int64(iw) * Int64(4), mem_space=1, dtype=Int32).store(
                            Int32(2143322048)
                        )
                        iw = iw + Int32(NT)
                L.bar(ROW_BAR, NT)
                if rtid == Int32(0):
                    L.publish_row(mRowStat.iterator.toint(), rb)
                rb = rb + Int32(self.num_sms)
        if cta == Int32(0):
            if rtid == Int32(0):
                mStatus[0] = refused
        if cta < oend:
            oq = cta
            while oq < oend:
                opos = Int32(mKvLens[oq]) - Int32(1)
                oreq = Int32(mReq[oq])
                og0, og1, og2, og3, og4 = _write_row_outputs_regs(
                    oq,
                    Int32(0),
                    sHand,
                    Int32(0),
                    Int32(1),
                    mTopk,
                    mGidx,
                    mBtSwa,
                    mBtC,
                    opos,
                    oreq,
                    mbs,
                    mbc,
                    rtid,
                )
                L.bar(ROW_BAR, NT)
                if rtid == Int32(0):
                    L.publish_row(mRowStat.iterator.toint(), oq)
                if cutlass.const_expr(self.attn):
                    if rtid == Int32(0):
                        _wait(sRowBar.iterator.raw_ptr() + Int32(5), Int32(0))
                        sRS[2] = sHI[9]
                        sRS[3] = deep_ok(sHI.iterator)
                    L.bar(ROW_BAR, NT)
                    tmem_att = sRS[2]
                    odeep = Int32(0)
                    if cutlass.const_expr(not self.no_deep):
                        if sRS[3] != Int32(0):
                            odeep = Int32(1)
                    if cutlass.const_expr(self.att_skip & 32 == 0):
                        aph = att_row(
                            rtid,
                            oq,
                            og0,
                            og1,
                            og2,
                            og3,
                            og4,
                            tmem_att,
                            odeep,
                            q_slot0,
                            kv_slot0,
                            deep_base,
                            deep_q,
                            abar,
                            abar + Int32(4),
                            abar + Int32(12),
                            aph,
                            tma_q,
                            tma_swa,
                            tma_cmp,
                            sink_base + oq * Int32(HEADS),
                            out_addr,
                            scale_log2,
                            sMsk,
                            sHI.iterator,
                            ROW_BAR,
                            self.att_skip,
                            not self.no_deep,
                            self.stamps,
                            mStamps,
                            cta * Int32(256) + Int32(128),
                            kv_slot7,
                            Int32(0),
                            abar + Int32(13),
                            self.ahint,
                        )
                oq = oq + Int32(self.num_sms)
            L.bar(ROW_BAR, NT)
        if rtid == Int32(0):
            cute.arch.fence_acq_rel_cta()
            cute.arch.atomic_exch(sHI.iterator + Int32(4), Int32(1), sem="release", scope="cta")
        adone = Int32(0)
        st_m = Float32(NEG_FLT_MAX)
        st_l = Float32(0.0)
        st_q = Int32(0)
        st_p = Int32(0)
        st_d = Int32(0)
        dq_n = Int32(0)
        while live != Int32(0):
            if rtid == Int32(0):
                prod = cute.arch.load(sHI.iterator, Int32, sem="acquire", scope="cta")
                fin = cute.arch.load(sHI.iterator + Int32(2), Int32, sem="acquire", scope="cta")
                ea = Int32(0)
                if cutlass.const_expr(self.etile):
                    ea = cute.arch.load(sHI.iterator + Int32(1), Int32, sem="acquire", scope="cta")
                dgo = Int32(0)
                if cutlass.const_expr(self.defer):
                    if dq_n > Int32(0):
                        dgo = deep_ok(sHI.iterator)
                        if cute.arch.load(
                            sHI.iterator + Int32(5), Int32, sem="acquire", scope="cta"
                        ) >= Int32(DEFER_LONG):
                            dgo = Int32(1)
                while (
                    (prod <= nh)
                    & (fin == Int32(0))
                    & ((ea == Int32(0)) | (adone != Int32(0)))
                    & (dgo == Int32(0))
                ):
                    L.nanosleep(Int32(128))
                    prod = cute.arch.load(sHI.iterator, Int32, sem="acquire", scope="cta")
                    fin = cute.arch.load(sHI.iterator + Int32(2), Int32, sem="acquire", scope="cta")
                    if cutlass.const_expr(self.etile):
                        ea = cute.arch.load(
                            sHI.iterator + Int32(1), Int32, sem="acquire", scope="cta"
                        )
                    if cutlass.const_expr(self.defer):
                        if dq_n > Int32(0):
                            dgo = deep_ok(sHI.iterator)
                            if cute.arch.load(
                                sHI.iterator + Int32(5), Int32, sem="acquire", scope="cta"
                            ) >= Int32(DEFER_LONG):
                                dgo = Int32(1)
                if prod <= nh:
                    cute.arch.fence_acq_rel_cta()
                    prod = cute.arch.load(sHI.iterator, Int32, sem="acquire", scope="cta")
                cute.arch.fence_acq_rel_cta()
                wk = Int32(0)
                if prod > nh:
                    # The leader consults pref before the row-wide work barrier.
                    _wait(sRowBar.iterator.raw_ptr() + (nh & Int32(1)), (nh >> Int32(1)) & Int32(1))
                    wk = Int32(1)
                    if cutlass.const_expr(self.defer):
                        if dq_n > Int32(0):
                            if sHQ[(nh & Int32(1)) * Int32(8) + Int32(6)] != Int32(0):
                                wk = Int32(3)
                elif dq_n > Int32(0):
                    wk = Int32(3)
                elif (ea != Int32(0)) & (adone == Int32(0)):
                    wk = Int32(2)
                sRS[0] = wk
            L.bar(ROW_BAR, NT)
            work = sRS[0]
            # Every reader captures work before the leader may publish again.
            L.bar(ROW_BAR, NT)
            if work == Int32(1):
                _wait(sRowBar.iterator.raw_ptr() + (nh & Int32(1)), (nh >> Int32(1)) & Int32(1))
            if work == Int32(0):
                live = Int32(0)
            elif cutlass.const_expr(self.row_noop):
                L.bar(ROW_BAR, NT)
                prims.mbarrier_arrive(sRowBar.iterator.raw_ptr() + Int32(2) + (nh & Int32(1)))
                nh = nh + Int32(1)
            else:
                hs = (nh & Int32(1)) * Int32(8)
                q = Int32(0)
                n = Int32(0)
                P = Int32(1)
                p = Int32(0)
                cbuf = Int32(0)
                bsel = Int32(0)
                pref = Int32(0)
                # The selector may rewrite these slots while deferred or early
                # attention runs. Only a published segment consumes its metadata.
                if work == Int32(1):
                    q = sHQ[hs + Int32(0)]
                    n = sHQ[hs + Int32(1)]
                    P = sHQ[hs + Int32(2)]
                    p = sHQ[hs + Int32(3)]
                    cbuf = sHQ[hs + Int32(4)]
                    bsel = sHQ[hs + Int32(5)]
                    pref = sHQ[hs + Int32(6)]
                done_row = Int32(0)
                gv0 = Int32(-1)
                gv1 = Int32(-1)
                gv2 = Int32(-1)
                gv3 = Int32(-1)
                gv4 = Int32(-1)
                j0 = Int32(0)
                j1 = Int32(NTILES)
                keep = Int32(0)
                dly_t = Int32(-1)
                if cutlass.const_expr(self.rdelay > 0):
                    if work == Int32(1):
                        if nh + Int32(2) < nseg_r:
                            dly_t = sHQ[hs + Int32(7)]
                if work == Int32(2):
                    q = sHQ[14]
                    pref = Int32(1)
                    gv0 = sGv[rtid]
                    done_row = Int32(1)
                    j1 = Int32(1)
                elif work == Int32(3):
                    dq_n = dq_n - Int32(1)
                    q = sDQ[dq_n]
                    gv0 = Int32(mGidx[q, rtid])
                    gv1 = Int32(mGidx[q, rtid + Int32(128)])
                    gv2 = Int32(mGidx[q, rtid + Int32(256)])
                    gv3 = Int32(mGidx[q, rtid + Int32(384)])
                    gv4 = Int32(mGidx[q, rtid + Int32(512)])
                    pref = Int32(0)
                    done_row = Int32(1)
                elif pref != Int32(0):
                    gv0 = sGv[rtid]
                    gv1 = sGv[rtid + Int32(128)]
                    gv2 = sGv[rtid + Int32(256)]
                    gv3 = sGv[rtid + Int32(384)]
                    gv4 = sGv[rtid + Int32(512)]
                    done_row = Int32(1)
                    if adone != Int32(0):
                        j0 = Int32(1)
                else:
                    if cutlass.const_expr(self.ring2):
                        if P == Int32(1):
                            keep = Int32(1)
                    sMb = cute.make_tensor(
                        sMain.iterator + bsel * Int32(MAIN_CAP), cute.make_layout(MAIN_CAP)
                    )
                    done_row, gv0, gv1, gv2, gv3, gv4 = _finalize_seg(
                        rtid,
                        ROW_BAR,
                        cta,
                        start_q,
                        q,
                        n,
                        P,
                        p,
                        sMb,
                        cbuf,
                        sHI.iterator + Int32(12) + bsel,
                        sHand,
                        sAttB,
                        sAttO,
                        sHistR,
                        sCkR,
                        sRres,
                        sRred,
                        sRwc,
                        cute.make_tensor(sRS.iterator + Int32(1), cute.make_layout(1)),
                        mKvLens,
                        mReq,
                        mBtSwa,
                        mBtC,
                        mTopk,
                        mGidx,
                        mRowStat,
                        mSched,
                        ws_seg_base,
                        ws_ctr_base,
                        mbs,
                        mbc,
                        self.num_sms,
                        self.skew_ns,
                        skew_seed,
                        nh * Int32(16) + Int32(3),
                        self.stamps,
                        mStamps,
                        cta * Int32(256) + Int32(148),
                        dly_t,
                        self.rdelay,
                        keep,
                        self.rsort,
                        sRowBar.iterator.raw_ptr() + Int32(2) + bsel,
                        sel_fast=self.sel_fast,
                    )
                if work == Int32(1):
                    if nh + Int32(2) == nseg_r:
                        # The selector takes sHand/sAttB/sAttO only for its last
                        # segment, after all 128 row threads finish the prior one
                        # (not awaited with sel_hand2).
                        prims.mbarrier_arrive(sRowBar.iterator.raw_ptr() + Int32(4))
                if cutlass.const_expr(self.attn):
                    if done_row != Int32(0):
                        if rtid == Int32(0):
                            _wait(sRowBar.iterator.raw_ptr() + Int32(5), Int32(0))
                            sRS[2] = sHI[9]
                            sRS[3] = deep_ok(sHI.iterator)
                            if cutlass.const_expr(self.defer):
                                sRS[5] = cute.arch.load(
                                    sHI.iterator, Int32, sem="acquire", scope="cta"
                                )
                                sRS[6] = cute.arch.load(
                                    sHI.iterator + Int32(5), Int32, sem="acquire", scope="cta"
                                )
                        if cutlass.const_expr(self.skew_ns > 0):
                            L.skew_sleep(
                                skew_seed + rtid, nh * Int32(16) + Int32(9), Int32(self.skew_ns)
                            )
                        L.bar(ROW_BAR, NT)
                        tmem_att = sRS[2]
                        dfr = Int32(0)
                        if cutlass.const_expr(self.defer):
                            if (
                                (work == Int32(1))
                                & (pref == Int32(0))
                                & (sRS[3] == Int32(0))
                                & (dq_n < Int32(self.dq_cap))
                            ):
                                if sRS[5] > nh + Int32(1):
                                    dfr = Int32(1)
                                elif sRS[6] < Int32(DEFER_LONG):
                                    dfr = Int32(1)
                        if dfr != Int32(0):
                            if rtid == Int32(0):
                                sDQ[dq_n] = q
                            dq_n = dq_n + Int32(1)
                        else:
                            deep0 = Int32(0)
                            if cutlass.const_expr(not self.no_deep):
                                if sRS[3] != Int32(0):
                                    deep0 = Int32(1)
                            ext_on = Int32(0)
                            if cutlass.const_expr(self.gath):
                                if pref != Int32(0):
                                    ext_on = Int32(1)
                                    deep0 = Int32(1)
                            if cutlass.const_expr(self.stamps):
                                if rtid == Int32(0):
                                    if work == Int32(2):
                                        mStamps[cta * 256 + 160] = cutlass.Int64(
                                            cute.arch.globaltimer()
                                        )
                                    elif (nh < Int32(8)) & (work == Int32(1)):
                                        mStamps[cta * 256 + 24 + nh] = cutlass.Int64(
                                            cute.arch.globaltimer()
                                        )
                            kv_b = None
                            two_a = None
                            if cutlass.const_expr(self.ring2):
                                kv_b = main_b + bsel * Int32(MAIN_CAP * 8)
                                two_a = keep
                                cute.arch.fence_view_async_shared()
                            if cutlass.const_expr(self.att_skip & 32 == 0):
                                if cutlass.const_expr(self.etile):
                                    i_m = Float32(NEG_FLT_MAX)
                                    i_l = Float32(0.0)
                                    i_q = Int32(0)
                                    i_p = Int32(0)
                                    i_d = deep0
                                    if j0 != Int32(0):
                                        i_m = st_m
                                        i_l = st_l
                                        i_q = st_q
                                        i_p = st_p
                                        i_d = st_d
                                    aph, st_m, st_l, st_q, st_p, st_d = att_row(
                                        rtid,
                                        q,
                                        gv0,
                                        gv1,
                                        gv2,
                                        gv3,
                                        gv4,
                                        tmem_att,
                                        deep0,
                                        q_slot0,
                                        kv_slot0,
                                        deep_base,
                                        deep_q,
                                        abar,
                                        abar + Int32(4),
                                        abar + Int32(12),
                                        aph,
                                        tma_q,
                                        tma_swa,
                                        tma_cmp,
                                        sink_base + q * Int32(HEADS),
                                        out_addr,
                                        scale_log2,
                                        sMsk,
                                        sHI.iterator,
                                        ROW_BAR,
                                        self.att_skip,
                                        not self.no_deep,
                                        self.stamps,
                                        mStamps,
                                        cta * Int32(256) + Int32(128),
                                        kv_slot7,
                                        ext_on,
                                        abar + Int32(13),
                                        self.ahint,
                                        True,
                                        j0,
                                        j1,
                                        (i_m, i_l, i_q, i_p, i_d),
                                        self.apace,
                                        kv_b,
                                        two_a,
                                    )
                                else:
                                    pl = None
                                    if cutlass.const_expr(self.pace_adapt):
                                        pl = nh + Int32(1)
                                        if work == Int32(3):
                                            pl = nh
                                    aph = att_row(
                                        rtid,
                                        q,
                                        gv0,
                                        gv1,
                                        gv2,
                                        gv3,
                                        gv4,
                                        tmem_att,
                                        deep0,
                                        q_slot0,
                                        kv_slot0,
                                        deep_base,
                                        deep_q,
                                        abar,
                                        abar + Int32(4),
                                        abar + Int32(12),
                                        aph,
                                        tma_q,
                                        tma_swa,
                                        tma_cmp,
                                        sink_base + q * Int32(HEADS),
                                        out_addr,
                                        scale_log2,
                                        sMsk,
                                        sHI.iterator,
                                        ROW_BAR,
                                        self.att_skip,
                                        not self.no_deep,
                                        self.stamps,
                                        mStamps,
                                        cta * Int32(256) + Int32(128),
                                        kv_slot7,
                                        ext_on,
                                        abar + Int32(13),
                                        self.ahint,
                                        False,
                                        None,
                                        None,
                                        None,
                                        self.apace,
                                        kv_b,
                                        two_a,
                                        pl,
                                    )
                    if cutlass.const_expr(self.stamps):
                        if work == Int32(1):
                            if nh < Int32(8):
                                mStamps[cta * 256 + 120 + nh] = cutlass.Int64(
                                    cute.arch.globaltimer()
                                )
                        else:
                            mStamps[cta * 256 + 161] = cutlass.Int64(cute.arch.globaltimer())
                if cutlass.const_expr(self.ring2):
                    if keep != Int32(0):
                        if rtid == Int32(0):
                            cute.arch.fence_view_async_shared()
                        prims.mbarrier_arrive(sRowBar.iterator.raw_ptr() + Int32(2) + bsel)
                if work == Int32(2):
                    adone = Int32(1)
                elif work == Int32(3):
                    if dq_n == Int32(0):
                        if rtid == Int32(0):
                            cute.arch.fence_acq_rel_cta()
                            cute.arch.atomic_exch(
                                sHI.iterator + Int32(15), nh, sem="release", scope="cta"
                            )
                else:
                    if dq_n == Int32(0):
                        if rtid == Int32(0):
                            cute.arch.fence_acq_rel_cta()
                            cute.arch.atomic_exch(
                                sHI.iterator + Int32(15), nh + Int32(1), sem="release", scope="cta"
                            )
                    nh = nh + Int32(1)
        if cutlass.const_expr(self.stamps):
            if rtid == Int32(0):
                mStamps[cta * 256 + 20] = cutlass.Int64(cute.arch.globaltimer())
                mStamps[cta * 256 + 14] = L.clk64()


_COMPILED = {}


def compile_indexer(mk: IndexerKernel):
    key = tuple(
        ((k, v) for k, v in sorted(vars(mk).items()) if isinstance(v, (int, float, bool, str)))
    )
    if key in _COMPILED:
        return _COMPILED[key]
    H, D = (mk.num_heads, mk.head_dim)
    fp4 = mk.fp4
    block_bytes = mk.phys_block_kv * ((D // 2 if fp4 else D) + 4)
    rt = cute.runtime
    sb = cute.sym_int()
    kv_fake = rt.make_fake_tensor(
        cutlass.Uint8, (cute.sym_int(), block_bytes), stride=(cute.sym_int64(), 1)
    )
    q_fake = rt.make_fake_compact_tensor(
        cutlass.Uint8, (H, D // 2 if fp4 else D, cute.sym_int()), stride_order=(1, 0, 2)
    )
    w_fake = rt.make_fake_compact_tensor(cutlass.Float32, (H, cute.sym_int()), stride_order=(0, 1))
    lg_fake = rt.make_fake_tensor(
        cutlass.Float32, (cute.sym_int(), cute.sym_int()), stride=(cute.sym_int64(), 1)
    )
    bt_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb, cute.sym_int()), stride_order=(1, 0))
    cl_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb,), stride_order=(0,))
    sm_fake = rt.make_fake_compact_tensor(cutlass.Int32, (cute.sym_int(), 2), stride_order=(1, 0))
    kvl_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb,), stride_order=(0,))
    req_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb,), stride_order=(0,))
    bts_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb, cute.sym_int()), stride_order=(1, 0))
    btc_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb, cute.sym_int()), stride_order=(1, 0))
    tk_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb, K_TOP), stride_order=(1, 0))
    gi_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb, 640), stride_order=(1, 0))
    wss_fake = rt.make_fake_compact_tensor(cutlass.Int64, (cute.sym_int(),), stride_order=(0,))
    wsc_fake = rt.make_fake_compact_tensor(cutlass.Int64, (cute.sym_int(),), stride_order=(0,))
    rs_fake = rt.make_fake_compact_tensor(cutlass.Int32, (sb, 16), stride_order=(1, 0))
    st_fake = rt.make_fake_compact_tensor(cutlass.Int64, (cute.sym_int(),), stride_order=(0,))
    qa_fake = rt.make_fake_compact_tensor(
        cutlass.Uint8, (cute.sym_int64(), 64, 512), stride_order=(2, 1, 0), assumed_align=16
    )
    sp_fake = rt.make_fake_compact_tensor(
        cutlass.Uint8, (cute.sym_int64(), 512), stride_order=(1, 0), assumed_align=16
    )
    cp_fake = rt.make_fake_compact_tensor(
        cutlass.Uint8, (cute.sym_int64(), 512), stride_order=(1, 0), assumed_align=16
    )
    sk_fake = rt.make_fake_compact_tensor(
        cutlass.Float32, (cute.sym_int(), 64), stride_order=(1, 0), assumed_align=16
    )
    ou_fake = rt.make_fake_compact_tensor(
        cutlass.BFloat16, (cute.sym_int(), 64, 512), stride_order=(2, 1, 0), assumed_align=16
    )
    stw_fake = rt.make_fake_compact_tensor(cutlass.Int32, (4,), stride_order=(0,))
    stream = rt.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        mk,
        kv_fake,
        q_fake,
        w_fake,
        lg_fake,
        bt_fake,
        cl_fake,
        sm_fake,
        kvl_fake,
        req_fake,
        bts_fake,
        btc_fake,
        tk_fake,
        gi_fake,
        wss_fake,
        wsc_fake,
        rs_fake,
        st_fake,
        qa_fake,
        sp_fake,
        cp_fake,
        sk_fake,
        ou_fake,
        stw_fake,
        cutlass.Float32(1.0),
        cutlass.Int32(1),
        cutlass.Int32(1),
        stream,
        *(
            (rt.make_fake_compact_tensor(cutlass.Int32, (H, cute.sym_int()), stride_order=(0, 1)),)
            if fp4
            else ()
        ),
        options="--enable-tvm-ffi --generate-line-info",
    )
    _COMPILED[key] = compiled
    return compiled
