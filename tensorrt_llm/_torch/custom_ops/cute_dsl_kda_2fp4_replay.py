# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused Conv4, native 2FP4 KDA replay, gated output RMSNorm, and window commit.

One CTA per (batch row, head). The CTA that appends the sixteenth history
entry of its head also commits the full window into both checkpoint terms,
so ordinary decode needs a single launch. History lengths are advanced by
the last CTA of each batch row to arrive, which keeps every CTA's length
read ahead of the update without a separate launch.

With PDL, everything except the current step's projections (slot mapping,
history length, checkpoint and history copies, Conv window and constants)
is issued before ``griddepcontrol.wait``; all stores follow it.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda_exp
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir as mlir_ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_tensor
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.experimental import primitives as prims

from ..flashinfer_utils import get_env_enable_pdl
from .cute_dsl_kda_2fp4_commit import (
    _decode8,
    _dequant8,
    _expand4,
    _f_abs,
    _f_div,
    _f_exp,
    _f_fma,
    _f_min,
    _f_mul,
    _f_neg,
    _ld_global_u32,
    _ld_global_v4,
    _pack_e2m1x8,
    _pack_e4m3x2,
    _pack_e4m3x4,
    _sel_eq0,
    _st_global_u32,
    _st_outer_pad,
    _tmem_st32,
    _tree_max,
    _warp_max_f32,
)

THREADS = 128
K = 128
N = 16
STATE_BYTES = 18448
VALUES_BYTES = 8192
PRIMARY_SCALE = 16384
RESIDUAL_SCALE = 17408
OUTER_SCALE = 18432
# Commit operand panels: K-major, no swizzle, 8-row x 16 B core matrices.
PANEL_LBO = 128
PANEL_SBO = 256
PANEL_BYTES = K * N * 2

_compiled = {}


def _ir(x, loc=None, ip=None):
    return x.ir_value(loc=loc, ip=ip)


@dsl_user_op
def _st_global_v2(addr: Int64, a: Uint32, b: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [_ir(Int64(addr), loc, ip), _ir(Uint32(a), loc, ip), _ir(Uint32(b), loc, ip)],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_u8(addr: Int64, v: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [_ir(Int64(addr), loc, ip), _ir(Uint32(v), loc, ip)],
        "{ .reg .b16 t; cvt.u16.u32 t, $1; st.global.b8 [$0], t; }",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_shared_v4(addr: Int32, a: Uint32, b: Uint32, c: Uint32, d: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            _ir(Int32(addr), loc, ip),
            _ir(Uint32(a), loc, ip),
            _ir(Uint32(b), loc, ip),
            _ir(Uint32(c), loc, ip),
            _ir(Uint32(d), loc, ip),
        ],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _pack_bf16x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """RN BF16 pair, ``lo`` in the low half (the lower K index)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [_ir(Float32(lo), loc, ip), _ir(Float32(hi), loc, ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# Batch indices, slot map and history lengths are a few bytes read by every
# layer; keep them in L2 across the weight traffic between KDA layers.
_EVICT_LAST = "createpolicy.fractional.L2::evict_last.b64 p, 1.0;"


@dsl_user_op
def _ld_keep_s32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [_ir(Int64(addr), loc, ip)],
            "{ .reg .b64 p; " + _EVICT_LAST + " ld.global.L2::cache_hint.b32 $0, [$1], p; }",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_keep_u8(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [_ir(Int64(addr), loc, ip)],
            "{ .reg .b64 p; .reg .b16 t; "
            + _EVICT_LAST
            + " ld.global.L2::cache_hint.b8 t, [$1], p; cvt.u32.u16 $0, t; }",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_keep_s32(addr: Int64, v: Int32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [_ir(Int64(addr), loc, ip), _ir(Int32(v), loc, ip)],
        "{ .reg .b64 p; " + _EVICT_LAST + " st.global.L2::cache_hint.b32 [$0], $1, p; }",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _addr(tensor, index):
    return Int64((tensor.iterator + index).toint())


@cute.jit
def _warp_sum_squares(value):
    partner = cute.arch.shuffle_sync_bfly(value * value, 16)
    total = cute.fma(value, value, partner)
    for offset in (8, 4, 2, 1):
        total = total + cute.arch.shuffle_sync_bfly(total, offset)
    return total


@cute.jit
def _warp_sum_products(lhs, rhs):
    partner = cute.arch.shuffle_sync_bfly(lhs * rhs, 16)
    total = cute.fma(lhs, rhs, partner)
    for offset in (8, 4, 2, 1):
        total = total + cute.arch.shuffle_sync_bfly(total, offset)
    return total


@cute.jit
def _smem_u32(ptr):
    return Int32(ptr.toint())


@cute.jit
def _panel_offset(row, chunk):
    return (row // 8) * PANEL_SBO + chunk * PANEL_LBO + (row % 8) * 16


@cute.jit
def _commit_window(
    tidx,
    warp,
    lane,
    tmem_raw,
    head_base,
    state_outer,
    cumulative,
    kn,
    update,
    primary_image,
    cache_hk,
    cache_hg,
    cache_hu,
    s_exp,
    s_red,
    commit_mbar,
):
    """Fold the 16-entry window into both checkpoint terms.

    The calculation and rounding are those of ``cute_dsl_kda_2fp4_commit``:
    the decayed primary+residual state seeds the TMEM accumulator, one BF16
    UMMA adds U^T @ bf16(K * exp(G_end - G)), and both FP4 terms are
    repacked with a shared outer scale. Thread ``tidx`` owns value row and
    key column ``tidx``. Entry 15 is the current token, still in registers.
    """
    v = tidx
    s_exp[v] = _f_exp(cumulative)
    # Residual row (prefetched to L2 at kernel start) and both scale rows.
    row_base = head_base + Int64(v) * 64
    residual_row = [_ld_global_v4(row_base + VALUES_BYTES + 16 * c) for c in range(4)]
    psw = [_ld_global_u32(head_base + PRIMARY_SCALE + Int64(v) * 8 + 4 * j) for j in range(2)]
    rsw = [_ld_global_u32(head_base + RESIDUAL_SCALE + Int64(v) * 8 + 4 * j) for j in range(2)]

    # B operand rows (key k): bf16(K_i[k] * exp(G_end[k] - G_i[k])).
    adj = []
    for i in cutlass.range_constexpr(N):
        kv = kn.to(Float32)
        gv = cumulative
        if cutlass.const_expr(i < N - 1):
            kv = cache_hk[i * K + v].to(Float32)
            gv = cache_hg[i * K + v]
        adj.append(_f_mul(kv, _f_exp(cumulative - gv)))
    # A operand rows (value v): cached BF16 updates.
    upd = []
    for i in cutlass.range_constexpr(N):
        uv = update.to(cutlass.BFloat16).to(Float32)
        if cutlass.const_expr(i < N - 1):
            uv = cache_hu[i * K + v].to(Float32)
        upd.append(uv)
    bws = []
    aws = []
    for chunk in cutlass.range_constexpr(2):
        bws.append(
            [_pack_bf16x2(adj[chunk * 8 + 2 * j], adj[chunk * 8 + 2 * j + 1]) for j in range(4)]
        )
        aws.append(
            [_pack_bf16x2(upd[chunk * 8 + 2 * j], upd[chunk * 8 + 2 * j + 1]) for j in range(4)]
        )
    # The G rows are consumed: they become the two 4 KB operand panels.
    cute.arch.barrier()
    panel_a = cutlass.Array(cache_hg.data_ptr(), shape=PANEL_BYTES // 2, dtype=cutlass.BFloat16)
    panel_b = cutlass.Array(
        cache_hg.subview(PANEL_BYTES // 4).data_ptr(),
        shape=PANEL_BYTES // 2,
        dtype=cutlass.BFloat16,
    )
    for chunk in cutlass.range_constexpr(2):
        off = _panel_offset(v, chunk)
        bw = bws[chunk]
        aw = aws[chunk]
        _st_shared_v4(_smem_u32(panel_b.data_ptr()) + off, bw[0], bw[1], bw[2], bw[3])
        _st_shared_v4(_smem_u32(panel_a.data_ptr()) + off, aw[0], aw[1], aw[2], aw[3])
    cute.arch.barrier()

    # Seed TMEM row v with dequant(primary + residual) * exp(G_end). The
    # primary row comes from the SW64 TMA image.
    swizzle_s64 = cutlass.Swizzle.from_name("s64b")
    ep = list(_expand4(psw[0], state_outer)) + list(_expand4(psw[1], state_outer))
    er = list(_expand4(rsw[0], state_outer)) + list(_expand4(rsw[1], state_outer))
    tmem_lane = ((tmem_raw >> Int32(16)) + warp * Int32(32)) << Int32(16)
    tmem_row = Uint32(tmem_lane | (tmem_raw & Int32(0xFFFF)))
    for c in cutlass.range_constexpr(4):
        pw_ptr = primary_image.subview(v * 64 + c * 16).data_ptr().apply_swizzle(swizzle_s64)
        pw = pw_ptr.load(count=16, alignment=16).bitcast(Uint32)
        rw = residual_row[c]
        seed = []
        for jj in cutlass.range_constexpr(4):
            g = 2 * c + jj // 2
            old8 = _dequant8(pw[jj], rw[jj], ep[g], er[g])
            for t in cutlass.range_constexpr(8):
                seed.append(_f_mul(old8[t], s_exp[32 * c + 8 * jj + t]))
        _tmem_st32(tmem_row + Uint32(32 * c), seed)
    cute.arch.fence_view_async_shared()
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    cute.arch.barrier()

    if warp == 0:
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        idesc = prims.Tcgen05InstrDesc.build(
            c_dtype=Float32,
            a_dtype=cutlass.BFloat16,
            b_dtype=cutlass.BFloat16,
            n_dim=K,
            m_dim=K,
        )
        desc_a = prims.Tcgen05SmemDesc.build(
            panel_a,
            leading_byte_offset=PANEL_LBO,
            stride_byte_offset=PANEL_SBO,
            layout=prims.Tcgen05SmemSwizzle.NONE,
        )
        desc_b = prims.Tcgen05SmemDesc.build(
            panel_b,
            leading_byte_offset=PANEL_LBO,
            stride_byte_offset=PANEL_SBO,
            layout=prims.Tcgen05SmemSwizzle.NONE,
        )
        if prims.elect_sync():
            prims.tcgen05_mma(
                prims.Tcgen05MMAKind.F16,
                prims.CTAGroup.CTA_1,
                cutlass.inttoptr(tmem_raw, 6, Float32),
                desc_a,
                desc_b,
                idesc,
                True,
            )
            prims.tcgen05_commit(commit_mbar, group=prims.CTAGroup.CTA_1)
    while not prims.mbarrier_try_wait_parity(commit_mbar, Int32(0)):
        pass
    prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)

    # Shared outer scale from the block amax of every 16-key group.
    bam = []
    for c in cutlass.range_constexpr(4):
        sn = prims.tcgen05_ld(
            "32x32b", cutlass.inttoptr(tmem_row + Uint32(32 * c), 6, Float32), num=32
        )
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        for h in cutlass.range_constexpr(2):
            bam.append(_tree_max([_f_abs(sn[16 * h + t]) for t in range(16)]))
    m = _warp_max_f32(_tree_max(bam))
    if lane == 0:
        s_red[warp] = m
    cute.arch.barrier()
    gmax = _tree_max([s_red[w] for w in range(4)])
    outer_new = _f_div(gmax, Float32(2688.0))
    t6 = _f_mul(Float32(6.0), _sel_eq0(outer_new, Float32(1.0), outer_new))
    inv_t6 = _f_div(Float32(1.0), t6)

    psc = [_f_min(Float32(448.0), _f_mul(bam[g], inv_t6)) for g in range(8)]
    pscp = [
        _pack_e4m3x4(psc[4 * j], psc[4 * j + 1], psc[4 * j + 2], psc[4 * j + 3]) for j in range(2)
    ]
    for j in cutlass.range_constexpr(2):
        _st_global_u32(head_base + PRIMARY_SCALE + Int64(v) * 8 + 4 * j, pscp[j])
    pe = list(_expand4(pscp[0], outer_new)) + list(_expand4(pscp[1], outer_new))

    for g in cutlass.range_constexpr(8):
        sn = prims.tcgen05_ld(
            "32x32b", cutlass.inttoptr(tmem_row + Uint32(16 * g), 6, Float32), num=16
        )
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        e = pe[g]
        pinv = Float32(0.0)
        if e != Float32(0.0):
            pinv = _f_div(Float32(1.0), e)
        pwords = []
        for h in cutlass.range_constexpr(2):
            nv = [_sel_eq0(e, Float32(0.0), _f_mul(sn[8 * h + t], pinv)) for t in range(8)]
            pwords.append(_pack_e2m1x8(nv))
        _st_global_v2(row_base + 8 * g, pwords[0], pwords[1])

        neg_e = _f_neg(e)
        rem = []
        for h in cutlass.range_constexpr(2):
            mg = _decode8(pwords[h])
            for t in cutlass.range_constexpr(8):
                rem.append(_f_fma(mg[t], neg_e, sn[8 * h + t]))
        rb = _tree_max([_f_abs(x) for x in rem])
        rs = _f_min(Float32(448.0), _f_mul(rb, inv_t6))
        rbyte = _pack_e4m3x2(rs, rs) & Uint32(0xFF)
        _st_global_u8(head_base + RESIDUAL_SCALE + Int64(v) * 8 + g, rbyte)
        re = _expand4(rbyte, outer_new)[0]
        rinv = Float32(0.0)
        if re != Float32(0.0):
            rinv = _f_div(Float32(1.0), re)
        rwords = []
        for h in cutlass.range_constexpr(2):
            nv = [_sel_eq0(re, Float32(0.0), _f_mul(rem[8 * h + t], rinv)) for t in range(8)]
            rwords.append(_pack_e2m1x8(nv))
        _st_global_v2(row_base + VALUES_BYTES + 8 * g, rwords[0], rwords[1])
    if tidx == 0:
        _st_outer_pad(head_base + OUTER_SCALE, outer_new)


@cute.jit
def _kda_scalar_active(
    tma_state: cutlass.GridConstant[cuda_exp.TensorMap],
    x,
    conv_state,
    conv_weight,
    gate,
    beta,
    z,
    norm_weight,
    pool,
    slot_map,
    hk,
    hu,
    hg,
    lengths,
    a_log,
    dt_bias,
    out,
    arrivals,
    slot,
    lower_bound: cutlass.Float32,
    norm_eps: cutlass.Float32,
    heads: cutlass.Constexpr,
    bounded: cutlass.Constexpr,
    pdl: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    batch = block // heads
    head = block - batch * heads

    live = _ld_keep_s32(_addr(slot_map, slot))
    pool_off = cute.crd2idx((slot, head, 0), pool.layout)
    pool_base = pool.iterator + pool_off

    length = _ld_keep_s32(_addr(lengths, live))
    commits = length == N - 1

    smem = utils.SmemAllocator()
    s_q = cutlass.Array(cutlass.BFloat16, K, space=cutlass.AddressSpace.smem, alignment=16)
    s_k = cutlass.Array(cutlass.BFloat16, K, space=cutlass.AddressSpace.smem, alignment=16)
    s_cumulative = cutlass.Array(cutlass.Float32, K, space=cutlass.AddressSpace.smem, alignment=16)
    # Four warp partials for two simultaneous reductions plus CTA scalars.
    s_red = smem.allocate_tensor(cutlass.Float32, cute.make_layout((16,)), byte_alignment=16)
    s_scalar = smem.allocate_tensor(cutlass.Float32, cute.make_layout((8,)), byte_alignment=16)
    # BF16-rounded history coefficients for K then Q.
    s_coef = cutlass.Array(cutlass.Float32, 2 * N, space=cutlass.AddressSpace.smem, alignment=16)

    mma_a = cutlass.Array(cutlass.Int8, 128 * 64, space=cutlass.AddressSpace.smem, alignment=128)
    mma_b = cutlass.Array(cutlass.Int8, 8 * 64, space=cutlass.AddressSpace.smem, alignment=128)
    mma_sfa = cutlass.Array(
        cutlass.Float8E4M3FN, 1024, space=cutlass.AddressSpace.smem, alignment=128
    )
    mma_sfb = cutlass.Array(
        cutlass.Float8E4M3FN, 1024, space=cutlass.AddressSpace.smem, alignment=128
    )
    acc_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    state_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    history_load_mbar = cutlass.Array(
        cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8
    )
    commit_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_holding = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)
    cache_hk = cutlass.Array(
        cutlass.BFloat16, N * K, space=cutlass.AddressSpace.smem, alignment=128
    )
    cache_hu = cutlass.Array(
        cutlass.BFloat16, N * K, space=cutlass.AddressSpace.smem, alignment=128
    )
    cache_hg = cutlass.Array(cutlass.Float32, N * K, space=cutlass.AddressSpace.smem, alignment=128)
    s_exp = cutlass.Array(mma_b.data_ptr(), shape=K, dtype=cutlass.Float32)

    if warp == 0:
        if prims.elect_sync():
            prims.mbarrier_init(acc_mbar, 1)
            prims.mbarrier_init(state_mbar, 1)
            prims.mbarrier_init(history_load_mbar, 1)
            prims.mbarrier_init(commit_mbar, 1)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)
    if warp == 0:
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(state_mbar, tma_state.global_tx_bytes())
            prims.cp_async_bulk_tensor_shared_cta_global(
                mma_a,
                tma_state.get_ptr(),
                (
                    cutlass.Int32(0),
                    cutlass.Int32(0),
                    head,
                    slot,
                ),
                state_mbar,
            )
            if length > 0:
                # Only accepted rows: the window is contiguous per (row, head).
                prims.mbarrier_arrive_expect_tx(history_load_mbar, length * (K * 8))
                row = cute.crd2idx((live, head, 0, 0), hk.layout)
                prims.cp_async_bulk_shared_cluster_global(
                    cache_hk, hk.iterator.raw_ptr() + row, history_load_mbar, length * (K * 2)
                )
                prims.cp_async_bulk_shared_cluster_global(
                    cache_hu, hu.iterator.raw_ptr() + row, history_load_mbar, length * (K * 2)
                )
                prims.cp_async_bulk_shared_cluster_global(
                    cache_hg, hg.iterator.raw_ptr() + row, history_load_mbar, length * (K * 4)
                )
            if commits:
                # Residual values and both scale rows, read only by the commit.
                prims.cp_async_bulk_prefetch(
                    pool_base.raw_ptr() + VALUES_BYTES, OUTER_SCALE - VALUES_BYTES
                )
    # A committing CTA holds a full 128x128 FP32 accumulator.
    tmem_cols = cutlass.Int32(32)
    if commits:
        tmem_cols = cutlass.Int32(128)
    if warp == 0:
        prims.tcgen05_alloc(tmem_holding, tmem_cols, group="cta_1")
    prims.barrier_cta_sync(0)
    tmem_raw = tmem_holding.load()
    prims.tcgen05_relinquish_alloc_permit(group="cta_1")

    # Loads that need at most the slot are issued ahead of the
    # slot -> live -> length chain and of every barrier.
    width = heads * K
    q_channel = head * K + tidx
    k_channel = width + head * K + tidx
    v_channel = 2 * width + head * K + tidx
    norm_bf16 = norm_weight[tidx]
    dt_value = dt_bias[head * K + tidx]
    a_log_value = a_log[head]
    state_sf = cute.recast_ptr(pool_base + PRIMARY_SCALE, dtype=cutlass.Int32)
    state_scale_words = (state_sf.raw_ptr() + tidx * 2).load(count=2, alignment=8)
    state_outer = cute.recast_ptr(pool_base + OUTER_SCALE, dtype=cutlass.Float32)[0]
    state_raw = conv_state.iterator.raw_ptr()
    q_state_ptr = state_raw + cute.crd2idx((slot, q_channel, 0), conv_state.layout)
    k_state_ptr = state_raw + cute.crd2idx((slot, k_channel, 0), conv_state.layout)
    v_state_ptr = state_raw + cute.crd2idx((slot, v_channel, 0), conv_state.layout)
    q_state = q_state_ptr.load(count=3, alignment=2)
    k_state = k_state_ptr.load(count=3, alignment=2)
    v_state = v_state_ptr.load(count=3, alignment=2)
    weight_raw = conv_weight.iterator.raw_ptr()
    wq = (weight_raw + q_channel * 4).load(count=4, alignment=8)
    wk = (weight_raw + k_channel * 4).load(count=4, alignment=8)
    wv = (weight_raw + v_channel * 4).load(count=4, alignment=8)

    prev_g_prefetch = cutlass.Float32(0.0)
    if length > 0:
        prev_g_prefetch = hg[(live, head, length - 1, tidx)]
    q0, q1, q2 = q_state[0], q_state[1], q_state[2]
    k0, k1, k2 = k_state[0], k_state[1], k_state[2]
    v0, v1, v2 = v_state[0], v_state[1], v_state[2]

    # Clear both SFB atoms (the K/Q scales are written after the next barrier).
    sfb_words = cutlass.inttoptr(
        mma_sfb.subview(tidx * 8).data_ptr().toint(), cutlass.AddressSpace.smem, cutlass.Int32
    )
    sfb_words.store(
        cutlass.Vector.from_elements((cutlass.Int32(0), cutlass.Int32(0)), dtype=cutlass.Int32),
        alignment=8,
    )

    if cutlass.const_expr(pdl):
        prims.griddepcontrol("wait")

    # Conv4 uses the old window, then publishes [old1, old2, raw].
    raw_q = x[(batch, q_channel)]
    raw_k = x[(batch, k_channel)]
    raw_v = x[(batch, v_channel)]
    z_bf16 = z[(batch, head, tidx)]
    beta_bf16 = beta[(batch, head)]
    gate_bf16 = gate[(batch, head * K + tidx)]

    aq = cute.fma(wq[0].to(cutlass.Float32), q0.to(cutlass.Float32), cutlass.Float32(0.0))
    aq = cute.fma(wq[1].to(cutlass.Float32), q1.to(cutlass.Float32), aq)
    aq = cute.fma(wq[2].to(cutlass.Float32), q2.to(cutlass.Float32), aq)
    aq = cute.fma(wq[3].to(cutlass.Float32), raw_q.to(cutlass.Float32), aq)
    ak = cute.fma(wk[0].to(cutlass.Float32), k0.to(cutlass.Float32), cutlass.Float32(0.0))
    ak = cute.fma(wk[1].to(cutlass.Float32), k1.to(cutlass.Float32), ak)
    ak = cute.fma(wk[2].to(cutlass.Float32), k2.to(cutlass.Float32), ak)
    ak = cute.fma(wk[3].to(cutlass.Float32), raw_k.to(cutlass.Float32), ak)
    av = cute.fma(wv[0].to(cutlass.Float32), v0.to(cutlass.Float32), cutlass.Float32(0.0))
    av = cute.fma(wv[1].to(cutlass.Float32), v1.to(cutlass.Float32), av)
    av = cute.fma(wv[2].to(cutlass.Float32), v2.to(cutlass.Float32), av)
    av = cute.fma(wv[3].to(cutlass.Float32), raw_v.to(cutlass.Float32), av)

    aq = aq * cute.rcp(cutlass.Float32(1.0) + cute.exp(-aq, fastmath=True))
    ak = ak * cute.rcp(cutlass.Float32(1.0) + cute.exp(-ak, fastmath=True))
    av = av * cute.rcp(cutlass.Float32(1.0) + cute.exp(-av, fastmath=True))
    qf = aq.to(cutlass.BFloat16).to(cutlass.Float32)
    kf = ak.to(cutlass.BFloat16).to(cutlass.Float32)
    v_bf16 = av.to(cutlass.BFloat16)
    # Each warp reduces its own 32 keys; lane 0 publishes the warp partial.
    q_partial = _warp_sum_squares(qf)
    k_partial = _warp_sum_squares(kf)
    if lane == 0:
        s_red[8 + warp] = q_partial
        s_red[12 + warp] = k_partial

    q_state_ptr.store(
        cutlass.Vector.from_elements((q1, q2, raw_q), dtype=cutlass.BFloat16),
        alignment=2,
    )
    k_state_ptr.store(
        cutlass.Vector.from_elements((k1, k2, raw_k), dtype=cutlass.BFloat16),
        alignment=2,
    )
    v_state_ptr.store(
        cutlass.Vector.from_elements((v1, v2, raw_v), dtype=cutlass.BFloat16),
        alignment=2,
    )
    cute.arch.barrier()

    qr = (s_red[8] + s_red[10]) + (s_red[9] + s_red[11])
    kr = (s_red[12] + s_red[14]) + (s_red[13] + s_red[15])
    qden = cutlass.Float32(0.0)
    kden = cutlass.Float32(0.0)
    if lane == 0:
        qden = cute.sqrt(qr + cutlass.Float32(1.0e-6), approx=True, ftz=True)
        kden = cute.sqrt(kr + cutlass.Float32(1.0e-6), approx=True, ftz=True)
    qden = cute.arch.shuffle_sync(qden, 0)
    kden = cute.arch.shuffle_sync(kden, 0)

    qn = cute.div(qf * cutlass.Float32(0.08838834764831845), qden, full=True).to(cutlass.BFloat16)
    kn = cute.div(kf, kden, full=True).to(cutlass.BFloat16)
    s_q[tidx] = qn
    s_k[tidx] = kn

    prev_g = prev_g_prefetch
    shifted = gate_bf16.to(cutlass.Float32) + dt_value
    a = cute.exp(a_log_value, fastmath=True)
    step = cutlass.Float32(0.0)
    if bounded:
        step = lower_bound * cute.div(
            cutlass.Float32(1.0),
            cutlass.Float32(1.0) + cute.exp(-(a * shifted), fastmath=True),
            full=True,
        )
    else:
        softplus = shifted
        if shifted <= cutlass.Float32(20.0):
            softplus = cute.log(
                cutlass.Float32(1.0) + cute.exp(shifted, fastmath=True),
                fastmath=True,
            )
        step = -a * softplus
    cumulative = prev_g + step
    s_cumulative[tidx] = cumulative
    decay = cute.exp(cumulative, fastmath=True)
    dyn_k = kn.to(cutlass.Float32) * decay
    dyn_q = qn.to(cutlass.Float32) * decay

    # Shared outer scale and kn.qn dot in one block reduction.
    local_max = cute.arch.fmax(cute.abs(dyn_k), cute.abs(dyn_q))
    wmax = cute.arch.warp_reduction_max(local_max)
    wdot = _warp_sum_products(kn.to(cutlass.Float32), qn.to(cutlass.Float32))
    if lane == 0:
        s_red[warp] = wmax
        s_red[4 + warp] = wdot
    cute.arch.barrier()
    if warp == 0:
        mr = cutlass.Float32(0.0)
        dr = cutlass.Float32(0.0)
        if lane < 4:
            mr = s_red[lane]
            dr = s_red[4 + lane]
        mr = cute.arch.warp_reduction_max(mr)
        dr = cute.arch.warp_reduction_sum(dr)
        if lane == 0:
            s_scalar[2] = mr * cutlass.Float32(1.0 / (6.0 * 448.0))
            s_scalar[3] = dr
    cute.arch.barrier()

    outer = s_scalar[2]
    safe_outer = outer
    if outer == cutlass.Float32(0.0):
        safe_outer = cutlass.Float32(1.0)
    # Each half-warp is one 16-element FP4 scale group.
    group_max_k = cute.arch.warp_reduction_max(cute.abs(dyn_k), threads_in_group=16)
    group_max_q = cute.arch.warp_reduction_max(cute.abs(dyn_q), threads_in_group=16)
    sfk_raw = cute.div(group_max_k, cutlass.Float32(6.0) * safe_outer, full=True)
    sfq_raw = cute.div(group_max_q, cutlass.Float32(6.0) * safe_outer, full=True)
    sfk_e4 = cute.arch.fmin(sfk_raw, cutlass.Float32(448.0)).to(cutlass.Float8E4M3FN)
    sfq_e4 = cute.arch.fmin(sfq_raw, cutlass.Float32(448.0)).to(cutlass.Float8E4M3FN)
    if (lane % 16) == 0:
        group = warp * 2 + lane // 16
        sf_offset_k = (group // 4) * 512 + (group % 4)
        sf_offset_q = sf_offset_k + 16
        mma_sfb.subview(sf_offset_k).store(sfk_e4)
        mma_sfb.subview(sf_offset_q).store(sfq_e4)

    sfk = sfk_e4.to(cutlass.Float32)
    sfq = sfq_e4.to(cutlass.Float32)
    denom_k = sfk * outer
    denom_q = sfq * outer
    normalized_k = cutlass.Float32(0.0)
    normalized_q = cutlass.Float32(0.0)
    if denom_k != cutlass.Float32(0.0):
        normalized_k = cute.div(dyn_k, denom_k, full=True)
    if denom_q != cutlass.Float32(0.0):
        normalized_q = cute.div(dyn_q, denom_q, full=True)

    # Native NVFP4 checkpoint projection. Operands use an s128b K-major
    # image padded privately to 256 logical K; only the first 128 K values
    # and two of the hardware-minimum eight N rows participate.
    swizzle_s64 = cutlass.Swizzle.from_name("s64b")

    # Dynamic K/Q pair: exact RNE satfinite E2M1x2, earlier K in low nibble.
    e2m1_type = mlir_ir.TypeAttr.get(cutlass.Float4E2M1FN.mlir_type)
    odd_k = cute.arch.shuffle_sync_down(normalized_k, 1)
    odd_q = cute.arch.shuffle_sync_down(normalized_q, 1)
    if (lane % 2) == 0:
        packed_k_ir = prims.dialect.convert_f32x2_to_f4x2(odd_k, normalized_k, e2m1_type)
        packed_q_ir = prims.dialect.convert_f32x2_to_f4x2(odd_q, normalized_q, e2m1_type)
        byte = tidx // 2
        (mma_b.subview(byte)).data_ptr().apply_swizzle(swizzle_s64).store(cutlass.Int8(packed_k_ir))
        (mma_b.subview(64 + byte)).data_ptr().apply_swizzle(swizzle_s64).store(
            cutlass.Int8(packed_q_ir)
        )
    if tidx < 24:
        # Private N padding rows 2..7 (bytes 128..511, any swizzle) must be true zero.
        zero = cutlass.Int32(0)
        cutlass.inttoptr(
            mma_b.subview(128 + tidx * 16).data_ptr().toint(),
            cutlass.AddressSpace.smem,
            cutlass.Int32,
        ).store(
            cutlass.Vector.from_elements((zero, zero, zero, zero), dtype=cutlass.Int32),
            alignment=16,
        )
    # Repack plain row-major E4M3 scales into two 128x4 scale atoms: each
    # row's four scales of one 64-key block are one word.
    for kblock in cutlass.range_constexpr(2):
        sf_offset = kblock * 512 + (tidx % 32) * 16 + (tidx // 32) * 4
        cutlass.inttoptr(
            mma_sfa.subview(sf_offset).data_ptr().toint(), cutlass.AddressSpace.smem, cutlass.Int32
        ).store(cutlass.Vector.from_elements((state_scale_words[kblock],), dtype=cutlass.Int32))
    while not prims.mbarrier_try_wait_parity(state_mbar, cutlass.Int32(0)):
        pass
    cute.arch.fence_view_async_shared()
    prims.barrier_cta_sync(0)

    if warp == 0:
        idesc = prims.Tcgen05MxOmmaInstrDesc.build(
            a_dtype=cutlass.Float4E2M1FN,
            b_dtype=cutlass.Float4E2M1FN,
            scale_format=0,
            n_dim=8,
            m_dim=128,
        )
        desc_a = prims.Tcgen05SmemDesc.build(
            mma_a,
            leading_byte_offset=0,
            stride_byte_offset=512,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
        )
        desc_b = prims.Tcgen05SmemDesc.build(
            mma_b,
            leading_byte_offset=0,
            stride_byte_offset=512,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
        )
        desc_sfa = prims.Tcgen05SmemDesc.build(
            mma_sfa,
            leading_byte_offset=16,
            stride_byte_offset=128,
            layout=0,
        )
        desc_sfb = prims.Tcgen05SmemDesc.build(
            mma_sfb,
            leading_byte_offset=16,
            stride_byte_offset=128,
            layout=0,
        )
        base_col = tmem_raw & cutlass.Int32(0xFFFF)
        base_row = tmem_raw >> cutlass.Int32(16)
        acc_addr = (base_row << cutlass.Int32(16)) | base_col
        sfa_addr = (base_row << cutlass.Int32(16)) | (base_col + cutlass.Int32(8))
        sfb_addr = (base_row << cutlass.Int32(16)) | (base_col + cutlass.Int32(16))
        acc_ptr = cutlass.inttoptr(acc_addr, 6, cutlass.Float32)
        sf_shape, sf_multicast = prims.S2TCopyMode.S2T_32x128b_WARPX4
        if prims.elect_sync():
            for kblock in cutlass.range_constexpr(2):
                sfa_ptr = cutlass.inttoptr(sfa_addr + kblock * 4, 6, cutlass.Int32)
                sfb_ptr = cutlass.inttoptr(sfb_addr + kblock * 4, 6, cutlass.Int32)
                prims.tcgen05_cp(
                    sf_shape,
                    sfa_ptr,
                    desc_sfa.advance_start_address(kblock * 512),
                    multicast=sf_multicast,
                )
                prims.tcgen05_cp(
                    sf_shape,
                    sfb_ptr,
                    desc_sfb.advance_start_address(kblock * 512),
                    multicast=sf_multicast,
                )
            for kblock in cutlass.range_constexpr(2):
                prims.tcgen05_mma_block_scale(
                    prims.MMABlockScaleKind.MXF4NVF4,
                    prims.CTAGroup.CTA_1,
                    acc_ptr,
                    desc_a.advance_start_address(kblock * 32),
                    desc_b.advance_start_address(kblock * 32),
                    idesc,
                    enable_input_d=(kblock != 0),
                    scale_a=cutlass.inttoptr(sfa_addr + kblock * 4, 6, cutlass.Int32),
                    scale_b=cutlass.inttoptr(sfb_addr + kblock * 4, 6, cutlass.Int32),
                    scale_vec_size=prims.Tcgen05MMABlockScale.BLOCK16,
                )
            prims.tcgen05_commit(acc_mbar, group=prims.CTAGroup.CTA_1)

    # History coefficients on the CUDA cores while the native MMA runs:
    # c[i] = adjusted_bf16[i] . (kn, qn) with FP32 sums, rounded to BF16,
    # then hist[v] = U[:, v] . c with FP32 accumulation.
    hist_k = cutlass.Float32(0.0)
    hist_q = cutlass.Float32(0.0)
    if length > 0:
        while not prims.mbarrier_try_wait_parity(history_load_mbar, cutlass.Int32(0)):
            pass
        # Eight threads per history row, sixteen keys each.
        hist_row = tidx // 8
        key0 = (tidx % 8) * 16
        dot_k = cutlass.Float32(0.0)
        dot_q = cutlass.Float32(0.0)
        if hist_row < length:
            base = hist_row * K + key0
            cached_k = [
                (cache_hk.data_ptr() + base + 8 * c).load(count=8, alignment=16) for c in range(2)
            ]
            cached_g = [
                (cache_hg.data_ptr() + base + 4 * c).load(count=4, alignment=16) for c in range(4)
            ]
            cum = [
                (s_cumulative.data_ptr() + key0 + 4 * c).load(count=4, alignment=16)
                for c in range(4)
            ]
            k_pair = [(s_k.data_ptr() + key0 + 8 * c).load(count=8, alignment=16) for c in range(2)]
            q_pair = [(s_q.data_ptr() + key0 + 8 * c).load(count=8, alignment=16) for c in range(2)]
            for j in cutlass.range_constexpr(16):
                adjusted = (
                    (
                        cached_k[j // 8][j % 8].to(cutlass.Float32)
                        * cute.exp(cum[j // 4][j % 4] - cached_g[j // 4][j % 4], fastmath=True)
                    )
                    .to(cutlass.BFloat16)
                    .to(cutlass.Float32)
                )
                dot_k = cute.fma(adjusted, k_pair[j // 8][j % 8].to(cutlass.Float32), dot_k)
                dot_q = cute.fma(adjusted, q_pair[j // 8][j % 8].to(cutlass.Float32), dot_q)
        for offset in (4, 2, 1):
            dot_k = dot_k + cute.arch.shuffle_sync_bfly(dot_k, offset)
            dot_q = dot_q + cute.arch.shuffle_sync_bfly(dot_q, offset)
        if (tidx % 8) == 0:
            s_coef[hist_row] = dot_k.to(cutlass.BFloat16).to(cutlass.Float32)
            s_coef[N + hist_row] = dot_q.to(cutlass.BFloat16).to(cutlass.Float32)
        cute.arch.barrier()
        for hist in cutlass.range(length, unroll=4):
            cached_u = cache_hu[hist * K + tidx].to(cutlass.Float32)
            hist_k = cute.fma(cached_u, s_coef[hist], hist_k)
            hist_q = cute.fma(cached_u, s_coef[N + hist], hist_q)

    while not prims.mbarrier_try_wait_parity(acc_mbar, cutlass.Int32(0)):
        pass
    base_col = tmem_raw & cutlass.Int32(0xFFFF)
    base_row = tmem_raw >> cutlass.Int32(16)
    tmem_row = base_row + warp * cutlass.Int32(32)
    tmem_addr = (tmem_row << cutlass.Int32(16)) | base_col
    checkpoint_vec = prims.tcgen05_ld(
        "32x32b",
        cutlass.inttoptr(tmem_addr, 6, cutlass.Float32),
        num=8,
    )
    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    checkpoint_rescale = state_outer * outer
    cp_k = checkpoint_vec[0] * checkpoint_rescale
    cp_q = checkpoint_vec[1] * checkpoint_rescale

    beta_v = beta_bf16.to(cutlass.Float32)
    beta_sigmoid = cute.div(
        cutlass.Float32(1.0),
        cutlass.Float32(1.0) + cute.exp(-beta_v, fastmath=True),
        full=True,
    )
    update = beta_sigmoid * (v_bf16.to(cutlass.Float32) - cp_k - hist_k)
    core = cp_q + hist_q + update * s_scalar[3]
    core_bf16 = core.to(cutlass.BFloat16)
    core_f = core_bf16.to(cutlass.Float32)
    rms_partial = _warp_sum_squares(core_f)
    if lane == 0:
        s_red[8 + warp] = rms_partial

    hk[(live, head, length, tidx)] = kn
    hu[(live, head, length, tidx)] = update.to(cutlass.BFloat16)
    hg[(live, head, length, tidx)] = cumulative
    cute.arch.barrier()

    # RMSNorm is intentionally after the BF16 core boundary.
    total = (s_red[8] + s_red[10]) + (s_red[9] + s_red[11])
    rstd = cutlass.Float32(0.0)
    if lane == 0:
        variance = cute.div(total, cutlass.Float32(128.0), full=True)
        rstd = cute.div(
            cutlass.Float32(1.0),
            cute.sqrt(
                variance + norm_eps,
                approx=True,
                ftz=True,
            ),
            full=True,
        )
    rstd = cute.arch.shuffle_sync(rstd, 0)
    zv = z_bf16.to(cutlass.Float32)
    zg = cute.div(
        cutlass.Float32(1.0),
        cutlass.Float32(1.0) + cute.exp(-zv, fastmath=True),
        full=True,
    )
    result = core_f * rstd * norm_bf16.to(cutlass.Float32) * zg
    out[(batch, head, tidx)] = result.to(cutlass.BFloat16)
    if cutlass.const_expr(pdl):
        prims.griddepcontrol("launch_dependents")

    if commits:
        _commit_window(
            tidx,
            warp,
            lane,
            tmem_raw,
            Int64(pool_base.toint()),
            state_outer,
            cumulative,
            kn,
            update,
            mma_a,
            cache_hk,
            cache_hg,
            cache_hu,
            s_exp,
            s_red,
            commit_mbar,
        )
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)

    prims.barrier_cta_sync(0)
    if warp == 0:
        prims.tcgen05_dealloc(
            cutlass.inttoptr(tmem_raw, 6, cutlass.Float32), tmem_cols, group="cta_1"
        )
    # Every CTA of this row consumed its length before arriving (relaxed
    # suffices); the last one to arrive advances it.
    if tidx == 0:
        arrived = prims.atomicrmw(
            "add",
            arrivals.iterator.raw_ptr() + batch,
            cutlass.Int32(1),
            mem_order="relaxed",
            syncscope="gpu",
        )
        if arrived == heads - 1:
            _st_keep_s32(_addr(arrivals, batch), cutlass.Int32(0))
            next_length = length + 1
            if commits:
                next_length = cutlass.Int32(0)
            _st_keep_s32(_addr(lengths, live), next_length)


@cute.kernel
def _kda_scalar_kernel(
    tma_state: cutlass.GridConstant[cuda_exp.TensorMap],
    x,
    conv_state,
    conv_weight,
    gate,
    beta,
    z,
    norm_weight,
    pool,
    slots,
    dummy,
    slot_map,
    hk,
    hu,
    hg,
    lengths,
    a_log,
    dt_bias,
    out,
    arrivals,
    lower_bound: cutlass.Float32,
    norm_eps: cutlass.Float32,
    heads: cutlass.Constexpr,
    bounded: cutlass.Constexpr,
    pdl: cutlass.Constexpr,
):
    tidx, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    batch = block // heads
    # Fetched with the dummy flag; only live rows dereference it.
    slot = _ld_keep_s32(_addr(slots, batch))
    if _ld_keep_u8(_addr(dummy, batch)) != cutlass.Int32(0):
        head = block - batch * heads
        if cutlass.const_expr(pdl):
            prims.griddepcontrol("wait")
        out[(batch, head, tidx)] = cutlass.BFloat16(0.0)
    else:
        _kda_scalar_active(
            tma_state,
            x,
            conv_state,
            conv_weight,
            gate,
            beta,
            z,
            norm_weight,
            pool,
            slot_map,
            hk,
            hu,
            hg,
            lengths,
            a_log,
            dt_bias,
            out,
            arrivals,
            slot,
            lower_bound,
            norm_eps,
            heads,
            bounded,
            pdl,
        )


@cute.jit
def _launch_scalar(
    x,
    conv_state,
    conv_weight,
    gate,
    beta,
    z,
    norm_weight,
    pool,
    slots,
    dummy,
    slot_map,
    hk,
    hu,
    hg,
    lengths,
    a_log,
    dt_bias,
    out,
    arrivals,
    lower_bound: cutlass.Float32,
    norm_eps: cutlass.Float32,
    heads: cutlass.Constexpr,
    bounded: cutlass.Constexpr,
    pdl: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    primary_layout = cute.make_layout(
        (64, 128, heads, pool.shape[0]),
        stride=(1, 64, STATE_BYTES, pool.layout.stride[0]),
    )
    primary_tensor = cute.make_tensor(pool.iterator, primary_layout)
    tma_state = cuda_exp.create_tensor_map_tiled_from_view(
        primary_tensor,
        box_dims=(64, 128, 1, 1),
        stride_order=(0, 1, 2, 3),
        swizzle=cuda_exp.TensorMapSwizzle.s64b,
    )
    kernel = _kda_scalar_kernel(
        tma_state,
        x,
        conv_state,
        conv_weight,
        gate,
        beta,
        z,
        norm_weight,
        pool,
        slots,
        dummy,
        slot_map,
        hk,
        hu,
        hg,
        lengths,
        a_log,
        dt_bias,
        out,
        arrivals,
        lower_bound,
        norm_eps,
        heads,
        bounded,
        pdl,
    )
    kernel.launch(
        grid=[x.shape[0] * heads, 1, 1],
        block=[THREADS, 1, 1],
        stream=stream,
        use_pdl=pdl,
    )


def _fake_tensor_args(heads: int) -> tuple:
    """Compile once per stable H while B/S/R and all padded strides stay runtime."""
    b = cute.sym_int32(divisibility=1)
    s = cute.sym_int32(divisibility=1)
    r = cute.sym_int32(divisibility=1)
    a = cute.sym_int32(divisibility=1)

    def row(divisibility: int = 1) -> cute.SymInt:
        return cute.sym_int64(divisibility=divisibility)

    width = heads * K
    channels = 3 * width
    hist_row = heads * N * K
    return (
        make_fake_tensor(cutlass.BFloat16, (b, channels), stride=(row(16), 1)),
        make_fake_tensor(cutlass.BFloat16, (s, channels, 3), stride=(row(16), 3, 1)),
        make_fake_tensor(cutlass.BFloat16, (channels, 4), stride=(4, 1)),
        make_fake_tensor(cutlass.BFloat16, (b, width), stride=(row(8), 1)),
        make_fake_tensor(cutlass.BFloat16, (b, heads), stride=(row(1), 1)),
        make_fake_tensor(cutlass.BFloat16, (b, heads, K), stride=(row(8), K, 1)),
        make_fake_tensor(cutlass.BFloat16, (K,), stride=(1,)),
        make_fake_tensor(cutlass.Uint8, (s, heads, STATE_BYTES), stride=(row(16), STATE_BYTES, 1)),
        make_fake_tensor(cutlass.Int32, (b,), stride=(1,)),
        make_fake_tensor(cutlass.Boolean, (b,), stride=(1,)),
        make_fake_tensor(cutlass.Int32, (s,), stride=(1,)),
        make_fake_tensor(cutlass.BFloat16, (r, heads, N, K), stride=(hist_row, N * K, K, 1)),
        make_fake_tensor(cutlass.BFloat16, (r, heads, N, K), stride=(hist_row, N * K, K, 1)),
        make_fake_tensor(cutlass.Float32, (r, heads, N, K), stride=(hist_row, N * K, K, 1)),
        make_fake_tensor(cutlass.Int32, (r,), stride=(1,)),
        make_fake_tensor(cutlass.Float32, (heads,), stride=(1,)),
        make_fake_tensor(cutlass.Float32, (width,), stride=(1,)),
        make_fake_tensor(cutlass.BFloat16, (b, heads, K), stride=(width, K, 1)),
        make_fake_tensor(cutlass.Int32, (a,), stride=(1,)),
    )


@torch.no_grad()
def fused_replay(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weight: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    pool: torch.Tensor,
    slots: torch.Tensor,
    dummy: torch.Tensor,
    slot_map: torch.Tensor,
    hk: torch.Tensor,
    hu: torch.Tensor,
    hg: torch.Tensor,
    lengths: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    norm_eps: float,
    out: torch.Tensor,
    arrivals: torch.Tensor,
) -> None:
    """Write BF16 [B,H,128] output, append or commit the window, and advance lengths.

    ``arrivals`` holds at least B zero-initialized int32 counters; every
    launch returns them to zero.
    """
    heads = a_log.numel()
    bounded = lower_bound is not None
    lower_bound = 0.0 if lower_bound is None else lower_bound
    pdl = get_env_enable_pdl()
    key = (heads, bounded, pdl)
    tensor_args = (
        x,
        conv_state,
        conv_weight,
        gate,
        beta,
        z,
        norm_weight,
        pool,
        slots,
        dummy,
        slot_map,
        hk,
        hu,
        hg,
        lengths,
        a_log,
        dt_bias,
        out,
        arrivals,
    )
    if key not in _compiled:
        compile_args = _fake_tensor_args(heads)
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        _compiled[key] = cute.compile(
            _launch_scalar,
            *compile_args,
            float(lower_bound),
            float(norm_eps),
            heads,
            bounded,
            pdl,
            stream,
            options="--enable-tvm-ffi",
        )
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    _compiled[key](*tensor_args, float(lower_bound), float(norm_eps), stream)
