# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP8 decode projection with its weights loaded before the PDL wait, bitwise equal to cuBLAS' nvjet
(``cublas_scaled_mm``): ``y[M, N] = bf16((x[M, K] @ W[N, K].T) * (sa * sb))`` in ``cta_group::2`` tcgen05 MMAs over
the whole K in order. Behind a producer that releases its dependents early, most weight K tiles are already in shared
memory when ``griddepcontrol.wait`` returns. Conv mode (``conv=``) also applies the GDN causal conv1d update of an MTP
target-verify step to the first output columns, bitwise as Triton's ``causal_conv1d_update`` (``supports_conv``).
"""

from __future__ import annotations

from typing import NamedTuple

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda_exp
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.experimental import primitives as prims

from ...flashinfer_utils import get_env_enable_pdl
from .low_m_mma_gemm import _gdn_conv_silu
from .moe_decode_front import _ld_f32

K_BOX = 128  # E4M3 per 128-byte TMA row (s128b swizzle)
K_GRANULE = 32  # K per tcgen05 f8f6f4 MMA
FEAT_PAIR = 128  # collective MMA M
FEAT_CTA = 64
W_TILE = FEAT_CTA * K_BOX  # 8 KB
SW_ALIGN = 1024
TMA_WARP, MMA_WARP, EPI_WARP0 = 0, 1, 2
TCOLS = 32
XG = 8  # K tiles per x TMA box / barrier
# x + weight slots + barriers per CTA: with its 1 KB reserved a CTA stays within the 196 KB shared-memory split beside
# three 512 B all-reduce CTAs, so it can start early on an SM that has not drained (an SM changes its split only idle).
SMEM_BUDGET = 190 * 1024
MAX_ROWS = 128


@dsl_user_op
def _mul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_bf16(addr: Int64, v: Float32, off: int = 0, *, loc=None, ip=None) -> None:
    """bf16(v) (round to nearest even) to [addr + off]; ``off`` is a compile-time byte offset folded into the store."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Float32(v).ir_value(loc=loc, ip=ip)],
        f"{{\n\t.reg .b16 h;\n\tcvt.rn.bf16.f32 h, $1;\n\tst.global.b16 [$0+{off}], h;\n\t}}",
        "l,f,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_bf16_exact(addr: Int64, v: Float32, off: int = 0, *, loc=None, ip=None) -> None:
    """An FP32 value that is exactly a BF16 value, stored as its high half to [addr + off] (no conversion; the same
    bits as rounding it)."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Float32(v).ir_value(loc=loc, ip=ip)],
        f"{{\n\t.reg .b32 r;\n\t.reg .b16 l, h;\n\tmov.b32 r, $1;\n\tmov.b32 {{l, h}}, r;\n\t"
        f"st.global.b16 [$0+{off}], h;\n\t}}",
        "l,f,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ld_bf16(addr: Int64, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 h, z;\n\tmov.b16 z, 0;\n\tld.global.b16 h, [$1];\n\t"
            "mov.b32 $0, {z, h};\n\t}",
            "=f,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_s32(addr: Int64, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.s32 $0, [$1];",
            "=r,l,~{memory}",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16r(v: Float32, *, loc=None, ip=None) -> Float32:
    """v rounded to BF16 (round to nearest even), returned as the exact FP32 value."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(v).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 b, z;\n\tmov.b16 z, 0;\n\tcvt.rn.bf16.f32 b, $1;\n\t"
            "mov.b32 $0, {z, b};\n\t}",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _resident(kt: int, tn: int, budget: int = SMEM_BUDGET) -> int:
    """Weight K tiles kept in smem (the rest stream through the same slots)."""
    return min(kt, (budget - kt * (tn // 2) * K_BOX - 2048) // W_TILE)


def _covers(k: int, n: int) -> bool:
    """Whole x TMA boxes of ``XG`` K tiles, whole 128-feature CTA pairs."""
    return k > 0 and k % (XG * K_BOX) == 0 and n > 0 and n % FEAT_PAIR == 0


def plan(m: int, k: int, n: int, budget: int = SMEM_BUDGET):
    """(token_tiles, tile_rows, TN, resident weight tiles) or None when the kernel does not cover the shape."""
    if not _covers(k, n) or m < 1 or m > MAX_ROWS:
        return None
    tiles = 1 if m <= 32 else (2 if m <= 64 else 3)
    ts = -(-m // tiles)
    tn = 32 if ts <= 32 else 48
    r_res = _resident(k // K_BOX, tn, budget)
    if r_res < 1:  # no weight slot to stream through
        return None
    return tiles, ts, tn, r_res


class ConvUpdate(NamedTuple):
    """Conv mode's operands (GDN in_proj_qkvz of an MTP target-verify step): ``weight`` [C, 4] BF16, ``state`` [slots,
    C, 3] BF16 (any strides), ``slots`` [R] int32 (negative = padding: no window / state writes), ``windows`` [>= R,
    >= tokens, C, 3] BF16, ``tokens`` per request."""

    weight: torch.Tensor
    state: torch.Tensor
    slots: torch.Tensor
    windows: torch.Tensor
    tokens: int


def _operands_ok(x: torch.Tensor, w: torch.Tensor) -> bool:
    """E4M3 x [M, K] and w [N, K], contiguous, at the compiled kernel's 128-byte alignment."""
    return (
        x.dtype == torch.float8_e4m3fn
        and w.dtype == torch.float8_e4m3fn
        and x.dim() == 2
        and w.dim() == 2
        and x.shape[1] == w.shape[1]
        and x.is_contiguous()
        and w.is_contiguous()
        and x.data_ptr() % 128 == 0
        and w.data_ptr() % 128 == 0
    )


def supports(x: torch.Tensor, w: torch.Tensor, budget: int = SMEM_BUDGET) -> bool:
    """Whether :func:`resident_fp8_gemm` takes x [M, K] and w [N, K] (``_operands_ok``) at a shape :func:`plan`
    covers."""
    return _operands_ok(x, w) and plan(x.shape[0], x.shape[1], w.shape[0], budget) is not None


def plan_conv(m: int, qt: int, k: int, n: int, budget: int = SMEM_BUDGET):
    """Conv mode (GDN in_proj_qkvz of an MTP target-verify step, m = requests x qt rows): (token_tiles, tile_rows, TN,
    resident weight tiles, rows per CTA half) with whole requests in every CTA half (each epilogue thread owns a
    channel's tokens of TN / 2 // qt requests; TN 32 while three token tiles suffice, else 48), or None."""
    if not _covers(k, n) or qt < 1 or qt > 16 or m % qt:
        return None
    for tn in (32, 48):
        rph = (tn // 2) // qt
        tiles = -(-(m // qt) // (2 * rph))
        if tiles <= 3:
            r_res = _resident(k // K_BOX, tn, budget)
            return (tiles, 2 * rph * qt, tn, r_res, rph * qt) if r_res >= 1 else None
    return None


def supports_conv(x: torch.Tensor, w: torch.Tensor, conv: ConvUpdate) -> bool:
    """Whether :func:`resident_fp8_gemm` takes x and w (``_operands_ok``) in conv mode: dtypes and shapes of
    ``conv``, rows = requests x tokens, whole 128-channel feature pairs inside the output columns; callers fall back
    to their unfused path otherwise."""
    if not _operands_ok(x, w):
        return False
    m, k = x.shape
    n = w.shape[0]
    qt = conv.tokens
    cw, cs, win, sl = conv.weight, conv.state, conv.windows, conv.slots
    c_dim = cw.shape[0]
    return (
        plan_conv(m, qt, k, n) is not None
        and cw.dtype == torch.bfloat16
        and cw.is_contiguous()
        and cw.dim() == 2
        and cw.shape[1] == 4
        and c_dim % FEAT_PAIR == 0
        and c_dim <= n
        and cs.dtype == torch.bfloat16
        and cs.dim() == 3
        and cs.shape[1] == c_dim
        and cs.shape[2] == 3
        and win.dtype == torch.bfloat16
        and win.dim() == 4
        and win.shape[0] >= m // qt
        and win.shape[1] >= qt
        and win.shape[2] == c_dim
        and win.shape[3] == 3
        and sl.dtype == torch.int32
        and sl.dim() == 1
        and sl.is_contiguous()
        and sl.shape[0] == m // qt
    )


@cute.kernel
def _kernel(
    tma_x_desc: cutlass.GridConstant[cuda_exp.TensorMap],
    tma_w_desc: cutlass.GridConstant[cuda_exp.TensorMap],
    gSa: cute.Tensor,
    gSb: cute.Tensor,
    gOut: cute.Tensor,
    gCw: cute.Tensor,
    gCs: cute.Tensor,
    gWin: cute.Tensor,
    gSlot: cute.Tensor,
    m_rows: cutlass.Constexpr[int],
    n_out: cutlass.Constexpr[int],
    kt: cutlass.Constexpr[int],
    ts: cutlass.Constexpr[int],
    tn: cutlass.Constexpr[int],
    r_res: cutlass.Constexpr[int],
    hrw: cutlass.Constexpr[int],
    cv: cutlass.Constexpr,
):
    AB: cutlass.Constexpr = tma_w_desc.dtype
    SWZ: cutlass.Constexpr = tma_w_desc.swizzle.to(prims.Tcgen05SmemSwizzle)
    XR: cutlass.Constexpr[int] = tn // 2  # x rows per CTA
    X_TILE: cutlass.Constexpr[int] = XR * K_BOX
    NKB: cutlass.Constexpr[int] = K_BOX // K_GRANULE
    S: cutlass.Constexpr[int] = kt - r_res  # streamed weight tiles
    NB: cutlass.Constexpr[int] = max(1, S)  # slot releases, one per streamed tile
    NPAIRS_F: cutlass.Constexpr[int] = n_out // FEAT_PAIR
    HALF: cutlass.Constexpr[int] = tn // 2  # accumulator columns per CTA (2x2 mapping)
    CHUNKS: cutlass.Constexpr = tuple(1 << b for b in range(6, -1, -1) if HALF & (1 << b))
    CONV: cutlass.Constexpr[bool] = cv is not None
    QT: cutlass.Constexpr[int] = cv[0] if CONV else 1
    RPH: cutlass.Constexpr[int] = (hrw // QT) if CONV else 1

    smem_w = cutlass.Array(AB, r_res * W_TILE, space=cutlass.AddressSpace.smem, alignment=SW_ALIGN)
    smem_x = cutlass.Array(AB, kt * X_TILE, space=cutlass.AddressSpace.smem, alignment=SW_ALIGN)
    w_full = cutlass.Array(cutlass.Int64, kt, space=cutlass.AddressSpace.smem, alignment=8)
    x_full = cutlass.Array(cutlass.Int64, kt // XG, space=cutlass.AddressSpace.smem, alignment=8)
    w_empty = cutlass.Array(cutlass.Int64, NB, space=cutlass.AddressSpace.smem, alignment=8)
    acc_bar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_i32 = cutlass.Array(Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)

    warp_idx = cute.arch.warp_idx()
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx % Int32(32)
    rank = cute.arch.block_idx_in_cluster()
    is_leader = rank == Int32(0)
    pair = bidx // Int32(2)
    tok_tile = pair // Int32(NPAIRS_F)
    fpair = pair % Int32(NPAIRS_F)
    feat0 = fpair * Int32(FEAT_PAIR) + rank * Int32(
        FEAT_CTA
    )  # this CTA's first output feature (W row)
    tok0 = tok_tile * Int32(ts)  # first token of the tile
    xrow0 = tok0 + rank * Int32(hrw)  # this CTA's first x row (B half)
    sa_ptr = Int64(gSa.iterator.toint())
    sb_ptr = Int64(gSb.iterator.toint())
    out_ptr = Int64(gOut.iterator.toint())

    # ---- barriers, TMEM (CTA / cluster state only: allowed before the dependency wait) ----
    if warp_idx == TMA_WARP:
        if prims.elect_sync():
            for t in cutlass.range_constexpr(kt):
                prims.mbarrier_init(w_full.subview(t), 1)
            for g in cutlass.range_constexpr(kt // XG):
                prims.mbarrier_init(x_full.subview(g), 1)
            for b in cutlass.range_constexpr(NB):
                prims.mbarrier_init(w_empty.subview(b), 1)
            prims.mbarrier_init(acc_bar, 1)
    prims.fence_mbarrier_init()
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()
    if warp_idx == MMA_WARP:
        prims.tcgen05_alloc(tmem_i32, TCOLS, group="cta_2")
    prims.barrier_cta_sync(0)
    tmem_raw = tmem_i32.load()
    tmem_ptr = cutlass.inttoptr(tmem_raw, 6, Float32)
    if warp_idx == MMA_WARP:
        prims.tcgen05_relinquish_alloc_permit(group="cta_2")

    x_ptr = tma_x_desc.get_ptr()
    w_ptr = tma_w_desc.get_ptr()
    w_bytes = tma_w_desc.global_tx_bytes()
    x_bytes = tma_x_desc.global_tx_bytes()

    # ---- TMA warp: resident weights + L2 prefetch before the wait, x right after it, then the streamed tiles ----
    if warp_idx == TMA_WARP:
        if prims.elect_sync():
            if is_leader:
                # the pair's transaction bytes land on the leader's barriers (cta_group::2 TMA)
                for t in cutlass.range_constexpr(kt):
                    prims.mbarrier_arrive_expect_tx(w_full.subview(t), 2 * w_bytes)
                for g in cutlass.range_constexpr(kt // XG):
                    prims.mbarrier_arrive_expect_tx(x_full.subview(g), 2 * x_bytes)
            for t in cutlass.range_constexpr(r_res):
                prims.cp_async_bulk_tensor_shared_cluster_global(
                    smem_w.subview(t * W_TILE),
                    w_ptr,
                    (Int32(t * K_BOX), feat0),
                    w_full.subview(t),
                    [],
                    l2_cache_hint=None,
                    group=prims.CTAGroup.CTA_2,
                )
            for t in cutlass.range_constexpr(r_res, kt):
                prims.cp_async_bulk_tensor_prefetch(
                    w_ptr, [Int32(t * K_BOX), feat0], [], l2_cache_hint=None
                )
            cute.arch.griddepcontrol_wait()
            for g in cutlass.range_constexpr(kt // XG):
                prims.cp_async_bulk_tensor_shared_cluster_global(
                    smem_x.subview(g * XG * X_TILE),
                    x_ptr,
                    (Int32(0), xrow0, Int32(g * XG)),
                    x_full.subview(g),
                    [],
                    group=prims.CTAGroup.CTA_2,
                )
        # streamed tile j (K tile r_res + j) into slot j % r_res once the MMAs of K tile j (the slot's previous
        # occupant) have completed; whole-warp polls, elect only around the issue
        for j in cutlass.range_constexpr(S):
            while not prims.mbarrier_try_wait_parity(w_empty.subview(j), 0, time_limit=10_000_000):
                pass
            if prims.elect_sync():
                prims.cp_async_bulk_tensor_shared_cluster_global(
                    smem_w.subview((j % r_res) * W_TILE),
                    w_ptr,
                    (Int32((r_res + j) * K_BOX), feat0),
                    w_full.subview(r_res + j),
                    [],
                    l2_cache_hint=None,
                    group=prims.CTAGroup.CTA_2,
                )

    # ---- MMA warp (leader of the pair issues): the K chain in order ----
    if warp_idx == MMA_WARP:
        if is_leader:
            idesc = prims.Tcgen05InstrDesc.build(
                c_dtype=Float32, a_dtype=AB, b_dtype=AB, n_dim=tn, m_dim=FEAT_PAIR
            )
            desc_w = prims.Tcgen05SmemDesc.build(
                start_address=smem_w,
                leading_byte_offset=K_BOX,
                stride_byte_offset=8 * K_BOX,
                layout=SWZ,
            )
            desc_x = prims.Tcgen05SmemDesc.build(
                start_address=smem_x,
                leading_byte_offset=K_BOX,
                stride_byte_offset=8 * K_BOX,
                layout=SWZ,
            )
            for t in cutlass.range_constexpr(kt):
                slot = t if t < r_res else (t - r_res) % r_res
                while not prims.mbarrier_try_wait_parity(
                    w_full.subview(t), 0, time_limit=10_000_000
                ):
                    pass
                if cutlass.const_expr(t % XG == 0):
                    while not prims.mbarrier_try_wait_parity(
                        x_full.subview(t // XG), 0, time_limit=10_000_000
                    ):
                        pass
                for kb in cutlass.range_constexpr(NKB):
                    if prims.elect_sync():
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F8F6F4,
                            prims.CTAGroup.CTA_2,
                            tmem_ptr,
                            desc_w.advance_start_address(Int32(slot * W_TILE + kb * K_GRANULE)),
                            desc_x.advance_start_address(Int32(t * X_TILE + kb * K_GRANULE)),
                            idesc,
                            (t > 0) or (kb > 0),
                        )
                if cutlass.const_expr(t < S):
                    if prims.elect_sync():
                        prims.tcgen05_commit(
                            w_empty.subview(t), multicast_mask=3, group=prims.CTAGroup.CTA_2
                        )
            if prims.elect_sync():
                prims.tcgen05_commit(acc_bar, multicast_mask=3, group=prims.CTAGroup.CTA_2)

    # ---- epilogue: TMEM -> * (sa * sb) -> BF16 -> y (conv mode: the causal conv1d update of the q|k|v columns) ----
    if warp_idx >= Int32(EPI_WARP0):
        cute.arch.griddepcontrol_wait()  # sa / sb / the conv state are data of earlier kernels
        alpha = _mul_rn(_ld_f32(sa_ptr), _ld_f32(sb_ptr))
        sp = warp_idx % Int32(4)  # TMEM lane quarter this warp may access
        m_stripe = sp % Int32(2)
        n_stripe = sp // Int32(2)
        feat = feat0 + m_stripe * Int32(32) + lane
        cv_w = []
        cv_slot = []
        cv_st = []
        if cutlass.const_expr(CONV):
            # After the wait, before the accumulator: hidden under the MMAs. Indices are clamped so z-channel CTAs
            # load harmlessly; they never use the values.
            cvf = cutlass.min(feat, Int32(cv[1] - 1))
            cw_addr = Int64(gCw.iterator.toint()) + Int64(cvf) * Int64(8)
            for i in cutlass.range_constexpr(4):
                cv_w.append(_ld_bf16(cw_addr + Int64(2 * i)))
            for j in cutlass.range_constexpr(RPH):
                req = tok_tile * Int32(2 * RPH) + n_stripe * Int32(RPH) + Int32(j)
                sl = _ld_s32(
                    Int64(gSlot.iterator.toint())
                    + Int64(cutlass.min(req, Int32(cv[2] - 1))) * Int64(4)
                )
                cv_slot.append(sl)
                st_addr = Int64(gCs.iterator.toint()) + (
                    Int64(cutlass.max(sl, Int32(0))) * Int64(cv[3][0])
                    + Int64(cvf) * Int64(cv[3][1])
                ) * Int64(2)
                stj = []
                for i in cutlass.range_constexpr(3):
                    stj.append(_ld_bf16(st_addr + Int64(2 * i * cv[3][2])))
                cv_st.append(stj)
        while not prims.mbarrier_try_wait_parity(acc_bar, 0, time_limit=10_000_000):
            pass
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        if tidx == Int32(EPI_WARP0 * 32):
            cute.arch.griddepcontrol_launch_dependents()
        base = prims.TmemAddr(tmem_raw)
        acc = cutlass.Array(Float32, HALF, alignment=32)
        for ci in cutlass.range_constexpr(len(CHUNKS)):  # tcgen05.ld widths must be powers of 2
            cwid = CHUNKS[ci]
            c0 = sum(CHUNKS[:ci])
            acc.store(
                prims.tcgen05_ld(
                    "32x32b",
                    prims.TmemAddr.from_row_col(
                        base.row_id + sp * Int32(32), base.col_id + Int32(c0)
                    ).as_ptr(Float32),
                    num=cwid,
                ),
                c0,
            )
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
        if cutlass.const_expr(CONV):
            if fpair < Int32(cv[1] // FEAT_PAIR):
                # GDN causal conv1d update of this channel for the RPH requests of this half: the outputs replace the
                # raw q|k|v projection, plus the window after every token and the conv state after the last one.
                win_base = Int64(gWin.iterator.toint())
                cs_base = Int64(gCs.iterator.toint())
                y = []
                for c in cutlass.range_constexpr(RPH * QT):
                    y.append(_bf16r(_mul_rn(acc[c], alpha)))
                for j in cutlass.range_constexpr(RPH):
                    req = tok_tile * Int32(2 * RPH) + n_stripe * Int32(RPH) + Int32(j)
                    if req < Int32(cv[2]):
                        ext = list(cv_st[j]) + y[j * QT : (j + 1) * QT]
                        # one base address per store group; the per-token / per-tap byte offsets are compile-time
                        ob = out_ptr + (
                            Int64(tok0 + n_stripe * Int32(hrw) + Int32(j * QT)) * Int64(n_out)
                            + Int64(feat)
                        ) * Int64(2)
                        for t in cutlass.range_constexpr(QT):
                            o = _gdn_conv_silu(
                                ext[t],
                                ext[t + 1],
                                ext[t + 2],
                                ext[t + 3],
                                cv_w[0],
                                cv_w[1],
                                cv_w[2],
                                cv_w[3],
                            )
                            _st_bf16_exact(ob, o, 2 * t * n_out)
                        if cv_slot[j] >= Int32(0):
                            wb = win_base + (
                                Int64(req) * Int64(cv[4][0]) + Int64(feat) * Int64(cv[4][2])
                            ) * Int64(2)
                            for t in cutlass.range_constexpr(QT):
                                for i in cutlass.range_constexpr(3):
                                    _st_bf16_exact(
                                        wb, ext[t + 1 + i], 2 * (t * cv[4][1] + i * cv[4][3])
                                    )
                            sb_ = cs_base + (
                                Int64(cv_slot[j]) * Int64(cv[3][0]) + Int64(feat) * Int64(cv[3][1])
                            ) * Int64(2)
                            for i in cutlass.range_constexpr(3):
                                _st_bf16_exact(sb_, ext[QT + i], 2 * i * cv[3][2])
            else:
                ob = out_ptr + (
                    Int64(tok0 + n_stripe * Int32(hrw)) * Int64(n_out) + Int64(feat)
                ) * Int64(2)
                for c in cutlass.range_constexpr(HALF):
                    tok_local = n_stripe * Int32(hrw) + Int32(c)
                    tok = tok0 + tok_local
                    if (Int32(c) < Int32(hrw)) & (tok_local < Int32(ts)) & (tok < Int32(m_rows)):
                        _st_bf16(ob, _mul_rn(acc[c], alpha), 2 * c * n_out)
        else:
            ob = out_ptr + (
                Int64(tok0 + n_stripe * Int32(hrw)) * Int64(n_out) + Int64(feat)
            ) * Int64(2)
            for c in cutlass.range_constexpr(HALF):
                tok_local = n_stripe * Int32(hrw) + Int32(c)
                tok = tok0 + tok_local
                if (Int32(c) < Int32(hrw)) & (tok_local < Int32(ts)) & (tok < Int32(m_rows)):
                    _st_bf16(ob, _mul_rn(acc[c], alpha), 2 * c * n_out)
    # Reconverge every warp before the CTA barrier (warp 0's lanes otherwise arrive in two groups).
    cute.arch.sync_warp()
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()
    if warp_idx == MMA_WARP:
        prims.tcgen05_dealloc(tmem_ptr, TCOLS, group="cta_2")


@cute.jit
def _launch(
    x: cute.Tensor,
    w: cute.Tensor,
    sa: cute.Tensor,
    sb: cute.Tensor,
    out: cute.Tensor,
    cw: cute.Tensor,
    cs: cute.Tensor,
    win: cute.Tensor,
    slot: cute.Tensor,
    stream: cuda.CUstream,
    m_rows: cutlass.Constexpr[int],
    n_out: cutlass.Constexpr[int],
    k_total: cutlass.Constexpr[int],
    tiles: cutlass.Constexpr[int],
    ts: cutlass.Constexpr[int],
    tn: cutlass.Constexpr[int],
    r_res: cutlass.Constexpr[int],
    hrw: cutlass.Constexpr[int],
    cv: cutlass.Constexpr,
    use_pdl: cutlass.Constexpr[bool],
):
    # [128 K][rows][K / 128] view of x: one box = 8 K tiles of this CTA's rows, landing as [8][rows][128]
    tma_x = cuda_exp.create_tensor_map_tiled(
        global_address=x.iterator.toint(),
        dtype=cutlass.Float8E4M3FN,
        global_dims=[K_BOX, m_rows, k_total // K_BOX],
        global_strides=[k_total // 16, K_BOX // 16],
        box_dims=[K_BOX, tn // 2, XG],
        swizzle=cuda_exp.TensorMapSwizzle.s128b,
    )
    tma_w = cuda_exp.create_tensor_map_tiled(
        global_address=w.iterator.toint(),
        dtype=cutlass.Float8E4M3FN,
        global_dims=[k_total, n_out],
        global_strides=[k_total // 16],
        box_dims=[K_BOX, FEAT_CTA],
        swizzle=cuda_exp.TensorMapSwizzle.s128b,
    )
    _kernel(
        tma_x,
        tma_w,
        sa,
        sb,
        out,
        cw,
        cs,
        win,
        slot,
        m_rows,
        n_out,
        k_total // K_BOX,
        ts,
        tn,
        r_res,
        hrw,
        cv,
    ).launch(
        grid=(2 * (n_out // FEAT_PAIR) * tiles, 1, 1),
        block=(6 * 32, 1, 1),
        cluster=(2, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


def _compile(
    m: int, k: int, n: int, tiles: int, ts: int, tn: int, r_res: int, hrw: int, cv, use_pdl: bool
):
    mk = cute.runtime.make_fake_compact_tensor
    u8 = cutlass.Uint8
    return cute.compile(
        _launch,
        mk(u8, (m, k), stride_order=(1, 0), assumed_align=128),
        mk(u8, (n, k), stride_order=(1, 0), assumed_align=128),
        mk(cutlass.Float32, (1,), assumed_align=4),
        mk(cutlass.Float32, (1,), assumed_align=4),
        mk(cutlass.BFloat16, (m, n), stride_order=(1, 0), assumed_align=16),
        mk(cutlass.BFloat16, (1,), assumed_align=2),
        mk(cutlass.BFloat16, (1,), assumed_align=2),
        mk(cutlass.BFloat16, (1,), assumed_align=2),
        mk(cutlass.Int32, (1,), assumed_align=4),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        m_rows=m,
        n_out=n,
        k_total=k,
        tiles=tiles,
        ts=ts,
        tn=tn,
        r_res=r_res,
        hrw=hrw,
        cv=cv,
        use_pdl=use_pdl,
        options="--enable-tvm-ffi",
    )


_COMPILED = {}


def _compiled(*key):
    """``_compile(*key)``, once per key; the first call of a key must precede CUDA graph capture."""
    fn = _COMPILED.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"compile the resident FP8 GEMM {key[:3]} before CUDA graph capture")
        fn = _COMPILED[key] = _compile(*key)
    return fn


def resident_fp8_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    sa: torch.Tensor,
    sb: torch.Tensor,
    out: torch.Tensor | None = None,
    conv: ConvUpdate | None = None,
    budget: int = SMEM_BUDGET,
) -> torch.Tensor:
    """y [M, N] BF16 = (x [M, K] E4M3 @ w [N, K].T E4M3) * sa * sb (sa, sb: FP32 device scalars).

    ``conv``: the first C output columns then hold the convolved q|k|v (silu), the windows and the state are updated,
    and the rows must be R x tokens. ``budget``: smem bytes for x + weight slots (SMEM_BUDGET)."""
    m, k = x.shape
    n = w.shape[0]
    assert x.dtype == torch.float8_e4m3fn and w.dtype == torch.float8_e4m3fn
    assert x.is_contiguous() and w.is_contiguous()
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    if conv is None:
        cfg = plan(m, k, n, budget)
        assert cfg is not None
        tiles, ts, tn, r_res = cfg
        hrw, cv = tn // 2, None
        # the conv operands are never read without conv: views of the output stand in for them
        head = torch.as_strided(out, (2,), (1,))
        cargs = (head[:1],) * 3 + (head.view(torch.int32),)
    else:
        qt = conv.tokens
        cfg = plan_conv(m, qt, k, n, budget)
        assert cfg is not None
        tiles, ts, tn, r_res, hrw = cfg
        assert supports_conv(x, w, conv)
        cw, cs, win, sl = conv.weight, conv.state, conv.windows, conv.slots
        cv = (qt, cw.shape[0], m // qt, tuple(cs.stride()), tuple(win.stride()))
        cargs = tuple(torch.as_strided(t, (1,), (1,)) for t in (cw, cs, win, sl))
    fn = _compiled(m, k, n, tiles, ts, tn, r_res, hrw, cv, get_env_enable_pdl())
    fn(
        x.view(torch.uint8),
        w.view(torch.uint8),
        sa.reshape(1).float(),
        sb.reshape(1).float(),
        out,
        *cargs,
    )
    return out
