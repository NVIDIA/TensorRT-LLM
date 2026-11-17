# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row-parallel FP8 projection (tcgen05) + MNNVL two-shot all-reduce + residual + RMSNorm in one launch.

``gemm_twoshot_allreduce_norm`` takes row-parallel decode projections (``o_proj`` / ``out_proj``) of up to
``MAX_ROWS`` rows where the library all-reduce runs two-shot + rmsNormLamport (above its one-shot message size; the
caller's ``min_rows``), in place of the projection on cuBLAS followed by those two launches:

1. GEMM: x [M, K] E4M3 @ W [H, K].T E4M3 times ``s_in * s_w``, FP32 accumulation, BF16 partial. tcgen05
   cta_group::1 MMA, M 64 (<= 64 rows) or 128 (rows past M are TMA zero fill), N 32, K-step 32; one CTA per 32
   output columns (H / 32 CTAs, all co-resident); all K stages resident in shared memory. Weight tiles load by TMA
   before ``griddepcontrol.wait``, activations after it. The partial is bitwise cuBLAS's (nvjet).
2. Scatter: column slice c (CTA c) is owned by rank ``c % W``. Every rank stores its BF16 slice tile into the
   owner's SCATTER stage, laid out [src rank][c / W][vector][row] so that each warp store is contiguous.
3. Owner: CTA c of rank ``c % W`` sums its slice over the ranks (FP32, rank order -> BF16, the library's two-shot
   arithmetic) and multicasts it into every rank's BROADCAST stage, [c][vector][row] (contiguous again).
4. Row: CTA r < M polls all H / 8 vectors of its row at once, adds the residual (BF16), and normalizes bit for bit
   as rmsNormLamport: its sum of squares is replayed in the order of the library's launch shape for M rows
   (``lib_norm_tree``: per vector 8 BF16 squares, the warp butterfly, the CTA's butterfly over its warp sums, the
   cluster's CTA sums in order), ``rsqrt(sum / H + eps)``, ``bf16((w * x) * rstd)``; optional FP8 =
   satfinite(norm * (1 / scale)).

It shares the library's Lamport triple buffer and ``buffer_flags`` (two stages of M x H BF16, as the CuTe
two-shot), so its launches interleave with the library / CuTe all-reduces in one graph. No arrival counter: CTA 0
rotates the flags only after its whole row arrived, which needs every rank's every CTA to have stored its tile,
i.e. to have read the flags.

Schedule (none of it changes the arithmetic):
- The owner phase runs on warps 0-1 of every CTA, one item per thread (``supports_rows``); in a row CTA the
  epilogue warps start polling their row only after the CTA's owner items are stored, so their spinning polls do
  not slow the owner warps.
- The activation TMA box holds ceil8(rows) rows, not the MMA's M; the M-128 MMA reads the rows past the box from
  the next stage, or for the last stage from a tail pad of smem_a, garbage that only reaches unstored rows.
- The Lamport row stride is padded to 8 rows, so every 512-byte warp store starts on a 128-byte line.
- The row poll is warp-uniform (vote.any), so the norm's shuffles run converged.
- Both TMA descriptors are prefetched, and the GEMM scales loaded, right after the dependency wait.
- M-128 launches (65+ rows) pad shared memory above half an SM: one CTA per SM, so a following call's early (PDL)
  CTAs do not share SMs with this call's tail.
"""

from __future__ import annotations

import functools
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda_exp
import torch
from cuda.bindings import driver as cuda
from cutlass import Float32, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.experimental import primitives as prims

from ...distributed.mnnvl_fused_allreduce import lib_norm_tree
from .mnnvl_allreduce_norm import (
    _NEG_ZERO,
    _add,
    _bf16r,
    _div_rn,
    _e4m3x2,
    _hi,
    _ld_s64,
    _ld_v4,
    _ld_volatile_v4,
    _lo,
    _mul,
    _pack_bf16,
    _rsqrt,
    _sanitize,
    _shfl_bfly,
    _st_s32,
    _st_v2,
    _st_v4,
)

MMA_K = 128  # FP8 elements per 128-byte TMA row (s128b)
K_GRANULE = 32  # FP8 K of one MMA
MMA_N = 32  # output columns per CTA
SW_ALIGN = 1024
TMA_WARP, MMA_WARP, EPI_WARP0 = 0, 1, 2
THREADS = 6 * 32
EPI_THREADS = 4 * 32
# Row phase: each 8-lane group of the epilogue warps holds whole library norm warps (32 vectors each, 4 per lane),
# the order their sums of squares are replayed in; their warp sums then fit one warp.
LIB_WARP_VECS = 32
MAX_LIB_WARPS = 32
MAX_ROWS = 128
NORM_BAR = 1  # named barrier of the four epilogue warps
OWN_WARPS = 2  # warps 0-1 (TMA, MMA) take the spread owner items
GATE_BAR = 3  # named barrier of the row gate over all THREADS (0 = CTA, 1 = NORM_BAR)


def _replayable(m: int, hidden: int, sms: int) -> bool:
    """One library thread per 16-byte vector, CTAs of whole 128-thread groups: the orders this kernel replays."""
    block, cluster, loads = lib_norm_tree(m, hidden, sms)
    return loads == 1 and block % 128 == 0 and block * cluster == hidden // 8


@functools.lru_cache(maxsize=None)
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@functools.lru_cache(maxsize=None)
def _onesm_bytes(device_index: int) -> int:
    """Shared memory of an M-128 CTA: half the SM's, so with the per-CTA reservation one CTA fits per SM."""
    return torch.cuda.get_device_properties(device_index).shared_memory_per_multiprocessor // 2


@dsl_user_op
def _vote_any(p: Int32, *, loc=None, ip=None) -> Int32:
    """1 if any lane of the (full) warp has p != 0, else 0."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(p).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred pa, pb;\n\tsetp.ne.s32 pa, $1, 0;\n\tvote.sync.any.pred pb, pa, 0xffffffff;\n\t"
            "selp.s32 $0, 1, 0, pb;\n\t}",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _shfl_idx(x: Float32, src: Int32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip), Int32(src).ir_value(loc=loc, ip=ip)],
            "shfl.sync.idx.b32 $0, $1, $2, 0x1f, 0xffffffff;",
            "=f,f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _flag_u32(word: Int32) -> Int64:
    """A ``buffer_flags`` word (uint32 on the host, held as Int32 here) zero-extended: the buffer and clear sizes
    reach 4 GiB."""
    return Int64(word) & Int64(0xFFFFFFFF)


@cute.jit
def _clear_dirty(
    own: Int64,
    fl,
    buf_bytes: Int64,
    dirty: Int32,
    dirty_stages: Int32,
    gtid: Int64,
    gthreads: Int64,
):
    """Clear this thread's share of the previous call's buffer (every dirty stage) to the Lamport sentinel."""
    neg0 = Int32(_NEG_ZERO)
    for s_i in cutlass.range_constexpr(4):
        if Int32(s_i) < dirty_stages:
            stage_bytes = buf_bytes // Int64(dirty_stages)
            stage = own + (Int64(dirty) * Int64(dirty_stages) + Int64(s_i)) * stage_bytes
            sb = _flag_u32(fl.subview(4 + s_i).load())
            clear_vecs = (sb + Int64(15)) // Int64(16)
            c = gtid
            while c < clear_vecs:
                _st_v4(stage + c * Int64(16), neg0, neg0, neg0, neg0)
                c = c + gthreads


@cute.kernel
def _kernel(
    tma_a_desc: cutlass.GridConstant[cuda_exp.TensorMap],
    tma_w_desc: cutlass.GridConstant[cuda_exp.TensorMap],
    gSin: cute.Tensor,
    gSw: cute.Tensor,
    gRes: cute.Tensor,
    gNw: cute.Tensor,
    gScale: cute.Tensor,
    gResOut: cute.Tensor,
    gOut: cute.Tensor,
    gQuant: cute.Tensor,
    gWs: cute.Tensor,
    rank: Int32,
    active_m: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    stages: cutlass.Constexpr[int],
    nranks: cutlass.Constexpr[int],
    norm_eps: cutlass.Constexpr[float],
    norm_out: cutlass.Constexpr[bool],
    quant: cutlass.Constexpr[bool],
    mma_m: cutlass.Constexpr[int],
    box_m: cutlass.Constexpr[int],
    lib_wpc: cutlass.Constexpr[int],
    lib_cluster: cutlass.Constexpr[int],
    onesm_bytes: cutlass.Constexpr[int],
):
    VPT: cutlass.Constexpr[int] = hidden // (8 * EPI_THREADS)  # row vectors per epilogue thread
    LIB_WARPS: cutlass.Constexpr[int] = hidden // (8 * LIB_WARP_VECS)  # library norm warps of a row
    LWG: cutlass.Constexpr[int] = VPT // 4  # library norm warps per 8-lane group
    AB: cutlass.Constexpr = tma_a_desc.dtype
    SWZ: cutlass.Constexpr = tma_a_desc.swizzle.to(prims.Tcgen05SmemSwizzle)
    MP: cutlass.Constexpr[int] = (active_m + 7) // 8 * 8  # Lamport row stride
    DESC_LEAD: cutlass.Constexpr[int] = MMA_K // 16
    DESC_STRIDE: cutlass.Constexpr[int] = 8 * MMA_K // 16
    A_STAGE: cutlass.Constexpr[int] = (
        box_m * MMA_K
    )  # activation rows TMA writes per stage (= mma_m unless tight)
    # The M-mma_m MMA reads mma_m rows from each stage's start: rows past box_m come from the next stage, and for the
    # last stage from this tail, so the over-read never leaves smem_a whatever follows it (garbage that only reaches
    # accumulator rows >= active_m, never stored). 0 unless the box is smaller than the MMA; a multiple of 1024 bytes.
    A_TAIL: cutlass.Constexpr[int] = (mma_m - box_m) * MMA_K
    W_STAGE: cutlass.Constexpr[int] = MMA_N * MMA_K
    ONESM_PAD: cutlass.Constexpr[int] = (
        max(0, onesm_bytes - stages * (A_STAGE + W_STAGE) - A_TAIL) if mma_m == 128 else 0
    )
    NKB: cutlass.Constexpr[int] = MMA_K // K_GRANULE
    ROWS_PER_SUB: cutlass.Constexpr[int] = mma_m // 4  # accumulator rows per TMEM subpartition
    VECS: cutlass.Constexpr[int] = hidden // 8  # 16-byte vectors of a row
    CVECS: cutlass.Constexpr[int] = MMA_N // 8  # vectors of a CTA's column slice
    ROW_BYTES: cutlass.Constexpr[int] = hidden * 2
    NCTAS: cutlass.Constexpr[int] = hidden // MMA_N
    CPR: cutlass.Constexpr[int] = NCTAS // nranks  # slices each rank owns
    SLICE_ITEMS: cutlass.Constexpr[int] = (
        CVECS * MP
    )  # (vector, row) slots of one slice in the Lamport layout
    OWN_TOTAL: cutlass.Constexpr[int] = (
        CPR * SLICE_ITEMS
    )  # slots a rank owns (= items without padding)
    PER_CTA: cutlass.Constexpr[int] = (
        OWN_TOTAL + NCTAS - 1
    ) // NCTAS  # owner items per CTA (<= 64)
    # the library norm's launch shape, as top-level constexprs (kernel parameters turn into Int32 inside dynamic
    # control flow; these stay Python ints there)
    LIB_WPC: cutlass.Constexpr[int] = lib_wpc  # warps per CTA
    LIB_CLUSTER: cutlass.Constexpr[int] = lib_cluster  # CTAs per row
    LOG_WPC: cutlass.Constexpr[int] = lib_wpc.bit_length() - 1

    smem_a = cutlass.Array(
        AB, stages * A_STAGE + A_TAIL, space=cutlass.AddressSpace.smem, alignment=SW_ALIGN
    )
    smem_w = cutlass.Array(
        AB, stages * W_STAGE, space=cutlass.AddressSpace.smem, alignment=SW_ALIGN
    )
    full_bar = cutlass.Array(cutlass.Int64, stages, space=cutlass.AddressSpace.smem, alignment=8)
    acc_bar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    tmem_i32 = cutlass.Array(Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)
    fl = cutlass.Array(Int32, 8, space=cutlass.AddressSpace.smem, alignment=16)
    nred = cutlass.Array(
        Float32, LIB_WARPS, space=cutlass.AddressSpace.smem, alignment=16
    )  # library warp sums
    sinkw = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.smem, alignment=16)
    if cutlass.const_expr(ONESM_PAD > 0):
        onesm = cutlass.Array(Int32, ONESM_PAD // 4, space=cutlass.AddressSpace.smem, alignment=16)
        if rank < Int32(0):  # never true: keeps the allocation referenced
            onesm.subview(ONESM_PAD // 4 - 1).store(rank)

    warp_idx = cute.arch.warp_idx()
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx % Int32(32)
    coord_n = bidx * Int32(MMA_N)
    is_epi = warp_idx >= Int32(EPI_WARP0)
    epi_tid = tidx - Int32(EPI_WARP0 * 32)
    # row-phase vector of this epilogue thread's j-th slot: vbase + 8 j, vbase = 32 LWG G + a, G = 4 w + lane / 8
    # (its 8-lane group, which holds library warps LWG G .. LWG G + LWG - 1), a = lane % 8
    group = Int32(4) * (warp_idx - Int32(EPI_WARP0)) + (lane >> Int32(3))
    vbase = Int32(8 * VPT) * group + (lane & Int32(7))
    # TMEM rows: lanes 0..ROWS_PER_SUB-1 of subpartition (warp % 4) hold rows (warp % 4) * ROWS_PER_SUB + lane
    my_row = (warp_idx % Int32(4)) * Int32(ROWS_PER_SUB) + lane
    row_valid = is_epi & (lane < Int32(ROWS_PER_SUB)) & (my_row < Int32(active_m))
    row_cta = bidx < Int32(active_m)  # this CTA normalizes row bidx
    owner = bidx % Int32(nranks)  # the rank that reduces column slice bidx
    lslice = bidx // Int32(nranks)  # its index among that rank's slices
    idesc = prims.Tcgen05InstrDesc.build(
        c_dtype=Float32, a_dtype=AB, b_dtype=AB, n_dim=MMA_N, m_dim=mma_m
    )

    # ---- barriers, TMEM ----
    if warp_idx == TMA_WARP:
        if prims.elect_sync():
            for s in cutlass.range_constexpr(stages):
                prims.mbarrier_init(full_bar.subview(s), 1)
            prims.mbarrier_init(acc_bar, 1)
    prims.fence_mbarrier_init()
    prims.barrier_cta_sync(0)
    if warp_idx == TMA_WARP:
        prims.tcgen05_alloc(tmem_i32, 32, group="cta_1")
    prims.barrier_cta_sync(0)
    tmem_raw = tmem_i32.load()
    tmem_ptr = cutlass.inttoptr(tmem_raw, 6, Float32)
    prims.tcgen05_relinquish_alloc_permit(group="cta_1")

    # ---- host-written state, before the dependency wait: workspace table, norm weight, the owner's buffer ----
    tbl = Int64(gWs.iterator.toint())
    own = _ld_s64(tbl)
    mcast = _ld_s64(tbl + Int64(8))
    flags = _ld_s64(tbl + Int64(16))
    peer = _ld_s64(tbl + Int64(24) + Int64(owner) * Int64(8))
    nwr = cutlass.Array(Int32, 4 * VPT, alignment=16)
    if is_epi:
        for j in cutlass.range_constexpr(VPT):
            a0, a1, a2, a3 = _ld_v4(
                Int64(gNw.iterator.toint()) + (Int64(vbase) + Int64(8 * j)) * Int64(16)
            )
            nwr[4 * j + 0] = a0
            nwr[4 * j + 1] = a1
            nwr[4 * j + 2] = a2
            nwr[4 * j + 3] = a3

    a_ptr = tma_a_desc.get_ptr()
    w_ptr = tma_w_desc.get_ptr()
    tma_bytes = tma_a_desc.global_tx_bytes() + tma_w_desc.global_tx_bytes()
    desc_a = prims.Tcgen05SmemDesc.build(
        start_address=smem_a,
        leading_byte_offset=DESC_LEAD * 16,
        stride_byte_offset=DESC_STRIDE * 16,
        layout=SWZ,
    )
    desc_w = prims.Tcgen05SmemDesc.build(
        start_address=smem_w,
        leading_byte_offset=DESC_LEAD * 16,
        stride_byte_offset=DESC_STRIDE * 16,
        layout=SWZ,
    )

    # ---- TMA: weight tiles, dependency wait, activation tiles ----
    if warp_idx == TMA_WARP:
        if prims.elect_sync():
            prims.prefetch_tensormap(a_ptr)
            prims.prefetch_tensormap(w_ptr)
            for s in cutlass.range_constexpr(stages):
                prims.mbarrier_arrive_expect_tx(full_bar.subview(s), tma_bytes)
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_w.subview(s * W_STAGE),
                    w_ptr,
                    (Int32(s * MMA_K), coord_n),
                    full_bar.subview(s),
                )
            cute.arch.griddepcontrol_wait()
            for s in cutlass.range_constexpr(stages):
                prims.cp_async_bulk_tensor_shared_cta_global(
                    smem_a.subview(s * A_STAGE),
                    a_ptr,
                    (Int32(s * MMA_K), Int32(0)),
                    full_bar.subview(s),
                )

    # ---- MMA ----
    if warp_idx == MMA_WARP:
        for s in cutlass.range_constexpr(stages):
            while not prims.mbarrier_try_wait_parity(full_bar.subview(s), 0, time_limit=10_000_000):
                pass
            for kb in cutlass.range_constexpr(NKB):
                off_a = Int32(s * A_STAGE + kb * K_GRANULE)
                off_w = Int32(s * W_STAGE + kb * K_GRANULE)
                if prims.elect_sync():
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F8F6F4,
                        prims.CTAGroup.CTA_1,
                        tmem_ptr,
                        desc_a.advance_start_address(off_a),
                        desc_w.advance_start_address(off_w),
                        idesc,
                        (s > 0) or (kb > 0),
                    )
        if prims.elect_sync():
            prims.tcgen05_commit(acc_bar, group=prims.CTAGroup.CTA_1)

    # ---- epilogue warps: dependency wait, residual of row bidx, the Lamport flags, the accumulator ----
    rres = cutlass.Array(Int32, 4 * VPT, alignment=16)
    acc = cutlass.Array(Float32, MMA_N, alignment=32)
    alr = cutlass.Array(Float32, 1)  # alpha, loaded right after the dependency wait
    row_off = Int64(bidx) * Int64(VECS) + Int64(
        vbase
    )  # vector index of this thread's j = 0 vector of row bidx
    if is_epi:
        cute.arch.griddepcontrol_wait()
        alr[0] = gSin[0] * gSw[0]
        if row_cta:
            for j in cutlass.range_constexpr(VPT):
                r0, r1, r2, r3 = _ld_v4(
                    Int64(gRes.iterator.toint()) + (row_off + Int64(8 * j)) * Int64(16)
                )
                rres[4 * j + 0] = r0
                rres[4 * j + 1] = r1
                rres[4 * j + 2] = r2
                rres[4 * j + 3] = r3
        if epi_tid == Int32(0):
            f0, f1, f2, f3 = _ld_volatile_v4(flags)
            f4, f5, f6, f7 = _ld_volatile_v4(flags + Int64(16))
            fl.subview(0).store(f0)
            fl.subview(1).store(f1)
            fl.subview(2).store(f2)
            fl.subview(3).store(f3)
            fl.subview(4).store(f4)
            fl.subview(5).store(f5)
            fl.subview(6).store(f6)
            fl.subview(7).store(f7)
        while not prims.mbarrier_try_wait_parity(acc_bar, 0, time_limit=10_000_000):
            pass
        base = prims.TmemAddr(tmem_raw)
        tsub = prims.TmemAddr.from_row_col(base.row_id, base.col_id).as_ptr(Float32)
        acc.store(prims.tcgen05_ld("32x32b", tsub, num=MMA_N), 0)
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    if warp_idx == TMA_WARP:
        prims.tcgen05_dealloc(tmem_ptr, 32, group="cta_1")

    cur = fl.subview(0).load()
    dirty = fl.subview(1).load()
    buf_bytes = _flag_u32(fl.subview(2).load())
    dirty_stages = fl.subview(3).load()
    data_off = Int64(cur) * buf_bytes
    half = buf_bytes // Int64(2)
    neg0 = Int32(_NEG_ZERO)

    # ---- scatter: this CTA's slice tile -> owner's SCATTER [src rank][slice / W][vector][row] (warp-contiguous) ----
    if row_valid:
        alpha = alr[0]
        dst = (
            peer
            + data_off
            + (
                (Int64(rank) * Int64(CPR) + Int64(lslice)) * Int64(CVECS) * Int64(MP)
                + Int64(my_row)
            )
            * Int64(16)
        )
        for v in cutlass.range_constexpr(CVECS):
            w0 = _sanitize(_pack_bf16(_mul(acc[8 * v + 0], alpha), _mul(acc[8 * v + 1], alpha)))
            w1 = _sanitize(_pack_bf16(_mul(acc[8 * v + 2], alpha), _mul(acc[8 * v + 3], alpha)))
            w2 = _sanitize(_pack_bf16(_mul(acc[8 * v + 4], alpha), _mul(acc[8 * v + 5], alpha)))
            w3 = _sanitize(_pack_bf16(_mul(acc[8 * v + 6], alpha), _mul(acc[8 * v + 7], alpha)))
            _st_v4(dst + Int64(16 * v * MP), w0, w1, w2, w3)

    # ---- PDL: the residual words are consumed (sunk to shared memory) by every thread, then release ----
    if is_epi:
        if row_cta:
            x = rres[0]
            for i in cutlass.range_constexpr(1, 4 * VPT):
                x = x ^ rres[i]
            sinkw.subview(0).store(x)
    prims.barrier_cta_sync(0)
    if tidx == Int32(0):
        cute.arch.griddepcontrol_launch_dependents()

    gtid = Int64(bidx) * Int64(THREADS) + Int64(tidx)
    gthreads = Int64(NCTAS * THREADS)
    _clear_dirty(own, fl, buf_bytes, dirty, dirty_stages, gtid, gthreads)

    # ---- owner: item g of this rank's CPR slices (g = local slice * 4 M + vector * M + row) on thread g % PER_CTA
    # of warps 0-1 in CTA g / PER_CTA: the slice summed over the ranks in rank order, multicast ----
    if warp_idx < Int32(OWN_WARPS):
        g = bidx * Int32(PER_CTA) + tidx
        live = (tidx < Int32(PER_CTA)) & (g < Int32(OWN_TOTAL))
        lsl = g // Int32(SLICE_ITEMS)
        it_ = g - lsl * Int32(SLICE_ITEMS)
        live = live & ((it_ % Int32(MP)) < Int32(active_m))  # padding rows are never written
        if live:
            sgot = cutlass.Array(Int32, 4 * nranks, alignment=16)
            spend = Int32(1)
            while spend != Int32(0):
                spend = Int32(0)
                for rk in cutlass.range_constexpr(nranks):
                    # SCATTER [src rank][local slice][vector][row] = rk * OWN_TOTAL + g
                    h0, h1, h2, h3 = _ld_volatile_v4(
                        own + data_off + (Int64(rk * OWN_TOTAL) + Int64(g)) * Int64(16)
                    )
                    sgot[4 * rk + 0] = h0
                    sgot[4 * rk + 1] = h1
                    sgot[4 * rk + 2] = h2
                    sgot[4 * rk + 3] = h3
                    if h0 == neg0:
                        spend = Int32(1)
                    if h1 == neg0:
                        spend = Int32(1)
                    if h2 == neg0:
                        spend = Int32(1)
                    if h3 == neg0:
                        spend = Int32(1)
            ssum = cutlass.Array(Int32, 4, alignment=16)
            for i in cutlass.range_constexpr(4):
                lo_sum = Float32(0.0)
                hi_sum = Float32(0.0)
                for rk in cutlass.range_constexpr(nranks):
                    lo_sum = _add(lo_sum, _lo(sgot[4 * rk + i]))
                    hi_sum = _add(hi_sum, _hi(sgot[4 * rk + i]))
                ssum[i] = _sanitize(_pack_bf16(lo_sum, hi_sum))
            # BROADCAST [slice][vector][row], slice = local slice * W + rank
            _st_v4(
                mcast
                + data_off
                + half
                + ((Int64(lsl) * Int64(nranks) + Int64(rank)) * Int64(SLICE_ITEMS) + Int64(it_))
                * Int64(16),
                ssum[0],
                ssum[1],
                ssum[2],
                ssum[3],
            )
        if row_cta:
            # this CTA's owner items are stored: release the epilogue warps' row polls
            cute.arch.barrier(barrier_id=GATE_BAR, number_of_threads=THREADS)

    # ---- row bidx: all its vectors at once, + residual, RMSNorm (rmsNormLamport arithmetic) ----
    if is_epi & row_cta:
        cute.arch.barrier(barrier_id=GATE_BAR, number_of_threads=THREADS)
        res = cutlass.Array(Float32, 8 * VPT, alignment=32)
        bwa = cutlass.Array(Int32, 4 * VPT, alignment=16)
        pend = Int32(1)
        while pend != Int32(0):
            pend = Int32(0)
            for j in cutlass.range_constexpr(VPT):
                # vector g = vbase + 8 j of the row = (slice g / 4, vector g % 4) -> [slice][vector][row]
                c0, c1, c2, c3 = _ld_volatile_v4(
                    own
                    + data_off
                    + half
                    + ((Int64(vbase) + Int64(8 * j)) * Int64(MP) + Int64(bidx)) * Int64(16)
                )
                bwa[4 * j + 0] = c0
                bwa[4 * j + 1] = c1
                bwa[4 * j + 2] = c2
                bwa[4 * j + 3] = c3
                if c0 == neg0:
                    pend = Int32(1)
                if c1 == neg0:
                    pend = Int32(1)
                if c2 == neg0:
                    pend = Int32(1)
                if c3 == neg0:
                    pend = Int32(1)
            pend = _vote_any(pend)
        for j in cutlass.range_constexpr(VPT):
            for i in cutlass.range_constexpr(4):
                res[8 * j + 2 * i] = _bf16r(_add(_lo(bwa[4 * j + i]), _lo(rres[4 * j + i])))
                res[8 * j + 2 * i + 1] = _bf16r(_add(_hi(bwa[4 * j + i]), _hi(rres[4 * j + i])))
        for j in cutlass.range_constexpr(VPT):
            _st_v4(
                Int64(gResOut.iterator.toint()) + (row_off + Int64(8 * j)) * Int64(16),
                _pack_bf16(res[8 * j + 0], res[8 * j + 1]),
                _pack_bf16(res[8 * j + 2], res[8 * j + 3]),
                _pack_bf16(res[8 * j + 4], res[8 * j + 5]),
                _pack_bf16(res[8 * j + 6], res[8 * j + 7]),
            )
        # rmsNormLamport's order (LIB_CLUSTER CTAs x LIB_WPC warps, a vector per library thread): this thread holds
        # vectors a + 8 (j % 4) of library warp LWG G + j / 4, so its xor-16 and xor-8 steps are local and xor 4, 2, 1
        # run across the 8-lane group; the CTA and cluster sums follow the barrier.
        qs = []
        for j in cutlass.range_constexpr(VPT):
            q = Float32(0.0)
            for e in cutlass.range_constexpr(8):
                q = _add(q, _bf16r(_mul(res[8 * j + e], res[8 * j + e])))
            qs.append(q)
        for lw in cutlass.range_constexpr(LWG):
            y = _add(_add(qs[4 * lw + 0], qs[4 * lw + 2]), _add(qs[4 * lw + 1], qs[4 * lw + 3]))
            y = _add(y, _shfl_bfly(y, Int32(4)))
            y = _add(y, _shfl_bfly(y, Int32(2)))
            y = _add(y, _shfl_bfly(y, Int32(1)))
            if (lane & Int32(7)) == Int32(0):
                if cutlass.const_expr(LWG == 1):
                    nred.subview(group).store(y)
                else:
                    nred.subview(Int32(LWG) * group + Int32(lw)).store(y)
        cute.arch.barrier(barrier_id=NORM_BAR, number_of_threads=EPI_THREADS)
        # lane W holds warp sum S_W (lanes past LIB_WARPS repeat and are never read); xor m for m = LIB_WPC / 2 .. 1 is
        # the CTA butterfly (the library's masks >= LIB_WPC add zeros); lane k LIB_WPC then holds CTA k's sum
        u = nred.subview(lane & Int32(LIB_WARPS - 1)).load()
        for st in cutlass.range_constexpr(LOG_WPC):
            u = _add(u, _shfl_bfly(u, Int32(LIB_WPC >> (st + 1))))
        total = Float32(0.0)
        for kc in cutlass.range_constexpr(LIB_CLUSTER):
            total = _add(total, _shfl_idx(u, Int32(kc * LIB_WPC)))
        rstd = _rsqrt(_add(_div_rn(total, Float32(float(hidden))), Float32(norm_eps)))
        inv = Float32(1.0)
        if cutlass.const_expr(quant):
            inv = _div_rn(Float32(1.0), gScale[0])
        for j in cutlass.range_constexpr(VPT):
            y = cutlass.Array(Float32, 8, alignment=32)
            for i in cutlass.range_constexpr(4):
                y[2 * i] = _bf16r(_mul(_mul(_lo(nwr[4 * j + i]), res[8 * j + 2 * i]), rstd))
                y[2 * i + 1] = _bf16r(_mul(_mul(_hi(nwr[4 * j + i]), res[8 * j + 2 * i + 1]), rstd))
            vrow = row_off + Int64(8 * j)
            if cutlass.const_expr(norm_out):
                _st_v4(
                    Int64(gOut.iterator.toint()) + vrow * Int64(16),
                    _pack_bf16(y[0], y[1]),
                    _pack_bf16(y[2], y[3]),
                    _pack_bf16(y[4], y[5]),
                    _pack_bf16(y[6], y[7]),
                )
            if cutlass.const_expr(quant):
                q01 = _e4m3x2(_mul(y[0], inv), _mul(y[1], inv))
                q23 = _e4m3x2(_mul(y[2], inv), _mul(y[3], inv))
                q45 = _e4m3x2(_mul(y[4], inv), _mul(y[5], inv))
                q67 = _e4m3x2(_mul(y[6], inv), _mul(y[7], inv))
                _st_v2(
                    Int64(gQuant.iterator.toint()) + vrow * Int64(8),
                    q01 | (q23 << 16),
                    q45 | (q67 << 16),
                )
        # ---- hand the triple buffer to the next call (see the module docstring for why no arrival counter) ----
        if bidx == Int32(0):
            if epi_tid == Int32(0):
                _st_s32(flags, (cur + Int32(1)) % Int32(3))
                _st_s32(flags + Int64(4), cur)
                _st_s32(flags + Int64(12), Int32(2))
                _st_s32(flags + Int64(16), Int32(MP * ROW_BYTES))
                _st_s32(flags + Int64(20), Int32(MP * ROW_BYTES))
                _st_s32(flags + Int64(24), Int32(0))
                _st_s32(flags + Int64(28), Int32(0))
                _st_s32(flags + Int64(32), Int32(0))


@cute.jit
def _launch(
    a: cute.Tensor,
    w: cute.Tensor,
    s_in: cute.Tensor,
    s_w: cute.Tensor,
    res: cute.Tensor,
    nw: cute.Tensor,
    scale: cute.Tensor,
    res_out: cute.Tensor,
    out: cute.Tensor,
    qout: cute.Tensor,
    table: cute.Tensor,
    rank: Int32,
    stream: cuda.CUstream,
    active_m: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    k_total: cutlass.Constexpr[int],
    nranks: cutlass.Constexpr[int],
    norm_eps: cutlass.Constexpr[float],
    norm_out: cutlass.Constexpr[bool],
    quant: cutlass.Constexpr[bool],
    lib_wpc: cutlass.Constexpr[int],
    lib_cluster: cutlass.Constexpr[int],
    onesm_bytes: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
):
    mma_m = 64 if active_m <= 64 else 128
    box_m = min(mma_m, (active_m + 7) // 8 * 8)
    tma_a = cuda_exp.create_tensor_map_tiled(
        global_address=a.iterator.toint(),
        dtype=cutlass.Float8E4M3FN,
        global_dims=[k_total, active_m],
        global_strides=[k_total // 16],
        box_dims=[MMA_K, box_m],
        swizzle=cuda_exp.TensorMapSwizzle.s128b,
    )
    tma_w = cuda_exp.create_tensor_map_tiled(
        global_address=w.iterator.toint(),
        dtype=cutlass.Float8E4M3FN,
        global_dims=[k_total, hidden],
        global_strides=[k_total // 16],
        box_dims=[MMA_K, MMA_N],
        swizzle=cuda_exp.TensorMapSwizzle.s128b,
    )
    _kernel(
        tma_a,
        tma_w,
        s_in,
        s_w,
        res,
        nw,
        scale,
        res_out,
        out,
        qout,
        table,
        rank,
        active_m,
        hidden,
        k_total // MMA_K,
        nranks,
        norm_eps,
        norm_out,
        quant,
        mma_m,
        box_m,
        lib_wpc,
        lib_cluster,
        onesm_bytes,
    ).launch(grid=(hidden // MMA_N, 1, 1), block=(THREADS, 1, 1), stream=stream, use_pdl=use_pdl)


def _compile(
    m: int,
    k: int,
    hidden: int,
    nranks: int,
    eps: float,
    norm_out: bool,
    quant: bool,
    sms: int,
    onesm_bytes: int,
    use_pdl: bool,
):
    block, cluster, loads = lib_norm_tree(m, hidden, sms)
    assert _replayable(m, hidden, sms), (m, hidden, sms, block, cluster, loads)
    mk = cute.runtime.make_fake_compact_tensor
    u8 = cutlass.Uint8
    # x and W only feed the host-built TMA descriptors, whose global address needs 16 bytes (``supports``).
    return cute.compile(
        _launch,
        mk(u8, (m, k), stride_order=(1, 0), assumed_align=16),
        mk(u8, (hidden, k), stride_order=(1, 0), assumed_align=16),
        mk(cutlass.Float32, (1,), assumed_align=4),
        mk(cutlass.Float32, (1,), assumed_align=4),
        mk(cutlass.BFloat16, (m, hidden), stride_order=(1, 0), assumed_align=16),
        mk(cutlass.BFloat16, (hidden,), assumed_align=16),
        mk(cutlass.Float32, (1,), assumed_align=4),
        mk(cutlass.BFloat16, (m, hidden), stride_order=(1, 0), assumed_align=16),
        mk(cutlass.BFloat16, (m, hidden), stride_order=(1, 0), assumed_align=16),
        mk(u8, (m, hidden), stride_order=(1, 0), assumed_align=16),
        mk(cutlass.Int64, (3 + nranks,), assumed_align=8),
        Int32(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        active_m=m,
        hidden=hidden,
        k_total=k,
        nranks=nranks,
        norm_eps=eps,
        norm_out=norm_out,
        quant=quant,
        lib_wpc=block // 32,
        lib_cluster=cluster,
        onesm_bytes=onesm_bytes,
        use_pdl=use_pdl,
        options="--enable-tvm-ffi",
    )


_COMPILED = {}


def _compiled(*key):
    """``_compile(*key)``, once per key (M is a compile-time constant of the kernel); the first call of a key must
    precede CUDA graph capture."""
    fn = _COMPILED.get(key)
    if fn is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"compile the GEMM + two-shot all-reduce + norm kernel {key[:4]} before CUDA graph capture"
            )
        fn = _COMPILED[key] = _compile(*key)
    return fn


def supports_hidden(hidden: int, nranks: int, sms: int) -> bool:
    """Whether the kernel takes this hidden size and TP group: whole library norm warps per 8-lane epilogue group
    and their warp sums within one warp, one CTA per 32 output columns, all co-resident, owned round-robin by the
    ranks."""
    return (
        hidden % (8 * EPI_THREADS * (LIB_WARP_VECS // 8)) == 0
        and hidden // (8 * LIB_WARP_VECS) <= MAX_LIB_WARPS
        and hidden // MMA_N <= sms
        and (hidden // MMA_N) % nranks == 0
        and nranks in (2, 4, 8, 16)
    )


def supports_rows(m: int, hidden: int, nranks: int, sms: int, min_rows: int) -> bool:
    """Whether a call of ``m`` rows fits: from ``min_rows`` (the library's first two-shot rows), every owner item (a
    rank's slices x vectors x padded rows) has a thread of the CTAs' ``OWN_WARPS`` warps, and the library norm's
    launch for ``m`` rows is one this kernel replays."""
    nctas = hidden // MMA_N
    own_total = (nctas // nranks) * (MMA_N // 8) * ((m + 7) // 8 * 8)
    return (
        min_rows <= m <= min(MAX_ROWS, nctas)
        and -(-own_total // nctas) <= OWN_WARPS * 32
        and _replayable(m, hidden, sms)
    )


def supports_problem(
    m: int, weight: torch.Tensor, hidden: int, nranks: int, sms: int, min_rows: int
) -> bool:
    """Whether a call of ``m`` rows with this E4M3 weight [H, K] fits (the activation not looked at); the weight's
    address is 16-byte aligned, as its TMA descriptor needs."""
    k = weight.shape[-1]
    return (
        weight.dtype == torch.float8_e4m3fn
        and weight.shape == (hidden, k)
        and weight.is_contiguous()
        and weight.data_ptr() % 16 == 0
        and k % MMA_K == 0
        and k // MMA_K <= 8
        and supports_hidden(hidden, nranks, sms)
        and supports_rows(m, hidden, nranks, sms, min_rows)
    )


def supports(
    x: torch.Tensor, weight: torch.Tensor, hidden: int, nranks: int, sms: int, min_rows: int
) -> bool:
    """Whether the launch takes this E4M3 activation and weight: ``supports_problem``, and x contiguous at a 16-byte
    aligned address (the compiled kernel's assumption for both)."""
    return (
        x.dim() == 2
        and x.dtype == torch.float8_e4m3fn
        and x.shape[1] == weight.shape[-1]
        and x.is_contiguous()
        and x.data_ptr() % 16 == 0
        and supports_problem(x.shape[0], weight, hidden, nranks, sms, min_rows)
    )


def workspace_bytes(m: int, hidden: int) -> int:  # rows padded to 8 (the Lamport row stride)
    """Lamport buffer bytes one call needs: two stages of M x H BF16."""
    return 2 * ((m + 7) // 8 * 8) * hidden * 2


def gemm_twoshot_allreduce_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_scale: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    table: torch.Tensor,
    rank: int,
    nranks: int,
    scale: Optional[torch.Tensor] = None,
    norm_out: bool = True,
) -> Tuple[torch.Tensor, ...]:
    """x [M, K] E4M3 (static ``input_scale``), weight [H, K] E4M3 (``weight_scale``), residual [M, H] BF16,
    norm_weight [H] BF16, table: the Lamport workspace table of ``_LamportTable`` (at least
    ``workspace_bytes(M, H)`` per buffer). Outputs as ``MnnvlAllReduceNorm.__call__``."""
    from ...flashinfer_utils import get_env_enable_pdl

    m, k = x.shape
    hidden = weight.shape[0]
    dev = residual.device
    fn = _compiled(
        m,
        k,
        hidden,
        nranks,
        float(eps),
        norm_out,
        scale is not None,
        _sm_count(dev.index),
        _onesm_bytes(dev.index),
        get_env_enable_pdl(),
    )
    res_out = torch.empty_like(residual)
    norm = torch.empty_like(residual) if norm_out else res_out
    quant = (
        torch.empty(residual.shape, dtype=torch.uint8, device=dev) if scale is not None else None
    )
    sc = scale if scale is not None else input_scale
    fn(
        x.view(torch.uint8),
        weight.view(torch.uint8),
        input_scale.reshape(1).float(),
        weight_scale.reshape(1).float(),
        residual,
        norm_weight.reshape(-1),
        sc.reshape(1).float(),
        res_out,
        norm,
        # without quant the FP8 output is never written: any [M, H] bytes stand in for it
        quant
        if quant is not None
        else res_out.view(torch.uint8).view(-1)[: m * hidden].view(m, hidden),
        table,
        rank,
    )
    if scale is None:
        return norm, res_out
    q = quant.view(torch.float8_e4m3fn)
    if norm_out:
        return norm, q, res_out
    return q, res_out
