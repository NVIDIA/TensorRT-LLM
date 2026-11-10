# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The BF16x3 router GEMM of ``ops/moe/router_gemm`` with its partial tile written through shared memory.

``Sm100BF16x3RouterGemm`` (``cute_dsl_kernels/blackwell/bf16x3_router_gemm``) ends every CTA with
an epilogue in which each of the four epilogue warps writes its accumulator lanes with one 4-byte
store per output element: 128 stores per thread of the ``[split, row, expert]`` partial at a 128-row
tile. Those narrow stores take most of the CTA's life at the decode row counts on R200. This
kernel is the same GEMM with the epilogue
split in two:

1. every epilogue thread reads its lane of the two accumulators from tensor memory, adds them per
   element (the served statements) and stores the FP32 sums into a ``[BN, 128]`` row-major tile in
   shared memory (row = the call's row, column = the expert);
2. after a barrier of the epilogue warps, the tile goes to global memory either as 16-byte vector
   stores (``EPI_MODE`` ``"v4"``: each warp writes 512 contiguous bytes of one row per instruction)
   or as one 512-byte bulk copy per valid row (``"bulk"``, ``cp.async.bulk`` from shared to global,
   waited for before the CTA exits).

Arithmetic: unchanged. The pipeline, the K tiles a CTA owns, the weight split, the MMAs, the
accumulators and the per-element add are the served kernel's; only the path of the finished FP32
values to the same global addresses changes. So the partials, and the sum kernel's logits, are
bit-identical to ``Sm100BF16x3RouterGemm``'s. ``weight_prologue=True`` also loads the CTA's first
weight tiles before the dependency wait (same bits).

Memory order and progress (PDL). Reads before the wait touch only the weight. The partials are
written only after the epilogue warps' ``griddepcontrol.wait`` (so a buffer the previous layer's sum
kernel read is written after that kernel completed). The shared tile is written by the generic
proxy and read either by the same warps after a named barrier (``"v4"``) or by the bulk-copy
(async) proxy after ``fence.proxy.async.shared::cta`` and the barrier (``"bulk"``); each issuing
thread waits for its bulk group's reads before the CTA's final barrier, so no shared memory is
released while a copy reads it, and the grid's completion (which the sum kernel's
``griddepcontrol.wait`` observes) includes the copies' writes. No warp waits on a producer that is
itself blocked, so a CTA completes for any number of K tiles.

Provenance: the feature branch's sm107 target module of the same name
(``muse_spark_open_816b_a42b_nvfp4__sm_107__tep/ops/moe/router_gemm_staged.py``, #21754, 736cfdc7c),
statement for statement; only this paragraph is new. In this target ``ops/moe/router_gemm`` runs it
for the router GEMM calls of ``STAGED_MIN_ROWS..STAGED_MAX_ROWS`` rows under
``router_staged_gemm`` (default off).
"""

from __future__ import annotations

from functools import cache

import cutlass
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import dsl_user_op

from tensorrt_llm._torch.cute_dsl_kernels.blackwell import router_tcgen05 as _tcgen05
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.bf16x3_router_gemm import (
    _decompose_fp32x2_to_3xbf16x2,
)
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.cute_ptx_utils import simple_tma_copy

#: Epilogue modes: 16-byte vector stores from the shared tile, or one bulk copy per row.
EPI_MODES = ("v4", "bulk")
#: Shared-memory slots for weight tiles when the weight prologue is on (every K tile a CTA owns
#: at up to 384 rows on 212 SMs).
W_SLOTS = 4


@dsl_user_op
def _st_shared_f32(addr: Int32, value: Float32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Float32(value).ir_value(loc=loc, ip=ip)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _copy_shared_to_global_v4(saddr: Int32, gaddr: Int64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int32(saddr).ir_value(loc=loc, ip=ip), Int64(gaddr).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .f32 a, b, c, d;\n\t"
        "ld.shared.v4.f32 {a, b, c, d}, [$0];\n\t"
        "st.global.v4.f32 [$1], {a, b, c, d};\n\t}",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _fence_proxy_async_shared(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [],
        "fence.proxy.async.shared::cta;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_shared_to_global_512(saddr: Int32, gaddr: Int64, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int64(gaddr).ir_value(loc=loc, ip=ip), Int32(saddr).ir_value(loc=loc, ip=ip)],
        "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], 512;\n\t"
        "cp.async.bulk.commit_group;\n\t"
        "cp.async.bulk.wait_group.read 0;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class Sm100BF16x3RouterGemmStaged:
    """``Sm100BF16x3RouterGemm`` (same tile, warps, pipeline, arithmetic) with a shared-memory epilogue."""

    def __init__(self, BN: int = 128, epi_mode: str = "v4", weight_prologue: bool = False) -> None:
        if epi_mode not in EPI_MODES:
            raise ValueError(f"epi_mode {epi_mode!r} not in {EPI_MODES}")
        if BN % 8:
            raise ValueError(f"BN {BN} must be a multiple of 8")
        self.cta_tile = (BN, 128, 64)
        self.num_stages = 2
        self.num_warps = 10
        self.epi_mode = epi_mode
        self.weight_prologue = weight_prologue

    @cute.jit
    def _make_tma(self, tensor: cute.Tensor, BM: int, BK: int, stages: int):
        op = cpasync.CopyBulkTensorTileG2SOp()
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        elems = 128 * 8 // tensor.element_type.width  # 128B
        slayout = cute.make_layout(
            (BM, (elems, BK // elems), stages),
            stride=(elems, (1, BM * elems), BM * BK),
        )
        slayout = cute.make_composed_layout(swizzle_128B, 0, slayout)
        return cpasync.make_tiled_tma_atom(op, tensor, slayout, (BM, BK))

    @cute.jit
    def __call__(
        self,
        X: cute.Tensor,
        W: cute.Tensor,
        out: cute.Tensor,
        split_k: Int32,
        stream: CUstream,
    ):
        BN, BM, BK = self.cta_tile
        w_stages = W_SLOTS if cutlass.const_expr(self.weight_prologue) else self.num_stages
        W_tma = self._make_tma(W, BM, BK, w_stages)
        X_tma = self._make_tma(X, BN, BK, self.num_stages)

        grid_m = cute.ceil_div(W.shape[0], BM)
        grid_n = cute.ceil_div(X.shape[0], BN)

        self.kernel(X_tma, W_tma, out).launch(
            grid=(grid_m, grid_n, split_k),
            block=(self.num_warps * 32, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.jit
    def _load_w(
        self, W_tma, gW_tiles, sW, w_full_mbar, w_free_mbar, tile_k, jt, WS: cutlass.Constexpr
    ):
        BN, BM, BK = self.cta_tile
        slot = jt % WS
        cute.arch.mbarrier_wait(w_free_mbar + slot, ((jt // WS) & 1) ^ 1)
        mbar = w_full_mbar + slot
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(mbar, BM * BK * 4)
        simple_tma_copy(W_tma.atom, gW_tiles[None, None, tile_k], sW[None, None, slot], mbar)

    @cute.kernel
    def kernel(self, X_tma: cpasync.TmaInfo, W_tma: cpasync.TmaInfo, out: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid_m, bid_n, bid_k = cute.arch.block_idx()
        _, _, split_k = cute.arch.grid_dim()

        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_id = cute.arch.lane_idx()

        BN, BM, BK = self.cta_tile
        STAGES = self.num_stages
        PROLOGUE = self.weight_prologue
        WS = W_SLOTS if cutlass.const_expr(PROLOGUE) else STAGES

        N, K = X_tma.tma_tensor.shape
        M, _ = W_tma.tma_tensor.shape
        k_tiles = cute.ceil_div(K, BK)

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            BFloat16,
            X_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=X_tma.smem_layout.inner,
        )
        sW = smem.allocate_tensor(
            Float32,
            W_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=W_tma.smem_layout.inner,
        )

        w_full_mbar = smem.allocate_array(Int64, WS)
        w_free_mbar = smem.allocate_array(Int64, WS)
        x_full_mbar = smem.allocate_array(Int64, STAGES)
        x_empty_mbar = smem.allocate_array(Int64, STAGES)
        d_full_mbar = smem.allocate_array(Int64, STAGES)
        d_empty_mbar = smem.allocate_array(Int64, STAGES)
        mma_mbar = smem.allocate_array(Int64, 1)
        taddr = smem.allocate(Int32, 4)

        BAR_PREP = 2
        BAR_EPI = 3
        BAR_PREP_DONE = 4

        acc_main = 0
        acc_res = BN
        w_tmem_base = BN * 2

        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(WS):
                    cute.arch.mbarrier_init(w_full_mbar + i, 1)
                    cute.arch.mbarrier_init(w_free_mbar + i, 1)
                for i in cutlass.range_constexpr(STAGES):
                    cute.arch.mbarrier_init(x_full_mbar + i, 1)
                    cute.arch.mbarrier_init(x_empty_mbar + i, 1)
                    cute.arch.mbarrier_init(d_full_mbar + i, 128)
                    cute.arch.mbarrier_init(d_empty_mbar + i, 1)
                cute.arch.mbarrier_init(mma_mbar, 1)
                cute.arch.mbarrier_init_fence()
            # All 512 columns, before any warp can write tensor memory.
            _tcgen05.alloc(taddr)
        elif warp_id == 1:
            cpasync.prefetch_descriptor(X_tma.atom)
            cpasync.prefetch_descriptor(W_tma.atom)
        cute.arch.sync_threads()

        if warp_id == 9:
            # TMA warp.
            gW_tiles = cute.local_tile(W_tma.tma_tensor, (BM, BK), (bid_m, None))
            gX_tiles = cute.local_tile(X_tma.tma_tensor, (BN, BK), (bid_n, None))
            if cutlass.const_expr(PROLOGUE):
                jt = Int32(0)
                for tile_k in cutlass.range(bid_k, k_tiles, split_k, unroll=1):
                    if jt < WS:
                        mbar = w_full_mbar + jt
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(mbar, BM * BK * 4)
                        simple_tma_copy(
                            W_tma.atom, gW_tiles[None, None, tile_k], sW[None, None, jt], mbar
                        )
                    jt = jt + 1
            cute.arch.griddepcontrol_wait()
            jt = Int32(0)
            for tile_k in cutlass.range(bid_k, k_tiles, split_k, unroll=1):
                if cutlass.const_expr(PROLOGUE):
                    if jt >= WS:
                        self._load_w(W_tma, gW_tiles, sW, w_full_mbar, w_free_mbar, tile_k, jt, WS)
                else:
                    self._load_w(W_tma, gW_tiles, sW, w_full_mbar, w_free_mbar, tile_k, jt, WS)
                stage = jt % STAGES
                cute.arch.mbarrier_wait(x_empty_mbar + stage, ((jt // STAGES) & 1) ^ 1)
                mbar = x_full_mbar + stage
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar, BN * BK * 2)
                simple_tma_copy(
                    X_tma.atom, gX_tiles[None, None, tile_k], sX[None, None, stage], mbar
                )
                jt = jt + 1

        elif warp_id == 8:
            # MMA warp: the served MMAs, in the served order.
            idesc = _tcgen05.make_bf16_idesc(BM, BN)
            sdesc = _tcgen05.make_sdesc_128B_swizzle(0)
            jm = Int32(0)
            for tile_k in cutlass.range(bid_k, k_tiles, split_k, unroll=1):
                stage = jm % STAGES
                parity = (jm // STAGES) & 1
                cute.arch.mbarrier_wait(x_full_mbar + stage, parity)
                cute.arch.mbarrier_wait(d_full_mbar + stage, parity)
                _tcgen05.fence_after_thread_sync()
                w_tmem = w_tmem_base + stage * (BK // 2 * 3)
                x_desc = sdesc | (sX[None, None, stage].iterator.toint() >> 4)
                for k in cutlass.range_constexpr(BK // 16):
                    enable_d = (tile_k > bid_k) or (k > 0)
                    _tcgen05.mma_ts_f16(acc_main, w_tmem + k * 8, x_desc, idesc, enable_d)
                    _tcgen05.mma_ts_f16(acc_res, w_tmem + 32 + k * 8, x_desc, idesc, enable_d)
                    _tcgen05.mma_ts_f16(acc_res, w_tmem + 64 + k * 8, x_desc, idesc, True)
                    x_desc += 32 >> 4
                _tcgen05.commit(x_empty_mbar + stage)
                _tcgen05.commit(d_empty_mbar + stage)
                jm = jm + 1
            _tcgen05.commit(mma_mbar)

        elif warp_id >= 4:
            # Prep warps: split each FP32 weight tile into 3 BF16 terms in tensor memory.
            warp_id_ = warp_id % 4
            op = cute.nvgpu.CopyUniversalOp()
            cp_atom = cute.make_copy_atom(op, Float32, num_bits_per_copy=128)
            row = warp_id_ * 32 + lane_id
            sW_view = cute.zipped_divide(sW[row, None, None], (4, 1))
            jp = Int32(0)
            for _ in cutlass.range(bid_k, k_tiles, split_k, unroll=1):
                slot = jp % WS
                stage = jp % STAGES
                if warp_id_ == 0:
                    cute.arch.mbarrier_wait(w_full_mbar + slot, (jp // WS) & 1)
                    cute.arch.mbarrier_wait(d_empty_mbar + stage, ((jp // STAGES) & 1) ^ 1)
                cute.arch.barrier(barrier_id=BAR_PREP, number_of_threads=128)
                _tcgen05.fence_after_thread_sync()
                w_tmem = w_tmem_base + stage * (BK // 2 * 3)
                for kblock in cutlass.range_constexpr(BK // 4):
                    w0 = cute.make_rmem_tensor(2, Uint32)
                    w1 = cute.make_rmem_tensor(2, Uint32)
                    w2 = cute.make_rmem_tensor(2, Uint32)
                    w_tmp = cute.make_rmem_tensor(4, Float32)
                    cute.copy(cp_atom, sW_view[None, (kblock, slot)], w_tmp)
                    w0[0], w1[0], w2[0] = _decompose_fp32x2_to_3xbf16x2(w_tmp[0], w_tmp[1])
                    w0[1], w1[1], w2[1] = _decompose_fp32x2_to_3xbf16x2(w_tmp[2], w_tmp[3])
                    tcol = kblock * 2
                    _tcgen05.st(warp_id_ * 32, w_tmem + 0 + tcol, "32x32b", 2, w0)
                    _tcgen05.st(warp_id_ * 32, w_tmem + 32 + tcol, "32x32b", 2, w1)
                    _tcgen05.st(warp_id_ * 32, w_tmem + 64 + tcol, "32x32b", 2, w2)
                _tcgen05.wait_st()
                _tcgen05.fence_before_thread_sync()
                cute.arch.mbarrier_arrive(d_full_mbar + stage)
                # Every prep thread has read the slot: it may be refilled.
                cute.arch.barrier(barrier_id=BAR_PREP_DONE, number_of_threads=128)
                if warp_id_ == 0:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(w_free_mbar + slot)
                jp = jp + 1

        else:
            # Epilogue warps.
            cute.arch.griddepcontrol_wait()
            if warp_id == 0:
                cute.arch.mbarrier_wait(mma_mbar, 0)
            cute.arch.barrier(barrier_id=BAR_EPI, number_of_threads=128)
            _tcgen05.fence_after_thread_sync()
            cute.arch.griddepcontrol_launch_dependents()

            # The weight slots are free once every MMA completed (the prep warps read every landed
            # tile before the MMAs that consumed it): they hold the [BN, 128] FP32 tile (<= 64 KiB).
            _fence_proxy_async_shared()
            s_base = sW.iterator.toint()
            WIDTH = 8
            for i in cutlass.range_constexpr(BN // WIDTH):
                tcol = i * WIDTH
                main_regs = cute.make_rmem_tensor(WIDTH, Float32)
                res_regs = cute.make_rmem_tensor(WIDTH, Float32)
                main_regs.store(_tcgen05.ld(warp_id * 32, tcol, "32x32b", WIDTH))
                res_regs.store(_tcgen05.ld(warp_id * 32, BN + tcol, "32x32b", WIDTH))
                _tcgen05.wait_ld()
                for j in cutlass.range(WIDTH, vectorize=True):
                    main_regs[j] += res_regs[j]
                for j in cutlass.range_constexpr(WIDTH):
                    # Shared tile: row = the call's row within the tile, column = the expert lane.
                    _st_shared_f32(s_base + ((tcol + j) * BM + tid) * 4, main_regs[j])

            if cutlass.const_expr(self.epi_mode == "bulk"):
                _fence_proxy_async_shared()
            cute.arch.barrier(barrier_id=BAR_EPI, number_of_threads=128)

            g_base = out.iterator.toint()
            if cutlass.const_expr(self.epi_mode == "v4"):
                # 32 float4 per 128-expert row; 128 threads cover 4 rows per pass.
                for p in cutlass.range_constexpr(BN // 4):
                    r = p * 4 + tid // 32
                    c4 = tid % 32
                    x_row = bid_n * BN + r
                    if x_row < N:
                        g_off = (Int64(bid_k) * N + x_row) * M + bid_m * BM + c4 * 4
                        _copy_shared_to_global_v4(
                            s_base + (r * BM + c4 * 4) * 4, g_base + g_off * 4
                        )
            else:
                if tid < BN:
                    x_row = bid_n * BN + tid
                    if x_row < N:
                        g_off = (Int64(bid_k) * N + x_row) * M + bid_m * BM
                        _bulk_shared_to_global_512(s_base + tid * BM * 4, g_base + g_off * 4)

            cute.arch.barrier(barrier_id=BAR_EPI, number_of_threads=128)
            if warp_id == 0:
                _tcgen05.dealloc()

    @cache
    @staticmethod
    def compile(BN: int = 128, K: int = 8192, epi_mode: str = "v4", weight_prologue: bool = False):
        N = cute.sym_int()
        M = cute.sym_int()
        SPLIT_K = cute.sym_int()
        X = cute.runtime.make_fake_compact_tensor(
            BFloat16, (N, K), stride_order=(1, 0), assumed_align=16
        )
        W = cute.runtime.make_fake_compact_tensor(
            Float32, (M, K), stride_order=(1, 0), assumed_align=16
        )
        out = cute.runtime.make_fake_compact_tensor(
            Float32, (SPLIT_K, N, M), stride_order=(2, 1, 0), assumed_align=16
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel = Sm100BF16x3RouterGemmStaged(BN, epi_mode, weight_prologue)
        return cute.compile(kernel, X, W, out, Int32(1), stream, options="--enable-tvm-ffi")
