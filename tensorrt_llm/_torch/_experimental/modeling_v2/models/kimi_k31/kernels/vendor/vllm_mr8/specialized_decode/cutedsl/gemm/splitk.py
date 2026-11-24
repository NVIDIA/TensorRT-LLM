# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from FlashInfer v0.7.0.post1 (Apache-2.0):
# flashinfer/gemm/kernels/dense_bf16_gemm_sm100_splitk.py
# Modified by NVIDIA for TensorRT-LLM (2026): import statements and the changes marked 'TensorRT-LLM local'.
"""FlashInfer 0.7.0.post1 ``SplitKDenseGemmKernel`` (``dense_bf16_gemm_sm100_splitk``)
reduced to our use (PDL, no bias) with the weight (kernel-A) TMA issued through a
device tensor map with an L2 evict_first policy, and its synchronization fixed: the
owner waits for the peers' ``st.async`` partials at ``.acquire.cluster``; the MMA warp
waits for TMEM-pointer phase 0 before its release arrival (else its phase-1 wait can
pass before the epilogue has read TMEM); ``tcgen05.fence::*_thread_sync`` around
every mbarrier hand-off of tcgen05 work.

Each cluster rank accumulates an exact K slice in FP32; peers publish partials to
rank 0 through DSMEM, which reduces, casts and stores once. Kernel-M carries public N
and kernel-N public M."""

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import dsl_user_op
from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
    OWNER_RANK,
    SplitKTactic,
    _store_shared_remote_v4,
)

from ..common.primitives import shared_address
from ..common.w8 import convert_block, stage_rows

CTA_K, ATOM_K = 128, 64  # K per tile; K per 128-byte swizzle atom (one A box)
# TensorRT-LLM local (K3.1 W8A16): shared memory a CTA may allocate (sm_100 opt-in
# limit less barriers and alignment slack). W8A16 splits it between the BF16 A/B ring
# (the tiles converted before the PDL wait) and the FP8 staging ring (the tiles in
# flight from global memory), giving up ring stages until staging holds as many.
SMEM_BUDGET = 227 * 1024 - 3 * 1024
W8_MAX_STAGES = 32
# W8A16 converter groups (128 threads each, the first the epilogue warps): group g
# converts K tiles g, g + W8_GROUPS, ... so the tiles' load-convert-fence chains overlap.
W8_GROUPS = 4

__all__ = ["SplitKTactic", "SplitKGemm"]


def _asm(code, constraints, operands, loc, ip):
    llvm.inline_asm(
        None, operands, code, constraints, has_side_effects=True,
        is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )  # fmt: skip


@dsl_user_op
def _acquire_tensor_map(tensor_map, *, loc=None, ip=None):
    code = "fence.proxy.tensormap::generic.acquire.sys [$0], 128; "
    _asm(code + "prefetch.tensormap [$0];", "l", [tensor_map.ir_value()], loc, ip)


@dsl_user_op
def _tma_load_evict_first(smem, mbar, tensor_map, k0, row0, *, loc=None, ip=None):
    """One {64, rows} box of A at (k0, row0), L2 evict_first."""
    code = "{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol, 1.0; "
    code += "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx"
    code += "::bytes.L2::cache_hint [$0], [$1, {$2, $3}], [$4], pol; }"
    operands = [smem.toint().ir_value(), tensor_map.ir_value(), k0.ir_value()]
    operands += [row0.ir_value(), mbar.toint().ir_value()]
    _asm(code, "r,l,r,r,r", operands, loc, ip)


@dsl_user_op
def _wait_cluster(mbar, phase, *, loc=None, ip=None):
    """``cute.arch.mbarrier_wait`` at ``.acquire.cluster`` (remote ``st.async``)."""
    code = "{ .reg .pred P1; LAB_WAIT: "
    code += "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [$0], $1, "
    code += "10000000; @P1 bra.uni DONE; bra.uni LAB_WAIT; DONE: }"
    _asm(code, "r,r", [mbar.toint().ir_value(), phase.ir_value()], loc, ip)


@dsl_user_op
def _fence_tcgen05(*, before=False, loc=None, ip=None):
    side = "before" if before else "after"
    _asm(f"tcgen05.fence::{side}_thread_sync;", "", [], loc, ip)


def _smem(dtype, size, alignment, layout=None):
    layout = cute.make_layout(size) if layout is None else layout
    return cute_ext.allocate(dtype, cute.AddressSpace.smem, layout, alignment=alignment)


class SplitKGemm:
    """BF16 GEMM with a cluster-local split-K reduction; ``a_map`` is the device
    address of A's tensor map (box {64, mma_m}, 128-byte swizzle).

    TensorRT-LLM local (K3.1 W8A16): with ``w8`` = (padded rows, K) A is a
    ``common.w8`` buffer at device address ``a_map`` (mA only gives the shape). The
    weight warp bulk-copies each K tile's E4M3 codes and UE8M0 scales into a
    deep staging ring, and ``W8_GROUPS`` converter groups (the first the epilogue
    warps, idle until the last MMA) convert them into the A stage as the same
    swizzled BF16 tile the TMA would load, before the PDL wait like the weights.
    Same MMAs, K order and split-K sums."""

    def __init__(self, *, tactic: SplitKTactic, w8=None):
        self.acc_dtype = cutlass.Float32
        self.cta_m, self.cta_n, self.cta_k = tactic.mma_m, tactic.mma_n, CTA_K
        self.num_ab_stage, self.split_k = tactic.ab_stages, tactic.split_k
        self.threads_per_cta, self.epilog_threads = 256, 128
        # ``trigger_after_wait`` (subclasses): the PDL dependents are released by the
        # activation warp right after its wait, instead of by the weight warp once
        # every weight tile is requested (possibly before the wait).
        self.trigger_after_wait = False
        self.mma_tiler_mn = (tactic.mma_m, tactic.mma_n)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape = (1, tactic.split_k, 1)
        values_per_thread = tactic.mma_m * tactic.mma_n // self.epilog_threads
        assert values_per_thread % 4 == 0, "remote stores move 4 values"
        self.mailbox_elements = (
            (tactic.split_k - 1) * self.epilog_threads * values_per_thread
        )
        self.expected_transaction_bytes = self.mailbox_elements * 4
        # TensorRT-LLM local (K3.1 W8A16): the converter threads and the staging ring's size.
        self.w8 = w8
        if w8 is not None:
            self.threads_per_cta = 128 + W8_GROUPS * self.epilog_threads
            self.w8_block = self.cta_m * 66  # one 64-wide K block: codes + scales
            self.w8_stage = self.w8_block * (self.cta_k // ATOM_K)
            a_b = (tactic.mma_m + tactic.mma_n) * self.cta_k * 2
            free = SMEM_BUDGET - self.mailbox_elements * 4

            def staging(ring):
                return min(W8_MAX_STAGES, (free - ring * a_b) // self.w8_stage)

            # fewer ring stages only change the pipelining (same MMAs and sums)
            while self.num_ab_stage > 2 and staging(self.num_ab_stage) < self.num_ab_stage:
                self.num_ab_stage -= 1
            self.w8_stages = staging(self.num_ab_stage)
            assert self.w8_stages >= 2, "W8A16 staging does not fit"

    @cute.experimental.jit
    def __call__(
        self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, bias: cute.Tensor,
        a_map: cutlass.Int64, stream: _cuda.CUstream,
    ):  # fmt: skip
        # Grid-y packs output-N tile and cluster rank.
        self.kernel(a, b, c, bias, a_map).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, mBias: cute.Tensor,
        a_map: cutlass.Int64,
    ):  # fmt: skip
        """A (M, K, L) and B (N, K, L) K-major, C (M, N, L) M-major; bias unused."""
        stages = self.num_ab_stage
        ab_dtype = mA.element_type
        majors = (utils.LayoutEnum.from_tensor(t).mma_major_mode() for t in (mA, mB))
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, *majors, self.acc_dtype, self.cta_group,
            self.mma_tiler_mn,
        )  # fmt: skip

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        shape = (tiled_mma, mnk_tiler, ab_dtype, stages)
        sA = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_a(*shape))
        sB = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_b(*shape))
        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma, self.mma_tiler_mn, acc_stage=1
        )
        bar_full = _smem(cutlass.Int64, stages, 8).iterator
        bar_empty = _smem(cutlass.Int64, stages, 8).iterator
        bar_tma_epilog = _smem(cutlass.Int64, 1, 8).iterator  # FlashInfer's bias
        bar_mma_epilog = _smem(cutlass.Int64, 1, 8).iterator
        bar_tmem_alloc = _smem(cutlass.Int64, 1, 8).iterator
        tmem_base_ptr = _smem(cutlass.Int32, 1, 4).iterator
        mailbox, bar_reduce = sA, bar_mma_epilog  # split 1: compile-time dead
        if cutlass.const_expr(self.split_k > 1):
            mailbox = _smem(cutlass.Float32, self.mailbox_elements, 128)
            bar_reduce = _smem(cutlass.Int64, 1, 8).iterator
        w8 = self.w8_allocate()  # TensorRT-LLM local (W8A16 staging; None otherwise)

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, self.full_arrivals())  # TensorRT-LLM local
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
                if cutlass.const_expr(self.split_k > 1):
                    # Owner arrival plus peer transaction-byte credits.
                    cute.arch.mbarrier_init(bar_reduce, 1)
                self.w8_init(w8)  # TensorRT-LLM local (W8A16)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_arrive_relaxed()  # peers' barriers before their stores
        else:
            cute.arch.barrier()

        # K splits evenly over the cluster (tactics are validated for the shapes).
        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        k_tile_start = split_rank * k_tile_count

        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()

        # Warp 3 is idle; warps 4-7 run the epilogue.
        tiles = (k_tile_start, k_tile_count)
        if warp_idx == 0:
            # TensorRT-LLM local (W8A16): the weight warp stages codes and scales instead.
            if cutlass.const_expr(self.w8 is not None):
                rows = ((bidx * self.cta_m, self.cta_m, 0),)
                self.w8_weight_warp(w8, a_map, rows, *tiles)
            else:
                row0 = bidx * self.cta_m
                self.weight_warp(bar_full, bar_empty, sA, a_map, row0, *tiles)
        elif warp_idx == 1:
            b_tile = cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx))
            b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
            self.activation_warp(bar_full, bar_empty, b_tile, sB, b_map, *tiles)
        elif warp_idx == 2:
            k_blocks = self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2])
            self.mma_warp(
                bar_full, bar_empty, bar_mma_epilog, bar_tmem_alloc, tiled_mma, sA, sB,
                tmem_base_ptr, acc_layout, k_blocks, k_tile_count,
            )  # fmt: skip
        elif warp_idx >= 4:
            # TensorRT-LLM local (W8A16): every converter group converts first; warps 4-7
            # then run the epilogue.
            if cutlass.const_expr(self.w8 is not None):
                self.w8_convert(w8, bar_full, bar_empty, sA, k_tile_count)
            if warp_idx < 8:
                tile = cute.local_tile(mC, (self.cta_m, self.cta_n), (bidx, n_idx, l_idx))
                self.epilog_warp(
                    bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr, acc_layout, tile,
                    cute.arch.thread_idx()[0] - 128, mC.element_type,
                    utils.LayoutEnum.from_tensor(mC), mailbox, bar_reduce, split_rank,
                )  # fmt: skip

    @cute.experimental.jit
    def weight_warp(
        self, bar_full, bar_empty, s_tile: cute.Tensor, a_map: cutlass.Int64,
        row0: cutlass.Int32, k_tile_start: cutlass.Int32, k_tile_count: cutlass.Int32,
    ):  # fmt: skip
        """A: two evict_first boxes per K tile, before the PDL wait (weights)."""
        stages = self.num_ab_stage
        stage_bytes = cute.size_in_bytes(
            s_tile.element_type, cute.slice_(s_tile.layout, (None, None, None, 0))
        )
        _acquire_tensor_map(a_map)
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bar_full + stage, stage_bytes)
            with cute.arch.elect_one():
                k0 = (k_tile_start + k_tile) * self.cta_k
                tile = s_tile[None, None, None, stage].iterator
                for atom in cutlass.range_constexpr(self.cta_k // ATOM_K):
                    box, mbar = tile + atom * self.cta_m * ATOM_K, bar_full + stage
                    k = k0 + atom * ATOM_K
                    _tma_load_evict_first(box, mbar, a_map, k, row0)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

        if cutlass.const_expr(not self.trigger_after_wait):
            cute.arch.griddepcontrol_launch_dependents()
        self.drain(bar_empty, empty_phase, k_tile_count)

    # ---- TensorRT-LLM local (K3.1 W8A16) ------------------------------------------
    def full_arrivals(self):
        """A stage's arrivals: the activation TMA's, plus the weight TMA's or (W8A16)
        one per converter thread."""
        return 2 if self.w8 is None else 1 + self.epilog_threads

    @cute.experimental.jit
    def w8_allocate(self):
        """(staging ring, its full barriers, its empty barriers), or None."""
        if cutlass.const_expr(self.w8 is None):
            return None
        staging = _smem(cutlass.Uint8, self.w8_stages * self.w8_stage, 1024)
        full = _smem(cutlass.Int64, self.w8_stages, 8).iterator
        empty = _smem(cutlass.Int64, self.w8_stages, 8).iterator
        return staging, full, empty

    @cute.experimental.jit
    def w8_init(self, w8):
        if cutlass.const_expr(self.w8 is not None):
            _, full, empty = w8
            for i in range(self.w8_stages):
                cute.arch.mbarrier_init(full + i, 1)
                cute.arch.mbarrier_init(empty + i, self.epilog_threads)

    @cute.experimental.jit
    def w8_weight_warp(
        self, w8, base: cutlass.Int64, parts,
        k_tile_start: cutlass.Int32, k_tile_count: cutlass.Int32,
    ):  # fmt: skip
        """Codes and scales of every K tile into the staging ring (``parts``: (first
        weight row, rows, staging row) copies per 64-wide K block)."""
        staging, full, empty = w8
        n_pad, k = self.w8
        base_smem = shared_address(staging.iterator)
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            st = k_tile % self.w8_stages
            cute.arch.mbarrier_wait(empty + st, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(full + st, self.w8_stage)
                block = base_smem + st * self.w8_stage
                for atom in cutlass.range_constexpr(self.cta_k // ATOM_K):
                    kb = (k_tile_start + k_tile) * (self.cta_k // ATOM_K) + atom
                    for row0, rows, at_row in parts:
                        stage_rows(
                            block + atom * self.w8_block, full + st, base, n_pad, k,
                            kb, row0, rows, at_row, self.cta_m,
                        )  # fmt: skip
            if st == self.w8_stages - 1:
                empty_phase = empty_phase ^ 1
        if cutlass.const_expr(not self.trigger_after_wait):
            cute.arch.griddepcontrol_launch_dependents()

    @cute.experimental.jit
    def w8_convert(
        self, w8, bar_full, bar_empty, s_tile: cute.Tensor, k_tile_count: cutlass.Int32,
    ):  # fmt: skip
        """Converter group g (warps 4 + 4g ..; the first is the epilogue's, before the
        epilogue): staged K tiles g, g + W8_GROUPS, ... -> their A stages."""
        staging, full, empty = w8
        stages = self.num_ab_stage
        base_smem = shared_address(staging.iterator)
        conv = cute.arch.thread_idx()[0] - 128
        epi_tid, group = conv % self.epilog_threads, conv // self.epilog_threads
        for k_tile in cutlass.range(group, k_tile_count, W8_GROUPS, unroll=1):
            stage, st = k_tile % stages, k_tile % self.w8_stages
            # the ring's previous use of the stage released (none in the first round)
            cute.arch.mbarrier_wait(bar_empty + stage, k_tile // stages % 2 ^ 1)
            cute.arch.mbarrier_wait(full + st, k_tile // self.w8_stages % 2)
            tile = shared_address(s_tile[None, None, None, stage].iterator)
            block = base_smem + st * self.w8_stage
            for atom in cutlass.range_constexpr(self.cta_k // ATOM_K):
                dst = tile + atom * self.cta_m * ATOM_K * 2
                convert_block(
                    block + atom * self.w8_block, dst, self.cta_m, self.epilog_threads,
                    epi_tid,
                )  # fmt: skip
            cute.arch.mbarrier_arrive(empty + st)
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.mbarrier_arrive(bar_full + stage)

    @cute.experimental.jit
    def activation_warp(
        self, bar_full, bar_empty, g_tile: cute.Tensor, s_tile: cute.Tensor,
        cta_v_map: cute.Layout, k_tile_start: cutlass.Int32,
        k_tile_count: cutlass.Int32,
    ):  # fmt: skip
        """B (the activations): after the PDL wait."""
        stages = self.num_ab_stage
        stage_bytes = cute.size_in_bytes(
            s_tile.element_type, cute.slice_(s_tile.layout, (None, None, None, 0))
        )
        cute.arch.griddepcontrol_wait()
        if cutlass.const_expr(self.trigger_after_wait):
            cute.arch.griddepcontrol_launch_dependents()
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bar_full + stage, stage_bytes)
            cute_ext.tma_load(
                g_tile[None, None, k_tile_start + k_tile],
                s_tile[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=cta_v_map,
                tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                update_expect_tx=False,
            )
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1
        self.drain(bar_empty, empty_phase, k_tile_count)

    @cute.experimental.jit
    def drain(self, bar_empty, empty_phase: cutlass.Int32, k_tile_count: cutlass.Int32):
        stages = self.num_ab_stage
        for tail in cutlass.range(stages, unroll=1):
            stage = (tail + k_tile_count) % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

    @cute.experimental.jit
    def mma_warp(
        self, bar_full, bar_empty, bar_mma_epilog, bar_tmem_alloc,
        tiled_mma: cute.TiledMma, sA: cute.Tensor, sB: cute.Tensor, tmem_base_ptr,
        acc_layout: cutlass.Constexpr, mma_inst_tile_k: cutlass.Constexpr,
        k_tile_count: cutlass.Int32,
    ):  # fmt: skip
        num_tmem_cols = 256
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr, is_two_cta=False)
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        accumulator = cute.make_tensor(tmem_ptr, acc_layout)[None, None, None, 0]
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        full_phase = cutlass.Int32(0)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % self.num_ab_stage
            cute.arch.mbarrier_wait(bar_full + stage, full_phase)
            _fence_tcgen05()
            for k_block in range(mma_inst_tile_k):
                accumulate = k_tile != 0 if k_block == 0 else True
                mma_atom.set(tcgen05.Field.ACCUMULATE, accumulate)
                cute_ext.dot(
                    mma_atom,
                    cute.append_ones(sA[None, None, k_block, stage], up_to_rank=3),
                    cute.append_ones(sB[None, None, k_block, stage], up_to_rank=3),
                    accumulator,
                )
            with cute.arch.elect_one():
                tcgen05.commit(bar_empty + stage, None, self.cta_group)
            if stage == self.num_ab_stage - 1:
                full_phase = full_phase ^ 1

        with cute.arch.elect_one():
            tcgen05.commit(bar_mma_epilog, None, self.cta_group)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)  # the release lands in phase 1
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        _fence_tcgen05()
        cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols, is_two_cta=False)

    @cute.experimental.jit
    def epilog_warp(
        self, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr,
        acc_layout: cutlass.Constexpr, gD_tile: cute.Tensor, epi_tid: cutlass.Int32,
        c_dtype: cutlass.Constexpr, d_layout: cutlass.Constexpr, mailbox, bar_reduce,
        split_rank: cutlass.Int32,
    ):  # fmt: skip
        # Wait until MMA publishes the TMEM base pointer.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        _fence_tcgen05()

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
            acc_layout,
        )[((None, None), 0, 0, 0)]
        epi_tile = (self.cta_m, self.cta_n)
        load_op = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k), d_layout, c_dtype, self.acc_dtype,
            epi_tile, False,
        )  # fmt: skip
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(load_op, acc_view)
        gD_epi = cute.flat_divide(gD_tile, epi_tile)

        # Match each epilogue thread's TMEM partition in RMEM.
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rmem = cute.AddressSpace.rmem
        rAcc = cute_ext.allocate(self.acc_dtype, rmem, rmem_layout, alignment=32)
        rD = cute_ext.allocate(c_dtype, rmem, rmem_layout, alignment=32)
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        _fence_tcgen05()
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        # Make tcgen05.ld visible before TMEM release and RMEM use.
        cute.arch.fence_view_async_tmem_load()
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        # Peers publish FP32 partials; only rank 0 reduces and stores.
        if cutlass.const_expr(self.split_k > 1):
            vpt = cutlass.const_expr(cute.size(rmem_layout))  # values per thread
            vpp = cutlass.const_expr(self.epilog_threads * vpt)  # values per peer
            if split_rank != OWNER_RANK:
                for i in cutlass.range_constexpr(0, vpt, 4):
                    words = (rAcc[i], rAcc[i + 1], rAcc[i + 2], rAcc[i + 3])
                    slot = mailbox.iterator + (split_rank - Int32(1)) * vpp
                    slot = slot + epi_tid * vpt + i
                    _store_shared_remote_v4(*words, slot, bar_reduce, Int32(OWNER_RANK))
            else:
                if epi_tid == 0:
                    expected = self.expected_transaction_bytes
                    cute.arch.mbarrier_arrive_and_expect_tx(bar_reduce, expected)
                _wait_cluster(bar_reduce, Int32(0))
                for peer in cutlass.range_constexpr(self.split_k - 1):
                    for i in cutlass.range_constexpr(vpt):
                        rAcc[i] = rAcc[i] + mailbox[peer * vpp + epi_tid * vpt + i]

        if split_rank == OWNER_RANK:
            rD.store(rAcc.load().to(c_dtype))
            # Preserve TMEM coordinates; the copy predicates output tails.
            cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])
