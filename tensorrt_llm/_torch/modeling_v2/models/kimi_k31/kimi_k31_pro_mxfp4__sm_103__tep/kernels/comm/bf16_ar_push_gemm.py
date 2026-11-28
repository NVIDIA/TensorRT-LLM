# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from FlashInfer (Apache-2.0): flashinfer/gemm/kernels/dense_bf16_gemm_sm100_splitk.py
"""Low-M BF16 projection whose epilogue pushes its output into the next MNNVL all-reduce (SM100 / SM103).

``out[M, N] = x[M, K] @ W[N, K]^T`` for a row-parallel BF16 projection followed by an MNNVL all-reduce (the Kimi K3.1
attention ``o_proj``): instead of storing ``out``, the owner CTA of each output tile stores its BF16 rows into the
Lamport slots of the all-reduce (:mod:`....kernels.comm.lamport_ar`), each token row to its owner's scatter slot
(two-shot) or to every rank's slot (one-shot, multicast), with ``-0`` sent as ``+0`` as the C++ all-reduce sends it. The
consumer then finishes the all-reduce (``lamport_ar.run_lamport_finish`` or the attention-residual mix with
``lamport_finish``): the GEMM output reaches the network without a store, a reload and a copy kernel.

The GEMM is a swap-AB tcgen05 split-K kernel:
kernel-M carries the output columns (weight rows), kernel-N the tokens, each cluster rank accumulates an exact K slice
in FP32 and the peers publish their partials to rank 0 through DSMEM; rank 0 adds them in rank order and rounds once
to BF16. The weight loads do not wait for the producer grid (PDL); the token loads do. The tactic table covers the
row counts where this GEMM + the finish beats cuBLAS + the C++ all-reduce (TP8 up to 72 rows, TP16 and TP32 up to
192).

Ordering (the rules of :mod:`....kernels.comm.lamport_ar`): the epilogue reads the workspace's rotation state after
its own ``griddepcontrol.wait``, and the grid releases its dependent only from the token producer, right after that
warp's wait, so a consumer that reads the state as soon as it is resident sees this call's state.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Optional

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.experimental import primitives as prims

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.comm import lamport_ar as _lam
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.comm.lamport_ar import (
    mbarrier_test_wait_acquire_cluster,
)

# ---------------------------------------------------------------------------
# Split-K GEMM helpers (shared with the fused MoE tail's up-projection)
# ---------------------------------------------------------------------------

#: Per-CTA SMEM capacity reported by CuTeDSL on SM100/SM103.
_SMEM_CAPACITY_BYTES = utils.get_smem_capacity_in_bytes("sm_100")

#: K extent of one CTA tile.
_CTA_K = 128

#: Bytes per FP32 partial exchanged through DSMEM.
_FP32_BYTES = 4

#: DSMEM mailbox base alignment, in bytes.
_MAILBOX_ALIGN_BYTES = 128

#: Size and alignment of one mbarrier, in bytes.
_MBARRIER_BYTES = 8

#: Size and alignment of the TMEM base pointer slot.
_TMEM_POINTER_BYTES = 4

#: Alignment of the staged output tile.
_TRANSPOSE_ALIGN_BYTES = 128

#: Bytes per BF16/FP16 element.
_AB_ELEMENT_BYTES = 2

#: Alignment of the A/B shared-memory buffers.
_AB_BUFFER_ALIGN_BYTES = 1024

#: Most A/B pipeline stages a tactic may use.
_MAX_AB_STAGES = 12


@dataclasses.dataclass(frozen=True)
class SplitKTactic:
    """One specialization; mma_m carries public N and mma_n carries public M."""

    mma_m: int
    mma_n: int
    split_k: int
    ab_stages: int


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _smem_bytes(
    tactic: SplitKTactic,
    ab_stages: int,
) -> int:
    """Mirror the device allocator's shared-memory layout."""
    cursor = (
        _align_up(
            tactic.mma_m * _CTA_K * _AB_ELEMENT_BYTES * ab_stages,
            _AB_BUFFER_ALIGN_BYTES,
        )
        + tactic.mma_n * _CTA_K * _AB_ELEMENT_BYTES * ab_stages
    )

    cursor = _align_up(cursor, _MBARRIER_BYTES)
    cursor += 2 * ab_stages * _MBARRIER_BYTES
    cursor += 3 * _MBARRIER_BYTES
    cursor = _align_up(cursor, _TMEM_POINTER_BYTES)
    cursor += _TMEM_POINTER_BYTES
    # Output tile staged for 16-byte multicast stores.
    cursor = (
        _align_up(cursor, _TRANSPOSE_ALIGN_BYTES) + tactic.mma_m * tactic.mma_n * _AB_ELEMENT_BYTES
    )

    if tactic.split_k == 1:
        return cursor

    return (
        _align_up(
            _align_up(cursor, _MAILBOX_ALIGN_BYTES)
            + (tactic.split_k - 1) * tactic.mma_m * tactic.mma_n * _FP32_BYTES,
            _MBARRIER_BYTES,
        )
        + _MBARRIER_BYTES
    )


def _max_ab_stages_for(
    tactic: SplitKTactic,
    smem_capacity: int,
) -> int:
    return next(
        (
            stages
            for stages in range(_MAX_AB_STAGES, -1, -1)
            if _smem_bytes(tactic, stages) <= smem_capacity
        ),
        0,
    )


@dsl_user_op
def _map_shared_rank(
    smem_ptr: "cute.Pointer",
    peer_cta_rank_in_cluster: "Int32",
    *,
    loc=None,
    ip=None,
) -> "Int32":
    """Map an SMEM pointer into a peer CTA's address space."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                smem_ptr.toint(loc=loc, ip=ip).ir_value(),
                peer_cta_rank_in_cluster.ir_value(),
            ],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _store_shared_remote_v4(
    value0,
    value1,
    value2,
    value3,
    smem_ptr: "cute.Pointer",
    mbar_ptr: "cute.Pointer",
    peer_cta_rank_in_cluster: "Int32",
    *,
    loc=None,
    ip=None,
) -> None:
    """Publish four FP32 partials into a peer's SMEM, crediting 16 bytes."""
    llvm.inline_asm(
        None,
        [
            _map_shared_rank(smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip).ir_value(),
            value0.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value1.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value2.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value3.bitcast(Int32).ir_value(loc=loc, ip=ip),
            _map_shared_rank(mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip).ir_value(),
        ],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
        "[$0], {$1, $2, $3, $4}, [$5];",
        "r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


#: Rank that gathers partials and stores the output.
OWNER_RANK = 0

#: Named barrier of the four epilogue warps (0 is the CTA barrier).
_EPILOG_BARRIER_ID = 3


@dsl_user_op
def _load_shared_u32x4(smem_ptr: "cute.Pointer", *, loc=None, ip=None):
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import vector

    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [smem_ptr.toint(loc=loc, ip=ip).ir_value()],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32()),
        [llvm.extractvalue(T.i32(), out, [i]) for i in range(4)],
    )
    return cute.TensorSSA(packed, 4, cutlass.Uint32)


# ---------------------------------------------------------------------------
# Push GEMM
# ---------------------------------------------------------------------------


#: Largest token count the kernel serves.
MAX_M = 192


class Bf16ArPushGemmKernel:
    """BF16 GEMM with a cluster-local split-K reduction whose epilogue pushes the BF16 output
    into the Lamport slots of the next MNNVL all-reduce (see the module docstring)."""

    def __init__(
        self,
        *,
        tactic: SplitKTactic,
        use_pdl: bool,
        hidden_dim: int,
        world: int,
        one_shot: bool,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.world = world
        self.one_shot = one_shot
        self.smem_bytes = min(
            _align_up(_smem_bytes(tactic, tactic.ab_stages), 1024) + 1024, _SMEM_CAPACITY_BYTES
        )
        self.acc_dtype = cutlass.Float32
        self.cta_m = tactic.mma_m
        self.cta_n = tactic.mma_n
        self.cta_k = _CTA_K
        self.num_ab_stage = tactic.ab_stages
        self.split_k = tactic.split_k
        self.use_pdl = use_pdl
        self.has_bias = False

        self.threads_per_cta = 256
        self.epilog_threads = 128
        self.mma_tiler_mn = (tactic.mma_m, tactic.mma_n)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tma_op = cute_ext.OperationTypeEnum.SM90_TMA_LOAD
        self.cluster_shape = (1, tactic.split_k, 1)

        values_per_thread = (tactic.mma_m * tactic.mma_n) // self.epilog_threads
        if values_per_thread % 4:
            raise ValueError(
                f"CTA tile ({tactic.mma_m}, {tactic.mma_n}) gives "
                f"{values_per_thread} "
                "values per epilogue thread; remote stores require a multiple of 4"
            )
        self.mailbox_elements = (tactic.split_k - 1) * self.epilog_threads * values_per_thread
        self.expected_transaction_bytes = self.mailbox_elements * _FP32_BYTES

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        peer_ptrs: cutlass.Int64,
        mc_base: cutlass.Int64,
        flags: cutlass.Int64,
        rank: cutlass.Int32,
        stream: _cuda.CUstream,
    ):
        # Grid-y packs output-N tile and cluster rank.
        self.kernel(a, b, c, peer_ptrs, mc_base, flags, rank).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            # The tactic's own footprint (not the whole SMEM) so small tiles
            # can run two CTAs per SM.
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major
        mB: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major
        mC: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M-major: only its tiling is used
        peer_ptrs: cutlass.Int64,
        mc_base: cutlass.Int64,
        flags: cutlass.Int64,
        rank: cutlass.Int32,
    ):
        """Allocate storage and dispatch the specialized warps."""
        stages = self.num_ab_stage

        ab_dtype = mA.element_type
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )

        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        sA = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )

        acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)
        c_tiler_mn = (self.cta_m, self.cta_n)

        bar_full = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_empty = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_tma_epilog = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_mma_epilog = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_tmem_alloc = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        tmem_base_ptr = cute_ext.allocate(
            cutlass.Int32,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_TMEM_POINTER_BYTES,
        ).iterator
        # Column-contiguous output tile: lets the owner store 16-byte vectors
        # (the TMEM partition gives each thread one column of many tokens).
        staging = cute_ext.allocate(
            mC.element_type,
            cute.AddressSpace.smem,
            cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.cta_m)),
            alignment=_TRANSPOSE_ALIGN_BYTES,
        )

        if cutlass.const_expr(self.split_k > 1):
            mailbox = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(self.mailbox_elements),
                alignment=_MAILBOX_ALIGN_BYTES,
            )
            bar_reduce = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_MBARRIER_BYTES,
            ).iterator
        else:
            # Dummy operands for the compile-time-elided reduction.
            mailbox = sA
            bar_reduce = bar_mma_epilog

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)

                if cutlass.const_expr(self.split_k > 1):
                    # Owner arrival plus peer transaction-byte credits.
                    cute.arch.mbarrier_init(bar_reduce, 1)

        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1):
            # Publish peer barriers before cross-CTA stores.
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        # Host validation guarantees an equal K partition; a K tail (one split only) is the last tile, whose
        # columns past K the TMA loads as zeros.
        k_tile_count = cute.ceil_div(cute.size(mA, mode=[1]), self.cta_k) // self.split_k
        k_tile_start = split_rank * k_tile_count

        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()

        # Warp 3 is idle; warps 4-7 run the epilogue.
        if warp_idx == 0:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(mA, (self.cta_m, self.cta_k), (bidx, None, l_idx)),
                sA,
                cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
                k_tile_start,
                k_tile_count,
                True,
            )
        elif warp_idx == 1:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx)),
                sB,
                cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B"),
                k_tile_start,
                k_tile_count,
                False,
            )
        elif warp_idx == 2:
            self.mma_warp(
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2]),
                k_tile_count,
            )
        elif warp_idx >= 4:
            self.epilog_warp(
                bar_tma_epilog,
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                cute.local_tile(mC, c_tiler_mn, (bidx, n_idx, l_idx)),
                cute.arch.thread_idx()[0] - 128,
                mC.element_type,
                utils.LayoutEnum.from_tensor(mC),
                mailbox,
                bar_reduce,
                split_rank,
                staging,
                cute.size(mC, mode=[1]),
                n_idx,
                bidx,
                peer_ptrs,
                mc_base,
                flags,
                rank,
            )

    @cute.experimental.jit
    def dma_warp(
        self,
        bar_full,
        bar_empty,
        bar_tma_epilog,
        g_tile: cute.Tensor,
        s_tile: cute.Tensor,
        cta_v_map: cute.Layout,
        k_tile_start: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        is_a: cutlass.Constexpr,
    ):
        stages = self.num_ab_stage
        if cutlass.const_expr(not is_a and self.use_pdl):
            cute.arch.griddepcontrol_wait()
            # The dependent (the consumer that finishes the all-reduce) reads the workspace's
            # rotation state as soon as it is resident: release it only once this grid has
            # passed its own wait, i.e. every earlier grid (the previous call included) is done.
            cute.arch.griddepcontrol_launch_dependents()

        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    cute.size_in_bytes(
                        s_tile.element_type,
                        cute.slice_(s_tile.layout, (None, None, None, 0)),
                    ),
                )
            cute_ext.tma_load(
                g_tile[None, None, k_tile_start + k_tile],
                s_tile[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

        self._drain_producer(bar_empty, empty_phase, k_tile_count)

    @cute.experimental.jit
    def _drain_producer(
        self,
        bar_empty,
        empty_phase: cutlass.Int32,
        k_tile_count: cutlass.Int32,
    ):
        stages = self.num_ab_stage
        for tail in cutlass.range(stages, unroll=1):
            stage = (tail + k_tile_count) % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1

    @cute.experimental.jit
    def mma_warp(
        self,
        bar_full,
        bar_empty,
        bar_mma_epilog,
        bar_tmem_alloc,
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,
        sB: cute.Tensor,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        mma_inst_tile_k: cutlass.Constexpr,
        k_tile_count: cutlass.Int32,
    ):
        num_tmem_cols = 256
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr, is_two_cta=False)
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        accumulator = cute.make_tensor(tmem_ptr, acc_layout)[None, None, None, 0]
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        full_phase = cutlass.Int32(0)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % self.num_ab_stage
            cute.arch.mbarrier_wait(bar_full + stage, full_phase)
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            for k_block in range(mma_inst_tile_k):
                if k_block == 0:
                    mma_atom.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                else:
                    mma_atom.set(tcgen05.Field.ACCUMULATE, True)
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
        # Phase 0 (the allocation, with the epilogue's first arrivals) completes before this warp's phase-1 arrival.
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols, is_two_cta=False)

    @cute.experimental.jit
    def epilog_warp(
        self,
        bar_tma_epilog,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,
        epi_tid: cutlass.Int32,
        c_dtype: cutlass.Constexpr,
        d_layout: cutlass.Constexpr,
        mailbox,
        bar_reduce,
        split_rank: cutlass.Int32,
        staging: cute.Tensor,
        rows: cutlass.Int32,
        n_idx: cutlass.Int32,
        m_idx: cutlass.Int32,
        peer_ptrs: cutlass.Int64,
        mc_base: cutlass.Int64,
        flags: cutlass.Int64,
        rank: cutlass.Int32,
    ):
        # Wait until MMA publishes the TMEM base pointer.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
            acc_layout,
        )[((None, None), 0, 0, 0)]

        epi_tile = (self.cta_m, self.cta_n)
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.cta_m, self.cta_n, self.cta_k),
                d_layout,
                c_dtype,
                self.acc_dtype,
                epi_tile,
                False,
            ),
            acc_view,
        )
        gD_epi = cute.flat_divide(gD_tile, epi_tile)

        # Match each epilogue thread's TMEM partition in RMEM.
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = cute_ext.allocate(
            self.acc_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        rD = cute_ext.allocate(
            c_dtype,
            cute.AddressSpace.rmem,
            rmem_layout,
            alignment=32,
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        # Make tcgen05.ld visible before TMEM release and RMEM use.
        cute.arch.fence_view_async_tmem_load()
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        # Peers publish FP32 partials; only rank 0 reduces and stores.
        if cutlass.const_expr(self.split_k > 1):
            assert cute.size(rmem_layout) == self.mailbox_elements // (
                (self.split_k - 1) * self.epilog_threads
            )
            values_per_thread = cutlass.const_expr(cute.size(rmem_layout))
            values_per_peer = cutlass.const_expr(self.epilog_threads * values_per_thread)
            if split_rank != OWNER_RANK:
                for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                    _store_shared_remote_v4(
                        rAcc[value_idx],
                        rAcc[value_idx + 1],
                        rAcc[value_idx + 2],
                        rAcc[value_idx + 3],
                        mailbox.iterator
                        + (split_rank - Int32(1)) * values_per_peer
                        + epi_tid * values_per_thread
                        + value_idx,
                        bar_reduce,
                        Int32(OWNER_RANK),
                    )
            else:
                if epi_tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bar_reduce, self.expected_transaction_bytes
                    )
                while mbarrier_test_wait_acquire_cluster(bar_reduce.toint(), Int32(0)) == 0:
                    pass
                for peer in cutlass.range_constexpr(self.split_k - 1):
                    for value_idx in cutlass.range_constexpr(values_per_thread):
                        rAcc[value_idx] = (
                            rAcc[value_idx]
                            + mailbox[
                                peer * values_per_peer + epi_tid * values_per_thread + value_idx
                            ]
                        )

        if split_rank == OWNER_RANK:
            # Round to BF16 (push_v4 sends -0 as +0, as the MNNVL all-reduce's copy phase does).
            rD.store(rAcc.load().to(c_dtype))
            cute_ext.partition_and_copy(thr_t2r, rD, staging)
            cute.arch.barrier(barrier_id=_EPILOG_BARRIER_ID, number_of_threads=self.epilog_threads)
            # The rotation state of the workspace is this call's once every earlier grid is done.
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            cur, _d, nbytes, _s, _c0, _c1, _c2, _c3 = _lam.lamport_state(flags)
            chunks_per_token = self.cta_m // 8
            total_chunks = self.cta_n * chunks_per_token
            for i in cutlass.range_constexpr(
                (total_chunks + self.epilog_threads - 1) // self.epilog_threads
            ):
                chunk = epi_tid + Int32(i * self.epilog_threads)
                token = chunk // Int32(chunks_per_token)
                column = (chunk - token * Int32(chunks_per_token)) * Int32(8)
                tok = n_idx * Int32(self.cta_n) + token
                if chunk < Int32(total_chunks) and tok < rows:
                    packed = _load_shared_u32x4(
                        staging.iterator + token * Int32(self.cta_m) + column
                    )
                    dst = _lam.push_slot(
                        peer_ptrs,
                        mc_base,
                        cur,
                        nbytes,
                        tok,
                        rank,
                        m_idx * Int32(self.cta_m) + column,
                        self.world,
                        self.hidden_dim,
                        self.one_shot,
                    )
                    _lam.push_v4(dst, packed[0], packed[1], packed[2], packed[3], self.one_shot)

        # The reduction mbarrier covers remote stores; no cluster barrier needed.


#: TP32 (K = 384) tactics above 48 rows. cuBLAS runs these shapes as one whole-K kernel at every row count (no K
#: split), which every single-pass tile here reproduces bit for bit; so push GEMM + finish is the bits of cuBLAS +
#: the engine two-shot all-reduce (whose 32 ranks the finish sums in the engine's two 16-rank chunks). 32 ranks, us
#: per site (graph of 24 layers, distinct weights) cuBLAS + engine AR -> push GEMM + finish kernel: M=49 11.34 ->
#: 10.11, M=64 11.89 -> 11.18, M=72 12.08 -> 12.36, M=80 12.39 -> 12.76, M=88 15.98 -> 13.47, M=96 16.67 -> 13.83,
#: M=128 17.96 -> 16.58, M=160 19.64 -> 18.88, M=190 23.74 -> 21.31, M=192 23.78 -> 22.05; most layers finish in
#: place under their attention-residual mix instead (oproj_push_attn_res_add_rmsnorm), which wins at every row count.
_TP32_WIDE = (
    (72, SplitKTactic(128, 64, 1, 3)),
    (96, SplitKTactic(128, 96, 1, 3)),
    (192, SplitKTactic(128, 64, 1, 3)),
)

#: Measured tactic per (N, K) and token count (``(max_m, tactic)`` rows, first row whose bound covers M), GB300,
#: 24-layer graph of GEMM + all-reduce with distinct weights, us per layer (cuBLAS + C++ MNNVL all-reduce -> this GEMM
#: + the Lamport finish). Past the last bound cuBLAS + the copy-kernel push stays faster.
_TACTICS = {
    # Kimi K3.1 KDA / MLA o_proj, TP8 (K = 12288 / 8): M=28 11.59 -> 9.58, M=48 13.32 -> 10.73, M=64 13.41 -> 11.19,
    # M=72 13.90 -> 12.89; M=84 13.20 vs 13.43 (loses from there). Split-K tiles (faster at M <= 16) sum the K halves
    # in another order than cuBLAS, so every row count up to 32 takes the single-split 32-row tile.
    (7168, 1536): (
        (32, SplitKTactic(64, 32, 1, 9)),
        (64, SplitKTactic(64, 64, 1, 6)),
        (72, SplitKTactic(64, 128, 1, 4)),
    ),
    # TP16 (K = 12288 / 16), 16 ranks: one token tile of the row count rounded up to 8 up to 80 rows (a second
    # 64-row tile re-streams the weight), then 128 x 64 tiles: M=1 7.79 -> 6.03, M=8 9.32 -> 7.47, M=16 9.74 ->
    # 8.09, M=24 10.41 -> 8.32, M=48 11.00 -> 9.11, M=64 11.40 -> 9.83, M=72 11.93 -> 10.69, M=80 12.26 -> 11.12,
    # M=96 15.30 -> 11.94, M=128 16.28 -> 13.02, M=160 17.88 -> 16.24, M=192 21.18 -> 18.07.
    (7168, 768): (
        (8, SplitKTactic(128, 8, 1, 6)),
        (16, SplitKTactic(64, 16, 1, 7)),
        (24, SplitKTactic(64, 24, 1, 7)),
        (32, SplitKTactic(64, 32, 1, 7)),
        (48, SplitKTactic(64, 48, 1, 6)),
        (64, SplitKTactic(64, 64, 1, 6)),
        (72, SplitKTactic(64, 72, 1, 5)),
        (80, SplitKTactic(64, 80, 1, 5)),
        (192, SplitKTactic(128, 64, 1, 3)),
    ),
    # Kimi K3.1 DFlash2 drafter (64 heads x 128, MLP 14336) at TP16 / TP32: o_proj K = 512 / 256, MLP down K = 896
    # (TP16; 448 at TP32 is not a whole number of K tiles). Same tiling rule as the TP16 o_proj; 4 ranks, us per
    # site cuBLAS + engine AR -> push GEMM + finish: K=512 M8 6.88 -> 5.13, M24 8.69 -> 7.16, M72 10.36 -> 8.67;
    # K=256 M8 5.86 -> 4.57, M72 9.50 -> 8.20; K=896 M8 7.74 -> 5.76, M24 9.55 -> 8.03, M72 11.83 -> 9.96.
    # The drafter's MLP down at TP8 (K = 1792), 8 ranks, us per sublayer with its epilogue (cuBLAS + engine AR +
    # epilogue -> push GEMM + finish + epilogue): M5 12.64 -> 12.09, M10 13.99 -> 12.46, M20 15.27 -> 12.81,
    # M30 15.62 -> 12.89.
    **{
        (7168, k): (
            (8, SplitKTactic(128, 8, 1, 7)),
            (16, SplitKTactic(64, 16, 1, 7)),
            (24, SplitKTactic(64, 24, 1, 7)),
            (32, SplitKTactic(64, 32, 1, 7)),
            (48, SplitKTactic(64, 48, 1, 6)),
            (64, SplitKTactic(64, 64, 1, 6)),
            (72, SplitKTactic(64, 72, 1, 5)),
            (80, SplitKTactic(64, 80, 1, 5)),
        )
        for k in (256, 512, 896, 1792)
    },
    # The drafter's MLP down at TP32 (K = 14336 / 32 = 448: three K tiles and a half, the half read as zeros past K):
    # the drafter rule.
    (7168, 448): (
        (8, SplitKTactic(128, 8, 1, 4)),
        (16, SplitKTactic(64, 16, 1, 4)),
        (24, SplitKTactic(64, 24, 1, 4)),
        (32, SplitKTactic(64, 32, 1, 4)),
    ),
    # TP32 (K = 12288 / 32: three K tiles), 32 ranks, us per site cuBLAS + engine AR -> push GEMM + finish:
    # M=1 8.24 -> 6.09, M=4 9.08 -> 7.33, M=8 9.43 -> 7.44, M=16 9.96 -> 7.99, M=32 10.56 -> 8.91,
    # M=48 11.36 -> 9.86. Above 48 rows: _TP32_WIDE.
    (7168, 384): (
        (2, SplitKTactic(64, 8, 1, 3)),
        (8, SplitKTactic(128, 8, 1, 3)),
        (16, SplitKTactic(64, 16, 1, 3)),
        (32, SplitKTactic(64, 32, 1, 3)),
        (48, SplitKTactic(64, 48, 1, 3)),
        *_TP32_WIDE,
    ),
}


def tactic_for(m: int, n: int, k: int) -> Optional[SplitKTactic]:
    """The tactic for ``m`` rows of an ``[N, K]`` weight, or ``None`` where the kernel is not tuned."""
    rows = _TACTICS.get((n, k))
    if rows is None or not 1 <= m <= min(MAX_M, rows[-1][0]):
        return None
    tactic = next(t for bound, t in rows if m <= bound)
    stages = min(tactic.ab_stages, _max_ab_stages_for(tactic, _SMEM_CAPACITY_BYTES))
    tactic = dataclasses.replace(tactic, ab_stages=stages)
    if stages < 1 or k % 8:
        return None
    # A K tail (zero-filled last tile) only without a K split.
    if k % _CTA_K and tactic.split_k != 1:
        return None
    if not k % _CTA_K and (k // _CTA_K) % tactic.split_k:
        return None
    return tactic


@cute.experimental.jit
def _launch(
    gemm_op: cutlass.Constexpr,
    weight: cute.Tensor,
    tokens: cute.Tensor,
    out: cute.Tensor,
    rows: Int32,
    peer_ptrs: cutlass.Int64,
    mc_base: cutlass.Int64,
    flags: cutlass.Int64,
    rank: Int32,
    hidden_dim: cutlass.Constexpr,
    stream: _cuda.CUstream,
):
    """``weight`` [N, K] and ``tokens`` [M, K] K-major; ``out`` [M, N] gives the output tiling only."""
    k = cute.size(weight, mode=[1])
    m_a = cute.make_tensor(weight.iterator, cute.make_layout((hidden_dim, k, 1), stride=(k, 1, 0)))
    m_b = cute.make_tensor(tokens.iterator, cute.make_layout((rows, k, 1), stride=(k, 1, 0)))
    m_c = cute.make_tensor(
        out.iterator, cute.make_layout((hidden_dim, rows, 1), stride=(1, hidden_dim, 0))
    )
    gemm_op(m_a, m_b, m_c, peer_ptrs, mc_base, flags, rank, stream)


@functools.cache
def _compiled(
    tactic: SplitKTactic,
    n: int,
    k: int,
    world: int,
    one_shot: bool,
    use_pdl: bool,
    device_index: int,
):
    device = torch.device("cuda", device_index)
    kernel = Bf16ArPushGemmKernel(
        tactic=tactic, use_pdl=use_pdl, hidden_dim=n, world=world, one_shot=one_shot
    )
    weight = torch.empty((n, k), dtype=torch.bfloat16, device=device)
    tokens = torch.empty((MAX_M, k), dtype=torch.bfloat16, device=device)
    out = torch.empty((MAX_M, n), dtype=torch.bfloat16, device=device)
    return cute_ext.compile(
        _launch,
        kernel,
        from_dlpack(weight, assumed_align=32),
        from_dlpack(tokens, assumed_align=32).mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1)
        ),
        from_dlpack(out, assumed_align=32).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1)),
        Int32(MAX_M),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        Int32(0),
        n,
        _cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
    )


def can_push(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Whether :func:`run_bf16_gemm_ar_push` serves ``x [M, K] @ weight[N, K]^T`` (BF16, contiguous, a tuned shape)."""
    if not (x.is_cuda and x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16):
        return False
    if x.dim() != 2 or weight.dim() != 2 or x.shape[1] != weight.shape[1]:
        return False
    if not (x.is_contiguous() and weight.is_contiguous()):
        return False
    if torch.cuda.get_device_capability(x.device) not in ((10, 0), (10, 3)):
        return False
    return tactic_for(x.shape[0], weight.shape[0], weight.shape[1]) is not None


def run_bf16_gemm_ar_push(
    x: torch.Tensor,
    weight: torch.Tensor,
    workspace: "_lam.LamportWorkspace",
    one_shot: bool,
    out: torch.Tensor,
    use_pdl: bool = True,
) -> None:
    """Push ``x @ weight^T`` (BF16 ``[M, N]``) into the Lamport slots of the next all-reduce on
    ``workspace`` (two-shot or ``one_shot`` transport); ``out`` (``[M, N]`` BF16, not written) is the
    all-reduce's output tensor, whose shape the kernel tiles. Check :func:`can_push` first."""
    m, k = x.shape
    n = weight.shape[0]
    tactic = tactic_for(m, n, k)
    if tactic is None or not 1 <= m <= MAX_M:
        raise ValueError(f"no tactic for M={m} N={n} K={k}")
    _lam._check(workspace, m, n, one_shot)
    compiled = _compiled(
        tactic, n, k, workspace.world, bool(one_shot), bool(use_pdl), x.device.index
    )
    compiled(
        from_dlpack(weight.detach(), assumed_align=32),
        from_dlpack(x, assumed_align=32).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1)),
        from_dlpack(out, assumed_align=32).mark_compact_shape_dynamic(mode=0, stride_order=(0, 1)),
        Int32(m),
        cutlass.Int64(workspace.peer_ptrs.data_ptr()),
        cutlass.Int64(workspace.mc_base),
        cutlass.Int64(workspace.flags.data_ptr()),
        Int32(workspace.rank),
        _cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
    )


__all__ = ["MAX_M", "can_push", "run_bf16_gemm_ar_push", "tactic_for"]
