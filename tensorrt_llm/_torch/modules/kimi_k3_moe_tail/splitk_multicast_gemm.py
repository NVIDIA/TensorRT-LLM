# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from FlashInfer (Apache-2.0; upstream notice: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.):
# https://github.com/flashinfer-ai/flashinfer/blob/4b6f2ead50642001033179b1f77cb443c794a9cb/flashinfer/gemm/kernels/dense_bf16_gemm_sm100_splitk.py
"""Low-M BF16 up-projection shard of the fused latent-MoE tail.

A tcgen05 GEMM with an in-kernel cluster split-K reduction (each cluster rank
accumulates an exact K slice in FP32, peers publish partials to rank 0 through
DSMEM). Rank 0 rounds the sum to BF16, adds this rank's BF16 shared-expert
shard in FP32, rounds again, turns -0 into +0 (the mailbox's empty marker) and
stores the tile through the multicast mapping of the symmetric mailbox, so the
store is the all-gather. The problem is swapped: kernel-M carries the shard's
output columns (the weight rows), kernel-N the tokens. The weight loads do not
wait for the producer grid (PDL); the token loads and the shared shard do.
"""

from __future__ import annotations

import dataclasses
import os

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch as _torch
from cutlass import BFloat16, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

#: Per-CTA SMEM capacity reported by CuTeDSL on SM100/SM103.
_SMEM_CAPACITY_BYTES = utils.get_smem_capacity_in_bytes("sm_100")

#: K extent of one CTA tile.
_CTA_K = 128

#: Kernel-M tiles; 64 increases CTA count for low-M decode shapes.
_SUPPORTED_MMA_M = (64, 128)

#: Kernel-N carries the tokens.
_SUPPORTED_MMA_N = (8, 16, 32, 64, 128)

#: Physical cluster-K sizes; split 1 compiles out the DSMEM path.
_SUPPORTED_SPLIT_K = (1, 2, 3, 4)

#: Largest token count this GEMM serves.
_MAX_M = 128

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

#: A/B pipeline stage bounds. One stage is useful only for a single K tile.
_MIN_AB_STAGES = 1
_DEFAULT_MIN_AB_STAGES = 2
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


def validate_tactic(
    tactic: SplitKTactic,
    m: int,
    n: int,
    k: int,
    *,
    smem_capacity: int = _SMEM_CAPACITY_BYTES,
) -> None:
    """Reject a tactic that cannot serve ``(m, n, k)``."""
    if tactic.mma_m not in _SUPPORTED_MMA_M:
        raise ValueError(f"unsupported mma_m={tactic.mma_m}")
    if tactic.mma_n not in _SUPPORTED_MMA_N:
        raise ValueError(f"unsupported mma_n={tactic.mma_n}")
    if tactic.split_k not in _SUPPORTED_SPLIT_K:
        raise ValueError(f"unsupported split_k={tactic.split_k}")
    if not _MIN_AB_STAGES <= tactic.ab_stages <= _MAX_AB_STAGES:
        raise ValueError(
            f"ab_stages must be in [{_MIN_AB_STAGES}, {_MAX_AB_STAGES}], got {tactic.ab_stages}"
        )
    if not 1 <= m <= _MAX_M:
        raise ValueError(f"this low-M policy requires 1 <= M <= {_MAX_M}, got {m}")
    if n <= 0:
        raise ValueError(f"N must be positive, got {n}")
    if k <= 0 or k % _CTA_K or (k // _CTA_K) % tactic.split_k:
        raise ValueError(
            f"K={k} with CTA_K={_CTA_K} does not divide evenly across split_k={tactic.split_k}"
        )
    smem_bytes = _smem_bytes(tactic, tactic.ab_stages)
    if smem_bytes > smem_capacity:
        raise ValueError(
            f"tactic {tactic} needs {smem_bytes} B of shared memory but only "
            f"{smem_capacity} B are available; max ab_stages is "
            f"{_max_ab_stages_for(tactic, smem_capacity)}"
        )


def autotune_tactics(
    m: int,
    n: int,
    k: int,
    *,
    smem_capacity: int = _SMEM_CAPACITY_BYTES,
) -> list[SplitKTactic]:
    """Return valid tactics in the shape-specific stage window."""
    tactics: list[SplitKTactic] = []
    for mma_m in _SUPPORTED_MMA_M:
        for mma_n in _SUPPORTED_MMA_N:
            for split_k in _SUPPORTED_SPLIT_K:
                min_stages = _MIN_AB_STAGES if k == _CTA_K else _DEFAULT_MIN_AB_STAGES
                base = SplitKTactic(mma_m, mma_n, split_k, min_stages)
                try:
                    validate_tactic(base, m, n, k, smem_capacity=smem_capacity)
                except ValueError:
                    continue
                max_stages = _max_ab_stages_for(base, smem_capacity)
                # Short K favors shallow pipelines; long K stays near the cap.
                tactics.extend(
                    dataclasses.replace(base, ab_stages=ab_stages)
                    for ab_stages in (
                        range(min_stages, min(max_stages, 6) + 1)
                        if k <= 4 * _CTA_K
                        else range(
                            min(max(5, max_stages - 2), max_stages),
                            max_stages + 1,
                        )
                    )
                )
    return tactics


def default_tactic(m: int, n: int, k: int) -> SplitKTactic:
    """Choose the default occupancy-oriented tactic."""
    if n <= 512:
        mma_m = 64
        mma_n = 16 if n == 512 and m > 24 else 8
        requested_split = 4
    else:
        mma_n = 8 if m <= 8 else 16 if m <= 16 else 32
        if n <= 3072:
            mma_m = 128 if m <= 16 else 64
            requested_split = 4 if m <= 16 else 2
        elif n < 8192:
            mma_m = 64
            requested_split = 2
        else:
            mma_m = 128 if k <= 1024 and m <= 24 else 64
            requested_split = 1

    if k <= 4 * _CTA_K:
        requested_split = 1
    split_k = next(
        split_k
        for split_k in reversed(_SUPPORTED_SPLIT_K)
        if split_k <= requested_split and (k // _CTA_K) % split_k == 0
    )
    tactic = SplitKTactic(mma_m, mma_n, split_k, _MIN_AB_STAGES)
    max_stages = _max_ab_stages_for(tactic, _SMEM_CAPACITY_BYTES)
    tactic = dataclasses.replace(
        tactic,
        ab_stages=(
            _MIN_AB_STAGES
            if k == _CTA_K
            else _DEFAULT_MIN_AB_STAGES
            if k <= 2 * _CTA_K and m > 8
            else min(max_stages, 6)
            if k <= 4 * _CTA_K
            else max_stages
        ),
    )
    validate_tactic(tactic, m, n, k)
    return tactic


__all__ = ["SplitKMulticastUpProjection", "SplitKTactic", "tactic_for_m"]


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


@dsl_user_op
def _store_global_u32x4(address: "cutlass.Int64", packed, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), *[packed[i].ir_value(loc=loc, ip=ip) for i in range(4)]],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class SplitKDenseGemmKernel:
    """Standalone BF16/FP16 GEMM with a cluster-local split-K reduction."""

    def __init__(
        self,
        *,
        tactic: SplitKTactic,
        use_pdl: bool,
        has_bias: bool,
        hidden_dim: int,
    ) -> None:
        self.hidden_dim = hidden_dim
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
        self.has_bias = has_bias

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
        bias: cute.Tensor,
        stream: _cuda.CUstream,
    ):
        # Grid-y packs output-N tile and cluster rank.
        self.kernel(a, b, c, bias).launch(
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
        mC: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L), M-major
        mBias: cute.Tensor,  # Broadcast bias; dead when has_bias=False
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

        # Host validation guarantees an equal, tail-free K partition.
        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
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
                cute.local_tile(mBias, c_tiler_mn, (bidx, n_idx, l_idx)),
                cute.arch.thread_idx()[0] - 128,
                mC.element_type,
                utils.LayoutEnum.from_tensor(mC),
                mailbox,
                bar_reduce,
                split_rank,
                staging,
                cute.size(mC, mode=[1]),
                n_idx,
                cute.size(mC, mode=[0]) - bidx * self.cta_m,
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

        if cutlass.const_expr(is_a and self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        if cutlass.const_expr(not is_a and self.has_bias):
            cute.arch.mbarrier_arrive(bar_tma_epilog)
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
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)

        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        accumulator = cute.make_tensor(tmem_ptr, acc_layout)[None, None, None, 0]
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        full_phase = cutlass.Int32(0)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % self.num_ab_stage
            cute.arch.mbarrier_wait(bar_full + stage, full_phase)
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
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
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
        gBias_tile: cute.Tensor,
        epi_tid: cutlass.Int32,
        c_dtype: cutlass.Constexpr,
        d_layout: cutlass.Constexpr,
        mailbox,
        bar_reduce,
        split_rank: cutlass.Int32,
        staging: cute.Tensor,
        rows: cutlass.Int32,
        n_idx: cutlass.Int32,
        m_valid: cutlass.Int32,
    ):
        """``m_valid``: this tile's output columns inside the shard (the last tile of a shard that is
        not a whole number of tiles, e.g. 224 columns at TP32, stores only those)."""
        # Wait until MMA publishes the TMEM base pointer.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

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

        if cutlass.const_expr(self.has_bias):
            bias_dtype = gBias_tile.element_type
            rBias = cute_ext.allocate(
                bias_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )
            rBiasAcc = cute_ext.allocate(
                self.acc_dtype,
                cute.AddressSpace.rmem,
                rmem_layout,
                alignment=32,
            )
            if split_rank == OWNER_RANK:
                cute.arch.mbarrier_wait(bar_tma_epilog, 0)
                cute_ext.partition_and_copy(
                    cute.make_tiled_copy_D(
                        cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), bias_dtype),
                        tiled_copy_t2r,
                    ).get_slice(epi_tid),
                    cute.flat_divide(gBias_tile, epi_tile)[None, None, 0, 0],
                    rBias,
                )
                rBiasAcc.store(rBias.load().to(self.acc_dtype))

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        # Make tcgen05.ld visible before TMEM release and RMEM use.
        cute.arch.fence_view_async_tmem_load()
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
                cute.arch.mbarrier_wait(bar_reduce, 0)
                for peer in cutlass.range_constexpr(self.split_k - 1):
                    for value_idx in cutlass.range_constexpr(values_per_thread):
                        rAcc[value_idx] = (
                            rAcc[value_idx]
                            + mailbox[
                                peer * values_per_peer + epi_tid * values_per_thread + value_idx
                            ]
                        )

        if split_rank == OWNER_RANK:
            # Round the GEMM to BF16, add the BF16 shared shard in FP32, round
            # again (the unfused tail's rounding points); -0 -> +0 keeps every
            # stored fragment distinct from the mailbox's empty marker.
            fused = (rAcc.load().to(c_dtype).to(self.acc_dtype) + rBiasAcc.load()).to(c_dtype)
            fused = cute.where(fused == cute.zeros_like(fused), cute.zeros_like(fused), fused)
            rD.store(fused)
            cute_ext.partition_and_copy(thr_t2r, rD, staging)
            cute.arch.barrier(barrier_id=_EPILOG_BARRIER_ID, number_of_threads=self.epilog_threads)
            chunks_per_token = self.cta_m // 8
            total_chunks = self.cta_n * chunks_per_token
            out_base = gD_tile.iterator.toint()
            for i in cutlass.range_constexpr(
                (total_chunks + self.epilog_threads - 1) // self.epilog_threads
            ):
                chunk = epi_tid + Int32(i * self.epilog_threads)
                token = chunk // Int32(chunks_per_token)
                column = (chunk - token * Int32(chunks_per_token)) * Int32(8)
                if (
                    chunk < Int32(total_chunks)
                    and n_idx * Int32(self.cta_n) + token < rows
                    and column < m_valid
                ):
                    packed = _load_shared_u32x4(
                        staging.iterator + token * Int32(self.cta_m) + column
                    )
                    _store_global_u32x4(
                        out_base
                        + (cutlass.Int64(token) * self.hidden_dim + cutlass.Int64(column)) * 2,
                        packed,
                    )

        # The reduction mbarrier covers remote stores; no cluster barrier needed.


_SUPPORTED_TORCH_DTYPES = (_torch.bfloat16, _torch.float16)


@cute.experimental.jit
def _launch(
    gemm_op: cutlass.Constexpr,
    weight: cute.Tensor,
    tokens: cute.Tensor,
    shared: cute.Tensor,
    rows: Int32,
    output_multicast: cutlass.Int64,
    hidden_dim: cutlass.Constexpr,
    shard_dim: cutlass.Constexpr,
    max_m: cutlass.Constexpr,
    stream: _cuda.CUstream,
):
    """``weight`` [shard, K] and ``tokens`` [M, K] K-major; ``shared`` the
    rank's [max_m, shard] slice (row stride ``hidden_dim``); the output is this
    rank's [M, shard] slice of the [max_m, hidden_dim] mailbox at
    ``output_multicast`` (multicast address of the slice's first element)."""
    k = cute.size(weight, mode=[1])
    m_a = cute.make_tensor(weight.iterator, cute.make_layout((shard_dim, k, 1), stride=(k, 1, 0)))
    m_b = cute.make_tensor(tokens.iterator, cute.make_layout((rows, k, 1), stride=(k, 1, 0)))
    m_c = cute.make_tensor(
        cute.make_ptr(BFloat16, output_multicast, cute.AddressSpace.gmem, assumed_align=16),
        cute.make_layout((shard_dim, rows, 1), stride=(1, hidden_dim, 0)),
    )
    m_s = cute.make_tensor(
        shared.iterator, cute.make_layout((shard_dim, max_m, 1), stride=(1, hidden_dim, 0))
    )
    gemm_op(m_a, m_b, m_c, m_s, stream)


def tactic_for_m(m: int, shard_dim: int) -> SplitKTactic:
    """Measured choice per token count (shard 896 at TP8, 448 at TP16)."""
    override = os.environ.get("KIMI_K3_FUSED_MOE_TAIL_GEMM_TACTIC")
    if override:
        mma_m, mma_n, split_k, stages = (int(v) for v in override.split(","))
        return SplitKTactic(mma_m, mma_n, split_k, stages)
    if m <= 32 and (shard_dim >= 896 or m <= 16):
        # One token tile; 14 x 4 CTAs at TP8 (all K stages resident).
        base = SplitKTactic(64, next(n for n in (8, 16, 32) if n >= m), 4, _MIN_AB_STAGES)
        return dataclasses.replace(
            base, ab_stages=min(_max_ab_stages_for(base, _SMEM_CAPACITY_BYTES), 7)
        )
    # Larger M: every K stage resident (measured: deeper K pipelines beat more
    # CTAs). TP8 (14 weight tiles): 32-token tiles, split 2 (one wave up to 128
    # tokens). TP16 (7 weight tiles): split 4 to keep enough CTAs.
    if shard_dim >= 896:
        return SplitKTactic(64, 32, 2, 7)
    return SplitKTactic(64, 16 if m <= 64 else 32, 4, 7)


_COMPILED: dict = {}


class SplitKMulticastUpProjection:
    """Compiled launchers per tactic for one (shard, latent, hidden, max_m)."""

    def __init__(self, *, latent_dim: int, hidden_dim: int, shard_dim: int, max_m: int) -> None:
        if latent_dim % _CTA_K:
            raise ValueError("latent_dim must be a multiple of 128")
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.shard_dim = shard_dim
        self.max_m = max_m

    def compile_for(self, m: int, device: _torch.device):
        tactic = tactic_for_m(m, self.shard_dim)
        if (self.latent_dim // _CTA_K) % tactic.split_k:
            raise ValueError(f"{tactic} does not split K={self.latent_dim}")
        if _smem_bytes(tactic, tactic.ab_stages) > _SMEM_CAPACITY_BYTES:
            raise ValueError(f"{tactic} exceeds shared memory")
        key = (device.index, tactic, self.latent_dim, self.hidden_dim, self.shard_dim, self.max_m)
        if key not in _COMPILED:
            kernel = SplitKDenseGemmKernel(
                tactic=tactic, use_pdl=True, has_bias=True, hidden_dim=self.hidden_dim
            )
            weight = _torch.empty(
                (self.shard_dim, self.latent_dim), dtype=_torch.bfloat16, device=device
            )
            tokens = _torch.empty(
                (self.max_m, self.latent_dim), dtype=_torch.bfloat16, device=device
            )
            shared = _torch.empty(
                (self.max_m, self.hidden_dim), dtype=_torch.bfloat16, device=device
            )
            _COMPILED[key] = cute_ext.compile(
                _launch,
                kernel,
                from_dlpack(weight, assumed_align=32),
                from_dlpack(tokens, assumed_align=32).mark_compact_shape_dynamic(
                    mode=0, stride_order=(0, 1)
                ),
                from_dlpack(shared[:, : self.shard_dim], assumed_align=32),
                Int32(self.max_m),
                cutlass.Int64(0),
                self.hidden_dim,
                self.shard_dim,
                self.max_m,
                _cuda.CUstream(_torch.cuda.current_stream(device).cuda_stream),
            )
        return _COMPILED[key]

    def __call__(self, compiled, weight, tokens, shared_shard, output_multicast: int) -> None:
        compiled(
            from_dlpack(weight.detach(), assumed_align=32),
            from_dlpack(tokens, assumed_align=32).mark_compact_shape_dynamic(
                mode=0, stride_order=(0, 1)
            ),
            from_dlpack(shared_shard, assumed_align=32),
            Int32(tokens.shape[0]),
            cutlass.Int64(output_multicast),
            _cuda.CUstream(_torch.cuda.current_stream(tokens.device).cuda_stream),
        )
