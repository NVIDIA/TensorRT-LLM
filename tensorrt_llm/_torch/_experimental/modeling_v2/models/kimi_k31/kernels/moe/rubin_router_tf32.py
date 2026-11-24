# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted from FlashInfer (Apache-2.0; upstream notice: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.):
# https://github.com/flashinfer-ai/flashinfer/blob/4b6f2ead50642001033179b1f77cb443c794a9cb/flashinfer/gemm/kernels/dense_bf16_gemm_sm100_splitk.py
# Modified by NVIDIA: TF32 FP32 router logits as 8 K-slice partials.
"""Kimi K3.1 FP32 router logits on TF32 tensor cores (SM107), as the 8 K-slice partials cuBLAS sums.

``torch.nn.functional.linear`` on FP32 ``x [M, 7168]`` and ``w [896, 7168]`` with TF32 enabled (the
container's ``TORCH_ALLOW_TF32_CUBLAS_OVERRIDE``) runs cuBLAS's ``nvjet_sm107_sss_tf32_*_splitK``
GEMM at 48 to 160 rows: 8 K slices of 896, whose partials ``splitKreduce`` adds in slice order.
:class:`RouterTf32SplitKKernel` computes each slice as one chain of ``tcgen05.mma.kind::tf32``
(K = 8 per instruction, in K order, the first one not accumulating) and leaves the sum to its
consumer: ``out[s, r, e]`` is slice ``s`` of row ``r`` and expert ``e``. When the operands are
exactly representable in TF32 (BF16 values widened to FP32, as K3.1's router input and weight
are), every partial has the bits of cuBLAS's, so ``((out[0] + out[1]) + ...) + out[7]`` has the
bits of ``F.linear`` (measured on SM107 at 48 to 160 rows). Operands with low mantissa bits are
truncated by the tensor cores here and are not handled the same way by cuBLAS.

The GEMM is swapped as in ``rubin_ull_k3_low_m_bf16_splitk``: kernel-M carries the experts
(``w`` rows, MMA M 128), kernel-N the token rows (MMA N 96). One CTA per (expert tile, row tile,
slice), ``7 x ceil(M / 96) x 8`` CTAs, no reduction in the kernel. Warp 0 loads the weight by TMA,
warp 1 the rows, warp 2 issues the MMAs into tensor memory, warps 4-7 store the partial tile.
Under PDL the weight is loaded before ``griddepcontrol.wait`` (only the rows wait for the grid
before) and the dependents launch once every weight load is issued.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack

#: K3.1's router: experts and hidden size.
NUM_EXPERTS = 896
HIDDEN = 7168
#: cuBLAS's K slices of this GEMM (``splitK`` 8) and their width.
SPLITS = 8
SLICE_K = HIDDEN // SPLITS
#: MMA tile: experts (kernel-M) x rows (kernel-N); K per pipeline stage.
TILE_E = 128
TILE_R = 96
TILE_K = 32
#: Pipeline stages of the weight and row tiles (28 KB each).
STAGES = 6

_AB_ALIGN = 1024
_MBAR = 8
_TMEM_PTR = 4
_TMEM_COLS = 256

assert NUM_EXPERTS % TILE_E == 0 and SLICE_K % TILE_K == 0


class RouterTf32SplitKKernel:
    """``SPLITS`` K-slice partials of ``x @ w.T`` on TF32 tensor cores, one CTA per tile and slice."""

    def __init__(self, use_pdl: bool):
        self.acc_dtype = cutlass.Float32
        self.cta_m, self.cta_n, self.cta_k = TILE_E, TILE_R, TILE_K
        self.num_ab_stage = STAGES
        self.use_pdl = use_pdl
        self.threads_per_cta = 256
        self.mma_tiler_mn = (TILE_E, TILE_R)
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tma_op = cute_ext.OperationTypeEnum.SM90_TMA_LOAD

    @cute.experimental.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, stream: _cuda.CUstream):
        self.kernel(a, b, c).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n),
                SPLITS,
            ),
            block=(self.threads_per_cta, 1, 1),
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor):
        """``mA`` (experts, K, 1) and ``mB`` (rows, K, 1) K-major; ``mC`` (experts, rows, slice)."""
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
        mnk_tiler = (self.cta_m, self.cta_n, self.cta_k)
        block_idx = cute.arch.block_idx()
        e_tile = block_idx[0]
        r_tile = block_idx[1]
        split = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        sA = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_ALIGN,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_ALIGN,
        )
        acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)
        bar_full = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(stages), alignment=_MBAR
        ).iterator
        bar_empty = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(stages), alignment=_MBAR
        ).iterator
        bar_mma_epilog = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=_MBAR
        ).iterator
        bar_tmem_alloc = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=_MBAR
        ).iterator
        tmem_base_ptr = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=_TMEM_PTR
        ).iterator

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    # Both loaders arrive on a stage; the MMA warp's commit empties it.
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                # The MMA warp and the 128 epilogue threads.
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // SPLITS
        k_tile_start = split * k_tile_count

        if warp_idx == 0:
            self.dma_warp(
                bar_full,
                bar_empty,
                cute.local_tile(mA, (self.cta_m, self.cta_k), (e_tile, None, 0)),
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
                cute.local_tile(mB, (self.cta_n, self.cta_k), (r_tile, None, 0)),
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
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                cute.local_tile(mC, (self.cta_m, self.cta_n), (e_tile, r_tile, split)),
                cute.arch.thread_idx()[0] - 128,
                utils.LayoutEnum.from_tensor(mC),
            )

    @cute.experimental.jit
    def dma_warp(
        self,
        bar_full,
        bar_empty,
        g_tile: cute.Tensor,
        s_tile: cute.Tensor,
        cta_v_map: cute.Layout,
        k_tile_start: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        is_a: cutlass.Constexpr,
    ):
        """TMA loads of one operand's K tiles into the stage ring; the rows wait for the grid
        before under PDL, the weight does not."""
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
                        s_tile.element_type, cute.slice_(s_tile.layout, (None, None, None, 0))
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
        # Drain: the CTA exits only after the MMAs released every stage it filled.
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
        """The slice's MMA chain in K order into one tensor-memory accumulator; the first MMA
        does not accumulate."""
        cute.arch.alloc_tmem(_TMEM_COLS, tmem_base_ptr, is_two_cta=False)
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
        cute.arch.dealloc_tmem(tmem_ptr, _TMEM_COLS, is_two_cta=False)

    @cute.experimental.jit
    def epilog_warp(
        self,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,
        epi_tid: cutlass.Int32,
        d_layout: cutlass.Constexpr,
    ):
        """The FP32 partial tile from tensor memory to global memory (rows past M not stored)."""
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr), acc_layout
        )[((None, None), 0, 0, 0)]
        epi_tile = (self.cta_m, self.cta_n)
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.cta_m, self.cta_n, self.cta_k),
                d_layout,
                self.acc_dtype,
                self.acc_dtype,
                epi_tile,
                False,
            ),
            acc_view,
        )
        gD_epi = cute.flat_divide(gD_tile, epi_tile)
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = cute_ext.allocate(self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)
        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute_ext.partition_and_copy(thr_t2r, rAcc, gD_epi[None, None, 0, 0])


@cute.experimental.jit
def _launch(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    stream: _cuda.CUstream,
):
    # torch views: a (1, E, K), b (1, R, K), c (S, R, E) -> kernel modes (E, K, 1), (R, K, 1),
    # (E, R, S).
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[1, 2, 0])),
        cute.make_tensor(c.iterator, cute.select(c.layout, mode=[2, 1, 0])),
        stream,
    )


def _dynamic(t: torch.Tensor, leading_dim: int):
    return from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=leading_dim)


@functools.cache
def _compiled(use_pdl: bool):
    kernel = RouterTf32SplitKKernel(use_pdl)
    a = torch.empty(1, TILE_E, SPLITS * TILE_K, device="cuda")
    b = torch.empty(1, 8, SPLITS * TILE_K, device="cuda")
    c = torch.empty(SPLITS, 8, TILE_E, device="cuda")
    stream = _cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute_ext.compile(_launch, kernel, _dynamic(a, 2), _dynamic(b, 2), _dynamic(c, 2), stream)


def run_router_tf32_partials(
    x: torch.Tensor, w: torch.Tensor, out: torch.Tensor, *, use_pdl: bool = True
) -> torch.Tensor:
    """Write the ``SPLITS`` slice partials of ``x @ w.T`` into ``out`` (inputs are not checked).

    Args:
        x: FP32 ``[M, 7168]`` rows, dense and 16-byte aligned.
        w: FP32 ``[896, 7168]`` router weight, dense and 16-byte aligned.
        out: FP32 ``[SPLITS, M, 896]``, dense: ``out[s]`` is K slice ``s``'s partial.
        use_pdl: Launch with programmatic dependent launch (the rows come from the kernel before).

    Returns:
        ``out``.
    """
    compiled = _compiled(use_pdl)
    stream = _cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    compiled(_dynamic(w.unsqueeze(0), 2), _dynamic(x.unsqueeze(0), 2), _dynamic(out, 2), stream)
    return out


__all__ = [
    "HIDDEN",
    "NUM_EXPERTS",
    "SLICE_K",
    "SPLITS",
    "RouterTf32SplitKKernel",
    "run_router_tf32_partials",
]
