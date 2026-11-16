# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The drafter's ``gate_up_proj`` with ``silu(gate) * up`` in its epilogue, on this target's split-K kernel.

``ops/gemm/dr3_fold_splitk_gemm``'s SF on ``ops/gemm/splitk_gemm.SplitKDenseGemmKernel`` (this target's copy of
flashinfer's kernel) instead of flashinfer's own, so that it takes the copy's 64-wide token tiles (``wide64``:
the TP8 drafter's 40..64-row blocks) and its oversized shared memory (``tep8_drafter_wide_gemms__DEEP_STAGES``'
stage counts); split_k 1 only (every drafter ``gate_up`` tactic). The weight's gate / up rows are interleaved per
kernel-M tile (``dr3_fold_splitk_gemm.interleave_gate_up``: ``mma_m / 2`` gate rows, then the matching up rows),
so a tile's owner holds both halves of ``mma_m / 2`` output columns. Everything up to the BF16 rounding of the
tile is the parent's (DMA warps with the weight before ``griddepcontrol.wait``, MMA warp, TMEM accumulator); the
epilogue then stages the rounded tile widened back to FP32 (exact) in shared memory, synchronizes its 128 threads
on a named barrier and applies the built-in Triton ``silu_and_mul`` statements as inline PTX
(``dr3_fold_splitk_gemm.SILU_PTX``), storing BF16 with ``cvt.rn``: bitwise the plain GEMM followed by
``drafter_silu_and_mul`` on the de-interleaved columns. A row permutation of the weight changes no column's
K-ordered sum, so the GEMM values are the plain kernel's for the original weight.

The staging buffer (``mma_m x mma_n`` FP32) comes on top of the parent's shared memory: ``fold_tactic`` lowers
the stage count until both fit the configured capacity.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute import experimental as _cute_ext

from . import splitk_gemm as _splitk
from .dr3_fold_splitk_gemm import (
    _EPILOG_THREADS,
    STAGE_BARRIER_ID,
    _silu_and_mul_f32,
    _store_bf16_rn,
    interleave_gate_up,
)

__all__ = ["fold_tactic", "interleave_gate_up", "silu_fold_gemm", "takes"]


def _stage_bytes(tactic) -> int:
    return tactic.mma_m * tactic.mma_n * 4


def fold_tactic(tactic) -> "_splitk.SplitKTactic | None":
    """``tactic`` (a split-K tuple / ``SplitKTactic``) with its stages lowered until the parent's shared memory plus
    the staging buffer fit the split-K kernel's configured capacity, or None (split_k > 1, no stage fits)."""
    t = _splitk.SplitKTactic(*tuple(tactic)) if not isinstance(tactic, _splitk.SplitKTactic) else tactic
    if t.split_k != 1 or t.mma_m not in (64, 128):
        return None
    _splitk._resolve_smem_capacity()
    for stages in range(t.ab_stages, 1, -1):
        candidate = _splitk.SplitKTactic(t.mma_m, t.mma_n, 1, stages)
        if _splitk._smem_bytes(candidate, stages) + _stage_bytes(candidate) + 1024 <= _splitk._SMEM_CAPACITY_BYTES:
            return candidate
    return None


def takes(x: torch.Tensor, weight: torch.Tensor, tactic) -> bool:
    """Whether ``silu_fold_gemm`` takes ``x @ weight.T`` (interleaved weight) at ``tactic`` (``fold_tactic``'s)."""
    return (
        tactic is not None
        and tactic.split_k == 1
        and tactic.mma_m in (64, 128)
        and x.dim() == 2
        and weight.dim() == 2
        and x.dtype == weight.dtype == torch.bfloat16
        and x.is_cuda
        and x.device == weight.device
        and x.is_contiguous()
        and weight.is_contiguous()
        and 1 <= x.shape[0] <= tactic.mma_n
        and x.shape[1] == weight.shape[1]
        and x.shape[1] % _splitk._CTA_K == 0
        and weight.shape[0] % (2 * tactic.mma_m) == 0
        and x.data_ptr() % 32 == 0
        and weight.data_ptr() % 32 == 0
        and not torch.is_grad_enabled()
    )


class SiluFoldKernel(_splitk.SplitKDenseGemmKernel):
    """``SplitKDenseGemmKernel`` (split_k 1) whose CTAs store ``silu(gate) * up`` of their tile.

    ``c`` is the activation output (kernel-M = half the GEMM's rows).
    """

    def __init__(self, *, tactic: "_splitk.SplitKTactic", use_pdl: bool) -> None:
        if tactic.split_k != 1:
            raise ValueError(f"SiluFoldKernel: split_k 1 only, got {tactic}")
        super().__init__(tactic=tactic, use_pdl=use_pdl, has_bias=False)
        self.half_m = tactic.mma_m // 2
        need = _splitk._smem_bytes(tactic, tactic.ab_stages) + _stage_bytes(tactic) + 1024
        self.launch_smem = (
            _splitk._SMEM_CAPACITY_BYTES if need > _splitk._OPTIN_SMEM_BYTES else _splitk._OPTIN_SMEM_BYTES
        )

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        stream: _cuda.CUstream,
    ):
        # The parent's grid over the GEMM's kernel-M (twice the output's), shared-memory request and PDL flag.
        self.kernel(a, b, c).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.half_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n),
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.launch_smem),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major: the interleaved weight
        mB: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major: the activation
        mC: cute.Tensor,  # (Gemm_M / 2, Gemm_N, Gemm_L), M-major: the SwiGLU output
    ):
        """The parent's kernel body (split_k 1); the output tile and the staging buffer differ."""
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
        n_idx = block_idx[1]
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        sA = _cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_splitk._AB_BUFFER_ALIGN_BYTES,
        )
        sB = _cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_splitk._AB_BUFFER_ALIGN_BYTES,
        )

        acc_layout = _cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)

        def barriers(count):
            return _cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(count),
                alignment=_splitk._MBARRIER_BYTES,
            ).iterator

        bar_full = barriers(stages)
        bar_empty = barriers(stages)
        bar_tma_epilog = barriers(1)
        bar_mma_epilog = barriers(1)
        bar_tmem_alloc = barriers(1)
        tmem_base_ptr = _cute_ext.allocate(
            cutlass.Int32,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_splitk._TMEM_POINTER_BYTES,
        ).iterator

        sStage = _cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout((self.cta_m, self.cta_n)),
            alignment=_splitk._MAILBOX_ALIGN_BYTES,
        )

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)

        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k
        k_tile_start = cutlass.Int32(0)

        if warp_idx == 0:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(mA, (self.cta_m, self.cta_k), (bidx, None, l_idx)),
                sA,
                _cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
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
                _cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B"),
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
            self.silu_epilog_warp(
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                cute.local_tile(mC, (self.half_m, self.cta_n), (bidx, n_idx, l_idx)),
                cute.size(mC, mode=[1]) - n_idx * self.cta_n,
                sStage,
                cute.arch.thread_idx()[0] - 128,
                utils.LayoutEnum.from_tensor(mC),
            )

    @cute.experimental.jit
    def silu_epilog_warp(
        self,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gOut_tile: cute.Tensor,
        rows_left: cutlass.Int32,
        sStage: cute.Tensor,
        epi_tid: cutlass.Int32,
        d_layout: cutlass.Constexpr,
    ):
        c_dtype = cutlass.BFloat16
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
        sStage_epi = cute.flat_divide(sStage, epi_tile)

        rmem_layout = _cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, sStage_epi, epi_tid)
        rAcc = _cute_ext.allocate(self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        rD = _cute_ext.allocate(c_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        rWide = _cute_ext.allocate(self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        _cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        # The plain kernel's BF16 values, widened back (exact) and staged in the accumulator's (M, N) layout.
        rD.store(rAcc.load().to(c_dtype))
        rWide.store(rD.load().to(self.acc_dtype))
        _cute_ext.partition_and_copy(thr_t2r, rWide, sStage_epi[None, None, 0, 0])
        cute.arch.barrier(barrier_id=STAGE_BARRIER_ID, number_of_threads=_EPILOG_THREADS)

        column = epi_tid % self.half_m
        first_row = epi_tid // self.half_m
        row_step = _EPILOG_THREADS // self.half_m
        for j in cutlass.range_constexpr(self.cta_n // row_step):
            row = first_row + j * row_step
            if row < rows_left:
                out = _silu_and_mul_f32(sStage[column, row], sStage[self.half_m + column, row])
                _store_bf16_rn(
                    out,
                    (gOut_tile.iterator + cute.crd2idx((column, row), gOut_tile.layout)).toint(),
                )


@cute.experimental.jit
def _bmm_silu(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    stream: _cuda.CUstream,
):
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0])),
        stream,
    )


@functools.cache
def _compiled(dtype: torch.dtype, tactic: "_splitk.SplitKTactic", use_pdl: bool, leading_dims: tuple):
    kernel = SiluFoldKernel(tactic=tactic, use_pdl=use_pdl)
    compile_tensors = _splitk._make_compile_repr_tensors(dtype, False, *leading_dims)
    stream = _cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return _cute_ext.compile(_bmm_silu, kernel, *compile_tensors[:3], stream)


def silu_fold_gemm(
    x: torch.Tensor, weight: torch.Tensor, tactic: "_splitk.SplitKTactic", enable_pdl: bool
) -> torch.Tensor:
    """``silu(x @ gate.T) * (x @ up.T)`` -> ``[M, N / 2]`` BF16 for an ``interleave_gate_up(.., tactic.mma_m)``
    weight, at ``tactic`` (a ``fold_tactic`` result). The first call per tactic, PDL flag and layout compiles
    (CuTe-DSL JIT) and must not be inside a CUDA-graph capture.

    Raises:
        ValueError: on operands or a tactic ``takes`` / ``validate_tactic`` refuse.
    """
    if not takes(x, weight, tactic):
        raise ValueError(
            f"silu_fold_gemm: unsupported operands x {tuple(x.shape)} {x.dtype}, weight {tuple(weight.shape)} "
            f"{weight.dtype}, tactic {tactic}"
        )
    m, k = x.shape
    n = weight.shape[0]
    _splitk.validate_tactic(tactic, m, n, k)
    out = torch.empty((m, n // 2), dtype=x.dtype, device=x.device)
    # run_splitk_dense's operand swap (``_to_cute_swap``).
    a_swap = weight.detach().t().unsqueeze(0).transpose(-2, -1)
    b_swap = x.detach().unsqueeze(0).transpose(-2, -1)
    c_swap = out.unsqueeze(0).transpose(-2, -1)
    leading_dims = tuple(_splitk._detect_leading_dim(t) for t in (a_swap, b_swap, c_swap))
    cute_tensors = tuple(
        _splitk._from_dlpack_dynamic(t, d) for t, d in zip((a_swap, b_swap, c_swap), leading_dims, strict=True)
    )
    compiled = _compiled(x.dtype, tactic, bool(enable_pdl), leading_dims)
    compiled(*cute_tensors, _cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream))
    return out
