# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""One low-M BF16 GEMM over row-packed weights that stores each row block as its own output tensor.

Adapted from flashinfer-ai/flashinfer v0.6.18, flashinfer/gemm/kernels/tgv_gemm_cute_ext.py (Apache-2.0).

``x [M, K] @ cat([W_0, W_1, W_2]).T`` -- several projections of the same input packed into one weight --
written as ``out_s = x @ W_s.T`` in separate contiguous tensors, each in its own dtype (BF16, or FP32 for a
router's logits). One launch streams all the weights instead of one kernel per projection, and no consumer
reads a strided column slice.

The kernel is FlashInfer's TGV GEMM (``flashinfer.gemm.kernels.tgv_gemm_cute_ext.TgvGemmCuteExtKernel``: one
CTA per (cta_m weight rows, cta_n tokens) tile over the full K, weights loaded before ``griddepcontrol.wait``)
with one change: the epilogue of a tile stores into the output of the row block the tile belongs to (row
blocks are multiples of cta_m rows). The FP32 accumulation is the TGV kernel's, so a BF16 block is bitwise
TGV's output for the packed weight, and an FP32 block is that accumulator before the cast.
"""

from __future__ import annotations

import functools
from typing import Optional, Sequence

import torch

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    import cutlass.utils.blackwell_helpers as sm100_utils
    from cutlass.cute import experimental as cute_ext
    from cutlass.cute.runtime import from_dlpack, make_fake_stream
    from flashinfer.gemm.kernels.tgv_gemm_cute_ext import (
        TgvGemmCuteExtKernel,
        WorkTileInfo,
        _resolve_tactic,
    )

    IS_AVAILABLE = True
except ImportError:  # pragma: no cover - FlashInfer / CuTe DSL without the TGV CuTe kernel
    TgvGemmCuteExtKernel = object
    IS_AVAILABLE = False

#: Largest number of row blocks one launch stores.
MAX_OUTPUTS = 3

_TORCH_TO_CUTLASS = {}
if IS_AVAILABLE:
    _TORCH_TO_CUTLASS = {torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32}


class _TgvMultiOutKernel(TgvGemmCuteExtKernel):
    """TGV with the epilogue store routed to one of up to three output tensors by the tile's weight-row block."""

    def __init__(self, block_tiles: tuple, grid_tiles: int, **tgv_kwargs):
        super().__init__(**tgv_kwargs)
        # Cumulative tile boundaries of the row blocks, e.g. (14, 70, 82): tiles [0, 14) -> out 0, ...
        self.block_ends = tuple(block_tiles)
        # Weight-row tiles of the packed weight (the launch rounds them up to a whole cluster).
        self.grid_tiles = grid_tiles

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L), K-major: the packed weight
        b: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L), K-major: the activations
        c0: cute.Tensor,  # (rows_0, Gemm_N, Gemm_L), M-major
        c1: cute.Tensor,
        c2: cute.Tensor,
        stream: cuda.CUstream,
    ):
        grid = cute.round_up(
            (
                self.grid_tiles,
                cute.ceil_div(b.layout.shape[0], self.cta_n),
                b.layout.shape[2],
            ),
            self.cluster_shape,
        )
        self.kernel(a, b, c0, c1, c2).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream,
            use_pdl=self.pdl_launch,
        )

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, mC0: cute.Tensor, mC1: cute.Tensor, mC2: cute.Tensor
    ):
        """TgvGemmCuteExtKernel.kernel (no bias) with the per-block output selection in the epilogue warps."""
        DMA_Stage = self.num_ab_stage
        a_major = utils.LayoutEnum.from_tensor(mA).mma_major_mode()
        b_major = utils.LayoutEnum.from_tensor(mB).mma_major_mode()
        ab_dtype = mA.element_type
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, a_major, b_major, self.acc_dtype, self.cta_group, self.mma_tiler_mn
        )
        num_mma_ctas = cute.size(tiled_mma.thr_id.shape)
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        mma_inst_tile_k = self.cta_k // mma_inst_shape_k
        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        a_tiler_mk = (self.cta_m, self.cta_k)
        b_tiler_nk = (self.cta_n // num_mma_ctas, self.cta_k)
        c_tiler_mn = (self.cta_m, self.cta_n)

        bidx, bidy, bidz = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        leader_rank = cutlass.Int32(0)
        if cutlass.const_expr(self.use_2cta):
            cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            is_leader = cta_rank_in_cluster == 0
        else:
            cta_rank_in_cluster = leader_rank
            is_leader = cutlass.Boolean(True)

        sA = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, DMA_Stage),
            alignment=1024,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, DMA_Stage),
            alignment=1024,
        )
        acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)

        def _bars(n):
            return cute_ext.allocate(
                cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(n), alignment=8
            ).iterator

        bar_full = _bars(DMA_Stage)
        bar_empty = _bars(DMA_Stage)
        bar_tma_epilog = _bars(1)
        bar_mma_epilog = _bars(1)
        bar_tmem_alloc = _bars(1)
        tmem_base_ptr = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(bar_full + i, 2 * num_mma_ctas)
                for i in range(DMA_Stage):
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 32 + 128)
        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.use_2cta):
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        work = WorkTileInfo(
            M_idx=bidx,
            N_idx=bidy,
            L_idx=bidz,
            K_idx_start=cutlass.Int32(0),
            K_idx_end=cute.ceil_div(cute.size(mA, mode=[1]), self.cta_k),
        )
        k_tile_count = work.K_idx_end - work.K_idx_start
        a_cta_v_map = cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A")
        b_cta_v_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
        gA_tile = cute.local_tile(mA, a_tiler_mk, (work.M_idx, None, work.L_idx))
        if cutlass.const_expr(self.use_2cta):
            gB_n_idx = work.N_idx * num_mma_ctas + cta_rank_in_cluster
        else:
            gB_n_idx = work.N_idx
        gB_tile = cute.local_tile(mB, b_tiler_nk, (gB_n_idx, None, work.L_idx))
        if cutlass.const_expr(self.use_2cta):
            cute.arch.cluster_wait()

        if warp_idx == 0:
            self.dma_a_warp(
                bar_full, bar_empty, leader_rank, gA_tile, sA, a_cta_v_map, k_tile_count
            )
        elif warp_idx == 1:
            self.dma_b_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                leader_rank,
                gB_tile,
                sB,
                b_cta_v_map,
                k_tile_count,
            )
        elif warp_idx == 2:
            self.mma_warp(
                is_leader,
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                mma_inst_tile_k,
                k_tile_count,
            )
        elif warp_idx >= 4:
            epi_tid = tidx - 128
            e0, e1 = self.block_ends[0], self.block_ends[1]
            barriers = (bar_tma_epilog, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr)
            if work.M_idx < e0:
                gD = cute.local_tile(mC0, c_tiler_mn, (work.M_idx, work.N_idx, work.L_idx))
                self._store_block(mC0, gD, *barriers, acc_layout, epi_tid)
            elif work.M_idx < e1:
                gD = cute.local_tile(mC1, c_tiler_mn, (work.M_idx - e0, work.N_idx, work.L_idx))
                self._store_block(mC1, gD, *barriers, acc_layout, epi_tid)
            else:
                gD = cute.local_tile(mC2, c_tiler_mn, (work.M_idx - e1, work.N_idx, work.L_idx))
                self._store_block(mC2, gD, *barriers, acc_layout, epi_tid)

    @cute.experimental.jit
    def _store_block(
        self,
        mC: cute.Tensor,
        gD_tile: cute.Tensor,
        bar_tma_epilog,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        epi_tid: cutlass.Int32,
    ):
        # has_bias is False: the bias tile is never read (the C tile stands in for it).
        self.epilog_warp(
            bar_tma_epilog,
            bar_mma_epilog,
            bar_tmem_alloc,
            tmem_base_ptr,
            acc_layout,
            gD_tile,
            gD_tile,
            epi_tid,
            mC.element_type,
            utils.LayoutEnum.from_tensor(mC),
        )


if IS_AVAILABLE:

    @cute.experimental.jit
    def _bmm_multi(
        gemm_op: cutlass.Constexpr,
        a: cute.Tensor,  # (L, rows, K)
        b: cute.Tensor,  # (L, K, M)
        c0: cute.Tensor,  # (L, rows_s, M)
        c1: cute.Tensor,
        c2: cute.Tensor,
        stream: cuda.CUstream,
    ):
        def mn(t):
            return cute.make_tensor(t.iterator, cute.select(t.layout, mode=[1, 2, 0]))

        gemm_op(
            mn(a),
            cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
            mn(c0),
            mn(c1),
            mn(c2),
            stream,
        )


def _cute(t: torch.Tensor, leading_dim: int):
    return from_dlpack(t, assumed_align=32).mark_layout_dynamic(leading_dim=leading_dim)


@functools.cache
def _compiled(tactic: int, use_pdl: bool, block_tiles: tuple, grid_tiles: int, out_dtypes: tuple):
    cta_m, cta_n, stages, use_2cta = _resolve_tactic(tactic)
    gemm = _TgvMultiOutKernel(
        block_tiles,
        grid_tiles,
        acc_dtype=cutlass.Float32,
        cta_m=cta_m,
        cta_n=cta_n,
        num_ab_stage=stages,
        use_2cta=use_2cta,
        use_pdl=use_pdl,
        has_bias=False,
    )
    m, rows, k = 8, 64, 128
    a = _cute(torch.empty(1, rows, k, dtype=torch.bfloat16, device="cuda"), 2)
    b = _cute(torch.empty(1, m, k, dtype=torch.bfloat16, device="cuda").transpose(1, 2), 1)
    cs = [
        _cute(torch.empty(1, m, rows, dtype=dt, device="cuda").transpose(1, 2), 1)
        for dt in out_dtypes
    ]
    return cute_ext.compile(_bmm_multi, gemm, a, b, *cs, make_fake_stream())


def tactic_cta_m(tactic: int) -> int:
    """Weight rows per CTA tile of a TGV tactic (each output block must be a multiple of it)."""
    return _resolve_tactic(tactic)[0]


def can_use(
    x: torch.Tensor, weight: torch.Tensor, outs: Sequence[torch.Tensor], tactic: int
) -> bool:
    """Whether :func:`tgv_multi_out_mm` serves these operands."""
    if not IS_AVAILABLE or not 1 <= len(outs) <= MAX_OUTPUTS:
        return False
    cta_m = tactic_cta_m(tactic)
    m, k = x.shape if x.dim() == 2 else (0, 0)
    return (
        m >= 1
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_cuda
        and x.is_contiguous()
        and weight.is_contiguous()
        and weight.dim() == 2
        and weight.shape[1] == k
        and k % 128 == 0
        and x.data_ptr() % 32 == 0
        and weight.data_ptr() % 32 == 0
        and sum(o.shape[-1] for o in outs) == weight.shape[0]
        and all(
            o.dim() == 2
            and o.shape[0] == m
            and o.shape[1] % cta_m == 0
            and o.is_contiguous()
            and o.dtype in _TORCH_TO_CUTLASS
            and o.data_ptr() % 32 == 0
            and o.device == x.device
            for o in outs
        )
    )


def tgv_multi_out_mm(
    x: torch.Tensor,
    weight: torch.Tensor,
    outs: Sequence[torch.Tensor],
    tactic: int,
    use_pdl: bool,
) -> None:
    """``outs[s] = x @ weight[r_s : r_s + outs[s].shape[1]].T`` for consecutive row blocks of ``weight``, in one
    TGV launch (tactic id as in FlashInfer's ``run_tgv_cute_ext``). Check :func:`can_use` first."""
    outs = list(outs)
    cta_m = tactic_cta_m(tactic)
    ends, acc = [], 0
    for o in outs:
        acc += o.shape[1] // cta_m
        ends.append(acc)
    if len(outs) < MAX_OUTPUTS:
        # Unused output slots take a one-tile scratch block: a 2-CTA launch rounds an odd tile count up by one
        # CTA, whose (zero, out-of-weight) tile then lands there instead of in a real output.
        scratch = torch.empty((x.shape[0], cta_m), dtype=torch.bfloat16, device=x.device)
        while len(outs) < MAX_OUTPUTS:
            outs.append(scratch)
            ends.append(acc + 1)
    compiled = _compiled(int(tactic), bool(use_pdl), tuple(ends), acc, tuple(o.dtype for o in outs))
    a = _cute(weight.unsqueeze(0), 2)
    b = _cute(x.unsqueeze(0).transpose(1, 2), 1)
    cs = [_cute(o.unsqueeze(0).transpose(1, 2), 1) for o in outs]
    compiled(a, b, *cs, cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream))


def multi_out_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    out_features: Sequence[int],
    out_dtypes: Sequence[torch.dtype],
    tactic: int,
    use_pdl: bool,
) -> Optional[list]:
    """Allocate the outputs and run :func:`tgv_multi_out_mm`; ``None`` if the operands are not served."""
    outs = [
        torch.empty((x.shape[0], n), dtype=dt, device=x.device)
        for n, dt in zip(out_features, out_dtypes)
    ]
    if not can_use(x, weight, outs, tactic):
        return None
    tgv_multi_out_mm(x, weight, outs, tactic, use_pdl)
    return outs


__all__ = [
    "IS_AVAILABLE",
    "MAX_OUTPUTS",
    "can_use",
    "multi_out_linear",
    "tactic_cta_m",
    "tgv_multi_out_mm",
]
