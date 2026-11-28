# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Builds on FlashInfer's cluster split-K CuTe DSL GEMM (Apache-2.0,
# ``flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk``): the kernel class is subclassed (its DMA, MMA and
# epilogue warps are reused); only the K partition of the cluster ranks differs.
"""Low-M BF16 GEMM with a cluster split-K over front-heavy K chunks (cuBLAS's partition).

FlashInfer's split-K kernel gives each of the ``split_k`` cluster ranks an equal share of the 128-wide K tiles, so
``split_k`` must divide them. This variant also takes the remainder: the first ``K tiles % split_k`` ranks take one
K tile more, which is how cuBLAS chunks a split-K GEMM. Each rank accumulates its chunk in order in FP32; rank 0 adds
the peers' partials in rank order and rounds once, as cuBLAS's split-K reduction does. So for a split count and
partition cuBLAS uses, the output is cuBLAS's bit for bit (e.g. the TEP16 KDA [q|k|v|g] 3072 x 7168 GEMM at 1-32 rows:
3 chunks of 19 / 19 / 18 K tiles).

FlashInfer serves its split-K kernel up to 32 tokens (one token tile of 8, 16 or 32). The kernel itself takes any
token tile (kernel-N) whose FP32 tile the epilogue holds in registers and the rank-0 mailbox holds in shared memory, and
any token count over several token tiles; this module also serves those wide configurations (:data:`WIDE_MMA_N_MAX`),
e.g. the TEP16 KDA gate pair [f_a | b] (136 x 7168) at 65-128 rows, where cuBLAS splits K in 4 even chunks.
"""

from __future__ import annotations

import functools
from typing import Tuple

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass.cute import experimental as cute_ext
from flashinfer.gemm.kernels import dense_bf16_gemm_sm100_splitk as _fs


class ChunkedSplitKGemmKernel(_fs.SplitKDenseGemmKernel):
    """FlashInfer's cluster split-K GEMM with front-heavy K chunks."""

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
            alignment=_fs._AB_BUFFER_ALIGN_BYTES,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_fs._AB_BUFFER_ALIGN_BYTES,
        )

        acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)
        c_tiler_mn = (self.cta_m, self.cta_n)

        bar_full = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_fs._MBARRIER_BYTES,
        ).iterator
        bar_empty = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_fs._MBARRIER_BYTES,
        ).iterator
        bar_tma_epilog = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_fs._MBARRIER_BYTES,
        ).iterator
        bar_mma_epilog = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_fs._MBARRIER_BYTES,
        ).iterator
        bar_tmem_alloc = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_fs._MBARRIER_BYTES,
        ).iterator
        tmem_base_ptr = cute_ext.allocate(
            cutlass.Int32,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_fs._TMEM_POINTER_BYTES,
        ).iterator

        mailbox = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(self.mailbox_elements),
            alignment=_fs._MAILBOX_ALIGN_BYTES,
        )
        bar_reduce = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_fs._MBARRIER_BYTES,
        ).iterator

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
                # Owner arrival plus peer transaction-byte credits.
                cute.arch.mbarrier_init(bar_reduce, 1)

        cute.arch.mbarrier_init_fence()
        # Publish peer barriers before cross-CTA stores.
        cute.arch.cluster_arrive_relaxed()

        # Front-heavy K chunks: the first (K tiles % split) ranks take one 128-wide K tile more.
        k_tiles = cute.size(mA, mode=[1]) // self.cta_k
        base = k_tiles // self.split_k
        extra = k_tiles - base * self.split_k
        k_tile_count = base
        k_tile_start = split_rank * base + extra
        if split_rank < extra:
            k_tile_count = base + 1
            k_tile_start = split_rank * base + split_rank

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
            )


#: (mma_m, mma_n, split_k, ab_stages) of the kernels this module serves.
SplitKConfig = Tuple[int, int, int, int]


@functools.cache
def _compiled(config: SplitKConfig, pdl: bool, leading_dims: Tuple[int, int, int], device: int):
    """The kernel compiled for ``config`` on ``device`` (one JIT per key; before CUDA-graph capture)."""
    kernel = ChunkedSplitKGemmKernel(tactic=_fs.SplitKTactic(*config), use_pdl=pdl, has_bias=False)
    compile_tensors = _fs._make_compile_repr_tensors(torch.bfloat16, False, *leading_dims)
    stream = _cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute_ext.compile(_fs._bmm_no_bias, kernel, *compile_tensors[:3], stream)


#: Largest wide token tile (kernel-N): its FP32 tile is held in the epilogue's registers and, from every peer rank, in
#: rank 0's shared-memory mailbox next to the A / B stages.
WIDE_MMA_N_MAX = 128


def validate_config(config: SplitKConfig, m: int, k: int) -> None:
    """Raise ValueError when ``config`` cannot serve ``[m, k]`` rows: FlashInfer's tiles up to its 32 rows (any K tile
    count), or a wide token tile (a multiple of 8 up to :data:`WIDE_MMA_N_MAX` on 64-row weight tiles, of 16 on 128-row
    ones) at any row count, with its stages and mailbox inside shared memory."""
    mma_m, mma_n, split_k, stages = config
    narrow = mma_n in _fs._SUPPORTED_MMA_N and m <= _fs._MAX_M
    wide = mma_n % (8 if mma_m == 64 else 16) == 0 and 8 <= mma_n <= WIDE_MMA_N_MAX
    if (
        mma_m not in _fs._SUPPORTED_MMA_M
        or not (narrow or wide)
        or split_k not in _fs._SUPPORTED_SPLIT_K
        or m < 1
        or k % _fs._CTA_K
        or k // _fs._CTA_K < split_k
        or not _fs._MIN_AB_STAGES <= stages <= _fs._MAX_AB_STAGES
        or _fs._smem_bytes(_fs.SplitKTactic(*config), stages) > _fs._SMEM_CAPACITY_BYTES
    ):
        raise ValueError(f"no chunked split-K config {config} for M={m}, K={k}")


def run_chunked_splitk(
    a: torch.Tensor, w_t: torch.Tensor, out: torch.Tensor, pdl: bool, config: SplitKConfig
) -> torch.Tensor:
    """``out = a @ w_t`` (``a [M, K]`` row-major, ``w_t [K, N]`` a row-major weight's transpose, BF16)."""
    validate_config(config, a.shape[0], a.shape[1])
    cute_tensors = _fs._to_cute_swap(a, w_t, out, None)
    compiled = _compiled(tuple(config), bool(pdl), cute_tensors[4], a.device.index)
    compiled(*cute_tensors[:3], _cuda.CUstream(torch.cuda.current_stream(a.device).cuda_stream))
    return out
