# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from CUTLASS utils.gemm.sm100.epilogue_tma_store.

from typing import Any, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Constexpr, Int32
from cutlass.cute.nvgpu import cpasync
from cutlass.utils.gemm.sm100 import (
    epilogue_smem_copy_and_partition,
    epilogue_tmem_copy_and_partition,
    transform_partitioned_tensor_layout,
)

WARP_SIZE = 32


@cute.jit
def epilogue_tma_store_fp8(
    gemm_kernel: Any,
    epi_tidx: Int32,
    warp_idx: Int32,
    tma_atom_c: cute.CopyAtom,
    # Input of epilogue
    tCtAcc_base: cute.Tensor,
    # Staging of epilogue
    sC: cute.Tensor,
    # Output of epilogue
    tCgC_base: cute.Tensor,
    epi_tile: cute.Tile,
    num_tiles_executed: Int32,
    epilogue_op: Constexpr,
    mma_tile_coord_mnl: Tuple[Int32, Int32, Int32],
    acc_consumer_state: pipeline.PipelineState,
    acc_pipeline: pipeline.PipelineAsync,
    c_pipeline: pipeline.PipelineTmaStore,
) -> pipeline.PipelineState:
    # Layout transformation for tCgC_base
    # ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, TILE_M, TILE_N, TILE_K)
    # -> ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), TILE_M, TILE_N, TILE_K)
    tCgC = transform_partitioned_tensor_layout(tCgC_base)

    # Layout transformation for tCtAcc_base
    # ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, STAGE)
    # -> ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), STAGE)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)

    tiled_copy_t2r, tTR_tAcc_base, tTR_rAcc = epilogue_tmem_copy_and_partition(
        gemm_kernel, epi_tidx, tCtAcc, tCgC, epi_tile, gemm_kernel.use_2cta_instrs
    )

    tTR_rC = cute.make_rmem_tensor(tTR_rAcc.shape, gemm_kernel.c_dtype)
    tiled_copy_r2s, tRS_rC, tRS_sC = epilogue_smem_copy_and_partition(
        gemm_kernel, tiled_copy_t2r, tTR_rC, epi_tidx, sC
    )

    # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
    tCgC_epi = cute.flat_divide(tCgC, epi_tile)
    # ((ATOM_V, REST_V), EPI_M, EPI_N)
    # ((ATOM_V, REST_V), EPI_M, EPI_N, RestM, RestN, RestL)
    bSG_sC, bSG_gC_partitioned = cpasync.tma_partition(
        tma_atom_c,
        0,
        cute.make_layout(1),
        cute.group_modes(sC, 0, 2),
        cute.group_modes(tCgC_epi, 0, 2),
    )

    epilog_sync_barrier = pipeline.NamedBarrier(
        barrier_id=gemm_kernel.epilog_sync_bar_id,
        num_threads=WARP_SIZE * len(gemm_kernel.epilogue_warp_id),
    )

    #
    # Slice to per mma tile index
    #
    # ((ATOM_V, REST_V), EPI_M, EPI_N)
    bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]

    # Set tensor memory buffer for current tile
    # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
    tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

    #
    # Wait for accumulator buffer full
    #
    acc_pipeline.consumer_wait(acc_consumer_state)

    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

    #
    # Store accumulator to global memory in subtiles
    #
    subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])  # type: ignore[union-attr]
    num_prev_subtiles = num_tiles_executed * subtile_cnt
    for subtile_idx in range(subtile_cnt):
        #
        # Load accumulator from tensor memory buffer to register
        #
        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]  # type: ignore[call-overload]
        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

        #
        # Convert to C type
        #
        acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
        # Match sm100.epilogue_tma_store: convert to BF16 before applying
        # epilogue_op (including GEMM alpha), then quantize with unit output
        # scale. Keep the intermediate BF16 rounding in registers only.
        acc_vec = epilogue_op(acc_vec.to(cutlass.BFloat16)).to(cutlass.Float32)
        acc_vec = cute.where(acc_vec > 448.0, 448.0, acc_vec)
        acc_vec = cute.where(acc_vec < -448.0, -448.0, acc_vec)
        acc_vec = acc_vec.to(cutlass.Float8E4M3FN)
        tRS_rC.store(acc_vec)

        #
        # Store C to shared memory
        #
        c_buffer = (num_prev_subtiles + subtile_idx) % gemm_kernel.num_c_stage
        cute.copy(tiled_copy_r2s, tRS_rC, tRS_sC[(None, None, None, c_buffer)])
        # Fence and barrier to make sure shared memory store is visible to TMA store
        cute.arch.fence_proxy(
            "async.shared",
            space="cta",
        )
        epilog_sync_barrier.arrive_and_wait()

        #
        # TMA store C to global memory
        #
        if warp_idx == gemm_kernel.epilogue_warp_id[0]:
            cute.copy(
                tma_atom_c,
                bSG_sC[(None, c_buffer)],
                bSG_gC[(None, subtile_idx)],
            )
            # Fence and barrier to make sure shared memory store is visible to TMA store
            c_pipeline.producer_commit()
            c_pipeline.producer_acquire()
        epilog_sync_barrier.arrive_and_wait()

    epilog_sync_barrier.arrive_and_wait()

    #
    # Async arrive accumulator buffer empty
    #
    with cute.arch.elect_one():
        acc_pipeline.consumer_release(acc_consumer_state)
    acc_consumer_state.advance()
    return acc_consumer_state
