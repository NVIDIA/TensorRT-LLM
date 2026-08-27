# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# Adapted from the CUTLASS CuTe DSL `cutlass.utils.gemm.sm100` helpers, with a
# persistent-loop `epilogue_tma_store` for the SM107 dense GEMM kernels.

from typing import Optional, Tuple, Union

import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.nvgpu import cpasync, tcgen05

try:
    from cutlass.cute.nvgpu.common import CacheEvictionPriority
except ImportError:
    CacheEvictionPriority = None
from cutlass.cutlass_dsl import Boolean, Constexpr, Int32, const_expr
from cutlass.utils.blackwell_helpers import get_smem_store_op, get_tmem_load_op
from cutlass.utils.dynamic_persistent_tile_scheduler import ClcDynamicPersistentTileScheduler
from cutlass.utils.static_persistent_tile_scheduler import StaticPersistentTileScheduler

_NO_ALLOCATE_KWARGS = (
    {"l1c_evict_priority": CacheEvictionPriority.NO_ALLOCATE}
    if CacheEvictionPriority is not None
    else {}
)

__all__ = [
    "epilogue_tma_store",
    "epilogue",
]


def transform_partitioned_tensor_layout(tensor: cute.Tensor) -> cute.Tensor:
    """
    Transform MMA layout from ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, ...rest)
    to ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), ...rest).

    This groups MMA_ATOM_M with MMA_M and MMA_ATOM_N with MMA_N.

    :param tensor: Input tensor with layout ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, ...rest)
    :type tensor: cute.Tensor
    :return: Transformed tensor with layout ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), ...rest)
    :rtype: cute.Tensor
    """
    layout = tensor.layout
    shape = layout.shape
    stride = layout.stride

    # Build new shape: ((shape[0][0], shape[1]), (shape[0][1], shape[2]), ...rest)
    new_shape = ((shape[0][0], shape[1]), (shape[0][1], shape[2]), *shape[3:])

    # Build new stride: ((stride[0][0], stride[1]), (stride[0][1], stride[2]), ...rest)
    new_stride = ((stride[0][0], stride[1]), (stride[0][1], stride[2]), *stride[3:])

    new_layout = cute.make_layout(shape=new_shape, stride=new_stride)
    return cute.make_tensor(tensor.iterator, new_layout)


def epilogue_tmem_copy_and_partition(
    gemm_kernel,
    tidx: Int32,
    tAcc: cute.Tensor,
    tCgC: cute.Tensor,
    epi_tile: cute.Tile,
    use_2cta_instrs: Union[Boolean, bool],
) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
    """
    Make tiledCopy for tensor memory load, then use it to partition tensor memory (source) and register
    array (destination).

    :param gemm_kernel: The kernel instance
    :type gemm_kernel: Any
    :param tidx: The thread index in epilogue warp groups
    :type tidx: Int32
    :param tAcc: The accumulator tensor to be copied and partitioned
    :type tAcc: cute.Tensor
    :param tCgC: The global tensor C to be copied and partitioned
    :type tCgC: cute.Tensor
    :param epi_tile: The epilogue tiler
    :type epi_tile: cute.Tile
    :param use_2cta_instrs: Whether use_2cta_instrs is enabled
    :type use_2cta_instrs: Union[Boolean, bool]

    :return: A tuple containing (tiled_copy_t2r, tTR_tAcc, tTR_rAcc) where:
        - tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
        - tTR_tAcc: The partitioned accumulator tensor
        - tTR_rAcc: The accumulated tensor in register used to hold t2r results
    :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
    """
    # Make tiledCopy for tensor memory load
    copy_atom_t2r = get_tmem_load_op(
        gemm_kernel.cta_tile_shape_mnk,
        gemm_kernel.c_layout,
        gemm_kernel.c_dtype,
        gemm_kernel.acc_dtype,
        epi_tile,
        use_2cta_instrs,
    )
    # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
    tAcc_epi = cute.flat_divide(
        tAcc,
        epi_tile,
    )
    # (EPI_TILE_M, EPI_TILE_N)
    tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])

    thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
    # (T2R, T2R_M, T2R_N, EPI_M, EPI_M, STAGE)
    tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

    # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, RestM, RestN, RestL)
    tCgC_epi = cute.flat_divide(tCgC, epi_tile)
    # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
    tTR_gC = thr_copy_t2r.partition_D(tCgC_epi)
    # (T2R, T2R_M, T2R_N)
    tTR_rAcc = cute.make_rmem_tensor(
        tTR_gC[(None, None, None, 0, 0, 0, 0, 0)].shape, gemm_kernel.acc_dtype
    )
    return tiled_copy_t2r, tTR_tAcc, tTR_rAcc


def epilogue_smem_copy_and_partition(
    gemm_kernel,
    tiled_copy_t2r: cute.TiledCopy,
    tTR_rC: cute.Tensor,
    tidx: Int32,
    sC: cute.Tensor,
) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
    """
    Make tiledCopy for shared memory store, then use it to partition register array (source) and
    shared memory (destination).

    :param tiled_copy_t2r: The tiled copy operation for tmem to register copy(t2r)
    :type tiled_copy_t2r: cute.TiledCopy
    :param tTR_rC: The partitioned accumulator tensor
    :type tTR_rC: cute.Tensor
    :param tidx: The thread index in epilogue warp groups
    :type tidx: Int32
    :param sC: The shared memory tensor to be copied and partitioned
    :type sC: cute.Tensor

    :return: A tuple containing (tiled_copy_r2s, tRS_rC, tRS_sC) where:
        - tiled_copy_r2s: The tiled copy operation for register to smem copy(r2s)
        - tRS_rC: The partitioned tensor C (register source)
        - tRS_sC: The partitioned tensor C (smem destination)
    :rtype: Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]
    """
    copy_atom_r2s = get_smem_store_op(
        gemm_kernel.c_layout, gemm_kernel.c_dtype, gemm_kernel.acc_dtype, tiled_copy_t2r
    )
    tiled_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, tiled_copy_t2r)
    # (R2S, R2S_M, R2S_N, PIPE_D)
    thr_copy_r2s = tiled_copy_r2s.get_slice(tidx)
    tRS_sC = thr_copy_r2s.partition_D(sC)
    # (R2S, R2S_M, R2S_N)
    tRS_rC = tiled_copy_r2s.retile(tTR_rC)
    return tiled_copy_r2s, tRS_rC, tRS_sC


@cute.jit
def epilogue_tma_store(
    gemm_kernel,
    epi_tidx: Int32,
    warp_idx: Int32,
    acc_pipeline: pipeline.PipelineAsync,
    tiled_mma: cute.TiledMma,
    tma_atom_c: cute.CopyAtom,
    # Input of epilogue
    tCtAcc_base: cute.Tensor,
    # Staging of epilogue
    sC: cute.Tensor,
    # Output of epilogue
    tCgC_base: cute.Tensor,
    epi_tile: cute.Tile,
    tile_sched: Union[StaticPersistentTileScheduler, ClcDynamicPersistentTileScheduler],
    epilogue_op: Constexpr,
    clc_pipeline: Union[pipeline.PipelineClcFetchAsync, None] = None,
    clc_consumer_state: Union[pipeline.PipelineState, None] = None,
    a_scale: Optional[cute.Tensor] = None,
    b_scale: Optional[cute.Tensor] = None,
) -> None:
    """Persistent TMA-store epilogue: drains every tile the scheduler hands out,
    staging TMEM accumulators through shared memory to global memory.
    """
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

    acc_consumer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, gemm_kernel.num_acc_stage
    )

    # Threads/warps participating in tma store pipeline
    c_producer_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        32 * len(gemm_kernel.epilogue_warp_id),
    )
    c_pipeline = pipeline.PipelineTmaStore.create(
        num_stages=gemm_kernel.num_c_stage, producer_group=c_producer_group
    )

    epilog_sync_barrier = pipeline.NamedBarrier(
        barrier_id=gemm_kernel.epilog_sync_bar_id,
        num_threads=32 * len(gemm_kernel.epilogue_warp_id),
    )

    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        # Get tile coord from tile scheduler
        cur_tile_coord = work_tile.tile_idx
        mma_tile_coord_mnl = (
            cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
            cur_tile_coord[1],
            cur_tile_coord[2],
        )

        #
        # Slice to per mma tile index
        #
        # ((ATOM_V, REST_V), EPI_M, EPI_N)
        bSG_gC = bSG_gC_partitioned[(None, None, None, *mma_tile_coord_mnl)]

        # Set tensor memory buffer for current tile
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
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
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        num_prev_subtiles = tile_sched.num_tiles_executed * subtile_cnt
        for subtile_idx in range(subtile_cnt):
            #
            # Load accumulator from tensor memory buffer to register
            #
            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

            #
            # Apply scale and convert to C type
            #
            acc_vec = tiled_copy_r2s.retile(tTR_rAcc).load()
            if a_scale is not None and b_scale is not None:
                acc_vec = (a_scale[0] * b_scale[0]) * acc_vec
            acc_vec = epilogue_op(acc_vec.to(gemm_kernel.c_dtype))
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

        #
        # Advance to next tile
        #
        # Check if tile_sched is StaticPersistentTileScheduler or any subclass inheriting from it
        if const_expr(isinstance(tile_sched, StaticPersistentTileScheduler)):
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
        elif const_expr(isinstance(tile_sched, ClcDynamicPersistentTileScheduler)):
            assert clc_pipeline is not None and clc_consumer_state is not None, (
                "a CLC scheduler requires clc_pipeline and clc_consumer_state"
            )
            clc_pipeline.consumer_wait(clc_consumer_state)
            work_tile = tile_sched.get_current_work()
            clc_pipeline.consumer_release(clc_consumer_state)
            clc_consumer_state.advance()
        else:
            raise TypeError(f"unsupported tile scheduler type: {type(tile_sched)}")

    # Wait for C store complete
    c_pipeline.producer_tail()


@cute.jit
def epilogue(
    gemm_kernel,
    epi_tidx: Int32,
    acc_pipeline: pipeline.PipelineAsync,
    tiled_mma: cute.TiledMma,
    tCtAcc_base: cute.Tensor,
    tCgC_base: cute.Tensor,
    epi_tile: cute.Tile,
    tile_sched: Union[StaticPersistentTileScheduler, ClcDynamicPersistentTileScheduler],
    epilogue_op: Constexpr,
    tmem_dealloc_barrier: pipeline.NamedBarrier,
    tCcC_base: cute.Tensor = None,
    mC_mnl: cute.Tensor = None,
    clc_pipeline: Union[pipeline.PipelineClcFetchAsync, None] = None,
    clc_consumer_state: Union[pipeline.PipelineState, None] = None,
    a_scale: Optional[cute.Tensor] = None,
    b_scale: Optional[cute.Tensor] = None,
) -> None:
    """
    Epilogue function that stores accumulator results directly to global memory.
    Used when TMA store is not enabled.

    :param gemm_kernel: The kernel instance
    :type gemm_kernel: Any
    :param epi_tidx: Thread index in epilogue warp groups
    :type epi_tidx: Int32
    :param acc_pipeline: Accumulator pipeline for async operations
    :type acc_pipeline: pipeline.PipelineAsync
    :param tiled_mma: The tiled MMA configuration
    :type tiled_mma: cute.TiledMma
    :param tCtAcc_base: Base accumulator tensor in tensor memory
    :type tCtAcc_base: cute.Tensor
    :param tCgC_base: The global memory tensor C to be copied and partitioned
    :type tCgC_base: cute.Tensor
    :param epi_tile: Epilogue tile configuration
    :type epi_tile: cute.Tile
    :param tile_sched: Tile scheduler for persistent scheduling
    :type tile_sched: Union[StaticPersistentTileScheduler, ClcDynamicPersistentTileScheduler]
    :param epilogue_op: Optional elementwise operation to apply
    :type epilogue_op: Constexpr
    :param tmem_dealloc_barrier: Barrier for tensor memory deallocation
    :type tmem_dealloc_barrier: pipeline.NamedBarrier
    :param tCcC_base: Identity/coordinate tensor C
    :type tCcC_base: cute.Tensor
    :param mC_mnl: Global memory tensor C (full tensor for predicate computation)
    :type mC_mnl: cute.Tensor
    :param clc_pipeline: Pipeline for dynamic persistent tile scheduling
    :type clc_pipeline: Union[pipeline.PipelineClcFetchAsync, None]
    :param clc_consumer_state: Consumer state for dynamic persistent tile scheduling
    :type clc_consumer_state: Union[pipeline.PipelineState, None]
    :param a_scale: Optional per-tensor scale applied to the accumulator
    :type a_scale: Optional[cute.Tensor]
    :param b_scale: Optional per-tensor scale applied to the accumulator
    :type b_scale: Optional[cute.Tensor]
    """

    # Layout transformation for tCgC_base
    # ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, TILE_M, TILE_N, TILE_K)
    # -> ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), TILE_M, TILE_N, TILE_K)
    tCgC = transform_partitioned_tensor_layout(tCgC_base)

    # Layout transformation for tCtAcc_base
    # ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, STAGE)
    # -> ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), STAGE)
    tCtAcc = transform_partitioned_tensor_layout(tCtAcc_base)

    #
    # Partition for epilogue
    #
    (
        tiled_copy_t2r,
        tTR_tAcc_base,
        tTR_rAcc,
    ) = epilogue_tmem_copy_and_partition(
        gemm_kernel, epi_tidx, tCtAcc, tCgC, epi_tile, gemm_kernel.use_2cta_instrs
    )

    gC_epi = cute.flat_divide(tCgC, epi_tile)
    # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, RestM, RestN, RestL)
    thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
    tTR_gC_partitioned = thr_copy_t2r.partition_D(gC_epi)
    # (T2R, T2R_M, T2R_N)
    tTR_rC = cute.make_rmem_tensor(
        tTR_gC_partitioned[(None, None, None, 0, 0, 0, 0, 0)].shape, gemm_kernel.c_dtype
    )

    mclD = cute.max_common_layout(
        tTR_rC.layout, tTR_gC_partitioned[(None, None, None, 0, 0, 0, 0, 0)].layout
    )
    num_bits_per_copy = min(
        tTR_gC_partitioned.iterator.alignment * 8,
        cute.size(mclD) * gemm_kernel.c_dtype.width,
        256,
    )

    # Cache policy selection for epilogue store:
    # - Use NO_ALLOCATE since this data is never reused after the store, would get better perf
    simt_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        gemm_kernel.c_dtype,
        num_bits_per_copy=num_bits_per_copy,
        **_NO_ALLOCATE_KWARGS,
    )
    use_predication = tCcC_base is not None and mC_mnl is not None

    if const_expr(use_predication):
        # Layout transformation for tCcC_base
        # ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, TILE_M, TILE_N, TILE_K)
        # -> ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), TILE_M, TILE_N, TILE_K)
        tCcC = transform_partitioned_tensor_layout(tCcC_base)
        cC_epi = cute.flat_divide(tCcC, epi_tile)
        tTR_cC_partitioned = thr_copy_t2r.partition_D(cC_epi)

    acc_consumer_state = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Consumer, gemm_kernel.num_acc_stage
    )

    work_tile = tile_sched.initial_work_tile_info()
    while work_tile.is_valid_tile:
        #
        # Pre-advance to next tile
        #
        if const_expr(isinstance(tile_sched, StaticPersistentTileScheduler)):
            tile_sched.advance_to_next_work()
            next_work_tile = tile_sched.get_current_work()

        # Get tile coord from current work tile
        cur_tile_coord = work_tile.tile_idx
        mma_tile_coord_mnl = (
            cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
            cur_tile_coord[1],
            cur_tile_coord[2],
        )

        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
        tTR_gC = tTR_gC_partitioned[
            (
                None,
                None,
                None,
                None,
                None,
                *mma_tile_coord_mnl,
            )
        ]
        if const_expr(use_predication):
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
            tTR_cC = tTR_cC_partitioned[
                (
                    None,
                    None,
                    None,
                    None,
                    None,
                    *mma_tile_coord_mnl,
                )
            ]
            tTR_cC = cute.group_modes(tTR_cC, 3, cute.rank(tTR_cC))

        # Set tensor memory buffer for current tile
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_M)
        tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]

        #
        # Wait for accumulator buffer full
        #
        acc_pipeline.consumer_wait(acc_consumer_state)

        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
        tTR_gC = cute.group_modes(tTR_gC, 3, cute.rank(tTR_gC))
        #
        # Store accumulator to global memory in subtiles
        #
        subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])
        for subtile_idx in range(subtile_cnt):
            #
            # Get the destination and coordinate slices for this subtile
            #
            tTR_gC_subtile = tTR_gC[(None, None, None, subtile_idx)]
            #
            # Load accumulator from tensor memory buffer to register
            #
            tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
            cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
            # Async arrive accumulator buffer empty
            # Release early for perf
            if subtile_idx == subtile_cnt - 1:
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

            #
            # Apply scale and convert to C type
            #
            acc_vec = tTR_rAcc.load()
            if a_scale is not None and b_scale is not None:
                acc_vec = (a_scale[0] * b_scale[0]) * acc_vec
            acc_vec = epilogue_op(acc_vec.to(gemm_kernel.c_dtype))
            tTR_rC.store(acc_vec)

            if const_expr(use_predication):
                # compute predicate
                tTR_cC_subtile = tTR_cC[(None, None, None, subtile_idx)]
                pred_C_shape = (1, *tTR_cC_subtile.shape[1:])
                pred_C = cute.make_rmem_tensor(pred_C_shape, Boolean)
                for m_idx in range(tTR_cC_subtile.shape[1]):
                    for n_idx in range(tTR_cC_subtile.shape[2]):
                        vector_first_coord = tTR_cC_subtile[(0, m_idx, n_idx)]
                        pred_C[(0, m_idx, n_idx)] = cute.elem_less(vector_first_coord, mC_mnl.shape)
                # Store C to global memory with predication
                cute.copy(simt_atom, tTR_rC, tTR_gC_subtile, pred=pred_C)
            else:
                # Store C directly to global memory
                cute.copy(simt_atom, tTR_rC, tTR_gC_subtile)

        if const_expr(isinstance(tile_sched, StaticPersistentTileScheduler)):
            work_tile = next_work_tile
        elif const_expr(isinstance(tile_sched, ClcDynamicPersistentTileScheduler)):
            assert clc_pipeline is not None and clc_consumer_state is not None, (
                "a CLC scheduler requires clc_pipeline and clc_consumer_state"
            )
            clc_pipeline.consumer_wait(clc_consumer_state)
            work_tile = tile_sched.get_current_work()
            clc_pipeline.consumer_release(clc_consumer_state)
            clc_consumer_state.advance()
        else:
            raise TypeError(f"unsupported tile scheduler type: {type(tile_sched)}")

    # Synchronize before TMEM dealloc (done by the caller)
    tmem_dealloc_barrier.arrive_and_wait()
