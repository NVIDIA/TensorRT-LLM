# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from CUTLASS utils.gemm.sm100.epilogue_tma_store.
"""Epilogue of the in-cluster K-split Rubin block-scaled dense GEMM.

``cluster_k = ck`` CTAs of one cluster (cluster z) compute the same output tile from disjoint
slices of K, each into its own tensor-memory accumulator. The epilogue reduces the ``ck``
FP32 partials through distributed shared memory (DSMEM) and stores the output tile once:

* the tile is split into epilogue subtiles (``epi_tile``) of ``EN`` columns; subtile ``s`` is
  owned by the CTA of K rank ``s % ck``;
* every CTA loads each subtile from tensor memory (all epilogue threads, one row each). A CTA
  that does not own the subtile pushes its 128 x ``EN`` FP32 partial into a mailbox in the
  owner's shared memory with ``st.async`` (16 bytes per instruction, completion credited to
  an mbarrier of the owner); the owner waits for the ``ck - 1`` partials of that subtile,
  adds them to its own in the fixed order K rank 0, 1, ..., ck - 1 (so the result does not
  depend on timing and is identical for every owner and run), converts, and stores the
  subtile through shared memory and TMA exactly like ``epilogue_tma_store``;
* a subtile with no valid output column (a partial N tile, e.g. a decode batch of a few
  tokens in a 64-column tile) is neither reduced nor stored.

One mailbox slot (the ``EN`` partial values of the 128 epilogue threads of one source CTA)
and one mbarrier exist per (owned subtile, source), written exactly once: the kernel runs
one output tile per cluster (non-persistent grid), so no slot is reused.
"""

from typing import Any, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Constexpr, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils.gemm.sm100 import (
    epilogue_smem_copy_and_partition,
    epilogue_tmem_copy_and_partition,
    transform_partitioned_tensor_layout,
)

from .streamk_layout import EPILOGUE_THREADS, mailbox_barriers

WARP_SIZE = 32
VALUES_PER_PUSH = 4  # FP32 values of one st.async.v4 (16 bytes)


@dsl_user_op
def map_shared_rank(smem_ptr: cute.Pointer, peer_cta_rank: Int32, *, loc=None, ip=None) -> Int32:
    """Shared-memory address of ``smem_ptr`` in the CTA ``peer_cta_rank`` of the cluster (mapa)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr.toint(loc=loc, ip=ip).ir_value(), peer_cta_rank.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_shared_remote_v4(
    value0: cutlass.Float32,
    value1: cutlass.Float32,
    value2: cutlass.Float32,
    value3: cutlass.Float32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store 4 FP32 values to ``smem_ptr`` of CTA ``peer_cta_rank`` and credit its mbarrier."""
    llvm.inline_asm(
        None,
        [
            map_shared_rank(smem_ptr, peer_cta_rank, loc=loc, ip=ip).ir_value(),
            value0.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value1.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value2.bitcast(Int32).ir_value(loc=loc, ip=ip),
            value3.bitcast(Int32).ir_value(loc=loc, ip=ip),
            map_shared_rank(mbar_ptr, peer_cta_rank, loc=loc, ip=ip).ir_value(),
        ],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
        "[$0], {$1, $2, $3, $4}, [$5];",
        "r,r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def epilogue_tma_store_streamk(
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
    epilogue_op: Constexpr,
    mma_tile_coord_mnl: Tuple[Int32, Int32, Int32],
    acc_consumer_state: pipeline.PipelineState,
    acc_pipeline: pipeline.PipelineAsync,
    c_pipeline: pipeline.PipelineTmaStore,
    # In-cluster K split
    mailbox: cute.Tensor,
    mailbox_bars: cute.Pointer,
    k_rank: Int32,
    cta_rank_in_cluster: Int32,
    n_extent: Int32,
) -> pipeline.PipelineState:
    ck = gemm_kernel.cluster_k  # K splits (static)
    epi_n = gemm_kernel.streamk_epi_n  # columns per epilogue subtile (static)
    tile_n = gemm_kernel.cta_tile_shape_mnk[1]
    rank_stride = gemm_kernel.cluster_shape_mn[0] * gemm_kernel.cluster_shape_mn[1]
    chunks = epi_n // VALUES_PER_PUSH  # 16-byte chunks per thread and subtile
    owned_max = mailbox_barriers(ck, epi_n, tile_n)
    slot_floats = epi_n * EPILOGUE_THREADS  # FP32 values of one (subtile, source) slot
    expected_bytes = (ck - 1) * slot_floats * 4  # bytes one owned subtile receives

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
    # FP32 sum of the K splits of one subtile (same layout as the loaded partial)
    tTR_rSum = cute.make_rmem_tensor(tTR_rAcc.shape, cutlass.Float32)

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

    # Subtiles with at least one valid output column (every CTA of the cluster computes the
    # same count: it depends on the tile only).
    valid_cols = cutlass.min(tile_n, n_extent - mma_tile_coord_mnl[1] * tile_n)
    n_valid_subtiles = (valid_cols + (epi_n - 1)) // epi_n

    # The owner of an owned subtile expects the (ck - 1) partials of the other K ranks: one
    # arrival (this thread) plus their transaction bytes. A peer may push before this point.
    for owned in cutlass.range_constexpr(owned_max):
        if epi_tidx == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(mailbox_bars + owned, expected_bytes)

    #
    # Wait for accumulator buffer full
    #
    acc_pipeline.consumer_wait(acc_consumer_state)

    tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
    bSG_gC = cute.group_modes(bSG_gC, 1, cute.rank(bSG_gC))

    for subtile_idx in range(n_valid_subtiles):
        #
        # Load accumulator (this CTA's partial over its K slice) from tensor memory
        #
        tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]  # type: ignore[call-overload]
        cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)

        owner = subtile_idx % ck
        owned = subtile_idx // ck  # index among the subtiles of the owner

        if owner != k_rank:
            #
            # Not the owner: push the partial into the owner's mailbox slot of this source.
            # Sources are numbered 0..ck-2 in K rank order, skipping the owner.
            #
            source = cutlass.select_(k_rank > owner, k_rank - 1, k_rank)
            owner_cta = cta_rank_in_cluster + (owner - k_rank) * rank_stride
            slot_base = (owned * (ck - 1) + source) * slot_floats + epi_tidx * VALUES_PER_PUSH
            for chunk in cutlass.range_constexpr(chunks):
                v = chunk * VALUES_PER_PUSH
                store_shared_remote_v4(
                    tTR_rAcc[v],
                    tTR_rAcc[v + 1],
                    tTR_rAcc[v + 2],
                    tTR_rAcc[v + 3],
                    mailbox.iterator + slot_base + chunk * (EPILOGUE_THREADS * VALUES_PER_PUSH),
                    mailbox_bars + owned,
                    owner_cta,
                )
        else:
            #
            # Owner: wait for the other K ranks, add in the order of the K rank (own partial in
            # its position), then convert and store like epilogue_tma_store.
            #
            cute.arch.mbarrier_wait(mailbox_bars + owned, 0)
            for rank in cutlass.range_constexpr(ck):
                if k_rank == rank:
                    for i in cutlass.range_constexpr(epi_n):
                        if cutlass.const_expr(rank == 0):
                            tTR_rSum[i] = tTR_rAcc[i]
                        else:
                            tTR_rSum[i] = tTR_rSum[i] + tTR_rAcc[i]
                else:
                    source = cutlass.select_(rank > k_rank, rank - 1, rank)
                    slot_base = (
                        owned * (ck - 1) + source
                    ) * slot_floats + epi_tidx * VALUES_PER_PUSH
                    for chunk in cutlass.range_constexpr(chunks):
                        for c in cutlass.range_constexpr(VALUES_PER_PUSH):
                            i = chunk * VALUES_PER_PUSH + c
                            value = mailbox[
                                slot_base + chunk * (EPILOGUE_THREADS * VALUES_PER_PUSH) + c
                            ]
                            if cutlass.const_expr(rank == 0):
                                tTR_rSum[i] = value
                            else:
                                tTR_rSum[i] = tTR_rSum[i] + value

            #
            # Convert to C type
            #
            acc_vec = tiled_copy_r2s.retile(tTR_rSum).load()
            acc_vec = epilogue_op(acc_vec.to(gemm_kernel.c_dtype))
            tRS_rC.store(acc_vec)

            #
            # Store C to shared memory
            #
            c_buffer = owned % gemm_kernel.num_c_stage
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
