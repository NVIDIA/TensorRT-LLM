# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Symbolic-T shared projection with cluster-local split-K.

This candidate increases small-prefill parallelism without a separate global
partial buffer, reduction kernel or memset. Each CTA owns a K slice; independent
distributed-shared-memory mailboxes collect all FP32 partials before the native
BF16 preactivation boundary. Adjacent gate/up rows are prepared once at model
loading. Up and down have separate callers and keep the local merge explicit.

The decode projection template is reused only for compile-time resource
accounting and constants; its kernels and dispatch remain unchanged. Runtime
T is 65..8192, clipped by TMA on load and masked on store. No per-T compilation,
native computation or hidden output initialization is used. PDL protects input
reads, and every CTA signals only after its output work has been issued.

This is an experimental alternative to the 2-CTA persistent schedule; it is
not used by default. Split-K changes floating-point association.
"""

import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda_exp
import torch
from cuda.bindings import driver as cuda
from cutlass.experimental import primitives as prims

from ..common import compat48 as _compat48
from ._projection_tcgen05_template import (
    EPI_WARP0,
    EPILOGUE_SWIGLU_INTERLEAVED,
    MMA_K,
    MMA_WARP,
    PREWAIT_TMA_A,
    PREWAIT_TMA_W,
    SWIZZLED_SMEM_ALIGNMENT,
    THREADS,
    TMA_WARP,
    TMEM_LD_WARPS,
    WARP_THREADS,
    ProjectionConfig,
)
from .moe_prefill_shared import _validate_arguments


@cute.kernel
def shared_splitk_kernel(
    tma_a_desc: cutlass.GridConstant[cuda_exp.TensorMap],
    tma_w_desc: cutlass.GridConstant[cuda_exp.TensorMap],
    c_ptr: cute.Tensor,
    stages: cutlass.Constexpr[int],
    split_k: cutlass.Constexpr[int],
    local_k_steps: cutlass.Constexpr[int],
    active_m: cutlass.Int32,
    n_cols: cutlass.Constexpr[int],
    output_n: cutlass.Constexpr[int],
    mma_n: cutlass.Constexpr[int],
    tmem_subs: cutlass.Constexpr[int],
    tmem_sub_last: cutlass.Constexpr[int],
    reduce_peers: cutlass.Constexpr[int],
    reducers: cutlass.Constexpr[int],
    chunk_cols: cutlass.Constexpr[int],
    output_chunk_cols: cutlass.Constexpr[int],
    tmem_cols_alloc: cutlass.Constexpr[int],
    mailbox_stride: cutlass.Constexpr[int],
    mailbox_values: cutlass.Constexpr[int],
    reduce_payload_bytes: cutlass.Constexpr[int],
    store_plan: cutlass.Constexpr[tuple],
    prewait_tma_operand: cutlass.Constexpr[str],
    epilogue: cutlass.Constexpr[str],
    out_fp32: cutlass.Constexpr[bool],
    prefetch_w_k_steps: cutlass.Constexpr[int],
    physical_m: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    AB_DTYPE: cutlass.Constexpr = tma_a_desc.dtype
    SMEM_SWIZZLE: cutlass.Constexpr = tma_a_desc.swizzle.to(prims.Tcgen05SmemSwizzle)
    ELEM_BYTES: cutlass.Constexpr[int] = AB_DTYPE.width // 8
    # Descriptor offsets use the s128b swizzle: each row contains eight 16B
    # atoms, and the stride advances eight rows. Pass the corresponding byte
    # offsets to the typed descriptor rather than confusing bytes and atoms.
    ROW_BYTES: cutlass.Constexpr[int] = MMA_K * ELEM_BYTES  # 128 for bf16 K=64
    DESC_LEAD: cutlass.Constexpr[int] = ROW_BYTES // 16  # 8
    DESC_STRIDE: cutlass.Constexpr[int] = 8 * MMA_K * ELEM_BYTES // 16  # 64
    K_GRANULE: cutlass.Constexpr[int] = 16  # bf16 MMA K granule

    # Operand rings are independent of the reduction mailbox. A peer may
    # finish its K slice and publish partials while this CTA still loads A/W.
    smem_a = cutlass.Array(
        AB_DTYPE,
        stages * physical_m * MMA_K,
        space=cutlass.AddressSpace.smem,
        alignment=SWIZZLED_SMEM_ALIGNMENT,
    )
    smem_w = cutlass.Array(
        AB_DTYPE,
        stages * mma_n * MMA_K,
        space=cutlass.AddressSpace.smem,
        alignment=SWIZZLED_SMEM_ALIGNMENT,
    )
    full_bar = cutlass.Array(cutlass.Int64, stages, space=cutlass.AddressSpace.smem, alignment=8)
    empty_bar = cutlass.Array(cutlass.Int64, stages, space=cutlass.AddressSpace.smem, alignment=8)
    acc_mbar = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    reduce_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem, alignment=8)
    # Dedicated storage permits early peer publication without overwriting
    # in-flight operand stages. The prologue cluster rendezvous establishes
    # mailbox lifetime, not completion of another CTA's K loop.
    reduce_mailbox = cutlass.Array(
        cutlass.Float32,
        mailbox_values,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    tmem_ptr_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=4)

    warp_idx = cute.arch.warp_idx()
    tidx, _, _ = cute.arch.thread_idx()
    bidx, bidy, _ = cute.arch.block_idx()
    lane = tidx % cutlass.Int32(WARP_THREADS)
    cta_rank = cute.arch.block_idx_in_cluster()

    coord_n = bidx * cutlass.Int32(mma_n)
    coord_m = bidy * cutlass.Int32(physical_m)
    # Each rank owns a contiguous K slice so its TMA boxes stay sequential.
    k_base = cta_rank * cutlass.Int32(local_k_steps * MMA_K)

    # Physical allocation (power of two); the drain walks only TMEM_COLS_LOGICAL.
    num_tmem_cols: cutlass.Constexpr[int] = tmem_cols_alloc
    idesc = prims.Tcgen05InstrDesc.build(
        c_dtype=cutlass.Float32,
        a_dtype=AB_DTYPE,
        b_dtype=AB_DTYPE,
        n_dim=mma_n,
        m_dim=physical_m,
    )

    # Pull this CTA's first weight tiles into L2 before anything else. The prefetch
    # takes no mbarrier and no completion, so it does not wait for the init below,
    # and the weight does not depend on the predecessor grid, so it does not wait
    # for griddepcontrol.wait either. Everything between here and the first TMA load
    # -- barrier init, TMEM allocation, two CTA barriers -- is the independent work
    # the DRAM round trip then hides behind.
    if cutlass.const_expr(prefetch_w_k_steps > 0):
        if warp_idx == TMA_WARP:
            if prims.elect_sync():
                for k in cutlass.range_constexpr(prefetch_w_k_steps):
                    prims.cp_async_bulk_tensor_prefetch(
                        tma_w_desc.get_ptr(),
                        (k_base + k * cutlass.Int32(MMA_K), coord_n),
                        [],
                    )

    if warp_idx == TMA_WARP:
        if prims.elect_sync():
            for s in cutlass.range_constexpr(stages):
                prims.mbarrier_init(full_bar.subview(s), 1)
                prims.mbarrier_init(empty_bar.subview(s), 1)
            prims.mbarrier_init(acc_mbar, 1)
            prims.mbarrier_init(reduce_full, 1)
    prims.fence_mbarrier_init()
    # Arm the reduce barrier before any peer can target it, then let the cluster
    # know this rank's mailbox is addressable. The matching wait is deferred to
    # just before the senders need rank 0's mapped address.
    if cutlass.const_expr(reduce_peers > 0):
        if warp_idx == TMA_WARP:
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(reduce_full, reduce_payload_bytes)
        # Safe in the prologue: with separate storage the arrive only announces
        # "my mailbox address is live", true from allocation.
        prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cta_sync(0)

    if warp_idx == TMA_WARP:
        prims.tcgen05_alloc(tmem_ptr_i32, num_tmem_cols, group="cta_1")
    prims.barrier_cta_sync(0)

    tmem_base_raw = tmem_ptr_i32.load()
    tmem_ptr = cutlass.inttoptr(tmem_base_raw, 6, cutlass.Float32)
    if warp_idx == TMA_WARP:
        # Allocation management is warp-collective. Release once from the
        # allocating warp after all threads can observe the allocated pointer.
        prims.tcgen05_relinquish_alloc_permit(group="cta_1")

    a_ptr = tma_a_desc.get_ptr()
    w_ptr = tma_w_desc.get_ptr()
    tma_bytes = tma_a_desc.global_tx_bytes() + tma_w_desc.global_tx_bytes()

    A_STAGE_ELEMS: cutlass.Constexpr[int] = physical_m * MMA_K
    W_STAGE_ELEMS: cutlass.Constexpr[int] = mma_n * MMA_K
    A_STAGE_BYTES: cutlass.Constexpr[int] = A_STAGE_ELEMS * ELEM_BYTES
    W_STAGE_BYTES: cutlass.Constexpr[int] = W_STAGE_ELEMS * ELEM_BYTES
    NUM_K_BLOCKS: cutlass.Constexpr[int] = MMA_K // K_GRANULE
    DESC_STEP_BYTES: cutlass.Constexpr[int] = K_GRANULE * ELEM_BYTES

    desc_a_root = prims.Tcgen05SmemDesc.build(
        start_address=smem_a,
        leading_byte_offset=DESC_LEAD * 16,
        stride_byte_offset=DESC_STRIDE * 16,
        layout=SMEM_SWIZZLE,
    )
    desc_w_root = prims.Tcgen05SmemDesc.build(
        start_address=smem_w,
        leading_byte_offset=DESC_LEAD * 16,
        stride_byte_offset=DESC_STRIDE * 16,
        layout=SMEM_SWIZZLE,
    )

    # ---- producer warp: TMA only ------------------------------------------
    if warp_idx == TMA_WARP:
        for k in cutlass.range(local_k_steps):
            stage = k % stages
            # A stage may only be refilled after the MMA warp releases it. In
            # the single-warp version program order guaranteed that; with a
            # separate producer the empty barrier is what orders it.
            if k >= stages:
                empty_phase = ((k - stages) // stages) % 2
                while not prims.mbarrier_try_wait_parity(
                    empty_bar.subview(stage), empty_phase, time_limit=10_000_000
                ):
                    pass
            if prims.elect_sync():
                prims.mbarrier_arrive_expect_tx(full_bar.subview(stage), tma_bytes)
                k_coord = k_base + k * cutlass.Int32(MMA_K)
                # Programmatic dependent launch wait is deliberately as late as
                # possible: one predecessor-independent operand TMA can start
                # first, then the first dependent operand TMA is protected by
                # the wait. The wait is needed only for k=0 because it waits
                # for the whole predecessor grid to finish and memory-flush.
                if cutlass.const_expr(prewait_tma_operand == PREWAIT_TMA_A):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_a.subview(stage * A_STAGE_ELEMS),
                        a_ptr,
                        (k_coord, coord_m),
                        full_bar.subview(stage),
                    )
                    if cutlass.const_expr(use_pdl):
                        if k == 0:
                            cute.arch.griddepcontrol_wait()
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_w.subview(stage * W_STAGE_ELEMS),
                        w_ptr,
                        (k_coord, coord_n),
                        full_bar.subview(stage),
                    )
                elif cutlass.const_expr(prewait_tma_operand == PREWAIT_TMA_W):
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_w.subview(stage * W_STAGE_ELEMS),
                        w_ptr,
                        (k_coord, coord_n),
                        full_bar.subview(stage),
                    )
                    if cutlass.const_expr(use_pdl):
                        if k == 0:
                            cute.arch.griddepcontrol_wait()
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_a.subview(stage * A_STAGE_ELEMS),
                        a_ptr,
                        (k_coord, coord_m),
                        full_bar.subview(stage),
                    )
                else:
                    if cutlass.const_expr(use_pdl):
                        if k == 0:
                            cute.arch.griddepcontrol_wait()
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_a.subview(stage * A_STAGE_ELEMS),
                        a_ptr,
                        (k_coord, coord_m),
                        full_bar.subview(stage),
                    )
                    prims.cp_async_bulk_tensor_shared_cta_global(
                        smem_w.subview(stage * W_STAGE_ELEMS),
                        w_ptr,
                        (k_coord, coord_n),
                        full_bar.subview(stage),
                    )

    # ---- MMA warp ----------------------------------------------------------
    if warp_idx == MMA_WARP:
        for k in cutlass.range(local_k_steps):
            stage = k % stages
            full_phase = (k // stages) % 2
            while not prims.mbarrier_try_wait_parity(
                full_bar.subview(stage), full_phase, time_limit=10_000_000
            ):
                pass
            scale_d = k > 0
            for kb in cutlass.range_constexpr(NUM_K_BLOCKS):
                off_a = cutlass.Int32(stage) * cutlass.Int32(A_STAGE_BYTES) + cutlass.Int32(
                    DESC_STEP_BYTES * kb
                )
                off_w = cutlass.Int32(stage) * cutlass.Int32(W_STAGE_BYTES) + cutlass.Int32(
                    DESC_STEP_BYTES * kb
                )
                if prims.elect_sync():
                    prims.tcgen05_mma(
                        prims.Tcgen05MMAKind.F16,
                        prims.CTAGroup.CTA_1,
                        tmem_ptr,
                        desc_a_root.advance_start_address(off_a),
                        desc_w_root.advance_start_address(off_w),
                        idesc,
                        scale_d if kb == 0 else True,
                    )
            if prims.elect_sync():
                prims.tcgen05_commit(empty_bar.subview(stage), group=prims.CTAGroup.CTA_1)
        if prims.elect_sync():
            prims.tcgen05_commit(acc_mbar, group=prims.CTAGroup.CTA_1)

    prims.barrier_cta_sync(0)
    while not prims.mbarrier_try_wait_parity(acc_mbar, 0, time_limit=10_000_000):
        pass

    # ---- drain TMEM -------------------------------------------------------
    # Four epilogue warps drain the physical TMEM subpartitions. Each owns
    # physical_m / 4 rows: 16 lanes for M64 and all 32 lanes for M128.
    is_epi_warp = (warp_idx >= EPI_WARP0) & (warp_idx < EPI_WARP0 + TMEM_LD_WARPS)
    # TMEM subpartition follows the warp's PHYSICAL position in its warpgroup
    # (warp_idx % 4), not its offset within the epilogue range. Warps 2..5 read
    # SP 2,3,0,1 -- so deriving the output row from (warp_idx - EPI_WARP0)
    # permutes the rows and silently writes correct data to wrong addresses.
    epi_slot = warp_idx % cutlass.Int32(4)
    tmem_base = prims.TmemAddr(tmem_base_raw)
    base_col = tmem_base.col_id
    tmem_row = tmem_base.row_id
    acc = cutlass.Array(cutlass.Float32, mma_n, alignment=32)
    if is_epi_warp:
        # Issue both TMEM loads before waiting so they overlap; a wait inside
        # the loop serialises them.
        for sub in cutlass.range_constexpr(tmem_subs):
            tmem_sub = prims.TmemAddr.from_row_col(tmem_row, base_col + sub * 32).as_ptr(
                cutlass.Float32
            )
            # The final load is short when MMA_N is not a multiple of 32.
            n_vals: cutlass.Constexpr[int] = 32 if sub < tmem_subs - 1 else tmem_sub_last
            acc.store(prims.tcgen05_ld("32x32b", tmem_sub, num=n_vals), sub * 32)
        prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)

    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    prims.barrier_cta_sync(0)
    if warp_idx == TMA_WARP:
        prims.tcgen05_dealloc(tmem_ptr, num_tmem_cols, group="cta_1")

    # ---- DSMEM split-K reduction -----------------------------------------
    is_owner_lane = is_epi_warp & (lane < cutlass.Int32((physical_m // 4)))
    if cutlass.const_expr(reduce_peers > 0):
        prims.barrier_cluster_wait()

        owner_id = epi_slot * cutlass.Int32((physical_m // 4)) + lane
        # Send every chunk this rank does NOT own to the CTA that does. Ranks at
        # or beyond REDUCERS own nothing and ship all of their chunks out.
        if is_owner_lane:
            for tgt in cutlass.range_constexpr(reducers):
                if cta_rank != cutlass.Int32(tgt):
                    # Slot index among the senders that target `tgt`: ranks are
                    # renumbered so each reducer sees a dense 0..PEERS-1 range.
                    slot = (
                        cta_rank - cutlass.Int32(1) if cta_rank > cutlass.Int32(tgt) else cta_rank
                    )
                    base = slot * cutlass.Int32(
                        physical_m * mailbox_stride
                    ) + owner_id * cutlass.Int32(mailbox_stride)
                    peer_full = prims.mapa(reduce_full, cutlass.Int32(tgt))
                    for chunk in cutlass.range_constexpr(chunk_cols // 4):
                        src = tgt * chunk_cols + chunk * 4
                        payload = acc.load(src, vector_size=4, alignment=16).bitcast(cutlass.Int32)
                        _compat48.store_async(
                            prims.mapa(
                                reduce_mailbox.data_ptr() + base + cutlass.Int32(chunk * 4),
                                cutlass.Int32(tgt),
                            ),
                            payload,
                            peer_full,
                        )

        # Reducers fold in what they received for their own chunk.
        #
        # Unroll the rank branches so every register-array index is constant.
        # A runtime acc[cta_rank * CHUNK + ...] index can instead require local
        # storage; do not infer a register layout from source array syntax.
        if cta_rank < cutlass.Int32(reducers):
            while not prims.mbarrier_try_wait_parity(reduce_full, 0):
                pass
            cute.arch.fence_acq_rel_cta()
            if is_owner_lane:
                for r in cutlass.range_constexpr(reducers):
                    if cta_rank == cutlass.Int32(r):
                        for peer in cutlass.range_constexpr(reduce_peers):
                            base = cutlass.Int32(
                                peer * physical_m * mailbox_stride
                            ) + owner_id * cutlass.Int32(mailbox_stride)
                            for chunk in cutlass.range_constexpr(chunk_cols // 4):
                                got = reduce_mailbox.subview(base + cutlass.Int32(chunk * 4)).load(
                                    vector_size=4, alignment=16
                                )
                                for i in cutlass.range_constexpr(4):
                                    dst = r * chunk_cols + chunk * 4 + i
                                    acc[dst] = acc[dst] + got[i]

    # ---- each reducer writes its own column slice ------------------------
    if cta_rank < cutlass.Int32(reducers):
        if is_owner_lane:
            gC = c_ptr.iterator.raw_ptr()
            row = coord_m + epi_slot * cutlass.Int32((physical_m // 4)) + lane
            # Rows >= active_m are MMA results over TMA-zero-filled A: harmless
            # numerically, but outside the caller's tensor.
            in_range = row < cutlass.Int32(active_m)
            # Vector stores preserve complete BF16 output chunks. Runtime row
            # masks exclude tail padding; reducer-owned slices do not overlap.
            for r in cutlass.range_constexpr(reducers):
                if cta_rank == cutlass.Int32(r):
                    if cutlass.const_expr(epilogue == EPILOGUE_SWIGLU_INTERLEAVED):
                        # Weight rows are load-time interleaved as
                        # (gate_0, up_0, gate_1, up_1, ...), so each pair is
                        # register-local after split-K reduction.
                        swi = cutlass.Array(
                            cutlass.Float32,
                            output_chunk_cols,
                            alignment=32,
                        )
                        for i in cutlass.range_constexpr(output_chunk_cols):
                            # Match the BF16 materialization in production's
                            # separate projection -> activation path.
                            gate = (
                                acc[r * chunk_cols + 2 * i].to(cutlass.BFloat16).to(cutlass.Float32)
                            )
                            up = (
                                acc[r * chunk_cols + 2 * i + 1]
                                .to(cutlass.BFloat16)
                                .to(cutlass.Float32)
                            )
                            swi[i] = (
                                gate
                                * cute.math.rcp(
                                    1.0
                                    + cute.math.exp(
                                        -gate,
                                        fastmath=True,
                                    ),
                                    fastmath=True,
                                )
                                * up
                            )
                        for off, width in store_plan:
                            col = coord_n // cutlass.Int32(2) + cutlass.Int32(
                                r * output_chunk_cols + off
                            )
                            out = (
                                swi.load(
                                    off,
                                    vector_size=width,
                                    alignment=16,
                                )
                                .to(cutlass.BFloat16)
                                .bitcast(cutlass.Int16)
                            )
                            if in_range and col + cutlass.Int32(width) <= cutlass.Int32(output_n):
                                prims.store_ext(
                                    out,
                                    gC + row * cutlass.Int32(output_n) + col,
                                )
                    else:
                        for off, width in store_plan:
                            src = r * chunk_cols + off
                            col = coord_n + cutlass.Int32(r * chunk_cols + off)
                            acc_vec = acc.load(
                                src,
                                vector_size=width,
                                alignment=16,
                            )
                            if cutlass.const_expr(out_fp32):
                                out = acc_vec.bitcast(cutlass.Int32)
                            else:
                                out = acc_vec.to(cutlass.BFloat16).bitcast(cutlass.Int16)
                            # Two independent masks. `in_range` drops rows past
                            # active_m; the column predicate drops the tail tile.
                            if in_range and col + cutlass.Int32(width) <= cutlass.Int32(output_n):
                                prims.store_ext(
                                    out,
                                    gC + row * cutlass.Int32(output_n) + col,
                                )

    # Release the dependent kernel. This sits outside the reducer branch: only
    # ranks < REDUCERS store, but every CTA in the grid has to signal before the
    # dependent grid is allowed to launch, so a signal inside the branch would
    # stall the successor behind the non-reducer ranks' exit instead.
    prims.barrier_cta_sync(0)
    if cutlass.const_expr(use_pdl):
        if tidx == 0:
            cute.arch.griddepcontrol_launch_dependents()


@cute.jit
def launch(
    a: cute.Tensor,
    w: cute.Tensor,
    c: cute.Tensor,
    stream: cuda.CUstream,
    stages: cutlass.Constexpr[int],
    split_k: cutlass.Constexpr[int],
    local_k_steps: cutlass.Constexpr[int],
    n_cols: cutlass.Constexpr[int],
    output_n: cutlass.Constexpr[int],
    k_total: cutlass.Constexpr[int],
    mma_n: cutlass.Constexpr[int],
    n_ctas: cutlass.Constexpr[int],
    tmem_subs: cutlass.Constexpr[int],
    tmem_sub_last: cutlass.Constexpr[int],
    reduce_peers: cutlass.Constexpr[int],
    reducers: cutlass.Constexpr[int],
    chunk_cols: cutlass.Constexpr[int],
    output_chunk_cols: cutlass.Constexpr[int],
    tmem_cols_alloc: cutlass.Constexpr[int],
    mailbox_stride: cutlass.Constexpr[int],
    mailbox_values: cutlass.Constexpr[int],
    reduce_payload_bytes: cutlass.Constexpr[int],
    store_plan: cutlass.Constexpr[tuple],
    prewait_tma_operand: cutlass.Constexpr[str],
    epilogue: cutlass.Constexpr[str],
    out_fp32: cutlass.Constexpr[bool],
    prefetch_w_k_steps: cutlass.Constexpr[int],
    physical_m: cutlass.Constexpr[int],
    use_pdl: cutlass.Constexpr[bool],
) -> None:
    active_m = a.shape[0]
    ELEM_BYTES: cutlass.Constexpr[int] = a.element_type.width // 8
    tma_a = cuda_exp.create_tensor_map_tiled(
        global_address=a.iterator.toint(),
        dtype=a.element_type,
        global_dims=[k_total, active_m],
        global_strides=[k_total * ELEM_BYTES // 16],
        box_dims=[MMA_K, physical_m],
        swizzle=cuda_exp.TensorMapSwizzle.s128b,
    )
    tma_w = cuda_exp.create_tensor_map_tiled(
        global_address=w.iterator.toint(),
        dtype=w.element_type,
        global_dims=[k_total, n_cols],
        global_strides=[k_total * ELEM_BYTES // 16],
        box_dims=[MMA_K, mma_n],
        swizzle=cuda_exp.TensorMapSwizzle.s128b,
    )
    shared_splitk_kernel(
        tma_a,
        tma_w,
        c,
        stages,
        split_k,
        local_k_steps,
        active_m,
        n_cols,
        output_n,
        mma_n,
        tmem_subs,
        tmem_sub_last,
        reduce_peers,
        reducers,
        chunk_cols,
        output_chunk_cols,
        tmem_cols_alloc,
        mailbox_stride,
        mailbox_values,
        reduce_payload_bytes,
        store_plan,
        prewait_tma_operand,
        epilogue,
        out_fp32,
        prefetch_w_k_steps,
        physical_m,
        use_pdl,
    ).launch(
        grid=(n_ctas, cute.ceil_div(active_m, physical_m), split_k),
        cluster=(1, 1, split_k),
        block=(THREADS, 1, 1),
        stream=stream,
        use_pdl=use_pdl,
    )


def compile_projection(
    *,
    up: bool,
    tile_n: int,
    stages: int,
    split_k: int,
    physical_m: int = 64,
    use_pdl: bool,
    output_alignment: int = 128,
):
    """Compile a fixed geometry with symbolic tokens and caller-owned output."""
    if torch.cuda.get_device_capability() != (10, 3):
        raise ValueError("shared split-K requires an SM103 device")
    if type(up) is not bool or type(use_pdl) is not bool:
        raise ValueError("up and use_pdl must be bool")
    if type(physical_m) is not int or physical_m not in (64, 128):
        raise ValueError("physical_m must be 64 or 128")
    if type(tile_n) is not int or tile_n not in (64, 128, 256):
        raise ValueError("tile_n must be 64, 128 or 256")
    if type(split_k) is not int or split_k not in (1, 2, 4, 8):
        raise ValueError("split_k must be 1, 2, 4 or 8")
    if type(stages) is not int or stages < 1:
        raise ValueError("stages must be positive")
    if type(output_alignment) is not int or output_alignment not in (16, 32, 64, 128):
        raise ValueError("output_alignment must be 16, 32, 64 or 128")
    columns, reduction = (1024, 6144) if up else (6144, 512)
    config = ProjectionConfig(
        module_name="moe_prefill_shared_" + ("up" if up else "down") + "_splitk",
        m=64,
        n=columns,
        k=reduction,
        out_fp32=False,
        mma_n=tile_n,
        stages=stages,
        split_k=split_k,
        standalone_perf="experimental",
        prewait_tma_operand=PREWAIT_TMA_W,
        epilogue="swiglu_interleaved" if up else "linear",
    )
    mailbox_values = max(1, config.reduce_peers * physical_m * config.mailbox_stride)
    smem_bytes = (
        stages * (physical_m + tile_n) * MMA_K * 2 + mailbox_values * 4 + 8 * (2 * stages + 2) + 4
    )
    if smem_bytes > 227 * 1024:
        raise ValueError("shared split-K exceeds the B300 shared-memory budget")
    fake = cute.runtime.make_fake_compact_tensor
    tokens = cute.sym_int32()
    a = fake(cutlass.BFloat16, (tokens, reduction), stride_order=(1, 0), assumed_align=128)
    b = fake(cutlass.BFloat16, (columns, reduction), stride_order=(1, 0), assumed_align=128)
    c = fake(
        cutlass.BFloat16,
        (tokens, config.output_n),
        stride_order=(1, 0),
        assumed_align=output_alignment,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    shared_splitk_kernel.set_name_prefix(config.kernel_name_prefix, remove_cutlass_symbol=True)
    raw = cute.compile(
        launch,
        a,
        b,
        c,
        stream,
        stages=stages,
        split_k=split_k,
        local_k_steps=config.local_k_steps,
        n_cols=columns,
        output_n=config.output_n,
        k_total=reduction,
        mma_n=tile_n,
        n_ctas=config.n_ctas,
        tmem_subs=config.tmem_subs,
        tmem_sub_last=config.tmem_sub_last,
        reduce_peers=config.reduce_peers,
        reducers=config.reducers,
        chunk_cols=config.chunk,
        output_chunk_cols=config.output_chunk,
        tmem_cols_alloc=config.tmem_cols_alloc,
        mailbox_stride=config.mailbox_stride,
        mailbox_values=mailbox_values,
        reduce_payload_bytes=config.reduce_peers * physical_m * config.chunk * 4,
        store_plan=config.store_plan,
        prewait_tma_operand=PREWAIT_TMA_W,
        epilogue=config.epilogue,
        out_fp32=False,
        prefetch_w_k_steps=0,
        physical_m=physical_m,
        use_pdl=use_pdl,
        options="--enable-tvm-ffi --generate-line-info",
    )

    def compiled(a_tensor: torch.Tensor, b_tensor: torch.Tensor, c_tensor: torch.Tensor):
        _validate_arguments(
            a_tensor,
            b_tensor,
            c_tensor,
            columns=columns,
            reduction=reduction,
            fused_swiglu=up,
            output_alignment=output_alignment,
        )
        return raw(a_tensor, b_tensor, c_tensor)

    return compiled
