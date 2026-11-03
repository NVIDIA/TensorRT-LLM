# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA decode FMHA, H16, PV reading V from the K slots, on CuTe DSL primitives.

The kernel of ``self_attn_fmha_sparse_mla_h16_qk576_v512_topk2048_fp8_bf16`` (the "base"
kernel, since removed from that module: nothing served ran it; the module's docstring still
describes the ABI, the warp roles, the split-K reducers and the valid-prefix masking, all
unchanged here) with one change in how a Kv128 tile reaches the PV MMA, for 16 heads only:

V from the K slots
------------------
The base kernel gathers every tile twice: five K128 slices (latent 0..511 and the rope
512..575) for QK into slots 0..4 of a nine-slot SMEM ring, then the four latent slices again
as V into slots 5..8 for PV. Both gathers write the same bytes in the same layout (128-byte
rows, SWIZZLE_128B): the QK operand reads a slice K-major, the PV operand reads the same
slice MN-major. Here PV reads V straight from the tile's own K latent slots and the V gather
is gone (1088 -> 576 gathered bytes per KV row). To keep the next tile's K load overlapping
the current tile's PV, tiles alternate between two five-slot sets (ten slots, 160 KB):
tile ``t`` uses set ``t % 2``; QK(t) releases the set's rope slot and PV(t) its four latent
slots, and the gather warps load tile ``t + 1`` into the other set once tile ``t - 1``
released it. The MMAs read the same fp8 bytes with the same descriptors' layouts, so every
output is bitwise identical to the base kernel's for the same specialization; only the SMEM
ring (203 KB instead of 187 KB) and the producer schedule change.

Measured (B300, CUDA-graph replay of 8 calls on 8 pool layers, 2048 valid rows per query,
lengths 3000..6007): MTP2 (``max_sq`` 2) 32 sequences 28.2 -> 20.7 us, MTP5 32 sequences
84.0 -> 56.1, 24 sequences 55.6 -> 35.3, MTP3 32 sequences 45.1 -> 31.2. When
most rows hit L2 (short, overlapping selections) the kernel is latency-bound and the two
variants tie within ~5 %.

The handles of ``ops/attention/cute_dsl_sparse_mla_decode`` build this module (H16 is the
only head count that op serves). Split-K selection, workspaces, reducers and the host ABI
are the base module's.
"""

from __future__ import annotations

import operator

import cuda.bindings.driver as cuda_driver
import cutlass
import cutlass.cute as cute
import cutlass.experimental.cuda as cuda
import torch
from cutlass.experimental import primitives as prims

from ..common import compat48 as _compat48
from . import self_attn_fmha_sparse_mla_h16_qk576_v512_topk2048_fp8_bf16 as _base
from .self_attn_fmha_sparse_mla_h16_qk576_v512_topk2048_fp8_bf16 import (
    ACC_DTYPE,
    BAR_SM0,
    CTA_N,
    EB,
    GRAN,
    H16_LOGICAL_TO_PACKED_STATS,
    I_STAGE_ELTS,
    I_TILES_PER_STAGE,
    INDEX_KIND_BASE_WITH_LAYER_OFFSET,
    INDEX_KIND_GLOBAL,
    INDEX_KINDS,
    IO_DTYPE,
    K_HDIM_STAGES,
    KV_POOL_TOKENS_PER_BLOCK,
    KV_STAGE_BYTES,
    KV_STRIDE,
    LBO_K,
    LBO_V,
    M_TILE,
    MAX_CTAS_KV,
    N_LAT,
    N_V,
    NUM_GATHER_WARPS,
    NUM_KV_TILES,
    NUM_TMEM_COLS,
    OUT_DTYPE,
    PV_M,
    QK_K,
    QK_N,
    SBO_LAT,
    SBO_P,
    SBO_ROPE,
    SBO_V,
    ST_C,
    ST_I,
    ST_P,
    ST_S,
    SUPPORTED_MAX_SQ,
    THREADS,
    TMEM_O_OFF,
    TOPK,
    V_HDIM_STAGES,
    V_KSTEP,
    W_CORR0,
    W_LOAD_GATHER0,
    W_LOAD_KQ,
    W_MMA_PV,
    W_MMA_QK,
    L,
    MlaFmhaRuntimeShape,
    MlaFmhaSpecialization,
    PreparedMlaFmhaRunner,
    R,
    _add_layer_offset_vector4,
    _fmax,
    _fragment_max4,
    _fragment_sum4,
    _mask_scores_after_valid_prefix_h16,
    _reduce_cga_output,
    _reduce_gmem_output,
    _reduce_two_cta_output,
    _scale_scores_packed,
    _store_single_cta_output,
    _tma_gather4_cta1,
    _warpgroup_row_max,
    _warpgroup_row_sum,
)

__all__ = [
    "INDEX_KIND_BASE_WITH_LAYER_OFFSET",
    "INDEX_KIND_GLOBAL",
    "NUM_HEADS",
    "compile",
    "precompile_for_serving",
    "prepare_runner",
]

NUM_HEADS: int = 16
#: Two five-slot tile sets: tile t uses set t % 2 (K latent 0..3, rope 4).
NUM_KV_SETS: int = 2
N_KV_SLOTS: int = NUM_KV_SETS * K_HDIM_STAGES
_COMPILE_CACHE: dict[tuple[object, ...], object] = {}


@cute.kernel
def _fmha_sparse_mla_wide_kernel(
    tma_ql: cutlass.GridConstant[cuda.TensorMap],
    tma_qr: cutlass.GridConstant[cuda.TensorMap],
    tma_kv: cutlass.GridConstant[cuda.TensorMap],
    sparse_indices: cute.Tensor,
    o: cute.Tensor,
    cu_q_seqlens: cute.Tensor,
    sequence_length: cute.Tensor,
    partial_o_workspace: cute.Tensor,
    partial_stats_workspace: cute.Tensor,
    completion_counters: cute.Tensor,
    layer_offset: cutlass.Int32,
    scale_log2: cutlass.Float32,
    all_sparse_indices_valid: cutlass.Constexpr[bool],
    num_ctas_kv: cutlass.Constexpr[int],
    use_gmem_reduction: cutlass.Constexpr[bool],
    add_layer_offset: cutlass.Constexpr[bool],
) -> None:
    num_heads: cutlass.Constexpr[int] = NUM_HEADS
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.warp_idx()
    task_x, _, batch = cute.arch.block_idx()
    if cutlass.const_expr(use_gmem_reduction):
        cta = task_x % cutlass.Int32(num_ctas_kv)
    else:
        cta = cute.arch.block_idx_in_cluster()
    query_in_batch = task_x // cutlass.Int32(num_ctas_kv)

    index_buffer_stages: cutlass.Constexpr[int] = TOPK // I_STAGE_ELTS if num_ctas_kv == 1 else ST_I
    num_kt_per_cta: cutlass.Constexpr[int] = (NUM_KV_TILES + num_ctas_kv - 1) // num_ctas_kv
    used_index_stages: cutlass.Constexpr[int] = (
        num_kt_per_cta + I_TILES_PER_STAGE - 1
    ) // I_TILES_PER_STAGE
    needs_tail_mask: cutlass.Constexpr[bool] = (
        not all_sparse_indices_valid or NUM_KV_TILES % num_ctas_kv != 0
    )
    num_scale_groups: cutlass.Constexpr[int] = num_heads // 4
    q_repeats: cutlass.Constexpr[int] = num_heads // 8
    scale_alignment: cutlass.Constexpr[int] = 16
    # As the base kernel: four-way partial unrolling of the single-CTA KV loop.
    kv_loop_unroll: cutlass.Constexpr[int] = 4 if num_ctas_kv == 1 else 1

    # ---- SMEM ------------------------------------------------------------
    sQ = cutlass.Array(IO_DTYPE, num_heads * L, space=cutlass.AddressSpace.smem, alignment=128)
    sQr = cutlass.Array(IO_DTYPE, num_heads * R, space=cutlass.AddressSpace.smem, alignment=128)
    sKv = cutlass.Array(
        IO_DTYPE,
        N_KV_SLOTS * QK_N * QK_K,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    sparse_stage = cutlass.Array(
        cutlass.Int32,
        index_buffer_stages * I_STAGE_ELTS,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    reduction_scratch = cutlass.Array(
        cutlass.Int32, 1, space=cutlass.AddressSpace.smem, alignment=32
    )
    sP = cutlass.Array(
        IO_DTYPE,
        ST_P * QK_N * num_heads,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    softmax_max = cutlass.Array(
        ACC_DTYPE,
        ST_S * 4 * num_heads,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    softmax_sum = cutlass.Array(
        ACC_DTYPE,
        4 * num_heads,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    compact_stats = cutlass.Array(
        ACC_DTYPE,
        ST_C * num_heads * 3,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    partial_o = cutlass.Array(
        OUT_DTYPE,
        num_heads * L,
        space=cutlass.AddressSpace.smem,
        alignment=128,
    )
    partial_stats = cutlass.Array(
        ACC_DTYPE,
        num_heads * 2,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    q_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    kv_full = cutlass.Array(cutlass.Int64, N_KV_SLOTS, space=cutlass.AddressSpace.smem)
    kv_empty = cutlass.Array(cutlass.Int64, N_KV_SLOTS, space=cutlass.AddressSpace.smem)
    index_full = cutlass.Array(cutlass.Int64, index_buffer_stages, space=cutlass.AddressSpace.smem)
    s_full = cutlass.Array(cutlass.Int64, ST_S, space=cutlass.AddressSpace.smem)
    s_empty = cutlass.Array(cutlass.Int64, ST_S, space=cutlass.AddressSpace.smem)
    p_full = cutlass.Array(cutlass.Int64, ST_P, space=cutlass.AddressSpace.smem)
    p_empty = cutlass.Array(cutlass.Int64, ST_P, space=cutlass.AddressSpace.smem)
    c_full = cutlass.Array(cutlass.Int64, ST_C, space=cutlass.AddressSpace.smem)
    c_empty = cutlass.Array(cutlass.Int64, ST_C, space=cutlass.AddressSpace.smem)
    o_full = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    o_empty = cutlass.Array(cutlass.Int64, 1, space=cutlass.AddressSpace.smem)
    tmem_i32 = cutlass.Array(cutlass.Int32, 1, space=cutlass.AddressSpace.smem)
    query_bounds = cutlass.Array(cutlass.Int32, 3, space=cutlass.AddressSpace.smem, alignment=16)

    if warp_idx == 0:
        if prims.elect_sync():
            prims.mbarrier_init(q_full, 1)
            for i in cutlass.range_constexpr(N_KV_SLOTS):
                prims.mbarrier_init(kv_full.data_ptr() + i, 1)
                prims.mbarrier_init(kv_empty.data_ptr() + i, 1)
            for i in cutlass.range_constexpr(index_buffer_stages):
                prims.mbarrier_init(index_full.data_ptr() + i, 1)
            for i in cutlass.range_constexpr(ST_S):
                prims.mbarrier_init(s_full.data_ptr() + i, 1)
                prims.mbarrier_init(s_empty.data_ptr() + i, 1)
            for i in cutlass.range_constexpr(ST_P):
                prims.mbarrier_init(p_full.data_ptr() + i, 4)
                prims.mbarrier_init(p_empty.data_ptr() + i, 1)
            for i in cutlass.range_constexpr(ST_C):
                prims.mbarrier_init(c_full.data_ptr() + i, 1)
                prims.mbarrier_init(c_empty.data_ptr() + i, 4)
            prims.mbarrier_init(o_full, 1)
            prims.mbarrier_init(o_empty, 4)
    prims.fence_mbarrier_init()
    prims.barrier_cluster_arrive_relaxed()

    if warp_idx == W_MMA_PV:
        prims.tcgen05_alloc(tmem_i32, NUM_TMEM_COLS, group="cta_1")
    if warp_idx == W_LOAD_KQ:
        # Only W9 reads the predecessor-owned query bounds (PDL wait).
        if prims.elect_sync():
            prims.griddepcontrol(prims.GridDepAction.WAIT)
            query_bounds[0] = cu_q_seqlens[batch] // cutlass.Int32(num_heads)
            query_bounds[1] = cu_q_seqlens[batch + cutlass.Int32(1)] // cutlass.Int32(num_heads)
            query_bounds[2] = sequence_length[batch]
    prims.barrier_cluster_wait()
    prims.barrier_cta_sync(0)
    tmem_raw = tmem_i32.load()
    tmem_ptr = cutlass.inttoptr(tmem_raw, 6, cutlass.Float32)
    if warp_idx == W_MMA_PV:
        prims.tcgen05_relinquish_alloc_permit(group="cta_1")

    q_begin = query_bounds[0]
    q_end = query_bounds[1]
    if query_in_batch >= q_end - q_begin:
        prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
        prims.barrier_cta_sync(0)
        if warp_idx == W_MMA_PV:
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            prims.tcgen05_dealloc(tmem_ptr, NUM_TMEM_COLS, group="cta_1")
        if tidx == 0:
            prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
        prims.exit()
    packed_q = q_begin + query_in_batch
    valid_kv = query_bounds[2] - (q_end - q_begin) + query_in_batch + cutlass.Int32(1)
    if valid_kv < cutlass.Int32(0):
        valid_kv = cutlass.Int32(0)
    if valid_kv > cutlass.Int32(TOPK):
        valid_kv = cutlass.Int32(TOPK)
    if tidx < cutlass.Int32(ST_S * num_heads):
        softmax_max[tidx] = ACC_DTYPE(-3.0e38)
    prims.barrier_cta_sync(0)

    first_kv_tile = cta * cutlass.Int32(num_kt_per_cta)
    num_kt = cutlass.Int32(num_kt_per_cta)
    if cutlass.const_expr(not all_sparse_indices_valid):
        # As the base kernel's H16 path: repartition the runtime-valid Kv128
        # tiles across split CTAs; an empty split participant (num_kt == 0)
        # publishes the neutral partial through the unchanged reducer.
        valid_tiles = (valid_kv + cutlass.Int32(QK_N - 1)) // cutlass.Int32(QK_N)
        if valid_tiles < cutlass.Int32(1):
            valid_tiles = cutlass.Int32(1)
        if cutlass.const_expr(num_ctas_kv == 1):
            num_kt = valid_tiles
        else:
            if valid_kv <= cutlass.Int32(TOPK - QK_N):
                first_kv_tile = (cta * valid_tiles) // cutlass.Int32(num_ctas_kv)
                num_kt = ((cta + cutlass.Int32(1)) * valid_tiles) // cutlass.Int32(
                    num_ctas_kv
                ) - first_kv_tile
    empty_split_participants: cutlass.Constexpr[bool] = (
        not all_sparse_indices_valid and num_ctas_kv > 1
    )
    q_bytes = (num_heads * L + num_heads * R) * EB

    if (warp_idx >= cutlass.Int32(W_LOAD_GATHER0)) and (
        warp_idx < cutlass.Int32(W_LOAD_GATHER0 + NUM_GATHER_WARPS)
    ):
        prims.setmaxregister(80, prims.SetMaxRegisterAction.DECREASE)
    if (warp_idx >= cutlass.Int32(8)) and (warp_idx < cutlass.Int32(12)):
        prims.setmaxregister(80, prims.SetMaxRegisterAction.DECREASE)

    # =====================================================================
    # W9 - Q TMA plus sparse-offset staging (unchanged from the base kernel).
    # =====================================================================
    if warp_idx == W_LOAD_KQ:
        if prims.elect_sync():
            prims.mbarrier_arrive_expect_tx(q_full, q_bytes)
        if prims.elect_sync():
            for c in cutlass.range_constexpr(N_LAT):
                prims.cp_async_bulk_tensor_shared_cta_global(
                    sQ.data_ptr() + c * (num_heads * QK_K),
                    tma_ql.get_ptr(),
                    (c * QK_K, 0, packed_q),
                    q_full,
                )
            prims.cp_async_bulk_tensor_shared_cta_global(
                sQr,
                tma_qr.get_ptr(),
                (0, 0, packed_q),
                q_full,
            )

        sparse_ptr = sparse_indices.iterator.raw_ptr()
        lane = tidx % cutlass.Int32(32)
        query_base = packed_q * cutlass.Int32(TOPK) + first_kv_tile * cutlass.Int32(QK_N)
        for i_stage in cutlass.range_constexpr(used_index_stages):
            for half in cutlass.range_constexpr(I_STAGE_ELTS // (32 * 4)):
                row = cutlass.Int32(half * 32 * 4) + lane * cutlass.Int32(4)
                if cutlass.const_expr(needs_tail_mask):
                    values = cutlass.Vector.from_elements(
                        tuple(cutlass.Int32(-1) for _ in range(4)),
                        dtype=cutlass.Int32,
                    )
                    sparse_offset = (
                        first_kv_tile * cutlass.Int32(QK_N)
                        + cutlass.Int32(i_stage * I_STAGE_ELTS)
                        + row
                    )
                    if sparse_offset < cutlass.Int32(TOPK):
                        values = (
                            sparse_ptr + query_base + cutlass.Int32(i_stage * I_STAGE_ELTS) + row
                        ).load(count=4, alignment=16)
                else:
                    values = (
                        sparse_ptr + query_base + cutlass.Int32(i_stage * I_STAGE_ELTS) + row
                    ).load(count=4, alignment=16)
                if cutlass.const_expr(add_layer_offset):
                    values = _add_layer_offset_vector4(values, layer_offset)
                (sparse_stage.data_ptr() + cutlass.Int32(i_stage * I_STAGE_ELTS) + row).store(
                    values, alignment=16
                )
            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
            if prims.elect_sync():
                prims.mbarrier_arrive(index_full.data_ptr() + i_stage, relaxed=True)

    # =====================================================================
    # W12-W15 - sparse gather4 producers: K only, into the tile's set.
    # =====================================================================
    if (warp_idx >= cutlass.Int32(W_LOAD_GATHER0)) and (
        warp_idx < cutlass.Int32(W_LOAD_GATHER0 + NUM_GATHER_WARPS)
    ):
        producer_rank = warp_idx - cutlass.Int32(W_LOAD_GATHER0)
        routed_rows = cutlass.Array(cutlass.Int32, 32, space=cutlass.AddressSpace.rmem)
        # Tile `t` goes to set `t % 2`; its slots were last used by tile `t - 2`
        # (use `t // 2`), whose PV released the latent slots and QK the rope slot.
        # An empty split participant (num_kt == 0) loads nothing.
        for kt in cutlass.range(0, num_kt, 1, unroll=1):
            idx_stage = kt // cutlass.Int32(I_TILES_PER_STAGE)
            while not prims.mbarrier_try_wait_parity(index_full.data_ptr() + idx_stage, 0):
                pass
            staged_base = (
                idx_stage * cutlass.Int32(I_STAGE_ELTS)
                + (kt % cutlass.Int32(I_TILES_PER_STAGE)) * cutlass.Int32(QK_N)
                + producer_rank * cutlass.Int32(CTA_N // NUM_GATHER_WARPS)
            )
            for g in cutlass.range_constexpr(CTA_N // (NUM_GATHER_WARPS * 4)):
                values = (sparse_stage.data_ptr() + staged_base + cutlass.Int32(g * 4)).load(
                    count=4, alignment=16
                )
                for j in cutlass.range_constexpr(4):
                    routed_rows[g * 4 + j] = values[j]
            set_base = (kt % cutlass.Int32(NUM_KV_SETS)) * cutlass.Int32(K_HDIM_STAGES)
            empty_phase = ((kt // cutlass.Int32(NUM_KV_SETS)) % cutlass.Int32(2)) ^ cutlass.Int32(1)
            for c in cutlass.range_constexpr(K_HDIM_STAGES):
                slot = set_base + cutlass.Int32(c)
                while not prims.mbarrier_try_wait_parity(kv_empty.data_ptr() + slot, empty_phase):
                    pass
                if producer_rank == cutlass.Int32(0):
                    if prims.elect_sync():
                        prims.mbarrier_arrive_expect_tx(kv_full.data_ptr() + slot, KV_STAGE_BYTES)
                for g in cutlass.range_constexpr(CTA_N // (NUM_GATHER_WARPS * 4)):
                    _tma_gather4_cta1(
                        sKv.data_ptr()
                        + slot * cutlass.Int32(CTA_N * QK_K)
                        + (
                            producer_rank * cutlass.Int32(CTA_N // NUM_GATHER_WARPS)
                            + cutlass.Int32(g * 4)
                        )
                        * cutlass.Int32(QK_K),
                        tma_kv.get_ptr(),
                        kv_full.data_ptr() + slot,
                        cutlass.Int32(c * QK_K),
                        routed_rows[g * 4],
                        routed_rows[g * 4 + 1],
                        routed_rows[g * 4 + 2],
                        routed_rows[g * 4 + 3],
                    )

    # =====================================================================
    # W8 - CTA1 QK over the tile's set; releases the rope slot.
    # =====================================================================
    if warp_idx == W_MMA_QK:
        idesc_qk = prims.Tcgen05InstrDesc.build(
            c_dtype=ACC_DTYPE,
            a_dtype=IO_DTYPE,
            b_dtype=IO_DTYPE,
            n_dim=num_heads,
            m_dim=M_TILE,
        )
        while not prims.mbarrier_try_wait_parity(q_full, 0):
            pass
        dq = _compat48.tcgen05_smem_desc(
            start_address=sQ,
            leading_byte_offset=LBO_K,
            stride_byte_offset=SBO_LAT,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        dqr = _compat48.tcgen05_smem_desc(
            start_address=sQr,
            leading_byte_offset=LBO_K,
            stride_byte_offset=SBO_ROPE,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_64B,
        )
        dk_root = _compat48.tcgen05_smem_desc(
            start_address=sKv,
            leading_byte_offset=LBO_K,
            stride_byte_offset=SBO_LAT,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        sst = cutlass.Int32(0)
        sph = cutlass.Int32(0)
        for kt in cutlass.range(0, num_kt, 1, unroll=kv_loop_unroll):
            while not prims.mbarrier_try_wait_parity(
                s_empty.data_ptr() + sst, sph ^ cutlass.Int32(1)
            ):
                pass
            set_base = (kt % cutlass.Int32(NUM_KV_SETS)) * cutlass.Int32(K_HDIM_STAGES)
            kvph = (kt // cutlass.Int32(NUM_KV_SETS)) % cutlass.Int32(2)
            stmem = cutlass.inttoptr(
                tmem_raw + sst * cutlass.Int32(num_heads),
                6,
                cutlass.Float32,
            )
            for c in cutlass.range_constexpr(K_HDIM_STAGES):
                slot = set_base + cutlass.Int32(c)
                while not prims.mbarrier_try_wait_parity(kv_full.data_ptr() + slot, kvph):
                    pass
                ko = slot * cutlass.Int32(CTA_N * QK_K * EB)
                if cutlass.const_expr(c < N_LAT):
                    qo: cutlass.Constexpr[int] = c * num_heads * QK_K * EB
                    for kb in cutlass.range_constexpr(QK_K // GRAN):
                        g: cutlass.Constexpr[int] = kb * GRAN * EB
                        if prims.elect_sync():
                            prims.tcgen05_mma(
                                prims.Tcgen05MMAKind.F8F6F4,
                                prims.CTAGroup.CTA_1,
                                stmem,
                                dk_root.advance_start_address(ko + g),
                                dq.advance_start_address(qo + g),
                                idesc_qk,
                                (c != 0) or (kb != 0),
                            )
                else:
                    for kb in cutlass.range_constexpr(R // GRAN):
                        g: cutlass.Constexpr[int] = kb * GRAN * EB
                        if prims.elect_sync():
                            prims.tcgen05_mma(
                                prims.Tcgen05MMAKind.F8F6F4,
                                prims.CTAGroup.CTA_1,
                                stmem,
                                dk_root.advance_start_address(ko + g),
                                dqr.advance_start_address(g),
                                idesc_qk,
                                True,
                            )
                    # The rope slot is QK's to release; PV releases the latent ones.
                    # Slots of tile t are reused by tile t + 2 only.
                    if prims.elect_sync():
                        if kt + cutlass.Int32(NUM_KV_SETS) < num_kt:
                            prims.tcgen05_commit(
                                kv_empty.data_ptr() + slot,
                                group=prims.CTAGroup.CTA_1,
                            )
            if prims.elect_sync():
                prims.tcgen05_commit(
                    s_full.data_ptr() + sst,
                    group=prims.CTAGroup.CTA_1,
                )
            sst += cutlass.Int32(1)
            if sst == cutlass.Int32(ST_S):
                sst = cutlass.Int32(0)
                sph ^= cutlass.Int32(1)

    # =====================================================================
    # W11 - CTA1 SwapsAb VP; V stripe s is the tile's K latent slot s.
    # =====================================================================
    if warp_idx == W_MMA_PV:
        idesc_pv = prims.Tcgen05InstrDesc.build(
            c_dtype=ACC_DTYPE,
            a_dtype=IO_DTYPE,
            b_dtype=IO_DTYPE,
            n_dim=num_heads,
            m_dim=PV_M,
            a_major=1,
            b_major=0,
        )
        dp_root = _compat48.tcgen05_smem_desc(
            start_address=sP,
            leading_byte_offset=16,
            stride_byte_offset=SBO_P,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        dv_root = _compat48.tcgen05_smem_desc(
            start_address=sKv,
            leading_byte_offset=LBO_V,
            stride_byte_offset=SBO_V,
            layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
        )
        pst = cutlass.Int32(0)
        pph = cutlass.Int32(0)
        for kt in cutlass.range(0, num_kt, 1, unroll=kv_loop_unroll):
            while not prims.mbarrier_try_wait_parity(p_full.data_ptr() + pst, pph):
                pass
            oph = kt % cutlass.Int32(2)
            while not prims.mbarrier_try_wait_parity(o_empty, oph ^ cutlass.Int32(1)):
                pass
            pbase = pst * cutlass.Int32(QK_N * num_heads)
            set_base = (kt % cutlass.Int32(NUM_KV_SETS)) * cutlass.Int32(K_HDIM_STAGES)
            kvph = (kt // cutlass.Int32(NUM_KV_SETS)) % cutlass.Int32(2)
            for s in cutlass.range_constexpr(V_HDIM_STAGES):
                slot = set_base + cutlass.Int32(s)
                # Complete since QK(kt) waited it; orders this warp's MMA reads.
                while not prims.mbarrier_try_wait_parity(kv_full.data_ptr() + slot, kvph):
                    pass
                vso = slot * cutlass.Int32(QK_N * QK_K * EB)
                otp = cutlass.inttoptr(
                    tmem_raw + TMEM_O_OFF + s * num_heads,
                    6,
                    cutlass.Float32,
                )
                for kb in cutlass.range_constexpr(QK_N // GRAN):
                    po: cutlass.Constexpr[int] = kb * GRAN * EB
                    vo: cutlass.Constexpr[int] = kb * V_KSTEP
                    if prims.elect_sync():
                        prims.tcgen05_mma(
                            prims.Tcgen05MMAKind.F8F6F4,
                            prims.CTAGroup.CTA_1,
                            otp,
                            dv_root.advance_start_address(vso + vo),
                            dp_root.advance_start_address(pbase + po),
                            idesc_pv,
                            (kt > 0) or (kb != 0),
                        )
                if prims.elect_sync():
                    if kt + cutlass.Int32(NUM_KV_SETS) < num_kt:
                        prims.tcgen05_commit(
                            kv_empty.data_ptr() + slot,
                            group=prims.CTAGroup.CTA_1,
                        )
            if prims.elect_sync():
                prims.tcgen05_commit(o_full, group=prims.CTAGroup.CTA_1)
            if prims.elect_sync():
                if kt + cutlass.Int32(ST_P) < num_kt:
                    prims.tcgen05_commit(
                        p_empty.data_ptr() + pst,
                        group=prims.CTAGroup.CTA_1,
                    )
            pst += cutlass.Int32(1)
            if pst == cutlass.Int32(ST_P):
                pst = cutlass.Int32(0)
                pph ^= cutlass.Int32(1)

    # =====================================================================
    # W0-W3 - online softmax (the base kernel's H16 path).
    # =====================================================================
    if warp_idx < cutlass.Int32(4):
        prims.setmaxregister(152, prims.SetMaxRegisterAction.INCREASE)
        lane = tidx % cutlass.Int32(32)
        col_group = lane % cutlass.Int32(4)
        row_idx = lane // cutlass.Int32(4)
        local_row_idx = row_idx % cutlass.Int32(4)
        base = prims.TmemAddr(tmem_raw)
        row0 = base.row_id + warp_idx * cutlass.Int32(32)
        addr0 = (row0 << cutlass.Int32(16)) | base.col_id
        addr1 = ((row0 + cutlass.Int32(16)) << cutlass.Int32(16)) | base.col_id
        prev_max = cutlass.Vector.from_elements(
            tuple(ACC_DTYPE(-3.0e38) for _ in range(num_scale_groups)),
            dtype=ACC_DTYPE,
        )
        prev_sum = cutlass.Vector.from_elements(
            tuple(ACC_DTYPE(0.0) for _ in range(num_scale_groups)), dtype=ACC_DTYPE
        )
        stage = cutlass.Int32(0)
        phase = cutlass.Int32(0)

        for kt in cutlass.range(0, num_kt, 1, unroll=kv_loop_unroll):
            while not prims.mbarrier_try_wait_parity(s_full.data_ptr() + stage, phase):
                pass
            frag0 = prims.tcgen05_ld(
                "16x256b",
                cutlass.inttoptr(addr0 + stage * cutlass.Int32(num_heads), 6, ACC_DTYPE),
                num=q_repeats,
            )
            frag1 = prims.tcgen05_ld(
                "16x256b",
                cutlass.inttoptr(addr1 + stage * cutlass.Int32(num_heads), 6, ACC_DTYPE),
                num=q_repeats,
            )
            prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
            scores = cutlass.Vector.from_elements(
                tuple(frag0[i] for i in range(4 * q_repeats))
                + tuple(frag1[i] for i in range(4 * q_repeats)),
                dtype=ACC_DTYPE,
            )
            if cutlass.const_expr(needs_tail_mask):
                scores = _mask_scores_after_valid_prefix_h16(
                    scores,
                    (first_kv_tile + kt) * cutlass.Int32(QK_N),
                    valid_kv,
                    warp_idx,
                    row_idx,
                    num_heads,
                )
            thread_max = _fragment_max4(scores)
            reduced_max = cutlass.Vector.from_elements(
                tuple(_warpgroup_row_max(thread_max[i]) for i in range(4)),
                dtype=ACC_DTYPE,
            )
            atomic_value = reduced_max[0]
            if local_row_idx == cutlass.Int32(1):
                atomic_value = reduced_max[1]
            elif local_row_idx == cutlass.Int32(2):
                atomic_value = reduced_max[2]
            elif local_row_idx == cutlass.Int32(3):
                atomic_value = reduced_max[3]
            atomic_value = _fmax(
                atomic_value,
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=atomic_value,
                    offset=16,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.BFLY,
                ),
            )
            if lane < cutlass.Int32(16):
                max_head = col_group * cutlass.Int32(4) + local_row_idx
                max_slot = (
                    stage * cutlass.Int32(4 * num_heads)
                    + warp_idx * cutlass.Int32(num_heads)
                    + max_head
                )
                softmax_max[max_slot] = atomic_value
            prims.barrier_cta_sync(BAR_SM0, thread_count=128)
            max_group = col_group * cutlass.Int32(num_scale_groups)
            max_stage = stage * cutlass.Int32(4 * num_heads)
            tile_max = (softmax_max.data_ptr() + max_stage + max_group).load(
                count=num_scale_groups, alignment=scale_alignment
            )
            for max_warp in cutlass.range_constexpr(1, 4):
                tile_max = cute.math.max(
                    tile_max,
                    (softmax_max.data_ptr() + max_stage + max_warp * num_heads + max_group).load(
                        count=num_scale_groups, alignment=scale_alignment
                    ),
                )
            if (warp_idx == cutlass.Int32(0)) and (lane == cutlass.Int32(0)):
                prims.mbarrier_arrive(s_empty.data_ptr() + stage, relaxed=True)

            while not prims.mbarrier_try_wait_parity(
                c_empty.data_ptr() + stage, phase ^ cutlass.Int32(1)
            ):
                pass
            while not prims.mbarrier_try_wait_parity(
                p_empty.data_ptr() + stage, phase ^ cutlass.Int32(1)
            ):
                pass

            new_max = cute.math.max(prev_max, tile_max)
            corr = cute.math.exp2((prev_max - new_max) * scale_log2, fastmath=True)
            scaled = _scale_scores_packed(scores, new_max, scale_log2, num_heads)
            probs = cute.math.exp2(scaled, fastmath=True)
            thread_sum = _fragment_sum4(probs)
            warp_sum = cutlass.Vector.from_elements(
                tuple(_warpgroup_row_sum(thread_sum[i]) for i in range(4)),
                dtype=ACC_DTYPE,
            )
            if lane < cutlass.Int32(4):
                (
                    softmax_sum.data_ptr()
                    + warp_idx * cutlass.Int32(num_heads)
                    + lane * cutlass.Int32(num_scale_groups)
                ).store(warp_sum, alignment=scale_alignment)
            # Reused max stages need all consumers to finish before reset.
            prims.barrier_cta_sync(BAR_SM0, thread_count=128)
            sum_base = col_group * cutlass.Int32(num_scale_groups)
            tile_sum = (softmax_sum.data_ptr() + sum_base).load(
                count=num_scale_groups, alignment=scale_alignment
            )
            for sum_warp in cutlass.range_constexpr(1, 4):
                tile_sum += (
                    softmax_sum.data_ptr() + sum_warp * cutlass.Int32(num_heads) + sum_base
                ).load(count=num_scale_groups, alignment=scale_alignment)
            prev_sum = prev_sum * corr + tile_sum
            prev_max = new_max

            stat_base = stage * cutlass.Int32(num_heads * 3)
            head_base = lane * cutlass.Int32(num_scale_groups)
            smem_swizzle = cutlass.Swizzle.from_name("s128b")
            for half in cutlass.range_constexpr(2):
                half_probs = cutlass.Vector.from_elements(
                    tuple(probs[half * 8 + i] for i in range(8)),
                    dtype=ACC_DTYPE,
                )
                packed_p = half_probs.to(IO_DTYPE).bitcast(cutlass.Int32)
                address_row = lane % cutlass.Int32(8)
                address_segment = (lane // cutlass.Int32(8)) % cutlass.Int32(2)
                row_start = (
                    stage * cutlass.Int32(num_heads * QK_N)
                    + (address_segment * cutlass.Int32(8) + address_row) * cutlass.Int32(QK_N)
                    + (warp_idx * cutlass.Int32(2) + cutlass.Int32(half)) * cutlass.Int32(16)
                )
                prims.stmatrix(
                    (sP.data_ptr() + row_start).apply_swizzle(smem_swizzle),
                    [packed_p[0], packed_p[1]],
                    prims.MMALayout.COL,
                    shape=prims.StoreShape.M16N8,
                )
            prims.fence_proxy(prims.Proxy.ASYNC_SHARED, space=prims.SharedSpace.shared_cta)
            if prims.elect_sync():
                prims.mbarrier_arrive(p_full.data_ptr() + stage)
            if (warp_idx == cutlass.Int32(0)) and (lane < cutlass.Int32(4)):
                (compact_stats.data_ptr() + stat_base + head_base).store(
                    prev_sum, alignment=scale_alignment
                )
                (compact_stats.data_ptr() + stat_base + cutlass.Int32(num_heads) + head_base).store(
                    prev_max, alignment=scale_alignment
                )
                (
                    compact_stats.data_ptr() + stat_base + cutlass.Int32(2 * num_heads) + head_base
                ).store(corr, alignment=scale_alignment)
            if (warp_idx == cutlass.Int32(0)) and (lane == cutlass.Int32(0)):
                prims.mbarrier_arrive(c_full.data_ptr() + stage)
            stage += cutlass.Int32(1)
            if stage == cutlass.Int32(ST_S):
                stage = cutlass.Int32(0)
                phase ^= cutlass.Int32(1)

    # =====================================================================
    # W4-7 - correction / rescale / epilogue (the base kernel's H16 path).
    # =====================================================================
    if (warp_idx >= cutlass.Int32(W_CORR0)) and (warp_idx < cutlass.Int32(W_CORR0 + 4)):
        prims.setmaxregister(176, prims.SetMaxRegisterAction.INCREASE)
        lane = tidx % cutlass.Int32(32)
        corr_warp = warp_idx - cutlass.Int32(W_CORR0)
        out_row = corr_warp * cutlass.Int32(32) + lane
        base = prims.TmemAddr(tmem_raw)
        my_addr = ((base.row_id + corr_warp * cutlass.Int32(32)) << 16) | base.col_id
        r_sum = cutlass.Vector.from_elements(
            tuple(ACC_DTYPE(0.0) for _ in range(num_heads)), dtype=ACC_DTYPE
        )
        r_max = cutlass.Vector.from_elements(
            tuple(ACC_DTYPE(-3.0e38) for _ in range(num_heads)), dtype=ACC_DTYPE
        )
        cst = cutlass.Int32(0)
        cph = cutlass.Int32(0)
        for kt in cutlass.range(0, num_kt, 1, unroll=kv_loop_unroll):
            while not prims.mbarrier_try_wait_parity(c_full.data_ptr() + cst, cph):
                pass
            stat_base = cst * cutlass.Int32(num_heads * 3)
            packed_sum = (compact_stats.data_ptr() + stat_base).load(count=num_heads, alignment=16)
            r_sum = cutlass.Vector.from_elements(
                tuple(packed_sum[packed_head] for packed_head in H16_LOGICAL_TO_PACKED_STATS),
                dtype=ACC_DTYPE,
            )
            packed_max = (compact_stats.data_ptr() + stat_base + num_heads).load(
                count=num_heads, alignment=16
            )
            r_max = cutlass.Vector.from_elements(
                tuple(packed_max[packed_head] for packed_head in H16_LOGICAL_TO_PACKED_STATS),
                dtype=ACC_DTYPE,
            )
            packed_corr = (compact_stats.data_ptr() + stat_base + 2 * num_heads).load(
                count=num_heads, alignment=16
            )
            corr = cutlass.Vector.from_elements(
                tuple(packed_corr[packed_head] for packed_head in H16_LOGICAL_TO_PACKED_STATS),
                dtype=ACC_DTYPE,
            )
            if prims.elect_sync():
                prims.mbarrier_arrive(c_empty.data_ptr() + cst, relaxed=True)

            oph = kt % cutlass.Int32(2)
            if kt > cutlass.Int32(0):
                while not prims.mbarrier_try_wait_parity(o_full, oph ^ cutlass.Int32(1)):
                    pass
                for s in cutlass.range_constexpr(N_V):
                    tp = cutlass.inttoptr(my_addr + TMEM_O_OFF + s * num_heads, 6, ACC_DTYPE)
                    tile_v = prims.tcgen05_ld("32x32b", tp, num=num_heads)
                    prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                    prims.tcgen05_st("32x32b", tp, tile_v * corr)
                prims.tcgen05_wait(prims.Tcgen05Wait.STORE)
                prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
                if prims.elect_sync():
                    prims.mbarrier_arrive(o_empty, relaxed=True)
            cst += cutlass.Int32(1)
            if cst == cutlass.Int32(ST_C):
                cst = cutlass.Int32(0)
                cph ^= cutlass.Int32(1)

        finalize_tiles = cutlass.Int32(1)
        if cutlass.const_expr(empty_split_participants):
            finalize_tiles = num_kt
            if num_kt == cutlass.Int32(0):
                # Empty split participant: publish the neutral partial
                # (O = 0, max = -3.0e38, sum = 0) through the reducer.
                for s in cutlass.range_constexpr(N_V):
                    out_col = s * PV_M + out_row
                    for head in cutlass.range_constexpr(num_heads):
                        partial_o[head * cutlass.Int32(L) + out_col] = OUT_DTYPE(0.0)
                if (corr_warp == cutlass.Int32(0)) and (lane < cutlass.Int32(num_heads)):
                    partial_stats[lane * cutlass.Int32(2)] = ACC_DTYPE(-3.0e38)
                    partial_stats[lane * cutlass.Int32(2) + cutlass.Int32(1)] = ACC_DTYPE(0.0)
        if finalize_tiles > cutlass.Int32(0):
            final_phase = (num_kt - cutlass.Int32(1)) % cutlass.Int32(2)
            while not prims.mbarrier_try_wait_parity(o_full, final_phase):
                pass
            if cutlass.const_expr(needs_tail_mask):
                safe_sum = cutlass.Vector.from_elements(
                    tuple(_fmax(r_sum[head], ACC_DTYPE(1.0e-30)) for head in range(num_heads)),
                    dtype=ACC_DTYPE,
                )
                inv = ACC_DTYPE(1.0) / safe_sum
            else:
                inv = ACC_DTYPE(1.0) / r_sum
            for s in cutlass.range_constexpr(N_V):
                tp = cutlass.inttoptr(my_addr + TMEM_O_OFF + s * num_heads, 6, ACC_DTYPE)
                tile_v = prims.tcgen05_ld("32x32b", tp, num=num_heads)
                prims.tcgen05_wait(prims.Tcgen05Wait.LOAD)
                tile_v = tile_v * inv
                out_col = s * PV_M + out_row
                for head in cutlass.range_constexpr(num_heads):
                    partial_o[head * cutlass.Int32(L) + out_col] = OUT_DTYPE(tile_v[head])
            if (corr_warp == cutlass.Int32(0)) and (lane < cutlass.Int32(num_heads)):
                partial_stats[lane * cutlass.Int32(2)] = r_max[lane]
                partial_stats[lane * cutlass.Int32(2) + cutlass.Int32(1)] = r_sum[lane]

    prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
    if cutlass.const_expr(num_ctas_kv == 1 or use_gmem_reduction):
        prims.barrier_cta_sync(0)
        if warp_idx == W_MMA_PV:
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            prims.tcgen05_dealloc(tmem_ptr, NUM_TMEM_COLS, group="cta_1")

    if cutlass.const_expr(use_gmem_reduction):
        _reduce_gmem_output(
            partial_o,
            partial_stats,
            partial_o_workspace,
            partial_stats_workspace,
            completion_counters,
            reduction_scratch,
            o,
            tidx,
            packed_q,
            cta,
            scale_log2,
            num_heads,
            num_ctas_kv,
        )
    elif cutlass.const_expr(num_ctas_kv == 1):
        _store_single_cta_output(partial_o, o, tidx, packed_q, num_heads)
    elif cutlass.const_expr(num_ctas_kv == 2):
        _reduce_two_cta_output(
            partial_o,
            partial_stats,
            o,
            tidx,
            packed_q,
            cta,
            scale_log2,
            num_heads,
        )
    else:
        _reduce_cga_output(
            partial_o,
            partial_stats,
            o,
            tidx,
            packed_q,
            cta,
            scale_log2,
            num_heads,
            num_ctas_kv,
        )
    prims.barrier_cta_sync(0)
    if cutlass.const_expr(num_ctas_kv > 1 and not use_gmem_reduction):
        if warp_idx == W_MMA_PV:
            prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)
            prims.tcgen05_dealloc(tmem_ptr, NUM_TMEM_COLS, group="cta_1")
    if tidx == 0:
        prims.griddepcontrol(prims.GridDepAction.LAUNCH_DEPENDENTS)
    if cutlass.const_expr(num_ctas_kv > 1 and not use_gmem_reduction):
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()


@cute.jit
def _launch_fmha_sparse_mla_wide(
    q_storage: cute.Tensor,
    kv_storage: cute.Tensor,
    sparse_indices: cute.Tensor,
    output: cute.Tensor,
    cu_q_seqlens: cute.Tensor,
    sequence_length: cute.Tensor,
    partial_o_workspace: cute.Tensor,
    partial_stats_workspace: cute.Tensor,
    completion_counters: cute.Tensor,
    layer_offset: cutlass.Int32,
    max_sq,
    stream: cuda_driver.CUstream,
    scale_log2: cutlass.Constexpr[float],
    all_sparse_indices_valid: cutlass.Constexpr[bool],
    num_ctas_kv: cutlass.Constexpr[int],
    use_gmem_reduction: cutlass.Constexpr[bool],
    add_layer_offset: cutlass.Constexpr[bool],
) -> None:
    num_heads = NUM_HEADS

    def q_tensor_map(offset, inner, box_inner, swizzle):
        return cuda.create_tensor_map_tiled(
            global_address=(q_storage.iterator + offset).toint(),
            dtype=IO_DTYPE,
            global_dims=[inner, num_heads, q_storage.shape[0]],
            global_strides=[
                KV_STRIDE * EB // 16,
                num_heads * KV_STRIDE * EB // 16,
            ],
            box_dims=[box_inner, num_heads, 1],
            swizzle=swizzle,
        )

    tma_ql = q_tensor_map(0, L, QK_K, cuda.TensorMapSwizzle.s128b)
    tma_qr = q_tensor_map(L, R, R, cuda.TensorMapSwizzle.s64b)
    tma_kv = cuda.create_tensor_map_tiled(
        global_address=kv_storage.iterator.toint(),
        dtype=IO_DTYPE,
        global_dims=[KV_STRIDE, kv_storage.shape[0]],
        global_strides=[KV_STRIDE * EB // 16],
        box_dims=[QK_K, 1],
        swizzle=cuda.TensorMapSwizzle.s128b,
    )
    _fmha_sparse_mla_wide_kernel(
        tma_ql,
        tma_qr,
        tma_kv,
        sparse_indices,
        output,
        cu_q_seqlens,
        sequence_length,
        partial_o_workspace,
        partial_stats_workspace,
        completion_counters,
        layer_offset,
        cutlass.Float32(scale_log2),
        all_sparse_indices_valid,
        num_ctas_kv,
        use_gmem_reduction,
        add_layer_offset,
    ).launch(
        grid=(num_ctas_kv * max_sq, 1, cu_q_seqlens.shape[0] - 1),
        block=(THREADS, 1, 1),
        cluster=(1 if use_gmem_reduction else num_ctas_kv, 1, 1),
        stream=stream,
        min_blocks_per_mp=1,
        use_pdl=True,
    )


def _compile_for(specialization: MlaFmhaSpecialization, device_index: int) -> object:
    specialization.validate()
    if specialization.num_heads != NUM_HEADS:
        raise ValueError(f"the wide kernel is H{NUM_HEADS} only, got {specialization.num_heads}")
    key = (
        device_index,
        specialization.all_sparse_indices_valid,
        specialization.q_scaling,
        specialization.num_ctas_kv,
        specialization.use_gmem_reduction,
        specialization.index_kind,
    )
    cached = _COMPILE_CACHE.get(key)
    if cached is not None:
        return cached
    cutlass.cuda.initialize_cuda_context(device_index)
    compiled = cute.compile(
        _launch_fmha_sparse_mla_wide,
        *_base._fake_tensors(
            NUM_HEADS,
            specialization.num_ctas_kv,
            specialization.use_gmem_reduction,
        ),
        cutlass.Int32(0),
        cute.sym_int32(),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=False),
        specialization.scale_log2,
        specialization.all_sparse_indices_valid,
        specialization.num_ctas_kv,
        specialization.use_gmem_reduction,
        specialization.add_layer_offset,
        options=_base._compile_options(),
    )
    _COMPILE_CACHE[key] = compiled
    return compiled


def compile(  # noqa: A001
    *,
    all_sparse_indices_valid: bool = True,
    num_heads: int = NUM_HEADS,
    q_scaling: float = 1.0,
    num_ctas_kv: int = 2,
    use_gmem_reduction: bool = False,
    device_index: int = 0,
    index_kind: str = INDEX_KIND_GLOBAL,
) -> object:
    """Compile one specialization (the base module's arguments; H16 only)."""
    return _compile_for(
        MlaFmhaSpecialization(
            all_sparse_indices_valid=all_sparse_indices_valid,
            num_heads=num_heads,
            q_scaling=q_scaling,
            num_ctas_kv=num_ctas_kv,
            use_gmem_reduction=use_gmem_reduction,
            index_kind=index_kind,
        ),
        device_index,
    )


def precompile_for_serving(
    *,
    max_batch_size: int,
    all_sparse_indices_valid: bool,
    num_heads: int = NUM_HEADS,
    q_scaling: float = 1.0,
    max_sq: int = _base.DEFAULT_MAX_SQ,
    max_seq_len_kv: int = TOPK,
    device_index: int = 0,
    index_kind: str = INDEX_KIND_GLOBAL,
) -> tuple[MlaFmhaSpecialization, ...]:
    """The base module's walk over batch sizes 1..`max_batch_size` (the same split-K and
    reducer picks), compiling this module's kernel for every distinct pick."""
    if max_batch_size <= 0:
        raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")
    if max_sq not in SUPPORTED_MAX_SQ:
        raise ValueError(f"max_sq must be one of {SUPPORTED_MAX_SQ}, got {max_sq}")
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    specializations: list[MlaFmhaSpecialization] = []
    for batch_size in range(1, max_batch_size + 1):
        num_ctas_kv = _base._select_num_ctas_kv(
            batch_size=batch_size,
            max_sq=max_sq,
            max_seq_len_kv=max_seq_len_kv,
            multiprocessor_count=sms,
        )
        use_gmem_reduction = num_ctas_kv > 1 and _base._select_gmem_reduction(
            batch_size=batch_size,
            max_sq=max_sq,
            max_active_clusters=_base._max_active_clusters(device_index, num_ctas_kv),
        )
        specialization = MlaFmhaSpecialization(
            all_sparse_indices_valid=all_sparse_indices_valid,
            num_heads=num_heads,
            q_scaling=q_scaling,
            num_ctas_kv=num_ctas_kv,
            use_gmem_reduction=use_gmem_reduction,
            index_kind=index_kind,
        )
        if specialization not in specializations:
            specializations.append(specialization)
    for specialization in specializations:
        _compile_for(specialization, device_index)
    return tuple(specializations)


def prepare_runner(
    quant_q_buffer: torch.Tensor,
    kv_pool_view: torch.Tensor,
    sparse_attn_indices: torch.Tensor,
    output: torch.Tensor,
    cu_q_seqlens: torch.Tensor,
    sequence_length: torch.Tensor,
    *,
    mla_bmm1_scale: torch.Tensor | None = None,
    mla_bmm2_scale: torch.Tensor | None = None,
    all_sparse_indices_valid: bool,
    q_scaling: float = 1.0,
    max_sq: int = _base.DEFAULT_MAX_SQ,
    num_ctas_kv: int | None = None,
    use_gmem_reduction: bool | None = None,
    refresh_stream: bool = False,
    index_kind: str = INDEX_KIND_GLOBAL,
    local_layer_idx: int | None = None,
) -> PreparedMlaFmhaRunner:
    """The base module's `prepare_runner` contract (same arguments, checks, split-K pick
    with the KV length defaulting to top_k, and workspaces) for this module's kernel, H16."""
    if index_kind not in INDEX_KINDS:
        raise ValueError(f"index_kind must be one of {INDEX_KINDS}, got {index_kind!r}")
    if index_kind == INDEX_KIND_BASE_WITH_LAYER_OFFSET:
        try:
            local_layer_idx_int = operator.index(local_layer_idx)
        except TypeError as exc:
            raise ValueError(
                "index_kind=BASE_WITH_LAYER_OFFSET requires a nonnegative "
                f"integral local_layer_idx, got {local_layer_idx!r}"
            ) from exc
        if local_layer_idx_int < 0:
            raise ValueError(
                "index_kind=BASE_WITH_LAYER_OFFSET requires a nonnegative "
                f"integral local_layer_idx, got {local_layer_idx!r}"
            )
        layer_offset = local_layer_idx_int * KV_POOL_TOKENS_PER_BLOCK
    else:
        if local_layer_idx is not None:
            raise ValueError(
                "local_layer_idx is only accepted with "
                "index_kind=BASE_WITH_LAYER_OFFSET (global rows carry the layer)"
            )
        layer_offset = 0
    if quant_q_buffer.device.type != "cuda":
        raise ValueError("quant_q_buffer must live on cuda")
    if not quant_q_buffer.is_contiguous():
        raise ValueError("quant_q_buffer must be contiguous")
    if not kv_pool_view.is_contiguous():
        raise ValueError("kv_pool_view must be contiguous")
    q_storage = _base._as_uint8_storage(quant_q_buffer, name="quant_q_buffer")
    kv_storage = _base._as_uint8_storage(kv_pool_view, name="kv_pool_view")
    if q_storage.ndim != 3 or int(q_storage.shape[2]) != KV_STRIDE:
        raise ValueError(
            "quant_q_buffer must have shape "
            f"[total_q, H, {KV_STRIDE}], got {tuple(q_storage.shape)}"
        )
    num_heads = int(q_storage.shape[1])
    if num_heads != NUM_HEADS:
        raise ValueError(f"the wide kernel is H{NUM_HEADS} only, got H{num_heads}")
    if kv_storage.numel() % KV_STRIDE:
        raise ValueError(
            f"kv_pool_view byte count must be divisible by {KV_STRIDE}, got {kv_storage.numel()}"
        )
    total_q = int(q_storage.shape[0])
    if sequence_length.ndim != 1:
        raise ValueError(f"sequence_length must be rank 1, got {tuple(sequence_length.shape)}")
    batch_size = int(sequence_length.shape[0])
    pool_rows = kv_storage.numel() // KV_STRIDE
    runtime_shape = MlaFmhaRuntimeShape(total_q, batch_size, pool_rows, max_sq)
    runtime_shape.validate()
    _base._require_tensor(
        sparse_attn_indices, name="sparse_attn_indices", dtype=torch.int32, shape=(total_q, TOPK)
    )
    _base._require_tensor(
        cu_q_seqlens, name="cu_q_seqlens", dtype=torch.int32, shape=(batch_size + 1,)
    )
    _base._require_tensor(
        sequence_length, name="sequence_length", dtype=torch.int32, shape=(batch_size,)
    )
    if output.dtype != torch.bfloat16 or not output.is_contiguous():
        raise TypeError("output must be contiguous torch.bfloat16")
    if output.numel() != total_q * num_heads * L:
        raise ValueError(
            f"output must contain {total_q * num_heads * L} BF16 elements, got {output.numel()}"
        )
    device = quant_q_buffer.device
    _base._require_same_device(
        (
            ("kv_pool_view", kv_pool_view),
            ("sparse_attn_indices", sparse_attn_indices),
            ("output", output),
            ("cu_q_seqlens", cu_q_seqlens),
            ("sequence_length", sequence_length),
        ),
        device,
    )
    for name, tensor in (
        ("quant_q_buffer", q_storage),
        ("kv_pool_view", kv_storage),
        ("sparse_attn_indices", sparse_attn_indices),
        ("output", output),
    ):
        if tensor.data_ptr() % 16:
            raise ValueError(f"{name} must be at least 16-byte aligned")
    _base._validate_scale_metadata(mla_bmm1_scale, name="mla_bmm1_scale", shape=(2,), device=device)
    _base._validate_scale_metadata(mla_bmm2_scale, name="mla_bmm2_scale", shape=(1,), device=device)

    device_index = _base._device_index(quant_q_buffer)
    if num_ctas_kv is None:
        num_ctas_kv = _base._select_num_ctas_kv(
            batch_size=batch_size,
            max_sq=max_sq,
            max_seq_len_kv=TOPK,
            multiprocessor_count=torch.cuda.get_device_properties(
                device_index
            ).multi_processor_count,
        )
    if not 1 <= num_ctas_kv <= MAX_CTAS_KV:
        raise ValueError(f"num_ctas_kv must be in [1, {MAX_CTAS_KV}], got {num_ctas_kv}")
    if use_gmem_reduction is None:
        use_gmem_reduction = num_ctas_kv > 1 and _base._select_gmem_reduction(
            batch_size=batch_size,
            max_sq=max_sq,
            max_active_clusters=_base._max_active_clusters(device_index, num_ctas_kv),
        )
    specialization = MlaFmhaSpecialization(
        all_sparse_indices_valid=all_sparse_indices_valid,
        num_heads=num_heads,
        q_scaling=q_scaling,
        num_ctas_kv=num_ctas_kv,
        use_gmem_reduction=use_gmem_reduction,
        index_kind=index_kind,
    )
    compiled = _compile_for(specialization, device_index)
    if use_gmem_reduction:
        partial_o_workspace = torch.empty(
            total_q, num_ctas_kv, num_heads, L, dtype=torch.bfloat16, device=device
        )
        partial_stats_workspace = torch.empty(
            total_q, num_ctas_kv, num_heads, 2, dtype=torch.float32, device=device
        )
        completion_counters = _base._completion_counters(total_q, device)
    else:
        # Constexpr-dead parameters keep one launch ABI across modes.
        partial_o_workspace = output.reshape(-1)[:1]
        partial_stats_workspace = sparse_attn_indices.view(torch.float32).reshape(-1)[:1]
        completion_counters = cu_q_seqlens[:1]
    stream = None if refresh_stream else _base._current_stream(device_index)
    return PreparedMlaFmhaRunner(
        compiled=compiled,
        args=(
            q_storage,
            kv_storage.reshape(pool_rows, KV_STRIDE),
            sparse_attn_indices,
            output.reshape(total_q, num_heads, L),
            cu_q_seqlens,
            sequence_length,
            partial_o_workspace,
            partial_stats_workspace,
            completion_counters,
            layer_offset,
            max_sq,
        ),
        device_index=device_index,
        stream=stream,
        runtime_shape=runtime_shape,
        specialization=specialization,
    )


# ----------------------------------------------------------------------------- CTX-HOST-DIET: steady state
#: TRTLLM_GLM52_CTX_HOST_DIET=1: what `prepare_runner` decided for the call signatures it has passed --
#: the specialization (split-K CTA count, GMEM reducer), the pool's row count, the device. A signature
#: fixes every input `prepare_runner` checks or decides on except the query-row count, the pointers and
#: the layer: batch size, index kind, max_sq, options, every dtype, stride and device, the pool's shape,
#: which scale arrays are given. `run_accepted` replays the decision with exactly the arguments
#: `prepare_runner` would bind, after the checks the signature leaves open. No torch tensor is held.
#: Bounded: a full table only misses (no eviction).
_ACCEPTED: dict[tuple, tuple[MlaFmhaSpecialization, int, int]] = {}
ACCEPTED_CAP = 64
#: Calls `run_accepted` sent back to `prepare_runner` (signature not accepted yet).
accepted_misses = 0


def _scale_signature(tensor: torch.Tensor | None):
    return None if tensor is None else (tensor.dtype, tensor.shape, tensor.stride(), tensor.device)


def call_signature(
    quant_q_buffer, kv_pool_view, sparse_attn_indices, output, cu_q_seqlens, sequence_length,
    mla_bmm1_scale, mla_bmm2_scale, options: tuple,
) -> tuple:  # fmt: skip
    """`options`: (all_sparse_indices_valid, max_sq, index_kind, local_layer_idx is None)."""
    return (
        options,
        sequence_length.shape,
        sequence_length.dtype,
        sequence_length.stride(),
        sequence_length.device,
        quant_q_buffer.dtype,
        quant_q_buffer.stride(),
        quant_q_buffer.device,
        kv_pool_view.dtype,
        kv_pool_view.shape,
        kv_pool_view.stride(),
        kv_pool_view.device,
        sparse_attn_indices.dtype,
        sparse_attn_indices.stride(),
        sparse_attn_indices.device,
        output.dtype,
        output.stride(),
        output.device,
        cu_q_seqlens.dtype,
        cu_q_seqlens.stride(),
        cu_q_seqlens.device,
        _scale_signature(mla_bmm1_scale),
        _scale_signature(mla_bmm2_scale),
    )


def run_accepted(
    kernels, quant_q_buffer, kv_pool_view, sparse_attn_indices, output, cu_q_seqlens,
    sequence_length, mla_bmm1_scale, mla_bmm2_scale, *, all_sparse_indices_valid: bool,
    max_sq: int, index_kind: str, local_layer_idx: int | None,
) -> tuple | None:  # fmt: skip
    """Launch `kernels[specialization]` as `prepare_runner(...)` and its runner would (with the
    runner's `compiled` replaced from `kernels`), when the call's signature was accepted (returns
    None); else launch nothing and return the signature, for `accept`. Raises on a call
    `prepare_runner` would reject for its row count, pointers or layer."""
    global accepted_misses
    options = (bool(all_sparse_indices_valid), int(max_sq), index_kind, local_layer_idx is None)
    sig = call_signature(
        quant_q_buffer, kv_pool_view, sparse_attn_indices, output, cu_q_seqlens, sequence_length,
        mla_bmm1_scale, mla_bmm2_scale, options,
    )  # fmt: skip
    decision = _ACCEPTED.get(sig)
    if decision is None:
        accepted_misses += 1
        return sig
    specialization, pool_rows, device_index = decision
    q_storage = _base._as_uint8_storage(quant_q_buffer, name="quant_q_buffer")
    kv_storage = _base._as_uint8_storage(kv_pool_view, name="kv_pool_view")
    total_q = q_storage.shape[0]
    batch_size = sequence_length.shape[0]
    if (
        q_storage.shape != (total_q, NUM_HEADS, KV_STRIDE)
        or not 0 < total_q <= batch_size * max_sq
        or sparse_attn_indices.shape != (total_q, TOPK)
        or cu_q_seqlens.shape != (batch_size + 1,)
        or output.numel() != total_q * NUM_HEADS * L
    ):
        raise ValueError(
            f"sparse MLA decode: {total_q} query rows over {batch_size} sequences do not fit "
            f"q {tuple(q_storage.shape)}, indices {tuple(sparse_attn_indices.shape)}, "
            f"cu_q {tuple(cu_q_seqlens.shape)}, output {output.numel()} elements"
        )
    if (
        q_storage.data_ptr() | kv_storage.data_ptr() | sparse_attn_indices.data_ptr()
    ) & 15 or output.data_ptr() & 15:
        raise ValueError(
            "quant_q_buffer, kv_pool_view, sparse_attn_indices and output must be 16-byte aligned"
        )
    if index_kind == INDEX_KIND_BASE_WITH_LAYER_OFFSET:
        try:
            layer = operator.index(local_layer_idx)
        except TypeError as exc:
            raise ValueError(
                "index_kind=BASE_WITH_LAYER_OFFSET requires a nonnegative "
                f"integral local_layer_idx, got {local_layer_idx!r}"
            ) from exc
        if layer < 0:
            raise ValueError(
                "index_kind=BASE_WITH_LAYER_OFFSET requires a nonnegative "
                f"integral local_layer_idx, got {local_layer_idx!r}"
            )
        layer_offset = layer * KV_POOL_TOKENS_PER_BLOCK
    else:
        layer_offset = 0
    device = quant_q_buffer.device
    num_ctas_kv = specialization.num_ctas_kv
    if specialization.use_gmem_reduction:
        partial_o_workspace = torch.empty(
            total_q, num_ctas_kv, NUM_HEADS, L, dtype=torch.bfloat16, device=device
        )
        partial_stats_workspace = torch.empty(
            total_q, num_ctas_kv, NUM_HEADS, 2, dtype=torch.float32, device=device
        )
        completion_counters = _base._completion_counters(total_q, device)
    else:
        partial_o_workspace = output.reshape(-1)[:1]
        partial_stats_workspace = sparse_attn_indices.view(torch.float32).reshape(-1)[:1]
        completion_counters = cu_q_seqlens[:1]
    kernels[specialization](
        q_storage,
        kv_storage.reshape(pool_rows, KV_STRIDE),
        sparse_attn_indices,
        output.reshape(total_q, NUM_HEADS, L),
        cu_q_seqlens,
        sequence_length,
        partial_o_workspace,
        partial_stats_workspace,
        completion_counters,
        layer_offset,
        max_sq,
        _base._current_stream(device_index),
    )
    return None


def accept(signature: tuple, runner: PreparedMlaFmhaRunner) -> None:
    """Record what `prepare_runner` decided for a signature it passed (see `run_accepted`)."""
    if len(_ACCEPTED) < ACCEPTED_CAP:
        _ACCEPTED[signature] = (
            runner.specialization,
            runner.runtime_shape.pool_rows,
            runner.device_index,
        )
