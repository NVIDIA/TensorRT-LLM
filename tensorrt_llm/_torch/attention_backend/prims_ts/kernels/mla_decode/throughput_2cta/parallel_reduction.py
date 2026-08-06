# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parallel standalone split-KV reduction for throughput 2CTA MLA."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.experimental import primitives as prims

from ...separate_reduction import finalize_log2_sum_exp, normalized_lse_weight
from ..helpers.constants import SPLIT_REDUCTION_SCALE_BARRIER_ID
from ..helpers.mask import MaskType, mask_visible_k_length
from ..helpers.math import ceil_div
from ..helpers.ops import fmax_f32, warp_reduce_max_f32, warp_reduce_sum_f32
from ..helpers.query import groups_tokens_heads_q_row_state, query_batch_bounds
from .work_partition import (
    runtime_row_prefix_active_split_count,
    runtime_split_kv_cap,
)

# One standalone-reducer thread loads an aligned BF16 vec4 partial-O fragment
# and accumulates its four values in FP32 registers. A 128-thread CTA therefore
# covers exactly one D=512 MLA row. These constants are deliberately local to
# the 2CTA policy; the reference reducer keeps its four-warp launch unchanged.
PARALLEL_REDUCTION_THREADS = 128
PARALLEL_REDUCTION_ELEMENTS_PER_THREAD = 4
PARALLEL_REDUCTION_HEAD_DIM = (
    PARALLEL_REDUCTION_THREADS * PARALLEL_REDUCTION_ELEMENTS_PER_THREAD
)


@cute.jit
def _parallel_reduction_row_state(
    groups_tokens_heads_q_ratio: cutlass.Constexpr[int],
    num_heads: cutlass.Constexpr[int],
    seq_len_q: cutlass.Constexpr[int],
    is_var_split_kv: cutlass.Constexpr[bool],
    cache_seqs,
    cu_seqlens_q,
    block_split_kvs,
    split_kv,
    effective_head_idx,
    seq_q_idx,
    batch_idx,
    cfg,
):
    """Return logical row coordinates and its runtime active split count.

    Keep this arithmetic identical to the serial reducer below.  A causal row
    can see fewer K tiles than the last row in its grouped producer tile, and a
    variable-length batch can expose fewer real partitions than the compiled
    static split capacity.  The parallel reducer must ignore both kinds of
    empty workspace rows.
    """

    (
        _,
        logical_head_idx,
        logical_q_idx,
        storage_q_idx,
        query_is_valid,
    ) = groups_tokens_heads_q_row_state(
        effective_head_idx,
        seq_q_idx,
        groups_tokens_heads_q_ratio,
        num_heads,
        seq_len_q,
        cu_seqlens_q,
        batch_idx,
    )
    # Producer and reducer share the same per-batch cap and configured-span group
    # partition. ``split_kv`` remains the grid/workspace capacity.
    split_kv_cap = runtime_split_kv_cap(
        split_kv,
        is_var_split_kv,
        block_split_kvs,
        batch_idx,
    )
    group_k = cache_seqs[batch_idx]
    row_k = group_k
    if cutlass.const_expr(cfg.mask_type == MaskType.CAUSAL.value and seq_len_q > 1):
        _, logical_seq_len_q = query_batch_bounds(
            cu_seqlens_q,
            batch_idx,
            seq_len_q,
        )
        _, _, group_last_logical_q_idx, _, _ = groups_tokens_heads_q_row_state(
            Int32(num_heads * groups_tokens_heads_q_ratio - 1),
            seq_q_idx,
            groups_tokens_heads_q_ratio,
            num_heads,
            seq_len_q,
            cu_seqlens_q,
            batch_idx,
        )
        group_k = mask_visible_k_length(
            cfg.mask_type,
            group_k,
            group_last_logical_q_idx,
            logical_seq_len_q,
        )
        row_k = mask_visible_k_length(
            cfg.mask_type,
            cache_seqs[batch_idx],
            logical_q_idx,
            logical_seq_len_q,
        )
    group_k_tile_total = (group_k + cfg.mma_qk_tiler[1] - 1) // cfg.mma_qk_tiler[1]
    row_k_tile_total = (row_k + cfg.mma_qk_tiler[1] - 1) // cfg.mma_qk_tiler[1]
    active_split_kv = runtime_row_prefix_active_split_count(
        row_k_tile_total,
        group_k_tile_total,
        split_kv_cap,
    )
    return (
        logical_head_idx,
        logical_q_idx,
        storage_q_idx,
        query_is_valid,
        active_split_kv,
    )


@cute.jit
def _store_parallel_reduction_result(
    output,
    lse,
    output_vals,
    global_lse,
    logical_head_idx,
    logical_q_idx,
    storage_q_idx,
    query_is_valid,
    element_idx,
    tidx,
    batch_idx,
    cu_seqlens_q,
):
    """Publish one normalized FP32 output fragment and final LSE."""

    if tidx == Int32(0) and query_is_valid:
        if cutlass.const_expr(cu_seqlens_q is not None):
            lse[logical_head_idx, storage_q_idx] = global_lse
        else:
            lse[logical_head_idx, logical_q_idx, batch_idx] = global_lse

    out_element_dtype = output.element_type
    output_regs = cutlass.Array(
        out_element_dtype,
        PARALLEL_REDUCTION_ELEMENTS_PER_THREAD,
        space=cutlass.AddressSpace.rmem,
    )
    for j in cutlass.range_constexpr(PARALLEL_REDUCTION_ELEMENTS_PER_THREAD):
        output_regs[j] = out_element_dtype(output_vals[j])

    if query_is_valid:
        for j in cutlass.range_constexpr(PARALLEL_REDUCTION_ELEMENTS_PER_THREAD):
            dim_idx = element_idx + Int32(j)
            if cutlass.const_expr(cu_seqlens_q is not None):
                output[logical_head_idx, dim_idx, storage_q_idx] = output_regs[j]
            else:
                output[logical_head_idx, dim_idx, logical_q_idx, batch_idx] = (
                    output_regs[j]
                )


@cute.jit
def run_parallel_reduction_kernel(
    groups_tokens_heads_q_ratio: cutlass.Constexpr[int],
    num_heads: cutlass.Constexpr[int],
    seq_len_q: cutlass.Constexpr[int],
    is_var_split_kv: cutlass.Constexpr[bool],
    output,
    lse,
    acc_output,
    acc_lse,
    split_kv,
    cache_seqs,
    cu_seqlens_q,
    block_split_kvs,
    cfg,
    actual_splits: cutlass.Constexpr[int],
    cluster_size: cutlass.Constexpr[int],
    slots_per_rank: cutlass.Constexpr[int],
):
    """Reduce one D=512 row cooperatively across a padded CTA cluster.

    Rank ``r`` owns ``slots_per_rank`` contiguous split slots.  Slots beyond
    ``actual_splits`` and row-specific inactive splits perform no GMEM access
    or arithmetic.  G1 writes its result directly; for G2/G4/G8 every rank
    publishes a neutral-or-valid ``(FP32 LSE, BF16 O[512])`` state to DSMEM and
    rank zero performs the final merge. Arithmetic remains FP32.

    The producer kernel does not emit a PDL signal, so this same-stream launch
    intentionally contains no ``griddepcontrol`` wait.
    """

    block_idx_x, seq_q_idx, batch_idx = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = tidx % Int32(cfg.threads_per_warp)
    cluster_rank = cute.arch.block_idx_in_cluster()
    effective_head_idx = block_idx_x // Int32(cluster_size)
    (
        logical_head_idx,
        logical_q_idx,
        storage_q_idx,
        query_is_valid,
        active_split_kv,
    ) = _parallel_reduction_row_state(
        groups_tokens_heads_q_ratio,
        num_heads,
        seq_len_q,
        is_var_split_kv,
        cache_seqs,
        cu_seqlens_q,
        block_split_kvs,
        split_kv,
        effective_head_idx,
        seq_q_idx,
        batch_idx,
        cfg,
    )

    element_idx = tidx * Int32(PARALLEL_REDUCTION_ELEMENTS_PER_THREAD)
    neg_inf = Float32(-Float32.inf)
    lse_slots_per_lane = ceil_div(slots_per_rank, cfg.threads_per_warp)
    smem_local_lse = cutlass.Array(
        Float32,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    smem_local_scale = cutlass.Array(
        Float32,
        slots_per_rank,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )

    # One warp computes the rank-local softmax state for this row.  All output
    # threads consume the same scales, so repeating LSE loads and SFU work in
    # every vector lane only adds instructions and register pressure.
    if warp_idx == Int32(0):
        lane_lse = cutlass.Array(
            Float32,
            lse_slots_per_lane,
            space=cutlass.AddressSpace.rmem,
        )
        local_lse_max = neg_inf
        for lane_slot_i in cutlass.range_constexpr(lse_slots_per_lane):
            local_slot_idx = lane_idx + Int32(lane_slot_i * cfg.threads_per_warp)
            split_idx = cluster_rank * Int32(slots_per_rank) + local_slot_idx
            active_slot = (
                query_is_valid
                & (local_slot_idx < Int32(slots_per_rank))
                & (split_idx < Int32(actual_splits))
                & (split_idx < active_split_kv)
            )
            lane_lse[lane_slot_i] = neg_inf
            # Keep the workspace access inside the dynamic predicate. Padded
            # ranks must not form a load from the unpadded producer allocation.
            if active_slot:
                lane_lse[lane_slot_i] = Float32(
                    acc_lse[effective_head_idx, split_idx, seq_q_idx, batch_idx]
                )
            local_lse_max = fmax_f32(
                local_lse_max,
                lane_lse[lane_slot_i],
            )

        local_lse_max = warp_reduce_max_f32(local_lse_max)
        local_exp_frame = local_lse_max if local_lse_max != neg_inf else Float32(0.0)
        lane_exp = cutlass.Array(
            Float32,
            lse_slots_per_lane,
            space=cutlass.AddressSpace.rmem,
        )
        local_sum_lse = Float32(0.0)
        for lane_slot_i in cutlass.range_constexpr(lse_slots_per_lane):
            lane_exp[lane_slot_i] = Float32(
                cute.math.exp2(
                    lane_lse[lane_slot_i] - local_exp_frame,
                    fastmath=True,
                )
            )
            local_sum_lse += lane_exp[lane_slot_i]
        local_sum_lse = warp_reduce_sum_f32(local_sum_lse)
        local_lse_value = finalize_log2_sum_exp(local_exp_frame, local_sum_lse)
        if lane_idx == Int32(0):
            smem_local_lse[0] = local_lse_value

        for lane_slot_i in cutlass.range_constexpr(lse_slots_per_lane):
            local_slot_idx = lane_idx + Int32(lane_slot_i * cfg.threads_per_warp)
            if local_slot_idx < Int32(slots_per_rank):
                smem_local_scale[local_slot_idx] = normalized_lse_weight(
                    lane_lse[lane_slot_i], local_lse_value
                )

    prims.barrier_cta_sync(SPLIT_REDUCTION_SCALE_BARRIER_ID)

    # Every thread owns one contiguous BF16 vec4 GMEM fragment and immediately
    # converts it to FP32 registers. The shared scales directly normalize the
    # rank-local result, so no second numerator array or reciprocal is needed.
    local_output = cutlass.Array(
        Float32,
        PARALLEL_REDUCTION_ELEMENTS_PER_THREAD,
        space=cutlass.AddressSpace.rmem,
    )
    for j in cutlass.range_constexpr(PARALLEL_REDUCTION_ELEMENTS_PER_THREAD):
        local_output[j] = Float32(0.0)

    acc_output_ptr = acc_output.iterator.raw_ptr()
    for slot_i in cutlass.range_constexpr(slots_per_rank):
        split_idx = cluster_rank * Int32(slots_per_rank) + Int32(slot_i)
        active_slot = (
            query_is_valid
            & (split_idx < Int32(actual_splits))
            & (split_idx < active_split_kv)
        )
        if active_slot:
            split_scale = Float32(smem_local_scale[slot_i])
            partial_offset = Int64(
                effective_head_idx * acc_output.stride[0]
                + split_idx * acc_output.stride[1]
                + element_idx * acc_output.stride[2]
                + seq_q_idx * acc_output.stride[3]
                + batch_idx * acc_output.stride[4]
            )
            partial_output = (
                (acc_output_ptr + partial_offset)
                .load(
                    count=PARALLEL_REDUCTION_ELEMENTS_PER_THREAD,
                    alignment=8,
                )
                .to(Float32)
            )
            for j in cutlass.range_constexpr(PARALLEL_REDUCTION_ELEMENTS_PER_THREAD):
                local_output[j] += partial_output[j] * split_scale

    local_lse_value = smem_local_lse[0]

    # Small split counts need no cluster exchange.  G1 still uses one CTA-local
    # scale barrier so its LSE/SFU work is shared rather than repeated 128 ways.
    if cutlass.const_expr(cluster_size == 1):
        _store_parallel_reduction_result(
            output,
            lse,
            local_output,
            local_lse_value,
            logical_head_idx,
            logical_q_idx,
            storage_q_idx,
            query_is_valid,
            element_idx,
            tidx,
            batch_idx,
            cu_seqlens_q,
        )
        return

    # G2+ publishes normalized BF16 O and one FP32 LSE scalar per rank. Every rank
    # writes the neutral (-inf, zero) state for padded or invalid rows and
    # participates in both cluster barriers, so rank zero's DSMEM pointers
    # cannot outlive peers.
    smem_output = cutlass.Array(
        acc_output.element_type,
        PARALLEL_REDUCTION_HEAD_DIM,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    local_output_bf16 = cutlass.Array(
        acc_output.element_type,
        PARALLEL_REDUCTION_ELEMENTS_PER_THREAD,
        space=cutlass.AddressSpace.rmem,
    )
    for j in cutlass.range_constexpr(PARALLEL_REDUCTION_ELEMENTS_PER_THREAD):
        local_output_bf16[j] = acc_output.element_type(local_output[j])
    (smem_output.data_ptr() + element_idx).store(
        local_output_bf16.data_ptr().load(
            count=PARALLEL_REDUCTION_ELEMENTS_PER_THREAD,
            alignment=8,
        ),
        alignment=8,
    )

    prims.barrier_cta_sync(0)
    # Rank zero immediately consumes each peer's published SMEM through DSMEM.
    # Use the ordered arrival; the relaxed form does not order prior writes.
    prims.barrier_cluster_arrive()
    prims.barrier_cluster_wait()

    smem_merged_lse = cutlass.Array(
        Float32,
        1,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    smem_peer_scale = cutlass.Array(
        Float32,
        cluster_size,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    if cluster_rank == Int32(0):
        # One warp merges the per-rank LSE scalars. Keep each mapa rank uniform
        # at its issue site, then let one lane retain that peer's value.
        if warp_idx == Int32(0):
            lane_peer_lse = neg_inf
            for peer_rank_i in cutlass.range_constexpr(cluster_size):
                peer_rank = Int32(peer_rank_i)
                peer_lse_ptr = prims.mapa(
                    smem_local_lse.data_ptr(),
                    peer_rank,
                )
                if lane_idx == peer_rank:
                    lane_peer_lse = peer_lse_ptr.load()

            merged_lse_max = warp_reduce_max_f32(lane_peer_lse)
            merged_exp_frame = (
                merged_lse_max if merged_lse_max != neg_inf else Float32(0.0)
            )
            lane_peer_exp = Float32(
                cute.math.exp2(
                    lane_peer_lse - merged_exp_frame,
                    fastmath=True,
                )
            )
            merged_sum_lse = warp_reduce_sum_f32(lane_peer_exp)
            merged_lse = finalize_log2_sum_exp(merged_exp_frame, merged_sum_lse)
            if lane_idx == Int32(0):
                smem_merged_lse[0] = merged_lse
            if lane_idx < Int32(cluster_size):
                smem_peer_scale[lane_idx] = normalized_lse_weight(
                    lane_peer_lse, merged_lse
                )

        prims.barrier_cta_sync(SPLIT_REDUCTION_SCALE_BARRIER_ID)

        merged_output = cutlass.Array(
            Float32,
            PARALLEL_REDUCTION_ELEMENTS_PER_THREAD,
            space=cutlass.AddressSpace.rmem,
        )
        for j in cutlass.range_constexpr(PARALLEL_REDUCTION_ELEMENTS_PER_THREAD):
            merged_output[j] = Float32(0.0)

        for peer_rank_i in cutlass.range_constexpr(cluster_size):
            peer_rank = Int32(peer_rank_i)
            peer_output = prims.mapa(smem_output.data_ptr(), peer_rank)
            peer_scale = Float32(smem_peer_scale[peer_rank_i])
            peer_output_vals = (
                (peer_output + element_idx)
                .load(
                    count=PARALLEL_REDUCTION_ELEMENTS_PER_THREAD,
                    alignment=8,
                )
                .to(Float32)
            )
            for j in cutlass.range_constexpr(PARALLEL_REDUCTION_ELEMENTS_PER_THREAD):
                merged_output[j] += peer_output_vals[j] * peer_scale

        _store_parallel_reduction_result(
            output,
            lse,
            merged_output,
            smem_merged_lse[0],
            logical_head_idx,
            logical_q_idx,
            storage_q_idx,
            query_is_valid,
            element_idx,
            tidx,
            batch_idx,
            cu_seqlens_q,
        )

    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()
