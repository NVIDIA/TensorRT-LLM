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

from ...separate_reduction import (
    finalize_log2_sum_exp,
    normalized_lse_weight,
    attention_sink_scale,
)
from ..helpers.constants import SPLIT_REDUCTION_SCALE_BARRIER_ID
from ..helpers.math import final_lse
from ..helpers.mask import MaskType, mask_visible_k_length
from ..helpers.math import ceil_div
from ..helpers.ops import fmax_f32, warp_reduce_max_f32, warp_reduce_sum_f32
from ..helpers.query import flat_query_row_state, query_batch_bounds
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
    tile_size_q: cutlass.Constexpr[int],
    num_heads: cutlass.Constexpr[int],
    seq_len_q: Int32,
    is_var_split_kv: cutlass.Constexpr[bool],
    cache_seqs,
    cu_seqlens_q,
    block_split_kvs,
    split_kv,
    row_in_tile,
    query_tile_idx,
    batch_idx,
    cfg,
    split_divisor=None,
):
    """Return logical row coordinates and its runtime active split count.

    Keep this arithmetic identical to the serial reducer below.  A causal row
    can see fewer K tiles than the last row in its producer tile, and a
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
    ) = flat_query_row_state(
        row_in_tile,
        query_tile_idx,
        tile_size_q,
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
    tile_k = cache_seqs[batch_idx]
    query_is_valid = query_is_valid & (tile_k > Int32(0))
    tile_k = cute.math.max(tile_k, Int32(0))
    row_k = tile_k
    if cutlass.const_expr(cfg.mask_type == MaskType.CAUSAL.value):
        _, logical_seq_len_q = query_batch_bounds(
            cu_seqlens_q,
            batch_idx,
            seq_len_q,
        )
        _, _, tile_last_logical_q_idx, _, _ = flat_query_row_state(
            Int32(tile_size_q - 1),
            query_tile_idx,
            tile_size_q,
            num_heads,
            seq_len_q,
            cu_seqlens_q,
            batch_idx,
        )
        tile_k = mask_visible_k_length(
            cfg.mask_type,
            tile_k,
            tile_last_logical_q_idx,
            logical_seq_len_q,
        )
        row_k = mask_visible_k_length(
            cfg.mask_type,
            cache_seqs[batch_idx],
            logical_q_idx,
            logical_seq_len_q,
        )
    tile_k_tile_total = (tile_k + cfg.mma_qk_tiler[1] - 1) // cfg.mma_qk_tiler[1]
    row_k_tile_total = (row_k + cfg.mma_qk_tiler[1] - 1) // cfg.mma_qk_tiler[1]
    active_split_kv = runtime_row_prefix_active_split_count(
        row_k_tile_total,
        tile_k_tile_total,
        split_kv_cap,
        split_divisor=split_divisor,
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
    finalize_attention: cutlass.Constexpr[bool] = False,
    output_quant: cutlass.Constexpr = None,
    output_quant_args=None,
    base_on_e: cutlass.Constexpr[bool] = False,
):
    """Publish one normalized FP32 output fragment and final LSE."""

    if cutlass.const_expr(lse is not None):
        if tidx == Int32(0) and query_is_valid:
            value = final_lse(global_lse, base_on_e or finalize_attention)
            if cutlass.const_expr(cu_seqlens_q is not None):
                lse[logical_head_idx, storage_q_idx] = value
            else:
                lse[logical_head_idx, logical_q_idx, batch_idx] = value

    if cutlass.const_expr(output_quant is not None):
        if query_is_valid:
            row = Int64(storage_q_idx) * output_quant.num_heads + logical_head_idx
            if cutlass.const_expr(cu_seqlens_q is None):
                row = (
                    Int64(batch_idx) * output.shape[2] + logical_q_idx
                ) * output_quant.num_heads + logical_head_idx
            output_quant.store(output_quant_args, output_vals, row, element_idx)
        return

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
def accumulate_parallel_reduction_slot(
    acc_output,
    smem_local_scale,
    local_output,
    row_in_tile,
    query_tile_idx,
    batch_idx,
    element_idx,
    slot_i,
    rank_split_begin,
    query_is_valid,
    active_split_kv,
    actual_splits: cutlass.Constexpr[int],
    finalize_attention: cutlass.Constexpr[bool],
):
    """Add one valid split's BF16 vec4 partial into the FP32 register fragment.

    ``slot_i`` indexes rank-local scales; ``rank_split_begin`` maps it into the
    producer workspace. Inactive slots never load workspace, and sparse
    finalization skips zero-weight partials. The updated fragment is returned
    so both static and runtime merge loops share identical accumulation.
    """
    acc_output_ptr = acc_output.iterator.raw_ptr()
    split_idx = rank_split_begin + Int32(slot_i)
    active_slot = (
        query_is_valid
        & (split_idx < Int32(actual_splits))
        & (split_idx < active_split_kv)
    )
    if active_slot:
        split_scale = Float32(smem_local_scale[slot_i])
        if not finalize_attention or split_scale > Float32(0):
            partial_offset = Int64(
                row_in_tile * acc_output.stride[0]
                + split_idx * acc_output.stride[1]
                + element_idx * acc_output.stride[2]
                + query_tile_idx * acc_output.stride[3]
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
    return local_output


@cute.jit
def run_parallel_reduction_kernel(
    num_heads: cutlass.Constexpr[int],
    seq_len_q: Int32,
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
    attention_sinks=None,
    finalize_attention: cutlass.Constexpr[bool] = False,
    output_quant: cutlass.Constexpr = None,
    output_quant_args=None,
    split_divisor=None,
):
    """Reduce one D=512 row cooperatively across a padded CTA cluster.

    Rank ``r`` owns at most ``slots_per_rank`` contiguous split slots, balanced
    over the live runtime prefix. Slots beyond ``actual_splits`` and
    row-specific inactive splits perform no GMEM access
    or arithmetic.  G1 writes its result directly; for G2/G4/G8 every rank
    publishes a neutral-or-valid ``(FP32 LSE, BF16 O[512])`` state to DSMEM and
    rank zero performs the final merge. Arithmetic remains FP32.

    For a one-wave producer, the caller waits for its post-TMEM dependent-launch
    signal before entering this body.
    """

    if cutlass.const_expr(finalize_attention and cluster_size != 1):
        raise ValueError("final attention output uses a single-CTA reducer per row")
    block_idx_x, query_tile_idx, batch_idx = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = tidx % Int32(cfg.threads_per_warp)
    cluster_rank = Int32(0) if cluster_size == 1 else cute.arch.block_idx_in_cluster()
    row_in_tile = block_idx_x // Int32(cluster_size)
    if cutlass.const_expr(finalize_attention):
        # A sparse query is already one virtual request. The producer writes
        # neutral LSE for skipped splits, so the compiled slot count suffices;
        # live length loads and dynamic split division would repeat per head.
        if cutlass.const_expr(
            not isinstance(seq_len_q, int) or seq_len_q != 1 or cu_seqlens_q is not None
        ):
            raise ValueError("final sparse reduction requires virtual query rows")
        logical_head_idx = query_tile_idx * Int32(cfg.mma_qk_tiler[0]) + row_in_tile
        logical_q_idx = Int32(0)
        storage_q_idx = batch_idx
        query_is_valid = logical_head_idx < Int32(num_heads)
        active_split_kv = Int32(actual_splits)
    else:
        (
            logical_head_idx,
            logical_q_idx,
            storage_q_idx,
            query_is_valid,
            active_split_kv,
        ) = _parallel_reduction_row_state(
            cfg.mma_qk_tiler[0],
            num_heads,
            seq_len_q,
            is_var_split_kv,
            cache_seqs,
            cu_seqlens_q,
            block_split_kvs,
            split_kv,
            row_in_tile,
            query_tile_idx,
            batch_idx,
            cfg,
            split_divisor=split_divisor,
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

    # Capacity sizes SMEM only. Partition the live prefix so a smaller runtime
    # count cannot strand ranks while the remaining ranks do extra work.
    live_splits = cute.math.min(active_split_kv, Int32(actual_splits))
    live_slots_per_rank = cute.ceil_div(live_splits, cluster_size)
    rank_split_begin = cluster_rank * live_slots_per_rank
    live_rank_splits = cute.math.max(
        cute.math.min(
            live_splits - rank_split_begin,
            live_slots_per_rank,
        ),
        Int32(0),
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
            if Int32(lane_slot_i * cfg.threads_per_warp) < live_rank_splits:
                local_slot_idx = lane_idx + Int32(lane_slot_i * cfg.threads_per_warp)
                split_idx = rank_split_begin + local_slot_idx
                active_slot = (
                    query_is_valid
                    & (local_slot_idx < live_rank_splits)
                    & (split_idx < Int32(actual_splits))
                    & (split_idx < active_split_kv)
                )
                lane_lse[lane_slot_i] = neg_inf
                # Keep the workspace access inside the dynamic predicate. Padded
                # ranks must not form a load from the unpadded producer allocation.
                if active_slot:
                    lane_lse[lane_slot_i] = Float32(
                        acc_lse[row_in_tile, split_idx, query_tile_idx, batch_idx]
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
            if Int32(lane_slot_i * cfg.threads_per_warp) < live_rank_splits:
                lane_exp[lane_slot_i] = Float32(
                    cute.math.exp2(
                        lane_lse[lane_slot_i] - local_exp_frame,
                        fastmath=True,
                    )
                )
                local_sum_lse += lane_exp[lane_slot_i]
        local_sum_lse = warp_reduce_sum_f32(local_sum_lse)
        # Keep this warp-local value distinct from the post-barrier SMEM load.
        stats_lse_value = finalize_log2_sum_exp(local_exp_frame, local_sum_lse)
        sink_scale = Float32(1)
        if cutlass.const_expr(finalize_attention and attention_sinks is not None):
            sink_scale = Float32(0)
            if query_is_valid:
                sink_scale = attention_sink_scale(
                    stats_lse_value, Float32(attention_sinks[logical_head_idx])
                )
        if lane_idx == Int32(0):
            smem_local_lse[0] = stats_lse_value

        for lane_slot_i in cutlass.range_constexpr(lse_slots_per_lane):
            if Int32(lane_slot_i * cfg.threads_per_warp) < live_rank_splits:
                local_slot_idx = lane_idx + Int32(lane_slot_i * cfg.threads_per_warp)
                if local_slot_idx < live_rank_splits:
                    weight = normalized_lse_weight(
                        lane_lse[lane_slot_i], stats_lse_value
                    )
                    if cutlass.const_expr(finalize_attention):
                        weight *= sink_scale
                    smem_local_scale[local_slot_idx] = weight

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

    if cutlass.const_expr(finalize_attention):
        # Sparse plans have fixed split capacity; stage each merge slot directly.
        # Resolve fixed slot addresses before generating the staged merge loop.
        for slot_i in cutlass.range_constexpr(slots_per_rank):
            local_output = accumulate_parallel_reduction_slot(
                acc_output,
                smem_local_scale,
                local_output,
                row_in_tile,
                query_tile_idx,
                batch_idx,
                element_idx,
                slot_i,
                rank_split_begin,
                query_is_valid,
                active_split_kv,
                actual_splits,
                finalize_attention,
            )
    else:
        # Dense plans keep runtime split pruning within their prepared capacity.
        for slot_i in cutlass.range(live_rank_splits, unroll=4):
            local_output = accumulate_parallel_reduction_slot(
                acc_output,
                smem_local_scale,
                local_output,
                row_in_tile,
                query_tile_idx,
                batch_idx,
                element_idx,
                slot_i,
                rank_split_begin,
                query_is_valid,
                active_split_kv,
                actual_splits,
                finalize_attention,
            )

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
            finalize_attention,
            output_quant,
            output_quant_args,
            base_on_e=cfg.lse_base_on_e,
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
            base_on_e=cfg.lse_base_on_e,
        )

    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()
