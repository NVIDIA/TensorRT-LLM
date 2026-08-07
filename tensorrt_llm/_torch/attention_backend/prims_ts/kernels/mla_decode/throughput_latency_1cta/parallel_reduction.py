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

"""Parallel standalone split-KV reduction for throughput-latency 1CTA MLA."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.experimental import primitives as prims

from ...separate_reduction import finalize_log2_sum_exp, normalized_lse_weight
from ..helpers.constants import SPLIT_REDUCTION_SCALE_BARRIER_ID
from ..helpers.math import ceil_div
from ..helpers.ops import (
    fmax_f32,
    warp_reduce_max_f32,
    warp_reduce_sum_f32,
)
from ..helpers.query import groups_tokens_heads_q_row_state, public_query_flat_row
from ..parallel_reduction_topology import ParallelReductionTopology
from .config import MlaConfig


GMEM_REDUCTION_WARP_LANES = 32
GMEM_REDUCTION_WARPS_PER_CTA = 4

# The parallel reducer always uses 128 threads. Q8/Q16/Q32 assign one scalar
# from a 128-element output band to each thread. Q64 uses up to one BF16 vec8
# per thread, but caps a slice at four D-per-CTA rows so the four warps can
# still compute one row's shared LSE state each.
PARALLEL_GMEM_REDUCTION_THREADS = 128
PARALLEL_GMEM_REDUCTION_ELEMENTS_PER_THREAD = 8
PARALLEL_GMEM_REDUCTION_SWAPS_ELEMENTS_PER_SLICE = 128
PARALLEL_GMEM_REDUCTION_ELEMENTS_PER_SLICE = (
    PARALLEL_GMEM_REDUCTION_THREADS * PARALLEL_GMEM_REDUCTION_ELEMENTS_PER_THREAD
)
_PARALLEL_GMEM_REDUCTION_SUPPORTED_SLICE_ELEMENTS = (
    PARALLEL_GMEM_REDUCTION_SWAPS_ELEMENTS_PER_SLICE,
    512,
    PARALLEL_GMEM_REDUCTION_ELEMENTS_PER_SLICE,
)


@cute.jit
def _split_o_row_base_offset(
    cfg: MlaConfig,
    batch_idx: Int32,
    q_idx: Int32,
    head_idx: Int32,
    dim_idx: Int32,
) -> Int64:
    """Return the S=0 split-O offset, widening before the first product."""

    return (
        (
            (Int64(batch_idx) * Int64(cfg.seq_len_q) + Int64(q_idx))
            * Int64(cfg.num_heads_q)
            + Int64(head_idx)
        )
        * Int64(cfg.num_ctas_per_seq_kv)
    ) * Int64(cfg.head_dim_v) + Int64(dim_idx)


@cute.jit
def _output_element_offset(
    cfg: MlaConfig,
    output_query_row: Int32,
    dim_idx: Int32,
) -> Int64:
    """Linearize one final-output element without a 32-bit row product."""

    return Int64(output_query_row) * Int64(cfg.head_dim_v) + Int64(dim_idx)


def _validate_parallel_reduction_slice(elements_per_slice: int) -> None:
    if elements_per_slice not in _PARALLEL_GMEM_REDUCTION_SUPPORTED_SLICE_ELEMENTS:
        raise ValueError(
            "parallel reducer supports only 128-, 512-, or 1,024-element slices"
        )


def supports_parallel_gmem_reduction(cfg: MlaConfig) -> bool:
    """Return whether ``cfg`` matches the parallel-reducer envelope."""

    return (
        cfg.use_multi_ctas_kv == 1
        and cfg.use_cluster_reduction != 1
        and cfg.head_dim_per_cta_v in (128, 256, 512)
        and cfg.tile_size_q in (8, 16, 32, 64)
        and 2 <= cfg.num_ctas_per_seq_kv <= 128
        and (
            (cfg.o_dtype == "bf16" and cfg.use_bf16_output == 1)
            or (cfg.o_dtype == "e4m3" and cfg.use_fp8_output == 1)
        )
        # Split partial O is always BF16 in initialize_workspace. Keep the
        # explicit byte check so a future workspace dtype change fails closed.
        and cfg.partial_o_dtype_bytes == 2
    )


def _parallel_reduction_effective_slice_elements(
    cfg: MlaConfig,
    elements_per_slice: int,
) -> int:
    """Cap one slice to the four row-statistics warps in a reducer CTA."""

    _validate_parallel_reduction_slice(elements_per_slice)
    return min(
        elements_per_slice,
        GMEM_REDUCTION_WARPS_PER_CTA * cfg.head_dim_per_cta_v,
    )


def parallel_gmem_reduction_launch_shape(
    cfg: MlaConfig,
    topology: ParallelReductionTopology,
    elements_per_slice: int = PARALLEL_GMEM_REDUCTION_ELEMENTS_PER_SLICE,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Return the parallel reducer grid and cluster shape.

    Grid X packs the cluster rank inside each output slice. Grid Y
    packs effective Q and the V head-dimension CTA so the helper remains exact
    about D-per-CTA rather than assuming total D is always 512.
    """

    elements_per_slice = _parallel_reduction_effective_slice_elements(
        cfg, elements_per_slice
    )
    output_slices = ceil_div(
        cfg.num_heads_q * cfg.head_dim_per_cta_v,
        elements_per_slice,
    )
    return (
        (
            output_slices * topology.cluster_size,
            cfg.seq_len_q * cfg.num_ctas_per_head_dim,
            cfg.batch_size,
        ),
        (topology.cluster_size, 1, 1),
    )


def parallel_gmem_reduction_base_clusters(
    cfg: MlaConfig,
    elements_per_slice: int = PARALLEL_GMEM_REDUCTION_ELEMENTS_PER_SLICE,
) -> int:
    """Return logical reducer clusters for the selected output slice."""

    elements_per_slice = _parallel_reduction_effective_slice_elements(
        cfg, elements_per_slice
    )
    output_slices = ceil_div(
        cfg.num_heads_q * cfg.head_dim_per_cta_v,
        elements_per_slice,
    )
    return output_slices * cfg.seq_len_q * cfg.num_ctas_per_head_dim * cfg.batch_size


def parallel_gmem_reduction_threads(elements_per_slice: int) -> int:
    """Return reducer threads for an output slice."""

    _validate_parallel_reduction_slice(elements_per_slice)
    return PARALLEL_GMEM_REDUCTION_THREADS


def parallel_gmem_reduction_elements_per_thread(elements_per_slice: int) -> int:
    """Return scalar output elements owned by one reducer thread.

    A 128-element band uses one scalar per thread. Wider slices assign BF16
    vec4 or vec8 fragments while retaining one shared-statistics warp per row.
    """

    return ceil_div(
        elements_per_slice,
        parallel_gmem_reduction_threads(elements_per_slice),
    )


@cute.jit
def _run_parallel_gmem_reduction_g1_shared_stats(
    output,
    lse,
    acc_output,
    acc_lse,
    cache_seqs,
    cu_seqlens_q,
    cfg,
    elements_per_slice: cutlass.Constexpr[int],
):
    """Reduce G1 partials with the compact row-shared schedule."""

    slice_idx, block_idx_y, batch_idx = cute.arch.block_idx()
    thread_idx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = thread_idx % Int32(GMEM_REDUCTION_WARP_LANES)
    head_dim_cta_idx = block_idx_y % Int32(cfg.num_ctas_per_head_dim)
    q_idx = block_idx_y // Int32(cfg.num_ctas_per_head_dim)
    elements_per_slice = _parallel_reduction_effective_slice_elements(
        cfg, elements_per_slice
    )
    rows_per_slice = ceil_div(elements_per_slice, cfg.head_dim_per_cta_v)
    reducer_threads = parallel_gmem_reduction_threads(elements_per_slice)
    output_elements_per_thread = ceil_div(elements_per_slice, reducer_threads)
    slice_element_base = slice_idx * Int32(elements_per_slice)

    smem_scale = cutlass.Array(
        Float32,
        rows_per_slice * cfg.num_ctas_per_seq_kv,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )

    # One statistics warp per covered row stripes over split slots, then
    # publishes normalized scales for all output threads to reuse.
    if warp_idx < Int32(rows_per_slice):
        stats_row = warp_idx
        stats_element = slice_element_base + stats_row * Int32(cfg.head_dim_per_cta_v)
        head_idx = stats_element // Int32(cfg.head_dim_per_cta_v)
        (
            storage_flat_query_row,
            _,
            _,
            _,
            valid_output_row,
        ) = groups_tokens_heads_q_row_state(
            head_idx,
            q_idx,
            cfg.groups_tokens_heads_q_ratio,
            cfg.logical_num_heads_q,
            cfg.logical_seq_len_q,
            cu_seqlens_q=cu_seqlens_q,
            batch_idx=batch_idx,
        )
        if head_idx < Int32(cfg.num_heads_q) and valid_output_row:
            row_lse = acc_lse[head_idx, None, q_idx, batch_idx]
            lse_per_lane = ceil_div(
                cfg.num_ctas_per_seq_kv,
                GMEM_REDUCTION_WARP_LANES,
            )
            lane_lse = cutlass.Array(
                Float32,
                lse_per_lane,
                space=cutlass.AddressSpace.rmem,
            )
            lse_max = Float32(-Float32.inf)
            for lane_slot_i in cutlass.range_constexpr(lse_per_lane):
                split_idx = lane_idx + Int32(lane_slot_i * GMEM_REDUCTION_WARP_LANES)
                lane_lse[lane_slot_i] = (
                    Float32(row_lse[split_idx])
                    if split_idx < Int32(cfg.num_ctas_per_seq_kv)
                    else Float32(-Float32.inf)
                )
                lse_max = fmax_f32(lse_max, lane_lse[lane_slot_i])

            lse_max = warp_reduce_max_f32(lse_max)
            lse_max = lse_max if lse_max != Float32(-Float32.inf) else Float32(0.0)
            lse_sum = Float32(0.0)
            for lane_slot_i in cutlass.range_constexpr(lse_per_lane):
                lse_sum += cute.math.exp2(
                    lane_lse[lane_slot_i] - lse_max,
                    fastmath=True,
                )
            lse_sum = warp_reduce_sum_f32(lse_sum)
            global_lse = finalize_log2_sum_exp(lse_max, lse_sum)
            if (
                lane_idx == Int32(0)
                and head_dim_cta_idx == Int32(0)
                and stats_element % Int32(cfg.head_dim_per_cta_v) == Int32(0)
            ):
                output_query_row = public_query_flat_row(
                    cfg,
                    storage_flat_query_row,
                    batch_idx,
                    cu_seqlens_q,
                )
                (lse.iterator.raw_ptr() + output_query_row).store(global_lse)

            for lane_slot_i in cutlass.range_constexpr(lse_per_lane):
                split_idx = lane_idx + Int32(lane_slot_i * GMEM_REDUCTION_WARP_LANES)
                if split_idx < Int32(cfg.num_ctas_per_seq_kv):
                    smem_scale[
                        stats_row * Int32(cfg.num_ctas_per_seq_kv) + split_idx
                    ] = normalized_lse_weight(lane_lse[lane_slot_i], global_lse)

    prims.barrier_cta_sync(
        barrier_id=SPLIT_REDUCTION_SCALE_BARRIER_ID,
        thread_count=reducer_threads,
    )

    thread_elem_offset = thread_idx * Int32(output_elements_per_thread)
    flat_element = slice_element_base + thread_elem_offset
    head_idx = flat_element // Int32(cfg.head_dim_per_cta_v)
    dim_in_cta = flat_element - head_idx * Int32(cfg.head_dim_per_cta_v)
    row_in_slice = (flat_element - slice_element_base) // Int32(cfg.head_dim_per_cta_v)
    dim_idx = head_dim_cta_idx * Int32(cfg.head_dim_per_cta_v) + dim_in_cta
    (
        storage_flat_query_row,
        _,
        _,
        _,
        valid_output_row,
    ) = groups_tokens_heads_q_row_state(
        head_idx,
        q_idx,
        cfg.groups_tokens_heads_q_ratio,
        cfg.logical_num_heads_q,
        cfg.logical_seq_len_q,
        cu_seqlens_q=cu_seqlens_q,
        batch_idx=batch_idx,
    )
    valid_output = (
        thread_elem_offset < Int32(elements_per_slice)
        and head_idx < Int32(cfg.num_heads_q)
        and dim_idx < Int32(cfg.head_dim_v)
        and valid_output_row
    )
    if valid_output:
        output_acc = cutlass.Array(
            Float32,
            output_elements_per_thread,
            space=cutlass.AddressSpace.rmem,
        )
        for elem_i in cutlass.range_constexpr(output_elements_per_thread):
            output_acc[elem_i] = Float32(0.0)
        # Form the 64-bit row base once. Split strides are compile-time
        # constants in this fixed-S kernel, avoiding repeated wide integer
        # linearization in the hot accumulation loop.
        acc_row_ptr = acc_output.iterator.raw_ptr() + _split_o_row_base_offset(
            cfg,
            batch_idx,
            q_idx,
            head_idx,
            dim_idx,
        )
        for split_i in cutlass.range_constexpr(cfg.num_ctas_per_seq_kv):
            split_idx = Int32(split_i)
            scale = Float32(
                smem_scale[row_in_slice * Int32(cfg.num_ctas_per_seq_kv) + split_idx]
            )
            split_ptr = acc_row_ptr + Int64(split_i * cfg.head_dim_v)
            if cutlass.const_expr(output_elements_per_thread == 1):
                output_acc[0] += Float32(split_ptr.load()) * scale
            else:
                partial_o = split_ptr.load(
                    count=output_elements_per_thread,
                    alignment=output_elements_per_thread * cfg.partial_o_dtype_bytes,
                ).to(Float32)
                for elem_i in cutlass.range_constexpr(output_elements_per_thread):
                    output_acc[elem_i] += partial_o[elem_i] * scale

        output_query_row = public_query_flat_row(
            cfg,
            storage_flat_query_row,
            batch_idx,
            cu_seqlens_q,
        )
        out_elem_offset = _output_element_offset(cfg, output_query_row, dim_idx)
        if cutlass.const_expr(output_elements_per_thread == 1):
            (output.iterator.raw_ptr() + out_elem_offset).store(
                output.element_type(output_acc[0])
            )
        else:
            output_regs = cutlass.Array(
                output.element_type,
                output_elements_per_thread,
                space=cutlass.AddressSpace.rmem,
            )
            for elem_i in cutlass.range_constexpr(output_elements_per_thread):
                output_regs[elem_i] = output.element_type(output_acc[elem_i])
            (output.iterator.raw_ptr() + out_elem_offset).store(
                output_regs.data_ptr().load(
                    count=output_elements_per_thread,
                    alignment=output_elements_per_thread * cfg.o_dtype_bytes,
                ),
                alignment=output_elements_per_thread * cfg.o_dtype_bytes,
            )


@cute.jit
def _run_parallel_gmem_reduction_shared_stats(
    output,
    lse,
    acc_output,
    acc_lse,
    cache_seqs,
    cu_seqlens_q,
    cfg,
    cluster_size: cutlass.Constexpr[int],
    slots_per_rank: cutlass.Constexpr[int],
    actual_splits: cutlass.Constexpr[int],
    elements_per_slice: cutlass.Constexpr[int],
):
    """Reduce split-KV with one cooperative statistics warp per output row.

    Every cluster rank owns at most four D-per-CTA rows and a cyclic subset of
    split slots. One warp calculates each row's statistics, then all output
    threads reuse the normalized scales while reducing BF16 fragments. Rank
    zero repeats the same row-shared scheme for the final DSMEM merge.
    """

    block_idx_x, block_idx_y, batch_idx = cute.arch.block_idx()
    thread_idx, _, _ = cute.arch.thread_idx()
    cluster_rank = cute.arch.block_idx_in_cluster()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = thread_idx % Int32(GMEM_REDUCTION_WARP_LANES)

    slice_idx = block_idx_x // Int32(cluster_size)
    head_dim_cta_idx = block_idx_y % Int32(cfg.num_ctas_per_head_dim)
    q_idx = block_idx_y // Int32(cfg.num_ctas_per_head_dim)
    local_slots = slots_per_rank
    elements_per_slice = _parallel_reduction_effective_slice_elements(
        cfg, elements_per_slice
    )
    rows_per_slice = ceil_div(elements_per_slice, cfg.head_dim_per_cta_v)
    reducer_threads = parallel_gmem_reduction_threads(elements_per_slice)
    output_elements_per_thread = ceil_div(elements_per_slice, reducer_threads)
    slice_element_base = slice_idx * Int32(elements_per_slice)

    neg_inf = Float32(-Float32.inf)
    smem_local_scale = cutlass.Array(
        Float32,
        rows_per_slice * local_slots,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    smem_local_lse = cutlass.Array(
        Float32,
        rows_per_slice,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )

    # Statistics warp r owns row r. This ownership is independent of vector
    # ownership; a vector warp can consume a different row fragment.
    if warp_idx < Int32(rows_per_slice):
        stats_row = warp_idx
        stats_element = slice_element_base + stats_row * Int32(cfg.head_dim_per_cta_v)
        stats_head_idx = stats_element // Int32(cfg.head_dim_per_cta_v)
        (
            _,
            _,
            _,
            _,
            valid_stats_row,
        ) = groups_tokens_heads_q_row_state(
            stats_head_idx,
            q_idx,
            cfg.groups_tokens_heads_q_ratio,
            cfg.logical_num_heads_q,
            cfg.logical_seq_len_q,
            cu_seqlens_q=cu_seqlens_q,
            batch_idx=batch_idx,
        )
        valid_stats = stats_head_idx < Int32(cfg.num_heads_q) and valid_stats_row
        lse_per_lane = ceil_div(local_slots, GMEM_REDUCTION_WARP_LANES)
        lane_lse = cutlass.Array(
            Float32,
            lse_per_lane,
            space=cutlass.AddressSpace.rmem,
        )
        lane_exp = cutlass.Array(
            Float32,
            lse_per_lane,
            space=cutlass.AddressSpace.rmem,
        )
        for lane_slot_i in cutlass.range_constexpr(lse_per_lane):
            lane_lse[lane_slot_i] = neg_inf
        local_lse = neg_inf
        if valid_stats:
            # Form the row view only after validating the grouped output row.
            # Invalid/padded rows publish a neutral state without an acc_lse
            # address or load.
            row_lse = acc_lse[stats_head_idx, None, q_idx, batch_idx]
            local_max = neg_inf
            for lane_slot_i in cutlass.range_constexpr(lse_per_lane):
                local_slot_idx = lane_idx + Int32(
                    lane_slot_i * GMEM_REDUCTION_WARP_LANES
                )
                split_idx = local_slot_idx * Int32(cluster_size) + cluster_rank
                if local_slot_idx < Int32(local_slots) and split_idx < Int32(
                    actual_splits
                ):
                    lane_lse[lane_slot_i] = Float32(row_lse[split_idx])
                local_max = fmax_f32(local_max, lane_lse[lane_slot_i])

            local_max = warp_reduce_max_f32(local_max)
            safe_local_max = local_max if local_max != neg_inf else Float32(0.0)
            local_sum = Float32(0.0)
            for lane_slot_i in cutlass.range_constexpr(lse_per_lane):
                lane_exp[lane_slot_i] = Float32(
                    cute.math.exp2(
                        lane_lse[lane_slot_i] - safe_local_max,
                        fastmath=True,
                    )
                )
                local_sum += lane_exp[lane_slot_i]
            local_sum = warp_reduce_sum_f32(local_sum)
            local_lse = finalize_log2_sum_exp(safe_local_max, local_sum)

        if lane_idx == Int32(0):
            smem_local_lse[stats_row] = local_lse

        for lane_slot_i in cutlass.range_constexpr(lse_per_lane):
            local_slot_idx = lane_idx + Int32(lane_slot_i * GMEM_REDUCTION_WARP_LANES)
            if local_slot_idx < Int32(local_slots):
                smem_local_scale[stats_row * Int32(local_slots) + local_slot_idx] = (
                    normalized_lse_weight(lane_lse[lane_slot_i], local_lse)
                )

    prims.barrier_cta_sync(
        barrier_id=SPLIT_REDUCTION_SCALE_BARRIER_ID,
        thread_count=reducer_threads,
    )

    thread_elem_offset = thread_idx * Int32(output_elements_per_thread)
    flat_element = slice_element_base + thread_elem_offset
    head_idx = flat_element // Int32(cfg.head_dim_per_cta_v)
    dim_in_cta = flat_element - head_idx * Int32(cfg.head_dim_per_cta_v)
    row_in_slice = (flat_element - slice_element_base) // Int32(cfg.head_dim_per_cta_v)
    dim_idx = head_dim_cta_idx * Int32(cfg.head_dim_per_cta_v) + dim_in_cta
    (
        storage_flat_query_row,
        _,
        _,
        _,
        valid_output_row,
    ) = groups_tokens_heads_q_row_state(
        head_idx,
        q_idx,
        cfg.groups_tokens_heads_q_ratio,
        cfg.logical_num_heads_q,
        cfg.logical_seq_len_q,
        cu_seqlens_q=cu_seqlens_q,
        batch_idx=batch_idx,
    )
    valid_output = (
        thread_elem_offset < Int32(elements_per_slice)
        and head_idx < Int32(cfg.num_heads_q)
        and dim_idx < Int32(cfg.head_dim_v)
        and valid_output_row
    )
    local_output = cutlass.Array(
        Float32,
        output_elements_per_thread,
        space=cutlass.AddressSpace.rmem,
    )
    for elem_i in cutlass.range_constexpr(output_elements_per_thread):
        local_output[elem_i] = Float32(0.0)
    if valid_output:
        # Offset to this rank once; each unrolled local slot then advances by
        # the compile-time cluster stride. This retains 64-bit safety without
        # paying a full wide linearization for every partial load.
        acc_row_ptr = (
            acc_output.iterator.raw_ptr()
            + _split_o_row_base_offset(
                cfg,
                batch_idx,
                q_idx,
                head_idx,
                dim_idx,
            )
            + Int64(cluster_rank) * Int64(cfg.head_dim_v)
        )
        for local_slot_i in cutlass.range_constexpr(local_slots):
            split_idx = Int32(local_slot_i * cluster_size) + cluster_rank
            if split_idx < Int32(actual_splits):
                scale = Float32(
                    smem_local_scale[
                        row_in_slice * Int32(local_slots) + Int32(local_slot_i)
                    ]
                )
                split_ptr = acc_row_ptr + Int64(
                    local_slot_i * cluster_size * cfg.head_dim_v
                )
                if cutlass.const_expr(output_elements_per_thread == 1):
                    local_output[0] += Float32(split_ptr.load()) * scale
                else:
                    partial_o = split_ptr.load(
                        count=output_elements_per_thread,
                        alignment=output_elements_per_thread
                        * cfg.partial_o_dtype_bytes,
                    ).to(Float32)
                    for elem_i in cutlass.range_constexpr(output_elements_per_thread):
                        local_output[elem_i] += partial_o[elem_i] * scale

    # Publish normalized O in BF16 while retaining FP32 accumulation.
    smem_o = cutlass.Array(
        acc_output.element_type,
        reducer_threads * output_elements_per_thread,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    smem_o_offset = thread_idx * Int32(output_elements_per_thread)
    if cutlass.const_expr(output_elements_per_thread == 1):
        smem_o[smem_o_offset] = acc_output.element_type(local_output[0])
    else:
        smem_output_regs = cutlass.Array(
            acc_output.element_type,
            output_elements_per_thread,
            space=cutlass.AddressSpace.rmem,
        )
        for elem_i in cutlass.range_constexpr(output_elements_per_thread):
            smem_output_regs[elem_i] = acc_output.element_type(local_output[elem_i])
        (smem_o.data_ptr() + smem_o_offset).store(
            smem_output_regs.data_ptr().load(
                count=output_elements_per_thread,
                alignment=output_elements_per_thread * cfg.partial_o_dtype_bytes,
            ),
            alignment=output_elements_per_thread * cfg.partial_o_dtype_bytes,
        )

    prims.barrier_cta_sync(0)
    prims.barrier_cluster_arrive()
    prims.barrier_cluster_wait()

    smem_global_lse = cutlass.Array(
        Float32,
        rows_per_slice,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    smem_peer_scale = cutlass.Array(
        Float32,
        rows_per_slice * cluster_size,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    if cluster_rank == Int32(0):
        if warp_idx < Int32(rows_per_slice):
            stats_row = warp_idx
            stats_element = slice_element_base + stats_row * Int32(
                cfg.head_dim_per_cta_v
            )
            stats_head_idx = stats_element // Int32(cfg.head_dim_per_cta_v)
            (
                _,
                _,
                _,
                _,
                valid_stats_row,
            ) = groups_tokens_heads_q_row_state(
                stats_head_idx,
                q_idx,
                cfg.groups_tokens_heads_q_ratio,
                cfg.logical_num_heads_q,
                cfg.logical_seq_len_q,
                cu_seqlens_q=cu_seqlens_q,
                batch_idx=batch_idx,
            )
            valid_stats = stats_head_idx < Int32(cfg.num_heads_q) and valid_stats_row
            peer_lse = neg_inf
            for peer_rank_i in cutlass.range_constexpr(cluster_size):
                if valid_stats and lane_idx == Int32(peer_rank_i):
                    peer_lse = (
                        prims.mapa(
                            smem_local_lse.data_ptr(),
                            Int32(peer_rank_i),
                        )
                        + stats_row
                    ).load()

            global_max = warp_reduce_max_f32(peer_lse)
            safe_global_max = global_max if global_max != neg_inf else Float32(0.0)
            peer_exp = Float32(
                cute.math.exp2(
                    peer_lse - safe_global_max,
                    fastmath=True,
                )
            )
            global_sum = warp_reduce_sum_f32(peer_exp)
            global_lse = finalize_log2_sum_exp(safe_global_max, global_sum)
            if lane_idx == Int32(0):
                smem_global_lse[stats_row] = global_lse
            if lane_idx < Int32(cluster_size):
                smem_peer_scale[stats_row * Int32(cluster_size) + lane_idx] = (
                    normalized_lse_weight(peer_lse, global_lse)
                )

        # Only rank zero executes this CTA-uniform branch and barrier.
        prims.barrier_cta_sync(0)

        global_output = cutlass.Array(
            Float32,
            output_elements_per_thread,
            space=cutlass.AddressSpace.rmem,
        )
        for elem_i in cutlass.range_constexpr(output_elements_per_thread):
            global_output[elem_i] = Float32(0.0)
        if valid_output:
            for peer_rank_i in cutlass.range_constexpr(cluster_size):
                peer_scale = Float32(
                    smem_peer_scale[
                        row_in_slice * Int32(cluster_size) + Int32(peer_rank_i)
                    ]
                )
                peer_o = prims.mapa(
                    smem_o.data_ptr(),
                    Int32(peer_rank_i),
                )
                if cutlass.const_expr(output_elements_per_thread == 1):
                    global_output[0] += (
                        Float32((peer_o + smem_o_offset).load()) * peer_scale
                    )
                else:
                    peer_output = (
                        (peer_o + smem_o_offset)
                        .load(
                            count=output_elements_per_thread,
                            alignment=(
                                output_elements_per_thread * cfg.partial_o_dtype_bytes
                            ),
                        )
                        .to(Float32)
                    )
                    for elem_i in cutlass.range_constexpr(output_elements_per_thread):
                        global_output[elem_i] += peer_output[elem_i] * peer_scale

            output_query_row = public_query_flat_row(
                cfg,
                storage_flat_query_row,
                batch_idx,
                cu_seqlens_q,
            )
            if head_dim_cta_idx == Int32(0) and dim_in_cta == Int32(0):
                (lse.iterator.raw_ptr() + output_query_row).store(
                    smem_global_lse[row_in_slice]
                )
            out_elem_offset = _output_element_offset(cfg, output_query_row, dim_idx)
            if cutlass.const_expr(output_elements_per_thread == 1):
                (output.iterator.raw_ptr() + out_elem_offset).store(
                    output.element_type(global_output[0])
                )
            else:
                output_regs = cutlass.Array(
                    output.element_type,
                    output_elements_per_thread,
                    space=cutlass.AddressSpace.rmem,
                )
                for elem_i in cutlass.range_constexpr(output_elements_per_thread):
                    output_regs[elem_i] = output.element_type(global_output[elem_i])
                (output.iterator.raw_ptr() + out_elem_offset).store(
                    output_regs.data_ptr().load(
                        count=output_elements_per_thread,
                        alignment=output_elements_per_thread * cfg.o_dtype_bytes,
                    ),
                    alignment=output_elements_per_thread * cfg.o_dtype_bytes,
                )

    # Every peer remains alive until rank zero finishes all DSMEM O loads.
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()


@cute.jit
def run_parallel_gmem_reduction_kernel(
    output,
    lse,
    acc_output,
    acc_lse,
    cache_seqs,
    cu_seqlens_q,
    cfg,
    cluster_size: cutlass.Constexpr[int],
    slots_per_rank: cutlass.Constexpr[int],
    actual_splits: cutlass.Constexpr[int],
    elements_per_slice: cutlass.Constexpr[int],
):
    """Reduce split-KV partials with row-shared local and peer statistics."""

    prims.griddepcontrol(kind=prims.GridDepAction.WAIT)

    if cutlass.const_expr(cluster_size == 1):
        _run_parallel_gmem_reduction_g1_shared_stats(
            output,
            lse,
            acc_output,
            acc_lse,
            cache_seqs,
            cu_seqlens_q,
            cfg,
            elements_per_slice,
        )
        return

    _run_parallel_gmem_reduction_shared_stats(
        output,
        lse,
        acc_output,
        acc_lse,
        cache_seqs,
        cu_seqlens_q,
        cfg,
        cluster_size,
        slots_per_rank,
        actual_splits,
        elements_per_slice,
    )
