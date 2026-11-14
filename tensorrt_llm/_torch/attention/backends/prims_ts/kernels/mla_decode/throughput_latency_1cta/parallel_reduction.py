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

from ...separate_reduction import (
    finalize_log2_sum_exp,
    normalized_lse_weight,
    attention_sink_scale,
)
from ..helpers.constants import (
    SPLIT_REDUCTION_SCALE_BARRIER_ID,
    WARP_REDUCTION_BFLY_DISTANCES,
)
from ..helpers.tile import runtime_split_count
from ..helpers.math import ceil_div, final_lse, ffma2
from ..helpers.ops import (
    fmax_f32,
    warp_reduce_max_f32,
    warp_reduce_sum_f32,
)
from ..helpers.query import flat_query_row_state, public_query_flat_row
from ..helpers.tile import runtime_request_publishes
from ..parallel_reduction_topology import ParallelReductionTopology
from .config import MlaConfig


GMEM_REDUCTION_WARP_LANES = 32
GMEM_REDUCTION_WARPS_PER_CTA = 4
# A runtime split count runs in the smallest fully unrolled window that holds
# it; only the slots above the previous window are predicated. One compiled
# reducer thus serves every count. Keep the set short: each window adds code
# that a launch of only a few short-lived CTAs must fetch.
GMEM_REDUCTION_RUNTIME_SPLIT_WINDOWS = (8, 16, 32, 48, 64, 96, 128)
# Counts above the largest window walk fixed unrolled chunks. Clamped loads keep
# a chunk's partials in flight together instead of one short unroll at a time.
GMEM_REDUCTION_RUNTIME_SPLIT_CHUNK = 16


def _runtime_split_windows(capacity: int) -> tuple[tuple[int, int], ...]:
    """Return (floor, window) pairs covering runtime counts up to ``capacity``."""

    sizes = [w for w in GMEM_REDUCTION_RUNTIME_SPLIT_WINDOWS if w < capacity]
    if capacity <= GMEM_REDUCTION_RUNTIME_SPLIT_WINDOWS[-1]:
        sizes.append(capacity)
    return tuple(zip([0, *sizes[:-1]], sizes, strict=True))


# The parallel reducer always uses 128 threads. Q8/Q16/Q32 assign one scalar
# from a 128-element output band to each thread. Q64 uses up to one BF16 vec8
# per thread, but caps a slice at four D-per-CTA rows so the four warps can
# still compute one row's shared LSE state each.
PARALLEL_GMEM_REDUCTION_THREADS = 128
# Register budget for prefetching each thread's split partials (BF16 elements).
_MAX_PREFETCH_PARTIAL_ELEMENTS = 32
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
        and cfg.num_ctas_per_seq_kv >= 2
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
    attention_sinks=None,
    finalize_attention: cutlass.Constexpr[bool] = False,
    output_quant: cutlass.Constexpr = None,
    output_quant_args=None,
    requested_splits=None,
    request_is_active=True,
):
    """Reduce G1 partials with the compact row-shared schedule."""

    split_limit = (
        Int32(cfg.num_ctas_per_seq_kv)
        if requested_splits is None
        else runtime_split_count(requested_splits)
    )

    slice_idx, block_idx_y, batch_idx = cute.arch.block_idx()
    thread_idx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = thread_idx % Int32(GMEM_REDUCTION_WARP_LANES)
    if cutlass.const_expr(finalize_attention):
        # Sparse slices contain complete heads. Map the head group back
        # to its physical producer tile; row validity masks a final odd head.
        heads_per_slice = elements_per_slice // cfg.head_dim_v
        slices_per_q_tile = cfg.tile_size_q // heads_per_slice
        q_idx = slice_idx // Int32(slices_per_q_tile)
        slice_idx = slice_idx % Int32(slices_per_q_tile)
        head_dim_cta_idx = Int32(0)
    else:
        head_dim_cta_idx = block_idx_y % Int32(cfg.num_ctas_per_head_dim)
        q_idx = block_idx_y // Int32(cfg.num_ctas_per_head_dim)
    elements_per_slice = _parallel_reduction_effective_slice_elements(
        cfg, elements_per_slice
    )
    rows_per_slice = ceil_div(elements_per_slice, cfg.head_dim_per_cta_v)
    reducer_threads = parallel_gmem_reduction_threads(elements_per_slice)
    output_elements_per_thread = ceil_div(elements_per_slice, reducer_threads)
    slice_element_base = slice_idx * Int32(elements_per_slice)

    thread_elem_offset = thread_idx * Int32(output_elements_per_thread)
    flat_element = slice_element_base + thread_elem_offset
    head_idx = flat_element // Int32(cfg.head_dim_per_cta_v)
    dim_in_cta = flat_element - head_idx * Int32(cfg.head_dim_per_cta_v)
    row_in_slice = (flat_element - slice_element_base) // Int32(cfg.head_dim_per_cta_v)
    dim_idx = head_dim_cta_idx * Int32(cfg.head_dim_per_cta_v) + dim_in_cta
    output_query_row_state = flat_query_row_state(
        head_idx,
        q_idx,
        cfg.tile_size_q,
        cfg.logical_num_heads_q,
        cfg.logical_seq_len_q,
        cu_seqlens_q=cu_seqlens_q,
        batch_idx=batch_idx,
    )
    valid_output = (
        thread_elem_offset < Int32(elements_per_slice)
        and head_idx < Int32(cfg.num_heads_q)
        and dim_idx < Int32(cfg.head_dim_v)
        and output_query_row_state[4]
    )
    # Sparse finalization must publish even if the primary source is empty:
    # the extra source may be live, and two empty sources produce zero output.
    if cutlass.const_expr(not finalize_attention):
        valid_output = valid_output & request_is_active
    # Issue every split's partial-O loads before the statistics round trip so
    # both global reads overlap. Invalid threads read the always-mapped first
    # element; masked splits are still skipped at use, never multiplied. Only
    # a fixed split count knows every slot at compile time.
    prefetch_partials = cutlass.const_expr(
        requested_splits is None
        and cfg.num_ctas_per_seq_kv * output_elements_per_thread
        <= _MAX_PREFETCH_PARTIAL_ELEMENTS
    )
    if cutlass.const_expr(prefetch_partials):
        prefetch_row = _split_o_row_base_offset(
            cfg, batch_idx, q_idx, head_idx, dim_idx
        )
        prefetch_ptr = acc_output.iterator.raw_ptr() + (
            prefetch_row if valid_output else Int64(0)
        )
        prefetched = cutlass.Array(
            acc_output.element_type,
            cfg.num_ctas_per_seq_kv * output_elements_per_thread,
            space=cutlass.AddressSpace.rmem,
        )
        for split_i in cutlass.range_constexpr(cfg.num_ctas_per_seq_kv):
            split_ptr = prefetch_ptr + Int64(split_i * cfg.head_dim_v)
            if cutlass.const_expr(output_elements_per_thread == 1):
                prefetched[split_i] = split_ptr.load()
            else:
                values = split_ptr.load(
                    count=output_elements_per_thread,
                    alignment=output_elements_per_thread * cfg.partial_o_dtype_bytes,
                )
                for elem_i in cutlass.range_constexpr(output_elements_per_thread):
                    prefetched[split_i * output_elements_per_thread + elem_i] = values[
                        elem_i
                    ]

    smem_scale = cutlass.Array(
        Float32,
        rows_per_slice * cfg.num_ctas_per_seq_kv,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )

    # One statistics warp per covered row stripes over split slots, then
    # publishes normalized scales for all output threads to reuse.
    # Keep branch-local names separate from the output phase below so CuTe
    # DSL 4.8 does not thread undefined values through inactive stats warps.
    if warp_idx < Int32(rows_per_slice):
        stats_row = warp_idx
        stats_element = slice_element_base + stats_row * Int32(cfg.head_dim_per_cta_v)
        stats_head_idx = stats_element // Int32(cfg.head_dim_per_cta_v)
        stats_query_row = flat_query_row_state(
            stats_head_idx,
            q_idx,
            cfg.tile_size_q,
            cfg.logical_num_heads_q,
            cfg.logical_seq_len_q,
            cu_seqlens_q=cu_seqlens_q,
            batch_idx=batch_idx,
        )
        valid_stats = stats_head_idx < Int32(cfg.num_heads_q) and stats_query_row[4]
        if cutlass.const_expr(not finalize_attention):
            valid_stats = valid_stats & request_is_active
        if valid_stats:
            row_lse = acc_lse[stats_head_idx, None, q_idx, batch_idx]
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
                if (
                    cutlass.const_expr(lane_slot_i == 0)
                    or Int32(lane_slot_i * GMEM_REDUCTION_WARP_LANES) < split_limit
                ):
                    stats_split_idx = lane_idx + Int32(
                        lane_slot_i * GMEM_REDUCTION_WARP_LANES
                    )
                    lane_lse[lane_slot_i] = (
                        Float32(row_lse[stats_split_idx])
                        if stats_split_idx < split_limit
                        else Float32(-Float32.inf)
                    )
                    lse_max = fmax_f32(lse_max, lane_lse[lane_slot_i])

            lse_max = warp_reduce_max_f32(lse_max)
            lse_max = lse_max if lse_max != Float32(-Float32.inf) else Float32(0.0)
            lse_sum = Float32(0.0)
            for lane_slot_i in cutlass.range_constexpr(lse_per_lane):
                if (
                    cutlass.const_expr(lane_slot_i == 0)
                    or Int32(lane_slot_i * GMEM_REDUCTION_WARP_LANES) < split_limit
                ):
                    lse_sum += cute.math.exp2(
                        lane_lse[lane_slot_i] - lse_max,
                        fastmath=True,
                    )
            lse_sum = warp_reduce_sum_f32(lse_sum)
            global_lse = finalize_log2_sum_exp(lse_max, lse_sum)
            stats_sink_scale = Float32(1)
            if cutlass.const_expr(finalize_attention and attention_sinks is not None):
                stats_sink_scale = attention_sink_scale(
                    global_lse, Float32(attention_sinks[stats_query_row[1]])
                )
            if (
                lane_idx == Int32(0)
                and head_dim_cta_idx == Int32(0)
                and stats_element % Int32(cfg.head_dim_per_cta_v) == Int32(0)
            ):
                stats_output_query_row = public_query_flat_row(
                    cfg,
                    stats_query_row[0],
                    batch_idx,
                    cu_seqlens_q,
                )
                if cutlass.const_expr(lse is not None):
                    (lse.iterator.raw_ptr() + stats_output_query_row).store(
                        final_lse(global_lse, cfg.lse_base_on_e or finalize_attention)
                    )

            for lane_slot_i in cutlass.range_constexpr(lse_per_lane):
                if (
                    cutlass.const_expr(lane_slot_i == 0)
                    or Int32(lane_slot_i * GMEM_REDUCTION_WARP_LANES) < split_limit
                ):
                    weight_split_idx = lane_idx + Int32(
                        lane_slot_i * GMEM_REDUCTION_WARP_LANES
                    )
                    if weight_split_idx < split_limit:
                        stats_weight = normalized_lse_weight(
                            lane_lse[lane_slot_i], global_lse
                        )
                        if cutlass.const_expr(finalize_attention):
                            stats_weight *= stats_sink_scale
                        smem_scale[
                            stats_row * Int32(cfg.num_ctas_per_seq_kv) + weight_split_idx
                        ] = stats_weight

    prims.barrier_cta_sync(
        barrier_id=SPLIT_REDUCTION_SCALE_BARRIER_ID,
        thread_count=reducer_threads,
    )

    if valid_output:
        output_acc = cutlass.Array(
            Float32,
            output_elements_per_thread,
            space=cutlass.AddressSpace.rmem,
        )
        for elem_i in cutlass.range_constexpr(output_elements_per_thread):
            output_acc[elem_i] = Float32(0.0)
        if cutlass.const_expr(prefetch_partials):
            for split_i in cutlass.range_constexpr(cfg.num_ctas_per_seq_kv):
                scale = Float32(
                    smem_scale[
                        row_in_slice * Int32(cfg.num_ctas_per_seq_kv) + Int32(split_i)
                    ]
                )
                # Sparse inactive splits may leave their O partial untouched.
                if not finalize_attention or scale > Float32(0):
                    base = split_i * output_elements_per_thread
                    if cutlass.const_expr(output_elements_per_thread == 1):
                        output_acc[0] += Float32(prefetched[base]) * scale
                    else:
                        for elem_i in cutlass.range_constexpr(
                            0, output_elements_per_thread, 2
                        ):
                            output_acc[elem_i], output_acc[elem_i + 1] = ffma2(
                                (
                                    Float32(prefetched[base + elem_i]),
                                    Float32(prefetched[base + elem_i + 1]),
                                ),
                                (scale, scale),
                                (output_acc[elem_i], output_acc[elem_i + 1]),
                            )
        else:
            # Scratch strides remain static for the prepared capacity. Form the
            # 64-bit row base once to avoid repeated address linearization.
            # Runtime counts up to the largest window stay on an unrolled path;
            # only counts above it use the runtime chunk loop. Padding slots are
            # never read: predicated slots clamp to the last live split.
            acc_row_ptr = acc_output.iterator.raw_ptr() + _split_o_row_base_offset(
                cfg,
                batch_idx,
                q_idx,
                head_idx,
                dim_idx,
            )
            # Pair FP32 channels while preserving each channel's split order.
            # Static splits unroll their exact count. A runtime count selects the
            # smallest window above it; slots at or below window_floor are live.
            if cutlass.const_expr(requested_splits is None):
                windows = ((cfg.num_ctas_per_seq_kv - 1, cfg.num_ctas_per_seq_kv),)
            else:
                windows = _runtime_split_windows(cfg.num_ctas_per_seq_kv)
            last_split = split_limit - Int32(1)
            for window_i in cutlass.range_constexpr(len(windows)):
                window_floor = windows[window_i][0]
                unrolled_splits = windows[window_i][1]
                if split_limit > Int32(window_floor) and split_limit <= Int32(
                    unrolled_splits
                ):
                    for split_i in cutlass.range_constexpr(unrolled_splits):
                        split_idx = Int32(split_i)
                        load_idx = split_idx
                        if cutlass.const_expr(split_i > window_floor):
                            # Clamp to a live split so the load needs no branch;
                            # its zero weight drops the duplicate contribution.
                            load_idx = split_idx if split_idx < split_limit else last_split
                        scale = Float32(
                            smem_scale[
                                row_in_slice * Int32(cfg.num_ctas_per_seq_kv) + load_idx
                            ]
                        )
                        if cutlass.const_expr(split_i > window_floor):
                            scale = scale if split_idx < split_limit else Float32(0)
                        # Sparse inactive splits may leave their O partial untouched.
                        if not finalize_attention or scale > Float32(0):
                            split_ptr = acc_row_ptr + Int64(load_idx) * Int64(
                                cfg.head_dim_v
                            )
                            if cutlass.const_expr(output_elements_per_thread == 1):
                                output_acc[0] += Float32(split_ptr.load()) * scale
                            else:
                                partial_o = split_ptr.load(
                                    count=output_elements_per_thread,
                                    alignment=output_elements_per_thread
                                    * cfg.partial_o_dtype_bytes,
                                ).to(Float32)
                                for elem_i in cutlass.range_constexpr(
                                    0, output_elements_per_thread, 2
                                ):
                                    output_acc[elem_i], output_acc[elem_i + 1] = ffma2(
                                        (partial_o[elem_i], partial_o[elem_i + 1]),
                                        (scale, scale),
                                        (output_acc[elem_i], output_acc[elem_i + 1]),
                                    )
            if cutlass.const_expr(windows[len(windows) - 1][1] < cfg.num_ctas_per_seq_kv):
                if split_limit > Int32(windows[len(windows) - 1][1]):
                    chunk = GMEM_REDUCTION_RUNTIME_SPLIT_CHUNK
                    num_chunks = (split_limit + Int32(chunk - 1)) // Int32(chunk)
                    for chunk_i in cutlass.range(num_chunks):
                        for lane_i in cutlass.range_constexpr(chunk):
                            split_idx = chunk_i * Int32(chunk) + Int32(lane_i)
                            load_idx = split_idx if split_idx < split_limit else last_split
                            scale = Float32(
                                smem_scale[
                                    row_in_slice * Int32(cfg.num_ctas_per_seq_kv) + load_idx
                                ]
                            )
                            scale = scale if split_idx < split_limit else Float32(0)
                            if not finalize_attention or scale > Float32(0):
                                split_ptr = acc_row_ptr + Int64(load_idx) * Int64(
                                    cfg.head_dim_v
                                )
                                if cutlass.const_expr(output_elements_per_thread == 1):
                                    output_acc[0] += Float32(split_ptr.load()) * scale
                                else:
                                    partial_o = split_ptr.load(
                                        count=output_elements_per_thread,
                                        alignment=output_elements_per_thread
                                        * cfg.partial_o_dtype_bytes,
                                    ).to(Float32)
                                    for elem_i in cutlass.range_constexpr(
                                        0, output_elements_per_thread, 2
                                    ):
                                        output_acc[elem_i], output_acc[elem_i + 1] = ffma2(
                                            (partial_o[elem_i], partial_o[elem_i + 1]),
                                            (scale, scale),
                                            (output_acc[elem_i], output_acc[elem_i + 1]),
                                        )

        output_query_row = public_query_flat_row(
            cfg,
            output_query_row_state[0],
            batch_idx,
            cu_seqlens_q,
        )
        out_elem_offset = _output_element_offset(cfg, output_query_row, dim_idx)
        if cutlass.const_expr(output_quant is not None):
            output_quant.store(output_quant_args, output_acc, output_query_row, dim_idx)
        elif cutlass.const_expr(output_elements_per_thread == 1):
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
    requested_splits=None,
    request_is_active=True,
):
    """Reduce split-KV with one cooperative statistics warp per output row.

    Every cluster rank owns at most four D-per-CTA rows and a cyclic subset of
    split slots. One warp calculates each row's statistics, then all output
    threads reuse the normalized scales while reducing BF16 fragments. Rank
    zero repeats the same row-shared scheme for the final DSMEM merge.
    """

    split_limit = (
        Int32(cfg.num_ctas_per_seq_kv)
        if requested_splits is None
        else runtime_split_count(requested_splits)
    )

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
        ) = flat_query_row_state(
            stats_head_idx,
            q_idx,
            cfg.tile_size_q,
            cfg.logical_num_heads_q,
            cfg.logical_seq_len_q,
            cu_seqlens_q=cu_seqlens_q,
            batch_idx=batch_idx,
        )
        valid_stats = stats_head_idx < Int32(cfg.num_heads_q) and valid_stats_row
        valid_stats = valid_stats & request_is_active
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
            # Form the row view only after validating the flat output row.
            # Invalid/padded rows publish a neutral state without an acc_lse
            # address or load.
            row_lse = acc_lse[stats_head_idx, None, q_idx, batch_idx]
            local_max = neg_inf
            for lane_slot_i in cutlass.range_constexpr(lse_per_lane):
                local_slot_idx = lane_idx + Int32(
                    lane_slot_i * GMEM_REDUCTION_WARP_LANES
                )
                split_idx = local_slot_idx * Int32(cluster_size) + cluster_rank
                if (
                    local_slot_idx < Int32(local_slots)
                    and split_idx < Int32(actual_splits)
                    and split_idx < split_limit
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
    ) = flat_query_row_state(
        head_idx,
        q_idx,
        cfg.tile_size_q,
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
    valid_output = valid_output & request_is_active
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
        # Each rank traverses its live cyclic split prefix, independent of capacity.
        for local_slot_i in cutlass.range(
            cute.ceil_div(split_limit - cluster_rank, cluster_size), unroll=32
        ):
            split_idx = Int32(local_slot_i * cluster_size) + cluster_rank
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
                    alignment=output_elements_per_thread * cfg.partial_o_dtype_bytes,
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
            ) = flat_query_row_state(
                stats_head_idx,
                q_idx,
                cfg.tile_size_q,
                cfg.logical_num_heads_q,
                cfg.logical_seq_len_q,
                cu_seqlens_q=cu_seqlens_q,
                batch_idx=batch_idx,
            )
            valid_stats = stats_head_idx < Int32(cfg.num_heads_q) and valid_stats_row
            valid_stats = valid_stats & request_is_active
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
                if cutlass.const_expr(lse is not None):
                    (lse.iterator.raw_ptr() + output_query_row).store(
                        final_lse(smem_global_lse[row_in_slice], cfg.lse_base_on_e)
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


FUSED_SPLIT_FINISH_ELEMENTS_PER_THREAD = 8


@cute.jit
def _fused_finish_row_stats(
    lse,
    acc_lse,
    cu_seqlens_q,
    cfg,
    attention_sinks,
    batch_idx: Int32,
    q_idx: Int32,
    scales,
    stats_row: Int32,
    lane_idx: Int32,
    publish_lse=True,
):
    """One head row's split weights and public LSE, as the finalizing reducer forms them."""

    splits = cfg.num_ctas_per_seq_kv
    lse_per_lane = ceil_div(splits, GMEM_REDUCTION_WARP_LANES)
    stats_query_row = flat_query_row_state(
        stats_row,
        q_idx,
        cfg.tile_size_q,
        cfg.logical_num_heads_q,
        cfg.logical_seq_len_q,
        cu_seqlens_q=cu_seqlens_q,
        batch_idx=batch_idx,
    )
    valid_stats = (
        stats_row < Int32(cfg.tile_size_q)
        and stats_row < Int32(cfg.num_heads_q)
        and stats_query_row[4]
    )
    if valid_stats:
        row_lse = acc_lse[stats_row, None, q_idx, batch_idx]
        lane_lse = cutlass.Array(Float32, lse_per_lane, space=cutlass.AddressSpace.rmem)
        lse_max = Float32(-Float32.inf)
        for lane_slot_i in cutlass.range_constexpr(lse_per_lane):
            stats_split_idx = lane_idx + Int32(lane_slot_i * GMEM_REDUCTION_WARP_LANES)
            lane_lse[lane_slot_i] = (
                Float32(row_lse[stats_split_idx])
                if stats_split_idx < Int32(splits)
                else Float32(-Float32.inf)
            )
            lse_max = fmax_f32(lse_max, lane_lse[lane_slot_i])
        lse_max = warp_reduce_max_f32(lse_max)
        lse_max = lse_max if lse_max != Float32(-Float32.inf) else Float32(0.0)
        lse_sum = Float32(0.0)
        for lane_slot_i in cutlass.range_constexpr(lse_per_lane):
            lse_sum += cute.math.exp2(lane_lse[lane_slot_i] - lse_max, fastmath=True)
        lse_sum = warp_reduce_sum_f32(lse_sum)
        global_lse = finalize_log2_sum_exp(lse_max, lse_sum)
        stats_sink_scale = Float32(1)
        if cutlass.const_expr(attention_sinks is not None):
            stats_sink_scale = attention_sink_scale(
                global_lse, Float32(attention_sinks[stats_query_row[1]])
            )
        if lane_idx == Int32(0) and publish_lse:
            if cutlass.const_expr(lse is not None):
                stats_output_query_row = public_query_flat_row(
                    cfg, stats_query_row[0], batch_idx, cu_seqlens_q
                )
                (lse.iterator.raw_ptr() + stats_output_query_row).store(
                    final_lse(global_lse, True)
                )
        for lane_slot_i in cutlass.range_constexpr(lse_per_lane):
            weight_split_idx = lane_idx + Int32(lane_slot_i * GMEM_REDUCTION_WARP_LANES)
            if weight_split_idx < Int32(splits):
                stats_weight = normalized_lse_weight(lane_lse[lane_slot_i], global_lse)
                stats_weight *= stats_sink_scale
                scales[stats_row * Int32(splits) + weight_split_idx] = stats_weight


@cute.jit
def _fused_finish_prefetch(
    acc_output,
    cfg,
    batch_idx: Int32,
    q_idx: Int32,
    flat_element: Int32,
    dim_base: Int32,
    dims: cutlass.Constexpr[int],
):
    """Every split's partial of one fragment, read before its weights are known.

    Rows past the tile's heads read the always-mapped first element instead; the merge skips
    them and every split whose weight is zero, so those values are never used.
    """
    splits = cfg.num_ctas_per_seq_kv
    elements = FUSED_SPLIT_FINISH_ELEMENTS_PER_THREAD
    head_idx = flat_element // Int32(dims)
    dim_idx = dim_base + (flat_element - head_idx * Int32(dims))
    valid = head_idx < Int32(cfg.tile_size_q) and head_idx < Int32(cfg.num_heads_q)
    row_ptr = acc_output.iterator.raw_ptr() + (
        _split_o_row_base_offset(cfg, batch_idx, q_idx, head_idx, dim_idx)
        if valid
        else Int64(0)
    )
    values = cutlass.Array(Float32, splits * elements, space=cutlass.AddressSpace.rmem)
    for split_i in cutlass.range_constexpr(splits):
        partial_o = (row_ptr + Int64(split_i * cfg.head_dim_v)).load(
            count=elements, alignment=elements * cfg.partial_o_dtype_bytes
        )
        for elem_i in cutlass.range_constexpr(elements):
            values[split_i * elements + elem_i] = Float32(partial_o[elem_i])
    return values


@cute.jit
def _fused_finish_fragment(
    output,
    acc_output,
    cu_seqlens_q,
    cfg,
    batch_idx: Int32,
    q_idx: Int32,
    scales,
    flat_element: Int32,
    dim_base: Int32,
    dims: cutlass.Constexpr[int],
    prefetched=None,
):
    """One 8-element output fragment summed over the splits in split order.

    ``flat_element`` counts the ``dims`` channels from ``dim_base`` of consecutive heads;
    ``prefetched`` holds its splits' partials when they were read ahead.
    """

    splits = cfg.num_ctas_per_seq_kv
    elements = FUSED_SPLIT_FINISH_ELEMENTS_PER_THREAD
    head_idx = flat_element // Int32(dims)
    dim_idx = dim_base + (flat_element - head_idx * Int32(dims))
    output_query_row_state = flat_query_row_state(
        head_idx,
        q_idx,
        cfg.tile_size_q,
        cfg.logical_num_heads_q,
        cfg.logical_seq_len_q,
        cu_seqlens_q=cu_seqlens_q,
        batch_idx=batch_idx,
    )
    valid_output = (
        head_idx < Int32(cfg.tile_size_q)
        and head_idx < Int32(cfg.num_heads_q)
        and output_query_row_state[4]
    )
    if valid_output:
        acc_row_ptr = acc_output.iterator.raw_ptr() + _split_o_row_base_offset(
            cfg, batch_idx, q_idx, head_idx, dim_idx
        )
        output_acc = cutlass.Array(Float32, elements, space=cutlass.AddressSpace.rmem)
        for elem_i in cutlass.range_constexpr(elements):
            output_acc[elem_i] = Float32(0.0)
        for split_i in cutlass.range_constexpr(splits):
            scale = Float32(scales[head_idx * Int32(splits) + Int32(split_i)])
            # Sparse inactive splits may leave their O partial untouched.
            if scale > Float32(0):
                if cutlass.const_expr(prefetched is None):
                    partial_o = (acc_row_ptr + Int64(split_i * cfg.head_dim_v)).load(
                        count=elements, alignment=elements * cfg.partial_o_dtype_bytes
                    ).to(Float32)
                else:
                    partial_o = cutlass.Array(
                        Float32, elements, space=cutlass.AddressSpace.rmem
                    )
                    for elem_i in cutlass.range_constexpr(elements):
                        partial_o[elem_i] = prefetched[split_i * elements + elem_i]
                for elem_i in cutlass.range_constexpr(0, elements, 2):
                    output_acc[elem_i], output_acc[elem_i + 1] = ffma2(
                        (partial_o[elem_i], partial_o[elem_i + 1]),
                        (scale, scale),
                        (output_acc[elem_i], output_acc[elem_i + 1]),
                    )
        output_query_row = public_query_flat_row(
            cfg, output_query_row_state[0], batch_idx, cu_seqlens_q
        )
        out_elem_offset = _output_element_offset(cfg, output_query_row, dim_idx)
        output_regs = cutlass.Array(output.element_type, elements, space=cutlass.AddressSpace.rmem)
        for elem_i in cutlass.range_constexpr(elements):
            output_regs[elem_i] = output.element_type(output_acc[elem_i])
        (output.iterator.raw_ptr() + out_elem_offset).store(
            output_regs.data_ptr().load(count=elements, alignment=elements * cfg.o_dtype_bytes),
            alignment=elements * cfg.o_dtype_bytes,
        )


@cute.jit
def run_fused_split_finish(
    output,
    lse,
    acc_output,
    acc_lse,
    cu_seqlens_q,
    cfg,
    attention_sinks,
    batch_idx: Int32,
    q_idx: Int32,
    scales,
    threads: cutlass.Constexpr[int],
    barrier_id: cutlass.Constexpr[int],
    dim_base=None,
    dims: cutlass.Constexpr[int] = None,
    publish_lse=True,
):
    """Merge one query tile's split partials in the CTA that finished them last.

    ``cfg`` is the finalizing reducer's configuration (complete D heads per slice, see
    ``_parallel_reduction_setup``). Head statistics and output elements are formed exactly as
    ``_run_parallel_gmem_reduction_g1_shared_stats`` forms them with ``finalize_attention``,
    so the output equals the standalone reducer's: warp ``w`` publishes the split weights of
    rows ``w, w + warps, ...`` into ``scales`` (shared FP32, ``tile_size_q * splits``), then
    each thread sums 8-element fragments over the splits in split order. All ``threads``
    threads of the CTA call it. ``dim_base`` and ``dims`` restrict the merge to those output
    channels; ``publish_lse`` gates the LSE stores.
    """
    if cutlass.const_expr(dims is None):
        dims = cfg.head_dim_v
        dim_base = Int32(0)

    warps = threads // GMEM_REDUCTION_WARP_LANES
    thread_idx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = thread_idx % Int32(GMEM_REDUCTION_WARP_LANES)
    rows = cfg.tile_size_q
    if cutlass.const_expr(isinstance(cfg.logical_seq_len_q, int)):
        # Later rows of a tile hold no logical query (every access of theirs is predicated
        # off), so the merge's rounds cover the live rows only.
        rows = min(rows, cfg.logical_num_heads_q * cfg.logical_seq_len_q)
    elements = FUSED_SPLIT_FINISH_ELEMENTS_PER_THREAD
    rounds = ceil_div(rows * dims, threads * elements)
    prefetched = None
    if cutlass.const_expr(rounds == 1):
        # A single round's partials are read before the weights' round trip, so both
        # reads overlap as in the standalone reducer.
        prefetched = _fused_finish_prefetch(
            acc_output, cfg, batch_idx, q_idx, thread_idx * Int32(elements), dim_base, dims
        )
    for row_i in cutlass.range_constexpr(ceil_div(rows, warps)):
        _fused_finish_row_stats(
            lse,
            acc_lse,
            cu_seqlens_q,
            cfg,
            attention_sinks,
            batch_idx,
            q_idx,
            scales,
            warp_idx + Int32(row_i * warps),
            lane_idx,
            publish_lse,
        )
    prims.barrier_cta_sync(barrier_id=barrier_id, thread_count=threads)
    for chunk_i in cutlass.range_constexpr(rounds):
        _fused_finish_fragment(
            output,
            acc_output,
            cu_seqlens_q,
            cfg,
            batch_idx,
            q_idx,
            scales,
            (Int32(chunk_i * threads) + thread_idx) * Int32(elements),
            dim_base,
            dims,
            prefetched,
        )


def _butterfly_lane_sum(values):
    """Sum ``values`` exactly as a warp butterfly sum forms it when lane ``i`` holds
    ``values[i]`` and every other lane zero.

    Each lane of such a reduction ends with the same total (addition is commutative), so
    this evaluates lane 0's addition tree; a zero lane adds nothing because every value
    here is a nonnegative ``exp2`` result.
    """

    def lane_value(lane, step):
        if step == 0:
            return values[lane] if lane < len(values) else None
        own = lane_value(lane, step - 1)
        peer = lane_value(lane ^ WARP_REDUCTION_BFLY_DISTANCES[step - 1], step - 1)
        if own is None:
            return peer
        if peer is None:
            return own
        return Float32(own + peer)

    return lane_value(0, len(WARP_REDUCTION_BFLY_DISTANCES))


@cute.jit
def run_cluster_push_finish(
    output,
    lse,
    push_smem,
    push_barrier,
    cfg,
    reducer_cfg,
    attention_sinks,
    batch_idx: Int32,
    q_idx: Int32,
    dim_base: Int32,
    publish_lse,
    threads: cutlass.Constexpr[int],
):
    """Merge the split partials pushed into this CTA's shared buffer, as the reducer would.

    ``push_smem`` holds, for ``cfg.num_ctas_per_seq_kv`` splits and ``cfg.cluster_split_rows``
    rows, the BF16 partial O of the CTA's ``cfg.head_dim_per_cta_v`` V columns, then each
    split's FP32 LSE (``cfg.cluster_split_push_bytes``); ``push_barrier`` completes once every
    byte has landed. Each thread
    merges ``FUSED_SPLIT_FINISH_ELEMENTS_PER_THREAD`` columns of one row and forms that
    row's statistics itself, in the order ``_run_parallel_gmem_reduction_g1_shared_stats``
    reduces them across a warp, so the result matches the finalizing reducer bitwise. A
    thread loads its sink before the wait and waits only if it owns columns. The first
    column's thread stores the row's LSE when ``publish_lse``. All ``threads`` threads of
    the CTA call it.
    """
    splits = cfg.num_ctas_per_seq_kv
    rows = cfg.cluster_split_rows
    dims = cfg.head_dim_per_cta_v
    elements = FUSED_SPLIT_FINISH_ELEMENTS_PER_THREAD
    o_bytes = splits * rows * dims * cfg.partial_o_dtype_bytes
    base = push_smem.data_ptr().toint(Int64)
    thread_idx, _, _ = cute.arch.thread_idx()

    def word_ptr(byte_offset, dtype):
        return cutlass.inttoptr(base + Int64(byte_offset), mem_space=3, dtype=dtype)

    for chunk_i in cutlass.range_constexpr(ceil_div(rows * dims, threads * elements)):
        flat_element = (Int32(chunk_i * threads) + thread_idx) * Int32(elements)
        row = flat_element // Int32(dims)
        column = flat_element - row * Int32(dims)
        if row < Int32(rows):
            state = flat_query_row_state(
                row,
                q_idx,
                reducer_cfg.tile_size_q,
                reducer_cfg.logical_num_heads_q,
                reducer_cfg.logical_seq_len_q,
                cu_seqlens_q=None,
                batch_idx=batch_idx,
            )
            if row < Int32(reducer_cfg.num_heads_q) and state[4]:
                sink = Float32(0)
                if cutlass.const_expr(attention_sinks is not None):
                    sink = Float32(attention_sinks[state[1]])
                out_row = public_query_flat_row(reducer_cfg, state[0], batch_idx, None)
                out_elem_offset = _output_element_offset(reducer_cfg, out_row, dim_base + column)
                ready = prims.mbarrier_try_wait_parity(push_barrier, 0, time_limit=0)
                while not ready:
                    ready = prims.mbarrier_try_wait_parity(
                        push_barrier, 0, time_limit=10_000_000
                    )
                split_lse = [
                    word_ptr(Int32(o_bytes) + (Int32(split_i * rows) + row) * Int32(4), Float32).load()
                    for split_i in range(splits)
                ]
                split_words = [
                    word_ptr(
                        ((Int32(split_i * rows) + row) * Int32(dims) + column)
                        * Int32(cfg.partial_o_dtype_bytes),
                        Int32,
                    ).load(count=4, alignment=16)
                    for split_i in range(splits)
                ]
                lse_max = Float32(-Float32.inf)
                for split_i in cutlass.range_constexpr(splits):
                    lse_max = fmax_f32(lse_max, split_lse[split_i])
                lse_max = lse_max if lse_max != Float32(-Float32.inf) else Float32(0.0)
                lse_sum = _butterfly_lane_sum(
                    [
                        Float32(cute.math.exp2(split_lse[split_i] - lse_max, fastmath=True))
                        for split_i in range(splits)
                    ]
                )
                global_lse = finalize_log2_sum_exp(lse_max, lse_sum)
                sink_scale = Float32(1)
                if cutlass.const_expr(attention_sinks is not None):
                    sink_scale = attention_sink_scale(global_lse, sink)
                if cutlass.const_expr(lse is not None):
                    if publish_lse and column == Int32(0):
                        (lse.iterator.raw_ptr() + out_row).store(final_lse(global_lse, True))
                output_acc = cutlass.Array(Float32, elements, space=cutlass.AddressSpace.rmem)
                for elem_i in cutlass.range_constexpr(elements):
                    output_acc[elem_i] = Float32(0.0)
                for split_i in cutlass.range_constexpr(splits):
                    scale = normalized_lse_weight(split_lse[split_i], global_lse) * sink_scale
                    # Sparse inactive splits push zero partials with a zero weight.
                    if scale > Float32(0):
                        words = split_words[split_i]
                        for word in cutlass.range_constexpr(4):
                            low = prims.mov_b32(Int32(words[word]) << Int32(16), target_type=Float32)
                            high = prims.mov_b32(
                                Int32(words[word]) & Int32(-65536), target_type=Float32
                            )
                            output_acc[2 * word], output_acc[2 * word + 1] = ffma2(
                                (low, high),
                                (scale, scale),
                                (output_acc[2 * word], output_acc[2 * word + 1]),
                            )
                output_regs = cutlass.Array(
                    output.element_type, elements, space=cutlass.AddressSpace.rmem
                )
                for elem_i in cutlass.range_constexpr(elements):
                    output_regs[elem_i] = output.element_type(output_acc[elem_i])
                (output.iterator.raw_ptr() + out_elem_offset).store(
                    output_regs.data_ptr().load(
                        count=elements, alignment=elements * reducer_cfg.o_dtype_bytes
                    ),
                    alignment=elements * reducer_cfg.o_dtype_bytes,
                )


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
    attention_sinks=None,
    finalize_attention: cutlass.Constexpr[bool] = False,
    output_quant: cutlass.Constexpr = None,
    output_quant_args=None,
    requested_splits=None,
):
    """Reduce split-KV partials with row-shared local and peer statistics."""

    if cutlass.const_expr(finalize_attention and cluster_size != 1):
        raise ValueError(
            "final attention output requires single-CTA reduction clusters"
        )
    prims.griddepcontrol(kind=prims.GridDepAction.WAIT)
    # The next PDL launch may start its prologue now; it waits for this
    # grid's completion before reading anything this reducer writes.
    prims.griddepcontrol(kind=prims.GridDepAction.LAUNCH_DEPENDENTS)

    # Reuse the CTA's predicate across the statistics barrier instead of
    # reloading metadata. Sparse finalization can have a live extra source.
    request_is_active = cutlass.Boolean(True)
    if cutlass.const_expr(not finalize_attention):
        request_is_active = runtime_request_publishes(
            cfg, cache_seqs, cute.arch.block_idx()[2]
        )

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
            attention_sinks,
            finalize_attention,
            output_quant,
            output_quant_args,
            requested_splits=requested_splits,
            request_is_active=request_is_active,
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
        requested_splits=requested_splits,
        request_is_active=request_is_active,
    )
