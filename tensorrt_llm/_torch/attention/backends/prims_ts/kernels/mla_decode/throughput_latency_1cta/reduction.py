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

"""Split-KV GMEM reduction helpers for throughput-latency 1CTA MLA."""

import cutlass
import cutlass.cute as cute
from cutlass.experimental import primitives as prims
from cutlass import Float32, Int32, Int64

from .config import MlaConfig
from ..helpers.math import ceil_div
from ..helpers.ops import (
    fmax_f32,
    warp_reduce_max_f32,
    warp_reduce_sum_f32,
    vector_from_scalars,
)
from ..helpers.query import (
    flat_query_row_state,
    public_query_flat_row,
    split_o_element_offset,
)
from ..helpers.tile import runtime_seq_len_kv_for_q


# Baseline split-KV reduction uses one 128-thread CTA.  Each thread owns eight
# BF16 elements, which is one 16B vector and covers a full 512-wide output head
# per CTA.
GMEM_REDUCTION_THREADS = 128
GMEM_REDUCTION_WARP_LANES = 32
GMEM_REDUCTION_WARPS_PER_CTA = GMEM_REDUCTION_THREADS // GMEM_REDUCTION_WARP_LANES
GMEM_REDUCTION_ELEMENTS_PER_THREAD = 8
GMEM_REDUCTION_VECTOR_BYTES = 16
GMEM_REDUCTION_VECTOR_HEAD_DIM = 512

# Keeps-MMA-AB split reduction uses a wider CTA so one slice can cover 512 16B
# vectors and split a large Q/head tile across multiple reducer CTAs.
GMEM_REDUCTION_SLICE_THREADS = 512

# Scalar stats are reduced in 128-wide bands to match the warp-reduction and
# SMEM exchange layout used by the O reduction.
GMEM_REDUCTION_SCALAR_DIM_TILE = 128

# The small-B fast path lets one reducer CTA cover two 512-wide BF16 heads.
GMEM_REDUCTION_HEADS_PER_CTA = 2

# CTA-local barrier used after writer warps publish split-KV softmax scales in
# SMEM and before all reducer threads consume those scales for O accumulation.
GMEM_REDUCTION_SCALE_BARRIER_ID = 4
GMEM_REDUCTION_HEAD_SEGMENTS_PER_CTA = (
    GMEM_REDUCTION_HEADS_PER_CTA
    * GMEM_REDUCTION_VECTOR_HEAD_DIM
    // (GMEM_REDUCTION_THREADS * GMEM_REDUCTION_ELEMENTS_PER_THREAD)
)
GMEM_REDUCTION_Q_ROWS_PER_CTA = (
    GMEM_REDUCTION_THREADS
    * GMEM_REDUCTION_ELEMENTS_PER_THREAD
    // GMEM_REDUCTION_VECTOR_HEAD_DIM
)


def uses_two_head_gmem_reduction(cfg: MlaConfig) -> bool:
    """Return whether one reducer CTA covers two full BF16 output heads."""
    return (
        cfg.head_dim_v == GMEM_REDUCTION_VECTOR_HEAD_DIM
        and cfg.o_dtype_bytes == 2
        and cfg.batch_size <= 2
    )


def uses_q_row_gmem_reduction(cfg: MlaConfig) -> bool:
    """Return whether one reducer CTA covers multiple Q rows for one head."""
    return (
        cfg.seq_len_q >= 2 * GMEM_REDUCTION_Q_ROWS_PER_CTA
        and cfg.seq_len_q % GMEM_REDUCTION_Q_ROWS_PER_CTA == 0
        and cfg.head_dim_v == GMEM_REDUCTION_VECTOR_HEAD_DIM
        and cfg.o_dtype_bytes == 2
    )


def uses_slice_split_gmem_reduction(cfg: MlaConfig) -> bool:
    """Return whether the slice-split reducer applies.

    The keeps-MMA-AB split-KV path writes one Q/head tile of BF16 partial O per
    split.  A 512-thread reducer CTA covers a contiguous slice of rows from that
    tile, with each thread loading/storing one 16B vector.  Multiple reducer CTAs
    can split the row slices for one Q/head tile when there are spare SMs.
    """

    return (
        cfg.kernel_variant == "keeps_mma_ab"
        and cfg.tile_size_q in (64, 128)
        and cfg.head_dim_per_cta_v in (64, 128, 256, 512)
        and cfg.partial_o_dtype_bytes == 2
    )


def slice_split_rows_per_slice(cfg: MlaConfig) -> int:
    """Return rows covered by one 512-thread reducer slice."""

    return (
        GMEM_REDUCTION_SLICE_THREADS
        * GMEM_REDUCTION_ELEMENTS_PER_THREAD
        // cfg.head_dim_per_cta_v
    )


def slice_split_num_slices(cfg: MlaConfig) -> int:
    """Return row-slice count in one keeps-MMA-AB Q/head tile."""

    return ceil_div(cfg.tile_size_q, slice_split_rows_per_slice(cfg))


def slice_split_num_reduction_ctas(
    cfg: MlaConfig,
    seq_len_q,
    batch_size,
    max_active_clusters,
) -> int:
    """Return reducer CTAs per Q/head tile, capped to roughly two SM waves."""

    base_ctas = (
        int(seq_len_q)
        * cfg.num_ctas_for_all_heads
        * cfg.num_ctas_per_head_dim
        * int(batch_size)
    )
    if base_ctas <= 0:
        return 1
    max_ctas_for_reduction = max(1, (int(max_active_clusters) * 2) // base_ctas)
    return min(max_ctas_for_reduction, slice_split_num_slices(cfg))


def gmem_reduction_launch_shape(
    cfg: MlaConfig,
    seq_len_q,
    batch_size,
    lse_width,
    max_active_clusters,
):
    """Return grid, dynamic SMEM bytes, block threads, and CTAs per tile."""

    if uses_slice_split_gmem_reduction(cfg):
        num_reduction_ctas = slice_split_num_reduction_ctas(
            cfg,
            seq_len_q,
            batch_size,
            max_active_clusters,
        )
        return (
            (
                int(seq_len_q) * num_reduction_ctas,
                cfg.num_ctas_for_all_heads * cfg.num_ctas_per_head_dim,
                int(batch_size),
            ),
            0,
            GMEM_REDUCTION_SLICE_THREADS,
            num_reduction_ctas,
        )

    lse_bytes = lse_width // 8
    if uses_two_head_gmem_reduction(cfg):
        return (
            (
                seq_len_q,
                ceil_div(cfg.num_heads_q, GMEM_REDUCTION_HEADS_PER_CTA)
                * GMEM_REDUCTION_HEAD_SEGMENTS_PER_CTA,
                batch_size,
            ),
            GMEM_REDUCTION_HEADS_PER_CTA * cfg.num_ctas_per_seq_kv * lse_bytes,
            GMEM_REDUCTION_THREADS,
            1,
        )
    if uses_q_row_gmem_reduction(cfg):
        return (
            (
                ceil_div(seq_len_q, GMEM_REDUCTION_Q_ROWS_PER_CTA),
                cfg.num_heads_q,
                batch_size,
            ),
            GMEM_REDUCTION_Q_ROWS_PER_CTA * cfg.num_ctas_per_seq_kv * lse_bytes,
            GMEM_REDUCTION_THREADS,
            1,
        )
    return (
        (
            seq_len_q,
            cfg.num_heads_q * ceil_div(cfg.head_dim_v, GMEM_REDUCTION_SCALAR_DIM_TILE),
            batch_size,
        ),
        cfg.num_ctas_per_seq_kv * lse_bytes,
        GMEM_REDUCTION_THREADS,
        1,
    )


@cute.jit
def runtime_seq_len_kv_for_reduction(
    cfg: MlaConfig,
    cache_seqs,
    batch_idx,
    cta_idx_q,
    cu_seqlens_q=None,
):
    """Return the runtime KV length visible to split-KV reduction."""
    return runtime_seq_len_kv_for_q(
        cfg,
        cache_seqs,
        batch_idx,
        cta_idx_q,
        cu_seqlens_q,
    )


@cute.jit
def run_gmem_reduction_kernel(
    kernel,
    output,
    lse,
    acc_output,
    acc_lse,
    cache_seqs,
    cu_seqlens_q,
    cfg,
    num_reduction_ctas,
):
    """Reduce split-KV partial O/LSE rows written by the 1CTA main kernel.

    Each reducer CTA owns either a full head row, a small group of Q rows, or a
    slice of a keeps-MMA-AB head tile.  For its owned rows it loads all split-KV
    partial LSE values, computes the row max, rescales each partial O by
    ``exp2(partial_lse - max_lse)``, sums the weighted O vectors and
    denominators, then writes final O and LSE. Runtime-pruned producers publish
    neutral partials, so the reducer retains configured, compile-time split
    geometry without consuming stale workspace from an earlier graph replay.

    The grid and workspace remain in physical flat-query tile coordinates.
    Every reducer variant maps those coordinates to fixed or compact-ragged
    public storage; inactive tail rows may synchronize and consume workspace
    slots but never publish O or LSE.
    """
    # Pair with the dense kernel PDL signal before reading partials.
    prims.griddepcontrol(kind=prims.GridDepAction.WAIT)
    q_idx, head_dim_tile_idx, batch_idx = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    lane_idx = tidx % Int32(GMEM_REDUCTION_WARP_LANES)

    use_two_head_reduction = uses_two_head_gmem_reduction(cfg)
    use_q_row_reduction = uses_q_row_gmem_reduction(cfg)
    use_slice_split_reduction = uses_slice_split_gmem_reduction(cfg)
    if cutlass.const_expr(use_slice_split_reduction):
        cta_idx_q = q_idx % Int32(cfg.seq_len_q)
        cta_idx_for_reduction = q_idx // Int32(cfg.seq_len_q)
        head_dim_cta_idx = head_dim_tile_idx % Int32(cfg.num_ctas_per_head_dim)
        head_group_idx = head_dim_tile_idx // Int32(cfg.num_ctas_per_head_dim)
        head_base_idx = head_group_idx * Int32(cfg.tile_size_q)
        head_dim_offset = head_dim_cta_idx * Int32(cfg.head_dim_per_cta_v)
        rows_per_slice = slice_split_rows_per_slice(cfg)
        num_slices = slice_split_num_slices(cfg)
        num_slices_per_cta = ceil_div(Int32(num_slices), num_reduction_ctas)
        start_slice_idx = cta_idx_for_reduction * num_slices_per_cta
        end_slice_idx = cute.math.min(
            start_slice_idx + num_slices_per_cta,
            Int32(num_slices),
        )
        acc_ptr = acc_output.iterator.raw_ptr()
        out_ptr = output.iterator.raw_ptr()
        for slice_idx in range(start_slice_idx, end_slice_idx):
            base_vec_offset = tidx * Int32(GMEM_REDUCTION_ELEMENTS_PER_THREAD)
            row_in_slice = base_vec_offset // Int32(cfg.head_dim_per_cta_v)
            dim_in_cta = base_vec_offset - row_in_slice * Int32(cfg.head_dim_per_cta_v)
            row_in_tile = slice_idx * Int32(rows_per_slice) + row_in_slice
            head_idx = head_base_idx + row_in_tile
            dim_idx = head_dim_offset + dim_in_cta
            (
                storage_flat_query_row,
                _,
                _,
                _,
                valid_output_row,
            ) = flat_query_row_state(
                head_idx,
                cta_idx_q,
                cfg.tile_size_q,
                cfg.logical_num_heads_q,
                cfg.logical_seq_len_q,
                cu_seqlens_q=cu_seqlens_q,
                batch_idx=batch_idx,
            )
            valid_row = head_idx < Int32(cfg.num_heads_q) and valid_output_row
            valid_dim = dim_idx < Int32(cfg.head_dim_v)

            if valid_row and valid_dim:
                row_lse = acc_lse[head_idx, None, cta_idx_q, batch_idx]
                local_lse = cutlass.Array(
                    kernel.lse_dtype,
                    cfg.num_ctas_per_seq_kv,
                    space=cutlass.AddressSpace.rmem,
                )
                lse_max = kernel.lse_dtype(-kernel.lse_dtype.inf)
                for split_idx in cutlass.range_constexpr(cfg.num_ctas_per_seq_kv):
                    local_lse[split_idx] = row_lse[split_idx]
                    lse_max = fmax_f32(lse_max, local_lse[split_idx])

                lse_max = (
                    lse_max
                    if lse_max != -kernel.lse_dtype.inf
                    else kernel.lse_dtype(0.0)
                )
                sum_lse = kernel.lse_dtype(0.0)
                for split_idx in cutlass.range_constexpr(cfg.num_ctas_per_seq_kv):
                    sum_lse += cute.math.exp2(
                        local_lse[split_idx] - lse_max,
                        fastmath=True,
                    )
                has_finite_mass = sum_lse == sum_lse and sum_lse != kernel.lse_dtype(
                    0.0
                )
                global_lse = (
                    lse_max + cute.math.log2(sum_lse, fastmath=True)
                    if has_finite_mass
                    else -kernel.lse_dtype.inf
                )
                if dim_in_cta == Int32(0):
                    output_query_row = public_query_flat_row(
                        cfg,
                        storage_flat_query_row,
                        batch_idx,
                        cu_seqlens_q,
                    )
                    (lse.iterator.raw_ptr() + output_query_row).store(global_lse)

                acc_vec = vector_from_scalars(
                    (
                        Float32(0.0),
                        Float32(0.0),
                        Float32(0.0),
                        Float32(0.0),
                        Float32(0.0),
                        Float32(0.0),
                        Float32(0.0),
                        Float32(0.0),
                    ),
                    dtype=Float32,
                )
                for split_idx in cutlass.range_constexpr(cfg.num_ctas_per_seq_kv):
                    scale = Float32(
                        cute.math.exp2(
                            local_lse[split_idx] - global_lse,
                            fastmath=True,
                        )
                        if has_finite_mass
                        else kernel.acc_dtype(0.0)
                    )
                    acc_elem_offset = split_o_element_offset(
                        cfg,
                        batch_idx,
                        cta_idx_q,
                        head_idx,
                        Int32(split_idx),
                        dim_idx,
                    )
                    partial_vec = (
                        (acc_ptr + acc_elem_offset)
                        .load(
                            count=GMEM_REDUCTION_ELEMENTS_PER_THREAD,
                            alignment=GMEM_REDUCTION_VECTOR_BYTES,
                        )
                        .to(Float32)
                    )
                    acc_vec = acc_vec + partial_vec * scale

                output_query_row = public_query_flat_row(
                    cfg,
                    storage_flat_query_row,
                    batch_idx,
                    cu_seqlens_q,
                )
                out_elem_offset = Int64(output_query_row) * Int64(
                    cfg.head_dim_v
                ) + Int64(dim_idx)
                (out_ptr + out_elem_offset).store(
                    acc_vec.to(output.element_type),
                    alignment=GMEM_REDUCTION_VECTOR_BYTES,
                )
        return

    if cutlass.const_expr(use_two_head_reduction):
        # Small-batch BF16 path: one CTA reduces two full output heads and each
        # thread writes 8 elements, matching a 16B GMEM vector store.
        head_group_idx = head_dim_tile_idx // Int32(
            GMEM_REDUCTION_HEAD_SEGMENTS_PER_CTA
        )
        segment_idx = head_dim_tile_idx - head_group_idx * Int32(
            GMEM_REDUCTION_HEAD_SEGMENTS_PER_CTA
        )
        smem_lse_scale = cutlass.Array(
            kernel.lse_dtype,
            GMEM_REDUCTION_HEADS_PER_CTA * cfg.num_ctas_per_seq_kv,
            space=cutlass.AddressSpace.smem,
            alignment=16,
        )

        for row_group in cutlass.range_constexpr(
            ceil_div(GMEM_REDUCTION_HEADS_PER_CTA, GMEM_REDUCTION_WARPS_PER_CTA)
        ):
            local_head_idx = Int32(row_group * GMEM_REDUCTION_WARPS_PER_CTA) + warp_idx
            global_head_idx = (
                head_group_idx * Int32(GMEM_REDUCTION_HEADS_PER_CTA) + local_head_idx
            )
            (
                storage_flat_query_row,
                _,
                _,
                _,
                valid_output_row,
            ) = flat_query_row_state(
                global_head_idx,
                q_idx,
                cfg.tile_size_q,
                cfg.logical_num_heads_q,
                cfg.logical_seq_len_q,
                cu_seqlens_q=cu_seqlens_q,
                batch_idx=batch_idx,
            )
            if (
                local_head_idx < Int32(GMEM_REDUCTION_HEADS_PER_CTA)
                and global_head_idx < Int32(cfg.num_heads_q)
                and valid_output_row
            ):
                row_lse = acc_lse[global_head_idx, None, q_idx, batch_idx]
                # Warp lanes stripe over split-KV LSE slots; each lane owns
                # split_idx = lane + n * warp_size.
                lse_per_thread = ceil_div(
                    cfg.num_ctas_per_seq_kv, GMEM_REDUCTION_WARP_LANES
                )
                local_lse = cutlass.Array(
                    kernel.lse_dtype,
                    lse_per_thread,
                    space=cutlass.AddressSpace.rmem,
                )
                lse_max = kernel.lse_dtype(-kernel.lse_dtype.inf)
                for i in cutlass.range_constexpr(lse_per_thread):
                    split_idx = lane_idx + Int32(i * GMEM_REDUCTION_WARP_LANES)
                    local_lse[i] = (
                        row_lse[split_idx]
                        if cute.elem_less(split_idx, Int32(cfg.num_ctas_per_seq_kv))
                        else -kernel.lse_dtype.inf
                    )
                    lse_max = fmax_f32(lse_max, local_lse[i])
                lse_max = warp_reduce_max_f32(lse_max)
                lse_max = (
                    lse_max
                    if lse_max != -kernel.lse_dtype.inf
                    else kernel.lse_dtype(0.0)
                )
                sum_lse = kernel.lse_dtype(0.0)
                for i in cutlass.range_constexpr(lse_per_thread):
                    sum_lse += cute.math.exp2(local_lse[i] - lse_max, fastmath=True)
                sum_lse = warp_reduce_sum_f32(sum_lse)
                has_finite_mass = sum_lse == sum_lse and sum_lse != kernel.lse_dtype(
                    0.0
                )
                global_lse = (
                    lse_max + cute.math.log2(sum_lse, fastmath=True)
                    if has_finite_mass
                    else -kernel.lse_dtype.inf
                )
                if lane_idx == Int32(0):
                    output_query_row = public_query_flat_row(
                        cfg,
                        storage_flat_query_row,
                        batch_idx,
                        cu_seqlens_q,
                    )
                    (lse.iterator.raw_ptr() + output_query_row).store(global_lse)
                for i in cutlass.range_constexpr(lse_per_thread):
                    split_idx = lane_idx + Int32(i * GMEM_REDUCTION_WARP_LANES)
                    if cute.elem_less(split_idx, Int32(cfg.num_ctas_per_seq_kv)):
                        smem_lse_scale[
                            local_head_idx * Int32(cfg.num_ctas_per_seq_kv) + split_idx
                        ] = (
                            cute.math.exp2(local_lse[i] - global_lse, fastmath=True)
                            if has_finite_mass
                            else kernel.acc_dtype(0.0)
                        )

        prims.barrier_cta_sync(
            barrier_id=GMEM_REDUCTION_SCALE_BARRIER_ID,
            thread_count=GMEM_REDUCTION_THREADS,
        )

        acc_ptr = acc_output.iterator.raw_ptr()
        vecs_per_head = cfg.head_dim_v // GMEM_REDUCTION_ELEMENTS_PER_THREAD
        base_vec_idx = segment_idx * Int32(GMEM_REDUCTION_THREADS) + tidx
        local_head_idx = base_vec_idx // Int32(vecs_per_head)
        dim_vec_idx = base_vec_idx - local_head_idx * Int32(vecs_per_head)
        dim_idx = dim_vec_idx * Int32(GMEM_REDUCTION_ELEMENTS_PER_THREAD)
        global_head_idx = (
            head_group_idx * Int32(GMEM_REDUCTION_HEADS_PER_CTA) + local_head_idx
        )
        (
            storage_flat_query_row,
            _,
            _,
            _,
            valid_output_row,
        ) = flat_query_row_state(
            global_head_idx,
            q_idx,
            cfg.tile_size_q,
            cfg.logical_num_heads_q,
            cfg.logical_seq_len_q,
            cu_seqlens_q=cu_seqlens_q,
            batch_idx=batch_idx,
        )
        if global_head_idx < Int32(cfg.num_heads_q) and valid_output_row:
            acc_vec = vector_from_scalars(
                (
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                ),
                dtype=Float32,
            )
            for split_idx in cutlass.range_constexpr(cfg.num_ctas_per_seq_kv):
                scale = Float32(
                    smem_lse_scale[
                        local_head_idx * Int32(cfg.num_ctas_per_seq_kv)
                        + Int32(split_idx)
                    ]
                )
                acc_elem_offset = split_o_element_offset(
                    cfg,
                    batch_idx,
                    q_idx,
                    global_head_idx,
                    Int32(split_idx),
                    dim_idx,
                )
                partial_vec = (
                    (acc_ptr + acc_elem_offset)
                    .load(
                        count=GMEM_REDUCTION_ELEMENTS_PER_THREAD,
                        alignment=16,
                    )
                    .to(Float32)
                )
                acc_vec = acc_vec + partial_vec * scale
            output_query_row = public_query_flat_row(
                cfg,
                storage_flat_query_row,
                batch_idx,
                cu_seqlens_q,
            )
            out_elem_offset = Int64(output_query_row) * Int64(cfg.head_dim_v) + Int64(
                dim_idx
            )
            (output.iterator.raw_ptr() + out_elem_offset).store(
                acc_vec.to(output.element_type),
                alignment=16,
            )
        return

    if cutlass.const_expr(use_q_row_reduction):
        # Multi-token-Q BF16 path: one CTA reduces several Q rows for one head,
        # again using 8 output elements per thread.
        slice_idx = q_idx
        head_idx = head_dim_tile_idx
        rows_per_slice = GMEM_REDUCTION_Q_ROWS_PER_CTA
        # Warp lanes stripe over split-KV LSE slots for each Q row in the slice.
        lse_per_thread = ceil_div(cfg.num_ctas_per_seq_kv, GMEM_REDUCTION_WARP_LANES)
        smem_lse_scale = cutlass.Array(
            kernel.lse_dtype,
            rows_per_slice * cfg.num_ctas_per_seq_kv,
            space=cutlass.AddressSpace.smem,
            alignment=16,
        )

        if warp_idx < Int32(rows_per_slice):
            row_q_idx = slice_idx * Int32(rows_per_slice) + warp_idx
            (
                storage_flat_query_row,
                _,
                _,
                _,
                valid_output_row,
            ) = flat_query_row_state(
                head_idx,
                row_q_idx,
                cfg.tile_size_q,
                cfg.logical_num_heads_q,
                cfg.logical_seq_len_q,
                cu_seqlens_q=cu_seqlens_q,
                batch_idx=batch_idx,
            )
            if valid_output_row:
                row_lse = acc_lse[head_idx, None, row_q_idx, batch_idx]
                local_lse = cutlass.Array(
                    kernel.lse_dtype,
                    lse_per_thread,
                    space=cutlass.AddressSpace.rmem,
                )
                lse_max = kernel.lse_dtype(-kernel.lse_dtype.inf)
                for i in cutlass.range_constexpr(lse_per_thread):
                    split_idx = lane_idx + Int32(i * GMEM_REDUCTION_WARP_LANES)
                    local_lse[i] = (
                        row_lse[split_idx]
                        if cute.elem_less(split_idx, Int32(cfg.num_ctas_per_seq_kv))
                        else -kernel.lse_dtype.inf
                    )
                    lse_max = fmax_f32(lse_max, local_lse[i])
                lse_max = warp_reduce_max_f32(lse_max)
                lse_max = (
                    lse_max
                    if lse_max != -kernel.lse_dtype.inf
                    else kernel.lse_dtype(0.0)
                )
                sum_lse = kernel.lse_dtype(0.0)
                for i in cutlass.range_constexpr(lse_per_thread):
                    sum_lse += cute.math.exp2(local_lse[i] - lse_max, fastmath=True)
                sum_lse = warp_reduce_sum_f32(sum_lse)
                has_finite_mass = sum_lse == sum_lse and sum_lse != kernel.lse_dtype(
                    0.0
                )
                global_lse = (
                    lse_max + cute.math.log2(sum_lse, fastmath=True)
                    if has_finite_mass
                    else -kernel.lse_dtype.inf
                )
                if lane_idx == Int32(0):
                    output_query_row = public_query_flat_row(
                        cfg,
                        storage_flat_query_row,
                        batch_idx,
                        cu_seqlens_q,
                    )
                    (lse.iterator.raw_ptr() + output_query_row).store(global_lse)
                for i in cutlass.range_constexpr(lse_per_thread):
                    split_idx = lane_idx + Int32(i * GMEM_REDUCTION_WARP_LANES)
                    if cute.elem_less(split_idx, Int32(cfg.num_ctas_per_seq_kv)):
                        smem_lse_scale[
                            warp_idx * Int32(cfg.num_ctas_per_seq_kv) + split_idx
                        ] = (
                            cute.math.exp2(local_lse[i] - global_lse, fastmath=True)
                            if has_finite_mass
                            else kernel.acc_dtype(0.0)
                        )

        prims.barrier_cta_sync(
            barrier_id=GMEM_REDUCTION_SCALE_BARRIER_ID,
            thread_count=GMEM_REDUCTION_THREADS,
        )

        base_vec_offset = tidx * Int32(GMEM_REDUCTION_ELEMENTS_PER_THREAD)
        row_in_slice = base_vec_offset // Int32(cfg.head_dim_v)
        dim_idx = base_vec_offset - row_in_slice * Int32(cfg.head_dim_v)
        q_idx = slice_idx * Int32(rows_per_slice) + row_in_slice
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
        if valid_output_row:
            acc_vec = vector_from_scalars(
                (
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                ),
                dtype=Float32,
            )
            acc_ptr = acc_output.iterator.raw_ptr()
            for split_idx in cutlass.range_constexpr(cfg.num_ctas_per_seq_kv):
                scale = Float32(
                    smem_lse_scale[
                        row_in_slice * Int32(cfg.num_ctas_per_seq_kv) + split_idx
                    ]
                )
                acc_elem_offset = split_o_element_offset(
                    cfg,
                    batch_idx,
                    q_idx,
                    head_idx,
                    Int32(split_idx),
                    dim_idx,
                )
                partial_vec = (
                    (acc_ptr + acc_elem_offset)
                    .load(
                        count=GMEM_REDUCTION_ELEMENTS_PER_THREAD,
                        alignment=16,
                    )
                    .to(Float32)
                )
                acc_vec = acc_vec + partial_vec * scale
            output_query_row = public_query_flat_row(
                cfg,
                storage_flat_query_row,
                batch_idx,
                cu_seqlens_q,
            )
            out_elem_offset = Int64(output_query_row) * Int64(cfg.head_dim_v) + Int64(
                dim_idx
            )
            (output.iterator.raw_ptr() + out_elem_offset).store(
                acc_vec.to(output.element_type),
                alignment=16,
            )
        return

    # Scalar fallback: one reducer coordinate owns one flat query row and one
    # head-dimension band, then publishes only after logical-row validation.
    reduction_dim_tiles = ceil_div(cfg.head_dim_v, GMEM_REDUCTION_SCALAR_DIM_TILE)
    head_idx = head_dim_tile_idx // Int32(reduction_dim_tiles)
    dim_tile_idx = head_dim_tile_idx - head_idx * Int32(reduction_dim_tiles)
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

    smem_lse_scale = cutlass.Array(
        kernel.lse_dtype,
        cfg.num_ctas_per_seq_kv,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )

    g_lse = acc_lse[head_idx, None, q_idx, batch_idx]
    if warp_idx == Int32(0):
        # One warp reduces split-KV statistics; lanes stripe split slots by
        # warp size before publishing per-split O rescale factors.
        lse_per_thread = ceil_div(cfg.num_ctas_per_seq_kv, GMEM_REDUCTION_WARP_LANES)
        local_lse = cutlass.Array(
            kernel.lse_dtype,
            lse_per_thread,
            space=cutlass.AddressSpace.rmem,
        )
        lse_max = kernel.lse_dtype(-kernel.lse_dtype.inf)
        for i in cutlass.range_constexpr(lse_per_thread):
            split_idx = lane_idx + Int32(i * GMEM_REDUCTION_WARP_LANES)
            local_lse[i] = (
                g_lse[split_idx]
                if valid_output_row
                and cute.elem_less(split_idx, Int32(cfg.num_ctas_per_seq_kv))
                else -kernel.lse_dtype.inf
            )
            lse_max = fmax_f32(lse_max, local_lse[i])
        lse_max = warp_reduce_max_f32(lse_max)
        lse_max = lse_max if lse_max != -kernel.lse_dtype.inf else kernel.lse_dtype(0.0)
        sum_lse = kernel.lse_dtype(0.0)
        for i in cutlass.range_constexpr(lse_per_thread):
            sum_lse += cute.math.exp2(local_lse[i] - lse_max, fastmath=True)
        sum_lse = warp_reduce_sum_f32(sum_lse)
        has_finite_mass = sum_lse == sum_lse and sum_lse != kernel.lse_dtype(0.0)
        global_lse = (
            lse_max + cute.math.log2(sum_lse, fastmath=True)
            if has_finite_mass
            else -kernel.lse_dtype.inf
        )
        if lane_idx == Int32(0) and valid_output_row:
            if dim_tile_idx == Int32(0):
                output_query_row = public_query_flat_row(
                    cfg,
                    storage_flat_query_row,
                    batch_idx,
                    cu_seqlens_q,
                )
                (lse.iterator.raw_ptr() + output_query_row).store(global_lse)
        for i in cutlass.range_constexpr(lse_per_thread):
            split_idx = lane_idx + Int32(i * GMEM_REDUCTION_WARP_LANES)
            if cute.elem_less(split_idx, Int32(cfg.num_ctas_per_seq_kv)):
                smem_lse_scale[split_idx] = (
                    cute.math.exp2(local_lse[i] - global_lse, fastmath=True)
                    if has_finite_mass
                    else kernel.acc_dtype(0.0)
                )

    prims.barrier_cta_sync(
        barrier_id=GMEM_REDUCTION_SCALE_BARRIER_ID,
        thread_count=GMEM_REDUCTION_THREADS,
    )

    dim_idx = dim_tile_idx * Int32(GMEM_REDUCTION_SCALAR_DIM_TILE) + tidx
    g_acc_o = acc_output[head_idx, None, None, q_idx, batch_idx]
    r_acc_o = cutlass.Array(kernel.acc_dtype, 1, space=cutlass.AddressSpace.rmem)
    out_element_dtype = output.element_type
    r_o = cutlass.Array(out_element_dtype, 1, space=cutlass.AddressSpace.rmem)
    r_acc_o[0] = kernel.acc_dtype(0.0)
    if valid_output_row:
        for split_idx in cutlass.range_constexpr(cfg.num_ctas_per_seq_kv):
            scale = Float32(smem_lse_scale[split_idx])
            if dim_idx < Int32(cfg.head_dim_v):
                r_acc_o[0] = r_acc_o[0] + Float32(g_acc_o[split_idx, dim_idx]) * scale
    r_o.store(r_acc_o.load(0, 1).to(out_element_dtype), 0)
    if dim_idx < Int32(cfg.head_dim_v) and valid_output_row:
        output_query_row = public_query_flat_row(
            cfg,
            storage_flat_query_row,
            batch_idx,
            cu_seqlens_q,
        )
        (
            output.iterator.raw_ptr()
            + Int64(output_query_row) * Int64(cfg.head_dim_v)
            + Int64(dim_idx)
        ).store(r_o[0])
