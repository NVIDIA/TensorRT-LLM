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

"""Split-KV reduction body for the throughput 2CTA MLA TS kernel."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.experimental import primitives as prims

from .config import (
    REDUCTION_ROWS_PER_CTA,
    REDUCTION_THREADS_PER_ROW,
    REDUCTION_VALUES_PER_THREAD,
    REDUCTION_VECTOR_BYTES,
)
from ..helpers.constants import SPLIT_REDUCTION_SCALE_BARRIER_ID
from ..helpers.math import ceil_div
from ..helpers.mask import MaskType, mask_visible_k_length
from ..helpers.ops import (
    fmax_f32,
    warp_reduce_max_f32,
    warp_reduce_sum_f32,
    vector_from_scalars,
)
from ..helpers.query import groups_tokens_heads_q_row_state, query_batch_bounds
from .work_partition import (
    runtime_row_prefix_active_split_count,
    runtime_split_kv_cap,
)


@cute.jit
def run_reduction_kernel(
    kernel,
    output,
    lse,
    acc_output,
    acc_lse,
    split_kv,
    cache_seqs,
    cu_seqlens_q,
    block_split_kvs,
    cfg,
    max_splits: cutlass.Constexpr[int],
):
    """Combine eight grouped-coordinate split rows per 512-thread CTA.

    Each 64-thread row group owns one D512 output row.  Its first warp computes
    FP32 LSE rescale factors, both warps consume one contiguous 16-byte BF16
    fragment per thread, and the final accumulation remains FP32.  Padded rows
    still participate in CTA synchronization but never publish public output.
    """
    effective_head_group_idx, seq_q_idx, batch_idx = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    row_in_cta = tidx // Int32(REDUCTION_THREADS_PER_ROW)
    row_thread_idx = tidx - row_in_cta * Int32(REDUCTION_THREADS_PER_ROW)
    row_warp_idx = cute.arch.make_warp_uniform(
        row_thread_idx // Int32(cfg.threads_per_warp)
    )
    lane_idx = row_thread_idx % Int32(cfg.threads_per_warp)

    effective_num_heads_q = Int32(kernel.num_heads * kernel.groups_tokens_heads_q_ratio)
    effective_head_idx = (
        effective_head_group_idx * Int32(REDUCTION_ROWS_PER_CTA) + row_in_cta
    )
    head_is_valid = effective_head_idx < effective_num_heads_q
    # The row helper performs control-flow and storage-coordinate arithmetic.
    # Clamp a tail CTA's padding rows before calling it, then predicate every
    # public/workspace access with ``head_is_valid``.
    safe_effective_head_idx = cute.math.min(
        effective_head_idx, effective_num_heads_q - Int32(1)
    )
    (
        storage_flat_query_row,
        _,
        logical_q_idx,
        _,
        mapped_query_is_valid,
    ) = groups_tokens_heads_q_row_state(
        safe_effective_head_idx,
        seq_q_idx,
        kernel.groups_tokens_heads_q_ratio,
        kernel.num_heads,
        kernel.seq_len_q,
        cu_seqlens_q,
        batch_idx,
    )
    query_is_valid = head_is_valid and mapped_query_is_valid
    public_flat_query_row = storage_flat_query_row
    if cutlass.const_expr(cu_seqlens_q is None):
        public_flat_query_row = public_flat_query_row + batch_idx * Int32(
            kernel.seq_len_q * kernel.num_heads
        )

    # The scalar split count remains the grid/workspace capacity. A variable-
    # split launch optionally contracts it with block_split_kvs[batch].
    if cutlass.const_expr(
        kernel.static_split_kv is not None and not kernel.is_var_split_kv
    ):
        split_kv_cap = Int32(max_splits)
    else:
        split_kv_cap = runtime_split_kv_cap(
            split_kv,
            kernel.is_var_split_kv,
            block_split_kvs,
            batch_idx,
        )
    # Producer splits are sized from the group's largest K domain, while each
    # logical row consumes only the prefix containing its visible K tiles.
    group_k = cache_seqs[batch_idx]
    row_k = group_k
    if cutlass.const_expr(
        cfg.mask_type == MaskType.CAUSAL.value and kernel.seq_len_q > 1
    ):
        _, logical_seq_len_q = query_batch_bounds(
            cu_seqlens_q,
            batch_idx,
            kernel.seq_len_q,
        )
        _, _, group_last_logical_q_idx, _, _ = groups_tokens_heads_q_row_state(
            Int32(kernel.num_heads * kernel.groups_tokens_heads_q_ratio - 1),
            seq_q_idx,
            kernel.groups_tokens_heads_q_ratio,
            kernel.num_heads,
            kernel.seq_len_q,
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
    # A causal row consumes the prefix of configured-span ranges intersecting
    # its visible K tiles. This must stay identical to the producer geometry.
    local_split_kv = runtime_row_prefix_active_split_count(
        row_k_tile_total,
        group_k_tile_total,
        split_kv_cap,
    )

    smem_lse_scale = cutlass.Array(
        kernel.lse_dtype,
        REDUCTION_ROWS_PER_CTA * max_splits,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    row_scale_offset = row_in_cta * Int32(max_splits)

    acc_lse_tile = acc_lse[safe_effective_head_idx, None, seq_q_idx, batch_idx]
    if row_warp_idx == 0:
        # The first warp for each row owns its log-sum-exp merge.  It publishes
        # one rescale factor per active split for the row's second warp too.
        lse_per_thread = ceil_div(max_splits, cfg.threads_per_warp)
        local_lse = cutlass.Array(kernel.lse_dtype, lse_per_thread)
        lse_max = kernel.lse_dtype(-kernel.lse_dtype.inf)
        for i in cutlass.range_constexpr(lse_per_thread):
            split_kv_idx = lane_idx + i * cfg.threads_per_warp
            active_slot = query_is_valid & cute.elem_less(
                split_kv_idx,
                local_split_kv,
            )
            local_lse[i] = -kernel.lse_dtype.inf
            # Keep the workspace access inside the dynamic predicate. Split
            # capacities need not be warp-aligned, so padded lanes must not
            # form an address beyond the producer allocation.
            if active_slot:
                local_lse[i] = acc_lse_tile[split_kv_idx]
            lse_max = fmax_f32(lse_max, local_lse[i])
        lse_max = warp_reduce_max_f32(lse_max)
        lse_max = lse_max if lse_max != -kernel.lse_dtype.inf else 0.0
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
        if lane_idx == 0 and query_is_valid:
            (lse.iterator.raw_ptr() + Int64(public_flat_query_row)).store(global_lse)
        for i in cutlass.range_constexpr(lse_per_thread):
            split_kv_idx = lane_idx + i * cfg.threads_per_warp
            if cute.elem_less(split_kv_idx, local_split_kv):
                smem_lse_scale[row_scale_offset + split_kv_idx] = (
                    cute.math.exp2(local_lse[i] - global_lse, fastmath=True)
                    if has_finite_mass
                    else kernel.lse_dtype(0.0)
                )

    # Eight independent writer warps publish scales before any row consumes
    # them.  Invalid/padded rows participate so this remains a full CTA barrier.
    prims.barrier_cta_sync(SPLIT_REDUCTION_SCALE_BARRIER_ID)

    element_idx = row_thread_idx * Int32(REDUCTION_VALUES_PER_THREAD)
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
    acc_output_ptr = acc_output.iterator.raw_ptr()
    if query_is_valid:
        partial_elem_offset_base = (
            Int64(batch_idx)
            * Int64(cute.size(acc_output.shape[3]))
            * Int64(effective_num_heads_q)
            * Int64(split_kv)
            * Int64(cfg.latent_dim)
            + Int64(seq_q_idx)
            * Int64(effective_num_heads_q)
            * Int64(split_kv)
            * Int64(cfg.latent_dim)
            + Int64(safe_effective_head_idx) * Int64(split_kv) * Int64(cfg.latent_dim)
            + Int64(element_idx)
        )
        partial_output_ptr = acc_output_ptr + partial_elem_offset_base
        # S2 is the common two-wave 2CTA decode reducer.  Materialize its two
        # fixed partials directly; larger reducers retain the compact dynamic
        # loop because fully unrolling S4 increases instruction pressure.
        if cutlass.const_expr(max_splits == 2):
            for i in cutlass.range_constexpr(max_splits):
                if Int32(i) < local_split_kv:
                    partial_vec = (
                        (partial_output_ptr + Int64(i * cfg.latent_dim))
                        .load(
                            count=REDUCTION_VALUES_PER_THREAD,
                            alignment=REDUCTION_VECTOR_BYTES,
                        )
                        .to(Float32)
                    )
                    scale = Float32(smem_lse_scale[row_scale_offset + Int32(i)])
                    acc_vec = acc_vec + partial_vec * scale
        else:
            for i in range(local_split_kv):
                partial_vec = (
                    (partial_output_ptr + Int64(i) * Int64(cfg.latent_dim))
                    .load(
                        count=REDUCTION_VALUES_PER_THREAD,
                        alignment=REDUCTION_VECTOR_BYTES,
                    )
                    .to(Float32)
                )
                scale = Float32(smem_lse_scale[row_scale_offset + i])
                acc_vec = acc_vec + partial_vec * scale

        output_elem_offset = Int64(public_flat_query_row) * Int64(
            cfg.latent_dim
        ) + Int64(element_idx)
        (output.iterator.raw_ptr() + output_elem_offset).store(
            acc_vec.to(output.element_type),
            alignment=REDUCTION_VECTOR_BYTES,
        )
