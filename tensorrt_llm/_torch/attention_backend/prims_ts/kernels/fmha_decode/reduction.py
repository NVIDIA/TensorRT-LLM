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

"""Standalone GMEM reducers for FMHA decode TS split-KV profiles.

The decode kernel publishes one normalized 16-bit O vector and one FP32
log2-LSE scalar per ``(batch, kv_head, split_kv, output_row)``. Reducer threads
own 16-byte output fragments and combine those normalized states with the
shared log2-LSE recurrence.

The production schedule is selected from the split count:

* S2-S4 use one 512-thread CTA per 8 KiB output slice and fold the exact split
  count in registers.
* S5 and larger use 128-thread CTAs over 2 KiB slices. Each cluster rank folds
  its split slots locally, publishes ``(LSE, O)`` through SMEM, and rank zero
  merges the distributed states. G16 uses a 4x4 two-level merge.

A 512-thread exact-split kernel remains as the serial reference schedule. Both
schedules merge the optional attention-sink denominator, pack the requested
output dtype, and write final O. Batch and KV head remain grid dimensions;
only split-KV is reduced here.
"""

import math

import cutlass
import cutlass.cute as cute
from cuda.bindings import driver as cuda_drv
from cutlass import Float32, Int32, Int64
from cutlass.experimental import primitives as prims

from .fmha_decode_constants import (
    FP32_BYTES,
    FP8_PACKED_OUTPUT_REGS_PER_THREAD,
    FP8_VALUES_PER_REG,
    FP16_VALUES_PER_REG,
    OUTPUT_VALUES_PER_THREAD,
    PACKED_OUTPUT_REGS_PER_THREAD,
    PACKED_REGISTER_BYTES,
    PARALLEL_REDUCTION_BYTES_PER_SLICE,
    PARALLEL_REDUCTION_FINAL_REDUCERS,
    PARALLEL_REDUCTION_LOAD_BATCH,
    PARALLEL_REDUCTION_THREADS_PER_CTA,
    PARTIAL_O_ELEMENT_BYTES,
    REDUCTION_BYTES_PER_SLICE,
    REDUCTION_BYTES_PER_THREAD,
    REDUCTION_THREADS_PER_CTA,
    SEPARATE_REDUCTION_LSE_VALUES_PER_ROW,
)
from .fmha_decode_config import FmhaDecodeConfig
from .fmha_decode_resources.helpers_common import (
    _attention_sink_head_stride,
    _local_head_from_q_output_row,
    _pack_float2_to_bf16,
    _pack_float2_to_fp16,
    _q_group_token_base,
    _q_logical_output_row_token_and_local_head,
    _q_logical_output_row_is_valid_for_seq,
    _q_physical_output_row_from_logical,
    _q_seq_bounds,
    fmul2,
)
from .fmha_decode_resources.helpers_kv_tile_idx import _runtime_active_splits_kv
from .fmha_decode_resources.helpers_softmax import _pack_float4_to_fp8_e4m3
from ..separate_reduction import (
    merge_log2_lse,
    unpack_normalized_vec8,
)


@cute.jit
def _separate_workspace_row_offset(
    logical_kv_idx: Int64,
    split_idx: Int32,
    row_idx: Int32,
    rows_per_split: Int32,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
) -> Int64:
    """Return a split/row offset without overflowing 32-bit products."""

    return (logical_kv_idx * Int64(cfg.max_splits_kv) + Int64(split_idx)) * Int64(
        rows_per_split
    ) + Int64(row_idx)


@cute.jit
def _reduction_q_group_idx(
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    h_r: Int32,
    logical_output_row_idx: Int32,
) -> Int32:
    """Recover the producer Q-group owning one logical scratch row."""
    token_idx, local_head_idx = _q_logical_output_row_token_and_local_head(
        cfg, h_r, logical_output_row_idx
    )
    if cutlass.const_expr(cfg.uses_nontrivial_grouped_q_layout):
        return token_idx // Int32(cfg.q_tokens_per_cta)
    if cutlass.const_expr(cfg.max_seq_len_q > 1):
        head_ctas_per_token = Int32(
            (cfg.heads_q_per_kv + cfg.tile_size_q - 1) // cfg.tile_size_q
        )
        return token_idx * head_ctas_per_token + local_head_idx // Int32(
            cfg.tile_size_q
        )
    return local_head_idx // Int32(cfg.tile_size_q)


@cute.jit
def _reduction_active_splits_kv(
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    g_seqlens_kv: cute.Pointer,
    b_idx: Int32,
    h_r: Int32,
    logical_output_row_idx: Int32,
    seq_len_q: Int32,
) -> Int32:
    """Recompute the producer's useful split prefix for one output row."""
    q_group_idx = _reduction_q_group_idx(cfg, h_r, logical_output_row_idx)
    return _runtime_active_splits_kv(
        cfg,
        Int32(g_seqlens_kv[b_idx]),
        seq_len_q,
        _q_group_token_base(cfg, q_group_idx),
    )


@cute.jit
def _reduce_exact_splits_body(
    o_iter: cute.Pointer,
    g_seqlens_kv: cute.Pointer,
    g_cu_seqlens_q: cute.Pointer,
    g_partial_o: cute.Pointer,
    g_partial_stats: cute.Pointer,
    g_attention_sinks: cute.Pointer,
    g_h_r: Int32,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    static_full_split_prefix: cutlass.Constexpr[bool] = False,
) -> None:
    """Fold every split in one 512-thread CTA over one 8 KiB output slice.

    ``g_partial_stats`` stores one log2-LSE scalar for each split and output
    row. ``g_partial_o`` stores the corresponding normalized 16-bit O fragment.
    This body is shared by the serial reference kernel and the compact S2-S4
    production schedule; PDL ordering remains in the production outer kernel.
    """

    thread_idx, _, _ = cute.arch.thread_idx()
    slice_idx, h_k_idx, b_idx = cute.arch.block_idx()
    _, grid_h_k, _ = cute.arch.grid_dim()
    # Flatten batch and KV-head so the partial buffers use one contiguous
    # logical tile index independent of the reducer launch grid layout.
    logical_kv_idx = Int64(b_idx) * Int64(grid_h_k) + Int64(h_k_idx)
    attention_sink_h_r = _attention_sink_head_stride(cfg, g_h_r)

    # One CTA owns one contiguous byte slice of the final output tile. Within
    # the CTA, each thread maps to one 16-byte fragment and derives both the
    # output row and the column offset inside that row from the byte offset.
    bytes_per_output_row = Int32(cfg.headdim * PARTIAL_O_ELEMENT_BYTES)
    bytes_per_thread = Int32(REDUCTION_BYTES_PER_THREAD)
    bytes_per_slice = Int32(REDUCTION_BYTES_PER_SLICE)
    reduce_base_offset = slice_idx * bytes_per_slice + thread_idx * bytes_per_thread
    reduce_row_idx = reduce_base_offset // bytes_per_output_row
    reduce_col_idx = (reduce_base_offset % bytes_per_output_row) // Int32(
        PARTIAL_O_ELEMENT_BYTES
    )
    q_token_offset, seq_len_q = _q_seq_bounds(cfg, g_cu_seqlens_q, b_idx)
    active_splits_kv = Int32(cfg.splits_kv)
    if cutlass.const_expr(not static_full_split_prefix):
        active_splits_kv = _reduction_active_splits_kv(
            cfg,
            g_seqlens_kv,
            b_idx,
            g_h_r,
            reduce_row_idx,
            seq_len_q,
        )
    valid_reduce_row = _q_logical_output_row_is_valid_for_seq(
        cfg,
        g_h_r,
        reduce_row_idx,
        seq_len_q,
    )
    output_row_idx = _q_physical_output_row_from_logical(
        cfg,
        g_h_r,
        grid_h_k,
        b_idx,
        h_k_idx,
        reduce_row_idx,
        q_token_offset,
    )
    attention_sink_head_idx = _local_head_from_q_output_row(
        cfg,
        g_h_r,
        reduce_row_idx,
    )

    output_vals = cutlass.Array(
        Float32, OUTPUT_VALUES_PER_THREAD, space=cutlass.AddressSpace.rmem
    )
    for elem_idx in cutlass.range_constexpr(OUTPUT_VALUES_PER_THREAD):
        output_vals[elem_idx] = Float32(0.0)

    global_lse = Float32(-Float32.inf)

    if valid_reduce_row:
        # The workspace retains configured-max strides, but only the runtime
        # active prefix was published by producer CTAs.
        for split_idx_i in cutlass.range_constexpr(cfg.max_splits_kv):
            split_idx = Int32(split_idx_i)
            split_is_active = split_idx < active_splits_kv
            if cutlass.const_expr(static_full_split_prefix):
                split_is_active = cutlass.const_expr(split_idx_i < cfg.splits_kv)
            if split_is_active:
                # LSE layout: [logical_kv][configured split][row].
                workspace_row = _separate_workspace_row_offset(
                    logical_kv_idx,
                    split_idx,
                    reduce_row_idx,
                    g_h_r,
                    cfg,
                )
                stats_offset = workspace_row * Int64(
                    SEPARATE_REDUCTION_LSE_VALUES_PER_ROW * FP32_BYTES
                )
                stats_src = cutlass.inttoptr(
                    g_partial_stats.toint() + stats_offset,
                    mem_space=1,
                    dtype=Float32,
                )
                partial_lse = stats_src.load()

                partial_o_offset = workspace_row * Int64(
                    cfg.headdim * PARTIAL_O_ELEMENT_BYTES
                ) + Int64(reduce_col_idx) * Int64(PARTIAL_O_ELEMENT_BYTES)
                partial_o_src = cutlass.inttoptr(
                    g_partial_o.toint() + partial_o_offset,
                    mem_space=1,
                    dtype=Int32,
                )
                loaded_partial_regs = partial_o_src.load(
                    count=PACKED_OUTPUT_REGS_PER_THREAD,
                    alignment=PACKED_OUTPUT_REGS_PER_THREAD * PACKED_REGISTER_BYTES,
                )

                new_lse, old_weight, partial_weight = merge_log2_lse(
                    global_lse,
                    partial_lse,
                )
                partial_vals = unpack_normalized_vec8(
                    loaded_partial_regs, cfg.use_bf16_separate_partial_o
                )
                for elem_idx in cutlass.range_constexpr(OUTPUT_VALUES_PER_THREAD):
                    output_vals[elem_idx] = (
                        output_vals[elem_idx] * old_weight
                        + partial_vals[elem_idx] * partial_weight
                    )
                global_lse = new_lse

        _store_parallel_reduction_output(
            output_vals,
            global_lse,
            o_iter,
            g_attention_sinks,
            output_row_idx,
            reduce_col_idx,
            h_k_idx,
            attention_sink_h_r,
            grid_h_k,
            attention_sink_head_idx,
            cfg,
        )


@cute.kernel
def decode_gen_separate_reduction_kernel(
    o_iter: cute.Pointer,
    g_seqlens_kv: cute.Pointer,
    g_cu_seqlens_q: cute.Pointer,
    g_partial_o: cute.Pointer,
    g_partial_stats: cute.Pointer,
    g_attention_sinks: cute.Pointer,
    g_h_r: Int32,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    static_full_split_prefix: cutlass.Constexpr[bool] = False,
) -> None:
    """Run the 512-thread exact-split reference schedule."""

    _reduce_exact_splits_body(
        o_iter,
        g_seqlens_kv,
        g_cu_seqlens_q,
        g_partial_o,
        g_partial_stats,
        g_attention_sinks,
        g_h_r,
        cfg,
        static_full_split_prefix,
    )


@cute.jit
def _attention_sink_log2_lse(
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    attention_sinks_ptr: cute.Pointer,
    logical_h_k_idx: Int32,
    h_r: Int32,
    num_heads_kv: Int32,
    local_head_idx: Int32,
) -> Float32:
    """Return the sink's LSE-domain denominator state, or neutral ``-inf``."""

    if cutlass.const_expr(not cfg.use_attention_sinks):
        return Float32(-Float32.inf)
    head_idx = cute.math.min(
        logical_h_k_idx * h_r + local_head_idx,
        h_r * num_heads_kv - Int32(1),
    )
    sink_ptr = cutlass.inttoptr(
        attention_sinks_ptr.toint() + cutlass.Int64(head_idx * Int32(FP32_BYTES)),
        mem_space=1,
        dtype=Float32,
    )
    sink_lse = sink_ptr.load() * Float32(1.4426950408889634)
    if cutlass.const_expr(cfg.use_fp8_qkv):
        sink_lse += Float32(math.log2(448.0))
    return sink_lse


@cute.jit
def _store_parallel_reduction_output(
    output_vals: cutlass.Array,
    global_lse: Float32,
    o_iter: cute.Pointer,
    g_attention_sinks: cute.Pointer,
    output_row_idx: Int32,
    reduce_col_idx: Int32,
    h_k_idx: Int32,
    attention_sink_h_r: Int32,
    grid_h_k: Int32,
    attention_sink_head_idx: Int32,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
) -> None:
    """Merge the optional sink and store one normalized output fragment.

    Producer CTAs already fold the softmax scale into log2-LSE and, for FP8
    output, fold the output quantization scale into normalized partial O. The
    standalone reducer therefore only merges normalized states here.
    """

    sink_lse = _attention_sink_log2_lse(
        cfg,
        g_attention_sinks,
        h_k_idx,
        attention_sink_h_r,
        grid_h_k,
        attention_sink_head_idx,
    )
    _, split_weight, _ = merge_log2_lse(global_lse, sink_lse)

    final_regs = cutlass.Array(
        Int32,
        PACKED_OUTPUT_REGS_PER_THREAD,
        space=cutlass.AddressSpace.rmem,
    )
    if cutlass.const_expr(cfg.use_fp8_output):
        for packed_idx in cutlass.range_constexpr(FP8_PACKED_OUTPUT_REGS_PER_THREAD):
            val_base = packed_idx * FP8_VALUES_PER_REG
            final_regs[packed_idx] = _pack_float4_to_fp8_e4m3(
                output_vals[val_base] * split_weight,
                output_vals[val_base + 1] * split_weight,
                output_vals[val_base + 2] * split_weight,
                output_vals[val_base + 3] * split_weight,
            )
    else:
        for packed_idx in cutlass.range_constexpr(PACKED_OUTPUT_REGS_PER_THREAD):
            val_base = packed_idx * FP16_VALUES_PER_REG
            final_pair = fmul2(
                (split_weight, split_weight),
                (output_vals[val_base], output_vals[val_base + 1]),
            )
            if cutlass.const_expr(cfg.use_bf16_output):
                final_regs[packed_idx] = _pack_float2_to_bf16(
                    final_pair[0], final_pair[1]
                )
            else:
                final_regs[packed_idx] = _pack_float2_to_fp16(
                    final_pair[0], final_pair[1]
                )

    # Widen before the first row-stride product. Large packed-Q batches can
    # exceed the signed-32-bit byte range even though each local row and
    # column coordinate is individually Int32.
    output_row_bytes = Int64(cfg.headdim * cfg.o_dtype_bytes)
    dst_offset = Int64(output_row_idx) * output_row_bytes + Int64(
        reduce_col_idx
    ) * Int64(cfg.o_dtype_bytes)
    dst_ptr = cutlass.inttoptr(
        o_iter.toint() + dst_offset,
        mem_space=1,
        dtype=Int32,
    )
    if cutlass.const_expr(cfg.use_fp8_output):
        dst_ptr.store(
            final_regs.data_ptr().load(
                count=FP8_PACKED_OUTPUT_REGS_PER_THREAD,
                alignment=PACKED_REGISTER_BYTES,
            ),
            alignment=FP8_PACKED_OUTPUT_REGS_PER_THREAD * PACKED_REGISTER_BYTES,
        )
    else:
        dst_ptr.store(
            final_regs.data_ptr().load(
                count=PACKED_OUTPUT_REGS_PER_THREAD,
                alignment=PACKED_REGISTER_BYTES,
            ),
            alignment=PACKED_OUTPUT_REGS_PER_THREAD * PACKED_REGISTER_BYTES,
        )


@cute.kernel
def decode_gen_parallel_separate_reduction_kernel(
    o_iter: cute.Pointer,
    g_seqlens_kv: cute.Pointer,
    g_cu_seqlens_q: cute.Pointer,
    g_partial_o: cute.Pointer,
    g_partial_stats: cute.Pointer,
    g_attention_sinks: cute.Pointer,
    g_h_r: Int32,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    static_full_split_prefix: cutlass.Constexpr[bool] = False,
) -> None:
    """Reduce split partials with a compact or clustered constexpr schedule.

    S2-S4 use one 512-thread CTA per 8 KiB output slice and reduce the exact
    split count directly. Larger schedules use 128-thread CTAs over 2 KiB
    slices. Each cluster rank owns 2, 4, or 8 split slots; G1 stores directly,
    G16 uses a two-level 4x4 merge, and G2/G4/G8 finalize through rank zero.
    """

    # Pair with the producer's launch-dependents signal before reading any
    # partial GMEM. The wait is CTA-convergent and stays outside both schedules.
    if cutlass.const_expr(cfg.use_parallel_separate_reduction_pdl):
        prims.griddepcontrol(kind=prims.GridDepAction.WAIT)

    if cutlass.const_expr(cfg.use_compact_parallel_reduction):
        _reduce_exact_splits_body(
            o_iter,
            g_seqlens_kv,
            g_cu_seqlens_q,
            g_partial_o,
            g_partial_stats,
            g_attention_sinks,
            g_h_r,
            cfg,
            static_full_split_prefix,
        )
        return

    thread_idx, _, _ = cute.arch.thread_idx()
    block_idx_x, h_k_idx, b_idx = cute.arch.block_idx()
    _, grid_h_k, _ = cute.arch.grid_dim()
    cluster_rank = cute.arch.block_idx_in_cluster()
    logical_kv_idx = Int64(b_idx) * Int64(grid_h_k) + Int64(h_k_idx)
    attention_sink_h_r = _attention_sink_head_stride(cfg, g_h_r)

    bytes_per_output_row = Int32(cfg.headdim * PARTIAL_O_ELEMENT_BYTES)
    slice_idx = block_idx_x // Int32(cfg.parallel_reduction_cluster_size)
    reduce_base_offset = slice_idx * Int32(
        PARALLEL_REDUCTION_BYTES_PER_SLICE
    ) + thread_idx * Int32(REDUCTION_BYTES_PER_THREAD)
    reduce_row_idx = reduce_base_offset // bytes_per_output_row
    reduce_col_idx = (reduce_base_offset % bytes_per_output_row) // Int32(
        PARTIAL_O_ELEMENT_BYTES
    )
    q_token_offset, seq_len_q = _q_seq_bounds(cfg, g_cu_seqlens_q, b_idx)
    active_splits_kv = Int32(cfg.splits_kv)
    if cutlass.const_expr(not static_full_split_prefix):
        active_splits_kv = _reduction_active_splits_kv(
            cfg,
            g_seqlens_kv,
            b_idx,
            g_h_r,
            reduce_row_idx,
            seq_len_q,
        )
    valid_reduce_row = _q_logical_output_row_is_valid_for_seq(
        cfg,
        g_h_r,
        reduce_row_idx,
        seq_len_q,
    )
    output_row_idx = _q_physical_output_row_from_logical(
        cfg,
        g_h_r,
        grid_h_k,
        b_idx,
        h_k_idx,
        reduce_row_idx,
        q_token_offset,
    )
    attention_sink_head_idx = _local_head_from_q_output_row(
        cfg,
        g_h_r,
        reduce_row_idx,
    )

    output_vals = cutlass.Array(
        Float32, OUTPUT_VALUES_PER_THREAD, space=cutlass.AddressSpace.rmem
    )
    for elem_idx in cutlass.range_constexpr(OUTPUT_VALUES_PER_THREAD):
        output_vals[elem_idx] = Float32(0.0)
    global_lse = Float32(-Float32.inf)

    # Batch up to four independent GMEM loads before folding them. The final
    # batch width is compile-time. Padded split slots stay neutral and never
    # form GMEM pointers, including non-power-of-two split counts.
    local_splits = cfg.parallel_reduction_splits_per_cta
    partial_lse = cutlass.Array(
        Float32, PARALLEL_REDUCTION_LOAD_BATCH, space=cutlass.AddressSpace.rmem
    )
    partial_regs = cutlass.Array(
        Int32,
        PARALLEL_REDUCTION_LOAD_BATCH * PACKED_OUTPUT_REGS_PER_THREAD,
        space=cutlass.AddressSpace.rmem,
    )
    for split_base_i in cutlass.range_constexpr(
        0, local_splits, PARALLEL_REDUCTION_LOAD_BATCH
    ):
        batch_width = min(PARALLEL_REDUCTION_LOAD_BATCH, local_splits - split_base_i)
        split_base = Int32(split_base_i)
        for jj in cutlass.range_constexpr(batch_width):
            split_idx = cluster_rank * Int32(local_splits) + split_base + Int32(jj)
            valid_split_idx = cutlass.const_expr(
                cfg.parallel_reduction_padded_splits == cfg.max_splits_kv
            ) or split_idx < Int32(cfg.max_splits_kv)
            active_split_idx = split_idx < active_splits_kv
            if cutlass.const_expr(static_full_split_prefix):
                active_split_idx = cutlass.Boolean(True)
            if valid_split_idx and active_split_idx and valid_reduce_row:
                workspace_row = _separate_workspace_row_offset(
                    logical_kv_idx,
                    split_idx,
                    reduce_row_idx,
                    g_h_r,
                    cfg,
                )
                stats_offset = workspace_row * Int64(
                    SEPARATE_REDUCTION_LSE_VALUES_PER_ROW * FP32_BYTES
                )
                stats_src = cutlass.inttoptr(
                    g_partial_stats.toint() + stats_offset,
                    mem_space=1,
                    dtype=Float32,
                )
                partial_lse[jj] = stats_src.load()

                partial_o_offset = workspace_row * Int64(
                    cfg.headdim * PARTIAL_O_ELEMENT_BYTES
                ) + Int64(reduce_col_idx) * Int64(PARTIAL_O_ELEMENT_BYTES)
                partial_o_src = cutlass.inttoptr(
                    g_partial_o.toint() + partial_o_offset,
                    mem_space=1,
                    dtype=Int32,
                )
                loaded_partial_regs = partial_o_src.load(
                    count=PACKED_OUTPUT_REGS_PER_THREAD,
                    alignment=PACKED_OUTPUT_REGS_PER_THREAD * PACKED_REGISTER_BYTES,
                )
                regs_base = jj * PACKED_OUTPUT_REGS_PER_THREAD
                for reg_idx in cutlass.range_constexpr(PACKED_OUTPUT_REGS_PER_THREAD):
                    partial_regs[regs_base + reg_idx] = loaded_partial_regs[reg_idx]

        for jj in cutlass.range_constexpr(batch_width):
            split_idx = cluster_rank * Int32(local_splits) + split_base + Int32(jj)
            valid_split_idx = cutlass.const_expr(
                cfg.parallel_reduction_padded_splits == cfg.max_splits_kv
            ) or split_idx < Int32(cfg.max_splits_kv)
            active_split_idx = split_idx < active_splits_kv
            if cutlass.const_expr(static_full_split_prefix):
                active_split_idx = cutlass.Boolean(True)
            if valid_split_idx and active_split_idx and valid_reduce_row:
                regs_base = jj * PACKED_OUTPUT_REGS_PER_THREAD
                loaded_partial_regs = (partial_regs.data_ptr() + Int32(regs_base)).load(
                    count=PACKED_OUTPUT_REGS_PER_THREAD,
                    alignment=PACKED_REGISTER_BYTES,
                )
                new_lse, old_weight, partial_weight = merge_log2_lse(
                    global_lse,
                    partial_lse[jj],
                )
                partial_vals = unpack_normalized_vec8(
                    loaded_partial_regs, cfg.use_bf16_separate_partial_o
                )
                for elem_idx in cutlass.range_constexpr(OUTPUT_VALUES_PER_THREAD):
                    output_vals[elem_idx] = (
                        output_vals[elem_idx] * old_weight
                        + partial_vals[elem_idx] * partial_weight
                    )
                global_lse = new_lse

    # G1 stores its register accumulator directly and compiles out SMEM
    # publication, mapa, and cluster barriers. The branch is CTA-uniform and
    # resolved at compile time.
    if cutlass.const_expr(cfg.parallel_reduction_cluster_size == 1):
        if valid_reduce_row:
            _store_parallel_reduction_output(
                output_vals,
                global_lse,
                o_iter,
                g_attention_sinks,
                output_row_idx,
                reduce_col_idx,
                h_k_idx,
                attention_sink_h_r,
                grid_h_k,
                attention_sink_head_idx,
                cfg,
            )
        return

    # Each G2+ thread publishes one normalized ``(LSE, O)`` state using the
    # profile's selected 16-bit partial type. Corresponding threads in every
    # rank map to the same output row, so row validity is cluster-uniform. A
    # rank with only padded split slots publishes the neutral ``(-inf, 0)``.
    smem_lse = cutlass.Array(
        Float32,
        SEPARATE_REDUCTION_LSE_VALUES_PER_ROW * PARALLEL_REDUCTION_THREADS_PER_CTA,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )
    smem_partial_o = cutlass.Array(
        Int32,
        PACKED_OUTPUT_REGS_PER_THREAD * PARALLEL_REDUCTION_THREADS_PER_CTA,
        space=cutlass.AddressSpace.smem,
        alignment=16,
    )

    stats_smem_offset = thread_idx * Int32(SEPARATE_REDUCTION_LSE_VALUES_PER_ROW)
    partial_o_smem_offset = thread_idx * Int32(PACKED_OUTPUT_REGS_PER_THREAD)
    if valid_reduce_row:
        (smem_lse.data_ptr() + stats_smem_offset).store(global_lse)
        packed_o = cutlass.Array(
            Int32,
            PACKED_OUTPUT_REGS_PER_THREAD,
            space=cutlass.AddressSpace.rmem,
        )
        for reg_idx in cutlass.range_constexpr(PACKED_OUTPUT_REGS_PER_THREAD):
            elem_base = reg_idx * FP16_VALUES_PER_REG
            if cutlass.const_expr(cfg.use_bf16_separate_partial_o):
                packed_o[reg_idx] = _pack_float2_to_bf16(
                    output_vals[elem_base], output_vals[elem_base + 1]
                )
            else:
                packed_o[reg_idx] = _pack_float2_to_fp16(
                    output_vals[elem_base], output_vals[elem_base + 1]
                )
        (smem_partial_o.data_ptr() + partial_o_smem_offset).store(
            packed_o.data_ptr().load(
                count=PACKED_OUTPUT_REGS_PER_THREAD,
                alignment=PACKED_REGISTER_BYTES,
            ),
            alignment=PACKED_OUTPUT_REGS_PER_THREAD * PACKED_REGISTER_BYTES,
        )

    # Publish every rank's local state before any distributed-SMEM read.
    prims.barrier_cta_sync(0)
    prims.barrier_cluster_arrive()
    prims.barrier_cluster_wait()

    # For G16, ranks 0/4/8/12 first reduce their own four-rank group and
    # overwrite that group's first slot. Keeping each reducer inside its source
    # group prevents one reducer from overwriting a slot that another reducer
    # may still be reading. G4/G8 skip this level and are consumed directly.
    if cutlass.const_expr(cfg.parallel_reduction_cluster_size == 16):
        is_stage_leader = (
            cluster_rank % Int32(PARALLEL_REDUCTION_FINAL_REDUCERS)
        ) == Int32(0)
        if is_stage_leader & valid_reduce_row:
            stage_vals = cutlass.Array(
                Float32, OUTPUT_VALUES_PER_THREAD, space=cutlass.AddressSpace.rmem
            )
            for elem_idx in cutlass.range_constexpr(OUTPUT_VALUES_PER_THREAD):
                stage_vals[elem_idx] = Float32(0.0)
            stage_lse = Float32(-Float32.inf)
            peer_base = cluster_rank
            for peer_offset_i in cutlass.range_constexpr(
                PARALLEL_REDUCTION_FINAL_REDUCERS
            ):
                peer_rank = peer_base + Int32(peer_offset_i)
                peer_lse = prims.mapa(smem_lse.data_ptr(), peer_rank)
                peer_partial_o = prims.mapa(smem_partial_o.data_ptr(), peer_rank)
                local_lse = (peer_lse + stats_smem_offset).load()
                loaded_partial_regs = (peer_partial_o + partial_o_smem_offset).load(
                    count=PACKED_OUTPUT_REGS_PER_THREAD,
                    alignment=PACKED_OUTPUT_REGS_PER_THREAD * PACKED_REGISTER_BYTES,
                )
                new_lse, old_weight, partial_weight = merge_log2_lse(
                    stage_lse,
                    local_lse,
                )
                partial_vals = unpack_normalized_vec8(
                    loaded_partial_regs, cfg.use_bf16_separate_partial_o
                )
                for elem_idx in cutlass.range_constexpr(OUTPUT_VALUES_PER_THREAD):
                    stage_vals[elem_idx] = (
                        stage_vals[elem_idx] * old_weight
                        + partial_vals[elem_idx] * partial_weight
                    )
                stage_lse = new_lse

            (smem_lse.data_ptr() + stats_smem_offset).store(stage_lse)
            stage_packed_o = cutlass.Array(
                Int32,
                PACKED_OUTPUT_REGS_PER_THREAD,
                space=cutlass.AddressSpace.rmem,
            )
            for reg_idx in cutlass.range_constexpr(PACKED_OUTPUT_REGS_PER_THREAD):
                elem_base = reg_idx * FP16_VALUES_PER_REG
                if cutlass.const_expr(cfg.use_bf16_separate_partial_o):
                    stage_packed_o[reg_idx] = _pack_float2_to_bf16(
                        stage_vals[elem_base], stage_vals[elem_base + 1]
                    )
                else:
                    stage_packed_o[reg_idx] = _pack_float2_to_fp16(
                        stage_vals[elem_base], stage_vals[elem_base + 1]
                    )
            (smem_partial_o.data_ptr() + partial_o_smem_offset).store(
                stage_packed_o.data_ptr().load(
                    count=PACKED_OUTPUT_REGS_PER_THREAD,
                    alignment=PACKED_REGISTER_BYTES,
                ),
                alignment=PACKED_OUTPUT_REGS_PER_THREAD * PACKED_REGISTER_BYTES,
            )

        prims.barrier_cta_sync(0)
        prims.barrier_cluster_arrive()
        prims.barrier_cluster_wait()

    final_input_partials = cfg.parallel_reduction_cluster_size
    if cutlass.const_expr(cfg.parallel_reduction_cluster_size == 16):
        final_input_partials = PARALLEL_REDUCTION_FINAL_REDUCERS

    if (cluster_rank == Int32(0)) & valid_reduce_row:
        final_vals = cutlass.Array(
            Float32, OUTPUT_VALUES_PER_THREAD, space=cutlass.AddressSpace.rmem
        )
        for elem_idx in cutlass.range_constexpr(OUTPUT_VALUES_PER_THREAD):
            final_vals[elem_idx] = Float32(0.0)
        final_lse = Float32(-Float32.inf)
        for peer_rank_i in cutlass.range_constexpr(final_input_partials):
            peer_rank = Int32(peer_rank_i)
            if cutlass.const_expr(cfg.parallel_reduction_cluster_size == 16):
                peer_rank = Int32(peer_rank_i * PARALLEL_REDUCTION_FINAL_REDUCERS)
            peer_lse = prims.mapa(smem_lse.data_ptr(), peer_rank)
            peer_partial_o = prims.mapa(smem_partial_o.data_ptr(), peer_rank)
            local_lse = (peer_lse + stats_smem_offset).load()
            loaded_partial_regs = (peer_partial_o + partial_o_smem_offset).load(
                count=PACKED_OUTPUT_REGS_PER_THREAD,
                alignment=PACKED_OUTPUT_REGS_PER_THREAD * PACKED_REGISTER_BYTES,
            )
            new_lse, old_weight, partial_weight = merge_log2_lse(
                final_lse,
                local_lse,
            )
            partial_vals = unpack_normalized_vec8(
                loaded_partial_regs, cfg.use_bf16_separate_partial_o
            )
            for elem_idx in cutlass.range_constexpr(OUTPUT_VALUES_PER_THREAD):
                final_vals[elem_idx] = (
                    final_vals[elem_idx] * old_weight
                    + partial_vals[elem_idx] * partial_weight
                )
            final_lse = new_lse

        _store_parallel_reduction_output(
            final_vals,
            final_lse,
            o_iter,
            g_attention_sinks,
            output_row_idx,
            reduce_col_idx,
            h_k_idx,
            attention_sink_h_r,
            grid_h_k,
            attention_sink_head_idx,
            cfg,
        )

    # Keep every peer CTA alive until rank zero has finished its DSMEM reads.
    prims.barrier_cluster_arrive_relaxed()
    prims.barrier_cluster_wait()


@cute.jit
def fmha_decode_separate_reduction_launch(
    problem_shape: tuple[Int32, Int32, Int32, Int32, Int32],
    o_iter: cute.Pointer,
    seqlens_kv_iter: cute.Pointer,
    cu_seqlens_q_iter: cute.Pointer,
    partial_o_iter: cute.Pointer,
    partial_stats_iter: cute.Pointer,
    attention_sinks_iter: cute.Pointer,
    scale_s: Float32,
    output_scale: Float32,
    stream: cuda_drv.CUstream,
    cfg: cutlass.Constexpr[FmhaDecodeConfig],
    static_full_split_prefix: cutlass.Constexpr[bool] = False,
) -> None:
    """Launch the standalone reducer after the main split-KV decode kernel.

    The serial reference grid is ``(slice, kv_head, batch)`` with 8 KiB slices.
    The production reducer keeps that geometry for compact S2-S4 and otherwise
    maps cluster ranks into grid.x for each 2 KiB slice. Producer CTAs already
    applied ``scale_s`` to log2-LSE and ``output_scale`` to normalized partial
    O. ``seqlens_kv_iter`` is internal reducer metadata used only to
    bound the runtime split prefix; the decode launch ABI remains unchanged.
    """
    b, h_q, h_k, _, _ = problem_shape
    h_r = h_q // h_k
    q_output_rows = h_r
    if cutlass.const_expr(cfg.max_seq_len_q > 1):
        q_output_rows = h_r * Int32(cfg.max_seq_len_q)
    bytes_per_slice = REDUCTION_BYTES_PER_SLICE
    grid_q_output_rows = q_output_rows
    if cutlass.const_expr(cfg.max_seq_len_q > 1 and cfg.heads_q_per_kv != 0):
        # Grouped multi-token Q profiles lay rows out by the configured
        # heads-per-KV group instead of the launch-time h_q / h_k ratio.
        grid_q_output_rows = cfg.heads_q_per_kv * cfg.max_seq_len_q
    # The reference schedule uses 512-thread CTAs over contiguous 8 KiB slices.
    # Multiple CTAs cover wide dimensions or grouped SQ rows without changing
    # the producer workspace layout.
    num_reduction_slices = max(
        (grid_q_output_rows * cfg.headdim * 2 + bytes_per_slice - 1) // bytes_per_slice,
        1,
    )
    if cutlass.const_expr(cfg.use_parallel_separate_reduction):
        cluster_size = cfg.parallel_reduction_cluster_size
        parallel_bytes_per_slice = cfg.parallel_reduction_bytes_per_slice
        num_parallel_slices = max(
            (
                grid_q_output_rows * cfg.headdim * PARTIAL_O_ELEMENT_BYTES
                + parallel_bytes_per_slice
                - 1
            )
            // parallel_bytes_per_slice,
            1,
        )
        decode_gen_parallel_separate_reduction_kernel(
            o_iter,
            seqlens_kv_iter,
            cu_seqlens_q_iter,
            partial_o_iter,
            partial_stats_iter,
            attention_sinks_iter,
            q_output_rows,
            cfg,
            static_full_split_prefix,
        ).launch(
            grid=(num_parallel_slices * cluster_size, h_k, b),
            block=[cfg.parallel_reduction_threads_per_cta, 1, 1],
            cluster=[cluster_size, 1, 1],
            stream=stream,
            use_pdl=cfg.use_parallel_separate_reduction_pdl,
        )
        return
    decode_gen_separate_reduction_kernel(
        o_iter,
        seqlens_kv_iter,
        cu_seqlens_q_iter,
        partial_o_iter,
        partial_stats_iter,
        attention_sinks_iter,
        q_output_rows,
        cfg,
        static_full_split_prefix,
    ).launch(
        grid=(num_reduction_slices, h_k, b),
        # Reduction parallelism is entirely in block.x; y/z are singleton
        # dimensions because grid.y/grid.z already carry head and batch.
        block=[REDUCTION_THREADS_PER_CTA, 1, 1],
        cluster=[1, 1, 1],
        stream=stream,
    )
