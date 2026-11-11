# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused GLM self-attention indexer K-cache update CuTe DSL example.

The backend sequence replaced by this example is:

    indexer_k BF16 [T, 128]
      -> LayerNorm over the indexer-K row
      -> RoPE on the first 64 K lanes
      -> TRT-LLM fused FP8 cat/scale
      -> TRT-LLM indexer K-cache scatter
      -> final uint8 K cache [num_blocks, block_size, 1, 132]

The fused kernel writes the final K-cache bytes directly. It intentionally
matches the backend's materialized BF16 precision boundaries before RoPE and
before FP8 quantization, so correctness is byte-for-byte equality of the final
cache image.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream, make_fake_tensor

if TYPE_CHECKING:
    import torch


_FUSED_CAT_WARP_SIZE = 32
_FUSED_CAT_PE_DIM = 64
_FUSED_CAT_NOPE_DIM = 64
_FUSED_CAT_HEAD_DIM = _FUSED_CAT_PE_DIM + _FUSED_CAT_NOPE_DIM
_FUSED_CAT_ELEMS_PER_THREAD = 4
_FUSED_CAT_FP8_MAX = 448.0
_FUSED_CAT_INV_FP8_MAX = 1.0 / _FUSED_CAT_FP8_MAX
_FUSED_CAT_MIN_AMAX = 1.0e-12
_FUSED_CAT_SCATTER_ROWS_PER_BLOCK = 8
_FUSED_CAT_SCATTER_ALLOWED_ROWS_PER_BLOCK = (1, 2, 4, 8, 16)
_FUSED_CAT_THREADS_PER_ROW = 16
_SMALL_TOKEN_THRESHOLD = 1024
_SMALL_TOKEN_ROWS_PER_BLOCK = 2
_INDEXER_K_NORM_EPS = 1.0e-6
_INDEXER_K_CACHE_UPDATE_COMPILE_CACHE: dict[
    tuple[int, bool, bool, int, bool, bool, bool], Callable
] = {}


@cute.jit
def _indexer_k_norm_value_from_registers(
    x: cutlass.Float32,
    weight: cutlass.Float32,
    bias: cutlass.Float32,
    mean: cutlass.Float32,
    inv_std: cutlass.Float32,
) -> cutlass.Float32:
    """Apply the materialized-BF16 LayerNorm boundary to register values."""
    out = (x - mean) * inv_std * weight + bias
    return out.to(cutlass.BFloat16).to(cutlass.Float32)


@cute.jit
def _subwarp16_reduction_sum(value: cutlass.Float32) -> cutlass.Float32:
    """Reduce independently within both 16-lane halves of a physical warp."""
    active_mask = cute.arch.activemask()
    value = value + cute.arch.shuffle_sync_bfly(value, 8, mask=active_mask)
    value = value + cute.arch.shuffle_sync_bfly(value, 4, mask=active_mask)
    value = value + cute.arch.shuffle_sync_bfly(value, 2, mask=active_mask)
    return value + cute.arch.shuffle_sync_bfly(value, 1, mask=active_mask)


@cute.jit
def _subwarp16_reduction_max(value: cutlass.Float32) -> cutlass.Float32:
    """Find the maximum independently within both physical subwarps."""
    active_mask = cute.arch.activemask()
    value = cutlass.max(
        value,
        cute.arch.shuffle_sync_bfly(value, 8, mask=active_mask),
    )
    value = cutlass.max(
        value,
        cute.arch.shuffle_sync_bfly(value, 4, mask=active_mask),
    )
    value = cutlass.max(
        value,
        cute.arch.shuffle_sync_bfly(value, 2, mask=active_mask),
    )
    return cutlass.max(
        value,
        cute.arch.shuffle_sync_bfly(value, 1, mask=active_mask),
    )


@cute.jit
def _quantize_pack_fp8x4(
    x0: cutlass.Float32,
    x1: cutlass.Float32,
    x2: cutlass.Float32,
    x3: cutlass.Float32,
    inv_scale: cutlass.Float32,
) -> cutlass.Int32:
    """Quantize four register values and pack the FP8 bytes into one word."""
    q0 = (x0 * inv_scale).to(cutlass.Float8E4M3FN).bitcast(cutlass.Int8)
    q1 = (x1 * inv_scale).to(cutlass.Float8E4M3FN).bitcast(cutlass.Int8)
    q2 = (x2 * inv_scale).to(cutlass.Float8E4M3FN).bitcast(cutlass.Int8)
    q3 = (x3 * inv_scale).to(cutlass.Float8E4M3FN).bitcast(cutlass.Int8)
    return (
        (cutlass.Int32(q0) & cutlass.Int32(0xFF))
        | ((cutlass.Int32(q1) & cutlass.Int32(0xFF)) << 8)
        | ((cutlass.Int32(q2) & cutlass.Int32(0xFF)) << 16)
        | ((cutlass.Int32(q3) & cutlass.Int32(0xFF)) << 24)
    )


@cute.jit
def _unpack_bf16x4(
    packed: cutlass.Uint64,
) -> tuple[
    cutlass.Float32,
    cutlass.Float32,
    cutlass.Float32,
    cutlass.Float32,
]:
    """Unpack one naturally aligned 64-bit input word into four FP32 values."""
    x0 = cutlass.Uint16(packed & cutlass.Uint64(0xFFFF))
    x1 = cutlass.Uint16((packed >> 16) & cutlass.Uint64(0xFFFF))
    x2 = cutlass.Uint16((packed >> 32) & cutlass.Uint64(0xFFFF))
    x3 = cutlass.Uint16(packed >> 48)
    return (
        x0.bitcast(cutlass.BFloat16).to(cutlass.Float32),
        x1.bitcast(cutlass.BFloat16).to(cutlass.Float32),
        x2.bitcast(cutlass.BFloat16).to(cutlass.Float32),
        x3.bitcast(cutlass.BFloat16).to(cutlass.Float32),
    )


@cute.kernel
def self_attn_indexer_k_norm_rope_fp8_cache_update(
    indexer_k: cute.Tensor,
    norm_weight: cute.Tensor,
    norm_bias: cute.Tensor,
    position_ids: cute.Tensor,
    rotary_cos_sin: cute.Tensor,
    k_cache: cute.Tensor,
    slot_mapping_fp8: cute.Tensor,
    slot_mapping_scale: cute.Tensor,
    num_tokens: cutlass.Int32,
    cache_dim_0: cutlass.Int32,
    cache_dim_1: cutlass.Int32,
    cache_dim_2: cutlass.Int32,
    cache_dim_3: cutlass.Int32,
    cache_stride_0: cutlass.Int64,
    cache_stride_1: cutlass.Int64,
    cache_stride_2: cutlass.Int64,
    cache_stride_3: cutlass.Int64,
    USE_UE8M0: cutlass.Constexpr[bool],
    IS_NEOX: cutlass.Constexpr[bool],
    ROWS_PER_BLOCK: cutlass.Constexpr[int],
    VECTORIZED_LOADS: cutlass.Constexpr[bool],
    CACHE_CONTIGUOUS: cutlass.Constexpr[bool],
    USE_PDL: cutlass.Constexpr[bool],
) -> None:
    block_row, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    warp_in_block = tidx // _FUSED_CAT_WARP_SIZE
    lane_in_warp = tidx % _FUSED_CAT_WARP_SIZE
    lane_in_row = lane_in_warp % cutlass.Int32(_FUSED_CAT_THREADS_PER_ROW)
    row_in_warp = lane_in_warp // cutlass.Int32(_FUSED_CAT_THREADS_PER_ROW)
    row_in_block = warp_in_block * cutlass.Int32(2) + row_in_warp
    token_idx = block_row * ROWS_PER_BLOCK + row_in_block

    if cutlass.const_expr(USE_PDL):
        # Every thread waits before it can issue an indexer-K load. This avoids
        # paying an additional CTA barrier on the latency-sensitive path.
        cute.arch.griddepcontrol_wait()

    if row_in_block < ROWS_PER_BLOCK and token_idx < num_tokens:
        input_ptr = indexer_k.iterator
        weight_ptr = norm_weight.iterator
        bias_ptr = norm_bias.iterator
        row_base = token_idx * _FUSED_CAT_HEAD_DIM

        # Physical lane i owns dimensions 4*i..4*i+3 and
        # 64+4*i..64+4*i+3.
        base_col = lane_in_row * _FUSED_CAT_ELEMS_PER_THREAD
        second_base_col = base_col + cutlass.Int32(_FUSED_CAT_HEAD_DIM // 2)
        if cutlass.const_expr(VECTORIZED_LOADS):
            input_word_ptr = cute.recast_ptr(
                (input_ptr + row_base + base_col).align(8),
                dtype=cutlass.Uint64,
            )
            x0, x1, x2, x3 = _unpack_bf16x4(input_word_ptr.load())
        else:
            x0 = (input_ptr + row_base + base_col).load().to(cutlass.Float32)
            x1 = (input_ptr + row_base + base_col + cutlass.Int32(1)).load().to(cutlass.Float32)
            x2 = (input_ptr + row_base + base_col + cutlass.Int32(2)).load().to(cutlass.Float32)
            x3 = (input_ptr + row_base + base_col + cutlass.Int32(3)).load().to(cutlass.Float32)

        if cutlass.const_expr(VECTORIZED_LOADS):
            input_word_ptr = cute.recast_ptr(
                (input_ptr + row_base + second_base_col).align(8),
                dtype=cutlass.Uint64,
            )
            y0, y1, y2, y3 = _unpack_bf16x4(input_word_ptr.load())
        else:
            y0 = (input_ptr + row_base + second_base_col).load().to(cutlass.Float32)
            y1 = (
                (input_ptr + row_base + second_base_col + cutlass.Int32(1))
                .load()
                .to(cutlass.Float32)
            )
            y2 = (
                (input_ptr + row_base + second_base_col + cutlass.Int32(2))
                .load()
                .to(cutlass.Float32)
            )
            y3 = (
                (input_ptr + row_base + second_base_col + cutlass.Int32(3))
                .load()
                .to(cutlass.Float32)
            )

        local_sum = cutlass.Float32(0.0)
        local_sum = local_sum + x0
        local_sum = local_sum + x1
        local_sum = local_sum + x2
        local_sum = local_sum + x3
        second_sum = cutlass.Float32(0.0)
        second_sum = second_sum + y0
        second_sum = second_sum + y1
        second_sum = second_sum + y2
        second_sum = second_sum + y3
        local_sum = local_sum + second_sum
        row_sum = _subwarp16_reduction_sum(local_sum)
        mean = row_sum * cutlass.Float32(1.0 / _FUSED_CAT_HEAD_DIM)

        local_sq = cutlass.Float32(0.0)
        centered = x0 - mean
        local_sq = local_sq + centered * centered
        centered = x1 - mean
        local_sq = local_sq + centered * centered
        centered = x2 - mean
        local_sq = local_sq + centered * centered
        centered = x3 - mean
        local_sq = local_sq + centered * centered
        second_sq = cutlass.Float32(0.0)
        centered = y0 - mean
        second_sq = second_sq + centered * centered
        centered = y1 - mean
        second_sq = second_sq + centered * centered
        centered = y2 - mean
        second_sq = second_sq + centered * centered
        centered = y3 - mean
        second_sq = second_sq + centered * centered
        local_sq = local_sq + second_sq
        row_sq = _subwarp16_reduction_sum(local_sq)
        variance = row_sq * cutlass.Float32(1.0 / _FUSED_CAT_HEAD_DIM)
        inv_std = cute.math.rsqrt(
            variance + cutlass.Float32(_INDEXER_K_NORM_EPS),
            fastmath=True,
        )

        weight0 = (weight_ptr + base_col).load().to(cutlass.Float32)
        weight1 = (weight_ptr + base_col + cutlass.Int32(1)).load().to(cutlass.Float32)
        weight2 = (weight_ptr + base_col + cutlass.Int32(2)).load().to(cutlass.Float32)
        weight3 = (weight_ptr + base_col + cutlass.Int32(3)).load().to(cutlass.Float32)
        bias0 = (bias_ptr + base_col).load().to(cutlass.Float32)
        bias1 = (bias_ptr + base_col + cutlass.Int32(1)).load().to(cutlass.Float32)
        bias2 = (bias_ptr + base_col + cutlass.Int32(2)).load().to(cutlass.Float32)
        bias3 = (bias_ptr + base_col + cutlass.Int32(3)).load().to(cutlass.Float32)
        x0 = _indexer_k_norm_value_from_registers(
            x0,
            weight0,
            bias0,
            mean,
            inv_std,
        )
        x1 = _indexer_k_norm_value_from_registers(
            x1,
            weight1,
            bias1,
            mean,
            inv_std,
        )
        x2 = _indexer_k_norm_value_from_registers(
            x2,
            weight2,
            bias2,
            mean,
            inv_std,
        )
        x3 = _indexer_k_norm_value_from_registers(
            x3,
            weight3,
            bias3,
            mean,
            inv_std,
        )
        weight0 = (weight_ptr + second_base_col).load().to(cutlass.Float32)
        weight1 = (weight_ptr + second_base_col + cutlass.Int32(1)).load().to(cutlass.Float32)
        weight2 = (weight_ptr + second_base_col + cutlass.Int32(2)).load().to(cutlass.Float32)
        weight3 = (weight_ptr + second_base_col + cutlass.Int32(3)).load().to(cutlass.Float32)
        bias0 = (bias_ptr + second_base_col).load().to(cutlass.Float32)
        bias1 = (bias_ptr + second_base_col + cutlass.Int32(1)).load().to(cutlass.Float32)
        bias2 = (bias_ptr + second_base_col + cutlass.Int32(2)).load().to(cutlass.Float32)
        bias3 = (bias_ptr + second_base_col + cutlass.Int32(3)).load().to(cutlass.Float32)
        y0 = _indexer_k_norm_value_from_registers(
            y0,
            weight0,
            bias0,
            mean,
            inv_std,
        )
        y1 = _indexer_k_norm_value_from_registers(
            y1,
            weight1,
            bias1,
            mean,
            inv_std,
        )
        y2 = _indexer_k_norm_value_from_registers(
            y2,
            weight2,
            bias2,
            mean,
            inv_std,
        )
        y3 = _indexer_k_norm_value_from_registers(
            y3,
            weight3,
            bias3,
            mean,
            inv_std,
        )

        position = (position_ids.iterator + token_idx).load()
        table_base = cutlass.Int64(position) * cutlass.Int64(_FUSED_CAT_PE_DIM)
        cos_sin_ptr = rotary_cos_sin.iterator
        if cutlass.const_expr(IS_NEOX):
            # The NeoX partner is 32 dimensions away, or eight lanes away for
            # this four-elements-per-lane ownership. Execute the shuffle in
            # the full warp before the PE-lane predicate.
            source_lane = row_in_warp * cutlass.Int32(16) + (lane_in_row ^ cutlass.Int32(8))
            active_mask = cute.arch.activemask()
            partner0 = cute.arch.shuffle_sync(x0, source_lane, mask=active_mask)
            partner1 = cute.arch.shuffle_sync(x1, source_lane, mask=active_mask)
            partner2 = cute.arch.shuffle_sync(x2, source_lane, mask=active_mask)
            partner3 = cute.arch.shuffle_sync(x3, source_lane, mask=active_mask)
            if base_col < _FUSED_CAT_PE_DIM:
                table_dim = base_col
                if table_dim >= cutlass.Int32(_FUSED_CAT_PE_DIM // 2):
                    table_dim = table_dim - cutlass.Int32(_FUSED_CAT_PE_DIM // 2)
                cos0 = (
                    (cos_sin_ptr + table_base + cutlass.Int64(table_dim)).load().to(cutlass.Float32)
                )
                cos1 = (
                    (cos_sin_ptr + table_base + cutlass.Int64(table_dim + 1))
                    .load()
                    .to(cutlass.Float32)
                )
                cos2 = (
                    (cos_sin_ptr + table_base + cutlass.Int64(table_dim + 2))
                    .load()
                    .to(cutlass.Float32)
                )
                cos3 = (
                    (cos_sin_ptr + table_base + cutlass.Int64(table_dim + 3))
                    .load()
                    .to(cutlass.Float32)
                )
                sin_base = table_base + cutlass.Int64(_FUSED_CAT_PE_DIM // 2)
                sin0 = (
                    (cos_sin_ptr + sin_base + cutlass.Int64(table_dim)).load().to(cutlass.Float32)
                )
                sin1 = (
                    (cos_sin_ptr + sin_base + cutlass.Int64(table_dim + 1))
                    .load()
                    .to(cutlass.Float32)
                )
                sin2 = (
                    (cos_sin_ptr + sin_base + cutlass.Int64(table_dim + 2))
                    .load()
                    .to(cutlass.Float32)
                )
                sin3 = (
                    (cos_sin_ptr + sin_base + cutlass.Int64(table_dim + 3))
                    .load()
                    .to(cutlass.Float32)
                )
                out0 = x0 * cos0 + partner0 * sin0
                out1 = x1 * cos1 + partner1 * sin1
                out2 = x2 * cos2 + partner2 * sin2
                out3 = x3 * cos3 + partner3 * sin3
                if base_col < cutlass.Int32(_FUSED_CAT_PE_DIM // 2):
                    out0 = x0 * cos0 - partner0 * sin0
                    out1 = x1 * cos1 - partner1 * sin1
                    out2 = x2 * cos2 - partner2 * sin2
                    out3 = x3 * cos3 - partner3 * sin3
                x0 = out0.to(cutlass.BFloat16).to(cutlass.Float32)
                x1 = out1.to(cutlass.BFloat16).to(cutlass.Float32)
                x2 = out2.to(cutlass.BFloat16).to(cutlass.Float32)
                x3 = out3.to(cutlass.BFloat16).to(cutlass.Float32)
        else:
            if base_col < _FUSED_CAT_PE_DIM:
                table_dim0 = base_col // cutlass.Int32(2)
                table_dim1 = table_dim0 + cutlass.Int32(1)
                cos0 = (
                    (cos_sin_ptr + table_base + cutlass.Int64(table_dim0))
                    .load()
                    .to(cutlass.Float32)
                )
                cos1 = (
                    (cos_sin_ptr + table_base + cutlass.Int64(table_dim1))
                    .load()
                    .to(cutlass.Float32)
                )
                sin_base = table_base + cutlass.Int64(_FUSED_CAT_PE_DIM // 2)
                sin0 = (
                    (cos_sin_ptr + sin_base + cutlass.Int64(table_dim0)).load().to(cutlass.Float32)
                )
                sin1 = (
                    (cos_sin_ptr + sin_base + cutlass.Int64(table_dim1)).load().to(cutlass.Float32)
                )
                out0 = x0 * cos0 - x1 * sin0
                out1 = x1 * cos0 + x0 * sin0
                out2 = x2 * cos1 - x3 * sin1
                out3 = x3 * cos1 + x2 * sin1
                x0 = out0.to(cutlass.BFloat16).to(cutlass.Float32)
                x1 = out1.to(cutlass.BFloat16).to(cutlass.Float32)
                x2 = out2.to(cutlass.BFloat16).to(cutlass.Float32)
                x3 = out3.to(cutlass.BFloat16).to(cutlass.Float32)

        ax0 = cute.math.absf(x0)
        ax1 = cute.math.absf(x1)
        ax2 = cute.math.absf(x2)
        ax3 = cute.math.absf(x3)
        local_max = ax0
        if ax1 > local_max:
            local_max = ax1
        if ax2 > local_max:
            local_max = ax2
        if ax3 > local_max:
            local_max = ax3

        ay0 = cute.math.absf(y0)
        ay1 = cute.math.absf(y1)
        ay2 = cute.math.absf(y2)
        ay3 = cute.math.absf(y3)
        if ay0 > local_max:
            local_max = ay0
        if ay1 > local_max:
            local_max = ay1
        if ay2 > local_max:
            local_max = ay2
        if ay3 > local_max:
            local_max = ay3
        amax = _subwarp16_reduction_max(local_max)
        safe_amax = amax
        if safe_amax < cutlass.Float32(_FUSED_CAT_MIN_AMAX):
            safe_amax = cutlass.Float32(_FUSED_CAT_MIN_AMAX)

        scale = safe_amax * cutlass.Float32(_FUSED_CAT_INV_FP8_MAX)
        if cutlass.const_expr(USE_UE8M0):
            scale_bits = scale.bitcast(cutlass.Int32)
            mantissa = scale_bits & cutlass.Int32(0x007FFFFF)
            exp_bits = scale_bits & cutlass.Int32(0x7F800000)
            if mantissa != cutlass.Int32(0):
                exp_bits += cutlass.Int32(0x00800000)
            scale = exp_bits.bitcast(cutlass.Float32)
            inv_scale_bits = cutlass.Int32(0x7F000000) - exp_bits
            inv_scale = inv_scale_bits.bitcast(cutlass.Float32)
        else:
            inv_scale = cutlass.Float32(1.0) / scale

        packed = _quantize_pack_fp8x4(x0, x1, x2, x3, inv_scale)
        second_packed = _quantize_pack_fp8x4(y0, y1, y2, y3, inv_scale)

        flat_idx_fp8_base = (slot_mapping_fp8.iterator + token_idx).load()
        flat_idx_scale_base = (slot_mapping_scale.iterator + token_idx).load()
        if flat_idx_fp8_base >= 0 and flat_idx_scale_base >= 0:
            if cutlass.const_expr(CACHE_CONTIGUOUS):
                dst_offset = flat_idx_fp8_base + cutlass.Int64(base_col)
            else:
                dst_offset = _indexer_scatter_flat_to_strided_offset(
                    flat_idx_fp8_base + cutlass.Int64(base_col),
                    cache_dim_0,
                    cache_dim_1,
                    cache_dim_2,
                    cache_dim_3,
                    cache_stride_0,
                    cache_stride_1,
                    cache_stride_2,
                    cache_stride_3,
                )
            dst_word_ptr = cute.recast_ptr(
                (k_cache.iterator + dst_offset).align(4),
                dtype=cutlass.Int32,
            )
            dst_word_ptr.store(packed)

            if cutlass.const_expr(CACHE_CONTIGUOUS):
                second_dst_offset = flat_idx_fp8_base + cutlass.Int64(second_base_col)
            else:
                second_dst_offset = _indexer_scatter_flat_to_strided_offset(
                    flat_idx_fp8_base + cutlass.Int64(second_base_col),
                    cache_dim_0,
                    cache_dim_1,
                    cache_dim_2,
                    cache_dim_3,
                    cache_stride_0,
                    cache_stride_1,
                    cache_stride_2,
                    cache_stride_3,
                )
            second_dst_word_ptr = cute.recast_ptr(
                (k_cache.iterator + second_dst_offset).align(4),
                dtype=cutlass.Int32,
            )
            second_dst_word_ptr.store(second_packed)

            if lane_in_row == 0:
                if cutlass.const_expr(CACHE_CONTIGUOUS):
                    dst_offset_scale = flat_idx_scale_base
                else:
                    dst_offset_scale = _indexer_scatter_flat_to_strided_offset(
                        flat_idx_scale_base,
                        cache_dim_0,
                        cache_dim_1,
                        cache_dim_2,
                        cache_dim_3,
                        cache_stride_0,
                        cache_stride_1,
                        cache_stride_2,
                        cache_stride_3,
                    )
                dst_scale_word_ptr = cute.recast_ptr(
                    (k_cache.iterator + dst_offset_scale).align(4),
                    dtype=cutlass.Int32,
                )
                dst_scale_word_ptr.store(scale.bitcast(cutlass.Int32))

    # Do not explicitly launch dependents here. The final cache store is at the
    # end of the kernel, so there is no useful producer tail to overlap. Kernel
    # completion provides the required implicit PDL release without paying for
    # an explicit griddepcontrol signal on this latency-sensitive path.


@cute.jit
def _indexer_scatter_flat_to_strided_offset(
    flat_idx: cutlass.Int64,
    cache_dim_0: cutlass.Int32,
    cache_dim_1: cutlass.Int32,
    cache_dim_2: cutlass.Int32,
    cache_dim_3: cutlass.Int32,
    cache_stride_0: cutlass.Int64,
    cache_stride_1: cutlass.Int64,
    cache_stride_2: cutlass.Int64,
    cache_stride_3: cutlass.Int64,
) -> cutlass.Int64:
    lower_dim_size = (
        cutlass.Int64(cache_dim_1) * cutlass.Int64(cache_dim_2) * cutlass.Int64(cache_dim_3)
    )
    lower_dims_contiguous = (
        cache_stride_3 == cutlass.Int64(1)
        and cache_stride_2 == cutlass.Int64(cache_dim_3)
        and cache_stride_1 == cutlass.Int64(cache_dim_2) * cutlass.Int64(cache_dim_3)
    )
    offset = cutlass.Int64(0)
    if lower_dims_contiguous:
        # TRT-LLM's per-layer view has a gap between blocks, but each block is
        # contiguous. Collapse the lower three dimensions to avoid the full
        # four-dimensional Int64 div/mod chain for every cache store.
        i0 = flat_idx // lower_dim_size
        intra_block = flat_idx - i0 * lower_dim_size
        offset = i0 * cache_stride_0 + intra_block
    else:
        i3 = flat_idx % cutlass.Int64(cache_dim_3)
        flat_idx = flat_idx // cutlass.Int64(cache_dim_3)
        i2 = flat_idx % cutlass.Int64(cache_dim_2)
        flat_idx = flat_idx // cutlass.Int64(cache_dim_2)
        i1 = flat_idx % cutlass.Int64(cache_dim_1)
        flat_idx = flat_idx // cutlass.Int64(cache_dim_1)
        i0 = flat_idx % cutlass.Int64(cache_dim_0)
        offset = (
            i0 * cache_stride_0 + i1 * cache_stride_1 + i2 * cache_stride_2 + i3 * cache_stride_3
        )
    return offset


@cute.jit
def _launch_indexer_k_norm_rope_fp8_cache_update(
    indexer_k: cute.Tensor,
    norm_weight: cute.Tensor,
    norm_bias: cute.Tensor,
    position_ids: cute.Tensor,
    rotary_cos_sin: cute.Tensor,
    k_cache: cute.Tensor,
    slot_mapping_fp8: cute.Tensor,
    slot_mapping_scale: cute.Tensor,
    num_tokens: cutlass.Int32,
    cache_stride_0: cutlass.Int64,
    cache_stride_1: cutlass.Int64,
    cache_stride_2: cutlass.Int64,
    cache_stride_3: cutlass.Int64,
    stream: cuda.CUstream,
    USE_UE8M0: cutlass.Constexpr[bool],
    IS_NEOX: cutlass.Constexpr[bool],
    ROWS_PER_BLOCK: cutlass.Constexpr[int],
    VECTORIZED_LOADS: cutlass.Constexpr[bool],
    CACHE_CONTIGUOUS: cutlass.Constexpr[bool],
    USE_PDL: cutlass.Constexpr[bool],
) -> None:
    cache_dim_0 = cute.size(k_cache, mode=[0])
    cache_dim_1 = cute.size(k_cache, mode=[1])
    cache_dim_2 = cute.size(k_cache, mode=[2])
    cache_dim_3 = cute.size(k_cache, mode=[3])

    self_attn_indexer_k_norm_rope_fp8_cache_update(
        indexer_k,
        norm_weight,
        norm_bias,
        position_ids,
        rotary_cos_sin,
        k_cache,
        slot_mapping_fp8,
        slot_mapping_scale,
        num_tokens,
        cache_dim_0,
        cache_dim_1,
        cache_dim_2,
        cache_dim_3,
        cache_stride_0,
        cache_stride_1,
        cache_stride_2,
        cache_stride_3,
        USE_UE8M0,
        IS_NEOX,
        ROWS_PER_BLOCK,
        VECTORIZED_LOADS,
        CACHE_CONTIGUOUS,
        USE_PDL,
    ).launch(
        grid=(cute.ceil_div(num_tokens, ROWS_PER_BLOCK), 1, 1),
        block=(
            _FUSED_CAT_WARP_SIZE
            * cute.ceil_div(
                ROWS_PER_BLOCK * _FUSED_CAT_THREADS_PER_ROW,
                _FUSED_CAT_WARP_SIZE,
            ),
            1,
            1,
        ),
        stream=stream,
        use_pdl=USE_PDL,
    )


def compile_indexer_k_norm_rope_fp8_cache_update(
    device_id: int = 0,
    use_ue8m0: bool = True,
    is_neox: bool = True,
    rows_per_block: int = _FUSED_CAT_SCATTER_ROWS_PER_BLOCK,
    *,
    vectorized_loads: bool = False,
    cache_contiguous: bool = False,
    use_pdl: bool = True,
) -> Callable:
    """Compile the fused kernel for one launch specialization.

    ``cache_contiguous`` defaults to the generic-stride path so separately
    compiled callables remain correct for every cache layout accepted by the
    public wrapper. ``use_pdl`` defaults on for the production launch chain:
    the kernel waits on its incoming dependency and relies on kernel completion
    to release a following PDL consumer. Callers can disable it for standalone
    use. The wrapper specializes these options automatically when it owns
    compilation.
    """
    if rows_per_block not in _FUSED_CAT_SCATTER_ALLOWED_ROWS_PER_BLOCK:
        raise ValueError(
            "rows_per_block must be one of "
            f"{_FUSED_CAT_SCATTER_ALLOWED_ROWS_PER_BLOCK}, got {rows_per_block}"
        )
    key = (
        device_id,
        bool(use_ue8m0),
        bool(is_neox),
        rows_per_block,
        bool(vectorized_loads),
        bool(cache_contiguous),
        bool(use_pdl),
    )
    if key in _INDEXER_K_CACHE_UPDATE_COMPILE_CACHE:
        return _INDEXER_K_CACHE_UPDATE_COMPILE_CACHE[key]

    cutlass.cuda.initialize_cuda_context(device_id)

    sym_num_tokens = cute.sym_int32()
    sym_max_positions = cute.sym_int32()
    sym_num_blocks = cute.sym_int32()
    sym_block_size = cute.sym_int32()
    sym_cache_token_bytes = cute.sym_int32()
    sym_cache_stride_0 = cute.sym_int64(symbol="indexer_k_cache_update_stride_0")
    sym_cache_stride_1 = cute.sym_int64(symbol="indexer_k_cache_update_stride_1")
    sym_cache_stride_2 = cute.sym_int64(symbol="indexer_k_cache_update_stride_2")

    fake_indexer_k = make_fake_compact_tensor(
        cutlass.BFloat16,
        (sym_num_tokens, _FUSED_CAT_HEAD_DIM),
        stride_order=(1, 0),
        assumed_align=8 if vectorized_loads else 2,
    )
    fake_norm_weight = make_fake_compact_tensor(
        cutlass.Float32,
        (_FUSED_CAT_HEAD_DIM,),
        assumed_align=4,
    )
    fake_norm_bias = make_fake_compact_tensor(
        cutlass.Float32,
        (_FUSED_CAT_HEAD_DIM,),
        assumed_align=4,
    )
    fake_position_ids = make_fake_compact_tensor(
        cutlass.Int32,
        (sym_num_tokens,),
        assumed_align=4,
    )
    fake_cos_sin = make_fake_compact_tensor(
        cutlass.Float32,
        (sym_max_positions, 2, _FUSED_CAT_PE_DIM // 2),
        stride_order=(2, 1, 0),
        assumed_align=4,
    )
    fake_k_cache = make_fake_tensor(
        cutlass.Uint8,
        (sym_num_blocks, sym_block_size, 1, sym_cache_token_bytes),
        stride=(sym_cache_stride_0, sym_cache_stride_1, sym_cache_stride_2, 1),
        assumed_align=4,
    )
    fake_slot_mapping_fp8 = make_fake_compact_tensor(
        cutlass.Int64,
        (sym_num_tokens,),
        assumed_align=8,
    )
    fake_slot_mapping_scale = make_fake_compact_tensor(
        cutlass.Int64,
        (sym_num_tokens,),
        assumed_align=8,
    )

    compiled = cute.compile(
        _launch_indexer_k_norm_rope_fp8_cache_update,
        fake_indexer_k,
        fake_norm_weight,
        fake_norm_bias,
        fake_position_ids,
        fake_cos_sin,
        fake_k_cache,
        fake_slot_mapping_fp8,
        fake_slot_mapping_scale,
        cutlass.Int32(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(0),
        cutlass.Int64(1),
        make_fake_stream(),
        bool(use_ue8m0),
        bool(is_neox),
        rows_per_block,
        bool(vectorized_loads),
        bool(cache_contiguous),
        bool(use_pdl),
        options="--enable-tvm-ffi",
    )
    _INDEXER_K_CACHE_UPDATE_COMPILE_CACHE[key] = compiled
    return compiled


def _current_cuda_stream(torch, device_id: int | None = None):
    stream = torch.cuda.current_stream(device_id)
    return cuda.CUstream(stream.cuda_stream)


def _require_cuda_contiguous(torch, tensor, name: str, dtype) -> None:
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}")
    if not tensor.is_cuda:
        raise TypeError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _require_current_cuda_device(
    torch,
    tensors: tuple[tuple["torch.Tensor", str], ...],
    current_device: int,
) -> None:
    for tensor, name in tensors:
        device_index = tensor.device.index
        if device_index is None:
            device_index = current_device
        if int(device_index) != int(current_device):
            raise ValueError(f"{name} must be on cuda:{current_device}, got {tensor.device}")


def _can_vectorize_loads(indexer_k) -> bool:
    """Return whether the packed indexer-K input load is naturally aligned."""
    input_address = indexer_k.data_ptr()
    return input_address % 8 == 0


def _select_rows_per_block(num_tokens: int) -> int:
    """Choose enough blocks to avoid underfilling small-token launches."""
    if num_tokens <= _SMALL_TOKEN_THRESHOLD:
        return _SMALL_TOKEN_ROWS_PER_BLOCK
    return _FUSED_CAT_SCATTER_ROWS_PER_BLOCK


def _has_any(torch, predicate) -> bool:
    return bool(torch.any(predicate).item())


def _require_indexer_k_cache_update_ranges(
    torch,
    position_ids,
    rotary_cos_sin,
    k_cache,
    slot_mapping_fp8,
    slot_mapping_scale,
    num_tokens: int,
) -> None:
    active_positions = position_ids[:num_tokens]
    if _has_any(
        torch,
        (active_positions < 0) | (active_positions >= rotary_cos_sin.shape[0]),
    ):
        raise ValueError("position_ids must be in [0, rotary_cos_sin.shape[0])")

    active_fp8 = slot_mapping_fp8[:num_tokens]
    active_scale = slot_mapping_scale[:num_tokens]
    if _has_any(torch, active_fp8 < -1) or _has_any(torch, active_scale < -1):
        raise ValueError("slot mapping entries must be -1 or non-negative offsets")

    valid_fp8 = active_fp8 >= 0
    valid_scale = active_scale >= 0
    if _has_any(torch, valid_fp8 != valid_scale):
        raise ValueError("slot_mapping_fp8 and slot_mapping_scale must have matching validity")

    logical_cache_elems = int(
        k_cache.shape[0] * k_cache.shape[1] * k_cache.shape[2] * k_cache.shape[3]
    )
    max_fp8_offset = logical_cache_elems - _FUSED_CAT_HEAD_DIM
    max_scale_offset = logical_cache_elems - 4
    if _has_any(
        torch,
        valid_fp8 & (((active_fp8 % 4) != 0) | (active_fp8 > max_fp8_offset)),
    ):
        raise ValueError(
            "slot_mapping_fp8 entries must be 4-byte aligned and leave room "
            f"for {_FUSED_CAT_HEAD_DIM} bytes"
        )
    if _has_any(
        torch,
        valid_scale & (((active_scale % 4) != 0) | (active_scale > max_scale_offset)),
    ):
        raise ValueError(
            "slot_mapping_scale entries must be 4-byte aligned and leave room for 4 bytes"
        )


def _require_indexer_k_cache_update_inputs(
    torch,
    indexer_k,
    norm_weight,
    norm_bias,
    position_ids,
    rotary_cos_sin,
    k_cache,
    slot_mapping_fp8,
    slot_mapping_scale,
) -> None:
    _require_cuda_contiguous(torch, indexer_k, "indexer_k", torch.bfloat16)
    _require_cuda_contiguous(torch, norm_weight, "norm_weight", torch.float32)
    _require_cuda_contiguous(torch, norm_bias, "norm_bias", torch.float32)
    _require_cuda_contiguous(torch, position_ids, "position_ids", torch.int32)
    _require_cuda_contiguous(torch, rotary_cos_sin, "rotary_cos_sin", torch.float32)
    if indexer_k.ndim != 2 or indexer_k.shape[1] != _FUSED_CAT_HEAD_DIM:
        raise ValueError(
            f"indexer_k must have shape [T,{_FUSED_CAT_HEAD_DIM}], got {tuple(indexer_k.shape)}"
        )
    if tuple(norm_weight.shape) != (_FUSED_CAT_HEAD_DIM,):
        raise ValueError(
            f"norm_weight must have shape [{_FUSED_CAT_HEAD_DIM}], got {tuple(norm_weight.shape)}"
        )
    if tuple(norm_bias.shape) != (_FUSED_CAT_HEAD_DIM,):
        raise ValueError(
            f"norm_bias must have shape [{_FUSED_CAT_HEAD_DIM}], got {tuple(norm_bias.shape)}"
        )
    if position_ids.shape != (indexer_k.shape[0],):
        raise ValueError(
            f"position_ids must have shape [{indexer_k.shape[0]}], got {tuple(position_ids.shape)}"
        )
    if rotary_cos_sin.ndim != 3 or tuple(rotary_cos_sin.shape[1:]) != (
        2,
        _FUSED_CAT_PE_DIM // 2,
    ):
        raise ValueError(
            "rotary_cos_sin must have shape [max_positions,2,32], "
            f"got {tuple(rotary_cos_sin.shape)}"
        )
    if k_cache.dtype != torch.uint8:
        raise TypeError(f"k_cache must be torch.uint8, got {k_cache.dtype}")
    if not k_cache.is_cuda:
        raise TypeError("k_cache must be a CUDA tensor")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError(
            "k_cache must be 4D [num_blocks, block_size, 1, per_token_size], "
            f"got {tuple(k_cache.shape)}"
        )
    if k_cache.shape[3] < _FUSED_CAT_HEAD_DIM + 4:
        raise ValueError(
            f"k_cache.shape[3] must be at least {_FUSED_CAT_HEAD_DIM + 4}, got {k_cache.shape[3]}"
        )
    if k_cache.stride(-1) != 1:
        raise ValueError(
            "k_cache must have innermost stride 1 for vectorized byte stores, "
            f"got stride={tuple(k_cache.stride())}"
        )

    for tensor, name in (
        (slot_mapping_fp8, "slot_mapping_fp8"),
        (slot_mapping_scale, "slot_mapping_scale"),
    ):
        if tensor.dtype != torch.int64:
            raise TypeError(f"{name} must be torch.int64, got {tensor.dtype}")
        if not tensor.is_cuda:
            raise TypeError(f"{name} must be a CUDA tensor")
        if tensor.ndim != 1:
            raise ValueError(f"{name} must be 1D, got {tuple(tensor.shape)}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")


def indexer_k_norm_rope_fp8_cache_update(
    indexer_k,
    norm_weight,
    norm_bias,
    position_ids,
    rotary_cos_sin,
    k_cache,
    slot_mapping_fp8,
    slot_mapping_scale,
    num_tokens: int,
    use_ue8m0: bool = True,
    is_neox: bool = True,
    compiled_fn: Callable | None = None,
    rows_per_block: int | None = None,
    validate: bool = True,
    *,
    use_pdl: bool = True,
) -> None:
    """Run fused GLM indexer-K norm + RoPE + FP8 cache update.

    When this wrapper owns compilation, ``rows_per_block=None`` selects two
    rows for up to 1024 tokens and eight rows for larger workloads. Callers
    that supply ``compiled_fn`` retain control of that specialization.
    """
    import torch

    current_device = torch.cuda.current_device()
    if validate:
        _require_indexer_k_cache_update_inputs(
            torch,
            indexer_k,
            norm_weight,
            norm_bias,
            position_ids,
            rotary_cos_sin,
            k_cache,
            slot_mapping_fp8,
            slot_mapping_scale,
        )
        if num_tokens < 0 or num_tokens > indexer_k.shape[0]:
            raise ValueError(f"num_tokens must be in [0, {indexer_k.shape[0]}], got {num_tokens}")
        if slot_mapping_fp8.shape[0] < num_tokens or slot_mapping_scale.shape[0] < num_tokens:
            raise ValueError("slot mapping tensors must contain at least num_tokens entries")
        _require_current_cuda_device(
            torch,
            (
                (indexer_k, "indexer_k"),
                (norm_weight, "norm_weight"),
                (norm_bias, "norm_bias"),
                (position_ids, "position_ids"),
                (rotary_cos_sin, "rotary_cos_sin"),
                (k_cache, "k_cache"),
                (slot_mapping_fp8, "slot_mapping_fp8"),
                (slot_mapping_scale, "slot_mapping_scale"),
            ),
            current_device,
        )
        _require_indexer_k_cache_update_ranges(
            torch,
            position_ids,
            rotary_cos_sin,
            k_cache,
            slot_mapping_fp8,
            slot_mapping_scale,
            num_tokens,
        )

    if compiled_fn is None:
        if rows_per_block is None:
            rows_per_block = _select_rows_per_block(num_tokens)
        compiled_fn = compile_indexer_k_norm_rope_fp8_cache_update(
            device_id=current_device,
            use_ue8m0=bool(use_ue8m0),
            is_neox=bool(is_neox),
            rows_per_block=rows_per_block,
            vectorized_loads=_can_vectorize_loads(indexer_k),
            cache_contiguous=bool(k_cache.is_contiguous()),
            use_pdl=bool(use_pdl),
        )

    active_rows = int(indexer_k.shape[0])
    active_slot_mapping_fp8 = slot_mapping_fp8[:active_rows]
    active_slot_mapping_scale = slot_mapping_scale[:active_rows]

    compiled_fn(
        indexer_k,
        norm_weight,
        norm_bias,
        position_ids,
        rotary_cos_sin,
        k_cache,
        active_slot_mapping_fp8,
        active_slot_mapping_scale,
        int(num_tokens),
        int(k_cache.stride(0)),
        int(k_cache.stride(1)),
        int(k_cache.stride(2)),
        int(k_cache.stride(3)),
        _current_cuda_stream(torch, current_device),
    )
