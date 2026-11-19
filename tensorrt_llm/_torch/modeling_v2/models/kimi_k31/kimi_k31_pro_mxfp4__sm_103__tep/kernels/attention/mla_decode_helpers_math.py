# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
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
"""Constants, math, dtype, low-level op and mask helpers for the MLA decode TS kernel.

Holds the shared instruction and layout constants, the dtype / descriptor / packed
math helpers, the reusable low-level operations, the mask policy shared by the
kernel and reference path, and the shared masking predicates.
"""

import math
from enum import Enum
from functools import partial

import cutlass
from cutlass import Float32, cute
from cutlass.experimental import primitives as prims

# CUDA warp geometry used by TMEM row addressing.
WARP_LANE_SHIFT = 5

# tcgen05 TMEM load/store instruction geometry used by MLA softmax, correction,
# and epilogue paths.
TCGEN05_32B_SHAPE = "32x32b"
TCGEN05_32B_REGS_PER_LOAD = 32

# Softmax uses base-two exponentials internally; public sparse LSE uses ln.
LN_2 = 0.6931471805599453
LOG2_E = 1.4426950408889634

# gather4 always copies four selected KV rows. This instruction width is
# independent of the number of issuing warps or the KV tile size.
TMA_GATHER_ROWS = 4

# Throughput 2CTA epilogue maps 128 local threads onto two 64-row groups and
# 128-column output halves for vectorized GMEM publication.
EPILOGUE_THREAD_TILE_THREADS = 128
EPILOGUE_THREAD_TILE_MASK = EPILOGUE_THREAD_TILE_THREADS - 1
EPILOGUE_ROW_THREADS = 64
EPILOGUE_ROW_MASK = EPILOGUE_ROW_THREADS - 1
EPILOGUE_COLUMN_GROUP_SHIFT = 6
FP8_OUTPUT_VECTOR_ELEMENTS = 16


fma_packed_f32x2 = partial(cute.arch.fma_packed_f32x2, rnd="rn")
add_packed_f32x2 = partial(cute.arch.add_packed_f32x2, rnd="rn")
mul_packed_f32x2 = partial(cute.arch.mul_packed_f32x2, rnd="rn")


def ceil_div(a, b):
    """Return integer ceil(a / b) for positive integer-like values."""
    return (a + b - 1) // b


def canonicalize_softmax_rescale_threshold_log2(threshold: object, *, qkv_dtype: str) -> float:
    """Range-check ``softmax_rescale_threshold_log2``; return it float32-rounded.

    Zero keeps the exact rescale.  A positive value is compiled into the
    kernel, which compares it with the log2-scaled row-max increase.  Any zero
    (including ``-0.0``) returns ``0.0`` so it selects the default kernel.
    """

    try:
        value = float(threshold)
    except OverflowError:
        value = math.inf
    if value == 0.0:
        # The default needs no per-dtype bound.
        return 0.0
    max_softmax_rescale_threshold_log2(qkv_dtype)
    return value


def qkv_dtype(cfg):
    """Return the shared Q/K/V element dtype for MLA decode examples."""
    if cfg.is_fp8_qkv():
        return cutlass.Float8E4M3FN
    return cutlass.BFloat16


def output_dtype(cfg):
    """Return the output element dtype for MLA decode examples."""
    return cutlass.BFloat16


def mma_kind_for_qkv(cfg):
    """Return the tcgen05 MMA kind for QK and PV operations."""
    if cfg.is_fp8_qkv():
        return prims.Tcgen05MMAKind.F8F6F4
    return prims.Tcgen05MMAKind.F16


def mma_k_step_for_qkv(cfg) -> int:
    """Return the K step consumed by one tcgen05 MMA instruction."""
    return 32 if cfg.is_fp8_qkv() else 16


def qkv_major_k_stride_bytes_for(cfg, head_dim: int) -> int:
    """Return the shared-memory major-K stride for one Q/K/V head dimension."""
    num_smem_cols = 128 // cfg.qkv_dtype_bytes
    rows_per_smem_row = max(1, num_smem_cols // head_dim)
    if rows_per_smem_row == 1:
        rows_per_swizzle_block = 8
    elif rows_per_smem_row == 2:
        rows_per_swizzle_block = 4
    return 128 * rows_per_swizzle_block


def qk_desc_layout_for_head_dim(cfg, head_dim: int):
    """Return the UMMA descriptor layout for a QK operand head dimension."""
    if cfg.is_fp8_qkv() and head_dim == 64:
        return 4
    return 2


def qk_desc_layout(cfg):
    """Return the UMMA descriptor layout for the latent QK operand."""
    return qk_desc_layout_for_head_dim(cfg, cfg.mma_qk_tiler_k)


def qk_desc_leading_byte_offset_for_head_dim(cfg, tile_rows: int, head_dim: int) -> int:
    """Return the descriptor leading byte offset for one QK head dimension."""
    if cfg.is_fp8_qkv():
        return tile_rows * head_dim * cfg.qkv_dtype_bytes
    return 16


def qk_desc_leading_byte_offset(cfg) -> int:
    """Return the descriptor leading byte offset for latent QK."""
    return qk_desc_leading_byte_offset_for_head_dim(
        cfg, cfg.mma_qk_tiler[0] // cfg.num_mma_ctas, cfg.mma_qk_tiler_k
    )


def qk_desc_stride_byte_offset_for_head_dim(cfg, head_dim: int) -> int:
    """Return the descriptor stride byte offset for one QK head dimension."""
    if cfg.is_fp8_qkv():
        return qkv_major_k_stride_bytes_for(cfg, head_dim)
    return 1024


def qk_desc_stride_byte_offset(cfg) -> int:
    """Return the descriptor stride byte offset for latent QK."""
    return qk_desc_stride_byte_offset_for_head_dim(cfg, cfg.mma_qk_tiler_k)


def p_desc_layout(cfg):
    """Return the UMMA descriptor layout for P in SMEM."""
    del cfg
    return 4


def p_desc_leading_byte_offset(cfg) -> int:
    """Return the descriptor leading byte offset for P in SMEM."""
    del cfg
    return 16


def p_desc_stride_byte_offset(cfg) -> int:
    """Return the descriptor stride byte offset for P in SMEM."""
    del cfg
    return 512


@cute.jit
def final_lse(value, base_on_e: cutlass.Constexpr[bool]):
    """Convert final statistics; partial/reduction statistics stay base2."""
    # Preserve the caller's base-2 value for subsequent split normalization.
    if cutlass.const_expr(base_on_e):
        return value * Float32(0.6931471805599453)
    return value


def max_softmax_rescale_threshold_log2(qkv_dtype: str) -> float:
    """Return the largest legal ``softmax_rescale_threshold_log2`` per P dtype.

    P reaches ``2**T``. ``"e4m3"`` caps at 8 binades: P is quantized at
    ``448 * 2**-T``, and a larger T would push more of the row into E4M3
    subnormals. ``"bf16"`` caps at 32 binades (P <= ``2**32``). Both bounds
    match the trtllm-gen FMHA ``skipCorrThreshold`` checks for E4M3 and BF16
    BMM2.
    """

    if qkv_dtype == "e4m3":
        return 8.0
    if qkv_dtype == "bf16":
        return 32.0
    raise ValueError(f"unsupported qkv_dtype={qkv_dtype!r}")


@cutlass.dsl_user_op
def fmax_f32(a, b, *, loc=None, ip=None):
    """Return the maximum of two values as Float32."""
    return Float32(
        cute.math.max(Float32(a), Float32(b), ftz=True, loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )


@cute.jit
def fp8_log2_quant_scale():
    """Return log2(448) for the E4M3 P scaling convention."""
    return Float32(8.8073549)


@cute.jit
def fp8_quant_scale_rcp():
    """Return the reciprocal of the shared E4M3 probability scale."""
    return Float32(1.0 / 448.0)


# A positive ``softmax_rescale_threshold_log2`` T keeps a row's max anchor while
# a K tile raises it by at most T binades, so P reaches 2**T. E4M3 P then uses
# the scale 448 * 2**-T, as Prims-TS FMHA decode does (``fp8_p_quant_scale``),
# so the scaled P stays within 448. This equals the trtllm-gen FMHA
# ``getSoftmaxPScale`` at T = 0 (448) and at T = 8 (1.75). T is a compile-time
# config value, so these are plain Python helpers rather than ``cute.jit``
# functions.


def fp8_log2_p_scale(cfg):
    """Return log2 of the E4M3 P scale selected by the config."""
    return fp8_log2_quant_scale()


def fp8_p_scale_rcp(cfg):
    """Return the reciprocal of ``fp8_p_scale``."""
    return fp8_quant_scale_rcp()


class MaskType(str, Enum):
    """Supported speculative-decode attention masks."""

    CAUSAL = "causal"
    DENSE = "dense"


def normalize_mask_type(mask_type: MaskType | str) -> str:
    """Return the canonical constexpr-safe string for ``mask_type``."""

    try:
        return MaskType(mask_type).value
    except (TypeError, ValueError) as error:
        supported = ", ".join(mask.value for mask in MaskType)
        raise ValueError(f"mask_type must be one of ({supported}), got {mask_type!r}") from error


def kv_tile_needs_right_mask(tile_offset_k, tile_size_kv, visible_k_end):
    """Return whether a KV tile crosses a row-visible right bound."""
    return tile_offset_k + tile_size_kv > visible_k_end
