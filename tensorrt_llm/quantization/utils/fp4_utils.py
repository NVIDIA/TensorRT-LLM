# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tensorrt_llm.models.modeling_utils import QuantConfig

# The declarations must be aligned with thUtils.h
SF_DTYPE = torch.uint8
FLOAT4_E2M1X2 = torch.uint8

# Number of elements sharing one NVFP4 E4M3 scale factor
# (SF_VEC_SIZE in cpp/tensorrt_llm/kernels/quantization.h).
NVFP4_SF_VEC_SIZE = 16

# Scale block widths the dense W4A16_NVFP4 Linear dequantization path accepts.
# W4A16_NVFP4 MoE remains restricted to 16-element blocks.
# The W4A4 NVFP4 GEMMs are hardware-bound to NVFP4_SF_VEC_SIZE; ModelOpt's
# nvfp4_*_weight_only recipes may export 32-element blocks.
W4A16_NVFP4_LINEAR_SF_VEC_SIZES = (16, 32)

# NVFP4_PLUS: E2M1 values with one UE5M3 scale per 32 values and no per-tensor
# scale. UE5M3 has 5 exponent bits (bias 15), 3 mantissa bits and subnormals;
# codes 0x00-0xFE are finite (0 to 114688) and 0xFF is NaN.
NVFP4_PLUS_SF_VEC_SIZE = 32
UE5M3_MAX = 114688.0
UE5M3_NAN = 0xFF

# For GEMM autotuning.
# Taken from https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/include/tensorrt_llm/runtime//modelConfig.h#L38
# TODO: move to model config, tune for blackwell hardware
FP4_BUCKETS = [64, 128, 256, 512, 1024]

# Export
float4_e2m1x2 = FLOAT4_E2M1X2
float4_sf_dtype = SF_DTYPE
fp4_buckets = FP4_BUCKETS

__all__ = [
    'float4_e2m1x2', 'float4_sf_dtype', 'pad_up', 'fp4_buckets',
    'NVFP4_SF_VEC_SIZE', 'W4A16_NVFP4_LINEAR_SF_VEC_SIZES',
    'nvfp4_scaling_vector_size', 'NVFP4_PLUS_SF_VEC_SIZE', 'UE5M3_MAX',
    'ue5m3_to_float', 'float_to_ue5m3', 'e8m0_to_ue5m3',
    'nvfp4_plus_quantize_ref', 'nvfp4_plus_dequantize_ref'
]


def nvfp4_scaling_vector_size(quant_config: QuantConfig | None) -> int:
    """Resolve a checkpoint's NVFP4 scale width without reinterpreting explicit values.

    An omitted group_size retains the standard NVFP4 width even though
    QuantConfig defaults to 128 for AWQ/GPTQ. Explicit values, including 128,
    are passed through for algorithm-specific validation.
    """
    if quant_config is None or "group_size" not in quant_config.model_fields_set:
        return NVFP4_SF_VEC_SIZE
    return NVFP4_SF_VEC_SIZE if quant_config.group_size is None else quant_config.group_size


def pad_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


class FP4GemmType(IntEnum):
    W4A4_NVFP4_NVFP4 = 0
    W4A8_MXFP4_MXFP8 = 1


def get_fp4_shape(input_shape, sf_vec_size, is_swizzled_layout=True):
    m = 1
    for i in range(len(input_shape) - 1):
        m *= input_shape[i]

    output_shape = [i for i in input_shape]
    output_shape[-1] //= 2

    scale_shape = pad_up(m, 128) * pad_up(
        input_shape[-1] // sf_vec_size,
        4) if is_swizzled_layout else m * (input_shape[-1] // sf_vec_size)
    return output_shape, scale_shape


def get_reorder_rows_for_gated_act_gemm_row_indices(x) -> torch.Tensor:
    """
    Reorders rows in the gemm/MOE_gemm weight matrix for min-latency
    [r0, r1, r2, r3, ..., rN/2, r(N/2+1), .. r(N-1)]
    to
    [r0, rN/2, r1, rN/2+1, ..., r(N/2-1), r(N-1)]
    """
    M = x.shape[0]
    assert M % 2 == 0, f"x.shape[0] must be even, not {M}"

    row_indices = torch.arange(M, dtype=torch.long)

    # We split into top half and bottom half, but if M is odd,
    # the bottom half is one row larger.
    top = row_indices[:(M + 1) // 2]  # round up
    bot = row_indices[(M + 1) // 2:]  # remainder

    # Create the output
    permuted_row_indices = torch.empty_like(row_indices)

    # We'll place rows of `top` and `bot` in alternation
    permuted_row_indices[0::2] = top
    permuted_row_indices[1::2] = bot

    return permuted_row_indices


def reorder_rows_for_gated_act_gemm(x):
    """
    PyTorch implementation of trt-llm gen `reorderRowsForGatedActGemm`
    """
    row_indices = get_reorder_rows_for_gated_act_gemm_row_indices(x)

    permute = lambda x: x[row_indices]

    return permute(x)


# yapf: disable
srcToDstBlk16RowMap = [
    0,  8,
    1,  9,
    2, 10,
    3, 11,
    4, 12,
    5, 13,
    6, 14,
    7, 15
]

srcToDstBlk32RowMap = [
    0,  8, 16, 24,
    1,  9, 17, 25,
    2, 10, 18, 26,
    3, 11, 19, 27,
    4, 12, 20, 28,
    5, 13, 21, 29,
    6, 14, 22, 30,
    7, 15, 23, 31
]
# yapf: enable


def get_shuffle_block_size(epilogue_tile_m: int) -> int:
    shuffle_block_size = 16
    if epilogue_tile_m % 128 == 0:
        shuffle_block_size = 32
    return shuffle_block_size


def get_shuffle_matrix_a_row_indices(input_tensor: torch.Tensor,
                                     epilogue_tile_m: int) -> torch.Tensor:
    """
    Higher-level PyTorch approach to reorder the rows in blocks of size 16 or 32.
    - We do NOT try to handle custom e2m1 memory usage (i.e. no 'K/2' bytes).
    - Instead, we purely reorder rows in a standard PyTorch shape [M, K].
    """
    # M from the input
    M = input_tensor.shape[0]

    # Choose block size 16 or 32
    shuffle_block_size = get_shuffle_block_size(epilogue_tile_m)
    row_map = (srcToDstBlk16RowMap
               if shuffle_block_size == 16 else srcToDstBlk32RowMap)

    assert M % shuffle_block_size == 0, f"input_tensor.shape[0] must be multiples of {shuffle_block_size}"

    # row_indices[new_row] = old_row
    # so row_indices is an array of size M telling us from which old_row
    # the new_row should be taken.
    row_indices = torch.empty(M, dtype=torch.long)

    for old_row in range(M):
        block_idx = old_row // shuffle_block_size
        row_in_block = old_row % shuffle_block_size
        mapped_row_in_block = row_map[row_in_block]

        new_row = block_idx * shuffle_block_size + mapped_row_in_block

        row_indices[new_row] = old_row

    return row_indices


def shuffle_matrix_a(input_tensor: torch.Tensor,
                     epilogue_tile_m: int) -> torch.Tensor:
    """
    PyTorch equivalent of trtllm-gen `shuffleMatrixA`
    """
    row_indices = get_shuffle_matrix_a_row_indices(input_tensor,
                                                   epilogue_tile_m)

    return torch.ops.trtllm.shuffle_matrix(input_tensor,
                                           row_indices.to(input_tensor.device))


def get_shuffle_matrix_sf_a_row_indices(
        input_tensor: torch.Tensor,
        epilogue_tile_m: int,
        num_elts_per_sf: int = 16) -> torch.Tensor:

    assert input_tensor.dtype == float4_sf_dtype
    assert num_elts_per_sf == 16 or num_elts_per_sf == 32

    assert input_tensor.dim(
    ) == 2, f"input_tensor should be a 2D tensor, not {input_tensor.dim()}"

    # M, K from the input
    M, K = input_tensor.shape
    assert M % 128 == 0
    assert K % 4 == 0

    row_indices = get_shuffle_matrix_a_row_indices(input_tensor,
                                                   epilogue_tile_m)

    return row_indices


def shuffle_matrix_sf_a(
    input_tensor: torch.Tensor,
    epilogue_tile_m: int,
    num_elts_per_sf: int = 16,
):
    """
    Cuda implementation of trtllm-gen `shuffleMatrixSfA` but with a caveat.
    `shuffleMatrixSfA` expects the input to be in 128x4 layout and then
    apply the same shuffling in `shuffleMatrixA` and writes out in 128x4
    layout.
    This function expects the input to be in linear layout. It's done this
    way because the scaling factors in the NVFP4 checkpoints are quantized
    and are in linear layout.
    This function doesn't add padding.
    """

    row_indices = get_shuffle_matrix_sf_a_row_indices(input_tensor,
                                                      epilogue_tile_m)

    w_shuffled = torch.ops.trtllm.shuffle_matrix(
        input_tensor, row_indices.to(input_tensor.device))

    # 128x4
    return torch.ops.trtllm.block_scale_interleave(w_shuffled)


def ue5m3_to_float(codes: torch.Tensor) -> torch.Tensor:
    """Decode UE5M3 scale codes (uint8) to float32; 0xFF decodes to NaN."""
    code = codes.to(torch.int32)
    exponent = code >> 3
    mantissa = code & 7
    normal = (((exponent + 112) << 23) | (mantissa << 20)).view(torch.float32)
    subnormal = mantissa.to(torch.float32) * 2.0**-17
    value = torch.where(exponent == 0, subnormal, normal)
    return value.masked_fill(code == UE5M3_NAN, float("nan"))


def float_to_ue5m3(value: torch.Tensor) -> torch.Tensor:
    """Round non-negative scales up to UE5M3 codes (uint8).

    Matches ``cvt.rp.ue5m3x2.f32`` for finite inputs, except that values above
    the largest finite UE5M3 value saturate to it instead of becoming NaN.
    """
    v = value.to(torch.float32).clamp(max=UE5M3_MAX)
    # Normal range: round the FP32 mantissa up to 3 bits (a carry moves into
    # the exponent), then rebias the exponent from 127 to 15.
    normal = ((v.view(torch.int32) + 0xFFFFF) >> 20) - (112 << 3)
    # Below 2**-14 the codes are multiples of 2**-17, so the code is the count.
    subnormal = torch.ceil(v * 2.0**17).to(torch.int32)
    code = torch.where(v < 2.0**-14, subnormal, normal).clamp(0, UE5M3_NAN - 1)
    return code.masked_fill(torch.isnan(v), UE5M3_NAN).to(torch.uint8)


def e8m0_to_ue5m3(e8m0: torch.Tensor) -> torch.Tensor:
    """Convert MXFP4 E8M0 block scales to UE5M3 codes without changing values.

    E8M0 code ``e`` is ``2**(e - 127)``; UE5M3 holds every power of two from
    2**-17 to 2**16. Smaller scales become 0: their blocks are below
    6 * 2**-18 in magnitude. Larger scales and NaN raise ``ValueError``.
    """
    exponent = e8m0.view(torch.uint8).to(torch.int32) - 127
    if bool((exponent > 16).any()):
        raise ValueError(
            "E8M0 scale above 2**16 (or NaN) cannot be represented in UE5M3")
    normal = (exponent + 15) << 3
    subnormal = 1 << (exponent + 17).clamp(min=0)
    code = torch.where(exponent >= -14, normal,
                       torch.where(exponent >= -17, subnormal, 0))
    return code.to(torch.uint8)


def _float_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round to E2M1 codes (uint8, 0-15) to nearest even, saturating at 6."""
    mag = x.abs()
    code = ((mag > 0.25).to(torch.int32) + (mag >= 0.75) + (mag > 1.25) +
            (mag >= 1.75) + (mag > 2.5) + (mag >= 3.5) + (mag > 5.0))
    return (code + torch.signbit(x) * 8).to(torch.uint8)


def nvfp4_plus_quantize_ref(
        x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` [..., K] (K % 32 == 0) to NVFP4_PLUS in plain PyTorch.

    Returns the packed E2M1 values [..., K // 2] (element 2j in the low nibble
    of byte j) and the UE5M3 scales [..., K // 32] in row-major order.
    """
    blocks = x.float().unflatten(-1, (-1, NVFP4_PLUS_SF_VEC_SIZE))
    sf = float_to_ue5m3(blocks.abs().amax(-1) / 6.0)
    scale = ue5m3_to_float(sf)
    inv_scale = torch.where(scale > 0, 1.0 / scale, torch.zeros_like(scale))
    e2m1 = _float_to_e2m1(blocks * inv_scale.unsqueeze(-1)).flatten(-2)
    return e2m1[..., 0::2] | (e2m1[..., 1::2] << 4), sf


_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5,
                -2.0, -3.0, -4.0, -6.0)


def nvfp4_plus_dequantize_ref(fp4: torch.Tensor,
                              sf: torch.Tensor) -> torch.Tensor:
    """Dequantize packed E2M1 [..., K // 2] with row-major UE5M3 scales [..., K // 32]."""
    lut = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=fp4.device)
    codes = torch.stack([fp4 & 0xF, fp4 >> 4], dim=-1).flatten(-2)
    values = lut[codes.long()].unflatten(-1, (-1, NVFP4_PLUS_SF_VEC_SIZE))
    return (values * ue5m3_to_float(sf).unsqueeze(-1)).flatten(-2)
