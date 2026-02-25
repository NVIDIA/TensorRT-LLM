# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Quantization operations for diffusion models.

Provides on-the-fly quantization functions for dynamic (load-time) quantization.
"""

from typing import Tuple

import torch

# FP8 E4M3 max value
FP8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max
# FP4 E2M1 max value
E2M1_MAX = 6.0


def quantize_fp8_per_tensor(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weight to FP8 E4M3 with per-tensor scale.

    Uses torch.ops.tensorrt_llm.quantize_e4m3_per_tensor CUDA kernel.

    Args:
        weight: Input weight tensor (BF16/FP16/FP32), shape (out_features, in_features)

    Returns:
        Tuple of:
            - qweight: Quantized weight (FP8 E4M3), same shape as input
            - weight_scale: Dequantization scale (FP32), shape (1, 1)
    """
    qweight, scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(weight)
    # Ensure scale is float32 and has shape (1, 1) for consistency
    return qweight, scale.to(torch.float32)


def quantize_fp8_blockwise(
    weight: torch.Tensor, block_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weight to FP8 E4M3 with 128x128 blockwise scales.

    This function converts BF16/FP16/FP32 weights to FP8 E4M3 format using
    per-block scale factors. The weight is divided into blocks of size
    (block_size, block_size) and each block has its own scale.

    Args:
        weight: Input weight tensor (BF16/FP16/FP32), shape (out_features, in_features)
        block_size: Block size for blockwise quantization (default: 128)

    Returns:
        Tuple of:
            - qweight: Quantized weight (FP8 E4M3), shape (out_features, in_features)
            - block_scales: Block-wise dequantization scales (FP32),
                shape (num_blocks_out, num_blocks_in)

    Note:
        - If dimensions are not divisible by block_size, the last block may be smaller
        - block_scales are dequantization scales (multiply to get back original scale)
        - This uses 128x128 block scaling compatible with Linear module's FP8_BLOCK_SCALES
    """
    out_features, in_features = weight.shape
    num_blocks_out = (out_features + block_size - 1) // block_size
    num_blocks_in = (in_features + block_size - 1) // block_size

    # Pad to multiple of block_size
    pad_out = num_blocks_out * block_size - out_features
    pad_in = num_blocks_in * block_size - in_features
    if pad_out > 0 or pad_in > 0:
        weight_padded = torch.nn.functional.pad(weight, (0, pad_in, 0, pad_out))
    else:
        weight_padded = weight

    # Reshape so each block becomes a row:
    # (out, in) -> (nb_out, bs, nb_in, bs) -> (nb_out, nb_in, bs, bs) -> (nb_out*nb_in, bs*bs)
    rows_per_block = (
        weight_padded.reshape(num_blocks_out, block_size, num_blocks_in, block_size)
        .permute(0, 2, 1, 3)
        .reshape(num_blocks_out * num_blocks_in, block_size * block_size)
    )

    # Single CUDA kernel: per-row FP8 quantization
    # quantize_e4m3_activation uses PER_TOKEN mode: one scale per row
    qrows, scales = torch.ops.tensorrt_llm.quantize_e4m3_activation(rows_per_block.contiguous())

    # Reshape back: (nb_out*nb_in, bs*bs) -> (nb_out, nb_in, bs, bs) -> (out_padded, in_padded)
    qweight_padded = (
        qrows.reshape(num_blocks_out, num_blocks_in, block_size, block_size)
        .permute(0, 2, 1, 3)
        .reshape(num_blocks_out * block_size, num_blocks_in * block_size)
    )

    # Remove padding and extract scales
    qweight = qweight_padded[:out_features, :in_features].contiguous()
    block_scales = scales.reshape(num_blocks_out, num_blocks_in).to(torch.float32)

    return qweight, block_scales


def quantize_nvfp4(
    weight: torch.Tensor, block_size: int = 16
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize weight to NVFP4 (FP4 E2M1) with blockwise scales.

    Uses torch.ops.trtllm.fp4_quantize CUDA kernel. This function performs
    dynamic weight quantization for NVFP4 format, producing:
    - Packed FP4 weights (2 values per byte)
    - Per-block FP8 scale factors
    - Global weight scale for alpha computation

    The scaling convention matches ModelOpt checkpoints:
    - weight_scale_2 = amax_weight / (448 * 6) (divisor form)

    Args:
        weight: Input weight tensor (BF16/FP16/FP32), shape (out_features, in_features)
        block_size: Block size for blockwise quantization (default: 16)

    Returns:
        Tuple of:
            - qweight: Packed FP4 weight, shape (out_features, in_features // 2)
            - weight_scale: Block-wise scales (FP8 E4M3), 2D LINEAR layout
                shape (out_features, in_features // block_size). Same format as
                ModelOpt checkpoints; needs block_scale_interleave before use
                in nvfp4_gemm.
            - weight_scale_2: Global weight scale (FP32), scalar tensor
                Stored as amax_weight / (448*6) to match ModelOpt convention.
    """
    out_features, in_features = weight.shape
    amax_weight = weight.float().abs().max()

    # Global scale for fp4_quantize kernel (multiplier form: (448*6) / amax)
    global_sf = FP8_E4M3_MAX * E2M1_MAX / amax_weight

    # Quantize using TRT-LLM kernel (isSfSwizzledLayout=False for LINEAR output)
    # LINEAR layout matches ModelOpt checkpoint format so that the downstream
    # weight loader in linear.py applies block_scale_interleave uniformly.
    qweight, weight_scale = torch.ops.trtllm.fp4_quantize(
        weight.to(torch.bfloat16), global_sf, block_size, False, False
    )

    # Reshape 1D LINEAR scale to 2D [out_features, in_features // block_size]
    # to match ModelOpt checkpoint format expected by block_scale_interleave
    scale_cols = in_features // block_size
    weight_scale = weight_scale.view(torch.float8_e4m3fn).reshape(out_features, scale_cols)

    # weight_scale_2 in divisor form (ModelOpt convention): amax / (448*6)
    # This matches what load_weight_scales() expects from calibrated checkpoints
    weight_scale_2 = amax_weight / (FP8_E4M3_MAX * E2M1_MAX)

    return qweight, weight_scale, weight_scale_2.to(torch.float32)
