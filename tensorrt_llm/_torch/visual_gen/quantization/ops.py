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
    weight_fp32 = weight.float()

    # Calculate number of blocks
    num_blocks_out = (out_features + block_size - 1) // block_size
    num_blocks_in = (in_features + block_size - 1) // block_size

    # Initialize outputs
    qweight = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    block_scales = torch.empty(
        (num_blocks_out, num_blocks_in), dtype=torch.float32, device=weight.device
    )

    # Quantize each block
    for i in range(num_blocks_out):
        row_start = i * block_size
        row_end = min((i + 1) * block_size, out_features)

        for j in range(num_blocks_in):
            col_start = j * block_size
            col_end = min((j + 1) * block_size, in_features)

            # Extract block
            block = weight_fp32[row_start:row_end, col_start:col_end]

            # Compute block scale
            max_val = block.abs().max()
            scale = (
                max_val / FP8_E4M3_MAX if max_val > 0 else torch.tensor(1.0, device=weight.device)
            )

            # Quantize block
            inv_scale = scale.reciprocal() if scale > 0 else torch.tensor(1.0, device=weight.device)
            qblock = (block * inv_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)

            # Store results
            qweight[row_start:row_end, col_start:col_end] = qblock
            block_scales[i, j] = scale.to(torch.float32)

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
            - weight_scale: Block-wise scales (FP8 E4M3), 1D tensor (already in correct layout for nvfp4_gemm)
            - weight_scale_2: Global weight scale (FP32), scalar tensor
                Stored as amax_weight / (448*6) to match ModelOpt convention.
    """
    out_features, in_features = weight.shape
    amax_weight = weight.float().abs().max()

    # Global scale for fp4_quantize kernel (multiplier form: (448*6) / amax)
    global_sf = FP8_E4M3_MAX * E2M1_MAX / amax_weight

    # Quantize using TRT-LLM kernel
    qweight, weight_scale = torch.ops.trtllm.fp4_quantize(
        weight.to(torch.bfloat16), global_sf, block_size, False
    )

    # View weight_scale as FP8 E4M3 dtype to match checkpoint format
    # The kernel returns uint8 but the values are FP8 E4M3 encoded
    # Keep as 1D - this matches the reference implementation and nvfp4_gemm expectations
    weight_scale = weight_scale.view(torch.float8_e4m3fn)

    # weight_scale_2 in divisor form (ModelOpt convention): amax / (448*6)
    # This matches what load_weight_scales() expects from calibrated checkpoints
    weight_scale_2 = amax_weight / (FP8_E4M3_MAX * E2M1_MAX)

    return qweight, weight_scale, weight_scale_2.to(torch.float32)
