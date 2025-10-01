from typing import List, Optional

import torch

from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import (
    cutlass_fp4_scale_to_modelopt_fp4_scale,
    unpack_uint8_to_int4_weight_2d,
)

# FP4 tables (E2M1)
e2m1_bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
e2m1_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])


# ===== Helpers =====
def _expect_single_scale(scales: List[Optional[torch.Tensor]], name: str) -> torch.Tensor:
    if len(scales) == 0 or scales[0] is None:
        raise ValueError(f"{name} must provide at least one scale tensor (scales[0]).")
    return scales[0]


def _to_fp8_fake(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x / scale).to(torch.float8_e4m3fn)


def _from_fp8(x_fp8: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return x_fp8.to(dtype) * scale


def _dequant_weight_fp8(
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    out_features: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return weight_fp8.to(dtype) * weight_scale


# The NVFP4 helpers below are adapted from modelopt.torch.quantization.qtensor.nvfp4_tensor.NVFP4QTensor
def _nvfp4_get_weights_scaling_factor(
    input: torch.Tensor,
    block_size: int,
    weights_scaling_factor_2: torch.Tensor | None = None,
    keep_high_precision: bool = False,
):
    """Returns quantized per block weight scaling factor."""
    if weights_scaling_factor_2 is None:
        # per-tensor scale-2 = amax / (6 * 448)
        weights_scaling_factor_2 = input.abs().amax().float() / (6.0 * 448.0)

    # Get per_block amax
    [n, k] = input.shape[-2:]
    assert block_size != 0, "Block size is zero. Cannot return per_block amax for given input."

    assert k % block_size == 0, (
        "Weight shape is not divisible for block size for block quantization."
    )

    input = input.reshape((*tuple(input.shape[:-2]), n, k // block_size, block_size))
    # Get per block amax
    per_block_amax = input.abs().amax(dim=-1).float()
    # Get per-block-scale
    per_block_scale = per_block_amax / 6.0
    # Quantize per_block_scale to FP8
    q_per_block_scale = per_block_scale / weights_scaling_factor_2
    # Set all zero values in scale to 1.0
    q_per_block_scale[per_block_scale == 0] = 1.0
    # Convert to torch.float8_e4m3fn
    if not keep_high_precision:
        q_per_block_scale = q_per_block_scale.to(torch.float8_e4m3fn)
    return q_per_block_scale, weights_scaling_factor_2


def _cast_fp4(weight: torch.Tensor):
    """Converts tensor to uint4."""
    # Get device
    device = weight.device

    # Define mask to perform rounding
    mask = torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.uint8).to(device)
    mask_shape = list(weight.shape)
    mask = mask.expand([*mask_shape, 7])

    sign_bit = (weight < 0).to(torch.uint8)

    weight_abs = weight.abs()  # avoid in-place modification to input
    # Calculate the ordinal value based on the bounds
    ord = torch.searchsorted(e2m1_bounds.to(device), weight_abs, out_int32=True).to(torch.uint8)
    # All values equal to e2m1_bounds at odd indices are rounded up and even indices are rounded down
    round = torch.any((weight_abs.unsqueeze(-1) == e2m1_bounds.to(device)) * mask, dim=-1)
    fp4_val = (sign_bit * 0b1000 + ord + round).to(torch.uint8)
    return fp4_val


def _quantize_nvfp4(
    input: torch.Tensor,
    block_size: int,
    weights_scaling_factor_2: torch.Tensor | None = None,
):
    """Converting a tensor to a quantized format based on NVFP4 quantization.

    Args:
        input (torch.Tensor): The input tensor to be quantized.
        block_size (int): The size of each block for quantization.
        weights_scaling_factor_2 (torch.Tensor): The per-tensor scaling factor for the weights.
    Returns:
    tuple: Contains quantized data and quantized per block scaling factor
    """

    weights_scaling_factor, weights_scaling_factor_2 = _nvfp4_get_weights_scaling_factor(
        input, block_size, weights_scaling_factor_2
    )

    # Reshape the weight and scale factors
    input = input.view((*tuple(input.shape[:-1]), -1, block_size))

    # Scale weights
    scaled_weight = input / (
        (weights_scaling_factor.to(torch.float32) * weights_scaling_factor_2).unsqueeze(-1)
    )

    # Reshape weights to original
    scaled_weight = scaled_weight.view((*tuple(scaled_weight.shape[:-2]), -1))

    # Cast weights to fp4
    q_weight = _cast_fp4(scaled_weight)
    # Pack weights
    packed_weight = (q_weight[..., 1::2] << 4) | q_weight[..., 0::2]
    return packed_weight, weights_scaling_factor


def _dequantize_nvfp4(
    quantized_t: torch.Tensor,  # [N, K/2] uint8
    scale_1: torch.Tensor,  # q_per_block_scale (FP8/FP32), flat or shaped
    scale_2: torch.Tensor,  # per-tensor scale-2 (FP32 scalar)
    orig_shape: tuple,  # (N, K)
    orig_dtype: torch.dtype,
) -> torch.Tensor:
    device = quantized_t.device
    N, K = orig_shape
    # slice/pad handling for the scale vector: take exactly N*K/16 entries
    num_blocks = N * (K // 16)
    s1 = scale_1.reshape(-1)[:num_blocks]

    high = (quantized_t >> 4) & 0x0F
    low = quantized_t & 0x0F
    idx = torch.empty(N, (K // 2) * 2, dtype=torch.long, device=device)
    idx[..., 0::2] = low.long()
    idx[..., 1::2] = high.long()

    vals = e2m1_values.to(device)[idx]  # [N, K], float32

    scale_real = (s1.to(torch.float32) * scale_2.to(torch.float32)).view(N, K // 16, 1)
    vals = vals.view(N, K // 16, 16) * scale_real
    return vals.view(N, K).to(orig_dtype)


@torch.library.custom_op("auto_deploy::torch_fake_quant_fp8_linear", mutates_args=())
def torch_fake_quant_fp8_linear(
    input: torch.Tensor,
    weight_quantized: torch.Tensor,
    bias: torch.Tensor,
    input_scale: List[torch.Tensor],
    weight_scale: List[torch.Tensor],
    input_zp: List[torch.Tensor],
    weight_zp: List[torch.Tensor],
) -> torch.Tensor:
    """
    Reference (eager) implementation for multiple quant formats via `format_type`.
    For FP8:
      - input_scale[0] and weight_scale[0] are required (amax/448 style)
      - input_zp / weight_zp ignored
    """
    if weight_quantized.dtype != torch.float8_e4m3fn:
        raise TypeError("FP8 path requires weight_quantized.dtype == float8_e4m3fn")
    s_in = _expect_single_scale(input_scale, "input_scale")
    s_w = _expect_single_scale(weight_scale, "weight_scale")

    in_dtype = input.dtype
    out_features, in_features = weight_quantized.shape

    input_fp8 = _to_fp8_fake(input, s_in)
    input_deq = _from_fp8(input_fp8, s_in, in_dtype)

    weight_deq = _dequant_weight_fp8(weight_quantized, s_w, out_features, in_dtype)

    out = torch.matmul(input_deq.reshape(-1, in_features), weight_deq.t())
    if bias is not None:
        out = out + bias
    return out.reshape(*input.shape[:-1], out_features)


@torch_fake_quant_fp8_linear.register_fake
def torch_fake_quant_fp8_linear(
    input: torch.Tensor,
    weight_quantized: torch.Tensor,
    bias: torch.Tensor,
    input_scale: List[torch.Tensor],
    weight_scale: List[torch.Tensor],
    input_zp: List[torch.Tensor],
    weight_zp: List[torch.Tensor],
) -> torch.Tensor:
    w = weight_quantized.to(input.dtype)
    return torch.ops.aten.linear(input, w, bias)


@torch.library.custom_op("auto_deploy::torch_fake_quant_nvfp4_linear", mutates_args=())
def torch_fake_quant_nvfp4_linear(
    input: torch.Tensor,
    weight_quantized: torch.Tensor,
    bias: torch.Tensor,
    input_scale: List[torch.Tensor],
    weight_scale: List[torch.Tensor],
    input_zp: List[torch.Tensor],
    weight_zp: List[torch.Tensor],
) -> torch.Tensor:
    """
    Reference (eager) implementation for multiple quant formats via `format_type`.
    For FP4:
      - input_scale[0]  = s_in2   (scalar, amax/(448*6))
      - weight_scale[0] = q_per_block_scale_w  (len >= N*K/16; may be padded)
      - weight_scale[1] = alpha = s_in2 * s_w2 (combined per-tensor scales)
    """
    if weight_quantized.dtype != torch.uint8:
        raise TypeError("NVFP4 path requires packed uint8 weights (2x FP4 per byte).")

    inv_x = _expect_single_scale(input_scale, "input_scale")
    if len(weight_scale) < 2 or weight_scale[0] is None or weight_scale[1] is None:
        raise ValueError(
            "NVFP4 needs weight_scale[0] (per-block vector) and weight_scale[1] (alpha)."
        )
    cutlass_qscale = weight_scale[0]
    alpha = weight_scale[1]

    if cutlass_qscale.dtype != torch.uint8:
        raise TypeError("NVFP4 expects CUTLASS per-block scale vector in uint8 (same as fused op).")

    inv_w = 1 / (inv_x * alpha)
    s2_x = 1.0 / inv_x
    s2_w = 1.0 / inv_w

    # Shapes
    in_dtype = input.dtype
    input_shape = input.shape
    N, K_packed = weight_quantized.shape[-2], weight_quantized.shape[-1]
    K = K_packed * 2
    assert K % 16 == 0, "NVFP4 requires K to be a multiple of 16"
    num_blocks_w = N * (K // 16)

    q_scale_w_slice = cutlass_fp4_scale_to_modelopt_fp4_scale(cutlass_qscale, (N, K))
    # (1) Dequantize weights with scale_1 = q_scale_w (sliced), scale_2 = s_w2
    q_scale_w_slice = q_scale_w_slice.reshape(-1)[:num_blocks_w]
    W_deq = _dequantize_nvfp4(weight_quantized, q_scale_w_slice, s2_w, (N, K), in_dtype)  # [N, K]

    # (2) Quantize+dequantize inputs with _quantize_nvfp4/_dequantize_nvfp4
    # Flatten batch for NVFP4 block processing
    X_2d = input.reshape(-1, K)

    X_packed, X_q_scale = _quantize_nvfp4(X_2d, block_size=16, weights_scaling_factor_2=s2_x)
    X_deq = _dequantize_nvfp4(X_packed, X_q_scale, s2_x, (X_2d.shape[0], K), in_dtype)  # [B, K]

    # (3) GEMM + bias (float GEMM with codec error baked in)
    out_2d = torch.matmul(X_deq, W_deq.t())  # [B, N]
    if bias is not None:
        out_2d = out_2d + bias
    return out_2d.reshape(*input_shape[:-1], N)


@torch_fake_quant_nvfp4_linear.register_fake
def torch_fake_quant_nvfp4_linear(
    input: torch.Tensor,
    weight_quantized: torch.Tensor,
    bias: torch.Tensor,
    input_scale: List[torch.Tensor],
    weight_scale: List[torch.Tensor],
    input_zp: List[torch.Tensor],
    weight_zp: List[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.aten.linear(input, weight_quantized.repeat(1, 2).to(input.dtype), bias)


@torch.library.custom_op("auto_deploy::torch_fake_quant_int4_linear", mutates_args=())
def torch_fake_quant_int4_linear(
    input: torch.Tensor,  # [..., K]
    weight_quantized: torch.Tensor,  # [N//2, K] unit8 (packed)
    bias: Optional[torch.Tensor],  # [N] or None
    input_scale: List[torch.Tensor],  # [ pre_quant_scale ]
    weight_scale: List[torch.Tensor],  # [ weight_scale ]
    input_zp: List[torch.Tensor],
    weight_zp: List[torch.Tensor],
) -> torch.Tensor:
    BLOCK_SIZE = 128
    # activation pre-scale
    pre_quant_scale = input_scale[0].to(dtype=input.dtype)
    x_scaled = torch.mul(input, pre_quant_scale)

    q_int4 = unpack_uint8_to_int4_weight_2d(weight_quantized, weight_scale[0])  # (N,K), int8
    amax_2d = (weight_scale[0] * 7.0).to(input.dtype)  # (N, K//128)

    scale_blocks = (7.0 / amax_2d).to(torch.float32)  # (N, K//128)
    scale_full = scale_blocks.repeat_interleave(BLOCK_SIZE, dim=1)  # (N,K)

    # Dequantize
    w_deq = (q_int4.to(torch.float32) / scale_full).to(input.dtype)

    return torch.ops.auto_deploy.torch_linear_simple.default(x_scaled, w_deq, bias)


@torch_fake_quant_int4_linear.register_fake
def _fake(
    input: torch.Tensor,
    weight_quantized: torch.Tensor,
    bias: Optional[torch.Tensor],
    input_scale: List[torch.Tensor],
    weight_scale: List[torch.Tensor],
    input_zp: List[torch.Tensor],
    weight_zp: List[torch.Tensor],
) -> torch.Tensor:
    N_half = weight_quantized.shape[-2]
    N = N_half * 2
    return torch.empty((*input.shape[:-1], N), dtype=input.dtype, device=input.device)
