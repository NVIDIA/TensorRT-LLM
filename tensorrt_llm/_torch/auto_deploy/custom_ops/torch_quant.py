from typing import List, Optional

import torch

# ===== Enums =====
FORMAT_FP8 = 0
FORMAT_NVFP4 = 1

# scale layouts
PER_TENSOR = 0


# ===== Helpers =====
def _expect_single_scale(scales: List[Optional[torch.Tensor]], name: str) -> torch.Tensor:
    if len(scales) == 0 or scales[0] is None:
        raise ValueError(f"{name} must provide at least one scale tensor (scales[0]).")
    return scales[0]


def _to_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (x / scale).to(torch.float8_e4m3fn)


def _from_fp8(x_fp8: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return x_fp8.to(dtype) * scale


def _dequant_weight_fp8(
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_type: int,
    out_features: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if weight_scale_type == PER_TENSOR:
        return weight_fp8.to(dtype) * weight_scale
    raise NotImplementedError(f"Unsupported weight_scale_type={weight_scale_type} for FP8.")


# ===== Single, format-aware op =====
@torch.library.custom_op("auto_deploy::custom_quant_linear", mutates_args=())
@torch.compile(dynamic=True)
def custom_quant_linear(
    input: torch.Tensor,
    weight_quantized: torch.Tensor,
    bias: torch.Tensor,  # Optional, no default
    input_scale: List[torch.Tensor],  # Tensor?[]  (REQUIRED: no default)
    weight_scale: List[torch.Tensor],  # Tensor?[]
    input_zp: List[torch.Tensor],  # Tensor?[]
    weight_zp: List[torch.Tensor],  # Tensor?[]
    format_type: int = FORMAT_FP8,  # which quant format this call uses
    input_scale_type: int = PER_TENSOR,
    weight_scale_type: int = PER_TENSOR,
    input_zp_type: int = 0,
    weight_zp_type: int = 0,
) -> torch.Tensor:
    """
    Reference (eager) implementation for multiple quant formats via `format_type`.
    For FP8:
      - input_scale[0] and weight_scale[0] are required (amax/448 style)
      - input_zp / weight_zp ignored
      - supports PER_TENSOR and PER_CHANNEL_OUT for weights
    """
    if format_type == FORMAT_FP8:
        if weight_quantized.dtype != torch.float8_e4m3fn:
            raise TypeError("FP8 path requires weight_quantized.dtype == float8_e4m3fn")
        s_in = _expect_single_scale(input_scale, "input_scale")
        s_w = _expect_single_scale(weight_scale, "weight_scale")

        in_dtype = input.dtype
        out_features, in_features = weight_quantized.shape

        input_fp8 = _to_fp8(input, s_in)
        input_deq = _from_fp8(input_fp8, s_in, in_dtype)

        weight_deq = _dequant_weight_fp8(
            weight_quantized, s_w, weight_scale_type, out_features, in_dtype
        )

        out = torch.matmul(input_deq.reshape(-1, in_features), weight_deq.t())
        if bias is not None:
            out = out + bias
        return out.reshape(*input.shape[:-1], out_features)

    elif format_type == FORMAT_NVFP4:
        raise NotImplementedError("NVFP4 path not implemented yet.")

    else:
        raise NotImplementedError(f"Unknown format_type={format_type}")


@custom_quant_linear.register_fake
def custom_quant_linear_fake(
    input: torch.Tensor,
    weight_quantized: torch.Tensor,
    bias: torch.Tensor,
    input_scale: List[torch.Tensor],
    weight_scale: List[torch.Tensor],
    input_zp: List[torch.Tensor],
    weight_zp: List[torch.Tensor],
    format_type: int = FORMAT_FP8,
    input_scale_type: int = PER_TENSOR,
    weight_scale_type: int = PER_TENSOR,
    input_zp_type: int = 0,
    weight_zp_type: int = 0,
) -> torch.Tensor:
    if format_type == FORMAT_FP8:
        w = weight_quantized.to(input.dtype)
        return torch.ops.aten.linear(input, w, bias)

    raise NotImplementedError(f"Fake mode not implemented for format_type={format_type}")
