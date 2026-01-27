"""Definition of the quant module that can be used for PTQ."""

import warnings
from typing import Optional

import torch
from flashinfer import bmm_fp8
from torch import nn

from .torch_libs.float8_python_api import addmm_float8_unwrapped

TRTLLM_FP4_OP_AVAILABLE = True

TRTLLM_NVFP4_SCALING_VECTOR_SIZE = 16
TRTLLM_NVFP4_ROW_SIZE = 128
TRTLLM_NVFP4_COLUMN_SIZE = 4
TRTLLM_NVFP4_PACKING_FACTOR = 2


@torch.library.custom_op("auto_deploy::torch_quant_fn", mutates_args=())
def quant_fn(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scaled_x = x / scale
    rounded_x = torch.round(scaled_x)
    rounded_x = scaled_x
    clamped_x = torch.clamp(rounded_x, -127, 128)
    y = clamped_x * scale
    return y


@quant_fn.register_fake
def quant_fn_fake(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


class QuantModule(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale))

    def forward(self, x: torch.Tensor):
        return torch.ops.auto_deploy.torch_quant_fn(x, self.scale)


FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
FP4_MAX = 6.0
FP4_GLOBAL_SCALE_MAX = FP8_MAX * FP4_MAX


def _to_fp8(x, scale):
    return (x / scale).clamp(FP8_MIN, FP8_MAX).to(torch.float8_e4m3fn)


@torch.library.custom_op("auto_deploy::trtllm_quant_fp8_linear", mutates_args=())
def trtllm_quant_fp8_linear(
    input: torch.Tensor,
    weight_fp8: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP8 linear op similar to torch.nn.linear using TensorRT-LLM FP8 operations.

    Args:
        input: unquantized input tensor
        weight_fp8: pre-quantized weight tensor, with dtype torch.float8_e4m3fn
        input_scale: (Optional) pre-computed scalar tensor for static quantization.
        weight_scale: scalar tensor for weight dequantization.

    Returns:
        The linear output with the original dtype as the input.
    """
    input_shape = input.shape
    input_dtype = input.dtype

    n = weight_fp8.shape[0]  # out_features
    k = weight_fp8.shape[1]  # in_features

    # Verify dimensions match
    assert input_shape[-1] == k, f"Input last dim {input_shape[-1]} must match weight last dim {k}"

    input = input.reshape(-1, k)

    # Calculate padding needed to reach next multiple of 16
    k_pad = (16 - k % 16) % 16  # Amount to pad K dimension
    n_pad = (16 - n % 16) % 16  # Amount to pad N dimension

    if k_pad != 0:
        # Pad input on the last dimension (K dimension)
        input = torch.nn.functional.pad(input, (0, k_pad), mode="constant", value=0).contiguous()
        # Pad weight on the last dimension (K dimension)
        weight_fp8 = torch.nn.functional.pad(
            weight_fp8, (0, k_pad), mode="constant", value=0
        ).contiguous()

    if n_pad != 0:
        # Pad weight on the first dimension (N dimension)
        weight_fp8 = torch.nn.functional.pad(
            weight_fp8, (0, 0, 0, n_pad), mode="constant", value=0
        ).contiguous()

    # Use TensorRT-LLM FP8 per-tensor quantization
    assert input_scale is not None
    input_fp8, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(input, input_scale)

    enable_cuda_core = False
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        enable_cuda_core = capability == (8, 9) or capability == (12, 0)
    # Use TensorRT-LLM FP8 scaled matrix multiply
    # Choose between CUDA core (for small M) and cuBLAS (for large M) implementations
    if (
        input_fp8.shape[0] <= 8 and enable_cuda_core
    ):  # NOTE: this kernel work with n % 2 == 0 as well??
        # Use CUDA core for small M dimension (better for small batch sizes)
        output = torch.ops.trtllm.cuda_scaled_mm(
            input_fp8,
            weight_fp8.t(),
            scale_a=input_scale,
            scale_b=weight_scale,
            bias=None,
            out_dtype=input_dtype,
        )
    else:
        # Use cuBLAS for large M dimension
        output = torch.ops.trtllm.cublas_scaled_mm(
            input_fp8,
            weight_fp8.t(),
            scale_a=input_scale,
            scale_b=weight_scale,
            bias=None,
            out_dtype=input_dtype,
        )

    # Remove padding from output if needed
    if n_pad != 0:
        output = output[..., :n]

    if bias is not None:
        output = output + bias
    return output.reshape(*input_shape[:-1], n)


@trtllm_quant_fp8_linear.register_fake
def trtllm_quant_fp8_linear_fake(
    input: torch.Tensor,
    weight_fp8: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.ops.aten.linear(input, weight_fp8.to(input.dtype), bias)


@torch.library.custom_op("auto_deploy::torch_quant_fp8_linear", mutates_args=())
@torch.compile(dynamic=True)
def fp8_linear(
    input: torch.Tensor,
    weight_fp8: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP8 linear op similar to torch.nn.linear.

    Args:
        input: unquantized input tensor
        weight_fp8: pre-quantized weight tensor, with dtype torch.float8_e4m3fn
        input_scale: a scalar tensor defined as amax / max value (448.0).
        weight_scale: a scalar tensor defined as amax / max value (448.0).

    Returns:
        The linear output with the original dtype as the input.
    """
    input_shape = input.shape
    weight_shape = weight_fp8.shape

    # Original dimensions
    n = weight_shape[0]  # out_features
    k = weight_shape[1]  # in_features

    # Verify dimensions match
    assert input_shape[-1] == k, f"Input last dim {input_shape[-1]} must match weight last dim {k}"

    # Calculate padding needed to reach next multiple of 16
    k_pad = (16 - k % 16) % 16  # Amount to pad K dimension
    n_pad = (16 - n % 16) % 16  # Amount to pad N dimension

    if k_pad != 0:
        # Pad input on the last dimension (K dimension)
        input = torch.nn.functional.pad(input, (0, k_pad), mode="constant", value=0).contiguous()
        # Pad weight on the last dimension (K dimension)
        weight_fp8 = torch.nn.functional.pad(
            weight_fp8, (0, k_pad), mode="constant", value=0
        ).contiguous()

    if n_pad != 0:
        # Pad weight on the first dimension (N dimension)
        weight_fp8 = torch.nn.functional.pad(
            weight_fp8, (0, 0, 0, n_pad), mode="constant", value=0
        ).contiguous()

    # Cuda graph compatibility
    assert input_scale is not None
    input_fp8 = _to_fp8(input, input_scale)

    weight_fp8_t = weight_fp8.reshape(-1, weight_fp8.shape[-1]).t()

    # If we have N padding, don't add bias in addmm (it won't match dimensions)
    # We'll add it after removing padding
    output = addmm_float8_unwrapped(
        input_fp8.reshape(-1, input.shape[-1]),
        input_scale,
        weight_fp8_t,
        weight_scale,
        input.dtype,
        bias=None if n_pad != 0 else bias,
        use_fast_accum=True,
    )

    # Remove padding from output if needed
    if n_pad != 0:
        output = output[..., :n]
        # Add bias after removing padding
        if bias is not None:
            output = output + bias

    return output.reshape(*input_shape[:-1], output.shape[-1])


@fp8_linear.register_fake
def fp8_linear_fake(
    input: torch.Tensor,
    weight_fp8: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.ops.aten.linear(input, weight_fp8.to(input.dtype), bias)


class FP8Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = self.weight.device
        weight_scale = torch.max(torch.abs(self.weight)).to(torch.float).to(device) / FP8_MAX
        self.weight = nn.Parameter((self.weight / weight_scale).to(torch.float8_e4m3fn))
        self.register_buffer(
            "input_scale", torch.tensor(1.0, device=self.weight.device, dtype=torch.float)
        )
        self.register_buffer("weight_scale", weight_scale)
        if self.bias is not None:
            self.bias = nn.Parameter(self.bias.to(torch.half))

    def forward(self, x):
        return torch.ops.auto_deploy.torch_quant_fp8_linear(
            x, self.weight, self.bias, self.input_scale, self.weight_scale
        )


@torch.library.custom_op("auto_deploy::torch_quant_nvfp4_linear", mutates_args=())
@torch.compile(dynamic=True)
def nvfp4_linear(
    input: torch.Tensor,
    weight_fp4: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP4 linear op similar to torch.nn.linear.

    Args:
        input: unquantized input tensor
        weight_fp4: pre-quantized weight tensor, with dtype torch.uint8 (1 uint8 == 2 elements)
        input_scale: a scalar tensor defined as per_tensor_amax / (FP8 max value (448.0) * FP4 max value (6.0)).
        weight_scale: a 1D tensor with shape (out_dim * in_dim / 16) padded to be multiple of (128 * 4).
            with value: per_block_amax / per_tensor_amax * FP8 max value (448.0)
        weight_scale_2: a scalar tensor defined as per_tensor_amax / (FP8 max value (448.0) * FP4 max value (6.0)).

    Returns:
        The linear output with the original dtype as the input.
    """
    assert TRTLLM_FP4_OP_AVAILABLE, "TRT-LLM FP4 operators are not available."

    input_shape = input.shape
    weight_shape = weight_fp4.shape

    n = weight_shape[0]
    k = input_shape[-1]
    assert k % 16 == 0
    assert weight_shape[-1] % 8 == 0
    assert weight_scale.numel() % (128 * 4) == 0

    input = input.reshape(-1, k)

    # FP4 compatibility
    assert input_scale is not None
    assert weight_scale is not None
    assert alpha is not None

    x_fp4, x_sf_block = torch.ops.trtllm.fp4_quantize(
        input, input_scale, TRTLLM_NVFP4_SCALING_VECTOR_SIZE, False
    )
    output = torch.ops.trtllm.nvfp4_gemm(
        x_fp4, weight_fp4, x_sf_block, weight_scale, alpha, input.dtype
    )

    if bias is not None:
        output = output + bias

    return output.reshape(*input_shape[:-1], n)


@nvfp4_linear.register_fake
def fp4_linear_fake(
    input: torch.Tensor,
    weight_fp4: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.ops.aten.linear(input, weight_fp4.repeat(1, 2).to(input.dtype), bias)


def is_column_major(tensor):
    rows, _ = tensor.shape[-2:]
    strides = tensor.stride()
    return strides[-2] == 1 and strides[-1] == rows


@torch.library.custom_op("auto_deploy::torch_quant_fp8_bmm", mutates_args=())
def fp8_bmm(
    input: torch.Tensor,
    mat2: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """FP8 BMM op similar to torch.bmm.

    Args:
        input: unquantized input tensor with shape (B, M, K)
        mat2: weight tensor with shape (B, K, N), with dtype torch.float8_e4m3fn,
            or torch.float16, or torch.bfloat16
        input_scale: a scalar tensor - the inverse scale for input quantization
        weight_scale: a scalar tensor - the inverse scale for weight quantization

    Returns:
        The BMM output with shape (B, M, N) and the original dtype as the input.
    """
    # Ensure input is contiguous
    input = input.contiguous()
    original_input_dtype = input.dtype

    # Convert input to fp8 using provided scale
    if input.dtype in [torch.float16, torch.bfloat16]:
        input_fp8 = _to_fp8(input, input_scale)
    else:
        assert input.dtype == torch.float8_e4m3fn
        input_fp8 = input

    # Convert mat2 to fp8 using provided scale
    if mat2.dtype in [torch.float16, torch.bfloat16]:
        mat2_fp8 = _to_fp8(mat2, weight_scale)
    else:
        assert mat2.dtype == torch.float8_e4m3fn
        mat2_fp8 = mat2

    # Ensure mat2 is contiguous in column-major format only if needed
    # Check if the tensor is already contiguous when transposed (i.e., already column-major)
    if not is_column_major(mat2_fp8):
        warnings.warn(
            "mat2 is not in column-major format, transposing it, this will cause performance degradation."
        )
        mat2_fp8 = mat2_fp8.transpose(-2, -1).contiguous().transpose(-2, -1)

    # Get dimensions
    B, M, K = input.shape
    B2, K2, N = mat2_fp8.shape
    assert B == B2, f"Batch dimensions must match: {B} vs {B2}"
    assert K == K2, f"Inner dimensions must match: {K} vs {K2}"

    output = torch.empty((B, M, N), dtype=original_input_dtype, device=input.device)
    bmm_fp8(
        input_fp8, mat2_fp8, input_scale.float(), weight_scale.float(), original_input_dtype, output
    )

    return output


@fp8_bmm.register_fake
def fp8_bmm_fake(
    input: torch.Tensor,
    mat2: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Fake implementation of fp8_bmm for testing and tracing."""
    # Use standard bmm
    return torch.bmm(input.to(torch.float), mat2.to(torch.float)).to(input.dtype)
