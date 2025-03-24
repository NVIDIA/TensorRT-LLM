"""Definition of the quant module that can be used for PTQ."""

from typing import Optional

import numpy as np
import torch
from torch import nn

from ..distributed import common as dist
from ..distributed import trtllm as trtllm_dist
from .torch_libs.float8_python_api import addmm_float8_unwrapped

try:
    from tensorrt_llm.quantization.utils.fp4_utils import FP4_BUCKETS as fp4_buckets

    TRTLLM_FP4_OP_AVAILABLE = True
except ImportError:
    TRTLLM_FP4_OP_AVAILABLE = False

TRTLLM_NVFP4_SCALING_VECTOR_SIZE = 16


@torch.library.custom_op("quant::quant_fn", mutates_args=())
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
        return torch.ops.quant.quant_fn(x, self.scale)


FP8_MIN = torch.finfo(torch.float8_e4m3fn).min
FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
FP4_MAX = 6.0
FP4_GLOBAL_SCALE_MAX = FP8_MAX * FP4_MAX


def _to_fp8(x, scale):
    return (x / scale).clamp(FP8_MIN, FP8_MAX).to(torch.float8_e4m3fn)


@torch.library.custom_op("quant::fp8_linear", mutates_args=())
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
    assert input.shape[-1] % 16 == 0
    assert weight_fp8.shape[-1] % 16 == 0

    input_shape = input.shape
    weight_shape = weight_fp8.shape

    # Cuda graph compatibility
    assert input_scale is not None
    input_fp8 = _to_fp8(input, input_scale)

    weight_fp8_t = weight_fp8.reshape(-1, weight_shape[-1]).t()
    output = addmm_float8_unwrapped(
        input_fp8.reshape(-1, input_shape[-1]),
        input_scale,
        weight_fp8_t,
        weight_scale,
        input.dtype,
        bias=bias,
        use_fast_accum=True,
    )

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


@torch.library.custom_op("quant::fused_fp8_linear_all_reduce", mutates_args=())
@torch.compile(dynamic=True)
def fused_fp8_linear_all_reduce(
    input: torch.Tensor,
    weight_fp8: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    out = torch.ops.quant.fp8_linear(input, weight_fp8, bias, input_scale, weight_scale)
    if trtllm_dist.is_trtllm_op_available():
        return trtllm_dist.trtllm_allreduce(out, op=dist.ReduceOp.SUM)
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out


@fused_fp8_linear_all_reduce.register_fake
def fused_fp8_linear_all_reduce_fake(
    input: torch.Tensor,
    weight_fp8: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.ops.quant.fp8_linear(input, weight_fp8, bias, input_scale, weight_scale)


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
        return torch.ops.quant.fp8_linear(
            x, self.weight, self.bias, self.input_scale, self.weight_scale
        )


@torch.library.custom_op("quant::fp4_linear", mutates_args=())
@torch.compile(dynamic=True)
def fp4_linear(
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
    if not hasattr(fp4_linear, "profiler"):
        fp4_linear.profiler = torch.classes.trtllm.FP4GemmRunner.get_instance(input.dtype)
        fp4_linear.profile_record = set()

    profiler = fp4_linear.profiler
    profile_record = fp4_linear.profile_record

    input_shape = input.shape
    weight_shape = weight_fp4.shape

    n = weight_shape[0]
    k = input_shape[-1]
    assert k % 16 == 0
    assert weight_shape[-1] % 8 == 0
    assert weight_scale.numel() % (128 * 4) == 0

    input = input.reshape(-1, k)

    # profile if needed and get the best config
    if (n, k) not in profile_record:
        profiler.run_profile(n, k, fp4_buckets)
        profile_record.add((n, k))
    best_config_id = profiler.get_best_config_id(np.prod(input_shape[:-1]), n, k)

    # FP4 compatibility
    assert input_scale is not None
    assert weight_scale is not None
    assert alpha is not None

    x_fp4, x_sf_block = torch.ops.trtllm.fp4_quantize(
        input, input_scale, TRTLLM_NVFP4_SCALING_VECTOR_SIZE, False
    )
    output = profiler.run_gemm(
        x_fp4, weight_fp4, x_sf_block, weight_scale, alpha, False, best_config_id
    )

    if bias is not None:
        output = output + bias

    return output.reshape(*input_shape[:-1], n)


@fp4_linear.register_fake
def fp4_linear_fake(
    input: torch.Tensor,
    weight_fp4: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    input_scale: Optional[torch.Tensor] = None,
    weight_scale: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.ops.aten.linear(input, weight_fp4.repeat(1, 2).to(input.dtype), bias)


QUANT_OPS = [torch.ops.quant.fp8_linear, torch.ops.quant.fp4_linear]
