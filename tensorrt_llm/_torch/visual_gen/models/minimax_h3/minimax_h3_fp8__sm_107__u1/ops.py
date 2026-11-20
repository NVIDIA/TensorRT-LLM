# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opaque compiler boundaries for the static-FP8 H3 block's fused operations."""

import torch

from .kernels import attention_output, down_residual, gate_up_swiglu
from .kernels.adaln import launch as adaln_launch
from .kernels.conversion import BLOCKS, THREADS, modules_for_device

FULL_SEQUENCE_LENGTHS = (37776, 37820, 38169)


@torch.library.custom_op("h3_fp8::qkv_gemm", mutates_args=())
def qkv_gemm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Keep the selected framework GEMM behind an opaque compiler boundary."""
    return torch._scaled_mm(
        activation,
        weight.t(),
        scale_a=input_scale,
        scale_b=weight_scale,
        out_dtype=torch.bfloat16,
        use_fast_accum=True,
    )


@qkv_gemm.register_fake
def _qkv_gemm_fake(activation, weight, input_scale, weight_scale):
    return torch.empty(
        (activation.shape[0], weight.shape[0]), device=activation.device, dtype=torch.bfloat16
    )


def supports_projection(
    activation: torch.Tensor,
    weight: torch.Tensor,
    hidden: torch.Tensor,
    modulation: torch.Tensor,
    indices: torch.Tensor,
    in_features: int,
) -> bool:
    """Keep the full-sequence kernel's dtype, shape and alignment contract."""
    rows = activation.shape[0]
    return (
        rows in FULL_SEQUENCE_LENGTHS
        and activation.shape == (rows, in_features)
        and weight.shape == (5376, in_features)
        and activation.dtype == weight.dtype == torch.float8_e4m3fn
        and hidden.dtype == torch.bfloat16
        and hidden.numel() == rows * 5376
        and activation.is_contiguous()
        and weight.is_contiguous()
        and hidden.is_contiguous()
        and modulation.ndim == 2
        and modulation.shape[1] == 6 * 5376
        and modulation.dtype in (torch.bfloat16, torch.float32)
        and modulation.is_contiguous()
        and modulation.data_ptr() % 32 == 0
        and modulation.stride(0) * modulation.element_size() % 32 == 0
        and indices.numel() == rows
        and indices.is_contiguous()
        and indices.dtype in (torch.int32, torch.int64)
    )


def _scaled_mm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.trtllm.cublas_scaled_mm(
        activation,
        weight.t(),
        scale_a=input_scale,
        scale_b=weight_scale,
        bias=None,
        out_dtype=torch.bfloat16,
    )


@torch.library.custom_op("h3_fp8::gate_up_swiglu", mutates_args=())
def gate_up(
    activation: torch.Tensor,
    packed_weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    output_scale: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty(
        (activation.shape[0], 14336), device=activation.device, dtype=torch.float8_e4m3fn
    )
    gate_up_swiglu._get_compiled(activation.device.index)(
        activation,
        packed_weight,
        input_scale.reshape(1),
        weight_scale.reshape(1),
        output_scale.reshape(1),
        output,
    )
    return output


@gate_up.register_fake
def _gate_up_fake(activation, packed_weight, input_scale, weight_scale, output_scale):
    return torch.empty(
        (activation.shape[0], 14336), device=activation.device, dtype=torch.float8_e4m3fn
    )


@torch.library.custom_op("h3_fp8::down_gated_residual", mutates_args=())
def down(
    activation: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    hidden: torch.Tensor,
    modulation: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    if supports_projection(activation, weight, hidden, modulation, indices, 14336):
        return down_residual.run(
            activation, weight, input_scale, weight_scale, hidden, modulation, indices
        ).reshape_as(hidden)
    projected = _scaled_mm(activation, weight, input_scale, weight_scale).reshape_as(hidden)
    return _gated_residual(hidden, projected, modulation, indices)


def _residual_expression(hidden, projected, modulation, indices):
    width = hidden.shape[-1]
    return hidden + modulation[:, 5 * width : 6 * width].index_select(0, indices) * projected


_gated_residual = torch.compile(_residual_expression, fullgraph=True, dynamic=False)


@down.register_fake
def _down_fake(activation, weight, input_scale, weight_scale, hidden, modulation, indices):
    return torch.empty_like(hidden)


@torch.library.custom_op("h3_fp8::norm", mutates_args=())
def norm(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    modulation: torch.Tensor,
    indices: torch.Tensor,
    reciprocal_scale: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return adaln_launch(hidden, None, weight, modulation, indices, reciprocal_scale, eps, 0, 1, 0)[
        1
    ]


@norm.register_fake
def _norm_fake(hidden, weight, modulation, indices, reciprocal_scale, eps):
    return torch.empty_like(hidden, dtype=torch.float8_e4m3fn)


@torch.library.custom_op("h3_fp8::attention_projection", mutates_args=())
def attention_projection(
    raw: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    modulation: torch.Tensor,
    indices: torch.Tensor,
    reciprocal_scale: torch.Tensor,
    quant_factor: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if raw.shape[-1] != 7168 or raw.numel() // 7168 != hidden.numel() // 5376:
        raise ValueError("Expected matching attention [*,7168] and hidden [*,5376] rows")
    quant, _ = modules_for_device(raw.device.index)
    if (
        raw.is_cuda
        and raw.dtype == torch.bfloat16
        and raw.is_contiguous()
        and raw.data_ptr() % 16 == 0
        and raw.numel() // 7168 in FULL_SEQUENCE_LENGTHS
    ):
        activation = quant.quantize_config(
            raw, input_scale, quant_factor, 0, quant.pdl, BLOCKS, THREADS
        )
    else:
        activation = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(raw, input_scale)[0]
    activation = activation.reshape(-1, 7168)
    if supports_projection(activation, weight, hidden, modulation, indices, 7168):
        residual = attention_output.run(
            activation, weight, input_scale, weight_scale, hidden, modulation, indices
        ).reshape_as(hidden)
        codes = adaln_launch(
            residual, None, norm_weight, modulation, indices, reciprocal_scale, eps, 0, 4, 3
        )[1]
        return residual, codes
    projected = _scaled_mm(activation, weight, input_scale, weight_scale).reshape_as(hidden)
    return adaln_launch(
        hidden, projected, norm_weight, modulation, indices, reciprocal_scale, eps, 2, 4, 3
    )


@attention_projection.register_fake
def _attention_projection_fake(
    raw,
    weight,
    input_scale,
    weight_scale,
    hidden,
    norm_weight,
    modulation,
    indices,
    reciprocal_scale,
    quant_factor,
    eps,
):
    return torch.empty_like(hidden), torch.empty_like(hidden, dtype=torch.float8_e4m3fn)


@torch.library.custom_op("h3_fp8::shared_output_cast", mutates_args=())
def output_cast(hidden: torch.Tensor) -> torch.Tensor:
    _, cast = modules_for_device(hidden.device.index)
    eligible = (
        hidden.dtype == torch.bfloat16
        and hidden.is_contiguous()
        and tuple(hidden.shape) in tuple((1, n, 5376) for n in FULL_SEQUENCE_LENGTHS)
        and hidden.data_ptr() % 16 == 0
    )
    return cast.cast(hidden, 1, BLOCKS, THREADS) if eligible else hidden.float()


@output_cast.register_fake
def _output_cast_fake(hidden):
    return torch.empty_like(hidden, dtype=torch.float32)
