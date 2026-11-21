# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opaque compiler boundaries for the static-FP8 H3 block's fused operations."""

import torch

from .kernels import gate_up_swiglu, projection_residual
from .kernels.adaln import launch as adaln_launch
from .kernels.conversion import BLOCKS, THREADS, modules_for_device


@torch.library.custom_op("h3_fp8::qkv_gemm", mutates_args=(), device_types="cuda")
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


def _cublas_scaled_mm(
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


@torch.library.custom_op("h3_fp8::gate_up_swiglu", mutates_args=(), device_types="cuda")
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


@torch.library.custom_op("h3_fp8::down_gated_residual", mutates_args=(), device_types="cuda")
def down(
    activation: torch.Tensor,
    weight: torch.Tensor,
    input_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    hidden: torch.Tensor,
    modulation: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    if projection_residual.supports(
        activation, weight, hidden, modulation, indices, projection_residual.DOWN_GATE
    ):
        return projection_residual.run(
            activation,
            weight,
            input_scale,
            weight_scale,
            hidden,
            modulation,
            indices,
            projection_residual.DOWN_GATE,
        ).reshape_as(hidden)
    projected = _cublas_scaled_mm(activation, weight, input_scale, weight_scale).reshape_as(hidden)
    return _gated_residual(hidden, projected, modulation, indices, projection_residual.DOWN_GATE)


def _gated_residual(hidden, projected, modulation, indices, gate_column):
    """The kernel's residual for tensors outside its layout contract: FP32, then one BF16 round."""
    width = hidden.shape[-1]
    gate = modulation[:, gate_column * width : (gate_column + 1) * width].index_select(0, indices)
    return torch.addcmul(hidden.float(), gate.float(), projected.float()).to(hidden.dtype)


@down.register_fake
def _down_fake(activation, weight, input_scale, weight_scale, hidden, modulation, indices):
    return torch.empty_like(hidden)


@torch.library.custom_op("h3_fp8::norm", mutates_args=(), device_types="cuda")
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


@torch.library.custom_op("h3_fp8::attention_projection", mutates_args=(), device_types="cuda")
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
        quant is not None
        and raw.is_cuda
        and raw.dtype == torch.bfloat16
        and raw.is_contiguous()
        and raw.data_ptr() % 16 == 0
    ):
        activation = quant.quantize_config(
            raw, input_scale, quant_factor, 0, quant.pdl, BLOCKS, THREADS
        )
    else:
        activation = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(raw, input_scale)[0]
    activation = activation.reshape(-1, 7168)
    if projection_residual.supports(
        activation, weight, hidden, modulation, indices, projection_residual.ATTENTION_OUTPUT_GATE
    ):
        residual = projection_residual.run(
            activation,
            weight,
            input_scale,
            weight_scale,
            hidden,
            modulation,
            indices,
            projection_residual.ATTENTION_OUTPUT_GATE,
        ).reshape_as(hidden)
        codes = adaln_launch(
            residual, None, norm_weight, modulation, indices, reciprocal_scale, eps, 0, 4, 3
        )[1]
        return residual, codes
    projected = _cublas_scaled_mm(activation, weight, input_scale, weight_scale).reshape_as(hidden)
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


@torch.library.custom_op("h3_fp8::shared_output_cast", mutates_args=(), device_types="cuda")
def output_cast(hidden: torch.Tensor) -> torch.Tensor:
    _, cast = modules_for_device(hidden.device.index)
    eligible = (
        cast is not None
        and hidden.dtype == torch.bfloat16
        and hidden.is_contiguous()
        and hidden.data_ptr() % 16 == 0
    )
    return cast.cast(hidden, 1, BLOCKS, THREADS) if eligible else hidden.float()


@output_cast.register_fake
def _output_cast_fake(hidden):
    return torch.empty_like(hidden, dtype=torch.float32)
