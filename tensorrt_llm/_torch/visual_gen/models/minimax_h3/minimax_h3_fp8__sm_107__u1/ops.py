# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opaque compiler boundaries for the static-FP8 H3 block's fused operations."""

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.swiglu import swiglu
from tensorrt_llm.logger import logger

from ..fused_adaln import h3_gate_res_norm_mod, h3_norm_mod
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


_UNFUSED_REPORTED: set[str] = set()


def _report_unfused(projection: str) -> None:
    """Warn the first time a projection runs unfused: correct, but slower on every later call too."""
    if projection not in _UNFUSED_REPORTED:
        _UNFUSED_REPORTED.add(projection)
        logger.warning(
            f"MiniMax-H3 static FP8: the {projection} projection received tensors outside the "
            "fused kernel's contract (dtypes, layout or 256-byte alignment) and runs on the "
            "unfused cuBLAS path; reported once"
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


def _valid_bf16_unit(unit, device):
    return (
        unit.device == device
        and unit.dtype == torch.float32
        and unit.shape == (1,)
        and unit.is_contiguous()
        and unit.data_ptr() % 4 == 0
    )


@torch.library.custom_op("h3_fp8::gate_up_swiglu_bf16", mutates_args=(), device_types="cuda")
def gate_up_bf16(
    activation: torch.Tensor,
    weight: torch.Tensor,
    unit_scale: torch.Tensor,
    swiglu_limit: float | None,
    swiglu_alpha: float | None,
    swiglu_beta: float | None,
) -> torch.Tensor:
    """Fuse the original BF16 GateUp projection and plain SwiGLU."""
    launch = gate_up_swiglu.precompiled_bf16(activation.device.index)
    if (
        launch is not None
        and activation.ndim == 2
        and activation.shape[0] > 0
        and activation.shape[1] == gate_up_swiglu.K_DIM
        and weight.shape == (gate_up_swiglu.N_TOT, gate_up_swiglu.K_DIM)
        and activation.device == weight.device
        and activation.dtype == weight.dtype == torch.bfloat16
        and activation.is_contiguous()
        and weight.is_contiguous()
        and activation.data_ptr() % 256 == 0
        and weight.data_ptr() % 256 == 0
        and _valid_bf16_unit(unit_scale, activation.device)
        and swiglu_limit in (None, float("inf"))
        and swiglu_alpha in (None, 1.0)
        and swiglu_beta in (None, 0.0)
    ):
        output = activation.new_empty((activation.shape[0], gate_up_swiglu.D_HALF))
        if output.data_ptr() % 256 == 0:
            launch(activation, weight, unit_scale, unit_scale, unit_scale, output)
            return output
    return swiglu(
        F.linear(activation, weight),
        swiglu_limit=swiglu_limit,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
    )


@gate_up_bf16.register_fake
def _gate_up_bf16_fake(activation, weight, unit_scale, swiglu_limit, swiglu_alpha, swiglu_beta):
    return activation.new_empty((*activation.shape[:-1], weight.shape[0] // 2))


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
    if activation.dtype == torch.float8_e4m3fn and projection_residual.supports(
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
    _report_unfused("Down")
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


def _require_bf16_projection(activation, weight):
    if activation.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise ValueError("BF16 projection requires BF16 activation and weight")


@torch.library.custom_op("h3_fp8::bf16_down_gated_residual", mutates_args=(), device_types="cuda")
def bf16_down(
    activation: torch.Tensor,
    weight: torch.Tensor,
    unit: torch.Tensor,
    hidden: torch.Tensor,
    modulation: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    _require_bf16_projection(activation, weight)
    gate_column = projection_residual.DOWN_GATE
    if _valid_bf16_unit(unit, activation.device) and projection_residual.supports(
        activation, weight, hidden, modulation, indices, gate_column
    ):
        return projection_residual.run(
            activation, weight, unit, unit, hidden, modulation, indices, gate_column
        ).reshape_as(hidden)
    projected = F.linear(activation, weight).reshape_as(hidden)
    return _gated_residual(hidden, projected, modulation, indices, gate_column)


@bf16_down.register_fake
def _bf16_down_fake(activation, weight, unit, hidden, modulation, indices):
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
    _report_unfused("attention output")
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


@torch.library.custom_op("h3_fp8::bf16_attention_projection", mutates_args=(), device_types="cuda")
def bf16_attention_projection(
    raw: torch.Tensor,
    weight: torch.Tensor,
    unit: torch.Tensor,
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    modulation: torch.Tensor,
    indices: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    _require_bf16_projection(raw, weight)
    if (
        raw.shape[-1:] != (7168,)
        or hidden.shape[-1:] != (5376,)
        or raw.shape[:-1] != hidden.shape[:-1]
    ):
        raise ValueError("Expected matching attention [*,7168] and hidden [*,5376] rows")
    activation = raw.reshape(-1, 7168)
    gate_column = projection_residual.ATTENTION_OUTPUT_GATE
    if _valid_bf16_unit(unit, activation.device) and projection_residual.supports(
        activation, weight, hidden, modulation, indices, gate_column
    ):
        residual = projection_residual.run(
            activation, weight, unit, unit, hidden, modulation, indices, gate_column
        ).reshape_as(hidden)
        return residual, h3_norm_mod(residual, norm_weight, modulation, indices, eps, 4, 3)
    projected = F.linear(raw, weight)
    return h3_gate_res_norm_mod(
        hidden, projected, norm_weight, modulation, indices, eps, gate_column, 4, 3
    )


@bf16_attention_projection.register_fake
def _bf16_attention_projection_fake(
    raw, weight, unit, hidden, norm_weight, modulation, indices, eps
):
    return torch.empty_like(hidden), torch.empty_like(hidden)


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
