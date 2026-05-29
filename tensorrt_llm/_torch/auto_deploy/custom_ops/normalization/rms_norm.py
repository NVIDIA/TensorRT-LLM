# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom operator for FlashInfer and Triton RMSNorm implementation."""

import flashinfer
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils

from ..quantization.quant import TRTLLM_NVFP4_SCALING_VECTOR_SIZE

try:
    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
except (ModuleNotFoundError, ImportError):
    import os

    def get_env_enable_pdl() -> bool:
        return os.environ.get("TRTLLM_ENABLE_PDL", "1") == "1"


try:
    from tensorrt_llm._torch.modules.mamba.layernorm_gated import _layer_norm_fwd
except (ModuleNotFoundError, ImportError):
    _layer_norm_fwd = None
from .triton_rms_norm import rms_norm


def _get_nvfp4_fake_shapes(x: torch.Tensor) -> tuple[tuple[int, ...], int]:
    output_shape, sf_size = fp4_utils.get_fp4_shape(x.shape, TRTLLM_NVFP4_SCALING_VECTOR_SIZE)
    return tuple(output_shape), sf_size


@torch.library.custom_op("auto_deploy::flashinfer_rms_norm", mutates_args=())
def flashinfer_rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Custom operator for FlashInfer RMSNorm implementation.

    Args:
        input: Input tensor to normalize.
        weight: Scaling weights for the normalized output.
        eps: Small constant for numerical stability.

    Returns:
        Normalized and scaled tensor using FlashInfer implementation.
    """
    # Flashinfer rmsnorm expects a 2D input
    input_flat = input.reshape(-1, input.shape[-1])
    rmsnorm_flat = flashinfer.norm.rmsnorm(input_flat, weight, eps, enable_pdl=get_env_enable_pdl())
    return rmsnorm_flat.reshape(input.shape)


@flashinfer_rmsnorm.register_fake
def _(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Fake implementation for the custom operator during tracing.

    Args:
        input: Input tensor to normalize.
        weight: Scaling weights for the normalized output.
        eps: Small constant for numerical stability.

    Returns:
        Empty tensor with same shape as input.
    """
    return torch.empty_like(input)


@torch.library.custom_op("auto_deploy::triton_rms_norm", mutates_args=())
def triton_rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Custom operator for Triton RMSNorm implementation.

    Args:
        input: Input tensor to normalize.
        weight: Scaling weights for the normalized output.
        eps: Small constant for numerical stability.

    Returns:
        Normalized and scaled tensor using Triton implementation.
    """
    return rms_norm(input, weight, eps)


@triton_rmsnorm.register_fake
def _(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Fake implementation for the custom operator during tracing."""
    return torch.empty_like(input)


@torch.library.custom_op("auto_deploy::torch_rmsnorm", mutates_args=())
def torch_rmsnorm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Custom operator for Torch RMSNorm implementation.

    Args:
        input: Input tensor to normalize.
        weight: Scaling weights for the normalized output.
        eps: Small constant for numerical stability.
    """
    # pre-allocate output to ensure same dtype+stride as input
    out = torch.empty_like(input)
    input = input.to(torch.float32)
    variance = input.pow(2).mean(-1, keepdim=True)
    input = input * torch.rsqrt(variance + eps)
    out.copy_((weight * input.to(out.dtype)))
    return out


@torch_rmsnorm.register_fake
def _(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Fake implementation for the custom operator during tracing."""
    return torch.empty_like(input)


@torch.library.custom_op("auto_deploy::torch_rmsnorm_gated", mutates_args=())
def torch_rmsnorm_gated(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor | None,
    eps: float,
    group_size: int,
    norm_before_gate: bool = False,
    tp_mode: str = "none",
    layer_type: str = "unknown",
) -> torch.Tensor:
    """Custom operator for Torch gated RMSNorm implementation.

    Group RMSNorm with optional SiLU gating, using pure PyTorch operations.

    Args:
        x: Input tensor of shape [..., H].
        weight: Scaling weights of shape [H].
        gate: Optional gate tensor with same shape as x, or None.
        eps: Small constant for numerical stability.
        group_size: Size of groups for grouped normalization. H must be divisible by group_size.
        norm_before_gate: If True, apply gating after normalization. If False, apply before.
        tp_mode: Tensor-parallel sharding hint for transforms.
        layer_type: Layer id hint for selective sharding (e.g. ``shard_layers``).

    Returns:
        Normalized and optionally gated tensor of shape like x.
    """
    dtype = x.dtype
    weight = weight.float()
    x = x.float()
    z = gate.float() if gate is not None else gate

    if z is not None and not norm_before_gate:
        x = x * F.silu(z)

    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = x * rstd * weight
    else:
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight

    if z is not None and norm_before_gate:
        out *= F.silu(z)

    return out.to(dtype)


@torch_rmsnorm_gated.register_fake
def _(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor | None,
    eps: float,
    group_size: int,
    norm_before_gate: bool = False,
    tp_mode: str = "none",
    layer_type: str = "unknown",
) -> torch.Tensor:
    """Fake implementation for the custom operator during tracing."""
    return x.new_empty(x.shape, dtype=x.dtype)


@torch.library.custom_op("auto_deploy::triton_rmsnorm_gated", mutates_args=())
def triton_rmsnorm_gated(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate: torch.Tensor | None,
    eps: float,
    group_size: int,
    norm_before_gate: bool = False,
    tp_mode: str = "none",
    layer_type: str = "unknown",
) -> torch.Tensor:
    """
    Group RMSNorm with optional SiLU gating, using Triton kernel `_layer_norm_fwd`.

    Shapes:
      x:     [..., H]
      gate:  same as x or None
      weight:[H]
      H % group_size == 0

    Returns:
      fp32 tensor of shape like x.
    """
    assert x.dim() >= 2, "x must be at least 2D"
    H = weight.numel()
    assert x.shape[-1] == H, "weight must match last dim (hidden size)"
    assert (H % group_size) == 0, f"H={H} must be divisible by group_size={group_size}"
    if gate is not None:
        assert gate.shape == x.shape, "gate must match x shape"

    # Flatten to (M, H), ensure last-dim contiguous, and run in fp32
    x_shape = x.shape
    x2 = x.reshape(-1, H)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()

    z2 = None
    if gate is not None:
        z2 = gate.reshape(-1, H)
        if z2.stride(-1) != 1:
            z2 = z2.contiguous()
    assert weight.is_contiguous(), "weight must be contiguous"

    out2, _, _ = _layer_norm_fwd(
        x2,
        weight,
        None,  # bias
        eps,
        z=z2,
        out=None,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        is_rms_norm=True,
    )

    return out2.reshape(x_shape)


@triton_rmsnorm_gated.register_fake
def _triton_rmsnorm_gated_meta(
    x,
    weight,
    gate,
    eps: float,
    group_size: int,
    norm_before_gate: bool = False,
    tp_mode: str = "none",
    layer_type: str = "unknown",
):
    assert x.dim() >= 2, "x must be at least 2D"
    H = x.shape[-1]
    assert weight.numel() == H, "weight must match last dim (hidden size)"
    assert (H % group_size) == 0, f"H={H} must be divisible by group_size={group_size}"
    if gate is not None:
        assert gate.shape == x.shape, "gate must match x shape"

    return x.new_empty(x.shape, dtype=x.dtype)


@torch.library.custom_op("auto_deploy::trtllm_fused_gated_rmsnorm_quant_nvfp4", mutates_args=())
def trtllm_fused_gated_rmsnorm_quant_nvfp4(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse gated RMSNorm and NVFP4 quantization using the TRT-LLM Torch kernel."""
    if weight.dtype in (torch.float16, torch.bfloat16):
        kernel_dtype = weight.dtype
    elif gate.dtype in (torch.float16, torch.bfloat16):
        kernel_dtype = gate.dtype
    else:
        kernel_dtype = x.dtype

    if x.dtype != kernel_dtype:
        x = x.to(kernel_dtype)
    if gate.dtype != kernel_dtype:
        gate = gate.to(kernel_dtype)
    if weight.dtype != kernel_dtype:
        weight = weight.to(kernel_dtype)

    x_shape = x.shape
    hidden_size = x_shape[-1]
    x_2d = x.reshape(-1, hidden_size)
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()

    gate_2d = gate.reshape(-1, hidden_size)
    if gate_2d.stride(-1) != 1:
        gate_2d = gate_2d.contiguous()

    fp4_i32, scale_factors = torch.ops.trtllm.fused_gated_rmsnorm_quant(
        x_2d, gate_2d, weight.contiguous(), group_size, eps, scale.contiguous()
    )
    fp4_u8 = fp4_i32.view(torch.uint8)
    return fp4_u8.reshape(*x_shape[:-1], hidden_size // 2), scale_factors


@trtllm_fused_gated_rmsnorm_quant_nvfp4.register_fake
def _trtllm_fused_gated_rmsnorm_quant_nvfp4_fake(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del gate, weight, scale, eps, group_size
    output_shape, sf_size = _get_nvfp4_fake_shapes(x)
    return x.new_empty(output_shape, dtype=torch.uint8), x.new_empty((sf_size,), dtype=torch.uint8)


def _run_trtllm_fused_add_rmsnorm_quant_nvfp4(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
    output_hp_norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    x_shape = x.shape
    hidden_size = x_shape[-1]
    x_2d = x.reshape(-1, hidden_size)
    if x_2d.stride(-1) != 1:
        x_2d = x_2d.contiguous()

    residual_2d = residual.reshape(-1, hidden_size)
    if residual_2d.dtype != x_2d.dtype:
        residual_2d = residual_2d.to(x_2d.dtype)
    if residual_2d.stride(-1) != 1:
        residual_2d = residual_2d.contiguous()

    if weight.dtype != x_2d.dtype:
        weight = weight.to(x_2d.dtype)

    fp4_i32, residual_out, scale_factors, norm_out = torch.ops.trtllm.fused_add_rms_norm_quant(
        x_2d,
        residual_2d,
        weight.contiguous(),
        scale.contiguous(),
        True,
        eps,
        output_hp_norm,
    )
    fp4_u8 = fp4_i32.view(torch.uint8).reshape(*x_shape[:-1], hidden_size // 2)
    residual_out = residual_out.reshape(x_shape)
    if norm_out is not None:
        norm_out = norm_out.reshape(x_shape)
    return fp4_u8, residual_out, scale_factors, norm_out


@torch.library.custom_op("auto_deploy::trtllm_fused_add_rmsnorm_quant_nvfp4", mutates_args=())
def trtllm_fused_add_rmsnorm_quant_nvfp4(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse residual add, RMSNorm, and NVFP4 quantization using a TRT-LLM kernel."""
    fp4_out, residual_out, scale_factors, _ = _run_trtllm_fused_add_rmsnorm_quant_nvfp4(
        x, residual, weight, scale, eps, False
    )
    return fp4_out, residual_out, scale_factors


@trtllm_fused_add_rmsnorm_quant_nvfp4.register_fake
def _trtllm_fused_add_rmsnorm_quant_nvfp4_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    del residual, weight, scale, eps
    output_shape, sf_size = _get_nvfp4_fake_shapes(x)
    return (
        x.new_empty(output_shape, dtype=torch.uint8),
        torch.empty_like(x),
        x.new_empty((sf_size,), dtype=torch.uint8),
    )


@torch.library.custom_op("auto_deploy::trtllm_fused_add_rmsnorm_out_quant_nvfp4", mutates_args=())
def trtllm_fused_add_rmsnorm_out_quant_nvfp4(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse residual add, RMSNorm, and NVFP4 quantization while keeping BF16 norm output."""
    fp4_out, residual_out, scale_factors, norm_out = _run_trtllm_fused_add_rmsnorm_quant_nvfp4(
        x, residual, weight, scale, eps, True
    )
    assert norm_out is not None
    return norm_out, fp4_out, residual_out, scale_factors


@trtllm_fused_add_rmsnorm_out_quant_nvfp4.register_fake
def _trtllm_fused_add_rmsnorm_out_quant_nvfp4_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del residual, weight, scale, eps
    output_shape, sf_size = _get_nvfp4_fake_shapes(x)
    return (
        torch.empty_like(x),
        x.new_empty(output_shape, dtype=torch.uint8),
        torch.empty_like(x),
        x.new_empty((sf_size,), dtype=torch.uint8),
    )


# Forked from:
# https://github.com/state-spaces/mamba/blob/6b32be06d026e170b3fdaf3ae6282c5a6ff57b06/mamba_ssm/ops/triton/layernorm_gated.py
# NOTES:
# 1. At time of writing (09/25/2025), the nano nemotron v2 modeling code expects `mamba_ssm`
#    to be installed so as to be able to make use of its grouped gated RMS norm operation.
#    We therefore replace it with one that uses einops + pytorch.
def gated_rms_norm_ref(
    x, weight, bias, z=None, eps=1e-6, group_size=None, norm_before_gate=True, upcast=True
):
    dtype = x.dtype
    # N = x.shape[-1]
    weight = weight.float()
    bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        z = z.float() if z is not None else z
    if z is not None and not norm_before_gate:
        x = x * F.silu(z)
    if group_size is None:
        rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
        out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    else:
        x_group = rearrange(x, "... (g d) -> ... g d", d=group_size)
        rstd = 1 / torch.sqrt((x_group.square()).mean(dim=-1, keepdim=True) + eps)
        out = rearrange(x_group * rstd, "... g d -> ... (g d)") * weight
        if bias is not None:
            out = out + bias
    if z is not None and norm_before_gate:
        out *= F.silu(z)
    return out.to(dtype)


# =============================================================================
# Sharded RMSNorm (for sharded activations)
# =============================================================================


@torch.library.custom_op("auto_deploy::sharded_rmsnorm", mutates_args=())
def sharded_rmsnorm(
    input: torch.Tensor, weight: torch.Tensor, eps: float, world_size: int
) -> torch.Tensor:
    """RMSNorm for sharded activations that need global reduction.

    When activations are sharded (split along the last dimension across devices),
    standard RMSNorm computes an incorrect local mean. This op uses all_reduce to compute
    the global mean of squared values across all shards, ensuring correct normalization.

    The computation is:
        1. Compute local sum of squares: sum(input^2) over local features
        2. All-reduce to get global sum of squares across all shards
        3. Compute global mean: global_sum / (local_dim * world_size)
        4. Normalize: input * rsqrt(global_mean + eps)
        5. Scale with local weight (weight is also column-sharded)

    Args:
        input: Input tensor, shape [..., local_hidden_size] where local_hidden_size
               is the shard of the full hidden dimension on this device.
        weight: Scaling weights, shape [local_hidden_size] (column-sharded).
        eps: Small constant for numerical stability.
        world_size: Number of devices across which the activation is sharded.

    Returns:
        Normalized and scaled tensor with same shape as input.
    """
    local_dim = input.shape[-1]

    # Cast to float32 for precision
    input_fp32 = input.to(torch.float32)

    # Compute local sum of squares (NOT mean - we need sum for all_reduce)
    local_sum_sq = input_fp32.pow(2).sum(-1, keepdim=True)

    # All-reduce to get global sum of squares
    global_sum_sq = local_sum_sq.clone()
    dist.all_reduce(global_sum_sq, op=dist.ReduceOp.SUM)

    # Compute global mean: global_sum / total_elements
    global_count = local_dim * world_size
    global_mean_sq = global_sum_sq / global_count

    # Normalize
    input_normalized = input_fp32 * torch.rsqrt(global_mean_sq + eps)

    # Apply weight (local weight since it's also column-sharded)
    out = weight * input_normalized.to(input.dtype)
    return out


@sharded_rmsnorm.register_fake
def _(input: torch.Tensor, weight: torch.Tensor, eps: float, world_size: int) -> torch.Tensor:
    """Fake implementation for tracing."""
    return torch.empty_like(input)
