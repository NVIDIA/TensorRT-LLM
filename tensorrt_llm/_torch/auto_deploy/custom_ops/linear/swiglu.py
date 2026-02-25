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

"""SwiGLU MLP custom operations for graph transformation.

This module provides custom operators for SwiGLU MLP fusion:
- torch_swiglu_mlp: Intermediate representation after pattern matching
- fused_swiglu_mlp: Fused implementation with concatenated gate+up weights
"""

from typing import Optional

import torch
import torch.nn.functional as F

try:
    from flashinfer.activation import silu_and_mul as _flashinfer_silu_and_mul
except ImportError:
    _flashinfer_silu_and_mul = None


def _silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU activation: split x in half, apply silu to first half, multiply with second half.

    Uses FlashInfer's fused kernel when available, falls back to manual implementation.
    """
    if _flashinfer_silu_and_mul is not None:
        return _flashinfer_silu_and_mul(x)
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


@torch.library.custom_op("auto_deploy::torch_swiglu_mlp", mutates_args=())
def torch_swiglu_mlp(
    input: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_bias: Optional[torch.Tensor],
    up_bias: Optional[torch.Tensor],
    down_bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Standardized SwiGLU MLP operation.

    Computes: silu(x @ gate.T + gate_bias) * (x @ up.T + up_bias) @ down.T + down_bias

    This is the intermediate representation used after pattern matching,
    before weight fusion is applied.

    Args:
        input: Input tensor of shape [..., hidden_size].
        gate_weight: Gate projection weight of shape [intermediate_size, hidden_size].
        up_weight: Up projection weight of shape [intermediate_size, hidden_size].
        down_weight: Down projection weight of shape [hidden_size, intermediate_size].
        gate_bias: Optional gate projection bias of shape [intermediate_size].
        up_bias: Optional up projection bias of shape [intermediate_size].
        down_bias: Optional down projection bias of shape [hidden_size].

    Returns:
        Output tensor of shape [..., hidden_size].
    """
    gate_out = F.linear(input, gate_weight, gate_bias)
    up_out = F.linear(input, up_weight, up_bias)
    hidden = F.silu(gate_out) * up_out
    return F.linear(hidden, down_weight, down_bias)


@torch_swiglu_mlp.register_fake
def _(
    input: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_bias: Optional[torch.Tensor],
    up_bias: Optional[torch.Tensor],
    down_bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Fake implementation for tracing."""
    # Output shape is [..., hidden_size] where hidden_size = down_weight.shape[0]
    output_shape = list(input.shape[:-1]) + [down_weight.shape[0]]
    return input.new_empty(output_shape, dtype=input.dtype)


@torch.library.custom_op("auto_deploy::fused_swiglu_mlp", mutates_args=())
def fused_swiglu_mlp(
    input: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_bias: Optional[torch.Tensor],
    down_bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Fused SwiGLU MLP with concatenated gate+up weights.

    Performs a single matmul for gate and up projections, then splits the result.
    Computes: silu(gate_out) * up_out @ down.T + down_bias
    where gate_out, up_out = split(x @ gate_up.T + gate_up_bias)

    Args:
        input: Input tensor of shape [..., hidden_size].
        gate_up_weight: Concatenated gate+up weight of shape [2*intermediate_size, hidden_size].
        down_weight: Down projection weight of shape [hidden_size, intermediate_size].
        gate_up_bias: Optional concatenated gate+up bias of shape [2*intermediate_size].
        down_bias: Optional down projection bias of shape [hidden_size].

    Returns:
        Output tensor of shape [..., hidden_size].
    """
    # Single matmul for both gate and up projections
    gate_up_out = F.linear(input, gate_up_weight, gate_up_bias)

    # Apply SwiGLU activation: split, silu(gate) * up (uses FlashInfer when available)
    hidden = _silu_and_mul(gate_up_out)

    # Down projection
    return F.linear(hidden, down_weight, down_bias)


@fused_swiglu_mlp.register_fake
def _(
    input: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_bias: Optional[torch.Tensor],
    down_bias: Optional[torch.Tensor],
) -> torch.Tensor:
    """Fake implementation for tracing."""
    # Output shape is [..., hidden_size] where hidden_size = down_weight.shape[0]
    output_shape = list(input.shape[:-1]) + [down_weight.shape[0]]
    return input.new_empty(output_shape, dtype=input.dtype)


# ── NVFP4 quantized SwiGLU ops ──────────────────────────────────────────────


@torch.library.custom_op("auto_deploy::torch_nvfp4_swiglu_mlp", mutates_args=())
def torch_nvfp4_swiglu_mlp(
    input: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_input_scale: torch.Tensor,
    gate_weight_scale: torch.Tensor,
    gate_alpha: torch.Tensor,
    up_input_scale: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_alpha: torch.Tensor,
    down_input_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_alpha: torch.Tensor,
) -> torch.Tensor:
    """NVFP4 quantized SwiGLU MLP operation (intermediate representation).

    Computes: silu(nvfp4_linear(x, gate)) * nvfp4_linear(x, up) -> nvfp4_linear(down)

    This is the intermediate representation used after pattern matching for NVFP4
    quantized checkpoints, before gate+up weight fusion is applied.

    Args:
        input: Input tensor of shape [..., hidden_size].
        gate_weight: FP4 packed gate weight [intermediate_size, hidden_size/2] uint8.
        up_weight: FP4 packed up weight [intermediate_size, hidden_size/2] uint8.
        down_weight: FP4 packed down weight [hidden_size, intermediate_size/2] uint8.
        gate_input_scale: Input scale for gate projection.
        gate_weight_scale: Per-block weight scale for gate projection.
        gate_alpha: Alpha (combined scale) for gate projection.
        up_input_scale: Input scale for up projection.
        up_weight_scale: Per-block weight scale for up projection.
        up_alpha: Alpha (combined scale) for up projection.
        down_input_scale: Input scale for down projection.
        down_weight_scale: Per-block weight scale for down projection.
        down_alpha: Alpha (combined scale) for down projection.

    Returns:
        Output tensor of shape [..., hidden_size].
    """
    gate_out = torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear(
        input,
        gate_weight,
        None,
        input_scale=[gate_input_scale],
        weight_scale=[gate_weight_scale, gate_alpha],
        input_zp=[],
        weight_zp=[],
    )
    up_out = torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear(
        input,
        up_weight,
        None,
        input_scale=[up_input_scale],
        weight_scale=[up_weight_scale, up_alpha],
        input_zp=[],
        weight_zp=[],
    )
    hidden = F.silu(gate_out) * up_out
    return torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear(
        hidden,
        down_weight,
        None,
        input_scale=[down_input_scale],
        weight_scale=[down_weight_scale, down_alpha],
        input_zp=[],
        weight_zp=[],
    )


@torch_nvfp4_swiglu_mlp.register_fake
def _(
    input: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_input_scale: torch.Tensor,
    gate_weight_scale: torch.Tensor,
    gate_alpha: torch.Tensor,
    up_input_scale: torch.Tensor,
    up_weight_scale: torch.Tensor,
    up_alpha: torch.Tensor,
    down_input_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_alpha: torch.Tensor,
) -> torch.Tensor:
    """Fake implementation for tracing."""
    # Output shape: [..., hidden_size] where hidden_size = down_weight.shape[0]
    output_shape = list(input.shape[:-1]) + [down_weight.shape[0]]
    return input.new_empty(output_shape, dtype=input.dtype)


@torch.library.custom_op("auto_deploy::fused_nvfp4_swiglu_mlp", mutates_args=())
def fused_nvfp4_swiglu_mlp(
    input: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_input_scale: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    gate_up_alpha: torch.Tensor,
    down_input_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_alpha: torch.Tensor,
) -> torch.Tensor:
    """Fused NVFP4 SwiGLU MLP with concatenated gate+up weights.

    Performs a single NVFP4 matmul for gate and up projections, then splits,
    applies SwiGLU activation, and does the down NVFP4 matmul.

    Args:
        input: Input tensor of shape [..., hidden_size].
        gate_up_weight: Concatenated FP4 packed gate+up weight
            [2*intermediate_size, hidden_size/2] uint8.
        down_weight: FP4 packed down weight [hidden_size, intermediate_size/2] uint8.
        gate_up_input_scale: Shared input scale for gate+up projection.
        gate_up_weight_scale: Concatenated per-block weight scale for gate+up.
        gate_up_alpha: Shared alpha for gate+up projection.
        down_input_scale: Input scale for down projection.
        down_weight_scale: Per-block weight scale for down projection.
        down_alpha: Alpha for down projection.

    Returns:
        Output tensor of shape [..., hidden_size].
    """
    # Single NVFP4 linear for both gate and up projections
    gate_up_out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        input,
        gate_up_weight,
        bias=None,
        input_scale=gate_up_input_scale,
        weight_scale=gate_up_weight_scale,
        alpha=gate_up_alpha,
    )

    # Apply SwiGLU activation: split, silu(gate) * up (uses FlashInfer when available)
    hidden = _silu_and_mul(gate_up_out)

    # Down projection
    return torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        hidden,
        down_weight,
        bias=None,
        input_scale=down_input_scale,
        weight_scale=down_weight_scale,
        alpha=down_alpha,
    )


@fused_nvfp4_swiglu_mlp.register_fake
def _(
    input: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    gate_up_input_scale: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    gate_up_alpha: torch.Tensor,
    down_input_scale: torch.Tensor,
    down_weight_scale: torch.Tensor,
    down_alpha: torch.Tensor,
) -> torch.Tensor:
    """Fake implementation for tracing."""
    # Output shape: [..., hidden_size] where hidden_size = down_weight.shape[0]
    output_shape = list(input.shape[:-1]) + [down_weight.shape[0]]
    return input.new_empty(output_shape, dtype=input.dtype)
