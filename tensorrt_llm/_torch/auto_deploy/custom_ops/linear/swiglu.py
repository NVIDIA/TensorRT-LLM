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

    # Split into gate and up outputs
    gate_out, up_out = gate_up_out.chunk(2, dim=-1)

    # Apply SwiGLU activation: silu(gate) * up
    hidden = F.silu(gate_out) * up_out

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
