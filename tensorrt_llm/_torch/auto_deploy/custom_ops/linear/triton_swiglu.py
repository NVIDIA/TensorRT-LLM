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

"""Triton kernel for SwiGLU activation: silu(gate) * up.

This kernel fuses the SiLU activation and element-wise multiply into a single
GPU kernel pass, avoiding a round-trip to global memory between the two ops.
The GEMM operations (gate projection, up projection, down projection) remain
as torch.nn.functional.linear calls backed by cuBLAS.
"""

from typing import Optional

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _swiglu_activation_kernel(
    gate_ptr,
    up_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused SwiGLU activation kernel: silu(gate) * up.

    Grid: 1D, one program per BLOCK_SIZE chunk of the flattened tensor.
    Both gate and up tensors must have the same shape and be contiguous.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load gate and up values
    gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0)
    up = tl.load(up_ptr + offsets, mask=mask, other=0.0)

    # Upcast to float32 for numerical stability
    gate_f32 = gate.to(tl.float32)
    up_f32 = up.to(tl.float32)

    # SiLU(gate) = gate * sigmoid(gate)
    sigmoid_gate = tl.sigmoid(gate_f32)
    silu_gate = gate_f32 * sigmoid_gate

    # SwiGLU = silu(gate) * up
    result = silu_gate * up_f32

    # Cast back to input dtype and store
    tl.store(output_ptr + offsets, result.to(gate.dtype), mask=mask)


def triton_swiglu_activation(gate: Tensor, up: Tensor) -> Tensor:
    """Launch the Triton SwiGLU activation kernel.

    Args:
        gate: Gate projection output, shape [..., intermediate_size].
        up: Up projection output, shape [..., intermediate_size].

    Returns:
        silu(gate) * up, same shape as input.
    """
    assert gate.shape == up.shape, "gate and up must have the same shape"

    # Ensure contiguous memory layout for Triton
    gate = gate.contiguous()
    up = up.contiguous()

    n_elements = gate.numel()
    output = torch.empty_like(gate)

    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _swiglu_activation_kernel[grid](
        gate,
        up,
        output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=3,
    )

    return output


def swiglu_mlp(
    input: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    gate_bias: Optional[Tensor],
    up_bias: Optional[Tensor],
    down_bias: Optional[Tensor],
) -> Tensor:
    """SwiGLU MLP with Triton-accelerated activation.

    Computes: silu(x @ gate.T + gate_bias) * (x @ up.T + up_bias) @ down.T + down_bias

    The gate and up projections use cuBLAS (via F.linear), the SwiGLU activation
    is fused using a Triton kernel, and the down projection uses cuBLAS.

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
    # Gate and up projections (cuBLAS)
    gate_out = F.linear(input, gate_weight, gate_bias)
    up_out = F.linear(input, up_weight, up_bias)

    # Fused SwiGLU activation (Triton)
    hidden = triton_swiglu_activation(gate_out, up_out)

    # Down projection (cuBLAS)
    return F.linear(hidden, down_weight, down_bias)
