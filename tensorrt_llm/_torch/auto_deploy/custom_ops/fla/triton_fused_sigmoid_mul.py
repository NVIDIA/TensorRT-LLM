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

"""Fused Triton kernel for sigmoid(gate) * x.

Handles two cases:
  - Same-shape: gate [M, D] * x [M, D]  (attention output gating)
  - Broadcast:  gate [M, 1] * x [M, D]  (shared expert gating)

Both cases are handled by a single kernel with a compile-time GATE_COLS
constant that selects the loading strategy.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_sigmoid_mul_kernel(
    out_ptr,
    gate_ptr,
    x_ptr,
    N_COLS: tl.constexpr,
    GATE_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused sigmoid-mul kernel.

    Grid: (M,) where M = number of rows.
    GATE_COLS: 1 for broadcast, N_COLS for same-shape.
    """
    row = tl.program_id(0)
    col_off = tl.arange(0, BLOCK_N)
    mask = col_off < N_COLS

    if GATE_COLS == 1:
        # Broadcast: load single gate value per row
        gate_val = tl.load(gate_ptr + row).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-gate_val))
    else:
        # Same-shape: load full row of gate values
        gate = tl.load(gate_ptr + row * N_COLS + col_off, mask=mask).to(tl.float32)
        sig = 1.0 / (1.0 + tl.exp(-gate))

    x = tl.load(x_ptr + row * N_COLS + col_off, mask=mask)
    out = (sig * x.to(tl.float32)).to(x.dtype)
    tl.store(out_ptr + row * N_COLS + col_off, out, mask=mask)


def _fused_sigmoid_mul_impl(
    gate: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Fused sigmoid(gate) * x implementation.

    Args:
        gate: [M, D] or [M, 1] - gate logits (before sigmoid)
        x:    [M, D]            - input to be gated

    Returns:
        [M, D] - sigmoid(gate) * x in x's dtype
    """
    orig_shape = x.shape
    # Flatten to 2D
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N_COLS = x_2d.shape

    gate_2d = gate.reshape(-1, gate.shape[-1])
    GATE_COLS = gate_2d.shape[-1]

    out = torch.empty_like(x_2d)

    BLOCK_N = triton.next_power_of_2(N_COLS)
    grid = (M,)
    _fused_sigmoid_mul_kernel[grid](
        out, gate_2d, x_2d, N_COLS, GATE_COLS, BLOCK_N, num_warps=min(8, max(1, BLOCK_N // 256))
    )

    return out.reshape(orig_shape)


@torch.library.custom_op("auto_deploy::fused_sigmoid_mul", mutates_args=())
def fused_sigmoid_mul(
    gate: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Fused sigmoid(gate) * x custom op.

    Computes sigmoid(gate) * x in a single kernel launch.
    Supports both same-shape (gate and x have the same shape) and
    broadcast (gate has trailing dim 1) modes.

    Args:
        gate: [*, D] or [*, 1] - gate logits (before sigmoid)
        x:    [*, D]           - input to be gated

    Returns:
        [*, D] - sigmoid(gate) * x in x's dtype
    """
    return _fused_sigmoid_mul_impl(gate, x)


@fused_sigmoid_mul.register_fake
def _fused_sigmoid_mul_fake(
    gate: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(x)
