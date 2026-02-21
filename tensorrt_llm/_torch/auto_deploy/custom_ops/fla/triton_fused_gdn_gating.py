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

"""Fused Triton kernel for GDN (Gated Delta Net) gating computation.

Computes both outputs in a single kernel launch:
  g    = -exp(A_log) * softplus(a + dt_bias)   (float32 output)
  beta = sigmoid(b)                             (input dtype output)

Ported from the production kernel at:
  tensorrt_llm/_torch/models/modeling_qwen3_next.py:350-397
Generalized from 2D [B, H] to 3D [B, S, H] for prefill support.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_gdn_gating_kernel(
    g_ptr,
    beta_ptr,
    A_log_ptr,
    a_ptr,
    dt_bias_ptr,
    b_ptr,
    NUM_HEADS: tl.constexpr,
    SP_BETA: tl.constexpr,
    THRESHOLD: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    """Fused GDN gating kernel.

    Grid: (B*S, cdiv(H, BLK_HEADS))
    """
    row = tl.program_id(0)  # flattened B*S index
    col_blk = tl.program_id(1)
    head_off = col_blk * BLK_HEADS + tl.arange(0, BLK_HEADS)
    mask = head_off < NUM_HEADS
    off = row * NUM_HEADS + head_off

    blk_A = tl.load(A_log_ptr + head_off, mask=mask)
    blk_a = tl.load(a_ptr + off, mask=mask)
    blk_bias = tl.load(dt_bias_ptr + head_off, mask=mask)
    blk_b = tl.load(b_ptr + off, mask=mask)

    # beta = sigmoid(b)
    bf = blk_b.to(tl.float32)
    blk_beta = 1.0 / (1.0 + tl.exp(-bf))

    # g = -exp(A_log) * softplus(a + dt_bias)
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    sp = tl.where(
        SP_BETA * x <= THRESHOLD,
        (1.0 / SP_BETA) * tl.log(1.0 + tl.exp(SP_BETA * x)),
        x,
    )
    blk_g = -tl.exp(blk_A.to(tl.float32)) * sp

    tl.store(g_ptr + off, blk_g.to(g_ptr.dtype.element_ty), mask=mask)
    tl.store(beta_ptr + off, blk_beta.to(beta_ptr.dtype.element_ty), mask=mask)


def _fused_gdn_gating_impl(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused GDN gating: computes g and beta in a single kernel launch.

    Args:
        A_log:   [H]        - log-space decay parameters (per head)
        a:       [B, S, H]  - gating input
        dt_bias: [H]        - dt bias parameters (per head)
        b:       [B, S, H]  - beta input

    Returns:
        g:    [B, S, H] float32 - gating values
        beta: [B, S, H] input dtype - sigmoid of b
    """
    orig_shape = a.shape
    # Flatten to 2D for kernel: [B*S, H]
    num_heads = orig_shape[-1]
    a_flat = a.reshape(-1, num_heads)
    b_flat = b.reshape(-1, num_heads)
    num_rows = a_flat.shape[0]

    g = torch.empty(num_rows, num_heads, dtype=torch.float32, device=a.device)
    beta = torch.empty(num_rows, num_heads, dtype=b.dtype, device=b.device)

    BLK_HEADS = 8
    grid = (num_rows, triton.cdiv(num_heads, BLK_HEADS))
    _fused_gdn_gating_kernel[grid](
        g, beta, A_log, a_flat, dt_bias, b_flat, num_heads, 1.0, 20.0, BLK_HEADS, num_warps=1
    )

    return g.reshape(orig_shape), beta.reshape(orig_shape)


@torch.library.custom_op("auto_deploy::fused_gdn_gating", mutates_args=())
def fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused GDN gating custom op.

    Computes in a single kernel launch:
      g    = -exp(A_log) * softplus(a + dt_bias)   (float32)
      beta = sigmoid(b)                             (input dtype)

    Args:
        A_log:   [H]        - log-space decay parameters
        a:       [B, S, H]  - gating input
        dt_bias: [H]        - dt bias parameters
        b:       [B, S, H]  - beta input

    Returns:
        Tuple of (g, beta) with shapes matching inputs.
    """
    return _fused_gdn_gating_impl(A_log, a, dt_bias, b)


@fused_gdn_gating.register_fake
def _fused_gdn_gating_fake(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.empty_like(a, dtype=torch.float32)
    beta = torch.empty_like(b)
    return g, beta
