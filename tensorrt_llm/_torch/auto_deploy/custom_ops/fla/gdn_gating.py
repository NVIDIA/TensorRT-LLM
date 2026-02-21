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

"""Custom ops for fused GDN gating computation.

Computes g = -exp(A_log) * softplus(a + dt_bias) in a single kernel,
collapsing 5-7 separate kernel launches into one.

Two ops are provided:
- torch_fused_gdn_gating: pure-torch source op (used in model forward)
- triton_fused_gdn_gating: Triton kernel op (swapped in via fusion transform)
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel (adapted from tensorrt_llm/_torch/models/modeling_qwen3_next.py)
# ---------------------------------------------------------------------------
@triton.jit
def _fused_gdn_gating_kernel(
    g_ptr,
    A_log_ptr,
    a_ptr,
    dt_bias_ptr,
    seq_len,
    NUM_HEADS: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_HEADS: tl.constexpr,
):
    """Triton kernel that computes g = -exp(A_log) * softplus(a + dt_bias).

    Grid: (batch, seq_len, cdiv(NUM_HEADS, BLK_HEADS))
    """
    i_b = tl.program_id(0)
    i_s = tl.program_id(1)
    i_d = tl.program_id(2)

    head_off = i_d * BLK_HEADS + tl.arange(0, BLK_HEADS)
    off = i_b * seq_len * NUM_HEADS + i_s * NUM_HEADS + head_off
    mask = head_off < NUM_HEADS

    blk_A_log = tl.load(A_log_ptr + head_off, mask=mask)
    blk_a = tl.load(a_ptr + off, mask=mask)
    blk_bias = tl.load(dt_bias_ptr + head_off, mask=mask)

    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)
    softplus_x = tl.where(
        beta * x <= threshold,
        (1 / beta) * tl.log(1 + tl.exp(beta * x)),
        x,
    )
    blk_g = -tl.exp(blk_A_log.to(tl.float32)) * softplus_x
    tl.store(g_ptr + off, blk_g.to(g_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# torch source op (used in model forward, later replaced by fusion transform)
# ---------------------------------------------------------------------------
@torch.library.custom_op("auto_deploy::torch_fused_gdn_gating", mutates_args=())
def torch_fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    """Pure-torch fused GDN gating: g = -exp(A_log) * softplus(a + dt_bias).

    Args:
        A_log:    [H]        - log of the decay parameter
        a:        [B, S, H]  - gating activation
        dt_bias:  [H]        - bias added before softplus
        beta:     softplus beta parameter (default 1.0)
        threshold: softplus threshold for numerical stability (default 20.0)

    Returns:
        g: [B, S, H] in float32
    """
    g = -torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias.float(), beta, threshold)
    return g


@torch_fused_gdn_gating.register_fake
def _torch_fused_gdn_gating_fake(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    """Fake implementation for torch.compile / export shape propagation.

    Returns:
        g: [B, S, H] in float32 (same shape as a, always float32)
    """
    return torch.empty_like(a, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Triton fused op (swapped in by FuseGdnGating transform)
# ---------------------------------------------------------------------------
@torch.library.custom_op("auto_deploy::triton_fused_gdn_gating", mutates_args=())
def triton_fused_gdn_gating(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    """Triton-fused GDN gating: g = -exp(A_log) * softplus(a + dt_bias).

    Handles both 2D [B*S, H] and 3D [B, S, H] inputs for ``a``.

    Args:
        A_log:    [H]        - log of the decay parameter
        a:        [B, S, H]  - gating activation (3D)
        dt_bias:  [H]        - bias added before softplus
        beta:     softplus beta parameter (default 1.0)
        threshold: softplus threshold for numerical stability (default 20.0)

    Returns:
        g: [B, S, H] in float32
    """
    orig_shape = a.shape
    if a.dim() == 2:
        # 2D input: treat as [B*S, 1, H]
        batch_size = a.shape[0]
        seq_len = 1
        num_heads = a.shape[1]
        a_flat = a.contiguous()
    else:
        batch_size, seq_len, num_heads = a.shape
        a_flat = a.reshape(batch_size * seq_len, num_heads).contiguous()

    g = torch.empty(batch_size * seq_len, num_heads, device=a.device, dtype=torch.float32)

    BLK_HEADS = 8
    grid = (batch_size, seq_len, triton.cdiv(num_heads, BLK_HEADS))

    _fused_gdn_gating_kernel[grid](
        g,
        A_log,
        a_flat,
        dt_bias,
        seq_len,
        num_heads,
        beta,
        threshold,
        BLK_HEADS,
        num_warps=1,
    )

    return g.reshape(orig_shape[:-1] + (num_heads,))


@triton_fused_gdn_gating.register_fake
def _triton_fused_gdn_gating_fake(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    """Fake implementation for torch.compile / export shape propagation.

    Returns:
        g: same shape as ``a``, in float32
    """
    return torch.empty_like(a, dtype=torch.float32)
