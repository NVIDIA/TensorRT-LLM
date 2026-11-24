# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""(Add +) or gated RMSNorm fused with dynamic per-token FP8 quantization.

The quantization matches ``vectorized_per_token_fp8_quant`` on the normed output.
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

FP8_E4M3_MAX = 448.0
FP8_MIN_SCALE = 1.0 / (FP8_E4M3_MAX * 512.0)


@triton.jit
def _quantize_fp8_per_token(y, MIN_SCALE: tl.constexpr):
    """y: fp32 values of one token (any shape). Returns (fp8 codes as fp32, scale)."""
    amax = tl.max(tl.abs(y))
    scale = tl.maximum(tl.math.div_rn(amax, 448.0), MIN_SCALE)
    q = tl.clamp(tl.math.div_rn(y, scale), -448.0, 448.0)
    return q, scale


@triton.jit
def _add_rms_norm_fp8_per_token_kernel(
    X,
    R,
    W,
    Y,
    Q,
    S,
    stride_x,
    stride_r,
    stride_y,
    stride_q,
    N,
    eps,
    BLOCK_N: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
    WEIGHT_IS_DELTA: tl.constexpr,
    MIN_SCALE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        x += tl.load(R + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(R + row * stride_r + cols, x.to(R.dtype.element_ty), mask=mask)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    if WEIGHT_IS_DELTA:
        w += 1.0
    y = (x * rstd * w).to(Y.dtype.element_ty)
    tl.store(Y + row * stride_y + cols, y, mask=mask)
    q, scale = _quantize_fp8_per_token(tl.where(mask, y.to(tl.float32), 0.0), MIN_SCALE)
    tl.store(Q + row * stride_q + cols, q.to(Q.dtype.element_ty), mask=mask)
    tl.store(S + row, scale)


@torch.library.custom_op("trtllm::add_rms_norm_fp8_per_token", mutates_args=("residual",))
def add_rms_norm_fp8_per_token(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    weight: torch.Tensor,
    eps: float,
    weight_is_delta: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (y, fp8(y), fp32 scale [M, 1]) for y = rmsnorm(x [+ residual]).

    Updates ``residual`` in place; ``weight_is_delta`` scales by (1 + weight).
    """
    M, N = x.shape
    assert x.stride(-1) == 1 and (residual is None or residual.stride(-1) == 1)
    y = torch.empty_like(x)
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty((M, 1), dtype=torch.float32, device=x.device)
    if M == 0:
        return y, q, scale
    block_n = triton.next_power_of_2(N)
    with torch.cuda.device(x.device.index):
        _add_rms_norm_fp8_per_token_kernel[(M,)](
            x,
            residual,
            weight,
            y,
            q,
            scale,
            x.stride(0),
            residual.stride(0) if residual is not None else 0,
            y.stride(0),
            q.stride(0),
            N,
            eps,
            BLOCK_N=block_n,
            HAS_RESIDUAL=residual is not None,
            WEIGHT_IS_DELTA=weight_is_delta,
            MIN_SCALE=FP8_MIN_SCALE,
            # Wide CTAs while the grid is small; narrower ones past ~2k rows.
            num_warps=16 if M <= 2048 else 4,
        )
    return y, q, scale


@add_rms_norm_fp8_per_token.register_fake
def _(x, residual, weight, eps, weight_is_delta):
    del residual, weight, eps, weight_is_delta
    return (
        torch.empty_like(x),
        torch.empty_like(x, dtype=torch.float8_e4m3fn),
        x.new_empty((x.shape[0], 1), dtype=torch.float32),
    )


@triton.jit
def _rms_norm_gated_fp8_per_token_kernel(
    X,
    W,
    Z,
    Q,
    S,
    stride_x_tok,
    stride_z_tok,
    stride_q_tok,
    eps,
    N: tl.constexpr,
    HEADS: tl.constexpr,
    GATE_SIGMOID: tl.constexpr,
    MIN_SCALE: tl.constexpr,
):
    tok = tl.program_id(0).to(tl.int64)
    off = tl.arange(0, HEADS)[:, None] * N + tl.arange(0, N)[None, :]
    x = tl.load(X + tok * stride_x_tok + off).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / N + eps)
    w = tl.load(W + tl.arange(0, N)).to(tl.float32)
    y = x * rstd[:, None] * w[None, :]
    z = tl.load(Z + tok * stride_z_tok + off).to(tl.float32)
    gate = tl.sigmoid(z)
    if not GATE_SIGMOID:
        gate *= z
    # Round to the activation dtype first, like the unfused path.
    y = (y * gate).to(X.dtype.element_ty).to(tl.float32)
    q, scale = _quantize_fp8_per_token(y, MIN_SCALE)
    tl.store(Q + tok * stride_q_tok + off, q.to(Q.dtype.element_ty))
    tl.store(S + tok, scale)


def rms_norm_gated_fp8_per_token_eligible(x: torch.Tensor, z: torch.Tensor) -> bool:
    """Shapes the per-token kernel handles: one power-of-two [heads, N] tile per token."""
    num_tokens, heads, n = z.shape
    return (
        x.is_contiguous()
        and z.stride(2) == 1
        and z.stride(1) == n
        and heads & (heads - 1) == 0
        and n & (n - 1) == 0
        and heads * n <= 8192
    )


@torch.library.custom_op("trtllm::rms_norm_gated_fp8_per_token", mutates_args=())
def rms_norm_gated_fp8_per_token(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    gate_activation: str = "silu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-head gated RMSNorm, then per-token FP8 quantization.

    x: [num_tokens * heads, N]; z: [num_tokens, heads, N], contiguous per token.
    Returns (fp8 [num_tokens, heads * N], fp32 scale [num_tokens, 1]).
    """
    if gate_activation not in ("silu", "sigmoid"):
        raise ValueError(f"gate_activation must be 'silu' or 'sigmoid', got {gate_activation!r}")
    num_tokens, heads, n = z.shape
    assert x.shape == (num_tokens * heads, n)
    q = torch.empty((num_tokens, heads * n), dtype=torch.float8_e4m3fn, device=x.device)
    scale = torch.empty((num_tokens, 1), dtype=torch.float32, device=x.device)
    if num_tokens == 0:
        return q, scale
    with torch.cuda.device(x.device.index):
        _rms_norm_gated_fp8_per_token_kernel[(num_tokens,)](
            x,
            weight.contiguous(),
            z,
            q,
            scale,
            heads * n,
            z.stride(0),
            q.stride(0),
            eps,
            N=n,
            HEADS=heads,
            GATE_SIGMOID=gate_activation == "sigmoid",
            MIN_SCALE=FP8_MIN_SCALE,
            num_warps=16 if num_tokens <= 2048 else 8,
        )
    return q, scale


@rms_norm_gated_fp8_per_token.register_fake
def _(x, z, weight, eps, gate_activation="silu"):
    del x, weight, eps, gate_activation
    num_tokens, heads, n = z.shape
    return (
        z.new_empty((num_tokens, heads * n), dtype=torch.float8_e4m3fn),
        z.new_empty((num_tokens, 1), dtype=torch.float32),
    )
