# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 SiTU activation and RMSNorm helpers.

Kimi K3 uses the ``situ`` activation (not SiLU/SwiGLU). ``SituAndMul``
computes ``beta * tanh(gate / beta) * sigmoid(gate)`` on the gate half
and optionally applies ``linear_beta * tanh(up / linear_beta)`` on the
up half, then multiplies.

Two activation paths coexist:

* the eager fp32 ``SituAndMul`` module — the byte-exact HF reference and
  fallback;
* the fused Triton ``trtllm::situ_and_mul`` custom op (same fp32 math
  in a single kernel, modeled on ``modules/swiglu.py``'s
  ``silu_and_mul_kernel``), enabled on :class:`SituAndMul`. The op is
  CUDA-graph-safe: no host synchronization and no data-dependent control flow.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

import torch
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]
import triton.language.extra.libdevice as tldevice  # type: ignore[import]
from torch import nn

from ...flashinfer_utils import IS_FLASHINFER_AVAILABLE

# Route the RMSNorm forward through flashinfer's single-kernel fused RMSNorm
# instead of the eager pow/mean/rsqrt/mul/cast chain. Set to "0" to fall back
# to the eager reference (the exact-parity rollback lever).
_FUSED_RMSNORM = os.environ.get("KIMI_K3_FUSED_RMSNORM", "1") == "1"


class SituAndMul(nn.Module):
    """K3 SiTU activation with gate/up multiplicative gating.

    Byte-identical to HF ``modeling_kimi.py``'s ``SituAndMul`` at
    lines 41-59. Runs the math in fp32 for numerical stability
    (matches HF), then casts back to the input's dtype. When
    ``use_fused_activation`` is enabled, CUDA inputs use the fused Triton
    implementation while CPU and meta inputs keep the eager reference path.
    """

    def __init__(
        self,
        *,
        beta: float = 1.0,
        linear_beta: Optional[float] = None,
        use_fused_activation: bool = False,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta
        self.use_fused_activation = use_fused_activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fused_activation and x.is_cuda:
            return torch.ops.trtllm.situ_and_mul(
                x.reshape(-1, x.shape[-1]), self.beta, self.linear_beta
            ).reshape(*x.shape[:-1], x.shape[-1] // 2)

        d = x.shape[-1] // 2
        gate = x[..., :d].to(torch.float32)
        up = x[..., d:].to(torch.float32)
        situ_a = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (situ_a * up).to(x.dtype)


@triton.jit
def situ_and_mul_kernel(
    o_ptr,
    o_stride,
    x_ptr,
    x_stride,
    d,
    beta,
    linear_beta,
    BLOCK_SIZE: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
) -> None:
    """Fused :class:`SituAndMul` on a packed ``[gate | up]`` row layout.

    Loads ``gate = x[i, :d]`` and ``up = x[i, d:2d]``, computes (fp32)
    ``beta * tanh(gate / beta) * sigmoid(gate) * up'`` with
    ``up' = linear_beta * tanh(up / linear_beta)`` when
    ``HAS_LINEAR_BETA`` else ``up``, and stores the product rounded to
    ``o_ptr``'s element type.
    """
    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)

    o_row_ptr = o_ptr + o_stride * i
    x_row_ptr = x_ptr + x_stride * i

    offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d

    gate = tl.load(x_row_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(x_row_ptr + offsets + d, mask=mask).to(tl.float32)

    situ_a = beta * tldevice.tanh(gate / beta) * tl.sigmoid(gate)
    if HAS_LINEAR_BETA:
        up = linear_beta * tldevice.tanh(up / linear_beta)
    result = situ_a * up

    tl.store(o_row_ptr + offsets, result, mask=mask)


@torch.library.custom_op("trtllm::situ_and_mul", mutates_args=())
def situ_and_mul(x: torch.Tensor, beta: float, linear_beta: Optional[float] = None) -> torch.Tensor:
    """Fused SiTU activation (single Triton kernel, fp32 internal math).

    Args:
        x: ``[num_tokens, 2 * d]`` packed ``[gate | up]`` GEMM output
           (fp16/bf16/fp32; the last dim must be contiguous).
        beta: SiTU gate ``beta`` (``activation_situ_beta``).
        linear_beta: optional up-half ``linear_beta``
           (``activation_situ_linear_beta``); ``None`` keeps the up half
           linear.

    Returns:
        ``[num_tokens, d]`` tensor in ``x``'s dtype, numerically matching
        the eager :class:`SituAndMul` reference.
    """
    b, n = x.shape

    assert n % 2 == 0
    d = n // 2

    o = torch.empty((b, d), dtype=x.dtype, device=x.device)

    def grid(meta: Mapping[str, int]) -> tuple[int, int]:
        return (b, triton.cdiv(d, meta["BLOCK_SIZE"]))

    situ_and_mul_kernel[grid](
        o_ptr=o,
        o_stride=o.stride(0),
        x_ptr=x,
        x_stride=x.stride(0),
        d=d,
        beta=float(beta),
        linear_beta=float(linear_beta) if linear_beta is not None else 1.0,
        BLOCK_SIZE=1024,
        HAS_LINEAR_BETA=linear_beta is not None,
    )

    return o


@situ_and_mul.register_fake
def _(x: torch.Tensor, beta: float, linear_beta: Optional[float] = None) -> torch.Tensor:
    b, n = x.shape

    assert n % 2 == 0

    return x.new_empty((b, n // 2))


class KimiK3RMSNorm(nn.Module):
    """RMSNorm matching HF ``KimiRMSNorm`` semantics exactly.

    HF ``KimiRMSNorm.forward``::

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + eps)
        return self.weight * hidden_states.to(input_dtype)

    ``self.weight`` in HF is initialised in the module's ambient dtype
    (bf16 or fp32). Callers pin the weight dtype here too so byte-exact
    parity holds regardless of the ambient dtype.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # flashinfer's fused RMSNorm does the same fp32-accumulate
        # normalization in one kernel, collapsing the eager
        # pow/mean/rsqrt/mul/cast launch chain. It is only valid for a CUDA
        # fp16/bf16 input whose dtype matches the weight; CPU / fp32 parity
        # paths, meta init, and the KIMI_K3_FUSED_RMSNORM=0 rollback keep the
        # exact eager math below.
        if (
            _FUSED_RMSNORM
            and IS_FLASHINFER_AVAILABLE
            and hidden_states.is_cuda
            and hidden_states.dtype in (torch.float16, torch.bfloat16)
            and self.weight.dtype == hidden_states.dtype
        ):
            from ...custom_ops import flashinfer_rmsnorm

            return flashinfer_rmsnorm(hidden_states.contiguous(), self.weight, self.eps)
        input_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + self.eps)
        return self.weight * h.to(input_dtype)
