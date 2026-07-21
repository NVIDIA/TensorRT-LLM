# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense MLP helper for the in-tree Kimi K3 MoE block.

Kimi K3 uses the ``situ`` activation (not SiLU/SwiGLU). ``SituAndMul``
computes ``beta * tanh(gate / beta) * sigmoid(gate)`` on the gate half
and optionally applies ``linear_beta * tanh(up / linear_beta)`` on the
up half, then multiplies. ``KimiK3MLP`` is the fused ``gate_up_proj +
down_proj`` layout used by the shared expert stack in HF
``KimiSparseMoeBlock`` — the same shape TRT-LLM's ``GatedMLP`` uses.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class SituAndMul(nn.Module):
    """K3 SiTU activation with gate/up multiplicative gating.

    Byte-identical to HF ``modeling_kimi.py``'s ``SituAndMul`` at
    lines 41-59. Runs the math in fp32 for numerical stability
    (matches HF), then casts back to the input's dtype.
    """

    def __init__(
        self,
        *,
        beta: float = 1.0,
        linear_beta: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.linear_beta = linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d].to(torch.float32)
        up = x[..., d:].to(torch.float32)
        situ_a = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (situ_a * up).to(x.dtype)


class NonSituActivation(nn.Module):
    """SiLU/SwiGLU activation used as the non-SiTU mutation control.

    Splits the last dim into gate/up, applies SiLU to the gate, and
    multiplies element-wise. Deliberately does NOT use the SiTU
    ``beta * tanh(gate/beta) * sigmoid(gate)`` recipe.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d]
        up = x[..., d:]
        return torch.nn.functional.silu(gate) * up


class KimiK3MLP(nn.Module):
    """K3 dense/shared-expert MLP module with TRT-LLM-style fused layout.

    Weight layout:

    * ``gate_up_proj``: ``nn.Linear(hidden_size, 2 * intermediate_size, bias=False)``.
      Rows ``[:intermediate_size]`` correspond to HF's ``gate`` (KimiMLP.gate_proj
      or KimiBlockSparseMLP.w1). Rows ``[intermediate_size:]`` correspond to
      HF's ``up`` (KimiMLP.up_proj or KimiBlockSparseMLP.w3).
    * ``down_proj``: ``nn.Linear(intermediate_size, hidden_size, bias=False)``.
      Matches HF ``KimiMLP.down_proj`` or ``KimiBlockSparseMLP.w2``.

    Forward: ``down_proj( activation( gate_up_proj(x) ) )``. Default
    ``activation`` is :class:`SituAndMul`; pass a different callable to
    run mutation controls (e.g. :class:`NonSituActivation` for a
    negative-control test).
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        situ_beta: float = 4.0,
        situ_linear_beta: Optional[float] = 25.0,
        activation: Optional[nn.Module] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_up_proj = nn.Linear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.activation = (
            activation
            if activation is not None
            else SituAndMul(beta=situ_beta, linear_beta=situ_linear_beta)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.gate_up_proj(x)
        h2 = self.activation(h1)
        return self.down_proj(h2)


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
        input_dtype = hidden_states.dtype
        h = hidden_states.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + self.eps)
        return self.weight * h.to(input_dtype)
