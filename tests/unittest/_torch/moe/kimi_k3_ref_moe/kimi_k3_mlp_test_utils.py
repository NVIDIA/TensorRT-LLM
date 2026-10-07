# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-only Kimi K3 MLP reference modules."""

import torch
from torch import nn

from tensorrt_llm._torch.modules.situ import SituAndMul


class NonSituActivation(nn.Module):
    """SiLU/SwiGLU activation used as a non-SiTU mutation control."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        gate = x[..., :d]
        up = x[..., d:]
        return torch.nn.functional.silu(gate) * up


class KimiK3MLP(nn.Module):
    """K3 MLP reference with a fused ``gate_up_proj`` weight layout."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        situ_beta: float = 4.0,
        situ_linear_beta: float | None = 25.0,
        activation: nn.Module | None = None,
        use_fused_activation: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if use_fused_activation and activation is not None:
            raise ValueError(
                "use_fused_activation only fuses the default SiTU activation; "
                "drop the custom activation module or the flag"
            )
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.use_fused_activation = use_fused_activation

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
            else SituAndMul(
                beta=situ_beta,
                linear_beta=situ_linear_beta,
                use_fused_activation=use_fused_activation,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation(self.gate_up_proj(x)))
