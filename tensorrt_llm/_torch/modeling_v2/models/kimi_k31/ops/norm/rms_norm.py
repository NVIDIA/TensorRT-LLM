# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 RMSNorm parameter holder: a norm weight and its epsilon, normalized by the catalog FlashInfer RMSNorm."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm


class K31RMSNorm(nn.Module):
    """Parameter holder + op call; not a layer.

    ``weight [hidden_size]`` (initialized to ones, as the engine ``RMSNorm``) and ``variance_epsilon``, under the
    engine module's names. The targets pass both to the norm ops they call (``flashinfer_rmsnorm``, the fused
    attention-residual and all-reduce norms); ``forward`` is ``flashinfer_rmsnorm(x, weight, variance_epsilon)``, the
    kernel the engine ``RMSNorm`` runs on a BF16 input without a residual.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype, device=device))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return flashinfer_rmsnorm(x, self.weight, self.variance_epsilon)
