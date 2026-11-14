# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RMSNorm on a K3-owned weight: the FlashInfer RMSNorm kernels through their catalog entries.

``rms_norm(x, w, eps)`` is the catalog's ``flashinfer_rmsnorm`` (``w * x / rms(x)``, FP32
statistics, the weight applied to the BF16-rounded normalized value); ``fused_add_rms_norm(x, r, w,
eps)`` is the catalog's ``flashinfer_fused_add_rmsnorm``, which writes ``r += x`` and
``x = rms_norm(r)`` in place. Both launch with PDL where ``TRTLLM_ENABLE_PDL`` is set (the op reads
it).

:class:`RMSNormWeight` holds the ``[hidden]`` weight under the parameter name ``weight`` and the
epsilon as ``variance_epsilon`` (also readable as ``eps``), the names the checkpoint loader and the
kernels that take a norm's weight read. The KDA output norm is one too: its kernel is
``rms_norm_gated_token_major`` (``modules/mamba/layernorm_gated.py``), called by the mixer on
``weight`` / ``eps``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.flashinfer_rmsnorm import (
    flashinfer_rmsnorm,
)

_FLASHINFER_DTYPES = (torch.bfloat16, torch.float16)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """``weight * x / sqrt(mean(x^2) + eps)`` over the last dim of a BF16 / FP16 ``x``."""
    if x.dtype not in _FLASHINFER_DTYPES:
        raise TypeError(f"rms_norm takes BF16 / FP16 activations, got {x.dtype}")
    return flashinfer_rmsnorm(x, weight, eps)


def fused_add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """In place: ``residual += x``, then ``x = rms_norm(residual)``. Returns ``(x, residual)``."""
    if x.dtype not in _FLASHINFER_DTYPES:
        raise TypeError(f"fused_add_rms_norm takes BF16 / FP16 activations, got {x.dtype}")
    return flashinfer_fused_add_rmsnorm(x, residual, weight, eps)


class RMSNormWeight(nn.Module):
    """The ``[hidden]`` RMSNorm weight (``weight``, initialized to ones) and its epsilon."""

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

    @property
    def eps(self) -> float:
        """``variance_epsilon``, under the name the gated (KDA ``o_norm``) and attention-residual
        norm readers use."""
        return self.variance_epsilon

    def forward(
        self, x: torch.Tensor, residual: Optional[torch.Tensor] = None
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return rms_norm(x, self.weight, self.variance_epsilon)
        return fused_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
