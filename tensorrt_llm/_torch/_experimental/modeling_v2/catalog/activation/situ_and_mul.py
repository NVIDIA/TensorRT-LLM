# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SiTU-gated multiply (Kimi K3's MLP activation) via the fused Triton situ_and_mul kernel."""

from typing import Optional

import torch

import tensorrt_llm._torch.modules.situ  # noqa: F401 — registers torch.ops.trtllm.situ_and_mul


def situ_and_mul(x: torch.Tensor, beta: float, linear_beta: Optional[float] = None) -> torch.Tensor:
    """Return `beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta)` for `[g | u] = x`.

    `g, u = x[:, :d], x[:, d:]` with `d = x.shape[-1] // 2`; `linear_beta=None` leaves `u` as is.
    A new `[M, d]` tensor in `x.dtype`.
    """
    return torch.ops.trtllm.situ_and_mul(x, beta, linear_beta)
