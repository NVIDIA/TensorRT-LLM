# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated RMS norm of per-head rows, with the gate read from a token-major [tokens, heads, N] view."""

from typing import Optional

import torch

import tensorrt_llm._torch.modules.mamba.layernorm_gated  # noqa: F401 — registers the op


def rms_norm_gated_token_major(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    fp8_scale: Optional[torch.Tensor] = None,
    gate_activation: str = "silu",
) -> torch.Tensor:
    """Return `rmsnorm(x) * weight * gate(z)` per row, `gate` = sigmoid ("sigmoid") or z * sigmoid(z) ("silu").

    Row `r = t * heads + h` of `x` [tokens * heads, N] is gated by `z[t, h, :]`. With `fp8_scale`
    the result is quantized to float8_e4m3fn in the same kernel. A new tensor.
    """
    return torch.ops.trtllm.rms_norm_gated_token_major(
        x, z, weight, eps, fp8_scale, gate_activation
    )
