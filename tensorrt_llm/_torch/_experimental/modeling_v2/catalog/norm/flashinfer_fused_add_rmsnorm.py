# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-place fused residual add + RMS normalization via the flashinfer kernel."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def flashinfer_fused_add_rmsnorm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> None:
    """In-place: residual += x; x = rmsnorm(residual) * weight. Returns None."""
    torch.ops.trtllm.flashinfer_fused_add_rmsnorm(x, residual, weight, eps)
