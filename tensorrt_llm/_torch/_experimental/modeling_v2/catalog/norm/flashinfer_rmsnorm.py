# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RMS normalization over the last dim via the flashinfer rmsnorm kernel."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def flashinfer_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Return `x / sqrt(mean(x^2, dim=-1) + eps) * weight` as a new tensor."""
    return torch.ops.trtllm.flashinfer_rmsnorm(x, weight, eps)
