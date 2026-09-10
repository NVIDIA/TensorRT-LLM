# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SiLU-gated multiply (SwiGLU activation) via the flashinfer silu_and_mul kernel."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def flashinfer_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """Return `silu(x[..., :d]) * x[..., d:]` with `d = x.shape[-1] // 2` as a new tensor."""
    return torch.ops.trtllm.flashinfer_silu_and_mul(x)
