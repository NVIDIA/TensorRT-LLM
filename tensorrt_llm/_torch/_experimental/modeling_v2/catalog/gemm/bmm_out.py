# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batched matmul into a caller-provided output buffer via the trtllm bmm_out op."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def bmm_out(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None:
    """Compute `out[i] = a[i] @ b[i]` for every batch index, in one bmm_out call."""
    # A wrong-shaped `out` is silently resized and re-allocated by the op,
    # detaching it from any buffer the caller aliased it with.
    assert a.dim() == 3 and b.dim() == 3 and out.dim() == 3, "a, b, out must be 3D"
    assert out.shape == (a.shape[0], a.shape[1], b.shape[2]), (
        "out must be [B, M, N] matching a [B, M, K] and b [B, K, N]"
    )
    # Mixed input dtypes are type-promoted instead of rejected, changing the
    # required out dtype; the catalog exposes only the single-dtype form.
    assert a.dtype == b.dtype == out.dtype, "a, b, out must share one dtype"
    torch.ops.trtllm.bmm_out(a, b, out)
