# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 decode GEMV: ``x @ weight^T`` for at most 8 bf16 rows on a CuTe DSL kernel (short-K or split-K, chosen
by the weight's shape)."""

import torch

import tensorrt_llm._torch.cute_dsl_kernels.k3_decode_gemv.op  # noqa: F401 — registers the op


def k3_decode_gemv(
    x: torch.Tensor, weight: torch.Tensor, trigger_early: bool = True
) -> torch.Tensor:
    """Return ``x @ weight^T`` as a new bf16 ``[M, N]`` tensor (fp32 accumulation, one bf16 rounding) in one
    k3_decode_gemv call."""
    return torch.ops.trtllm.k3_decode_gemv(x, weight, trigger_early)
