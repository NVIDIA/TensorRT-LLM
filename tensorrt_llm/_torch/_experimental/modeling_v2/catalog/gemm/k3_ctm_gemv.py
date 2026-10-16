# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 decode GEMV x @ weight^T for M <= 8 bf16 tokens via the CTM (CuTe DSL) kernel."""

import torch

import tensorrt_llm._torch.cute_dsl_kernels.k3_ctm_gemv.op  # noqa: F401  (registers torch.ops.trtllm.k3_ctm_gemv*)


def k3_ctm_gemv(
    x: torch.Tensor,
    weight: torch.Tensor,
    trigger_early: bool = True,
    split: int = 1,
    push: bool = False,
) -> torch.Tensor:
    """Return `bf16(x @ weight.T)`, fp32-accumulated, in one k3_ctm_gemv call."""
    return torch.ops.trtllm.k3_ctm_gemv(
        x, weight, trigger_early=trigger_early, split=split, push=push
    )
