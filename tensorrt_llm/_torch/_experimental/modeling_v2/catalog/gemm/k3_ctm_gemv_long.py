# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 long-K decode GEMV x @ weight^T for M <= 8 tokens via the split-K CTM (CuTe DSL) kernel."""

import torch

import tensorrt_llm._torch.cute_dsl_kernels.k3_ctm_gemv.op  # noqa: F401  (registers torch.ops.trtllm.k3_ctm_gemv*)


def k3_ctm_gemv_long(
    x: torch.Tensor,
    weight: torch.Tensor,
    sig_col0: int = -1,
    split: int = 6,
    ring: int = 5,
    trigger_early: bool = True,
    push: bool = False,
) -> torch.Tensor:
    """Return `bf16(x @ weight.T)`, columns >= `sig_col0` (if >= 0) as `bf16(sigmoid(.))`, in one call."""
    return torch.ops.trtllm.k3_ctm_gemv_long(
        x,
        weight,
        sig_col0=sig_col0,
        split=split,
        ring=ring,
        trigger_early=trigger_early,
        push=push,
    )
