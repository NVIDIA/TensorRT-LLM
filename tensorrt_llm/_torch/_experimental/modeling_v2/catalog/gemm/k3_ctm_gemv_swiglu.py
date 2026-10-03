# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU down projection silu_and_mul(gu) @ weight^T for M <= 8 tokens via the CTM (CuTe DSL) kernel."""

import torch

import tensorrt_llm._torch.cute_dsl_kernels.k3_ctm_gemv.op  # noqa: F401  (registers torch.ops.trtllm.k3_ctm_gemv*)


def k3_ctm_gemv_swiglu(
    gu: torch.Tensor,
    weight: torch.Tensor,
    trigger_early: bool = True,
    split: int = 2,
    push: bool = False,
) -> torch.Tensor:
    """Return `bf16(silu_and_mul(gu) @ weight.T)`, gate columns first, in one k3_ctm_gemv_swiglu call."""
    return torch.ops.trtllm.k3_ctm_gemv_swiglu(
        gu, weight, trigger_early=trigger_early, split=split, push=push
    )
