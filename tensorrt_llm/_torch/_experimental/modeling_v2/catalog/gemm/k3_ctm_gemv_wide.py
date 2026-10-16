# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3 wide decode GEMV x @ weight^T for M <= 64 tokens via the split-K CTM (CuTe DSL) kernel."""

import torch

import tensorrt_llm._torch.cute_dsl_kernels.k3_ctm_gemv.op  # noqa: F401  (registers torch.ops.trtllm.k3_ctm_gemv*)


def k3_ctm_gemv_wide(
    x: torch.Tensor,
    weight: torch.Tensor,
    sig_col0: int = -1,
    out_fp32: bool = False,
) -> torch.Tensor:
    """Return `x @ weight.T` in bf16 (columns >= `sig_col0` as sigmoid) or fp32, in one call."""
    return torch.ops.trtllm.k3_ctm_gemv_wide(x, weight, sig_col0=sig_col0, out_fp32=out_fp32)
