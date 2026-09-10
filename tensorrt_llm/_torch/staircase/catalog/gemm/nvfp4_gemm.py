# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 x NVFP4 dense GEMM in nn.Linear layout via the trtllm unified nvfp4_gemm op."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def nvfp4_gemm(
    act_fp4: torch.Tensor,
    weight: torch.Tensor,
    act_sf: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    output_buffer_kind: int = 0,
    allowed_backends: str = "cutlass,cublaslt,cuda_core",
    group: list[int] | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return `alpha * (act @ weight.T) (+ bias)` over block-scaled NVFP4 operands, one op call."""
    # Every guard is pure metadata, and every guarded violation was observed on
    # this machine to be read wrongly *without raising* by at least one backend
    # the default `allowed_backends` string can select: the cutlass path checks
    # contiguity and raises, cublaslt does not, and none of the three default
    # backends rejects an oversized `alpha`.
    assert act_fp4.is_contiguous() and weight.is_contiguous(), (
        "act_fp4 [M, K/2] and weight [N, K/2] must be contiguous; the cublaslt "
        "backend ignores strides and returns wrong results"
    )
    assert act_sf.is_contiguous() and weight_scale.is_contiguous(), (
        "act_sf and weight_scale must be contiguous; the cublaslt backend "
        "ignores strides and returns wrong results"
    )
    assert alpha.numel() == 1, (
        "alpha must hold exactly one element; extra elements are silently "
        "ignored (this build has no per-token alpha)"
    )
    return torch.ops.trtllm.nvfp4_gemm(
        act_fp4,
        weight,
        act_sf,
        weight_scale,
        alpha,
        output_dtype,
        output_buffer_kind,
        allowed_backends,
        group,
        bias,
    )
