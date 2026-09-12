# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""General matmul with optional fused bias via the trtllm cuBLASLt gemm op."""

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def cublas_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    output_buffer_kind: int = 0,
    group: list[int] | None = None,
) -> torch.Tensor:
    """Return `mat_a @ mat_b (+ bias)`, fp32-accumulated, in one cublas_mm call."""
    # The kernel reads mat_a as dense row-major and mat_b as dense
    # column-major; other layouts produce silently wrong results.
    assert mat_a.is_contiguous(), "mat_a must be dense row-major [M, K]"
    assert mat_b.stride(0) == 1 and mat_b.stride(1) == mat_b.shape[0], (
        "mat_b must be dense column-major [K, N] (e.g. weight.t())"
    )
    if bias is not None:
        out_dt = out_dtype if out_dtype is not None else mat_a.dtype
        # A bias in the wrong dtype or shape is accepted by the op and
        # produces silently wrong results; with fp32 inputs the bias is
        # accepted but silently ignored.
        assert mat_a.dtype != torch.float32, "bias is silently ignored for fp32 inputs"
        assert bias.dtype == out_dt, "bias dtype must equal the output dtype"
        assert bias.shape == (mat_b.shape[1],) and bias.is_contiguous(), (
            "bias must be a contiguous [N] tensor"
        )
    return torch.ops.trtllm.cublas_mm(mat_a, mat_b, bias, out_dtype, output_buffer_kind, group)
