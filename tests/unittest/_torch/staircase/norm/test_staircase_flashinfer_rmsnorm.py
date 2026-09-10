# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_rmsnorm catalog entry."""

import torch

from tensorrt_llm._torch.staircase.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm

assert torch.cuda.is_available(), "flashinfer_rmsnorm requires a CUDA device"


def _ref_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """fp32-accumulated reference: x / sqrt(mean(x^2, -1) + eps) * weight."""
    xf = x.float()
    normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (normed * weight.float()).to(x.dtype)


def _check(x: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    out = flashinfer_rmsnorm(x, weight, eps)
    ref = _ref_rmsnorm(x, weight, eps)
    assert out.shape == x.shape and out.dtype == x.dtype
    torch.testing.assert_close(out, ref)


def test_bf16_2d() -> None:
    torch.manual_seed(0)
    # decode-like (few tokens) and prefill-like (many tokens) shapes
    for num_tokens, hidden in [(1, 4096), (4, 5120), (2048, 4096)]:
        x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")
        _check(x, w, 1e-6)


def test_bf16_3d_qk_norm_shape() -> None:
    torch.manual_seed(1)
    x = torch.randn(16, 32, 128, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    _check(x, w, 1e-5)


def test_bf16_unaligned_hidden() -> None:
    # hidden sizes not divisible by the 128-bit vector width
    torch.manual_seed(2)
    for hidden in [111, 1152]:
        x = torch.randn(16, hidden, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")
        _check(x, w, 1e-6)


def test_bf16_strided_rows() -> None:
    # last-dim contiguous slice of a wider buffer (row stride != hidden)
    torch.manual_seed(3)
    buf = torch.randn(8, 8192, dtype=torch.bfloat16, device="cuda")
    x = buf[:, :4096]
    assert not x.is_contiguous() and x.stride(-1) == 1
    w = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
    _check(x, w, 1e-6)


def test_fp16_2d() -> None:
    torch.manual_seed(4)
    for num_tokens, hidden in [(2, 4096), (1024, 2048)]:
        x = torch.randn(num_tokens, hidden, dtype=torch.float16, device="cuda")
        w = torch.randn(hidden, dtype=torch.float16, device="cuda")
        _check(x, w, 1e-6)


def test_fp32_2d() -> None:
    torch.manual_seed(5)
    for num_tokens, hidden in [(2, 4096), (1024, 2048)]:
        x = torch.randn(num_tokens, hidden, dtype=torch.float32, device="cuda")
        w = torch.randn(hidden, dtype=torch.float32, device="cuda")
        _check(x, w, 1e-6)
