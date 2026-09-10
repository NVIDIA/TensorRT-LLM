# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_fused_add_rmsnorm catalog entry."""

import torch

from tensorrt_llm._torch.staircase.catalog.norm.flashinfer_fused_add_rmsnorm import (
    flashinfer_fused_add_rmsnorm,
)

assert torch.cuda.is_available(), "flashinfer_fused_add_rmsnorm requires a CUDA device"


def _ref(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32-accumulated reference for the fused op.

    h = fp32(x) + fp32(residual); the norm reads the fp32 h, not the
    rounded residual output, matching the kernel.
    """
    h = x.float() + residual.float()
    normed = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + eps)
    x_out = (normed * weight.float()).to(x.dtype)
    residual_out = h.to(residual.dtype)
    return x_out, residual_out


def _check(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
    ref_x, ref_res = _ref(x, residual, weight, eps)
    flashinfer_fused_add_rmsnorm(x, residual, weight, eps)
    torch.testing.assert_close(residual, ref_res)
    torch.testing.assert_close(x, ref_x)


def test_bf16_2d() -> None:
    torch.manual_seed(0)
    # decode-like (few tokens) and prefill-like (many tokens) shapes
    for num_tokens, hidden in [(1, 4096), (4, 5120), (2048, 4096)]:
        x = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
        r = torch.randn(num_tokens, hidden, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")
        _check(x, r, w, 1e-6)


def test_bf16_unaligned_hidden() -> None:
    # hidden sizes not divisible by the 128-bit vector width
    torch.manual_seed(1)
    for hidden in [111, 1152]:
        x = torch.randn(16, hidden, dtype=torch.bfloat16, device="cuda")
        r = torch.randn(16, hidden, dtype=torch.bfloat16, device="cuda")
        w = torch.randn(hidden, dtype=torch.bfloat16, device="cuda")
        _check(x, r, w, 1e-6)


def test_bf16_strided_rows() -> None:
    # last-dim contiguous slices of wider buffers (row stride != hidden);
    # row stride 8192 is divisible by the kernel vector size
    torch.manual_seed(2)
    x_buf = torch.randn(8, 8192, dtype=torch.bfloat16, device="cuda")
    r_buf = torch.randn(8, 8192, dtype=torch.bfloat16, device="cuda")
    x, r = x_buf[:, :4096], r_buf[:, :4096]
    assert not x.is_contiguous() and x.stride(-1) == 1
    w = torch.randn(4096, dtype=torch.bfloat16, device="cuda")
    # snapshot the untouched right halves before the in-place call
    right_before = torch.cat([x_buf[:, 4096:], r_buf[:, 4096:]]).clone()
    _check(x, r, w, 1e-6)
    # mutation must land in the parent buffers' left halves only
    assert torch.equal(torch.cat([x_buf[:, 4096:], r_buf[:, 4096:]]), right_before)


def test_fp16_2d() -> None:
    torch.manual_seed(3)
    for num_tokens, hidden in [(2, 4096), (1024, 2048)]:
        x = torch.randn(num_tokens, hidden, dtype=torch.float16, device="cuda")
        r = torch.randn(num_tokens, hidden, dtype=torch.float16, device="cuda")
        w = torch.randn(hidden, dtype=torch.float16, device="cuda")
        _check(x, r, w, 1e-6)


def test_fp32_2d() -> None:
    torch.manual_seed(4)
    for num_tokens, hidden in [(2, 4096), (1024, 2048)]:
        x = torch.randn(num_tokens, hidden, dtype=torch.float32, device="cuda")
        r = torch.randn(num_tokens, hidden, dtype=torch.float32, device="cuda")
        w = torch.randn(hidden, dtype=torch.float32, device="cuda")
        _check(x, r, w, 1e-6)
