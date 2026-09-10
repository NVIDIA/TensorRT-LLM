# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the flashinfer_silu_and_mul catalog entry."""

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.staircase.catalog.activation.flashinfer_silu_and_mul import (
    flashinfer_silu_and_mul,
)

assert torch.cuda.is_available(), "flashinfer_silu_and_mul requires a CUDA device"


def _ref_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """fp32-accumulated reference: silu(x[..., :d]) * x[..., d:]."""
    gate, up = x.float().chunk(2, dim=-1)
    return (F.silu(gate) * up).to(x.dtype)


def _check(x: torch.Tensor) -> None:
    out = flashinfer_silu_and_mul(x)
    ref = _ref_silu_and_mul(x)
    assert out.shape == x.shape[:-1] + (x.shape[-1] // 2,)
    assert out.dtype == x.dtype
    torch.testing.assert_close(out, ref)


def test_bf16_2d() -> None:
    torch.manual_seed(0)
    # decode-like (few tokens) and prefill-like (many tokens) shapes;
    # last dim is 2 * intermediate_size (gate half then up half)
    for num_tokens, two_d in [(1, 8192), (4, 28672), (2048, 8192)]:
        x = torch.randn(num_tokens, two_d, dtype=torch.bfloat16, device="cuda")
        _check(x)


def test_bf16_3d() -> None:
    torch.manual_seed(1)
    x = torch.randn(4, 16, 2048, dtype=torch.bfloat16, device="cuda")
    _check(x)


def test_bf16_edge_sizes() -> None:
    # 16 is the smallest legal last dim (d = 8 = one vector);
    # 16400 gives d = 8200 > 8192 = blockDim * vec_size, exercising the
    # scalar remainder loop after the vectorized loop
    torch.manual_seed(2)
    for two_d in [16, 16400]:
        x = torch.randn(16, two_d, dtype=torch.bfloat16, device="cuda")
        _check(x)


def test_fp16_2d() -> None:
    torch.manual_seed(3)
    for num_tokens, two_d in [(2, 8192), (1024, 4096)]:
        x = torch.randn(num_tokens, two_d, dtype=torch.float16, device="cuda")
        _check(x)
