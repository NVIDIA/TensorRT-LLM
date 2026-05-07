#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the BF16 GEMM path in the DeepSeek-V4 Compressor wkv_gate.

The wkv_gate weight now matches the upstream V4 checkpoint dtype (bf16). The
GEMM runs in that dtype and the compressor kernels consume bf16 kv_score input.
"""

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modules.linear import Linear

DEVICE = "cuda"
# Typical DeepSeek-V4 dimensions
DIM = 4096
HEAD_DIM = 512
STATE_DIM = 2 * HEAD_DIM  # overlap=True for compress_ratio=4
OUT_DIM = STATE_DIM * 2  # wkv + gate


def _create_wkv_gate(dtype: torch.dtype = torch.bfloat16) -> Linear:
    """Create a Linear layer matching the compressor's wkv_gate."""
    gate = Linear(
        DIM,
        OUT_DIM,
        bias=False,
        dtype=dtype,
        quant_config=None,
        skip_create_weights_in_init=False,
        use_custom_cublas_mm=True,
    ).to(DEVICE)
    gate.weight.data.normal_(0, 0.02)
    return gate


def _bf16_path(x: torch.Tensor, wkv_gate: Linear) -> torch.Tensor:
    """Default path: bf16 GEMM via F.linear."""
    return F.linear(x.to(wkv_gate.weight.dtype), wkv_gate.weight)


def _fp32_reference(x: torch.Tensor, wkv_gate: Linear) -> torch.Tensor:
    """FP32 reference path."""
    return F.linear(x.float(), wkv_gate.weight.float())


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    yield
    torch.cuda.empty_cache()


@pytest.mark.parametrize("num_tokens", [1, 16, 128, 512])
def test_bf16_path_close_to_fp32_reference(num_tokens):
    """BF16 GEMM is close enough to an FP32 reference for the compressor's use.

    BF16 has 8-bit mantissa precision, but the input is already bf16 (matching
    upstream V4) so the additional precision loss from a bf16 GEMM is bounded
    by the input precision.
    """
    bf16_gate = _create_wkv_gate(torch.bfloat16)
    fp32_gate = _create_wkv_gate(torch.float32)
    # Make weights numerically equivalent (rounded to bf16 in both copies)
    fp32_gate.weight.data.copy_(bf16_gate.weight.data.float())

    x = torch.randn(num_tokens, DIM, device=DEVICE, dtype=torch.bfloat16)

    with torch.no_grad():
        out_bf16 = _bf16_path(x, bf16_gate)
        out_fp32 = _fp32_reference(x, fp32_gate)

    assert out_bf16.shape == out_fp32.shape
    assert out_bf16.dtype == torch.bfloat16

    a, b = out_bf16.float().flatten(), out_fp32.flatten()
    cos_sim = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    assert cos_sim >= 0.99, f"cos_sim={cos_sim:.6f}"

    max_diff = (a - b).abs().max().item()
    scale = max(a.abs().max().item(), b.abs().max().item(), 1e-3)
    rel_err = max_diff / scale
    assert rel_err <= 0.1, f"rel_err={rel_err:.6f}, max_diff={max_diff:.6f}"
