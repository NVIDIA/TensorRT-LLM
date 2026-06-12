# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the Triton kernel backend of torch_rope_with_qk_interleaving."""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401 (triggers op registration)
from tensorrt_llm._torch.auto_deploy.custom_ops.rope.triton_rope_with_qk_interleaving import (
    rope_with_qk_interleaving_fused,
)


def _get_tolerances(dtype):
    """Return (rtol, atol) appropriate for the given dtype."""
    if dtype == torch.bfloat16:
        return 1e-2, 1e-2
    return 1e-3, 1e-3


def _make_cos_sin(batch, seq_len, head_dim, dtype, device):
    """Create duplicated cos/sin tensors matching DeepSeek RoPE convention."""
    inv_freq = 1.0 / (
        10000
        ** (
            torch.arange(0, head_dim // 2, dtype=torch.float32, device=device)
            / (head_dim // 2)
        )
    )
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # [seq_len, head_dim//2]
    cos_half = torch.cos(angles)
    sin_half = torch.sin(angles)
    # Duplicate to [seq_len, head_dim] as in DeepSeek
    cos = torch.cat([cos_half, cos_half], dim=-1).to(dtype)
    sin = torch.cat([sin_half, sin_half], dim=-1).to(dtype)
    # Expand to [batch, seq_len, head_dim]
    cos = cos.unsqueeze(0).expand(batch, -1, -1)
    sin = sin.unsqueeze(0).expand(batch, -1, -1)
    return cos, sin


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "batch, seq_len, n_heads, head_dim",
    [
        (1, 1, 1, 64),
        (2, 4, 3, 64),
        (2, 8, 8, 128),
        (1, 16, 4, 256),
    ],
)
def test_triton_matches_torch_reference(batch, seq_len, n_heads, head_dim, dtype):
    """Triton interleaved RoPE output matches the torch reference."""
    torch.manual_seed(42)
    device = "cuda"

    q = torch.randn(batch, seq_len, n_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seq_len, n_heads, head_dim, dtype=dtype, device=device)
    cos, sin = _make_cos_sin(batch, seq_len, head_dim, dtype, device)

    # unsqueeze_dim=2 for BSND layout
    expected_q, expected_k = torch.ops.auto_deploy.torch_rope_with_qk_interleaving(
        q, k, cos, sin, 2
    )
    actual_q, actual_k = torch.ops.auto_deploy.triton_rope_with_qk_interleaving(
        q, k, cos, sin, 2
    )

    rtol, atol = _get_tolerances(dtype)
    torch.testing.assert_close(actual_q, expected_q, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_k, expected_k, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "batch, seq_len, n_heads, head_dim",
    [
        (2, 4, 3, 64),
        (1, 8, 8, 128),
    ],
)
def test_triton_matches_torch_bnsd_layout(batch, seq_len, n_heads, head_dim, dtype):
    """Triton interleaved RoPE works with BNSD layout (unsqueeze_dim=1)."""
    torch.manual_seed(42)
    device = "cuda"

    # BNSD layout: [batch, n_heads, seq_len, head_dim]
    q = torch.randn(batch, n_heads, seq_len, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, n_heads, seq_len, head_dim, dtype=dtype, device=device)
    cos, sin = _make_cos_sin(batch, seq_len, head_dim, dtype, device)

    # unsqueeze_dim=1 for BNSD layout
    expected_q, expected_k = torch.ops.auto_deploy.torch_rope_with_qk_interleaving(
        q, k, cos, sin, 1
    )
    actual_q, actual_k = torch.ops.auto_deploy.triton_rope_with_qk_interleaving(
        q, k, cos, sin, 1
    )

    rtol, atol = _get_tolerances(dtype)
    torch.testing.assert_close(actual_q, expected_q, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_k, expected_k, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "batch, seq_len, n_heads, head_dim",
    [
        (1, 5, 3, 64),
        (2, 7, 4, 128),
        (1, 13, 2, 192),
    ],
)
def test_triton_non_power_of_2_seq_len(batch, seq_len, n_heads, head_dim):
    """Triton kernel handles non-power-of-2 sequence lengths correctly."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    q = torch.randn(batch, seq_len, n_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seq_len, n_heads, head_dim, dtype=dtype, device=device)
    cos, sin = _make_cos_sin(batch, seq_len, head_dim, dtype, device)

    expected_q, expected_k = torch.ops.auto_deploy.torch_rope_with_qk_interleaving(
        q, k, cos, sin, 2
    )
    actual_q, actual_k = torch.ops.auto_deploy.triton_rope_with_qk_interleaving(
        q, k, cos, sin, 2
    )

    rtol, atol = _get_tolerances(dtype)
    torch.testing.assert_close(actual_q, expected_q, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_k, expected_k, rtol=rtol, atol=atol)


def test_triton_large_realistic_shape():
    """Triton kernel works on large realistic shapes (DeepSeek-V3 like)."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    batch, seq_len, n_heads, head_dim = 2, 128, 16, 128
    q = torch.randn(batch, seq_len, n_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seq_len, n_heads, head_dim, dtype=dtype, device=device)
    cos, sin = _make_cos_sin(batch, seq_len, head_dim, dtype, device)

    expected_q, expected_k = torch.ops.auto_deploy.torch_rope_with_qk_interleaving(
        q, k, cos, sin, 2
    )
    actual_q, actual_k = torch.ops.auto_deploy.triton_rope_with_qk_interleaving(
        q, k, cos, sin, 2
    )

    torch.testing.assert_close(actual_q, expected_q, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_k, expected_k, rtol=1e-2, atol=1e-2)


def test_triton_launcher_directly():
    """Test the Triton launcher function directly (bypassing custom op registration)."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float16

    batch, seq_len, n_heads, head_dim = 2, 4, 3, 64
    q = torch.randn(batch, seq_len, n_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(batch, seq_len, n_heads, head_dim, dtype=dtype, device=device)
    cos, sin = _make_cos_sin(batch, seq_len, head_dim, dtype, device)

    expected_q, expected_k = torch.ops.auto_deploy.torch_rope_with_qk_interleaving(
        q, k, cos, sin, 2
    )
    actual_q, actual_k = rope_with_qk_interleaving_fused(q, k, cos, sin, 2)

    torch.testing.assert_close(actual_q, expected_q, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual_k, expected_k, rtol=1e-3, atol=1e-3)


def test_triton_3d_input():
    """Triton kernel handles 3D input [B*S, N, D] correctly."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    total_tokens, n_heads, head_dim = 16, 8, 128
    q = torch.randn(total_tokens, n_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_tokens, n_heads, head_dim, dtype=dtype, device=device)
    # cos/sin for 3D: [total_tokens, head_dim], unsqueeze_dim=1 adds head dim
    inv_freq = 1.0 / (
        10000
        ** (
            torch.arange(0, head_dim // 2, dtype=torch.float32, device=device)
            / (head_dim // 2)
        )
    )
    positions = torch.arange(total_tokens, dtype=torch.float32, device=device)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)
    cos = torch.cat([torch.cos(angles), torch.cos(angles)], dim=-1).to(dtype)
    sin = torch.cat([torch.sin(angles), torch.sin(angles)], dim=-1).to(dtype)

    expected_q, expected_k = torch.ops.auto_deploy.torch_rope_with_qk_interleaving(
        q, k, cos, sin, 1
    )
    actual_q, actual_k = torch.ops.auto_deploy.triton_rope_with_qk_interleaving(
        q, k, cos, sin, 1
    )

    rtol, atol = _get_tolerances(dtype)
    torch.testing.assert_close(actual_q, expected_q, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_k, expected_k, rtol=rtol, atol=atol)
