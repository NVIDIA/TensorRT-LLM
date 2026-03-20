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

"""Correctness tests for the Triton RoPE kernel with explicit cos/sin tensors.

Compares triton_rope_with_explicit_cos_sin against torch_rope_with_explicit_cos_sin
across multiple dtypes, shapes, layouts, and edge cases.
"""

import pytest
import torch

# Trigger registration of all custom ops
import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401


def _precompute_cos_sin(seq_len, head_dim, dtype=torch.float32, device="cuda"):
    """Precompute cos/sin tensors in the HF style (duplicated halves)."""
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # [S, D/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [S, D]
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos, sin


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "batch,num_heads,seq_len,head_dim,unsqueeze_dim",
    [
        (1, 1, 1, 64, 1),  # minimal shape
        (2, 8, 16, 64, 1),  # BNSD layout (unsqueeze_dim=1)
        (2, 8, 16, 64, 2),  # BSND layout (unsqueeze_dim=2)
        (4, 32, 8, 128, 1),  # larger heads
        (1, 4, 7, 64, 1),  # non-power-of-2 seq_len
        (2, 3, 5, 128, 2),  # non-power-of-2 seq_len and odd num_heads
    ],
)
def test_triton_rope_matches_torch(dtype, batch, num_heads, seq_len, head_dim, unsqueeze_dim):
    """Triton kernel output matches torch reference within tolerance."""
    torch.manual_seed(42)
    device = "cuda"

    if unsqueeze_dim == 1:
        # BNSD layout: [B, N, S, D]
        q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    else:
        # BSND layout: [B, S, N, D]
        q = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch, seq_len, num_heads, head_dim, device=device, dtype=dtype)

    cos, sin = _precompute_cos_sin(seq_len, head_dim, dtype=dtype, device=device)
    # Expand to [B, S, D]
    cos = cos.unsqueeze(0).expand(batch, -1, -1)
    sin = sin.unsqueeze(0).expand(batch, -1, -1)

    q_ref, k_ref = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(
        q, k, cos, sin, unsqueeze_dim
    )
    q_tri, k_tri = torch.ops.auto_deploy.triton_rope_with_explicit_cos_sin(
        q, k, cos, sin, unsqueeze_dim
    )

    # bf16 needs looser tolerance: Triton kernel computes in float32 while
    # the torch reference computes in the native dtype, and bf16 has only
    # ~7 mantissa bits (machine eps ~0.008).
    rtol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    atol = 1e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(q_tri, q_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(k_tri, k_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_triton_rope_different_head_dims(head_dim):
    """Triton kernel works correctly across different head dimensions."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    batch, num_heads, seq_len = 2, 8, 16

    q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    cos, sin = _precompute_cos_sin(seq_len, head_dim, dtype=dtype, device=device)
    cos = cos.unsqueeze(0).expand(batch, -1, -1)
    sin = sin.unsqueeze(0).expand(batch, -1, -1)

    q_ref, k_ref = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(q, k, cos, sin, 1)
    q_tri, k_tri = torch.ops.auto_deploy.triton_rope_with_explicit_cos_sin(q, k, cos, sin, 1)

    torch.testing.assert_close(q_tri, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_tri, k_ref, rtol=1e-2, atol=1e-2)


def test_triton_rope_different_qk_heads():
    """Triton kernel works when q and k have different number of heads (GQA)."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    batch, seq_len, head_dim = 2, 16, 64
    num_q_heads = 32
    num_kv_heads = 8

    q = torch.randn(batch, num_q_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, num_kv_heads, seq_len, head_dim, device=device, dtype=dtype)

    cos, sin = _precompute_cos_sin(seq_len, head_dim, dtype=dtype, device=device)
    cos = cos.unsqueeze(0).expand(batch, -1, -1)
    sin = sin.unsqueeze(0).expand(batch, -1, -1)

    q_ref, k_ref = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(q, k, cos, sin, 1)
    q_tri, k_tri = torch.ops.auto_deploy.triton_rope_with_explicit_cos_sin(q, k, cos, sin, 1)

    torch.testing.assert_close(q_tri, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k_tri, k_ref, rtol=1e-2, atol=1e-2)


def test_triton_rope_large_realistic():
    """Triton kernel works on large realistic shapes (LLaMA-style)."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16
    batch, num_heads, seq_len, head_dim = 1, 32, 2048, 128

    q = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, num_heads, seq_len, head_dim, device=device, dtype=dtype)

    cos, sin = _precompute_cos_sin(seq_len, head_dim, dtype=dtype, device=device)
    cos = cos.unsqueeze(0).expand(batch, -1, -1)
    sin = sin.unsqueeze(0).expand(batch, -1, -1)

    q_ref, k_ref = torch.ops.auto_deploy.torch_rope_with_explicit_cos_sin(q, k, cos, sin, 1)
    q_tri, k_tri = torch.ops.auto_deploy.triton_rope_with_explicit_cos_sin(q, k, cos, sin, 1)

    torch.testing.assert_close(q_tri, q_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(k_tri, k_ref, rtol=2e-2, atol=2e-2)
