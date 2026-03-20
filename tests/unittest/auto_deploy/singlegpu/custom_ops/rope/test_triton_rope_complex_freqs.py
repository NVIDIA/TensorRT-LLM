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

"""Correctness tests for the Triton complex-multiplication RoPE kernel.

Validates the triton_rope_with_complex_freqs custom op against the
torch_rope_with_complex_freqs reference implementation.
"""

import pytest
import torch

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401  (triggers op registration)


def _precompute_freqs_cis(seq_len: int, head_dim: int, device: str = "cuda") -> torch.Tensor:
    """Compute complex frequency tensor for RoPE.

    Returns a complex tensor of shape (seq_len, head_dim // 2).
    """
    inv_freq = 1.0 / (
        10000
        ** (torch.arange(0, head_dim // 2, dtype=torch.float32, device=device) / (head_dim // 2))
    )
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0)  # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "batch,seq_len,n_heads,n_kv_heads,head_dim",
    [
        (1, 1, 4, 4, 64),  # single token, single batch
        (2, 4, 8, 8, 64),  # small, equal q/k heads
        (2, 4, 8, 2, 64),  # GQA (fewer kv heads)
        (4, 16, 8, 8, 128),  # medium
        (1, 32, 4, 4, 256),  # large head_dim
    ],
)
def test_triton_rope_complex_matches_torch_bsnd(
    dtype, batch, seq_len, n_heads, n_kv_heads, head_dim
):
    """Triton complex RoPE output matches torch reference in BSND layout."""
    torch.manual_seed(42)
    device = "cuda"

    xq = torch.randn(batch, seq_len, n_heads, head_dim, dtype=dtype, device=device)
    xk = torch.randn(batch, seq_len, n_kv_heads, head_dim, dtype=dtype, device=device)
    freqs_cis = _precompute_freqs_cis(seq_len, head_dim, device=device)
    freqs_cis = freqs_cis.unsqueeze(0).expand(batch, -1, -1)  # (B, S, D//2)

    expected_q, expected_k = torch.ops.auto_deploy.torch_rope_with_complex_freqs(
        xq, xk, freqs_cis, 2
    )
    actual_q, actual_k = torch.ops.auto_deploy.triton_rope_with_complex_freqs(xq, xk, freqs_cis, 2)

    torch.testing.assert_close(actual_q, expected_q, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual_k, expected_k, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "batch,seq_len,n_heads,n_kv_heads,head_dim",
    [
        (2, 4, 8, 8, 64),
        (2, 4, 8, 2, 128),
        (4, 16, 8, 8, 256),
    ],
)
def test_triton_rope_complex_matches_torch_bnsd(
    dtype, batch, seq_len, n_heads, n_kv_heads, head_dim
):
    """Triton complex RoPE output matches torch reference in BNSD layout."""
    torch.manual_seed(42)
    device = "cuda"

    xq = torch.randn(batch, n_heads, seq_len, head_dim, dtype=dtype, device=device)
    xk = torch.randn(batch, n_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    freqs_cis = _precompute_freqs_cis(seq_len, head_dim, device=device)
    freqs_cis = freqs_cis.unsqueeze(0).expand(batch, -1, -1)  # (B, S, D//2)

    expected_q, expected_k = torch.ops.auto_deploy.torch_rope_with_complex_freqs(
        xq, xk, freqs_cis, 1
    )
    actual_q, actual_k = torch.ops.auto_deploy.triton_rope_with_complex_freqs(xq, xk, freqs_cis, 1)

    # bf16 has lower mantissa precision than fp16; use relaxed tolerances for large head_dim
    rtol = 2e-2 if dtype == torch.bfloat16 else 1e-3
    atol = 2e-2 if dtype == torch.bfloat16 else 1e-3
    torch.testing.assert_close(actual_q, expected_q, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual_k, expected_k, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "batch,seq_len,n_heads,head_dim",
    [
        (1, 7, 4, 64),  # non-power-of-2 seq_len
        (3, 4, 5, 96),  # non-power-of-2 head_dim (D//2=48)
        (2, 13, 3, 64),  # non-power-of-2 seq_len
    ],
)
def test_triton_rope_complex_non_power_of_2(batch, seq_len, n_heads, head_dim):
    """Triton kernel handles non-power-of-2 dimensions correctly."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    xq = torch.randn(batch, seq_len, n_heads, head_dim, dtype=dtype, device=device)
    xk = torch.randn(batch, seq_len, n_heads, head_dim, dtype=dtype, device=device)
    freqs_cis = _precompute_freqs_cis(seq_len, head_dim, device=device)
    freqs_cis = freqs_cis.unsqueeze(0).expand(batch, -1, -1)

    expected_q, expected_k = torch.ops.auto_deploy.torch_rope_with_complex_freqs(
        xq, xk, freqs_cis, 2
    )
    actual_q, actual_k = torch.ops.auto_deploy.triton_rope_with_complex_freqs(xq, xk, freqs_cis, 2)

    torch.testing.assert_close(actual_q, expected_q, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual_k, expected_k, rtol=1e-3, atol=1e-3)


def test_triton_rope_complex_large():
    """Triton kernel works on large realistic shapes."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.bfloat16

    batch, seq_len, n_heads, n_kv_heads, head_dim = 2, 128, 32, 8, 128

    xq = torch.randn(batch, seq_len, n_heads, head_dim, dtype=dtype, device=device)
    xk = torch.randn(batch, seq_len, n_kv_heads, head_dim, dtype=dtype, device=device)
    freqs_cis = _precompute_freqs_cis(seq_len, head_dim, device=device)
    freqs_cis = freqs_cis.unsqueeze(0).expand(batch, -1, -1)

    expected_q, expected_k = torch.ops.auto_deploy.torch_rope_with_complex_freqs(
        xq, xk, freqs_cis, 2
    )
    actual_q, actual_k = torch.ops.auto_deploy.triton_rope_with_complex_freqs(xq, xk, freqs_cis, 2)

    torch.testing.assert_close(actual_q, expected_q, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(actual_k, expected_k, rtol=1e-2, atol=1e-2)


def test_triton_rope_complex_float32_input():
    """Triton kernel handles float32 input correctly."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32

    batch, seq_len, n_heads, head_dim = 2, 4, 4, 64

    xq = torch.randn(batch, seq_len, n_heads, head_dim, dtype=dtype, device=device)
    xk = torch.randn(batch, seq_len, n_heads, head_dim, dtype=dtype, device=device)
    freqs_cis = _precompute_freqs_cis(seq_len, head_dim, device=device)
    freqs_cis = freqs_cis.unsqueeze(0).expand(batch, -1, -1)

    expected_q, expected_k = torch.ops.auto_deploy.torch_rope_with_complex_freqs(
        xq, xk, freqs_cis, 2
    )
    actual_q, actual_k = torch.ops.auto_deploy.triton_rope_with_complex_freqs(xq, xk, freqs_cis, 2)

    torch.testing.assert_close(actual_q, expected_q, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual_k, expected_k, rtol=1e-5, atol=1e-5)


def test_triton_rope_complex_single_head_single_batch():
    """Edge case: single head, single batch, single sequence position."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float16

    xq = torch.randn(1, 1, 1, 64, dtype=dtype, device=device)
    xk = torch.randn(1, 1, 1, 64, dtype=dtype, device=device)
    freqs_cis = _precompute_freqs_cis(1, 64, device=device)
    freqs_cis = freqs_cis.unsqueeze(0)  # (1, 1, 32)

    expected_q, expected_k = torch.ops.auto_deploy.torch_rope_with_complex_freqs(
        xq, xk, freqs_cis, 2
    )
    actual_q, actual_k = torch.ops.auto_deploy.triton_rope_with_complex_freqs(xq, xk, freqs_cis, 2)

    torch.testing.assert_close(actual_q, expected_q, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(actual_k, expected_k, rtol=1e-3, atol=1e-3)
