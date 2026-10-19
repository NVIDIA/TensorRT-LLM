# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLA context K packing: the Triton kernel is bitwise the cat it replaces."""

import pytest
import torch

from tensorrt_llm._torch.attention.mla import _pack_context_k


def _reference(kv, k_pe, num_heads, nope_dim, rope_dim):
    k_nope = kv[:, : num_heads * nope_dim].view(-1, num_heads, nope_dim)
    k_pe = k_pe.reshape(-1, 1, rope_dim).expand(-1, num_heads, -1)
    return torch.cat((k_nope, k_pe), dim=-1).reshape(-1, num_heads * (nope_dim + rope_dim))


def _inputs(num_tokens, num_heads, nope_dim, rope_dim, v_dim, dtype, device):
    kv = torch.randn(num_tokens, num_heads * (nope_dim + v_dim), dtype=dtype, device=device)
    # k_pe as the MLA context path sees it: a column slice of the latent cache rows.
    latent = torch.randn(num_tokens, 512 + rope_dim, dtype=dtype, device=device)
    k_pe = latent[:, 512:]
    return kv, k_pe


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_heads", [1, 16, 128])
@pytest.mark.parametrize("num_tokens", [0, 1, 63, 64, 1000, 8191])
def test_pack_context_k_matches_cat(num_tokens: int, num_heads: int, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    nope_dim, rope_dim, v_dim = 128, 64, 128
    kv, k_pe = _inputs(num_tokens, num_heads, nope_dim, rope_dim, v_dim, dtype, "cuda")
    k = _pack_context_k(kv, k_pe, num_heads, nope_dim, rope_dim)
    assert k.shape == (num_tokens, num_heads * (nope_dim + rope_dim))
    assert k.is_contiguous()
    assert torch.equal(k, _reference(kv, k_pe, num_heads, nope_dim, rope_dim))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_pack_context_k_contiguous_k_pe() -> None:
    kv = torch.randn(300, 16 * 256, dtype=torch.bfloat16, device="cuda")
    k_pe = torch.randn(300, 64, dtype=torch.bfloat16, device="cuda")
    k = _pack_context_k(kv, k_pe, 16, 128, 64)
    assert torch.equal(k, _reference(kv, k_pe, 16, 128, 64))


@pytest.mark.parametrize(
    "nope_dim,rope_dim,device",
    [(128, 64, "cpu"), (96, 64, "cuda"), (128, 48, "cuda")],
)
def test_pack_context_k_fallback(nope_dim: int, rope_dim: int, device: str) -> None:
    """CPU tensors and non-power-of-two head dims take the cat fallback."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    kv, k_pe = _inputs(70, 4, nope_dim, rope_dim, 128, torch.bfloat16, device)
    k = _pack_context_k(kv, k_pe, 4, nope_dim, rope_dim)
    assert torch.equal(k, _reference(kv, k_pe, 4, nope_dim, rope_dim))
