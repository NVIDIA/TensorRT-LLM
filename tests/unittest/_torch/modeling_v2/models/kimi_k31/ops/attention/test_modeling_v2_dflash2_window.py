# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 MLA drafter's exact per-query sliding window: the eager decode over its index lists
equals dense attention under the window mask."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.attention.dflash2_window import (
    dflash2_window_rows,
    sparse_mla_rows_eager,
)

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _paged(batch, max_pages, seed=0):
    """A random page table: request b's position page k is pool page ``table[b, k]``."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(batch * max_pages, generator=g)
    return perm.view(batch, max_pages).to(_DEVICE)


def _dense_reference(query, latents, ctx_len, block_size, window, scale, latent_dim):
    """Per-request dense attention with a [query, key] causal-window mask, FP32."""
    batch = latents.shape[0]
    out = []
    for b in range(batch):
        keys = latents[b].float()  # [T, D] positions 0..T-1
        t = torch.arange(keys.shape[0], device=keys.device)
        for j in range(block_size):
            p = int(ctx_len[b]) + j
            mask = (t <= p) & (t >= p - window + 1)
            q = query[b * block_size + j].float()  # [H, D]
            s = (q @ keys.T) * scale
            s = s.masked_fill(~mask, float("-inf"))
            out.append(torch.softmax(s, -1) @ keys[:, :latent_dim])
    return torch.stack(out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("pool_dtype", [torch.float8_e4m3fn])
def test_eager_window_decode_matches_dense_mask(pool_dtype):
    torch.manual_seed(0)
    window, block_size, page_size, heads, latent_dim, rope = 16, 5, 4, 3, 32, 8
    dim = latent_dim + rope
    ctx_len = torch.tensor([2, 15, 40], device=_DEVICE)
    batch = ctx_len.shape[0]
    total = int(ctx_len.max()) + block_size
    max_pages = (total + page_size - 1) // page_size
    table = _paged(batch, max_pages, seed=1)
    latents = torch.randn(batch, max_pages * page_size, dim, device=_DEVICE)
    pool = torch.zeros(batch * max_pages, page_size, dim, device=_DEVICE)
    for b in range(batch):
        pool[table[b]] = latents[b].view(max_pages, page_size, dim)
    pool = pool.to(pool_dtype)
    keys = pool[table.view(-1)].view(batch, -1, dim).float()  # what the pool holds
    query = torch.randn(batch * block_size, heads, dim, device=_DEVICE, dtype=torch.bfloat16)
    rows, _ = dflash2_window_rows(ctx_len, table, block_size, page_size, window, window)
    got = sparse_mla_rows_eager(query, pool.view(-1, dim), rows, 0.13, latent_dim)
    ref = _dense_reference(query, keys, ctx_len, block_size, window, 0.13, latent_dim)
    torch.testing.assert_close(got.float(), ref, atol=2e-2, rtol=2e-2)
