# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the fused MiniMax-M3 MSA paged-cache scatter."""

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.common import write_kv_slots
from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.msa_scatter import (
    fused_write_layer_caches,
)

cuda_required = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _reference_write(k_cache, v_cache, idx_cache, slots, k, v, idx_k):
    num_tokens = int(slots.shape[0])
    num_heads, head_dim = int(k_cache.shape[1]), int(k_cache.shape[3])
    write_kv_slots(k_cache, slots, k.reshape(num_tokens, num_heads, head_dim), layout="HND")
    write_kv_slots(v_cache, slots, v.reshape(num_tokens, num_heads, head_dim), layout="HND")
    if idx_k is not None:
        write_kv_slots(idx_cache, slots, idx_k.reshape(num_tokens, 1, head_dim), layout="HND")


@cuda_required
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("with_idx", [True, False])
def test_fused_scatter_matches_reference(cache_dtype, num_kv_heads, with_idx):
    torch.manual_seed(0)
    device = "cuda"
    num_pages, tokens_per_block, head_dim = 6, 32, 128
    num_tokens = 17
    inner = num_kv_heads * head_dim

    # Paged HND caches carved from a pool with a coalescing axis, so the
    # views are non-contiguous like production get_buffers(...) output.
    pool = torch.zeros(
        num_pages, 2, num_kv_heads, tokens_per_block, head_dim, dtype=cache_dtype, device=device
    )
    k_cache, v_cache = pool[:, 0], pool[:, 1]
    idx_pool = torch.zeros(
        num_pages, 2, 1, tokens_per_block, head_dim, dtype=torch.bfloat16, device=device
    )
    idx_cache = idx_pool[:, 0]

    # Strided sources: rows sliced out of a wider fused-projection tensor.
    qkv = torch.randn(num_tokens, 3 * inner + 64, dtype=torch.bfloat16, device=device)
    k = qkv[:, :inner]
    v = qkv[:, inner : 2 * inner]
    idx_k = qkv[:, 2 * inner : 2 * inner + head_dim] if with_idx else None

    slots = torch.randperm(num_pages * tokens_per_block, device=device)[:num_tokens].to(
        torch.int32
    )

    ref_pool = pool.clone()
    ref_idx_pool = idx_pool.clone()
    _reference_write(
        ref_pool[:, 0], ref_pool[:, 1], ref_idx_pool[:, 0], slots, k, v, idx_k
    )

    wrote = fused_write_layer_caches(
        k_cache, v_cache, idx_cache if with_idx else None, slots, k, v, idx_k
    )
    assert wrote

    torch.testing.assert_close(pool.to(torch.float32), ref_pool.to(torch.float32))
    torch.testing.assert_close(idx_pool, ref_idx_pool)


@cuda_required
def test_fused_scatter_rejects_unsupported_layout():
    device = "cuda"
    pool = torch.zeros(4, 2, 2, 16, 128, dtype=torch.bfloat16, device=device)
    slots = torch.zeros(3, dtype=torch.int32, device=device)
    # Channel-strided source rows cannot be consumed; the caller must fall
    # back to the legacy writes.
    bad = torch.randn(3, 2 * 128 * 2, dtype=torch.bfloat16, device=device)[:, ::2]
    ok = fused_write_layer_caches(pool[:, 0], pool[:, 1], None, slots, bad, bad, None)
    assert not ok


@cuda_required
def test_fused_scatter_int64_offset_no_overflow():
    """Write to a paged cache whose element offset for a high page exceeds
    2**31, forcing the kernel's int64 address arithmetic. With int32 the
    ``page * page_stride`` product would wrap negative and corrupt/OOB. Guarded:
    allocates ~4.3 GB of fp8 cache, so it skips on small GPUs."""
    device = "cuda"
    num_kv_heads, tokens_per_block, head_dim = 1, 128, 128
    page_stride = num_kv_heads * tokens_per_block * head_dim  # elements per page
    # Enough pages that the last page's element offset lands just past 2**31.
    num_pages = (1 << 31) // page_stride + 64
    dtype = torch.float8_e4m3fn  # 1 byte/elem -> ~2.15 GB per cache
    need_bytes = 2 * num_pages * page_stride + (1 << 28)
    free, _ = torch.cuda.mem_get_info()
    if free < need_bytes:
        pytest.skip(f"needs ~{need_bytes / 1024 ** 3:.1f} GB free, "
                    f"have {free / 1024 ** 3:.1f} GB")

    k_cache = torch.zeros(
        num_pages, num_kv_heads, tokens_per_block, head_dim, dtype=dtype, device=device
    )
    v_cache = torch.zeros_like(k_cache)
    high_page = num_pages - 1
    assert high_page * page_stride > (1 << 31)  # int32 arithmetic overflows here
    slot = high_page * tokens_per_block + 7
    slots = torch.tensor([slot], dtype=torch.int32, device=device)

    inner = num_kv_heads * head_dim
    k = torch.randn(1, inner, dtype=torch.bfloat16, device=device)
    v = torch.randn(1, inner, dtype=torch.bfloat16, device=device)
    assert fused_write_layer_caches(k_cache, v_cache, None, slots, k, v, None)

    # The write must land exactly at (high_page, head 0, within 7); an int32
    # overflow would have scattered it to a wrapped offset (or faulted).
    exp_k = k.reshape(1, num_kv_heads, head_dim)[0, 0].to(dtype).to(torch.float32)
    exp_v = v.reshape(1, num_kv_heads, head_dim)[0, 0].to(dtype).to(torch.float32)
    torch.testing.assert_close(k_cache[high_page, 0, 7].to(torch.float32), exp_k)
    torch.testing.assert_close(v_cache[high_page, 0, 7].to(torch.float32), exp_v)


@cuda_required
def test_fused_scatter_boundary_slots():
    """Exercise the extremes of the slot -> (page, within) split: the first
    slot (page 0, within 0), the last valid slot (max page, within tpb-1), and
    the two page-boundary neighbours. Confirms the div/mod addressing at the
    valid-range edges matches the legacy path. (The wrapper is only ever called
    with a live-token slice, so out-of-range / sentinel slots never reach the
    kernel; this pins the corners of the valid range.)"""
    torch.manual_seed(0)
    device = "cuda"
    num_pages, tokens_per_block, head_dim, num_kv_heads = 6, 32, 128, 4
    inner = num_kv_heads * head_dim
    pool = torch.zeros(
        num_pages, 2, num_kv_heads, tokens_per_block, head_dim,
        dtype=torch.bfloat16, device=device,
    )
    k_cache, v_cache = pool[:, 0], pool[:, 1]

    last = num_pages * tokens_per_block - 1
    slots = torch.tensor(
        [0, tokens_per_block - 1, tokens_per_block, last],
        dtype=torch.int32, device=device,
    )
    num_tokens = int(slots.shape[0])
    qkv = torch.randn(num_tokens, 3 * inner, dtype=torch.bfloat16, device=device)
    k, v = qkv[:, :inner], qkv[:, inner:2 * inner]

    ref_pool = pool.clone()
    _reference_write(ref_pool[:, 0], ref_pool[:, 1], None, slots, k, v, None)
    assert fused_write_layer_caches(k_cache, v_cache, None, slots, k, v, None)
    torch.testing.assert_close(pool.to(torch.float32), ref_pool.to(torch.float32))
