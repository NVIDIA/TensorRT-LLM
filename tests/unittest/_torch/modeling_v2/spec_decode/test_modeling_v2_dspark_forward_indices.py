# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""catalog/spec_decode/dspark_forward_indices: the paged dflash_forward's index prelude in one launch is bit for bit
the stock lines of DFlashForCausalLM.dflash_forward (cache_seqlens, cache_batch_idx, seq_lens_after, kv_indptr,
kv_last_page_len, append_positions, query_positions_flat), eager and in a CUDA graph."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.spec_decode.dspark_forward_indices import (
    forward_indices,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Triton)")


def _stock(num_ctx, batch_idx, query_positions, block_size, pages_per_slot, page_size):
    """The stock lines of dflash_forward's paged branch, verbatim."""
    B = query_positions.shape[0]
    cache_seqlens_i32 = num_ctx[:B].to(torch.int32)
    cache_batch_idx_i32 = batch_idx.to(torch.int32)
    seq_lens_after = cache_seqlens_i32 + block_size
    kv_indptr = torch.arange(
        0, (B + 1) * pages_per_slot, pages_per_slot, dtype=torch.int32, device="cuda"
    )
    kv_last_page_len = ((seq_lens_after - 1) % page_size) + 1
    block_offsets = torch.arange(block_size, dtype=torch.int32, device="cuda")
    append_positions = (cache_seqlens_i32.view(-1, 1) + block_offsets).reshape(-1).contiguous()
    query_positions_flat_i32 = query_positions.reshape(-1).to(torch.int32)
    return (
        cache_seqlens_i32,
        cache_batch_idx_i32,
        seq_lens_after,
        kv_indptr,
        kv_last_page_len,
        append_positions,
        query_positions_flat_i32,
    )


def _case(B, block, dtype, seed, strided):
    g = torch.Generator(device="cuda").manual_seed(seed)
    num_ctx = torch.randint(0, 1 << 20, (B,), generator=g, device="cuda").to(dtype)
    num_ctx[0] = 0  # an empty context: seq_lens_after = block
    batch_idx = torch.randperm(B + 5, generator=g, device="cuda")[:B].to(dtype)
    if strided:
        q = torch.randint(0, 1 << 21, (block, B), generator=g, device="cuda").to(dtype).t()
    else:
        q = torch.randint(0, 1 << 21, (B, block), generator=g, device="cuda").to(dtype)
    return num_ctx, batch_idx, q


def _equal(a, b):
    return all(
        x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y) for x, y in zip(a, b)
    )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("block", [3, 4, 7, 8])
@pytest.mark.parametrize("page_size,pages_per_slot", [(64, 66), (32, 2049), (16, 5)])
@pytest.mark.parametrize("strided", [False, True])
def test_indices_are_the_stock_lines_bit_for_bit(dtype, block, page_size, pages_per_slot, strided):
    for B in (1, 2, 3, 8, 16, 33):
        for seed in range(3):
            num_ctx, batch_idx, q = _case(B, block, dtype, seed, strided)
            ref = _stock(num_ctx, batch_idx, q, block, pages_per_slot, page_size)
            got = forward_indices(num_ctx, batch_idx, q, block, pages_per_slot, page_size)
            assert got is not None
            torch.cuda.synchronize()
            assert _equal(ref, got), (B, seed)


def test_indices_in_a_cuda_graph():
    num_ctx, batch_idx, q = _case(8, 4, torch.int64, 0, False)
    forward_indices(num_ctx, batch_idx, q, 4, 66, 64)  # compile outside capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            got = forward_indices(num_ctx, batch_idx, q, 4, 66, 64)
    torch.cuda.current_stream().wait_stream(s)
    n2, b2, q2 = _case(8, 4, torch.int64, 1, False)
    num_ctx.copy_(n2), batch_idx.copy_(b2), q.copy_(q2)
    g.replay()
    ref = _stock(num_ctx, batch_idx, q, 4, 66, 64)
    torch.cuda.synchronize()
    assert _equal(ref, got)


def test_unsupported_layouts_keep_the_stock_path():
    num_ctx, batch_idx, q = _case(2, 4, torch.int64, 0, False)
    assert forward_indices(num_ctx, batch_idx, q, 4, 66, 64) is not None
    assert forward_indices(num_ctx.float(), batch_idx, q, 4, 66, 64) is None
    assert forward_indices(num_ctx, batch_idx, q[:, :3], 4, 66, 64) is None
    assert forward_indices(num_ctx, batch_idx[:1], q, 4, 66, 64) is None
