# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""catalog/spec_decode/dspark_block_tables: the one-launch draft block-table refresh writes, bit for bit, what
DFlashWorker._refresh_ctx_block_tables writes (decoded tables and placeholder counts), eager and in a CUDA graph, with
the per-row kernel and with S10c-B2's (row, chunk) grid."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.spec_decode.dspark_block_tables import (
    enable_parallel_refresh,
    refresh_block_tables,
)
from tensorrt_llm._torch.speculative.dflash import DFlashWorker

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Triton)")


@pytest.fixture(params=[False, True], ids=["per_row", "grid2d"])
def parallel(request):
    enable_parallel_refresh(request.param)
    yield request.param
    enable_parallel_refresh(False)


def _worker(rows, width, pool_idx, divisor, fused):
    worker = object.__new__(DFlashWorker)
    worker._ctx_block_tables = torch.full((rows, width), 77, dtype=torch.int32, device="cuda")
    worker._ctx_block_counts = torch.full((rows,), 77, dtype=torch.int64, device="cuda")
    worker._ctx_pool_idx = pool_idx
    worker._ctx_block_divisor = divisor
    worker._fused_block_refresh = refresh_block_tables if fused else None
    return worker


def _offsets(pools, rows, width, dtype, seed, high=1 << 20):
    g = torch.Generator(device="cuda").manual_seed(seed)
    off = torch.randint(0, high, (pools, rows, 2, width), generator=g, device="cuda").to(dtype)
    # placeholders past each row's allocation (negative), at a random length per row
    lens = torch.randint(0, width + 1, (pools, rows, 1, 1), generator=g, device="cuda")
    cols = torch.arange(width, device="cuda").view(1, 1, 1, -1)
    return torch.where(cols < lens, off, torch.full_like(off, -1))


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("width", [7, 129, 1100, 4500])
@pytest.mark.parametrize("divisor", [1, 2, 10])
@pytest.mark.parametrize("pools,pool_idx", [(1, 0), (3, 2)])
@pytest.mark.parametrize("high", [1 << 20, (1 << 31) - 1])
def test_refresh_is_the_stock_refresh_bit_for_bit(
    parallel, dtype, width, divisor, pools, pool_idx, high
):
    rows = 9
    for seed in range(3):
        src = _offsets(pools, rows, width, dtype, seed, high)
        md = SimpleNamespace(draft_kv_cache_block_offsets=src)
        for num_seqs in (1, 5, rows):
            ref = _worker(rows, width, pool_idx, divisor, fused=False)
            got = _worker(rows, width, pool_idx, divisor, fused=True)
            assert DFlashWorker._refresh_ctx_block_tables(ref, md, num_seqs)
            assert DFlashWorker._refresh_ctx_block_tables(got, md, num_seqs)
            torch.cuda.synchronize()
            assert torch.equal(ref._ctx_block_tables, got._ctx_block_tables), (seed, num_seqs)
            assert torch.equal(ref._ctx_block_counts, got._ctx_block_counts), (seed, num_seqs)
            assert torch.equal(
                src, _offsets(pools, rows, width, dtype, seed, high)
            )  # the metadata is not edited


@pytest.mark.parametrize("width", [64, 3000])
def test_refresh_in_a_cuda_graph(parallel, width):
    rows, num_seqs = 8, 6
    src = _offsets(2, rows, width, torch.int32, 0)
    md = SimpleNamespace(draft_kv_cache_block_offsets=src)
    got = _worker(rows, width, 1, 2, fused=True)
    DFlashWorker._refresh_ctx_block_tables(got, md, num_seqs)  # compile outside capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            DFlashWorker._refresh_ctx_block_tables(got, md, num_seqs)
    torch.cuda.current_stream().wait_stream(s)
    src.copy_(_offsets(2, rows, width, torch.int32, 1))  # a later step's offsets, same buffer
    got._ctx_block_tables.fill_(77)
    g.replay()
    ref = _worker(rows, width, 1, 2, fused=False)
    DFlashWorker._refresh_ctx_block_tables(ref, md, num_seqs)
    torch.cuda.synchronize()
    assert torch.equal(ref._ctx_block_tables, got._ctx_block_tables)
    assert torch.equal(ref._ctx_block_counts, got._ctx_block_counts)


def test_unsupported_layouts_keep_the_stock_path():
    src = _offsets(1, 4, 16, torch.int32, 0)
    tables = torch.zeros(4, 16, dtype=torch.int32, device="cuda")
    counts = torch.zeros(4, dtype=torch.int64, device="cuda")
    assert refresh_block_tables(src, 0, 4, 2, tables, counts)
    assert not refresh_block_tables(src, 0, 4, 2, tables[:, :8], counts)  # width mismatch
    assert not refresh_block_tables(src, 0, 4, 2, tables, counts.int())  # counts not int64
    assert not refresh_block_tables(src.float(), 0, 4, 2, tables, counts)  # not integer offsets
    assert not refresh_block_tables(src, 0, 4, 0, tables, counts)  # no divisor
