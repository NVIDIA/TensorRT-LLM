# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ops/dspark/dspark_block_tables: the one-launch draft block-table refresh (the family op
minimax_m31_dspark_refresh_block_tables) writes, bit for bit, what the engine's stock decode
(``decode_draft_block_table``) writes for tables whose placeholders follow each row's allocation, eager and in a CUDA
graph, with the per-row and the 2D-grid launch, on every launch path of the op (up to 256 entries a warp per row at
1 / 2 / 4 / 8 entries per lane, 4 rows per CTA; up to 512 two warps per row, 4 rows per CTA; above, the count CTA +
16-B vector / scalar decode CTAs)."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark import (
    dspark_block_tables as bt,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark.dspark_block_tables import (
    enable_parallel_refresh,
    refresh_block_tables,
)
from tensorrt_llm._torch.speculative.spec_tail_kernels import decode_draft_block_table
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="the DSpark block-table refresh op runs on sm_107 (Rubin) only",
)


@pytest.fixture(params=[False, True], ids=["per_row", "grid2d"])
def parallel(request):
    enable_parallel_refresh(request.param)
    yield request.param
    enable_parallel_refresh(False)


def _buffers(rows, width):
    return (
        torch.full((rows, width), 77, dtype=torch.int32, device="cuda"),
        torch.full((rows,), 77, dtype=torch.int64, device="cuda"),
    )


def _stock(src, pool_idx, num_seqs, divisor, tables, counts):
    """The engine's refresh: field 0 of the pool's rows, decoded by ``decode_draft_block_table``."""
    decode_draft_block_table(src[pool_idx, :num_seqs, 0], divisor, tables, counts)


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
        for num_seqs in (1, 5, rows):
            ref_tables, ref_counts = _buffers(rows, width)
            got_tables, got_counts = _buffers(rows, width)
            _stock(src, pool_idx, num_seqs, divisor, ref_tables, ref_counts)
            assert refresh_block_tables(src, pool_idx, num_seqs, divisor, got_tables, got_counts)
            torch.cuda.synchronize()
            assert torch.equal(ref_tables[:num_seqs], got_tables[:num_seqs]), (seed, num_seqs)
            assert torch.equal(ref_counts[:num_seqs], got_counts[:num_seqs]), (seed, num_seqs)
            # the offsets are not edited
            assert torch.equal(src, _offsets(pools, rows, width, dtype, seed, high))


@pytest.mark.parametrize("width", [64, 3000])
def test_refresh_in_a_cuda_graph(parallel, width):
    rows, num_seqs = 8, 6
    src = _offsets(2, rows, width, torch.int32, 0)
    got_tables, got_counts = _buffers(rows, width)
    assert refresh_block_tables(
        src, 1, num_seqs, 2, got_tables, got_counts
    )  # compile outside capture
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            refresh_block_tables(src, 1, num_seqs, 2, got_tables, got_counts)
    torch.cuda.current_stream().wait_stream(s)
    src.copy_(_offsets(2, rows, width, torch.int32, 1))  # a later step's offsets, same buffer
    got_tables.fill_(77)
    g.replay()
    ref_tables, ref_counts = _buffers(rows, width)
    _stock(src, 1, num_seqs, 2, ref_tables, ref_counts)
    torch.cuda.synchronize()
    assert torch.equal(ref_tables[:num_seqs], got_tables[:num_seqs])
    assert torch.equal(ref_counts[:num_seqs], got_counts[:num_seqs])


def test_unsupported_layouts_keep_the_stock_path():
    src = _offsets(1, 4, 16, torch.int32, 0)
    tables = torch.zeros(4, 16, dtype=torch.int32, device="cuda")
    counts = torch.zeros(4, dtype=torch.int64, device="cuda")
    assert refresh_block_tables(src, 0, 4, 2, tables, counts)
    assert not refresh_block_tables(src, 0, 4, 2, tables[:, :8], counts)  # width mismatch
    assert not refresh_block_tables(src, 0, 4, 2, tables, counts.int())  # counts not int64
    assert not refresh_block_tables(src.float(), 0, 4, 2, tables, counts)  # not integer offsets
    assert not refresh_block_tables(src, 0, 4, 0, tables, counts)  # no divisor


def test_cpp_op_no_triton():
    """The module carries no Triton kernel and the refresh is the C++ op."""
    import pathlib

    src = pathlib.Path(bt.__file__).read_text()
    assert "import triton" not in src and "triton.jit" not in src
    assert hasattr(family_ops(), "minimax_m31_dspark_refresh_block_tables")


def _extreme(pools, rows, width, dtype, seed):
    """Values over the dtype's whole range (int64: past int32, so the int32 store truncates), with zeros, -1
    placeholders and the maximum mixed in."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    lo, hi = (-(2**31), 2**31 - 1) if dtype == torch.int32 else (-(2**62), 2**62)
    v = torch.randint(
        lo, hi, (pools, rows, 2, width), generator=g, device="cuda", dtype=torch.int64
    )
    v.view(-1)[::7] = 0
    v.view(-1)[1::11] = -1
    v.view(-1)[2::13] = hi
    return v.to(dtype)


def _reference(src, divisor):
    v = src.to(torch.int64)
    return (v.clamp(min=0) // divisor).to(torch.int32), (v >= 0).sum(-1)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(
    "width", [1, 7, 32, 33, 48, 64, 65, 128, 129, 257, 512, 513, 1030, 2051, 8196]
)
@pytest.mark.parametrize("num_seqs", [1, 2, 6])
@pytest.mark.parametrize("padded", [False, True], ids=["dense", "padded_tables"])
@pytest.mark.parametrize("divisor", [32, 3, 2**31 + 5])
def test_cpp_paths_bit_for_bit(dtype, width, num_seqs, padded, divisor):
    """Every launch path (tiny: 4 rows per CTA, num_seqs 6 = two CTAs, every entries-per-lane boundary; two warps per
    row above 256; count CTA with a scalar head / tail; 16-B vector decode at aligned rows, with a width % 4 tail;
    scalar decode at odd / padded rows), power-of-two / odd / > 2^31 divisors: the written rows equal the integer
    reference, rows past num_seqs keep their sentinel, counts are stored (not accumulated)."""
    _check_cpp(dtype, width, num_seqs, padded, divisor)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("width", [257, 384, 511, 512])
@pytest.mark.parametrize("num_seqs", [1, 2, 4, 5, 6])
@pytest.mark.parametrize("padded", [False, True], ids=["dense", "padded_tables"])
@pytest.mark.parametrize("divisor", [32, 3])
def test_cpp_two_warps_per_row(dtype, width, num_seqs, padded, divisor):
    """256 < width <= 512: two warps per row, 4 rows per CTA; num_seqs 5 / 6 = a partial second CTA (its idle warps
    still reach the barrier), the second warp's count added through shared memory."""
    _check_cpp(dtype, width, num_seqs, padded, divisor)


def _check_cpp(dtype, width, num_seqs, padded, divisor):
    rows_cap, pool_idx = 7, 1
    off = _extreme(3, rows_cap, width, dtype, width * 31 + num_seqs)
    buf = torch.full(
        (rows_cap, width + (13 if padded else 0)), 0x5A5A5A5A, dtype=torch.int32, device="cuda"
    )
    tables = buf[:, :width]
    counts = torch.full((rows_cap,), -7, dtype=torch.int64, device="cuda")
    assert refresh_block_tables(off, pool_idx, num_seqs, divisor, tables, counts)
    want_t, want_c = _reference(off[pool_idx, :num_seqs, 0], divisor)
    torch.cuda.synchronize()
    assert torch.equal(tables[:num_seqs], want_t)
    assert torch.equal(counts[:num_seqs], want_c)
    assert bool((buf[num_seqs:] == 0x5A5A5A5A).all()) and bool((counts[num_seqs:] == -7).all())
    if padded:
        assert bool((buf[:, width:] == 0x5A5A5A5A).all())


def test_cpp_layouts_the_op_declines_keep_the_stock_path():
    src = _offsets(1, 4, 16, torch.int32, 0)
    tables = torch.zeros(4, 16, dtype=torch.int32, device="cuda")
    counts = torch.zeros(8, dtype=torch.int64, device="cuda")
    assert not refresh_block_tables(src, 0, 4, 2, tables, counts[::2])  # counts not contiguous
    assert not refresh_block_tables(
        src, 0, 4, 2, tables[:3], counts
    )  # fewer table rows than num_seqs
    assert refresh_block_tables(src, 0, 0, 2, tables, counts)  # nothing to refresh
