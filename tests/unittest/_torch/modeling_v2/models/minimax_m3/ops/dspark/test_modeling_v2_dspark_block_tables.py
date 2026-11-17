# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ops/dspark/dspark_block_tables: the one-launch draft block-table refresh (the family op
minimax_m31_dspark_refresh_block_tables) writes, bit for bit, what the engine's stock decode
(``decode_draft_block_table``) writes for tables whose placeholders follow each row's allocation, eager and in a CUDA
graph, with the per-row and the 2D-grid launch, on every launch path of the op (up to 256 entries a warp per row at
1 / 2 / 4 / 8 entries per lane, 4 rows per CTA; up to 512 two warps per row, 4 rows per CTA; above, the count CTA +
16-B vector / scalar decode CTAs).

``dspark_drafter_inputs.drafter_inputs_and_refresh`` (one launch for the drafter's four step inputs and this refresh)
writes, bit for bit, what ``drafter_inputs`` plus ``refresh_block_tables`` write."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.dspark.dspark_block_tables import (
    enable_parallel_refresh,
    refresh_block_tables,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.dspark.dspark_drafter_inputs import (
    drafter_inputs,
    drafter_inputs_and_refresh,
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
@pytest.mark.parametrize("width", [7, 4500])
@pytest.mark.parametrize("divisor", [10])
@pytest.mark.parametrize("pools,pool_idx", [(3, 2)])
@pytest.mark.parametrize("high", [(1 << 31) - 1])
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
@pytest.mark.parametrize("width", [1, 257, 513, 8196])
@pytest.mark.parametrize("num_seqs", [6])
@pytest.mark.parametrize("padded", [True], ids=["padded_tables"])
@pytest.mark.parametrize("divisor", [2**31 + 5])
def test_cpp_paths_bit_for_bit(dtype, width, num_seqs, padded, divisor):
    """Every launch path (tiny: 4 rows per CTA, num_seqs 6 = two CTAs, every entries-per-lane boundary; two warps per
    row above 256; count CTA with a scalar head / tail; 16-B vector decode at aligned rows, with a width % 4 tail;
    scalar decode at padded rows), a > 2^31 divisor: the written rows equal the integer
    reference, rows past num_seqs keep their sentinel, counts are stored (not accumulated)."""
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


VOCAB, HIDDEN, MASK_ID, SLOTS, MAX_CTX = 1000, 6144, 997, 20, 4096


def _drafter_args(n, block, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    table = torch.randn(VOCAB, HIDDEN, generator=g, device="cuda").bfloat16()
    accepted = torch.randint(0, VOCAB, (n, block), generator=g, device="cuda").int()
    accepted[0, :] = -1
    nacc = torch.randint(1, block + 1, (n,), generator=g, device="cuda").int()
    slots = torch.randperm(SLOTS, generator=g, device="cuda")[:n]
    ctx_len = torch.randint(0, MAX_CTX + 1, (SLOTS,), generator=g, device="cuda").long()
    return (
        accepted,
        nacc,
        slots,
        ctx_len,
        MAX_CTX,
        table,
        table[MASK_ID].contiguous(),
        block,
        block,
    )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("n,block", [(1, 8), (8, 7)])
def test_inputs_and_refresh_is_inputs_plus_refresh(parallel, dtype, n, block):
    width, divisor, pools, pool_idx = 4500, 2, 3, 2
    rows = n + 2
    args = _drafter_args(n, block, seed=width + n)
    off = _offsets(pools, rows, width, dtype, seed=divisor + width)
    want_t, want_c = _buffers(rows, width)
    assert refresh_block_tables(off, pool_idx, n, divisor, want_t, want_c)
    want = drafter_inputs(*args)
    got_t, got_c = _buffers(rows, width)
    got = drafter_inputs_and_refresh(*args, off, pool_idx, n, divisor, got_t, got_c)
    assert got is not None and len(got) == len(want) == 4
    for x, y in zip(got, want):
        assert x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
    assert torch.equal(got_t, want_t) and torch.equal(got_c, want_c)
