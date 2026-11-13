# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ops/dspark/dspark_drafter_inputs.drafter_inputs_and_refresh: one launch writes, bit for bit, the
four step inputs of drafter_inputs and the block tables / placeholder counts of dspark_block_tables.refresh_block_tables
(with the per-row kernel and with the 2D grid), over table widths below and above one refresh chunk, int32 / int64
offsets with placeholders, eagerly and replayed from a CUDA graph; it declines, writing nothing, when the refresh rows
are not the input rows or a layout is not one it takes."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark.dspark_block_tables import (
    enable_parallel_refresh,
    refresh_block_tables,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.dspark.dspark_drafter_inputs import (
    drafter_inputs,
    drafter_inputs_and_refresh,
)
from tensorrt_llm._utils import get_sm_version

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="the DSpark glue ops run on sm_107 (Rubin) only",
)

VOCAB, HIDDEN, MASK_ID, SLOTS, MAX_CTX = 1000, 6144, 997, 20, 4096


@pytest.fixture(params=[False, True], ids=["per_row", "grid2d"])
def parallel(request):
    enable_parallel_refresh(request.param)
    yield request.param
    enable_parallel_refresh(False)


def _inputs(n, block, seed):
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


def _offsets(pools, rows, width, dtype, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    off = torch.randint(0, 1 << 20, (pools, rows, 2, width), generator=g, device="cuda")
    off[..., width // 2 :] = -1  # placeholders past each row's pages
    off[:, 0, 0, 0] = -7
    return off.to(dtype)


def _refresh_buffers(rows, width):
    return (
        torch.full((rows, width), 77, dtype=torch.int32, device="cuda"),
        torch.full((rows,), 77, dtype=torch.int64, device="cuda"),
    )


def _equal(a, b):
    assert len(a) == len(b) == 4, f"expected 4 drafter inputs, got {len(a)} and {len(b)}"
    return all(
        x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y) for x, y in zip(a, b)
    )


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("width", [7, 129, 1100, 4500])
@pytest.mark.parametrize("divisor", [1, 2, 10])
@pytest.mark.parametrize("pools,pool_idx", [(1, 0), (3, 2)])
@pytest.mark.parametrize("n,block", [(1, 8), (3, 8), (8, 7)])
def test_combined_is_inputs_plus_refresh(
    parallel, dtype, width, divisor, pools, pool_idx, n, block
):
    rows = n + 2
    args = _inputs(n, block, seed=width + n)
    off = _offsets(pools, rows, width, dtype, seed=divisor + width)
    want_t, want_c = _refresh_buffers(rows, width)
    assert refresh_block_tables(off, pool_idx, n, divisor, want_t, want_c)
    want = drafter_inputs(*args)
    got_t, got_c = _refresh_buffers(rows, width)
    got = drafter_inputs_and_refresh(*args, off, pool_idx, n, divisor, got_t, got_c)
    assert got is not None
    assert _equal(got, want)
    assert torch.equal(got_t, want_t) and torch.equal(got_c, want_c)


def test_combined_in_a_cuda_graph():
    n, block, width, divisor = 1, 8, 3000, 2
    args = _inputs(n, block, seed=3)
    off = _offsets(1, n + 1, width, torch.int32, seed=4)
    got_t, got_c = _refresh_buffers(n + 1, width)
    drafter_inputs_and_refresh(*args, off, 0, n, divisor, got_t, got_c)  # compile outside capture
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g):
            out = drafter_inputs_and_refresh(*args, off, 0, n, divisor, got_t, got_c)
    torch.cuda.current_stream().wait_stream(s)
    for rep in range(4):
        fresh = _inputs(n, block, seed=50 + rep)
        for dst, src in zip(args[:4], fresh[:4]):
            dst.copy_(src)
        off.copy_(_offsets(1, n + 1, width, torch.int32, seed=60 + rep))
        g.replay()
        torch.cuda.synchronize()
        want_t, want_c = _refresh_buffers(n + 1, width)
        refresh_block_tables(off, 0, n, divisor, want_t, want_c)
        assert _equal(out, drafter_inputs(*args)), f"replay {rep}"
        assert torch.equal(got_t[:n], want_t[:n]) and torch.equal(got_c[:n], want_c[:n]), (
            f"replay {rep}"
        )


def test_declines_without_writing():
    n, block, width = 3, 8, 64
    args = _inputs(n, block, seed=9)
    off = _offsets(1, n + 1, width, torch.int32, seed=10)
    for case in ("rows", "divisor", "tables_dtype"):
        t, c = _refresh_buffers(n + 1, width)
        num_seqs, divisor = (n + 1, 2) if case == "rows" else (n, 0 if case == "divisor" else 2)
        if case == "tables_dtype":
            t = t.long()
        assert drafter_inputs_and_refresh(*args, off, 0, num_seqs, divisor, t, c) is None, case
        assert (t == 77).all() and (c == 77).all(), case
