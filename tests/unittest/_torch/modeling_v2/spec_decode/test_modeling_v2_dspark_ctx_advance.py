# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""catalog/spec_decode/dspark_ctx_advance: the one-launch context advance + read view leaves the context lengths and
the read block table exactly as DFlashWorker._advance_ctx_len and the generation branch's read-view lines
(dflash_allocated_ctx_limit, dflash_rebased_ctx_view) do, and returns the same advertised lengths; eager and in a CUDA
graph."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.spec_decode.dspark_ctx_advance import (
    advance_ctx_and_read_view,
)
from tensorrt_llm._torch.speculative.dflash import (
    DFlashWorker,
    dflash_allocated_ctx_limit,
    dflash_rebased_ctx_view,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Triton)")

PAGE, BLOCK = 64, 3


def _state(n, width, max_ctx, seed, dummy_rows):
    g = torch.Generator(device="cuda").manual_seed(seed)
    num_slots = n + 4
    dummy = num_slots - 1
    slots = torch.randperm(num_slots - 1, generator=g, device="cuda")[:n]
    for r in range(dummy_rows):
        slots[-1 - r] = dummy
    ctx_len = torch.randint(0, max_ctx + 1, (num_slots,), generator=g, device="cuda").long()
    ctx_len[slots[0]] = max_ctx - 1  # runs into the max_ctx clamp
    ctx_len[dummy] = 0
    nacc = torch.randint(1, 5, (n,), generator=g, device="cuda").long()
    rows_total = n + 2
    counts = torch.randint(0, width + 1, (rows_total,), generator=g, device="cuda").long()
    base = torch.randint(0, 6, (num_slots,), generator=g, device="cuda").long()
    base[slots[1 % n]] = (1 << 20) // PAGE  # a context shorter than its base
    tables = torch.randint(0, 1 << 16, (rows_total, width), generator=g, device="cuda").int()
    read = torch.full_like(tables, -9)
    rows = torch.arange(1, n + 1, device="cuda")
    return dict(
        slots=slots,
        rows=rows,
        nacc=nacc,
        ctx_len=ctx_len,
        max_ctx=max_ctx,
        dummy=dummy,
        counts=counts,
        base=base,
        tables=tables,
        read=read,
    )


def _stock(st):
    """_advance_ctx_len + the generation branch's read-view lines, verbatim."""
    w = object.__new__(DFlashWorker)
    w._ctx_len, w._max_ctx, w._dummy_slot = st["ctx_len"], st["max_ctx"], st["dummy"]
    DFlashWorker._advance_ctx_len(w, st["slots"], st["nacc"])
    num_ctx = w._ctx_len[st["slots"]]
    allocated = dflash_allocated_ctx_limit(st["counts"], PAGE, BLOCK)
    num_ctx = torch.minimum(num_ctx, allocated[st["rows"]])
    base_pages = st["base"][st["slots"]]
    st["read"][st["rows"]], num_ctx = dflash_rebased_ctx_view(
        st["tables"][st["rows"]], base_pages, num_ctx, PAGE
    )
    return num_ctx


def _fused(st):
    return advance_ctx_and_read_view(
        st["slots"],
        st["rows"],
        st["nacc"],
        st["ctx_len"],
        st["max_ctx"],
        st["dummy"],
        st["counts"],
        st["base"],
        st["tables"],
        st["read"],
        PAGE,
        BLOCK,
    )


def _copy(st):
    return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in st.items()}


@pytest.mark.parametrize("width", [5, 33, 2112])
@pytest.mark.parametrize("n,dummy_rows", [(1, 0), (4, 1), (8, 3), (16, 0)])
def test_advance_is_the_stock_advance_bit_for_bit(width, n, dummy_rows):
    for seed in range(4):
        st = _state(n, width, 135168, seed, dummy_rows)
        ref, got = _copy(st), _copy(st)
        ref_ctx = _stock(ref)
        got_ctx = _fused(got)
        torch.cuda.synchronize()
        assert got_ctx is not None and got_ctx.dtype == ref_ctx.dtype
        assert torch.equal(ref_ctx, got_ctx), seed
        assert torch.equal(ref["ctx_len"], got["ctx_len"]), seed
        assert torch.equal(ref["read"], got["read"]), seed


def test_advance_in_a_cuda_graph():
    st = _state(8, 64, 4096, 0, 2)
    got = _copy(st)
    _fused(got)  # compile outside capture
    got = _copy(st)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        with torch.cuda.graph(g, stream=s):
            out = _fused(got)
    torch.cuda.current_stream().wait_stream(s)
    for k in (
        "ctx_len",
        "read",
    ):  # the capture ran nothing: reset to the step's inputs, then replay
        got[k].copy_(st[k])
    g.replay()
    ref = _copy(st)
    ref_ctx = _stock(ref)
    torch.cuda.synchronize()
    assert torch.equal(ref_ctx, out)
    assert torch.equal(ref["ctx_len"], got["ctx_len"]) and torch.equal(ref["read"], got["read"])


def test_unsupported_layouts_keep_the_stock_path():
    st = _state(2, 8, 4096, 0, 0)
    assert _fused(_copy(st)) is not None
    bad = _copy(st)
    bad["read"] = bad["read"].long()
    assert _fused(bad) is None
    bad = _copy(st)
    bad["ctx_len"] = bad["ctx_len"].int()
    assert _fused(bad) is None
