# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the csa2_chunk_window entry: window rows of a context chunk (history copy and per-slot row index).

Integer index arithmetic and a row copy: every cell compares the kernel with the plain-torch reference bit for bit.
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_chunk_window import (  # noqa: E501
    csa2_chunk_window,
)

BLOCKS = 48


def _ids(c):
    return [x.why[:48] for x in c]


def _inputs(spec, gen):
    reqs, window = spec["reqs"], spec["window"]
    nreq = len(reqs)
    pages = nreq * BLOCKS + 4
    table = (
        torch.randperm(pages, generator=gen, device=DEV)[: nreq * BLOCKS].view(nreq, BLOCKS).int()
    )
    table = torch.where(
        torch.rand(nreq, BLOCKS, generator=gen, device=DEV) < spec.get("unmapped", 0.0), -1, table
    ).int()
    pool = torch.randint(0, 256, (pages, 128, 528), generator=gen, device=DEV, dtype=torch.uint8)
    pos = torch.cat([torch.arange(h, h + r) for h, r in reqs]).int().to(DEV)
    cs = torch.cat([torch.full((r,), h) for h, r in reqs]).int().to(DEV)
    rid = torch.cat([torch.full((r,), i) for i, (_, r) in enumerate(reqs)]).to(DEV)
    req_cs = torch.tensor([h for h, _ in reqs], dtype=torch.int64, device=DEV)
    floor = None
    if "floor" in spec:
        floor = (
            torch.cat([torch.full((r,), f) for f, (_, r) in zip(spec["floor"], reqs)])
            .long()
            .to(DEV)
        )
    return pos, rid, cs, req_cs, pool, table, window, floor


@pytest.mark.parametrize("cell", csa2_chunk_window.CELLS, ids=_ids(csa2_chunk_window.CELLS))
def test_chunk_window_cells(cell, certifying):  # noqa: F811
    certifying(csa2_chunk_window)
    args = _inputs(cell.spec, gen_for(cell))
    with validating(csa2_chunk_window):
        got = csa2_chunk_window(*args)
    ref = csa2_chunk_window.reference(*args)
    for name, g, r in zip(("old_rows", "swa_idx"), got, ref):
        assert g.shape == r.shape and g.dtype == r.dtype, (name, g.shape, r.shape, g.dtype, r.dtype)
        assert torch.equal(g, r), name
    assert (got[1] < 0).any() and (got[1] >= 0).any(), (
        "the cell must exercise masked and live slots"
    )


def test_chunk_window_worked_example():
    """One request with history 3 and two rows, window 128: the slots before position 0 are -1, the history row of
    position 2 is `n + 126`, and the rows of the step are 0 and 1."""
    table = torch.zeros(1, 2, dtype=torch.int32, device=DEV)
    pool = torch.randint(0, 256, (1, 128, 528), dtype=torch.uint8, device=DEV)
    pos = torch.tensor([3, 4], dtype=torch.int32, device=DEV)
    rid = torch.zeros(2, dtype=torch.int64, device=DEV)
    cs = torch.tensor([3, 3], dtype=torch.int32, device=DEV)
    req_cs = torch.tensor([3], dtype=torch.int64, device=DEV)
    with validating(csa2_chunk_window):
        old, idx = csa2_chunk_window(pos, rid, cs, req_cs, pool, table, 128)
    assert idx[0, 127].item() == 0 and idx[1, 127].item() == 1 and idx[1, 126].item() == 0
    assert (
        idx[0, 126].item() == 2 + 126
    )  # the history row of position 2: past the 2 step rows, slot 126
    assert idx[0, :124].eq(-1).all()
    assert torch.equal(old[126], pool[0, 2])  # position 2 is the last history row of the request


def test_cuda_graph_replay_follows_device_data():
    """A prefill graph replays with the next chunk's positions: the launches read device data only."""
    gen = torch.Generator(device=DEV).manual_seed(5)
    a = _inputs(dict(reqs=[(300, 40), (0, 20)], window=128), gen)
    b = _inputs(dict(reqs=[(500, 40), (60, 20)], window=128), gen)
    static = [t.clone() if isinstance(t, torch.Tensor) else t for t in a]
    csa2_chunk_window(*static)  # kernel JIT
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = csa2_chunk_window(*static)
    for dst, src in zip(static, b):
        if isinstance(dst, torch.Tensor):
            dst.copy_(src)
    g.replay()
    torch.cuda.synchronize()
    ref = csa2_chunk_window.reference(*b)
    assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])


def test_strided_req_cs_is_rejected():
    """The kernel reads `req_cs` with unit stride: a strided view (request 1 would read 0, not 800) is rejected."""
    pos = torch.tensor([300, 800], dtype=torch.int32, device=DEV)
    rid = torch.tensor([0, 1], device=DEV)
    pool = torch.zeros(2, 128, 528, dtype=torch.uint8, device=DEV)
    table = torch.zeros(2, 8, dtype=torch.int32, device=DEV)
    backing = torch.tensor([300, 0, 800, 0], device=DEV)
    csa2_chunk_window.is_valid(pos, rid, pos.long(), backing[::2].contiguous(), pool, table, 128)
    with pytest.raises(AssertionError, match="contiguous"):
        csa2_chunk_window.is_valid(pos, rid, pos.long(), backing[::2], pool, table, 128)
