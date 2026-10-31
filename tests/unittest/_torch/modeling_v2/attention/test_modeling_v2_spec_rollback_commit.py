# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the spec_rollback_commit variant: `spec_rollback_stage` and
`spec_rollback_commit`, every cell through the entry's own reference, bitwise (both are data
movement).

A cell is one verify step at the call site's layout: `n` generation requests of `p` rows, each
request's rows at consecutive positions on pages of their own, int32 `accepted` / `slots` /
`ids` and int64 `pages` / `within` as `SpeculativeRollback` hands them. The index-K planes are
`[pages, 32, 1, 128]` bf16 views addressed through `plane_table`: separate buffers in the tp1
cells, the KV cache manager's coalesced pool (12 planes side by side in a 49152-element page) at
a tep4 rank's measured page count in the unified target's. Request 0's rows straddle the pool's first
and last page and it accepts one row, so its rejected rows are restored on the last page, past
2^31 elements from a plane's base. The pools are filled with a NaN pattern `randn` never draws
and the trunk's writes are put at every row's slot, so a whole-pool count shows the commit
wrote nothing else; the n-gram, conv and position pools are compared whole.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_rollback_planes import (  # noqa: E501
    plane_table,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.spec_rollback_commit import (  # noqa: E501
    ROW,
    spec_rollback_commit,
    spec_rollback_stage,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "spec_rollback_commit requires a CUDA device"

DEV = "cuda"
TPB = 32
#: The tp1 cells' PLE geometry and plane pool, where a cell does not name its own.
TP1_GEOMETRY = dict(context=2, channels=640, state_len=3, layout="separate", pool_pages=600)
#: `max_seq_len` under MTP: the last position a verify row can hold is one less.
MTP_MAX_SEQ_LEN = 16384
#: The vocabulary the ids are drawn from.
VOCAB = 248320
#: bf16 NaN with a payload: the index pool's fill.
SENTINEL = 0x7FA5
#: The position plane's fill; positions stay below 2^24.
POSITION_FILL = 0x7FA5A5A5
#: One plane's page: `tokens_per_block` 128-wide rows.
PLANE_PAGE = TPB * ROW


def _seed(why: str) -> int:
    return zlib.crc32(why.encode()) % 2**31


def _geometry(spec: dict) -> dict:
    g = dict(TP1_GEOMETRY, **spec)
    g.setdefault("state_slots", 2 * spec["n"] + 4)
    return g


def _slots(n: int, state_slots: int, gen: torch.Generator) -> torch.Tensor:
    """`n` distinct recurrent slots; at an engine-sized pool its first and last among them."""
    if state_slots < n + 2 or n < 2:
        picked = torch.randperm(state_slots - 1, generator=gen)[:n] + 1
    else:
        inner = torch.randperm(state_slots - 2, generator=gen)[: n - 2] + 1
        picked = torch.cat([torch.tensor([0, state_slots - 1]), inner])
        picked = picked[torch.randperm(n, generator=gen)]
    return picked.to(DEV, torch.int32)


def _ple_pools(g: dict, gen: torch.Generator):
    ngram_pool = torch.randint(0, 1 << 40, (g["state_slots"], g["context"]), generator=gen)
    conv_pool = torch.randn(g["state_slots"], g["channels"], g["state_len"], generator=gen)
    return ngram_pool.to(DEV), conv_pool.to(DEV, torch.bfloat16)


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t


def _equal(got: torch.Tensor, want: torch.Tensor, what: str) -> None:
    assert got.dtype == want.dtype and got.shape == want.shape, (what, got.shape, want.shape)
    differ = (_bits(got) != _bits(want)).reshape(got.shape[0], -1).any(dim=1)
    assert not bool(differ.any()), f"{what}: {int(differ.sum())} of {got.shape[0]} rows differ"


class _Planes:
    """One rank's index-K planes, every element at `SENTINEL` until a slot is written."""

    def __init__(self, count: int, pages: int, layout: str):
        if layout == "separate":
            self.buffers = [
                torch.empty(pages, TPB, 1, ROW, dtype=torch.bfloat16, device=DEV)
                for _ in range(count)
            ]
            self.views = list(self.buffers)
        else:
            assert layout == "coalesced", layout
            page = torch.empty(pages, count * PLANE_PAGE, dtype=torch.bfloat16, device=DEV)
            self.buffers = [page]
            self.views = [
                page[:, k * PLANE_PAGE : (k + 1) * PLANE_PAGE].view(pages, TPB, 1, ROW)
                for k in range(count)
            ]
        self.fill()

    def fill(self) -> None:
        for b in self.buffers:
            b.view(torch.int16).fill_(SENTINEL)

    def written(self) -> int:
        """Elements anywhere in the pool that no longer hold the fill."""
        total = 0
        for b in self.buffers:
            for chunk in b.view(torch.int16).reshape(b.shape[0], -1).split(8192):
                total += int((chunk != SENTINEL).sum())
        return total

    def rows(self, pages: torch.Tensor, within: torch.Tensor) -> torch.Tensor:
        return torch.stack([v[pages, within].reshape(-1, ROW) for v in self.views])


class _Step:
    """One verify step's staged state, the pools it commits into, and the trunk's writes."""

    def __init__(self, spec: dict, seed: int):
        g = _geometry(spec)
        self.n, self.p, self.count = spec["n"], spec["p"], spec["planes"]
        self.planes = _Planes(self.count, g["pool_pages"], g["layout"])
        self.position = torch.empty(g["pool_pages"], TPB, 3, dtype=torch.int32, device=DEV)
        self.ptrs, self.strides = plane_table(self.planes.views)
        gen = torch.Generator().manual_seed(seed)
        self.ngram_pool, self.conv_pool = _ple_pools(g, gen)
        self.g = g
        n, p, rows = self.n, self.p, self.n * self.p
        self.accepted = torch.empty(n, dtype=torch.int32, device=DEV)
        self.slots = torch.empty(n, dtype=torch.int32, device=DEV)
        self.pages = torch.empty(rows, dtype=torch.int64, device=DEV)
        self.within = torch.empty(rows, dtype=torch.int64, device=DEV)
        self.stream = torch.empty(n, g["context"] + p, dtype=torch.int64, device=DEV)
        self.conv_prev = torch.empty(
            n, g["channels"], g["state_len"], dtype=torch.bfloat16, device=DEV
        )
        self.conv_x = torch.empty(n, p, g["channels"], dtype=torch.bfloat16, device=DEV)
        self.position_old = torch.empty(rows, 3, dtype=torch.int32, device=DEV)
        self.index_old = torch.empty(self.count, rows, ROW, dtype=torch.bfloat16, device=DEV)
        self.draw(gen)

    def draw(self, gen: torch.Generator) -> None:
        """Draw every staged input into its buffer (in place, so a captured graph re-reads it)
        and put the trunk's writes at the rows' slots of a freshly filled index and position
        pool."""
        g, n, p = self.g, self.n, self.p
        pool_pages = g["pool_pages"]
        accepted = torch.randint(1, p + 1, (n,), generator=gen)
        accepted[0] = 1
        if n > 1:
            accepted[1] = p
        self.accepted.copy_(accepted)
        self.slots.copy_(_slots(n, g["state_slots"], gen))

        # Request 0 straddles its first two blocks, mapped to the pool's first and last page;
        # every other block of every request gets a page of its own.
        starts = torch.randint(0, MTP_MAX_SEQ_LEN - p + 1, (n,), generator=gen)
        starts[0] = TPB - p // 2
        positions = starts[:, None] + torch.arange(p)[None, :]
        blocks = positions // TPB
        free = torch.randperm(pool_pages - 2, generator=gen) + 1
        page_of = {}
        pages = torch.empty(n, p, dtype=torch.int64)
        for i in range(n):
            for t in range(p):
                key = (i, int(blocks[i, t]))
                if key not in page_of:
                    if i == 0:
                        page_of[key] = 0 if int(blocks[i, t]) == 0 else pool_pages - 1
                    else:
                        page_of[key] = int(free[len(page_of)])
                pages[i, t] = page_of[key]
        self.pages.copy_(pages.reshape(-1))
        self.within.copy_((positions % TPB).reshape(-1))

        rows = n * p
        self.stream.copy_(torch.randint(0, 1 << 40, tuple(self.stream.shape), generator=gen))
        self.conv_prev.copy_(torch.randn(tuple(self.conv_prev.shape), generator=gen))
        self.conv_x.copy_(torch.randn(tuple(self.conv_x.shape), generator=gen))
        self.position_old.copy_(torch.randint(0, 1 << 20, (rows, 3), generator=gen))
        self.index_old.copy_(torch.randn(self.count, rows, ROW, generator=gen))

        # What the trunk wrote at every row: kept where the request accepted the row.
        self.planes.fill()
        self.position.fill_(POSITION_FILL)
        trunk = torch.randn(self.count, rows, ROW, generator=gen).to(DEV, torch.bfloat16)
        for k, view in enumerate(self.planes.views):
            view[self.pages, self.within] = trunk[k].reshape(rows, 1, ROW)
        self.position[self.pages, self.within] = torch.randint(
            0, 1 << 20, (rows, 3), generator=gen
        ).to(DEV, torch.int32)

    def call_args(self, index_old=None, conv_prev=None) -> tuple:
        return (
            self.accepted,
            self.slots,
            self.ngram_pool,
            self.stream,
            self.conv_pool,
            self.conv_prev if conv_prev is None else conv_prev,
            self.conv_x,
            self.position,
            self.position_old,
            self.pages,
            self.within,
            self.ptrs,
            self.strides,
            self.index_old if index_old is None else index_old,
        )

    def reference(self):
        return spec_rollback_commit.reference(
            self.accepted,
            self.slots,
            self.ngram_pool,
            self.stream,
            self.conv_pool,
            self.conv_prev,
            self.conv_x,
            self.position,
            self.position_old,
            self.pages,
            self.within,
            self.planes.views,
            self.index_old,
        )

    def snapshot(self):
        return self.ngram_pool.clone(), self.conv_pool.clone(), self.position.clone()

    def check(self, ref, before) -> None:
        """The named slots hold `ref`, bit for bit, and nothing else in any pool moved."""
        ngram, conv, pos, index = ref
        ngram_before, conv_before, pos_before = before
        slots = self.slots.long()
        want = ngram_before.clone()
        want[slots] = ngram
        _equal(self.ngram_pool, want, "n-gram pool")
        want = conv_before.clone()
        want[slots] = conv
        _equal(self.conv_pool, want, "PLE conv pool")
        want = pos_before.clone()
        want[self.pages, self.within] = pos
        _equal(self.position, want, "position plane")
        _equal(self.planes.rows(self.pages, self.within), index, "index-K planes")
        rows = self.n * self.p
        assert self.planes.written() == self.count * rows * ROW, (
            "the commit wrote index-K pool elements outside the rows' slots"
        )


@pytest.mark.parametrize(
    "cell", spec_rollback_stage.CELLS, ids=[c.why for c in spec_rollback_stage.CELLS]
)
def test_stage_cells(cell, certifying) -> None:  # noqa: F811
    certifying(spec_rollback_stage)
    spec = cell.spec
    n, p = spec["n"], spec["p"]
    g = _geometry(spec)
    gen = torch.Generator().manual_seed(_seed(cell.why))
    ngram_pool, conv_pool = _ple_pools(g, gen)
    slots = _slots(n, g["state_slots"], gen)
    ids = torch.randint(0, VOCAB, (n * p,), generator=gen).to(DEV, torch.int32)
    stream, conv_prev = spec_rollback_stage(ngram_pool, conv_pool, slots, ids, p)
    want_stream, want_prev = spec_rollback_stage.reference(ngram_pool, conv_pool, slots, ids, p)
    _equal(stream, want_stream, "stream")
    _equal(conv_prev, want_prev, "conv_prev")


@pytest.mark.parametrize(
    "cell", spec_rollback_commit.CELLS, ids=[c.why for c in spec_rollback_commit.CELLS]
)
def test_commit_cells(cell, certifying) -> None:  # noqa: F811
    certifying(spec_rollback_commit)
    step = _Step(cell.spec, _seed(cell.why))
    ref = step.reference()
    before = step.snapshot()
    spec_rollback_commit(*step.call_args())
    torch.cuda.synchronize()
    step.check(ref, before)


def test_commit_graph_replay_rereads_its_inputs() -> None:
    """One graph captured at the call site's 1024 rows (128 requests x 8, the 10240 x 9 conv
    state over 1874 slots), replayed after every staged input, slot and page is redrawn in place:
    what the decode target's captured verify step does each iteration."""
    spec = dict(spec_rollback_commit.CELLS[-1].spec, pool_pages=4096)
    step = _Step(spec, 4242)
    args = step.call_args()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        spec_rollback_commit(*args)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        spec_rollback_commit(*args)
    for seed in (7, 8):
        step.draw(torch.Generator().manual_seed(seed))
        ref = step.reference()
        before = step.snapshot()
        graph.replay()
        torch.cuda.synchronize()
        step.check(ref, before)


_GUARD_SPEC = dict(n=8, p=4, planes=12)


def test_guard_refuses_a_strided_index_old() -> None:
    """A `[planes, rows, ROW]` view of a row-major `[rows, planes, ROW]` buffer is read as if
    packed: the rejected rows get other rows' and planes' values, and nothing raises."""
    step = _Step(_GUARD_SPEC, 11)
    rows = step.n * step.p
    buf = torch.empty(rows, step.count, ROW, dtype=torch.bfloat16, device=DEV)
    strided = buf.permute(1, 0, 2)
    strided.copy_(step.index_old)
    assert not strided.is_contiguous()
    ref = step.reference()
    spec_rollback_commit(*step.call_args(index_old=strided))
    torch.cuda.synchronize()
    got = step.planes.rows(step.pages, step.within)
    assert not torch.equal(_bits(got), _bits(ref[3])), "the hazard this guard covers is gone"

    with validating(spec_rollback_commit), pytest.raises(AssertionError, match="index_old must"):
        spec_rollback_commit(*step.call_args(index_old=strided))


def test_guard_refuses_a_strided_conv_prev() -> None:
    """A `[n, C, state_len]` view of a `[n, state_len, C]` buffer is read as packed rows: the
    conv pool gets the wrong state, and nothing raises."""
    step = _Step(_GUARD_SPEC, 12)
    shape = step.conv_prev.shape
    buf = torch.empty(shape[0], shape[2], shape[1], dtype=torch.bfloat16, device=DEV)
    strided = buf.transpose(1, 2)
    strided.copy_(step.conv_prev)
    assert not strided.is_contiguous()
    ref = step.reference()
    spec_rollback_commit(*step.call_args(conv_prev=strided))
    torch.cuda.synchronize()
    got = step.conv_pool[step.slots.long()]
    assert not torch.equal(_bits(got), _bits(ref[1])), "the hazard this guard covers is gone"

    with validating(spec_rollback_commit), pytest.raises(AssertionError, match="conv_prev must"):
        spec_rollback_commit(*step.call_args(conv_prev=strided))
