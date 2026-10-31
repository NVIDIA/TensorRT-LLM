# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qsa_rollback_planes variant: `qsa_rollback_gather` and
`qsa_rollback_restore`, each cell through the entry's own reference, bitwise (both are copies).

The pool is filled with a NaN pattern `randn` never draws, then every slot a cell names is
written, so a whole-pool count of what no longer holds the fill shows that a call wrote nothing
outside those slots. The tp1 cells run over separate contiguous planes and over the packed
side-plane page of an FP8 / NVFP4 pool; the unified target's cells over core's coalesced bf16 pool at a
tep4 rank's measured page count, with the requests' pages drawn across it and the pool's first
and last page among them, so the last is addressed past 2^31 elements from a plane's base.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.qsa_rollback_planes import (  # noqa: E501
    ROW,
    plane_table,
    qsa_rollback_gather,
    qsa_rollback_restore,
)

__extra_import_path__ = [".."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "qsa_rollback_planes requires a CUDA device"

DEV = "cuda"
TPB = 32
#: The tp1 cells' pool pages.
SMALL_POOL = 600
#: One plane's page: `tokens_per_block` 128-wide rows.
PLANE_PAGE = TPB * ROW
#: The QSA position page an FP8 / NVFP4 pool packs after the 12 index pages (384 bytes).
SIDE_PLANES_PAD = 192
#: `max_seq_len` under MTP: the last position a verify row can hold is one less.
MTP_MAX_SEQ_LEN = 16384
#: bf16 NaN with a payload: the pool's fill.
SENTINEL = 0x7FA5


class _Pool:
    """One rank's index planes, every element at `SENTINEL` until a slot is written.

    `contiguous`: one `[pages, 32, 1, 128]` buffer a plane. `shared_page`: the planes side by
    side in one page with the position page after them. `coalesced`: core's KV cache manager V2
    pool, layer k's buffer at element 4096 k of a 49152-element page.
    """

    def __init__(self, spec: dict, layout: str):
        count, pages = spec["planes"], spec.get("pool_pages", SMALL_POOL)
        if layout == "contiguous":
            self.buffers = [
                torch.empty(pages, TPB, 1, ROW, dtype=torch.bfloat16, device=DEV)
                for _ in range(count)
            ]
            self.planes = list(self.buffers)
        else:
            pad = SIDE_PLANES_PAD if layout == "shared_page" else 0
            page = torch.empty(pages, count * PLANE_PAGE + pad, dtype=torch.bfloat16, device=DEV)
            self.buffers = [page]
            self.planes = [
                page[:, k * PLANE_PAGE : (k + 1) * PLANE_PAGE].view(pages, TPB, 1, ROW)
                for k in range(count)
            ]
        for b in self.buffers:
            b.view(torch.int16).fill_(SENTINEL)
        self.pages = pages

    def written(self) -> int:
        """Elements anywhere in the pool that no longer hold the fill."""
        total = 0
        for b in self.buffers:
            for chunk in b.view(torch.int16).reshape(b.shape[0], -1).split(8192):
                total += int((chunk != SENTINEL).sum())
        return total


def _layouts(cell):
    return (cell.spec["layout"],) if "layout" in cell.spec else ("contiguous", "shared_page")


def _cases(op):
    return [(c, layout) for c in op.CELLS for layout in _layouts(c)]


def _ids(cases):
    return [f"{c.why[:44]}-{layout}" for c, layout in cases]


GATHER_CASES = _cases(qsa_rollback_gather)
RESTORE_CASES = _cases(qsa_rollback_restore)


def _verify_slots(spec, seed):
    """`requests` x `p` consecutive positions each, every request's QSA pages its own physical
    pages of the pool. Request 0 straddles a page boundary from the pool's first page into its
    last; the last request ends at position 16383."""
    n, p, pool = spec["requests"], spec["p"], spec["pool_pages"]
    starts = (
        [TPB - p // 2]
        + [(29 + 127 * s) % (MTP_MAX_SEQ_LEN - p) for s in range(1, n - 1)]
        + [MTP_MAX_SEQ_LEN - p]
    )
    positions = torch.tensor(starts)[:, None] + torch.arange(p)[None, :]
    logical = positions // TPB
    rest = torch.randperm(pool - 2, generator=torch.Generator().manual_seed(seed)) + 1
    physical = torch.cat([torch.tensor([0, pool - 1]), rest])
    pages = torch.empty(n, p, dtype=torch.int64)
    cursor = 0
    for s in range(n):
        for lp in logical[s].unique().tolist():
            pages[s][logical[s] == lp] = physical[cursor]
            cursor += 1
    return pages.reshape(-1).to(DEV), (positions % TPB).reshape(-1).to(DEV)


def _slots(spec, seed, g):
    if "requests" in spec:
        return _verify_slots(spec, seed)
    rows = spec["rows"]
    pages = torch.randperm(SMALL_POOL, generator=torch.Generator().manual_seed(rows))[:rows]
    within = torch.randint(0, TPB, (rows,), generator=g, device=DEV)
    return pages.to(DEV).to(torch.int64), within.to(torch.int64)


def _rejected(spec, g):
    """The call site's mask for a verify step (bool, `ordinal >= accepted`; one request rejects
    every draft, one none), or the tp1 cells' random int32 one."""
    if spec.get("rejected") != "bool":
        return (torch.rand(spec["rows"], generator=g, device=DEV) < 0.5).to(torch.int32)
    n, p = spec["requests"], spec["p"]
    accepted = torch.randint(1, p + 1, (n,), generator=g, device=DEV)
    accepted[0], accepted[1] = 1, p
    return (torch.arange(p, device=DEV)[None, :] >= accepted[:, None]).reshape(-1)


def _fill(planes, pages, within, g) -> None:
    """Write fresh values into every slot of every plane (a step's index store)."""
    for t in planes:
        t[pages, within] = torch.randn(pages.shape[0], 1, ROW, generator=g, device=DEV).to(t.dtype)


def _slots_now(planes, pages, within) -> torch.Tensor:
    return torch.stack([t[pages, within].reshape(-1, ROW) for t in planes])


def _assert_far_page(spec, pool, far_pages) -> None:
    """A measured-pool cell addresses the pool's last page, past 2^31 elements from the base of
    every plane, where a 32-bit offset would wrap."""
    if "pool_pages" not in spec:
        return
    last = int(far_pages.max())
    assert last == pool.pages - 1, f"the pool's last page {pool.pages - 1} is unread ({last})"
    assert last * pool.planes[-1].stride(0) >= 2**31


def _setup(cell, layout):
    seed = zlib.crc32(f"{cell.why}/{layout}".encode()) % 2**31
    g = torch.Generator(device=DEV).manual_seed(seed)
    pool = _Pool(cell.spec, layout)
    pages, within = _slots(cell.spec, seed, g)
    assert pages.shape == (cell.spec["rows"],)
    _fill(pool.planes, pages, within, g)
    return g, pool, pages, within


@pytest.mark.parametrize("cell, layout", GATHER_CASES, ids=_ids(GATHER_CASES))
def test_gather_cells(cell, layout, certifying):  # noqa: F811
    certifying(qsa_rollback_gather)
    _, pool, pages, within = _setup(cell, layout)
    ptrs, strides = plane_table(pool.planes)
    count, rows = len(pool.planes), pages.shape[0]

    want = qsa_rollback_gather.reference(pool.planes, pages, within, count)
    got = qsa_rollback_gather(ptrs, strides, pages, within, count)

    assert got.shape == (count, rows, ROW) and got.dtype == torch.bfloat16
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    assert torch.equal(
        _slots_now(pool.planes, pages, within).view(torch.int16), want.view(torch.int16)
    )
    assert pool.written() == count * rows * ROW, "the gather wrote into the pool"
    _assert_far_page(cell.spec, pool, pages)


@pytest.mark.parametrize("cell, layout", RESTORE_CASES, ids=_ids(RESTORE_CASES))
def test_restore_cells(cell, layout, certifying):  # noqa: F811
    certifying(qsa_rollback_restore)
    g, pool, pages, within = _setup(cell, layout)
    ptrs, strides = plane_table(pool.planes)
    count, rows = len(pool.planes), pages.shape[0]
    # What the begin step's gather captured (its own cells hold it to this), then the step
    # overwrites every slot.
    old = _slots_now(pool.planes, pages, within)
    _fill(pool.planes, pages, within, g)
    rejected = _rejected(cell.spec, g)
    assert bool(rejected.bool().any()) and not bool(rejected.bool().all())

    want = qsa_rollback_restore.reference(pool.planes, pages, within, rejected, old)
    assert qsa_rollback_restore(ptrs, strides, pages, within, rejected, old) is None

    got = _slots_now(pool.planes, pages, within)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    assert pool.written() == count * rows * ROW, "the restore wrote outside the rows' slots"
    _assert_far_page(cell.spec, pool, pages[rejected.bool()])


def test_cuda_graph_replay():
    """The verify step is graph-captured: both calls at the envelope's 1024 rows, captured
    once, then replayed over a new step's slots written into the captured inputs."""
    (cell,) = [c for c in qsa_rollback_gather.CELLS if c.spec["rows"] == 1024]
    spec = cell.spec
    g, pool, pages, within = _setup(cell, spec["layout"])
    ptrs, strides = plane_table(pool.planes)
    count, rows = len(pool.planes), pages.shape[0]
    rejected = _rejected(spec, g)
    old = torch.empty(count, rows, ROW, dtype=torch.bfloat16, device=DEV)

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # Triton compiles on the first launch, outside the capture
        old.copy_(qsa_rollback_gather(ptrs, strides, pages, within, count))
        qsa_rollback_restore(ptrs, strides, pages, within, rejected, old)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    gather_graph, restore_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(gather_graph):
        gathered = qsa_rollback_gather(ptrs, strides, pages, within, count)
    with torch.cuda.graph(restore_graph, pool=gather_graph.pool()):
        qsa_rollback_restore(ptrs, strides, pages, within, rejected, old)

    new_pages, new_within = _verify_slots(spec, 991)
    assert not torch.equal(new_pages, pages)
    pages.copy_(new_pages)
    within.copy_(new_within)
    rejected.copy_(_rejected(spec, g))
    _fill(pool.planes, pages, within, g)

    want_old = qsa_rollback_gather.reference(pool.planes, pages, within, count)
    gather_graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(gathered.view(torch.int16), want_old.view(torch.int16))

    old.copy_(gathered)
    _fill(pool.planes, pages, within, g)
    want = qsa_rollback_restore.reference(pool.planes, pages, within, rejected, old)
    restore_graph.replay()
    torch.cuda.synchronize()
    got = _slots_now(pool.planes, pages, within)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def _small():
    spec = dict(planes=12, rows=16)
    g = torch.Generator(device=DEV).manual_seed(5)
    pool = _Pool(spec, "contiguous")
    pages, within = _slots(spec, 5, g)
    _fill(pool.planes, pages, within, g)
    return g, pool, pages, within


def test_guard_refuses_a_strided_old():
    """The kernel reads `old` as dense `[L, R, 128]`: a strided view restores the wrong values,
    with no error."""
    g, pool, pages, within = _small()
    ptrs, strides = plane_table(pool.planes)
    count, rows = len(pool.planes), pages.shape[0]
    wide = torch.randn(count, rows, 2 * ROW, generator=g, device=DEV).to(torch.bfloat16)
    old = wide[..., :ROW]
    assert not old.is_contiguous()
    rejected = torch.ones(rows, dtype=torch.bool, device=DEV)

    want = qsa_rollback_restore.reference(pool.planes, pages, within, rejected, old)
    qsa_rollback_restore(ptrs, strides, pages, within, rejected, old)
    got = _slots_now(pool.planes, pages, within)
    assert not torch.equal(got.view(torch.int16), want.view(torch.int16)), (
        "the hazard this guard covers is gone"
    )

    with validating(qsa_rollback_restore), pytest.raises(AssertionError, match="contiguous"):
        qsa_rollback_restore(ptrs, strides, pages, within, rejected, old)


def test_guard_refuses_more_planes_than_the_table_holds():
    """Not driven unguarded: the kernel would read addresses past the table's end."""
    _, pool, pages, within = _small()
    ptrs, strides = plane_table(pool.planes[:-1])
    with validating(qsa_rollback_gather), pytest.raises(AssertionError, match="num_planes"):
        qsa_rollback_gather(ptrs, strides, pages, within, len(pool.planes))


def test_guard_refuses_a_short_within():
    """Not driven unguarded: rows are counted off `pages`, so `within` would be read past its
    end."""
    _, pool, pages, within = _small()
    ptrs, strides = plane_table(pool.planes)
    with validating(qsa_rollback_gather), pytest.raises(AssertionError, match=r"one \[R\] pair"):
        qsa_rollback_gather(ptrs, strides, pages, within[:-1], len(pool.planes))
