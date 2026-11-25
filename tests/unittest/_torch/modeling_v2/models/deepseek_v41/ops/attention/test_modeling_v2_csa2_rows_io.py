# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the csa2_rows_io entries: row addressing, gather and masked scatter over paged pools.

Everything here is integer index arithmetic and byte / fp32 copies, so every cell gates the op
against its loop-over-elements reference bit for bit.
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_rows_io import (  # noqa: E501
    csa2_gather_rows,
    csa2_gather_rows_footer,
    csa2_row_addr,
    csa2_scatter_rows,
    csa2_scatter_rows_footer,
)


def _ids(c):
    return [x.why[:44] for x in c]


def _pool(spec, gen):
    shape = (spec["pages"], spec["rows"], spec["w"])
    if spec["dtype"] == torch.uint8:
        return torch.randint(0, 256, shape, generator=gen, device=DEV, dtype=torch.uint8)
    return torch.randn(shape, generator=gen, device=DEV, dtype=spec["dtype"])


@pytest.mark.parametrize("cell", csa2_row_addr.CELLS, ids=_ids(csa2_row_addr.CELLS))
def test_row_addr_cells(cell, certifying):  # noqa: F811
    certifying(csa2_row_addr)
    s, gen = cell.spec, gen_for(cell)
    req, blocks, rpp, n = s["requests"], s["blocks"], s["rows_per_page"], s["n"]
    table = torch.randperm(req * blocks + 4, generator=gen, device=DEV)[: req * blocks].view(
        req, blocks
    )
    table = table.int()
    table = torch.where(torch.rand(req, blocks, generator=gen, device=DEV) < 0.2, -1, table).int()
    rid = torch.randint(0, req, (n,), generator=gen, device=DEV)
    row = torch.randint(-3, blocks * rpp + 2 * rpp, (n,), generator=gen, device=DEV)
    # A small cell (the 8-row decode batch) may draw no invalid or no valid row, so both outcomes
    # are pinned rather than left to the draw: row 0 is negative (invalid), row 1 sits in block 0
    # of its request, which is mapped to a page (valid).
    row[0] = -1
    table[rid[1], 0] = req * blocks + 3
    row[1] = rpp // 2
    got = csa2_row_addr(table, rid, row, rpp)
    assert torch.equal(got, csa2_row_addr.reference(table, rid, row, rpp))
    assert (got == -1).any() and (got >= 0).any(), "the cell must exercise both outcomes"


def test_row_addr_worked_example():
    table = torch.tensor([[5, 2, -1], [0, 7, 1]], dtype=torch.int32, device=DEV)
    rid = torch.tensor([0, 0, 0, 1, 1, 0], device=DEV)
    row = torch.tensor([0, 5, 9, 6, -1, 100], device=DEV)
    with validating(csa2_row_addr):
        assert csa2_row_addr(table, rid, row, 4).tolist() == [20, 9, -1, 30, -1, -1]


@pytest.mark.parametrize("cell", csa2_gather_rows.CELLS, ids=_ids(csa2_gather_rows.CELLS))
def test_gather_cells(cell, certifying):  # noqa: F811
    certifying(csa2_gather_rows)
    s, gen = cell.spec, gen_for(cell)
    pool = _pool(s, gen)
    n = s["n"] if isinstance(s["n"], tuple) else (s["n"],)
    addr = torch.randint(-1, s["pages"] * s["rows"], n, generator=gen, device=DEV)
    got = csa2_gather_rows(pool, addr)
    assert got.shape == (*n, s["w"])
    assert torch.equal(got, csa2_gather_rows.reference(pool, addr))
    assert not got[addr < 0].any(), "-1 addresses give zero rows"


@pytest.mark.parametrize("cell", csa2_scatter_rows.CELLS, ids=_ids(csa2_scatter_rows.CELLS))
def test_scatter_cells(cell, certifying):  # noqa: F811
    certifying(csa2_scatter_rows)
    s, gen = cell.spec, gen_for(cell)
    pool = _pool(s, gen)
    n = s["n"]
    addr = torch.randperm(s["pages"] * s["rows"], generator=gen, device=DEV)[:n]
    if s["masked"] != "first":
        addr = torch.where(torch.rand(n, generator=gen, device=DEV) < 0.1, -1, addr)
    rows = _pool(dict(s, pages=1, rows=n), gen)[0]
    if s["masked"] == "first":
        keep = torch.ones(n, dtype=torch.bool, device=DEV)
        keep[0] = False
    else:
        keep = torch.rand(n, generator=gen, device=DEV) >= s["masked"]
    ref_pool = pool.clone()
    out = csa2_scatter_rows(pool, addr, rows, keep)
    assert out is pool, "the scatter writes in place and returns the pool"
    csa2_scatter_rows.reference(ref_pool, addr, rows, keep)
    assert torch.equal(pool, ref_pool)


def test_scatter_with_first_entry_skipped_is_deterministic():
    pool = torch.zeros(2, 2, 2, device=DEV)
    addr = torch.tensor([3, 1, 1, 2], device=DEV)
    rows = torch.tensor([[9.0, 9], [1, 1], [1, 1], [2, 2]], device=DEV)
    keep = torch.tensor([False, True, True, True], device=DEV)
    with validating(csa2_scatter_rows):
        csa2_scatter_rows(pool, addr, rows, keep)
    flat = pool.view(-1, 2)
    assert flat[1].tolist() == [1, 1] and flat[2].tolist() == [2, 2] and flat[3].tolist() == [0, 0]


def _footer_addrs(spec, gen):
    a = spec["addrs"]
    if a == "random":
        return torch.randint(-1, spec["pages"] * 64, (spec["n"],), generator=gen, device=DEV)
    if a == "consecutive":
        return torch.arange(spec["n"], device=DEV)
    return torch.tensor(a, device=DEV)


@pytest.mark.parametrize(
    "cell", csa2_gather_rows_footer.CELLS, ids=_ids(csa2_gather_rows_footer.CELLS)
)
def test_gather_footer_cells(cell, certifying):  # noqa: F811
    certifying(csa2_gather_rows_footer)
    s, gen = cell.spec, gen_for(cell)
    pool = torch.randint(0, 256, (s["pages"], 64, 68), generator=gen, device=DEV, dtype=torch.uint8)
    addr = _footer_addrs(s, gen)
    got = csa2_gather_rows_footer(pool, addr)
    assert torch.equal(got, csa2_gather_rows_footer.reference(pool, addr))
    assert not got[addr < 0].any()


@pytest.mark.parametrize(
    "cell", csa2_scatter_rows_footer.CELLS, ids=_ids(csa2_scatter_rows_footer.CELLS)
)
def test_scatter_footer_cells(cell, certifying):  # noqa: F811
    certifying(csa2_scatter_rows_footer)
    s, gen = cell.spec, gen_for(cell)
    pool = torch.randint(0, 256, (s["pages"], 64, 68), generator=gen, device=DEV, dtype=torch.uint8)
    addr = _footer_addrs(s, gen)
    n = addr.shape[0]
    rows = torch.randint(0, 256, (n, 68), generator=gen, device=DEV, dtype=torch.uint8)
    keep = torch.ones(n, dtype=torch.bool, device=DEV)
    if s["keep"] == "middle_skipped":
        keep[1] = False
    elif s["keep"] == "first_skipped":
        keep[0] = False
    orig, ref_pool = pool.clone(), pool.clone()
    csa2_scatter_rows_footer(pool, addr, rows, keep)
    csa2_scatter_rows_footer.reference(ref_pool, addr, rows, keep)
    assert torch.equal(pool, ref_pool)
    if (
        s["keep"] == "middle_skipped"
    ):  # the skipped entry's row (data bytes and footer scales) keeps its old bytes
        assert torch.equal(pool.view(-1)[64:128], orig.view(-1)[64:128])
        assert torch.equal(pool.view(-1)[4096 + 4 : 4096 + 8], orig.view(-1)[4096 + 4 : 4096 + 8])


def test_footer_roundtrip_and_layout():
    pool = torch.zeros(3, 128, 68, dtype=torch.uint8, device=DEV)
    rows = torch.randint(0, 255, (5, 68), dtype=torch.uint8, device=DEV)
    addr = torch.tensor([0, 1, 63, 64, 130], device=DEV)
    with validating(csa2_scatter_rows_footer, csa2_gather_rows_footer):
        csa2_scatter_rows_footer(pool, addr, rows, torch.ones(5, dtype=torch.bool, device=DEV))
        assert torch.equal(csa2_gather_rows_footer(pool, addr), rows)
        empty = csa2_gather_rows_footer(pool, torch.tensor([-1], device=DEV))
    flat = pool.view(-1)
    page = 4352
    assert torch.equal(flat[64:128], rows[1, :64])  # row 1: data bytes at 1*64
    assert torch.equal(flat[4096 + 4 : 4096 + 8], rows[1, 64:])  # its scales in the footer
    assert torch.equal(flat[page : page + 64], rows[3, :64])  # row 64 = first row of page 1
    assert torch.equal(empty, torch.zeros(1, 68, dtype=torch.uint8, device=DEV))


def test_scatter_rejections():
    pool = torch.zeros(2, 4, 3, device=DEV)
    rows = torch.ones(2, 3, device=DEV)
    ok = torch.tensor([True, True], device=DEV)
    addr = torch.tensor([0, 1], device=DEV)
    csa2_scatter_rows.is_valid(pool, addr, rows, ok)  # the accepted case
    # nothing kept is a no-op, accepted (both scatters write nothing then; see test_fully_masked_scatter_is_a_no_op)
    csa2_scatter_rows.is_valid(pool, torch.tensor([-1, -1], device=DEV), rows, ok)
    csa2_scatter_rows.is_valid(pool, addr, rows, ~ok)
    with pytest.raises(AssertionError, match="silently convert"):
        csa2_scatter_rows.is_valid(pool, addr, rows.to(torch.bfloat16), ok)
    with pytest.raises(AssertionError, match="winner is undefined"):
        csa2_scatter_rows.is_valid(
            pool, torch.tensor([1, 1], device=DEV), torch.stack([rows[0], 2 * rows[1]]), ok
        )
    with pytest.raises(AssertionError):
        csa2_scatter_rows_footer.is_valid(
            torch.zeros(100, dtype=torch.uint8, device=DEV), addr, rows, ok
        )


def test_gather_and_addr_rejections():
    with pytest.raises(AssertionError, match="integer"):
        csa2_gather_rows.is_valid(torch.zeros(2, 2, 2, device=DEV), torch.tensor([0.5], device=DEV))
    with pytest.raises(AssertionError, match="integer"):
        csa2_row_addr.is_valid(
            torch.zeros(1, 2, dtype=torch.int32),
            torch.zeros(1, dtype=torch.long),
            torch.tensor([1.5]),
            4,
        )
    with pytest.raises(AssertionError):
        csa2_gather_rows_footer.is_valid(torch.zeros(10, dtype=torch.uint8), torch.tensor([0]))


@pytest.mark.parametrize(
    "addr, keep",
    [([3], [False]), ([-1], [True]), ([-1, 70, 70], [True, False, False])],
    ids=["skipped-row", "unmapped-row", "unmapped-and-skipped"],
)
def test_fully_masked_scatter_is_a_no_op(addr, keep):
    """No kept entry with a valid address: neither scatter writes a byte (the footer scatter used to redirect
    every entry to entry 0: a skipped row 3 was overwritten, and an address of -1 wrote into the last native
    page through negative offsets)."""
    gen = torch.Generator(device=DEV).manual_seed(7)
    addr = torch.tensor(addr, device=DEV)
    keep = torch.tensor(keep, device=DEV)
    rows = torch.randint(0, 256, (addr.shape[0], 68), generator=gen, device=DEV, dtype=torch.uint8)
    pool = torch.randint(1, 256, (2, 64, 68), generator=gen, device=DEV, dtype=torch.uint8)
    orig = pool.clone()
    with validating(csa2_scatter_rows_footer):
        csa2_scatter_rows_footer(pool, addr, rows, keep)
    assert torch.equal(pool, orig)
    assert torch.equal(csa2_scatter_rows_footer.reference(orig.clone(), addr, rows, keep), orig)
    if DEV == "cuda":  # the Triton scatter
        flat_pool = torch.randint(1, 256, (2, 64, 68), generator=gen, device=DEV, dtype=torch.uint8)
        flat_orig = flat_pool.clone()
        with validating(csa2_scatter_rows):
            csa2_scatter_rows(flat_pool, addr, rows, keep)
        assert torch.equal(flat_pool, flat_orig)


def test_row_addr_of_a_table_without_blocks():
    """A `[requests, 0]` table maps no row: every address is -1, as the reference says."""
    table = torch.zeros(1, 0, dtype=torch.int32, device=DEV)
    rid = torch.zeros(3, dtype=torch.long, device=DEV)
    row = torch.tensor([0, 5, -1], device=DEV)
    with validating(csa2_row_addr):
        got = csa2_row_addr(table, rid, row, 4)
    assert got.tolist() == [-1, -1, -1] == csa2_row_addr.reference(table, rid, row, 4).tolist()
