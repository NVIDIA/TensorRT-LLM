# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Row addressing, gather and masked scatter over a paged byte/float
pool (plain torch; the scatter is one Triton launch)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# ---- the engine's native footer index pages ---------------------------------------------------
# One native page holds 64 rows: 64 x 64 packed fp4 data bytes, then 64 x 4 scale bytes (the layout the paged
# FP4 MQA-logits kernels read in place). A 68-byte row is `data[64] ++ scale[4]`; flat row `r` of the pool is
# row `r % 64` of native page `r // 64` (a GLOBAL page is consecutive native pages).
_NATIVE_ROWS, _DATA, _PAGE = 64, 64, 64 * 68


def _footer_offsets(addr: torch.Tensor):
    page, i = torch.div(addr, _NATIVE_ROWS, rounding_mode="floor"), addr % _NATIVE_ROWS
    base = page * _PAGE
    d = (base + i * _DATA).unsqueeze(-1) + torch.arange(_DATA, device=addr.device)
    s = (base + _NATIVE_ROWS * _DATA + i * 4).unsqueeze(-1) + torch.arange(4, device=addr.device)
    return d, s


def _scatter_is_valid(pool, addr, rows, keep, name):
    assert addr.dim() == 1 and rows.shape[0] == addr.shape[0] and keep.shape == addr.shape, (
        "addr [n], rows [n, w], keep [n]"
    )
    assert rows.dtype == pool.dtype, (
        f"rows are {rows.dtype} but the pool is {pool.dtype}: the write would silently convert"
    )
    assert pool.is_contiguous(), (
        f"{name}: the pool must be contiguous (it is written through a flat view)"
    )
    ok = keep & (addr >= 0)
    # Kept entries sharing an address must carry the same bytes, or which write wins is undefined.
    a, r = addr[ok], rows[ok]
    order = torch.argsort(a, stable=True)
    a, r = a[order], r[order]
    same = a[1:] == a[:-1]
    assert bool((r[1:][same] == r[:-1][same]).all()), (
        f"{name}: kept entries with one address carry different rows (the winner is undefined)"
    )


class Csa2RowAddr(OpWrapper):
    """Flat row of logical row `row` of request `rid`: `table[rid, row // rpp] * rpp + row % rpp`.

    `table` is int32 `[requests, blocks]` with a negative entry for an unmapped block; the result is
    -1 for those, for negative `row`, and for blocks past the table.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="tiny table with unmapped blocks, negative rows and rows past the table",
            spec=dict(requests=2, blocks=3, rows_per_page=4, n=24, kind="edges"),
        ),
        Cell(
            why="SWA pool addressing: 128 rows per page, a decode batch of 8 requests",
            spec=dict(requests=8, blocks=40, rows_per_page=128, n=8, kind="random"),
        ),
        Cell(
            why="compressed pool at ratio 2: 64 rows per page, a prefill chunk of 4096 tokens",
            spec=dict(requests=2, blocks=64, rows_per_page=64, n=4096, kind="random"),
        ),
    )
    note = "Pure index arithmetic: any table dtype the engine produces (int32) is certified."

    def __call__(
        self, table: torch.Tensor, rid: torch.Tensor, row: torch.Tensor, rows_per_page: int
    ) -> torch.Tensor:
        blk = torch.div(row, rows_per_page, rounding_mode="floor").long()
        if table.shape[1] == 0:  # no block is mapped: every row is outside the table
            return torch.full(blk.shape, -1, dtype=torch.int64, device=row.device)
        inside = (row >= 0) & (blk < table.shape[1])
        page = table[rid.long(), blk.clamp(0, table.shape[1] - 1)].long()
        addr = page * rows_per_page + (row.long() - blk * rows_per_page)
        return torch.where(inside & (page >= 0), addr, torch.full_like(addr, -1))

    def reference(
        self, table: torch.Tensor, rid: torch.Tensor, row: torch.Tensor, rows_per_page: int
    ) -> torch.Tensor:
        """One element at a time, in Python integers."""
        t = table.cpu().tolist()
        out = []
        for r, x in zip(rid.cpu().tolist(), row.cpu().tolist()):
            blk, off = divmod(x, rows_per_page)
            if x < 0 or blk >= len(t[r]) or t[r][blk] < 0:
                out.append(-1)
            else:
                out.append(t[r][blk] * rows_per_page + off)
        return torch.tensor(out, dtype=torch.int64, device=row.device)

    def is_valid(
        self, table: torch.Tensor, rid: torch.Tensor, row: torch.Tensor, rows_per_page: int
    ) -> None:
        assert table.dim() == 2, "table is [requests, blocks]"
        assert not row.dtype.is_floating_point, (
            "row must be integer: a fractional row is floored into a wrong address"
        )
        assert rows_per_page > 0


class Csa2GatherRows(OpWrapper):
    """`pool` is `[pages, rows, w]` (any dtype); rows of flat address `addr` (any shape); -1 gives zeros."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="fp32 compressor-state rows [pages, 128, 512] with -1 addresses",
            spec=dict(pages=6, rows=128, w=512, dtype=torch.float32, n=40),
        ),
        Cell(
            why="uint8 SWA byte rows (528 wide), 2-D address grid",
            spec=dict(pages=6, rows=128, w=528, dtype=torch.uint8, n=(16, 8)),
        ),
    )
    note = "Addresses past the pool are not certified (a device-side assert on CUDA)."

    def __call__(self, pool: torch.Tensor, addr: torch.Tensor) -> torch.Tensor:
        flat = pool.view(-1, pool.shape[-1])
        out = flat[addr.clamp(min=0).reshape(-1)].view(*addr.shape, pool.shape[-1])
        return out.masked_fill((addr < 0).unsqueeze(-1), 0)

    def reference(self, pool: torch.Tensor, addr: torch.Tensor) -> torch.Tensor:
        flat = pool.reshape(-1, pool.shape[-1])
        out = torch.zeros(*addr.shape, pool.shape[-1], dtype=pool.dtype, device=pool.device)
        for idx in torch.nonzero(addr >= 0).tolist():
            out[tuple(idx)] = flat[int(addr[tuple(idx)])]
        return out

    def is_valid(self, pool: torch.Tensor, addr: torch.Tensor) -> None:
        assert not addr.dtype.is_floating_point and addr.dtype != torch.bool, "addr must be integer"
        assert pool.is_contiguous(), "the pool is read through a flat view"


@triton.jit
def _scatter_rows_kernel(
    POOL, ADDR, ROWS, KEEP, row_stride, WORDS: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.program_id(0)
    a = tl.load(ADDR + i).to(tl.int64)
    ok = (tl.load(KEEP + i) != 0) & (a >= 0)
    offs = tl.arange(0, BLOCK)
    m = (offs < WORDS) & ok
    v = tl.load(ROWS + i.to(tl.int64) * row_stride + offs, mask=m)
    tl.store(POOL + a * WORDS + offs, v, mask=m)


def _words(t: torch.Tensor) -> torch.Tensor:
    """`t` (rows of unit column stride) as 32-bit words when its row bytes allow, else bytes (bit copies either way)."""
    b = t.view(torch.uint8) if t.dtype.itemsize == 1 else t
    nbytes = t.shape[-1] * t.dtype.itemsize
    if nbytes % 4 == 0:
        return b.view(torch.int32) if t.dtype.itemsize != 4 else t.view(torch.int32)
    return t.view(torch.uint8)


class Csa2ScatterRows(OpWrapper):
    """Write `rows[i]` at flat address `addr[i]` where `keep[i]` and `addr[i] >= 0`; in place on `pool`.

    One launch, a program per entry that copies its row only when kept (no host read, no full-width
    temporaries): static shapes, CUDA-graph safe. Kept entries sharing an address carry identical bytes.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="SWA write of a decode step: 8 rows, some masked out of the window",
            spec=dict(pages=8, rows=128, w=528, dtype=torch.uint8, n=8, masked=0.25),
        ),
        Cell(
            why="compressor-state rows, fp32, first entry skipped (the redirect target is not entry 0)",
            spec=dict(pages=4, rows=128, w=512, dtype=torch.float32, n=16, masked="first"),
        ),
        Cell(
            why="a 4096-token prefill chunk with half the rows outside the window",
            spec=dict(pages=40, rows=128, w=528, dtype=torch.uint8, n=4096, masked=0.5),
        ),
    )
    note = "Entries with the same address must carry the same bytes; `is_valid` states it."

    def __call__(
        self, pool: torch.Tensor, addr: torch.Tensor, rows: torch.Tensor, keep: torch.Tensor
    ) -> torch.Tensor:
        n = addr.shape[0]
        if n:
            flat = _words(pool.view(-1, pool.shape[-1]))
            src = rows if rows.stride(-1) == 1 else rows.contiguous()
            words = _words(src)
            w = flat.shape[-1]
            with torch.cuda.device(pool.device):
                k = keep.contiguous()
                _scatter_rows_kernel[(n,)](
                    flat,
                    addr.contiguous(),
                    words,
                    k.view(torch.uint8) if k.dtype == torch.bool else k,
                    words.stride(0),
                    WORDS=w,
                    BLOCK=triton.next_power_of_2(w),
                )
        return pool

    def reference(
        self, pool: torch.Tensor, addr: torch.Tensor, rows: torch.Tensor, keep: torch.Tensor
    ) -> torch.Tensor:
        """Entry by entry, in order, on `pool` itself (pass a copy)."""
        flat = pool.view(-1, pool.shape[-1])
        for i in range(addr.shape[0]):
            if bool(keep[i]) and int(addr[i]) >= 0:
                flat[int(addr[i])] = rows[i]
        return pool

    def is_valid(
        self, pool: torch.Tensor, addr: torch.Tensor, rows: torch.Tensor, keep: torch.Tensor
    ) -> None:
        _scatter_is_valid(pool, addr, rows, keep, "csa2_scatter_rows")


class Csa2GatherRowsFooter(OpWrapper):
    """Rows `[..., 68]` of flat row address `addr` from a native footer-page pool (uint8); -1 gives zeros."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="rows across the first three native pages, including page edges 63 | 64",
            spec=dict(pages=3, n=6, addrs=(0, 1, 63, 64, 130, -1)),
        ),
        Cell(
            why="a decode-step read: 128 query rows' worth of addresses over 4 GLOBAL pages",
            spec=dict(pages=8, n=128, addrs="random"),
        ),
    )
    note = "Pools whose size is not a whole number of 64-row native pages are not certified."

    def __call__(self, pool: torch.Tensor, addr: torch.Tensor) -> torch.Tensor:
        flat = pool.reshape(-1)
        d, s = _footer_offsets(addr.clamp(min=0))
        out = torch.cat(
            [
                flat[d.reshape(-1)].view(*addr.shape, _DATA),
                flat[s.reshape(-1)].view(*addr.shape, 4),
            ],
            dim=-1,
        )
        return out.masked_fill((addr < 0).unsqueeze(-1), 0)

    def reference(self, pool: torch.Tensor, addr: torch.Tensor) -> torch.Tensor:
        flat = pool.reshape(-1)
        out = torch.zeros(*addr.shape, 68, dtype=torch.uint8, device=pool.device)
        for idx in torch.nonzero(addr >= 0).tolist():
            a = int(addr[tuple(idx)])
            page, i = divmod(a, _NATIVE_ROWS)
            base = page * _PAGE
            out[tuple(idx)][:_DATA] = flat[base + i * _DATA : base + (i + 1) * _DATA]
            sb = base + _NATIVE_ROWS * _DATA + i * 4
            out[tuple(idx)][_DATA:] = flat[sb : sb + 4]
        return out

    def is_valid(self, pool: torch.Tensor, addr: torch.Tensor) -> None:
        assert pool.dtype == torch.uint8 and pool.numel() % _PAGE == 0, (
            "a footer pool is uint8 bytes in whole 64-row native pages"
        )
        assert not addr.dtype.is_floating_point and addr.dtype != torch.bool


class Csa2ScatterRowsFooter(OpWrapper):
    """`csa2_scatter_rows` for a native footer-page pool.

    Static shapes: a skipped entry is redirected to the first kept one (same address, same bytes); when no
    entry is kept, every entry rewrites the bytes already at its own (clamped) row, so the pool is unchanged.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="rows on both sides of the 63 | 64 native-page edge",
            spec=dict(pages=3, n=5, addrs=(0, 1, 63, 64, 130), keep="all"),
        ),
        Cell(
            why="a skipped entry between kept ones leaves its row untouched",
            spec=dict(pages=2, n=3, addrs=(0, 1, 2), keep="middle_skipped"),
        ),
        Cell(
            why="a ratio-1 prefill chunk: 2048 consecutive rows, the first skipped",
            spec=dict(pages=40, n=2048, addrs="consecutive", keep="first_skipped"),
        ),
    )
    note = "Pools whose size is not a whole number of 64-row native pages are not certified."

    def __call__(
        self, pool: torch.Tensor, addr: torch.Tensor, rows: torch.Tensor, keep: torch.Tensor
    ) -> torch.Tensor:
        ok = keep & (addr >= 0)
        any_ok = ok.any()
        first = torch.argmax(ok.to(torch.int8)).reshape(1)
        a = torch.where(
            any_ok, torch.where(ok, addr, addr.index_select(0, first)), addr.clamp(min=0)
        )
        flat = pool.view(-1)
        d, s = _footer_offsets(a)
        old = torch.cat(
            [flat[d.reshape(-1)].view(-1, _DATA), flat[s.reshape(-1)].view(-1, 4)], dim=-1
        )
        r = torch.where(
            any_ok, torch.where(ok.unsqueeze(-1), rows, rows.index_select(0, first)), old
        )
        flat[d.reshape(-1)] = r[:, :_DATA].reshape(-1)
        flat[s.reshape(-1)] = r[:, _DATA:].reshape(-1)
        return pool

    def reference(
        self, pool: torch.Tensor, addr: torch.Tensor, rows: torch.Tensor, keep: torch.Tensor
    ) -> torch.Tensor:
        flat = pool.view(-1)
        for i in range(addr.shape[0]):
            if bool(keep[i]) and int(addr[i]) >= 0:
                page, j = divmod(int(addr[i]), _NATIVE_ROWS)
                base = page * _PAGE
                flat[base + j * _DATA : base + (j + 1) * _DATA] = rows[i, :_DATA]
                sb = base + _NATIVE_ROWS * _DATA + j * 4
                flat[sb : sb + 4] = rows[i, _DATA:]
        return pool

    def is_valid(
        self, pool: torch.Tensor, addr: torch.Tensor, rows: torch.Tensor, keep: torch.Tensor
    ) -> None:
        assert pool.dtype == torch.uint8 and pool.numel() % _PAGE == 0
        assert rows.shape[-1] == 68, "footer rows are 68 bytes: 64 data + 4 scale"
        _scatter_is_valid(pool, addr, rows, keep, "csa2_scatter_rows_footer")


csa2_row_addr = Csa2RowAddr()
csa2_gather_rows = Csa2GatherRows()
csa2_scatter_rows = Csa2ScatterRows()
csa2_gather_rows_footer = Csa2GatherRowsFooter()
csa2_scatter_rows_footer = Csa2ScatterRowsFooter()
