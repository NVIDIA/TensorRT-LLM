# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-step SWA window slots of every layer, in one launch.

Wraps the engine's `kernel.refresh_swa_slots` (CuTe DSL `SwaRefreshKernel`). For every query row it
resolves, through the layer's own page table, the flat pool row of each of the `window` positions
ending at the row's position (`reads`), the row of the position itself (`writes`) and the number of
compressed rows visible to it at the layer's ratio (`visible`). Doing this once per step replaces the
per-layer index arithmetic (about twenty small launches per layer).
"""

from __future__ import annotations

from typing import Tuple

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


def csa2_swa_table_descriptors(tables: torch.Tensor) -> torch.Tensor:
    """`[S, requests, blocks]` int32 page tables of S layers -> the kernel's int64 `[S, 4]` descriptors
    `(pointer, row stride, rows, columns)`.

    A setup helper, not an entry: the descriptors hold raw pointers, so the target builds them once,
    after the tables' storage is allocated, and keeps both alive together.
    """
    assert tables.dtype == torch.int32 and tables.dim() == 3 and tables.stride(2) == 1
    rows = [
        (tables[i].data_ptr(), tables.stride(1), tables.shape[1], tables.shape[2])
        for i in range(tables.shape[0])
    ]
    return torch.tensor(rows, dtype=torch.int64).to(tables.device)


class Csa2SwaSlots(OpWrapper):
    """`reads [S, n, window]`, `writes [S, n]` (int64 flat pool rows, -1 = none) and `visible [L, n]` (int64).

    `pos` int32 `[n]` is the position of each query row and `rid` int64 `[n]` its page-table row. `tables` is the
    S layers' `[S, requests, blocks]` int32 page tables (`block` pool rows per page, 128 at every cell) and `desc`
    their `csa2_swa_table_descriptors` (the kernel reads the tables through it); `floors` int64 `[requests]` are the
    lowest positions read or written per request (0 for a request that never rolled its window); `ratios` int32 `[L]`
    the compression ratio of every layer (L >= S; layers past S own no SWA pool and get only `visible`).
    `reads[s, i, c]` is the pool row of position `pos[i] - window + 1 + c` (-1 before position 0 or on an unmapped
    page), `writes[s, i]` that of `pos[i]` itself, `visible[l, i]` is `(pos[i] + 1) // ratio` (0 at ratio 0).
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="tep8 decode: one request, six rows (1 + 5 drafts), history inside the first page",
            spec=dict(layers=6, swa_layers=6, nreq=1, rows=6, start=(100,), window=128),
        ),
        Cell(
            why="tep8 decode: four requests of six rows with histories across page edges, 43 layers",
            spec=dict(
                layers=43, swa_layers=43, nreq=4, rows=6, start=(5, 127, 300, 5000), window=128
            ),
        ),
        Cell(
            why="dep8 decode: 320 requests of four rows (1280 rows), unmapped pages, 43 layers",
            spec=dict(layers=43, swa_layers=43, nreq=320, rows=4, start="random", window=128),
        ),
        Cell(
            why="dep32 decode: 25 requests of six rows (150 rows), a layer without SWA pool",
            spec=dict(layers=8, swa_layers=6, nreq=25, rows=6, start="random", window=128),
        ),
    )
    note = "Pure index arithmetic: every cell is compared bit for bit with the reference."

    def __call__(
        self,
        pos: torch.Tensor,
        rid: torch.Tensor,
        tables: torch.Tensor,
        desc: torch.Tensor,
        ratios: torch.Tensor,
        floors: torch.Tensor,
        window: int,
        block: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import refresh_swa_slots

        n, swa_layers, layers = pos.shape[0], desc.shape[0], ratios.shape[0]
        dev = pos.device
        reads = torch.empty(swa_layers, n, window, dtype=torch.int64, device=dev)
        writes = torch.empty(swa_layers, n, dtype=torch.int64, device=dev)
        visible = torch.empty(layers, n, dtype=torch.int64, device=dev)
        with torch.cuda.device(dev):
            refresh_swa_slots(pos, rid, floors, floors, desc, ratios, reads, writes, visible, block)
        return reads, writes, visible

    def reference(
        self,
        pos: torch.Tensor,
        rid: torch.Tensor,
        tables: torch.Tensor,
        desc: torch.Tensor,
        ratios: torch.Tensor,
        floors: torch.Tensor,
        window: int,
        block: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Plain torch, one layer at a time: window positions, page lookup, validity masks."""
        dev = pos.device
        p, r = pos.long(), rid.long()
        logical = p[:, None] - window + 1 + torch.arange(window, device=dev)  # [n, window]
        blk, off = torch.div(logical, block, rounding_mode="floor"), logical % block
        reads, writes = [], []
        for t in tables:
            inside = (logical >= 0) & (blk < t.shape[1]) & (logical >= floors[r][:, None])
            page = t[r[:, None], blk.clamp(0, t.shape[1] - 1)].long()
            slot = torch.where(inside & (page >= 0), page * block + off, -1)
            reads.append(slot)
            writes.append(slot[:, -1])
        ratio = ratios.long()[:, None]
        visible = torch.where(ratio > 0, (p[None] + 1) // ratio.clamp(min=1), 0)
        return torch.stack(reads), torch.stack(writes), visible

    def is_valid(
        self,
        pos: torch.Tensor,
        rid: torch.Tensor,
        tables: torch.Tensor,
        desc: torch.Tensor,
        ratios: torch.Tensor,
        floors: torch.Tensor,
        window: int,
        block: int,
    ) -> None:
        assert pos.dtype == torch.int32 and pos.is_contiguous(), "pos is contiguous int32 [n]"
        assert rid.dtype == torch.int64 and rid.is_contiguous() and rid.shape == pos.shape, (
            "rid is int64 [n]"
        )
        assert tables.dtype == torch.int32 and tables.dim() == 3 and tables.stride(2) == 1, (
            "tables are int32 [S, requests, blocks]"
        )
        assert desc.dtype == torch.int64 and desc.shape == (tables.shape[0], 4), (
            "desc is int64 [S, 4]"
        )
        assert desc[:, 0].tolist() == [tables[i].data_ptr() for i in range(tables.shape[0])], (
            "desc must point at these tables (rebuild it with csa2_swa_table_descriptors)"
        )
        assert ratios.dtype == torch.int32 and ratios.shape[0] >= desc.shape[0], (
            "ratios is int32 [L], L >= S"
        )
        assert floors.dtype == torch.int64 and floors.is_contiguous(), "floors is int64 [requests]"
        assert 0 < window <= 1024 and block > 0
        assert bool((rid >= 0).all()) and bool((rid < floors.numel()).all()), (
            "every row needs a request in [0, requests): the kernel treats others as unmapped"
        )


csa2_swa_slots = Csa2SwaSlots()
