# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's per-query sliding-window index lists as one CuTe DSL launch."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.spec.dflash2_window_rows import dflash2_window_rows_into
from ..attention.dflash2_window import dflash2_window_rows as _torch_dflash2_window_rows

#: Kimi K3.1 DFlash2 drafter: sliding_window 1024, index lists as wide as the window.
_WINDOW = 1024


def _spec(batch: int, width: int, ctx: str, page_size: int = 64, **overrides) -> dict:
    spec = dict(
        batch=batch,
        width=width,
        ctx=ctx,
        page_size=page_size,
        window=_WINDOW,
        index_width=_WINDOW,
        dtype=torch.int64,
        spare_pages=4,
        table="own",
        ring=None,
    )
    spec.update(overrides)
    return spec


class K31DFlash2WindowRows(OpWrapper):
    """``(rows, lengths)``: each block query's causal sliding window as global pool rows.

    The query at position ``p = ctx_len[b] + j`` attends ``[max(0, p - window + 1), p]``; its
    list holds those positions' pool rows (``page * page_size + offset``, ascending) and ``-1``
    after its live length. With the worker's context ring (``ring_rows``, ``ops/spec/dflash2_ring``)
    position ``p`` is read from table row ``p mod ring_rows``. Bitwise the torch reference
    (``dflash2_window_rows``), which is what the drafter's window decode reads; int32 outputs,
    request-major query rows.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="c1, width 5, a 3-token context: the window floors at position 0",
            spec=_spec(1, 5, ctx="short"),
        ),
        Cell(
            why="c1, width 8, a long context: full 1024-row windows across 64-token pages",
            spec=_spec(1, 8, ctx="long"),
        ),
        Cell(
            why="64 requests at width 5, mixed lengths 0..20000, 64-token pages",
            spec=_spec(64, 5, ctx="mixed"),
        ),
        Cell(
            why="64 requests at width 8, mixed lengths, 32-token pages",
            spec=_spec(64, 8, ctx="mixed", page_size=32),
        ),
        Cell(
            why="contexts straddling the window edge: lengths 1023/1024 and full windows mixed",
            spec=_spec(7, 5, ctx="edge"),
        ),
        Cell(
            why="int32 lengths and page tables, positions past the table clamp to its last page",
            spec=_spec(5, 8, ctx="mixed", dtype=torch.int32, spare_pages=-3),
        ),
        Cell(
            why="page tables as strided rows of a wider shared table (no gather copy)",
            spec=_spec(9, 8, ctx="mixed", table="strided"),
        ),
        Cell(
            why="context ring, width 5, 64-token pages: 1088 rows per slot, mixed lengths wrap",
            spec=_spec(64, 5, ctx="mixed", ring=1088, spare_pages=0),
        ),
        Cell(
            why="context ring, width 8, 32-token pages, int32: windows straddling the wrap",
            spec=_spec(7, 8, ctx="edge", page_size=32, ring=1056, spare_pages=0, dtype=torch.int32),
        ),
        Cell(
            why="context ring, c1 width 5, a long context: every window wraps",
            spec=_spec(1, 5, ctx="long", ring=1088, spare_pages=0),
        ),
        Cell(
            why="sm_107 DEP16 generation: 12 requests a rank at width 8, mixed lengths",
            spec=_spec(12, 8, ctx="mixed"),
        ),
    )

    note = """
    TP and the drafter's head split do not change the lists (one set for all layers and heads),
    so the cells span the batch (1..64 requests), the width (5 / 8), the page size of the draft
    pool (32 / 64 tokens), a dense table and the context ring, and the integer dtypes the worker
    hands over. One launch in place of
    the reference's chain of index ops (see the test's launch count); no host synchronization.

    The kernel compiles once per (block_size, page_size, window, width, ring_rows, dtypes) on
    the first call, which must be eager; CUDA-graph capture replays the launch.
    """

    def __call__(
        self,
        ctx_len: torch.Tensor,
        page_tables: torch.Tensor,
        block_size: int,
        page_size: int,
        window: int,
        width: int,
        ring_rows: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if width < window:
            raise ValueError(f"index width {width} cannot hold a {window}-position window")
        if ring_rows and ring_rows < window + block_size - 1:
            raise ValueError(
                f"a {ring_rows}-row ring cannot hold a {window}-position window and its block"
            )
        num_rows = ctx_len.shape[0] * block_size
        rows = torch.empty(num_rows, width, dtype=torch.int32, device=ctx_len.device)
        lengths = torch.empty(num_rows, dtype=torch.int32, device=ctx_len.device)
        dflash2_window_rows_into(
            ctx_len, page_tables, block_size, page_size, window, rows, lengths, ring_rows or 0
        )
        return rows, lengths

    def reference(
        self,
        ctx_len: torch.Tensor,
        page_tables: torch.Tensor,
        block_size: int,
        page_size: int,
        window: int,
        width: int,
        ring_rows: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The drafter's own torch index lists (``ops/attention/dflash2_window.py``)."""
        return _torch_dflash2_window_rows(
            ctx_len, page_tables, block_size, page_size, window, width, ring_rows=ring_rows
        )

    def is_valid(
        self,
        ctx_len: torch.Tensor,
        page_tables: torch.Tensor,
        block_size: int,
        page_size: int,
        window: int,
        width: int,
        ring_rows: Optional[int] = None,
    ) -> None:
        assert ctx_len.dtype in (torch.int32, torch.int64), f"integer ctx_len, got {ctx_len.dtype}"
        assert page_tables.dtype in (torch.int32, torch.int64), (
            f"integer page tables, got {page_tables.dtype}"
        )
        assert ctx_len.dim() == 1 and page_tables.dim() == 2, "ctx_len [B], page_tables [B, pages]"
        assert page_tables.shape[0] == ctx_len.shape[0], (
            f"one page-table row per request: {page_tables.shape[0]} rows for "
            f"{ctx_len.shape[0]} requests"
        )
        # An empty table has no last page to clamp to: the reference gathers out of range.
        assert page_tables.shape[1] >= 1, "page tables need at least one page column"
        assert ring_rows is None or ring_rows >= 0, f"ring_rows must be >= 0, got {ring_rows}"


dflash2_window_rows = K31DFlash2WindowRows()
