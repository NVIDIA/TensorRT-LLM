# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-row paged-indexer descriptors of one index owner, in one launch.

Wraps the engine's `kernel.fill_indexer_descriptors` (CuTe DSL `IndexerDescriptorsKernel`): from the owner's
GLOBAL page table and each row's request and visible length it produces what the paged FP4 MQA-logits
kernels and the selection kernels read (the row's native block table, its context length, its visible
length and a per-page validity byte). One launch per owner and step replaces about ten small launches per
indexer layer. With `masked` the same launch also writes the engine's DeepGEMM-masked table (attention TP: the paged
logits kernel scores an admitted row's unallocated pages as -inf itself, no mask launch).
"""

from __future__ import annotations

from typing import Tuple

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_NATIVE_ROWS = 64  # rows of a native index page (the layout the paged logits kernels read)


class Csa2IndexerDescriptors(OpWrapper):
    """`(blocks [n, cap] int32, context [n, 1] int32, visible [n] int32, valid [n, cap] bool)`.

    `table` is the owner's `[requests, pages]` int32 GLOBAL page table (-1 = unmapped); a GLOBAL page holds
    `128 // ratio` rows, i.e. `sub = 128 // ratio // 64` native pages, so `cap = pages * sub`. `rid` int64 `[n]`
    is each row's request, `visible` int64 `[n]` its number of visible compressed rows (`(pos + 1) // ratio`).
    `blocks[i]` lists the native pages of the row's request (0 for an unmapped one), `valid[i]` marks the mapped
    ones, `visible[i]` is `visible` (0 for a row outside the table), `context[i]` is `max(visible, 1)`.

    `masked=True` appends `masked_blocks [n, cap]` int32: `blocks` with each unmapped page of a row inside the table
    flagged `DEEPGEMM_MASKED_PAGE` (`blocks | 1 << 30`), the table the engine's attention-TP indexer hands DeepGEMM's
    paged logits so that they score those columns -inf (`csa2_indexer_masked_block_table`, #22366).
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="tep8 decode, ratio 2: one request of six rows, a long history",
            spec=dict(ratio=2, nreq=1, rows=6, pages=40, unmapped=0.0),
        ),
        Cell(
            why="ratio 1: four requests of six rows, unmapped pages inside the table",
            spec=dict(ratio=1, nreq=4, rows=6, pages=96, unmapped=0.3),
        ),
        Cell(
            why="dep8 decode, ratio 2: 320 requests of four rows (1280 rows) over a deep table",
            spec=dict(ratio=2, nreq=320, rows=4, pages=500, unmapped=0.1),
        ),
        Cell(
            why="dep32 decode, ratio 1: 25 requests of six rows, rows without a request",
            spec=dict(ratio=1, nreq=25, rows=6, pages=300, unmapped=0.1, stray=True),
        ),
        Cell(
            why="tep8 decode, ratio 2, attention TP: the DeepGEMM-masked table too, unmapped pages, a stray row",
            spec=dict(ratio=2, nreq=1, rows=6, pages=80, unmapped=0.2, stray=True, masked=True),
        ),
        Cell(
            why="tep8 decode, ratio 1, attention TP: the DeepGEMM-masked table of four requests",
            spec=dict(ratio=1, nreq=4, rows=6, pages=96, unmapped=0.3, masked=True),
        ),
    )
    note = "Pure index arithmetic: every cell is compared bit for bit with the reference."

    def __call__(
        self,
        table: torch.Tensor,
        rid: torch.Tensor,
        visible: torch.Tensor,
        ratio: int,
        masked: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import (
            fill_indexer_descriptors,
        )

        n, dev = rid.shape[0], table.device
        sub = 128 // ratio // _NATIVE_ROWS
        cap = table.shape[1] * sub
        blocks = torch.empty(n, cap, dtype=torch.int32, device=dev)
        context = torch.empty(n, 1, dtype=torch.int32, device=dev)
        vis32 = torch.empty(n, dtype=torch.int32, device=dev)
        valid = torch.empty(n, cap, dtype=torch.bool, device=dev)
        masked_blocks = torch.empty(n, cap, dtype=torch.int32, device=dev) if masked else None
        with torch.cuda.device(dev):
            fill_indexer_descriptors(
                table, rid, visible, 0, sub, blocks, context, vis32, valid, masked_blocks
            )
        if masked:
            return blocks, context, vis32, valid, masked_blocks
        return blocks, context, vis32, valid

    def reference(
        self,
        table: torch.Tensor,
        rid: torch.Tensor,
        visible: torch.Tensor,
        ratio: int,
        masked: bool = False,
    ) -> Tuple[torch.Tensor, ...]:
        """Native page ids and validity by repeating each GLOBAL page `sub` times, then one gather per row."""
        sub = 128 // ratio // _NATIVE_ROWS
        requests = table.shape[0]
        native = table.long()[:, :, None] * sub + torch.arange(sub, device=table.device)
        native = native.reshape(requests, -1)
        ok = (table >= 0)[:, :, None].expand(-1, -1, sub).reshape(requests, -1)
        native = torch.where(ok, native, 0)
        row_ok = (rid >= 0) & (rid < requests)
        r = rid.clamp(0, requests - 1)
        blocks = torch.where(row_ok[:, None], native[r], 0).int()
        vis = torch.where(row_ok, visible, 0).int()
        valid = ok[r] & row_ok[:, None]
        if masked:  # the engine's torch path (`_fill_indexer_descriptors`): admitted rows' unbacked pages flagged
            from tensorrt_llm._torch.attention.backends.sparse.csa2.kernel import (
                DEEPGEMM_MASKED_PAGE,
            )

            unbacked = row_ok[:, None] & ~ok[r]
            flagged = torch.where(unbacked, blocks | DEEPGEMM_MASKED_PAGE, blocks)
            return blocks, vis.clamp_min(1)[:, None], vis, valid, flagged
        return blocks, vis.clamp_min(1)[:, None], vis, valid

    def is_valid(
        self,
        table: torch.Tensor,
        rid: torch.Tensor,
        visible: torch.Tensor,
        ratio: int,
        masked: bool = False,
    ) -> None:
        assert ratio in (1, 2), (
            "the index owners have compression ratio 1 or 2 (128 // ratio rows per page)"
        )
        assert (
            table.dtype == torch.int32
            and table.dim() == 2
            and (table.shape[1] <= 1 or table.stride(1) == 1)
        )
        assert (
            rid.dtype == torch.int64 and visible.dtype == torch.int64 and rid.shape == visible.shape
        )
        assert rid.is_contiguous() and visible.is_contiguous()


csa2_indexer_descriptors = Csa2IndexerDescriptors()
