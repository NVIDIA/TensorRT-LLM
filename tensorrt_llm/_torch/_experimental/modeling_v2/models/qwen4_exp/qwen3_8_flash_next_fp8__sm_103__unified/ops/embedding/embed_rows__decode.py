# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A decode step's token embedding in one launch: the rows of the table at the step's ids,
optionally with negative ids read as row 0, written `copies` times side by side (once per
hyper-connection stream).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

BLOCK = 512


# `ids` is a view of the step's token buffer at any row (a gather index, read by scalar loads).
@triton.jit(do_not_specialize_on_alignment=["ids"])
def _embed_rows_kernel(
    ids,
    table,
    out,
    stride_table,
    stride_out,
    H: tl.constexpr,
    COPIES: tl.constexpr,
    CLAMP: tl.constexpr,
    BLOCK_: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_ + tl.arange(0, BLOCK_)
    mask = cols < H
    token = tl.load(ids + row).to(tl.int64)
    if CLAMP:
        token = tl.maximum(token, 0)
    vals = tl.load(table + token * stride_table + cols, mask=mask)
    for c in tl.static_range(COPIES):
        tl.store(out + row.to(tl.int64) * stride_out + c * H + cols, vals, mask=mask)


class EmbedRows(OpWrapper):
    """`out = embed_rows(ids, table, copies, clamp)`: `[T, copies * H]`, block `c` of row `t`
    `table[ids[t]]` (`table[max(ids[t], 0)]` with `clamp`); `ids` `[T]` int32 / int64
    contiguous, `table` `[V, H]` with unit column stride. Bitwise the composed `F.embedding`
    path. Ids past the table (or negative without `clamp`) read out of bounds: the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(why="batch-4 verify: 12 rows over 4 streams", spec=dict(rows=12, copies=4, clamp=False)),
        Cell(why="draft decode step: 1 row, clamped", spec=dict(rows=1, copies=1, clamp=True)),
    )  # fmt: skip

    note = """
    Source: one Triton kernel (`_embed_rows_kernel`), grid `(T, ceil(H / 512))`, JIT-compiled per
    (H, copies, clamp, id dtype); the engine's warmup forward compiles it before capture. Copies
    only: bitwise the composed path for every in-range id.
    """

    def __call__(
        self, ids: torch.Tensor, table: torch.Tensor, copies: int, clamp: bool
    ) -> torch.Tensor:
        rows = ids.shape[0]
        hidden = table.shape[1]
        out = torch.empty([rows, copies * hidden], dtype=table.dtype, device=table.device)
        if rows:
            _embed_rows_kernel[(rows, triton.cdiv(hidden, BLOCK))](
                ids,
                table,
                out,
                table.stride(0),
                out.stride(0),
                H=hidden,
                COPIES=copies,
                CLAMP=clamp,
                BLOCK_=BLOCK,
                num_warps=4,
            )
        return out

    def reference(
        self, ids: torch.Tensor, table: torch.Tensor, copies: int, clamp: bool
    ) -> torch.Tensor:
        h = torch.nn.functional.embedding(ids.clamp_min(0) if clamp else ids, table)
        return torch.cat([h] * copies, -1)

    def is_valid(self, ids: torch.Tensor, table: torch.Tensor, copies: int, clamp: bool) -> None:
        assert ids.dim() == 1 and ids.is_contiguous() and ids.dtype in (torch.int32, torch.int64), (
            ids.shape,
            ids.dtype,
        )
        assert table.dim() == 2 and table.stride(1) == 1, (tuple(table.shape), table.stride())
        assert copies >= 1


embed_rows = EmbedRows()
