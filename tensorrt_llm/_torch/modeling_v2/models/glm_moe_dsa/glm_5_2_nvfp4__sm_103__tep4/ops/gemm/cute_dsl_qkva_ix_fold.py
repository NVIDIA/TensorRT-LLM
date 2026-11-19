# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The wide decode qkva projection with the DSA indexer's key / head-weight projection folded
in: one GEMM writes the qkva output and the indexer's keys and head weights."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.gemm import self_attn_kv_a_proj_ix_fold_bf16 as _kernel

#: The fold's cells: the MTP3 / MTP5 wide decode row counts (32 sequences x 4 / 6), the qkva
#: projection's MTP cells. A row count between 64 and 192 runs on the next cell up
#: (`at_rows`), as the qkva cells do.
MTP_ROWS = _kernel.MTP_DECODE_M_VALUES
#: qkva output columns; the indexer's key and head-weight columns; the hidden size.
QKVA_N, IX_KEY, IX_HEADS, K = _kernel.QKVA_N, _kernel.IX_KEY, _kernel.IX_HEADS, _kernel.K
#: The stacked weight's first indexer row (zero rows pad qkva up to it) and its row count.
IX_ROW0, N_TOTAL = _kernel.IX_ROW0, _kernel.N_TOTAL


@dataclass(frozen=True)
class QkvaIxFoldRunner:
    """One compiled fold cell launched on `rows` rows.

    `launch(x, w_ext, out)` runs the cell's compiled instance on `x`'s rows (at most the
    cell's); `kb` / `wq` are the cell's own persistent side buffers (bf16 `[cell rows, 128]`,
    fp32 `[cell rows, 32]`, compiled into the kernel), which every call overwrites."""

    rows: int
    launch: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], object]
    kb: torch.Tensor
    wq: torch.Tensor


def stack_weight(qkva: torch.Tensor, ix: torch.Tensor) -> torch.Tensor:
    """`[qkva ; 0 ; ix]`, a new contiguous bf16 `[2848, 6144]` tensor: the qkva projection's
    row-major `[2624, 6144]` weight, 64 zero rows, then from row `IX_ROW0` the indexer's bf16
    `[key (128) | head weights (32), 6144]` rows (`ix_fused_bf16`)."""
    if tuple(qkva.shape) != (QKVA_N, K) or tuple(ix.shape) != (IX_KEY + IX_HEADS, K):
        raise ValueError(
            f"stack_weight takes qkva [{QKVA_N}, {K}] and ix [{IX_KEY + IX_HEADS}, {K}], got "
            f"{tuple(qkva.shape)} and {tuple(ix.shape)}"
        )
    if qkva.dtype != torch.bfloat16 or ix.dtype != torch.bfloat16:
        raise ValueError(f"stack_weight takes bf16 weights, got {qkva.dtype} / {ix.dtype}")
    pad = torch.zeros(IX_ROW0 - QKVA_N, K, dtype=qkva.dtype, device=qkva.device)
    return torch.cat([qkva, pad, ix], 0).contiguous()


def build(rows: int, device: torch.device | int) -> QkvaIxFoldRunner:
    """Compile the fold at `rows` (one of `MTP_ROWS`) with its own side buffers, and launch
    it once on zeros so the module loads before any CUDA-graph capture. Raises if `rows`
    has no cell or the compile fails."""
    rows = int(rows)
    if rows not in MTP_ROWS:
        raise ValueError(f"qkva indexer fold: no cell at {rows} rows; cells are {MTP_ROWS}")
    dev = torch.device("cuda", device) if isinstance(device, int) else device
    with torch.cuda.device(dev):
        kb = torch.zeros([rows, IX_KEY], dtype=torch.bfloat16, device=dev)
        wq = torch.zeros([rows, IX_HEADS], dtype=torch.float32, device=dev)
        compiled = _kernel.compile(rows, kb, wq)
        x = torch.zeros([rows, K], dtype=torch.bfloat16, device=dev)
        w = torch.zeros([N_TOTAL, K], dtype=torch.bfloat16, device=dev)
        out = torch.empty([rows, QKVA_N], dtype=torch.bfloat16, device=dev)
        compiled.launch_rows(x, w, out)
        torch.cuda.synchronize(dev)
    return QkvaIxFoldRunner(rows, compiled.launch_rows, kb, wq)


def build_all(
    device: torch.device | int, rows: tuple[int, ...] = MTP_ROWS
) -> dict[int, QkvaIxFoldRunner]:
    """`build` at every row count in `rows`, `{rows: runner}`. Raises on any failure: the
    target then keeps the two GEMMs (it builds this under `_try_build`)."""
    return {int(m): build(m, device) for m in rows}


def at_rows(runner: QkvaIxFoldRunner, rows: int) -> QkvaIxFoldRunner | None:
    """`runner`'s cell launched on `rows` rows (`1..runner.rows`), the same compiled
    instance and side buffers: each row's outputs are bitwise the cell's for that row. None
    outside that range. Builds nothing (safe inside a CUDA-graph capture)."""
    rows = int(rows)
    if not 0 < rows <= runner.rows:
        return None
    return QkvaIxFoldRunner(rows, runner.launch, runner.kb, runner.wq)


class CuteDslQkvaIxFold(OpWrapper):
    """`x @ w_ext.T` split three ways: `out` (the qkva projection), keys, head weights.

    `x` is the bf16 normalized hidden state `[M, 6144]` (contiguous); `w_ext` the stacked
    bf16 `[2848, 6144]` weight (`stack_weight`). Writes `out[:, :] = (x @ w_ext[:2624].T)` in
    bf16 -- bit for bit the qkva projection's MTP cell at the same cell and rows -- and
    returns `(out, key, head_weights)`: `key = bf16(x @ w_ext[2688:2816].T)` `[M, 128]` and
    `head_weights = x @ w_ext[2816:].T` in fp32 `[M, 32]`, accumulated as the split-output
    indexer projection (`cute_dsl_indexer_pre_proj_split`) accumulates them: 8 fp32 chains of
    exact bf16 products summed in chain order. `key` / `head_weights` are views of the
    runner's persistent buffers: the next call overwrites them.
    """

    ARCHS = frozenset({Arch.SM_103})

    # GLM-5.2 MTP3 / MTP5 wide decode steps: the 128 / 192-row batches and the partial
    # batches between 64 and 192 rows on the next cell up.
    CELLS: tuple[Cell, ...] = (
        Cell(why="MTP5 full decode batch: 32 x 6 rows", spec=dict(rows=192, cell=192)),
        Cell(why="MTP3 full decode batch: 32 x 4 rows", spec=dict(rows=128, cell=128)),
        Cell(
            why="MTP5 at 25 sequences (150 rows) on the 192 cell: ragged last tile",
            spec=dict(rows=150, cell=192),
        ),
        Cell(why="one row past the 128 cell, on the 192 cell", spec=dict(rows=129, cell=192)),
        Cell(why="191 rows on the 192 cell: one row short", spec=dict(rows=191, cell=192)),
        Cell(
            why="65 rows (one past the decode cells) on the 128 cell", spec=dict(rows=65, cell=128)
        ),
        Cell(why="MTP3 at 17 sequences (68 rows) on the 128 cell", spec=dict(rows=68, cell=128)),
        Cell(why="97 rows: an odd count inside the 128 cell", spec=dict(rows=97, cell=128)),
    )

    note = """
    `build_all(device)` compiles the 128 and 192-row cells (each with its own side
    buffers); `at_rows(cell, rows)` runs a cell on fewer rows. The qkva part is the qkva
    projection's MTP cell's schedule, so its bytes equal that cell's; the indexer columns
    follow the split-output indexer projection's (`cute_dsl_indexer_pre_proj_split`) chains
    and reduction order. The side buffers are fixed per cell: consume the returned
    `key` / `head_weights` before the next call on the same cell.
    """

    def __call__(
        self,
        x: torch.Tensor,
        w_ext: torch.Tensor,
        out: torch.Tensor,
        runner: QkvaIxFoldRunner,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = int(x.shape[0])
        runner.launch(x, w_ext, out)
        return out, runner.kb[:rows], runner.wq[:rows]

    def reference(
        self,
        x: torch.Tensor,
        w_ext: torch.Tensor,
        out: torch.Tensor,
        runner: QkvaIxFoldRunner,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(qkva bf16, key bf16, head weights fp32)` from one fp64 product, each rounded
        once; `out` and `runner` are not read."""
        full = x.double() @ w_ext.double().t()
        qkva = full[:, :QKVA_N]
        key, heads = torch.split(full[:, IX_ROW0:], [IX_KEY, IX_HEADS], -1)
        return qkva.to(torch.bfloat16), key.to(torch.bfloat16), heads.float()


cute_dsl_qkva_ix_fold = CuteDslQkvaIxFold()
