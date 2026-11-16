# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The wide decode q_b projection and the DSA indexer's wq_b projection in one GEMM: both read
the same normalized q_a, so one launch writes q and the indexer's q."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.gemm import self_attn_q_b_ix_wq_b_bf16 as _kernel

#: The cells: the MTP3 / MTP5 wide decode row counts, the q_b / wq_b projections' MTP cells.
#: A row count between 64 and 192 runs on the next cell up (`at_rows`), as those cells do.
MTP_ROWS = _kernel.MTP_DECODE_M_VALUES
#: Output features of each projection; their shared input features.
N, K = _kernel.N, _kernel.K


@dataclass(frozen=True)
class QbIxQbRunner:
    """One compiled cell launched on `rows` rows: `launch(q_a, w2, out2)`."""

    rows: int
    launch: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], None]


def stack_weight(q_b: torch.Tensor, ix_wq_b: torch.Tensor) -> torch.Tensor:
    """`[q_b ; wq_b]` as a new contiguous bf16 `[2, 4096, 2048]` tensor: q_b's row-major
    `[4096, 2048]` weight (this rank's heads), then the indexer's wq_b."""
    for name, w in (("q_b", q_b), ("ix_wq_b", ix_wq_b)):
        if tuple(w.shape) != (N, K) or w.dtype != torch.bfloat16:
            raise ValueError(
                f"stack_weight takes bf16 [{N}, {K}] weights; {name} is {w.dtype} {tuple(w.shape)}"
            )
    return torch.stack([q_b, ix_wq_b]).contiguous()


def build(rows: int, device: torch.device | int) -> QbIxQbRunner:
    """Compile the cell at `rows` (one of `MTP_ROWS`) and launch it once on zeros, so the
    module loads before any CUDA-graph capture. Raises on a row count without a cell or a
    failed compile."""
    rows = int(rows)
    if rows not in MTP_ROWS:
        raise ValueError(f"q_b | wq_b fold: no cell at {rows} rows; cells are {MTP_ROWS}")
    dev = torch.device("cuda", device) if isinstance(device, int) else device
    with torch.cuda.device(dev):
        compiled = _kernel.compile(rows, dev)
        compiled.launch_rows(
            torch.zeros([rows, K], dtype=torch.bfloat16, device=dev),
            torch.zeros([2, N, K], dtype=torch.bfloat16, device=dev),
            torch.empty([2, rows, N], dtype=torch.bfloat16, device=dev),
        )
        torch.cuda.synchronize(dev)
    return QbIxQbRunner(rows, compiled.launch_rows)


def build_all(
    device: torch.device | int, rows: tuple[int, ...] = MTP_ROWS
) -> dict[int, QbIxQbRunner]:
    """`build` at every row count in `rows`, `{rows: runner}`. Raises on any failure (the
    target builds this under `_try_build` and then keeps the two GEMMs)."""
    return {int(m): build(m, device) for m in rows}


def at_rows(runner: QbIxQbRunner, rows: int) -> QbIxQbRunner | None:
    """`runner`'s cell launched on `rows` rows (`1..runner.rows`): each row's outputs are
    bitwise the cell's for that row. None outside that range. Builds nothing."""
    rows = int(rows)
    if not 0 < rows <= runner.rows:
        return None
    return QbIxQbRunner(rows, runner.launch)


class CuteDslQbIxQb(OpWrapper):
    """`(q, ix_q) = (q_a @ w2[0].T, q_a @ w2[1].T)` into `out2[0]`, `out2[1]`.

    `q_a` is the bf16 normalized `[M, 2048]` (unit stride along K); `w2` the stacked bf16
    `[2, 4096, 2048]` weight (`stack_weight`); `out2` a bf16 `[2, M, 4096]` buffer. Each output
    element is one fp32 accumulation over K in the q_b / wq_b MTP cells' order, rounded once
    to bf16: bit for bit those cells' outputs at the same rows. Returns `(out2[0], out2[1])`.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="MTP5 full decode batch: 32 x 6 rows", spec=dict(rows=192, cell=192)),
        Cell(why="MTP3 full decode batch: 32 x 4 rows", spec=dict(rows=128, cell=128)),
        Cell(why="MTP5 at 25 sequences (150 rows) on the 192 cell", spec=dict(rows=150, cell=192)),
        Cell(why="one row past the 128 cell, on the 192 cell", spec=dict(rows=129, cell=192)),
        Cell(why="191 rows on the 192 cell: one row short", spec=dict(rows=191, cell=192)),
        Cell(why="65 rows on the 128 cell", spec=dict(rows=65, cell=128)),
        Cell(why="MTP3 at 17 sequences (68 rows) on the 128 cell", spec=dict(rows=68, cell=128)),
        Cell(why="97 rows: an odd count inside the 128 cell", spec=dict(rows=97, cell=128)),
    )

    note = """
    `build_all(device)` compiles the 128 and 192-row cells; `at_rows(cell, rows)` runs a cell
    on fewer rows. The outputs equal the q_b and wq_b MTP cells' (`cute_dsl_projection`'s
    `q_b` / `ix_qb` at 128 / 192 rows and their re-pinned runners in between) bit for bit.
    """

    def __call__(
        self,
        q_a: torch.Tensor,
        w2: torch.Tensor,
        out2: torch.Tensor,
        runner: QbIxQbRunner,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        runner.launch(q_a, w2, out2)
        return out2[0], out2[1]

    def reference(
        self,
        q_a: torch.Tensor,
        w2: torch.Tensor,
        out2: torch.Tensor,
        runner: QbIxQbRunner,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(q, ix_q)` from fp64 products rounded once to bf16; `out2` and `runner` unread."""
        x = q_a.double()
        return (
            (x @ w2[0].double().t()).to(torch.bfloat16),
            (x @ w2[1].double().t()).to(torch.bfloat16),
        )


cute_dsl_q_b_ix_wq_b = CuteDslQbIxQb()
