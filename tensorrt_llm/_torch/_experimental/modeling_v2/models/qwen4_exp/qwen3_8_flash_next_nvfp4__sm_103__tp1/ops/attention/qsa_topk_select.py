# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA block selection: an exact per-row top-k over a caller-supplied column window."""

from __future__ import annotations

from typing import NamedTuple

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: This checkpoint's block budget: `indexer_budget` 2048 tokens over
#: `indexer_compress_ratio` 4. The op's own default, 2048, is the *token* budget.
_BLOCK_TOPK = 512
#: Score-row width the target hands over: a 260-page block table at 32 tokens
#: per block, compressed 4:1.
_SCORE_COLUMNS = 260 * 32 // 4


class TopkExpectation(NamedTuple):
    """What a correct selection must satisfy.

    The op's output order is unspecified, so `reference` cannot return the
    indices themselves; it returns each row's expected top values plus the
    frame the test needs to read the op's offsets back as values.
    """

    #: `[rows, index_topk]` fp32: the window's top values, descending, `-inf`-padded.
    values: torch.Tensor
    #: The logits as fp32 with every out-of-window column at `-inf`.
    logits: torch.Tensor
    #: int64 `row_starts`, to lift window-local offsets to columns.
    row_starts: torch.Tensor
    #: int64 window length per row, `max(row_ends - row_starts, 0)`.
    window: torch.Tensor


class QsaTopkSelect(OpWrapper):
    """Each row's top-`index_topk` window-local column offsets, written into `indices`.

    Wraps `torch.ops.trtllm.indexer_topk_prefill` (CUDA, compiled into the
    wheel). For row `r` it selects the `index_topk` largest values of
    `logits[r, row_starts[r]:row_ends[r]]` and writes their offsets *relative to
    `row_starts[r]`* into `indices[r]`: `min(index_topk, window)` distinct
    offsets as a prefix, `-1` in every remaining slot. Every slot of every row
    is written, so `indices` needs no pre-fill. Returns `indices`.

    Exact, not approximate: the kernel resolves the threshold over the full 32
    bits of the fp32 pattern. The **order of the offsets is unspecified** and
    not stable between identical launches (candidates are appended through a
    shared-memory atomic; measured at `[8, 2048]`, `k=512`: four launches, four
    permutations of one selected set). Ties are broken arbitrarily within the
    tied value.

    Fusion boundary: selection only. Scoring, the causal bound the caller
    encodes in `row_ends`, and the block-to-token expansion are outside.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The target calls this with `index_topk = 512` over the `[rows, 2080]`
    # score matrix `qsa_paged_index_scores` returns with
    # `only_visible_blocks=True` -- so every column at or past a row's
    # `row_end` is allocator garbage. Those cells poison it with NaN: a NaN
    # inside the window would be selected (see `note`), so they prove the
    # window is honoured, not merely that the tail happened to be small.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: one row at position 7777, 2080 cols, k=512, NaN tail",
            spec=dict(
                rows=1,
                cols=_SCORE_COLUMNS,
                index_topk=_BLOCK_TOPK,
                windows=("positions", (7777,)),
                logits="relu",
                tail="nan",
            ),
        ),
        Cell(
            why="decode MTP: one request's two rows (7777, 7778), NaN tail",
            spec=dict(
                rows=2,
                cols=_SCORE_COLUMNS,
                index_topk=_BLOCK_TOPK,
                windows=("positions", (7777, 7778)),
                logits="relu",
                tail="nan",
            ),
        ),
        Cell(
            why="decode MTP x4: two requests below the budget get -1 padding",
            spec=dict(
                rows=8,
                cols=_SCORE_COLUMNS,
                index_topk=_BLOCK_TOPK,
                windows=("positions", (7777, 7778, 4000, 4001, 1500, 1501, 300, 301)),
                logits="relu",
                tail="nan",
            ),
        ),
        Cell(
            why="context: 8191 rows at positions 0..8190 (warm-up capture)",
            spec=dict(
                rows=8191,
                cols=_SCORE_COLUMNS,
                index_topk=_BLOCK_TOPK,
                windows=("chunk", 0),
                logits="relu",
                tail="nan",
            ),
        ),
        Cell(
            why="context: 1024 rows from position 4000 over 2048 cols, relu ties",
            spec=dict(
                rows=1024,
                cols=2048,
                index_topk=_BLOCK_TOPK,
                windows=("chunk", 4000),
                logits="relu",
                tail=None,
            ),
        ),
        Cell(
            why="nonzero row_starts: offsets come back window-local",
            spec=dict(
                rows=4,
                cols=512,
                index_topk=32,
                windows=("explicit", (0, 37, 100, 480), (512, 300, 140, 512)),
                logits="randn",
                tail=None,
            ),
        ),
        Cell(
            why="strided logits: column stride 2 takes the scalar load path",
            spec=dict(
                rows=5,
                cols=512,
                index_topk=32,
                windows=("explicit", (0, 11, 0, 64, 7), (512, 400, 33, 512, 8)),
                logits="randn",
                tail=None,
                strided=True,
            ),
        ),
        Cell(
            why="windows of 0, 1, k-1, k, 1 and full width: -1 suffix",
            spec=dict(
                rows=6,
                cols=128,
                index_topk=16,
                windows=("explicit", (0, 0, 0, 5, 9, 0), (0, 1, 15, 21, 10, 128)),
                logits="randn",
                tail=None,
            ),
        ),
        Cell(
            why="budget wider than every window and than the matrix",
            spec=dict(
                rows=3,
                cols=64,
                index_topk=_BLOCK_TOPK,
                windows=("explicit", (0, 0, 0), (64, 7, 0)),
                logits="randn",
                tail=None,
            ),
        ),
        Cell(
            why="row tier: 12287 rows, all on the insertion-sort variant",
            spec=dict(
                rows=6 * 2048 - 1,
                cols=512,
                index_topk=64,
                windows=("random_ends",),
                logits="randn",
                tail=None,
            ),
        ),
        Cell(
            why="row tier: 12416 rows, rows >= 12288 take the radix variant",
            spec=dict(
                rows=6 * 2048 + 128,
                cols=512,
                index_topk=64,
                windows=("random_ends",),
                logits="randn",
                tail=None,
            ),
        ),
    )

    note = """
    Metadata consumed: none. Stateless; every input is an argument.

    Preconditions (the op checks none of these):
      - `0 <= row_starts[r] <= row_ends[r] <= logits.shape[1]`. An empty window
        is legal and yields an all-`-1` row; a `row_end` past the last column
        reads out of bounds (not probed).
      - Logits inside the window are finite. `-inf` is never selected, but a
        NaN ranks above every finite value and IS selected: NaN in columns 0-3
        of a 32-wide window with `k=8` returned all four. Outside the window
        anything goes -- the cells poison it with NaN.
      - `index_topk` is the BLOCK budget (512 here). The default, 2048, is the
        token budget, and with a 512-wide buffer it is the layout fault
        `is_valid` refuses.

    Loud rejections (measured): bf16/fp16 `logits` ("expected scalar type
    Float"), int64 `indices` ("expected scalar type Int"), `row_starts` /
    `row_ends` shorter than the row count ("one entry per row").

    Dispatch is by row index, not column count: rows below 12288 run an
    insertion-sort variant, rows at or above a radix variant, in one launch.
    This target's `max_num_tokens` is 8192, so serving only reaches the first;
    the cells cover both.

    QSA's stock indexer picks the CuTe DSL sibling
    `torch.ops.trtllm.cute_dsl_indexer_topk_prefill_blackwell` on this GPU; it
    was measured to select identical sets (never identical permutations). This
    entry wraps the compiled CUDA op so a first captured decode JIT-compiles
    nothing. `torch.ops.trtllm.indexer_topk_decode` is the request-grouped
    sibling; QSA does not use it.

    CUDA graphs: captured at `[8, 2048]`, `k=512`, and replayed against new
    logits and a new `row_ends` through the captured addresses. Bitwise
    comparison of two replays is not a valid check (order is unspecified).

    Hard limits: not probed. `indices` is addressed as `row * index_topk +
    slot`; the largest certified extent is 12416 x 64.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
        index_topk: int = 2048,
    ) -> torch.Tensor:
        torch.ops.trtllm.indexer_topk_prefill(logits, row_starts, row_ends, indices, index_topk)
        return indices

    def reference(
        self,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
        index_topk: int = 2048,
    ) -> TopkExpectation:
        """Each window's top values, sorted. `indices` (the output buffer) is ignored."""
        num_rows, num_cols = logits.shape
        columns = torch.arange(num_cols, device=logits.device)[None, :]
        starts, ends = row_starts.long()[:, None], row_ends.long()[:, None]
        in_window = (columns >= starts) & (columns < ends)
        masked = logits.float().masked_fill(~in_window, float("-inf"))
        values = torch.sort(masked, dim=1, descending=True).values
        if num_cols < index_topk:
            values = torch.nn.functional.pad(
                values, (0, index_topk - num_cols), value=float("-inf")
            )
        window = (row_ends.long() - row_starts.long()).clamp_min(0)
        return TopkExpectation(values[:, :index_topk], masked, row_starts.long(), window)

    def is_valid(
        self,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
        index_topk: int = 2048,
    ) -> None:
        # The op checks only `indices.size(1) >= index_topk`, then writes a flat
        # run of `index_topk`-wide rows and ignores the real row stride. Measured
        # with a [4, 12] buffer at k=8: row 1's answer landed across row 0's
        # columns 8-11 and its own 0-3, and the last row was never written.
        assert indices.dim() == 2 and indices.shape[1] == index_topk, (
            f"indices must be [num_rows, index_topk]; got {tuple(indices.shape)} "
            f"for index_topk={index_topk}"
        )
        assert indices.is_contiguous(), "indices must be contiguous"


qsa_topk_select = QsaTopkSelect()
