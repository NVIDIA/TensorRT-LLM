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
#: The unified target's widest score row: the tune envelope's `max_seq_len`
#: 262144 is an 8192-page block table at 32 tokens per block, compressed 4:1.
#: Core's 9216 (288 pages, 2304 columns) and MTP's 16384 (512 pages, 4096
#: columns) are narrower; the kernel picks its path by each row's window, so
#: the windows below span theirs.
_ENVELOPE_SEQ_LEN = 262144
_ENVELOPE_COLUMNS = _ENVELOPE_SEQ_LEN // 4
#: 128 decode requests: the window edges first -- empty (position 2), one
#: block, 511 / 512 / 513 blocks around the budget (512 is the kernel's
#: copy-through shortcut, 513 its first histogram row), core's and MTP's widest
#: rows, the envelope's less one column -- then 120 spread to the envelope.
_DECODE_POSITIONS = (2, 3, 2046, 2047, 2051, 9215, 16383, _ENVELOPE_SEQ_LEN - 2) + tuple(
    4096 + (_ENVELOPE_SEQ_LEN - 1 - 4096) * i // 119 for i in range(120)
)
#: 128 verify requests at max_draft_len 7, 8 consecutive positions each: one
#: from position 0, two straddling the budget, core's and MTP's last rows, the
#: envelope's last (a full 65536-column window), then 122 spread between.
_VERIFY_STARTS = (0, 2040, 2044, 9208, 16376, _ENVELOPE_SEQ_LEN - 8) + tuple(
    4096 + (_ENVELOPE_SEQ_LEN - 16 - 4096) * i // 121 for i in range(122)
)
#: Rows from which a contiguous launch runs the CuTe DSL sibling: the CUDA op
#: serves rows at or past 12288 with its radix variant, which is the slower of
#: the two (16384 rows: 310 vs 135 us at k=512 over 2592 columns, GB300);
#: below it the two run alike and the CUDA op needs no JIT compilation.
_CUTE_DSL_MIN_ROWS = 12288


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
    wheel) and, for a contiguous launch of `_CUTE_DSL_MIN_ROWS` rows or more,
    its CuTe DSL sibling `torch.ops.trtllm.cute_dsl_indexer_topk_prefill_blackwell`
    (the same contract). For row `r` it selects the `index_topk` largest values of
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
            why="decode MTP7: 32 requests x 8 rows = 256 rows, some below the budget, NaN tail",
            spec=dict(
                rows=256,
                cols=_SCORE_COLUMNS,
                index_topk=_BLOCK_TOPK,
                windows=(
                    "positions",
                    tuple(600 + 233 * g + j for g in range(32) for j in range(8)),
                ),
                logits="relu",
                tail="nan",
            ),
        ),
        Cell(
            why="context: 8191 rows at positions 0..8190 (warm-up capture), the CUDA op",
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
            why="context: 16384 rows at positions 0..16383, NaN tail (CuTe DSL op)",
            spec=dict(
                rows=16384,
                cols=_SCORE_COLUMNS,
                index_topk=_BLOCK_TOPK,
                windows=("chunk", 0),
                logits="relu",
                tail="nan",
            ),
        ),
        # The unified target (tp4 / tep4 / dep4): the selection is per row and
        # replicated across ranks, so the member changes only the row count.
        Cell(
            why="decode: one row at the 262144-token envelope, a 65535-block window, NaN tail",
            spec=dict(
                rows=1,
                cols=_ENVELOPE_COLUMNS,
                index_topk=_BLOCK_TOPK,
                windows=("positions", (_ENVELOPE_SEQ_LEN - 2,)),
                logits="relu",
                tail="nan",
            ),
        ),
        Cell(
            why="decode batch 128: windows empty, 511 / 512 / 513 blocks and up to the envelope",
            spec=dict(
                rows=128,
                cols=_ENVELOPE_COLUMNS,
                index_topk=_BLOCK_TOPK,
                windows=("positions", _DECODE_POSITIONS),
                logits="relu",
                tail="nan",
            ),
        ),
        Cell(
            why="verify MTP7 at 128 requests: 1024 rows, windows up to the envelope's 65536",
            spec=dict(
                rows=1024,
                cols=_ENVELOPE_COLUMNS,
                index_topk=_BLOCK_TOPK,
                windows=("positions", tuple(s + j for s in _VERIFY_STARTS for j in range(8))),
                logits="relu",
                tail="nan",
            ),
        ),
        Cell(
            why="context chunk: max_num_tokens 8192 rows ending a 262144-token prompt",
            spec=dict(
                rows=8192,
                cols=_ENVELOPE_COLUMNS,
                index_topk=_BLOCK_TOPK,
                windows=("chunk", _ENVELOPE_SEQ_LEN - 8192),
                logits="relu",
                tail="nan",
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
        of a 32-wide window with `k=8` returned all four (measured, not pinned
        by a test). Outside the window anything goes -- the cells poison it
        with NaN.
      - `index_topk` is the BLOCK budget (512 here). The default, 2048, is the
        token budget, and with a 512-wide buffer it is the layout fault
        `is_valid` refuses.

    Loud rejections (measured, not pinned by a test): bf16/fp16 `logits`
    ("expected scalar type Float"), int64 `indices` ("expected scalar type
    Int"), `row_starts` / `row_ends` shorter than the row count ("one entry per
    row").

    The CUDA op dispatches by row index, not column count: rows below 12288 run
    an insertion-sort variant, rows at or above a radix variant, in one launch.
    A launch of `_CUTE_DSL_MIN_ROWS` (12288) rows or more with contiguous
    logits runs the CuTe DSL sibling instead -- QSA's stock indexer's choice on
    this GPU, measured to select identical sets (never identical permutations;
    ties, as in the CUDA op, broken arbitrarily). The tuned config's
    `max_num_tokens` 16384 reaches it at prefill-heavy mixed steps; smaller
    launches, every decode one among them, keep the compiled CUDA op, so a
    first captured decode JIT-compiles nothing. `torch.ops.trtllm.indexer_topk_decode` is the request-grouped
    sibling; QSA does not use it.

    The CUDA op picks its path per row by the window's length, never by the
    row width: a window of at most `index_topk` copies offsets 0..len-1
    through (unsorted, no histogram); a longer one runs the histogram passes
    over the fp32 bits into a 2048-item final buffer. The width enters only as
    the row stride, and a row that does not start 16-byte aligned is read with
    a scalar head and tail. So the unified target's cells, at the 262144-token
    envelope's 65536-column rows, take windows from empty through 1, 511, 512,
    513, core's 2304 (`max_seq_len` 9216) and MTP's 4096 (16384) to 65535 and
    65536 -- rows 1, 128 (decode), 1024 (MTP7 verify at 128 requests) and 8192
    (a `max_num_tokens` context chunk). All four run the compiled-in CUDA op
    (below 12288 rows); the JIT-compiled CuTe DSL sibling runs at the 16384-row
    cell, on sm_103 in the same test file. Attention DP and TP do not change the
    op's shape, only its row count. Measured on GB300, warm, CUDA events:
    1 x 65536 26.6 us, 128 x 65536 26.8 us, 1024 x 65536 70.4 us, 8192 x 65536
    885.4 us.

    A tied class far wider than the final buffer -- this checkpoint's relu
    scores leave rows with fewer positive blocks than the budget -- was probed
    (not a cell): 0, 1, 100, 511, 512, 513 and 3000 positive scores in a
    65536-column window, the rest exactly 0, at 1, 128 and 1024 rows. Every row
    returned 512 distinct in-window offsets whose values are the window's top
    512.

    CUDA graphs: captured at `[256, 2048]` (32 requests at max_draft_len 7) and
    at 1, 128 and 1024 rows of 65536 columns, `k=512`, each over full windows,
    and replayed against new logits and a new `row_ends` through the captured
    addresses (the 128-row replay at 511 blocks, the copy-through path).
    Bitwise comparison of two replays is not a valid check (order is
    unspecified).

    Hard limits: `indices` is addressed as `row * index_topk + slot` and the
    logits row as `row * stride(0)`, both in 64-bit; `num_columns` and
    `stride(0)` pass through int32, so a row past 2^31 - 1 elements is out of
    reach (the envelope's is 65536). The largest certified extents are logits
    8192 x 65536 (2^29 elements) and `indices` 16384 x 512.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
        index_topk: int = 2048,
    ) -> torch.Tensor:
        if logits.shape[0] >= _CUTE_DSL_MIN_ROWS and logits.is_contiguous():
            torch.ops.trtllm.cute_dsl_indexer_topk_prefill_blackwell(
                logits, row_starts, row_ends, indices, index_topk
            )
        else:
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
