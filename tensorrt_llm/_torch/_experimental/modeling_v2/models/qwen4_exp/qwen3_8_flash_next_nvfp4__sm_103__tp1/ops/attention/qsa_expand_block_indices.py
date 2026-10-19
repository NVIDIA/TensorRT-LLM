# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA Top-K group ids expanded into raw KV token indices, plus the causal tail."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# Tier-2 sourcing: a vendored Triton callable reached as ordinary Python. QSA's
# kernel plane has no torch.ops surface.
from tensorrt_llm._torch.attention.backends.sparse.qsa import kernels as _qsa_kernels

#: This checkpoint's selection geometry: `indexer_compress_ratio` and
#: `indexer_budget`, so `block_topk` 512 and an output 2051 wide.
_CKPT = dict(compress_ratio=4, token_topk=2048)


def _spec(batch, **overrides) -> dict:
    """A cell spec at this checkpoint's geometry and the engine's metadata dtypes.

    `batch` is one of:
      ("sweep", n)                         n rows at 4096 + 13 i, length = position + 1
      ("serving", requests, context_lens)  one packed step: context chunks, then
                                           two MTP rows per remaining request
      ("rows", positions)                  explicit positions, length = position + 1
    """
    base = dict(
        batch=batch,
        position_dtype=torch.int64,  # qsa_logical_positions
        length_dtype=torch.int32,  # qsa_sequence_lengths
        layout="dense",
        **_CKPT,
    )
    return {**base, **overrides}


def _geometry(compress_ratio: int, token_topk: int) -> dict:
    return _spec(("rows", (4096, 97, 30)), compress_ratio=compress_ratio, token_topk=token_topk)


class QsaExpandBlockIndices(OpWrapper):
    """Expand each row's selected compressed groups into token ids, then append
    the incomplete group at the causal frontier.

    Wraps `tensorrt_llm._torch.attention.backends.sparse.qsa.kernels.
    triton_expand_qsa_block_indices`: one Triton launch,
    `_expand_qsa_block_indices_kernel`. Per row (one per query token), with
    `final_topk = token_topk + compress_ratio - 1`:

      1. column `c < token_topk` holds member `c % cr` of group
         `block_indices[row, c // cr]`; `-1` if that group is `-1` or the
         token is `>= sequence_lengths[row]`;
      2. the last `(query_positions[row] + 1) % cr` visible tokens -- the
         group the indexer never scored -- go in the columns right after the
         selected groups, i.e. from column `(# non-negative groups) * cr`;
      3. everything else is `-1`.

    Selection order is kept (nothing sorted, compacted or de-duplicated).
    `query_positions` feeds the tail only; the expansion is masked by
    `sequence_lengths` alone, so the causal bound on the *selection* is the
    producer's. Returns a fresh contiguous int32 `[rows, final_topk]`; inputs
    are read-only.

    Fusion boundary: index arithmetic and the tail only. Scoring, Top-K, the
    per-row causal group bound and the geometry are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        # ---- decode rows: the column-split launch below 128 rows -------------
        Cell(why="decode: one row, columns split across programs", spec=_spec(("sweep", 1))),
        Cell(why="decode MTP: one request's two rows", spec=_spec(("sweep", 2))),
        Cell(why="decode: one row for each of 32 request slots", spec=_spec(("sweep", 32))),
        Cell(why="127 rows: the last row count on the column split", spec=_spec(("sweep", 127))),
        Cell(why="128 rows: the first row count on whole-row programs", spec=_spec(("sweep", 128))),
        Cell(why="256 rows on whole-row programs", spec=_spec(("sweep", 256))),
        # ---- serving steps: sequence_lengths > query_position + 1 inside chunks
        Cell(
            why="mixed 1334 rows: 4 chunks + 28 MTP decodes, 32 slots",
            spec=_spec(("serving", 32, (900, 200, 150, 28))),
        ),
        Cell(
            why="mixed 8190 rows: 4 chunks + 28 MTP decodes, 32 slots",
            spec=_spec(("serving", 32, (8000, 100, 30, 4))),
        ),
        Cell(
            why="warm-up capture: 8191 rows of one request from 0",
            spec=_spec(("serving", 1, (8191,))),
        ),
        Cell(
            why="warm-up capture: 2 x 4096 rows, the [8192, 512] buffer's height",
            spec=_spec(("serving", 2, (4096, 4096))),
        ),
        Cell(
            why="tail widths 0..3 across 4096, and positions 0, 3, 7",
            spec=_spec(("rows", (4095, 4096, 4097, 4098, 4099, 7, 3, 0))),
        ),
        # ---- geometry is a free axis: neither cr nor block_topk need be 2^k ---
        Cell(why="geometry cr 2, block_topk 8", spec=_geometry(2, 16)),
        Cell(why="geometry cr 2, block_topk 5", spec=_geometry(2, 10)),
        Cell(why="geometry cr 2, block_topk 1024", spec=_geometry(2, 2048)),
        Cell(why="geometry cr 3, block_topk 7", spec=_geometry(3, 21)),
        Cell(why="geometry cr 3, block_topk 8", spec=_geometry(3, 24)),
        Cell(why="geometry cr 4, block_topk 3", spec=_geometry(4, 12)),
        Cell(why="geometry cr 4, block_topk 511", spec=_geometry(4, 2044)),
        Cell(why="geometry cr 5, block_topk 8", spec=_geometry(5, 40)),
        Cell(why="geometry cr 8, block_topk 8", spec=_geometry(8, 64)),
        Cell(why="geometry cr 16, block_topk 128", spec=_geometry(16, 2048)),
        # ---- metadata dtypes and layouts ------------------------------------
        Cell(
            why="int32 positions, int32 lengths",
            spec=_spec(("rows", (4096, 97)), position_dtype=torch.int32),
        ),
        Cell(
            why="int64 positions, int64 lengths",
            spec=_spec(("rows", (4096, 97)), length_dtype=torch.int64),
        ),
        Cell(
            why="int32 positions, int64 lengths",
            spec=_spec(("rows", (4096, 97)), position_dtype=torch.int32, length_dtype=torch.int64),
        ),
        Cell(
            why="a row slice of the engine's taller qsa_topk_indices buffer",
            spec=_spec(("sweep", 4), layout="row_slice"),
        ),
        Cell(
            why="a column slice of a wider buffer: stride(0) != block_topk",
            spec=_spec(("sweep", 4), layout="column_slice"),
        ),
        Cell(why="zero rows: a shaped empty, no launch", spec=_spec(("sweep", 0))),
    )

    note = """
    Metadata consumed: none -- every input is an argument. The target passes
    `qsa_topk_indices[:rows]` (int32, written by `qsa_topk_select`),
    `qsa_logical_positions` (int64) and `qsa_sequence_lengths` (int32: the
    request's whole KV length broadcast over its rows, so inside a context
    chunk it exceeds `position + 1` on every row but the last).

    Preconditions that are values, so not `is_valid` guards (both measured to
    fail silently):
      - `block_indices` is prefix-packed: every group id before every `-1`.
        The tail's column comes from the *count* of valid groups, so `[-1, 3]`
        at position 15, length 16, cr 4, topk 8 returns all `-1` -- tokens
        12..15 lost. `qsa_topk_select` emits packed rows.
      - selected groups are complete and visible (`g < (pos + 1) // cr`). An
        incomplete group comes back with `-1` holes in place and its real
        tokens repeated in the tail; a group past the query's horizon but
        inside `sequence_lengths` is expanded verbatim.
    Duplicate group ids expand to duplicate tokens.

    Every metadata violation is rejected loudly by the callable (rank, width
    vs geometry, int32-only `block_indices`, int32/int64 metadata, one entry
    per row, `stride(1) == 1`, contiguous metadata, CUDA), so `is_valid` adds
    nothing. A one-element strided metadata view reports contiguous and is
    accepted -- benign at one row.

    Launch: one program per (row, 256 columns) below 128 rows, one per row
    from 128 up; 127 and 128 rows agree on their shared rows. No autotune.
    CUDA-graph capturable; a replay recomputes from the live buffers, and the
    captured row count fixes the launch shape.

    Hard limits (read from the kernel, not probed): the row offset is
    `program_id * stride(0)` in int32, so `rows * stride(0) >= 2^31` (4 Mi
    rows at width 512) would wrap. Certified extent: 8192 rows x 2051 columns.

    Not certified: multiple devices, `torch.compile`, rows above 8192.
    """

    def __call__(
        self,
        block_indices: torch.Tensor,
        query_positions: torch.Tensor,
        sequence_lengths: torch.Tensor,
        *,
        compress_ratio: int,
        token_topk: int,
    ) -> torch.Tensor:
        return _qsa_kernels.triton_expand_qsa_block_indices(
            block_indices,
            query_positions,
            sequence_lengths,
            compress_ratio=compress_ratio,
            token_topk=token_topk,
        )

    def reference(
        self,
        block_indices: torch.Tensor,
        query_positions: torch.Tensor,
        sequence_lengths: torch.Tensor,
        *,
        compress_ratio: int,
        token_topk: int,
    ) -> torch.Tensor:
        """The expansion as the contract states it, in int64 torch."""
        rows, block_topk = block_indices.shape
        final_topk = token_topk + compress_ratio - 1
        device = block_indices.device
        blocks = block_indices.long()
        lengths = sequence_lengths.long()
        offsets = torch.arange(compress_ratio, device=device)

        expanded = (blocks[:, :, None] * compress_ratio + offsets).reshape(rows, token_topk)
        chosen = (blocks >= 0).repeat_interleave(compress_ratio, dim=1)
        keep = chosen & (expanded < lengths[:, None])
        out = torch.full((rows, final_topk), -1, dtype=torch.int64, device=device)
        out[:, :token_topk] = torch.where(keep, expanded, -1)

        first_free = (blocks >= 0).sum(dim=1) * compress_ratio
        visible = query_positions.long() + 1
        tail_start = visible // compress_ratio * compress_ratio
        row = torch.arange(rows, device=device)
        for offset in range(compress_ratio - 1):
            column = first_free + offset
            token = tail_start + offset
            live = (offset < visible - tail_start) & (token < lengths) & (column < final_topk)
            out[row[live], column[live]] = token[live]
        return out.to(torch.int32)


qsa_expand_block_indices = QsaExpandBlockIndices()
