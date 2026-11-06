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


#: First positions of 32 MTP verify blocks in the sparse branch (KV past 2048).
_VERIFY_STARTS = tuple(2100 + (499 * g) % 6000 for g in range(32))


def _spec(batch, **overrides) -> dict:
    """A cell spec at this checkpoint's geometry and the engine's metadata dtypes.

    `batch` is one of:
      ("sweep", n)                         n rows at 4096 + 13 i, length = position + 1
      ("serving", requests, context_lens)  one packed step: context chunks, then
                                           two MTP rows per remaining request
      ("verify", starts, p)                one MTP verify block of p rows per start,
                                           every row's length the block's end
    """
    base = dict(
        batch=batch,
        position_dtype=torch.int64,  # qsa_logical_positions
        length_dtype=torch.int32,  # qsa_sequence_lengths
        **_CKPT,
    )
    return {**base, **overrides}


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
        # ---- decode rows: the column-split launch below 128 rows, whole rows from 128
        Cell(why="decode: one row, columns split across programs", spec=_spec(("sweep", 1))),
        # ---- MTP verify blocks (max_draft_len up to 7): p rows per request, every
        # row's length the block's end, so only the last row has length == position + 1.
        Cell(
            why="verify MTP7: 32 requests x 8 = 256 rows, whole-row programs",
            spec=_spec(("verify", _VERIFY_STARTS, 8)),
        ),
        # ---- context rows: sequence_lengths > query_position + 1 inside chunks; chunk
        # 0 starts at position 0, so its rows also carry every tail width 0..3.
        Cell(
            why="warm-up capture: 2 x 4096 rows, the [8192, 512] buffer's height",
            spec=_spec(("serving", 2, (4096, 4096))),
        ),
    )

    note = """
    Metadata consumed: none -- every input is an argument. The target passes
    `qsa_topk_indices[:rows]` (int32, written by `qsa_topk_select`),
    `qsa_logical_positions` (int64) and `qsa_sequence_lengths` (int32: the
    request's whole KV length broadcast over its rows, so inside a context
    chunk it exceeds `position + 1` on every row but the last).

    Preconditions that are values, so not `is_valid` guards (both fail
    silently; no test pins them):
      - `block_indices` is prefix-packed: every group id before every `-1`.
        The tail's column comes from the *count* of valid groups, so `[-1, 3]`
        at position 15, length 16, cr 4, topk 8 returns all `-1` -- tokens
        12..15 lost. `qsa_topk_select` emits packed rows.
      - selected groups are complete and visible (`g < (pos + 1) // cr`). An
        incomplete group comes back with `-1` holes in place and its real
        tokens repeated in the tail; a group past the query's horizon but
        inside `sequence_lengths` is expanded verbatim.
    Duplicate group ids expand to duplicate tokens.

    The callable rejects metadata violations loudly (rank, width vs geometry,
    int32-only `block_indices`, int32/int64 metadata, one entry per row,
    `stride(1) == 1`, contiguous metadata, CUDA), so `is_valid` adds nothing;
    the test pins the width-vs-geometry rejection. A one-element strided
    metadata view reports contiguous and is accepted -- benign at one row.

    Launch: one program per (row, 256 columns) below 128 rows, one per row
    from 128 up. No autotune. CUDA-graph capturable (captured at 256 rows); a
    replay recomputes from the live buffers, and the captured row count fixes
    the launch shape.

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
