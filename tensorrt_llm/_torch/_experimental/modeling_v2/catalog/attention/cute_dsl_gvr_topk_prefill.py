# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefill top-k of the DSA indexer, the self-sampling GVR engine: per query row, the best keys
of its window, in one streaming pass over the row."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k import selfsampling_topk_run_prefill
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k.gvr_topk_decode_self_sampling_host import (
    WS_BYTES,
)

from .._op import Arch, Cell, OpWrapper

#: The DSA indexer of Mistral Large 4: 2048 positions per row.
_INDEXER = dict(top_k=2048)

#: The bytes of scratch one call takes (`workspace`): the engine's per-device slab.
GVR_WORKSPACE_BYTES = WS_BYTES


#: The engine's row-count tiers (`_prefill_tier`): a launch of at most 148 rows (tier 0), of
#: 149..296 (tier 1) or of more (tier 2); the smallest row count of each.
_TIER_ROWS = (1, 149, 297)
#: The widths whose tier-0 launches the engine compiles apart (its power-of-two bucket of the
#: width, capped at 32768): (2048, 4096], (4096, 8192], (8192, 8448] (8449..16384 take tier 1's
#: plan at top_k 2048) and from 16385 on, one each.
_TIER0_WIDTHS = (4096, 8192, 8448, 32768)


def gvr_workspace(device: torch.device) -> torch.Tensor:
    """A zeroed `GVR_WORKSPACE_BYTES` scratch slab for `cute_dsl_gvr_topk_prefill` on `device`.

    The kernel restores every word it consumes to zero before it returns, so one zeroing
    serves every later call on the stream that owns the slab; calls on two streams at once
    need a slab each."""
    return torch.zeros(GVR_WORKSPACE_BYTES // 4, dtype=torch.int32, device=device)


def gvr_warmup(
    top_k: int, max_cols: int, max_elems: int, device: torch.device, workspace: torch.Tensor
) -> int:
    """Compile now every engine launch a caller's `cute_dsl_gvr_topk_prefill` calls can reach,
    for calls of at most `max_cols` columns whose `[rows, W]` blocks hold at most `max_elems`
    scores (a logits budget), so that no served step compiles. Returns the warm calls made.

    The engine compiles on first use, about 2 s each on sm_103: one launcher per row-count tier
    (and, at tier 0, per width bucket), and in its deterministic mode two variants of each,
    chosen per launch by the row stride (a bitmap-assisted one up to 256 x the tier's block
    size of columns, 65536 at tier 2, 131072 at tier 1 and 262144 at tier 0, a plain one past
    it; `get_compiled` in `gvr_topk_decode_self_sampling.py`). So each tier is called once at a
    narrow stride and once at the widest its calls reach: `max_cols` at tier 0 (one row),
    `max_elems // rows` at tiers 1 and 2, within `max_cols` (a budget of 2**28 reaches tier 1's
    149 rows only at `max_cols` and below). Blocks of at most `top_k` columns need none
    (`__call__` answers them without the engine). Each call scores zeros, `[rows, W]` within
    `max(max_elems, max_cols)` elements, freed before the next; the calls run on the current
    stream with `workspace`."""
    calls = {(1, w) for w in _TIER0_WIDTHS}
    calls.add((1, max_cols))
    for rows in _TIER_ROWS[1:]:
        calls.add((rows, 2 * top_k))
        calls.add((rows, min(max_elems // rows, max_cols)))
    calls = sorted((r, w) for r, w in calls if top_k < w <= max_cols)
    for rows, width in calls:
        stride = (width + 255) // 256 * 256 + 256
        scores = torch.zeros(rows, stride, dtype=torch.float32, device=device)
        starts = torch.zeros(rows, dtype=torch.int32, device=device)
        ends = torch.full((rows,), width, dtype=torch.int32, device=device)
        out = torch.empty(rows, top_k, dtype=torch.int32, device=device)
        cute_dsl_gvr_topk_prefill(scores[:, :width], starts, ends, out, top_k, workspace)
        del scores, starts, ends, out
    torch.cuda.synchronize(device)
    return len(calls)


class CuteDslGvrTopkPrefill(OpWrapper):
    """Select each prefill row's `top_k` best keys inside its window, as window-local indices.

    The contract of `cute_dsl_indexer_topk_prefill`: `input_values` is `[rows, W]` fp32 scores
    over one K block; row `r` sees columns `[row_starts[r], row_ends[r])`. With `len =
    row_ends[r] - row_starts[r]`: when `len <= top_k`, `output_indices[r]` is `0..len-1` in
    order, then -1 (a window with `len <= 0` is all -1); otherwise it holds the `top_k`
    columns of the window with the largest scores, minus `row_starts[r]`, and no -1.

    One launch of the CuTe DSL self-sampling GVR top-k in its prefill form
    (`selfsampling_topk_run_prefill`, `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/
    gvr_topk_decode_self_sampling_host.py`), the engine's `TopK` module's
    `TopKImplementation.CUTE_DSL_GVR` prefill path, in its deterministic mode: one CTA per
    row (three row-count tiers) samples the row, derives a ladder of candidate thresholds
    from the sample's histogram and harvests the winners in one streaming pass. Exactness
    rests on count-crossing invariants (the harvest checks the count above each rung), not
    on the estimate. Deterministic: the selected set is the scores above the k-th score plus
    the smallest columns among those equal to it, and the order is a fixed function of the
    input. Selection only, fp32 compares throughout: the scores (`fp8_mqa_logits`) and the
    mapping of local indices to pool rows are the caller's.

    `output_indices` (`[rows, top_k]` int32, contiguous, 16-byte aligned) is written in place
    and returned; `workspace` is the caller's `gvr_workspace` slab, or None for the engine's
    per-device one.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The dep4 target's calls (dep8 alike), one per query tile of an indexer
    # prefill block, on fp8_mqa_logits' row-padded output: at the 1M envelope
    # (max_seq_len 1048576, chunked prefill in 32768-token chunks) its tiles are
    # floor(2**26 / W) rows of a block of W keys, at max_seq_len 32768
    # floor(2**24 / W). `blocks` lists one call per K block; a block lists
    # `(cached, new, rows_taken)` per prompt (`rows_taken` keeps that prompt's
    # last rows, a row tile). The engine picks its plan by row count (tier 0 at
    # up to 148 rows, 1 at 149..296, 2 above; tier 0 at widths of 8449..16384
    # takes tier 1's plan at k = 2048) and, at tier 0, by the power-of-two width
    # bucket up to 32768; the cells drive each tier. `values` names the score pattern the test
    # writes into the windows ("logits" scores them with fp8_mqa_logits over
    # random e4m3 keys, as the target does).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="dep4 1M, the last 64-row tile of the 32nd chunk of a 1M prompt: tier 0, "
            "bucket 32768, windows of 1048513..1048576 keys",
            spec=dict(blocks=([(1015808, 32768, 64)],), values="distinct", **_INDEXER),
        ),
        Cell(
            why="dep4 1M, the same tile scored by fp8_mqa_logits over random keys",
            spec=dict(blocks=([(1015808, 32768, 64)],), values="logits", **_INDEXER),
        ),
        Cell(
            why="dep4 1M, the 64-row tile with narrow scores: every key in one coarse bin",
            spec=dict(blocks=([(1015808, 32768, 64)],), values="narrow", **_INDEXER),
        ),
        Cell(
            why="dep4 1M, the 64-row tile with tied scores (4 values): the k-th score is "
            "shared by about a quarter of each window",
            spec=dict(blocks=([(1015808, 32768, 64)],), values="ties", **_INDEXER),
        ),
        Cell(
            why="dep4 1M, a 15-row final chunk at 1048559 keys, alone: a short tile",
            spec=dict(blocks=([(1048544, 15, None)],), values="distinct", **_INDEXER),
        ),
        Cell(
            why="dep4 1M, the 128-row tile at 524288 keys: tier 0's widest row count",
            spec=dict(blocks=([(491520, 32768, 128)],), values="logits", **_INDEXER),
        ),
        Cell(
            why="dep4 1M, the 256-row tile at 262144 keys: tier 1",
            spec=dict(blocks=([(229376, 32768, 256)],), values="logits", **_INDEXER),
        ),
        Cell(
            why="dep4 1M, the 512-row tile at 131072 keys: tier 2",
            spec=dict(blocks=([(98304, 32768, 512)],), values="distinct", **_INDEXER),
        ),
        Cell(
            why="dep4 1M, the 1024-row tile at 65536 keys, windows crossing 2**16",
            spec=dict(blocks=([(32768, 32768, 1024)],), values="spread", **_INDEXER),
        ),
        Cell(
            why="dep4 1M, the first 2048-row tile of a fresh 32768-token chunk: windows of "
            "1..2048 keys, every row short (identity, then -1)",
            spec=dict(blocks=([(0, 2048, None)],), values="distinct", **_INDEXER),
        ),
        Cell(
            why="dep4 1M, the second 2048-row tile of a fresh chunk: windows of 2049..4096, "
            "the first ranked rows, a 4096-column block (tier 2)",
            spec=dict(blocks=([(0, 4096, 2048)],), values="distinct", **_INDEXER),
        ),
        Cell(
            why="dep4 1M, the last 2048-row tile of a fresh 32768-token chunk",
            spec=dict(blocks=([(0, 32768, 2048)],), values="logits", **_INDEXER),
        ),
        Cell(
            why="dep4 32K, the last 512-row tile of a whole 32768-token prompt (2**24)",
            spec=dict(blocks=([(0, 32768, 512)],), values="distinct", **_INDEXER),
        ),
        Cell(
            why="dep4 32K, 16 packed prompts in one 32768-column block, the last 142 rows of "
            "the 8193-token prompt and the 882 of the last one: windows starting at 23693 and "
            "31886",
            spec=dict(
                blocks=(
                    [
                        (0, n, 0)
                        for n in (
                            1,
                            7,
                            33,
                            100,
                            255,
                            257,
                            513,
                            1023,
                            1025,
                            2047,
                            2049,
                            4095,
                            4097,
                            8191,
                        )
                    ]
                    + [(0, 8193, 142), (0, 882, None)],
                ),
                values="distinct",
                **_INDEXER,
            ),  # fmt: skip
        ),
        Cell(
            why="dep4 1M, 16 packed chunk tails in one 1048576-key block, its last 64 rows: "
            "four requests' windows starting at multiples of 65536",
            spec=dict(
                blocks=([(65520, 16, 0)] * 12 + [(65520, 16, None)] * 4,),
                values="distinct",
                **_INDEXER,
            ),
        ),
        Cell(
            why="dep4 1M, a short tail beside a fresh prompt in one block: rows of a "
            "1048400-key window and of windows of 1..155 keys in one call",
            spec=dict(
                blocks=([(1048400, 100, 40), (0, 155, None)],), values="distinct", **_INDEXER
            ),
        ),
        # The tep4 target's calls (tep8 alike): its 2**28-element tiles at the 1M envelope,
        # cut from the first row of the rank's share of a chunk under the engine's query split,
        # and the pieces of a packed chunk cut per request past max_seq_len.
        Cell(
            why="tep4 1M, a q-split share's 256-row tile at 1048576 keys: tier 1",
            spec=dict(blocks=([(1015808, 32768, 256)],), values="logits", **_INDEXER),
        ),
        Cell(
            why="tep4 1M, q-split rank 0's 70-row last tile at 1024000 keys: tier 0",
            spec=dict(blocks=([(1015808, 8192, 70)],), values="distinct", **_INDEXER),
        ),
        Cell(
            why="tep4 1M, q-split rank 1's 260-row tile at 1032192 keys, narrow scores",
            spec=dict(blocks=([(1015808, 16384, 260)],), values="narrow", **_INDEXER),
        ),
        Cell(
            why="tep4 1M, a q-split share's 512-row tile at 524288 keys: tier 2",
            spec=dict(blocks=([(491520, 32768, 512)],), values="logits", **_INDEXER),
        ),
        Cell(
            why="tep4 1M, q-split rank 3's whole 8192-row share of a first chunk at 32768 keys",
            spec=dict(blocks=([(0, 32768, 8192)],), values="spread", **_INDEXER),
        ),
        Cell(
            why="tep8 1M, q-split rank 7's whole 4096-row share of a first chunk at 32768 keys",
            spec=dict(blocks=([(0, 32768, 4096)],), values="distinct", **_INDEXER),
        ),
        Cell(
            why="tep4 1M, an unsplit 8191-row tail chunk's last 255 rows at 1048576 keys",
            spec=dict(blocks=([(1040385, 8191, 255)],), values="distinct", **_INDEXER),
        ),
        Cell(
            why="tep4 1M, 16 packed chunk tails in one 983296-key block, 256 rows (2**28)",
            spec=dict(blocks=([(61440, 16, None)] * 16,), values="distinct", **_INDEXER),
        ),
        Cell(
            why="tep4 1M, the packed-chunk cut: a ~1M request's 100-token tail alone, 100 rows "
            "against its own 1048500 keys",
            spec=dict(blocks=([(1048400, 100, None)],), values="ties", **_INDEXER),
        ),
    )

    note = """
    Metadata consumed: none. The windows are the caller's: an indexer prefill
    chunk's `cu_seqlen_ks` / `cu_seqlen_ke`, sliced to the row tile, in the K
    block's column coordinates. The engine reads no device value on the host
    and trusts `0 <= row_starts <= row_ends <= W`; `is_valid` checks it.

    Preconditions, each raised by the engine's host checks: fp32
    `input_values` with unit column stride, a row stride that is a multiple of
    4 and a 16-byte aligned base (fp8_mqa_logits' row-padded view is read in
    place, its stride align(W + 256, 256)); int32, contiguous, 1-D
    `row_starts` / `row_ends` of `rows` entries; `output_indices` int32,
    contiguous, `[rows, k]` with k a multiple of 4, 16-byte aligned (a row
    slice of a `[T, k]` buffer is). `top_k` is not read: k is
    `output_indices.shape[1]`, which `is_valid` holds to `top_k`. A call with
    zero rows returns without a launch.

    The engine compiles one launcher per (tier, k, width bucket at tier 0,
    deterministic) on the first call that needs it, and in deterministic mode
    two variants of each by the call's row stride, and caches them for the
    process, about 2 s per compile on sm_103 (measured: 4.4 s for the first,
    2.1 to 2.6 s for each later one); it refuses to compile under CUDA-graph
    capture. Prefill steps are eager; `gvr_warmup` compiles every launch a
    caller's tiles reach ahead of serving. A block of at most `top_k` columns
    does not reach the engine: every row of it is short, and `__call__`
    writes the identity rows itself (the `TopK` module routes such blocks
    away from the engine too).
    """

    def __call__(
        self,
        input_values: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        output_indices: torch.Tensor,
        top_k: int,
        workspace: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_values.shape[1] <= top_k:
            # Every window lies inside the block, so every row is short: the identity, then -1.
            ident = torch.arange(top_k, dtype=torch.int32, device=output_indices.device)
            empty = torch.full((), -1, dtype=torch.int32, device=output_indices.device)
            lengths = (row_ends - row_starts)[:, None]
            return torch.where(ident < lengths, ident, empty, out=output_indices)
        selfsampling_topk_run_prefill(
            input_values,
            row_starts,
            row_ends,
            output_indices,
            workspace=workspace,
            deterministic=True,
        )
        return output_indices

    def reference(
        self,
        input_values: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        output_indices: torch.Tensor,
        top_k: int,
        workspace: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """A fresh `[rows, top_k]` int32 selection.

        `output_indices` and `workspace` are not read. A long row holds `torch.topk`'s
        winners in descending score order; the engine promises its own fixed order and keeps
        the lowest columns among keys tied at the k-th score, so the set is exact only on
        rows without such a tie.
        """
        rows, width = input_values.shape
        device = input_values.device
        starts = row_starts.long()[:, None]
        ends = row_ends.long()[:, None]
        cols = torch.arange(width, device=device)
        inside = (cols >= starts) & (cols < ends)
        scores = torch.where(inside, input_values.float(), float("-inf"))
        out = torch.full((rows, top_k), -1, dtype=torch.int64, device=device)
        ranked = min(top_k, width)
        if ranked:
            out[:, :ranked] = scores.topk(ranked, dim=1).indices - starts
        lengths = (ends - starts).squeeze(1)
        short = lengths <= top_k
        ident = torch.arange(top_k, device=device)
        out[short] = torch.where(ident < lengths[short, None], ident, -1)
        return out.to(torch.int32)

    def is_valid(
        self,
        input_values: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        output_indices: torch.Tensor,
        top_k: int,
        workspace: Optional[torch.Tensor] = None,
    ) -> None:
        # k is the output's width: a wider buffer selects more than top_k per row.
        assert output_indices.shape[1] == top_k, (
            f"output_indices must be exactly top_k = {top_k} wide; got "
            f"{tuple(output_indices.shape)}, whose width the engine selects"
        )
        # The engine trusts the windows: a start below zero or an end past W reads another
        # row's scores and ranks them as this row's.
        if row_starts.numel():
            assert int(row_starts.min()) >= 0, "a window starts before its row"
            assert bool((row_starts <= row_ends).all()), "a window ends before it starts"
            assert int(row_ends.max()) <= input_values.shape[1], (
                f"a window ends past the {input_values.shape[1]}-column K block"
            )
        if workspace is not None:
            assert workspace.numel() * workspace.element_size() >= GVR_WORKSPACE_BYTES, (
                f"workspace holds {workspace.numel() * workspace.element_size()} bytes; the "
                f"engine needs {GVR_WORKSPACE_BYTES}"
            )


cute_dsl_gvr_topk_prefill = CuteDslGvrTopkPrefill()
