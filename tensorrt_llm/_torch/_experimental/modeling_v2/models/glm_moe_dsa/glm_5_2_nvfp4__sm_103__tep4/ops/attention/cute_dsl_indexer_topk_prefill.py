# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefill top-k of the DSA indexer in CuTe DSL: per query row, the best keys of its window."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention import self_attn_indexer_topk_prefill as _kernel
from .cute_dsl_indexer_topk_decode import select_rows

#: Threads per row the short-window arm covers `top_k` with: 512 x 4 = 2048.
_SHORT_WINDOW_ELEMENTS_PER_THREAD = 4


def _next_power_of_2(value: int) -> int:
    return 1 if value <= 1 else 1 << (value - 1).bit_length()


@dataclass(frozen=True)
class CompiledIndexerTopkPrefill:
    """The short-window launcher and one general-arm pair per power-of-two width.

    `general[w]` serves every K-block width in `(w / 2, w]` above `top_k`;
    `max_columns` is the widest block any of them takes.
    """

    short_window: Callable
    general: Mapping[int, tuple[Callable, Callable]]
    top_k: int
    max_columns: int


def build(device_id: int, *, top_k: int, max_columns: int) -> CompiledIndexerTopkPrefill:
    """Compile the short-window arm and the general arm for every width up to `max_columns`.

    The general arm compiles per power-of-two K-block width (each a top-k
    kernel plus its localize kernel), from the first power of two above `top_k`
    to the one at or above `max_columns`: at top_k 2048 and 8192 columns that
    is 4096 and 8192. A few seconds per width, cached per process by the kernel
    module, so building again with a larger `max_columns` compiles only the new
    widths. Allocates nothing.
    """
    short = _kernel.compile_indexer_topk_prefill_short_window(
        device_id, int(top_k), _SHORT_WINDOW_ELEMENTS_PER_THREAD
    )
    general = {}
    width = _next_power_of_2(int(top_k) + 1)
    while width <= _next_power_of_2(int(max_columns)):
        general[width] = _kernel.compile_indexer_topk_prefill_general(
            device_id, torch.float32, max_num_columns=width, top_k=int(top_k)
        )
        width <<= 1
    widest = max(general, default=int(top_k))
    return CompiledIndexerTopkPrefill(
        short_window=short, general=general, top_k=int(top_k), max_columns=widest
    )


class CuteDslIndexerTopkPrefill(OpWrapper):
    """Select each prefill row's `top_k` best keys inside its window, as window-local indices.

    `logits` is `[rows, W]` fp32 over one K block; row `r` sees columns
    `[row_starts[r], row_ends[r])`. With `len = row_ends[r] - row_starts[r]`:
    when `len <= top_k`, `indices[r]` is `0..len-1` then -1; otherwise it holds
    the `top_k` columns of the window with the largest values, minus
    `row_starts[r]`, in unspecified order.

    The arm is chosen from the block width: `W <= top_k` takes the short-window
    kernel, which never reads `logits`; a wider block takes the general arm,
    which selects per row (long rows as absolute columns, short rows already
    local) and then localises the long rows only.

    `indices` (`[rows, top_k]` int32) is written in place and returned.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The target calls this per prefill chunk, per row tile, on fp8_mqa_logits'
    # output: one K block holds one request's whole KV (cached + new) or
    # several packed requests' back to back, so a later request's windows start
    # past zero. `requests` lists `(cached, new, rows_taken)` per request in
    # block order; `rows_taken` keeps the last rows of each when set.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="short-window arm: a 1024-token prompt, block no wider than top-k",
            spec=dict(requests=[(0, 1024, None)], max_columns=16384),
        ),
        Cell(
            why="packed requests: long and short rows, later windows start past 0",
            spec=dict(
                requests=[(2900, 100, None), (0, 1500, None), (2000, 600, None)],
                max_columns=16384,
            ),
        ),
        Cell(
            why="a block in (2048, 4096]: the 4096-wide general build",
            spec=dict(requests=[(2500, 300, None)], max_columns=16384),
        ),
        Cell(
            why="a block of exactly 8192 columns: the widest load-time build",
            spec=dict(requests=[(7000, 1192, 256)], max_columns=16384),
        ),
        Cell(
            why="a cached prefix past 8192: a width the call site builds on demand",
            spec=dict(requests=[(11000, 1000, 128)], max_columns=16384),
        ),
    )

    note = """
    `build(device_id, top_k=, max_columns=)` compiles the short-window arm and
    every general width up to `max_columns`, and returns the handle `__call__`
    takes as `compiled`. The target builds it at load for 8192 columns. A K
    block wider than `compiled.max_columns` (a long cached prefix, or several
    packed requests) has no launcher: the call site builds a wider handle
    first. Context steps are never captured, so that compile cannot break a
    graph; it adds to that step's latency once per new power-of-two width.

    Only fp32 logits and the general ("utility") long-row arm are certified:
    that is what the target runs. The module's histogram long-row arm is a
    separate specialisation for exactly 8192 columns that the target does not
    call.

    The general arm allocates a `[rows, 2, W]` int32 workspace per call, and a
    non-contiguous `logits` is copied first.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
        *,
        top_k: int,
        compiled: CompiledIndexerTopkPrefill,
    ) -> torch.Tensor:
        width = int(logits.shape[1])
        if width <= top_k:
            return _kernel.indexer_topk_prefill_short_window_run(
                logits,
                row_starts,
                row_ends,
                indices,
                top_k=top_k,
                elements_per_thread=_SHORT_WINDOW_ELEMENTS_PER_THREAD,
                compiled_fn=compiled.short_window,
            )
        return _kernel.indexer_topk_prefill_general_run(
            logits,
            row_starts,
            row_ends,
            indices,
            top_k=top_k,
            compiled_fns=compiled.general[_next_power_of_2(width)],
        )

    def reference(
        self,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
        *,
        top_k: int,
        compiled: CompiledIndexerTopkPrefill,
    ) -> torch.Tensor:
        """A fresh `[rows, top_k]` int32 selection; `indices` and `compiled` are not read.

        Ties at the k-th value go to the lower column here; the kernel does not
        promise that, so the reference is exact only for rows without such a tie.
        """
        return select_rows(logits, row_starts.long(), row_ends.long(), top_k)

    def is_valid(
        self,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
        *,
        top_k: int,
        compiled: CompiledIndexerTopkPrefill,
    ) -> None:
        # The launchers bake top_k into their row stride and shared memory.
        assert top_k == compiled.top_k, f"compiled for top_k={compiled.top_k}, called with {top_k}"
        # Windows are read with no bound: a start past the end, or an end past
        # the block, reads another row's scores or past the tensor.
        if row_starts.numel():
            assert bool((row_starts >= 0).all()) and bool((row_starts <= row_ends).all()), (
                "every window needs 0 <= row_start <= row_end"
            )
            assert int(row_ends.max()) <= logits.shape[1], (
                f"a window ends at {int(row_ends.max())}, past the {logits.shape[1]}-column K block"
            )


cute_dsl_indexer_topk_prefill = CuteDslIndexerTopkPrefill()
