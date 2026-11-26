# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.2's token embedding gather, in one CuTe DSL launch; and `embed`, the gather of a step
of any row count in slices the kernel takes."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.embedding import embedding_m64_vocab154880_h6144_bf16 as _kernel

Runner = _kernel.GLMEmbeddingAdapter


def build(weight: torch.Tensor) -> Runner:
    """Compile the kernel's seven row-count schedules and bind them to `weight`.

    Call once, at load. Returns the runner the op takes as `runner`. `weight` is
    the full replicated `[154880, 6144]` bf16 table, contiguous on the GPU; the
    runner keeps a reference to it (no copy). Costs seven CuTe DSL compiles the
    first time; the kernel module caches them, so a second runner is cheap.
    """
    return _kernel.make_trtllm_adapter(weight)


class CuteDslEmbedding(OpWrapper):
    """`weight[input_ids]` as a new contiguous `[M, 6144]` bf16 tensor.

    `input_ids` is a contiguous int32 `[M]` with 1 <= M <= 8192. An id outside
    `[0, 154880)` yields an all-zero row rather than an error.
    """

    ARCHS = frozenset({Arch.SM_103})

    # GLM-5.2 embeds every step's tokens (and the MTP draft step's), so M is
    # the step's row count: decode up to 64 rows, prefill up to max_num_tokens
    # 8192. The runner picks one of seven schedules from M; the cells hit each,
    # plus the partially filled last block of the two multi-row schedules.
    CELLS: tuple[Cell, ...] = (
        Cell(why="decode, one token: 2 CTAs per row, 8-wide loads", spec=dict(rows=1)),
        Cell(why="4 rows: 4 CTAs per row, 16-wide loads", spec=dict(rows=4)),
        Cell(why="8 rows: its own 2-CTA-per-row schedule", spec=dict(rows=8)),
        Cell(why="24 rows: its own 1-CTA-per-row schedule", spec=dict(rows=24)),
        Cell(
            why="a row count with no schedule of its own, 7, takes the 32-row one",
            spec=dict(rows=7),
        ),
        Cell(why="the full decode batch, 64 rows: 4 rows per CTA", spec=dict(rows=64)),
        Cell(
            why="33 rows: the first 4-rows-per-CTA count, one row in its last CTA",
            spec=dict(rows=33),
        ),
        Cell(
            why="4097 rows: the first bulk-copy count, last CTA half full",
            spec=dict(rows=4097),
        ),
        Cell(why="max_num_tokens, 8192 rows, bulk copy", spec=dict(rows=8192)),
        Cell(
            why="ids outside the vocabulary (negative and >= 154880) give zero rows",
            spec=dict(rows=64, invalid=True),
        ),
    )

    note = """
    The target builds one runner at load and calls it at every row count; there
    is no stock-op fallback at the call site. M above 8192, M = 0 and int64 ids
    are rejected by the kernel's wrapper. The table is replicated on every rank
    (no vocabulary sharding, no collective).
    """

    def __call__(
        self, input_ids: torch.Tensor, weight: torch.Tensor, runner: Runner
    ) -> torch.Tensor:
        return runner(input_ids)

    def reference(
        self, input_ids: torch.Tensor, weight: torch.Tensor, runner: Runner
    ) -> torch.Tensor:
        """The gathered rows, zero where the id is out of range. `runner` is ignored."""
        ids = input_ids.long()
        valid = (ids >= 0) & (ids < weight.shape[0])
        rows = weight[ids.clamp(0, weight.shape[0] - 1)]
        return torch.where(valid.unsqueeze(-1), rows, torch.zeros_like(rows))

    def is_valid(self, input_ids: torch.Tensor, weight: torch.Tensor, runner: Runner) -> None:
        # The runner gathers from the table it was built with and never looks at
        # `weight`. A runner built on another tensor -- a stale copy, a table
        # from before a reload -- returns plausible embeddings of the wrong table.
        assert runner.weight is weight, (
            "runner was built on a different weight tensor; it ignores the one passed"
        )


cute_dsl_embedding = CuteDslEmbedding()


#: The kernel's widest row count. A step above it (a `max_num_tokens` past 8192) gathers in
#: slices.
MAX_ROWS = 8192


class Embed(OpWrapper):
    """The embedding rows for `input_ids` of any count: `cute_dsl_embedding` on slices of up
    to `MAX_ROWS` ids, concatenated. `runner` is `build`'s."""

    def __call__(
        self, input_ids: torch.Tensor, weight: torch.Tensor, runner: Runner
    ) -> torch.Tensor:
        if input_ids.shape[0] <= MAX_ROWS:
            return cute_dsl_embedding(input_ids, weight, runner)
        return torch.cat(
            [cute_dsl_embedding(ids, weight, runner) for ids in input_ids.split(MAX_ROWS)]
        )

    def reference(
        self, input_ids: torch.Tensor, weight: torch.Tensor, runner: Runner
    ) -> torch.Tensor:
        """`cute_dsl_embedding`'s reference over all the ids."""
        return cute_dsl_embedding.reference(input_ids, weight, runner)


embed = Embed()
