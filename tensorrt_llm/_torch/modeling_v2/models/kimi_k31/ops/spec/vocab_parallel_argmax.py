# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Greedy token per row from this rank's vocab shard of the LM-head logits, without gathering them
(Triton local argmax, one float32 all-reduce of the (value, index) pairs, Triton pair reduction)."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.speculative.vocab_parallel import (
    _MAX_VOCAB,
    _local_argmax_kernel,
    _PairExchange,
    _select_tokens,
)

#: Kimi K3.1's vocabulary.
VOCAB = 163840


def _spec(tp_size: int, rows: int, **overrides) -> dict:
    spec = dict(tp_size=tp_size, rows=rows, vocab=VOCAB, dtype=torch.bfloat16, data="randn")
    spec.update(overrides)
    return spec


class VocabParallelArgmax(OpWrapper):
    """``argmax(cat(local_0, ..., local_{tp-1}, -1), -1)`` for the TP group's vocab shards, from
    this rank's ``local [rows, shard]`` (unit inner stride) alone: rank ``tp_rank`` holds global
    columns ``[tp_rank * shard, (tp_rank + 1) * shard)``. Returns ``[rows]`` in ``out_dtype``, the
    same ids on every rank, bit-identical to ``torch.argmax`` over the gathered logits (NaN beats
    every number, equal maxima go to the lowest global index).

    ``all_reduce`` sums a float32 ``[rows, width]`` buffer over the TP group (the engine's
    ``AllReduce`` at an exact strategy); None with ``tp_size == 1`` exchanges nothing.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"attention TP{tp}: target verify of {rows} rows (shard {VOCAB // tp})",
                spec=_spec(tp, rows),
            )
            for tp, rows in ((2, 6), (4, 40), (8, 9), (8, 72), (8, 576), (16, 288), (32, 9))
        ),
        Cell(
            why="one shard (no exchange): the kernels alone over the full vocab",
            spec=_spec(1, 64),
        ),
        Cell(
            why="a context-only step: one row per request",
            spec=_spec(8, 3),
        ),
        Cell(
            why="ties: the row maximum repeated on several ranks goes to the lowest global index",
            spec=_spec(8, 40, data="ties"),
        ),
        Cell(
            why="NaN rows: a NaN beats every number, the first NaN wins",
            spec=_spec(4, 16, data="nan"),
        ),
        Cell(
            why="fp32 shard logits and an int32 result",
            spec=_spec(8, 40, dtype=torch.float32, out_dtype=torch.int32),
        ),
    )
    note = """
    The Kimi K3.1 target reaches it through its DFlash worker under attention TP (tp > 1, no
    attention DP) when the LM head is an evenly vocab-sharded gathered head and the deployment
    has neither penalties nor a guided decoder (the worker's VocabParallelSelection): the
    all-greedy target tokens of each step (`_sample_tokens_for_batch` on `VocabShardedLogits`,
    rows = contexts + generations x (K + 1)), and plain DFlash's greedy draft tokens (not
    DFlash2, whose candidate selector picks the drafts). K3.1's vocab is 163840, so the shard is
    163840 / tp; the logits are the BF16 LM-head output.

    Launch shapes depend only on (rows, shard), nothing synchronizes with the host: CUDA-graph
    capturable. The pairs travel as float32: a token index is exact below 2**24 and `x + 0` is
    exact, so the all-reduce sum acts as an all-gather. `all_reduce` must therefore be an exact
    sum (not a low-precision strategy). It must also stay off the one-shot Lamport kernel of the
    TP group's IPC workspace: that kernel records the size to clear in elements of its caller's
    dtype, so a float32 call there leaves stale words that a later bf16 all-reduce on the same
    workspace reads as a peer's data. The worker's head builds the MNNVL all-reduce on a TP group
    across nodes and NCCL elsewhere.

    ARCHS lists only what was driven: sm_103 (GB300) and sm_107 (VR200). The test drives the multi-rank
    exchange in one process: each rank's local kernels run on its shard, the buffers are summed
    in rank order (what the all-reduce returns), and every rank's reduction is checked.
    """

    def __call__(
        self,
        local: torch.Tensor,
        all_reduce: Optional[Callable[[torch.Tensor], torch.Tensor]],
        tp_size: int,
        tp_rank: int,
        out_dtype: torch.dtype = torch.int64,
    ) -> torch.Tensor:
        return _select_tokens(
            _local_argmax_kernel,
            local,
            (),
            tp_rank * local.shape[-1],
            _PairExchange(all_reduce, tp_size, tp_rank),
            out_dtype,
        )

    def reference(
        self,
        per_rank_local: Sequence[torch.Tensor],
        all_reduce: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        tp_size: Optional[int] = None,
        tp_rank: Optional[int] = None,
        out_dtype: torch.dtype = torch.int64,
    ) -> torch.Tensor:
        """``torch.argmax`` of the gathered float32 logits.

        The documented collective exception to mirroring: the result depends on every rank's
        shard, so this takes the TP group's ``local`` tensors in rank order. ``all_reduce``,
        ``tp_size`` and ``tp_rank`` are accepted and ignored (every rank gets the same ids).
        """
        gathered = torch.cat([t.float() for t in per_rank_local], dim=-1)
        return gathered.argmax(dim=-1).to(out_dtype)

    def is_valid(
        self,
        local: torch.Tensor,
        all_reduce: Optional[Callable[[torch.Tensor], torch.Tensor]],
        tp_size: int,
        tp_rank: int,
        out_dtype: torch.dtype = torch.int64,
    ) -> None:
        # Indices travel as float32 through the exchange: past 2**24 two neighbouring ids
        # round to one value and the winner is silently the wrong token.
        assert local.shape[-1] * tp_size < _MAX_VOCAB, (
            f"vocab {local.shape[-1] * tp_size} must stay below 2**24; the pair exchange "
            "carries indices as float32"
        )
        # With tp_size > 1 and no all-reduce the reduction reads only this rank's pairs
        # (and the zero-padded slots of the others) and returns a shard-local argmax.
        assert tp_size == 1 or all_reduce is not None, (
            "tp_size > 1 needs the group's all_reduce; without it each rank returns the "
            "argmax of its own shard"
        )


vocab_parallel_argmax = VocabParallelArgmax()
