# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Target probabilities of a 33-96-row verify on the fused sampler's multi-CTA family: row chunks of at most 32 rows,
issued on parallel streams."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.common.multi_stream import do_multi_stream
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.fused_compute_probs_from_logits import (  # noqa: E501
    sampling_spec,
)

#: Most rows the fused sampler gives its multi-CTA family (several CTAs per row; probabilities over a vocabulary of at
#: least ``SPLIT_MIN_VOCAB``); a larger call takes one CTA per row.
SPLIT_MAX_ROWS = 32
SPLIT_MIN_VOCAB = 65536
#: Most row chunks a call is split into (one stream each); a larger call stays one call. At top-p 0.95 three
#: parallel chunks still save ~0.1 ms at 65-72 rows but only ~0.04 at 96; four save ~0.03 at 97-128.
MAX_CHUNKS = 3


def split_row_chunks(rows: int) -> list[tuple[int, int]]:
    """``[begin, end)`` bounds of the fewest even row chunks of at most ``SPLIT_MAX_ROWS`` rows."""
    count = -(-rows // SPLIT_MAX_ROWS)
    base, extra = divmod(rows, count)
    bounds, begin = [], 0
    for i in range(count):
        end = begin + base + (1 if i < extra else 0)
        bounds.append((begin, end))
        begin = end
    return bounds


class SplitStreams:
    """The auxiliary streams (one per chunk past the first) and the fork / join events of a split call."""

    def __init__(self, count: int = MAX_CHUNKS - 1):
        self.streams = [torch.cuda.Stream() for _ in range(count)]
        self.fork = torch.cuda.Event()
        self.joins = [torch.cuda.Event() for _ in range(count)]


def _engine(logits, temperatures, top_ks, top_ps, min_ps):
    return torch.ops.trtllm.fused_compute_probs_from_logits(
        logits, temperatures, top_ks, top_ps, min_ps
    )


class FusedComputeProbsSplit(OpWrapper):
    """``[rows, vocab]`` float32: the fused sampler's filtered target distribution of every row (as
    ``fused_compute_probs_from_logits``), each row computed by the sampler's multi-CTA family.

    A call of ``SPLIT_MAX_ROWS < rows <= MAX_CHUNKS * SPLIT_MAX_ROWS`` rows over a vocabulary of at least
    ``SPLIT_MIN_VOCAB`` runs as even row chunks of at most ``SPLIT_MAX_ROWS`` rows, one sampler call each: the first
    on the current stream, the others on ``streams``' auxiliary streams (one after another on the current stream
    without ``streams``, outside multi-stream execution or under torch.compile), concatenated in row order. Other
    calls are one sampler call. Arguments as ``fused_compute_probs_from_logits``.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TEP16 w5 B7 verify (35 rows) at the card's T=1.0, top-p 0.95: two chunks",
            spec=sampling_spec(35, top_p=0.95),
        ),
        Cell(
            why="w8 verify of 8 requests (64 rows) at T=1.0, top-p 0.95: two full chunks",
            spec=sampling_spec(64, top_p=0.95),
        ),
        Cell(
            why="a mixed 48-row batch: per-row filters and greedy-sentinel rows (T=1e-5)",
            spec=sampling_spec(48, top_k=50, top_p=0.9, min_p=0.02, greedy_rows=6, mixed=True),
        ),
        Cell(
            why="every filter neutral at 60 rows (T=0.7): the softmax sum alone sets every value",
            spec=sampling_spec(60, temperature=0.7),
        ),
        Cell(
            why="w8 verify of 9 requests (72 rows), three chunks, top-k 20 with top-p 0.95 at T=0.6",
            spec=sampling_spec(72, temperature=0.6, top_k=20, top_p=0.95),
        ),
        Cell(
            why="96 rows, three full chunks, mixed filters",
            spec=sampling_spec(96, top_k=50, top_p=0.9, min_p=0.02, greedy_rows=12, mixed=True),
        ),
    )
    note = """
    The Kimi K3.1 tep target's worker `_compute_probs_from_logits` calls it for the target distributions of DFlash2
    block verification (spec_dflash2_verify `dflash2_sample_and_accept`) when advanced_sampling_mode is FULL, the
    default: the gathered float32 logits of the generation rows, `num_gens x (K + 1)` (TEP16 w5 B7-B12 = 35-60
    rows, w8 B5-B8 = 40-64), with the per-token sampling tensors of the spec metadata.

    Measured on GB300 (one call vs parallel chunks, us, CUDA graph): top-p 0.95 rows 271 vs 135-173 at 33-64
    rows, 271 vs 174-178 at 65-72, 283 vs 247 at 96; rows without a filter, which the one-CTA family computes
    in one cheap pass, 41 vs 33-37 at 33-35 rows but 41 vs 44-57 at 40-64 and 61 vs 88 at 96.

    Each row's values are bit for bit what the sampler gives that row in a call of at most 32 rows: the family a
    row gets does not depend on the rows beside it, and the chunks are independent calls. A single call of 33-96
    rows instead takes one CTA per row, whose softmax sum is reduced in a different order: rows without a filter
    then differ in the last bits, and a top-p row can differ in the one token on its cut. That family is also not
    run-to-run reproducible on top-p rows (its radix histogram accumulates masses with float atomics); the
    multi-CTA family shares the caveat. No RNG; CUDA-graph capturable, the streams forked and joined in the graph.

    `reference` is each row's own one-row sampler call, which the test compares bit for bit.
    """

    def __call__(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
        streams: Optional[SplitStreams] = None,
    ) -> torch.Tensor:
        rows, vocab = logits.shape
        if rows <= SPLIT_MAX_ROWS or rows > MAX_CHUNKS * SPLIT_MAX_ROWS or vocab < SPLIT_MIN_VOCAB:
            return _engine(logits, temperatures, top_ks, top_ps, min_ps)
        bounds = split_row_chunks(rows)

        def chunk(i: int) -> torch.Tensor:
            b, e = bounds[i]
            return _engine(logits[b:e], temperatures[b:e], top_ks[b:e], top_ps[b:e], min_ps[b:e])

        parallel = (
            streams is not None
            and len(streams.streams) >= len(bounds) - 1
            and do_multi_stream()
            and not torch.compiler.is_compiling()
        )
        if not parallel:
            return torch.cat([chunk(i) for i in range(len(bounds))])
        current = torch.cuda.current_stream()
        streams.fork.record(current)
        outs: list[Optional[torch.Tensor]] = [None] * len(bounds)
        for i in range(1, len(bounds)):
            aux = streams.streams[i - 1]
            with torch.cuda.stream(aux):
                streams.fork.wait(aux)
                outs[i] = chunk(i)
                streams.joins[i - 1].record(aux)
        outs[0] = chunk(0)
        for i in range(1, len(bounds)):
            streams.joins[i - 1].wait(current)
        if not torch.cuda.is_current_stream_capturing():
            # Allocated in auxiliary-stream order and read on this stream.
            for t in outs[1:]:
                t.record_stream(current)
        return torch.cat(outs)

    def reference(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
        streams: Optional[SplitStreams] = None,
    ) -> torch.Tensor:
        """Each row's own one-row sampler call (the multi-CTA family), concatenated."""
        return torch.cat(
            [
                _engine(
                    logits[r : r + 1],
                    temperatures[r : r + 1],
                    top_ks[r : r + 1],
                    top_ps[r : r + 1],
                    min_ps[r : r + 1],
                )
                for r in range(logits.shape[0])
            ]
        )

    def is_valid(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        top_ps: torch.Tensor,
        min_ps: torch.Tensor,
        streams: Optional[SplitStreams] = None,
    ) -> None:
        assert logits.dim() == 2 and logits.is_contiguous(), (
            "logits must be contiguous [rows, vocab]"
        )
        rows = logits.shape[0]
        for t in (temperatures, top_ks, top_ps, min_ps):
            assert t.dim() == 1 and t.shape[0] >= rows and t.is_contiguous(), (
                "per-row tensors need one contiguous entry per row"
            )


fused_compute_probs_split = FusedComputeProbsSplit()
