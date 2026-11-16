# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A sequence-parallel context step's reduce-scatters overlapped with the work around them.

What it changes. On a sequence-parallel context step (``forward.ContextSequenceParallel``)
every layer reduce-scatters two BF16 partial sums on the main stream, each exposed: the
output projection's (``[rows, 8192]``, issued by the built-in decoder layer) and the
experts' (``[rows, 4096]``, ``moe.ContextSequenceParallelMoE``). Three levers, all off by
default (``levers.RULES``: on only by their knobs), overlap them:

* ``context_sp_oproj_overlap`` (``CHUNKS``, ``MIN_ROWS``, ``HIGH_PRIORITY``) turns on the
  built-in overlapped output projection (``MuseSparkOpenAttention._overlap_sp_output_projection``,
  the switches ``MUSESPARK_OPEN_OVERLAP_SP_PROJECTION`` and ``..._GRAPH``) on every layer, in
  captured steps too, for steps of ``MIN_ROWS`` rows or more: the projection runs on
  ``CHUNKS`` owner-major row chunks and each chunk's reduce-scatter runs on a side stream
  (``HIGH_PRIORITY``: a high-priority one) while the next chunk's GEMM runs.
* ``context_sp_tail_pipeline`` (on the projection overlap and the router-input join) has
  each layer join and prepare its MoE input chunk by chunk behind those reduce-scatters
  (``router_input.tail_pipeline``): chunk ``c``'s attention-side join, pooler GEMM and NVFP4
  quantization run while chunk ``c + 1`` is still being reduced; the router GEMM runs once
  on the whole shard, as before.
* ``context_sp_moe_rs_overlap`` (``CHUNKS``, ``MIN_ROWS``, ``HIGH_PRIORITY``) reduce-scatters
  the experts' combined partial sums in owner-major row chunks on a side stream, and unpools
  chunk ``c`` on the main stream while chunk ``c + 1`` is reduced (``reduce_scatter_unpool``).

The side streams are created on the first prefill step (``CommOverlap.build``, from
``ContextSequenceParallel.reserve``), before any CUDA graph is captured. Every side stream
forks from the main stream after the producer it reads and is joined (events, then a
``wait_stream``) before anything reads what it wrote, so the breakable prefill graph captures
the forks and joins (its ``wait_stream`` hook joins a forked stream at a segment end too).

Ordering against the copy-engine gather (``ce_gather``): a peer writes this rank's receive
buffer for layer ``L + 1`` only after its layer ``L`` passed both reduce-scatters, which
need this rank's contributions. Chunked, each contribution is still issued (on the side
stream) after this rank's layer ``L`` row-block QKV and flag resets (the side stream forks
from the main stream after them), and the next gather is enqueued after the main stream
joined every chunk, so the invariant holds unchanged.

Numerics: bitwise by construction where (a) the BF16 GEMMs that run on row chunks (the
output projection, the pooler, the unpooler) compute every row as the whole call does, and
(b) NCCL reduces every element in the same order for a chunk as for the whole buffer. Both
held on GB300 (the built-in 4-rank test ``test_overlapped_projection_chunks_bitwise``; the
pooler at 1024/1024, 512/1536, 1023/1025 splits); on sm107 they are what the GPU tests of
this module and the built-in one check. Where either fails the levers are lossless-numeric:
the same FP32 GEMM accumulation and BF16 reduce-scatter adds in another order, nothing
approximated.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch

from tensorrt_llm.logger import logger

#: ``context_sp_oproj_overlap``: the output projection's chunks (0: the lever is off), the
#: fewest rows of a step it takes, and whether its reduce-scatters run on a high-priority stream.
OPROJ_CHUNKS = 0
OPROJ_MIN_ROWS = 8192
OPROJ_HIGH_PRIORITY = True
#: ``context_sp_tail_pipeline``: the attention-side join and MoE input preparation per chunk.
TAIL_PIPELINE = False
#: ``context_sp_moe_rs_overlap``: the experts' reduce-scatter chunks (0: off), the fewest rows
#: of a step it takes, and whether they run on a high-priority stream.
MOE_RS_CHUNKS = 0
MOE_RS_MIN_ROWS = 8192
MOE_RS_HIGH_PRIORITY = True

#: The most chunks either overlap takes: with the path's smallest step (``forward.MIN_ROWS``,
#: a 256-row shard) every chunk keeps more rows than the residual join's cluster-boundary
#: band (32), which the tail pipeline's fused join needs.
MAX_CHUNKS = 4


def configure(
    *,
    oproj_chunks: int = 0,
    oproj_min_rows: int = 8192,
    oproj_high_priority: bool = True,
    tail_pipeline: bool = False,
    moe_rs_chunks: int = 0,
    moe_rs_min_rows: int = 8192,
    moe_rs_high_priority: bool = True,
) -> None:
    """Set what ``CommOverlap.install`` wires (``lever_settings.apply``, from the plan).

    Raises:
        ValueError: for a chunk count outside ``0..MAX_CHUNKS``, a row floor below one, or
            the tail pipeline without the projection overlap.
    """
    global OPROJ_CHUNKS, OPROJ_MIN_ROWS, OPROJ_HIGH_PRIORITY, TAIL_PIPELINE
    global MOE_RS_CHUNKS, MOE_RS_MIN_ROWS, MOE_RS_HIGH_PRIORITY
    for name, chunks in (("oproj_chunks", oproj_chunks), ("moe_rs_chunks", moe_rs_chunks)):
        if not 0 <= int(chunks) <= MAX_CHUNKS:
            raise ValueError(
                f"[musespark_open] context_sequence_parallel overlap: {name} must be in "
                f"0..{MAX_CHUNKS}, got {chunks}"
            )
    for name, rows in (("oproj_min_rows", oproj_min_rows), ("moe_rs_min_rows", moe_rs_min_rows)):
        if int(rows) < 1:
            raise ValueError(
                f"[musespark_open] context_sequence_parallel overlap: {name} must be positive, "
                f"got {rows}"
            )
    if tail_pipeline and not oproj_chunks:
        raise ValueError(
            "[musespark_open] context_sequence_parallel overlap: the tail pipeline runs behind the "
            "overlapped output projection (context_sp_oproj_overlap)"
        )
    OPROJ_CHUNKS = int(oproj_chunks)
    OPROJ_MIN_ROWS = int(oproj_min_rows)
    OPROJ_HIGH_PRIORITY = bool(oproj_high_priority)
    TAIL_PIPELINE = bool(tail_pipeline)
    MOE_RS_CHUNKS = int(moe_rs_chunks)
    MOE_RS_MIN_ROWS = int(moe_rs_min_rows)
    MOE_RS_HIGH_PRIORITY = bool(moe_rs_high_priority)
    if OPROJ_CHUNKS or MOE_RS_CHUNKS:
        logger.info(
            "[musespark_open] context_sequence_parallel overlap: o_proj reduce-scatter "
            + (
                f"in {OPROJ_CHUNKS} chunks from {OPROJ_MIN_ROWS} rows"
                + (" with the tail pipeline" if TAIL_PIPELINE else "")
                if OPROJ_CHUNKS
                else "unchanged"
            )
            + "; experts reduce-scatter "
            + (
                f"in {MOE_RS_CHUNKS} chunks from {MOE_RS_MIN_ROWS} rows"
                if MOE_RS_CHUNKS
                else "unchanged"
            )
        )


def chunk_bounds(shard: int, chunks: int) -> List[int]:
    """Chunk ``c`` covers rows ``[bounds[c], bounds[c + 1])`` of every owner's ``shard`` rows.

    The built-in overlapped projection's split (``ceil(c * shard / chunks)``), with at most
    ``shard`` chunks, so no chunk is empty.
    """
    chunks = max(1, min(int(chunks), int(shard)))
    return [-(-shard * c // chunks) for c in range(chunks + 1)]


def _stream(high_priority: bool) -> torch.cuda.Stream:
    """A side stream of the current device, at the device's highest priority when asked.

    The lower number is the higher priority; ``priority_range`` (where this torch has it) is
    ``(lowest, highest)``, and ``-1`` is a high priority on every torch.
    """
    if not high_priority:
        return torch.cuda.Stream()
    priority_range = getattr(torch.cuda.Stream, "priority_range", None)
    highest = priority_range()[1] if callable(priority_range) else -1
    return torch.cuda.Stream(priority=highest)


def reduce_scatter_unpool(
    combined: torch.Tensor,
    mapping,
    chunks: int,
    stream: torch.cuda.Stream,
    unpool: Callable[[torch.Tensor], torch.Tensor],
    reducescatter: Optional[Callable] = None,
) -> torch.Tensor:
    """``unpool(reducescatter(combined))`` with the reduce-scatter in owner-major row chunks.

    ``combined`` is ``[world * shard, width]``, owner-major (rank ``r``'s rows are
    ``[r * shard, (r + 1) * shard)``), already padded. Chunk ``c`` packs rows ``[lo, hi)`` of
    every owner's block (``chunk_bounds``) and is reduce-scattered on ``stream`` after the
    main stream's pack; the main stream unpools chunk ``c`` after its reduce-scatter's event,
    while chunk ``c + 1`` is still being reduced, and the chunks' outputs are concatenated in
    row order. Returns ``[shard, unpooled width]``, this rank's rows.

    Every element's reduce-scatter adds the same partials as the whole buffer's call; every
    row's unpool is ``unpool`` of that row (see the module docstring for when that is bitwise).
    """
    if reducescatter is None:
        from tensorrt_llm._torch.distributed import reducescatter
    world = int(mapping.tp_size)
    if combined.shape[0] % world:
        raise ValueError(
            f"[musespark_open] reduce_scatter_unpool: {combined.shape[0]} rows do not split "
            f"over {world} ranks"
        )
    shard = combined.shape[0] // world
    bounds = chunk_bounds(shard, chunks)
    main = torch.cuda.current_stream(combined.device)
    pending: List[Tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]] = []
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        if len(bounds) == 2:
            packed = combined
        else:
            packed = torch.cat([combined[r * shard + lo : r * shard + hi] for r in range(world)], 0)
        stream.wait_stream(main)
        with torch.cuda.stream(stream):
            reduced = reducescatter(packed, mapping, dim=0)
            event = torch.cuda.Event()
            event.record(stream)
        # ``packed`` (main-stream memory the collective reads) is kept until the main stream
        # has waited for every chunk, so its next reuse on main is ordered after the read.
        pending.append((packed, reduced, event))
    outputs = []
    for _, reduced, event in pending:
        main.wait_event(event)
        outputs.append(unpool(reduced))
    # The reduced chunks were allocated on ``stream`` and read on main: order their next reuse
    # on ``stream`` after these reads.
    stream.wait_stream(main)
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, 0)


class CommOverlap:
    """The overlap levers of one model's sequence-parallel path: what ``install`` wired, and
    the side streams ``build`` makes on the first prefill step."""

    def __init__(self) -> None:
        self.oproj_chunks = OPROJ_CHUNKS
        self.oproj_min_rows = OPROJ_MIN_ROWS
        self.oproj_high_priority = OPROJ_HIGH_PRIORITY
        self.tail_pipeline = TAIL_PIPELINE
        self.moe_rs_chunks = MOE_RS_CHUNKS
        self.moe_rs_min_rows = MOE_RS_MIN_ROWS
        self.moe_rs_high_priority = MOE_RS_HIGH_PRIORITY
        self.layers: list = []
        self.oproj_stream: Optional[torch.cuda.Stream] = None
        self.moe_rs_stream: Optional[torch.cuda.Stream] = None

    def __bool__(self) -> bool:
        return bool(self.oproj_chunks or self.moe_rs_chunks)

    def install(self, model) -> None:
        """Set every layer's overlap attributes; the streams come with ``build``.

        Until ``build`` the projection keeps the model's stream (``aux_stream``) and the
        experts' reduce-scatter its unchunked call, so nothing runs chunked before the side
        streams exist.
        """
        self.layers = list(model.layers)
        for layer in self.layers:
            if self.oproj_chunks:
                layer._overlap_sp_projection_enabled = True
                layer._overlap_sp_projection_graph = True
                layer._overlap_sp_projection_min_rows = self.oproj_min_rows
                layer.self_attn._sp_projection_chunks = self.oproj_chunks
            moe = layer.feed_forward
            moe._sp_rs_chunks = 0
            moe._sp_rs_min_rows = self.moe_rs_min_rows
            moe._sp_rs_stream = None

    def build(self) -> None:
        """Make the side streams (eager, before any capture) and hand them to the layers."""
        if self.oproj_chunks and self.oproj_stream is None:
            self.oproj_stream = _stream(self.oproj_high_priority)
            for layer in self.layers:
                layer.self_attn._sp_projection_stream = self.oproj_stream
        if self.moe_rs_chunks and self.moe_rs_stream is None:
            self.moe_rs_stream = _stream(self.moe_rs_high_priority)
            for layer in self.layers:
                moe = layer.feed_forward
                moe._sp_rs_stream = self.moe_rs_stream
                moe._sp_rs_chunks = self.moe_rs_chunks
