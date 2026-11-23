# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rows of a context step whose drafter context is written, when only a window is kept. Pure.

A generation server's paged drafter reads the drafter context rows of the last ``window``
positions before a request's prompt end, and nothing below. When the context server's KV
cache manager keeps only such a window (``KVCacheManagerV2.enable_draft_context_window``,
which also guards reuse so that nobody reads a row that was not written), the model engine
names, per context request, the first prompt position whose row is written
(``SpecMetadata.draft_context_keep_from``). A context step then writes the rows of positions
at or after it and nothing else: the captures cross the stage boundaries for those rows (and
the lead rows below) only, and the last stage projects those rows and stores the kept ones.

``plan_window`` turns the step's context rows into a ``WindowPlan``: per request, how many of
its leading rows are dropped and where its kept rows sit in the step and among the carried rows.
Every stage plans from the same schedule (the chunk sizes, chunk starts and keep positions the
model engine prepared for the micro-batch), so the two ends of an edge agree on the message.

**Lead rows.** The kept rows' drafter context must be bit for bit the whole write's, and its
GEMMs (the projection over the step's rows, the fused K/V GEMM over each request's rows) are
not row-independent at every row count: below ``MIN_GEMM_ROWS`` rows cuBLAS can pick a split-K
kernel, which sums the reduction dimension in another order than the kernels it picks for the
whole step. So a request whose kept rows are fewer than ``MIN_GEMM_ROWS`` (and fewer than its
chunk) also carries the rows just ahead of them (its ``lead``), up to ``MIN_GEMM_ROWS`` rows or
its whole chunk: its K/V GEMM then runs over its whole chunk, as the whole write's does, or over
at least ``MIN_GEMM_ROWS`` rows. When every request that keeps a row carries its whole chunk and
the step still carries fewer than ``min(step rows, MIN_GEMM_ROWS)`` rows, requests that keep no
row carry their last rows until it does, so the projection also runs over the whole step or over
at least ``MIN_GEMM_ROWS`` rows. Lead rows cross the stage boundaries and are projected (and, for
a request that keeps rows, run through its K/V GEMM), and are never stored. Every other op of the
write (hidden_norm, the K copy, k_norm, RoPE) works on each row alone, so they change no kept
row.
"""

from __future__ import annotations

import dataclasses
from typing import Any, List, Optional, Sequence, Tuple

import torch

#: The fewest rows the window write runs a projection or K/V GEMM over, unless it runs over
#: every row of the step (projection) or of the request's chunk (K/V GEMM) as the whole write
#: does. From this many rows on, cuBLAS runs these GEMM shapes on the served GPU without split-K,
#: and a row's projection and K/V come out with the same bits at every row count up to a step's
#: (the window-shape GPU tests check every row count).
MIN_GEMM_ROWS = 512


@dataclasses.dataclass(frozen=True)
class WindowPlan:
    """The kept rows of one context step, and the rows carried to compute them.

    ``lengths`` are the context requests' rows in step order, ``starts`` the prompt position
    of each request's first row, ``skips`` how many of its leading rows are dropped
    (``0 <= skip <= length``) and ``leads`` how many of the dropped rows just ahead of its kept
    rows are carried and computed but not stored (``0 <= lead <= skip``).
    """

    lengths: Tuple[int, ...]
    starts: Tuple[int, ...]
    skips: Tuple[int, ...]
    leads: Tuple[int, ...]

    @property
    def kept(self) -> Tuple[int, ...]:
        """The kept (stored) rows of each request."""
        return tuple(length - skip for length, skip in zip(self.lengths, self.skips))

    @property
    def carried(self) -> Tuple[int, ...]:
        """The carried rows of each request: its lead rows, then its kept rows."""
        return tuple(kept + lead for kept, lead in zip(self.kept, self.leads))

    @property
    def count(self) -> int:
        """The carried rows of the step."""
        return sum(self.carried)

    def ranges(self) -> List[Tuple[int, int]]:
        """``[begin, end)`` step rows of each request's carried rows (empty ones included)."""
        out, offset = [], 0
        for length, skip, lead in zip(self.lengths, self.skips, self.leads):
            out.append((offset + skip - lead, offset + length))
            offset += length
        return out

    def offsets(self) -> List[int]:
        """Where each request's carried rows start among the carried rows of the step; its kept
        rows start ``lead`` rows later."""
        out, offset = [], 0
        for carried in self.carried:
            out.append(offset)
            offset += carried
        return out

    def span(self) -> Optional[Tuple[int, int]]:
        """``(begin, count)`` when the carried rows are the one run of step rows ``[begin,
        begin + count)`` (the requests' non-empty carried ranges follow each other), else None
        (also when the step carries no row). Such rows are a slice of every per-row tensor of
        the step, with the rows ``index`` selects in the same order."""
        begin = end = None
        for first, last in self.ranges():
            if first == last:
                continue
            if begin is None:
                begin, end = first, last
            elif first == end:
                end = last
            else:
                return None
        return None if begin is None else (begin, end - begin)

    def index(self, device: Any) -> torch.Tensor:
        """The carried step rows in order, ``int64 [count]`` on ``device``."""
        parts = [torch.arange(begin, end, dtype=torch.int64) for begin, end in self.ranges()]
        rows = torch.cat(parts) if parts else torch.zeros(0, dtype=torch.int64)
        device = torch.device(device)
        if device.type == "cpu":
            return rows
        from tensorrt_llm._utils import maybe_pin_memory

        return maybe_pin_memory(rows).to(device, non_blocking=True)


def plan_leads(
    lengths: Sequence[int], skips: Sequence[int], floor: Optional[int] = None
) -> Tuple[int, ...]:
    """The lead rows of requests with ``lengths`` rows that drop their first ``skips`` rows.

    A request that keeps ``kept > 0`` rows carries ``min(length, max(kept, floor))`` rows, so
    its K/V GEMM runs over its whole chunk or over at least ``floor`` rows. If the step then
    carries fewer than ``min(sum(lengths), floor)`` rows, no request carries ``floor`` rows, so
    every request that keeps a row carries its whole chunk and the rows the step lacks are rows
    of requests that keep none: those carry their last rows, in step order, until the step
    carries ``min(sum(lengths), floor)`` rows. A step that keeps no row carries none. ``floor``
    is ``MIN_GEMM_ROWS`` when None.
    """
    floor = MIN_GEMM_ROWS if floor is None else int(floor)
    kept = [int(length) - int(skip) for length, skip in zip(lengths, skips)]
    leads = [
        min(int(skip), max(floor - count, 0)) if count else 0 for skip, count in zip(skips, kept)
    ]
    carried = sum(kept) + sum(leads)
    need = min(sum(int(length) for length in lengths), floor) - carried
    if carried == 0 or need <= 0:
        return tuple(leads)
    for i, (length, count) in enumerate(zip(lengths, kept)):
        if need == 0:
            break
        if count == 0:
            leads[i] = min(int(length), need)
            need -= leads[i]
    return tuple(leads)


def plan_window(
    lengths: Sequence[int],
    starts: Sequence[int],
    keep_from: Sequence[int],
    floor: Optional[int] = None,
) -> WindowPlan:
    """The ``WindowPlan`` of context requests with ``lengths`` rows starting at prompt position
    ``starts``, each keeping the rows of positions ``>= keep_from``, with the lead rows of
    ``plan_leads`` (``floor`` rows, ``MIN_GEMM_ROWS`` when None).

    Raises:
        ValueError: when the three lists differ in length.
    """
    if not len(lengths) == len(starts) == len(keep_from):
        raise ValueError(
            f"{len(lengths)} chunk lengths, {len(starts)} chunk starts and {len(keep_from)} "
            "keep positions for one step's context requests"
        )
    skips = tuple(
        min(max(int(first) - int(start), 0), int(length))
        for length, start, first in zip(lengths, starts, keep_from)
    )
    return WindowPlan(
        tuple(int(length) for length in lengths),
        tuple(int(start) for start in starts),
        skips,
        plan_leads(lengths, skips, floor),
    )


def step_plan(attn_metadata: Any, spec_metadata: Any) -> Optional[WindowPlan]:
    """The ``WindowPlan`` of a step, or None when the step writes every row.

    None for a step without context rows, without speculative metadata or capture buffer, or
    whose metadata names no keep positions (``draft_context_keep_from``). The chunk starts are
    the host position ids of each request's first row (the drafter context write reads the
    same ones), else the KV cache's cached-token counts.
    """
    num_contexts = int(getattr(attn_metadata, "num_contexts", 0) or 0)
    if num_contexts == 0 or spec_metadata is None:
        return None
    keep_from = getattr(spec_metadata, "draft_context_keep_from", None)
    if keep_from is None or getattr(spec_metadata, "captured_hidden_states", None) is None:
        return None
    if len(keep_from) != num_contexts:
        raise ValueError(
            f"{len(keep_from)} drafter context keep positions for a step of {num_contexts} "
            "context requests"
        )
    lengths = [int(n) for n in attn_metadata.seq_lens[:num_contexts].tolist()]
    host_positions = getattr(spec_metadata, "host_position_ids", None)
    if host_positions is not None:
        starts, offset = [], 0
        for length in lengths:
            starts.append(int(host_positions[offset]) if length else 0)
            offset += length
    else:
        cached = attn_metadata.kv_cache_params.num_cached_tokens_per_seq
        starts = [int(cached[i]) for i in range(num_contexts)]
    return plan_window(lengths, starts, keep_from)
