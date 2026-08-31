# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph replay of the greedy decode tail.

The tail of a single-token greedy step is three strictly dependent device
operations -- the split argmax, the merge-and-scatter, and the four-byte
read-back of the sampled token. None of them is large enough to cover the
launch of the next, so the host pays a full launch latency for each and the
device idles in between. Replaying them from a graph costs one launch and
spaces the nodes at graph-node latency instead.

A capture bakes in the addresses of the tensors it was traced with, so the
graph is keyed on them and re-captured when they move -- which happens
whenever the batch changes, not every step. The read-back destination must
outlive the replay, so this owns a small ring of pinned buffers, one captured
graph each, and hands out a slot per step. A slot is reusable only once the
caller releases it; if a caller never does, the ring runs dry and the tail
falls back to running eagerly.
"""

from __future__ import annotations

import gc
from typing import Optional

import torch

from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.logger import logger

from .greedy_sample_kernels import ARGMAX_SPLITS, greedy_argmax_scatter

__all__ = ["GreedyTailGraph"]

#: Sampled-token buffers held simultaneously. The overlap scheduler keeps one
#: sample state in flight while it builds the next, so two are load-bearing;
#: the rest are headroom for schedulers that hold on to more.
RING_SIZE = 4

#: Steps a set of input addresses must repeat before it is captured. Capture is
#: only worth its cost, and only safe to attempt, for inputs that are going to
#: stay put; until then the caller runs the tail itself.
WARMUP_STEPS = 2


class GreedyTailGraph:
    """Replays argmax + scatter + token read-back from a captured graph."""

    def __init__(self) -> None:
        self._enabled = True
        self._steps_on_key = 0
        self._key: Optional[tuple] = None
        self._shape: Optional[tuple] = None
        self._graphs: list[Optional[torch.cuda.CUDAGraph]] = [None] * RING_SIZE
        self._held: set[int] = set()
        self._host: list[torch.Tensor] = []
        self._next_tokens: Optional[torch.Tensor] = None
        self._partials: Optional[torch.Tensor] = None
        self._capture_stream: Optional[torch.cuda.Stream] = None

    @staticmethod
    def _key_for(
        logits: torch.Tensor,
        new_tokens: torch.Tensor,
        dest_indices: torch.Tensor,
        beam_width: int,
    ) -> tuple:
        return (
            logits.data_ptr(),
            logits.shape,
            logits.stride(),
            logits.dtype,
            new_tokens.data_ptr(),
            new_tokens.shape,
            new_tokens.stride(),
            new_tokens.dtype,
            dest_indices.data_ptr(),
            dest_indices.shape,
            dest_indices.dtype,
            beam_width,
        )

    def _rekey(self, num_rows: int, dtype: torch.dtype, device: torch.device) -> bool:
        """Invalidate the captured graphs for a new set of input addresses.

        Returns False when the tail must stay eager from now on.
        """
        self._steps_on_key = 0
        self._graphs = [None] * RING_SIZE

        shape = (num_rows, dtype, device)
        if shape == self._shape:
            # Same buffers, different input addresses: the slots handed out
            # already stay valid, only the graphs have to be re-traced.
            return True

        # Everything the captured nodes write is allocated here, before any
        # capture, and is not re-allocated while a graph holds its address.
        self._next_tokens = torch.empty(num_rows, dtype=dtype, device=device)
        self._partials = torch.empty((num_rows, ARGMAX_SPLITS), dtype=torch.int64, device=device)
        self._host = [
            torch.empty(num_rows, dtype=dtype, device="cpu", pin_memory=prefer_pinned())
            for _ in range(RING_SIZE)
        ]
        if not all(host.is_pinned() for host in self._host):
            # A pageable destination makes the read-back a blocking copy,
            # which cannot be captured.
            self._enabled = False
            return False
        # The buffers a held slot points at have just been replaced, so the
        # states holding them keep their own (still valid) tensors and the
        # ring starts empty again.
        self._held.clear()
        self._shape = shape
        return True

    def _issue(
        self,
        logits: torch.Tensor,
        new_tokens: torch.Tensor,
        dest_indices: torch.Tensor,
        beam_width: int,
        slot: int,
    ) -> None:
        greedy_argmax_scatter(
            logits,
            new_tokens,
            dest_indices,
            beam_width,
            out=self._next_tokens,
            partials=self._partials,
        )
        self._host[slot].copy_(self._next_tokens, non_blocking=True)

    def _capture(
        self,
        logits: torch.Tensor,
        new_tokens: torch.Tensor,
        dest_indices: torch.Tensor,
        beam_width: int,
        slot: int,
    ) -> torch.cuda.CUDAGraph:
        # The kernels must already be compiled and their modules loaded before
        # capture begins, so issue the sequence once first. It is a pure
        # function of the logits, so running it twice is harmless.
        self._issue(logits, new_tokens, dest_indices, beam_width, slot)

        if self._capture_stream is None:
            self._capture_stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        # Freeing device or pinned memory invalidates an in-progress capture,
        # and the cyclic collector can run at any allocation, so hold it off
        # for the handful of launches the capture takes.
        collecting = gc.isenabled()
        gc.disable()
        try:
            # torch.cuda.graph() is not used here: its preamble synchronizes
            # the device and drops both allocator caches to make room for the
            # graph pool, which is far too expensive to pay while serving and
            # buys nothing for a body that allocates nothing. "thread_local"
            # leaves other threads free to keep issuing their own CUDA work.
            with torch.cuda.stream(self._capture_stream):
                graph.capture_begin(capture_error_mode="thread_local")
                try:
                    self._issue(logits, new_tokens, dest_indices, beam_width, slot)
                finally:
                    graph.capture_end()
        finally:
            if collecting:
                gc.enable()
        return graph

    def run(
        self,
        logits: torch.Tensor,
        new_tokens: torch.Tensor,
        dest_indices: torch.Tensor,
        beam_width: int,
    ) -> Optional[tuple[torch.Tensor, int]]:
        """Sample one token per row and read it back, replaying if possible.

        Returns the pinned host tensor holding the sampled tokens and the ring
        slot to hand back to :meth:`release`, or None when the caller has to
        run the tail itself. As with any non-blocking read-back, the host
        tensor may only be read once the stream this was issued to has been
        synchronized.
        """
        if not self._enabled:
            return None

        key = self._key_for(logits, new_tokens, dest_indices, beam_width)
        if key != self._key:
            if not self._rekey(logits.shape[0], new_tokens.dtype, new_tokens.device):
                return None
            self._key = key

        self._steps_on_key += 1
        if self._steps_on_key <= WARMUP_STEPS:
            return None

        slot = next((i for i in range(RING_SIZE) if i not in self._held), None)
        if slot is None:
            return None

        graph = self._graphs[slot]
        if graph is None:
            if torch.cuda.is_current_stream_capturing():
                # A capture cannot be nested inside another one.
                return None
            try:
                graph = self._capture(logits, new_tokens, dest_indices, beam_width, slot)
            except RuntimeError as error:
                # A tail that cannot be captured still has to produce tokens,
                # and retrying a capture that failed once is not worth the
                # risk, so drop to the caller's eager path for good.
                logger.warning(f"Greedy sampling tail will not be CUDA-graphed: {error}")
                self._enabled = False
                self._graphs = [None] * RING_SIZE
                return None
            self._graphs[slot] = graph

        graph.replay()
        self._held.add(slot)
        return self._host[slot], slot

    def release(self, slot: int) -> None:
        """Return a slot handed out by :meth:`run` to the ring."""
        self._held.discard(slot)
