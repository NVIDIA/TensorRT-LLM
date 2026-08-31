# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-graph replay of the device work of a generation-step prepare.

Preparing an unchanged generation batch issues a handful of tiny device
operations -- advance the positions, gather the committed token, refresh the
sequence lengths, the block table and the sparse-attention token mapping, stage
the model's inputs. Collected with ``begin_device_work()`` they can be issued
back to back, but each still costs a launch, and none of them is big enough to
cover the next, so the device idles between them. Replaying them from a graph
costs one launch and spaces the nodes at graph-node latency instead.

A capture bakes in the addresses of the tensors it was traced with, so the
graph is keyed on what can move between steps and re-captured when it does --
which happens when the batch changes, not every step. The operations write
buffers that are allocated once and reused, which is what makes them
replayable at all; the values they read are refreshed in place by the host
half of the same prepare, before the replay runs.

Once a graph exists the step's operations are not recorded again: the caller's
region drops them and the replay is what performs them. Collecting them only to
throw the list away is pure host cost on the decode critical path, which is a
material part of what a step this small is worth. The sequence they form is a
property of the key, but it is re-recorded and checked against the captured one
every ``RECHECK_STEPS`` steps rather than being taken on faith.

The capture is taken on the caller's stream rather than a private one, because
some of the collected work is issued to a stream of its own choosing (the KV
cache manager copies block offsets on its execution stream) and would escape a
capture taken anywhere else.
"""

from __future__ import annotations

import gc
from typing import List, Optional

import torch

from tensorrt_llm.logger import logger

from ..utils import DeviceWorkItem, begin_device_work, end_device_work, run_device_work_items

__all__ = ["SteadyGenPrepGraph"]

#: Steps a key must repeat before its work is captured. Capture is only worth
#: its cost, and only safe to attempt, for a step shape that is going to stay
#: put; until then the work is issued as it was collected.
WARMUP_STEPS = 2

#: How often a replaying step records its operations anyway and checks them
#: against the captured graph. Cheap enough to be free at this interval, and
#: the only thing standing between a step that quietly changed shape and a
#: graph that no longer matches it.
RECHECK_STEPS = 64


class SteadyGenPrepGraph:
    """Replays the collected device work of a steady-state generation prepare."""

    def __init__(self) -> None:
        self._enabled = True
        self._key: Optional[tuple] = None
        self._steps_on_key = 0
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._captured_len = -1

    @property
    def enabled(self) -> bool:
        return self._enabled

    def begin(self, key: tuple) -> Optional[List[DeviceWorkItem]]:
        """Open the collection region for this step.

        Returns the list to collect this step's device work into, or None when
        it is already captured under `key` and :meth:`issue` will replay it.
        """
        if key != self._key:
            self._key = key
            self._steps_on_key = 0
            self._graph = None
            self._captured_len = -1
        self._steps_on_key += 1
        collect = self._graph is None or self._steps_on_key % RECHECK_STEPS == 0
        return begin_device_work(collect)

    def end(self) -> None:
        """Close the region opened by :meth:`begin`. Always call this."""
        end_device_work()

    def issue(self, work: Optional[List[DeviceWorkItem]]) -> bool:
        """Perform this step's device work; True if it was replayed."""
        if work is None:
            self._graph.replay()
            return True

        if self._graph is not None:
            # A recheck step: the graph performs exactly the operations it was
            # traced from, so it is only this step's work while the step still
            # produces that same sequence.
            if len(work) == self._captured_len:
                self._graph.replay()
                return True
            logger.warning(
                f"Generation-step prepare now issues {len(work)} operations "
                f"where its graph was captured from {self._captured_len}; "
                "re-capturing."
            )
            self._graph = None
            self._captured_len = -1
            self._steps_on_key = 0

        if (
            not self._enabled
            or not work
            or self._steps_on_key <= WARMUP_STEPS
            or torch.cuda.is_current_stream_capturing()
        ):
            # The last is a capture already in progress; one cannot nest.
            run_device_work_items(work)
            return False

        try:
            graph = self._capture(work)
        except RuntimeError as error:
            # A prepare that cannot be captured still has to happen, and
            # retrying a capture that failed once is not worth the risk, so
            # drop to issuing the work for good. Nothing runs on the device
            # while a capture is being taken, so the failed attempt has left
            # this step's work undone.
            logger.warning(f"Generation-step prepare will not be CUDA-graphed: {error}")
            self._enabled = False
            run_device_work_items(work)
            return False

        self._graph = graph
        self._captured_len = len(work)
        # A capture records without running, so this step's work has not
        # happened yet; the replay is what performs it.
        graph.replay()
        return True

    def _capture(self, work: List[DeviceWorkItem]) -> torch.cuda.CUDAGraph:
        graph = torch.cuda.CUDAGraph()
        # Freeing device or pinned memory invalidates an in-progress capture,
        # and the cyclic collector can run at any allocation, so hold it off
        # for the handful of launches the capture takes.
        collecting = gc.isenabled()
        gc.disable()
        try:
            # torch.cuda.graph() is not used here: it captures on a stream of
            # its own, which work bound to another stream would escape, and
            # its preamble synchronizes the device and drops both allocator
            # caches to make room for the graph pool, which is far too
            # expensive to pay while serving and buys nothing for a body that
            # allocates nothing. "thread_local" leaves other threads free to
            # keep issuing their own CUDA work.
            graph.capture_begin(capture_error_mode="thread_local")
            try:
                run_device_work_items(work)
            finally:
                graph.capture_end()
        finally:
            if collecting:
                gc.enable()
        return graph
