# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Delayed reclamation of eager attention scratch at model-forward boundaries."""

import weakref
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

import torch

from tensorrt_llm.logger import logger

from ..modules.multi_stream_utils import do_multi_stream

if TYPE_CHECKING:
    from .backends.trtllm import TrtllmAttentionMetadata


class WorkspaceShrinkPolicy:
    """Shrink after three underfilled forwards using their maximum demand."""

    def __init__(self, floor_bytes: int) -> None:
        if floor_bytes < 0:
            raise ValueError("Workspace floor must be nonnegative")
        self.floor_bytes = floor_bytes
        self.reset()

    def reset(self) -> None:
        self.remaining = 3
        self._window_max = 0

    def finish_forward(self, required_bytes: int, capacity_bytes: int, *, grew: bool) -> int | None:
        if required_bytes < 0 or capacity_bytes < max(required_bytes, self.floor_bytes):
            raise ValueError("Workspace capacity is below its requirement or warmup floor")
        if required_bytes == 0:
            return None
        if grew or capacity_bytes == self.floor_bytes or required_bytes == capacity_bytes:
            self.reset()
            return None
        self._window_max = max(self._window_max, required_bytes)
        self.remaining -= 1
        if self.remaining:
            return None
        target = max(self.floor_bytes, self._window_max)
        self.reset()
        return target


class EagerWorkspaceReclaimer:
    """Reclaim single-stream eager scratch, preserving the warmup floor."""

    def __init__(self, metadata: "TrtllmAttentionMetadata") -> None:
        if metadata.workspace is None:
            raise ValueError("Warmup must initialize eager workspace before reclamation")
        self.policy = WorkspaceShrinkPolicy(metadata.workspace.untyped_storage().nbytes())
        self._metadata = weakref.ref(metadata)
        self._stream: torch.cuda.Stream | None = None
        self._active = False
        self._disabled = False

    @contextmanager
    def forward(self, metadata: "TrtllmAttentionMetadata") -> Iterator[None]:
        if self._active:
            raise RuntimeError("Overlapping forwards cannot share eager attention workspace")
        if metadata.is_cuda_graph or torch.cuda.is_current_stream_capturing():
            yield
            return
        if metadata is not self._metadata():
            raise RuntimeError("Eager workspace metadata changed after warmup")
        if self._disabled or not metadata.workspace_reclaimable:
            self.policy.reset()
            yield
            return
        stream = torch.cuda.current_stream(torch.cuda.current_device())
        if do_multi_stream() or (self._stream is not None and stream != self._stream):
            self._disabled = True
            logger.warning("Disabling eager workspace reclamation after a stream change")
            yield
            return
        self._stream = stream
        if metadata.workspace is None:
            raise RuntimeError("Eager workspace was removed outside its reclaimer")
        capacity_before = metadata.workspace.untyped_storage().nbytes()
        # Protect the consumer if warmup allocated this storage on another stream.
        metadata.workspace.record_stream(self._stream)
        # At the warmup floor nothing can be reclaimed. A growing forward
        # resets the policy anyway; start collecting demand on the next one.
        collect_demand = capacity_before > self.policy.floor_bytes
        if collect_demand:
            # Retain storage; grow-only layer sizing accumulates this forward's peak.
            metadata.workspace.resize_(0)
        self._active = True
        completed = False
        try:
            yield
            completed = True
        finally:
            self._active = False
            if not completed or not metadata.workspace_reclaimable or not collect_demand:
                self.policy.reset()
            else:
                capacity = metadata.workspace.untyped_storage().nbytes()
                target = self.policy.finish_forward(
                    metadata.workspace.numel() * metadata.workspace.element_size(),
                    capacity,
                    grew=capacity > capacity_before,
                )
                if target is not None:
                    device = metadata.workspace.device
                    # resize_(smaller) retains storage. Release first, then
                    # allocate; pure scratch has no payload to preserve.
                    metadata.workspace = None
                    metadata.workspace = torch.empty(target, dtype=torch.uint8, device=device)
                    logger.debug(
                        f"Shrank eager attention workspace from {capacity} to {target} bytes"
                    )
