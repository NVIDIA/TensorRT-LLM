# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack a decode step's small per-step input copies into one upload and one scatter launch.

Between two decode-graph replays, the next step's metadata reaches the device as a chain of tiny copies (positions,
KV lengths, cache slots, and the graph runner's static-input copies). Each is a separate operation on the stream the
next graph waits for, so the GPU idles for the whole chain.

While a pack is open (``begin`` ... ``flush``), ``copy_(dst, src)`` records a host-to-device copy by snapshotting the
host bytes into a pinned staging buffer, and a device-to-device copy as a (dst, src) descriptor. ``flush`` issues one
host-to-device copy of [descriptors | data] into a device staging buffer and one ``trtllm::step_inputs_scatter``
launch, which writes every staged host copy and then every device copy (in that order, so a device copy may read a
destination a host copy just filled). With no pack open, ``copy_`` is ``dst.copy_(src, non_blocking=True)``.

Each engine that armed ``RuntimeHooks.pack_step_inputs`` owns one ``StepInputPack`` (on its device) and makes it the
active pack for its forward (``active``); the module functions act on the active pack of the calling context, so
executors on other threads or devices never share one. The engine opens it only when the step will reach the flush (a
replay of an already captured graph). Route a copy through ``copy_`` only when no device work reads or writes its
destination between the copy site and the flush: the pack moves the copy to the flush point. A copy that cannot be
packed (non-contiguous, device-side dtype change) flushes the pack first and then runs as a plain copy, which keeps
the original order. A pack holds every tensor it staged until it is issued, so a temporary source outlives its copy.
"""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Iterator, Optional

import torch

from tensorrt_llm._utils import prefer_pinned


@torch.library.register_fake("trtllm::step_inputs_scatter")
def _step_inputs_scatter_fake(stage: torch.Tensor) -> None:
    # Writes the staged rows into their destinations; returns nothing.
    return None


class StepInputPack:
    """One engine's pack of deferred step-input copies (see the module docstring)."""

    CAPACITY = 1 << 16
    MAX_DESC = 256
    # Pinned staging buffers in rotation: a forced flush reopens the pack without waiting on the upload just issued.
    RING = 4
    HEADER_BYTES = 16
    DESC_BYTES = 24
    ALIGN = 16

    def __init__(self, device: torch.device) -> None:
        self.host = [
            torch.empty(self.CAPACITY, dtype=torch.uint8, pin_memory=prefer_pinned())
            for _ in range(self.RING)
        ]
        self.events: list[Optional[torch.cuda.Event]] = [None] * self.RING
        self.dev = torch.empty(self.CAPACITY, dtype=torch.uint8, device=device)
        self.data_start = self.HEADER_BYTES + self.DESC_BYTES * self.MAX_DESC
        self.parity = 0
        self.open = False
        self.h2d: list[tuple[int, int, int]] = []
        self.d2d: list[tuple[int, int, int]] = []
        # Every tensor a staged copy reads or writes, kept alive until the pack is issued.
        self.staged: list[torch.Tensor] = []
        self.used = self.data_start

    def begin(self) -> None:
        if self.open:
            self.flush()
        event = self.events[self.parity]
        if event is not None:
            # The upload that last read this pinned buffer has run.
            event.synchronize()
        self._clear()
        self.open = True

    def _clear(self) -> None:
        self.h2d.clear()
        self.d2d.clear()
        self.staged.clear()
        self.used = self.data_start

    def stage(self, dst: torch.Tensor, src: torch.Tensor) -> bool:
        if not dst.is_cuda or dst.numel() != src.numel():
            return False
        nbytes = dst.numel() * dst.element_size()
        if src.device.type == "cpu":
            # A contiguous destination is one descriptor; a 2-D slice with contiguous rows (a page table's leading
            # rows and columns) is one per row.
            if dst.is_contiguous():
                rows, row_bytes, row_stride = 1, nbytes, 0
            elif dst.dim() == 2 and dst.stride(1) == 1:
                rows = dst.shape[0]
                row_bytes = dst.shape[1] * dst.element_size()
                row_stride = dst.stride(0) * dst.element_size()
            else:
                return False
            offset = (self.used + self.ALIGN - 1) // self.ALIGN * self.ALIGN
            if (
                offset + nbytes > self.CAPACITY
                or len(self.h2d) + len(self.d2d) + rows > self.MAX_DESC
            ):
                return False
            # A dtype change converts on the host, as copy_ would on the device.
            src = src.to(dst.dtype).contiguous()
            self.host[self.parity][offset : offset + nbytes].copy_(src.view(-1).view(torch.uint8))
            base = dst.data_ptr()
            for row in range(rows):
                self.h2d.append((base + row * row_stride, offset + row * row_bytes, row_bytes))
            self.staged.append(dst)
            self.used = offset + nbytes
            return True
        if len(self.h2d) + len(self.d2d) >= self.MAX_DESC:
            return False
        if src.is_cuda and src.dtype == dst.dtype and src.is_contiguous() and dst.is_contiguous():
            self.d2d.append((dst.data_ptr(), src.data_ptr(), nbytes))
            self.staged += (dst, src)
            return True
        return False

    def flush(self) -> None:
        if not self.open:
            return
        self.open = False
        if not self.h2d and not self.d2d:
            return
        host = self.host[self.parity]
        desc = [len(self.h2d), len(self.d2d)] + [
            value for copy in self.h2d + self.d2d for value in copy
        ]
        host[: 8 * len(desc)].view(torch.int64).copy_(torch.tensor(desc, dtype=torch.int64))
        # Descriptors and data in one upload; the gap between them is unused padding.
        self.dev[: self.used].copy_(host[: self.used], non_blocking=True)
        torch.ops.trtllm.step_inputs_scatter(self.dev)
        event = self.events[self.parity]
        if event is None:
            event = self.events[self.parity] = torch.cuda.Event()
        event.record()
        self.parity = (self.parity + 1) % self.RING
        # The scatter is queued on the current stream: the caching allocator keeps the staged tensors' memory for it.
        self._clear()

    def discard(self) -> None:
        """Close the pack without issuing it (its step failed)."""
        self.open = False
        self._clear()


_active: ContextVar[Optional[StepInputPack]] = ContextVar("trtllm_step_input_pack", default=None)


@contextlib.contextmanager
def active(pack: Optional[StepInputPack]) -> Iterator[None]:
    """Make ``pack`` the calling context's pack (None: no packing) for the duration."""
    token = _active.set(pack)
    try:
        yield
    finally:
        _active.reset(token)


def begin() -> None:
    """Open the active pack for the coming copies. Call only when the step will reach ``flush``."""
    pack = _active.get()
    if pack is not None:
        pack.begin()


def is_open() -> bool:
    pack = _active.get()
    return pack is not None and pack.open


def copy_(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """``dst.copy_(src, non_blocking=True)``, deferred into the active open pack when there is one."""
    pack = _active.get()
    if pack is not None and pack.open and not torch.cuda.is_current_stream_capturing():
        if pack.stage(dst, src):
            return dst
        # Keep the order: everything staged so far lands before this copy, and the copies after it pack again.
        pack.flush()
        out = dst.copy_(src, non_blocking=True)
        pack.begin()
        return out
    return dst.copy_(src, non_blocking=True)


def discard() -> None:
    """Close the active pack without issuing it (its step failed)."""
    pack = _active.get()
    if pack is not None:
        pack.discard()


def flush() -> None:
    """Issue the active open pack (one upload and one scatter launch on the current stream) and close it."""
    pack = _active.get()
    if pack is not None:
        pack.flush()
