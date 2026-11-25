# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded pinned staging for asynchronous host-to-device metadata uploads."""

from dataclasses import dataclass, field

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import (
    CUASSERT,
    cudart,
    current_cuda_stream,
    maybe_pin_memory,
    prefer_pinned,
)

_STAGING_RING = 2
_COPY_BLOCK = 1024


def copy_host_to_device(store: dict, key, destination: torch.Tensor, value: torch.Tensor) -> None:
    """Asynchronously copy a host tensor into ``destination`` through pinned staging.

    Pageable or strided sources make ``copy_(non_blocking=True)`` a synchronous
    staged copy inside the driver (a strided source is first materialized as a
    pageable temporary), so the value is first written into a contiguous pinned
    buffer. Each key owns a small ring of buffers with a CUDA event recorded
    after every copy: a buffer is only rewritten once the copy that last read
    it has completed, so the host may run ahead of the device (overlap
    scheduling, chunked prefill) without corrupting a copy still in flight.
    """
    if value.numel() == 0:
        return
    if (
        not destination.is_cuda
        or not prefer_pinned()
        or (value.is_pinned() and value.is_contiguous())
    ):
        # Host destinations (tests) and already pinned sources need no staging.
        destination.copy_(value, non_blocking=destination.is_cuda)
        return
    slot = (key, value.dtype)
    ring = store.get(slot)
    if ring is None:
        ring = store[slot] = {
            "buffers": [None] * _STAGING_RING,
            "events": [None] * _STAGING_RING,
            "next": 0,
        }
    index = ring["next"]
    ring["next"] = (index + 1) % _STAGING_RING
    event = ring["events"][index]
    if event is not None:
        event.synchronize()
    buffer = ring["buffers"][index]
    if buffer is None or buffer.numel() < value.numel():
        # A normal tensor: callers write it both inside and outside inference mode.
        with torch.inference_mode(False):
            buffer = maybe_pin_memory(
                torch.empty(
                    max(value.numel(), 2 * buffer.numel() if buffer is not None else 0),
                    dtype=value.dtype,
                    device="cpu",
                )
            )
        ring["buffers"][index] = buffer
    view = buffer[: value.numel()].view(value.shape)
    view.copy_(value)
    # Tensor.copy_ holds the pinned block until the copy ran, even if the ring is dropped.
    destination.copy_(view, non_blocking=True)
    if event is None:
        event = ring["events"][index] = torch.cuda.Event()
    event.record(current_cuda_stream(destination.device.index))


@triton.jit(do_not_specialize=["count"], do_not_specialize_on_alignment=["count"])
def _copy_kernel(source, destination, count, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < count
    tl.store(destination + offsets, tl.load(source + offsets, mask=mask), mask=mask)


@dataclass
class _UploadSlot:
    pinned: torch.Tensor | None = None
    staged: torch.Tensor | None = None
    # Recorded after the copy that last read ``pinned``.
    uploaded: torch.cuda.Event = field(default_factory=torch.cuda.Event)
    # Recorded on ``reader`` after the work that last read ``staged``.
    released: torch.cuda.Event = field(default_factory=torch.cuda.Event)
    reader: torch.cuda.Stream | None = None


class HostUploader:
    """Host-to-device uploads that keep copy-engine work off the current stream.

    An upload issues its copy on a private stream, where it can run while the
    current stream is still busy (under the previous CUDA graph replay), and
    makes the current stream wait for it. With a destination, one kernel then
    copies the values into it at this point of the current stream, so its
    readers see them exactly where ``destination.copy_(value, non_blocking=True)``
    would have put them. Without one, the device staging buffer is returned; the
    work reading it must be enqueued on the current stream before the next
    upload with that key.

    Each key owns two slots of a pinned and a device buffer: a pinned buffer is
    rewritten only after the copy that read it has completed, a device buffer
    only after its stream has passed the work that read it. During graph
    capture, without pinned memory, off the uploader's device, or for a
    destination of another dtype, size or layout, the values take a plain
    ``copy_(non_blocking=True)``.
    """

    def __init__(self) -> None:
        self._stream = torch.cuda.Stream() if prefer_pinned() else None
        self._rings: dict[str, tuple[_UploadSlot, _UploadSlot]] = {}
        self._next: dict[str, int] = {}

    def synchronize(self) -> None:
        """Finish uploads before the owner releases the pinned staging buffers."""
        for ring in self._rings.values():
            for slot in ring:
                slot.uploaded.synchronize()

    def upload(
        self, key: str, value: torch.Tensor, destination: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Upload host ``value``; returns the device tensor holding it (``destination`` if given)."""
        if not self._can_stage(value, destination):
            if destination is None:
                device = "cuda" if self._stream is None else self._stream.device
                destination = torch.empty(value.shape, dtype=value.dtype, device=device)
            destination.copy_(maybe_pin_memory(value), non_blocking=True)
            return destination
        current = current_cuda_stream(self._stream.device_index)
        ring = self._rings.get(key)
        if ring is None:
            ring = self._rings[key] = (_UploadSlot(), _UploadSlot())
        index = self._next.get(key, 0)
        self._next[key] = 1 - index
        slot, previous = ring[index], ring[1 - index]
        # The readers of the previous upload were enqueued before this call.
        if previous.reader is not None:
            previous.released.record(previous.reader)
        slot.uploaded.synchronize()
        count = value.numel()
        if slot.pinned is None or slot.pinned.dtype != value.dtype or slot.pinned.numel() < count:
            size = max(count, 2 * slot.pinned.numel() if slot.pinned is not None else 0)
            if slot.reader is not None:
                slot.staged.record_stream(slot.reader)
            # Normal tensors: uploads run both inside and outside inference mode.
            with torch.inference_mode(False):
                slot.pinned = torch.empty(size, dtype=value.dtype, pin_memory=prefer_pinned())
                slot.staged = torch.empty(size, dtype=value.dtype, device=current.device)
            # A fresh device allocation is free from this point of the current stream on.
            slot.released.record(current)
        slot.pinned[:count].copy_(value.reshape(count))
        self._stream.wait_event(slot.released)
        # Invisible to the pinned allocator: a slot frees ``pinned`` only after ``uploaded``.
        CUASSERT(
            cudart.cudaMemcpyAsync(
                slot.staged.data_ptr(),
                slot.pinned.data_ptr(),
                count * value.element_size(),
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                self._stream.cuda_stream,
            )
        )
        slot.uploaded.record(self._stream)
        current.wait_event(slot.uploaded)
        slot.reader = current
        if destination is None:
            return slot.staged[:count].view(value.shape)
        _copy_kernel[(triton.cdiv(count, _COPY_BLOCK),)](
            slot.staged, destination, count, BLOCK=_COPY_BLOCK
        )
        return destination

    def _can_stage(self, value: torch.Tensor, destination: torch.Tensor | None) -> bool:
        if (
            self._stream is None
            or value.numel() == 0
            # The raw copy runs in the calling thread's current context.
            or torch._C._cuda_getDevice() != self._stream.device_index
            or torch.cuda.is_current_stream_capturing()
        ):
            return False
        return destination is None or (
            destination.device == self._stream.device
            and destination.dtype == value.dtype
            and destination.numel() == value.numel()
            and destination.is_contiguous()
        )
