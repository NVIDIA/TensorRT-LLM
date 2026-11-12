# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded pinned staging for asynchronous host-to-device uploads."""

import bisect
import collections
import itertools
import threading
from typing import Optional, Sequence

import torch

from tensorrt_llm._utils import maybe_pin_memory, prefer_pinned

_STAGING_RING = 2
# Pinned staging of each device's `StagedUploader`: at most this many buffers of this size.
_STAGING_BUFFER_BYTES = 64 << 20
_STAGING_BUFFER_COUNT = 4


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
    destination.copy_(view, non_blocking=True)
    if event is None:
        event = ring["events"][index] = torch.cuda.Event()
    event.record(torch.cuda.current_stream(destination.device))


class PinnedStagingPool:
    """At most ``buffer_count`` pinned buffers of ``buffer_bytes``, each allocated once and reused.

    A buffer is reused once the event recorded after its copy has completed.
    When every buffer is still in flight, the host waits for the oldest one's
    event; those events are on the uploader's copy stream, which never waits
    for compute, so the wait lasts at most the copies queued ahead of it.
    Nothing else is allocated, so pinned memory never exceeds
    ``buffer_count * buffer_bytes``, and only as many buffers are made as the
    copies have needed at once.
    """

    def __init__(
        self, buffer_bytes: int = _STAGING_BUFFER_BYTES, buffer_count: int = _STAGING_BUFFER_COUNT
    ) -> None:
        self.buffer_bytes = buffer_bytes
        self.buffer_count = buffer_count
        self.buffers: list[torch.Tensor] = []
        self._free: list[torch.Tensor] = []
        self._in_flight: collections.deque[tuple[torch.Tensor, torch.cuda.Event]] = (
            collections.deque()
        )
        self._spare_events: list[torch.cuda.Event] = []
        # Acquisitions that found every buffer in flight and waited for the oldest copy.
        self.waits = 0

    def acquire(self) -> torch.Tensor:
        """Return a staging buffer (uint8, ``buffer_bytes``) that no queued copy still reads."""
        while self._in_flight and self._in_flight[0][1].query():
            self._retire_oldest()
        if not self._free:
            if len(self.buffers) < self.buffer_count:
                buffer = torch.empty(
                    self.buffer_bytes, dtype=torch.uint8, pin_memory=prefer_pinned()
                )
                self.buffers.append(buffer)
                return buffer
            self.waits += 1
            self._in_flight[0][1].synchronize()
            self._retire_oldest()
        return self._free.pop()

    def release(self, buffer: torch.Tensor, stream: torch.cuda.Stream) -> None:
        """Hand ``buffer`` back; it is reused after the work queued on ``stream`` so far."""
        event = self._spare_events.pop() if self._spare_events else torch.cuda.Event()
        event.record(stream)
        self._in_flight.append((buffer, event))

    def _retire_oldest(self) -> None:
        buffer, event = self._in_flight.popleft()
        self._spare_events.append(event)
        self._free.append(buffer)


def _device_index(device: int | str | torch.device) -> int:
    device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"Staged uploads need a CUDA device, got {device}")
    return torch.cuda.current_device() if device.index is None else device.index


class StagedUploader:
    """Host-to-device uploads of one CUDA device, on that device's own copy stream.

    The copy stream never waits for a compute stream. The batch is allocated on
    it, so the allocator gives it a block that no compute work can still be
    using (the native allocator keeps such blocks in the copy stream's own
    pool), and each staging range is copied as soon as the host has laid it
    out, even while the consumer stream is busy.
    The consumer waits for the copies with an event, and the batch is recorded
    on the consumer stream, so the allocator reuses its block only after the
    consumer is done with it.
    """

    def __init__(
        self, device: int | str | torch.device, pool: Optional[PinnedStagingPool] = None
    ) -> None:
        self.device = torch.device("cuda", _device_index(device))
        self.copy_stream = torch.cuda.Stream(self.device)
        self.pool = pool if pool is not None else PinnedStagingPool()
        self._ready: Optional[torch.cuda.Event] = None
        self._lock = threading.Lock()
        # Pinned sources copied without staging, kept until the event after their copies.
        self._held: collections.deque[tuple[torch.cuda.Event, list[torch.Tensor]]] = (
            collections.deque()
        )

    def upload_pieces(
        self,
        shape: Sequence[int],
        dtype: torch.dtype,
        pieces: Sequence[tuple[torch.Tensor, Sequence[int]]],
        fill_value: Optional[int | float] = None,
    ) -> torch.Tensor:
        """Return a new tensor on this device made of host pieces, ready for the current stream.

        Each piece is ``(value, start)``: ``value`` lands at ``start[d]`` of every
        dim ``d``. Elements no piece covers get ``fill_value`` (left as they are
        when it is None, for pieces that cover every element). The tensor is
        written in ranges of at most one staging buffer, each laid out on the
        host, padding included, and copied right away on the copy stream while
        the host lays out the next one. A pinned piece that lands as one run of
        bytes (the row layout) is copied straight from its storage instead, and
        kept alive until that copy is done. Nothing is copied on the device and
        no device temporary is made. Pageable sources are only read during this
        call; the host never waits for the consumer stream.
        """
        consumer = torch.cuda.current_stream(self.device)
        self.reclaim()
        with self._lock, torch.cuda.stream(self.copy_stream):
            destination = torch.empty(tuple(shape), dtype=dtype, device=self.device)
            held: list[torch.Tensor] = []
            if destination.numel():
                values = _rows_in_order(destination, pieces)
                if values is not None:
                    _stage_rows(destination, values, self.pool, self.copy_stream, held)
                else:
                    _stage_boxes(destination, pieces, fill_value, self.pool, self.copy_stream)
            if held:
                # Recorded before the consumer's event, so it is done once the consumer's work is.
                copied = torch.cuda.Event()
                copied.record(self.copy_stream)
                self._held.append((copied, held))
            if self._ready is None:
                self._ready = torch.cuda.Event()
            self._ready.record(self.copy_stream)
            consumer.wait_event(self._ready)
        destination.record_stream(consumer)
        return destination

    def reclaim(self) -> None:
        """Drop the pinned sources whose copies are done."""
        with self._lock:
            while self._held and self._held[0][0].query():
                self._held.popleft()

    def reserved_bytes(self) -> int:
        """GPU memory the native allocator keeps in the copy stream's pool: batches in use or
        cached for reuse, which other streams cannot take. cudaMallocAsync has one pool for all
        streams of a device (and no block snapshot), so it keeps nothing apart: 0."""
        if torch.cuda.get_allocator_backend() != "native":
            return 0
        handle = self.copy_stream.cuda_stream
        return sum(
            segment["total_size"]
            for segment in torch.cuda.memory_snapshot()
            if segment["device"] == self.device.index and segment["stream"] == handle
        )


_STAGED_UPLOADERS: dict[int, StagedUploader] = {}
_STAGED_UPLOADERS_LOCK = threading.Lock()


def staged_uploader(device: int | str | torch.device) -> StagedUploader:
    """The `StagedUploader` of a CUDA device, made on first use."""
    index = _device_index(device)
    uploader = _STAGED_UPLOADERS.get(index)
    if uploader is None:
        with _STAGED_UPLOADERS_LOCK:
            uploader = _STAGED_UPLOADERS.get(index)
            if uploader is None:
                uploader = _STAGED_UPLOADERS[index] = StagedUploader(index)
    return uploader


def reclaim_staged_uploads() -> None:
    """Drop the pinned sources of every uploader whose copies are done, even when no upload
    follows; the executor calls it after each sampler wait."""
    for uploader in list(_STAGED_UPLOADERS.values()):
        uploader.reclaim()


def _rows_in_order(
    destination: torch.Tensor, pieces: Sequence[tuple[torch.Tensor, Sequence[int]]]
) -> Optional[list[torch.Tensor]]:
    """The values in dim-0 order when they are contiguous and fill ``destination`` row after
    row, so the batch is their bytes one after another; otherwise None."""
    ordered = sorted(pieces, key=lambda piece: piece[1][0])
    trailing, row = tuple(destination.shape[1:]), 0
    for value, start in ordered:
        if (
            start[0] != row
            or any(start[1:])
            or tuple(value.shape[1:]) != trailing
            or value.dim() != destination.dim()
            or value.dtype != destination.dtype
            or not value.is_contiguous()
        ):
            return None
        row += value.shape[0]
    return [value for value, _ in ordered] if row == destination.shape[0] else None


def _stage_rows(
    destination: torch.Tensor,
    values: Sequence[torch.Tensor],
    pool: PinnedStagingPool,
    stream: torch.cuda.Stream,
    held: list[torch.Tensor],
) -> None:
    """Copy contiguous dim-0 pieces as one run of bytes, cut into whole staging buffers: small
    pieces share a buffer and large ones span several, with no per-row or per-dim work.
    Pinned pieces are copied straight from their storage and added to ``held``."""
    target = destination.view(-1).view(torch.uint8)
    buffer, start, filled, offset = None, 0, 0, 0
    try:
        for value in values:
            source = value.view(-1).view(torch.uint8)
            if not source.numel():
                continue
            if value.is_pinned():
                if buffer is not None:
                    target[start : start + filled].copy_(buffer[:filled], non_blocking=True)
                    pool.release(buffer, stream)
                    buffer = None
                target[offset : offset + source.numel()].copy_(source, non_blocking=True)
                held.append(value)
                offset += source.numel()
                continue
            position = 0
            while position < source.numel():
                if buffer is None:
                    buffer, start, filled = pool.acquire(), offset, 0
                take = min(pool.buffer_bytes - filled, source.numel() - position)
                buffer[filled : filled + take].copy_(source[position : position + take])
                filled += take
                position += take
                offset += take
                if filled == pool.buffer_bytes:
                    target[start : start + filled].copy_(buffer, non_blocking=True)
                    pool.release(buffer, stream)
                    buffer = None
        if buffer is not None:
            target[start : start + filled].copy_(buffer[:filled], non_blocking=True)
    finally:
        if buffer is not None:
            pool.release(buffer, stream)


def _stage_boxes(
    destination: torch.Tensor,
    pieces: Sequence[tuple[torch.Tensor, Sequence[int]]],
    fill_value: Optional[int | float],
    pool: PinnedStagingPool,
    stream: torch.cuda.Stream,
) -> None:
    """Copy pieces placed anywhere (another item axis, padding, strided views) range by range:
    one index on the dims before a level dim, rows of that dim, the trailing dims whole."""
    shape, ndim = tuple(destination.shape), destination.dim()
    itemsize = destination.element_size()
    buffer_numel = pool.buffer_bytes // itemsize
    suffix = [1] * (ndim + 1)  # suffix[d]: elements per index of dim d - 1
    for dim in range(ndim - 1, -1, -1):
        suffix[dim] = suffix[dim + 1] * shape[dim]
    level = next(dim for dim in range(ndim) if suffix[dim + 1] <= buffer_numel)
    rows = buffer_numel // suffix[level + 1]
    boxes = sorted(
        (
            (tuple(start), tuple(s + n for s, n in zip(start, value.shape)), value)
            for value, start in pieces
        ),
        key=lambda box: box[0][0],
    )
    first_rows = [box[0][0] for box in boxes]
    # Pieces that do not share a row of dim 0 (the usual item axis) are found by bisection.
    rows_apart = all(a[1][0] <= b[0][0] for a, b in zip(boxes, boxes[1:]))
    last_rows = [box[1][0] for box in boxes]
    for prefix in itertools.product(*(range(size) for size in shape[:level])):
        for begin in range(0, shape[level], rows):
            end = min(begin + rows, shape[level])
            low = prefix + (begin,) + (0,) * (ndim - level - 1)
            high = tuple(index + 1 for index in prefix) + (end,) + shape[level + 1 :]
            target = destination[prefix + (slice(begin, end),)]
            buffer = pool.acquire()
            try:
                staged = buffer[: target.numel() * itemsize].view(destination.dtype)
                box = staged.view(tuple(hi - lo for lo, hi in zip(low, high)))
                if fill_value is not None:
                    box.fill_(fill_value)
                stop = bisect.bisect_left(first_rows, high[0])
                start = bisect.bisect_right(last_rows, low[0]) if rows_apart else 0
                for piece_low, piece_high, value in boxes[start:stop]:
                    source, region = value, box
                    for dim in range(ndim):
                        lo, hi = max(piece_low[dim], low[dim]), min(piece_high[dim], high[dim])
                        if lo >= hi:
                            break
                        if hi - lo != value.shape[dim]:
                            source = source.narrow(dim, lo - piece_low[dim], hi - lo)
                        if hi - lo != high[dim] - low[dim]:
                            region = region.narrow(dim, lo - low[dim], hi - lo)
                    else:
                        region.copy_(source)
                target.copy_(staged.view(target.shape), non_blocking=True)
            finally:
                pool.release(buffer, stream)
