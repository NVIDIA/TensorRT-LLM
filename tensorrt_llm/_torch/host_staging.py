# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded pinned staging for asynchronous host-to-device uploads."""

import bisect
import collections
import itertools
import threading
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

import torch
import triton
import triton.language as tl

from tensorrt_llm._utils import maybe_pin_memory, prefer_pinned

_STAGING_RING = 2
_COPY_BLOCK = 1024
# Pinned staging of each device's `StagedUploader`: at most this many buffers of this size.
_STAGING_BUFFER_BYTES = 64 << 20
_STAGING_BUFFER_COUNT = 4
# Encodes whose batches a device's copy-stream pool holds at most, by default: the one being
# uploaded and the last one, which its encoder may still be reading (`StagedUploader.end_encode`).
MAX_ENCODES_IN_FLIGHT = 2


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
    destination.copy_(view, non_blocking=True)
    if event is None:
        event = ring["events"][index] = torch.cuda.Event()
    event.record(torch.cuda.current_stream(destination.device))


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
    capture, without pinned memory, or for a destination of another dtype,
    size or layout, the values take a plain ``copy_(non_blocking=True)``.
    """

    def __init__(self) -> None:
        self._stream = torch.cuda.Stream() if prefer_pinned() else None
        self._rings: dict[str, tuple[_UploadSlot, _UploadSlot]] = {}
        self._next: dict[str, int] = {}

    def upload(
        self, key: str, value: torch.Tensor, destination: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Upload host ``value``; returns the device tensor holding it (``destination`` if given)."""
        if not self._can_stage(value, destination):
            if destination is None:
                destination = torch.empty(value.shape, dtype=value.dtype, device="cuda")
            destination.copy_(maybe_pin_memory(value), non_blocking=True)
            return destination
        current = torch.cuda.current_stream(self._stream.device)
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
        host = slot.pinned[:count].view(value.shape)
        host.copy_(value)
        staged = slot.staged[:count].view(value.shape)
        self._stream.wait_event(slot.released)
        with torch.cuda.stream(self._stream):
            staged.copy_(host, non_blocking=True)
        slot.uploaded.record(self._stream)
        current.wait_event(slot.uploaded)
        slot.reader = current
        if destination is None:
            return staged
        _copy_kernel[(triton.cdiv(count, _COPY_BLOCK),)](
            staged, destination, count, BLOCK=_COPY_BLOCK
        )
        return destination

    def _can_stage(self, value: torch.Tensor, destination: torch.Tensor | None) -> bool:
        if self._stream is None or value.numel() == 0 or torch.cuda.is_current_stream_capturing():
            return False
        return destination is None or (
            destination.device == self._stream.device
            and destination.dtype == value.dtype
            and destination.numel() == value.numel()
            and destination.is_contiguous()
        )


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
                # A normal tensor: uploads run both inside and outside inference mode.
                with torch.inference_mode(False):
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
    So that the pool holds the batches of at most `max_encodes_in_flight`
    encodes, the host waits for the consumer of older encodes before it makes
    a batch (`end_encode`).
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
        # Consumer streams of the batches made since the last `end_encode`, and for each ended
        # encode the events recorded on its consumer streams once it dropped its batches.
        self._readers: set[torch.cuda.Stream] = set()
        self._ended: collections.deque[list[torch.cuda.Event]] = collections.deque()
        # The KV-cache estimate sets it (1 when each executor iteration waits for its GPU work
        # before the next encode) and reserves that many pools.
        self.max_encodes_in_flight = MAX_ENCODES_IN_FLIGHT

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
        call. The host waits for a consumer stream only when
        `max_encodes_in_flight` ended encodes may still read their batches.
        """
        consumer = torch.cuda.current_stream(self.device)
        self.reclaim()
        with self._lock, torch.cuda.stream(self.copy_stream):
            while len(self._ended) >= self.max_encodes_in_flight:
                for event in self._ended.popleft():
                    event.synchronize()
            self._readers.add(consumer)
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

    def end_encode(self) -> None:
        """End the encode that read the batches made since the last call; it has dropped them.

        Events recorded now on their consumer streams complete once the native
        allocator can reuse their blocks.
        """
        with self._lock:
            if not self._readers:
                return
            events = [torch.cuda.Event() for _ in self._readers]
            for event, stream in zip(events, self._readers):
                event.record(stream)
            self._readers.clear()
            self._ended.append(events)

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


def _clear_error_frames(error: BaseException) -> None:
    """Clear the variables of the frames in the tracebacks of ``error`` and of every error it
    wraps (``__cause__`` and ``__context__``, each error once). The errors themselves stay."""
    seen, pending = set(), [error]
    while pending:
        error = pending.pop()
        if error is not None and id(error) not in seen:
            seen.add(id(error))
            traceback.clear_frames(error.__traceback__)
            pending += (error.__cause__, error.__context__)


@contextmanager
def staged_encode() -> Iterator[None]:
    """One encode of the encoder entry point: `StagedUploader.end_encode` of every uploader when
    the block exits, also after an error, outside stream capture. The encode must have dropped
    its batches by then, so they may live only in the calls made in the block. The tracebacks
    of an error and of the errors it wraps still hold those calls' frames, so their variables
    are cleared first."""
    try:
        yield
    except BaseException as error:
        _clear_error_frames(error)
        raise
    finally:
        uploaders = list(_STAGED_UPLOADERS.values())
        if uploaders and not torch.cuda.is_current_stream_capturing():
            for uploader in uploaders:
                uploader.end_encode()


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
