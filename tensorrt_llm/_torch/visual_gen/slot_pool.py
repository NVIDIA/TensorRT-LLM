# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Delivery slots: the storage a session result is written into and read from.

The executor acquires one slot before an advance, the pipeline writes its streams
into it, and the slot stays held until the client has moved past the result it
carries. K slots form the pool. Host-placed streams live in shared host memory that
the worker page-locks, so the pipeline's device-to-host copy is asynchronous and the
client maps the same bytes without a copy. Device-placed streams live in device
memory shared through CUDA IPC.
"""

from __future__ import annotations

import ctypes
import itertools
import math
import weakref
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import torch

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm._torch.visual_gen.pipeline import SessionCapabilities, SessionStreamSpec
from tensorrt_llm._torch.visual_gen.session_protocol import (
    ReturnEventHandles,
    SharedTensorHandle,
    SlotHandles,
)
from tensorrt_llm.logger import logger


class StreamBuffer:
    """Fixed storage for one stream in one slot: a flat byte tensor allocated here,
    sized for the stream's largest result, exported once so the client can map it,
    page-locked when it lives on the host. Hands out views of a prefix in the
    shape of one result; a view is the storage, no copy."""

    def __init__(self, nbytes: int, device: torch.device, *, local: bool) -> None:
        self.storage = torch.empty(nbytes, dtype=torch.uint8, device=device)
        # The export comes first: sharing host memory moves it, and the client
        # must map the address that is then page-locked.
        self.handle: SharedTensorHandle = SharedTensorContainer.from_tensor(
            self.storage, local=local
        ).dump_to_dict()
        # Host pages are pinned (cudaHostRegister) so device-to-host copies into
        # them run as asynchronous DMA. The pin is undone when this object dies,
        # before the storage it holds can be freed, and never at interpreter
        # exit, where the process reclaims it anyway. Device memory needs none.
        if device.type != "cuda" and _page_lock(self.storage):
            weakref.finalize(self, _page_unlock, self.storage).atexit = False

    def release_handle(self) -> None:
        """Consume the exported handle when no client will map it."""
        SharedTensorContainer.release_dict(self.handle)

    @property
    def device(self) -> torch.device:
        return self.storage.device

    @property
    def nbytes(self) -> int:
        return self.storage.numel()

    def view(self, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        """A tensor of ``shape`` and ``dtype`` over the start of the storage. Raises
        ``ValueError`` when it does not fit."""
        return make_view(self.storage, shape, dtype)

    def is_view_of(self, tensor: torch.Tensor) -> bool:
        """Whether ``tensor`` is what a pipeline may report as written: a contiguous
        view that starts at the first byte of this storage, which is the only
        layout the client can rebuild from a shape and a dtype."""
        return (
            tensor.untyped_storage().data_ptr() == self.storage.untyped_storage().data_ptr()
            and tensor.data_ptr() == self.storage.data_ptr()
            and tensor.is_contiguous()
        )


def make_view(storage: torch.Tensor, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """A tensor of ``shape`` and ``dtype`` over the first bytes of a flat uint8
    ``storage``; the same bytes, no copy. Raises ``ValueError`` when it does not fit."""
    nbytes = math.prod(int(d) for d in shape) * dtype.itemsize
    if nbytes > storage.numel():
        raise ValueError(f"{nbytes} bytes requested, the buffer holds {storage.numel()}")
    return storage[:nbytes].view(dtype).view(tuple(shape))


# One slot: a StreamBuffer per declared stream, by name. slot["video"] is the
# buffer; the pipeline writes into a view of it and reports that view in its
# outcome. A slot keeps no record of writes.
DeliverySlot = Dict[str, StreamBuffer]


def _stream_buffer(spec: SessionStreamSpec, device: torch.device, *, local: bool) -> StreamBuffer:
    """The buffer one slot holds for one declared stream: the declared byte size,
    on the worker's device or on the host as the stream's placement says."""
    where = device if spec.placement == "device" else torch.device("cpu")
    return StreamBuffer(spec.max_bytes, where, local=local)


class SlotPool:
    """K delivery slots for one session, allocated on the worker.

    ``export_handles()`` hands the client one shared-tensor handle per slot per
    stream; the client maps them once for the session. A slot is handed out by
    ``acquire()`` with its index, the name the client knows it by, after
    ``free_count`` said one is free, and comes back through ``release(index)``
    when the client has moved past its result. The pool is the only owner of
    positions; the storage goes when the pool does.
    """

    def __init__(
        self,
        capabilities: SessionCapabilities,
        num_slots: int,
        device: torch.device,
        *,
        local: bool = False,
    ) -> None:
        if num_slots < 1:
            raise ValueError("a session needs at least one delivery slot")
        self._slots: List[DeliverySlot] = []
        try:
            for _ in range(num_slots):
                slot: DeliverySlot = {}
                self._slots.append(slot)
                for spec in capabilities.streams:
                    slot[spec.name] = _stream_buffer(spec, device, local=local)
        except BaseException:
            self.release_handles()
            raise
        # Indices of the slots nobody holds, oldest return first.
        self._free: Deque[int] = deque(range(num_slots))
        # One event per slot when a stream lives in device memory. The client
        # records a slot's event on its stream as it returns the slot, and the
        # worker's stream waits on that event before the slot is written again:
        # CUDA orders work within one stream of one process only. Host-placed
        # streams need none, the client reads its mapping of them synchronously.
        self._device = device
        self._local = local
        self._return_events: List[torch.cuda.Event] = []
        on_device = any(spec.placement == "device" for spec in capabilities.streams)
        if device.type == "cuda" and on_device:
            with torch.cuda.device(device):
                for _ in range(num_slots):
                    event = torch.cuda.Event(enable_timing=False, interprocess=not local)
                    event.record()  # creates it on this device, already complete
                    self._return_events.append(event)

    def __len__(self) -> int:
        return len(self._slots)

    @property
    def free_count(self) -> int:
        return len(self._free)

    @property
    def nbytes(self) -> int:
        """Bytes held by every buffer of every slot."""
        return sum(buffer.nbytes for slot in self._slots for buffer in slot.values())

    def acquire(self) -> Tuple[int, DeliverySlot]:
        """A free slot with its index. Ask ``free_count`` first: an empty pool is
        a programming error here, not a condition to wait on."""
        if not self._free:
            raise RuntimeError("every delivery slot is held")
        slot_id = self._free.popleft()
        if self._return_events:
            self._return_events[slot_id].wait(torch.cuda.current_stream(self._device))
        return slot_id, self._slots[slot_id]

    def wait_for_readers(self) -> None:
        """Queue, on the current stream, a wait for the reads the client queued
        on every slot it returned. Call before the pool is dropped, so memory
        the pool frees is not written under those reads."""
        for event in self._return_events:
            event.wait(torch.cuda.current_stream(self._device))

    def release(self, slot_id: int) -> None:
        if not 0 <= slot_id < len(self._slots):
            raise ValueError(f"slot {slot_id} does not exist")
        if slot_id in self._free:
            raise ValueError(f"slot {slot_id} returned twice")
        self._free.append(slot_id)

    def release_handles(self) -> None:
        """Consume every exported handle of a pool whose handles will not reach
        a client (the open failed), so the exports stop holding the storage."""
        for slot in self._slots:
            for buffer in slot.values():
                buffer.release_handle()

    def export_handles(self) -> List[SlotHandles]:
        """One handle dict per slot per stream, for ``SessionOpened``. Each handle is
        consumed exactly once by the client's mapping."""
        return [{name: buffer.handle for name, buffer in slot.items()} for slot in self._slots]

    def export_return_events(self) -> Optional[ReturnEventHandles]:
        """What ``SessionOpened`` carries for the client to reach the per-slot
        events: one handle per slot, or a token for a client in this process.
        ``None`` when no stream lives in device memory."""
        if not self._return_events:
            return None
        if self._local:
            token = next(_local_tokens)
            _local_pools[token] = self
            return token
        return [event.ipc_handle() for event in self._return_events]


# Pools whose client lives in this process, by token. The handle of a CUDA event
# cannot be opened by the process that created it, so such a client takes the
# pool's own event objects.
_local_pools: "weakref.WeakValueDictionary[int, SlotPool]" = weakref.WeakValueDictionary()
_local_tokens = itertools.count()


def _page_lock(storage: torch.Tensor) -> bool:
    """Pin an existing host allocation so device-to-host copies into it are
    asynchronous. Shared memory cannot be allocated pinned, so it is registered
    after the fact; a failure leaves the copy synchronous and is logged once.
    Returns whether the storage is now registered."""
    if not torch.cuda.is_available() or storage.numel() == 0:
        return False
    try:
        rc = torch.cuda.cudart().cudaHostRegister(storage.data_ptr(), storage.numel(), 0)
    except Exception as exc:  # the runtime binding may be missing on some builds
        logger.warning(f"slot pool: cudaHostRegister unavailable ({exc}); copies stay synchronous")
        return False
    if int(rc) != 0:
        logger.warning(
            f"slot pool: cudaHostRegister failed with {int(rc)}; copies stay synchronous"
        )
        _clear_cuda_error()
        return False
    return True


def _clear_cuda_error() -> None:
    """Reset the CUDA runtime's last error on this thread. A runtime call that
    fails leaves its error behind, and PyTorch raises it from the next kernel
    launch, whatever that kernel is."""
    try:
        with open("/proc/self/maps") as maps:
            runtimes = {line.split()[-1] for line in maps if "libcudart" in line}
        for path in runtimes:
            ctypes.CDLL(path).cudaGetLastError()  # returns the error and resets it
    except (OSError, AttributeError) as exc:
        logger.warning(f"slot pool: could not reset the CUDA runtime's last error: {exc}")


def _page_unlock(storage: torch.Tensor) -> None:
    """Undo ``_page_lock`` before the storage is freed: a freed range that is
    still registered poisons the next allocation that lands on the same
    addresses (``cudaErrorHostMemoryAlreadyRegistered`` at its registration)."""
    try:
        rc = torch.cuda.cudart().cudaHostUnregister(storage.data_ptr())
    except Exception as exc:
        logger.warning(f"slot pool: cudaHostUnregister failed: {exc}")
        return
    if int(rc) != 0:
        logger.warning(f"slot pool: cudaHostUnregister failed with {int(rc)}")
        _clear_cuda_error()


def map_slots(handles: List[SlotHandles]) -> List[Dict[str, torch.Tensor]]:
    """The client's mapping of a session's slot pool from the handles in
    ``SessionOpened``: ``mapped[slot_id][stream]`` is a flat tensor over the
    worker's buffer. Each handle maps exactly once; call this once per session.
    When a handle fails to map, the ones not yet mapped are consumed too and the
    error is raised."""
    pending = [(i, name, handle) for i, slot in enumerate(handles) for name, handle in slot.items()]
    mapped: List[Dict[str, torch.Tensor]] = [{} for _ in handles]
    try:
        while pending:
            i, name, handle = pending.pop(0)
            mapped[i][name] = SharedTensorContainer.from_dict(handle).get_local_view()
    except BaseException:
        for _, _, handle in pending:
            try:
                SharedTensorContainer.release_dict(handle)
            except Exception as exc:
                logger.warning(f"slot pool: could not release an unmapped handle: {exc}")
        raise
    return mapped


def open_return_events(
    handles: Optional[ReturnEventHandles], slots: List[Dict[str, torch.Tensor]]
) -> Optional[List[torch.cuda.Event]]:
    """The client's side of the pool's per-slot events, from what
    ``SessionOpened`` carries for them, on the device the mapped slots live on.
    ``None`` when the worker sent nothing."""
    if handles is None:
        return None
    if isinstance(handles, int):
        pool = _local_pools.pop(handles, None)
        if pool is None:
            raise RuntimeError("the slot pool of this session is gone")
        return pool._return_events
    device = next(t.device for slot in slots for t in slot.values() if t.is_cuda)
    return [torch.cuda.Event.from_ipc_handle(device, handle) for handle in handles]


__all__ = [
    "DeliverySlot",
    "SlotPool",
    "StreamBuffer",
    "make_view",
    "map_slots",
    "open_return_events",
]
