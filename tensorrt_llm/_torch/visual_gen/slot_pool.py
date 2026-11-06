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

import math
import weakref
from collections import deque
from typing import Deque, Dict, List, Tuple

import torch

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm._torch.visual_gen.pipeline import SessionCapabilities, SessionStreamSpec
from tensorrt_llm._torch.visual_gen.session_protocol import SharedTensorHandle, SlotHandles
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
        """Whether ``tensor`` views this storage: what a pipeline may report as written."""
        return tensor.untyped_storage().data_ptr() == self.storage.untyped_storage().data_ptr()


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
        self._slots: List[DeliverySlot] = [
            {spec.name: _stream_buffer(spec, device, local=local) for spec in capabilities.streams}
            for _ in range(num_slots)
        ]
        # Indices of the slots nobody holds, oldest return first.
        self._free: Deque[int] = deque(range(num_slots))

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
        return slot_id, self._slots[slot_id]

    def release(self, slot_id: int) -> None:
        if not 0 <= slot_id < len(self._slots):
            raise ValueError(f"slot {slot_id} does not exist")
        if slot_id in self._free:
            raise ValueError(f"slot {slot_id} returned twice")
        self._free.append(slot_id)

    def export_handles(self) -> List[SlotHandles]:
        """One handle dict per slot per stream, for ``SessionOpened``. Each handle is
        consumed exactly once by the client's mapping."""
        return [{name: buffer.handle for name, buffer in slot.items()} for slot in self._slots]


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
        return False
    return True


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


def map_slots(handles: List[SlotHandles]) -> List[Dict[str, torch.Tensor]]:
    """The client's mapping of a session's slot pool from the handles in
    ``SessionOpened``: ``mapped[slot_id][stream]`` is a flat tensor over the
    worker's buffer. Each handle maps exactly once; call this once per session."""
    return [
        {
            name: SharedTensorContainer.from_dict(handle).get_local_view()
            for name, handle in slot.items()
        }
        for slot in handles
    ]


__all__ = [
    "DeliverySlot",
    "SlotPool",
    "StreamBuffer",
    "make_view",
    "map_slots",
]
