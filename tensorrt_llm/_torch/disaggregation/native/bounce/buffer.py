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
"""Bounce buffers for coalescing cache data. On NVLink-fabric platforms the region is fabric-VMM so
the write can ride the fast intra-node fabric. Elsewhere FABRIC handles do not exist, and a VMM
region would carry no shareable handle at all: peer processes could not map it, and the staging hop
itself would fall back to slow non-IPC transports. The region is therefore a legacy cuMemAlloc
allocation there, which stays shareable through classic CUDA IPC. Either way it is allocated once at
setup and reused."""

import threading
import time
from typing import Dict, Optional, Tuple

import cuda.bindings.driver as drv

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.base.agent import RegMemoryDescs
from tensorrt_llm.runtime.kv_cache_manager_v2._cuda_virt_mem import (
    PooledPhysMemAllocator,
    VirtMem,
    is_fabric_handle_supported,
)
from tensorrt_llm.runtime.kv_cache_manager_v2._utils import _unwrap

_MIB = 1024 * 1024


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


class _LegacyDeviceMem:
    """Minimal VirtMem stand-in backed by one legacy cuMemAlloc allocation. Legacy allocations are
    shareable via cuIpcGetMemHandle on every platform, which VMM allocations are not once the FABRIC
    handle type is unavailable."""

    __slots__ = ("address",)

    address: Optional[drv.CUdeviceptr]

    def __init__(self, size: int) -> None:
        self.address = _unwrap(drv.cuMemAlloc(size))

    def destroy(self) -> None:
        if self.address is not None:
            _unwrap(drv.cuMemFree(self.address))
            self.address = None


class Buffer:
    """One contiguous region for coalescing cache data. The physical chunk size must match the
    cache pool's chunk size, which the C++ splitter relies on."""

    __slots__ = ("_device_id", "_name", "_size", "_vm")

    def __init__(self, capacity_bytes: int, phys_chunk_size: int, name: str = "kv_bounce"):
        if capacity_bytes <= 0 or phys_chunk_size <= 0:
            raise ValueError(
                f"Buffer: capacity_bytes={capacity_bytes}, "
                f"phys_chunk_size={phys_chunk_size} must both be > 0"
            )
        vm_size = _div_up(capacity_bytes, phys_chunk_size) * phys_chunk_size
        if is_fabric_handle_supported():
            allocator = PooledPhysMemAllocator(phys_chunk_size)
            # back the whole region up front so its address is writable and stable for life
            self._vm = VirtMem(vm_size, allocator, init_num_phys_mem=vm_size // phys_chunk_size)
            self._device_id = allocator.device_id
            kind = "fabric"
        else:
            self._vm = _LegacyDeviceMem(vm_size)
            self._device_id = int(_unwrap(drv.cuCtxGetDevice()))
            kind = "legacy"
        self._size = vm_size
        self._name = name
        logger.info(
            f"[kv-bounce] allocated {kind} bounce buffer '{name}': "
            f"{vm_size / _MIB:.1f} MiB @ 0x{int(self._vm.address):x} "
            f"(chunk={phys_chunk_size // _MIB}MiB, dev={self._device_id})"
        )

    @property
    def base_ptr(self) -> int:
        return int(self._vm.address)

    @property
    def size(self) -> int:
        return self._size

    @property
    def device_id(self) -> int:
        return self._device_id

    def reg_descs(self) -> "RegMemoryDescs":
        # the type is the string "VRAM", not the enum, because the agent upper-cases it
        return RegMemoryDescs("VRAM", [(self.base_ptr, self._size, self._device_id, self._name)])

    def close(self) -> None:
        vm = getattr(self, "_vm", None)
        if vm is not None:
            vm.destroy()
            self._vm = None  # type: ignore[assignment]

    def __del__(self):
        # A destructor must never raise, but a leaked region should be visible, so log the failure.
        try:
            self.close()
        except Exception as e:
            logger.debug(f"[kv-bounce] buffer '{getattr(self, '_name', '?')}' cleanup failed: {e}")


# region starts are rounded to this for copy alignment (negligible waste)
_ALIGN = 512


class SlotAllocator:
    """First-fit allocator over one bounce buffer. Regions may be freed in any order, and first-fit
    reuses a hole freed out of order rather than skipping it. Reserve is thread-safe and blocking.
    The whole buffer is one registration, so a write can stripe across the network links."""

    __slots__ = ("_buf", "_cap", "_cv", "_in_use", "_quarantine", "_next_slot_id")

    def __init__(self, capacity_bytes: int, phys_chunk_size: int, name: str = "kv_bounce"):
        if capacity_bytes <= 0:
            raise ValueError(f"SlotAllocator: capacity_bytes={capacity_bytes} must be > 0")
        self._buf = Buffer(capacity_bytes, phys_chunk_size, name=name)
        self._cap = self._buf.size  # rounded up to a chunk multiple
        self._in_use: Dict[int, Tuple[int, int]] = {}  # each live slot maps to its start and size
        # Quarantined slots not yet reusable: an orphaned writer's write may still be landing
        # and cannot be aborted, so each is held out of the pool until its deadline passes.
        self._quarantine: Dict[int, Tuple[int, int, float]] = {}
        self._next_slot_id = 0
        self._cv = threading.Condition(threading.Lock())

    @property
    def capacity(self) -> int:
        return self._cap

    def _occupied(self):
        """Ranges that must not be handed out: live and quarantined, treated the same."""
        for s, n in self._in_use.values():
            yield s, n
        for s, n, _dl in self._quarantine.values():
            yield s, n

    def _find_free_start(self, size: int) -> Optional[int]:
        """Lowest free gap large enough, or None if none fits. Live and quarantined regions both
        block reuse, so an out-of-order-freed hole is reused but a quarantined one is not."""
        cursor = 0
        for s, n in sorted(self._occupied()):
            if s - cursor >= size:
                return cursor
            cursor = max(cursor, s + n)
        return cursor if self._cap - cursor >= size else None

    def reserve(self, size: int, timeout: Optional[float] = None) -> Optional[Tuple[int, int]]:
        """Reserve a contiguous region, or None if it can never fit or nothing frees within the
        timeout."""
        size = _div_up(size, _ALIGN) * _ALIGN
        if size <= 0 or size > self._cap:
            return None
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while True:
                start = self._find_free_start(size)
                if start is not None:
                    slot_id = self._next_slot_id
                    self._next_slot_id += 1
                    self._in_use[slot_id] = (start, size)
                    return slot_id, self._buf.base_ptr + start
                if deadline is None:
                    self._cv.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not self._cv.wait(timeout=remaining):
                        return None

    def release(self, slot_id: int) -> None:
        with self._cv:
            self._in_use.pop(slot_id, None)
            self._cv.notify_all()

    def quarantine(self, slot_id: int, grace_s: float) -> None:
        """Hold a slot out of the free pool for the grace period instead of releasing it, because its
        region may still be under an in-doubt write. An infinite grace holds it until close."""
        with self._cv:
            entry = self._in_use.pop(slot_id, None)
            if entry is not None:
                start, size = entry
                # a finite time plus infinity is infinity, so an infinite grace never expires
                deadline = time.monotonic() + grace_s
                self._quarantine[slot_id] = (start, size, deadline)
            self._cv.notify_all()

    def reclaim_expired(self) -> int:
        """Return quarantined slots past their deadline to the free pool and report how many. Runs
        off a timer, not tied to reserve, so it makes progress even when the arena is full."""
        now = time.monotonic()
        with self._cv:
            expired = [sid for sid, (_s, _n, dl) in self._quarantine.items() if dl <= now]
            for sid in expired:
                del self._quarantine[sid]
            if expired:
                self._cv.notify_all()
        return len(expired)

    @property
    def quarantined_bytes(self) -> int:
        """Bytes currently held in quarantine, for observability."""
        return sum(n for _s, n, _dl in self._quarantine.values())

    def reg_descs(self) -> "RegMemoryDescs":
        return self._buf.reg_descs()

    def close(self) -> None:
        self._buf.close()
