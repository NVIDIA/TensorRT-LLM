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
"""Fabric-VMM bounce buffers (KVCacheManagerV2 VirtMem): fabric regions let the
NIXL WRITE ride cuda_ipc/MNNVL (a plain cudaMalloc buffer can't), allocated once
at setup and reused."""

import threading
import time
from typing import Dict, Optional, Tuple

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.base.agent import MemoryType, RegMemoryDescs  # noqa: F401
from tensorrt_llm.runtime.kv_cache_manager_v2._cuda_virt_mem import PooledPhysMemAllocator, VirtMem

_MIB = 1024 * 1024


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


class Buffer:
    """One contiguous fabric-VMM region for coalescing KV; phys_chunk_size MUST
    equal the KV pool's chunk size for the C++ VmmDescSplitter uniform-chunk invariant."""

    __slots__ = ("_device_id", "_name", "_size", "_vm")

    def __init__(self, capacity_bytes: int, phys_chunk_size: int, name: str = "kv_bounce"):
        if capacity_bytes <= 0 or phys_chunk_size <= 0:
            raise ValueError(
                f"Buffer: capacity_bytes={capacity_bytes}, "
                f"phys_chunk_size={phys_chunk_size} must both be > 0"
            )
        vm_size = _div_up(capacity_bytes, phys_chunk_size) * phys_chunk_size
        allocator = PooledPhysMemAllocator(phys_chunk_size)
        # Fully back the region up front so the base ptr is writable + stable for life.
        self._vm = VirtMem(vm_size, allocator, init_num_phys_mem=vm_size // phys_chunk_size)
        self._size = vm_size
        self._device_id = allocator.device_id
        self._name = name
        logger.info(
            f"[kv-bounce] allocated fabric bounce buffer '{name}': "
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
        # type is the STRING "VRAM" (the NIXL agent calls .upper() on it), not the enum.
        return RegMemoryDescs("VRAM", [(self.base_ptr, self._size, self._device_id, self._name)])

    def close(self) -> None:
        vm = getattr(self, "_vm", None)
        if vm is not None:
            vm.destroy()
            self._vm = None  # type: ignore[assignment]

    def __del__(self):
        # Broad except is deliberate (a destructor must never raise), but log the teardown
        # failure instead of silently swallowing it so a leaked VMM region is observable.
        try:
            self.close()
        except Exception as e:
            logger.debug(f"[kv-bounce] buffer '{getattr(self, '_name', '?')}' cleanup failed: {e}")


# Round region starts to this for copy/WRITE alignment (negligible waste).
_ALIGN = 512


class SlotAllocator:
    """First-fit allocator over ONE fabric buffer: variable-size contiguous regions placed in the
    lowest free gap large enough. Regions may be released in any order; first-fit scans the live
    regions so a hole freed out of order is reused (a bump cursor would skip it and could wait
    even with space free). One Condition makes reserve thread-safe and blocking. One Buffer =>
    one NIXL registration, so the WRITE can stripe across NIC rails."""

    __slots__ = ("_buf", "_cap", "_cv", "_in_use", "_next_slot_id")

    def __init__(self, capacity_bytes: int, phys_chunk_size: int, name: str = "kv_bounce"):
        if capacity_bytes <= 0:
            raise ValueError(f"SlotAllocator: capacity_bytes={capacity_bytes} must be > 0")
        self._buf = Buffer(capacity_bytes, phys_chunk_size, name=name)
        self._cap = self._buf.size  # rounded up to a chunk multiple by Buffer
        self._in_use: Dict[int, Tuple[int, int]] = {}  # slot_id -> (start, size)
        self._next_slot_id = 0
        self._cv = threading.Condition(threading.Lock())

    @property
    def capacity(self) -> int:
        return self._cap

    def _find_free_start(self, size: int) -> Optional[int]:
        """Lowest start of a free gap >= size across the ring, or None if none currently fits.
        Scans live regions in order so a hole freed out of order is reused, never skipped."""
        cursor = 0
        for s, n in sorted(self._in_use.values()):
            if s - cursor >= size:
                return cursor
            cursor = max(cursor, s + n)
        return cursor if self._cap - cursor >= size else None

    def reserve(self, size: int, timeout: Optional[float] = None) -> Optional[Tuple[int, int]]:
        """Reserve a contiguous size-byte region -> (slot_id, device_addr), or None if it can't ever
        fit or no space frees within timeout."""
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

    def reg_descs(self) -> "RegMemoryDescs":
        # ONE registration for the whole region (the multi-rail key).
        return self._buf.reg_descs()

    def close(self) -> None:
        self._buf.close()
