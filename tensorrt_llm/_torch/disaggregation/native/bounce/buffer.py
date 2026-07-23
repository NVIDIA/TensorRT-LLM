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
"""Fabric-VMM bounce buffers. A fabric region lets the write ride the fast intra-node fabric, which a
plain device allocation cannot; it is allocated once at setup and reused."""

import threading
import time
from typing import Dict, Optional, Tuple

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.base.agent import RegMemoryDescs
from tensorrt_llm.runtime.kv_cache_manager_v2._cuda_virt_mem import PooledPhysMemAllocator, VirtMem

_MIB = 1024 * 1024

# A Buffer that was not explicitly closed has no proof that transport users
# are quiescent. Retain its VirtMem object until process exit instead of
# allowing destructor order to unmap a potentially live one-sided target.
_UNSAFE_VMM_RETENTION: list[VirtMem] = []


def _div_up(a: int, b: int) -> int:
    return (a + b - 1) // b


class Buffer:
    """One contiguous fabric region for coalescing cache data. The physical chunk size must match the
    cache pool's chunk size, which the C++ splitter relies on."""

    __slots__ = ("_device_id", "_name", "_size", "_vm")

    def __init__(self, capacity_bytes: int, phys_chunk_size: int, name: str = "kv_bounce"):
        if capacity_bytes <= 0 or phys_chunk_size <= 0:
            raise ValueError(
                f"Buffer: capacity_bytes={capacity_bytes}, "
                f"phys_chunk_size={phys_chunk_size} must both be > 0"
            )
        vm_size = _div_up(capacity_bytes, phys_chunk_size) * phys_chunk_size
        allocator = PooledPhysMemAllocator(phys_chunk_size)
        # back the whole region up front so its address is writable and stable for life
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
        # the type is the string "VRAM", not the enum, because the agent upper-cases it
        return RegMemoryDescs("VRAM", [(self.base_ptr, self._size, self._device_id, self._name)])

    def close(self) -> None:
        vm = getattr(self, "_vm", None)
        if vm is not None:
            vm.destroy()
            self._vm = None  # type: ignore[assignment]

    def __del__(self):
        # Explicit owners call close() only after lifecycle drain. A destructor
        # has no such evidence, so it must fail safe by intentionally leaking
        # the mapping until process teardown.
        try:
            vm = getattr(self, "_vm", None)
            if vm is None:
                return
            retained = globals().get("_UNSAFE_VMM_RETENTION")
            if retained is not None:
                retained.append(vm)
                self._vm = None  # type: ignore[assignment]
            logger.warning(
                f"[kv-bounce] retaining unclosed buffer '{getattr(self, '_name', '?')}' "
                "until process exit"
            )
        except Exception as e:
            logger.debug(f"[kv-bounce] buffer retention failed: {e}")


# region starts are rounded to this for copy alignment (negligible waste)
_ALIGN = 512


class SlotAllocator:
    """First-fit allocator over one fabric buffer. Regions may be freed in any order, and first-fit
    reuses a hole freed out of order rather than skipping it. Reserve is thread-safe and blocking.
    The whole buffer is one registration, so a write can stripe across the network links."""

    __slots__ = (
        "_buf",
        "_cap",
        "_closed",
        "_cv",
        "_in_use",
        "_quarantine",
        "_next_slot_id",
    )

    def __init__(self, capacity_bytes: int, phys_chunk_size: int, name: str = "kv_bounce"):
        if capacity_bytes <= 0:
            raise ValueError(f"SlotAllocator: capacity_bytes={capacity_bytes} must be > 0")
        self._buf = Buffer(capacity_bytes, phys_chunk_size, name=name)
        self._cap = self._buf.size  # rounded up to a chunk multiple
        self._in_use: Dict[int, Tuple[int, int]] = {}  # each live slot maps to its start and size
        # Quarantined slots are never made reusable by elapsed time. A caller must provide explicit
        # quiescence evidence through release_quarantined(), or close must refuse the arena.
        self._quarantine: Dict[int, Tuple[int, int]] = {}
        self._next_slot_id = 0
        self._closed = False
        self._cv = threading.Condition(threading.Lock())

    @property
    def capacity(self) -> int:
        return self._cap

    def _occupied(self):
        """Ranges that must not be handed out: live and quarantined, treated the same."""
        for s, n in self._in_use.values():
            yield s, n
        for s, n in self._quarantine.values():
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
                if self._closed:
                    return None
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

    def quarantine(self, slot_id: int) -> None:
        """Hold a slot indefinitely because an in-doubt remote writer may still access it."""
        with self._cv:
            entry = self._in_use.pop(slot_id, None)
            if entry is not None:
                self._quarantine[slot_id] = entry
            self._cv.notify_all()

    def release_quarantined(self, slot_id: int) -> bool:
        """Release one quarantined slot after the caller proves remote access is quiescent."""
        with self._cv:
            released = self._quarantine.pop(slot_id, None) is not None
            if released:
                self._cv.notify_all()
            return released

    @property
    def quarantined_bytes(self) -> int:
        """Bytes currently held in quarantine, for observability."""
        with self._cv:
            return sum(n for _s, n in self._quarantine.values())

    @property
    def has_outstanding(self) -> bool:
        """Whether a live or quarantined slot still owns any part of the arena."""
        with self._cv:
            return bool(self._in_use or self._quarantine)

    def reg_descs(self) -> "RegMemoryDescs":
        return self._buf.reg_descs()

    def close(self) -> None:
        with self._cv:
            if self._in_use or self._quarantine:
                raise RuntimeError(
                    "[kv-bounce] cannot close an arena with live or quarantined slots"
                )
            # Close admission while holding the allocator gate. A blocked or
            # racing reserve must observe this state before the VMM is unmapped.
            self._closed = True
            self._cv.notify_all()
        self._buf.close()
