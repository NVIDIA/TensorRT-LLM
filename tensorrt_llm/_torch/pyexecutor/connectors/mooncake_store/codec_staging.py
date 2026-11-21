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
"""Pinned staging for pages stored in the cache's cold-page encoding.

With a `KvCachePageCodec`, a stored page is not the GPU page's bytes but the
codec's encoding of them, so it cannot be read or written in place. Each page
passes through an arena of registered pinned memory instead: encoded from its
GPU slot into the arena before a put, decoded back out of it after a get.

The codec addresses a batch as ``base + slot * page_bytes(layer group)``, so a
batch is laid out one layer group at a time, each group's pages packed at that
group's encoded size from an aligned base. The store sees one buffer per page.
"""

import ctypes
import mmap
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch

try:
    from cuda.bindings import driver as cuda_driver
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cuda as cuda_driver
    from cuda import cudart

from tensorrt_llm.logger import logger

from ...kv_cache.host_memory import MADV_HUGEPAGE, bind_to_numa_node, populate

__all__ = [
    "ENCODED_GROUP_ALIGNMENT",
    "EncodedBatch",
    "EncodedGroup",
    "EncodedStagingArena",
    "plan_encoded_batches",
]

#: Alignment of each layer group's region in a batch. Codecs write in wide
#: vectors (NVFP4 moves 16 bytes at a time), and 256 keeps every region on a
#: boundary any of them could ask for.
ENCODED_GROUP_ALIGNMENT = 256


def _align(value: int, alignment: int = ENCODED_GROUP_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass
class EncodedGroup:
    """One layer group's pages in a batch: what the codec is called with."""

    layer_group_id: int
    #: Address of the group's first slot in the arena.
    base: int
    #: Arena slot of each page, counted from `base` in units of the encoded size.
    slots: List[int] = field(default_factory=list)
    #: GPU page slot of each page, matching `slots`.
    pages: List[int] = field(default_factory=list)


@dataclass
class EncodedBatch:
    """Pages moved by one store call, in the order the caller listed them."""

    keys: List[str] = field(default_factory=list)
    #: Arena address and encoded size of each page, matching `keys`.
    addresses: List[int] = field(default_factory=list)
    sizes: List[int] = field(default_factory=list)
    groups: List[EncodedGroup] = field(default_factory=list)


def plan_encoded_batches(
    items: Sequence[Tuple[str, int, int]],
    page_bytes: Callable[[int], int],
    arena_base: int,
    arena_bytes: int,
    max_pages: int,
) -> Iterator[EncodedBatch]:
    """Split pages into batches that each fit the arena, and lay each one out.

    Args:
        items: `(key, layer group, GPU page slot)` per page, in transfer order.
        page_bytes: Encoded page size of a layer group.
        arena_base: First byte of the arena.
        arena_bytes: Arena capacity.
        max_pages: Most pages per batch, the store call's batch size.

    Yields:
        Batches whose pages keep their order from `items`, the last one possibly
        short. A page larger than the whole arena raises, since no batch could
        ever hold it.
    """
    if max_pages <= 0:
        raise ValueError(f"max_pages must be > 0, got {max_pages}")
    pending: List[Tuple[str, int, int]] = []
    group_bytes: Dict[int, int] = {}

    def footprint(groups: Dict[int, int]) -> int:
        return sum(_align(size) for size in groups.values())

    for key, layer_group_id, page in items:
        size = page_bytes(layer_group_id)
        if _align(size) > arena_bytes:
            raise ValueError(
                f"an encoded page of layer group {layer_group_id} is {size} B, more "
                f"than the {arena_bytes} B staging arena"
            )
        trial = dict(group_bytes)
        trial[layer_group_id] = trial.get(layer_group_id, 0) + size
        if pending and (len(pending) >= max_pages or footprint(trial) > arena_bytes):
            yield _lay_out(pending, page_bytes, arena_base)
            pending, trial = [], {layer_group_id: size}
        pending.append((key, layer_group_id, page))
        group_bytes = trial
    if pending:
        yield _lay_out(pending, page_bytes, arena_base)


def _lay_out(
    pending: Sequence[Tuple[str, int, int]],
    page_bytes: Callable[[int], int],
    arena_base: int,
) -> EncodedBatch:
    groups: Dict[int, EncodedGroup] = {}
    counts: Dict[int, int] = {}
    for _, layer_group_id, _ in pending:
        counts[layer_group_id] = counts.get(layer_group_id, 0) + 1
    cursor = arena_base
    for layer_group_id, count in counts.items():
        groups[layer_group_id] = EncodedGroup(layer_group_id=layer_group_id, base=cursor)
        cursor += _align(count * page_bytes(layer_group_id))
    batch = EncodedBatch(groups=list(groups.values()))
    for key, layer_group_id, page in pending:
        group = groups[layer_group_id]
        slot = len(group.slots)
        size = page_bytes(layer_group_id)
        group.slots.append(slot)
        group.pages.append(page)
        batch.keys.append(key)
        batch.addresses.append(group.base + slot * size)
        batch.sizes.append(size)
    return batch


#: Mapping granularity of the NUMA-placed buffer, so huge pages can back it.
_HUGE_PAGE_BYTES = 2 << 20


def gpu_host_numa_node(device: int) -> Optional[int]:
    """The host NUMA node the driver reports closest to `device`, or None when it reports none."""
    err, node = cuda_driver.cuDeviceGetAttribute(
        cuda_driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device
    )
    if err != cuda_driver.CUresult.CUDA_SUCCESS or node < 0:
        return None
    return int(node)


def resident_numa_nodes(address: int, nbytes: int, step: int = 1 << 20) -> Optional[set]:
    """The NUMA nodes a sample of a range's pages live on, or None when the kernel will not say.

    Binding a range reports no failure (libnuma prints one), so this is how to
    tell where the pages actually went.
    """
    try:
        libnuma = ctypes.CDLL("libnuma.so.1")
    except OSError:
        return None
    count = max(1, nbytes // step)
    pages = (ctypes.c_void_p * count)(*[address + index * step for index in range(count)])
    status = (ctypes.c_int * count)()
    # With no target nodes, numa_move_pages only reports where each page is.
    if libnuma.numa_move_pages(0, count, pages, None, status, 0) != 0:
        return None
    return set(status)


class PinnedHostBuffer:
    """Page-locked host memory on the NUMA node of the current GPU.

    Torch's pinned allocator places pages wherever the allocating thread runs,
    which on a two-socket Grace node is often the socket away from the GPU, so
    every encode and decode through it crosses the socket link (about 1.6x
    slower on GB300). The buffer is mapped, bound to the GPU's node before it is
    touched, faulted in, then registered with CUDA. Where that cannot be done it
    falls back to torch's pinned memory, and without a GPU to plain memory.
    """

    def __init__(self, nbytes: int):
        if nbytes <= 0:
            raise ValueError(f"nbytes must be > 0, got {nbytes}")
        self.nbytes = int(nbytes)
        #: NUMA node every sampled page of the buffer was found on after binding,
        #: or None when it was not placed there (or where it went is unknown).
        self.numa_node: Optional[int] = None
        self._map: Optional[mmap.mmap] = None
        self._registered = 0
        self._tensor: Optional[torch.Tensor] = None
        if torch.cuda.is_available():
            try:
                self._map_on_gpu_node()
                return
            except (OSError, RuntimeError) as error:
                logger.warning(
                    f"mooncake-store: staging falls back to torch pinned memory, which may sit "
                    f"on the socket away from the GPU: {error}"
                )
                self._release_map()
        self._tensor = torch.empty(
            self.nbytes, dtype=torch.uint8, pin_memory=torch.cuda.is_available()
        )
        self.address = int(self._tensor.data_ptr())

    def _map_on_gpu_node(self) -> None:
        size = -(-self.nbytes // _HUGE_PAGE_BYTES) * _HUGE_PAGE_BYTES
        # Without the driver's answer the pages are left to first touch: a
        # guessed node can be the far socket.
        node = gpu_host_numa_node(torch.cuda.current_device())
        self._map = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        self.address = ctypes.addressof(ctypes.c_char.from_buffer(self._map))
        try:
            # Fewer, larger pages for CUDA and the NIC to register.
            self._map.madvise(MADV_HUGEPAGE)
        except OSError:
            pass
        bound = node is not None and bind_to_numa_node(self.address, size, node)
        populate(self._map, self.address, size)
        status = cudart.cudaHostRegister(
            self.address, size, cudart.cudaHostRegisterPortable | cudart.cudaHostRegisterMapped
        )[0]
        if status != cudart.cudaError_t.cudaSuccess:
            # Cleared so a later, unrelated launch check does not report it.
            cudart.cudaGetLastError()
            raise RuntimeError(f"cudaHostRegister of {size} B failed with {status}")
        self._registered = size
        if bound:
            placed = resident_numa_nodes(self.address, size)
            if placed == {node}:
                self.numa_node = node
            elif placed is not None:
                logger.warning(
                    f"mooncake-store: staging bound to NUMA node {node} landed on nodes "
                    f"{sorted(placed)}; binding memory may not be permitted here"
                )

    def _release_map(self) -> None:
        if self._registered:
            cudart.cudaHostUnregister(self.address)
            self._registered = 0
        if self._map is not None:
            try:
                self._map.close()
            except BufferError:
                # A ctypes view still exports it; the mapping goes with the object.
                pass
            self._map = None

    def close(self) -> None:
        """Unpin and free the memory. Idempotent."""
        self._release_map()
        self._tensor = None


class EncodedStagingArena:
    """A registered pinned buffer encoded pages pass through, one per direction.

    Loads run on the executor thread and saves on the connector's background
    thread, so each direction owns an arena and neither takes a lock.
    """

    def __init__(self, *, capacity_bytes: int, store, label: str):
        if capacity_bytes <= 0:
            raise ValueError(f"capacity_bytes must be > 0, got {capacity_bytes}")
        # Pinned so the store can register it and the codec reach it from the
        # GPU; both read and write it while no Python object refers to it.
        self._buffer: Optional[PinnedHostBuffer] = PinnedHostBuffer(int(capacity_bytes))
        self._capacity = int(capacity_bytes)
        self._base = self._buffer.address
        status = store.register_buffer(self._base, self._capacity)
        if status != 0:
            self._buffer.close()
            self._buffer = None
            raise RuntimeError(
                f"MooncakeDistributedStore.register_buffer failed with status {status} "
                f"for the {label} encoded-page staging arena at "
                f"[{self._base:#x}, {self._base + self._capacity:#x}). Host memory "
                "registration failing points at the pool or the fabric."
            )
        logger.info(
            f"mooncake-store {label} encoded staging: {self._capacity / 1024**2:.1f} MiB "
            f"on NUMA node {self._buffer.numa_node}"
        )

    @property
    def base(self) -> int:
        """Address of the arena's first byte."""
        return self._base

    @property
    def capacity(self) -> int:
        """Arena size in bytes."""
        return 0 if self._buffer is None else self._capacity

    def free_after_store_closed(self) -> None:
        """Free the buffer once the store that registered it is closed. Idempotent.

        Closing the store drops its registrations, so there is nothing to hand
        back; freeing before that would leave the store able to reach memory
        handed out again.
        """
        if self._buffer is not None:
            self._buffer.close()
            self._buffer = None
