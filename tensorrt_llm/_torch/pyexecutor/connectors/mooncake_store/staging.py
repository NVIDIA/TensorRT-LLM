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
"""Pinned host slots that stand in for GPU pages when the pool cannot reach them.

The connector's default path registers the KV pools themselves with Mooncake, so
the store reads and writes device memory directly. That needs the HCA to be able
to pin GPU pages -- GPUDirect RDMA, via ``nvidia_peermem`` or dma-buf. Where that
is unavailable, ``ibv_reg_mr`` fails on every pool range and the connector cannot
start at all.

Staging trades a copy for that dependency. Mooncake is given a pinned host buffer
instead of the pools, and each page passes through a slot in it: gathered from its
device regions before a write, scattered back to them after a read. The store then
only ever registers host memory, which needs no GPUDirect.

A slot holds the page's regions concatenated in region order, which is precisely
the payload the zero-copy path would have produced from the same regions. The
stored bytes are therefore identical either way, so a pool written by one path is
readable by the other -- including by another engine sharing the pool.

Copies go through ``cudaMemcpyAsync`` rather than the batched Triton kernel in
``disaggregation/native/bounce/gather_scatter.py``. That kernel is the better tool
for device-to-device gather, but here one side is host memory: the copy engines
move it over the host link by DMA, whereas a kernel would do it with scattered
stores from the SMs.
"""

from typing import List, Optional, Sequence, Tuple

import torch

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

from tensorrt_llm._utils import CUASSERT
from tensorrt_llm.logger import logger

__all__ = ["HostStagingPool", "plan_slot_geometry", "sync_stream"]

#: Stated rather than inferred from the pointers: the direction is known at each
#: call site, and saying so keeps a copy from being misread if a host pointer is
#: ever outside the unified address space.
_DEVICE_TO_HOST = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
_HOST_TO_DEVICE = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice


def _memcpy_async(dst: int, src: int, size: int, kind, stream: int) -> None:
    """One asynchronous copy between a device page and a host slot."""
    status = cudart.cudaMemcpyAsync(int(dst), int(src), int(size), kind, stream)[0]
    if status == cudart.cudaError_t.cudaSuccess:
        return
    # Raised here rather than through CUASSERT so the operands are in the
    # message. A bare cudaErrorInvalidValue from a copy says nothing about which
    # of the three plausible causes it was.
    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    raise RuntimeError(
        f"cudaMemcpyAsync failed with {status} staging a KV page: "
        f"dst={int(dst):#x} src={int(src):#x} size={size} "
        f"stream={int(stream):#x} current_device={device}. An invalid value here "
        "is usually a stream created on a different device than the pages, which "
        "happens when a thread issues the copy without inheriting the rank's "
        "device -- torch's current device is thread-local."
    )


def sync_stream(stream: int) -> None:
    """Wait for a stream's copies to finish, given its raw handle."""
    CUASSERT(cudart.cudaStreamSynchronize(stream))


def plan_slot_geometry(
    max_bytes_per_page: int,
    transfer_batch_size: int,
    budget_bytes: int,
) -> Tuple[int, int]:
    """Choose how many pages may be staged at once, and how wide a slot is.

    A slot has to hold the largest page any layer group produces, so the page size
    is a floor on the allocation: a budget below one page is raised to one rather
    than refused, since the alternative is not starting.

    Args:
        max_bytes_per_page: Largest page payload across layer groups.
        transfer_batch_size: Pages the connector puts in one store call. There is
            no point staging more than that.
        budget_bytes: Ceiling on this pool's pinned allocation.

    Returns:
        Slot width in bytes, and the number of slots.
    """
    if max_bytes_per_page <= 0:
        raise ValueError(f"max_bytes_per_page must be > 0, got {max_bytes_per_page}")
    if transfer_batch_size <= 0:
        raise ValueError(f"transfer_batch_size must be > 0, got {transfer_batch_size}")

    affordable = budget_bytes // max_bytes_per_page
    num_slots = max(1, min(transfer_batch_size, affordable))
    return max_bytes_per_page, num_slots


class HostStagingPool:
    """A registered pinned buffer, sliced into per-page slots.

    One pool serves one direction. Loads run on the executor thread and saves on
    the connector's background thread, so sharing slots between them would need a
    lock on the transfer path for no benefit -- the two pools are independent.
    """

    def __init__(
        self,
        *,
        slot_bytes: int,
        num_slots: int,
        store,
        label: str,
    ):
        self._slot_bytes = int(slot_bytes)
        self._num_slots = int(num_slots)
        self._label = label

        # Pinned unconditionally, unlike the ``prefer_pinned`` heuristic used for
        # transfer buffers elsewhere: this memory is handed to the store to
        # register, so page-locking it is a correctness property of the
        # registration rather than a copy-speed preference.
        pin = torch.cuda.is_available()
        self._buffer = torch.empty(
            self._slot_bytes * self._num_slots, dtype=torch.uint8, pin_memory=pin
        )
        self._base = int(self._buffer.data_ptr())

        status = store.register_buffer(self._base, self._buffer.numel())
        if status != 0:
            raise RuntimeError(
                f"MooncakeDistributedStore.register_buffer failed with status "
                f"{status} for the {label} host staging buffer at "
                f"[{self._base:#x}, {self._base + self._buffer.numel():#x}). Host "
                f"memory registration failing points at the pool or the fabric "
                f"rather than at GPUDirect, which is what staging avoids."
            )
        logger.info(
            f"mooncake-store {label} staging: {self._num_slots} slots x "
            f"{self._slot_bytes} B = {self._buffer.numel() / 1024**2:.1f} MiB pinned "
            f"(pinned={pin})"
        )

    @property
    def num_slots(self) -> int:
        """Pages this pool can hold at once."""
        return self._num_slots

    @property
    def slot_bytes(self) -> int:
        """Capacity of one slot."""
        return self._slot_bytes

    def slot_address(self, index: int) -> int:
        """Address of slot ``index``."""
        if not 0 <= index < self._num_slots:
            raise IndexError(f"slot {index} out of range [0, {self._num_slots})")
        return self._base + index * self._slot_bytes

    def _check_fits(self, total: int) -> None:
        if total > self._slot_bytes:
            raise ValueError(
                f"a {total} B page does not fit the {self._slot_bytes} B "
                f"{self._label} staging slot; the pool was sized from the layout's "
                "largest page, so this means the layout changed after registration"
            )

    def gather(
        self,
        index: int,
        addresses: Sequence[int],
        sizes: Sequence[int],
        stream: int,
    ) -> Tuple[int, int]:
        """Copy one page's device regions into slot ``index``, concatenated.

        Args:
            index: Slot to fill.
            addresses: Device addresses of the page's regions, in region order.
            sizes: Byte counts matching ``addresses``.
            stream: CUDA stream handle the copies are issued on.

        Returns:
            The slot's address and the total bytes written, ready to hand to the
            store as a single buffer.
        """
        total = sum(sizes)
        self._check_fits(total)
        destination = self.slot_address(index)
        offset = 0
        for address, size in zip(addresses, sizes, strict=True):
            _memcpy_async(destination + offset, address, size, _DEVICE_TO_HOST, stream)
            offset += size
        return destination, total

    def scatter(
        self,
        index: int,
        addresses: Sequence[int],
        sizes: Sequence[int],
        stream: int,
    ) -> None:
        """Copy slot ``index`` back out to one page's device regions.

        The inverse of :meth:`gather`, walking the regions in the same order so
        the split matches the concatenation the slot holds.
        """
        self._check_fits(sum(sizes))
        source = self.slot_address(index)
        offset = 0
        for address, size in zip(addresses, sizes, strict=True):
            _memcpy_async(address, source + offset, size, _HOST_TO_DEVICE, stream)
            offset += size

    def reserve(self, total: int) -> None:
        """Assert a page of ``total`` bytes is stageable, without copying."""
        self._check_fits(total)


def stage_batch_for_put(
    pool: HostStagingPool,
    addresses: Sequence[Sequence[int]],
    sizes: Sequence[Sequence[int]],
    stream: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Gather a batch of device pages into slots and describe them for the store.

    Args:
        pool: Slots to stage through. The batch must not exceed its slot count.
        addresses: Per-page device region addresses.
        sizes: Per-page device region sizes.
        stream: Stream the copies are issued on. The caller must synchronize it
            before the store reads the slots.

    Returns:
        Per-page address and size lists, each a single staged buffer.
    """
    if len(addresses) > pool.num_slots:
        raise ValueError(
            f"batch of {len(addresses)} pages exceeds {pool.num_slots} staging slots"
        )
    staged_addresses: List[List[int]] = []
    staged_sizes: List[List[int]] = []
    for index, (page_addresses, page_sizes) in enumerate(zip(addresses, sizes, strict=True)):
        slot, total = pool.gather(index, page_addresses, page_sizes, stream)
        staged_addresses.append([slot])
        staged_sizes.append([total])
    return staged_addresses, staged_sizes


def describe_batch_for_get(
    pool: HostStagingPool,
    sizes: Sequence[Sequence[int]],
) -> Tuple[List[List[int]], List[List[int]]]:
    """Describe slots for the store to read a batch into, before scattering.

    Unlike the put direction there is nothing to copy first: the slots are the
    destination, and :func:`unstage_batch_after_get` moves the bytes on once the
    store has filled them.
    """
    if len(sizes) > pool.num_slots:
        raise ValueError(f"batch of {len(sizes)} pages exceeds {pool.num_slots} staging slots")
    staged_addresses: List[List[int]] = []
    staged_sizes: List[List[int]] = []
    for index, page_sizes in enumerate(sizes):
        total = sum(page_sizes)
        pool.reserve(total)
        staged_addresses.append([pool.slot_address(index)])
        staged_sizes.append([total])
    return staged_addresses, staged_sizes


def unstage_batch_after_get(
    pool: HostStagingPool,
    addresses: Sequence[Sequence[int]],
    sizes: Sequence[Sequence[int]],
    stream: int,
    only: Optional[Sequence[int]] = None,
) -> None:
    """Scatter filled slots back to their device pages.

    Args:
        pool: Slots the store just wrote into.
        addresses: Per-page device region addresses.
        sizes: Per-page device region sizes.
        stream: Stream the copies are issued on. The caller must synchronize it
            before the pages are read.
        only: Slot indices to scatter. Defaults to all of them; a caller that
            knows some reads failed passes the rest so a failed page is not
            written over its device slot with whatever the slot held.
    """
    indices = range(len(addresses)) if only is None else only
    for index in indices:
        pool.scatter(index, addresses[index], sizes[index], stream)
