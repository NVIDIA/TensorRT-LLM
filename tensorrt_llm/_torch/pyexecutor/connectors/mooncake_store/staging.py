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

Registering the KV pools themselves needs the HCA to pin GPU pages, which means
GPUDirect RDMA via `nvidia_peermem` or dma-buf. Where that is unavailable,
`ibv_reg_mr` fails on every pool range and the connector cannot start.

Staging trades a copy for that dependency. Mooncake is given a pinned host
buffer instead of the pools, and each page passes through a slot in it:
gathered from its device regions before a write, scattered back after a read.

A slot holds the page's regions concatenated in region order, which is the same
payload the zero-copy path produces, so a pool written by one path is readable
by the other.

Copies go through `cudaMemcpyAsync` rather than the batched Triton kernel in
`disaggregation/native/bounce/gather_scatter.py`, since one side here is host
memory and the copy engines move that over the host link by DMA.
"""

import os
import threading
from itertools import chain
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart

from tensorrt_llm._utils import CUASSERT
from tensorrt_llm.logger import logger

__all__ = [
    "BATCH_COPY_ENV",
    "MAX_STAGING_BUFFER_BYTES",
    "HostStagingPool",
    "batch_copy_mode",
    "plan_slot_geometry",
    "sync_stream",
]

#: Opt-in batched staging copies (ldfix F1). Unset or "0" keeps the per-page, per-region `cudaMemcpyAsync` loop
#: below, unchanged. Otherwise a whole batch's copies are planned at once (numpy over the flattened region lists)
#: and issued as:
#:   "batch": one `cudaMemcpyBatchAsync` (CUDA 12.8+ runtime and cuda-python; never on the legacy NULL stream);
#:   "2d":    one `cudaMemcpy2DAsync` per run of pages whose region sits at a constant device and host stride (the
#:            common case: consecutive page indices of one region), plain `cudaMemcpyAsync` for the rest;
#:   "1" / "auto": "batch" where available, else "2d" (a failing batch call also falls back to "2d" for good).
#: Same bytes to the same places: a batch whose destinations overlap (where only the loop's order would decide the
#: result) goes through the loop. The point is the GIL: at ~10k Python-level CUDA calls per 1.4 GB load the loader
#: thread starves the executor's forward launch (lanes/mc/STORE_CTX_OVERHEAD.md); cuda-python drops the GIL inside
#: each runtime call, so issuing a handful of calls leaves the executor the interpreter.
BATCH_COPY_ENV = "TRTLLM_MOONCAKE_STORE_BATCH_COPY"


def _parse_batch_copy(value: Optional[str]) -> Optional[str]:
    value = (value or "0").strip().lower()
    if value in ("", "0", "off", "false"):
        return None
    if value in ("1", "on", "true", "auto"):
        return "auto"
    if value in ("batch", "2d"):
        return value
    raise ValueError(f"{BATCH_COPY_ENV} must be 0, 1/auto, batch or 2d; got {value!r}")


#: None (the per-region loop), "auto", "batch" or "2d". Read once at import; tests patch it.
_BATCH_COPY: Optional[str] = _parse_batch_copy(os.environ.get(BATCH_COPY_ENV))
#: Set when a `cudaMemcpyBatchAsync` call failed, so "auto" stops trying it.
_BATCH_API_BROKEN = False
_BATCH_ATTR = None
_MODE_LOGGED = set()
_MODE_LOCK = threading.Lock()


def batch_copy_mode() -> Optional[str]:
    """The configured batched-copy mode (None when the per-region loop is in use)."""
    return _BATCH_COPY

#: Ceiling on the pinned allocation per direction, bounding what a large page
#: times a large batch could demand. The memory is pinned for the process's
#: lifetime and comes out of the DRAM the pool segment is charged against.
MAX_STAGING_BUFFER_BYTES = 1 << 30

#: Stated explicitly rather than inferred from the pointers, which would be
#: wrong for a host pointer outside the unified address space.
_DEVICE_TO_HOST = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
_HOST_TO_DEVICE = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice


def _memcpy_async(dst: int, src: int, size: int, kind, stream: int) -> None:
    """One asynchronous copy between a device page and a host slot."""
    status = cudart.cudaMemcpyAsync(int(dst), int(src), int(size), kind, stream)[0]
    if status == cudart.cudaError_t.cudaSuccess:
        return
    # Raised here rather than through CUASSERT so the operands appear in the
    # message; a bare cudaErrorInvalidValue names no cause.
    device = torch.cuda.current_device() if torch.cuda.is_available() else None
    raise RuntimeError(
        f"cudaMemcpyAsync failed with {status} staging a KV page: "
        f"dst={int(dst):#x} src={int(src):#x} size={size} "
        f"stream={int(stream):#x} current_device={device}. An invalid value here "
        "is usually a stream created on a different device than the pages, which "
        "happens when a thread issues the copy without inheriting the rank's "
        "device, since torch's current device is thread-local."
    )


def _memcpy2d_async(
    dst: int, dpitch: int, src: int, spitch: int, width: int, height: int, kind, stream: int
) -> None:
    """`height` rows of `width` bytes, rows `dpitch` / `spitch` apart, as one asynchronous copy."""
    status = cudart.cudaMemcpy2DAsync(
        int(dst), int(dpitch), int(src), int(spitch), int(width), int(height), kind, stream
    )[0]
    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"cudaMemcpy2DAsync failed with {status} staging KV pages: dst={int(dst):#x} "
            f"dpitch={dpitch} src={int(src):#x} spitch={spitch} width={width} height={height} "
            f"stream={int(stream):#x}"
        )


def _batch_api_available() -> bool:
    return hasattr(cudart, "cudaMemcpyBatchAsync") and hasattr(cudart, "cudaMemcpyAttributes")


def _memcpy_batch_async(dsts: List[int], srcs: List[int], sizes: List[int], stream: int) -> None:
    """All of a batch's copies as one `cudaMemcpyBatchAsync` on `stream` (stream-ordered source access).

    Raises RuntimeError on a non-success status; the caller falls back to the 2D path.
    """
    global _BATCH_ATTR
    if _BATCH_ATTR is None:
        attr = cudart.cudaMemcpyAttributes()
        attr.srcAccessOrder = cudart.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
        try:
            attr.flags = cudart.cudaMemcpyFlags.cudaMemcpyFlagPreferOverlapWithCompute
        except AttributeError:
            pass
        _BATCH_ATTR = attr
    status = cudart.cudaMemcpyBatchAsync(
        dsts, srcs, sizes, len(dsts), [_BATCH_ATTR], [0], 1, stream
    )[0]
    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaMemcpyBatchAsync of {len(dsts)} KV page regions failed with {status}")


def _log_mode_once(mode: str) -> None:
    with _MODE_LOCK:
        if mode in _MODE_LOGGED:
            return
        _MODE_LOGGED.add(mode)
    logger.warning(f"mooncake-store staging: {BATCH_COPY_ENV}={_BATCH_COPY}, issuing copies as '{mode}'")


def _plan_copies(
    slots: Sequence[int], addresses: Sequence[Sequence[int]], sizes: Sequence[Sequence[int]]
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Flatten pages into (device address, host address, size, region position) arrays, page-major.

    Page `i`'s regions land in `slots[i]` back to back, in region order: the same placement the
    per-region loop produces. Returns None for an empty batch.
    """
    if len(addresses) != len(sizes) or any(len(a) != len(s) for a, s in zip(addresses, sizes)):
        raise ValueError("every page needs one size per region address")
    counts = np.fromiter((len(page) for page in sizes), dtype=np.int64, count=len(sizes))
    if counts.size == 0 or int(counts.sum()) == 0:
        return None
    flat_sizes = np.fromiter(chain.from_iterable(sizes), dtype=np.int64, count=int(counts.sum()))
    flat_dev = np.fromiter(chain.from_iterable(addresses), dtype=np.int64, count=int(counts.sum()))
    page_of = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    first = (np.cumsum(counts) - counts)[page_of]  # per region: index of its page's first region
    exclusive = np.cumsum(flat_sizes) - flat_sizes
    offsets = exclusive - exclusive[first]
    host = np.asarray(slots, dtype=np.int64)[page_of] + offsets
    position = np.arange(flat_sizes.size, dtype=np.int64) - first
    keep = flat_sizes > 0  # an empty region moves no bytes
    if not keep.all():
        flat_dev, host, flat_sizes, position = flat_dev[keep], host[keep], flat_sizes[keep], position[keep]
        if flat_sizes.size == 0:
            return None
    return flat_dev, host, flat_sizes, position


def _overlaps(dst: np.ndarray, size: np.ndarray) -> bool:
    """Whether any two destination ranges overlap (then copy order matters, so keep the loop)."""
    if dst.size < 2:
        return False
    order = np.argsort(dst, kind="stable")
    ends = dst[order] + size[order]
    return bool(np.any(ends[:-1] > dst[order][1:]))


#: Conservative ceiling on a 2D copy's pitch (cudaDevAttrMaxPitch is 2^31 - 1 on current GPUs).
_MAX_PITCH = (1 << 31) - 1


def _issue_2d(
    dst: np.ndarray, src: np.ndarray, size: np.ndarray, position: np.ndarray, kind, stream: int
) -> int:
    """Issue the copies as 2D runs: same region position and width, constant dst and src strides.

    Returns the number of runtime calls issued.
    """
    # Region position first, then page order: consecutive page indices of one region line up.
    order = np.lexsort((np.arange(dst.size), size, position))
    d = dst[order].tolist()
    s = src[order].tolist()
    w = size[order].tolist()
    p = position[order].tolist()
    calls = 0
    n = len(d)
    i = 0
    while i < n:
        width = w[i]
        j = i + 1
        dpitch = spitch = 0
        if j < n and p[j] == p[i] and w[j] == width:
            dpitch = d[j] - d[i]
            spitch = s[j] - s[i]
            if width <= dpitch <= _MAX_PITCH and width <= spitch <= _MAX_PITCH:
                j += 1
                while (
                    j < n
                    and p[j] == p[i]
                    and w[j] == width
                    and d[j] - d[j - 1] == dpitch
                    and s[j] - s[j - 1] == spitch
                ):
                    j += 1
            else:
                j = i + 1
        rows = j - i
        if rows == 1:
            _memcpy_async(d[i], s[i], width, kind, stream)
        else:
            _memcpy2d_async(d[i], dpitch, s[i], spitch, width, rows, kind, stream)
        calls += 1
        i = j
    return calls


def _batched_copy(
    slots: Sequence[int],
    addresses: Sequence[Sequence[int]],
    sizes: Sequence[Sequence[int]],
    to_host: bool,
    stream: int,
) -> bool:
    """Copy pages between device regions and host slots as a batch; False when the caller must use the loop.

    `to_host` gathers device regions into the slots (save), otherwise the slots are scattered back (load).
    """
    global _BATCH_API_BROKEN
    plan = _plan_copies(slots, addresses, sizes)
    if plan is None:
        return True
    device, host, size, position = plan
    dst, src = (host, device) if to_host else (device, host)
    if _overlaps(dst, size):
        return False
    kind = _DEVICE_TO_HOST if to_host else _HOST_TO_DEVICE
    mode = _BATCH_COPY
    if mode == "auto":
        mode = "batch" if _batch_api_available() and not _BATCH_API_BROKEN else "2d"
    # cudaMemcpyBatchAsync refuses the legacy NULL stream (synchronous loads may run on it).
    if mode == "batch" and (int(stream) == 0 or not _batch_api_available() or _BATCH_API_BROKEN):
        mode = "2d"
    if mode == "batch":
        try:
            _memcpy_batch_async(dst.tolist(), src.tolist(), size.tolist(), stream)
            _log_mode_once("batch")
            return True
        except RuntimeError as exc:
            # Copies are idempotent (same source bytes), so re-issuing all of them as 2D runs is safe even if
            # the batch call issued some before failing.
            _BATCH_API_BROKEN = True
            logger.warning(f"mooncake-store staging: {exc}; using cudaMemcpy2DAsync runs from now on")
    _issue_2d(dst, src, size, position, kind, stream)
    _log_mode_once("2d")
    return True


def sync_stream(stream: int) -> None:
    """Wait for a stream's copies to finish, given its raw handle."""
    CUASSERT(cudart.cudaStreamSynchronize(stream))


def plan_slot_geometry(
    max_bytes_per_page: int,
    transfer_batch_size: int,
    budget_bytes: int = MAX_STAGING_BUFFER_BYTES,
) -> Tuple[int, int]:
    """Choose how many pages may be staged at once, and how wide a slot is.

    The geometry follows from the layout rather than from configuration: a slot
    has to hold the largest page any layer group produces, and there is no
    point holding more slots than a transfer batch has pages. A budget below
    one page is raised to one rather than refused.

    Args:
        max_bytes_per_page: Largest page payload across layer groups.
        transfer_batch_size: Pages the connector puts in one store call.
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

    One pool serves one direction. Loads run on the executor thread and saves
    on the connector's background thread, so sharing slots would need a lock on
    the transfer path for no benefit.
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
        self._store = store
        self._label = label

        # Page-locking is a correctness requirement here rather than a
        # copy-speed preference: this memory is handed to the store to register.
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
        logger.warning(
            f"mooncake-store {label} staging: {self._num_slots} slots x "
            f"{self._slot_bytes} B = {self._buffer.numel() / 1024**2:.1f} MiB pinned "
            f"(pinned={pin})"
        )

    def close(self) -> None:
        """Hand the registration back before the buffer is freed.

        The store registers an address range rather than the tensor, so a
        buffer freed while it still holds one leaves the fabric able to reach
        memory the allocator has since handed out again. Idempotent, and a
        failed unregistration keeps the buffer.
        """
        if self._buffer is None:
            return
        status = self._store.unregister_buffer(self._base)
        if status != 0:
            raise RuntimeError(
                f"MooncakeDistributedStore.unregister_buffer failed with status "
                f"{status} for the {self._label} host staging buffer at "
                f"{self._base:#x}. The buffer is kept alive rather than freed."
            )
        self._buffer = None

    @property
    def num_slots(self) -> int:
        """Pages this pool can hold at once."""
        return self._num_slots

    @property
    def slot_bytes(self) -> int:
        """Capacity of one slot."""
        return self._slot_bytes

    def slot_address(self, index: int) -> int:
        """Address of slot `index`."""
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
        """Copy one page's device regions into slot `index`, concatenated.

        Args:
            index: Slot to fill.
            addresses: Device addresses of the page's regions, in region order.
            sizes: Byte counts matching `addresses`.
            stream: CUDA stream handle the copies are issued on.

        Returns:
            The slot's address and the total bytes written, ready to hand to
            the store as a single buffer.
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
        """Copy slot `index` back out to one page's device regions.

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
        """Assert a page of `total` bytes is stageable, without copying."""
        self._check_fits(total)

    def gather_many(
        self,
        indices: Sequence[int],
        addresses: Sequence[Sequence[int]],
        sizes: Sequence[Sequence[int]],
        stream: int,
    ) -> List[Tuple[int, int]]:
        """:meth:`gather` for several pages; batched when `BATCH_COPY_ENV` is on.

        Returns (slot address, total bytes) per page, as :meth:`gather` does.
        """
        totals = [sum(page_sizes) for page_sizes in sizes]
        for total in totals:
            self._check_fits(total)
        slots = [self.slot_address(index) for index in indices]
        if len(slots) != len(addresses) or len(slots) != len(sizes):
            raise ValueError("gather_many needs one slot index per page")
        if _BATCH_COPY is None or not _batched_copy(slots, addresses, sizes, True, stream):
            for index, page_addresses, page_sizes in zip(indices, addresses, sizes):
                self.gather(index, page_addresses, page_sizes, stream)
        return list(zip(slots, totals))

    def scatter_many(
        self,
        indices: Sequence[int],
        addresses: Sequence[Sequence[int]],
        sizes: Sequence[Sequence[int]],
        stream: int,
    ) -> None:
        """:meth:`scatter` for several pages (slot `indices[i]` to page i); batched when `BATCH_COPY_ENV` is on."""
        for page_sizes in sizes:
            self._check_fits(sum(page_sizes))
        slots = [self.slot_address(index) for index in indices]
        if len(slots) != len(addresses) or len(slots) != len(sizes):
            raise ValueError("scatter_many needs one slot index per page")
        if _BATCH_COPY is None or not _batched_copy(slots, addresses, sizes, False, stream):
            for index, page_addresses, page_sizes in zip(indices, addresses, sizes):
                self.scatter(index, page_addresses, page_sizes, stream)


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
        raise ValueError(f"batch of {len(addresses)} pages exceeds {pool.num_slots} staging slots")
    staged_addresses: List[List[int]] = []
    staged_sizes: List[List[int]] = []
    if _BATCH_COPY is not None:
        if len(addresses) != len(sizes):
            raise ValueError(f"{len(addresses)} page address lists for {len(sizes)} size lists")
        placed = pool.gather_many(range(len(addresses)), addresses, sizes, stream)
        return [[slot] for slot, _ in placed], [[total] for _, total in placed]
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
    destination, and :func:`unstage_batch_after_get` moves the bytes on once
    the store has filled them.
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
            knows some reads failed passes the rest, so no device page is
            overwritten with whatever a failed slot held.
    """
    indices = range(len(addresses)) if only is None else only
    if _BATCH_COPY is not None:
        indices = list(indices)
        pool.scatter_many(
            indices, [addresses[i] for i in indices], [sizes[i] for i in indices], stream
        )
        return
    for index in indices:
        pool.scatter(index, addresses[index], sizes[index], stream)
