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
"""Host memory placement for KV cache donations (``remote_donor.py``): fault-in, NUMA binding and topology, and the
chunk placement of a fabric donation over the host's NUMA nodes.

A fabric chunk the driver refuses is retried; with reclaim-and-compact passes on
(``RemoteFabricTransportConfig.compact_host_memory``), the node is then compacted for it under a per-node flock that
donors on one host take in turn. Without them, what fits is published and the job prolog is left to compact.
"""

import contextlib
import ctypes
import errno
import fcntl
import mmap
import os
import time
from collections.abc import Callable, Iterator
from types import ModuleType
from typing import TYPE_CHECKING, Optional

from tensorrt_llm._torch.mmap_utils import madvise_range
from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from cuda.bindings.driver import CUdevice

_GIB = 1 << 30


def _log(message: str) -> None:
    logger.info(f"KVCM2 remote donor: {message}")


# Linux values (asm-generic, so x86_64 and aarch64 alike); Python's mmap module lacks some of them before 3.13.
MADV_HUGEPAGE = 14
_POPULATE_SLICE = 64 << 20


def populate(region: mmap.mmap, address: int, nbytes: int) -> None:
    """Make every page of `region` (at `address`) resident: huge pages where the kernel has them (MADV_HUGEPAGE is set
    first), 4 KiB pages elsewhere. MADV_POPULATE_WRITE does it in the kernel without holding the GIL; a kernel without
    it gets the region zeroed slice by slice. Any other madvise error (ENOMEM, EHWPOISON) raises."""
    try:
        madvise_range(address, nbytes, "populate_write")
        return
    except OSError as error:
        if error.errno not in (errno.EINVAL, errno.ENOSYS):
            raise
    zeros = bytes(_POPULATE_SLICE)
    view = memoryview(region)
    try:
        for offset in range(0, nbytes, _POPULATE_SLICE):
            end = min(offset + _POPULATE_SLICE, nbytes)
            view[offset:end] = zeros[: end - offset]
    finally:
        view.release()


def bind_to_numa_node(address: int, nbytes: int, node: int) -> bool:
    """Bind [address, address + nbytes) to NUMA node `node` with libnuma, before it is touched."""
    try:
        libnuma = ctypes.CDLL("libnuma.so.1")
    except OSError:
        return False
    if libnuma.numa_available() < 0:
        return False
    libnuma.numa_tonode_memory.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    libnuma.numa_tonode_memory.restype = None
    libnuma.numa_tonode_memory(
        ctypes.c_void_p(address), ctypes.c_size_t(nbytes), ctypes.c_int(node)
    )
    return True


# Seconds a refused fabric chunk is retried with backoff before a pass: the driver refuses transiently while reclaiming.
_REFUSAL_RETRY_S = 2.0
# A reclaim-and-compact pass (compact_node) covers the donation still missing, at least this share of the node reserve
# and at most _MAX_PASS_CHUNKS chunks, and leaves this share of the reserve free above the reserve and the chunk.
_PASS_RESERVE_SHARE = 1 / 3
_MAX_PASS_CHUNKS = 64
# Pass bounds: per node, and per refused chunk; past the first bound only while each pass adds a chunk's worth of
# >= 2 MiB blocks.
_MAX_PASSES_PER_NODE = 64
_MAX_PASSES_PER_REFUSAL = 3
_MAX_PASSES_PER_REFUSAL_PROGRESSING = 16


def node_meminfo(node: int) -> dict:
    """GiB per field of /sys/devices/system/node/node<node>/meminfo; empty if unreadable."""
    out = {}
    try:
        with open(f"/sys/devices/system/node/node{node}/meminfo") as f:
            for line in f:
                parts = line.split()  # "Node 0 MemFree:  123 kB"
                if len(parts) >= 4 and parts[3].isdigit():
                    out[parts[2].rstrip(":")] = int(parts[3]) * 1024 / _GIB
    except OSError:
        return {}
    return out


def _numa_nodes() -> list:
    try:
        names = os.listdir("/sys/devices/system/node")
    except OSError:
        return []
    return sorted(int(n[4:]) for n in names if n.startswith("node") and n[4:].isdigit())


def _reclaimable_gib(info: dict) -> float:
    """Clean file pages the kernel can drop, which cuMemCreate's host allocations reclaim (shmem sits on the
    anonymous LRU, outside these counts)."""
    return max(
        0.0,
        info.get("Active(file)", 0.0) + info.get("Inactive(file)", 0.0) - info.get("Dirty", 0.0),
    )


def host_numa_nodes() -> list:
    """The NUMA nodes that have memory (GPU memory partitions show up as nodes with MemTotal 0)."""
    return [n for n in _numa_nodes() if node_meminfo(n).get("MemTotal", 0.0) > 0]


def contiguous_free_gib(
    node: int, min_bytes: int = 2 << 20, text: Optional[str] = None, page: Optional[int] = None
) -> float:
    """GiB of free memory on NUMA node `node` held in blocks of at least `min_bytes` (/proc/buddyinfo: free blocks of
    each order, per zone); -1 when unreadable. A refusal while this is near zero points at fragmentation."""
    try:
        if text is None:
            with open("/proc/buddyinfo") as f:
                text = f.read()
        page = page or os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return -1.0
    total = 0
    for line in text.splitlines():
        parts = line.replace(",", " ").split()  # "Node 0, zone   Normal   n0 n1 ..."
        if len(parts) < 5 or parts[0] != "Node" or not parts[1].isdigit() or int(parts[1]) != node:
            continue
        for order, count in enumerate(parts[4:]):
            if count.isdigit() and (page << order) >= min_bytes:
                total += int(count) * (page << order)
    return total / _GIB


def host_numa_node_of(drv: ModuleType, device: "CUdevice") -> int:
    """The host NUMA node closest to CUDA device `device`; the first node with memory if the driver reports none."""
    err, node = drv.cuDeviceGetAttribute(
        drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device
    )
    if err == drv.CUresult.CUDA_SUCCESS and node >= 0:
        return node
    nodes = host_numa_nodes()
    return nodes[0] if nodes else 0


def compact_node(node: int, gib: float) -> None:
    """Reclaim and compact memory on NUMA node `node`: fault in `gib` GiB of node-bound anonymous memory with huge
    pages (direct reclaim and compaction), then unmap it, leaving about that much free in 2 MiB blocks. Raises when
    libnuma cannot bind the region, since unbound it would land anywhere."""
    nbytes = int(gib * _GIB) // (2 << 20) * (2 << 20)
    region = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    try:
        anchor = ctypes.c_char.from_buffer(region)
        address = ctypes.addressof(anchor)
        del anchor  # the buffer export must end before close()
        if not bind_to_numa_node(address, nbytes, node):
            raise RuntimeError("libnuma is unavailable, so the region cannot be bound to the node")
        try:
            region.madvise(MADV_HUGEPAGE)
        except OSError:
            pass
        populate(region, address, nbytes)
    finally:
        region.close()


@contextlib.contextmanager
def node_compaction_lock(directory: str, host: str, node: int) -> Iterator[bool]:
    """Hold the exclusive flock of NUMA node `node` of `host` in `directory`; yields False, holding nothing, when the
    file system refuses it (no pass may then run on the node)."""
    path = os.path.join(directory, f"compact.{host}.node{node}.lock")
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    except OSError as error:
        _log(f"cannot open {path} ({error}); no reclaim-and-compact pass on node {node}")
        yield False
        return
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as error:
            _log(f"cannot lock {path} ({error}); no reclaim-and-compact pass on node {node}")
            yield False
            return
        yield True
    finally:
        os.close(fd)  # drops the lock


def place_chunks(
    n_chunks: int,
    nodes: list,
    create: Callable[[int], Optional[object]],
    *,
    chunk_gib: float,
    reserve_gib: float,
    node_lock: Callable[[int], contextlib.AbstractContextManager[bool]],
    compact_passes: bool = True,
    meminfo: Optional[Callable[[int], dict]] = None,
    compact: Optional[Callable[[int, float], None]] = None,
    contiguous: Optional[Callable[[int], float]] = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> tuple[list, dict[int, int]]:
    """Place up to `n_chunks` chunks over `nodes`, the preferred node first; `create(node)` returns a handle, or None
    when the driver refuses. A node takes chunks while it keeps `reserve_gib`; a refused chunk is retried, then, with
    `compact_passes`, compacted for under `node_lock(node)` (see the constants), which donors sharing the node take
    in turn; a node that still refuses is left, and swept once more when no node takes the chunk. Returns (handles,
    {node: placed})."""
    meminfo = meminfo or node_meminfo
    compact = compact or compact_node
    contiguous = contiguous or contiguous_free_gib
    handles = []
    placement = {n: 0 for n in nodes}
    passes = {n: 0 for n in nodes}
    pass_failed = set()
    min_pass_gib = margin_gib = reserve_gib * _PASS_RESERVE_SHARE
    max_pass_gib = _MAX_PASS_CHUNKS * chunk_gib

    def room(node: int) -> float:
        info = meminfo(node)
        return info.get("MemFree", 0.0) + _reclaimable_gib(info) if info else float("inf")

    def recover(node: int) -> Optional[object]:
        delay, deadline = 0.05, clock() + _REFUSAL_RETRY_S
        while clock() < deadline:
            sleep(delay)
            delay = min(2 * delay, 0.5)
            handle = create(node)
            if handle is not None:
                return handle
        if not compact_passes:
            return None
        tries = 0
        while (
            tries < _MAX_PASSES_PER_REFUSAL_PROGRESSING
            and passes[node] < _MAX_PASSES_PER_NODE
            and node not in pass_failed
        ):
            tries += 1
            with node_lock(node) as locked:
                if not locked:
                    pass_failed.add(node)
                    return None
                # Sized from the node's memory once the lock is held: another donor may have used it meanwhile.
                missing = (n_chunks - len(handles)) * chunk_gib
                size = min(
                    max(min_pass_gib, missing),
                    max_pass_gib,
                    room(node) - reserve_gib - chunk_gib - margin_gib,
                )
                if size < chunk_gib:
                    return None
                before = contiguous(node)
                try:
                    compact(node, size)
                except Exception as error:  # noqa: BLE001 - no pass is possible here; the chunk moves on
                    _log(f"reclaim-and-compact pass on node {node} failed: {error!r}")
                    pass_failed.add(node)
                    return None
                passes[node] += 1
                gained = contiguous(node) - before
                handle = create(node)
            if handle is not None:
                return handle
            if tries >= _MAX_PASSES_PER_REFUSAL and gained < chunk_gib:
                return None  # the passes stopped bringing contiguous memory
        return None

    exhausted = set()
    swept = False
    while len(handles) < n_chunks:
        handle = None
        for node in nodes:
            if node in exhausted or room(node) - chunk_gib < reserve_gib:
                continue  # refused for good, or the node keeps its reserve for the serving process
            handle = create(node)
            if handle is None:
                handle = recover(node)
                if handle is None:
                    exhausted.add(node)
            if handle is not None:
                placement[node] += 1
                break
        if handle is None:
            if swept:
                break
            # A node given up on can recover after the donor left it (compaction and other donors' frees go on).
            swept = True
            revived = {
                n
                for n in exhausted
                if contiguous(n) >= chunk_gib and room(n) - chunk_gib >= reserve_gib
            }
            if not revived:
                break
            exhausted -= revived
            continue
        handles.append(handle)
    if any(passes.values()):
        _log(f"reclaim-and-compact passes per NUMA node: {passes}")
    return handles, placement
