# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Page-locked host memory for Engram tables."""

import contextlib
import ctypes
import fcntl
import mmap
import os
import signal
import subprocess  # nosec B404 - runs this package's eviction helper with sys.executable and fixed arguments
import sys
import time
import weakref
from contextlib import ExitStack

import pynvml
import torch
from cuda.bindings import driver as cuda

from tensorrt_llm.logger import logger

_POSIX_FD = cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
# "0" turns off page-cache eviction, and the lock around it, for a node that is out of memory.
_EVICT_ENV = "TRTLLM_ENGRAM_EVICT_PAGE_CACHE"
# Eviction faults memory in 4 GiB steps until about 16 GiB of clean cache is left on the node.
_EVICT_CHUNK = 4 << 30  # 4 GiB
_EVICT_MARGIN = 16 << 30  # 16 GiB
# One eviction at a time for every process that sees this sysfs, so only one NUMA node is ever drained.
_EVICT_LOCK = "/sys/devices/system/node/online"
_EVICT_LOCK_WAIT_S = 600  # 10 min
# Caps how long a stuck child keeps every other rank's allocation waiting on the lock.
_EVICT_TIMEOUT_S = 120  # 2 min


class HostNumaOutOfMemoryError(MemoryError):
    """CUDA could not allocate physical host memory on the requested NUMA node."""


def round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _check(result: tuple, call: str) -> object:
    if result[0] != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{call} failed: {result[0]}")
    return result[1] if len(result) > 1 else None


def host_numa_node(device: int) -> int | None:
    """Return the host NUMA node of ``device`` if VMM can allocate there, else None."""
    attribute = cuda.CUdevice_attribute
    supported = _check(
        cuda.cuDeviceGetAttribute(
            attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device
        ),
        "cuDeviceGetAttribute",
    )
    if not supported:
        return None
    node = _check(
        cuda.cuDeviceGetAttribute(attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device),
        "cuDeviceGetAttribute",
    )
    # -1: the device has no host NUMA affinity.
    return node if node >= 0 else None


def node_gpu_numa_nodes() -> list[int]:
    """Closest host NUMA node of every GPU of this node, hidden ones included (-1 if NVML gives none);
    empty if NVML cannot tell."""
    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError:
        return []
    try:
        nodes = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            mask = pynvml.nvmlDeviceGetMemoryAffinity(handle, 1, pynvml.NVML_AFFINITY_SCOPE_NODE)[0]
            nodes.append((mask & -mask).bit_length() - 1)
        return nodes
    except pynvml.NVMLError:
        return []
    finally:
        with contextlib.suppress(pynvml.NVMLError):
            pynvml.nvmlShutdown()


def _set_access(address: int, size: int, device: int, flags) -> tuple:
    """Give ``device`` and the host ``flags`` access to a mapped range; returns the CUDA result."""
    access = []
    for location_type, location_id in (
        (cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE, device),
        (cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST, 0),
    ):
        descriptor = cuda.CUmemAccessDesc()
        descriptor.location.type = location_type
        descriptor.location.id = location_id
        descriptor.flags = flags
        access.append(descriptor)
    return cuda.cuMemSetAccess(address, size, access, len(access))


def _release(device: int, address: int, size: int, handle) -> None:
    torch.cuda.synchronize(device)
    _check(cuda.cuMemUnmap(address, size), "cuMemUnmap")
    _check(cuda.cuMemRelease(handle), "cuMemRelease")
    _check(cuda.cuMemAddressFree(address, size), "cuMemAddressFree")


class HostNumaMem:
    """A CUDA VMM mapping of page-locked host memory on ``numa_node``, shared by ``device``
    and the host at one address; other processes can map ``shareable`` memory via ``export_fd``."""

    def __init__(self, size: int, device: int, numa_node: int, shareable: bool = False) -> None:
        self.numa_node = numa_node
        properties = cuda.CUmemAllocationProp()
        properties.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        properties.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
        properties.location.id = numa_node
        if shareable:
            properties.requestedHandleTypes = _POSIX_FD
        self.granularity = _check(
            cuda.cuMemGetAllocationGranularity(
                properties,
                cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            ),
            "cuMemGetAllocationGranularity",
        )
        self.size = round_up(size, self.granularity)
        with ExitStack() as cleanup:
            self.address = int(
                _check(
                    cuda.cuMemAddressReserve(self.size, self.granularity, 0, 0),
                    "cuMemAddressReserve",
                )
            )
            cleanup.callback(
                lambda: _check(cuda.cuMemAddressFree(self.address, self.size), "cuMemAddressFree")
            )
            result = cuda.cuMemCreate(self.size, properties, 0)
            if result[0] == cuda.CUresult.CUDA_ERROR_OUT_OF_MEMORY:
                raise HostNumaOutOfMemoryError(
                    f"cuMemCreate failed: {result[0]}; device={device}, numa_node={numa_node}, "
                    f"requested_bytes={size}, aligned_bytes={self.size}"
                )
            handle = _check(result, "cuMemCreate")
            cleanup.callback(lambda: _check(cuda.cuMemRelease(handle), "cuMemRelease"))
            _check(cuda.cuMemMap(self.address, self.size, 0, handle, 0), "cuMemMap")
            cleanup.callback(lambda: _check(cuda.cuMemUnmap(self.address, self.size), "cuMemUnmap"))
            _check(
                _set_access(
                    self.address,
                    self.size,
                    device,
                    cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
                ),
                "cuMemSetAccess",
            )
            self._handle = handle
            finalizer = weakref.finalize(self, _release, device, self.address, self.size, handle)
            # Process exit reclaims the mapping; releasing it then would race driver teardown.
            finalizer.atexit = False
            cleanup.pop_all()

    def export_fd(self) -> int:
        """A new POSIX descriptor of this memory; the caller closes it."""
        return int(
            _check(
                cuda.cuMemExportToShareableHandle(self._handle, _POSIX_FD, 0),
                "cuMemExportToShareableHandle",
            )
        )


def evict_numa_node_page_cache(node: int, lock_fd: int) -> bool:
    """Evict host NUMA ``node``'s clean page cache from a child process; False if the child could not run."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_evict_page_cache.py")
    command = [sys.executable, script, str(node), str(_EVICT_CHUNK), str(_EVICT_MARGIN)]
    try:
        # The child inherits the lock, so it stays held even if this process dies first.
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            pass_fds=(lock_fd,),
            timeout=_EVICT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        logger.warning(f"Engram: page-cache eviction on NUMA node {node} failed: {error}")
        return False
    return result.returncode in (0, -signal.SIGKILL)


@contextlib.contextmanager
def _eviction_lock(mode: int):
    """Hold the eviction lock in ``mode``; yields ``(fd, waited)``, or None if it cannot be taken in time."""
    try:
        lock = open(_EVICT_LOCK)
    except OSError as error:
        logger.debug(f"Engram: no page-cache eviction lock: {error}")
        yield None
        return
    with lock:
        deadline, waited = time.monotonic() + _EVICT_LOCK_WAIT_S, False
        while True:
            try:
                fcntl.flock(lock, mode | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    logger.warning("Engram: timed out waiting for the page-cache eviction lock")
                    yield None
                    return
                waited = True
                time.sleep(0.5)
            except OSError as error:
                logger.debug(f"Engram: no page-cache eviction lock: {error}")
                yield None
                return
        yield lock.fileno(), waited


def host_numa_mem(size: int, device: int, numa_node: int, shareable: bool = False) -> HostNumaMem:
    """``HostNumaMem``, retried after evicting the node's page cache if the node is out of memory,
    unless ``TRTLLM_ENGRAM_EVICT_PAGE_CACHE=0``."""
    if os.environ.get(_EVICT_ENV, "1").lower() in ("0", "false"):
        return HostNumaMem(size, device, numa_node, shareable=shareable)
    try:
        # Shared: wait out an eviction in progress, whose child this allocation could push into the OOM killer.
        with _eviction_lock(fcntl.LOCK_SH):
            return HostNumaMem(size, device, numa_node, shareable=shareable)
    except HostNumaOutOfMemoryError as error:
        # cuMemCreate needs 2 MiB blocks, which a node full of page cache often cannot provide.
        with _eviction_lock(fcntl.LOCK_EX) as taken:
            if taken is None:
                raise
            lock_fd, waited = taken
            if waited:
                # The process that held the lock may have evicted this node already.
                with contextlib.suppress(HostNumaOutOfMemoryError):
                    return HostNumaMem(size, device, numa_node, shareable=shareable)
            logger.warning(
                f"Engram: {error}; evicting the page cache of NUMA node {numa_node}, which every "
                f"process on the node loses ({_EVICT_ENV}=0 turns this off)"
            )
            if not evict_numa_node_page_cache(numa_node, lock_fd):
                raise
            logger.info(f"Engram: evicted page cache on NUMA node {numa_node}, retrying")
            return HostNumaMem(size, device, numa_node, shareable=shareable)


class ImportedHostMem:
    """Memory another process exported with ``HostNumaMem.export_fd``, mapped read-only for
    ``device`` and the host; the mapping keeps it alive after the exporter exits."""

    def __init__(self, fd: int, size: int, granularity: int, device: int) -> None:
        self.size = size
        with ExitStack() as cleanup:
            self.address = int(
                _check(cuda.cuMemAddressReserve(size, granularity, 0, 0), "cuMemAddressReserve")
            )
            cleanup.callback(
                lambda: _check(cuda.cuMemAddressFree(self.address, size), "cuMemAddressFree")
            )
            handle = _check(
                cuda.cuMemImportFromShareableHandle(fd, _POSIX_FD),
                "cuMemImportFromShareableHandle",
            )
            cleanup.callback(lambda: _check(cuda.cuMemRelease(handle), "cuMemRelease"))
            _check(cuda.cuMemMap(self.address, size, 0, handle, 0), "cuMemMap")
            cleanup.callback(lambda: _check(cuda.cuMemUnmap(self.address, size), "cuMemUnmap"))
            _check(
                _set_access(
                    self.address, size, device, cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READ
                ),
                "cuMemSetAccess",
            )
            finalizer = weakref.finalize(self, _release, device, self.address, size, handle)
            finalizer.atexit = False
            cleanup.pop_all()


def _unregister(mapping: mmap.mmap, addresses: list[int]) -> None:
    for address in reversed(addresses):
        _check(cuda.cuMemHostUnregister(address), "cuMemHostUnregister")
    mapping.close()


def _release_registered(
    device: int, mapping: mmap.mmap, addresses: list[int], fd: int | None
) -> None:
    torch.cuda.synchronize(device)
    _unregister(mapping, addresses)
    if fd is not None:
        os.close(fd)


class HostRegisteredMem:
    """CUDA-registered host memory under the process NUMA policy, with huge pages requested: anonymous,
    a ``shareable`` memfd, or another process's memfd ``fd``."""

    def __init__(
        self, size: int, device: int, shareable: bool = False, fd: int | None = None
    ) -> None:
        if size <= 0:
            raise ValueError("Engram host mapping size must be positive")
        self.size = round_up(size, mmap.PAGESIZE)
        self._fd = None
        addresses = []
        with ExitStack() as cleanup:
            if fd is not None:
                # Not every device can register read-only host memory, so the mapping is writable.
                mapping = mmap.mmap(
                    fd, self.size, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE
                )
            elif shareable:
                self._fd = os.memfd_create(
                    "trtllm-engram-table", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING
                )
                cleanup.callback(os.close, self._fd)
                os.ftruncate(self._fd, self.size)
                # Importers must not resize the memory this process maps.
                fcntl.fcntl(
                    self._fd,
                    fcntl.F_ADD_SEALS,
                    fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SEAL,
                )
                mapping = mmap.mmap(
                    self._fd,
                    self.size,
                    flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                )
            else:
                mapping = mmap.mmap(
                    -1,
                    self.size,
                    flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                    prot=mmap.PROT_READ | mmap.PROT_WRITE,
                )
            cleanup.callback(_unregister, mapping, addresses)
            self.address = ctypes.addressof(ctypes.c_uint8.from_buffer(mapping))
            try:
                mapping.madvise(mmap.MADV_HUGEPAGE)
            except (AttributeError, OSError) as error:
                logger.debug(
                    f"Engram registered host mapping could not request huge pages: {error}"
                )
            # Bound registration size for kernels that cannot register >2 GiB at once.
            chunk = 2 << 30
            flags = cuda.CU_MEMHOSTREGISTER_PORTABLE | cuda.CU_MEMHOSTREGISTER_DEVICEMAP
            for offset in range(0, self.size, chunk):
                address = self.address + offset
                _check(
                    cuda.cuMemHostRegister(address, min(chunk, self.size - offset), flags),
                    "cuMemHostRegister",
                )
                addresses.append(address)
            finalizer = weakref.finalize(
                self, _release_registered, device, mapping, addresses, self._fd
            )
            finalizer.atexit = False
            cleanup.pop_all()

    def export_fd(self) -> int:
        """A new descriptor of ``shareable`` memory; the caller closes it."""
        return os.dup(self._fd)
