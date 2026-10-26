# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Page-locked host memory for Engram tables."""

import ctypes
import mmap
import weakref
from contextlib import ExitStack

import torch
from cuda.bindings import driver as cuda

from tensorrt_llm.logger import logger


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
    return _check(
        cuda.cuDeviceGetAttribute(attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID, device),
        "cuDeviceGetAttribute",
    )


def _release(device: int, address: int, size: int, handle) -> None:
    torch.cuda.synchronize(device)
    _check(cuda.cuMemUnmap(address, size), "cuMemUnmap")
    _check(cuda.cuMemRelease(handle), "cuMemRelease")
    _check(cuda.cuMemAddressFree(address, size), "cuMemAddressFree")


class HostNumaMem:
    """A CUDA VMM mapping of page-locked host memory on ``numa_node``, shared by ``device``
    and the host at one address."""

    def __init__(self, size: int, device: int, numa_node: int) -> None:
        properties = cuda.CUmemAllocationProp()
        properties.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        properties.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA
        properties.location.id = numa_node
        granularity = _check(
            cuda.cuMemGetAllocationGranularity(
                properties,
                cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
            ),
            "cuMemGetAllocationGranularity",
        )
        self.size = round_up(size, granularity)
        with ExitStack() as cleanup:
            self.address = int(
                _check(
                    cuda.cuMemAddressReserve(self.size, granularity, 0, 0), "cuMemAddressReserve"
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
            access = []
            for location_type, location_id in (
                (cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE, device),
                (cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST, 0),
            ):
                descriptor = cuda.CUmemAccessDesc()
                descriptor.location.type = location_type
                descriptor.location.id = location_id
                descriptor.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
                access.append(descriptor)
            _check(
                cuda.cuMemSetAccess(self.address, self.size, access, len(access)), "cuMemSetAccess"
            )
            finalizer = weakref.finalize(self, _release, device, self.address, self.size, handle)
            # Process exit reclaims the mapping; releasing it then would race driver teardown.
            finalizer.atexit = False
            cleanup.pop_all()


def _unregister(mapping: mmap.mmap, addresses: list[int]) -> None:
    for address in reversed(addresses):
        _check(cuda.cuMemHostUnregister(address), "cuMemHostUnregister")
    mapping.close()


def _release_registered(device: int, mapping: mmap.mmap, addresses: list[int]) -> None:
    torch.cuda.synchronize(device)
    _unregister(mapping, addresses)


class HostRegisteredMem:
    """CUDA-registered anonymous memory allocated under the process NUMA policy.

    The OS supplies a page-aligned mapping. Transparent huge pages are requested
    but their size and availability are left to the kernel.
    """

    def __init__(self, size: int, device: int) -> None:
        if size <= 0:
            raise ValueError("Engram host mapping size must be positive")
        self.size = round_up(size, mmap.PAGESIZE)
        mapping = mmap.mmap(
            -1,
            self.size,
            flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        addresses = []
        with ExitStack() as cleanup:
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
            finalizer = weakref.finalize(self, _release_registered, device, mapping, addresses)
            finalizer.atexit = False
            cleanup.pop_all()
