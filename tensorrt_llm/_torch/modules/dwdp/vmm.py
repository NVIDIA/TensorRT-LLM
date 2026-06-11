# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""CUDA Virtual Memory Management (VMM) utilities for DWDP.

This module provides low-level CUDA VMM operations including:
- Page alignment utilities
- Allocation property configuration
- Granularity queries
- Handle creation and mapping
- Tensor creation from VA pointers
"""

from __future__ import annotations

import functools
import platform
from typing import Tuple

import torch

from tensorrt_llm.logger import logger

try:
    from cuda.bindings import driver as cuda

    logger.debug("[DWDP vmm] Using cuda.bindings.driver")
except ImportError:
    from cuda import cuda

    logger.debug("[DWDP vmm] Falling back to legacy `cuda` bindings (cuda.bindings not available)")


def check_cu_result(cu_func_ret):
    """Check CUDA driver API result and raise on error.

    Args:
        cu_func_ret: Return value from CUDA driver API call.

    Returns:
        Extracted result(s) from the tuple, or None if only status.

    Raises:
        RuntimeError: If CUDA call failed.
    """
    if isinstance(cu_func_ret, tuple):
        cu_result, *others = cu_func_ret
        if cu_result != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA error: {cu_result}")
        if len(others) == 1:
            return others[0]
        elif len(others) > 1:
            return tuple(others)
        else:
            return None
    else:
        if cu_func_ret != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CUDA error: {cu_func_ret}")
        return None


def align_up(value: int, alignment: int) -> int:
    """Align value up to the nearest multiple of alignment.

    Args:
        value: Value to align.
        alignment: Alignment boundary (must be power of 2).

    Returns:
        Value aligned up to alignment boundary.

    Raises:
        ValueError: If alignment is not a positive power of 2.
    """
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError(f"alignment must be a positive power of 2, got {alignment}")
    return ((value + alignment - 1) // alignment) * alignment


def align_down(value: int, alignment: int) -> int:
    """Align value down to the nearest multiple of alignment.

    Args:
        value: Value to align.
        alignment: Alignment boundary (must be power of 2).

    Returns:
        Value aligned down to alignment boundary.

    Raises:
        ValueError: If alignment is not a positive power of 2.
    """
    if alignment <= 0 or (alignment & (alignment - 1)) != 0:
        raise ValueError(f"alignment must be a positive power of 2, got {alignment}")
    return (value // alignment) * alignment


def get_allocation_prop(device_id: int, fabric_only: bool = True) -> cuda.CUmemAllocationProp:
    """Get allocation property for MNNVL memory.

    Args:
        device_id: CUDA device ordinal.
        fabric_only: If True, force the fabric handle type (used internally
            for granularity queries — the value isn't consumed by an actual
            export, so the choice is arbitrary as long as it's non-zero).
            If False, pick the peer-shareable handle type appropriate for
            the current arch via :func:`peer_handle_type` so that the
            ``cuMemCreate`` call matches what ``transport.py`` will later
            export — see the docstring of :func:`peer_handle_type` for the
            arch table.

    Returns:
        CUmemAllocationProp configured for the device.
    """
    location = cuda.CUmemLocation()
    location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    location.id = device_id

    allocation_prop = cuda.CUmemAllocationProp()
    allocation_prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    allocation_prop.location = location

    if fabric_only:
        allocation_prop.requestedHandleTypes = (
            cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        )
    else:
        # Single source of truth for the arch -> handle-type mapping.
        # ``transport.py`` independently calls ``peer_handle_type()`` to
        # decide the export/import side; routing the create side through
        # the same helper guarantees the two cannot drift.
        allocation_prop.requestedHandleTypes = peer_handle_type()

    return allocation_prop


@functools.lru_cache(maxsize=None)
def _get_allocation_granularity_cached(device_id: int) -> int:
    """Cached implementation of get_allocation_granularity (thread-safe via lru_cache)."""
    allocation_prop = get_allocation_prop(device_id)
    option = cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
    return check_cu_result(cuda.cuMemGetAllocationGranularity(prop=allocation_prop, option=option))


def get_allocation_granularity(device_id: int, use_cache: bool = True) -> int:
    """Get allocation granularity for VMM on the specified device.

    Args:
        device_id: CUDA device ordinal.
        use_cache: If True, cache and reuse granularity per device (thread-safe).

    Returns:
        Allocation granularity in bytes (typically 2MB on GB200).
    """
    if use_cache:
        return _get_allocation_granularity_cached(device_id)

    allocation_prop = get_allocation_prop(device_id)
    option = cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED
    return check_cu_result(cuda.cuMemGetAllocationGranularity(prop=allocation_prop, option=option))


def get_access_desc(device_id: int) -> cuda.CUmemAccessDesc:
    """Get memory access descriptor for read/write access.

    Args:
        device_id: CUDA device ordinal.

    Returns:
        CUmemAccessDesc for read/write access.
    """
    location = cuda.CUmemLocation()
    location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    location.id = device_id

    madesc = cuda.CUmemAccessDesc()
    madesc.location = location
    madesc.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

    return madesc


def peer_handle_type() -> cuda.CUmemAllocationHandleType:
    """Return the appropriate peer-shareable handle type for the current arch.

    On aarch64 (GB200) we use ``CU_MEM_HANDLE_TYPE_FABRIC`` which routes
    through the IMEX channel.  On x86_64 (B200/H100/H200 single-node hosts
    with no IMEX) ``CU_MEM_HANDLE_TYPE_FABRIC`` is denied with
    ``CUDA_ERROR_NOT_PERMITTED`` (800), so we use
    ``CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR`` and exchange the FDs
    between sibling MPI workers via ``pidfd_open`` / ``pidfd_getfd``
    (mirrors ``MnnvlMemory.get_allocation_prop`` in ``_mnnvl_utils.py``).
    """
    arch = platform.machine().lower()
    if "aarch64" in arch:
        return cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
    return cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR


def create_fabric_handle(size: int, device_id: int) -> int:
    """Create a peer-shareable memory handle.

    The handle type is selected by ``peer_handle_type()``: FABRIC on
    aarch64 (GB200), POSIX_FILE_DESCRIPTOR on x86_64.  The function name
    is preserved for backward compatibility.

    Args:
        size: Size in bytes (must be aligned to granularity).
        device_id: CUDA device ordinal.

    Returns:
        Handle as integer.

    Raises:
        RuntimeError: If creation fails.
    """
    allocation_prop = get_allocation_prop(device_id, fabric_only=False)
    handle = check_cu_result(cuda.cuMemCreate(size, allocation_prop, flags=0))
    return int(handle)


def create_local_handle(size: int, device_id: int) -> int:
    """Create a local (non-shareable) memory handle.

    Uses CU_MEM_HANDLE_TYPE_NONE — the handle cannot be exported to other
    processes but avoids consuming NVLink fabric routing table entries.
    Use this for buffers that only need local GPU access (e.g., page pool
    double buffers that are written via P2P copy and read locally).

    Args:
        size: Size in bytes (must be aligned to granularity).
        device_id: CUDA device ordinal.

    Returns:
        Handle as integer.

    Raises:
        RuntimeError: If creation fails.
    """
    allocation_prop = get_allocation_prop(device_id, fabric_only=False)
    # Override to HANDLE_TYPE_NONE (no shareable handle)
    allocation_prop.requestedHandleTypes = cuda.CUmemAllocationHandleType(0)
    handle = check_cu_result(cuda.cuMemCreate(size, allocation_prop, flags=0))
    return int(handle)


def release_handle(handle: int) -> None:
    """Release a memory handle.

    Args:
        handle: Handle to release.
    """
    if handle != 0:
        check_cu_result(cuda.cuMemRelease(handle))


def reserve_va(size: int, granularity: int) -> int:
    """Reserve virtual address space.

    Args:
        size: Size in bytes.
        granularity: Alignment granularity.

    Returns:
        Virtual address as integer.
    """
    va = check_cu_result(cuda.cuMemAddressReserve(size, granularity, 0, 0))
    return int(va)


def free_va(va: int, size: int) -> None:
    """Free reserved virtual address space.

    Args:
        va: Virtual address to free.
        size: Size of the reserved region.
    """
    if va != 0:
        device_ptr = cuda.CUdeviceptr(va)
        check_cu_result(cuda.cuMemAddressFree(device_ptr, size))


def map_handle(va: int, size: int, handle: int, offset: int = 0) -> None:
    """Map a memory handle to virtual address.

    Args:
        va: Virtual address to map to.
        size: Size in bytes.
        handle: Memory handle.
        offset: Offset within the handle (typically 0 for fabric handles).
    """
    check_cu_result(cuda.cuMemMap(va, size, offset, handle, 0))


def unmap_va(va: int, size: int) -> None:
    """Unmap virtual address region.

    Args:
        va: Virtual address to unmap.
        size: Size in bytes.
    """
    check_cu_result(cuda.cuMemUnmap(va, size))


def set_access(va: int, size: int, device_id: int) -> None:
    """Set memory access permissions.

    Args:
        va: Virtual address.
        size: Size in bytes.
        device_id: CUDA device ordinal.
    """
    madesc = get_access_desc(device_id)
    check_cu_result(cuda.cuMemSetAccess(va, size, [madesc], 1))


def tensor_from_ptr(
    ptr: int,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device_id: int,
) -> torch.Tensor:
    """Create a torch tensor from a CUDA virtual address pointer.

    This creates a tensor view over existing CUDA memory. The caller is
    responsible for ensuring the memory remains valid for the tensor's lifetime.

    Args:
        ptr: CUDA virtual address pointer.
        shape: Shape of the tensor.
        dtype: Data type of the tensor.
        device_id: CUDA device ordinal.

    Returns:
        Torch tensor viewing the memory at ptr.

    Raises:
        ValueError: If ptr is 0 or shape is invalid.
    """
    if ptr == 0:
        raise ValueError("Cannot create tensor from null pointer")

    numel = 1
    for dim in shape:
        if dim <= 0:
            raise ValueError(f"All dimensions must be positive, got shape={shape}")
        numel *= dim

    element_size = torch.tensor([], dtype=dtype).element_size()
    total_bytes = numel * element_size

    # Use DLPack approach for robust pointer wrapping
    # This mimics the pack_strided_memory pattern from _dlpack_utils
    try:
        from tensorrt_llm._dlpack_utils import create_dlpack_capsule

        # For contiguous memory, we create a single segment
        capsule_wrapper = create_dlpack_capsule(
            ptr=ptr,
            segment_size=total_bytes,
            segment_stride=0,  # Not used for single segment
            num_segments=1,
            torch_dtype=dtype,
            dev_id=device_id,
        )

        # Convert via DLPack
        tensor = torch.utils.dlpack.from_dlpack(capsule_wrapper.capsule)
        # Keep reference to capsule to prevent GC
        tensor._capsule_wrapper = capsule_wrapper

        # Reshape to desired shape
        return tensor.reshape(shape)

    except ImportError:
        logger.info(
            "[DWDP vmm] tensorrt_llm._dlpack_utils not available; "
            "falling back to _tensor_from_ptr_internal (D2D copy, not zero-copy)"
        )
        return _tensor_from_ptr_internal(ptr, shape, dtype, device_id)


def _tensor_from_ptr_internal(
    ptr: int,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device_id: int,
) -> torch.Tensor:
    """Internal tensor creation using PyTorch storage.

    Used as fallback when tensorrt_llm._dlpack_utils is not available.
    This uses PyTorch's internal _UntypedStorage to wrap a CUDA pointer.
    """
    import ctypes

    numel = 1
    for dim in shape:
        numel *= dim

    element_size = torch.tensor([], dtype=dtype).element_size()
    total_bytes = numel * element_size

    # Create an untyped storage from the CUDA pointer
    # Using torch.cuda.cudart() to access the storage creation
    device = torch.device(f"cuda:{device_id}")

    # Create storage using ctypes to call internal PyTorch function
    # This approach uses torch.Storage.from_buffer equivalent for CUDA
    try:
        # Get the data pointer of the new storage and copy memory reference
        # Note: This creates a view, we need to use set_ instead
        tensor = torch.empty(shape, dtype=dtype, device=device)

        # Use ctypes to set the data pointer directly
        # This is a workaround - in production, use tensorrt_llm._dlpack_utils
        data_ptr_func = ctypes.pythonapi.PyLong_AsVoidPtr
        data_ptr_func.argtypes = [ctypes.py_object]
        data_ptr_func.restype = ctypes.c_void_p

        # For now, create the tensor and copy the data using cudaMemcpy
        # This is not zero-copy but safe for testing
        # In production with full tensorrt_llm, the DLPack path will be used
        cuda.cuMemcpyDtoD(tensor.data_ptr(), ptr, total_bytes)

        return tensor

    except Exception as e:
        logger.warning(
            f"[DWDP vmm] _tensor_from_ptr_internal UntypedStorage path failed ({e!r}); "
            "falling back to empty-tensor + cuMemcpyDtoD (not zero-copy)"
        )
        tensor = torch.empty(shape, dtype=dtype, device=device)
        cuda.cuMemcpyDtoD(tensor.data_ptr(), ptr, total_bytes)
        return tensor


class VMMHandle:
    """RAII wrapper for a CUDA VMM physical memory handle.

    This class manages the lifecycle of a CUDA memory handle,
    ensuring proper cleanup on destruction.

    Attributes:
        handle: The raw CUDA handle as integer.
        size: Size of the allocated memory in bytes.
        device_id: CUDA device ordinal.
    """

    __slots__ = ("_handle", "_size", "_device_id", "_released")

    def __init__(self, size: int, device_id: int):
        """Create a new VMM handle.

        Args:
            size: Size in bytes (will be aligned to granularity).
            device_id: CUDA device ordinal.
        """
        granularity = get_allocation_granularity(device_id)
        aligned_size = align_up(size, granularity)

        self._handle = create_fabric_handle(aligned_size, device_id)
        self._size = aligned_size
        self._device_id = device_id
        self._released = False

    @property
    def handle(self) -> int:
        """Get the raw handle value."""
        if self._released:
            raise RuntimeError("Handle has been released")
        return self._handle

    @property
    def size(self) -> int:
        """Get the allocated size."""
        return self._size

    @property
    def device_id(self) -> int:
        """Get the device ID."""
        return self._device_id

    def release(self) -> None:
        """Release the handle. Idempotent."""
        if not self._released:
            release_handle(self._handle)
            self._released = True

    def __del__(self):
        """Clean up on destruction."""
        try:
            self.release()
        except Exception:
            pass  # Ignore errors during destruction

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release handle."""
        self.release()
        return False


class VARegion:
    """RAII wrapper for a CUDA virtual address region.

    This class manages the lifecycle of a reserved VA region,
    including mapping/unmapping of handles.

    Attributes:
        va: The virtual address as integer.
        size: Total size of the region in bytes.
        device_id: CUDA device ordinal.
    """

    __slots__ = ("_va", "_size", "_device_id", "_granularity", "_mappings", "_released")

    def __init__(self, size: int, device_id: int):
        """Reserve a VA region.

        Args:
            size: Size in bytes.
            device_id: CUDA device ordinal.
        """
        self._device_id = device_id
        self._granularity = get_allocation_granularity(device_id)
        aligned_size = align_up(size, self._granularity)

        self._va = reserve_va(aligned_size, self._granularity)
        self._size = aligned_size
        self._mappings: list[Tuple[int, int]] = []  # List of (offset, size) mappings
        self._released = False

    @property
    def va(self) -> int:
        """Get the base virtual address."""
        if self._released:
            raise RuntimeError("VA region has been released")
        return self._va

    @property
    def size(self) -> int:
        """Get the total size."""
        return self._size

    def map(self, offset: int, size: int, handle: int, handle_offset: int = 0) -> int:
        """Map a handle at an offset within this VA region.

        Args:
            offset: Offset within this VA region.
            size: Size to map.
            handle: Memory handle to map.
            handle_offset: Offset within the handle.

        Returns:
            Virtual address of the mapping.

        Raises:
            ValueError: If the mapping would exceed the region bounds.
        """
        if offset + size > self._size:
            raise ValueError(
                f"Mapping at offset={offset} size={size} exceeds region size={self._size}"
            )

        va = self._va + offset
        map_handle(va, size, handle, handle_offset)
        set_access(va, size, self._device_id)
        self._mappings.append((offset, size))

        return va

    def unmap_all(self) -> None:
        """Unmap all mappings in this region."""
        for offset, size in self._mappings:
            try:
                unmap_va(self._va + offset, size)
            except Exception:
                pass  # Best effort cleanup
        self._mappings.clear()

    def release(self) -> None:
        """Release the VA region. Idempotent."""
        if not self._released:
            self.unmap_all()
            free_va(self._va, self._size)
            self._released = True

    def __del__(self):
        """Clean up on destruction."""
        try:
            self.release()
        except Exception:
            pass  # Ignore errors during destruction

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release region."""
        self.release()
        return False
