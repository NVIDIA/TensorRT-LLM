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

from typing import Type

import cuda.bindings.driver as drv

from ._common import MemAddress
from ._exceptions import CuError
from ._utils import ItemHolderWithSharedPool, PooledFactoryBase, _unwrap, div_up


def _is_prop_supported(prop: drv.CUmemAllocationProp) -> bool:
    err, handle = drv.cuMemCreate(2 << 20, prop, 0)
    err_int = int(err)
    if err_int == int(drv.CUresult.CUDA_SUCCESS):
        _unwrap(drv.cuMemRelease(handle))
        return True
    # Note: OOM is intentionally not caught here — OOM on a 2 MiB probe
    # indicates a fundamental resource problem, not an unsupported property.
    elif err_int in (
        int(drv.CUresult.CUDA_ERROR_NOT_PERMITTED),
        int(drv.CUresult.CUDA_ERROR_NOT_SUPPORTED),
        int(drv.CUresult.CUDA_ERROR_INVALID_DEVICE),
        int(drv.CUresult.CUDA_ERROR_INVALID_VALUE),
    ):
        return False
    else:
        raise CuError(err)


# Physical memory
class NativePhysMemAllocator:
    __slots__ = ("_device_id", "_size", "_prop", "_outstanding_handles")

    _device_id: int
    _size: int
    _prop: drv.CUmemAllocationProp
    _outstanding_handles: set[int]  # allocated but not released

    def __init__(self, size: int) -> None:
        self._device_id = int(_unwrap(drv.cuCtxGetDevice()))  # pyright: ignore
        self._size = size
        prop = drv.CUmemAllocationProp()
        prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = self._device_id
        prop.allocFlags.gpuDirectRDMACapable = 1
        prop.requestedHandleTypes = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        if not _is_prop_supported(prop):
            prop.requestedHandleTypes = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
            if not _is_prop_supported(prop):
                prop.allocFlags.gpuDirectRDMACapable = 0
                if not _is_prop_supported(prop):
                    raise ValueError("Failed to create physical memory allocation property")
        self._prop = prop
        self._outstanding_handles = set()

    def allocate(self) -> drv.CUmemGenericAllocationHandle:
        handle: drv.CUmemGenericAllocationHandle = _unwrap(
            drv.cuMemCreate(self._size, self._prop, 0)
        )
        int_handle = int(handle)  # pyright: ignore
        assert (int_handle not in self._outstanding_handles) and int_handle != 0
        self._outstanding_handles.add(int_handle)
        return handle

    def release(self, handle: drv.CUmemGenericAllocationHandle) -> None:
        if handle == drv.CUmemGenericAllocationHandle(0):
            return
        assert int(handle) in self._outstanding_handles
        self._outstanding_handles.remove(int(handle))
        try:
            _unwrap(drv.cuMemRelease(handle))
        except:
            print(
                f"Failed to release handle {handle}. num_outstanding = {len(self._outstanding_handles)}"
            )
            raise

    @property
    def device_id(self) -> int:
        return self._device_id

    @property
    def size(self) -> int:
        return self._size


class PhysMem(ItemHolderWithSharedPool[drv.CUmemGenericAllocationHandle]):
    __slots__ = ()


class PooledPhysMemAllocator(PooledFactoryBase[drv.CUmemGenericAllocationHandle, PhysMem]):
    _Holder: Type[PhysMem] = PhysMem
    __slots__ = ("device_id", "phys_mem_size")
    device_id: int
    phys_mem_size: int

    def __init__(self, phys_mem_size: int) -> None:
        """phys_mem_size is the size of each physical memory chunk."""
        raw_alloc = NativePhysMemAllocator(phys_mem_size)
        self.device_id = raw_alloc.device_id
        self.phys_mem_size = phys_mem_size
        super().__init__(lambda: raw_alloc.allocate(), lambda handle: raw_alloc.release(handle))


# Virtual memory
class VirtMem:
    # NOTE: `_page_map` backs the *sparse* mapping mode used by the contiguous
    # primary KV cache (per-sequence arenas, see `_sequence_arena.py` and
    # `contiguous_primary_kvcache/DESIGN.md` §4.1-§4.2). It maps physical chunks
    # at arbitrary granularity-aligned offsets inside the reservation, in
    # contrast to `_pm_stack` which is tail-only LIFO. A single `VirtMem`
    # instance uses one mode or the other, never both (asserted below).
    __slots__ = ("_vm_size", "_allocator", "_address", "_pm_stack", "_access_desc", "_page_map")
    _vm_size: int
    _allocator: PooledPhysMemAllocator
    _address: drv.CUdeviceptr
    _pm_stack: list[PhysMem]
    _access_desc: drv.CUmemAccessDesc
    # chunk_index -> mapped PhysMem, for sparse mode. chunk_index = byte_offset // phys_mem_size.
    _page_map: dict[int, PhysMem]

    def __init__(
        self, vm_size: int, phys_mem_allocator: PooledPhysMemAllocator, init_num_phys_mem: int = 0
    ):
        assert vm_size % phys_mem_allocator.phys_mem_size == 0
        self._allocator = phys_mem_allocator
        device_id = phys_mem_allocator.device_id
        self._address = _unwrap(drv.cuMemAddressReserve(vm_size, 0, 0, 0))
        self._vm_size = vm_size
        self._pm_stack = []
        self._page_map = {}
        self._access_desc = drv.CUmemAccessDesc()
        self._access_desc.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        self._access_desc.location.id = device_id
        self._access_desc.flags = drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self.extend(init_num_phys_mem)

    @property
    def phys_mem_size(self) -> int:
        return self._allocator.phys_mem_size

    def destroy(self) -> None:
        if self._vm_size == 0:
            return
        _unwrap(drv.cuCtxSynchronize())
        while self._pm_stack:
            self._pop().close()
        # Tear down any sparse mappings (see `_page_map`). Unmap each chunk's VA
        # and return its handle to the shared pool.
        for chunk_index, phys_mem in self._page_map.items():
            vm_ptr = drv.CUdeviceptr(int(self._address) + self.phys_mem_size * chunk_index)
            _unwrap(drv.cuMemUnmap(vm_ptr, self.phys_mem_size))
            phys_mem.close()
        self._page_map = {}
        _unwrap(drv.cuMemAddressFree(self._address, self._vm_size))
        self._address = drv.CUdeviceptr(0)
        self._vm_size = 0

    def __del__(self) -> None:
        self.destroy()

    def extend(self, num_phys_mem: int) -> None:
        old_num_phys_mem = self.num_phys_mem
        try:
            for _ in range(num_phys_mem):
                self._push(self._allocator.create())
        except (
            Exception
        ):  # to make realloc behave like normal realloc, we need to rollback if out of memory
            while self.num_phys_mem > old_num_phys_mem:
                self._pop().close()
            raise

    def shrink(self, num_phys_mem: int) -> None:
        _unwrap(drv.cuCtxSynchronize())
        for _ in range(num_phys_mem):
            self._pop().close()

    # Different from normal realloc, this function never changes the pointer.
    def realloc(self, num_bytes: int) -> None:
        required_num_phys_mem = div_up(num_bytes, self.phys_mem_size)
        if required_num_phys_mem > self.num_phys_mem:
            self.extend(required_num_phys_mem - self.num_phys_mem)
        elif required_num_phys_mem < self.num_phys_mem:
            self.shrink(self.num_phys_mem - required_num_phys_mem)

    def _push(self, phy_mem: PhysMem) -> None:
        # Tail-stack and sparse modes are mutually exclusive on one reservation.
        assert not self._page_map, "cannot use tail-stack extend() on a sparse-mapped VirtMem"
        phys_mem_size = self.phys_mem_size
        assert phys_mem_size * (len(self._pm_stack) + 1) <= self._vm_size
        vm_ptr = drv.CUdeviceptr(self.address + phys_mem_size * len(self._pm_stack))
        _unwrap(drv.cuMemMap(vm_ptr, phys_mem_size, 0, phy_mem.handle, 0))
        _unwrap(drv.cuMemSetAccess(vm_ptr, phys_mem_size, (self._access_desc,), 1))
        self._pm_stack.append(phy_mem)

    def _pop(self) -> PhysMem:
        assert self._pm_stack
        phys_mem_size = self.phys_mem_size
        vm_ptr = drv.CUdeviceptr(self.address + phys_mem_size * (len(self._pm_stack) - 1))
        _unwrap(drv.cuMemUnmap(vm_ptr, phys_mem_size))
        return self._pm_stack.pop()

    @property
    def mapped_bytes(self) -> int:
        return self.phys_mem_size * self.num_phys_mem

    @property
    def virtual_bytes(self) -> int:
        return self._vm_size

    @property
    def num_phys_mem(self) -> int:
        return len(self._pm_stack)

    @property
    def address(self) -> MemAddress:
        return MemAddress(int(self._address))

    # ------------------------------------------------------------------
    # Sparse mapping mode (contiguous primary KV cache)
    #
    # Maps/unmaps physical chunks at arbitrary granularity-aligned offsets
    # inside the reservation, so per-sequence arenas can page memory in on
    # demand ahead of the write frontier and recycle it on free. See
    # `_sequence_arena.py` and DESIGN.md §4.2. Callers must ensure that
    # `unmap_range` is only issued once no in-flight GPU work can touch the
    # range (deferred-reclaim gating lives in the arena layer, DESIGN.md §4.2
    # "Shrink/free"): `cuMemUnmap` of a still-referenced range is an IMA.
    # ------------------------------------------------------------------

    def _chunk_index(self, byte_offset: int) -> int:
        phys_mem_size = self.phys_mem_size
        assert byte_offset % phys_mem_size == 0, "byte_offset must be granularity-aligned"
        assert 0 <= byte_offset < self._vm_size
        return byte_offset // phys_mem_size

    def map_range(self, byte_offset: int, num_chunks: int) -> None:
        """Map ``num_chunks`` physical chunks into the reservation starting at
        ``byte_offset`` (must be a multiple of ``phys_mem_size``).

        Chunks are pulled from the shared physical pool. Access is granted with
        a single ``cuMemSetAccess`` spanning the whole contiguous run (its cost
        is per-mapping, not per-byte -- see the microbenchmark in
        ``contiguous_primary_kvcache/``). On OOM the partial mapping is rolled
        back so the call behaves atomically.
        """
        assert not self._pm_stack, "cannot use sparse map_range() on a tail-stack VirtMem"
        assert num_chunks >= 0
        if num_chunks == 0:
            return
        phys_mem_size = self.phys_mem_size
        start_chunk = self._chunk_index(byte_offset)
        assert (start_chunk + num_chunks) * phys_mem_size <= self._vm_size
        mapped: list[int] = []
        try:
            for i in range(num_chunks):
                chunk_index = start_chunk + i
                assert chunk_index not in self._page_map, f"chunk {chunk_index} already mapped"
                phys_mem = self._allocator.create()
                vm_ptr = drv.CUdeviceptr(int(self._address) + phys_mem_size * chunk_index)
                _unwrap(drv.cuMemMap(vm_ptr, phys_mem_size, 0, phys_mem.handle, 0))
                self._page_map[chunk_index] = phys_mem
                mapped.append(chunk_index)
            span_ptr = drv.CUdeviceptr(int(self._address) + phys_mem_size * start_chunk)
            _unwrap(
                drv.cuMemSetAccess(span_ptr, phys_mem_size * num_chunks, (self._access_desc,), 1)
            )
        except Exception:  # roll back to keep map_range atomic on OOM
            for chunk_index in mapped:
                vm_ptr = drv.CUdeviceptr(int(self._address) + phys_mem_size * chunk_index)
                _unwrap(drv.cuMemUnmap(vm_ptr, phys_mem_size))
                self._page_map.pop(chunk_index).close()
            raise

    def unmap_range(self, byte_offset: int, num_chunks: int) -> None:
        """Unmap ``num_chunks`` chunks starting at ``byte_offset`` and return
        their physical handles to the shared pool.

        The caller MUST guarantee no in-flight GPU work references this range
        (deferred-reclaim gating, DESIGN.md §4.2). ``cuMemUnmap`` is host-side
        and not stream-ordered.
        """
        assert num_chunks >= 0
        if num_chunks == 0:
            return
        phys_mem_size = self.phys_mem_size
        start_chunk = self._chunk_index(byte_offset)
        for i in range(num_chunks):
            chunk_index = start_chunk + i
            phys_mem = self._page_map.pop(chunk_index, None)
            assert phys_mem is not None, f"chunk {chunk_index} is not mapped"
            vm_ptr = drv.CUdeviceptr(int(self._address) + phys_mem_size * chunk_index)
            _unwrap(drv.cuMemUnmap(vm_ptr, phys_mem_size))
            phys_mem.close()

    def is_mapped(self, byte_offset: int) -> bool:
        return self._chunk_index(byte_offset) in self._page_map

    @property
    def num_sparse_chunks(self) -> int:
        """Number of chunks currently mapped via the sparse path."""
        return len(self._page_map)
