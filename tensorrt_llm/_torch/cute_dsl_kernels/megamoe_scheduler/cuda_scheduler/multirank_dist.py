"""CUDA fabric allocation shared across expert-parallel processes."""

from __future__ import annotations

import torch
from cuda.bindings import driver as cuda


def _check(result: object) -> object:
    if isinstance(result, tuple):
        error, *values = result
    else:
        error, values = result, []
    if int(error) != int(cuda.CUresult.CUDA_SUCCESS):
        _, name = cuda.cuGetErrorName(error)
        _, description = cuda.cuGetErrorString(error)
        if isinstance(name, bytes):
            name = name.decode()
        if isinstance(description, bytes):
            description = description.decode()
        raise RuntimeError(f"CUDA driver call failed: {name}: {description}")
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


class FabricSymmetricBuffer:
    """One CUDA-fabric allocation mapped read/write into every EP process."""

    def __init__(self, nbytes: int, device: int):
        self.device = device
        properties = cuda.CUmemAllocationProp()
        properties.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        properties.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        properties.location.id = device
        properties.requestedHandleTypes = (
            cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
        )
        self._properties = properties
        self.granularity = int(
            _check(
                cuda.cuMemGetAllocationGranularity(
                    properties,
                    cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
                )
            )
        )
        self.size = (nbytes + self.granularity - 1) // self.granularity
        self.size *= self.granularity
        self._mappings: list[int] = []
        self._reservations: list[int] = []
        self._imported: list[object] = []
        self._closed = False
        self._handle = None
        self.ptr = 0
        self.shareable = b""
        try:
            self._handle = _check(cuda.cuMemCreate(self.size, properties, 0))
            self.ptr = self._map(self._handle)
            _check(cuda.cuMemsetD8(cuda.CUdeviceptr(self.ptr), 0, self.size))
            exported = _check(
                cuda.cuMemExportToShareableHandle(
                    self._handle,
                    cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
                    0,
                )
            )
            self.shareable = bytes(exported.data)
        except BaseException:
            self.close()
            raise

    def _map(self, handle: object) -> int:
        address = int(
            _check(cuda.cuMemAddressReserve(self.size, self.granularity, 0, 0))
        )
        self._reservations.append(address)
        _check(cuda.cuMemMap(address, self.size, 0, handle, 0))
        self._mappings.append(address)
        access = cuda.CUmemAccessDesc()
        access.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = self.device
        access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        _check(cuda.cuMemSetAccess(address, self.size, [access], 1))
        return int(address)

    def import_peer(self, shareable: bytes) -> int:
        handle = _check(
            cuda.cuMemImportFromShareableHandle(
                shareable,
                cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC,
            )
        )
        self._imported.append(handle)
        return self._map(handle)

    def close(self) -> None:
        """Release every mapping and fabric handle after device work is drained."""
        if self._closed:
            return
        with torch.cuda.device(self.device):
            while self._mappings:
                address = self._mappings[-1]
                _check(cuda.cuMemUnmap(address, self.size))
                self._mappings.pop()
            while self._reservations:
                address = self._reservations[-1]
                _check(cuda.cuMemAddressFree(address, self.size))
                self._reservations.pop()
            while self._imported:
                handle = self._imported[-1]
                _check(cuda.cuMemRelease(handle))
                self._imported.pop()
            if self._handle is not None:
                _check(cuda.cuMemRelease(self._handle))
                self._handle = None
            self.ptr = 0
            self.shareable = b""
            self._closed = True


__all__ = ["FabricSymmetricBuffer"]
