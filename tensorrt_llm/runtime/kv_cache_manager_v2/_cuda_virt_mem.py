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


import enum
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final, Optional, Protocol

import cuda.bindings.driver as drv

from ._common import MemAddress
from ._utils import (
    ItemHolderWithSharedPool,
    PooledFactoryBase,
    _unwrap,
    div_up,
    get_current_device_id,
    round_up,
)

MEM_DEBUG: Final[int] = int(os.environ.get("V2_KV_CACHE_MEM_DEBUG", "0")) == 1
_PROBE_SIZE: Final[int] = 2 << 20
_GPU_DIRECT_RDMA_OVER_PCIE_USAGE: Final[int] = 0x4
_GPU_DIRECT_RDMA_WITH_LOCALIZED_MEMORY_ATTRIBUTE: Final[int] = 161
_UNSUPPORTED_ALLOCATION_ERRORS: Final[frozenset[int]] = frozenset(
    {
        int(drv.CUresult.CUDA_ERROR_INVALID_DEVICE),
        int(drv.CUresult.CUDA_ERROR_INVALID_VALUE),
        int(drv.CUresult.CUDA_ERROR_NOT_PERMITTED),
        int(drv.CUresult.CUDA_ERROR_NOT_SUPPORTED),
    }
)


class _LocalizationHandle(Protocol):
    def supports_localization(self) -> bool: ...

    def supports_memory_localization(self) -> bool: ...

    def try_create_localized_allocation_handle(
        self,
        size: int,
        locality_domain_id: int,
        requested_handle_types: int,
        gpu_direct_rdma_capable: bool,
        usage: int,
    ) -> tuple[int, int]: ...

    def try_get_localized_allocation_granularity(
        self,
        locality_domain_id: int,
        requested_handle_types: int,
        gpu_direct_rdma_capable: bool,
        usage: int,
    ) -> tuple[int, int]: ...

    def create_localized_allocation_handle(
        self,
        size: int,
        locality_domain_id: int,
        requested_handle_types: int,
        gpu_direct_rdma_capable: bool,
        usage: int,
    ) -> int: ...


@dataclass(frozen=True)
class _AllocationConfig:
    requested_handle_types: int
    gpu_direct_rdma_capable: bool
    usage: int
    granularity: int


class LocalizationMode(str, enum.Enum):
    AUTO = "auto"
    OFF = "off"
    MOCK = "mock"


def _get_localization_mode() -> LocalizationMode:
    """Return the configured localization mode.

    ``TRT_LLM_MOCK_LOCALIZATION_SUPPORT`` controls the mode:
      - unset: probe hardware support
      - ``0``: force non-localized behavior
      - ``1``: force localized logic while using non-localized allocations
    """
    mode = os.environ.get("TRT_LLM_MOCK_LOCALIZATION_SUPPORT")
    if mode is None:
        return LocalizationMode.AUTO
    if mode == "0":
        return LocalizationMode.OFF
    if mode == "1":
        return LocalizationMode.MOCK
    raise ValueError("TRT_LLM_MOCK_LOCALIZATION_SUPPORT must be '0' or '1' when set")


def _get_localization_handle() -> _LocalizationHandle:
    import tensorrt_llm.bindings.internal.runtime as _tbr

    return _tbr.LocalizationHandle()


def _get_current_context() -> drv.CUcontext:
    return _unwrap(drv.cuCtxGetCurrent())


@contextmanager
def _activate_cuda_context(context: drv.CUcontext, device_id: int) -> Iterator[None]:
    current = _get_current_context()
    context_was_pushed = int(current) != int(context)
    if context_was_pushed:
        _unwrap(drv.cuCtxPushCurrent(context))
    try:
        current_device_id = get_current_device_id()
        if current_device_id != device_id:
            raise RuntimeError(
                "CUDA context device changed while using virtual memory: "
                f"expected={device_id}, current={current_device_id}"
            )
        yield
    finally:
        if context_was_pushed:
            popped = _unwrap(drv.cuCtxPopCurrent())
            if int(popped) != int(context):
                raise RuntimeError("CUDA context stack was modified unexpectedly")


def _location_to_locality_domain_id(location: int) -> int:
    if location == _LOCALITY_DOMAIN_LOC_0:
        return 0
    if location == _LOCALITY_DOMAIN_LOC_1:
        return 1
    raise ValueError(f"Unsupported locality domain localization location: {location}")


def _create_localized_allocation_handle(
    localization_handle: _LocalizationHandle,
    size: int,
    prop: drv.CUmemAllocationProp,
    locality_domain_id: int,
) -> drv.CUmemGenericAllocationHandle:
    handle = localization_handle.create_localized_allocation_handle(
        size,
        locality_domain_id,
        int(prop.requestedHandleTypes),
        bool(prop.allocFlags.gpuDirectRDMACapable),
        int(prop.allocFlags.usage),
    )
    return drv.CUmemGenericAllocationHandle(handle)


def _supports_memory_localization(handle: _LocalizationHandle) -> bool:
    memory_support = getattr(handle, "supports_memory_localization", None)
    if memory_support is not None:
        return bool(memory_support())
    return bool(handle.supports_localization())


def is_device_localization_supported(device_id: int) -> bool:
    """Return whether the given device supports locality domain memory localization.

    Callers still need to opt in through ``GpuCacheTierConfig.enable_locality_domains``.
    ``TRT_LLM_MOCK_LOCALIZATION_SUPPORT=1`` forces the localized path for
    tests. ``TRT_LLM_MOCK_LOCALIZATION_SUPPORT=0`` forces the non-localized
    path even on localization-capable hardware. Unset probes real hardware
    support.
    """
    mode = _get_localization_mode()
    if mode is LocalizationMode.OFF:
        return False
    if mode is LocalizationMode.MOCK:
        return True

    current_device_id = get_current_device_id()
    if device_id != current_device_id:
        if MEM_DEBUG:
            print(
                "is_device_localization_supported called for non-current device: "
                f"requested={device_id} current={current_device_id}"
            )
        return False

    try:
        handle = _get_localization_handle()
    except (AttributeError, ImportError):
        return False
    return _supports_memory_localization(handle)


def _raise_cuda_error(error_code: int) -> None:
    _unwrap(drv.CUresult(error_code))
    raise AssertionError("Expected a failing CUDA result")


def _make_allocation_prop(device_id: int, config: _AllocationConfig) -> drv.CUmemAllocationProp:
    prop = drv.CUmemAllocationProp()
    prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device_id
    prop.requestedHandleTypes = config.requested_handle_types
    prop.allocFlags.gpuDirectRDMACapable = int(config.gpu_direct_rdma_capable)
    prop.allocFlags.usage = config.usage
    return prop


def _allocation_candidates(
    gpu_direct_rdma_with_localized_memory: bool,
) -> tuple[tuple[int, bool, int], ...]:
    fabric = int(drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC)
    posix_fd = int(drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
    no_handle = int(drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE)
    gdr_usage = 0 if gpu_direct_rdma_with_localized_memory else _GPU_DIRECT_RDMA_OVER_PCIE_USAGE
    # Prefer shareable handle types so UCX cuda_ipc can export the memory for
    # intra-node zero-copy transfers: FABRIC (MNNVL) > POSIX_FILE_DESCRIPTOR
    # (pidfd-based, UCX >= 1.22) > NONE, and within each handle type prefer
    # gpuDirectRDMACapable over not.
    return (
        (fabric, True, gdr_usage),
        (fabric, False, 0),
        (posix_fd, True, gdr_usage),
        (posix_fd, False, 0),
        (no_handle, True, gdr_usage),
        (no_handle, False, 0),
    )


def _query_localized_gdr_support(device_id: int) -> bool:
    err, supported = drv.cuDeviceGetAttribute(
        _GPU_DIRECT_RDMA_WITH_LOCALIZED_MEMORY_ATTRIBUTE, device_id
    )
    error_code = int(err)
    if error_code == int(drv.CUresult.CUDA_SUCCESS):
        return bool(supported)
    if error_code in _UNSUPPORTED_ALLOCATION_ERRORS:
        return False
    _raise_cuda_error(error_code)


def _try_get_default_allocation_granularity(
    device_id: int,
    requested_handle_types: int,
    gpu_direct_rdma_capable: bool,
    usage: int,
) -> tuple[int, int]:
    config = _AllocationConfig(
        requested_handle_types=requested_handle_types,
        gpu_direct_rdma_capable=gpu_direct_rdma_capable,
        usage=usage,
        granularity=1,
    )
    prop = _make_allocation_prop(device_id, config)
    err, granularity = drv.cuMemGetAllocationGranularity(
        prop, drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
    )
    return int(err), int(granularity or 0)


def _try_create_default_allocation_handle(
    size: int,
    device_id: int,
    requested_handle_types: int,
    gpu_direct_rdma_capable: bool,
    usage: int,
) -> tuple[int, int]:
    config = _AllocationConfig(
        requested_handle_types=requested_handle_types,
        gpu_direct_rdma_capable=gpu_direct_rdma_capable,
        usage=usage,
        granularity=1,
    )
    prop = _make_allocation_prop(device_id, config)
    err, handle = drv.cuMemCreate(size, prop, 0)
    return int(err), int(handle or 0)


def _try_get_localized_allocation_granularity(
    localization_handle: _LocalizationHandle | None,
    device_id: int,
    locality_domain_id: int,
    requested_handle_types: int,
    gpu_direct_rdma_capable: bool,
    usage: int,
) -> tuple[int, int]:
    if _get_localization_mode() is LocalizationMode.MOCK:
        return _try_get_default_allocation_granularity(
            device_id,
            requested_handle_types,
            gpu_direct_rdma_capable,
            usage,
        )
    if localization_handle is None:
        raise RuntimeError("Missing locality domain memory-localization handle")
    result = localization_handle.try_get_localized_allocation_granularity(
        locality_domain_id,
        requested_handle_types,
        gpu_direct_rdma_capable,
        usage,
    )
    return int(result[0]), int(result[1])


def _try_create_localized_allocation_handle(
    localization_handle: _LocalizationHandle | None,
    size: int,
    device_id: int,
    locality_domain_id: int,
    requested_handle_types: int,
    gpu_direct_rdma_capable: bool,
    usage: int,
) -> tuple[int, int]:
    if _get_localization_mode() is LocalizationMode.MOCK:
        return _try_create_default_allocation_handle(
            size,
            device_id,
            requested_handle_types,
            gpu_direct_rdma_capable,
            usage,
        )
    if localization_handle is None:
        raise RuntimeError("Missing locality domain memory-localization handle")
    result = localization_handle.try_create_localized_allocation_handle(
        size,
        locality_domain_id,
        requested_handle_types,
        gpu_direct_rdma_capable,
        usage,
    )
    return int(result[0]), int(result[1])


def _select_localized_allocation_config(
    size: int,
    device_id: int,
    localization_handle: _LocalizationHandle | None,
) -> _AllocationConfig:
    if size <= 0:
        raise ValueError("Allocation size must be positive")
    localized_gdr_supported = _query_localized_gdr_support(device_id)
    locations = tuple(
        _location_to_locality_domain_id(location) for location in _LOCALITY_DOMAIN_LOCATIONS
    )

    for requested_handle_types, gpu_direct_rdma_capable, usage in _allocation_candidates(
        localized_gdr_supported
    ):
        granularities = []
        candidate_supported = True
        for locality_domain_id in locations:
            error_code, granularity = _try_get_localized_allocation_granularity(
                localization_handle,
                device_id,
                locality_domain_id,
                requested_handle_types,
                gpu_direct_rdma_capable,
                usage,
            )
            if error_code in _UNSUPPORTED_ALLOCATION_ERRORS:
                candidate_supported = False
                break
            if error_code != int(drv.CUresult.CUDA_SUCCESS):
                _raise_cuda_error(error_code)
            if granularity <= 0:
                raise RuntimeError("CUDA returned an invalid allocation granularity")
            granularities.append(granularity)
        if not candidate_supported:
            continue

        common_granularity = math.lcm(*granularities)
        probe_size = round_up(_PROBE_SIZE, common_granularity)
        probe_handles: list[drv.CUmemGenericAllocationHandle] = []
        try:
            for locality_domain_id in locations:
                error_code, raw_handle = _try_create_localized_allocation_handle(
                    localization_handle,
                    probe_size,
                    device_id,
                    locality_domain_id,
                    requested_handle_types,
                    gpu_direct_rdma_capable,
                    usage,
                )
                if error_code in _UNSUPPORTED_ALLOCATION_ERRORS:
                    candidate_supported = False
                    break
                if error_code != int(drv.CUresult.CUDA_SUCCESS):
                    _raise_cuda_error(error_code)
                if raw_handle == 0:
                    raise RuntimeError("CUDA returned a null allocation handle")
                probe_handles.append(drv.CUmemGenericAllocationHandle(raw_handle))
        finally:
            for probe_handle in probe_handles:
                _unwrap(drv.cuMemRelease(probe_handle))

        if candidate_supported:
            return _AllocationConfig(
                requested_handle_types=requested_handle_types,
                gpu_direct_rdma_capable=gpu_direct_rdma_capable,
                usage=usage,
                granularity=common_granularity,
            )

    raise ValueError("No physical-memory allocation property is supported by both locality domains")


def _select_default_allocation_config(size: int, device_id: int) -> _AllocationConfig:
    if size <= 0:
        raise ValueError("Allocation size must be positive")
    for requested_handle_types, gpu_direct_rdma_capable, usage in _allocation_candidates(True):
        error_code, granularity = _try_get_default_allocation_granularity(
            device_id,
            requested_handle_types,
            gpu_direct_rdma_capable,
            usage,
        )
        if error_code in _UNSUPPORTED_ALLOCATION_ERRORS:
            continue
        if error_code != int(drv.CUresult.CUDA_SUCCESS):
            _raise_cuda_error(error_code)
        if granularity <= 0:
            raise RuntimeError("CUDA returned an invalid allocation granularity")

        probe_size = round_up(_PROBE_SIZE, granularity)
        error_code, raw_handle = _try_create_default_allocation_handle(
            probe_size,
            device_id,
            requested_handle_types,
            gpu_direct_rdma_capable,
            usage,
        )
        if error_code in _UNSUPPORTED_ALLOCATION_ERRORS:
            continue
        if error_code != int(drv.CUresult.CUDA_SUCCESS):
            _raise_cuda_error(error_code)
        if raw_handle == 0:
            raise RuntimeError("CUDA returned a null allocation handle")
        _unwrap(drv.cuMemRelease(drv.CUmemGenericAllocationHandle(raw_handle)))
        return _AllocationConfig(
            requested_handle_types=requested_handle_types,
            gpu_direct_rdma_capable=gpu_direct_rdma_capable,
            usage=usage,
            granularity=granularity,
        )

    raise ValueError("Failed to create physical memory allocation property")


# Physical memory
class NativePhysMemAllocator:
    __slots__ = (
        "_context",
        "_device_id",
        "_localization_handle",
        "_size",
        "_prop",
        "_outstanding_handles",
        "_outstanding_handles_location",
        "_support_localization",
        "_location",
    )

    _context: drv.CUcontext
    _device_id: int
    _localization_handle: _LocalizationHandle | None
    _size: int
    _prop: drv.CUmemAllocationProp
    _outstanding_handles: set[int]  # allocated but not released
    _outstanding_handles_location: dict[int, Optional[int]]  # handle_int -> location
    _support_localization: bool
    _location: Optional[
        int
    ]  # locality domain location baked in at construction; None for non-localized

    def __init__(
        self,
        size: int,
        support_localization: bool = False,
        location: int | None = None,
        *,
        allocation_config: _AllocationConfig | None = None,
        context: drv.CUcontext | None = None,
        device_id: int | None = None,
        localization_handle: _LocalizationHandle | None = None,
    ) -> None:
        self._context = _get_current_context() if context is None else context
        self._device_id = get_current_device_id() if device_id is None else device_id
        self._support_localization = support_localization
        self._location = location
        self._localization_handle = localization_handle

        if self._support_localization:
            if location not in _LOCALITY_DOMAIN_LOCATIONS:
                raise ValueError(f"Invalid locality domain location: {location}")
            if _get_localization_mode() is LocalizationMode.MOCK:
                localization_supported = True
            elif self._localization_handle is not None:
                localization_supported = _supports_memory_localization(self._localization_handle)
            else:
                localization_supported = is_device_localization_supported(self._device_id)
            if not localization_supported:
                raise ValueError("Localization is not supported on this device")
            if (
                self._localization_handle is None
                and _get_localization_mode() is not LocalizationMode.MOCK
            ):
                self._localization_handle = _get_localization_handle()
            if MEM_DEBUG:
                print(
                    f"[] Localization mode: device_id={self._device_id} "
                    f"chunk_size={size} location={location}"
                )
        else:
            if location is not None:
                raise ValueError("A locality domain requires localized allocation")
            if MEM_DEBUG:
                print(f"[] device_id={self._device_id} chunk_size={size}")

        if allocation_config is None:
            if self._support_localization:
                allocation_config = _select_localized_allocation_config(
                    size,
                    self._device_id,
                    self._localization_handle,
                )
            else:
                allocation_config = _select_default_allocation_config(size, self._device_id)
        self._size = round_up(size, allocation_config.granularity)
        prop = _make_allocation_prop(self._device_id, allocation_config)
        if MEM_DEBUG:
            print(
                f"allocation prop: "
                f"requestedHandleTypes={prop.requestedHandleTypes} "
                f"gpuDirectRDMACapable={prop.allocFlags.gpuDirectRDMACapable} "
                f"usage={prop.allocFlags.usage} granularity={allocation_config.granularity} "
                f"aligned_size={self._size}"
            )
        self._prop = prop
        self._outstanding_handles = set()
        self._outstanding_handles_location = {}

    def _default_allocate(self) -> drv.CUmemGenericAllocationHandle:
        # Replace drv.cuMemCreate with the non-localized Rubin API when available.
        if MEM_DEBUG:
            print(f"_default_allocate: location={self._location} size={self._size}")
        return _unwrap(drv.cuMemCreate(self._size, self._prop, 0))

    def _localized_allocate(self) -> drv.CUmemGenericAllocationHandle:
        if MEM_DEBUG:
            print(f"_localized_allocate: location={self._location} size={self._size}")
        if _get_localization_mode() is LocalizationMode.MOCK:
            return _unwrap(drv.cuMemCreate(self._size, self._prop, 0))

        assert self._location is not None
        if self._localization_handle is None:
            raise RuntimeError("Missing locality domain memory-localization handle")
        return _create_localized_allocation_handle(
            self._localization_handle,
            self._size,
            self._prop,
            _location_to_locality_domain_id(self._location),
        )

    def allocate(self) -> drv.CUmemGenericAllocationHandle:
        with _activate_cuda_context(self._context, self._device_id):
            if self._support_localization:
                handle = self._localized_allocate()
            else:
                handle = self._default_allocate()

        int_handle = int(handle)  # pyright: ignore
        assert (int_handle not in self._outstanding_handles) and int_handle != 0
        self._outstanding_handles.add(int_handle)
        self._outstanding_handles_location[int_handle] = self._location
        if MEM_DEBUG:
            print(
                f"allocated handle={int_handle:#x} location={self._location} "
                f"num_outstanding={len(self._outstanding_handles)}"
            )
        return handle

    def release(self, handle: drv.CUmemGenericAllocationHandle) -> None:
        if handle == drv.CUmemGenericAllocationHandle(0):
            return
        int_handle = int(handle)  # pyright: ignore
        assert int_handle in self._outstanding_handles
        location = self._outstanding_handles_location[int_handle]
        if MEM_DEBUG:
            print(
                f"release: handle={int_handle:#x} location={location} "
                f"num_outstanding={len(self._outstanding_handles)}"
            )
        try:
            with _activate_cuda_context(self._context, self._device_id):
                _unwrap(drv.cuMemRelease(handle))
        except Exception:
            print(
                f"failed to release handle={int_handle:#x} "
                f"location={location} num_outstanding={len(self._outstanding_handles)}"
            )
            raise
        self._outstanding_handles_location.pop(int_handle)
        self._outstanding_handles.remove(int_handle)

    @property
    def device_id(self) -> int:
        return self._device_id

    @property
    def context(self) -> drv.CUcontext:
        return self._context

    @property
    def size(self) -> int:
        return self._size


class PhysMem(ItemHolderWithSharedPool[drv.CUmemGenericAllocationHandle]):
    __slots__ = ()


# Internal pool-location sentinels. They map to public CUDA locality-domain
# ordinals 0 and 1 through _location_to_locality_domain_id().
_LOCALITY_DOMAIN_LOC_0: int = 1
_LOCALITY_DOMAIN_LOC_1: int = 2
_LOCALITY_DOMAIN_LOCATIONS: tuple[int, ...] = (_LOCALITY_DOMAIN_LOC_0, _LOCALITY_DOMAIN_LOC_1)


class _SinglePool(PooledFactoryBase[drv.CUmemGenericAllocationHandle, PhysMem]):
    """Pooled physical memory factory for one locality domain."""

    _Holder = PhysMem

    def __init__(self, raw_alloc: NativePhysMemAllocator) -> None:
        super().__init__(lambda: raw_alloc.allocate(), lambda handle: raw_alloc.release(handle))


class PooledPhysMemAllocator:
    """Pooled physical memory allocator supporting 1 or N locality domains.

    Default construction creates a single pool (``num_locality_domains == 1``) for
    non-localized hardware.  Use :meth:`create_localized` to create an
    N-pool allocator (``num_locality_domains == N``) where each pool targets a
    specific locality domain via ``NativePhysMemAllocator(location=...)``.

    All public operations accept an optional ``locality_domain_id`` (default 0) so
    that existing single-pool call sites remain unchanged.
    """

    __slots__ = ("context", "device_id", "phys_mem_size", "_pools")
    context: drv.CUcontext
    device_id: int
    phys_mem_size: int
    _pools: list[_SinglePool]

    def __init__(self, phys_mem_size: int) -> None:
        """Create a single-pool (non-localized) allocator."""
        raw_alloc = NativePhysMemAllocator(phys_mem_size)
        self.context = raw_alloc.context
        self.device_id = raw_alloc.device_id
        self.phys_mem_size = raw_alloc.size
        self._pools = [_SinglePool(raw_alloc)]

    @classmethod
    def create_localized(cls, phys_mem_size: int) -> "PooledPhysMemAllocator":
        """Create a two-pool allocator with per-locality domain memory localization."""
        obj = cls.__new__(cls)
        context = _get_current_context()
        device_id = get_current_device_id()
        mode = _get_localization_mode()
        localization_handle = None if mode is LocalizationMode.MOCK else _get_localization_handle()
        if localization_handle is not None and not _supports_memory_localization(
            localization_handle
        ):
            raise ValueError("Localization is not supported on this device")
        allocation_config = _select_localized_allocation_config(
            phys_mem_size,
            device_id,
            localization_handle,
        )
        raw_allocs = [
            NativePhysMemAllocator(
                size=phys_mem_size,
                support_localization=True,
                location=loc,
                allocation_config=allocation_config,
                context=context,
                device_id=device_id,
                localization_handle=localization_handle,
            )
            for loc in _LOCALITY_DOMAIN_LOCATIONS
        ]
        if any(raw_alloc.size != raw_allocs[0].size for raw_alloc in raw_allocs):
            raise RuntimeError("Localized allocation sizes differ across locality domain domains")
        obj.context = context
        obj.device_id = raw_allocs[0].device_id
        obj.phys_mem_size = raw_allocs[0].size
        obj._pools = [_SinglePool(ra) for ra in raw_allocs]
        return obj

    @property
    def num_locality_domains(self) -> int:
        return len(self._pools)

    def create(self, locality_domain_id: int = 0) -> PhysMem:
        """Allocate a physical memory chunk from the specified locality domain's pool."""
        assert 0 <= locality_domain_id < len(self._pools), (
            f"locality_domain_id {locality_domain_id} out of range [0, {len(self._pools)})"
        )
        return self._pools[locality_domain_id].create()

    def clear(self) -> None:
        """Release all cached physical memory handles from all locality domain pools."""
        for pool in self._pools:
            pool.clear()


# Large fixed VA offset separating locality domain regions within a multi-locality domain VirtMem.
# The first pool's mapped bytes can never realistically reach this value.
LOCALIZATION_OFFSET: Final[int] = 1 << 40  # 1 TiB


# Virtual memory
class VirtMem:
    """Virtual memory supporting 1 or N locality domains within a single VA reservation.

    Each locality domain owns an independent region of the VA space with its own physical
    memory stack.  The number of locality domains is derived from the
    ``PooledPhysMemAllocator`` passed at construction time
    (``allocator.num_locality_domains``).

    VA layout (N locality domains)::

        locality_domain 0: [base,                              base + vm_sizes[0])
        locality_domain 1: [base + 1 * LOCALIZATION_OFFSET,    base + 1 * LOCALIZATION_OFFSET + vm_sizes[1])
        locality_domain k: [base + k * LOCALIZATION_OFFSET,    base + k * LOCALIZATION_OFFSET + vm_sizes[k])

    For a single locality domain (the default, N == 1) this simplifies to a contiguous
    region ``[base, base + vm_sizes[0])`` with no gap.

    All public operations accept an optional ``locality_domain_id`` (default 0) so that
    existing single-locality domain call sites remain unchanged.
    """

    __slots__ = (
        "_vm_sizes",
        "_allocator",
        "_address",
        "_pm_stacks",
        "_access_desc",
        "_context",
        "_device_id",
        "_reservation_size",
    )
    _vm_sizes: list[int]  # len == num_locality_domains
    _allocator: PooledPhysMemAllocator
    _address: drv.CUdeviceptr
    _pm_stacks: list[list[PhysMem]]  # len == num_locality_domains
    _access_desc: drv.CUmemAccessDesc
    _context: drv.CUcontext
    _device_id: int
    _reservation_size: int

    def __init__(
        self,
        vm_sizes: int | list[int],
        phys_mem_allocator: PooledPhysMemAllocator,
        init_num_phys_mem: int | list[int] = 0,
    ) -> None:
        num_locality_domains = phys_mem_allocator.num_locality_domains
        # Normalise scalar args to per-locality domain lists.
        if isinstance(vm_sizes, int):
            vm_sizes = [vm_sizes]
        if isinstance(init_num_phys_mem, int):
            init_num_phys_mem = [init_num_phys_mem] * num_locality_domains
        assert (
            len(vm_sizes) == num_locality_domains and len(init_num_phys_mem) == num_locality_domains
        ), (
            f"Expected {num_locality_domains} elements (num_locality_domains={num_locality_domains}), "
            f"got vm_sizes={len(vm_sizes)}, init_num_phys_mem={len(init_num_phys_mem)}"
        )

        phys_mem_size = phys_mem_allocator.phys_mem_size
        for vs in vm_sizes:
            assert vs % phys_mem_size == 0
        if num_locality_domains > 1:
            assert LOCALIZATION_OFFSET % phys_mem_size == 0
            for vs in vm_sizes:
                assert vs <= LOCALIZATION_OFFSET

        self._allocator = phys_mem_allocator
        self._context = phys_mem_allocator.context
        self._device_id = phys_mem_allocator.device_id
        self._address = drv.CUdeviceptr(0)
        self._vm_sizes = list(vm_sizes)
        self._pm_stacks = [[] for _ in range(num_locality_domains)]
        reservation_size = (
            vm_sizes[0]
            if num_locality_domains == 1
            else (num_locality_domains - 1) * LOCALIZATION_OFFSET + vm_sizes[-1]
        )
        self._reservation_size = 0

        # Reserve VA: single locality domain gets a tight reservation; multi-locality domain uses
        # LOCALIZATION_OFFSET gaps to place each region at offset k * LOCALIZATION_OFFSET.
        with _activate_cuda_context(self._context, self._device_id):
            self._address = _unwrap(drv.cuMemAddressReserve(reservation_size, 0, 0, 0))
        self._reservation_size = reservation_size
        self._access_desc = drv.CUmemAccessDesc()
        self._access_desc.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        self._access_desc.location.id = self._device_id
        self._access_desc.flags = drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE

        try:
            for uid, n in enumerate(init_num_phys_mem):
                if n:
                    self.extend(n, locality_domain_id=uid)
        except Exception:
            self._release_reservation(synchronize=False)
            raise

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_locality_domains(self) -> int:
        return len(self._pm_stacks)

    @property
    def phys_mem_size(self) -> int:
        return self._allocator.phys_mem_size

    @property
    def address(self) -> MemAddress:
        """Base address of the entire reservation (== locality_domain_address(0))."""
        return MemAddress(int(self._address))

    def locality_domain_address(self, locality_domain_id: int = 0) -> MemAddress:
        """Base virtual address for the given locality domain's region."""
        return MemAddress(int(self._address) + locality_domain_id * LOCALIZATION_OFFSET)

    def virtual_bytes(self, locality_domain_id: int = 0) -> int:
        """VA region size for the given locality domain."""
        return self._vm_sizes[locality_domain_id]

    def num_phys_mem(self, locality_domain_id: int = 0) -> int:
        return len(self._pm_stacks[locality_domain_id])

    def mapped_bytes(self, locality_domain_id: int = 0) -> int:
        return self.phys_mem_size * self.num_phys_mem(locality_domain_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def destroy(self) -> None:
        if self._reservation_size == 0:
            return
        self._release_reservation(synchronize=True)

    def _release_reservation(self, synchronize: bool) -> None:
        with _activate_cuda_context(self._context, self._device_id):
            if synchronize:
                _unwrap(drv.cuCtxSynchronize())
            for uid in range(self.num_locality_domains):
                while self._pm_stacks[uid]:
                    self._pop(uid).close()
            _unwrap(drv.cuMemAddressFree(self._address, self._reservation_size))
        self._address = drv.CUdeviceptr(0)
        self._vm_sizes = [0] * self.num_locality_domains
        self._reservation_size = 0

    def __del__(self) -> None:
        self.destroy()

    # ------------------------------------------------------------------
    # Core operations (all take locality_domain_id, defaulting to 0)
    # ------------------------------------------------------------------

    def _push(self, phy_mem: PhysMem, locality_domain_id: int = 0) -> None:
        phys_mem_size = self.phys_mem_size
        pm_stack = self._pm_stacks[locality_domain_id]
        assert phys_mem_size * (len(pm_stack) + 1) <= self._vm_sizes[locality_domain_id]
        vm_ptr = drv.CUdeviceptr(
            self.locality_domain_address(locality_domain_id) + phys_mem_size * len(pm_stack)
        )
        with _activate_cuda_context(self._context, self._device_id):
            try:
                _unwrap(drv.cuMemMap(vm_ptr, phys_mem_size, 0, phy_mem.handle, 0))
            except Exception:
                phy_mem.close()
                raise
            try:
                _unwrap(drv.cuMemSetAccess(vm_ptr, phys_mem_size, (self._access_desc,), 1))
            except Exception:
                try:
                    _unwrap(drv.cuMemUnmap(vm_ptr, phys_mem_size))
                except Exception:
                    # Keep ownership so the outer transaction can retry unmapping.
                    pm_stack.append(phy_mem)
                else:
                    phy_mem.close()
                raise
            pm_stack.append(phy_mem)

    def _pop(self, locality_domain_id: int = 0) -> PhysMem:
        pm_stack = self._pm_stacks[locality_domain_id]
        assert pm_stack
        phys_mem_size = self.phys_mem_size
        vm_ptr = drv.CUdeviceptr(
            self.locality_domain_address(locality_domain_id) + phys_mem_size * (len(pm_stack) - 1)
        )
        with _activate_cuda_context(self._context, self._device_id):
            _unwrap(drv.cuMemUnmap(vm_ptr, phys_mem_size))
        return pm_stack.pop()

    def extend(self, num_phys_mem: int, locality_domain_id: int = 0) -> None:
        assert 0 <= locality_domain_id < self.num_locality_domains, (
            f"locality_domain_id {locality_domain_id} out of range [0, {self.num_locality_domains})"
        )
        pm_stack = self._pm_stacks[locality_domain_id]
        old_num = len(pm_stack)
        try:
            for _ in range(num_phys_mem):
                self._push(self._allocator.create(locality_domain_id), locality_domain_id)
        except Exception:
            # Rollback to make realloc behave like normal realloc on OOM.
            while len(self._pm_stacks[locality_domain_id]) > old_num:
                self._pop(locality_domain_id).close()
            raise

    def shrink(self, num_phys_mem: int, locality_domain_id: int = 0) -> None:
        assert 0 <= locality_domain_id < self.num_locality_domains, (
            f"locality_domain_id {locality_domain_id} out of range [0, {self.num_locality_domains})"
        )
        with _activate_cuda_context(self._context, self._device_id):
            _unwrap(drv.cuCtxSynchronize())
        for _ in range(num_phys_mem):
            self._pop(locality_domain_id).close()

    # Different from normal realloc, this function never changes the pointer.
    def realloc(self, num_bytes: int, locality_domain_id: int = 0) -> None:
        required_num_phys_mem = div_up(num_bytes, self.phys_mem_size)
        current = self.num_phys_mem(locality_domain_id)
        if required_num_phys_mem > current:
            self.extend(required_num_phys_mem - current, locality_domain_id)
        elif required_num_phys_mem < current:
            self.shrink(current - required_num_phys_mem, locality_domain_id)
