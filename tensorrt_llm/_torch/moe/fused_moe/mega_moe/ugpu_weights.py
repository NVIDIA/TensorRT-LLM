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
"""Load-time placement for MegaMoE's single-grid die-aware kernels.

This module accepts TensorRT-LLM's natural packed weight layout, before the custom op's
transpose view. It never changes expert scales or activation quantization.

VMM memory is outside PyTorch's caching allocator. The MegaMoE backend keeps
the returned owner for the lifetime of all parameter aliases and CUDA graphs,
i.e. until process exit. A tensor finalizer cannot prove that no graph still
uses the memory, so this module never unmaps memory from __del__.
"""

from __future__ import annotations

import ctypes
import warnings
from dataclasses import dataclass
from typing import Literal, Sequence

import torch

__all__ = [
    "LocalizedMegaMoEWeights",
    "PackedWeightLayout",
    "localize_megamoe_weights",
    "plan_packed_weight_layout",
    "reorder_packed_weight",
]

_WEIGHT_NAMES = ("fc1_weight", "fc1_weight_sf", "fc2_weight", "fc2_weight_sf")
_LOCALITY_DOMAIN_ATTRIBUTE = 22


@dataclass(frozen=True)
class PackedWeightLayout:
    """Byte layout and two VMM mappings for one packed weight or scale tensor."""

    mode: Literal["expert_half", "n_split"]
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    payload_bytes: int
    reserved_bytes: int
    die1_offset: int
    tensor_offset: int
    boundary_spill_bytes: int
    granularity: int

    @property
    def spans(self) -> tuple[tuple[int, int, int], ...]:
        """Return (die ordinal, VA offset, byte length) for both mappings."""
        return (
            (0, 0, self.die1_offset),
            (1, self.die1_offset, self.reserved_bytes - self.die1_offset),
        )


def plan_packed_weight_layout(
    shape: Sequence[int],
    *,
    mode: Literal["expert_half", "n_split"],
    granularity: int,
) -> PackedWeightLayout:
    """Plan contiguous byte tensors: weights [E, N, K/2], or SF [E, flat].

    n_split moves each expert's first N/2 rows before every expert's remaining
    rows. Its front pad aligns the die boundary exactly; expert_half retains
    the original layout and reports the last expert's possible page spill.
    """
    shape = tuple(shape)
    if len(shape) not in (2, 3) or any(type(value) is not int or value <= 0 for value in shape):
        raise ValueError("Packed weights need positive integer [E,N,K/2] or [E,flat] dimensions")
    if type(granularity) is not int or granularity < 128 or granularity & (granularity - 1):
        raise ValueError(
            "Localized allocation granularity must be a power of two of at least 128 bytes"
        )
    payload = 1
    for size in shape:
        payload *= size
    if mode == "expert_half":
        if shape[0] % 2:
            raise ValueError("expert_half requires an even local expert count")
        exact_boundary = payload // 2
        boundary = exact_boundary // granularity * granularity
        reserved = (payload + granularity - 1) // granularity * granularity
        if boundary == 0 or boundary == reserved:
            raise ValueError("expert_half tensor is too small for two localized VMM mappings")
        return PackedWeightLayout(
            mode,
            shape,
            shape,
            payload,
            reserved,
            boundary,
            0,
            exact_boundary - boundary,
            granularity,
        )
    if mode != "n_split":
        raise ValueError(f"Unknown MegaMoE weight placement mode {mode!r}")
    if shape[1] % (256 if len(shape) == 3 else 2):
        raise ValueError("n_split needs weight rows divisible by 256 and an even SF width")
    half_bytes = payload // 2
    span = (half_bytes + granularity - 1) // granularity * granularity
    pad = span - half_bytes
    if pad % 128:
        raise ValueError("n_split tensor base must remain 128-byte aligned for TMA")
    target_shape = (shape[0] * 2, shape[1] // 2, *shape[2:])
    return PackedWeightLayout(
        mode, shape, target_shape, payload, 2 * span, span, pad, 0, granularity
    )


def reorder_packed_weight(
    source: torch.Tensor, destination: torch.Tensor, plan: PackedWeightLayout
) -> None:
    """Copy final packed bytes into a planned tensor; runs on CPU or CUDA.

    No unpacking, re-quantization, SF swizzle, or gate/up interleave is applied.
    Both arguments must be contiguous uint8 tensors of the planned shapes.
    """
    for name, tensor, shape in (
        ("source", source, plan.source_shape),
        ("destination", destination, plan.target_shape),
    ):
        if (
            tensor.dtype != torch.uint8
            or not tensor.is_contiguous()
            or tuple(tensor.shape) != shape
        ):
            raise ValueError(f"{name} must be contiguous uint8 with shape {shape}")
    if source.device != destination.device:
        raise ValueError("Packed placement must remain on the same device")
    if source.untyped_storage().data_ptr() == destination.untyped_storage().data_ptr():
        raise ValueError("Packed layout conversion requires a distinct destination allocation")
    if plan.mode == "expert_half":
        destination.copy_(source)
        return
    experts, rows = plan.source_shape[:2]
    halves = destination.view(2, experts, rows // 2, *plan.source_shape[2:])
    halves[0].copy_(source[:, : rows // 2])
    halves[1].copy_(source[:, rows // 2 :])


class _RawDeviceBytes:
    def __init__(self, pointer: int, size: int):
        self.__cuda_array_interface__ = {
            "shape": (size,),
            "typestr": "|u1",
            "data": (pointer, False),
            "version": 2,
            "strides": None,
        }


class _CudaVmm:
    """Small CUDA driver boundary; all allocations are explicit and testable."""

    def __init__(self, device: int):
        import cuda.bindings.driver as cuda

        if not 0 <= device <= 255:
            raise ValueError("CUDA locality-domain allocation requires an 8-bit device ordinal")
        self.cuda = cuda
        self.device = device
        self.libcuda = ctypes.CDLL("libcuda.so.1")
        self.libcuda.cuPointerGetAttribute.restype = ctypes.c_int
        self.libcuda.cuPointerGetAttribute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_ulonglong,
        ]

    @staticmethod
    def _check(result, operation):
        status, *values = result
        code = int(getattr(status, "value", status))
        if code:
            raise RuntimeError(f"{operation} failed with CUDA driver status {code}")
        return values[0] if len(values) == 1 else tuple(values)

    def _properties(self, die):
        cuda = self.cuda
        prop = cuda.CUmemAllocationProp()
        prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.requestedHandleTypes = cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
        location_type = getattr(
            cuda.CUmemLocationType, "CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN", None
        )
        prop.location.type = (
            location_type if location_type is not None else cuda.CUmemLocationType(6)
        )
        prop.location.id = self.device | (die << 8)
        return prop

    def granularity(self):
        cuda = self.cuda
        flag = cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        counts = [
            int(
                self._check(
                    cuda.cuMemGetAllocationGranularity(self._properties(d), flag),
                    "cuMemGetAllocationGranularity",
                )
            )
            for d in range(2)
        ]
        if counts[0] != counts[1]:
            raise RuntimeError("The two locality domains report different VMM granularities")
        return counts[0]

    def reserve(self, size, granularity):
        return int(
            self._check(
                self.cuda.cuMemAddressReserve(size, granularity, self.cuda.CUdeviceptr(0), 0),
                "cuMemAddressReserve",
            )
        )

    def create(self, size, die):
        return self._check(self.cuda.cuMemCreate(size, self._properties(die), 0), "cuMemCreate")

    def map(self, address, size, handle):
        self._check(
            self.cuda.cuMemMap(self.cuda.CUdeviceptr(address), size, 0, handle, 0), "cuMemMap"
        )

    def allow_access(self, address, size):
        cuda = self.cuda
        access = cuda.CUmemAccessDesc()
        access.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = self.device
        access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self._check(
            cuda.cuMemSetAccess(cuda.CUdeviceptr(address), size, [access], 1), "cuMemSetAccess"
        )

    def alias(self, address, size):
        tensor = torch.as_tensor(
            _RawDeviceBytes(address, size), device=torch.device("cuda", self.device)
        )
        if tensor.data_ptr() != address or tensor.numel() != size:
            raise RuntimeError("PyTorch did not alias the localized virtual address range")
        return tensor

    def locality(self, address):
        ordinal = ctypes.c_int(-1)
        status = self.libcuda.cuPointerGetAttribute(
            ctypes.byref(ordinal), _LOCALITY_DOMAIN_ATTRIBUTE, ctypes.c_ulonglong(address)
        )
        if status:
            raise RuntimeError(
                f"Locality-domain pointer query failed with CUDA driver status {status}"
            )
        return ordinal.value

    def synchronize(self):
        torch.cuda.synchronize(self.device)

    def unmap(self, address, size):
        self._check(self.cuda.cuMemUnmap(self.cuda.CUdeviceptr(address), size), "cuMemUnmap")

    def release(self, handle):
        self._check(self.cuda.cuMemRelease(handle), "cuMemRelease")

    def free_address(self, address, size):
        self._check(
            self.cuda.cuMemAddressFree(self.cuda.CUdeviceptr(address), size), "cuMemAddressFree"
        )


class _PlacedAllocation:
    def __init__(self, name, plan, driver):
        self.name, self.plan, self.driver = name, plan, driver
        self.address = 0
        self.handles = []
        self.mappings = []
        self.tensor = None

    def allocate(self, source):
        plan, driver = self.plan, self.driver
        self.address = driver.reserve(plan.reserved_bytes, plan.granularity)
        for die, offset, size in plan.spans:
            handle = driver.create(size, die)
            self.handles.append(handle)
            driver.map(self.address + offset, size, handle)
            self.mappings.append((offset, size))
        driver.allow_access(self.address, plan.reserved_bytes)
        # Verify every mapped granule, including front/tail padding. A driver
        # lacking the query is a failed placement proof, never a silent pass.
        for die, offset, size in plan.spans:
            for page in range(offset, offset + size, plan.granularity):
                actual = driver.locality(self.address + page)
                if actual != die:
                    raise RuntimeError(
                        f"{self.name}: expected die {die}, got {actual} at page {page}"
                    )
        storage = driver.alias(self.address, plan.reserved_bytes)
        self.tensor = storage[plan.tensor_offset : plan.tensor_offset + plan.payload_bytes].view(
            plan.target_shape
        )
        reorder_packed_weight(source, self.tensor, plan)

    def close(self):
        """Release every mapping, handle and address range; return driver errors."""
        self.tensor = None
        errors = []
        for operation, args in (
            *((self.driver.unmap, (self.address + offset, size)) for offset, size in self.mappings),
            *((self.driver.release, (handle,)) for handle in self.handles),
            *(
                ((self.driver.free_address, (self.address, self.plan.reserved_bytes)),)
                if self.address
                else ()
            ),
        ):
            try:
                operation(*args)
            except RuntimeError as error:
                errors.append(str(error))
        self.mappings, self.handles, self.address = [], [], 0
        return errors


class LocalizedMegaMoEWeights:
    """Own four localized tensors for the lifetime of the MegaMoE module.

    The named tensor properties are borrowed views; they and all their aliases
    become invalid after close(). Dropping a temporary tensor is not proof that
    a CUDA graph stopped using its address, so an unclosed owner only warns.
    """

    def __init__(self, mode, driver):
        self.mode = mode
        self._driver = driver
        self._records = []
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def _tensor(self, index):
        if self._closed or len(self._records) != 4:
            raise RuntimeError("Localized MegaMoE weights are not live")
        return self._records[index].tensor

    fc1_weight = property(lambda self: self._tensor(0))
    fc1_weight_sf = property(lambda self: self._tensor(1))
    fc2_weight = property(lambda self: self._tensor(2))
    fc2_weight_sf = property(lambda self: self._tensor(3))

    def close(self) -> None:
        """Synchronize, then release all mappings (best effort).

        Never call while any graph can replay or another host thread can submit
        work using these tensors.
        """
        if self._closed:
            return
        self._driver.synchronize()
        errors = [error for record in self._records for error in record.close()]
        self._closed = True
        if errors:
            raise RuntimeError("Localized MegaMoE teardown failed: " + "; ".join(errors))

    def __del__(self):
        if not self._closed and self._records:
            warnings.warn(
                "Localized MegaMoE owner discarded without close(); retaining its VMM "
                "memory because graph/alias lifetime is unknown",
                ResourceWarning,
            )


def localize_megamoe_weights(
    fc1_weight: torch.Tensor,
    fc1_weight_sf: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_weight_sf: torch.Tensor,
    *,
    mode: Literal["expert_half", "n_split"],
) -> LocalizedMegaMoEWeights:
    """Place four already-packed uint8 tensors once, during model loading.

    Weights have shapes [E, 2I, H/2] and [E, H, I/2]; their blocked SF tensors
    have shapes [E, flat]. All must be contiguous and on one CUDA device.
    Return natural layouts for expert_half or [2E, N/2, K/2]/[2E, flat/2]
    for n_split. The existing custom-op transpose then supplies K-major views.

    The caller replaces the four original parameters after success; this
    function retains neither the sources nor a second full packed weight bank.
    Alphas, norm constants and global activation scales are not inputs.
    """
    tensors = (fc1_weight, fc1_weight_sf, fc2_weight, fc2_weight_sf)
    device = fc1_weight.device
    for name, tensor, rank in zip(_WEIGHT_NAMES, tensors, (3, 2, 3, 2)):
        if tensor.dtype != torch.uint8 or tensor.ndim != rank or not tensor.is_contiguous():
            raise ValueError(f"{name} must be a contiguous rank-{rank} packed uint8 tensor")
        if not tensor.is_cuda or tensor.device != device:
            raise ValueError("All MegaMoE packed tensors must be on the same CUDA device")
        if tensor.shape[0] != fc1_weight.shape[0]:
            raise ValueError("All MegaMoE packed tensors must have the same local expert count")
    if (
        fc1_weight.shape[1] != 4 * fc2_weight.shape[2]
        or fc2_weight.shape[1] != 2 * fc1_weight.shape[2]
    ):
        raise ValueError("FC1/FC2 packed hidden/intermediate dimensions disagree")
    hidden = fc2_weight.shape[1]
    intermediate = fc1_weight.shape[1] // 2
    expected_sf = (
        ((2 * intermediate + 127) // 128 * 128) * ((hidden + 63) // 64 * 4),
        ((hidden + 127) // 128 * 128) * ((intermediate + 63) // 64 * 4),
    )
    if (fc1_weight_sf.shape[1], fc2_weight_sf.shape[1]) != expected_sf:
        raise ValueError("FC1/FC2 blocked SF widths do not match the packed weight geometry")
    with torch.cuda.device(device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Localize MegaMoE weights during loading, before CUDA graph capture")
        return _place_weights(tensors, mode=mode, driver=_CudaVmm(device.index))


def _place_weights(tensors, *, mode, driver):
    granularity = driver.granularity()
    plans = tuple(
        plan_packed_weight_layout(tuple(tensor.shape), mode=mode, granularity=granularity)
        for tensor in tensors
    )
    owner = LocalizedMegaMoEWeights(mode, driver)
    succeeded = False
    try:
        for name, source, plan in zip(_WEIGHT_NAMES, tensors, plans):
            record = _PlacedAllocation(name, plan, driver)
            owner._records.append(record)
            record.allocate(source)
        succeeded = True
        return owner
    finally:
        if not succeeded:
            owner.close()
