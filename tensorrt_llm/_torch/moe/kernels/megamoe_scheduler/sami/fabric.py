# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA Fabric allocation descriptors.

The live-weight arena allocates FABRIC-exportable memory and maps it into every
EP process; these are the allocation properties and access descriptor it
uses. This module is imported lazily by the arena to avoid eager native
initialization at module import time.
"""

from __future__ import annotations

from cuda.bindings import driver as cuda

FABRIC = cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC


def _read_write_access(device: int) -> cuda.CUmemAccessDesc:
    descriptor = cuda.CUmemAccessDesc()
    descriptor.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    descriptor.location.id = int(device)
    descriptor.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    return descriptor


def _allocation_properties(device: int) -> cuda.CUmemAllocationProp:
    properties = cuda.CUmemAllocationProp()
    properties.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    properties.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    properties.location.id = int(device)
    properties.requestedHandleTypes = FABRIC
    return properties


__all__ = ["FABRIC"]
