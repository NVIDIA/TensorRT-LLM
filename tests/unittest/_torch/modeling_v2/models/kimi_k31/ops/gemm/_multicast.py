# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A one-device CUDA multicast buffer: a multicast address whose ``multimem`` stores land in one
local allocation, read back through its unicast mapping. Lets a single-GPU test drive a kernel's
one-shot (multicast) push path."""

from __future__ import annotations

import torch
from cuda.bindings import driver as cu


def _ok(result):
    err, *rest = result if isinstance(result, tuple) else (result,)
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {err}")
    return rest[0] if len(rest) == 1 else rest


class _Raw:
    def __init__(self, ptr: int, nbytes: int, device: int) -> None:
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
            "strides": None,
        }
        self.device = device


class MulticastBuffer:
    """``nbytes`` (rounded up to the multicast granularity) of device memory with a unicast view
    ``uc`` (uint8 tensor) and a multicast address ``mc`` (int)."""

    def __init__(self, nbytes: int, device: int | None = None) -> None:
        device = torch.cuda.current_device() if device is None else device
        torch.cuda.init()
        _ok(cu.cuInit(0))
        dev = _ok(cu.cuDeviceGet(device))
        attr = cu.CUdevice_attribute
        if not _ok(cu.cuDeviceGetAttribute(attr.CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, dev)):
            raise RuntimeError("the device does not support multicast objects")
        types = cu.CUmemAllocationHandleType
        # Fabric handles where the driver offers them (as the engine's multicast buffers), else
        # POSIX file descriptors: the first that yields a multicast object.
        candidates = [types.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR]
        if _ok(cu.cuDeviceGetAttribute(attr.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, dev)):
            candidates.insert(0, types.CU_MEM_HANDLE_TYPE_FABRIC)
        # A team of this device alone, else this device and a peer whose copy is never read (the
        # driver may want two members).
        teams = [(device,)]
        if torch.cuda.device_count() > 1:
            teams.append((device, (device + 1) % torch.cuda.device_count()))
        failures = []
        for team in teams:
            for handle in candidates:
                try:
                    self._create(nbytes, team, handle)
                    return
                except RuntimeError as error:
                    failures.append(f"team {team} {handle}: {error}")
        raise RuntimeError("no multicast object: " + "; ".join(failures))

    def _create(self, nbytes: int, team: tuple[int, ...], handle) -> None:
        device = team[0]

        def alloc_prop(index: int):
            prop = cu.CUmemAllocationProp()
            prop.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            prop.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            prop.location.id = index
            prop.requestedHandleTypes = handle
            return prop

        prop = alloc_prop(device)
        alloc_gran = _ok(
            cu.cuMemGetAllocationGranularity(
                prop, cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
            )
        )
        mc_prop = cu.CUmulticastObjectProp()
        mc_prop.numDevices = len(team)
        # A bit mask (unsigned long long), not the enum: pass the enum's integer value.
        mc_prop.handleTypes = int(getattr(handle, "value", handle))
        mc_prop.flags = 0
        size = -(-nbytes // alloc_gran) * alloc_gran
        mc_prop.size = size
        gran = _ok(
            cu.cuMulticastGetGranularity(
                mc_prop, cu.CUmulticastGranularity_flags.CU_MULTICAST_GRANULARITY_MINIMUM
            )
        )
        size = -(-size // gran) * gran
        mc_prop.size = size
        mc_handle = _ok(cu.cuMulticastCreate(mc_prop))
        mems = []
        try:
            for index in team:
                _ok(cu.cuMulticastAddDevice(mc_handle, _ok(cu.cuDeviceGet(index))))
            for index in team:
                mems.append(_ok(cu.cuMemCreate(size, alloc_prop(index), 0)))
                _ok(cu.cuMulticastBindMem(mc_handle, 0, mems[-1], 0, size, 0))
        except RuntimeError:
            for mem in mems:
                cu.cuMemRelease(mem)
            cu.cuMemRelease(mc_handle)
            raise
        mem = mems[0]
        self._others, self._team = mems[1:], team
        access = cu.CUmemAccessDesc()
        access.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = device
        access.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        self._mc_handle, self._mem, self._ptrs = mc_handle, mem, []
        for h in (mem, mc_handle):
            ptr = _ok(cu.cuMemAddressReserve(size, max(gran, alloc_gran), 0, 0))
            _ok(cu.cuMemMap(ptr, size, 0, h, 0))
            _ok(cu.cuMemSetAccess(ptr, size, [access], 1))
            self._ptrs.append(ptr)
        self.size = size
        self.mc = int(self._ptrs[1])
        self.uc = torch.as_tensor(_Raw(int(self._ptrs[0]), size, device), device=f"cuda:{device}")

    def close(self) -> None:
        torch.cuda.synchronize()
        for ptr in self._ptrs:
            _ok(cu.cuMemUnmap(ptr, self.size))
            _ok(cu.cuMemAddressFree(ptr, self.size))
        for index in self._team:
            _ok(cu.cuMulticastUnbind(self._mc_handle, _ok(cu.cuDeviceGet(index)), 0, self.size))
        for mem in (self._mem, *self._others):
            _ok(cu.cuMemRelease(mem))
        _ok(cu.cuMemRelease(self._mc_handle))
