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
"""Which remote-level transports this machine can run, for the remote-level tests."""

import functools


@functools.cache
def gpu_count() -> int:
    try:
        from cuda.bindings import driver as drv

        if drv.cuInit(0)[0] != drv.CUresult.CUDA_SUCCESS:
            return 0
        err, count = drv.cuDeviceGetCount()
        return count if err == drv.CUresult.CUDA_SUCCESS else 0
    except ImportError:
        return 0


@functools.cache
def nixl_available() -> bool:
    try:
        from tensorrt_llm.tensorrt_llm_transfer_agent_binding import NixlTransferAgent  # noqa: F401
    except ImportError:
        return False
    return True


@functools.cache
def fabric_available() -> bool:
    """Whether device 0 can export a host-NUMA fabric allocation, as a fabric donor does (driver, Grace, IMEX)."""
    if gpu_count() == 0:
        return False
    from cuda.bindings import driver as drv

    from tensorrt_llm._torch.pyexecutor.kv_cache.host_memory import host_numa_node_of
    from tensorrt_llm._torch.pyexecutor.kv_cache.remote_donor import FabricDonorTransport

    ok = drv.CUresult.CUDA_SUCCESS
    err, device = drv.cuDeviceGet(0)
    attr = drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED
    if err != ok or drv.cuDeviceGetAttribute(attr, device)[1] == 0:
        return False
    err, ctx = drv.cuDevicePrimaryCtxRetain(device)
    if err != ok:
        return False
    drv.cuCtxPushCurrent(ctx)
    try:
        prop = FabricDonorTransport._fabric_prop(drv, host_numa_node_of(drv, device))
        err, gran = drv.cuMemGetAllocationGranularity(
            prop, drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        )
        if err != ok:
            return False
        err, handle = drv.cuMemCreate(gran, prop, 0)
        if err != ok:
            return False
        try:
            fabric = drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC
            return drv.cuMemExportToShareableHandle(handle, fabric, 0)[0] == ok
        finally:
            drv.cuMemRelease(handle)
    finally:
        drv.cuCtxPopCurrent()
        drv.cuDevicePrimaryCtxRelease(device)


def available_transports() -> list:
    """The remote-level transports a test can run here: none without a GPU."""
    if gpu_count() == 0:
        return []
    return [t for t, ok in (("nixl", nixl_available), ("fabric", fabric_available)) if ok()]
