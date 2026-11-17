/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/executor/cache_transmission/nixl_utils/bounce/BounceArena.h"

#include "tensorrt_llm/batch_manager/cacheTransBuffer.h" // kv_cache_manager::FabricMemory (full def)
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"

#include <cuda.h>

#include <cstring>

namespace tensorrt_llm::executor::kv_cache::bounce
{

BounceArena::BounceArena(std::size_t bytes, int deviceId, bool allowFabric)
    : mDeviceId(deviceId)
    , mBytes(bytes)
{
    TLLM_CHECK_WITH_INFO(bytes > 0, "BounceArena: bytes must be > 0");
    TLLM_CUDA_CHECK(cudaSetDevice(mDeviceId));

    // On MNNVL parts (GH200/GB200) the arena (RDMA src/dst, NIXL-registered) must be fabric memory
    // to be reachable over the NVLink fabric + GPUDirect-RDMA capable. Fabric memory is only
    // compiled in on those aarch64 parts; on x86 the fallback is always cudaMalloc, so
    // disable_fabric_memory (allowFabric=false) has no effect there. It also falls back when the
    // runtime reports no fabric support or when fabric is force-disabled.
    if (allowFabric && tensorrt_llm::batch_manager::kv_cache_manager::FabricMemory::supportFabricMemory())
    {
        mFabric = std::make_unique<tensorrt_llm::batch_manager::kv_cache_manager::FabricMemory>(bytes);
        mBase = mFabric->getPtr();
        TLLM_LOG_DEBUG("BounceArena: %zuB backed by fabric memory", bytes);
    }
    else
    {
        TLLM_CUDA_CHECK(cudaMalloc(&mBase, bytes));
    }
}

BounceArena::~BounceArena()
{
    if (!mFabric && mBase != nullptr)
    {
        // Select the owning device before freeing (multi-GPU: cudaFree otherwise targets the thread's
        // current device). Fabric-backed arena is freed by ~FabricMemory (mFabric). A dtor can't
        // throw, so use the project's warn-only cleanup check rather than discarding the result.
        TLLM_CUDA_CHECK_WARN(cudaSetDevice(mDeviceId));
        TLLM_CUDA_CHECK_WARN(cudaFree(mBase));
    }
}

namespace
{
std::string cuErrorName(CUresult r)
{
    char const* name = nullptr;
    (void) cuGetErrorName(r, &name);
    return name != nullptr ? name : "unknown CUresult";
}

/// Makes `deviceId` current for the scope and restores the caller's device afterwards (these calls run
/// on the caller's thread, e.g. loadRemoteAgent's).
class ScopedDevice
{
public:
    explicit ScopedDevice(int deviceId)
    {
        mRestore = cudaGetDevice(&mPrevious) == cudaSuccess;
        mOk = cudaSetDevice(deviceId) == cudaSuccess;
        if (!mOk)
        {
            (void) cudaGetLastError();
        }
    }

    ~ScopedDevice()
    {
        if (mRestore)
        {
            (void) cudaSetDevice(mPrevious);
        }
    }

    ScopedDevice(ScopedDevice const&) = delete;
    ScopedDevice& operator=(ScopedDevice const&) = delete;

    [[nodiscard]] bool ok() const noexcept
    {
        return mOk;
    }

private:
    int mPrevious{0};
    bool mRestore{false};
    bool mOk{false};
};
} // namespace

bool BounceArena::exportFabric(std::uint64_t& allocBase, std::uint64_t& allocBytes, std::string& handle) const
{
    if (mFabric == nullptr)
    {
        return false;
    }
    ScopedDevice const device(mDeviceId);
    if (!device.ok())
    {
        TLLM_LOG_WARNING("BounceArena: cudaSetDevice(%d) failed: the arena is not exported", mDeviceId);
        return false;
    }
    CUdeviceptr base = 0;
    std::size_t size = 0;
    CUresult r = cuMemGetAddressRange(&base, &size, reinterpret_cast<CUdeviceptr>(mBase));
    if (r != CUDA_SUCCESS || base != reinterpret_cast<CUdeviceptr>(mBase) || size < mBytes)
    {
        TLLM_LOG_WARNING("BounceArena: cuMemGetAddressRange failed (%s): the arena is not exported for push gather",
            cuErrorName(r).c_str());
        return false;
    }
    CUmemGenericAllocationHandle h{};
    r = cuMemRetainAllocationHandle(&h, mBase);
    if (r != CUDA_SUCCESS)
    {
        TLLM_LOG_WARNING(
            "BounceArena: cuMemRetainAllocationHandle failed (%s): the arena is not exported", cuErrorName(r).c_str());
        return false;
    }
    CUmemFabricHandle fabricHandle{};
    r = cuMemExportToShareableHandle(&fabricHandle, h, CU_MEM_HANDLE_TYPE_FABRIC, 0);
    (void) cuMemRelease(h);
    if (r != CUDA_SUCCESS)
    {
        TLLM_LOG_WARNING("BounceArena: cuMemExportToShareableHandle(FABRIC) failed (%s): the arena is not exported",
            cuErrorName(r).c_str());
        return false;
    }
    allocBase = static_cast<std::uint64_t>(base);
    allocBytes = static_cast<std::uint64_t>(size);
    handle.assign(reinterpret_cast<char const*>(&fabricHandle), sizeof(fabricHandle));
    return true;
}

std::shared_ptr<PeerArenaMapping> PeerArenaMapping::map(
    std::uint64_t remoteBase, std::uint64_t allocBytes, std::string const& handle, int deviceId, std::string& error)
{
    if (handle.size() != sizeof(CUmemFabricHandle) || allocBytes == 0)
    {
        error = "the peer exported no fabric handle";
        return nullptr;
    }
    ScopedDevice const device(deviceId);
    if (!device.ok())
    {
        error = "cudaSetDevice";
        return nullptr;
    }
    CUmemFabricHandle fabricHandle{};
    std::memcpy(&fabricHandle, handle.data(), sizeof(fabricHandle));
    CUmemGenericAllocationHandle h{};
    CUresult r = cuMemImportFromShareableHandle(&h, &fabricHandle, CU_MEM_HANDLE_TYPE_FABRIC);
    if (r != CUDA_SUCCESS)
    {
        error = "cuMemImportFromShareableHandle(FABRIC): " + cuErrorName(r);
        return nullptr;
    }
    CUmemAllocationProp prop{};
    std::size_t granularity = 0;
    r = cuMemGetAllocationPropertiesFromHandle(&prop, h);
    if (r == CUDA_SUCCESS)
    {
        r = cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    }
    if (r != CUDA_SUCCESS || granularity == 0 || allocBytes % granularity != 0)
    {
        (void) cuMemRelease(h);
        error = "allocation granularity: " + cuErrorName(r);
        return nullptr;
    }
    std::shared_ptr<PeerArenaMapping> m(new PeerArenaMapping());
    CUdeviceptr va = 0;
    r = cuMemAddressReserve(&va, allocBytes, granularity, 0, 0);
    if (r != CUDA_SUCCESS)
    {
        (void) cuMemRelease(h);
        error = "cuMemAddressReserve: " + cuErrorName(r);
        return nullptr;
    }
    m->mLocalBase = static_cast<std::uint64_t>(va);
    m->mBytes = allocBytes;
    m->mRemoteBase = remoteBase;
    r = cuMemMap(va, allocBytes, 0, h, 0);
    (void) cuMemRelease(h); // the mapping holds the reference
    if (r != CUDA_SUCCESS)
    {
        error = "cuMemMap: " + cuErrorName(r);
        return nullptr; // the destructor frees the reservation
    }
    m->mMapped = true;
    CUmemAccessDesc access{};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = deviceId;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    r = cuMemSetAccess(va, allocBytes, &access, 1);
    if (r != CUDA_SUCCESS)
    {
        error = "cuMemSetAccess: " + cuErrorName(r);
        return nullptr;
    }
    return m;
}

PeerArenaMapping::~PeerArenaMapping()
{
    if (mMapped)
    {
        (void) cuMemUnmap(static_cast<CUdeviceptr>(mLocalBase), mBytes);
    }
    if (mLocalBase != 0)
    {
        (void) cuMemAddressFree(static_cast<CUdeviceptr>(mLocalBase), mBytes);
    }
}

} // namespace tensorrt_llm::executor::kv_cache::bounce
