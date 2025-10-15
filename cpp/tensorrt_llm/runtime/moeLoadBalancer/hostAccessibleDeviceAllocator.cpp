/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string.h>

#include <cstddef>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <numaif.h>
#include <sys/mman.h>

#include "gdrwrap.h"
#include "hostAccessibleDeviceAllocator.h"
#include "topologyDetector.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"

namespace tensorrt_llm::runtime
{

class NumaHugePagePoolAllocator
{
public:
    NumaHugePagePoolAllocator(NumaHugePagePoolAllocator const&) = delete;
    void operator=(NumaHugePagePoolAllocator const&) = delete;

    static NumaHugePagePoolAllocator& getInstance();

    void* allocate(size_t memorySize);

    void free(void* ptr);

private:
    static constexpr size_t kHugePageSize = 512LL * 1024 * 1024;
    static constexpr size_t kAlignSize = 1024;
    static constexpr size_t kReservedVirtualMemorySize = 256LL * 1024 * 1024 * 1024;
    NumaHugePagePoolAllocator() = default;

    void maybeInit();
    void shutdown();

    uint8_t* mMmapBasePtr = nullptr; // allocated memory address range, not aligned.
    uint8_t* mBasePtr = nullptr;     // aligned memory address range.
    size_t mAllocatedSize = 0;       // aligned to kAlignSize
    size_t mMappedSize = 0;          // aligned to kHugePageSize

    int mDevId = -1;
    int mGpuMemNumaId = -1;

    std::mutex mMutex{};
    bool mIsInited = false;
};

NumaHugePagePoolAllocator& NumaHugePagePoolAllocator::getInstance()
{
    static NumaHugePagePoolAllocator instance;
    instance.maybeInit();
    return instance;
}

static void allocateAlignedHugePage(void* hintAddr, size_t sizeBytes, int numaNodeId)
{
    void* alignedAddr
        = mmap(hintAddr, sizeBytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED, -1, 0);
    if (alignedAddr == MAP_FAILED)
    {
        TLLM_THROW("mmap aligned failed.");
        return;
    }
    TLLM_CHECK_WITH_INFO(alignedAddr == hintAddr, "alignedAddr=%p, but hintAddr=%p", alignedAddr, hintAddr);

    void* addr = alignedAddr;

    // NUMA bind
    unsigned long nodemask = 1UL << numaNodeId;
    long mbind_ret = mbind(addr, sizeBytes, MPOL_BIND, &nodemask, sizeof(nodemask) * 8, 0);
    if (mbind_ret != 0)
    {
        TLLM_THROW("mbind failed.");
        munmap(addr, sizeBytes);
        return;
    }

    // Request THP
    if (madvise(addr, sizeBytes, MADV_HUGEPAGE) != 0)
    {
        TLLM_THROW("madvise(MADV_HUGEPAGE) failed.");
    }

    // Touch memory to actually allocate
    memset(addr, 0, sizeBytes);
}

void* NumaHugePagePoolAllocator::allocate(size_t memorySize)
{
    std::unique_lock<std::mutex> lock(mMutex);
    size_t alignedMemorySize = tensorrt_llm::common::divUp(memorySize, kAlignSize) * kAlignSize;
    size_t totalAllocatedSize = mAllocatedSize + alignedMemorySize;
    if (totalAllocatedSize > mMappedSize)
    {
        // we need to map new pages.
        size_t newMapSize
            = tensorrt_llm::common::divUp(totalAllocatedSize - mMappedSize, kHugePageSize) * kHugePageSize;
        if (mMappedSize != 0)
        {
            TLLM_CUDA_CHECK(cudaHostUnregister(mBasePtr));
        }
        allocateAlignedHugePage(mBasePtr + mMappedSize, newMapSize, mGpuMemNumaId);
        mMappedSize += newMapSize;
        TLLM_CUDA_CHECK(cudaHostRegister(mBasePtr, mMappedSize, cudaHostRegisterDefault));
    }
    uint8_t* ptr = mBasePtr + mAllocatedSize;
    mAllocatedSize += alignedMemorySize;
    return ptr;
}

void NumaHugePagePoolAllocator::free(void* ptr)
{
    // TODO: we don't actually free up memory since reuse is not implemented, and our use case is for weights, which are
    // not released until exit.
    (void) ptr;
}

void NumaHugePagePoolAllocator::maybeInit()
{
    std::unique_lock<std::mutex> lock(mMutex);

    if (mIsInited)
    {
        return;
    }

    TLLM_CUDA_CHECK(cudaGetDevice(&mDevId));
    mGpuMemNumaId = TopologyDetector::getInstance().getCurrentGpuMemoryNumaId();
    TLLM_CHECK_WITH_INFO(mGpuMemNumaId >= 0, "NUMA memory not supported.");
    // allocate a range of virtual address
    mMmapBasePtr = static_cast<uint8_t*>(
        mmap(NULL, kReservedVirtualMemorySize + kHugePageSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    // aligned to huge page boundary
    size_t offset = reinterpret_cast<uint64_t>(mMmapBasePtr) % kHugePageSize;
    mBasePtr = mMmapBasePtr;
    if (offset > 0)
    {
        mBasePtr += kHugePageSize - offset;
    }
    mIsInited = true;
}

void NumaHugePagePoolAllocator::shutdown()
{
    munmap(mMmapBasePtr, kReservedVirtualMemorySize + kHugePageSize);
}

bool HostAccessibleDeviceAllocator::mAllowManagedFallback = false;

bool HostAccessibleDeviceAllocator::isSupported()
{
    if (TopologyDetector::getInstance().getCurrentGpuMemoryNumaId() >= 0)
    {
        // we are on systems that GPU memory is also a NUMA node.
        return true;
    }
    if (!tensorrt_llm::runtime::gdrcopy::isInitialized() && !tensorrt_llm::runtime::gdrcopy::initialize())
    {
        // system don't support GDRCopy.
        return mAllowManagedFallback;
    }
    return true;
}

void HostAccessibleDeviceAllocator::init()
{
    TLLM_CHECK(mIsInited == false);

    if (getenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK") != nullptr)
    {
        if (std::string(getenv("TLLM_HOST_ACCESSIBLE_ALLOW_MANAGED_FALLBACK")) == "1")
        {
            mAllowManagedFallback = true;
        }
    }

    TLLM_CUDA_CHECK(cudaGetDevice(&mDevId));
    mGpuMemNumaId = TopologyDetector::getInstance().getCurrentGpuMemoryNumaId();
    if (mGpuMemNumaId < 0)
    {
        // We only use GDRCopy when there is no NUMA node for GPU memory.
        bool gdrCopyInitedSuccess = true;
        if (!tensorrt_llm::runtime::gdrcopy::isInitialized() && !tensorrt_llm::runtime::gdrcopy::initialize())
        {
            gdrCopyInitedSuccess = false;
        }
        if (gdrCopyInitedSuccess)
        {
            mGdrHandle = tensorrt_llm::runtime::gdrcopy::open();
        }
    }
    mIsInited = true;
}

void HostAccessibleDeviceAllocator::shutdown()
{
    if (mIsInited == false)
    {
        return;
    }
    // We should close GDRCopy handle in the last MoeLoadBalancer,
    // But there might be some allocated memory not freed, so we can't close GDRCopy handle.
    // So for now, we don't close GDRCopy handle.
#if 0
    if (mGdrHandle != nullptr) {
        tensorrt_llm::runtime::gdrcopy::close(mGdrHandle);
        mGdrHandle = nullptr;
    }
#endif
    mIsInited = false;
}

HostAccessibleDeviceAllocator& HostAccessibleDeviceAllocator::getInstance()
{
    static HostAccessibleDeviceAllocator instance;
    return instance;
}

void HostAccessibleDeviceAllocator::IncRefCount()
{
    std::lock_guard<std::mutex> lock(mRefMutex);
    if (mLoadBalancerCount == 0)
    {
        init();
    }
    mLoadBalancerCount++;
}

void HostAccessibleDeviceAllocator::DecRefCount()
{
    std::lock_guard<std::mutex> lock(mRefMutex);
    mLoadBalancerCount--;
    if (mLoadBalancerCount == 0)
    {
        shutdown();
    }
}

void HostAccessibleDeviceAllocator::recordAllocation(
    void* devPtr, size_t memorySize, void* hostPtr, gdrcopy::GdrMemDesc* memDesc)
{
    std::unique_lock<std::shared_mutex> lock(mAllocationsMutex);
    mDeviceAllocations[devPtr] = {memorySize, hostPtr, devPtr, memDesc};
    mHostAllocations[hostPtr] = {memorySize, hostPtr, devPtr, memDesc};
}

HostAccessibleDeviceAllocator::AllocationInfo HostAccessibleDeviceAllocator::getAllocationInfoFromHostPtr(
    void const* hostPtr)
{
    std::shared_lock<std::shared_mutex> lock(mAllocationsMutex);
    if (mHostAllocations.empty())
    {
        return HostAccessibleDeviceAllocator::AllocationInfo{0, nullptr, nullptr, nullptr};
    }
    auto it = mHostAllocations.upper_bound(hostPtr);
    if (it == mHostAllocations.begin())
    {
        return HostAccessibleDeviceAllocator::AllocationInfo{0, nullptr, nullptr, nullptr};
        ;
    }
    --it;
    return it->second;
}

HostAccessibleDeviceAllocator::AllocationInfo HostAccessibleDeviceAllocator::getAllocationInfoFromDevPtr(
    void const* devPtr)
{
    std::shared_lock<std::shared_mutex> lock(mAllocationsMutex);
    if (mDeviceAllocations.empty())
    {
        return HostAccessibleDeviceAllocator::AllocationInfo{0, nullptr, nullptr, nullptr};
    }
    auto it = mDeviceAllocations.upper_bound(devPtr);
    if (it == mDeviceAllocations.begin())
    {
        return HostAccessibleDeviceAllocator::AllocationInfo{0, nullptr, nullptr, nullptr};
        ;
    }
    --it;
    return it->second;
}

void* HostAccessibleDeviceAllocator::getHostPtr(void* devPtr)
{
    auto allocationInfo = getAllocationInfoFromDevPtr(devPtr);
    if (allocationInfo.devPtr == nullptr)
    {
        return nullptr;
    }
    void* recordedDevPtr = allocationInfo.devPtr;
    size_t recordedSize = allocationInfo.size;
    void* recordedHostPtr = allocationInfo.hostPtr;

    auto pDev = static_cast<char*>(devPtr);
    auto pRecordedDev = static_cast<char*>(recordedDevPtr);

    if (pDev >= pRecordedDev && pDev < (pRecordedDev + recordedSize))
    {
        ptrdiff_t offset = pDev - pRecordedDev;
        return static_cast<char*>(recordedHostPtr) + offset;
    }

    return nullptr;
}

void HostAccessibleDeviceAllocator::memcpyToDevice(void* dst, void const* src, size_t size)
{
    if (mGdrHandle != nullptr)
    {
        auto allocationInfo = getAllocationInfoFromHostPtr(dst);
        TLLM_CHECK(allocationInfo.hostPtr != nullptr);
        TLLM_CHECK(allocationInfo.memDesc != nullptr);
        tensorrt_llm::runtime::gdrcopy::copy_to_mapping(allocationInfo.memDesc->gdrMh, dst, src, size);
    }
    else
    {
        memcpy(dst, src, size);
    }
}

void* HostAccessibleDeviceAllocator::allocate(size_t memorySize)
{
    int currentDevId = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&currentDevId));
    TLLM_CHECK_WITH_INFO(currentDevId == mDevId,
        "HostAccessibleDeviceAllocator is not initialized for the current device, currentDevId=%d, mDevId=%d",
        currentDevId, mDevId);
    TLLM_CHECK_WITH_INFO(isSupported(), "HostAccessibleDeviceAllocator is not supported on the current system.");
    void* devPtr = nullptr;
    void* hostPtr = nullptr;
    gdrcopy::GdrMemDesc* memDesc = nullptr;
    if (mGpuMemNumaId >= 0)
    {
        // devPtr = TopologyDetector::getInstance().allocateCurrentGpuNumaMemory(memorySize);
        devPtr = NumaHugePagePoolAllocator::getInstance().allocate(memorySize);
        hostPtr = devPtr;
    }
    else if (mGdrHandle)
    {
        gdrcopy::gdrCudaMalloc(&hostPtr, &devPtr, memorySize, &memDesc, mGdrHandle);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(
            mAllowManagedFallback, "HostAccessibleDeviceAllocator is not supported on the current system.");
        TLLM_CUDA_CHECK(cudaMallocManaged(&devPtr, memorySize));
        TLLM_CUDA_CHECK(cudaMemAdvise(
            devPtr, memorySize, cudaMemAdviseSetPreferredLocation, {cudaMemLocationTypeDevice, currentDevId}));
        hostPtr = devPtr;
    }
    recordAllocation(devPtr, memorySize, hostPtr, memDesc);
    return devPtr;
}

void HostAccessibleDeviceAllocator::free(void* ptr)
{
    std::unique_lock<std::shared_mutex> lock(mAllocationsMutex);
    auto it = mDeviceAllocations.find(ptr);
    if (it != mDeviceAllocations.end())
    {
        auto const& allocInfo = it->second;
        if (allocInfo.memDesc)
        {
            gdrcopy::gdrCudaFree(allocInfo.memDesc, mGdrHandle);
        }
        else if (mGpuMemNumaId >= 0)
        {
            // TopologyDetector::getInstance().freeCurrentGpuNumaMemory(const_cast<void*>(it->first), allocInfo.size);
            NumaHugePagePoolAllocator::getInstance().free(const_cast<void*>(it->first));
        }
        else
        {
            TLLM_CHECK_WITH_INFO(
                mAllowManagedFallback, "HostAccessibleDeviceAllocator is not supported on the current system.");
            TLLM_CUDA_CHECK(cudaFree(ptr));
        }
        void* hostPtr = it->second.hostPtr;
        TLLM_CHECK_WITH_INFO(mHostAllocations.count(hostPtr) == 1, "host pointer not recorded.");
        mDeviceAllocations.erase(it);
        mHostAllocations.erase(hostPtr);
    }
    else
    {
        TLLM_LOG_WARNING("Attempted to free a pointer that was not allocated by HostAccessibleDeviceAllocator.");
    }
}

} // namespace tensorrt_llm::runtime
