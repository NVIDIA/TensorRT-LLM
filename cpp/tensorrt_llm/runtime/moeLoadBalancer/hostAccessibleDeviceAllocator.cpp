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

#include <cstddef>
#include <cuda_runtime_api.h>

#include "gdrwrap.h"
#include "hostAccessibleDeviceAllocator.h"
#include "topologyDetector.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"

namespace tensorrt_llm::runtime
{

void HostAccessibleDeviceAllocator::init()
{
    TLLM_CHECK(mIsInited == false);
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
    std::lock_guard<std::mutex> lock(mAllocationsMutex);
    mAllocations[devPtr] = {memorySize, hostPtr, memDesc};
}

void* HostAccessibleDeviceAllocator::getHostPtr(void* devPtr)
{
    std::lock_guard<std::mutex> lock(mAllocationsMutex);
    if (mAllocations.empty())
    {
        return nullptr;
    }

    auto it = mAllocations.upper_bound(devPtr);
    if (it == mAllocations.begin())
    {
        return nullptr;
    }

    --it;

    void* recordedDevPtr = it->first;
    auto const& allocationInfo = it->second;
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
        devPtr = TopologyDetector::getInstance().allocateCurrentGpuNumaMemory(memorySize);
        hostPtr = devPtr;
    }
    else
    {
        gdrcopy::gdrCudaMalloc(&hostPtr, &devPtr, memorySize, &memDesc, mGdrHandle);
    }
    recordAllocation(devPtr, memorySize, hostPtr, memDesc);
    return devPtr;
}

void HostAccessibleDeviceAllocator::free(void* ptr)
{
    std::lock_guard<std::mutex> lock(mAllocationsMutex);
    auto it = mAllocations.find(ptr);
    if (it != mAllocations.end())
    {
        auto const& allocInfo = it->second;
        if (allocInfo.memDesc)
        {
            gdrcopy::gdrCudaFree(allocInfo.memDesc, mGdrHandle);
        }
        else
        {
            TopologyDetector::getInstance().freeCurrentGpuNumaMemory(it->first, allocInfo.size);
        }
        mAllocations.erase(it);
    }
    else
    {
        TLLM_LOG_WARNING("Attempted to free a pointer that was not allocated by HostAccessibleDeviceAllocator.");
    }
}

} // namespace tensorrt_llm::runtime
