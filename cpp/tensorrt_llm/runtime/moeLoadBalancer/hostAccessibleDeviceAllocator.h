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

#pragma once

#include <map>
#include <mutex>
#include <utility>

#include "gdrwrap.h"
#include "topologyDetector.h"

namespace tensorrt_llm::runtime
{

class MoeLoadBalancer;

namespace unit_tests
{
class HostAccessibleDeviceAllocatorTest;
}

class HostAccessibleDeviceAllocator
{
public:
    // Delete the copy constructor and copy assignment operator to prevent cloning.
    HostAccessibleDeviceAllocator(HostAccessibleDeviceAllocator const&) = delete;
    void operator=(HostAccessibleDeviceAllocator const&) = delete;

    /**
     * @brief Get the single instance of the HostAccessibleDeviceAllocator.
     *
     * @return HostAccessibleDeviceAllocator& Reference to the singleton instance.
     */
    static HostAccessibleDeviceAllocator& getInstance();

    /**
     * @brief Allocate host accessible memory on the device.
     *
     * @param memorySize The size of the memory to allocate.
     * @return void* Pointer to the allocated memory.
     */
    void* allocate(size_t memorySize);

    /**
     * @brief Free the allocated memory.
     *
     * @param ptr Pointer to the memory to free.
     */
    void free(void* ptr);

    /**
     * @brief Get the host-accessible pointer for a given device pointer.
     *
     * @param devPtr The device pointer to look up. It can be a pointer inside a recorded allocation.
     * @return void* The corresponding host-accessible pointer, or nullptr if not found.
     */
    void* getHostPtr(void* devPtr);

    /**
     * @brief Check if host accessible memory is supported on the current system.
     * @note This function should be called after IncRefCount() by some MoeLoadBalancer.
     *
     * @return bool True if host accessible memory is supported, false otherwise.
     */
    bool isSupported() const
    {
        return mGpuMemNumaId >= 0 || mGdrHandle != nullptr;
    }

private:
    struct AllocationInfo
    {
        size_t size;
        void* hostPtr;
        gdrcopy::GdrMemDesc* memDesc;
    };

    /**
     * @brief Private constructor to prevent direct instantiation.
     *
     * Initialization logic for the allocator (like initializing GDRCopy)
     * can be placed here.
     */
    HostAccessibleDeviceAllocator() = default;

    /**
     * @brief Initialize the allocator.
     */
    void init();

    /**
     * @brief Shutdown the allocator.
     */
    void shutdown();

    friend class tensorrt_llm::runtime::MoeLoadBalancer;
    friend class tensorrt_llm::runtime::unit_tests::HostAccessibleDeviceAllocatorTest;

    /**
     * @brief Increment the reference count of the load balancer.
     * This Allocator is shared by multiple MoeLoadBalancers, so we need to
     * increment the reference count when a new MoeLoadBalancer is created.
     * They may share the same GDR handle.
     */
    void IncRefCount();

    /**
     * @brief Decrement the reference count of the load balancer.
     * This Allocator is shared by multiple MoeLoadBalancers, so we need to
     * decrement the reference count when a MoeLoadBalancer is destroyed.
     * If the reference count is 0, we need to close the GDR handle.
     */
    void DecRefCount();

    /**
     * @brief Record a device memory allocation and its corresponding host-accessible pointer.
     *
     * @param devPtr The device pointer of the allocated memory.
     * @param memorySize The size of the allocated memory.
     * @param hostPtr The corresponding host-accessible pointer.
     * @param memDesc Optional GDR memory descriptor if allocated with GDRCopy.
     */
    void recordAllocation(void* devPtr, size_t memorySize, void* hostPtr, gdrcopy::GdrMemDesc* memDesc = nullptr);

    // if GPU memory has NUMA id, then CPU can direct access that. We should use this.
    int mGpuMemNumaId = -1;
    // if Not, we should use GDRCopy
    gdr_t mGdrHandle = nullptr;

    int mDevId = -1;

    bool mIsInited = false;
    std::mutex mRefMutex;
    int mLoadBalancerCount = 0;

    std::mutex mAllocationsMutex;
    std::map<void*, AllocationInfo> mAllocations;
};

} // namespace tensorrt_llm::runtime
