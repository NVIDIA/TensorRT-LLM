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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/moeLoadBalancer/hostAccessibleDeviceAllocator.h"
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

namespace tensorrt_llm::runtime::unit_tests
{

// Kernel to verify data written by the host on the device.
// It checks if each element in the array devPtr has the expected value.
__global__ void verifyDataOnDevice(int const* devPtr, int size, bool* result)
{
    bool success = true;
    for (int i = 0; i < size; ++i)
    {
        if (devPtr[i] != i)
        {
            success = false;
            break;
        }
    }
    *result = success;
}

// Kernel to write data to device memory from the device.
// This is used to test if host can read data written by the device.
__global__ void writeDataOnDevice(int* devPtr, int size)
{
    for (int i = 0; i < size; ++i)
    {
        devPtr[i] = size - i;
    }
}

class HostAccessibleDeviceAllocatorTest : public ::testing::Test
{
protected:
    // SetUp is called before each test case.
    void SetUp() override
    {
        // Get the allocator instance, which is a singleton.
        allocator = &HostAccessibleDeviceAllocator::getInstance();
        // The allocator is initialized on the first IncRefCount call.
        // This is to simulate a component (like MoeLoadBalancer) starting to use the allocator.
        allocator->IncRefCount();
    }

    // TearDown is called after each test case.
    void TearDown() override
    {
        // Decrement the reference count.
        // The allocator will be shut down when the count reaches zero.
        allocator->DecRefCount();
    }

    HostAccessibleDeviceAllocator* allocator;
};

// Test case to check the basic allocation and free functionality.
TEST_F(HostAccessibleDeviceAllocatorTest, AllocationAndFree)
{
    // Skip the test if host-accessible memory is not supported on the current system.
    if (!allocator->isSupported())
    {
        GTEST_SKIP() << "Host accessible memory is not supported on this system.";
    }

    constexpr size_t allocSize = 1024;
    void* devPtr = allocator->allocate(allocSize);
    ASSERT_NE(devPtr, nullptr);

    // Free the allocated memory.
    allocator->free(devPtr);

    // Test freeing a pointer that was not allocated by this allocator.
    // This should not cause a crash but should be handled gracefully (e.g., by logging a warning).
    int* dummyPtr = nullptr;
    TLLM_CUDA_CHECK(cudaMalloc(&dummyPtr, 16));
    allocator->free(dummyPtr);
    TLLM_CUDA_CHECK(cudaFree(dummyPtr));
}

// Test case to verify data access between the host and the device.
TEST_F(HostAccessibleDeviceAllocatorTest, HostDeviceDataAccess)
{
    if (!allocator->isSupported())
    {
        GTEST_SKIP() << "Host accessible memory is not supported on this system.";
    }

    constexpr size_t numInts = 256;
    constexpr size_t allocSize = numInts * sizeof(int);

    void* devPtr = allocator->allocate(allocSize);
    ASSERT_NE(devPtr, nullptr);

    void* hostPtr = allocator->getHostPtr(devPtr);
    ASSERT_NE(hostPtr, nullptr);

    // 1. Write from the host, then read and verify from the device.
    int* hostIntPtr = static_cast<int*>(hostPtr);
    for (size_t i = 0; i < numInts; ++i)
    {
        hostIntPtr[i] = i;
    }

    // Use a CUDA kernel to verify the data on the device.
    bool deviceVerificationResult = false;
    bool* d_result;
    TLLM_CUDA_CHECK(cudaMalloc(&d_result, sizeof(bool)));
    verifyDataOnDevice<<<1, 1>>>(static_cast<int*>(devPtr), numInts, d_result);
    TLLM_CUDA_CHECK(cudaDeviceSynchronize());
    TLLM_CUDA_CHECK(cudaMemcpy(&deviceVerificationResult, d_result, sizeof(bool), cudaMemcpyDeviceToHost));
    TLLM_CUDA_CHECK(cudaFree(d_result));
    EXPECT_TRUE(deviceVerificationResult);

    // 2. Write from the device, then read and verify from the host.
    writeDataOnDevice<<<1, 1>>>(static_cast<int*>(devPtr), numInts);
    TLLM_CUDA_CHECK(cudaDeviceSynchronize());

    for (size_t i = 0; i < numInts; ++i)
    {
        EXPECT_EQ(hostIntPtr[i], numInts - i);
    }

    allocator->free(devPtr);
}

// Test getting a host pointer for an offset device pointer.
TEST_F(HostAccessibleDeviceAllocatorTest, GetHostPtrForOffset)
{
    if (!allocator->isSupported())
    {
        GTEST_SKIP() << "Host accessible memory is not supported on this system.";
    }

    constexpr size_t allocSize = 1024;
    void* baseDevPtr = allocator->allocate(allocSize);
    ASSERT_NE(baseDevPtr, nullptr);

    void* baseHostPtr = allocator->getHostPtr(baseDevPtr);
    ASSERT_NE(baseHostPtr, nullptr);

    // Check a pointer with an offset within the allocation.
    ptrdiff_t offset = 128;
    void* offsetDevPtr = static_cast<char*>(baseDevPtr) + offset;
    void* offsetHostPtr = allocator->getHostPtr(offsetDevPtr);

    ASSERT_NE(offsetHostPtr, nullptr);
    EXPECT_EQ(offsetHostPtr, static_cast<char*>(baseHostPtr) + offset);

    // Check a pointer to the last byte of the allocation.
    offset = allocSize - 1;
    offsetDevPtr = static_cast<char*>(baseDevPtr) + offset;
    offsetHostPtr = allocator->getHostPtr(offsetDevPtr);
    ASSERT_NE(offsetHostPtr, nullptr);
    EXPECT_EQ(offsetHostPtr, static_cast<char*>(baseHostPtr) + offset);

    // Check a pointer just outside the allocation boundary. It should not be found.
    void* outsideDevPtr = static_cast<char*>(baseDevPtr) + allocSize;
    void* outsideHostPtr = allocator->getHostPtr(outsideDevPtr);
    EXPECT_EQ(outsideHostPtr, nullptr);

    allocator->free(baseDevPtr);
}

// Test multiple allocations and frees.
TEST_F(HostAccessibleDeviceAllocatorTest, MultipleAllocations)
{
    if (!allocator->isSupported())
    {
        GTEST_SKIP() << "Host accessible memory is not supported on this system.";
    }

    constexpr int numAllocs = 10;
    constexpr size_t baseSize = 64;
    std::vector<void*> devPtrs(numAllocs);
    std::vector<size_t> sizes(numAllocs);

    // Allocate multiple blocks of memory.
    for (int i = 0; i < numAllocs; ++i)
    {
        sizes[i] = baseSize * (i + 1);
        devPtrs[i] = allocator->allocate(sizes[i]);
        ASSERT_NE(devPtrs[i], nullptr);
    }

    // Verify each allocation by writing and reading a value.
    for (int i = 0; i < numAllocs; ++i)
    {
        void* hostPtr = allocator->getHostPtr(devPtrs[i]);
        ASSERT_NE(hostPtr, nullptr);
        // Do a small write/read test.
        static_cast<char*>(hostPtr)[0] = static_cast<char>(i);
        EXPECT_EQ(static_cast<char*>(hostPtr)[0], static_cast<char>(i));
    }

    // Free all allocated blocks.
    for (int i = 0; i < numAllocs; ++i)
    {
        allocator->free(devPtrs[i]);
    }
}

// Test that getHostPtr returns nullptr for a pointer that was not allocated
// by this allocator.
TEST_F(HostAccessibleDeviceAllocatorTest, GetHostPtrForUnallocated)
{
    if (!allocator->isSupported())
    {
        GTEST_SKIP() << "Host accessible memory is not supported on this system.";
    }
    // Use an arbitrary pointer value.
    void* devPtr = reinterpret_cast<void*>(0xDEADBEEF);
    EXPECT_EQ(allocator->getHostPtr(devPtr), nullptr);
}

} // namespace tensorrt_llm::runtime::unit_tests
