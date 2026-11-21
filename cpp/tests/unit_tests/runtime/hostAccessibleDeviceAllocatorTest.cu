/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/runtime/moeLoadBalancer/hostAccessibleDeviceAllocator.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <numeric>
#include <string>
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

namespace
{

//! Set an environment variable for the scope of a test and restore its previous value after.
class ScopedEnv
{
public:
    ScopedEnv(char const* name, char const* value)
        : mName(name)
    {
        char const* previous = std::getenv(name);
        mHadPrevious = previous != nullptr;
        mPrevious = mHadPrevious ? previous : "";
        setenv(name, value, 1);
    }

    ~ScopedEnv()
    {
        if (mHadPrevious)
        {
            setenv(mName.c_str(), mPrevious.c_str(), 1);
        }
        else
        {
            unsetenv(mName.c_str());
        }
    }

private:
    std::string mName;
    bool mHadPrevious;
    std::string mPrevious;
};

} // namespace

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

    // Initialize the allocator again with GDRCopy forced, so a test runs on GDRCopy wherever gdrdrv is loaded, even
    // where GPU memory is a NUMA node. Test bodies are subclasses, which the allocator's friendship does not reach.
    void reopenWithGdrCopyForced()
    {
        allocator->DecRefCount();
        ScopedEnv const forceGdrcopy("TRTLLM_EPLB_FORCE_GDRCOPY", "1");
        allocator->IncRefCount();
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

// Without a NUMA node for GPU memory (e.g. CDMM mode) and without GDRCopy, host accessible memory falls back to
// managed memory on hardware-coherent platforms (e.g. GB200/GB300), where the host can access device-resident managed
// memory directly. The test only runs in that configuration, e.g. on a CDMM node where /dev/gdrdrv is absent.
TEST_F(HostAccessibleDeviceAllocatorTest, ManagedMemoryFallbackWithoutGdrCopy)
{
    int deviceId = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&deviceId));
    int directAccess = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&directAccess, cudaDevAttrDirectManagedMemAccessFromHost, deviceId));
    if (directAccess == 0)
    {
        GTEST_SKIP() << "Host cannot directly access device-resident managed memory on this system.";
    }
    if (tensorrt_llm::common::getEnvEplbForceGdrcopy())
    {
        GTEST_SKIP() << "GDRCopy is forced by TRTLLM_EPLB_FORCE_GDRCOPY.";
    }
    if (TopologyDetector::getInstance().getCurrentGpuMemoryNumaId() >= 0 || gdrcopy::initialize())
    {
        GTEST_SKIP() << "GPU memory is a NUMA node or GDRCopy is available, so the managed memory fallback is unused.";
    }
    ASSERT_TRUE(allocator->isSupported());

    constexpr size_t numInts = 256;
    constexpr size_t allocSize = numInts * sizeof(int);
    void* devPtr = allocator->allocate(allocSize);
    ASSERT_NE(devPtr, nullptr);
    void* hostPtr = allocator->getHostPtr(devPtr);
    ASSERT_NE(hostPtr, nullptr);

    // 1. Copy from the host with memcpyToDevice, then verify on the device.
    std::vector<int> pattern(numInts);
    std::iota(pattern.begin(), pattern.end(), 0);
    allocator->memcpyToDevice(hostPtr, pattern.data(), allocSize);
    bool deviceVerificationResult = false;
    bool* dResult = nullptr;
    TLLM_CUDA_CHECK(cudaMalloc(&dResult, sizeof(bool)));
    verifyDataOnDevice<<<1, 1>>>(static_cast<int*>(devPtr), numInts, dResult);
    TLLM_CUDA_CHECK(cudaDeviceSynchronize());
    TLLM_CUDA_CHECK(cudaMemcpy(&deviceVerificationResult, dResult, sizeof(bool), cudaMemcpyDeviceToHost));
    TLLM_CUDA_CHECK(cudaFree(dResult));
    EXPECT_TRUE(deviceVerificationResult);

    // 2. Write from the device, then read and verify from the host.
    writeDataOnDevice<<<1, 1>>>(static_cast<int*>(devPtr), numInts);
    TLLM_CUDA_CHECK(cudaDeviceSynchronize());
    int const* hostIntPtr = static_cast<int const*>(hostPtr);
    for (size_t i = 0; i < numInts; ++i)
    {
        EXPECT_EQ(hostIntPtr[i], static_cast<int>(numInts - i));
    }

    allocator->free(devPtr);
}

// With fault tolerance, the online EPLB weight updates copy each row at its device address. Check that this writes the
// bytes that a copy through the row's host pointer writes, for a row inside the allocation.
TEST_F(HostAccessibleDeviceAllocatorTest, MemcpyToDeviceAddressMatchesMemcpyToDevice)
{
    if (!allocator->isSupported())
    {
        GTEST_SKIP() << "Host accessible memory is not supported on this system.";
    }

    constexpr size_t numInts = 256;
    constexpr size_t allocSize = numInts * sizeof(int);
    constexpr size_t rowOffset = 64;
    constexpr size_t rowInts = 96;
    std::vector<int> pattern(rowInts);
    std::iota(pattern.begin(), pattern.end(), 7);

    char* byHostPtr = static_cast<char*>(allocator->allocate(allocSize));
    char* byDeviceAddress = static_cast<char*>(allocator->allocate(allocSize));
    ASSERT_NE(byHostPtr, nullptr);
    ASSERT_NE(byDeviceAddress, nullptr);
    ASSERT_NE(allocator->getHostPtr(byHostPtr), nullptr);
    ASSERT_NE(allocator->getHostPtr(byDeviceAddress), nullptr);
    TLLM_CUDA_CHECK(cudaMemset(byHostPtr, 0, allocSize));
    TLLM_CUDA_CHECK(cudaMemset(byDeviceAddress, 0, allocSize));
    TLLM_CUDA_CHECK(cudaDeviceSynchronize());

    allocator->memcpyToDevice(static_cast<char*>(allocator->getHostPtr(byHostPtr)) + rowOffset * sizeof(int),
        pattern.data(), rowInts * sizeof(int));
    allocator->memcpyToDeviceAddress(byDeviceAddress + rowOffset * sizeof(int), pattern.data(), rowInts * sizeof(int));

    std::vector<int> expected(numInts);
    std::vector<int> actual(numInts);
    TLLM_CUDA_CHECK(cudaMemcpy(expected.data(), byHostPtr, allocSize, cudaMemcpyDeviceToHost));
    TLLM_CUDA_CHECK(cudaMemcpy(actual.data(), byDeviceAddress, allocSize, cudaMemcpyDeviceToHost));
    EXPECT_EQ(actual, expected);
    EXPECT_EQ(actual[rowOffset], 7);
    EXPECT_EQ(actual[rowOffset - 1], 0);

    allocator->free(byHostPtr);
    allocator->free(byDeviceAddress);
}

// With GDRCopy, a copy at a device address splits across adjacent host mappings, as the chunks of a backed spare slot
// are mapped one by one, and is refused outside every mapping.
TEST_F(HostAccessibleDeviceAllocatorTest, MemcpyToDeviceAddressSplitsAcrossMappings)
{
    reopenWithGdrCopyForced();
    if (!allocator->isSupported() || !allocator->usesGdrCopy())
    {
        GTEST_SKIP() << "Device memory is not mapped for the host with GDRCopy on this system.";
    }

    constexpr size_t kPageSize = 1UL << 16;
    void* raw = nullptr;
    TLLM_CUDA_CHECK(cudaMalloc(&raw, 4 * kPageSize));
    auto* const base = reinterpret_cast<char*>((reinterpret_cast<uintptr_t>(raw) + kPageSize - 1) & ~(kPageSize - 1));
    TLLM_CUDA_CHECK(cudaMemset(base, 0, 3 * kPageSize));
    TLLM_CUDA_CHECK(cudaDeviceSynchronize());
    allocator->mapDeviceMemory(base, kPageSize);
    allocator->mapDeviceMemory(base + kPageSize, kPageSize);
    ASSERT_NE(allocator->getHostPtr(base), nullptr);
    ASSERT_NE(allocator->getHostPtr(base + kPageSize), nullptr);

    // A row that ends 256 bytes into the second mapping.
    constexpr size_t kRowBytes = 1024;
    char* const row = base + kPageSize - (kRowBytes - 256);
    std::vector<uint8_t> pattern(kRowBytes);
    std::iota(pattern.begin(), pattern.end(), 1);
    allocator->memcpyToDeviceAddress(row, pattern.data(), kRowBytes);
    std::vector<uint8_t> written(kRowBytes + 2);
    TLLM_CUDA_CHECK(cudaMemcpy(written.data(), row - 1, written.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(written.front(), 0);
    EXPECT_EQ(written.back(), 0);
    EXPECT_TRUE(std::equal(pattern.begin(), pattern.end(), written.begin() + 1));

    // The third page is not mapped for the host.
    EXPECT_THROW(allocator->memcpyToDeviceAddress(base + 2 * kPageSize, pattern.data(), kRowBytes),
        tensorrt_llm::common::TllmException);
    EXPECT_THROW(allocator->memcpyToDeviceAddress(base + 2 * kPageSize - 256, pattern.data(), kRowBytes),
        tensorrt_llm::common::TllmException);

    allocator->unmapDeviceMemory(base);
    allocator->unmapDeviceMemory(base + kPageSize);
    TLLM_CUDA_CHECK(cudaFree(raw));
}

} // namespace tensorrt_llm::runtime::unit_tests
