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

#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <vector>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/runtime/locality_domain/locality_domain_utils.h"

using namespace tensorrt_llm::locality_domain;

// Memory allocation type for dual-stream tests
enum class MemoryAllocationType
{
    SAME,      // Allocate on the same LOCALITY_DOMAIN as the stream
    DIFFERENT, // Allocate on a different LOCALITY_DOMAIN from the stream
    NORMAL     // Use regular cudaMalloc (no LOCALITY_DOMAIN localization)
};

// CUDA kernel for int4 memory copy
// Each thread processes 8 int4 elements to hide memory latency
// Use strided access to maintain coalesced memory access within a warp
__global__ void memcpyInt4Kernel(int4* dst, int4 const* src, size_t numInt4Elements)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

// Process 8 int4 elements per thread with strided access
// This ensures threads in the same warp access consecutive memory addresses
#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        size_t idx = tid + i * stride;
        if (idx < numInt4Elements)
        {
            dst[idx] = src[idx];
        }
    }
}

class LocalizationTest : public ::testing::Test
{
protected:
    static constexpr size_t kGiB = 1024ULL * 1024ULL * 1024ULL;
    static constexpr size_t kMinPerfAllocationSize = 256ULL * 1024ULL * 1024ULL;

    struct DeviceAllocation
    {
        LocalizationHandle* handle = nullptr;
        MemoryAllocationType type = MemoryAllocationType::NORMAL;
        void* ptr = nullptr;

        DeviceAllocation() = default;
        DeviceAllocation(DeviceAllocation const&) = delete;
        DeviceAllocation& operator=(DeviceAllocation const&) = delete;

        ~DeviceAllocation() noexcept
        {
            resetNoThrow();
        }

        void allocate(
            LocalizationHandle* newHandle, size_t size, MemoryAllocationType newType, int streamLocalityDomainId)
        {
            reset();
            handle = newHandle;
            type = newType;
            switch (type)
            {
            case MemoryAllocationType::SAME: handle->localityDomainMalloc(&ptr, size, streamLocalityDomainId); break;
            case MemoryAllocationType::DIFFERENT:
                handle->localityDomainMalloc(&ptr, size, 1 - streamLocalityDomainId); // 0->1, 1->0
                break;
            case MemoryAllocationType::NORMAL: TLLM_CUDA_CHECK(cudaMalloc(&ptr, size)); break;
            }
        }

        void reset()
        {
            if (ptr == nullptr)
            {
                return;
            }
            if (type == MemoryAllocationType::NORMAL)
            {
                TLLM_CUDA_CHECK(cudaFree(ptr));
            }
            else
            {
                handle->localityDomainFree(ptr);
            }
            ptr = nullptr;
        }

        void resetNoThrow() noexcept
        {
            if (ptr == nullptr)
            {
                return;
            }
            if (type == MemoryAllocationType::NORMAL)
            {
                auto result = cudaFree(ptr);
                if (result != cudaSuccess)
                {
                    ADD_FAILURE() << "cudaFree failed during cleanup: " << cudaGetErrorString(result);
                }
            }
            else
            {
                try
                {
                    handle->localityDomainFree(ptr);
                }
                catch (...)
                {
                    ADD_FAILURE() << "localityDomainFree failed during cleanup";
                }
            }
            ptr = nullptr;
        }
    };

    struct StreamHolder
    {
        CUstream stream = nullptr;
        //! True only for streams this holder created. Borrowed localized streams are owned by the
        //! process-lifetime localization resource and must stay unowned here.
        bool owned = false;

        StreamHolder() = default;
        StreamHolder(StreamHolder const&) = delete;
        StreamHolder& operator=(StreamHolder const&) = delete;

        ~StreamHolder() noexcept
        {
            if (owned && stream != nullptr)
            {
                auto const result = cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream));
                if (result != cudaSuccess)
                {
                    ADD_FAILURE() << "cudaStreamDestroy failed during cleanup: " << cudaGetErrorString(result);
                }
            }
        }
    };

    std::unique_ptr<LocalizationHandle> mHandle;
    bool mSupportsLocalityDomain = false;

    void SetUp() override
    {
        // Create handle
        mHandle = std::make_unique<LocalizationHandle>();

        // Check if LOCALITY_DOMAIN is supported
        mSupportsLocalityDomain = mHandle->supportsLocalization();
        if (!mSupportsLocalityDomain)
        {
            TLLM_LOG_WARNING(
                "LOCALITY_DOMAIN localization is not supported on this device, skipping LOCALITY_DOMAIN-specific "
                "tests.");
        }
    }

    std::optional<size_t> chooseAllocationSize(size_t requestedSize, int allocationCount)
    {
        size_t freeMem = 0;
        size_t totalMem = 0;
        TLLM_CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
        (void) totalMem;

        size_t maxPerAllocation = freeMem / static_cast<size_t>(allocationCount * 2);
        size_t selectedSize = std::min(requestedSize, maxPerAllocation);
        selectedSize = (selectedSize / sizeof(int4)) * sizeof(int4);
        if (selectedSize < kMinPerfAllocationSize)
        {
            return std::nullopt;
        }
        return selectedSize;
    }

    void TearDown() override
    {
        // Handle will be automatically destroyed
    }

    // Helper function: Run memory copy test and return elapsed time (milliseconds)
    float runMemcpyTest(void* dst, void const* src, size_t sizeBytes, CUstream stream, int iterations)
    {
        size_t numInt4Elements = sizeBytes / sizeof(int4);
        int threadsPerBlock = 256;
        // Each thread processes 8 int4 elements
        int numBlocks = (numInt4Elements + threadsPerBlock * 8 - 1) / (threadsPerBlock * 8);

        cudaEvent_t start, stop;
        TLLM_CUDA_CHECK(cudaEventCreate(&start));
        TLLM_CUDA_CHECK(cudaEventCreate(&stop));

        // Warmup
        memcpyInt4Kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
            reinterpret_cast<int4*>(dst), reinterpret_cast<int4 const*>(src), numInt4Elements);
        TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));

        // Actual performance test
        TLLM_CUDA_CHECK(cudaEventRecord(start, stream));
        for (int i = 0; i < iterations; i++)
        {
            memcpyInt4Kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
                reinterpret_cast<int4*>(dst), reinterpret_cast<int4 const*>(src), numInt4Elements);
        }
        TLLM_CUDA_CHECK(cudaEventRecord(stop, stream));
        TLLM_CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0;
        TLLM_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        TLLM_CUDA_CHECK(cudaEventDestroy(start));
        TLLM_CUDA_CHECK(cudaEventDestroy(stop));

        return milliseconds;
    }

    // Helper function: Run dual-stream memory copy test and return elapsed time (milliseconds)
    float runDualStreamMemcpyTest(void* dst0, void const* src0, void* dst1, void const* src1, size_t sizeBytes,
        CUstream stream0, CUstream stream1, int iterations)
    {
        size_t numInt4Elements = sizeBytes / sizeof(int4);
        int threadsPerBlock = 256;
        // Each thread processes 8 int4 elements
        int numBlocks = (numInt4Elements + threadsPerBlock * 8 - 1) / (threadsPerBlock * 8);

        cudaEvent_t start, stop, sync0, sync1;
        TLLM_CUDA_CHECK(cudaEventCreate(&start));
        TLLM_CUDA_CHECK(cudaEventCreate(&stop));
        TLLM_CUDA_CHECK(cudaEventCreate(&sync0));
        TLLM_CUDA_CHECK(cudaEventCreate(&sync1));

        // Warmup
        memcpyInt4Kernel<<<numBlocks, threadsPerBlock, 0, stream0>>>(
            reinterpret_cast<int4*>(dst0), reinterpret_cast<int4 const*>(src0), numInt4Elements);
        memcpyInt4Kernel<<<numBlocks, threadsPerBlock, 0, stream1>>>(
            reinterpret_cast<int4*>(dst1), reinterpret_cast<int4 const*>(src1), numInt4Elements);
        TLLM_CUDA_CHECK(cudaStreamSynchronize(stream0));
        TLLM_CUDA_CHECK(cudaStreamSynchronize(stream1));

        // Actual performance test
        TLLM_CUDA_CHECK(cudaEventRecord(start, stream0)); // Record to default stream

        for (int i = 0; i < iterations; i++)
        {
            memcpyInt4Kernel<<<numBlocks, threadsPerBlock, 0, stream0>>>(
                reinterpret_cast<int4*>(dst0), reinterpret_cast<int4 const*>(src0), numInt4Elements);
            memcpyInt4Kernel<<<numBlocks, threadsPerBlock, 0, stream1>>>(
                reinterpret_cast<int4*>(dst1), reinterpret_cast<int4 const*>(src1), numInt4Elements);

            // Record completion of both streams
            TLLM_CUDA_CHECK(cudaEventRecord(sync0, stream0));
            TLLM_CUDA_CHECK(cudaEventRecord(sync1, stream1));

            TLLM_CUDA_CHECK(cudaStreamWaitEvent(stream0, sync1));
            TLLM_CUDA_CHECK(cudaStreamWaitEvent(stream1, sync0));
        }

        TLLM_CUDA_CHECK(cudaEventRecord(stop, stream0));
        TLLM_CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0;
        TLLM_CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

        TLLM_CUDA_CHECK(cudaEventDestroy(start));
        TLLM_CUDA_CHECK(cudaEventDestroy(stop));
        TLLM_CUDA_CHECK(cudaEventDestroy(sync0));
        TLLM_CUDA_CHECK(cudaEventDestroy(sync1));

        return milliseconds;
    }

    // Helper function: Run dual-stream performance test with configurable memory allocation
    void runDualStreamPerformanceTest(MemoryAllocationType srcType, MemoryAllocationType dstType, char const* testName)
    {
        auto sizeBytesOpt = chooseAllocationSize(10ULL * kGiB, 4);
        if (!sizeBytesOpt.has_value())
        {
            GTEST_SKIP() << "Not enough free GPU memory for dual-stream performance test";
        }
        size_t const sizeBytes = sizeBytesOpt.value();
        int const iterations = 10;

        // Allocate memory for stream 0 (LOCALITY_DOMAIN 0)
        DeviceAllocation devSrc0;
        DeviceAllocation devDst0;
        devSrc0.allocate(mHandle.get(), sizeBytes, srcType, 0);
        devDst0.allocate(mHandle.get(), sizeBytes, dstType, 0);

        // Allocate memory for stream 1 (LOCALITY_DOMAIN 1)
        DeviceAllocation devSrc1;
        DeviceAllocation devDst1;
        devSrc1.allocate(mHandle.get(), sizeBytes, srcType, 1);
        devDst1.allocate(mHandle.get(), sizeBytes, dstType, 1);

        // Initialize source memory
        TLLM_CUDA_CHECK(cudaMemset(devSrc0.ptr, 0x42, sizeBytes));
        TLLM_CUDA_CHECK(cudaMemset(devSrc1.ptr, 0x43, sizeBytes));

        // Create two localized streams on different locality domains
        StreamHolder stream0{mHandle->createLocalizedStream(0)};
        StreamHolder stream1{mHandle->createLocalizedStream(1)};

        // Run test
        float totalTime = runDualStreamMemcpyTest(
            devDst0.ptr, devSrc0.ptr, devDst1.ptr, devSrc1.ptr, sizeBytes, stream0.stream, stream1.stream, iterations);
        float avgTime = totalTime / iterations;
        // Bandwidth accounts for both read and write (2x data movement) on both streams.
        float bandwidth = (2.0 * 2.0 * sizeBytes / static_cast<double>(kGiB)) / (avgTime / 1000.0); // GB/s

        TLLM_LOG_INFO("%s:", testName);
        TLLM_LOG_INFO("  Allocation size per buffer: %.2f GiB", sizeBytes / static_cast<double>(kGiB));
        TLLM_LOG_INFO("  Total time: %.2f ms", totalTime);
        TLLM_LOG_INFO("  Average time per iteration: %.2f ms", avgTime);
        TLLM_LOG_INFO("  Bandwidth: %.2f GB/s", bandwidth);
    }
};

// Test 1: Create and destroy handle
TEST_F(LocalizationTest, CreateAndDestroyHandle)
{
    // Handle is created in SetUp, just verify it's valid
    ASSERT_NE(mHandle, nullptr);

    // TearDown will handle destruction automatically
}

// Test 3: Allocate and free memory on LOCALITY_DOMAIN 0
TEST_F(LocalizationTest, MallocFreeOnLocalityDomain0)
{
    if (!mSupportsLocalityDomain)
    {
        GTEST_SKIP() << "LOCALITY_DOMAIN not supported, skipping test";
    }

    void* devPtr = nullptr;
    size_t size = 1024 * 1024; // 1 MB

    // Allocate memory on LOCALITY_DOMAIN 0
    ASSERT_NO_THROW(mHandle->localityDomainMalloc(&devPtr, size, 0));
    ASSERT_NE(devPtr, nullptr);

#if CUDA_VERSION >= 13040
    int localityDomain = -1;
    ASSERT_EQ(cuPointerGetAttribute(
                  &localityDomain, CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL, reinterpret_cast<CUdeviceptr>(devPtr)),
        CUDA_SUCCESS);
    EXPECT_EQ(localityDomain, 0);
#endif

    // Free memory
    ASSERT_NO_THROW(mHandle->localityDomainFree(devPtr));
}

// Test 4: Allocate and free memory on LOCALITY_DOMAIN 1
TEST_F(LocalizationTest, MallocFreeOnLocalityDomain1)
{
    if (!mSupportsLocalityDomain)
    {
        GTEST_SKIP() << "LOCALITY_DOMAIN not supported, skipping test";
    }

    void* devPtr = nullptr;
    size_t size = 1024 * 1024; // 1 MB

    // Allocate memory on LOCALITY_DOMAIN 1
    ASSERT_NO_THROW(mHandle->localityDomainMalloc(&devPtr, size, 1));
    ASSERT_NE(devPtr, nullptr);

#if CUDA_VERSION >= 13040
    int localityDomain = -1;
    ASSERT_EQ(cuPointerGetAttribute(
                  &localityDomain, CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL, reinterpret_cast<CUdeviceptr>(devPtr)),
        CUDA_SUCCESS);
    EXPECT_EQ(localityDomain, 1);
#endif

    // Free memory
    ASSERT_NO_THROW(mHandle->localityDomainFree(devPtr));
}

// Test 5: Create localized stream on LOCALITY_DOMAIN 0
TEST_F(LocalizationTest, CreateLocalizedStreamLocalityDomain0)
{
    if (!mSupportsLocalityDomain)
    {
        GTEST_SKIP() << "LOCALITY_DOMAIN not supported, skipping test";
    }

    // Create stream localized to LOCALITY_DOMAIN 0
    CUstream stream = mHandle->createLocalizedStream(0);
    ASSERT_NE(stream, nullptr);
    EXPECT_EQ(mHandle->createLocalizedStream(0), stream);
}

// Test 6: Create localized stream on LOCALITY_DOMAIN 1
TEST_F(LocalizationTest, CreateLocalizedStreamLocalityDomain1)
{
    if (!mSupportsLocalityDomain)
    {
        GTEST_SKIP() << "LOCALITY_DOMAIN not supported, skipping test";
    }

    // Create stream localized to LOCALITY_DOMAIN 1
    CUstream stream = mHandle->createLocalizedStream(1);
    ASSERT_NE(stream, nullptr);
    EXPECT_EQ(mHandle->createLocalizedStream(1), stream);
}

// Bandwidth checks allocate tens of GiB and are disabled in normal unit-test runs.
// Run them explicitly with --gtest_also_run_disabled_tests when benchmarking locality-domain locality.

// Performance Test 1: Single stream, copy 20GB memory
TEST_F(LocalizationTest, DISABLED_PerformanceTestSingleStream20GB)
{
    auto sizeBytesOpt = chooseAllocationSize(20ULL * kGiB, 2);
    if (!sizeBytesOpt.has_value())
    {
        GTEST_SKIP() << "Not enough free GPU memory for single-stream performance test";
    }
    size_t const sizeBytes = sizeBytesOpt.value();
    int const iterations = 10;

    // Allocate memory
    DeviceAllocation devSrc;
    DeviceAllocation devDst;
    devSrc.allocate(mHandle.get(), sizeBytes, MemoryAllocationType::NORMAL, 0);
    devDst.allocate(mHandle.get(), sizeBytes, MemoryAllocationType::NORMAL, 0);

    // Initialize source memory
    TLLM_CUDA_CHECK(cudaMemset(devSrc.ptr, 0x42, sizeBytes));

    // Create stream
    StreamHolder stream;
    TLLM_CUDA_CHECK(cudaStreamCreate(reinterpret_cast<cudaStream_t*>(&stream.stream)));
    stream.owned = true;

    // Run test
    float totalTime = runMemcpyTest(devDst.ptr, devSrc.ptr, sizeBytes, stream.stream, iterations);
    float avgTime = totalTime / iterations;
    // Bandwidth accounts for both read and write (2x data movement)
    float bandwidth = (2.0 * sizeBytes / static_cast<double>(kGiB)) / (avgTime / 1000.0); // GB/s

    TLLM_LOG_INFO("Performance Test 1 - Single Stream (20GB):");
    TLLM_LOG_INFO("  Allocation size per buffer: %.2f GiB", sizeBytes / static_cast<double>(kGiB));
    TLLM_LOG_INFO("  Total time: %.2f ms", totalTime);
    TLLM_LOG_INFO("  Average time per iteration: %.2f ms", avgTime);
    TLLM_LOG_INFO("  Bandwidth: %.2f GB/s", bandwidth);
}

// Performance Test 2: Dual streams, source and destination on the same LOCALITY_DOMAIN as stream
TEST_F(LocalizationTest, DISABLED_PerformanceTestDualStreamSameLocalityDomain)
{
    if (!mSupportsLocalityDomain)
    {
        GTEST_SKIP() << "LOCALITY_DOMAIN not supported, skipping test";
    }

    runDualStreamPerformanceTest(MemoryAllocationType::SAME, MemoryAllocationType::SAME,
        "Performance Test 2 - Dual Stream Same LOCALITY_DOMAIN (2x10GB)");
}

// Performance Test 3: Dual streams, source on same locality domains as stream and destination on different locality
// domains
TEST_F(LocalizationTest, DISABLED_PerformanceTestDualStreamSrcSameDstDifferentLocalityDomain)
{
    if (!mSupportsLocalityDomain)
    {
        GTEST_SKIP() << "LOCALITY_DOMAIN not supported, skipping test";
    }

    runDualStreamPerformanceTest(MemoryAllocationType::SAME, MemoryAllocationType::DIFFERENT,
        "Performance Test 3 - Dual Stream Src Same Dst Different LOCALITY_DOMAIN (2x10GB)");
}

// Performance Test 3: Dual streams, source on different locality domains as stream and destination on same locality
// domains
TEST_F(LocalizationTest, DISABLED_PerformanceTestDualStreamSrcDifferentDstSameLocalityDomain)
{
    if (!mSupportsLocalityDomain)
    {
        GTEST_SKIP() << "LOCALITY_DOMAIN not supported, skipping test";
    }

    runDualStreamPerformanceTest(MemoryAllocationType::DIFFERENT, MemoryAllocationType::SAME,
        "Performance Test 3 - Dual Stream Src Different Dst Same LOCALITY_DOMAIN (2x10GB)");
}

// Performance Test 3: Dual streams, source and destination on different locality domains as stream
TEST_F(LocalizationTest, DISABLED_PerformanceTestDualStreamDifferentLocalityDomain)
{
    if (!mSupportsLocalityDomain)
    {
        GTEST_SKIP() << "LOCALITY_DOMAIN not supported, skipping test";
    }

    runDualStreamPerformanceTest(MemoryAllocationType::DIFFERENT, MemoryAllocationType::DIFFERENT,
        "Performance Test 3 - Dual Stream Src and Dst Different LOCALITY_DOMAIN (2x10GB)");
}

// Performance Test 4: Dual localized streams with regular cudaMalloc memory
TEST_F(LocalizationTest, DISABLED_PerformanceTestDualLocalizedStreamRegularMemory)
{
    if (!mSupportsLocalityDomain)
    {
        GTEST_SKIP() << "LOCALITY_DOMAIN not supported, skipping test";
    }

    runDualStreamPerformanceTest(MemoryAllocationType::NORMAL, MemoryAllocationType::NORMAL,
        "Performance Test 4 - Dual Localized Stream with Regular Memory (2x10GB)");
}
