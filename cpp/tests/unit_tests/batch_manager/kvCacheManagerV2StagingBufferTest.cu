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

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/coldPageCodec.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/coldPageCopy.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/stagingBuffer.h"
#include "tensorrt_llm/common/tllmException.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

struct CudaDeleter
{
    void operator()(void* ptr) const noexcept
    {
        cudaFree(ptr);
    }
};

using CudaAllocation = std::unique_ptr<void, CudaDeleter>;

CudaAllocation allocateCuda(size_t numBytes)
{
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, numBytes) != cudaSuccess)
    {
        return CudaAllocation{};
    }
    return CudaAllocation{ptr};
}

struct HostGate
{
    std::atomic<bool> open{false};

    static void CUDART_CB wait(void* data)
    {
        auto& gate = *static_cast<HostGate*>(data);
        while (!gate.open.load(std::memory_order_acquire))
        {
            std::this_thread::yield();
        }
    }

    void release() noexcept
    {
        open.store(true, std::memory_order_release);
    }
};

struct HostGateReleaseGuard
{
    HostGateReleaseGuard(HostGate& first, HostGate& second, HostGate& third)
        : first(first)
        , second(second)
        , third(third)
    {
    }

    ~HostGateReleaseGuard()
    {
        releaseAll();
        cudaDeviceSynchronize();
    }

    void releaseAll() noexcept
    {
        first.release();
        second.release();
        third.release();
    }

    HostGate& first;
    HostGate& second;
    HostGate& third;
};

__global__ void copyPageIndexPairs(PageIndexPair const* input, PageIndexPair* output, size_t count)
{
    size_t const index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count)
    {
        output[index] = input[index];
    }
}

TEST(KvCacheManagerV2StagingBufferTest, SynchronousAcquireProtectsPinnedHostOverwrite)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    constexpr size_t kBytes = 4096;
    StagingBufferManager manager(kBytes, StagingBufferMemory::kPinnedHost);
    EXPECT_EQ(manager.memory(), StagingBufferMemory::kPinnedHost);
    EXPECT_NE(manager.baseAddress(), 0);
    CudaAllocation device = allocateCuda(kBytes);
    ASSERT_NE(device, nullptr);

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    std::vector<uint8_t> expected(kBytes);
    for (size_t index = 0; index < expected.size(); ++index)
    {
        expected[index] = static_cast<uint8_t>(index);
    }

    {
        auto staging = manager.acquire(kBytes, kBytes, 1, 1, std::nullopt);
        EXPECT_FALSE(staging.stream().has_value());
        std::memcpy(reinterpret_cast<void*>(staging.address()), expected.data(), expected.size());
        staging.setStream(reinterpret_cast<CUstream>(stream));
        ASSERT_TRUE(staging.stream().has_value());
        ASSERT_EQ(cudaMemcpyAsync(device.get(), reinterpret_cast<void const*>(staging.address()), kBytes,
                      cudaMemcpyHostToDevice, stream),
            cudaSuccess);
    }

    {
        auto staging = manager.acquire(kBytes, kBytes, 1, 1, std::nullopt);
        EXPECT_FALSE(staging.stream().has_value());
        std::memset(reinterpret_cast<void*>(staging.address()), 0, kBytes);
    }

    std::vector<uint8_t> actual(kBytes);
    ASSERT_EQ(cudaMemcpy(actual.data(), device.get(), kBytes, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(actual, expected);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(KvCacheManagerV2StagingBufferTest, DeviceBackingSupportsStreamAndSynchronousHandoffs)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    constexpr size_t kBytes = 4096;
    constexpr uint8_t kPattern = 0x5A;
    StagingBufferManager manager(kBytes, StagingBufferMemory::kDevice);
    EXPECT_EQ(manager.memory(), StagingBufferMemory::kDevice);
    EXPECT_NE(manager.baseAddress(), 0);

    cudaStream_t firstStream = nullptr;
    cudaStream_t secondStream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&firstStream), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&secondStream), cudaSuccess);

    std::vector<uint8_t> actual(kBytes);
    {
        auto staging = manager.acquire(kBytes, kBytes, 1, 1, reinterpret_cast<CUstream>(firstStream));
        ASSERT_EQ(
            cudaMemsetAsync(reinterpret_cast<void*>(staging.address()), kPattern, kBytes, firstStream), cudaSuccess);
        staging.setStream(reinterpret_cast<CUstream>(secondStream));
        ASSERT_EQ(cudaMemcpyAsync(actual.data(), reinterpret_cast<void const*>(staging.address()), kBytes,
                      cudaMemcpyDeviceToHost, secondStream),
            cudaSuccess);
    }

    {
        auto staging = manager.acquire(kBytes, kBytes, 1, 1, std::nullopt);
        EXPECT_FALSE(staging.stream().has_value());
    }
    EXPECT_EQ(actual, std::vector<uint8_t>(kBytes, kPattern));

    {
        auto staging = manager.acquire(kBytes, kBytes, 1, 1, reinterpret_cast<CUstream>(firstStream));
        ASSERT_EQ(cudaMemsetAsync(reinterpret_cast<void*>(staging.address()), 0x33, kBytes, firstStream), cudaSuccess);
        staging.setStream(std::nullopt);
        EXPECT_FALSE(staging.stream().has_value());
    }

    ASSERT_EQ(cudaStreamDestroy(firstStream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(secondStream), cudaSuccess);
}

TEST(KvCacheManagerV2StagingBufferTest, EphemeralHostIndicesUploadIntoDeviceRing)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    constexpr size_t kNumPairs = 20000;
    std::vector<PageIndexPair> expected(kNumPairs);
    for (size_t index = 0; index < expected.size(); ++index)
    {
        expected[index] = PageIndexPair{static_cast<int32_t>(index * 3), static_cast<int32_t>(index * 7)};
    }
    std::vector<PageIndexPair> input = expected;
    CudaAllocation output = allocateCuda(expected.size() * sizeof(PageIndexPair));
    ASSERT_NE(output, nullptr);

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    constexpr size_t kChunkBytes = 64u << 10u;
    StagingBufferManager deviceManager(256u << 10u, StagingBufferMemory::kDevice);
    size_t offset = 0;
    while (offset < input.size())
    {
        size_t const remainingBytes = (input.size() - offset) * sizeof(PageIndexPair);
        auto device = deviceManager.acquire(sizeof(PageIndexPair), std::min(remainingBytes, kChunkBytes),
            sizeof(PageIndexPair), alignof(PageIndexPair), reinterpret_cast<CUstream>(stream));
        size_t const chunkPairs = std::min(input.size() - offset, device.size() / sizeof(PageIndexPair));
        detail::copyPageIndicesToDevice(static_cast<CUdeviceptr>(device.address()), input.data() + offset, chunkPairs,
            reinterpret_cast<CUstream>(stream));

        // The helper captures the pageable host source before returning.
        std::fill_n(input.data() + offset, chunkPairs, PageIndexPair{-1, -1});

        constexpr uint32_t kThreads = 256;
        uint32_t const blocks = static_cast<uint32_t>((chunkPairs + kThreads - 1) / kThreads);
        copyPageIndexPairs<<<blocks, kThreads, 0, stream>>>(reinterpret_cast<PageIndexPair const*>(device.address()),
            static_cast<PageIndexPair*>(output.get()) + offset, chunkPairs);
        ASSERT_EQ(cudaGetLastError(), cudaSuccess);
        offset += chunkPairs;
    }

    std::vector<PageIndexPair> actual(kNumPairs);
    ASSERT_EQ(cudaMemcpyAsync(
                  actual.data(), output.get(), actual.size() * sizeof(PageIndexPair), cudaMemcpyDeviceToHost, stream),
        cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (size_t index = 0; index < expected.size(); ++index)
    {
        EXPECT_EQ(actual[index].dst, expected[index].dst);
        EXPECT_EQ(actual[index].src, expected[index].src);
    }
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST(KvCacheManagerV2StagingBufferTest, PacksAndAlignsByteRangesWithUnitGranularity)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    StagingBufferManager manager(64, StagingBufferMemory::kPinnedHost);
    auto first = manager.acquire(7, 7, 1, 1, std::nullopt);
    auto second = manager.acquire(9, 9, 1, 1, std::nullopt);
    auto aligned = manager.acquire(16, 16, 16, 16, std::nullopt);

    EXPECT_EQ(second.address(), first.address() + first.size());
    EXPECT_EQ(aligned.address() % 16, 0);
    EXPECT_EQ(first.size(), 7);
    EXPECT_EQ(second.size(), 9);
    EXPECT_EQ(aligned.size(), 16);
}

TEST(KvCacheManagerV2StagingBufferTest, ReclaimsInOrderAndWrapsAtByteBoundary)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    StagingBufferManager manager(32, StagingBufferMemory::kPinnedHost);
    MemAddress const baseAddress = manager.baseAddress();
    {
        auto first = manager.acquire(20, 20, 1, 1, std::nullopt);
        EXPECT_EQ(first.address(), baseAddress);
    }
    {
        auto second = manager.acquire(8, 8, 1, 1, std::nullopt);
        EXPECT_EQ(second.address(), baseAddress + 20);
    }

    auto wrapped = manager.acquire(8, 8, 1, 1, std::nullopt);
    EXPECT_EQ(wrapped.address(), baseAddress);
}

TEST(KvCacheManagerV2StagingBufferTest, SkipsLiveRangeAndUsesLaterRetiredRun)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    StagingBufferManager manager(64, StagingBufferMemory::kPinnedHost);
    MemAddress const baseAddress = manager.baseAddress();
    auto first = std::make_unique<StagingBuffer>(manager, 16, 16, 1, 1, std::nullopt);
    {
        auto tail = manager.acquire(48, 48, 1, 1, std::nullopt);
        EXPECT_EQ(tail.address(), baseAddress + 16);
    }

    auto skipped = manager.acquire(8, 8, 1, 1, std::nullopt);
    EXPECT_EQ(skipped.address(), baseAddress + 16);
}

TEST(KvCacheManagerV2StagingBufferTest, RejectsExhaustionWithoutChangingAllocationCursor)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    StagingBufferManager manager(32, StagingBufferMemory::kPinnedHost);
    MemAddress const baseAddress = manager.baseAddress();
    auto full = std::make_unique<StagingBuffer>(manager, 32, 32, 1, 1, std::nullopt);

    EXPECT_THROW(manager.acquire(1, 1, 1, 1, std::nullopt), tensorrt_llm::common::TllmException);
    full.reset();

    auto reused = manager.acquire(32, 32, 1, 1, std::nullopt);
    EXPECT_EQ(reused.address(), baseAddress);
}

TEST(KvCacheManagerV2StagingBufferTest, AlignmentPaddingPreservesEachPreviousRangeEvent)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    using namespace std::chrono_literals;

    constexpr size_t kBytes = 96;
    constexpr size_t kAlignment = 32;
    StagingBufferManager manager(kBytes, StagingBufferMemory::kDevice);
    MemAddress const baseAddress = manager.baseAddress();
    ASSERT_EQ(baseAddress % kAlignment, 0);

    cudaStream_t firstStream = nullptr;
    cudaStream_t secondStream = nullptr;
    cudaStream_t thirdStream = nullptr;
    cudaStream_t setupStream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&firstStream), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&secondStream), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&thirdStream), cudaSuccess);
    ASSERT_EQ(cudaStreamCreate(&setupStream), cudaSuccess);

    HostGate firstGate;
    HostGate secondGate;
    HostGate thirdGate;
    HostGateReleaseGuard gateGuard(firstGate, secondGate, thirdGate);

    {
        auto first = manager.acquire(8, 8, 1, 1, reinterpret_cast<CUstream>(firstStream));
        EXPECT_EQ(first.address(), baseAddress);
        ASSERT_EQ(cudaLaunchHostFunc(firstStream, HostGate::wait, &firstGate), cudaSuccess);
    }
    {
        auto second = manager.acquire(8, 8, 1, 1, reinterpret_cast<CUstream>(secondStream));
        EXPECT_EQ(second.address(), baseAddress + 8);
        ASSERT_EQ(cudaLaunchHostFunc(secondStream, HostGate::wait, &secondGate), cudaSuccess);
    }
    {
        auto third = manager.acquire(80, 80, 1, 1, reinterpret_cast<CUstream>(thirdStream));
        EXPECT_EQ(third.address(), baseAddress + 16);
        ASSERT_EQ(cudaLaunchHostFunc(thirdStream, HostGate::wait, &thirdGate), cudaSuccess);
    }

    // Reuse three bytes from the first old range, then align the next allocation to byte 32.
    // The skipped [3, 32) interval crosses all three old temporal ranges.
    {
        auto prefix = manager.acquire(3, 3, 1, 1, reinterpret_cast<CUstream>(setupStream));
        EXPECT_EQ(prefix.address(), baseAddress);
    }
    auto aligned = std::make_unique<StagingBuffer>(
        manager, kAlignment, kAlignment, kAlignment, kAlignment, reinterpret_cast<CUstream>(setupStream));
    EXPECT_EQ(aligned->address(), baseAddress + kAlignment);
    {
        auto suffix = manager.acquire(32, 32, 1, 1, reinterpret_cast<CUstream>(setupStream));
        EXPECT_EQ(suffix.address(), baseAddress + 64);
    }
    {
        auto prefix = manager.acquire(3, 3, 1, 1, reinterpret_cast<CUstream>(setupStream));
        EXPECT_EQ(prefix.address(), baseAddress);
    }

    auto verifyPaddingFragment = [&](size_t size, MemAddress expectedAddress, HostGate& gate)
    {
        auto acquisition = std::async(std::launch::async,
            [&manager, size]
            {
                if (cudaSetDevice(0) != cudaSuccess)
                {
                    return MemAddress{0};
                }
                auto staging = manager.acquire(size, size, 1, 1, std::nullopt);
                return staging.address();
            });

        EXPECT_EQ(acquisition.wait_for(100ms), std::future_status::timeout);
        gate.release();
        std::future_status const status = acquisition.wait_for(30s);
        EXPECT_EQ(status, std::future_status::ready);
        if (status != std::future_status::ready)
        {
            gateGuard.releaseAll();
        }
        EXPECT_EQ(acquisition.get(), expectedAddress);
    };

    verifyPaddingFragment(5, baseAddress + 3, firstGate);
    verifyPaddingFragment(8, baseAddress + 8, secondGate);
    verifyPaddingFragment(16, baseAddress + 16, thirdGate);

    aligned.reset();

    gateGuard.releaseAll();
    ASSERT_EQ(cudaStreamDestroy(setupStream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(thirdStream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(secondStream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(firstStream), cudaSuccess);
}

TEST(KvCacheManagerV2StagingBufferTest, RoundsAvailableSizeToGranularity)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    StagingBufferManager manager(64, StagingBufferMemory::kPinnedHost);
    auto prefix = manager.acquire(5, 5, 1, 1, std::nullopt);
    auto rounded = manager.acquire(12, 60, 6, 1, std::nullopt);
    EXPECT_EQ(rounded.size(), 54);
    EXPECT_EQ(rounded.size() % 6, 0);
}

TEST(KvCacheManagerV2StagingBufferTest, RejectsInvalidRangeRequests)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    EXPECT_THROW((StagingBufferManager{0, StagingBufferMemory::kPinnedHost}), tensorrt_llm::common::TllmException);
    EXPECT_THROW((StagingBufferManager{4096, static_cast<StagingBufferMemory>(-1)}), std::invalid_argument);

    StagingBufferManager manager(4096, StagingBufferMemory::kPinnedHost);
    EXPECT_THROW(manager.acquire(0, 1, 1, 1, std::nullopt), tensorrt_llm::common::TllmException);
    EXPECT_THROW(manager.acquire(4097, 4097, 1, 1, std::nullopt), tensorrt_llm::common::TllmException);
    EXPECT_THROW(manager.acquire(2, 1, 1, 1, std::nullopt), tensorrt_llm::common::TllmException);
    EXPECT_THROW(manager.acquire(1, 1, 0, 1, std::nullopt), tensorrt_llm::common::TllmException);
    EXPECT_THROW(manager.acquire(1, 1, 1, 0, std::nullopt), tensorrt_llm::common::TllmException);
    EXPECT_THROW(manager.acquire(9, 15, 8, 1, std::nullopt), tensorrt_llm::common::TllmException);
    EXPECT_THROW(manager.acquire(1, 1, 1, 3, std::nullopt), tensorrt_llm::common::TllmException);
}

TEST(KvCacheManagerV2StagingBufferTest, DestructionWaitsForRetiredEvents)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    using namespace std::chrono_literals;

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    HostGate gate;
    auto manager = std::make_shared<StagingBufferManager>(4096, StagingBufferMemory::kPinnedHost);
    std::weak_ptr<StagingBufferManager> weakManager = manager;
    {
        auto staging = manager->acquire(4096, 4096, 1, 1, reinterpret_cast<CUstream>(stream));
        ASSERT_EQ(cudaLaunchHostFunc(stream, HostGate::wait, &gate), cudaSuccess);
    }

    auto destroyManager = std::async(std::launch::async, [manager = std::move(manager)]() mutable { manager.reset(); });
    EXPECT_EQ(destroyManager.wait_for(100ms), std::future_status::timeout);
    gate.release();
    ASSERT_EQ(destroyManager.wait_for(30s), std::future_status::ready);
    destroyManager.get();
    EXPECT_TRUE(weakManager.expired());
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

} // namespace
