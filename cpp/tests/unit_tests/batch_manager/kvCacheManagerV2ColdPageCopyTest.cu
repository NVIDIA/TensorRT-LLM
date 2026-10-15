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

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/batchedPageCopy.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/hostMem.h"

#include "kvCacheManagerV2TestUtils.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;
using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2::test;

#if CUDA_VERSION < 12080
TEST(KvCacheManagerV2ColdPageCopyTest, KernelFallbackCapturesMultipleIndexSegments)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    constexpr size_t kNumPairs = 1025;
    std::vector<PageIndexPair> expected(kNumPairs);
    for (size_t index = 0; index < expected.size(); ++index)
    {
        expected[index] = PageIndexPair{static_cast<int32_t>(index * 3), static_cast<int32_t>(index * 7)};
    }
    std::vector<PageIndexPair> input = expected;
    CudaAllocation device = allocateCuda(kNumPairs * sizeof(PageIndexPair));
    ASSERT_NE(device.get(), nullptr);

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    detail::copyPageIndicesToDeviceWithKernel(
        reinterpret_cast<CUdeviceptr>(device.get()), input.data(), input.size(), reinterpret_cast<CUstream>(stream));
    std::fill(input.begin(), input.end(), PageIndexPair{-1, -1});

    std::vector<PageIndexPair> actual(kNumPairs);
    ASSERT_EQ(
        cudaMemcpyAsync(actual.data(), device.get(), kNumPairs * sizeof(PageIndexPair), cudaMemcpyDeviceToHost, stream),
        cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    for (size_t index = 0; index < expected.size(); ++index)
    {
        EXPECT_EQ(actual[index].dst, expected[index].dst);
        EXPECT_EQ(actual[index].src, expected[index].src);
    }
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}
#endif

// Page sizes of a production hybrid-attention recipe: an attention page and the conv-state and SSM-state pages of
// its recurrent layers.
constexpr size_t kAttentionPageBytes = 884736;
constexpr size_t kConvStateBytes = 15261696;
constexpr size_t kSsmStateBytes = 217055232;

size_t roundUpToHostMemAlignment(size_t numBytes)
{
    return (numBytes + HostMem::kAlignment - 1) / HostMem::kAlignment * HostMem::kAlignment;
}

// Pseudo-random 64-bit words from a fixed seed, so no two slots of a buffer carry the same bytes and a permuted or
// slot-0 read cannot pass as a correct copy.
void fillPattern(unsigned char* bytes, size_t numBytes, uint64_t seed)
{
    std::mt19937_64 rng(seed);
    size_t const numWords = numBytes / sizeof(uint64_t);
    for (size_t i = 0; i < numWords; ++i)
    {
        uint64_t const word = rng();
        std::memcpy(bytes + i * sizeof(word), &word, sizeof(word));
    }
    for (size_t i = numWords * sizeof(uint64_t); i < numBytes; ++i)
    {
        bytes[i] = static_cast<unsigned char>(rng());
    }
}

// The index pairs in the memory space the copier wants: host memory on the copy-engine path, device memory on the
// kernel path.
class PairArray
{
public:
    PairArray(BatchedPageCopier const& copier, std::vector<PageIndexPair> pairs)
        : mHost(std::move(pairs))
    {
        if (copier.pageIndexLocation() == PageIndexLocation::kDevice)
        {
            mDevice = allocateCuda(mHost.size() * sizeof(PageIndexPair));
            if (mDevice.get() != nullptr)
            {
                EXPECT_EQ(cudaMemcpy(mDevice.get(), mHost.data(), mHost.size() * sizeof(PageIndexPair),
                              cudaMemcpyHostToDevice),
                    cudaSuccess);
            }
        }
    }

    [[nodiscard]] PageIndexPair const* data() const noexcept
    {
        return mDevice.get() != nullptr ? static_cast<PageIndexPair const*>(mDevice.get()) : mHost.data();
    }

private:
    std::vector<PageIndexPair> mHost;
    CudaAllocation mDevice;
};

PoolCopyArgs poolArgs(
    CUdeviceptr dstBase, CUdeviceptr srcBase, size_t pageBytes, PageIndexPair const* pairs, size_t numPairs)
{
    PoolCopyArgs args{};
    args.dstBase = dstBase;
    args.srcBase = srcBase;
    args.dstStride = pageBytes;
    args.srcStride = pageBytes;
    args.bytesPerPage = static_cast<uint32_t>(pageBytes);
    args.pairs = pairs;
    args.numPairs = static_cast<uint32_t>(numPairs);
    return args;
}

// The knob is read when a copier is constructed, so copiers on either side of it can coexist in one process.
TEST(KvCacheManagerV2BatchedPageCopierTest, PlatformSelectionFollowsTheKnob)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    bool const chunked = HostMem::shouldUseChunkedRegistration();
    {
        ScopedEnv unset(kPageCopyPreferCopyEngineEnv, std::nullopt);
        BatchedPageCopier const copier;
        EXPECT_EQ(copier.kernelEnabled(), copier.topology().coherentLink || chunked);
        EXPECT_EQ(
            copier.pageIndexLocation(), copier.kernelEnabled() ? PageIndexLocation::kDevice : PageIndexLocation::kHost);
    }
    {
        ScopedEnv prefer(kPageCopyPreferCopyEngineEnv, "1");
        BatchedPageCopier const copier;
        // The copy engines everywhere, except where chunked registration forbids them.
        EXPECT_EQ(copier.kernelEnabled(), chunked);
        EXPECT_EQ(copier.pageIndexLocation(), chunked ? PageIndexLocation::kDevice : PageIndexLocation::kHost);
    }
    {
        ScopedEnv other(kPageCopyPreferCopyEngineEnv, "yes"); // only "1" enables it
        BatchedPageCopier const copier;
        EXPECT_EQ(copier.kernelEnabled(), copier.topology().coherentLink || chunked);
    }
}

// With the knob set, launch() moves pages through the copy engines on a coherent link, index pairs in host memory
// as pageIndexLocation() demands, byte-exact and honouring a non-identity pair list. Where chunked registration keeps
// the kernel the same test exercises the kernel.
TEST(KvCacheManagerV2BatchedPageCopierTest, PreferredCopyEngineLaunchRoundTrips)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ScopedEnv prefault("TLLM_KV_CACHE_MANAGER_V2_PREFAULT_THREADS", "0");
    ScopedEnv thp("TLLM_KV_CACHE_MANAGER_V2_THP", "0");
    ScopedEnv prefer(kPageCopyPreferCopyEngineEnv, "1");
    BatchedPageCopier const copier;

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);

    struct Shape
    {
        size_t pageBytes;
        size_t numPages;
    };

    for (Shape const shape : {Shape{kSsmStateBytes, 1}, Shape{kConvStateBytes, 12}, Shape{kAttentionPageBytes, 140}})
    {
        size_t const total = shape.pageBytes * shape.numPages;
        HostMem host(roundUpToHostMemAlignment(total));
        HostMem hostBack(roundUpToHostMemAlignment(total));
        CudaAllocation device = allocateCuda(total);
        ASSERT_NE(device.get(), nullptr);
        auto* hostBytes = reinterpret_cast<unsigned char*>(host.address());
        auto* hostBackBytes = reinterpret_cast<unsigned char*>(hostBack.address());
        fillPattern(hostBytes, total, shape.numPages);
        std::memset(hostBackBytes, 0, total);
        ASSERT_EQ(cudaMemsetAsync(device.get(), 0xEE, total, stream), cudaSuccess);

        // Onboard with a permutation: device page (5 i + 3) mod n receives host page i. Offload with the identity, so
        // hostBack page (5 i + 3) mod n must equal host page i. 5 is coprime to 1, 12 and 140.
        std::vector<PageIndexPair> onboardPairs;
        std::vector<PageIndexPair> offloadPairs;
        std::vector<size_t> destinationOf(shape.numPages);
        for (size_t i = 0; i < shape.numPages; ++i)
        {
            destinationOf[i] = (5 * i + 3) % shape.numPages;
            onboardPairs.push_back(PageIndexPair{static_cast<int32_t>(destinationOf[i]), static_cast<int32_t>(i)});
            offloadPairs.push_back(PageIndexPair{static_cast<int32_t>(i), static_cast<int32_t>(i)});
        }
        PairArray const onboardIndex(copier, onboardPairs);
        PairArray const offloadIndex(copier, offloadPairs);

        auto const deviceBase = reinterpret_cast<CUdeviceptr>(device.get());
        copier.launch(poolArgs(deviceBase, static_cast<CUdeviceptr>(host.address()), shape.pageBytes,
                          onboardIndex.data(), shape.numPages),
            CopyDirection::kH2D, reinterpret_cast<CUstream>(stream));
        copier.launch(poolArgs(static_cast<CUdeviceptr>(hostBack.address()), deviceBase, shape.pageBytes,
                          offloadIndex.data(), shape.numPages),
            CopyDirection::kD2H, reinterpret_cast<CUstream>(stream));
        ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

        for (size_t i = 0; i < shape.numPages; ++i)
        {
            EXPECT_EQ(std::memcmp(hostBytes + i * shape.pageBytes, hostBackBytes + destinationOf[i] * shape.pageBytes,
                          shape.pageBytes),
                0)
                << "pageBytes=" << shape.pageBytes << " page " << i << " -> " << destinationOf[i];
        }
    }
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

// Informational only: GB/s of one 217 MB page per direction through the platform default and through the knob, from
// two copiers in the same process. Not asserted; both sit at the link ceiling on an idle GPU, the difference shows
// under SM load.
TEST(KvCacheManagerV2BatchedPageCopierTest, PathBandwidthInfo)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ScopedEnv prefault("TLLM_KV_CACHE_MANAGER_V2_PREFAULT_THREADS", "0");
    ScopedEnv thp("TLLM_KV_CACHE_MANAGER_V2_THP", "1");

    size_t const pageBytes = kSsmStateBytes;
    HostMem host(roundUpToHostMemAlignment(pageBytes));
    CudaAllocation device = allocateCuda(pageBytes);
    ASSERT_NE(device.get(), nullptr);
    std::memset(reinterpret_cast<void*>(host.address()), 0x3C, pageBytes);
    auto const deviceBase = reinterpret_cast<CUdeviceptr>(device.get());
    auto const hostBase = static_cast<CUdeviceptr>(host.address());

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    ASSERT_EQ(cudaEventCreate(&start), cudaSuccess);
    ASSERT_EQ(cudaEventCreate(&stop), cudaSuccess);

    auto measure = [&](char const* label, BatchedPageCopier const& copier)
    {
        std::vector<PageIndexPair> const identity{PageIndexPair{0, 0}};
        PairArray const index(copier, identity);
        auto const onboard = poolArgs(deviceBase, hostBase, pageBytes, index.data(), 1);
        auto const offload = poolArgs(hostBase, deviceBase, pageBytes, index.data(), 1);
        auto timeGbPerSec = [&](PoolCopyArgs const& args, CopyDirection direction) -> double
        {
            constexpr int kWarmup = 3;
            constexpr int kIters = 10;
            for (int i = 0; i < kWarmup; ++i)
            {
                copier.launch(args, direction, reinterpret_cast<CUstream>(stream));
            }
            EXPECT_EQ(cudaEventRecord(start, stream), cudaSuccess);
            for (int i = 0; i < kIters; ++i)
            {
                copier.launch(args, direction, reinterpret_cast<CUstream>(stream));
            }
            EXPECT_EQ(cudaEventRecord(stop, stream), cudaSuccess);
            EXPECT_EQ(cudaEventSynchronize(stop), cudaSuccess);
            float ms = 0.0F;
            EXPECT_EQ(cudaEventElapsedTime(&ms, start, stop), cudaSuccess);
            return static_cast<double>(pageBytes) * kIters / (static_cast<double>(ms) * 1e-3) / 1e9;
        };
        double const h2d = timeGbPerSec(onboard, CopyDirection::kH2D);
        double const d2h = timeGbPerSec(offload, CopyDirection::kD2H);
        std::cout << "PathBandwidthInfo: " << label << " path=" << (copier.kernelEnabled() ? "kernel" : "copy-engine")
                  << " coherentLink=" << copier.topology().coherentLink
                  << " ctas(d2h/h2d)=" << copier.config(CopyDirection::kD2H).ctas << "/"
                  << copier.config(CopyDirection::kH2D).ctas << ", " << pageBytes << " B page: host->device " << h2d
                  << " GB/s, device->host " << d2h << " GB/s" << std::endl;
    };
    {
        ScopedEnv unset(kPageCopyPreferCopyEngineEnv, std::nullopt);
        BatchedPageCopier const copier;
        measure("default", copier);
    }
    {
        ScopedEnv prefer(kPageCopyPreferCopyEngineEnv, "1");
        BatchedPageCopier const copier;
        measure("prefer-copy-engine", copier);
    }

    ASSERT_EQ(cudaEventDestroy(start), cudaSuccess);
    ASSERT_EQ(cudaEventDestroy(stop), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

} // namespace
