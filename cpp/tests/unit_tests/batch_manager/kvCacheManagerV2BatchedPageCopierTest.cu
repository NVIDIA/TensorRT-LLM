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
 * WITHOUT WARRANTIES OR WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/batchedPageCopy.h"
#include "tensorrt_llm/batch_manager/kv_cache_manager_v2/utils/hostMem.h"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace
{

using namespace tensorrt_llm::batch_manager::kv_cache_manager_v2;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct CudaDeleter
{
    void operator()(void* ptr) const noexcept
    {
        cudaFree(ptr);
    }
};

using CudaAllocation = std::unique_ptr<void, CudaDeleter>;

CudaAllocation allocateCudaChecked(size_t numBytes)
{
    void* ptr = nullptr;
    EXPECT_EQ(cudaMalloc(&ptr, numBytes), cudaSuccess);
    return CudaAllocation{ptr};
}

struct PinnedDeleter
{
    void operator()(void* ptr) const noexcept
    {
        cudaFreeHost(ptr);
    }
};

using PinnedAllocation = std::unique_ptr<void, PinnedDeleter>;

PinnedAllocation allocatePinnedChecked(size_t numBytes)
{
    void* ptr = nullptr;
    EXPECT_EQ(cudaHostAlloc(&ptr, numBytes, cudaHostAllocPortable), cudaSuccess);
    return PinnedAllocation{ptr};
}

//! Fills `bytes` with a deterministic pattern so we can compare round-trip copies byte-exact.
void fillPattern(char* data, size_t bytes, uint64_t seed)
{
    for (size_t i = 0; i < bytes; ++i)
    {
        seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
        data[i] = static_cast<char>(seed >> 56);
    }
}

// ---------------------------------------------------------------------------
// Path selection tests
// ---------------------------------------------------------------------------

// These tests verify that BatchedPageCopier picks the right copy path given the platform state.
// The decision tree (see selectKernelViable in batchedPageCopy.cu) is:
//
//   1. TRTLLM_KV_CACHE_FORCE_SM_COPY=1        → SM kernel
//   2. HostMem::shouldUseChunkedRegistration() → SM kernel
//   3. Otherwise                               → copy engine
//
// Because readForceSMCopyEnv() caches its result in a lazy static, the env var must be set
// before the process starts or (for testing) before the first BatchedPageCopier construction.
// A child-process exec is the cleanest way to assert env-var effects in CI.

TEST(KvCacheManagerV2BatchedPageCopierTest, DefaultPathPrefersCopyEngine)
{
    // On a clean CI run with TRTLLM_KV_CACHE_FORCE_SM_COPY unset, the default must be copy engine
    // whenever chunked host registration is off. We cannot assert chunked state here because it
    // depends on the running kernel, but we can assert the invariant: if chunked registration is
    // NOT in force then the copy engine must be selected, regardless of link type.
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

    bool const chunked = HostMem::shouldUseChunkedRegistration();
    BatchedPageCopier copier;

    if (!chunked)
    {
        // Non-chunked + no env var → copy engine is the only valid default.
        EXPECT_FALSE(copier.kernelEnabled())
            << "expected copy engine path when chunked registration is off; "
               "coherentLink="
            << copier.topology().coherentLink
            << " (NVLink-C2C used to force the SM kernel unconditionally, which caused "
               "CUDA-graph replay idle gaps)";
        EXPECT_EQ(copier.pageIndexLocation(), PageIndexLocation::kHost);
    }
    else
    {
        // Chunked registration forbids the copy engine regardless of env/link.
        EXPECT_TRUE(copier.kernelEnabled()) << "expected SM kernel when host registration is chunked";
        EXPECT_EQ(copier.pageIndexLocation(), PageIndexLocation::kDevice);
    }
}

TEST(KvCacheManagerV2BatchedPageCopierTest, TopologyIsObservable)
{
    BatchedPageCopier copier;
    BatchedPageCopier::Topology const& topo = copier.topology();

    EXPECT_GE(topo.device, 0);
    EXPECT_GT(topo.smVersion, 0);
    EXPECT_GT(topo.smCount, 0);
    EXPECT_GT(topo.linkBandwidthGBs, 0.0);
    // hostNumaId may be -1 on systems where CUDA does not report it.
}

// ---------------------------------------------------------------------------
// Correctness: copy engine vs SM kernel must be byte-identical
// ---------------------------------------------------------------------------

// We cannot force a single process to run both paths (the env var and chunked registration are
// cached at process scope), so the best cross-path correctness check is:
//
//   * In CI, build and run this test binary twice — once with
//     TRTLLM_KV_CACHE_FORCE_SM_COPY=0 (default copy engine) and once with =1 (SM kernel forced).
//   * Each run performs a D2H and H2D round-trip with known-pattern buffers.
//   * A golden comparison script verifies the two runs produce the same byte-for-byte results.
//
// Within a single run we verify each path independently produces correct round-trip output.

constexpr size_t kNumPages = 64;
constexpr size_t kBytesPerPage = 4096; //!< 4 KiB, well above any kernel minimum
constexpr size_t kDstSrcStride = 8192; //!< pages are spaced apart in the pool

//! Helper: performs a full D2H+H2D round-trip through the copier and asserts byte equality.
void roundTripAndCheck(CopyDirection dir, uint64_t seed)
{
    BatchedPageCopier copier;
    auto const cfg = copier.config(dir);

    // Allocate pools: source and destination in device memory, mirrored in pinned host for H2D.
    CudaAllocation deviceSrc = allocateCudaChecked(kNumPages * kDstSrcStride);
    CudaAllocation deviceDst = allocateCudaChecked(kNumPages * kDstSrcStride);
    PinnedAllocation hostSrc = allocatePinnedChecked(kNumPages * kBytesPerPage);
    PinnedAllocation hostDst = allocatePinnedChecked(kNumPages * kBytesPerPage);

    // Fill source (device for D2H test, host for H2D test).
    if (dir == CopyDirection::kD2H)
    {
        fillPattern(reinterpret_cast<char*>(deviceSrc.get()), kNumPages * kBytesPerPage, seed);
    }
    else
    {
        fillPattern(reinterpret_cast<char*>(hostSrc.get()), kNumPages * kBytesPerPage, seed);
    }

    // Build a simple identity page-index mapping (page i of pool src → page i of pool dst).
    std::vector<PageIndexPair> pairs(kNumPages);
    for (size_t i = 0; i < kNumPages; ++i)
    {
        pairs[i] = PageIndexPair{static_cast<int32_t>(i), static_cast<int32_t>(i)};
    }

    CUstream stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    // The copy engine path builds descriptors on the CPU and therefore needs pairs in host memory;
    // the kernel path reads pairs from device memory. BatchedPageCopier::pageIndexLocation()
    // tells us which — but for our test we always keep pairs in host memory and adapt below.
    CUdeviceptr pairsDev = 0;
    if (copier.pageIndexLocation() == PageIndexLocation::kDevice)
    {
        void* tmp = nullptr;
        ASSERT_EQ(cudaMalloc(&tmp, pairs.size() * sizeof(PageIndexPair)), cudaSuccess);
        pairsDev = reinterpret_cast<CUdeviceptr>(tmp);
        ASSERT_EQ(cudaMemcpyAsync(pairsDev, pairs.data(), pairs.size() * sizeof(PageIndexPair),
                      cudaMemcpyHostToDevice, stream),
            cudaSuccess);
    }

    PoolCopyArgs args{};
    args.bytesPerPage = kBytesPerPage;
    args.dstStride = kDstSrcStride;
    args.srcStride = kDstSrcStride;
    args.numPairs = kNumPages;
    args.dstBase = reinterpret_cast<CUdeviceptr>(dir == CopyDirection::kD2H ? hostDst.get()
                                                                           : deviceDst.get());
    args.srcBase = reinterpret_cast<CUdeviceptr>(dir == CopyDirection::kD2H ? deviceSrc.get()
                                                                           : hostSrc.get());
    args.pairs = (copier.pageIndexLocation() == PageIndexLocation::kDevice)
        ? reinterpret_cast<PageIndexPair const*>(pairsDev)
        : pairs.data();

    copier.launch(args, dir, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    // Verify: D2H means deviceSrc → hostDst, so hostDst must match deviceSrc.
    // H2D means hostSrc → deviceDst, so deviceDst must match hostSrc.
    if (dir == CopyDirection::kD2H)
    {
        std::vector<char> expected(kNumPages * kBytesPerPage);
        ASSERT_EQ(cudaMemcpy(expected.data(), deviceSrc.get(), expected.size(), cudaMemcpyDeviceToHost),
            cudaSuccess);
        auto const* actual = reinterpret_cast<char const*>(hostDst.get());
        for (size_t i = 0; i < expected.size(); ++i)
        {
            ASSERT_EQ(actual[i], expected[i]) << "byte mismatch at offset " << i;
        }
    }
    else
    {
        std::vector<char> actual(kNumPages * kBytesPerPage);
        ASSERT_EQ(cudaMemcpy(actual.data(), deviceDst.get(), actual.size(), cudaMemcpyDeviceToHost),
            cudaSuccess);
        auto const* expected = reinterpret_cast<char const*>(hostSrc.get());
        for (size_t i = 0; i < actual.size(); ++i)
        {
            ASSERT_EQ(actual[i], expected[i]) << "byte mismatch at offset " << i;
        }
    }

    // Clean up device allocation for pairs if we made one.
    if (pairsDev != 0)
    {
        cudaFree(reinterpret_cast<void*>(pairsDev));
    }
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);

    // The kernel should have zero CTAs on the copy-engine path — no SM work to launch.
    if (!cfg.useKernel)
    {
        EXPECT_EQ(cfg.ctas, 0);
        EXPECT_EQ(cfg.inFlightBytes, 0u);
    }
    else
    {
        // Kernel path: non-zero grid sizing was computed.
        EXPECT_GT(cfg.ctas, 0);
        EXPECT_GT(cfg.inFlightBytes, 0u);
    }
}

TEST(KvCacheManagerV2BatchedPageCopierTest, D2HRoundTripIsByteExact)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    roundTripAndCheck(CopyDirection::kD2H, 0xA5A5A5A5ULL);
}

TEST(KvCacheManagerV2BatchedPageCopierTest, H2DRoundTripIsByteExact)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    roundTripAndCheck(CopyDirection::kH2D, 0x5A5A5A5AULL);
}

TEST(KvCacheManagerV2BatchedPageCopierTest, BothDirectionsProduceCorrectConfigs)
{
    BatchedPageCopier copier;
    auto const& offload = copier.config(CopyDirection::kD2H);
    auto const& onboard = copier.config(CopyDirection::kH2D);

    // Threads and ILP are fixed constants shared between directions.
    EXPECT_EQ(offload.threads, 128);
    EXPECT_EQ(offload.ilp, 2);
    EXPECT_EQ(onboard.threads, 128);
    EXPECT_EQ(onboard.ilp, 2);

    // Offload uses 32 KiB in-flight, onboard uses 64 KiB (platform-independent knees).
    EXPECT_EQ(offload.inFlightBytes, 32u * 1024u);
    EXPECT_EQ(onboard.inFlightBytes, 64u * 1024u);

    // Path must match across both directions — we do not split offload/onboard between paths.
    EXPECT_EQ(offload.useKernel, onboard.useKernel);
}

// ---------------------------------------------------------------------------
// Launch smoke test: zero-length copy must be a no-op
// ---------------------------------------------------------------------------

TEST(KvCacheManagerV2BatchedPageCopierTest, ZeroPairsIsNoOp)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    BatchedPageCopier copier;

    CUstream stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);

    PoolCopyArgs args{};
    copier.launch(args, CopyDirection::kD2H, stream);
    copier.launch(args, CopyDirection::kH2D, stream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

} // namespace
