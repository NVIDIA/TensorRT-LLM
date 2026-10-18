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

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#if CUDA_VERSION < 12080
#include <algorithm>
#endif
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
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

struct CudaHostDeleter
{
    void operator()(void* ptr) const noexcept
    {
        cudaFreeHost(ptr);
    }
};

using CudaHostAllocation = std::unique_ptr<void, CudaHostDeleter>;

CudaHostAllocation allocateMappedHost(size_t numBytes)
{
    void* ptr = nullptr;
    if (cudaHostAlloc(&ptr, numBytes, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess)
    {
        return CudaHostAllocation{};
    }
    return CudaHostAllocation{ptr};
}

class ScopedEnv
{
public:
    ScopedEnv(char const* name, std::optional<std::string> value)
        : mName(name)
    {
        if (char const* oldValue = std::getenv(name); oldValue != nullptr)
        {
            mOldValue = oldValue;
        }
        if (value.has_value())
        {
            if (::setenv(name, value->c_str(), /*overwrite=*/1) != 0)
            {
                throw std::system_error(errno, std::generic_category(), "setenv failed");
            }
        }
        else if (::unsetenv(name) != 0)
        {
            throw std::system_error(errno, std::generic_category(), "unsetenv failed");
        }
    }

    ScopedEnv(ScopedEnv const&) = delete;
    ScopedEnv& operator=(ScopedEnv const&) = delete;

    ~ScopedEnv()
    {
        if (mOldValue.has_value())
        {
            ::setenv(mName.c_str(), mOldValue->c_str(), /*overwrite=*/1);
        }
        else
        {
            ::unsetenv(mName.c_str());
        }
    }

private:
    std::string mName;
    std::optional<std::string> mOldValue;
};

TEST(KvCacheManagerV2ColdPageCopyTest, CopyEngineIsDefaultOnEveryLink)
{
    for (bool const coherentLink : {false, true})
    {
        EXPECT_FALSE(detail::selectCopyKernel(coherentLink, /*chunkedRegistration=*/false,
            /*kernelOnCoherentLink=*/false))
            << "coherentLink=" << coherentLink;
    }
}

TEST(KvCacheManagerV2ColdPageCopyTest, CopyKernelOptInAppliesOnlyToCoherentLinks)
{
    EXPECT_TRUE(detail::selectCopyKernel(/*coherentLink=*/true, /*chunkedRegistration=*/false,
        /*kernelOnCoherentLink=*/true));
    EXPECT_FALSE(detail::selectCopyKernel(/*coherentLink=*/false, /*chunkedRegistration=*/false,
        /*kernelOnCoherentLink=*/true));
}

TEST(KvCacheManagerV2ColdPageCopyTest, ChunkedRegistrationAlwaysSelectsCopyKernel)
{
    for (bool const coherentLink : {false, true})
    {
        for (bool const kernelOnCoherentLink : {false, true})
        {
            EXPECT_TRUE(detail::selectCopyKernel(coherentLink, /*chunkedRegistration=*/true, kernelOnCoherentLink))
                << "coherentLink=" << coherentLink << " kernelOnCoherentLink=" << kernelOnCoherentLink;
        }
    }
}

class BatchedPageCopierPathTest : public ::testing::TestWithParam<std::optional<std::string>>
{
};

TEST_P(BatchedPageCopierPathTest, SelectsPathAndRoundTripsPages)
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ScopedEnv env(detail::kCoherentCopyKernelEnv, GetParam());
    BatchedPageCopier const copier;

    bool const kernelOnCoherentLink = GetParam().has_value() && *GetParam() == "1";
    bool const expectKernel = detail::selectCopyKernel(
        copier.topology().coherentLink, HostMem::shouldUseChunkedRegistration(), kernelOnCoherentLink);
    EXPECT_EQ(copier.kernelEnabled(), expectKernel);
    EXPECT_EQ(copier.config(CopyDirection::kD2H).useKernel, expectKernel);
    EXPECT_EQ(copier.config(CopyDirection::kH2D).useKernel, expectKernel);
    EXPECT_EQ(copier.pageIndexLocation(), expectKernel ? PageIndexLocation::kDevice : PageIndexLocation::kHost);

    constexpr uint32_t kBytesPerPage = 64U * 1024U;
    constexpr int32_t kNumPages = 8;
    constexpr size_t kPoolBytes = static_cast<size_t>(kBytesPerPage) * kNumPages;
    constexpr size_t kNumWords = kPoolBytes / sizeof(uint64_t);

    std::vector<uint64_t> expected(kNumWords);
    for (size_t index = 0; index < kNumWords; ++index)
    {
        expected[index] = (static_cast<uint64_t>(index) * 0x9E3779B97F4A7C15ULL) ^ 0xD1B54A32D192ED03ULL;
    }

    CudaAllocation deviceSrc = allocateCuda(kPoolBytes);
    CudaAllocation deviceDst = allocateCuda(kPoolBytes);
    CudaHostAllocation host = allocateMappedHost(kPoolBytes);
    ASSERT_NE(deviceSrc, nullptr);
    ASSERT_NE(deviceDst, nullptr);
    ASSERT_NE(host, nullptr);
    void* hostDevicePtr = nullptr;
    ASSERT_EQ(cudaHostGetDevicePointer(&hostDevicePtr, host.get(), 0), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(deviceSrc.get(), expected.data(), kPoolBytes, cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(deviceDst.get(), 0, kPoolBytes), cudaSuccess);

    // Offload page i to host slot (kNumPages - 1 - i), then onboard host slot j to device page j
    // through a second permutation; the composition maps every page back to its own index.
    std::vector<PageIndexPair> offloadPairs(kNumPages);
    std::vector<PageIndexPair> onboardPairs(kNumPages);
    for (int32_t page = 0; page < kNumPages; ++page)
    {
        offloadPairs[page] = PageIndexPair{kNumPages - 1 - page, page};
        onboardPairs[page] = PageIndexPair{page, kNumPages - 1 - page};
    }

    CudaAllocation devicePairs = allocateCuda(2 * kNumPages * sizeof(PageIndexPair));
    ASSERT_NE(devicePairs, nullptr);
    PageIndexPair const* offloadArg = offloadPairs.data();
    PageIndexPair const* onboardArg = onboardPairs.data();
    if (copier.pageIndexLocation() == PageIndexLocation::kDevice)
    {
        auto* staged = static_cast<PageIndexPair*>(devicePairs.get());
        ASSERT_EQ(cudaMemcpy(staged, offloadPairs.data(), kNumPages * sizeof(PageIndexPair), cudaMemcpyHostToDevice),
            cudaSuccess);
        ASSERT_EQ(cudaMemcpy(staged + kNumPages, onboardPairs.data(), kNumPages * sizeof(PageIndexPair),
                      cudaMemcpyHostToDevice),
            cudaSuccess);
        offloadArg = staged;
        onboardArg = staged + kNumPages;
    }

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    auto const cuStream = reinterpret_cast<CUstream>(stream);

    PoolCopyArgs offload{};
    offload.dstBase = reinterpret_cast<CUdeviceptr>(hostDevicePtr);
    offload.srcBase = reinterpret_cast<CUdeviceptr>(deviceSrc.get());
    offload.dstStride = kBytesPerPage;
    offload.srcStride = kBytesPerPage;
    offload.bytesPerPage = kBytesPerPage;
    offload.pairs = offloadArg;
    offload.numPairs = kNumPages;
    copier.launch(offload, CopyDirection::kD2H, cuStream);

    PoolCopyArgs onboard = offload;
    onboard.dstBase = reinterpret_cast<CUdeviceptr>(deviceDst.get());
    onboard.srcBase = reinterpret_cast<CUdeviceptr>(hostDevicePtr);
    onboard.pairs = onboardArg;
    copier.launch(onboard, CopyDirection::kH2D, cuStream);
    ASSERT_EQ(cudaStreamSynchronize(stream), cudaSuccess);

    auto const* hostWords = static_cast<uint64_t const*>(host.get());
    constexpr size_t kWordsPerPage = kBytesPerPage / sizeof(uint64_t);
    for (int32_t page = 0; page < kNumPages; ++page)
    {
        size_t const hostSlot = static_cast<size_t>(kNumPages - 1 - page);
        ASSERT_EQ(hostWords[hostSlot * kWordsPerPage], expected[static_cast<size_t>(page) * kWordsPerPage])
            << "offload page " << page;
    }

    std::vector<uint64_t> actual(kNumWords);
    ASSERT_EQ(cudaMemcpy(actual.data(), deviceDst.get(), kPoolBytes, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(actual, expected);
    ASSERT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

INSTANTIATE_TEST_SUITE_P(CoherentCopyKernelEnv, BatchedPageCopierPathTest,
    ::testing::Values(std::nullopt, std::optional<std::string>{"0"}, std::optional<std::string>{"1"}),
    [](::testing::TestParamInfo<std::optional<std::string>> const& info)
    { return info.param.has_value() ? "Value" + *info.param : std::string{"Unset"}; });

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
    ASSERT_NE(device, nullptr);

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

} // namespace
