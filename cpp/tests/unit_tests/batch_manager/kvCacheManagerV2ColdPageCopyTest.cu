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

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#if CUDA_VERSION < 12080
#include <algorithm>
#endif
#include <array>
#include <cstddef>
#if CUDA_VERSION < 12080
#include <cstdint>
#endif
#include <memory>
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
