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

#include "coldPageCopy.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaDriverWrapper.h"
#if CUDA_VERSION < 12080
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#endif
#include <limits>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2::detail
{
#if CUDA_VERSION < 12080
namespace
{

constexpr size_t kPageIndexKernelParamBytes = 2U << 10U;
constexpr size_t kPageIndicesPerKernel = kPageIndexKernelParamBytes / sizeof(PageIndexPair);
constexpr uint32_t kPageIndexCopyThreads = 256;

using PageIndexKernelParams = std::array<PageIndexPair, kPageIndicesPerKernel>;
static_assert(sizeof(PageIndexKernelParams) == kPageIndexKernelParamBytes);
static_assert(kPageIndicesPerKernel <= kPageIndexCopyThreads);

#if CUDA_VERSION >= 11070
#define TLLM_KVCM2_GRID_CONSTANT __grid_constant__
#else
#define TLLM_KVCM2_GRID_CONSTANT
#endif

__global__ void copyPageIndicesKernel(
    PageIndexPair* dst, PageIndexKernelParams const TLLM_KVCM2_GRID_CONSTANT src, size_t count)
{
    size_t const index = threadIdx.x;
    if (index < count)
    {
        dst[index] = src[index];
    }
}

#undef TLLM_KVCM2_GRID_CONSTANT

} // namespace

void copyPageIndicesToDeviceWithKernel(
    CUdeviceptr dst, PageIndexPair const* src, size_t numPageIndices, CUstream stream)
{
    TLLM_CHECK_WITH_INFO(dst != 0 && src != nullptr, "Page-index copy requires valid source and destination");

    PageIndexKernelParams params{};
    size_t offset = 0;
    while (offset < numPageIndices)
    {
        size_t const count = std::min(numPageIndices - offset, kPageIndicesPerKernel);
        std::copy_n(src + offset, count, params.begin());
        copyPageIndicesKernel<<<1, kPageIndexCopyThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            reinterpret_cast<PageIndexPair*>(dst) + offset, params, count);
        TLLM_CUDA_CHECK(cudaGetLastError());
        offset += count;
    }
}

void copyColdPageDataBatchWithMemcpyAsync(
    CUdeviceptr* dsts, CUdeviceptr* srcs, size_t* sizes, size_t count, CUstream stream)
{
    TLLM_CHECK_WITH_INFO(
        count == 0 || (dsts != nullptr && srcs != nullptr && sizes != nullptr), "Invalid cold-page copy batch");
    for (size_t index = 0; index < count; ++index)
    {
        TLLM_CU_CHECK(cuMemcpyAsync(dsts[index], srcs[index], sizes[index], stream));
    }
}
#endif

void copyPageIndicesToDevice(CUdeviceptr dst, PageIndexPair const* src, size_t numPageIndices, CUstream stream)
{
    if (numPageIndices == 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(
        numPageIndices <= std::numeric_limits<size_t>::max() / sizeof(PageIndexPair), "Page-index array is too large");
    TLLM_CHECK_WITH_INFO(dst != 0 && src != nullptr, "Page-index copy requires valid source and destination");

#if CUDA_VERSION >= 12080
    size_t numBytes = numPageIndices * sizeof(PageIndexPair);
    CUdeviceptr srcAddress = reinterpret_cast<CUdeviceptr>(src);
    CUmemcpyAttributes attributes{};
    attributes.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_DURING_API_CALL;
    attributes.srcLocHint.type = CU_MEM_LOCATION_TYPE_HOST;
    attributes.dstLocHint.type = CU_MEM_LOCATION_TYPE_DEVICE;
    attributes.flags = CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE;
    size_t firstCopy = 0;
#if CUDA_VERSION < 13000
    size_t failIdx = std::numeric_limits<size_t>::max();
    TLLM_CU_CHECK(cuMemcpyBatchAsync(&dst, &srcAddress, &numBytes, 1, &attributes, &firstCopy, 1, &failIdx, stream));
#else
    TLLM_CU_CHECK(cuMemcpyBatchAsync(&dst, &srcAddress, &numBytes, 1, &attributes, &firstCopy, 1, stream));
#endif
#else
    copyPageIndicesToDeviceWithKernel(dst, src, numPageIndices, stream);
#endif
}

void copyColdPageDataBatch(CUdeviceptr* dsts, CUdeviceptr* srcs, size_t* sizes, size_t count, CUstream stream)
{
    if (count == 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(dsts != nullptr && srcs != nullptr && sizes != nullptr, "Invalid cold-page copy batch");

#if CUDA_VERSION >= 12080
    CUmemcpyAttributes attributes{};
    attributes.srcAccessOrder = CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
    attributes.flags = CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE;
    size_t firstCopy = 0;
#if CUDA_VERSION < 13000
    size_t failIdx = std::numeric_limits<size_t>::max();
    TLLM_CU_CHECK(cuMemcpyBatchAsync(dsts, srcs, sizes, count, &attributes, &firstCopy, 1, &failIdx, stream));
#else
    TLLM_CU_CHECK(cuMemcpyBatchAsync(dsts, srcs, sizes, count, &attributes, &firstCopy, 1, stream));
#endif
#else
    copyColdPageDataBatchWithMemcpyAsync(dsts, srcs, sizes, count, stream);
#endif
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2::detail
