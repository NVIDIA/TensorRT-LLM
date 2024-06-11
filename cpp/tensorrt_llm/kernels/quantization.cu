/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/quantization.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

__global__ void quantizedKernel(char4* dst, float4 const* src, const int64_t sizeDiv4, float const* scalePtr)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeDiv4; idx += blockDim.x * gridDim.x)
    {
        float const scale = __ldg(scalePtr);
        char4 tmp;
        const float4 floatTmp = __ldg(src + idx);
        tmp.x = cuda_cast<int8_t>(floatTmp.x * scale);
        tmp.y = cuda_cast<int8_t>(floatTmp.y * scale);
        tmp.z = cuda_cast<int8_t>(floatTmp.z * scale);
        tmp.w = cuda_cast<int8_t>(floatTmp.w * scale);
        dst[idx] = tmp;
    }
}

__global__ void quantizedKernel(char4* dst, half2 const* src, const int64_t sizeDiv4, float const* scalePtr)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeDiv4; idx += blockDim.x * gridDim.x)
    {
        float const scale = __ldg(scalePtr);
        char4 tmp;
        int srcId = idx << 1;

        const uint2 h2 = __ldg(reinterpret_cast<uint2 const*>(src + srcId));

        const half2 half2Tmp = reinterpret_cast<half2 const&>(h2.x);
        const half2 half2Tmp2 = reinterpret_cast<half2 const&>(h2.y);

        tmp.x = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp.x) * scale);
        tmp.y = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp.y) * scale);
        tmp.z = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp2.x) * scale);
        tmp.w = cuda_cast<int8_t>(cuda_cast<float>(half2Tmp2.y) * scale);
        dst[idx] = tmp;
    }
}

#ifdef ENABLE_BF16
__global__ void quantizedKernel(char4* dst, __nv_bfloat162 const* src, const int64_t sizeDiv4, float const* scalePtr)
{
    for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < sizeDiv4; idx += blockDim.x * gridDim.x)
    {
        float const scale = __ldg(scalePtr);
        char4 tmp;
        int srcId = idx << 1;

        const uint2 h2 = __ldg(reinterpret_cast<uint2 const*>(src + srcId));

        const __nv_bfloat162 bfloat162Tmp = reinterpret_cast<__nv_bfloat162 const&>(h2.x);
        const __nv_bfloat162 bfloat162Tmp2 = reinterpret_cast<__nv_bfloat162 const&>(h2.y);

        tmp.x = cuda_cast<int8_t>(cuda_cast<float>(bfloat162Tmp.x) * scale);
        tmp.y = cuda_cast<int8_t>(cuda_cast<float>(bfloat162Tmp.y) * scale);
        tmp.z = cuda_cast<int8_t>(cuda_cast<float>(bfloat162Tmp2.x) * scale);
        tmp.w = cuda_cast<int8_t>(cuda_cast<float>(bfloat162Tmp2.y) * scale);

        dst[idx] = tmp;
    }
}
#endif

template <typename T>
void invokeQuantization(
    int8_t* dst, T const* src, const int64_t size, float const* scalePtr, cudaStream_t stream, int maxGridSize)
{
    TLLM_CHECK_WITH_INFO(size % 4 == 0, "[ERROR][invokeQuantization] size should be a multiple of 4.\n");

    int numBlocks{static_cast<int>((size + 255) / 256)};
    dim3 grid(std::min(numBlocks, maxGridSize));
    TLLM_CHECK_WITH_INFO(grid.x <= maxGridSize, "[ERROR][invokeQuantization] grid max size is exceeded\n");
    dim3 block(64);
    if (std::is_same_v<T, float>)
    {
        quantizedKernel<<<grid, block, 0, stream>>>((char4*) dst, (float4 const*) src, size / 4, scalePtr);
    }
    else if (std::is_same_v<T, half>)
    {
        quantizedKernel<<<grid, block, 0, stream>>>((char4*) dst, (half2 const*) src, size / 4, scalePtr);
    }
#ifdef ENABLE_BF16
    else if (std::is_same_v<T, __nv_bfloat16>)
    {
        quantizedKernel<<<grid, block, 0, stream>>>((char4*) dst, (__nv_bfloat162 const*) src, size / 4, scalePtr);
    }
#endif
}

template void invokeQuantization<float>(
    int8_t* dst, float const* src, const int64_t size, float const* scalePtr, cudaStream_t stream, int maxGridSize);

template void invokeQuantization<half>(
    int8_t* dst, half const* src, const int64_t size, float const* scalePtr, cudaStream_t stream, int maxGridSize);

#ifdef ENABLE_BF16
template void invokeQuantization<__nv_bfloat16>(int8_t* dst, __nv_bfloat16 const* src, const int64_t size,
    float const* scalePtr, cudaStream_t stream, int maxGridSize);
#endif

template <typename T>
__global__ void perTokenQuantization(
    int8_t* dst, T const* src, const int64_t numRows, const int64_t numCols, float* scalePtr)
{
    T const* srcRow = src + blockIdx.x * numCols;
    int8_t* dstRow = dst + blockIdx.x * numCols;

    T localMax = 1e-6f;
    for (int i = threadIdx.x; i < numCols; i += blockDim.x)
    {
        localMax = cuda_max(localMax, cuda_abs(srcRow[i]));
    }
    float const rowMax = blockAllReduceMax(cuda_cast<float>(localMax));

    if (threadIdx.x == 0)
    {
        scalePtr[blockIdx.x] = rowMax / 127.f;
    }

    float const scaleOrigQuant = 127.f / rowMax;
    for (int i = threadIdx.x; i < numCols; i += blockDim.x)
    {
        dstRow[i] = cuda_cast<int8_t>(cuda_cast<float>(srcRow[i]) * scaleOrigQuant);
    }
}

template <typename T>
void invokePerTokenQuantization(
    int8_t* dst, T const* src, const int64_t numRows, const int64_t numCols, float* scalePtr, cudaStream_t stream)
{
    // each block is responsible for a single row
    const dim3 block(512);
    const dim3 grid(numRows);

    perTokenQuantization<<<grid, block, 0, stream>>>(dst, src, numRows, numCols, scalePtr);
}

#define INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(T)                                                                   \
    template void invokePerTokenQuantization(                                                                          \
        int8_t* dst, const T* src, const int64_t numRows, const int64_t numCols, float* scalePtr, cudaStream_t stream)

INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(float);
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(half);
#ifdef ENABLE_BF16
INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION(__nv_bfloat16);
#endif

} // namespace kernels
} // namespace tensorrt_llm
