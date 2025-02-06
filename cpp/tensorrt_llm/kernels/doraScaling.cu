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

#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"

// TODO(oargov): literally zero performance optimization work was put into these kernels and their launch parameters,
// since they should hopefully be fused to some gemm eventually.
namespace tensorrt_llm::kernels
{
template <typename T>
__global__ void tokenPerChannelScaleKernel(size_t const numModules, size_t const numTokens,
    int64_t const* __restrict__ cumModuleSizes, T const* __restrict__ a, T const* const* __restrict__ scales,
    T* __restrict__ result)
{
    /*
     * This kernel applies DoRA scaling to LoRA output.
     * Like LoRA, each token in the batch may target a different adapter.
     * Each adapter may also have multiple modules, for example: QKV projection will have a different scale for Q, K and
     * V, but they will be concatenated into a single input vector.
     * `scales` is a vector of pointers to DoRA magnitude vectors. Each token will have `numModules` pointers, and
     * pointers for the same module are next to each other. For example:
     * scales = [token0_module0_ptr, token1_module0_ptr, ..., token0_module1_ptr, token1_module1_ptr, ...]
     */
    auto const threadId = blockIdx.x * blockDim.x + threadIdx.x;
    // number of columns in the input
    auto const numChannels = cumModuleSizes[numModules - 1];
    // current thread's column
    auto const channelId = threadId % numChannels;
    // current thread's token
    auto const tokenId = threadId / numChannels;
    // offset the input column to fit in the scaling vector's column in case of multiple modules
    int64_t scaleChannelOffset = 0;

    T const* scale = nullptr;

    // this loop searches for the module the current column is a part of, in case of multiple modules
    for (auto moduleId = 0; moduleId < numModules; moduleId++)
    {
        if (channelId < cumModuleSizes[moduleId])
        {
            // pick the proper scale for the token and module
            scale = scales[numTokens * moduleId + tokenId];
            break;
        }
        // adjust scale offset
        scaleChannelOffset = cumModuleSizes[moduleId];
    }

    if (threadId < numChannels * numTokens)
    {
        // apply scaling if scale is not null (it is null in case of a non-DoRA adapter)
        result[threadId] = scale == nullptr ? a[threadId] : a[threadId] * scale[channelId - scaleChannelOffset];
    }
}

template <typename T>
void tokenPerChannelScale(int64_t const numel, size_t const numModules, size_t const numTokens,
    int64_t const* __restrict__ cumModuleSizes, T const* __restrict__ a, T const* const* __restrict__ scale_ptrs,
    T* __restrict__ result, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((numel + 255) / 256);

    tokenPerChannelScaleKernel<T>
        <<<grid, block, 0, stream>>>(numModules, numTokens, cumModuleSizes, a, scale_ptrs, result);
}

template void tokenPerChannelScale<half>(int64_t const numel, size_t const numModules, size_t const numTokens,
    int64_t const* __restrict__ cumModuleSizes, half const* __restrict__ a, half const* const* __restrict__ scale_ptrs,
    half* __restrict__ result, cudaStream_t stream);

#ifdef ENABLE_BF16
template void tokenPerChannelScale<nv_bfloat16>(int64_t const numel, size_t const numModules, size_t const numTokens,
    int64_t const* __restrict__ cumModuleSizes, nv_bfloat16 const* __restrict__ a,
    nv_bfloat16 const* const* __restrict__ scale_ptrs, nv_bfloat16* __restrict__ result, cudaStream_t stream);
#endif

} // namespace tensorrt_llm::kernels
