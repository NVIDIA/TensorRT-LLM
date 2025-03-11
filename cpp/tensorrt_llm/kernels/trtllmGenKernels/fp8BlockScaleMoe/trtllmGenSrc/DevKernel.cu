/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "DevKernel.h"

//// FIX
#include "Utils.h"  // #include <trtllm/dev/Utils.h>
#include "macros.h" // #include <utils/macros.h>
// #include <trtllm/gen/GenCtx.h>

#include <cub/cub.cuh>

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace moe::dev
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace activation
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void activationKernel(KernelParams params)
{
    using Type = typename KernelParams::Type;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // immediately trigger the secondary kernel when using PDL, then wait on primary
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
        cudaGridDependencySynchronize();
    }
#endif

    int const numIters = params.outerDim * params.innerDim / 2;
    for (int tid = threadIdx.x + blockDim.x * blockIdx.x; tid < numIters; tid += blockDim.x * gridDim.x)
    {
        int const innerIdx = tid % (params.innerDim / 2);
        int const outerIdx = tid / (params.innerDim / 2);
        int const baseIdx = outerIdx * params.innerDim + innerIdx;
        float x1 = (float) params.inPtr[baseIdx];
        float x2 = (float) params.inPtr[baseIdx + params.innerDim / 2];

        float act = trtllm::dev::silu(x1);
        Type out = (Type) (act * x2);

        int const outIdx = outerIdx * (params.innerDim / 2) + innerIdx;
        params.outPtr[outIdx] = out;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void activationDeepSeekKernel(KernelParams params)
{
    using Type = typename KernelParams::Type;
    using BlockReduce = cub::BlockReduce<float, 128>;

    __shared__ float s_scaleOut;
    __shared__ typename BlockReduce::TempStorage temp_storage;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // immediately trigger the secondary kernel when using PDL, then wait on primary
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
        cudaGridDependencySynchronize();
    }
#endif

    int const numIters = params.outerDim * params.innerDim / 2;
    for (int tid = threadIdx.x + blockDim.x * blockIdx.x; tid < numIters; tid += blockDim.x * gridDim.x)
    {
        int const innerIdx = tid % (params.innerDim / 2);
        int const outerIdx = tid / (params.innerDim / 2);
        int const baseIdx = outerIdx * params.innerDim + innerIdx;

        int const permutedIdx = outerIdx;
        int const expandedIdx = params.permutedIdxToExpandedIdx[permutedIdx];
        if (expandedIdx == -1)
            continue;

        int const scale1_idx = outerIdx + params.totalNumPaddedTokens * (innerIdx / 128);
        int const scale2_idx
            = outerIdx + params.totalNumPaddedTokens * ((innerIdx / 128) + (params.innerDim / 2 / 128));
        float const scale1 = params.inDqSfsPtr[scale1_idx];
        float const scale2 = params.inDqSfsPtr[scale2_idx];

        float x1 = scale1 * (float) params.inPtr[baseIdx];
        float x2 = scale2 * (float) params.inPtr[baseIdx + params.innerDim / 2];

        float act = trtllm::dev::silu(x1);
        float out = act * x2;

        // The largest (finite) value that can be represented using E4m3.
        float constexpr E4m3MaxVal{448.f};

        // Compute the absolute max
        float aMax = BlockReduce(temp_storage).Reduce(fabsf(out), cub::Max());
        if (threadIdx.x == 0)
        {
            s_scaleOut = aMax / E4m3MaxVal;
            int const scaleOut_idx = outerIdx + params.totalNumPaddedTokens * (innerIdx / 128);
            params.outDqSfsPtr[scaleOut_idx] = aMax / E4m3MaxVal;
        }
        __syncthreads();
        float const scaleOut = s_scaleOut;
        __syncthreads();
        int const outIdx = outerIdx * (params.innerDim / 2) + innerIdx;
        params.outPtr[outIdx] = (Type) (out / scaleOut);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream)
{
    if (data.mUseDeepSeekFp8)
    {
        int const numThreads = 128;
        int const numElems = (data.outerDim * data.innerDim / 2);
        int const numBlocks = (numElems - 1 + numThreads) / numThreads;

        LAUNCH(data, activationDeepSeekKernel, numBlocks, numThreads, 0, stream);
    }
    else
    {
        int const numThreads = 256;
        int const numElems = (data.outerDim * data.innerDim / 2);
        int const numBlocks = (numElems - 1 + numThreads) / numThreads;

        LAUNCH(data, activationKernel, numBlocks, numThreads, 0, stream);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace activation

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace permute
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void permuteKernel(KernelParams params)
{
    using Type = typename KernelParams::Type;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // immediately trigger the secondary kernel when using PDL, then wait on primary
    if constexpr (KernelParams::UsePdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
        cudaGridDependencySynchronize();
    }
#endif

    for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens; tokenIdx += gridDim.y)
    {
        // Loop over hidden dim
        for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.hiddenDim;
             hiddenIdx += blockDim.x * gridDim.x)
        {

            // Load chunk of token into registers
            const Type data = params.inPtr[tokenIdx * params.hiddenDim + hiddenIdx];

            // Write to topK places
            for (int k = 0; k < params.topK; k++)
            {
                int const expandedIdx = tokenIdx * params.topK + k;
                int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];
                params.outPtr[permutedIdx * params.hiddenDim + hiddenIdx] = data;
            }
        }
        if (params.useDeepSeekFp8)
        {
            for (int scaleIdx = threadIdx.x + blockDim.x * blockIdx.x; scaleIdx < params.hiddenDim / 128;
                 scaleIdx += blockDim.x * gridDim.x)
            {
                for (int k = 0; k < params.topK; k++)
                {
                    int const expandedIdx = tokenIdx * params.topK + k;
                    int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];

                    int const idx_in = tokenIdx + params.numTokens * scaleIdx;
                    int const idx_out = permutedIdx + params.totalNumPaddedTokens * scaleIdx;

                    params.outDqSfsPtr[idx_out] = params.inDqSfsPtr[idx_in];
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream)
{
    int const numThreads = 256;
    int const numBlocksX = (data.hiddenDim - 1 + numThreads) / numThreads;
    int const numBlocksY = data.numTokens;
    dim3 numBlocks(numBlocksX, numBlocksY);

    LAUNCH(data, permuteKernel, numBlocks, numThreads, 0, stream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace permute

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace finalize
{

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace tg = trtllm::gen;

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void finalizeKernel(KernelParams params)
{
    using Type = typename KernelParams::Type;
    using TypeExpW = typename KernelParams::TypeExpW;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // wait on primary kernel when using PDL
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }
#endif

    for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens; tokenIdx += gridDim.y)
    {
        // Loop over hidden dim
        for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.hiddenDim;
             hiddenIdx += blockDim.x * gridDim.x)
        {

            // Accumulate chunk of token into registers
            float data = 0.0F;

            // Write to topK places
            for (int k = 0; k < params.topK; k++)
            {
                int const expandedIdx = tokenIdx * params.topK + k;

                const TypeExpW scale = params.expertWeightsPtr[expandedIdx];

                int const permuteIdx = params.expandedIdxToPermutedIdx[expandedIdx];
                data += float{scale} * float{params.inPtr[permuteIdx * params.hiddenDim + hiddenIdx]};
            }

            params.outPtr[tokenIdx * params.hiddenDim + hiddenIdx] = static_cast<Type>(data);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename KernelParams>
__global__ void finalizeDeepSeekKernel(KernelParams params)
{
    using Type = typename KernelParams::Type;
    using BlockReduce = cub::BlockReduce<float, 128>;

    __shared__ float s_scaleOut;
    __shared__ typename BlockReduce::TempStorage temp_storage;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // wait on primary kernel when using PDL
    if constexpr (KernelParams::UsePdl)
    {
        cudaGridDependencySynchronize();
    }
#endif

    for (int tokenIdx = blockIdx.y; tokenIdx < params.numTokens; tokenIdx += gridDim.y)
    {
        // Loop over hidden dim
        for (int hiddenIdx = threadIdx.x + blockDim.x * blockIdx.x; hiddenIdx < params.hiddenDim;
             hiddenIdx += blockDim.x * gridDim.x)
        {

            // Accumulate chunk of token into registers
            float acc = 0.0f;

            for (int k = 0; k < params.topK; k++)
            {
                int const expandedIdx = tokenIdx * params.topK + k;
                int const permutedIdx = params.expandedIdxToPermutedIdx[expandedIdx];

                int const scaleIdx = permutedIdx + params.totalNumPaddedTokens * (hiddenIdx / 128);
                float const blockScale = params.inDqSfsPtr ? params.inDqSfsPtr[scaleIdx] : 1;

                float const expertProb = (float) params.expertWeightsPtr[tokenIdx * params.topK + k];

                float const scale = expertProb * blockScale;
                acc += scale * static_cast<float>(params.inPtr[permutedIdx * params.hiddenDim + hiddenIdx]);
            }

            // The largest (finite) value that can be represented using E4m3.
            float constexpr E4m3MaxVal{448.f};

            // Compute the absolute max
            float aMax = BlockReduce(temp_storage).Reduce(fabsf(acc), cub::Max());

            if (threadIdx.x == 0)
            {
                if (params.outDqSfsPtr)
                {
                    s_scaleOut = aMax / E4m3MaxVal;
                    int const scaleOut_idx = tokenIdx + hiddenIdx / 128 * params.numTokens;
                    params.outDqSfsPtr[scaleOut_idx] = aMax / E4m3MaxVal;
                }
                else
                {
                    s_scaleOut = 1.0f;
                }
            }
            __syncthreads();
            float const scaleOut = s_scaleOut;
            __syncthreads();
            params.outPtr[tokenIdx * params.hiddenDim + hiddenIdx] = (Type) (acc / scaleOut);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void run(Data const& data, void* stream)
{
    if (data.mUseDeepSeekFp8)
    {
        int const numThreads = 128;
        int const numBlocksX = (data.hiddenDim - 1 + numThreads) / numThreads;
        int const numBlocksY = data.numTokens;
        dim3 numBlocks(numBlocksX, numBlocksY);

        LAUNCH_EXPW(data, finalizeDeepSeekKernel, numBlocks, numThreads, 0, stream);
    }
    else
    {
        int const numThreads = 256;
        int const numBlocksX = (data.hiddenDim - 1 + numThreads) / numThreads;
        int const numBlocksY = data.numTokens;
        dim3 numBlocks(numBlocksX, numBlocksY);

        LAUNCH_EXPW(data, finalizeKernel, numBlocks, numThreads, 0, stream);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace finalize

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace moe::dev
