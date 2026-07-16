/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/samplerKernels/toppDecayKernels.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

__global__ void toppDecayUpdateKernel(float* __restrict__ runtimeTopP, float const* __restrict__ initialTopP,
    float const* __restrict__ topPDecay, float const* __restrict__ topPMin, int32_t const* __restrict__ resetIds,
    bool const* __restrict__ isDecaySlot, int32_t const* __restrict__ stepTokens, int64_t stepTokenStride,
    int64_t const* __restrict__ sampledSlots, int32_t numSampled)
{
    int32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numSampled)
    {
        return;
    }
    int64_t const slot = sampledSlots[i];
    if (!isDecaySlot[slot]) // on-device decay gate
    {
        return;
    }
    int32_t const tok = stepTokens[slot * stepTokenStride]; // gather in-kernel (strided new_tokens view)
    int32_t const rid = resetIds[slot];
    float updated;
    if (rid >= 0 && tok == rid)
    {
        updated = initialTopP[slot];
    }
    else
    {
        updated = fmaxf(runtimeTopP[slot] * topPDecay[slot], topPMin[slot]);
    }
    runtimeTopP[slot] = updated;
}

__global__ void toppDecayGatherKernel(float* __restrict__ rowTopP, float const* __restrict__ runtimeTopP,
    bool const* __restrict__ isDecaySlot, float const* __restrict__ staticTopP, int64_t const* __restrict__ slots,
    int32_t numRows)
{
    int32_t const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numRows)
    {
        return;
    }
    int64_t const slot = slots[i];
    rowTopP[i] = isDecaySlot[slot] ? runtimeTopP[slot] : staticTopP[i];
}

} // namespace

void invokeToppDecayUpdate(float* runtimeTopP, float const* initialTopP, float const* topPDecay, float const* topPMin,
    int32_t const* resetIds, bool const* isDecaySlot, int32_t const* stepTokens, int64_t stepTokenStride,
    int64_t const* sampledSlots, int32_t numSampled, cudaStream_t stream)
{
    if (numSampled == 0)
    {
        return;
    }
    constexpr int32_t kBlock = 256;
    int32_t const grid = (numSampled + kBlock - 1) / kBlock;
    toppDecayUpdateKernel<<<grid, kBlock, 0, stream>>>(runtimeTopP, initialTopP, topPDecay, topPMin, resetIds,
        isDecaySlot, stepTokens, stepTokenStride, sampledSlots, numSampled);
}

void invokeToppDecayGather(float* rowTopP, float const* runtimeTopP, bool const* isDecaySlot, float const* staticTopP,
    int64_t const* slots, int32_t numRows, cudaStream_t stream)
{
    if (numRows == 0)
    {
        return;
    }
    constexpr int32_t kBlock = 256;
    int32_t const grid = (numRows + kBlock - 1) / kBlock;
    toppDecayGatherKernel<<<grid, kBlock, 0, stream>>>(rowTopP, runtimeTopP, isDecaySlot, staticTopP, slots, numRows);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
