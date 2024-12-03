/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/speculativeDecoding/medusaDecodingKernels.h"
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::kernels::speculative_decoding
{
namespace
{
__global__ void scatterMedusaDraftTokens(TokenIdType* treeDraftIds, TokenIdType const* sourceDraftIds,
    SizeType32 const* treeIds, SizeType32 const* tokensPerStepData, SizeType32 const* batchSlots,
    SizeType32 maxDecodingTokens)
{
    auto const batchIdx = static_cast<SizeType32>(blockIdx.x);
    auto const batchSlot = batchSlots[batchIdx];
    auto const tokensPerStep = tokensPerStepData[batchSlot];
    auto const maxDecodingDraftTokens = maxDecodingTokens - 1;
    for (auto index = static_cast<SizeType32>(threadIdx.x); index < tokensPerStep - 1;
         index += static_cast<SizeType32>(blockDim.x))
    {
        auto const indexInTree = treeIds[batchSlot * maxDecodingDraftTokens + index];
        auto const treeDraftIdx = batchSlot * maxDecodingDraftTokens + index;
        auto const sourceDraftIdx = batchSlot * maxDecodingTokens + indexInTree;
        treeDraftIds[treeDraftIdx] = sourceDraftIds[sourceDraftIdx];
    }
}
} // namespace

void scatterMedusaDraftTokens(TokenIdType* treeDraftIds, TokenIdType const* sourceDraftIds, SizeType32 const* treeIds,
    SizeType32 const* tokensPerStep, SizeType32 const* batchSlots, SizeType32 maxDecodingTokens, SizeType32 batchSize,
    cudaStream_t stream)
{
    constexpr SizeType32 BLOCK_SIZE = 256;
    scatterMedusaDraftTokens<<<batchSize, BLOCK_SIZE, 0, stream>>>(
        treeDraftIds, sourceDraftIds, treeIds, tokensPerStep, batchSlots, maxDecodingTokens);
}
} // namespace tensorrt_llm::kernels::speculative_decoding
