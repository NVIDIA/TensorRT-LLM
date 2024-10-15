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
#include "tensorrt_llm/kernels/speculativeDecoding/eagleDecodingKernels.h"
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
template <typename T, int BLOCK_SIZE>
__global__ void assembleTargetLogitsOffsets(T const** logitsPtrs, SizeType32* decodingTokens, T const* logits,
    SizeType32 const* draftDecodingTokens, SizeType32 batchSize, SizeType32 maxDecodingTokens,
    SizeType32 vocabSizePadded)
{
    typedef cub::BlockScan<SizeType32, BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage tempStorage;

    auto const tix = static_cast<SizeType32>(threadIdx.x);

    SizeType32 numDecodingTokens{0};
    if (tix < batchSize)
    {
        numDecodingTokens = draftDecodingTokens[tix] + 1;
        decodingTokens[tix] = numDecodingTokens;
    }

    SizeType32 logitsOffset{0};
    BlockScan(tempStorage).ExclusiveSum(numDecodingTokens, logitsOffset);

    if (tix < batchSize)
    {
        for (SizeType32 ti = 0; ti < numDecodingTokens; ++ti)
        {
            logitsPtrs[tix * maxDecodingTokens + ti] = logits + (logitsOffset + ti) * vocabSizePadded;
        }
    }
}
} // namespace

template <typename T>
void invokeAssembleTargetLogitsOffsets(T const** logitsPtrs, SizeType32* decodingTokens, T const* logits,
    SizeType32 const* draftDecodingTokens, SizeType32 batchSize, SizeType32 maxDecodingTokens,
    SizeType32 vocabSizePadded, cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 512;
    TLLM_CHECK_WITH_INFO(
        batchSize <= BLOCK_SIZE, "Batch size larger than %d is not supported for EAGLE yet", batchSize);
    assembleTargetLogitsOffsets<T, BLOCK_SIZE><<<1, BLOCK_SIZE, 0, stream>>>(
        logitsPtrs, decodingTokens, logits, draftDecodingTokens, batchSize, maxDecodingTokens, vocabSizePadded);

    sync_check_cuda_error();
}

template void invokeAssembleTargetLogitsOffsets(float const** logitsPtrs, SizeType32* decodingTokens,
    float const* logits, SizeType32 const* draftDecodingTokens, SizeType32 batchSize, SizeType32 maxDecodingTokens,
    SizeType32 vocabSizePadded, cudaStream_t stream);
template void invokeAssembleTargetLogitsOffsets(__half const** logitsPtrs, SizeType32* decodingTokens,
    __half const* logits, SizeType32 const* draftDecodingTokens, SizeType32 batchSize, SizeType32 maxDecodingTokens,
    SizeType32 vocabSizePadded, cudaStream_t stream);

namespace
{
template <int BLOCK_SIZE>
__global__ void selectLastAccTokenAndComputeIndicesCumSum(TokenIdType* lastAcceptedTokenIds,
    SizeType32* exclusiveSumLastAcceptedIndices, SizeType32 const* draftDecodingTokens,
    TokenIdType const* acceptedTokenIds, SizeType32 const* acceptedLengths, SizeType32 const* bestPathIds,
    SizeType32 const* paths, SizeType32 batchSize, SizeType32 maxDecodingTokens, SizeType32 maxPathLen)
{
    typedef cub::BlockScan<SizeType32, BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage tempStorage;

    auto const tix = static_cast<SizeType32>(threadIdx.x);
    SizeType32 decodingTokens{0};
    SizeType32 lastTokenId{0};
    if (tix < batchSize)
    {
        auto const acceptedLen = acceptedLengths[tix];
        lastAcceptedTokenIds[tix] = acceptedTokenIds[tix * maxPathLen + acceptedLen - 1];
        auto const bestPathId = bestPathIds[tix];
        auto const pathIdx = flat_index3(tix, bestPathId, acceptedLen - 1, maxDecodingTokens, maxPathLen);
        lastTokenId = paths[pathIdx];
        decodingTokens = draftDecodingTokens[tix] + 1;
    }

    BlockScan(tempStorage).ExclusiveSum(decodingTokens, decodingTokens);

    if (tix < batchSize)
    {
        exclusiveSumLastAcceptedIndices[tix] = decodingTokens + lastTokenId;
    }
}
} // namespace

void invokeSelectLastAccTokenAndComputeIndicesCumSum(TokenIdType* lastAcceptedTokenIds,
    SizeType32* exclusiveSumLastAcceptedIndices, SizeType32 const* draftDecodingTokens,
    TokenIdType const* acceptedTokenIds, SizeType32 const* acceptedLengths, SizeType32 const* bestPathIds,
    SizeType32 const* paths, SizeType32 batchSize, SizeType32 maxDecodingTokens, SizeType32 maxPathLen,
    cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 512;
    TLLM_CHECK_WITH_INFO(
        batchSize <= BLOCK_SIZE, "Batch size larger than %d is not supported for EAGLE yet", batchSize);
    selectLastAccTokenAndComputeIndicesCumSum<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, stream>>>(lastAcceptedTokenIds,
        exclusiveSumLastAcceptedIndices, draftDecodingTokens, acceptedTokenIds, acceptedLengths, bestPathIds, paths,
        batchSize, maxDecodingTokens, maxPathLen);
}

} // namespace tensorrt_llm::kernels::speculative_decoding
