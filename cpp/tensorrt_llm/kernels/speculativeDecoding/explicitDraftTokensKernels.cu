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
#include "tensorrt_llm/kernels/speculativeDecoding/explicitDraftTokensKernels.h"
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
size_t invokeScanSpecDecodingGenerationLengths(void* __restrict__ scanTempStorage, size_t scanTempStorageBytes,
    SizeType32 const* __restrict__ specDecodingGenerationLengths,
    SizeType32* __restrict__ maxSpecDecodingGenerationLengths, SizeType32 batchSize, cudaStream_t stream)
{
    cub::DeviceScan::InclusiveSum(scanTempStorage, scanTempStorageBytes, specDecodingGenerationLengths,
        maxSpecDecodingGenerationLengths, batchSize, stream);
    return scanTempStorageBytes;
}

size_t invokeReduceMaxSpecDecodingGenerationLengths(void* __restrict__ reduceMaxTempStorage,
    size_t reduceTempStorageBytes, SizeType32 const* __restrict__ specDecodingGenerationLengths,
    SizeType32* __restrict__ scannedSpecDecodingGenerationLengths, SizeType32 batchSize, cudaStream_t stream)
{
    cub::DeviceReduce::Max(reduceMaxTempStorage, reduceTempStorageBytes, specDecodingGenerationLengths,
        scannedSpecDecodingGenerationLengths, batchSize, stream);
    return reduceTempStorageBytes;
}

// inclusive prefix sum specDecodingGenerationLengths and reduce max specDecodingGenerationLengths
void invokeScanReduceSpecDecodingGenerationLengths(SizeType32 batchSize,
    SizeType32 const* __restrict__ specDecodingGenerationLengths, void* __restrict__ scanTempStorage,
    size_t scanTempStorageBytes, SizeType32* __restrict__ scanedSpecDecodingGenerationLengths,
    void* __restrict__ reduceMaxTempStorage, size_t reduceMaxTempStorageBytes,
    SizeType32* maxSpecDecodingGenerationLengths, cudaStream_t stream)
{
    invokeScanSpecDecodingGenerationLengths(scanTempStorage, scanTempStorageBytes, specDecodingGenerationLengths,
        scanedSpecDecodingGenerationLengths, batchSize, stream);
    invokeReduceMaxSpecDecodingGenerationLengths(reduceMaxTempStorage, reduceMaxTempStorageBytes,
        specDecodingGenerationLengths, maxSpecDecodingGenerationLengths, batchSize, stream);
}

////////////////////////

namespace
{

template <typename T>
inline __device__ __host__ T divUp(T m, T n)
{
    return (m + n - 1) / n;
}

__device__ SizeType32 positivePowerOfTwo(SizeType32 n)
{
    if (n == 0)
    {
        return 1;
    }
    if (n == 1)
    {
        return 2;
    }
    SizeType32 res = 1;
    SizeType32 i = n;
    SizeType32 x = 2;
    while (i)
    {
        if (i & 0x1)
        {
            res *= x;
        }
        x *= x;
        i >>= 1;
    }
    return res;
}

__global__ void getSpecDecodingPackedMask(SizeType32 const* __restrict__ specDecodingCumGenerationLengths,
    SizeType32 const* __restrict__ specDecodingMaxGenerationLengths, bool const* __restrict__ specDecodingMask,
    SizeType32 const* __restrict__ batchSlots, SizeType32 maxDraftTokens,
    SizeType32* __restrict__ specDecodingPackedMask)
{
    auto const batchIdx = static_cast<SizeType32>(blockIdx.y);
    auto const tokenIdx = static_cast<SizeType32>(blockIdx.x);

    auto const numTokens = (batchIdx == 0)
        ? specDecodingCumGenerationLengths[0]
        : specDecodingCumGenerationLengths[batchIdx] - specDecodingCumGenerationLengths[batchIdx - 1];
    if (tokenIdx >= numTokens)
    {
        return;
    }

    auto const maxGenerationLength = specDecodingMaxGenerationLengths[0];
    auto const numPackedMasks = divUp(maxDraftTokens, 32);
    auto const outputStartId = batchSlots ? (batchSlots[batchIdx] * (maxDraftTokens + 1))
                                          : ((batchIdx == 0) ? 0 : specDecodingCumGenerationLengths[batchIdx - 1]);
    auto* outputPtr = specDecodingPackedMask + (outputStartId + tokenIdx) * numPackedMasks;
    if (tokenIdx == 0)
    {
        for (auto maskId = static_cast<SizeType32>(threadIdx.x); maskId < numPackedMasks;
             maskId += static_cast<SizeType32>(blockDim.x))
        {
            outputPtr[maskId] = maskId == 0 ? 1 : 0;
        }
        return;
    }
    else
    {
        bool const* specDecodingMaskPtr = specDecodingMask + batchIdx * maxGenerationLength * maxGenerationLength
            + tokenIdx * maxGenerationLength + 1;
        extern __shared__ char shSpecDecodingMask[];
        if (threadIdx.x == 0)
        {
            shSpecDecodingMask[maxGenerationLength - 1] = '1';
        }
        for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < maxGenerationLength - 1;
             ti += static_cast<SizeType32>(blockDim.x))
        {
            auto const shIndex = maxGenerationLength - 1 - ti - 1;
            shSpecDecodingMask[shIndex] = specDecodingMaskPtr[ti] ? '1' : '0';
        }
        __syncthreads();
        for (auto maskId = static_cast<SizeType32>(threadIdx.x); maskId < numPackedMasks;
             maskId += static_cast<SizeType32>(blockDim.x))
        {
            if (maskId * 32 >= maxGenerationLength)
            {
                outputPtr[maskId] = 0;
                return;
            }
            else
            {
                auto const shSpecDecodingMaskIndexStart
                    = ((maxGenerationLength - (maskId + 1) * 32) < 0) ? 0 : (maxGenerationLength - (maskId + 1) * 32);
                auto const shSpecDecodingMaskIndexEnd = maxGenerationLength - (maskId * 32 + 1) + 1;

                auto const validNumBits = shSpecDecodingMaskIndexEnd - shSpecDecodingMaskIndexStart;
                auto const firstBit1 = (shSpecDecodingMask[shSpecDecodingMaskIndexStart] == '1') ? true : false;
                SizeType32 mask31bits = 0;
                if (validNumBits != 1)
                {
                    for (auto i = shSpecDecodingMaskIndexStart + 1; i < shSpecDecodingMaskIndexEnd; i++)
                    {
                        auto const index = (validNumBits - 1) - (i - shSpecDecodingMaskIndexStart - 1) - 1;
                        mask31bits += (shSpecDecodingMask[i] == '1') ? positivePowerOfTwo(index) : 0;
                    }
                }
                SizeType32 mask32bits;
                if (validNumBits == 32)
                {
                    mask32bits = firstBit1 ? mask31bits - positivePowerOfTwo(validNumBits - 1) : mask31bits;
                }
                else
                {
                    mask32bits = firstBit1 ? mask31bits + positivePowerOfTwo(validNumBits - 1) : mask31bits;
                }
                outputPtr[maskId] = mask32bits;
            }
        }
    }
}
} // namespace

void invokeConvertSpecDecodingMaskToPackedMask(SizeType32 batchSize,
    SizeType32 const* __restrict__ specDecodingCumGenerationLengths,
    SizeType32 const* __restrict__ specDecodingMaxGenerationLengths, bool const* __restrict__ specDecodingMask,
    SizeType32 const* __restrict__ batchSlots, SizeType32 maxDraftTokens, SizeType32 maxGenerationLength,
    SizeType32* __restrict__ specDecodingPackedMask, cudaStream_t stream)
{
    dim3 block(32);
    dim3 grid(maxGenerationLength, batchSize);
    size_t shmSize = maxGenerationLength * sizeof(char);
    getSpecDecodingPackedMask<<<grid, block, shmSize, stream>>>(specDecodingCumGenerationLengths,
        specDecodingMaxGenerationLengths, specDecodingMask, batchSlots, maxDraftTokens, specDecodingPackedMask);
}

namespace
{
template <typename T>
__global__ void extractExplicitDraftTokens(ExtractExplicitDraftTokensParams<T> params)
{
    auto const bid = static_cast<SizeType32>(blockIdx.x);
    auto const batchSlot = params.batchSlots ? params.batchSlots[bid] : bid;

    // Get accepted path len.
    // This tensor comes directly from engine and has linear batch index.
    auto const bestPathLength = params.bestPathLengths[bid];
    // Get accepted path idx.
    // This tensor comes directly from engine and has linear batch index.
    auto const bestPathIdx = params.bestPathIndices[bid];
    // Get current seq len (w/o newly accepted tokens).
    auto const curSeqLen = params.sequenceLengths[batchSlot];

    // Get output ids.
    auto* outputIdsRequest = params.outputIds + batchSlot * params.maxSeqLen;

    // First assemble accepted tokens and write them to output ids.
    for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < bestPathLength; ti += static_cast<SizeType32>(blockDim.x))
    {
        TokenIdType acceptedToken;
        // Read the last accepted token
        if (ti == bestPathLength - 1)
        {
            // Last accepted token is the first new draft token.
            // This tensor comes directly from engine and has linear batch index.
            auto const pathOffset = flat_index3(bid, 0, 0, params.numPaths, params.maxPathLength);
            // Read last accept token from new draft tokens.
            acceptedToken = params.nextDraftTokens[pathOffset];
        }
        else
        {
            // Read 1:bestPathLength slice of last draft tokens at best path idx.
            // This tensor comes directly from engine and has linear batch index.
            auto const pathOffset = flat_index3(bid, bestPathIdx, ti + 1, params.numPaths, params.maxPathLength);
            // Read accepted token from last draft tokens.
            acceptedToken = params.lastDraftTokens[pathOffset];
        }
        // Save accepted tokens to output ids.
        outputIdsRequest[curSeqLen + ti] = acceptedToken;
    }

    // Copy draft tokens and indices
    for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < params.numPaths * params.maxPathLength;
         ti += static_cast<SizeType32>(blockDim.x))
    {
        params.unpackedNextDraftTokens[batchSlot * params.numPaths * params.maxPathLength + ti]
            = params.nextDraftTokens[bid * params.numPaths * params.maxPathLength + ti];
        params.unpackedNextDraftIndices[batchSlot * params.numPaths * params.maxPathLength + ti]
            = params.inputUnpackedNextDraftIndices[bid * params.numPaths * params.maxPathLength + ti];
    }

    auto const numNextDraftTokens = (bid == 0)
        ? params.generationLengthInclusiveSum[0]
        : params.generationLengthInclusiveSum[bid] - params.generationLengthInclusiveSum[bid - 1];
    auto const startId = (bid == 0) ? 0 : params.generationLengthInclusiveSum[bid - 1];

    // Copy new draft tokens.
    for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < numNextDraftTokens - 1;
         ti += static_cast<SizeType32>(blockDim.x))
    {
        // Extract per request draft tokens from packed flat tokens where the 1st token is the "golden" token from
        // primary head.
        params.outputNextDraftTokens[batchSlot * params.numPaths * (params.maxPathLength - 1) + ti]
            = params.nextFlatTokens[startId + 1 + ti];
    }
    // Copy new pos ids.
    auto const maxDecodingTokens = (params.numPaths * (params.maxPathLength - 1) + 1);
    for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < numNextDraftTokens;
         ti += static_cast<SizeType32>(blockDim.x))
    {
        params.outputPositionIds[batchSlot * maxDecodingTokens + ti] = params.packedPositionIds[startId + ti];
    }

    for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < params.numPaths * (params.maxPathLength - 1);
         ti += static_cast<SizeType32>(blockDim.x))
    {
        // Generate new random data for token verification.
        // This tensor goes directly to engine and has linear batch index.
        auto const offset = flat_index2(batchSlot, ti, params.numPaths * (params.maxPathLength - 1));
        params.randDataVerification[offset] = static_cast<T>(curand_uniform(params.curandState + batchSlot));
    }

    // When all threads are done.
    __syncthreads();
    if (threadIdx.x == 0)
    {
        // Update pos id base.
        // This tensor goes directly to engine and has linear batch index.
        params.outputPositionIdsBase[batchSlot] = params.inputPositionIdsBase[bid] + bestPathLength;

        // Set number of accepted tokens at this iteration.
        params.acceptedLengths[batchSlot] = bestPathLength;

        // Set number of draft tokens for the next iteration.
        params.nextDraftLengths[batchSlot] = numNextDraftTokens - 1;

        // Generate new random data for sampling.
        // This tensor goes directly to engine and has linear batch index.
        params.randDataSample[batchSlot] = static_cast<T>(curand_uniform(params.curandState + batchSlot));
        // Increase seqLen by accepted len.
        params.sequenceLengths[batchSlot] = curSeqLen + bestPathLength;

        // Copy temperature.
        params.outputTemperatures[batchSlot] = params.inputTemperatures[batchSlot];
    }
}
} // namespace

template <typename T>
void invokeExtractExplicitDraftTokens(ExtractExplicitDraftTokensParams<T> const& params, cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 128;
    extractExplicitDraftTokens<<<params.batchSize, BLOCK_SIZE, 0, stream>>>(params);
}

template void invokeExtractExplicitDraftTokens(
    ExtractExplicitDraftTokensParams<float> const& params, cudaStream_t stream);
template void invokeExtractExplicitDraftTokens(
    ExtractExplicitDraftTokensParams<half> const& params, cudaStream_t stream);

namespace
{
template <typename VecT>
__global__ void copyProbs(uint8_t const* srcData, uint8_t* dstData, SizeType32 const* inputBatchSlots,
    SizeType32 const* outputBatchSlots, SizeType32 sizeInBytes)
{
    auto constexpr VEC_ELTS = static_cast<SizeType32>(sizeof(VecT));
    auto const bid = static_cast<SizeType32>(blockIdx.y);
    auto const intputBatchSlot = inputBatchSlots ? inputBatchSlots[bid] : bid;
    auto const outputBatchSlot = outputBatchSlots ? outputBatchSlots[bid] : bid;
    auto const srcStartIdx = intputBatchSlot * sizeInBytes;
    auto const dstStartIdx = outputBatchSlot * sizeInBytes;
    auto const tidx = (static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * VEC_ELTS;
    auto const stride = static_cast<std::size_t>(blockDim.x) * gridDim.x * VEC_ELTS;
    auto const srcEndIdx = srcStartIdx + sizeInBytes;

    auto srcIdx = srcStartIdx + tidx;
    auto dstIdx = dstStartIdx + tidx;

    for (; srcIdx < srcEndIdx; srcIdx += stride, dstIdx += stride)
    {
        *reinterpret_cast<VecT*>(&dstData[dstIdx]) = *reinterpret_cast<VecT const*>(&srcData[srcIdx]);
    }
}
} // namespace

void invokeCopyProbs(uint8_t const* srcDataPtr, uint8_t* dstDataPtr, SizeType32 const* inputBatchSlots,
    SizeType32 const* outputBatchSlots, SizeType32 batchSize, SizeType32 copyRowSizeInBytes, cudaStream_t stream)
{
    auto copyProbsInvocation = copyProbs<uint8_t>;
    if (copyRowSizeInBytes % 16 == 0)
    {
        copyProbsInvocation = copyProbs<uint4>;
    }
    else if (copyRowSizeInBytes % 8 == 0)
    {
        copyProbsInvocation = copyProbs<uint2>;
    }
    else if (copyRowSizeInBytes % 4 == 0)
    {
        copyProbsInvocation = copyProbs<uint32_t>;
    }
    else if (copyRowSizeInBytes % 2 == 0)
    {
        copyProbsInvocation = copyProbs<uint16_t>;
    }

    dim3 const blockSize{256};
    SizeType32 constexpr BLOCKS_PER_ROW{32};
    dim3 const gridSize{BLOCKS_PER_ROW, static_cast<uint32_t>(batchSize)};
    copyProbsInvocation<<<gridSize, blockSize, 0, stream>>>(
        srcDataPtr, dstDataPtr, inputBatchSlots, outputBatchSlots, copyRowSizeInBytes);
}

template <typename T>
void invokeCopyProbs(ExtractExplicitDraftTokensParams<T> const& params, cudaStream_t stream)
{
    auto srcDataPtr = reinterpret_cast<uint8_t const*>(params.nextDraftProbs);
    auto dstDataPtr = reinterpret_cast<uint8_t*>(params.outputDraftProbs);
    auto const numCopyElems = params.numPaths * (params.maxPathLength - 1) * params.vocabSize;
    auto const copyRowSizeInBytes = numCopyElems * sizeof(T);

    invokeCopyProbs(srcDataPtr, dstDataPtr, nullptr, params.batchSlots, params.batchSize, copyRowSizeInBytes, stream);
}

template void invokeCopyProbs(ExtractExplicitDraftTokensParams<float> const& params, cudaStream_t stream);
template void invokeCopyProbs(ExtractExplicitDraftTokensParams<half> const& params, cudaStream_t stream);

namespace
{
template <typename T>
__global__ void packExplicitDraftTokens(PackExplicitDraftTokensParams<T> params)
{
    auto const batchIdx = static_cast<SizeType32>(blockIdx.x);
    auto const batchSlot = params.batchSlots ? params.batchSlots[batchIdx] : batchIdx;

    if (threadIdx.x == 0)
    {
        params.outputPositionIdsBase[batchIdx] = params.inputPositionIdsBase[batchSlot];
        params.outputGenerationLengths[batchIdx] = params.inputGenerationLengths[batchSlot];
        params.outputRandomDataSample[batchIdx] = params.inputRandomDataSample[batchSlot];
        params.outputTemperatures[batchIdx] = params.inputTemperatures[batchSlot];
    }

    // Copy random validation data.
    auto const numDecodingDraftTokens = params.numPaths * (params.maxPathLength - 1);
    auto outputRandomDataValidation = params.outputRandomDataValidation + batchIdx * numDecodingDraftTokens;
    auto inputRandomDataValidation = params.inputRandomDataValidation + batchSlot * numDecodingDraftTokens;
    for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < numDecodingDraftTokens;
         ti += static_cast<SizeType32>(blockDim.x))
    {
        outputRandomDataValidation[ti] = inputRandomDataValidation[ti];
    }

    // Copy draft tokens and indices
    auto const numUnpackedTokens = numDecodingDraftTokens + params.numPaths;
    auto outputNextDraftTokens = params.outputNextDraftTokens + batchIdx * numUnpackedTokens;
    auto outputNextDraftIndices = params.outputNextDraftIndices + batchIdx * numUnpackedTokens;
    auto const inputNextDraftTokens = params.inputNextDraftTokens + batchSlot * numUnpackedTokens;
    auto const inputNextDraftIndices = params.inputNextDraftIndices + batchSlot * numUnpackedTokens;
    for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < numUnpackedTokens;
         ti += static_cast<SizeType32>(blockDim.x))
    {
        outputNextDraftTokens[ti] = inputNextDraftTokens[ti];
        outputNextDraftIndices[ti] = inputNextDraftIndices[ti];
    }

    auto const maxGenerationLength = params.maxGenerationLength[0];
    auto const maxDecodingTokens = numDecodingDraftTokens + 1;
    auto const numPackedMasks = divUp(maxGenerationLength, 32);
    auto const outputMaskStartId = (batchIdx == 0) ? 0 : params.cumSumGenerationLengths[batchIdx - 1];
    auto const numTokens = (batchIdx == 0)
        ? params.cumSumGenerationLengths[0]
        : params.cumSumGenerationLengths[batchIdx] - params.cumSumGenerationLengths[batchIdx - 1];
    // Copy packed masks.
    // Masks are placed next to each other with offsets of cumSumGenerationLengths[bi-1]
    auto const inputPackedMask = params.inputPackedMask + batchSlot * numPackedMasks * maxDecodingTokens;
    auto outputPackedMask = params.outputPackedMask + outputMaskStartId * numPackedMasks;
    for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < numTokens * numPackedMasks;
         ti += static_cast<SizeType32>(blockDim.x))
    {
        outputPackedMask[ti] = inputPackedMask[ti];
    }

    // Copy pos offsets. Copy only for maxGenerationLength
    auto outputPositionOffsets = params.outputPositionOffsets + batchIdx * maxGenerationLength;
    auto const inputPositionOffsets = params.inputPositionOffsets + batchSlot * maxDecodingTokens;
    for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < maxGenerationLength;
         ti += static_cast<SizeType32>(blockDim.x))
    {
        outputPositionOffsets[ti] = inputPositionOffsets[ti];
    }
}
} // namespace

template <typename T>
void invokePackExplicitDraftTokens(PackExplicitDraftTokensParams<T> const& params, cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 128;
    packExplicitDraftTokens<<<params.batchSize, BLOCK_SIZE, 0, stream>>>(params);
}

template void invokePackExplicitDraftTokens(PackExplicitDraftTokensParams<float> const& params, cudaStream_t stream);
template void invokePackExplicitDraftTokens(PackExplicitDraftTokensParams<half> const& params, cudaStream_t stream);

template <typename T>
void invokeCopyProbs(PackExplicitDraftTokensParams<T> const& params, cudaStream_t stream)
{
    auto srcDataPtr = reinterpret_cast<uint8_t const*>(params.inputDraftProbs);
    auto dstDataPtr = reinterpret_cast<uint8_t*>(params.outputDraftProbs);
    auto const numCopyElems = params.numPaths * (params.maxPathLength - 1) * params.vocabSize;
    auto const copyRowSizeInBytes = numCopyElems * sizeof(T);

    invokeCopyProbs(srcDataPtr, dstDataPtr, params.batchSlots, nullptr, params.batchSize, copyRowSizeInBytes, stream);
}

template void invokeCopyProbs(PackExplicitDraftTokensParams<float> const& params, cudaStream_t stream);
template void invokeCopyProbs(PackExplicitDraftTokensParams<half> const& params, cudaStream_t stream);
} // namespace tensorrt_llm::kernels::speculative_decoding
