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

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/kernels/speculativeDecoding/explicitDraftTokensKernels.h"
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif
#if ENABLE_BF16
#include <cuda_bf16.h>
#endif // ENABLE_BF16

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::kernels::speculative_decoding
{
size_t invokeScanGenerationLengths(void* __restrict__ scanTempStorage, size_t scanTempStorageBytes,
    SizeType32 const* __restrict__ generationLengths, SizeType32* __restrict__ scannedGenerationLengths,
    SizeType32 batchSize, cudaStream_t stream)
{
    cub::DeviceScan::InclusiveSum(
        scanTempStorage, scanTempStorageBytes, generationLengths, scannedGenerationLengths, batchSize, stream);
    return scanTempStorageBytes;
}

size_t invokeReduceMaxGenerationLengths(void* __restrict__ reduceMaxTempStorage, size_t reduceTempStorageBytes,
    SizeType32 const* __restrict__ generationLengths, SizeType32* __restrict__ maxGenerationLengths,
    SizeType32 batchSize, cudaStream_t stream)
{
    cub::DeviceReduce::Max(
        reduceMaxTempStorage, reduceTempStorageBytes, generationLengths, maxGenerationLengths, batchSize, stream);
    return reduceTempStorageBytes;
}

// inclusive prefix sum generationLengths and reduce max generationLengths
size_t invokeScanReduceGenerationLengths(SizeType32 batchSize, SizeType32 const* __restrict__ generationLengths,
    void* scanReduceTempStorage, size_t scanReduceTempStorageBytes, SizeType32* __restrict__ scannedGenerationLengths,
    SizeType32* maxGenerationLengths, cudaStream_t stream)
{
    auto scanTempStorageBytes = invokeScanGenerationLengths(scanReduceTempStorage, scanReduceTempStorageBytes,
        generationLengths, scannedGenerationLengths, batchSize, stream);
    auto reduceTempStorageBytes = invokeReduceMaxGenerationLengths(
        scanReduceTempStorage, scanReduceTempStorageBytes, generationLengths, maxGenerationLengths, batchSize, stream);
    return std::max(scanTempStorageBytes, reduceTempStorageBytes);
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

__global__ void getPackedMask(SizeType32 const* __restrict__ cumGenerationLengths,
    SizeType32 const* __restrict__ maxGenerationLengths, bool const* __restrict__ mask,
    SizeType32 const* __restrict__ batchSlots, SizeType32 maxDraftTokens, SizeType32* __restrict__ packedMask)
{
    auto const batchIdx = static_cast<SizeType32>(blockIdx.y);
    auto const tokenIdx = static_cast<SizeType32>(blockIdx.x);

    auto const numTokens = (batchIdx == 0) ? cumGenerationLengths[0]
                                           : cumGenerationLengths[batchIdx] - cumGenerationLengths[batchIdx - 1];
    if (tokenIdx >= numTokens)
    {
        return;
    }

    auto const maxGenerationLength = maxGenerationLengths[0];
    auto const numPackedMasks = divUp(maxDraftTokens + 1, 32);

    auto const outputStartId = batchSlots ? (batchSlots[batchIdx] * (maxDraftTokens + 1))
                                          : ((batchIdx == 0) ? 0 : cumGenerationLengths[batchIdx - 1]);
    auto* outputPtr = packedMask + (outputStartId + tokenIdx) * numPackedMasks;
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
        bool const* maskPtr
            = mask + batchIdx * maxGenerationLength * maxGenerationLength + tokenIdx * maxGenerationLength + 1;
        extern __shared__ char shMask[];
        if (threadIdx.x == 0)
        {
            shMask[maxGenerationLength - 1] = '1';
        }
        for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < maxGenerationLength - 1;
             ti += static_cast<SizeType32>(blockDim.x))
        {
            auto const shIndex = maxGenerationLength - 1 - ti - 1;
            shMask[shIndex] = maskPtr[ti] ? '1' : '0';
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
                auto const shMaskIndexStart
                    = ((maxGenerationLength - (maskId + 1) * 32) < 0) ? 0 : (maxGenerationLength - (maskId + 1) * 32);
                auto const shMaskIndexEnd = maxGenerationLength - (maskId * 32 + 1) + 1;

                auto const validNumBits = shMaskIndexEnd - shMaskIndexStart;
                auto const firstBit1 = (shMask[shMaskIndexStart] == '1') ? true : false;
                SizeType32 mask31bits = 0;
                if (validNumBits != 1)
                {
                    for (auto i = shMaskIndexStart + 1; i < shMaskIndexEnd; i++)
                    {
                        auto const index = (validNumBits - 1) - (i - shMaskIndexStart - 1) - 1;
                        mask31bits += (shMask[i] == '1') ? positivePowerOfTwo(index) : 0;
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

void invokeConvertMaskToPackedMask(SizeType32 batchSize, SizeType32 const* __restrict__ cumGenerationLengths,
    SizeType32 const* __restrict__ maxGenerationLengths, bool const* __restrict__ mask,
    SizeType32 const* __restrict__ batchSlots, SizeType32 maxDraftTokens, SizeType32 maxGenerationLength,
    SizeType32* __restrict__ packedMask, cudaStream_t stream)
{
    dim3 block(32);
    dim3 grid(maxGenerationLength, batchSize);
    size_t shmSize = maxGenerationLength * sizeof(char);
    getPackedMask<<<grid, block, shmSize, stream>>>(
        cumGenerationLengths, maxGenerationLengths, mask, batchSlots, maxDraftTokens, packedMask);
}

namespace
{
template <typename T>
__global__ void fillContextBuffers(FillContextExplicitDraftTokensParams<T> params)
{
    auto const bid = static_cast<SizeType32>(blockIdx.x);
    auto const batchSlot = params.batchSlots[bid];

    if (threadIdx.x == 0)
    {
        // Generate new random data for sampling.
        params.randDataSample[batchSlot] = static_cast<T>(curand_uniform(params.curandState + batchSlot));

        // Copy temperature.
        params.outputTemperatures[batchSlot] = __frcp_rn(params.inputTemperatures[batchSlot]);
    }
}
} // namespace

template <typename T>
void invokeFillContextBuffers(FillContextExplicitDraftTokensParams<T> const& params, cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 32;
    fillContextBuffers<<<params.batchSize, BLOCK_SIZE, 0, stream>>>(params);
}

template void invokeFillContextBuffers(FillContextExplicitDraftTokensParams<float> const& params, cudaStream_t stream);
template void invokeFillContextBuffers(FillContextExplicitDraftTokensParams<half> const& params, cudaStream_t stream);
#if ENABLE_BF16
template void invokeFillContextBuffers(
    FillContextExplicitDraftTokensParams<__nv_bfloat16> const& params, cudaStream_t stream);
#endif // ENABLE_BF16

namespace
{
// params.skipVerification == true must be similar to fillContextBuffers
// params.skipVerification == false must be similar to extractExplicitDraftTokens
template <typename T>
__global__ void fillRandData(FillRandDataExplicitDraftTokensParams<T> const params)
{
    if (threadIdx.x == 0)
    {
        auto const bid = static_cast<SizeType32>(blockIdx.x);
        auto const batchSlot = params.batchSlots ? params.batchSlots[bid] : bid;

        auto curandState = params.curandState[batchSlot];

        // Generate new random data for sampling.
        params.randDataSample[batchSlot] = static_cast<T>(curand_uniform(&curandState));

        if (!params.skipVerification)
        {
            for (auto idx = 0; idx < params.numPaths * params.draftLength; idx++)
            {
                // Generate new random data for token verification.
                auto const offset = flat_index2(batchSlot, idx, params.numPaths * params.draftLength);
                params.randDataVerification[offset] = static_cast<T>(curand_uniform(&curandState));
            }
        }

        params.curandState[batchSlot] = curandState;
    }
}
} // namespace

template <typename T>
void invokeFillRandData(FillRandDataExplicitDraftTokensParams<T> const& params, cudaStream_t stream)
{
    params.checkParams();

    SizeType32 constexpr BLOCK_SIZE = 32;
    fillRandData<<<params.batchSize, BLOCK_SIZE, 0, stream>>>(params);
}

template void invokeFillRandData(FillRandDataExplicitDraftTokensParams<float> const& params, cudaStream_t stream);
template void invokeFillRandData(FillRandDataExplicitDraftTokensParams<half> const& params, cudaStream_t stream);
#if ENABLE_BF16
template void invokeFillRandData(
    FillRandDataExplicitDraftTokensParams<__nv_bfloat16> const& params, cudaStream_t stream);
#endif // ENABLE_BF16

namespace
{
template <typename T>
__global__ void extractExplicitDraftTokens(ExtractExplicitDraftTokensParams<T> params)
{
    auto const bid = static_cast<SizeType32>(blockIdx.x);
    auto const batchSlot = params.batchSlots[bid];

    // Get accepted path len.
    // This tensor comes directly from engine and has linear batch index.
    auto const bestPathLength = params.bestPathLengths[bid];
    // Get accepted path idx.
    // This tensor comes directly from engine and has linear batch index.
    auto const bestPathIdx = params.bestPathIndices[bid];
    // Get current seq len (w/o newly accepted tokens).
    auto const curSeqLen = params.sequenceLengths[batchSlot];
    // `last*` tensors do not have data for context requests.
    auto const lastTensorBid = bid - params.numContextRequests;

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
            auto const pathOffset
                = flat_index3(lastTensorBid, bestPathIdx, ti + 1, params.numPaths, params.maxPathLength);
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
        if (lastTensorBid >= 0)
        {
            params.outputLastDraftIndices[batchSlot * params.numPaths * params.maxPathLength + ti]
                = params.lastDraftIndices[lastTensorBid * params.numPaths * params.maxPathLength + ti];
        }
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
        params.outputPositionIds[batchSlot * maxDecodingTokens + ti] = params.packedPositionIds[startId + ti] - 1;
    }

    // When all threads are done.
    __syncthreads();
    if (threadIdx.x == 0)
    {
        // Update pos id base.
        params.outputPositionIdsBase[batchSlot] = params.inputPositionIdsBase[bid] + bestPathLength;

        // Set number of accepted tokens at this iteration.
        params.acceptedLengths[batchSlot] = bestPathLength;

        // Set number of draft tokens for the next iteration.
        params.prevDraftLengths[batchSlot] = params.nextDraftLengths[batchSlot];

        // Set number of draft tokens for the next iteration.
        params.nextDraftLengths[batchSlot] = numNextDraftTokens - 1;

        // Set number of tokens passed to the engine per request for the next iteration.
        params.outputGenerationLengths[batchSlot] = numNextDraftTokens;

        auto curandState = params.curandState[batchSlot];
        // Generate new random data for sampling.
        params.randDataSample[batchSlot] = static_cast<T>(curand_uniform(&curandState));
        for (auto idx = 0; idx < params.numPaths * (params.maxPathLength - 1); idx++)
        {
            // Generate new random data for token verification.
            auto const offset = flat_index2(batchSlot, idx, params.numPaths * (params.maxPathLength - 1));
            params.randDataVerification[offset] = static_cast<T>(curand_uniform(&curandState));
        }
        params.curandState[batchSlot] = curandState;

        // Increase seqLen by accepted len.
        params.sequenceLengths[batchSlot] = curSeqLen + bestPathLength;

        // Copy temperature.
        params.outputTemperatures[batchSlot] = __frcp_rn(params.inputTemperatures[batchSlot]);

        // Copy best path index.
        params.outputBestPathIndices[batchSlot] = bestPathIdx;
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
#if ENABLE_BF16
template void invokeExtractExplicitDraftTokens(
    ExtractExplicitDraftTokensParams<__nv_bfloat16> const& params, cudaStream_t stream);
#endif // ENABLE_BF16

namespace
{
template <typename VecT>
__global__ void copyProbs(uint8_t const* srcData, uint8_t* dstData, SizeType32 const* inputBatchSlots,
    SizeType32 const* outputBatchSlots, SizeType32 sizeInBytes, SizeType32 inputBatchIdxOffset)
{
    auto constexpr VEC_ELTS = static_cast<SizeType32>(sizeof(VecT));
    auto const inputBid = static_cast<SizeType32>(blockIdx.y) + inputBatchIdxOffset;
    auto const outputBid = static_cast<SizeType32>(blockIdx.y);
    auto const intputBatchSlot = inputBatchSlots ? inputBatchSlots[inputBid] : inputBid;
    auto const outputBatchSlot = outputBatchSlots ? outputBatchSlots[outputBid] : outputBid;
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
    SizeType32 const* outputBatchSlots, SizeType32 batchSize, SizeType32 inputBatchIdxOffset,
    SizeType32 copyRowSizeInBytes, cudaStream_t stream)
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
        srcDataPtr, dstDataPtr, inputBatchSlots, outputBatchSlots, copyRowSizeInBytes, inputBatchIdxOffset);
}

template <typename T>
void invokeCopyProbs(ExtractExplicitDraftTokensParams<T> const& params, cudaStream_t stream)
{
    auto srcDataPtr = reinterpret_cast<uint8_t const*>(params.nextDraftProbs);
    auto dstDataPtr = reinterpret_cast<uint8_t*>(params.outputDraftProbs);
    auto const numCopyElems = params.numPaths * (params.maxPathLength - 1) * params.vocabSize;
    auto const copyRowSizeInBytes = numCopyElems * sizeof(T);

    invokeCopyProbs(
        srcDataPtr, dstDataPtr, nullptr, params.batchSlots, params.batchSize, 0, copyRowSizeInBytes, stream);
}

template void invokeCopyProbs(ExtractExplicitDraftTokensParams<float> const& params, cudaStream_t stream);
template void invokeCopyProbs(ExtractExplicitDraftTokensParams<half> const& params, cudaStream_t stream);
#if ENABLE_BF16
template void invokeCopyProbs(ExtractExplicitDraftTokensParams<__nv_bfloat16> const& params, cudaStream_t stream);
#endif // ENABLE_BF16

namespace
{
template <typename T>
__global__ void packGenerationLengths(PackExplicitDraftTokensParams<T> params)
{
    auto const batchIdx = static_cast<SizeType32>(blockIdx.x);
    auto const batchSlot = params.batchSlots[batchIdx];

    auto const isGenerationRequest = batchIdx >= params.numContextRequests;
    auto const genIdx = batchIdx - params.numContextRequests;

    if (threadIdx.x == 0 && isGenerationRequest)
    {
        params.outputGenerationLengths[genIdx] = params.inputGenerationLengths[batchSlot];
    }
}
} // namespace

template <typename T>
void invokePackGenerationLengths(PackExplicitDraftTokensParams<T> const& params, cudaStream_t stream)
{
    SizeType32 constexpr BLOCK_SIZE = 32;
    packGenerationLengths<<<params.batchSize, BLOCK_SIZE, 0, stream>>>(params);
}

template void invokePackGenerationLengths(PackExplicitDraftTokensParams<float> const& params, cudaStream_t stream);
template void invokePackGenerationLengths(PackExplicitDraftTokensParams<half> const& params, cudaStream_t stream);
#if ENABLE_BF16
template void invokePackGenerationLengths(
    PackExplicitDraftTokensParams<__nv_bfloat16> const& params, cudaStream_t stream);
#endif // ENABLE_BF16

namespace
{
template <typename T>
__global__ void packExplicitDraftTokens(PackExplicitDraftTokensParams<T> params)
{
    auto const batchIdx = static_cast<SizeType32>(blockIdx.x);
    auto const batchSlot = params.batchSlots[batchIdx];

    auto const isGenerationRequest = batchIdx >= params.numContextRequests;
    auto const genIdx = batchIdx - params.numContextRequests;

    if (threadIdx.x == 0)
    {
        params.outputPositionIdsBase[batchIdx] = params.inputPositionIdsBase[batchSlot];
        params.outputRandomDataSample[batchIdx] = params.inputRandomDataSample[batchSlot];
        params.outputTemperatures[batchIdx] = params.inputTemperatures[batchSlot];
    }

    if (isGenerationRequest)
    {
        // Copy random validation data.
        auto const numDecodingDraftTokens = params.numPaths * (params.maxPathLength - 1);
        auto outputRandomDataValidation = params.outputRandomDataValidation + genIdx * numDecodingDraftTokens;
        auto const inputRandomDataValidation = params.inputRandomDataValidation + batchSlot * numDecodingDraftTokens;
        for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < numDecodingDraftTokens;
             ti += static_cast<SizeType32>(blockDim.x))
        {
            outputRandomDataValidation[ti] = inputRandomDataValidation[ti];
        }

        // Copy draft tokens and indices
        auto const numUnpackedTokens = numDecodingDraftTokens + params.numPaths;
        auto outputNextDraftTokens = params.outputNextDraftTokens + genIdx * numUnpackedTokens;
        auto outputNextDraftIndices = params.outputNextDraftIndices + genIdx * numUnpackedTokens;
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
        auto const numPackedMasks = divUp(maxDecodingTokens, 32);
        auto const outputStartId = (genIdx == 0) ? 0 : params.cumSumGenerationLengths[genIdx - 1];
        auto const numTokens = (genIdx == 0)
            ? params.cumSumGenerationLengths[0]
            : params.cumSumGenerationLengths[genIdx] - params.cumSumGenerationLengths[genIdx - 1];
        // Copy packed masks.
        // Masks are placed next to each other with offsets of cumSumGenerationLengths[bi-1]
        auto const inputPackedMask = params.inputPackedMask + batchSlot * numPackedMasks * maxDecodingTokens;
        auto outputPackedMask = params.outputPackedMask + outputStartId * numPackedMasks;
        for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < numTokens * numPackedMasks;
             ti += static_cast<SizeType32>(blockDim.x))
        {
            outputPackedMask[ti] = inputPackedMask[ti];
        }
        auto const inputPositionIds = params.inputPositionIds + batchSlot * maxDecodingTokens;
        auto outputPositionIds = params.outputPositionIds + params.numContextTokens + outputStartId;
        for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < numTokens; ti += static_cast<SizeType32>(blockDim.x))
        {
            outputPositionIds[ti] = inputPositionIds[ti];
        }

        // Copy pos offsets. Copy only for maxGenerationLength
        auto const basePosId = params.outputPositionIdsBase[batchIdx];
        auto outputPositionOffsets = params.outputPositionOffsets + genIdx * maxGenerationLength;
        for (auto ti = static_cast<SizeType32>(threadIdx.x); ti < maxGenerationLength;
             ti += static_cast<SizeType32>(blockDim.x))
        {
            outputPositionOffsets[ti] = inputPositionIds[ti] - basePosId + 1;
        }
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
#if ENABLE_BF16
template void invokePackExplicitDraftTokens(
    PackExplicitDraftTokensParams<__nv_bfloat16> const& params, cudaStream_t stream);
#endif // ENABLE_BF16

template <typename T>
void invokeCopyProbs(PackExplicitDraftTokensParams<T> const& params, cudaStream_t stream)
{
    auto srcDataPtr = reinterpret_cast<uint8_t const*>(params.inputDraftProbs);
    auto dstDataPtr = reinterpret_cast<uint8_t*>(params.outputDraftProbs);
    auto const numCopyElems = params.numPaths * (params.maxPathLength - 1) * params.vocabSize;
    auto const copyRowSizeInBytes = numCopyElems * sizeof(T);

    invokeCopyProbs(srcDataPtr, dstDataPtr, params.batchSlots, nullptr, params.numGenerationRequests,
        params.numContextRequests, copyRowSizeInBytes, stream);
}

template void invokeCopyProbs(PackExplicitDraftTokensParams<float> const& params, cudaStream_t stream);
template void invokeCopyProbs(PackExplicitDraftTokensParams<half> const& params, cudaStream_t stream);
#if ENABLE_BF16
template void invokeCopyProbs(PackExplicitDraftTokensParams<__nv_bfloat16> const& params, cudaStream_t stream);
#endif // ENABLE_BF16

} // namespace tensorrt_llm::kernels::speculative_decoding
