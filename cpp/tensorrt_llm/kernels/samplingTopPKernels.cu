/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
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
#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/samplingTopKKernels.h"
#include "tensorrt_llm/kernels/samplingTopPKernels.h"

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::kernels
{
__global__ void topPInitialize(TokenIdType* topPIdValBuf, SizeType32* topPOffsetBuf, SizeType32* beginTopPOffsetBuf,
    SizeType32 batchSize, SizeType32 vocabSize)
{
    auto const tid = static_cast<SizeType32>(threadIdx.x);
    auto const bid = static_cast<SizeType32>(blockIdx.x);

    if (bid == 0)
    {
        for (auto i = tid; i < batchSize + 1; i += static_cast<SizeType32>(blockDim.x))
        {
            // Inclusive sum of offsets to vocab rows
            topPOffsetBuf[i] = i * vocabSize;
            beginTopPOffsetBuf[i] = topPOffsetBuf[i];
        }
    }

    auto index = tid + bid * static_cast<SizeType32>(blockDim.x);

    while (index < batchSize * vocabSize)
    {
        // Set value at {bi, vi} position to vi
        topPIdValBuf[index] = index % vocabSize;
        index += static_cast<SizeType32>(blockDim.x * gridDim.x);
    }
}

void invokeTopPInitialize(TokenIdType* topPIdValBuf, SizeType32* topPOffsetBuf, SizeType32* beginTopPOffsetBuf,
    SizeType32 batchSize, SizeType32 vocabSize, cudaStream_t stream)
{
    // vocabSize: the column number of logits_buffer for top_p sampling
    // TODO: launch based on available resources
    topPInitialize<<<32, 512, 0, stream>>>(topPIdValBuf, topPOffsetBuf, beginTopPOffsetBuf, batchSize, vocabSize);
}

template <typename T, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void topPBeamTopKKernel(T const* probs, // prob.
    TokenIdType* topKTmpIdBuf, T* topKTmpValBuf, FinishedState const* finishedInput, SizeType32 vocabSize,
    SizeType32* offsetBuf, SizeType32* beginOffsetBuf, float const* topPs, bool const* skipDecode,
    SizeType32 const* batchSlots)
{
    /**
     * Kernel performs top 1 search and saves the token with largest probability if it exceeds probability threshold
     */
    SizeType32 constexpr MAX_K = 1;
    auto const threadId = static_cast<SizeType32>(threadIdx.x);
    auto const batchId = static_cast<SizeType32>(blockIdx.x);
    auto const batchSlot = batchSlots[batchId];

    // Skip decoding kernel if configured
    if ((skipDecode != nullptr && skipDecode[batchSlot])
        || (finishedInput != nullptr && finishedInput[batchSlot].isSkipDecoding()))
    {
        // Required to skip radix sort
        beginOffsetBuf[batchId] += vocabSize;
        return;
    }

    float pThreshold = topPs[batchSlot];

    typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    TopK<T, MAX_K> partial;

    bool const IS_FP16 = std::is_same<T, half>::value;
    T const MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

#pragma unroll
    for (SizeType32 i = 0; i < MAX_K; ++i)
    {
        partial.p[i] = -1;
        partial.u[i] = -MAX_T_VAL;
    }

#pragma unroll
    for (SizeType32 elemId = static_cast<SizeType32>(threadId); elemId < vocabSize; elemId += THREADBLOCK_SIZE)
    {
        auto index = elemId + batchId * vocabSize;
        partial.insert(probs[index], elemId);
    }

    TopK<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

    if (threadId == 0)
    {
        beginOffsetBuf[batchId] = offsetBuf[batchId];
        T sumProb = (T) (0.0f);

#pragma unroll
        for (SizeType32 i = 0; i < MAX_K; i++)
        {
            sumProb += total.u[i];
        }

        if ((float) sumProb >= pThreshold)
        {
            beginOffsetBuf[batchId] += vocabSize;
            auto index = batchId * vocabSize;

#pragma unroll
            for (SizeType32 i = 0; i < MAX_K; ++i)
            {
                topKTmpIdBuf[index + i] = total.p[i];
                topKTmpValBuf[index + i] = total.u[i];
            }
        }
    }
}

struct BlockPrefixCallbackOp
{
    // Running prefix
    float running_total;

    // Constructor
    __device__ BlockPrefixCallbackOp(float running_total)
        : running_total(running_total)
    {
    }

    // Callback operator to be entered by the first warp of threads in the block.
    // Thread-0 is responsible for returning a value for seeding the block-wide
    // scan.
    __device__ float operator()(float block_aggregate)
    {
        float old_prefix = running_total;
        running_total += block_aggregate;
        return old_prefix;
    }
};

template <typename T>
__device__ void epilogue(SizeType32 batchId, SizeType32 currentStep, SizeType32 offset, TokenIdType** ids,
    TokenIdType const* sortedIdVals, T const* sortedProbs, float* cumLogProbs, float* outputLogProbs,
    TokenIdType const* endIds, SizeType32* sequenceLengths, FinishedState* finishedOutput, SizeType32 maxBatchSize)
{
    ids[batchId][currentStep] = sortedIdVals[offset];

    if (cumLogProbs != nullptr || outputLogProbs != nullptr)
    {
        float lprob = logf(sortedProbs[offset]);
        if (cumLogProbs != nullptr)
        {
            cumLogProbs[batchId] += lprob;
        }
        if (outputLogProbs != nullptr)
        {
            outputLogProbs[sequenceLengths[batchId] * maxBatchSize + batchId] = lprob;
        }
    }
    if (finishedOutput != nullptr && endIds != nullptr)
    {
        if (ids[batchId][currentStep] == endIds[batchId])
        {
            finishedOutput[batchId].setFinishedEOS();
            // Do not increase seq len when EOS is generated. Seq len should always contain only tokens to be outputted
        }
        else
        {
            // We don't need to set output finished state as it is assumed to be in non finished state
            sequenceLengths[batchId] += 1;
        }
    }
}

template <typename T, int blockSize>
__global__ void topPSsampling(T const* sortedProbs, TokenIdType const* sortedIdVals, TokenIdType* ids,
    TokenIdType** idsPtrs, SizeType32* sequenceLength, FinishedState const* finishedInput,
    FinishedState* finishedOutput, float* cumLogProbs, float* outputLogProbs, SizeType32 const* beginOffsetBuf,
    SizeType32 const* offsetBuf, SizeType32 vocabSize, curandState_t* curandState, float const* randomVals,
    float const* topPs, TokenIdType const* endIds, SizeType32 maxBatchSize, bool const* skipDecode,
    SizeType32 const* batchSlots, bool returnAllSelectedTokensFlag, bool const* returnAllSelectedTokensPerSlot,
    SizeType32 maxSeqLen, TokenIdType* outputIdCurrentStep, bool const* skipOutputIdCurrentStep)
{
    /**
     * Each block processes one request row sorted in descending order by probabilities.
     * All threads within block compute running sum of probabilities until one of the threads exceeds the randomly
     * chosen probability threshold. Thread that crossed probaility threshold writes the corresponding token to the
     * output.
     */

    __shared__ float randNumS;
    __shared__ float randNumS2;

    auto const tid = static_cast<SizeType32>(threadIdx.x);
    auto const batchId = static_cast<SizeType32>(blockIdx.x);
    auto const batchSlot = batchSlots[batchId];
    // Skip kernel if this sampling method is not chosen
    FinishedState const finishState = finishedInput != nullptr ? finishedInput[batchSlot] : FinishedState::empty();
    if ((skipDecode != nullptr && skipDecode[batchSlot]) || (finishState.isSkipDecoding()))
    {
        return;
    }

    // Exit early if sequence has finished
    if (finishState.isFinished())
    {
        if (tid == 0)
        {
            if (finishedOutput != nullptr)
            {
                finishedOutput[batchSlot] = finishState;
            }
        }
        return;
    }

    auto const probThreshold = topPs[batchSlot];
    auto const currentStep = sequenceLength == nullptr ? 0 : sequenceLength[batchSlot];
    auto* outputIdsRequestPtr = idsPtrs == nullptr ? ids + batchSlot * maxSeqLen : idsPtrs[batchSlot];
    auto const returnAllSelectedTokens = returnAllSelectedTokensPerSlot != nullptr
        ? returnAllSelectedTokensPerSlot[batchSlot]
        : returnAllSelectedTokensFlag;
    bool const sampleTokenInSelected = returnAllSelectedTokens && outputIdCurrentStep && curandState
        && skipOutputIdCurrentStep && !skipOutputIdCurrentStep[batchSlot];

    // With P in (0.0; 1.0] we draw a random number P' in range (0.0; P]
    // We will sum all probs moving from the largest probability to the smallest and
    // will choose the token which probability makes cumulative probability sum to exceed P'
    if (threadIdx.x == 0)
    {
        // if we want to return all top p indices, we should not do random sampling for probThreshold
        auto const randomNumber = randomVals ? randomVals[batchSlot] : curand_uniform(curandState + batchSlot);
        randNumS = returnAllSelectedTokens ? probThreshold : randomNumber * probThreshold;
        randNumS2 = sampleTokenInSelected ? curand_uniform(curandState + batchSlot) * probThreshold : 0.0f;
    }

    // if beginOffsetBuf and offsetBuf of sorting have same value,
    // this means that we have find best one in topPBeamTopKKernel
    // So, we can skip this sampling.
    if (beginOffsetBuf[batchId] == offsetBuf[batchId])
    {
        if (tid == 0)
        {
            auto offset = batchId * vocabSize;
            if (returnAllSelectedTokens)
            {
                outputIdsRequestPtr[currentStep] = sortedIdVals[offset];
            }
            else
            {
                epilogue(batchSlot, currentStep, offset, idsPtrs, sortedIdVals, sortedProbs, cumLogProbs,
                    outputLogProbs, endIds, sequenceLength, finishedOutput, maxBatchSize);
            }
        }
        return;
    }

    typedef cub::BlockScan<float, blockSize> BlockScan;
    __shared__ typename BlockScan::TempStorage tempStorage;
    // Initialize running total
    BlockPrefixCallbackOp prefixOp(0);

    __syncthreads();

    auto offset = batchId * vocabSize;
    outputIdsRequestPtr[currentStep] = sortedIdVals[offset];
    auto end = ((vocabSize + blockSize - 1) / blockSize) * blockSize;
    SizeType32 selectedTokenId = 0;
    // Cumulative sum
    float threadOffset = 0;
    SizeType32 count = 0;
    // For sampleTokenInSelected == True
    SizeType32 selectedTokenId2 = 0;
    SizeType32 count2 = 0;
    for (int vi = tid; vi < end; vi += blockSize)
    {
        auto threadProb = (vi < vocabSize) ? static_cast<float>(sortedProbs[offset + vi]) : 0.f;
        BlockScan(tempStorage).InclusiveSum(threadProb, threadOffset, prefixOp);
        count = __syncthreads_count(randNumS <= threadOffset);
        selectedTokenId = vi;
        if (sampleTokenInSelected && count2 == 0)
        {
            count2 = __syncthreads_count(randNumS2 <= threadOffset);
            selectedTokenId2 = vi;
        }
        if (count != 0)
        {
            break;
        }
    }

    selectedTokenId = min(selectedTokenId, vocabSize - 1);

    if (returnAllSelectedTokens)
    {
        __shared__ SizeType32 sharedSelectedTokenId;
        if (sampleTokenInSelected && (threadIdx.x == min(blockDim.x - count2, blockDim.x - 1)))
        {
            selectedTokenId2 = min(selectedTokenId2, vocabSize - 1);
            outputIdCurrentStep[batchSlot] = sortedIdVals[offset + selectedTokenId2];
        }
        if (threadIdx.x == min(blockDim.x - count, blockDim.x - 1))
        {
            sharedSelectedTokenId = selectedTokenId;
        }
        __syncthreads();
        for (int vi = tid; vi <= sharedSelectedTokenId; vi += blockSize)
        {
            outputIdsRequestPtr[vi] = sortedIdVals[offset + vi];
        }
        if (tid == 0 && sharedSelectedTokenId != end - 1)
        {
            outputIdsRequestPtr[sharedSelectedTokenId + 1] = -1; // a boundary to record the end of all selected top Ps.
        }
    }
    else
    {
        // select first thread exceeded the prob threshold or the last thread in case of P=1.0f
        if (threadIdx.x == min(blockDim.x - count, blockDim.x - 1))
        {
            epilogue(batchSlot, currentStep, offset + selectedTokenId, idsPtrs, sortedIdVals, sortedProbs, cumLogProbs,
                outputLogProbs, endIds, sequenceLength, finishedOutput, maxBatchSize);
        }
    }
}

template <typename T>
std::vector<size_t> getTopPWorkspaceSizes(SizeType32 batchSize, SizeType32 vocabSize)
{
    auto const sortedLogProbBufSize = sizeof(T) * batchSize * vocabSize;
    auto const sortedIdValsBufSize = sizeof(TokenIdType) * batchSize * vocabSize;
    auto const topPIdValsSize = sizeof(TokenIdType) * batchSize * vocabSize;
    auto const topPOffsetSize = sizeof(SizeType32) * (batchSize + 1);
    auto const beginTopPOffsetSize = sizeof(SizeType32) * (batchSize + 1);

    size_t cubTempStorageSize;
    tensorrt_llm::common::check_cuda_error(cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr,
        cubTempStorageSize, static_cast<T*>(nullptr), static_cast<T*>(nullptr), static_cast<SizeType32*>(nullptr),
        static_cast<SizeType32*>(nullptr), static_cast<SizeType32>(vocabSize * batchSize), batchSize,
        static_cast<SizeType32*>(nullptr), static_cast<SizeType32*>(nullptr),
        0,             // begin_bit
        sizeof(T) * 8, // end_bit = sizeof(KeyT) * 8
        0));           // cudaStream_t

    return {cubTempStorageSize, sortedLogProbBufSize, sortedIdValsBufSize, topPIdValsSize, topPOffsetSize,
        beginTopPOffsetSize};
}

template std::vector<size_t> getTopPWorkspaceSizes<float>(SizeType32 batchSize, SizeType32 vocabSize);
template std::vector<size_t> getTopPWorkspaceSizes<half>(SizeType32 batchSize, SizeType32 vocabSize);

[[nodiscard]] std::vector<size_t> getTopPInitWorkspaceSizes(SizeType32 batchSize)
{
    auto const tempTopKsBufSize = batchSize * sizeof(SizeType32);
    auto const tempTopPsBufSize = batchSize * sizeof(float);
    auto const tempTopPDecayBufSize = batchSize * sizeof(float);
    auto const tempTopPMinBufSize = batchSize * sizeof(float);
    auto const tempTopPResetIdsBufSize = batchSize * sizeof(TokenIdType);

    return {tempTopKsBufSize, tempTopPsBufSize, tempTopPDecayBufSize, tempTopPMinBufSize, tempTopPResetIdsBufSize};
}

template <typename T>
size_t getTopPWorkspaceSize(SizeType32 batchSize, SizeType32 vocabSizePadded)
{
    auto const workspaceSizes = getTopPWorkspaceSizes<T>(batchSize, vocabSizePadded);
    auto const initWorkspaceSizes = getTopPInitWorkspaceSizes(batchSize);
    return std::max(tensorrt_llm::common::calcAlignedSize(workspaceSizes, 256),
        tensorrt_llm::common::calcAlignedSize(initWorkspaceSizes, 256));
}

template size_t getTopPWorkspaceSize<float>(SizeType32 batchSize, SizeType32 vocabSizePadded);
template size_t getTopPWorkspaceSize<half>(SizeType32 batchSize, SizeType32 vocabSizePadded);

template <typename T>
void invokeBatchTopPSampling(TopPSamplingKernelParams<T> const& params, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    params.checkParams();

    auto const workspaceSizes = getTopPWorkspaceSizes<T>(params.batchSize, params.vocabSizePadded);

    std::vector<void*> alignedPointers;
    calcAlignedPointers(alignedPointers, params.workspace, workspaceSizes);

    auto cubTempStorage = static_cast<void*>(alignedPointers[0]);
    auto sortedProbs = static_cast<T*>(alignedPointers[1]);
    auto sortedIdVals = static_cast<TokenIdType*>(alignedPointers[2]);
    auto idVals = static_cast<TokenIdType*>(alignedPointers[3]);
    auto offsetBuf = static_cast<SizeType32*>(alignedPointers[4]);
    auto beginOffsetBuf = static_cast<SizeType32*>(alignedPointers[5]);

    invokeTopPInitialize(idVals, offsetBuf, beginOffsetBuf, params.batchSize, params.vocabSizePadded, stream);
    sync_check_cuda_error(stream);

    SizeType32 constexpr BLOCK_SIZE = 256;
    // Performs Top K=1 search.
    // If the most probable token exceeds P, we skip sorting by setting beginOffsetBuf[bi] = offsetBuf[bi]
    topPBeamTopKKernel<T, BLOCK_SIZE><<<params.batchSize, BLOCK_SIZE, 0, stream>>>(params.probs, sortedIdVals,
        sortedProbs, params.finishedInput, params.vocabSizePadded, offsetBuf, beginOffsetBuf, params.topPs,
        params.skipDecode, params.batchSlots);
    sync_check_cuda_error(stream);

    // Sort tokens by probability in descending order
    auto cubWorkspaceSize = workspaceSizes[0];
    check_cuda_error(cub::DeviceSegmentedRadixSort::SortPairsDescending(cubTempStorage, cubWorkspaceSize, params.probs,
        sortedProbs, idVals, sortedIdVals, params.vocabSizePadded * params.batchSize, params.batchSize, beginOffsetBuf,
        offsetBuf + 1,
        0,                                      // begin_bit
        static_cast<SizeType32>(sizeof(T) * 8), // end_bit = sizeof(KeyT) * 8
        stream));                               // cudaStream_t

    SizeType32 constexpr SAMPLING_BLOCK_SIZE = 256;
    dim3 grid(params.batchSize);
    // Sample with Top P given sorted tokens
    topPSsampling<T, SAMPLING_BLOCK_SIZE><<<grid, SAMPLING_BLOCK_SIZE, 0, stream>>>(sortedProbs, sortedIdVals,
        params.outputIds, params.outputIdsPtrs, params.sequenceLength, params.finishedInput, params.finishedOutput,
        params.cumLogProbs, params.outputLogProbs, beginOffsetBuf, offsetBuf + 1, params.vocabSizePadded,
        params.curandState, params.randomVals, params.topPs, params.endIds, params.maxBatchSize, params.skipDecode,
        params.batchSlots, params.returnAllSelectedTokens, params.returnAllSelectedTokensPerSlot, params.maxSeqLen,
        params.outputIdCurrentStep, params.skipOutputIdCurrentStep);
    sync_check_cuda_error(stream);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template void invokeBatchTopPSampling(TopPSamplingKernelParams<float> const& params, cudaStream_t stream);

template void invokeBatchTopPSampling(TopPSamplingKernelParams<half> const& params, cudaStream_t stream);

__global__ void computeToppDecay(float* runtimeTopP, float const* runtimeInitialTopP, TokenIdType const** outputIds,
    float const* topPDecay, float const* topPMin, TokenIdType const* topPResetIds, SizeType32 const* sequenceLengths,
    SizeType32 const* batchSlots, SizeType32 localBatchSize)
{
    auto const idx = static_cast<SizeType32>(blockDim.x * blockIdx.x + threadIdx.x);
    if (idx >= localBatchSize)
    {
        return;
    }
    auto const batchSlot = batchSlots[idx];
    auto const currentStep{sequenceLengths[batchSlot]};
    if (outputIds[batchSlot][currentStep] == topPResetIds[batchSlot])
    {
        runtimeTopP[batchSlot] = runtimeInitialTopP[batchSlot];
    }
    else
    {
        runtimeTopP[batchSlot] = max(runtimeTopP[batchSlot] * topPDecay[batchSlot], topPMin[batchSlot]);
    }
}

void invokeComputeToppDecay(float* runtimeTopP, float const* runtimeInitialTopP, TokenIdType const** outputIds,
    float const* topPDecay, float const* topPMin, TokenIdType const* topPResetIds, SizeType32 const* sequenceLengths,
    SizeType32 const* batchSlots, SizeType32 localBatchSize, cudaStream_t stream)
{
    dim3 block(std::min(localBatchSize, 512));
    dim3 grid((localBatchSize + block.x - 1) / block.x);
    computeToppDecay<<<grid, block, 0, stream>>>(runtimeTopP, runtimeInitialTopP, outputIds, topPDecay, topPMin,
        topPResetIds, sequenceLengths, batchSlots, localBatchSize);
}

__global__ void setTopPRuntimeArgs(SizeType32 batchSize, SizeType32 const* batchSlots,
    ScatterDecodingParamEntry<SizeType32> topK, ScatterDecodingParamEntry<float> topP, bool* skipDecode,
    float* initialTopPBuf)
{
    auto index = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
    for (SizeType32 bi = index; bi < batchSize; bi += static_cast<SizeType32>(gridDim.x * blockDim.x))
    {
        setupTopKTopPRuntimeArgOne(bi, topK, topP, batchSlots, nullptr, skipDecode, initialTopPBuf);
    }
}

void invokeSetTopPRuntimeArgs(SizeType32 batchSize, ScatterDecodingParamEntry<SizeType32> topK,
    ScatterDecodingParamEntry<float> topP, bool* skipDecodePtr, float* initialTopPPtr, SizeType32 const* batchSlotsPtr,
    bool onDevice, cudaStream_t stream)
{
    if (onDevice)
    {
        dim3 block(std::min(static_cast<uint32_t>(batchSize), 256u));
        dim3 grid(divUp(static_cast<uint32_t>(batchSize), block.x));
        setTopPRuntimeArgs<<<grid, block, 0, stream>>>(
            batchSize, batchSlotsPtr, topK, topP, skipDecodePtr, initialTopPPtr);
    }
    else
    {
        for (int bi = 0; bi < batchSize; ++bi)
        {
            setupTopKTopPRuntimeArgOne(bi, topK, topP, batchSlotsPtr, nullptr, skipDecodePtr, nullptr);
        }
    }
}

} // namespace tensorrt_llm::kernels
