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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/samplingTopPKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{
__global__ void topPInitialize(
    int* topPIdValBuf, int* topPOffsetBuf, int* beginTopPOffsetBuf, int const batchSize, int const vocabSize)
{
    auto const tid = static_cast<int32_t>(threadIdx.x);
    auto const bid = static_cast<int32_t>(blockIdx.x);

    if (bid == 0)
    {
        for (auto i = tid; i < batchSize + 1; i += static_cast<int32_t>(blockDim.x))
        {
            // Inclusive sum of offsets to vocab rows
            topPOffsetBuf[i] = i * vocabSize;
            beginTopPOffsetBuf[i] = topPOffsetBuf[i];
        }
    }

    auto index = tid + bid * static_cast<int32_t>(blockDim.x);

    while (index < batchSize * vocabSize)
    {
        // Set value at {bi, vi} position to vi
        topPIdValBuf[index] = index % vocabSize;
        index += static_cast<int32_t>(blockDim.x * gridDim.x);
    }
}

void invokeTopPInitialize(int* topPIdValBuf, int* topPOffsetBuf, int* beginTopPOffsetBuf, size_t const batchSize,
    int const vocabSize, cudaStream_t stream)
{
    // vocabSize: the column number of logits_buffer for top_p sampling
    // TODO(nkorobov): launch based on available resources
    topPInitialize<<<32, 512, 0, stream>>>(topPIdValBuf, topPOffsetBuf, beginTopPOffsetBuf, batchSize, vocabSize);
}

template <typename T, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void topPBeamTopKKernel(T const* logProbs, // prob.
    int* topKTmpIdBuf, T* topKTmpValBuf, FinishedState const* finishedInput, int const vocabSize, int* offsetBuf,
    int* beginOffsetBuf, float const topP, float const* topPs, bool const* skipDecode, int const* batchSlots)
{
    /**
     * Kernel performs top 1 search and saves the token with largest probability if it exceeds probability threshold
     */
    int constexpr MAX_K = 1;
    auto const threadId = static_cast<int32_t>(threadIdx.x);
    auto const batchId = static_cast<int32_t>(blockIdx.x);
    auto const batchSlot = batchSlots != nullptr ? batchSlots[batchId] : batchId;

    // Skip decoding kernel if configured
    if ((skipDecode != nullptr && skipDecode[batchSlot])
        || (finishedInput != nullptr && finishedInput[batchSlot].isSkipDecoding()))
    {
        // Required to skip radix sort
        beginOffsetBuf[batchId] += vocabSize;
        return;
    }

    float pThreshold = (topPs != nullptr) ? topPs[batchSlot] : topP;

    typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    TopK<T, MAX_K> partial;

    bool const IS_FP16 = std::is_same<T, half>::value;
    T const MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

#pragma unroll
    for (int i = 0; i < MAX_K; ++i)
    {
        partial.p[i] = -1;
        partial.u[i] = -MAX_T_VAL;
    }

#pragma unroll
    for (int elemId = threadId; elemId < vocabSize; elemId += THREADBLOCK_SIZE)
    {
        int index = elemId + batchId * vocabSize;
        partial.insert(logProbs[index], elemId);
    }

    TopK<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

    if (threadId == 0)
    {
        beginOffsetBuf[batchId] = offsetBuf[batchId];
        T sumProb = (T) (0.0f);

#pragma unroll
        for (int i = 0; i < MAX_K; i++)
        {
            sumProb += total.u[i];
        }

        if ((float) sumProb >= pThreshold)
        {
            beginOffsetBuf[batchId] += vocabSize;
            int index = batchId * vocabSize;

#pragma unroll
            for (int i = 0; i < MAX_K; ++i)
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
__device__ void epilogue(int batchId, int currentStep, int offset, int** ids, int* sortedIdVals, T* sortedLogProbs,
    float* cumLogProbs, float* outputLogProbs, int const* endIds, int* sequenceLengths, FinishedState* finishedOutput,
    int maxBatchSize)
{
    ids[batchId][currentStep] = sortedIdVals[offset];

    if (cumLogProbs != nullptr || outputLogProbs != nullptr)
    {
        float lprob = logf(sortedLogProbs[offset]);
        if (cumLogProbs != nullptr)
        {
            cumLogProbs[batchId] += lprob;
        }
        if (outputLogProbs != nullptr)
        {
            outputLogProbs[sequenceLengths[batchId] * maxBatchSize + batchId] = lprob;
        }
    }
    if (sequenceLengths != nullptr && finishedOutput != nullptr)
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
__global__ void topPSsampling(T* sortedLogProbs, int* sortedIdVals, int** ids, int* sequenceLength,
    FinishedState const* finishedInput, FinishedState* finishedOutput, float* cumLogProbs, float* outputLogProbs,
    int const* beginOffsetBuf, int const* offsetBuf, int const vocabSize, curandState_t* curandstate, float const topP,
    float const* topPs, int const* endIds, int maxBatchSize, bool const* skipDecode, int const* batchSlots)
{
    /**
     * Each block processes one request row sorted in descending order by probabilities.
     * All threads within block compute running sum of probabilities until one of the threads exceeds the randomly
     * chosen probability threshold. Thread that crossed probaility threshold writes the corresponding token to the
     * output.
     */

    __shared__ float randNumS;

    auto const tid = static_cast<int32_t>(threadIdx.x);
    auto const batchId = static_cast<int32_t>(blockIdx.x);
    auto const batchSlot = batchSlots != nullptr ? batchSlots[batchId] : batchId;
    // Skip kernel if this sampling method is not chosen
    const FinishedState finishState = finishedInput != nullptr ? finishedInput[batchSlot] : FinishedState::empty();
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

    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = blockSize / WARP_SIZE;
    int const laneId = threadIdx.x % WARP_SIZE;
    int const warpId = threadIdx.x / WARP_SIZE;
    float const probThreshold = (topPs != nullptr) ? topPs[batchSlot] : topP;
    int const currentStep = sequenceLength[batchSlot];

    // With P in (0.0; 1.0] we draw a random number P' in range (0.0; P]
    // We will sum all probs moving from the largest probability to the smallest and
    // will choose the token which probability makes cumulative probability sum to exceed P'
    if (threadIdx.x == 0)
    {
        randNumS = curand_uniform(curandstate + blockIdx.x) * probThreshold;
    }

    // if beginOffsetBuf and offsetBuf of sorting have same value,
    // this means that we have find best one in topPBeamTopKKernel
    // So, we can skip this sampling.
    if (beginOffsetBuf[batchId] == offsetBuf[batchId])
    {
        if (tid == 0)
        {
            int offset = batchId * vocabSize;
            epilogue(batchSlot, currentStep, offset, ids, sortedIdVals, sortedLogProbs, cumLogProbs, outputLogProbs,
                endIds, sequenceLength, finishedOutput, maxBatchSize);
        }
        return;
    }

    typedef cub::BlockScan<float, blockSize> BlockScan;
    __shared__ typename BlockScan::TempStorage tempStorage;
    __shared__ uint32_t selectedShared[NUM_WARPS];
    // Initialize running total
    BlockPrefixCallbackOp prefixOp(0);

    if (laneId == 0)
    {
        selectedShared[warpId] = 0;
    }

    __syncthreads();

    int offset = batchId * vocabSize;
    ids[batchSlot][currentStep] = sortedIdVals[offset];
    int end = ((vocabSize + blockSize - 1) / blockSize) * blockSize;
    int selectedTokenId = 0;
    // Cumulative sum
    float threadOffset = 0;
    int count = 0;
    for (int vi = tid; vi < end; vi += blockSize)
    {
        float threadProb = (vi < vocabSize) ? (float) sortedLogProbs[offset + vi] : 0.f;
        BlockScan(tempStorage).InclusiveSum(threadProb, threadOffset, prefixOp);
        count = __syncthreads_count(randNumS <= threadOffset);
        selectedTokenId = vi;
        if (count != 0)
        {
            break;
        }
    }

    // select first thread exceeded the prob threshold or the last thread in case of P=1.0f
    if (threadIdx.x == min(blockDim.x - count, blockDim.x - 1))
    {
        epilogue(batchSlot, currentStep, offset + selectedTokenId, ids, sortedIdVals, sortedLogProbs, cumLogProbs,
            outputLogProbs, endIds, sequenceLength, finishedOutput, maxBatchSize);
    }
}

template <typename T>
std::vector<size_t> getTopPWorkspaceSizes(int32_t batchSize, int32_t vocabSize)
{
    auto const sortedLogProbBufSize = sizeof(T) * batchSize * vocabSize;      // type T
    auto const sortedIdValsBufSize = sizeof(int32_t) * batchSize * vocabSize; // type int

    size_t cubTempStorageSize;
    tensorrt_llm::common::check_cuda_error(cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr,
        cubTempStorageSize, static_cast<T*>(nullptr), static_cast<T*>(nullptr), static_cast<int32_t*>(nullptr),
        static_cast<int32_t*>(nullptr), static_cast<int32_t>(vocabSize * batchSize), batchSize,
        static_cast<int32_t*>(nullptr), static_cast<int32_t*>(nullptr),
        0,             // begin_bit
        sizeof(T) * 8, // end_bit = sizeof(KeyT) * 8
        0));           // cudaStream_t

    return {cubTempStorageSize, sortedLogProbBufSize, sortedIdValsBufSize};
}

template std::vector<size_t> getTopPWorkspaceSizes<float>(int32_t batchSize, int32_t vocabSize);
template std::vector<size_t> getTopPWorkspaceSizes<half>(int32_t batchSize, int32_t vocabSize);

template <typename T>
size_t getTopPWorkspaceSize(int32_t batchSize, int32_t vocabSizePadded)
{
    auto const workspaceSizes = getTopPWorkspaceSizes<T>(batchSize, vocabSizePadded);
    return tensorrt_llm::common::calcAlignedSize(workspaceSizes, 256);
}

template size_t getTopPWorkspaceSize<float>(int32_t batchSize, int32_t vocabSizePadded);
template size_t getTopPWorkspaceSize<half>(int32_t batchSize, int32_t vocabSizePadded);

template <typename T>
void invokeBatchTopPSampling(void* workspace, int** outputIds, int* sequenceLength, FinishedState const* finishedInput,
    FinishedState* finishedOutput, float* cumLogProbs, float* outputLogProbs, T const* logProbs, int32_t const* idVals,
    int* offsetBuf, int* beginOffsetBuf, curandState_t* curandstate, int const batchSize, int maxBatchSize,
    size_t const vocabSize, int const* endIds, float const maxTopP, float const* topPs, cudaStream_t stream,
    bool const* skipDecode, int const* batchSlots)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    auto const workspaceSizes = getTopPWorkspaceSizes<T>(batchSize, vocabSize);

    std::vector<void*> alignedPointers;
    calcAlignedPointers(alignedPointers, workspace, workspaceSizes);

    auto cubTempStorage = static_cast<void*>(alignedPointers[0]);
    auto sortedLogProbs = static_cast<T*>(alignedPointers[1]);
    auto sortedIdVals = static_cast<int32_t*>(alignedPointers[2]);

    int constexpr BLOCK_SIZE = 256;
    // Performs Top K=1 search.
    // If the most probable token exceeds P, we skip sorting by setting beginOffsetBuf[bi] = offsetBuf[bi]
    topPBeamTopKKernel<T, BLOCK_SIZE><<<batchSize, BLOCK_SIZE, 0, stream>>>(logProbs, sortedIdVals, sortedLogProbs,
        finishedInput, vocabSize, offsetBuf, beginOffsetBuf, maxTopP, topPs, skipDecode, batchSlots);
    sync_check_cuda_error();

    // Sort tokens by probability in descending order
    auto cubWorkspaceSize = workspaceSizes[0];
    check_cuda_error(
        cub::DeviceSegmentedRadixSort::SortPairsDescending(cubTempStorage, cubWorkspaceSize, logProbs, sortedLogProbs,
            idVals, sortedIdVals, static_cast<int32_t>(vocabSize * batchSize), batchSize, beginOffsetBuf, offsetBuf + 1,
            0,                                   // begin_bit
            static_cast<int32_t>(sizeof(T) * 8), // end_bit = sizeof(KeyT) * 8
            stream));                            // cudaStream_t

    int constexpr SAMPLING_BLOCK_SIZE = 256;
    dim3 grid(batchSize);
    // Sample with Top P given sorted tokens
    topPSsampling<T, SAMPLING_BLOCK_SIZE><<<grid, SAMPLING_BLOCK_SIZE, 0, stream>>>(sortedLogProbs, sortedIdVals,
        outputIds, sequenceLength, finishedInput, finishedOutput, cumLogProbs, outputLogProbs, beginOffsetBuf,
        offsetBuf + 1, vocabSize, curandstate, maxTopP, topPs, endIds, maxBatchSize, skipDecode, batchSlots);
    sync_check_cuda_error();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template void invokeBatchTopPSampling(void* workspace, int** outputIds, int* sequenceLength,
    FinishedState const* finishedInput, FinishedState* finishedOutput, float* cumLogProbs, float* outputLogProbs,
    float const* logProbs, int32_t const* idVals, int* offsetBuf, int* beginOffsetBuf, curandState_t* curandstate,
    int const batchSize, int maxBatchSize, size_t const vocabSizePadded, int const* endIds, float const maxTopP,
    float const* topPs, cudaStream_t stream, bool const* skipDecode, int const* batchSlots);

template void invokeBatchTopPSampling(void* workspace, int** outputIds, int* sequenceLength,
    FinishedState const* finishedInput, FinishedState* finishedOutput, float* cumLogProbs, float* outputLogProbs,
    half const* logProbs, int32_t const* idVals, int* offsetBuf, int* beginOffsetBuf, curandState_t* curandstate,
    int const batchSize, int maxBatchSize, size_t const vocabSizePadded, int const* endIds, float const maxTopP,
    float const* topPs, cudaStream_t stream, bool const* skipDecode, int const* batchSlots);

template <typename T>
void invokeTopPSampling(void* workspace, int** outputIds, int* sequenceLength, FinishedState const* finishedInput,
    FinishedState* finishedOutput, float* cumLogProbs, float* outputLogProbs, T const* logProbs, int32_t const* idVals,
    int* offsetBuf, int* beginOffsetBuf, curandState_t* curandstate, int const batchSize, int maxBatchSize,
    size_t const vocabSizePadded, int const* endIds, float const topP, cudaStream_t stream, bool const* skipDecode,
    int const* batchSlots)
{
    invokeBatchTopPSampling(workspace, outputIds, sequenceLength, finishedInput, finishedOutput, cumLogProbs,
        outputLogProbs, logProbs, idVals, offsetBuf, beginOffsetBuf, curandstate, batchSize, maxBatchSize,
        vocabSizePadded, endIds, topP, nullptr, stream, skipDecode, batchSlots);
}

template void invokeTopPSampling(void* workspace, int** outputIds, int* sequenceLength,
    FinishedState const* finishedInput, FinishedState* finishedOutput, float* cumLogProbs, float* outputLogProbs,
    float const* logProbs, int32_t const* idVals, int* offsetBuf, int* beginOffsetBuf, curandState_t* curandstate,
    int const batchSize, int maxBatchSize, size_t const vocabSizePadded, int const* endIds, float const topP,
    cudaStream_t stream, bool const* skipDecode, int const* batchSlots);

template void invokeTopPSampling(void* workspace, int** outputIds, int* sequenceLength,
    FinishedState const* finishedInput, FinishedState* finishedOutput, float* cumLogProbs, float* outputLogProbs,
    half const* logProbs, int32_t const* idVals, int* offsetBuf, int* beginOffsetBuf, curandState_t* curandstate,
    int const batchSize, int maxBatchSize, size_t const vocabSizePadded, int const* endIds, float const topP,
    cudaStream_t stream, bool const* skipDecode, int const* batchSlots);

__global__ void computeToppDecay(float* runtimeTopP, float const* runtimeInitialTopP, int const** outputIds,
    float const* topPDecay, float const* topPMin, int32_t const* topPResetIds, int const* sequenceLengths,
    int const* batchSlots)
{
    int const idx = blockDim.x * blockIdx.x + threadIdx.x;
    auto const batchSlot = batchSlots != nullptr ? batchSlots[idx] : idx;
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

void invokeComputeToppDecay(float* runtimeTopP, float const* runtimeInitialTopP, int const** outputIds,
    float const* topPDecay, float const* topPMin, int32_t const* topPResetIds, int const* sequenceLengths,
    int const* batchSlots, int const localBatchSize, cudaStream_t stream)
{
    dim3 block(min(localBatchSize, 512));
    dim3 grid((localBatchSize + block.x - 1) / block.x);
    computeToppDecay<<<grid, block, 0, stream>>>(
        runtimeTopP, runtimeInitialTopP, outputIds, topPDecay, topPMin, topPResetIds, sequenceLengths, batchSlots);
}

} // namespace kernels
} // namespace tensorrt_llm
