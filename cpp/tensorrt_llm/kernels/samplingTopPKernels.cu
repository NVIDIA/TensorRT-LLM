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

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
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
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (bid == 0)
    {
        for (int i = tid; i < batchSize + 1; i += blockDim.x)
        {
            // Inclusive sum of offsets to vocab rows
            topPOffsetBuf[i] = i * vocabSize;
            beginTopPOffsetBuf[i] = topPOffsetBuf[i];
        }
    }

    int index = tid + bid * blockDim.x;

    while (index < batchSize * vocabSize)
    {
        // Set value at {bi, vi} position to vi
        topPIdValBuf[index] = index % vocabSize;
        index += blockDim.x * gridDim.x;
    }
}

void invokeTopPInitialize(int* topPIdValBuf, int* topPOffsetBuf, int* beginTopPOffsetBuf, size_t const batchSize,
    int const vocabSize, cudaStream_t stream)
{
    // vocabSize: the column number of logits_buffer for top_p sampling
    topPInitialize<<<32, 512, 0, stream>>>(topPIdValBuf, topPOffsetBuf, beginTopPOffsetBuf, batchSize, vocabSize);
}

template <typename T, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void topPBeamTopKKernel(T const* logProbs, // prob.
    int* topKTmpIdBuf, T* topKTmpValBuf, FinishedState const* finishedInput, int const vocabSize, int* offsetBuf,
    int* beginOffsetBuf, float const topP, float const* topPs, bool const* skipDecode)
{
    /**
     * Kernel performs top 1 search and saves the token with largest probability if it exceeds probability threshold
     */
    int constexpr MAX_K = 1;
    int threadId = threadIdx.x;
    int batchId = blockIdx.x;

    // Skip decoding kernel if configured
    if ((skipDecode != nullptr && skipDecode[batchId])
        || (finishedInput != nullptr && finishedInput[batchId].isSkipDecoding()))
    {
        // Required to skip radix sort
        beginOffsetBuf[batchId] += vocabSize;
        return;
    }

    float pThreshold = (topPs != nullptr) ? topPs[batchId] : topP;

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
    float* cumLogProbs, float* outputLogProbs, int const* endIds, int* sequenceLengths, FinishedState* finishedOutput)
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
            outputLogProbs[batchId] = lprob;
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
    float const* topPs, int const* endIds, int const batchSize, bool const* skipDecode)
{
    /**
     * Each block processes one request row sorted in descending order by probabilities.
     * All threads within block compute running sum of probabilities until one of the threads exceeds the randomly
     * chosen probability threshold. Thread that crossed probaility threshold writes the corresponding token to the
     * output.
     */

    __shared__ float randNumS;

    int const tid = threadIdx.x;
    int const batchId = blockIdx.x;
    // Skip kernel if this sampling method is not chosen
    FinishedState const finishState = finishedInput != nullptr ? finishedInput[batchId] : FinishedState::empty();
    if ((skipDecode != nullptr && skipDecode[batchId]) || (finishState.isSkipDecoding()))
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
                finishedOutput[batchId] = finishState;
            }
            ids[batchId][sequenceLength[batchId]] = endIds[batchId];
        }
        return;
    }

    int constexpr WARP_SIZE = 32;
    int constexpr NUM_WARPS = blockSize / WARP_SIZE;
    int const laneId = threadIdx.x % WARP_SIZE;
    int const warpId = threadIdx.x / WARP_SIZE;
    float const probThreshold = (topPs != nullptr) ? topPs[batchId] : topP;
    int const currentStep = sequenceLength[batchId];

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
            epilogue(batchId, currentStep, offset, ids, sortedIdVals, sortedLogProbs, cumLogProbs, outputLogProbs,
                endIds, sequenceLength, finishedOutput);
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
    ids[batchId][currentStep] = sortedIdVals[offset];
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
        epilogue(batchId, currentStep, offset + selectedTokenId, ids, sortedIdVals, sortedLogProbs, cumLogProbs,
            outputLogProbs, endIds, sequenceLength, finishedOutput);
    }
}

template <typename T>
void invokeBatchTopPSampling(void* workspace, size_t& workspaceSize, size_t& cubTempStorageSize, int** outputIds,
    int* sequenceLength, FinishedState const* finishedInput, FinishedState* finishedOutput, float* cumLogProbs,
    float* outputLogProbs, T const* logProbs, int const* idVals, int* offsetBuf, int* beginOffsetBuf,
    curandState_t* curandstate, int const batchSize, size_t const vocabSizePadded, int const* endIds,
    float const maxTopP, float const* topPs, cudaStream_t stream, bool const* skipDecode)
{
    int const vocabSize = vocabSizePadded;

    size_t sortedLogProbBufSize = batchSize * vocabSize * sizeof(T);  // type T
    size_t sortedIdValsBufSize = batchSize * vocabSize * sizeof(int); // type int
    sortedLogProbBufSize = divUp(sortedLogProbBufSize, 256) * 256;
    sortedIdValsBufSize = divUp(sortedIdValsBufSize, 256) * 256;

    void* cubTempStorage = workspace;
    T* sortedLogProbs = (T*) ((char*) cubTempStorage + cubTempStorageSize);
    int* sortedIdVals = (int*) ((char*) sortedLogProbs + sortedLogProbBufSize);

    if (workspace == nullptr)
    {
        check_cuda_error(cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, cubTempStorageSize, logProbs,
            (T*) nullptr, idVals, (int*) nullptr, vocabSize * batchSize, batchSize, beginOffsetBuf, offsetBuf + 1,
            0,             // begin_bit
            sizeof(T) * 8, // end_bit = sizeof(KeyT) * 8
            stream));      // cudaStream_t
        cubTempStorageSize = divUp(cubTempStorageSize, 256) * 256;
        workspaceSize = sortedLogProbBufSize + sortedIdValsBufSize + cubTempStorageSize;
        return;
    }

    int constexpr BLOCK_SIZE = 256;
    // Performs Top K=1 search.
    // If the most probable token exceeds P, we skip sorting by setting beginOffsetBuf[bi] = offsetBuf[bi]
    topPBeamTopKKernel<T, BLOCK_SIZE><<<batchSize, BLOCK_SIZE, 0, stream>>>(logProbs, sortedIdVals, sortedLogProbs,
        finishedInput, vocabSize, offsetBuf, beginOffsetBuf, maxTopP, topPs, skipDecode);

    // Sort tokens by probability in descending order
    check_cuda_error(cub::DeviceSegmentedRadixSort::SortPairsDescending(cubTempStorage, cubTempStorageSize, logProbs,
        sortedLogProbs, idVals, sortedIdVals, vocabSize * batchSize, batchSize, beginOffsetBuf, offsetBuf + 1,
        0,             // begin_bit
        sizeof(T) * 8, // end_bit = sizeof(KeyT) * 8
        stream));      // cudaStream_t

    int constexpr SAMPLING_BLOCK_SIZE = 256;
    dim3 grid(batchSize);
    // Sample with Top P given sorted tokens
    topPSsampling<T, SAMPLING_BLOCK_SIZE><<<grid, SAMPLING_BLOCK_SIZE, 0, stream>>>(sortedLogProbs, sortedIdVals,
        outputIds, sequenceLength, finishedInput, finishedOutput, cumLogProbs, outputLogProbs, beginOffsetBuf,
        offsetBuf + 1, vocabSize, curandstate, maxTopP, topPs, endIds, batchSize, skipDecode);
}

template void invokeBatchTopPSampling(void* workspace, size_t& workspaceSize, size_t& cubTempStorageSize,
    int** outputIds, int* sequenceLength, FinishedState const* finishedInput, FinishedState* finishedOutput,
    float* cumLogProbs, float* outputLogProbs, float const* logProbs, int const* idVals, int* offsetBuf,
    int* beginOffsetBuf, curandState_t* curandstate, int const batchSize, size_t const vocabSizePadded,
    int const* endIds, float const maxTopP, float const* topPs, cudaStream_t stream, bool const* skipDecode);

template void invokeBatchTopPSampling(void* workspace, size_t& workspaceSize, size_t& cubTempStorageSize,
    int** outputIds, int* sequenceLength, FinishedState const* finishedInput, FinishedState* finishedOutput,
    float* cumLogProbs, float* outputLogProbs, half const* logProbs, int const* idVals, int* offsetBuf,
    int* beginOffsetBuf, curandState_t* curandstate, int const batchSize, size_t const vocabSizePadded,
    int const* endIds, float const maxTopP, float const* topPs, cudaStream_t stream, bool const* skipDecode);

template <typename T>
void invokeTopPSampling(void* workspace, size_t& workspaceSize, size_t& cubTempStorageSize, int** outputIds,
    int* sequenceLength, FinishedState const* finishedInput, FinishedState* finishedOutput, float* cumLogProbs,
    float* outputLogProbs, T const* logProbs, int const* idVals, int* offsetBuf, int* beginOffsetBuf,
    curandState_t* curandstate, int const batchSize, size_t const vocabSizePadded, int const* endIds, float const topP,
    cudaStream_t stream, bool const* skipDecode)
{
    invokeBatchTopPSampling(workspace, workspaceSize, cubTempStorageSize, outputIds, sequenceLength, finishedInput,
        finishedOutput, cumLogProbs, outputLogProbs, logProbs, idVals, offsetBuf, beginOffsetBuf, curandstate,
        batchSize, vocabSizePadded, endIds, topP, nullptr, stream, skipDecode);
}

template void invokeTopPSampling(void* workspace, size_t& workspaceSize, size_t& cubTempStorageSize, int** outputIds,
    int* sequenceLength, FinishedState const* finishedInput, FinishedState* finishedOutput, float* cumLogProbs,
    float* outputLogProbs, float const* logProbs, int const* idVals, int* offsetBuf, int* beginOffsetBuf,
    curandState_t* curandstate, int const batchSize, size_t const vocabSizePadded, int const* endIds, float const topP,
    cudaStream_t stream, bool const* skipDecode);

template void invokeTopPSampling(void* workspace, size_t& workspaceSize, size_t& cubTempStorageSize, int** outputIds,
    int* sequenceLength, FinishedState const* finishedInput, FinishedState* finishedOutput, float* cumLogProbs,
    float* outputLogProbs, half const* logProbs, int const* idVals, int* offsetBuf, int* beginOffsetBuf,
    curandState_t* curandstate, int const batchSize, size_t const vocabSizePadded, int const* endIds, float const topP,
    cudaStream_t stream, bool const* skipDecode);

__global__ void computeToppDecay(float* runtimeTopP, float const* runtimeInitialTopP, int const** outputIds,
    float const* topPDecay, float const* topPMin, int32_t const* topPResetIds, int const* sequenceLengths)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    auto const currentStep{sequenceLengths[idx]};
    if (outputIds[idx][currentStep] == topPResetIds[idx])
    {
        runtimeTopP[idx] = runtimeInitialTopP[idx];
    }
    else
    {
        runtimeTopP[idx] = max(runtimeTopP[idx] * topPDecay[idx], topPMin[idx]);
    }
}

void invokeComputeToppDecay(float* runtimeTopP, float const* runtimeInitialTopP, int const** outputIds,
    float const* topPDecay, float const* topPMin, int32_t const* topPResetIds, int const* sequenceLengths,
    int const local_batchSize, cudaStream_t stream)
{
    dim3 block(min(local_batchSize, 512));
    dim3 grid((local_batchSize + block.x - 1) / block.x);
    computeToppDecay<<<grid, block, 0, stream>>>(
        runtimeTopP, runtimeInitialTopP, outputIds, topPDecay, topPMin, topPResetIds, sequenceLengths);
}

} // namespace kernels
} // namespace tensorrt_llm
