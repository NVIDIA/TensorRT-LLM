/*
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
#include <cub/block/block_shuffle.cuh> // Why is it not in the monolithic cub.cuh?
#else
#include "3rdparty/cub/cub.cuh"
#include "3rdparty/cub/block/block_shuffle.cuh" // Why is it not in the monolithic cub.cuh?
#endif

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/samplingMinPKernels.h"

using namespace tensorrt_llm::common;
using namespace tensorrt_llm::runtime;

#define DEBUG_MINP 0

namespace tensorrt_llm::kernels
{
// Shared state for MinP sampling kernels
template <typename T, int THREADBLOCK_SIZE>
struct BlockScanShuffleStorage
{
    union {
        typename cub::BlockScan<T, THREADBLOCK_SIZE>::TempStorage scan;
        typename cub::BlockShuffle<T, THREADBLOCK_SIZE>::TempStorage shuffle;
    };
};

template <typename T, int THREADBLOCK_SIZE>
__global__ void fusedMinPSsampling(T const* probs, TokenIdType* outputIds, TokenIdType** outputIdsPtrs,
    SizeType32* sequenceLengths, FinishedState const* finishedInput, FinishedState* finishedOutput,
    float* cumLogProbs, float* outputLogProbs, SizeType32 vocabSize, curandState_t* curandState,
    float const* randomVals, float const* minPs, float const* temperatures, TokenIdType const* endIds,
    SizeType32 maxBatchSize, SizeType32 const* batchSlots, bool returnAllSelectedTokens,
    SizeType32 maxSeqLen, TokenIdType* outputIdCurrentStep, bool const* skipOutputIdCurrentStep)
{
    auto const tid = static_cast<SizeType32>(threadIdx.x);
    auto const batchId = static_cast<SizeType32>(blockIdx.x);
    auto const batchSlot = batchSlots[batchId];

#if DEBUG_MINP
    if (tid == 0)
    {
        printf("Begin batch slot %d sequence length %d\n", batchSlot, sequenceLengths[batchSlot]);
    }
#endif

    // Skip kernel if this sampling method is not chosen
    FinishedState const finishState = finishedInput != nullptr ? finishedInput[batchSlot] : FinishedState::empty();
    if (finishState.isSkipDecoding())
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

#if DEBUG_MINP
            printf("Batch slot %d already finished\n", batchSlot);
#endif
        }
        return;
    }

    // Common stride for all arrays
    const int probsBeginIdx = batchId * vocabSize;
    const int probsEndIdx = (batchId + 1) * vocabSize;

    // Each thread computes local maximum across its assigned probabilities
    float threadMaxProb = -FLT_MAX;

    #pragma unroll
    for (int idx = probsBeginIdx + tid; idx < probsEndIdx; idx += THREADBLOCK_SIZE)
    {
        auto const prob = static_cast<float>(probs[idx]);
        threadMaxProb = max(threadMaxProb, prob);
    }

    // Find global maximum probability across all threads in block
    threadMaxProb = blockReduceMax<float>(threadMaxProb);
    __shared__ float sCutoffP;
    __shared__ float sInvTemp;

    if (tid == 0)
    {
        // Probs below this value will be ignored
        sCutoffP = threadMaxProb * (minPs != nullptr ? minPs[batchSlot] : 0.0f);

        // Inverse temperature for scaling probabilities
        sInvTemp = 1.0f / (temperatures != nullptr ? temperatures[batchSlot] : 1.0f);

#if DEBUG_MINP
        printf("Batch slot %d maxP %f cutoffP %f\n", batchSlot, threadMaxProb, sCutoffP);
#endif
    }
    __syncthreads();

#if DEBUG_MINP
    // Print how many probabilities are above the cutoff
    int threadNumProbsAboveCutoff = 0;

    #pragma unroll
    for (int idx = probsBeginIdx + tid; idx < probsEndIdx; idx += THREADBLOCK_SIZE)
    {
        if (static_cast<float>(probs[idx]) >= sCutoffP)
        {
            threadNumProbsAboveCutoff++;
        }
    }

    threadNumProbsAboveCutoff = blockReduceSum<int>(threadNumProbsAboveCutoff);

    if (tid == 0)
    {
        printf("Batch slot %d numProbsAboveCutoff %d\n", batchSlot, threadNumProbsAboveCutoff);
    }
#endif

    // Adjust the probabilities and sum the ones passing the cutoff
    float threadScaledProbsSum = 0.f;

    #pragma unroll
    for (int idx = probsBeginIdx + tid; idx < probsEndIdx; idx += THREADBLOCK_SIZE)
    {
        auto const prob = static_cast<float>(probs[idx]);
        auto const scaledProb = (prob < sCutoffP) ? 0.0f : powf(prob, sInvTemp);
        threadScaledProbsSum += scaledProb;
    }

    // Find global sum and prefix sum of adjusted probabilities
    using BlockScan = cub::BlockScan<float, THREADBLOCK_SIZE>;
    using BlockShuffle = cub::BlockShuffle<float, THREADBLOCK_SIZE>;
    __shared__ BlockScanShuffleStorage<float, THREADBLOCK_SIZE> tempStorage;

    float threadScaledProbsIncl = 0.f;
    float threadScaledProbsExcl = 0.f;
    float scaledProbsSum = 0.f;

    BlockScan(tempStorage.scan).InclusiveSum(threadScaledProbsSum, threadScaledProbsIncl, scaledProbsSum);
    __syncthreads(); // We are aliasing the shared memory
    BlockShuffle(tempStorage.shuffle).Offset(threadScaledProbsIncl, threadScaledProbsExcl, -1);

    // Select a random point in the distribution
    __shared__ float sRandomPoint;

    if (tid == 0)
    {
        // Rescale uniform random val to be within the sum of included adjusted probabilities
        float randomVal = randomVals != nullptr ? randomVals[batchSlot] : curand_uniform(&curandState[batchSlot]);
        sRandomPoint = randomVal * scaledProbsSum;

#if DEBUG_MINP
        printf("Batch slot %d scaledProbsSum %f randomPoint %f\n", batchSlot, scaledProbsSum, sRandomPoint);
#endif
    }
    __syncthreads();

    // All but one warps will reliably terminate on this condition
    if (sRandomPoint < threadScaledProbsExcl || sRandomPoint >= threadScaledProbsIncl)
    {
        return;
    }

    // Convert global random point to local range of current thread
    float randomLocalOffset = sRandomPoint - threadScaledProbsExcl;
    float randomLocalScalar = randomLocalOffset / (threadScaledProbsIncl - threadScaledProbsExcl);
    float randomLocalPoint = randomLocalScalar * threadScaledProbsSum;

#if DEBUG_MINP
    printf("Batch slot %d threadScaledProbsExcl %f threadScaledProbsIncl %f threadScaledProbsSum %f\n",
        batchSlot, threadScaledProbsExcl, threadScaledProbsIncl, threadScaledProbsSum);

    printf("Batch slot %d randomLocalOffset %f randomLocalScalar %f randomLocalPoint %f\n",
        batchSlot, randomLocalOffset, randomLocalScalar, randomLocalPoint);
#endif

    // Find the selected token id and write it to the output buffer
    threadScaledProbsSum = 0.f;

    auto const curSeqLen = sequenceLengths == nullptr ? 0 : sequenceLengths[batchSlot];
    auto* outPtr = outputIdsPtrs == nullptr ? outputIds + batchSlot * maxSeqLen : outputIdsPtrs[batchSlot];

    for (int idx = probsBeginIdx + tid; idx < probsEndIdx; idx += THREADBLOCK_SIZE)
    {
        auto const prob = static_cast<float>(probs[idx]);
        auto const scaledProb = (prob < sCutoffP) ? 0.0f : powf(prob, sInvTemp);
        threadScaledProbsSum += scaledProb;

        // We are summing again in the same order, so this is guaranteed to be entered
        if (randomLocalPoint < threadScaledProbsSum)
        {
            auto const selectedTokenIdx = idx - probsBeginIdx;
            outPtr[curSeqLen] = selectedTokenIdx;

#if DEBUG_MINP
            printf("Batch slot %d selected token %d original prob %f scaled prob %f normalized %f\n",
                batchSlot, selectedTokenIdx, prob, scaledProb, scaledProb / scaledProbsSum);
#endif

            if (!returnAllSelectedTokens && sequenceLengths != nullptr && finishedOutput != nullptr && endIds != nullptr)
            {
                if (selectedTokenIdx == endIds[batchSlot])
                {
                    // This request has finished
                    finishedOutput[batchSlot].setFinishedEOS();
                }
                else
                {
                    // This request must generate more tokens
                    sequenceLengths[batchSlot] += 1;
                }
            }
            return;
        }
    }

    // This should never be reached
    outPtr[curSeqLen] = vocabSize - 1;
}

template <typename T>
std::vector<size_t> getMinPWorkspaceSizes(SizeType32 batchSize, SizeType32 vocabSize)
{
    return {};
}

template std::vector<size_t> getMinPWorkspaceSizes<float>(SizeType32 batchSize, SizeType32 vocabSize);
template std::vector<size_t> getMinPWorkspaceSizes<half>(SizeType32 batchSize, SizeType32 vocabSize);

template <typename T>
std::vector<size_t> getMinPInitWorkspaceSizes(SizeType32 batchSize)
{
    auto const tempMinPsBufSize = batchSize * sizeof(float);
    auto const tempTemperaturesBufSize = batchSize * sizeof(float);

    return {tempMinPsBufSize, tempTemperaturesBufSize};
}

template std::vector<size_t> getMinPInitWorkspaceSizes<float>(SizeType32 batchSize);
template std::vector<size_t> getMinPInitWorkspaceSizes<half>(SizeType32 batchSize);

template <typename T>
size_t getMinPWorkspaceSize(SizeType32 batchSize, SizeType32 vocabSizePadded)
{
    auto const workspaceSizes = getMinPWorkspaceSizes<T>(batchSize, vocabSizePadded);
    auto const initWorkspaceSizes = getMinPInitWorkspaceSizes<T>(batchSize);

    return std::max(tensorrt_llm::common::calcAlignedSize(workspaceSizes, 256),
        tensorrt_llm::common::calcAlignedSize(initWorkspaceSizes, 256));
}

template size_t getMinPWorkspaceSize<float>(SizeType32 batchSize, SizeType32 vocabSizePadded);
template size_t getMinPWorkspaceSize<half>(SizeType32 batchSize, SizeType32 vocabSizePadded);

template <typename T>
void invokeBatchMinPSampling(MinPSamplingKernelParams<T> const& params, cudaStream_t stream)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    params.checkParams();

    // Sample with Min P filter and late temperature in single pass
    SizeType32 constexpr SAMPLING_BLOCK_SIZE = 1024;
    dim3 grid(params.batchSize);
    fusedMinPSsampling<T, SAMPLING_BLOCK_SIZE><<<grid, SAMPLING_BLOCK_SIZE, 0, stream>>>(params.probs,
        params.outputIds, params.outputIdsPtrs, params.sequenceLength, params.finishedInput, params.finishedOutput,
        params.cumLogProbs, params.outputLogProbs, params.vocabSizePadded, params.curandState, params.randomVals,
        params.minPs, params.temperatures, params.endIds, params.maxBatchSize, params.batchSlots, params.returnAllSelectedTokens,
        params.maxSeqLen, params.outputIdCurrentStep, params.skipOutputIdCurrentStep);

    sync_check_cuda_error();

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

template void invokeBatchMinPSampling(MinPSamplingKernelParams<float> const& params, cudaStream_t stream);

template void invokeBatchMinPSampling(MinPSamplingKernelParams<half> const& params, cudaStream_t stream);

__device__ __host__ inline void setupMinPRuntimeArg(runtime::SizeType32 batchIndex,
    ScatterDecodingParamEntry<float> minP, ScatterDecodingParamEntry<float> temperature,
    runtime::SizeType32 const* batchSlots)
{
    auto const batchSlot = batchSlots[batchIndex];
    auto const p = minP.mVector == nullptr ? minP.mScalar : minP.mVector[batchIndex];
    auto const t = temperature.mVector == nullptr ? temperature.mScalar : temperature.mVector[batchIndex];

    if (minP.mTarget != nullptr)
    {
        minP.mTarget[batchSlot] = p;
    }

    if (temperature.mTarget != nullptr)
    {
        temperature.mTarget[batchSlot] = t;
    }
}

__global__ void setMinPRuntimeArgs(SizeType32 batchSize, ScatterDecodingParamEntry<float> minP,
    ScatterDecodingParamEntry<float> temperature, SizeType32 const* batchSlotsPtr)
{
    auto index = static_cast<SizeType32>(blockIdx.x * blockDim.x + threadIdx.x);
    for (SizeType32 bi = index; bi < batchSize; bi += static_cast<SizeType32>(gridDim.x * blockDim.x))
    {
        setupMinPRuntimeArg(bi, minP, temperature, batchSlotsPtr);
    }
}

void invokeSetMinPRuntimeArgs(SizeType32 batchSize, ScatterDecodingParamEntry<float> minP,
    ScatterDecodingParamEntry<float> temperature, SizeType32 const* batchSlotsPtr,
    cudaStream_t stream)
{
    dim3 block(std::min(static_cast<uint32_t>(batchSize), 256u));
    dim3 grid(divUp(static_cast<uint32_t>(batchSize), block.x));
    setMinPRuntimeArgs<<<grid, block, 0, stream>>>(
        batchSize, minP, temperature, batchSlotsPtr);
}

} // namespace tensorrt_llm::kernels
