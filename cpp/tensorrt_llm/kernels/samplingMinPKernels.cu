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
#else
#include "3rdparty/cub/cub.cuh"
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
template <typename T, int THREADBLOCK_SIZE>
__global__ void fusedMinPSsampling(T const* probs, T* adjustedProbs, TokenIdType* outputIds,
    TokenIdType** outputIdsPtrs, SizeType32* sequenceLengths, FinishedState const* finishedInput,
    FinishedState* finishedOutput, float* cumLogProbs, float* outputLogProbs, SizeType32 vocabSize,
    curandState_t* curandState, float const* randomVals, float const* minPs, float const* temperatures,
    TokenIdType const* endIds, SizeType32 maxBatchSize, SizeType32 const* batchSlots, bool returnAllSelectedTokens,
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

    // Each thread computes local maximum across its assigned probabilities
    float threadMax = -FLT_MAX;
    const int probsBeginIdx = batchId * vocabSize;
    const int probsEndIdx = (batchId + 1) * vocabSize;

    #pragma unroll
    for (int idx = probsBeginIdx + tid; idx < probsEndIdx; idx += THREADBLOCK_SIZE)
    {
        float prob = static_cast<float>(probs[idx]);
        threadMax = max(threadMax, prob);
    }

    // Find global maximum probability across all threads in block
    threadMax = blockReduceMax<float>(threadMax);
    __shared__ float sCutoffP;

    if (tid == 0)
    {
        sCutoffP = threadMax * (minPs != nullptr ? minPs[batchSlot] : 0.0f);

#if DEBUG_MINP
        printf("Batch slot %d maxP %f cutoffP %f\n", batchSlot, threadMax, sCutoffP);
#endif
    }
    __syncthreads();

    // Adjust the probabilities and cache them
    float threadAdjustedProbsSum = 0.0f;
    float invTemp = 1.0f / (temperatures != nullptr ? temperatures[batchSlot] : 1.0f);

    #pragma unroll
    for (int idx = probsBeginIdx + tid; idx < probsEndIdx; idx += THREADBLOCK_SIZE)
    {
        float prob = static_cast<float>(probs[idx]);
        prob = (prob < sCutoffP) ? 0.0f : powf(prob, invTemp);
        adjustedProbs[idx] = static_cast<T>(prob);
        threadAdjustedProbsSum += prob;
    }

    // Find global sum of adjusted probabilities and determine quantization scale factor
    threadAdjustedProbsSum = blockReduceSum<float>(threadAdjustedProbsSum);
    __shared__ float sAdjustedProbsSum;
    __shared__ float sQuantizeScaleFactor;

    if (tid == 0)
    {
        sAdjustedProbsSum = threadAdjustedProbsSum;

        // Do division with doubles and round down to avoid special cases like
        // 4294967295 / 32768 giving us 131072 rather than the desired 131071
        sQuantizeScaleFactor = __double2float_rd((double)(UINT32_MAX - vocabSize) / (double)threadAdjustedProbsSum);

#if DEBUG_MINP
        printf("Batch slot %d adjustedProbsSum %f quantizeScaleFactor %f\n", batchSlot, threadAdjustedProbsSum, sQuantizeScaleFactor);
#endif
    }
    __syncthreads();

    // We will now quantize the probabilities to integers to avoid numerical errors
    // when trying to find the selected point in the prefix sum of the probabilities.
    // We map the adjusted distribution between [0, UINT32_MAX] to avoid overflow.

    // Compute the sum of the quantized probabilities for each thread
    uint32_t threadQuantProbsSum = 0;

    #pragma unroll
    for (int idx = probsBeginIdx + tid; idx < probsEndIdx; idx += THREADBLOCK_SIZE)
    {
        float prob = static_cast<float>(adjustedProbs[idx]);
        threadQuantProbsSum += __float2uint_rd(prob * sQuantizeScaleFactor);
    }

    // Compute a global prefix sum of the quantized probabilities
    uint32_t threadQuantProbsPrefix;
    uint32_t totalQuantProbsSum;

    using BlockScan = cub::BlockScan<uint32_t, THREADBLOCK_SIZE>;
    __shared__ typename BlockScan::TempStorage tempStorage;

    BlockScan(tempStorage).ExclusiveSum(threadQuantProbsSum, threadQuantProbsPrefix, totalQuantProbsSum);

    // Select a random point in the distribution
    __shared__ uint32_t sRandomPoint;

    if (tid == 0)
    {
        // Rescale uniform random val to be within the sum of quantized probabilities
        float randomVal = randomVals != nullptr ? randomVals[batchSlot] : curand_uniform(&curandState[batchSlot]);
        sRandomPoint = min(__float2uint_rd(randomVal * totalQuantProbsSum), totalQuantProbsSum - 1);

#if DEBUG_MINP
        printf("Batch slot %d totalQuantProbsSum %u randomPoint %u\n", batchSlot, totalQuantProbsSum, sRandomPoint);
#endif
    }
    __syncthreads();

    // All but one warps will terminate on this condition
    if (sRandomPoint < threadQuantProbsPrefix || sRandomPoint >= threadQuantProbsPrefix + threadQuantProbsSum)
    {
        return;
    }

    // Find the selected token id and write it to the output buffer
    threadQuantProbsSum = threadQuantProbsPrefix;

    for (int idx = probsBeginIdx + tid; idx < probsEndIdx; idx += THREADBLOCK_SIZE)
    {
        float prob = static_cast<float>(adjustedProbs[idx]);
        uint32_t quantProb = __float2uint_rd(prob * sQuantizeScaleFactor);

        if (sRandomPoint < threadQuantProbsSum + quantProb)
        {
            auto const selectedTokenIdx = idx - probsBeginIdx;
            auto const curSeqLen = sequenceLengths == nullptr ? 0 : sequenceLengths[batchSlot];
            auto* outPtr = outputIdsPtrs == nullptr ? outputIds + batchSlot * maxSeqLen : outputIdsPtrs[batchSlot];
            outPtr[curSeqLen] = selectedTokenIdx;

#if DEBUG_MINP
            printf("Batch slot %d selected token %d original prob %f adjusted prob %f normalized %f\n",
                batchSlot, selectedTokenIdx, static_cast<float>(probs[idx]), prob, prob / sAdjustedProbsSum);

            printf("Batch slot %d thread index %d prefix %d sum %u quant prob %u\n",
                batchSlot, tid, threadQuantProbsPrefix, threadQuantProbsSum, quantProb);
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

        threadQuantProbsSum += quantProb;
    }
}

template <typename T>
std::vector<size_t> getMinPWorkspaceSizes(SizeType32 batchSize, SizeType32 vocabSize)
{
    auto const adjustedProbBufSize = sizeof(T) * batchSize * vocabSize;

    return {adjustedProbBufSize};
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
    auto const workspaceSizes = getMinPWorkspaceSizes<T>(params.batchSize, params.vocabSizePadded);

    std::vector<void*> alignedPointers;
    calcAlignedPointers(alignedPointers, params.workspace, workspaceSizes);

    auto adjustedProbs = static_cast<T*>(alignedPointers[0]);

    // Sample with Min P filter and late temperature in single pass
    SizeType32 constexpr SAMPLING_BLOCK_SIZE = 1024;
    dim3 grid(params.batchSize);
    fusedMinPSsampling<T, SAMPLING_BLOCK_SIZE><<<grid, SAMPLING_BLOCK_SIZE, 0, stream>>>(params.probs, adjustedProbs,
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
