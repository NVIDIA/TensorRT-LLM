/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <assert.h>
#include <float.h>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/samplingPenaltyKernels.h"

namespace tensorrt_llm
{
namespace kernels
{

// TODO Add half2 implementation
template <typename T>
__global__ void applyTemperaturePenalty(T* logits, const T* bias, const float temperatureInverse, const int m,
    const int vocabSize, const int vocabSizePadded)
{
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? 65504.F : FLT_MAX;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < m * vocabSizePadded;
         index += blockDim.x * gridDim.x)
    {
        T biasVal = bias == nullptr ? (T) (0.0f) : bias[index % vocabSizePadded];
        if (index % vocabSizePadded < vocabSize)
        {
            logits[index] = (logits[index] + biasVal) * (T) temperatureInverse;
        }
        else
        {
            logits[index] = -MAX_T_VAL;
        }
    }
}

template <>
__global__ void applyTemperaturePenalty(half2* logits, const half2* bias, const float temperatureInverse,
    const int batchSize, const int vocabSize, const int vocabSizePaddeded)
{
    assert(vocabSize % 2 == 0);
    assert(vocabSizePaddeded % 2 == 0);
    const half2 maskVal = __float2half2_rn(-65504.0f);
    const half2 tempInv = __float2half2_rn(temperatureInverse);

    const int halfVocabSize = vocabSize / 2;
    const int halfVocabSizePaddeded = vocabSizePaddeded / 2;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < batchSize * halfVocabSizePaddeded;
         index += blockDim.x * gridDim.x)
    {
        int vocabIdx = index % halfVocabSizePaddeded;
        half2 logit = vocabIdx < halfVocabSize ? __ldg(&logits[index]) : maskVal;
        if (vocabIdx < halfVocabSize)
        {
            if (bias != nullptr)
            {
                logit = __hadd2(logit, bias[vocabIdx]);
            }
            logits[index] = __hmul2(logit, tempInv);
        }
    }
}

template <typename T>
void invokeApplyTemperaturePenalty(T* logits, const T* bias, const float temperature, const int batchSize,
    const int vocabSize, const int vocabSizePadded, cudaStream_t stream)
{
    dim3 block(min(vocabSizePadded, 1024));
    dim3 grid(min(batchSize * vocabSizePadded / block.x, 65536));
    const T temperatureInverse = (T) (1.f / (temperature + 1e-6f));
    if (std::is_same<T, half>::value && vocabSize % 2 == 0 && vocabSizePadded % 2 == 0)
    {
        applyTemperaturePenalty<<<grid, block, 0, stream>>>(reinterpret_cast<half2*>(logits),
            reinterpret_cast<const half2*>(bias), temperatureInverse, batchSize, vocabSize, vocabSizePadded);
    }
    else
    {
        applyTemperaturePenalty<T>
            <<<grid, block, 0, stream>>>(logits, bias, temperatureInverse, batchSize, vocabSize, vocabSizePadded);
    }
}

template void invokeApplyTemperaturePenalty(float* logits, const float* bias, const float temperature,
    const int batchSize, const int vocabSize, const int vocabSizePadded, cudaStream_t stream);

template void invokeApplyTemperaturePenalty(half* logits, const half* bias, const float temperature,
    const int batchSize, const int vocabSize, const int vocabSizePadded, cudaStream_t stream);

template <typename T>
__global__ void batchApplyTemperaturePenalty(T* logits, const T* bias, const float* temperatures, const int batchSize,
    const int vocabSize, const int vocabSizePadded)
{
    // TODO: Add macro or device function to get MAX_T_VAL.
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? 65504.F : FLT_MAX;
    extern __shared__ float invTemperatures[];
    if (threadIdx.x < batchSize)
    {
        invTemperatures[threadIdx.x] = 1.0f / (temperatures[threadIdx.x] + 1e-6f);
    }
    __syncthreads();

    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < batchSize * vocabSizePadded;
         index += blockDim.x * gridDim.x)
    {
        int batchIdx = index / vocabSizePadded;
        int vocabIdx = index % vocabSizePadded;
        T logit = (vocabIdx < vocabSize) ? logits[index] : -MAX_T_VAL;
        if (vocabIdx < vocabSize)
        {
            if (bias != nullptr)
            {
                logit += bias[vocabIdx];
            }
            logit *= invTemperatures[batchIdx];
        }
        logits[index] = logit;
    }
}

__global__ void batchApplyTemperaturePenalty_h2(half2* logits, const half2* bias, const float* temperatures,
    const int batchSize, const int vocabSize, const int vocabSizePaddeded)
{
    assert(vocabSize % 2 == 0);
    assert(vocabSizePaddeded % 2 == 0);
    extern __shared__ half2 h2InvTemperatures[];
    if (threadIdx.x < batchSize)
    {
        h2InvTemperatures[threadIdx.x] = __float2half2_rn(1.f / (temperatures[threadIdx.x] + 1e-6f));
    }
    __syncthreads();

    const half2 maskVal = __float2half2_rn(-65504.0f);
    const int halfVocabSize = vocabSize / 2;
    const int halfVocabSizePaddeded = vocabSizePaddeded / 2;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < batchSize * halfVocabSizePaddeded;
         index += blockDim.x * gridDim.x)
    {
        int batchIdx = index / halfVocabSizePaddeded;
        int vocabIdx = index % halfVocabSizePaddeded;
        half2 logit = vocabIdx < halfVocabSize ? __ldg(&logits[index]) : maskVal;
        if (vocabIdx < halfVocabSize)
        {
            if (bias != nullptr)
            {
                logit = __hadd2(logit, bias[vocabIdx]);
            }
            logits[index] = __hmul2(logit, h2InvTemperatures[batchIdx]);
        }
    }
}

template <typename T>
void invokeBatchApplyTemperaturePenalty(T* logits, const T* bias, const float* temperatures, const int batchSize,
    const int vocabSize, const int vocabSizePadded, cudaStream_t stream)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    dim3 block(min(vocabSizePadded, 1024));
    dim3 grid(min(batchSize * vocabSizePadded / block.x, 65536));
    if (std::is_same<T, half>::value && vocabSize % 2 == 0 && vocabSizePadded % 2 == 0)
    {
        size_t smemSize = sizeof(half2) * batchSize;
        batchApplyTemperaturePenalty_h2<<<grid, block, smemSize, stream>>>(reinterpret_cast<half2*>(logits),
            reinterpret_cast<const half2*>(bias), temperatures, batchSize, vocabSize, vocabSizePadded);
    }
    else
    {
        size_t smemSize = sizeof(float) * batchSize;
        batchApplyTemperaturePenalty<T>
            <<<grid, block, smemSize, stream>>>(logits, bias, temperatures, batchSize, vocabSize, vocabSizePadded);
    }
}

template void invokeBatchApplyTemperaturePenalty(float* logits, const float* bias, const float* temperatures,
    const int batchSize, const int vocabSize, const int vocabSizePadded, cudaStream_t stream);

template void invokeBatchApplyTemperaturePenalty(half* logits, const half* bias, const float* temperatures,
    const int batchSize, const int vocabSize, const int vocabSizePadded, cudaStream_t stream);

template <typename T, RepetitionPenaltyType penaltyType>
__global__ void batchApplyRepetitionPenalty(T* logits, const float* penalties, const int** outputIds,
    const int* sequenceLengths, const int batchSize, const int vocabSize, const int maxSeqLen)
{
    extern __shared__ float penaltyLogits[];
    int* penaltyIndices = (int*) (penaltyLogits + maxSeqLen);
    const int batchIdx = blockIdx.x;
    const float penalty = penalties[batchIdx];
    const int currentStep = sequenceLengths[batchIdx];

    logits += batchIdx * vocabSize;

    // Phase 1. Find indices to penalize and keep the penalized values.
    // A vocab id can appear multiple times but should be penalized once.
    for (int index = threadIdx.x; index < currentStep; index += blockDim.x)
    {
        // outputIds shape: (batchSize, input_len + output_len)
        int penaltyIndex = outputIds[batchIdx][blockIdx.y * maxSeqLen + index];
        penaltyIndices[index] = penaltyIndex;
        if (penaltyIndex >= vocabSize)
        {
            continue;
        }
        float logit = (float) logits[penaltyIndex];
        if (penaltyType == RepetitionPenaltyType::Additive)
        {
            penaltyLogits[index] = logit - penalty;
        }
        else if (penaltyType == RepetitionPenaltyType::Multiplicative)
        {
            penaltyLogits[index] = logit < 0.0f ? logit * penalty : logit / penalty;
        }
        else if (penaltyType == RepetitionPenaltyType::None)
        {
            penaltyLogits[index] = logit;
        }
        else
        {
            // Unsupported type
            assert(false);
        }
    }

    if (blockDim.x > 32)
    {
        __syncthreads();
    }

    // Phase 2. Replace a logit value by the penalized one.
    for (int index = threadIdx.x; index < currentStep; index += blockDim.x)
    {
        if (penaltyIndices[index] >= vocabSize)
        {
            continue;
        }
        logits[penaltyIndices[index]] = penaltyLogits[index];
    }
}

template <typename T>
void invokeBatchApplyRepetitionPenalty(T* logits, const float* penalties, const int** outputIds,
    const int* sequenceLengths, const int batchSize, const int vocabSize, RepetitionPenaltyType penaltyType,
    int maxSeqLen, cudaStream_t stream)
{
    TLLM_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    dim3 block(min(maxSeqLen, 1024));
    dim3 grid(batchSize);
    // FIXME(nkorobov): with long sequences we might hit upper smem limit
    size_t smemSize = maxSeqLen * (sizeof(float) + sizeof(int));
    if (penaltyType == RepetitionPenaltyType::Additive)
    {
        if (smemSize >= 46 * 1024)
        {
            /* Set 46KB threshold here because we have to take static/driver shared memory into consideration. */
            cudaError_t res = cudaFuncSetAttribute(batchApplyRepetitionPenalty<T, RepetitionPenaltyType::Additive>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smemSize);
            TLLM_CHECK_WITH_INFO(res == cudaSuccess,
                "Sequence Length is too long for the batchApplyRepetitionPenalty kernel (not enough shared memory).");
        }
        batchApplyRepetitionPenalty<T, RepetitionPenaltyType::Additive><<<grid, block, smemSize, stream>>>(
            logits, penalties, outputIds, sequenceLengths, batchSize, vocabSize, maxSeqLen);
    }
    else if (penaltyType == RepetitionPenaltyType::Multiplicative)
    {
        if (smemSize >= 46 * 1024)
        {
            /* Set 46KB threshold here because we have to take static/driver shared memory into consideration. */
            cudaError_t res
                = cudaFuncSetAttribute(batchApplyRepetitionPenalty<T, RepetitionPenaltyType::Multiplicative>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smemSize);
            TLLM_CHECK_WITH_INFO(res == cudaSuccess,
                "Sequence Length is too long for the batchApplyRepetitionPenalty kernel (not enough shared memory).");
        }
        batchApplyRepetitionPenalty<T, RepetitionPenaltyType::Multiplicative><<<grid, block, smemSize, stream>>>(
            logits, penalties, outputIds, sequenceLengths, batchSize, vocabSize, maxSeqLen);
    }
    else if (penaltyType == RepetitionPenaltyType::None)
    {
        // do nothing
    }
}

template void invokeBatchApplyRepetitionPenalty(float* logits, const float* penalties, const int** outputIds,
    const int* sequenceLengths, const int batchSize, const int vocabSize, RepetitionPenaltyType penaltyType,
    int maxSeqLen, cudaStream_t stream);

template void invokeBatchApplyRepetitionPenalty(half* logits, const float* penalties, const int** outputIds,
    const int* sequenceLengths, const int batchSize, const int vocabSize, RepetitionPenaltyType penaltyType,
    int maxSeqLen, cudaStream_t stream);

template <typename T>
__global__ void batchApplyMinLengthPenalty(T* logits, const int* minLengths, const int* endIds,
    const int* sequenceLengths, const int* contextLengths, const int vocabSizePaddeded)
{
    int bid = threadIdx.x + blockIdx.x * blockDim.x; // batch index
    auto const contextLength{contextLengths == nullptr ? 0 : contextLengths[bid]};
    // This kernel is called before sequenceLengths is incrememnted.
    // We need +1 because sequenceLengths = contextLength + numGenTokens - 1, which is equal to the length of k/v
    // caches.
    if (sequenceLengths[bid] + 1 - contextLength < minLengths[bid])
    {
        T maskVal = (std::is_same<T, half>::value) ? -65504.0f : -FLT_MAX;
        logits[bid * vocabSizePaddeded + endIds[bid]] = maskVal;
    }
}

template <typename T>
void invokeMinLengthPenalty(T* logits, const int* minLengths, const int* endIds, const int* sequneceLengths,
    const int* contextLengths, const int batchSize, const int vocabSizePaddeded, cudaStream_t stream)

{
    const int blockSize = min(batchSize, 1024);
    const int gridSize = (batchSize + blockSize - 1) / blockSize;
    batchApplyMinLengthPenalty<<<gridSize, blockSize, 0, stream>>>(
        logits, minLengths, endIds, sequneceLengths, contextLengths, vocabSizePaddeded);
}

template void invokeMinLengthPenalty(float* logits, const int* minLengths, const int* endIds,
    const int* sequneceLengths, const int* contextLengths, const int batchSize, const int vocabSizePaddeded,
    cudaStream_t stream);

template void invokeMinLengthPenalty(half* logits, const int* minLengths, const int* endIds, const int* sequneceLengths,
    const int* contextLengths, const int batchSize, const int vocabSizePaddeded, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
