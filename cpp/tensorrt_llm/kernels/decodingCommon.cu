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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include <stdio.h>

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

__global__ void curandInitialize(curandState_t* state, const int size, const uint64_t randomSeed)
{
    if (threadIdx.x + blockIdx.x * blockDim.x < size)
    {
        curand_init(randomSeed, 0, 0, &state[blockIdx.x * blockDim.x + threadIdx.x]);
    }
}

void invokeCurandInitialize(
    curandState_t* state, const size_t batchSize, const uint64_t randomSeed, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((int) (ceil(batchSize * 1.0 / 256)));
    curandInitialize<<<grid, block, 0, stream>>>(state, batchSize, randomSeed);
}

__global__ void curandBatchInitialize(curandState_t* states, const int size, const uint64_t* randomSeeds)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        curand_init(randomSeeds[idx], 0, 0, &states[idx]);
    }
}

void invokeCurandBatchInitialize(
    curandState_t* states, const size_t batchSize, const uint64_t* randomSeeds, cudaStream_t stream)
{
    dim3 block(256);
    dim3 grid((int) (ceil(batchSize * 1.0 / 256)));
    curandBatchInitialize<<<grid, block, 0, stream>>>(states, batchSize, randomSeeds);
}

template <typename T>
__global__ void addBiasSoftMax(T* logits, T* probs, const T* bias, const int* endIds, const FinishedState* finished,
    const int vocabSize, const int vocabSizePadded)
{
    int bid = blockIdx.x;
    const FinishedState finishState = finished != nullptr ? finished[bid] : FinishedState::empty();
    if (finishState.isSkipDecoding())
    {
        return;
    }

    bool finish = finishState.isFinished();
    int offset = bid * vocabSizePadded;

    float maxVal = -1 * FLT_MAX;
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
    __shared__ float sMaxVal;
    __shared__ float sSumVal;

    for (int tid = threadIdx.x; tid < vocabSizePadded; tid += blockDim.x)
    {
        if (tid < vocabSize)
        {
            if (finish && endIds != nullptr)
            {
                logits[offset + tid] = (tid == endIds[bid]) ? MAX_T_VAL : -MAX_T_VAL;
            }
            else
            {
                T bias_val = (bias != nullptr) ? bias[tid] : (T) 0.0f;
                logits[offset + tid] += bias_val;
            }
        }
        else
        {
            logits[offset + tid] = -MAX_T_VAL;
        }
        maxVal = max(maxVal, (float) logits[offset + tid]);
    }

    maxVal = blockReduceMax<float>((float) maxVal);
    if (threadIdx.x == 0)
    {
        sMaxVal = maxVal;
    }
    __syncthreads();

    float sumVal = 0.0f;
    for (int tid = threadIdx.x; tid < vocabSizePadded; tid += blockDim.x)
    {
        probs[offset + tid] = __expf((float) logits[offset + tid] - sMaxVal);
        sumVal += (float) probs[offset + tid];
    }

    sumVal = blockReduceSum<float>(sumVal);
    if (threadIdx.x == 0)
    {
        sSumVal = sumVal;
    }
    __syncthreads();

    for (int tid = threadIdx.x; tid < vocabSizePadded; tid += blockDim.x)
    {
        probs[offset + tid] = ((float) probs[offset + tid] / (sSumVal + 1e-6f));
    }
}

template <typename T>
void invokeAddBiasSoftMax(T* logits, T* probs, const T* bias, const int* endIds, const FinishedState* finished,
    const int batchSize, const int vocabSize, const int vocabSizePadded, cudaStream_t stream)
{
    dim3 grid(batchSize);
    auto const vocabRoundedToWarp = roundUp(vocabSize, 32);
    dim3 block(min(vocabRoundedToWarp, 1024));
    // vocabSize, e.g., 30000, 7000.... vocabSize is usually very big.
    addBiasSoftMax<<<grid, block, 0, stream>>>(logits, probs, bias, endIds, finished, vocabSize, vocabSizePadded);
}

template void invokeAddBiasSoftMax(float* logits, float* probs, const float* bias, const int* endIds,
    const FinishedState* finished, const int m, const int nPadded, const int n, cudaStream_t stream);

template void invokeAddBiasSoftMax(half* logits, half* probs, const half* bias, const int* endIds,
    const FinishedState* finished, const int m, const int nPadded, const int n, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
