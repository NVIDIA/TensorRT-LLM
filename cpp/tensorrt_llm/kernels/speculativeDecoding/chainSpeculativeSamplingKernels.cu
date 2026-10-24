/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "chainSpeculativeSamplingKernels.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/fusedSampling/fusedSamplingKernels.h"

#include <cub/cub.cuh>
#include <curand_kernel.h>

#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::speculative_decoding
{

namespace
{

constexpr int32_t kBlockSize = 1024;

struct Float2SumOp
{
    __device__ __forceinline__ float2 operator()(float2 a, float2 b) const
    {
        return make_float2(a.x + b.x, a.y + b.y);
    }
};

struct MaxOp
{
    __device__ __forceinline__ int32_t operator()(int32_t a, int32_t b) const
    {
        return a > b ? a : b;
    }
};

template <int32_t VEC_SIZE>
__device__ __forceinline__ void loadProbs(float const* row, uint32_t base, uint32_t vocabSize, float (&out)[VEC_SIZE])
{
    if constexpr (VEC_SIZE == 4)
    {
        if (base + VEC_SIZE <= vocabSize)
        {
            float4 const v = *reinterpret_cast<float4 const*>(row + base);
            out[0] = v.x;
            out[1] = v.y;
            out[2] = v.z;
            out[3] = v.w;
            return;
        }
    }
#pragma unroll
    for (int32_t j = 0; j < VEC_SIZE; ++j)
    {
        out[j] = base + j < vocabSize ? row[base + j] : 0.0f;
    }
}

//! The distribution a request resamples from: max(0, target - draft) at the first rejected
//! position, or the target itself when every draft token was accepted (draft == nullptr) or
//! when the residual has no mass.
template <int32_t VEC_SIZE>
__device__ __forceinline__ void loadResidual(
    float const* target, float const* draft, bool targetOnly, uint32_t base, uint32_t vocabSize, float (&out)[VEC_SIZE])
{
    loadProbs<VEC_SIZE>(target, base, vocabSize, out);
    if (targetOnly)
    {
        return;
    }
    float d[VEC_SIZE];
    loadProbs<VEC_SIZE>(draft, base, vocabSize, d);
#pragma unroll
    for (int32_t j = 0; j < VEC_SIZE; ++j)
    {
        out[j] = fmaxf(out[j] - d[j], 0.0f);
    }
}

template <int32_t BLOCK_SIZE, int32_t VEC_SIZE>
__global__ void __launch_bounds__(BLOCK_SIZE) chainSpeculativeSamplingKernel(ChainSpeculativeSamplingParams params)
{
    using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
    using BlockReducePair = cub::BlockReduce<float2, BLOCK_SIZE>;
    using BlockReduceInt = cub::BlockReduce<int32_t, BLOCK_SIZE>;
    using BlockScan = cub::BlockScan<float, BLOCK_SIZE>;

    __shared__ union
    {
        typename BlockReduce::TempStorage reduce;
        typename BlockReducePair::TempStorage reducePair;
        typename BlockReduceInt::TempStorage reduceInt;
        typename BlockScan::TempStorage scan;
    } tempStorage;

    __shared__ int32_t sPos;
    __shared__ float sU;
    __shared__ bool sTargetOnly;
    __shared__ float sTotal;
    __shared__ float sTileSum[2];
    __shared__ int32_t sWinner;
    __shared__ int32_t sLastValid;

    int64_t const row = blockIdx.x;
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    int32_t const numDraft = params.numDraftTokens;
    uint32_t const vocabSize = static_cast<uint32_t>(params.vocabSize);
    int32_t* outTokens = params.outputTokenIds + row * (numDraft + 1);

    if (tid == 0)
    {
        curandStatePhilox4_32_10_t state;
        curand_init(static_cast<uint64_t>(params.seed[row]),
            kPerRowSubsequenceBase | static_cast<uint64_t>(params.offset[row]), 0, &state);
        int32_t pos = numDraft;
        for (int32_t i = 0; i < numDraft; ++i)
        {
            int32_t const token = params.draftTokenIds[row * numDraft + i];
            float const pTarget = params.targetProbs[(row * (numDraft + 1) + i) * vocabSize + token];
            float const pDraft = params.draftProbs[(row * numDraft + i) * vocabSize + token];
            if (curand_uniform(&state) * pDraft < pTarget)
            {
                outTokens[i] = token;
            }
            else
            {
                pos = i;
                break;
            }
        }
        sPos = pos;
        // curand_uniform is in (0, 1]; the inverse-CDF search below wants [0, 1).
        sU = 1.0f - curand_uniform(&state);
        sWinner = static_cast<int32_t>(vocabSize);
    }
    __syncthreads();

    int32_t const pos = sPos;
    float const* target = params.targetProbs + (row * (numDraft + 1) + pos) * vocabSize;
    float const* draft = pos < numDraft ? params.draftProbs + (row * numDraft + pos) * vocabSize : nullptr;
    uint32_t const tileSize = BLOCK_SIZE * VEC_SIZE;

    // x: target mass, y: residual mass.
    float2 sums = make_float2(0.0f, 0.0f);
    for (uint32_t base = tid * VEC_SIZE; base < vocabSize; base += tileSize)
    {
        float t[VEC_SIZE];
        loadProbs<VEC_SIZE>(target, base, vocabSize, t);
        float d[VEC_SIZE] = {};
        if (draft != nullptr)
        {
            loadProbs<VEC_SIZE>(draft, base, vocabSize, d);
        }
#pragma unroll
        for (int32_t j = 0; j < VEC_SIZE; ++j)
        {
            sums.x += t[j];
            sums.y += fmaxf(t[j] - d[j], 0.0f);
        }
    }
    float2 const blockSums = BlockReducePair(tempStorage.reducePair).Reduce(sums, Float2SumOp{});
    if (tid == 0)
    {
        sTargetOnly = draft == nullptr || !(blockSums.y > 0.0f);
        sTotal = sTargetOnly ? blockSums.x : blockSums.y;
    }
    __syncthreads();

    // Inverse-CDF search: every tile is only summed, and the one tile the threshold falls in is
    // scanned to find the token.
    bool const targetOnly = sTargetOnly;
    float const threshold = sU * sTotal;
    float prefix = 0.0f;
    int32_t localLastValid = -1;
    int32_t parity = 0;
    for (uint32_t tileBase = 0; tileBase < vocabSize; tileBase += tileSize, parity ^= 1)
    {
        uint32_t const base = tileBase + tid * VEC_SIZE;
        float value[VEC_SIZE];
        loadResidual<VEC_SIZE>(target, draft, targetOnly, base, vocabSize, value);
#pragma unroll
        for (int32_t j = 0; j < VEC_SIZE; ++j)
        {
            if (value[j] > 0.0f)
            {
                localLastValid = static_cast<int32_t>(base + j);
            }
        }
        float const tileSum = BlockReduce(tempStorage.reduce).Sum(value);
        if (tid == 0)
        {
            sTileSum[parity] = tileSum;
        }
        __syncthreads();
        if (prefix + sTileSum[parity] > threshold)
        {
            float inclusive[VEC_SIZE];
            BlockScan(tempStorage.scan).InclusiveSum(value, inclusive);
#pragma unroll
            for (int32_t j = 0; j < VEC_SIZE; ++j)
            {
                if (value[j] > 0.0f && prefix + inclusive[j] > threshold)
                {
                    atomicMin(&sWinner, static_cast<int32_t>(base + j));
                    break;
                }
            }
            __syncthreads();
            break;
        }
        prefix += sTileSum[parity];
    }

    if (sWinner == static_cast<int32_t>(vocabSize))
    {
        // Float rounding can leave the summed prefix just short of the threshold; the last
        // token with mass is then the one the threshold falls on.
        int32_t const lastValid = BlockReduceInt(tempStorage.reduceInt).Reduce(localLastValid, MaxOp{});
        if (tid == 0)
        {
            sLastValid = lastValid;
        }
    }

    if (tid == 0)
    {
        int32_t const token = sWinner < static_cast<int32_t>(vocabSize) ? sWinner : max(sLastValid, 0);
        outTokens[pos] = token;
        for (int32_t i = pos + 1; i <= numDraft; ++i)
        {
            outTokens[i] = -1;
        }
        params.outputNumTokens[row] = pos + 1;
    }
}

bool isAligned(void const* ptr, uintptr_t alignment)
{
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

} // namespace

void invokeChainSpeculativeSampling(ChainSpeculativeSamplingParams const& params, cudaStream_t stream)
{
    if (params.batchSize == 0)
    {
        return;
    }
    dim3 const grid(params.batchSize);
    dim3 const block(kBlockSize);
    bool const vectorized = params.vocabSize % 4 == 0 && isAligned(params.targetProbs, sizeof(float4))
        && isAligned(params.draftProbs, sizeof(float4));
    if (vectorized)
    {
        chainSpeculativeSamplingKernel<kBlockSize, 4><<<grid, block, 0, stream>>>(params);
    }
    else
    {
        chainSpeculativeSamplingKernel<kBlockSize, 1><<<grid, block, 0, stream>>>(params);
    }
    sync_check_cuda_error(stream);
}

} // namespace kernels::speculative_decoding

TRTLLM_NAMESPACE_END
