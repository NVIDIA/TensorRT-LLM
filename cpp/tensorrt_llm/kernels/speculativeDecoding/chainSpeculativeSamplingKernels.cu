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
//! Consecutive tiles a pass loads before reducing any of them, their sums folded in one block
//! reduction: a 248320-token row is 16 such groups on the aligned path instead of 61 tiles.
constexpr int32_t kTilesInFlight = 4;

//! Per-tile sums of a group of tiles, reduced over the block together.
template <int32_t TILES>
struct TileSums
{
    float v[TILES];
};

template <int32_t TILES>
struct TileSumsOp
{
    __device__ __forceinline__ TileSums<TILES> operator()(TileSums<TILES> const& a, TileSums<TILES> const& b) const
    {
        TileSums<TILES> r;
#pragma unroll
        for (int32_t k = 0; k < TILES; ++k)
        {
            r.v[k] = a.v[k] + b.v[k];
        }
        return r;
    }
};

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

//! A tile is BLOCK_SIZE x VEC_SIZE consecutive probabilities, thread t holding VEC_SIZE of them
//! from t x VEC_SIZE (coalesced). Both passes walk the row TILES tiles at a time: all of a
//! group's loads are issued before any of its sums is formed, and the group's per-tile sums
//! take one block reduction. The inverse-CDF search keeps its tile order and the scan of the
//! one tile the threshold falls in.
template <int32_t BLOCK_SIZE, int32_t VEC_SIZE, int32_t TILES>
__global__ void __launch_bounds__(BLOCK_SIZE) chainSpeculativeSamplingKernel(ChainSpeculativeSamplingParams params)
{
    using BlockReduceTiles = cub::BlockReduce<TileSums<TILES>, BLOCK_SIZE>;
    using BlockReducePair = cub::BlockReduce<float2, BLOCK_SIZE>;
    using BlockReduceInt = cub::BlockReduce<int32_t, BLOCK_SIZE>;
    using BlockScan = cub::BlockScan<float, BLOCK_SIZE>;

    __shared__ union
    {
        typename BlockReduceTiles::TempStorage reduceTiles;
        typename BlockReducePair::TempStorage reducePair;
        typename BlockReduceInt::TempStorage reduceInt;
        typename BlockScan::TempStorage scan;
    } tempStorage;

    __shared__ int32_t sPos;
    __shared__ float sU;
    __shared__ bool sTargetOnly;
    __shared__ float sTotal;
    __shared__ float sTileSum[2][TILES];
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
    uint32_t const groupSize = tileSize * TILES;

    // x: target mass, y: residual mass.
    float2 sums = make_float2(0.0f, 0.0f);
    for (uint32_t groupBase = 0; groupBase < vocabSize; groupBase += groupSize)
    {
        float t[TILES][VEC_SIZE];
        float d[TILES][VEC_SIZE] = {};
#pragma unroll
        for (int32_t k = 0; k < TILES; ++k)
        {
            loadProbs<VEC_SIZE>(target, groupBase + k * tileSize + tid * VEC_SIZE, vocabSize, t[k]);
        }
        if (draft != nullptr)
        {
#pragma unroll
            for (int32_t k = 0; k < TILES; ++k)
            {
                loadProbs<VEC_SIZE>(draft, groupBase + k * tileSize + tid * VEC_SIZE, vocabSize, d[k]);
            }
        }
#pragma unroll
        for (int32_t k = 0; k < TILES; ++k)
        {
#pragma unroll
            for (int32_t j = 0; j < VEC_SIZE; ++j)
            {
                sums.x += t[k][j];
                sums.y += fmaxf(t[k][j] - d[k][j], 0.0f);
            }
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
    for (uint32_t groupBase = 0; groupBase < vocabSize; groupBase += groupSize, parity ^= 1)
    {
        float value[TILES][VEC_SIZE];
        int32_t tileLastValid[TILES];
        TileSums<TILES> mine;
#pragma unroll
        for (int32_t k = 0; k < TILES; ++k)
        {
            uint32_t const base = groupBase + k * tileSize + tid * VEC_SIZE;
            loadResidual<VEC_SIZE>(target, draft, targetOnly, base, vocabSize, value[k]);
            tileLastValid[k] = -1;
            mine.v[k] = 0.0f;
#pragma unroll
            for (int32_t j = 0; j < VEC_SIZE; ++j)
            {
                if (value[k][j] > 0.0f)
                {
                    tileLastValid[k] = static_cast<int32_t>(base + j);
                }
                mine.v[k] += value[k][j];
            }
        }
        TileSums<TILES> const groupSums = BlockReduceTiles(tempStorage.reduceTiles).Reduce(mine, TileSumsOp<TILES>{});
        if (tid == 0)
        {
#pragma unroll
            for (int32_t k = 0; k < TILES; ++k)
            {
                sTileSum[parity][k] = groupSums.v[k];
            }
        }
        __syncthreads();
        // The tiles in order: the first whose sum carries the prefix past the threshold is
        // scanned; the ones before it only add to the prefix.
        int32_t hit = -1;
#pragma unroll
        for (int32_t k = 0; k < TILES; ++k)
        {
            if (hit < 0)
            {
                if (prefix + sTileSum[parity][k] > threshold)
                {
                    hit = k;
                }
                else
                {
                    prefix += sTileSum[parity][k];
                }
            }
        }
#pragma unroll
        for (int32_t k = 0; k < TILES; ++k)
        {
            if (hit < 0 || k <= hit)
            {
                localLastValid = max(localLastValid, tileLastValid[k]);
            }
        }
        if (hit >= 0)
        {
            float hitValue[VEC_SIZE];
#pragma unroll
            for (int32_t k = 0; k < TILES; ++k)
            {
                if (k == hit)
                {
#pragma unroll
                    for (int32_t j = 0; j < VEC_SIZE; ++j)
                    {
                        hitValue[j] = value[k][j];
                    }
                }
            }
            uint32_t const base = groupBase + hit * tileSize + tid * VEC_SIZE;
            float inclusive[VEC_SIZE];
            BlockScan(tempStorage.scan).InclusiveSum(hitValue, inclusive);
#pragma unroll
            for (int32_t j = 0; j < VEC_SIZE; ++j)
            {
                if (hitValue[j] > 0.0f && prefix + inclusive[j] > threshold)
                {
                    atomicMin(&sWinner, static_cast<int32_t>(base + j));
                    break;
                }
            }
            __syncthreads();
            break;
        }
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
        chainSpeculativeSamplingKernel<kBlockSize, 4, kTilesInFlight><<<grid, block, 0, stream>>>(params);
    }
    else
    {
        chainSpeculativeSamplingKernel<kBlockSize, 1, kTilesInFlight><<<grid, block, 0, stream>>>(params);
    }
    sync_check_cuda_error(stream);
}

} // namespace kernels::speculative_decoding

TRTLLM_NAMESPACE_END
