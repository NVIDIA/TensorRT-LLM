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

// ST-01-REJ: the stock chainSpeculativeSamplingKernel<1024, 4> on ctasPerRow CTAs
// per request, bitwise. v2 (bitwise v1 and stock, 24.8 vs 32.3 us at one request)
// moves loads only, every add and compare in its v1 order: the acceptance loop's probabilities are loaded in parallel
// first, the chain loop is unrolled, and the leader reads the per-tile values from shared memory. See
// chainSpecSamplingMcSm103.h for the contract. Compiled with the repo's default flags, as
// kernels/speculativeDecoding/chainSpeculativeSamplingKernels.cu.

#include "tensorrt_llm/kernels/fusedSampling/fusedSamplingKernels.h"
#include "tensorrt_llm/kernels/minimaxM31/chainSpecSamplingMcSm103.h"

#include <cuda/atomic>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <cstdint>

#include <cub/cub.cuh>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

namespace
{

using speculative_decoding::ChainSpeculativeSamplingParams;

constexpr int32_t kBlockSize = 1024;
constexpr int32_t kVecSize = 4;
constexpr int32_t kTileSize = kBlockSize * kVecSize;
constexpr int32_t kMaxTiles = kChainSpecMcMaxVocab / kTileSize;

//! One request's cross-CTA state.
struct RowWorkspace
{
    //! The stock kernel's per-thread (target, residual) mass chains.
    float2 chain[kBlockSize];
    //! Per tile: the stock per-tile residual (or target) sum, and the last index with mass (-1 if none).
    float tileSum[kMaxTiles];
    int32_t tileLast[kMaxTiles];
    float total;
    int32_t targetOnly;
    //! The row barrier: CTAs arrived in this generation, and the generation (only increases).
    unsigned arrive;
    unsigned gen;
    int32_t pad[3];
};

// ---------------------------------------------------------------------------------------------------------------------
// Identical to kernels/speculativeDecoding/chainSpeculativeSamplingKernels.cu (Float2SumOp, loadProbs, loadResidual).
// ---------------------------------------------------------------------------------------------------------------------
struct Float2SumOp
{
    __device__ __forceinline__ float2 operator()(float2 a, float2 b) const
    {
        return make_float2(a.x + b.x, a.y + b.y);
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

// ---------------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ unsigned loadAcquire(unsigned* p)
{
    return cuda::atomic_ref<unsigned, cuda::thread_scope_device>(*p).load(cuda::memory_order_acquire);
}

//! A barrier of the request's G CTAs; the last CTA to arrive runs leader() (all its threads) before releasing the
//! others.
template <typename F>
__device__ __forceinline__ void rowSync(RowWorkspace* w, int32_t G, F leader)
{
    __shared__ unsigned sGen;
    __shared__ int32_t sLast;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        sGen = loadAcquire(&w->gen);
        sLast = atomicAdd(&w->arrive, 1u) == static_cast<unsigned>(G - 1);
    }
    __syncthreads();
    if (sLast)
    {
        __threadfence();
        leader();
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0)
        {
            atomicExch(&w->arrive, 0u);
            cuda::atomic_ref<unsigned, cuda::thread_scope_device>(w->gen).store(sGen + 1, cuda::memory_order_release);
        }
    }
    else if (threadIdx.x == 0)
    {
        while (loadAcquire(&w->gen) == sGen)
        {
            __nanosleep(20);
        }
    }
    __syncthreads();
}

__global__ void __launch_bounds__(kBlockSize, 1)
    chainSpecSamplingMcKernel(ChainSpeculativeSamplingParams params, RowWorkspace* ws)
{
    using BlockReduce = cub::BlockReduce<float, kBlockSize>;
    using BlockReducePair = cub::BlockReduce<float2, kBlockSize>;
    using BlockScan = cub::BlockScan<float, kBlockSize>;

    __shared__ union
    {
        typename BlockReduce::TempStorage reduce;
        typename BlockReducePair::TempStorage reducePair;
        typename BlockScan::TempStorage scan;
    } tempStorage;

    __shared__ int32_t sPos;
    __shared__ float sU;
    __shared__ int32_t sWinner;
    __shared__ float sPrefix;
    __shared__ int32_t sTile;
    __shared__ int32_t sTileLast;

    int64_t const row = blockIdx.y;
    int32_t const G = static_cast<int32_t>(gridDim.x);
    int32_t const cta = static_cast<int32_t>(blockIdx.x);
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    int32_t const numDraft = params.numDraftTokens;
    uint32_t const vocabSize = static_cast<uint32_t>(params.vocabSize);
    RowWorkspace* w = ws + row;
    int32_t* outTokens = params.outputTokenIds + row * (numDraft + 1);

    // 1. Acceptance: the stock tid-0 loop, identical in every CTA of the request. The probabilities it may read are
    //    loaded by numDraft threads in parallel first (the loop's values and order unchanged).
    __shared__ float sPTarget[kChainSpecMcMaxDraft];
    __shared__ float sPDraft[kChainSpecMcMaxDraft];
    if (tid < numDraft)
    {
        int32_t const token = params.draftTokenIds[row * numDraft + tid];
        sPTarget[tid] = params.targetProbs[(row * (numDraft + 1) + tid) * vocabSize + token];
        sPDraft[tid] = params.draftProbs[(row * numDraft + tid) * vocabSize + token];
    }
    __syncthreads();
    if (tid == 0)
    {
        curandStatePhilox4_32_10_t state;
        curand_init(static_cast<uint64_t>(params.seed[row]),
            kPerRowSubsequenceBase | static_cast<uint64_t>(params.offset[row]), 0, &state);
        int32_t pos = numDraft;
        for (int32_t i = 0; i < numDraft; ++i)
        {
            float const pTarget = sPTarget[i];
            float const pDraft = sPDraft[i];
            if (!(curand_uniform(&state) * pDraft < pTarget))
            {
                pos = i;
                break;
            }
        }
        sPos = pos;
        // curand_uniform is in (0, 1]; the inverse-CDF search below wants [0, 1).
        sU = 1.0f - curand_uniform(&state);
    }
    __syncthreads();

    int32_t const pos = sPos;
    float const* target = params.targetProbs + (row * (numDraft + 1) + pos) * vocabSize;
    float const* draft = pos < numDraft ? params.draftProbs + (row * numDraft + pos) * vocabSize : nullptr;

    // 2. The stock per-thread (target, residual) chains, spread over the request's CTAs; one CTA reduces them.
    int32_t const chainsPerCta = kBlockSize / G;
    if (tid < chainsPerCta)
    {
        int32_t const t = cta * chainsPerCta + tid;
        // x: target mass, y: residual mass.
        float2 sums = make_float2(0.0f, 0.0f);
#pragma unroll 8
        for (uint32_t base = t * kVecSize; base < vocabSize; base += kTileSize)
        {
            float tv[kVecSize];
            loadProbs<kVecSize>(target, base, vocabSize, tv);
            float dv[kVecSize] = {};
            if (draft != nullptr)
            {
                loadProbs<kVecSize>(draft, base, vocabSize, dv);
            }
#pragma unroll
            for (int32_t j = 0; j < kVecSize; ++j)
            {
                sums.x += tv[j];
                sums.y += fmaxf(tv[j] - dv[j], 0.0f);
            }
        }
        w->chain[t] = sums;
    }
    rowSync(w, G,
        [&]
        {
            float2 const blockSums
                = BlockReducePair(tempStorage.reducePair).Reduce(__ldcg(&w->chain[tid]), Float2SumOp{});
            if (tid == 0)
            {
                int32_t const targetOnly = draft == nullptr || !(blockSums.y > 0.0f);
                w->targetOnly = targetOnly;
                w->total = targetOnly ? blockSums.x : blockSums.y;
            }
        });
    bool const targetOnly = __ldcg(&w->targetOnly) != 0;

    // 3. Tile sums (the stock per-tile cub reduce) and each tile's last index with mass (the stock fallback takes the
    //    max over the tiles it visited, i.e. up to and including the tile where the threshold falls).
    int32_t const numTiles = static_cast<int32_t>((vocabSize + kTileSize - 1) / kTileSize);
    for (int32_t tile = cta; tile < numTiles; tile += G)
    {
        if (tid == 0)
        {
            sTileLast = -1;
        }
        __syncthreads();
        uint32_t const base = tile * kTileSize + tid * kVecSize;
        float value[kVecSize];
        loadResidual<kVecSize>(target, draft, targetOnly, base, vocabSize, value);
        int32_t localLast = -1;
#pragma unroll
        for (int32_t j = 0; j < kVecSize; ++j)
        {
            if (value[j] > 0.0f)
            {
                localLast = static_cast<int32_t>(base + j);
            }
        }
        if (localLast >= 0)
        {
            atomicMax(&sTileLast, localLast);
        }
        float const tileSum = BlockReduce(tempStorage.reduce).Sum(value);
        __syncthreads();
        if (tid == 0)
        {
            w->tileSum[tile] = tileSum;
            w->tileLast[tile] = sTileLast;
        }
        __syncthreads();
    }

    // 4. One CTA: the tile prefix in tile order, then the stock in-tile scan; it writes the outputs.
    rowSync(w, G,
        [&]
        {
            float const threshold = sU * __ldcg(&w->total);
            // The per-tile values in shared memory, read by numTiles threads in parallel.
            __shared__ float sTileSum[kMaxTiles];
            __shared__ int32_t sTileLastOf[kMaxTiles];
            if (tid < numTiles)
            {
                sTileSum[tid] = __ldcg(&w->tileSum[tid]);
                sTileLastOf[tid] = __ldcg(&w->tileLast[tid]);
            }
            __syncthreads();
            if (tid == 0)
            {
                float prefix = 0.0f;
                int32_t found = -1;
                for (int32_t tile = 0; tile < numTiles; ++tile)
                {
                    float const tileSum = sTileSum[tile];
                    if (prefix + tileSum > threshold)
                    {
                        found = tile;
                        break;
                    }
                    prefix += tileSum;
                }
                sTile = found;
                sPrefix = prefix;
                sWinner = static_cast<int32_t>(vocabSize);
            }
            __syncthreads();
            if (sTile >= 0)
            {
                uint32_t const base = sTile * kTileSize + tid * kVecSize;
                float value[kVecSize];
                loadResidual<kVecSize>(target, draft, targetOnly, base, vocabSize, value);
                float inclusive[kVecSize];
                BlockScan(tempStorage.scan).InclusiveSum(value, inclusive);
                float const prefix = sPrefix;
#pragma unroll
                for (int32_t j = 0; j < kVecSize; ++j)
                {
                    if (value[j] > 0.0f && prefix + inclusive[j] > threshold)
                    {
                        atomicMin(&sWinner, static_cast<int32_t>(base + j));
                        break;
                    }
                }
            }
            __syncthreads();
            if (tid == 0)
            {
                // Float rounding can leave the summed prefix just short of the threshold; the last token with mass
                // (over the tiles the stock loop visits) is then the one the threshold falls on.
                int32_t const upto = sTile >= 0 ? sTile : numTiles - 1;
                int32_t lastValid = -1;
                for (int32_t tile = 0; tile <= upto; ++tile)
                {
                    lastValid = max(lastValid, sTileLastOf[tile]);
                }
                int32_t const token = sWinner < static_cast<int32_t>(vocabSize) ? sWinner : max(lastValid, 0);
                for (int32_t i = 0; i < pos; ++i)
                {
                    outTokens[i] = params.draftTokenIds[row * numDraft + i];
                }
                outTokens[pos] = token;
                for (int32_t i = pos + 1; i <= numDraft; ++i)
                {
                    outTokens[i] = -1;
                }
                params.outputNumTokens[row] = pos + 1;
            }
        });
}

} // namespace

cudaError_t invokeChainSpecSamplingMc(
    ChainSpeculativeSamplingParams const& params, int ctasPerRow, void* workspace, cudaStream_t stream)
{
    if (params.batchSize <= 0)
    {
        return cudaSuccess;
    }
    if (ctasPerRow < 1 || ctasPerRow > kBlockSize || kBlockSize % ctasPerRow != 0 || params.vocabSize <= 0
        || params.vocabSize % kVecSize != 0 || params.vocabSize > kChainSpecMcMaxVocab || params.numDraftTokens < 0
        || params.numDraftTokens > kChainSpecMcMaxDraft || params.batchSize > 65535)
    {
        return cudaErrorInvalidValue;
    }
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(static_cast<unsigned>(ctasPerRow), static_cast<unsigned>(params.batchSize));
    cfg.blockDim = dim3(kBlockSize);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    return cudaLaunchKernelEx(&cfg, chainSpecSamplingMcKernel, params, static_cast<RowWorkspace*>(workspace));
}

size_t chainSpecSamplingMcWorkspaceBytesPerRow()
{
    return sizeof(RowWorkspace);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
