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

// REJ_MC: the stock chainSpeculativeSamplingKernel<1024, 4> on ctasPerRow CTAs
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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>

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
    //! v3 only: the target-only variant of tileSum / tileLast (v3 computes both before it knows targetOnly; tileSum /
    //! tileLast then hold the residual variant).
    float tileSumTarget[kMaxTiles];
    int32_t tileLastTarget[kMaxTiles];
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

// Request row's draft distribution at position i: params.draftProbs[row, i, :] of a contiguous [batch, numDraft, vocab]
// buffer (rows.rowIds null), or, slot-indexed, rows.rowIds[row] * rows.rowStride + i * rows.posStride into a larger
// [slots, maxDraft, >= vocab] buffer (the gather the caller would otherwise copy out first; the same floats).
__device__ __forceinline__ float const* draftRow(ChainSpeculativeSamplingParams const& params,
    ChainSpecMcDraftRows const& rows, int64_t row, int32_t i, int32_t numDraft, uint32_t vocabSize)
{
    if (rows.rowIds == nullptr)
    {
        return params.draftProbs + (row * numDraft + i) * static_cast<int64_t>(vocabSize);
    }
    return params.draftProbs + rows.rowIds[row] * rows.rowStride + i * rows.posStride;
}

__global__ void __launch_bounds__(kBlockSize, 1)
    chainSpecSamplingMcKernel(ChainSpeculativeSamplingParams params, RowWorkspace* ws, ChainSpecMcDraftRows rows)
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
        sPDraft[tid] = draftRow(params, rows, row, tid, numDraft, vocabSize)[token];
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
    float const* draft = pos < numDraft ? draftRow(params, rows, row, pos, numDraft, vocabSize) : nullptr;

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

// ---------------------------------------------------------------------------------------------------------------------
// v3 (VR). The v2 stamps put the chains at 10.4 of 21.3 us per call at one
// request: only 1024 / G threads of each CTA walk them, each through ~49 dependent loads. v3 keeps every add, compare
// and their order (v2's and the stock kernel's) and moves only loads and the cross-CTA schedule:
//   A. all 1024 threads stage the CTA's chain elements in shared memory (chunkK 4096-strides at a time), then the chain
//      threads run the stock chain loop over them;
//   B. every CTA computes both stock tile variants (target only, residual) before the row barrier, so the barrier's
//      leader runs the final search directly: one barrier per call, and the other CTAs exit on arrival;
//   C. the leader's tile prefix runs unrolled from shared memory (same adds, same first-crossing exit), the fallback's
//      last index with mass is an integer max (order-free), and numDraft + 1 threads store the outputs from the draft
//      ids loaded in phase 1.
// ---------------------------------------------------------------------------------------------------------------------

//! Staged chains: float4 slots per probability row (target, draft) in dynamic shared memory, 64 KiB each.
constexpr int32_t kStageVecsPerRow = 4096;
//! Tiles a CTA holds in registers at once while it sums both tile variants.
constexpr int32_t kTileRegs = 2;

//! The row barrier without a release: the last CTA to arrive runs leader() (all its threads) and resets the barrier
//! (arrival count, generation; the same state rowSync leaves, so v2 and v3 launches may share a workspace); the other
//! CTAs return at once.
template <typename F>
__device__ __forceinline__ void rowArrive(RowWorkspace* w, int32_t G, F leader)
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
    if (!sLast)
    {
        return;
    }
    __threadfence();
    leader();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        atomicExch(&w->arrive, 0u);
        cuda::atomic_ref<unsigned, cuda::thread_scope_device>(w->gen).store(sGen + 1, cuda::memory_order_release);
    }
}

__global__ void __launch_bounds__(kBlockSize, 1) chainSpecSamplingMcKernelV3(
    ChainSpeculativeSamplingParams params, RowWorkspace* ws, ChainSpecMcDraftRows rows, int32_t chunkK)
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

    //! [chunkK x chainsPerCta] float4 of the target, then the same of the draft.
    extern __shared__ float4 sStage[];

    __shared__ int32_t sPos;
    __shared__ float sU;
    __shared__ int32_t sTileLastOfTile;
    __shared__ int32_t sDraftToken[kChainSpecMcMaxDraft];

    int64_t const row = blockIdx.y;
    int32_t const G = static_cast<int32_t>(gridDim.x);
    int32_t const cta = static_cast<int32_t>(blockIdx.x);
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    int32_t const numDraft = params.numDraftTokens;
    uint32_t const vocabSize = static_cast<uint32_t>(params.vocabSize);
    RowWorkspace* w = ws + row;
    int32_t* outTokens = params.outputTokenIds + row * (numDraft + 1);

    // 1. Acceptance: v2's (the stock tid-0 loop over probabilities loaded in parallel); the draft ids stay in shared
    //    memory for the outputs.
    __shared__ float sPTarget[kChainSpecMcMaxDraft];
    __shared__ float sPDraft[kChainSpecMcMaxDraft];
    if (tid < numDraft)
    {
        int32_t const token = params.draftTokenIds[row * numDraft + tid];
        sDraftToken[tid] = token;
        sPTarget[tid] = params.targetProbs[(row * (numDraft + 1) + tid) * vocabSize + token];
        sPDraft[tid] = draftRow(params, rows, row, tid, numDraft, vocabSize)[token];
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
    float const* draft = pos < numDraft ? draftRow(params, rows, row, pos, numDraft, vocabSize) : nullptr;
    int32_t const chainsPerCta = kBlockSize / G;
    int32_t const numTiles = static_cast<int32_t>((vocabSize + kTileSize - 1) / kTileSize);

    // B (loads): the CTA's first kTileRegs tiles (tiles cta, cta + G, ...), issued with A's staged loads.
    float tileTarget[kTileRegs][kVecSize];
    float tileDraft[kTileRegs][kVecSize];
    auto loadTiles = [&](int32_t tile0)
    {
#pragma unroll
        for (int32_t r = 0; r < kTileRegs; ++r)
        {
            int32_t const tile = tile0 + r * G;
#pragma unroll
            for (int32_t j = 0; j < kVecSize; ++j)
            {
                tileTarget[r][j] = 0.0f;
                tileDraft[r][j] = 0.0f;
            }
            if (tile < numTiles)
            {
                uint32_t const base = tile * kTileSize + tid * kVecSize;
                loadProbs<kVecSize>(target, base, vocabSize, tileTarget[r]);
                if (draft != nullptr)
                {
                    loadProbs<kVecSize>(draft, base, vocabSize, tileDraft[r]);
                }
            }
        }
    };
    loadTiles(cta);

    // 2 (A). The stock chains: chain t adds the elements at 4t + 4096k in k order (x: target mass, y: residual mass).
    //    The CTA's chains are t = cta * chainsPerCta + c; all threads stage their vec4s, chunkK strides at a time, and
    //    thread c < chainsPerCta adds them in the stock order (a stride past the vocab ends the chain, as the stock
    //    loop's bound; the vocab is a multiple of 4, so a staged vec4 is whole or absent).
    float4* const sTarget = sStage;
    float4* const sDraft = sStage + chunkK * chainsPerCta;
    int32_t const numK = numTiles;
    float2 sums = make_float2(0.0f, 0.0f);
    for (int32_t k0 = 0; k0 < numK; k0 += chunkK)
    {
        int32_t const kc = min(chunkK, numK - k0);
        for (int32_t i = tid; i < kc * chainsPerCta; i += kBlockSize)
        {
            int32_t const k = k0 + i / chainsPerCta;
            int32_t const t = cta * chainsPerCta + i % chainsPerCta;
            uint32_t const base = static_cast<uint32_t>(t) * kVecSize + static_cast<uint32_t>(k) * kTileSize;
            float4 tv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 dv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (base < vocabSize)
            {
                tv = *reinterpret_cast<float4 const*>(target + base);
                if (draft != nullptr)
                {
                    dv = *reinterpret_cast<float4 const*>(draft + base);
                }
            }
            sTarget[i] = tv;
            sDraft[i] = dv;
        }
        __syncthreads();
        if (tid < chainsPerCta)
        {
            uint32_t const t = static_cast<uint32_t>(cta * chainsPerCta + tid);
            for (int32_t kk = 0; kk < kc; ++kk)
            {
                if (t * kVecSize + static_cast<uint32_t>(k0 + kk) * kTileSize >= vocabSize)
                {
                    break;
                }
                float4 const a = sTarget[kk * chainsPerCta + tid];
                float tv[kVecSize] = {a.x, a.y, a.z, a.w};
                float dv[kVecSize] = {};
                if (draft != nullptr)
                {
                    float4 const b = sDraft[kk * chainsPerCta + tid];
                    dv[0] = b.x;
                    dv[1] = b.y;
                    dv[2] = b.z;
                    dv[3] = b.w;
                }
#pragma unroll
                for (int32_t j = 0; j < kVecSize; ++j)
                {
                    sums.x += tv[j];
                    sums.y += fmaxf(tv[j] - dv[j], 0.0f);
                }
            }
        }
        __syncthreads();
    }
    if (tid < chainsPerCta)
    {
        w->chain[cta * chainsPerCta + tid] = sums;
    }

    // 3 (B). Both stock tile variants of the CTA's tiles: the stock per-tile cub reduce and last index with mass over
    // the
    //    target (targetOnly) and over max(0, target - draft) (the residual; only when there is a draft row).
    auto tileSums = [&](int32_t tile, float(&value)[kVecSize], float* sumOut, int32_t* lastOut)
    {
        if (tid == 0)
        {
            sTileLastOfTile = -1;
        }
        __syncthreads();
        uint32_t const base = tile * kTileSize + tid * kVecSize;
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
            atomicMax(&sTileLastOfTile, localLast);
        }
        float const tileSum = BlockReduce(tempStorage.reduce).Sum(value);
        __syncthreads();
        if (tid == 0)
        {
            sumOut[tile] = tileSum;
            lastOut[tile] = sTileLastOfTile;
        }
        __syncthreads();
    };
    for (int32_t tile0 = cta; tile0 < numTiles; tile0 += kTileRegs * G)
    {
        if (tile0 != cta)
        {
            loadTiles(tile0);
        }
#pragma unroll
        for (int32_t r = 0; r < kTileRegs; ++r)
        {
            int32_t const tile = tile0 + r * G;
            if (tile >= numTiles)
            {
                break;
            }
            tileSums(tile, tileTarget[r], w->tileSumTarget, w->tileLastTarget);
            if (draft != nullptr)
            {
                float residual[kVecSize];
#pragma unroll
                for (int32_t j = 0; j < kVecSize; ++j)
                {
                    residual[j] = fmaxf(tileTarget[r][j] - tileDraft[r][j], 0.0f);
                }
                tileSums(tile, residual, w->tileSum, w->tileLast);
            }
        }
    }

    // 4. The last CTA to arrive: the chains' reduce, the tile prefix in tile order, the stock in-tile scan, the
    // outputs.
    rowArrive(w, G,
        [&]
        {
            __shared__ int32_t sTargetOnly;
            __shared__ float sTotal;
            __shared__ float sTileSum[kMaxTiles];
            __shared__ int32_t sTileLastOf[kMaxTiles];
            __shared__ float sPrefix;
            __shared__ int32_t sTile;
            __shared__ int32_t sWinner;
            __shared__ int32_t sLastValid;
            __shared__ int32_t sToken;
            // Both variants' per-tile values, loaded with the chains for the reduce.
            float sumTarget = 0.0f;
            float sumResidual = 0.0f;
            int32_t lastTarget = -1;
            int32_t lastResidual = -1;
            if (tid < numTiles)
            {
                sumTarget = __ldcg(&w->tileSumTarget[tid]);
                lastTarget = __ldcg(&w->tileLastTarget[tid]);
                if (draft != nullptr)
                {
                    sumResidual = __ldcg(&w->tileSum[tid]);
                    lastResidual = __ldcg(&w->tileLast[tid]);
                }
            }
            float2 const blockSums
                = BlockReducePair(tempStorage.reducePair).Reduce(__ldcg(&w->chain[tid]), Float2SumOp{});
            if (tid == 0)
            {
                int32_t const targetOnly = draft == nullptr || !(blockSums.y > 0.0f);
                sTargetOnly = targetOnly;
                sTotal = targetOnly ? blockSums.x : blockSums.y;
                sWinner = static_cast<int32_t>(vocabSize);
                sLastValid = -1;
            }
            __syncthreads();
            bool const targetOnly = sTargetOnly != 0;
            if (tid < numTiles)
            {
                sTileSum[tid] = targetOnly ? sumTarget : sumResidual;
                sTileLastOf[tid] = targetOnly ? lastTarget : lastResidual;
            }
            __syncthreads();
            float const threshold = sU * sTotal;
            if (tid == 0)
            {
                // The stock walk: the first tile whose prefix + sum crosses the threshold, prefix summed in tile order.
                float prefix = 0.0f;
                int32_t found = -1;
#pragma unroll
                for (int32_t tile = 0; tile < kMaxTiles; ++tile)
                {
                    if (tile < numTiles && found < 0)
                    {
                        float const tileSum = sTileSum[tile];
                        if (prefix + tileSum > threshold)
                        {
                            found = tile;
                        }
                        else
                        {
                            prefix += tileSum;
                        }
                    }
                }
                sTile = found;
                sPrefix = prefix;
            }
            __syncthreads();
            // The fallback's last index with mass over the tiles the stock loop visits (a max, in any order).
            int32_t const upto = sTile >= 0 ? sTile : numTiles - 1;
            if (tid <= upto && sTileLastOf[tid] >= 0)
            {
                atomicMax(&sLastValid, sTileLastOf[tid]);
            }
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
                sToken = sWinner < static_cast<int32_t>(vocabSize) ? sWinner : max(sLastValid, 0);
                params.outputNumTokens[row] = pos + 1;
            }
            __syncthreads();
            if (tid <= numDraft)
            {
                outTokens[tid] = tid < pos ? sDraftToken[tid] : (tid == pos ? sToken : -1);
            }
        });
}

//! REJ v3 on sm_107 devices unless TRTLLM_M31_DISABLE_REJ_V3=1 (the v2 kernel); other devices launch v2. The switch
//! is read once per process and the compute capability once per device.
bool rejV3Enabled()
{
    static bool const allowed = []
    {
        char const* env = std::getenv("TRTLLM_M31_DISABLE_REJ_V3");
        return env == nullptr || std::string(env) != "1";
    }();
    if (!allowed)
    {
        return false;
    }
    static std::once_flag smOnce[64];
    static bool sm107[64] = {};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess)
    {
        return false;
    }
    std::call_once(smOnce[device % 64],
        [device]
        {
            int major = 0;
            int minor = 0;
            cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
            cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
            sm107[device % 64] = major == 10 && minor == 7;
        });
    return sm107[device % 64];
}

} // namespace

cudaError_t invokeChainSpecSamplingMc(ChainSpeculativeSamplingParams const& params, int ctasPerRow, void* workspace,
    cudaStream_t stream, ChainSpecMcDraftRows const& rows)
{
    if (params.batchSize <= 0)
    {
        return cudaSuccess;
    }
    if (ctasPerRow < 1 || ctasPerRow > kBlockSize || kBlockSize % ctasPerRow != 0 || params.vocabSize <= 0
        || params.vocabSize % kVecSize != 0 || params.vocabSize > kChainSpecMcMaxVocab || params.numDraftTokens < 0
        || params.numDraftTokens > kChainSpecMcMaxDraft || params.batchSize > 65535
        || (rows.rowIds != nullptr
            && (rows.posStride < params.vocabSize || rows.rowStride < rows.posStride * params.numDraftTokens)))
    {
        return cudaErrorInvalidValue;
    }
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(static_cast<unsigned>(ctasPerRow), static_cast<unsigned>(params.batchSize));
    cfg.blockDim = dim3(kBlockSize);
    cfg.dynamicSmemBytes = 0;
    cfg.stream = stream;
    if (!rejV3Enabled())
    {
        return cudaLaunchKernelEx(&cfg, chainSpecSamplingMcKernel, params, static_cast<RowWorkspace*>(workspace), rows);
    }
    // v3 stages chunkK 4096-strides of the CTA's chains per pass (all of them at 16+ CTAs per request).
    int32_t const chainsPerCta = kBlockSize / ctasPerRow;
    int32_t const numK = (params.vocabSize + kTileSize - 1) / kTileSize;
    int32_t const chunkK = std::min(numK, std::max(1, kStageVecsPerRow / chainsPerCta));
    constexpr int kMaxStageBytes = 2 * kStageVecsPerRow * static_cast<int>(sizeof(float4));
    // The attribute is per device; set it on each device's first launch.
    static std::once_flag smemOnce[64];
    int device = 0;
    if (cudaError_t const err = cudaGetDevice(&device); err != cudaSuccess)
    {
        return err;
    }
    cudaError_t attrErr = cudaSuccess;
    std::call_once(smemOnce[device % 64],
        [&]
        {
            attrErr = cudaFuncSetAttribute(
                chainSpecSamplingMcKernelV3, cudaFuncAttributeMaxDynamicSharedMemorySize, kMaxStageBytes);
        });
    if (attrErr != cudaSuccess)
    {
        return attrErr;
    }
    cfg.dynamicSmemBytes = 2 * static_cast<size_t>(chunkK) * chainsPerCta * sizeof(float4);
    return cudaLaunchKernelEx(
        &cfg, chainSpecSamplingMcKernelV3, params, static_cast<RowWorkspace*>(workspace), rows, chunkK);
}

size_t chainSpecSamplingMcWorkspaceBytesPerRow()
{
    return sizeof(RowWorkspace);
}

int chainSpecSamplingMcKernelVersion()
{
    return rejV3Enabled() ? 3 : 2;
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
