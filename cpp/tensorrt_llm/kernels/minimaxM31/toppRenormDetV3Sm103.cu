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

// TOPP_DET v3: toppRenormDetSm103.cu (v1.2) with the redundant selection and an optional programmatic dependent
// launch; V4K8: the one-sync list path and 8 privatized pass histograms (see the header). See toppRenormDetV3Sm103.h
// for the contract (bitwise v1.2's output). Compiled with --use_fast_math (set on this source in
// kernels/CMakeLists.txt), as FlashInfer's JIT compiles it.

#include "tensorrt_llm/kernels/minimaxM31/toppRenormDetV3Sm103.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/atomic>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <strings.h>

#include <cub/cub.cuh>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

namespace toppv3
{

// ---------------------------------------------------------------------------------------------------------------------
// Vendored from FlashInfer, include/flashinfer/air_top_p.cuh (FlashInfer v0.6.15; identical in v0.6.18), namespace
// flashinfer::sampling::air_top_p. Copyright (c) 2026 by FlashInfer team, licensed under the Apache License,
// Version 2.0. Copied verbatim (only the pieces this kernel uses; formatting kept) so that the bucket keys, masks
// and the deterministic histogram -> float conversion are FlashInfer's own; FlashInfer's headers are not on the
// TensorRT-LLM C++ include path.
// ---------------------------------------------------------------------------------------------------------------------
namespace fi_air_top_p
{
// clang-format off
static constexpr int BITS_PER_PASS = 11;
static constexpr int NUM_BUCKETS = 1 << BITS_PER_PASS;

template <typename T>
static constexpr int NUM_PASSES = (sizeof(T) * 8 + BITS_PER_PASS - 1) / BITS_PER_PASS;

template <typename T>
__device__ int constexpr calcStartBit(int pass) {
  int startBit = static_cast<int>(sizeof(T) * 8) - (pass + 1) * BITS_PER_PASS;
  return startBit < 0 ? 0 : startBit;
}

template <typename T>
__device__ uint32_t constexpr calcMask(int pass) {
  int numBits = calcStartBit<T>(pass - 1) - calcStartBit<T>(pass);
  return (1 << numBits) - 1;
}

template <typename T>
__device__ typename cub::Traits<T>::UnsignedBits twiddleIn(T key, bool selectMin) {
  auto bits = reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(key);
  bits = cub::Traits<T>::TwiddleIn(bits);
  if (!selectMin) bits = ~bits;
  return bits;
}

template <typename T>
__device__ T twiddleOut(typename cub::Traits<T>::UnsignedBits bits, bool selectMin) {
  if (!selectMin) bits = ~bits;
  bits = cub::Traits<T>::TwiddleOut(bits);
  return reinterpret_cast<T&>(bits);
}

__device__ inline uint32_t calcMantissa(float value) {
  union {
    uint32_t bits;
    float value;
  } input;
  input.value = value;
  constexpr uint32_t numMantissa = 23;
  return input.bits & ((1u << numMantissa) - 1);
}

__device__ inline uint32_t calcExponent(float value) {
  union {
    uint32_t bits;
    float value;
  } input;
  input.value = value;
  constexpr uint32_t numMantissa = 23;
  return input.bits & ~((1u << numMantissa) - 1);
}

__device__ inline float calcFloatValue(uint32_t count, uint32_t exponent, uint64_t bitSum) {
  constexpr uint32_t numTotalBits = 64;
  constexpr uint32_t numMantissa = 23;
  uint64_t extraInMantissa = (bitSum >> numMantissa);
  extraInMantissa = (exponent == 0) ? extraInMantissa : extraInMantissa + count;
  uint32_t numExtra = numTotalBits - __clzll(extraInMantissa);
  int numNorm = (exponent == 0) ? 0 : -1;
  exponent = exponent + ((numExtra + numNorm) << numMantissa);
  uint32_t mantissa;
  if (extraInMantissa != 0) {
    int numMove = numMantissa - (numExtra - 1);
    uint32_t mask = (1u << (numExtra - 1)) - 1;
    extraInMantissa = extraInMantissa & mask;
    if (numMove > 0) {
      extraInMantissa = extraInMantissa << numMove;
      mask = (1u << numMantissa) - 1;
      mantissa = ((bitSum & mask) >> (numExtra - 1)) | extraInMantissa;
    } else {
      mantissa = extraInMantissa >> (-1 * numMove);
    }
  } else {
    mantissa = bitSum;
  }
  uint32_t bitFloat = exponent | mantissa;
  return reinterpret_cast<float&>(bitFloat);
}

// clang-format on
} // namespace fi_air_top_p

// ---------------------------------------------------------------------------------------------------------------------
// End of the vendored FlashInfer code.
// ---------------------------------------------------------------------------------------------------------------------

namespace fa = fi_air_top_p;
namespace cg = cooperative_groups;

constexpr int kThreads = 1024;
constexpr int kBuckets = fa::NUM_BUCKETS; // 2048
constexpr int kSmemBytes = 32768;

// Privatized pass histograms per CTA: kCopies (a kernel template parameter) copies, one per group of warps; 8 (V4K8,
// default) or 4 (TRTLLM_M31_DISABLE_TOPPDET_K8).
template <int kCopies>
constexpr size_t dynSmemBytes()
{
    return kCopies * kBuckets * (sizeof(unsigned long long) + sizeof(int));
}

constexpr int kSyncs = fa::NUM_PASSES<float> + 1; // one per radix pass, then the norm
// Candidate list: passes 1 .. NUM_PASSES - 1 run locally on the members of pass 0's threshold bucket when it holds at
// most this many elements (the list lives in the workspace, then in the dynamic shared memory of each CTA).
constexpr int kCandMax = 4096;
// V4 list path: pass 0 also lists the elements >= 2^-12 (a pass-0 bucket edge; probabilities summing to 1 hold at most
// 4096 of them) with their indices; pass-0 histograms and lists are double-buffered by launch parity.
constexpr int kListMax = 4096;
constexpr float kListMin = 0.000244140625f; // 2^-12

// One row's persistent state. Zero once; every launch leaves it zeroed except the arrival counters, which only grow.
struct RowWorkspace
{
    unsigned long long hist[fa::NUM_PASSES<float>][kBuckets]; // per pass: det mantissa-sum histogram (uint64 HisT)
    int cnt[fa::NUM_PASSES<float>][kBuckets];                 // per pass: count histogram
    float chain[kThreads];                                    // the 1024 per-thread chain sums of the kept values
    unsigned long long arrive[kSyncs];                        // per sync: CTAs arrived, monotone across launches
    float cand[kCandMax];                                     // the candidate list (pass 0's bucket members)
    unsigned nCand;                                           // its length, zero between launches
    int kept;
    int pad[2];
    unsigned long long hist0x[2][kBuckets];
    int cnt0x[2][kBuckets];
    float lval[2][kListMax];
    int lidx[2][kListMax];
    unsigned nList[2];
    unsigned pad2[2];
};

// Arrive at sync `si` and wait until all G CTAs of the row have arrived. The counter is never reset: each CTA adds
// kThreads / G (G divides kThreads), so every launch adds exactly kThreads per sync it runs whatever its G (a row's
// CTAs all take the same branch) and the counter is a multiple of kThreads between launches; the target is the next
// one.
__device__ __forceinline__ void allArrive(RowWorkspace* w, int G, int si)
{
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        unsigned long long const step = static_cast<unsigned long long>(kThreads / G);
        unsigned long long const old = atomicAdd(&w->arrive[si], step);
        unsigned long long const target = (old / kThreads + 1) * kThreads;
        cuda::atomic_ref<unsigned long long, cuda::thread_scope_device> ref(w->arrive[si]);
        while (ref.load(cuda::memory_order_acquire) < target)
        {
            __nanosleep(20);
        }
    }
    __syncthreads();
}

// FlashInfer's last-block selection for `pass` (air_top_p.cuh AirTopPRenormRadixKernel, IsDeterministic branch), on
// the row's global integer histograms of this pass, run by EVERY CTA of the row (identical inputs, identical code, so
// identical bits). The running sum and key prefix are CTA-local (shared memory) instead of the workspace. histVal and
// cntS (the bucket counts, cached for the sequential scan) reuse the CTA's scratch.
//
// kGlobal: the histograms are the row's global ones (read through L2); otherwise this CTA's shared-memory histograms of
// the candidate list, which hold the same integers.
template <bool kGlobal>
__device__ __forceinline__ void selectPassOn(unsigned long long const* hist, int const* cnt, int pass, float p,
    float* histVal, int* cntS, float* sSum, unsigned* sKbits)
{
    constexpr int WARP_SIZE = 32;
    constexpr int WARP_COUNT = kBuckets / WARP_SIZE;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float warpSum[WARP_COUNT];
    __shared__ cuda::atomic<float, cuda::thread_scope_block> blockSum;
    for (int i = threadIdx.x; i < kBuckets; i += blockDim.x)
    {
        unsigned long long value = kGlobal ? __ldcg(&hist[i]) : hist[i];
        int count = kGlobal ? __ldcg(&cnt[i]) : cnt[i];
        cntS[i] = count;
        if (count != 0)
        {
            uint32_t sb = fa::calcStartBit<float>(pass);
            uint32_t bv = *sKbits;
            if (pass == 0)
            {
                bv = i << sb;
            }
            histVal[i] = fa::calcFloatValue(
                static_cast<uint32_t>(count), fa::calcExponent(fa::twiddleOut<float>(bv, false)), value);
        }
        else
        {
            histVal[i] = 0.0f;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kBuckets; i += blockDim.x)
    {
        cg::reduce_store_async(warp, warpSum + i / WARP_SIZE, histVal[i], cg::plus<float>{});
    }
    __syncthreads();
    if (threadIdx.x < WARP_SIZE)
    {
        cg::reduce_store_async(warp, blockSum, warpSum[threadIdx.x], cg::plus<float>{});
        cg::reduce_update_async(warp, blockSum, warpSum[threadIdx.x + WARP_SIZE], cg::plus<float>{});
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
        float currentSum = pass == 0 ? static_cast<float>(blockSum) * p : *sSum;
        float prev = 0;
        int targetStep = 0;
        for (int i = 0; i < WARP_COUNT; i++)
        {
            if (warpSum[i])
            {
                targetStep = i;
                if ((prev + warpSum[i]) >= currentSum)
                {
                    break;
                }
                prev += warpSum[i];
            }
        }
        int targetIdx = 0;
        for (int i = targetStep * WARP_SIZE; i < kBuckets; i++)
        {
            if (cntS[i])
            {
                targetIdx = i;
                if ((prev + histVal[i]) >= currentSum)
                {
                    break;
                }
                prev += histVal[i];
            }
        }
        *sSum = currentSum - prev;
        unsigned kb = pass == 0 ? 0u : *sKbits;
        *sKbits = kb | (static_cast<unsigned>(targetIdx) << fa::calcStartBit<float>(pass));
    }
    __syncthreads();
}

__device__ __forceinline__ void selectPass(
    RowWorkspace* w, int pass, float p, float* histVal, int* cntS, float* sSum, unsigned* sKbits)
{
    selectPassOn<true>(w->hist[pass], w->cnt[pass], pass, p, histVal, cntS, sSum, sKbits);
}

// CTA 0 clears pass `pass`'s global histograms once every CTA of the row has finished reading them.
__device__ __forceinline__ void clearPass(RowWorkspace* w, int pass)
{
    if (blockIdx.x == 0)
    {
        for (int i = threadIdx.x; i < kBuckets; i += blockDim.x)
        {
            w->hist[pass][i] = 0;
            w->cnt[pass][i] = 0;
        }
    }
}

// V4 list path, run by every CTA of the row after pass 0 (one barrier so far) when the pass-0 threshold bucket lies at
// or above 2^-12: then every element of that bucket and above, so every kept element, is in the row's list. The
// remaining radix passes run on the list (the same integer histograms as v3's candidate passes), FlashInfer's 1024
// chain sums are rebuilt from the kept members (chain c = the kept values at indices c, c + 1024, ... in ascending
// order, as v3's chains), reduced by the same cub block sum, and each CTA writes its own slice: zeros, then its kept
// values.
template <bool kPdl>
__device__ __noinline__ void listPath(RowWorkspace* w, int par, int nL, unsigned kbits, float p, float const* in,
    float* o, int d, int row, int cta, int G, int* keptOut, unsigned long long* hsCopies, int* csCopies,
    unsigned char* smem, float* sSum, unsigned* sKbits)
{
    int const tid = threadIdx.x;
    float* lv = reinterpret_cast<float*>(hsCopies + kBuckets); // 16 KB
    int* li = reinterpret_cast<int*>(hsCopies + 2 * kBuckets); // 16 KB
    for (int i = tid; i < nL; i += kThreads)
    {
        lv[i] = __ldcg(&w->lval[par][i]);
        li[i] = __ldcg(&w->lidx[par][i]);
    }
    __syncthreads();
    for (int lp = 1; lp < fa::NUM_PASSES<float>; ++lp)
    {
        for (int i = tid; i < kBuckets; i += kThreads)
        {
            hsCopies[i] = 0;
            csCopies[i] = 0;
        }
        __syncthreads();
        int const lsb = fa::calcStartBit<float>(lp);
        uint32_t const lmask = fa::calcMask<float>(lp);
        int const lpsb = fa::calcStartBit<float>(lp - 1);
        for (int i = tid; i < nL; i += kThreads)
        {
            float const x = lv[i];
            auto const bits = fa::twiddleIn(x, false);
            if (((bits >> lpsb) << lpsb) == kbits)
            {
                int const b = static_cast<int>((bits >> lsb) & lmask);
                atomicAdd(hsCopies + b, static_cast<unsigned long long>(fa::calcMantissa(x)));
                atomicAdd(csCopies + b, 1);
            }
        }
        __syncthreads();
        selectPassOn<false>(hsCopies, csCopies, lp, p, reinterpret_cast<float*>(smem),
            reinterpret_cast<int*>(smem + kBuckets * sizeof(float)), sSum, sKbits);
        kbits = *sKbits;
    }
    float const thr = fa::twiddleOut<float>(kbits, false);
    // Compact the kept members (csCopies is free now: 32 KB = 4096 values + 4096 indices).
    float* kv = reinterpret_cast<float*>(csCopies);
    int* ki = csCopies + kListMax;
    __shared__ int nK;
    if (tid == 0)
    {
        nK = 0;
    }
    __syncthreads();
    for (int i = tid; i < nL; i += kThreads)
    {
        if (lv[i] >= thr)
        {
            int const s = atomicAdd(&nK, 1);
            kv[s] = lv[i];
            ki[s] = li[i];
        }
    }
    __syncthreads();
    int const nKept = nK;
    // Chain tid: its kept values in ascending index order, accumulated from 0 as v3's chains.
    float acc = 0.0f;
    int last = -1;
    while (true)
    {
        int best = INT_MAX;
        float bv = 0.0f;
        for (int i = 0; i < nKept; ++i)
        {
            int const idx = ki[i];
            if ((idx & (kThreads - 1)) == tid && idx > last && idx < best)
            {
                best = idx;
                bv = kv[i];
            }
        }
        if (best == INT_MAX)
        {
            break;
        }
        acc += bv;
        last = best;
    }
    typedef cub::BlockReduce<float, 1024> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float sNormL;
    float const totalSum = BlockReduce(temp).Sum(acc);
    if (tid == 0)
    {
        sNormL = (totalSum > 1e-8f) ? (1.0f / totalSum) : 1.0f;
        if (cta == 0 && keptOut)
        {
            keptOut[row] = nKept;
        }
    }
    __syncthreads();
    if constexpr (kPdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
    float const norm = sNormL;
    float4* o4 = reinterpret_cast<float4*>(o);
    int const n4 = d / 4;
    for (int j = cta * kThreads + tid; j < n4; j += G * kThreads)
    {
        o4[j] = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    __syncthreads();
    for (int i = tid; i < nKept; i += kThreads)
    {
        int const idx = ki[i];
        if (((idx / 4) / kThreads) % G == cta)
        {
            o[idx] = kv[i] * norm;
        }
    }
}

template <bool kPdl, bool kCandList, bool kList, int kCopies, bool kChainPf>
__global__ void __launch_bounds__(kThreads, 1) toppRenormDetV3Kernel(
    float const* __restrict__ probs, float* __restrict__ out, float const* topP, int d, RowWorkspace* ws, int* keptOut)
{
    __shared__ __align__(16) unsigned char smem[kSmemBytes];
    // Privatized pass histograms: kCopies copies, one per group of warps (dynamic shared memory).
    extern __shared__ __align__(16) unsigned char dyn[];
    unsigned long long* hsCopies = reinterpret_cast<unsigned long long*>(dyn);
    int* csCopies = reinterpret_cast<int*>(dyn + kCopies * kBuckets * sizeof(unsigned long long));
    int const row = blockIdx.y;
    int const G = gridDim.x;
    int const cta = blockIdx.x;
    int const tid = threadIdx.x;
    int const lane = tid & 31;
    RowWorkspace* w = ws + row;
    float const* in = probs + static_cast<size_t>(row) * d;
    float* o = out + static_cast<size_t>(row) * d;
    __shared__ float sSum;
    __shared__ unsigned sKbits;
    if (tid == 0)
    {
        sSum = 0.0f;
        sKbits = 0u;
    }
    // Everything above is CTA-private; probs, top_p and the shared workspace are only touched after the wait.
    if constexpr (kPdl)
    {
        cudaGridDependencySynchronize();
    }
    __syncthreads();
    // V4: launch index from the grow-only pass-0 counter (every earlier launch has completed; this launch adds less
    // than kThreads until all of its CTAs arrive), so every CTA of the row gets the same parity.
    int const par = static_cast<int>((__ldcg(&w->arrive[0]) / kThreads) & 1ull);
    // Spread over the row's CTAs (each clears 2048 / G buckets), so no CTA's binning is delayed.
    for (int i = cta * kThreads + tid; i < kBuckets; i += G * kThreads)
    {
        w->hist0x[par ^ 1][i] = 0;
        w->cnt0x[par ^ 1][i] = 0;
    }
    if (cta == 0 && tid == 0)
    {
        w->nList[par ^ 1] = 0;
    }
    float const p = topP[row];
    float4 const* in4 = reinterpret_cast<float4 const*>(in);
    int const n4 = d / 4;
    int const stride4 = G * kThreads;
    int const start4 = cta * kThreads + tid;

    unsigned long long* hs = reinterpret_cast<unsigned long long*>(smem);           // 16 KB
    int* cs = reinterpret_cast<int*>(smem + kBuckets * sizeof(unsigned long long)); // 8 KB
    unsigned kbits = 0;

    // top_p >= 1 (the keep-all exception): no radix passes, the threshold is -inf.
    bool const keepAll = p >= 1.0f;
    for (int pass = 0; pass < (keepAll ? 0 : fa::NUM_PASSES<float>); ++pass)
    {
        for (int i = tid; i < kCopies * kBuckets; i += kThreads)
        {
            hsCopies[i] = 0;
            csCopies[i] = 0;
        }
        __syncthreads();
        int const copy = (tid >> 5) % kCopies;
        unsigned long long* myHs = hsCopies + copy * kBuckets;
        int* myCs = csCopies + copy * kBuckets;
        // S32 (with the 8 copies): the 64-bit mantissa-sum space holds two 32-bit sums per bucket instead (hi = m >>
        // 11, then lo = m & 0x7ff), two cheap shared atomics in place of a contended 64-bit one; the merge rebuilds the
        // exact sum (hi << 11) + lo. Per copy and bucket at most d / G elements, so both stay below 2^32 while d / G <
        // 2^20.
        bool const s32 = kCopies == 8 && d / G < (1 << 20);
        unsigned* myHi = reinterpret_cast<unsigned*>(hsCopies) + copy * kBuckets;
        unsigned* myLo = reinterpret_cast<unsigned*>(hsCopies) + kCopies * kBuckets + copy * kBuckets;
        int const sb = fa::calcStartBit<float>(pass);
        uint32_t const mask = fa::calcMask<float>(pass);
        int const psb = pass == 0 ? 0 : fa::calcStartBit<float>(pass - 1);
        // Warp-uniform trip count: every lane runs the same iterations (masked), so the warp collectives are
        // full-warp.
        for (int j0 = start4 - lane; j0 < n4; j0 += stride4)
        {
            int const j = j0 + lane;
            bool const valid = j < n4;
            float4 v = valid ? in4[j] : make_float4(0.f, 0.f, 0.f, 0.f);
            float const e[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                float const x = e[k];
                if (kList && pass == 0)
                {
                    // Warp-aggregated append: one global atomic per warp and element slot (the list's order is
                    // irrelevant: integer histograms, index-ordered chains).
                    bool const want = valid && x >= kListMin;
                    unsigned const m = __ballot_sync(0xffffffffu, want);
                    if (m)
                    {
                        int const leader = __ffs(m) - 1;
                        unsigned base = 0;
                        if (lane == leader)
                        {
                            base = atomicAdd(&w->nList[par], static_cast<unsigned>(__popc(m)));
                        }
                        base = __shfl_sync(0xffffffffu, base, leader);
                        if (want)
                        {
                            unsigned const slot = base + __popc(m & ((1u << lane) - 1u));
                            if (slot < kListMax)
                            {
                                w->lval[par][slot] = x;
                                w->lidx[par][slot] = 4 * j + k;
                            }
                        }
                    }
                }
                auto const bits = fa::twiddleIn(x, false);
                bool const ok = valid && (pass == 0 || ((bits >> psb) << psb) == kbits);
                int const b = ok ? static_cast<int>((bits >> sb) & mask) : -1;
                unsigned const m = ok ? fa::calcMantissa(x) : 0u;
                // Adaptive per warp (integer sums, so order-free and exact): a warp whose 32 lanes share one bucket
                // (near-uniform rows) adds it with ONE aggregated atomic; otherwise every eligible lane adds into its
                // warp group's private copy (few lanes collide on rows spread over many buckets).
                int const b0 = __shfl_sync(0xffffffffu, b, 0);
                if (__all_sync(0xffffffffu, b == b0))
                {
                    if (b0 >= 0)
                    {
                        if (s32)
                        {
                            unsigned const sh = __reduce_add_sync(0xffffffffu, m >> 11);
                            unsigned const sl = __reduce_add_sync(0xffffffffu, m & 0x7ffu);
                            if (lane == 0)
                            {
                                atomicAdd(myHi + b0, sh);
                                atomicAdd(myLo + b0, sl);
                                atomicAdd(myCs + b0, 32);
                            }
                        }
                        else
                        {
                            unsigned const s = __reduce_add_sync(0xffffffffu, m);
                            if (lane == 0)
                            {
                                atomicAdd(myHs + b0, static_cast<unsigned long long>(s));
                                atomicAdd(myCs + b0, 32);
                            }
                        }
                    }
                }
                else if (ok)
                {
                    if (s32)
                    {
                        atomicAdd(myHi + b, m >> 11);
                        atomicAdd(myLo + b, m & 0x7ffu);
                    }
                    else
                    {
                        atomicAdd(myHs + b, static_cast<unsigned long long>(m));
                    }
                    atomicAdd(myCs + b, 1);
                }
            }
        }
        __syncthreads();
        for (int i = tid; i < kBuckets; i += kThreads)
        {
            unsigned long long h = 0;
            int c = 0;
            unsigned const* hiS = reinterpret_cast<unsigned const*>(hsCopies);
#pragma unroll
            for (int k = 0; k < kCopies; ++k)
            {
                h += s32 ? (static_cast<unsigned long long>(hiS[k * kBuckets + i]) << 11)
                        + hiS[kCopies * kBuckets + k * kBuckets + i]
                         : hsCopies[k * kBuckets + i];
                c += csCopies[k * kBuckets + i];
            }
            hs[i] = h;
            cs[i] = c;
        }
        __syncthreads();
        for (int i = tid; i < kBuckets; i += kThreads)
        {
            if (hs[i] != 0)
            {
                atomicAdd(pass == 0 ? &w->hist0x[par][i] : &w->hist[pass][i], hs[i]);
            }
            if (cs[i] != 0)
            {
                atomicAdd(pass == 0 ? &w->cnt0x[par][i] : &w->cnt[pass][i], cs[i]);
            }
        }
        allArrive(w, G, pass);
        if (pass > 1)
        {
            clearPass(w, pass - 1); // every CTA finished selectPass(pass - 1) before arriving here
        }
        if (pass == 0)
        {
            selectPassOn<true>(w->hist0x[par], w->cnt0x[par], 0, p, reinterpret_cast<float*>(smem),
                reinterpret_cast<int*>(smem + kBuckets * sizeof(float)), &sSum, &sKbits);
        }
        else
        {
            selectPass(w, pass, p, reinterpret_cast<float*>(smem),
                reinterpret_cast<int*>(smem + kBuckets * sizeof(float)), &sSum, &sKbits);
        }
        kbits = sKbits;
        if (kList && pass == 0)
        {
            unsigned const nL = __ldcg(&w->nList[par]);
            float const lowEdge = fa::twiddleOut<float>(kbits | ((1u << fa::calcStartBit<float>(0)) - 1u), false);
            if (nL <= static_cast<unsigned>(kListMax) && lowEdge >= kListMin)
            {
                listPath<kPdl>(w, par, static_cast<int>(nL), kbits, p, in, o, d, row, cta, G, keptOut, hsCopies,
                    csCopies, smem, &sSum, &sKbits);
                return;
            }
        }
        if constexpr (kCandList)
        {
            // Pass 0's threshold bucket count (cached by selectPass) is the same in every CTA of the row, so all of
            // them take the same branch. A small bucket: list its members once, then finish the radix search locally.
            int const sb0 = fa::calcStartBit<float>(0);
            int const nCand = reinterpret_cast<int const*>(smem + kBuckets * sizeof(float))[kbits >> sb0];
            if (pass == 0 && nCand <= kCandMax)
            {
                for (int j0 = start4 - lane; j0 < n4; j0 += stride4)
                {
                    int const j = j0 + lane;
                    if (j < n4)
                    {
                        float4 const v = in4[j];
                        float const e[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                        for (int k = 0; k < 4; ++k)
                        {
                            if (((fa::twiddleIn(e[k], false) >> sb0) << sb0) == kbits)
                            {
                                w->cand[atomicAdd(&w->nCand, 1u)] = e[k];
                            }
                        }
                    }
                }
                allArrive(w, G, 1);
                // The list in this CTA's dynamic shared memory (privatized copy 1's space), the local histograms in
                // copy 0's: the same integer histograms the global passes 1 .. NUM_PASSES - 1 would build.
                float* candS = reinterpret_cast<float*>(hsCopies + kBuckets);
                for (int i = tid; i < nCand; i += kThreads)
                {
                    candS[i] = __ldcg(&w->cand[i]);
                }
                __syncthreads();
                for (int lp = 1; lp < fa::NUM_PASSES<float>; ++lp)
                {
                    for (int i = tid; i < kBuckets; i += kThreads)
                    {
                        hsCopies[i] = 0;
                        csCopies[i] = 0;
                    }
                    __syncthreads();
                    int const lsb = fa::calcStartBit<float>(lp);
                    uint32_t const lmask = fa::calcMask<float>(lp);
                    int const lpsb = fa::calcStartBit<float>(lp - 1);
                    for (int i = tid; i < nCand; i += kThreads)
                    {
                        float const x = candS[i];
                        auto const bits = fa::twiddleIn(x, false);
                        if (((bits >> lpsb) << lpsb) == kbits)
                        {
                            int const b = static_cast<int>((bits >> lsb) & lmask);
                            atomicAdd(hsCopies + b, static_cast<unsigned long long>(fa::calcMantissa(x)));
                            atomicAdd(csCopies + b, 1);
                        }
                    }
                    __syncthreads();
                    selectPassOn<false>(hsCopies, csCopies, lp, p, reinterpret_cast<float*>(smem),
                        reinterpret_cast<int*>(smem + kBuckets * sizeof(float)), &sSum, &sKbits);
                    kbits = sKbits;
                }
                break;
            }
        }
    }

    // Apply: the threshold, FlashInfer's per-thread chain sums (spread over the G CTAs), the cub block sum, then the
    // renorm write.
    float const thr = keepAll ? -INFINITY : fa::twiddleOut<float>(kbits, false);
    int const cpc = kThreads / G; // chains per CTA
    int const c0 = cta * cpc;
    int const kTile = (kSmemBytes / 4) / cpc;
    int const kTotal = (d + kThreads - 1) / kThreads;
    float* stage = reinterpret_cast<float*>(smem);
    float acc = 0.0f;
    int kept = 0;
    // TOPP_V6 (kChainPf): each chain loads its own values straight from global memory, kPf loads in flight, and adds
    // them in the same ascending-k order as the staged loop (the same values, the same fp32 adds and kept counts, so
    // the same bits); no shared staging and no per-tile barriers. Kill TRTLLM_M31_DISABLE_TOPP_V6=1 (the staged loop).
    if constexpr (kChainPf)
    {
        constexpr int kPf = 16;
        if (tid < cpc)
        {
            int const cbase = c0 + tid;
            int k = 0;
            for (; k + kPf <= kTotal; k += kPf)
            {
                float v[kPf];
#pragma unroll
                for (int u = 0; u < kPf; ++u)
                {
                    int const i = cbase + kThreads * (k + u);
                    v[u] = i < d ? __ldg(in + i) : 0.0f;
                }
#pragma unroll
                for (int u = 0; u < kPf; ++u)
                {
                    int const i = cbase + kThreads * (k + u);
                    if (i < d && v[u] >= thr)
                    {
                        acc += v[u];
                        ++kept;
                    }
                }
            }
            for (; k < kTotal; ++k)
            {
                int const i = cbase + kThreads * k;
                float const val = i < d ? __ldg(in + i) : 0.0f;
                if (i < d && val >= thr)
                {
                    acc += val;
                    ++kept;
                }
            }
        }
    }
    for (int k0 = 0; !kChainPf && k0 < kTotal; k0 += kTile)
    {
        int const kn = min(kTile, kTotal - k0);
        for (int q = tid; q < kn * cpc; q += kThreads)
        {
            int const kk = q / cpc, c = q - kk * cpc;
            int const i = c0 + c + kThreads * (k0 + kk);
            stage[q] = i < d ? in[i] : 0.0f;
        }
        __syncthreads();
        if (tid < cpc)
        {
            for (int kk = 0; kk < kn; ++kk)
            {
                int const i = c0 + tid + kThreads * (k0 + kk);
                float const val = stage[kk * cpc + tid];
                if (i < d && val >= thr)
                {
                    acc += val;
                    ++kept;
                }
            }
        }
        __syncthreads();
    }
    if (tid < cpc)
    {
        w->chain[c0 + tid] = acc;
    }
    typedef cub::BlockReduce<float, 1024> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ int ctaKept;
    if (tid == 0)
    {
        ctaKept = 0;
    }
    __syncthreads();
    if (tid < cpc && kept)
    {
        atomicAdd(&ctaKept, kept);
    }
    __syncthreads();
    if (tid == 0 && ctaKept)
    {
        atomicAdd(&w->kept, ctaKept);
    }
    allArrive(w, G, fa::NUM_PASSES<float>);
    if (!keepAll)
    {
        clearPass(w, fa::NUM_PASSES<float> - 1); // every CTA finished the last selectPass before arriving here
    }
    __shared__ float sNorm;
    {
        float totalSum = BlockReduce(temp).Sum(__ldcg(&w->chain[tid]));
        if (tid == 0)
        {
            sNorm = (totalSum > 1e-8f) ? (1.0f / totalSum) : 1.0f;
        }
    }
    __syncthreads();
    if (cta == 0 && tid == 0)
    {
        if (keptOut)
        {
            keptOut[row] = __ldcg(&w->kept);
        }
        w->kept = 0; // only CTA 0 reads it; the next launch adds to it after its own grid-dependency wait
        if constexpr (kCandList)
        {
            w->nCand = 0; // every CTA copied the list before arriving at the norm barrier
        }
    }
    // The workspace is left ready for the next launch here (the last clear above), so dependents may start; their
    // grid-dependency wait still covers this launch's renorm write.
    if constexpr (kPdl)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
    float const norm = sNorm;
    float4* o4 = reinterpret_cast<float4*>(o);
    for (int j = start4; j < n4; j += stride4)
    {
        float4 v = in4[j];
        float4 r;
        r.x = (v.x >= thr) ? v.x * norm : 0.0f;
        r.y = (v.y >= thr) ? v.y * norm : 0.0f;
        r.z = (v.z >= thr) ? v.z * norm : 0.0f;
        r.w = (v.w >= thr) ? v.w * norm : 0.0f;
        o4[j] = r;
    }
}

} // namespace toppv3

namespace
{
// TOPP_V6: the chain-prefetch summation on sm_107 devices unless TRTLLM_M31_DISABLE_TOPP_V6 is set to 1 / true / on /
// yes; other devices run the staged loop. The switch is read once per process and the compute capability once per
// device.
bool chainPrefetchOn()
{
    static bool const allowed = []
    {
        char const* v = std::getenv("TRTLLM_M31_DISABLE_TOPP_V6");
        if (v == nullptr)
        {
            return true;
        }
        return !(std::strcmp(v, "1") == 0 || strcasecmp(v, "true") == 0 || strcasecmp(v, "on") == 0
            || strcasecmp(v, "yes") == 0);
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

template <bool kPdl, bool kCandList, bool kList, int kCopies, bool kChainPf>
cudaError_t launchV3Kernel(cudaLaunchConfig_t& cfg, float const* probs, float* out, float const* topP, int d,
    toppv3::RowWorkspace* ws, int* keptOut)
{
    auto* kernel = toppv3::toppRenormDetV3Kernel<kPdl, kCandList, kList, kCopies, kChainPf>;
    static cudaError_t const attrErr = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(toppv3::dynSmemBytes<kCopies>()));
    if (attrErr != cudaSuccess)
    {
        return attrErr;
    }
    cfg.dynamicSmemBytes = toppv3::dynSmemBytes<kCopies>();
    return cudaLaunchKernelEx(&cfg, kernel, probs, out, topP, d, ws, keptOut);
}

template <bool kPdl, bool kCandList, bool kList, int kCopies>
cudaError_t launchV3(cudaLaunchConfig_t& cfg, float const* probs, float* out, float const* topP, int d,
    toppv3::RowWorkspace* ws, int* keptOut)
{
    return chainPrefetchOn()
        ? launchV3Kernel<kPdl, kCandList, kList, kCopies, true>(cfg, probs, out, topP, d, ws, keptOut)
        : launchV3Kernel<kPdl, kCandList, kList, kCopies, false>(cfg, probs, out, topP, d, ws, keptOut);
}

template <bool kPdl, bool kCandList>
cudaError_t launchV3Copies(cudaLaunchConfig_t& cfg, bool listPath, bool wideHist, float const* probs, float* out,
    float const* topP, int d, toppv3::RowWorkspace* ws, int* keptOut)
{
    if (listPath)
    {
        return wideHist ? launchV3<kPdl, kCandList, true, 8>(cfg, probs, out, topP, d, ws, keptOut)
                        : launchV3<kPdl, kCandList, true, 4>(cfg, probs, out, topP, d, ws, keptOut);
    }
    return wideHist ? launchV3<kPdl, kCandList, false, 8>(cfg, probs, out, topP, d, ws, keptOut)
                    : launchV3<kPdl, kCandList, false, 4>(cfg, probs, out, topP, d, ws, keptOut);
}
} // namespace

cudaError_t invokeToppRenormDetV3(float const* probs, float* out, float const* topP, int rows, int d, int ctasPerRow,
    void* workspace, int* keptOut, bool pdl, bool candidateList, bool listPath, bool wideHist, cudaStream_t stream)
{
    using namespace toppv3;
    if (rows <= 0 || d <= 0)
    {
        return cudaSuccess;
    }
    if (ctasPerRow < 1 || ctasPerRow > kThreads || kThreads % ctasPerRow != 0 || d % 4 != 0)
    {
        return cudaErrorInvalidValue;
    }
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(static_cast<unsigned>(ctasPerRow), static_cast<unsigned>(rows));
    cfg.blockDim = dim3(kThreads);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    auto* ws = static_cast<RowWorkspace*>(workspace);
    if (pdl)
    {
        return candidateList ? launchV3Copies<true, true>(cfg, listPath, wideHist, probs, out, topP, d, ws, keptOut)
                             : launchV3Copies<true, false>(cfg, listPath, wideHist, probs, out, topP, d, ws, keptOut);
    }
    return candidateList ? launchV3Copies<false, true>(cfg, listPath, wideHist, probs, out, topP, d, ws, keptOut)
                         : launchV3Copies<false, false>(cfg, listPath, wideHist, probs, out, topP, d, ws, keptOut);
}

size_t toppRenormDetV3WorkspaceBytesPerRow()
{
    return sizeof(toppv3::RowWorkspace);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
