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

// ST-01 v2: FlashInfer AirTopPRenormProb<IsDeterministic = true, float> in one
// persistent launch, with the per-row keep-all bypass for top_p >= 1. See toppRenormDetSm103.h for the contract.
// Compiled with --use_fast_math (set on this source in kernels/CMakeLists.txt), as FlashInfer's JIT compiles it.

#include "tensorrt_llm/kernels/minimaxM31/toppRenormDetSm103.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/atomic>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include <cub/cub.cuh>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace minimaxM31
{

namespace
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
constexpr int kCopies = 4;                // privatized pass histograms
constexpr size_t kDynSmemBytes = kCopies * kBuckets * (sizeof(unsigned long long) + sizeof(int)); // 96 KB

// One row's persistent state. Zero once; every launch leaves it zeroed except `gen` (monotonic) and the per-launch
// scalars it rewrites before reading.
struct RowWorkspace
{
    unsigned long long hist[kBuckets]; // the pass's det mantissa-sum histogram (FlashInfer's uint64 HisT)
    int cnt[kBuckets];                 // the pass's count histogram
    float chain[kThreads];             // the 1024 per-thread chain sums of the kept values
    unsigned arrive;                   // group barrier: CTAs arrived
    unsigned gen;                      // group barrier: generation
    unsigned kbits;                    // the selected key prefix (FlashInfer's kthValueBits)
    float sum;                         // the remaining target sum (FlashInfer's counter->sum)
    float norm;
    int kept;
    int pad[2];
};

__device__ __forceinline__ unsigned ldAcquire(unsigned* p)
{
    return cuda::atomic_ref<unsigned, cuda::thread_scope_device>(*p).load(cuda::memory_order_acquire);
}

__device__ __forceinline__ void stRelease(unsigned* p, unsigned v)
{
    cuda::atomic_ref<unsigned, cuda::thread_scope_device>(*p).store(v, cuda::memory_order_release);
}

// Group barrier of the row's G CTAs; the last CTA to arrive runs `leader` (whole CTA) before releasing the others.
template <typename F>
__device__ __forceinline__ void groupSync(RowWorkspace* w, int G, F leader)
{
    __shared__ unsigned sGen;
    __shared__ int sLast;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        unsigned gen = ldAcquire(&w->gen);
        sGen = gen;
        unsigned old = atomicAdd(&w->arrive, 1u);
        sLast = (old == static_cast<unsigned>(G - 1));
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
            stRelease(&w->gen, sGen + 1);
        }
    }
    else if (threadIdx.x == 0)
    {
        while (ldAcquire(&w->gen) == sGen)
        {
            __nanosleep(20);
        }
    }
    __syncthreads();
}

// FlashInfer's last-block selection for `pass` (air_top_p.cuh AirTopPRenormRadixKernel, IsDeterministic branch), on
// the row's global integer histograms. histVal and cntS (the bucket counts, cached for the sequential scan) reuse the
// CTA's scratch.
__device__ __forceinline__ void selectPass(RowWorkspace* w, int pass, float p, float* histVal, int* cntS)
{
    constexpr int WARP_SIZE = 32;
    constexpr int WARP_COUNT = kBuckets / WARP_SIZE;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    __shared__ float warpSum[WARP_COUNT];
    __shared__ cuda::atomic<float, cuda::thread_scope_block> blockSum;
    for (int i = threadIdx.x; i < kBuckets; i += blockDim.x)
    {
        unsigned long long value = __ldcg(&w->hist[i]);
        int count = __ldcg(&w->cnt[i]);
        cntS[i] = count;
        if (count != 0)
        {
            uint32_t sb = fa::calcStartBit<float>(pass);
            uint32_t bv = __ldcg(&w->kbits);
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
        float currentSum = pass == 0 ? static_cast<float>(blockSum) * p : __ldcg(&w->sum);
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
        w->sum = currentSum - prev;
        unsigned kb = pass == 0 ? 0u : __ldcg(&w->kbits);
        w->kbits = kb | (static_cast<unsigned>(targetIdx) << fa::calcStartBit<float>(pass));
    }
    __syncthreads();
    for (int i = threadIdx.x; i < kBuckets; i += blockDim.x)
    {
        w->hist[i] = 0;
        w->cnt[i] = 0;
    }
}

__global__ void __launch_bounds__(kThreads, 1) toppRenormDetKernel(
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
                        unsigned const s = __reduce_add_sync(0xffffffffu, m);
                        if (lane == 0)
                        {
                            atomicAdd(myHs + b0, static_cast<unsigned long long>(s));
                            atomicAdd(myCs + b0, 32);
                        }
                    }
                }
                else if (ok)
                {
                    atomicAdd(myHs + b, static_cast<unsigned long long>(m));
                    atomicAdd(myCs + b, 1);
                }
            }
        }
        __syncthreads();
        for (int i = tid; i < kBuckets; i += kThreads)
        {
            unsigned long long h = 0;
            int c = 0;
#pragma unroll
            for (int k = 0; k < kCopies; ++k)
            {
                h += hsCopies[k * kBuckets + i];
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
                atomicAdd(&w->hist[i], hs[i]);
            }
            if (cs[i] != 0)
            {
                atomicAdd(&w->cnt[i], cs[i]);
            }
        }
        groupSync(w, G,
            [&] {
                selectPass(w, pass, p, reinterpret_cast<float*>(smem),
                    reinterpret_cast<int*>(smem + kBuckets * sizeof(float)));
            });
        kbits = __ldcg(&w->kbits);
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
    for (int k0 = 0; k0 < kTotal; k0 += kTile)
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
    groupSync(w, G,
        [&]
        {
            float totalSum = BlockReduce(temp).Sum(__ldcg(&w->chain[tid]));
            if (tid == 0)
            {
                w->norm = (totalSum > 1e-8f) ? (1.0f / totalSum) : 1.0f;
                if (keptOut)
                {
                    keptOut[row] = __ldcg(&w->kept);
                }
                w->kept = 0;
            }
        });
    float const norm = __ldcg(&w->norm);
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

} // namespace

cudaError_t invokeToppRenormDet(float const* probs, float* out, float const* topP, int rows, int d, int ctasPerRow,
    void* workspace, int* keptOut, cudaStream_t stream)
{
    if (rows <= 0 || d <= 0)
    {
        return cudaSuccess;
    }
    if (ctasPerRow < 1 || ctasPerRow > kThreads || kThreads % ctasPerRow != 0 || d % 4 != 0)
    {
        return cudaErrorInvalidValue;
    }
    static cudaError_t const attrErr = cudaFuncSetAttribute(
        toppRenormDetKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kDynSmemBytes));
    if (attrErr != cudaSuccess)
    {
        return attrErr;
    }
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(static_cast<unsigned>(ctasPerRow), static_cast<unsigned>(rows));
    cfg.blockDim = dim3(kThreads);
    cfg.dynamicSmemBytes = kDynSmemBytes;
    cfg.stream = stream;
    return cudaLaunchKernelEx(
        &cfg, toppRenormDetKernel, probs, out, topP, d, static_cast<RowWorkspace*>(workspace), keptOut);
}

size_t toppRenormDetWorkspaceBytesPerRow()
{
    return sizeof(RowWorkspace);
}

} // namespace minimaxM31
} // namespace kernels

TRTLLM_NAMESPACE_END
