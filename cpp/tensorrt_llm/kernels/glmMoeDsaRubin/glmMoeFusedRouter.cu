/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
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

// Router GEMV + DeepSeek-style routing (sigmoid + bias, top-8 of 256 experts, scaled sum normalization) in one kernel
// for 1-8 tokens, with the outputs of routingIndicesSmallTokenKernel.
//
// Logits. Every logit is the value glmRouterGemm.cu's router_gemm_kernel computes (bit for bit). There, thread t
// (0..127) of the CTA of expert e accumulates acc = fma(a, w, acc) over k = ki * 1024 + 8 t + j (ki = 0..5, j = 0..7),
// in that order; warp s (the "slice" of threads 32 s .. 32 s + 31) adds its lanes with the xor butterfly 16, 8, 4, 2, 1
// (lane 0's value), and the four slice sums are added to 0.0f in slice order. Here a GEMV CTA covers one slice of 8
// experts: it stages the slice's activations of every token in shared memory (24 KB at 8 tokens), runs the same
// per-thread chains (fma.rn.f32.bf16 computes fma(float(a), float(w), acc) without conversion instructions), adds the
// 32 positions with the same pairs as the butterfly and publishes the slice sums. 128 CTAs cover 256 experts x 4
// slices, and each CTA stages a quarter of the activations instead of all of them.
//
// Handoff. The slice sums go to one of two exchange buffers in the workspace (by the parity of the epoch word) whose
// idle value is -0.0. A sum of -0.0 is published as +0.0, which gives the same logit, since the four sums are added to
// 0.0f. The routing CTA polls its token rows until no -0.0 is left, so no fence or counter is needed. It resets the
// other buffer, consumed by the previous launch, while it waits, and advances the epoch at its end. Launches that share
// a workspace must therefore run in stream order.
//
// Routing. Phase A of routingIndicesSmallTokenKernel (one warp per token: sigmoid + bias, the packed-key top-K, the
// scaled sum normalization; the same operations in the same order). The metadata then follows from the set of
// selected local experts after a single barrier: with no more tokens than tile rows every selected local expert owns
// exactly one CTA tile, so its tile index is its rank among the selected local experts and its rows start at
// rank * tile, which is what the small-token kernel's scans compute.

#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmMoeFusedRouter.h"
#include "tensorrt_llm/kernels/moe/trtllmGen/routing/RoutingCustomPolicy.cuh"
#include "tensorrt_llm/kernels/moe/trtllmGen/routing/RoutingKernel.h"

#include <cooperative_groups.h>

// The kernel is file-local to the routing namespace whose policies and helpers it uses.
namespace moe::dev::routing
{
namespace routingCustom
{
namespace
{

using FusedExpertSelect = TopKExpertSelect<SigmoidBiasPreprocess, ScaledSumNormalizePostprocess>;
using FusedKernelParams = KernelParams<float, __nv_bfloat16, NumExperts256Experts, NumTop8Experts, FusedExpertSelect>;

constexpr int kNumExperts = 256;
constexpr int kTopK = 8;
constexpr int kMaxTokens = 8;
constexpr int kHiddenDim = 6144;
constexpr int kEltsPerVec = 8;                                         // bf16 per 16-byte load, as in the router GEMV
constexpr int kKIterElts = kEltsPerVec * 128;                          // elements per k-iteration of the router GEMV
constexpr int kKIters = kHiddenDim / kKIterElts;                       // 6
constexpr int kSlices = 4;                                             // warps per expert in the router GEMV
constexpr int kExpertsPerCta = 8;
constexpr int kNumGemvCtas = kNumExperts / kExpertsPerCta * kSlices;   // 128
constexpr int kNumThreads = 256;
constexpr int kSliceVecs = kKIters * WarpSize;                         // 16-byte vectors of one token's slice
constexpr int kActBytes = kMaxTokens * kSliceVecs * 16;                // 24 KB
constexpr int kPartBytes = WarpSize * kExpertsPerCta * kMaxTokens * 4; // 8 KB
constexpr int kSmemBytes = kActBytes + kPartBytes;
constexpr int kExchangeFloats = kMaxTokens * kNumExperts * kSlices;
constexpr int kKIdxRowBytes = 8;
constexpr uint32_t kNegZeroBits = 0x80000000u;

static_assert(kNumExperts == kNumThreads, "the routing CTA runs one thread per expert");
static_assert(
    2 * kExchangeFloats + 4 == tensorrt_llm::kernels::glm_moe_dsa::kFusedRouterWorkspaceWords, "workspace layout");

struct FusedRouterParams
{
    FusedKernelParams routing;
    __nv_bfloat16 const* hiddenStates;
    __nv_bfloat16 const* routerWeight;
    int32_t* workspace;
};

__device__ __forceinline__ uint32_t smemAddr(void const* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void gridDependencyWait()
{
    // The memory clobber keeps the compiler from moving memory accesses across the wait.
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

// Non-coherent 16-byte load of router weights.
__device__ __forceinline__ uint4 loadWeight16(void const* ptr)
{
    uint4 v;
    asm volatile("ld.global.nc.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr));
    return v;
}

__device__ __forceinline__ uint32_t loadVolatile(uint32_t const* ptr)
{
    uint32_t v;
    asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(ptr) : "memory");
    return v;
}

__device__ __forceinline__ float4 loadVolatile4(float const* ptr)
{
    float4 v;
    asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                 : "l"(ptr)
                 : "memory");
    return v;
}

// A slice sum for the routing CTA, which polls it with volatile loads.
__device__ __forceinline__ void storeRelaxed(float* ptr, float v)
{
    asm volatile("st.relaxed.gpu.global.f32 [%0], %1;" ::"l"(ptr), "f"(v) : "memory");
}

__device__ __forceinline__ bool isNegZero(float v)
{
    return __float_as_uint(v) == kNegZeroBits;
}

// acc = fma(a[j], w[j], acc) for j = 0..7, in that order, on 8 packed bf16 pairs.
__device__ __forceinline__ void fmaBf16x8(float& acc, uint4 const& a, uint4 const& w)
{
    uint32_t const aw[4] = {a.x, a.y, a.z, a.w};
    uint32_t const ww[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int q = 0; q < 4; q++)
    {
        asm("{\n"
            ".reg .b16 al, ah, bl, bh;\n"
            "mov.b32 {al, ah}, %1;\n"
            "mov.b32 {bl, bh}, %2;\n"
            "fma.rn.f32.bf16 %0, al, bl, %0;\n"
            "fma.rn.f32.bf16 %0, ah, bh, %0;\n"
            "}\n"
            : "+f"(acc)
            : "r"(aw[q]), "r"(ww[q]));
    }
}

// The slice GEMV of one CTA: slice r of experts 8 g .. 8 g + 7. Thread mapping: warp w covers positions 8 (w & 3) ..
// + 7 of the slice and experts 4 (w >> 2) .. + 3, so the 4 expert groups of a warp read the same 128 bytes of
// activations. Returns through emit(eLocal, m, sliceSum) once per (expert, token).
template <int kNumTokens, typename Emit>
__device__ __forceinline__ void sliceGemv(uint8_t* smem, __nv_bfloat16 const* hiddenStates, uint4 const (&w)[kKIters],
    int r, int t, int eLocal, Emit const& emit)
{
    uint4* act = reinterpret_cast<uint4*>(smem);
    float* part = reinterpret_cast<float*>(smem + kActBytes);

    // Stage the slice: token m, k-iteration ki, positions 32 r .. 32 r + 31.
    constexpr int kVecs = kNumTokens * kSliceVecs;
#pragma unroll
    for (int j = 0; j < (kVecs + kNumThreads - 1) / kNumThreads; ++j)
    {
        int const c = threadIdx.x + kNumThreads * j;
        if (c < kVecs)
        {
            int const m = c / kSliceVecs;
            int const rem = c - m * kSliceVecs;
            __nv_bfloat16 const* src = hiddenStates + m * kHiddenDim + (rem / WarpSize) * kKIterElts
                + (WarpSize * r + rem % WarpSize) * kEltsPerVec;
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smemAddr(&act[c])), "l"(src) : "memory");
        }
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
    __syncthreads();

    // The per-thread chains of router thread 32 r + t for one expert and every token.
    float acc[kMaxTokens] = {};
#pragma unroll
    for (int ki = 0; ki < kKIters; ki++)
    {
#pragma unroll
        for (int m = 0; m < kNumTokens; m++)
        {
            fmaBf16x8(acc[m], act[(m * kKIters + ki) * WarpSize + t], w[ki]);
        }
    }

    // The butterfly of the slice: 4 threads per (expert, token) fold the positions {sub, sub + 4, ..., sub + 28}
    // through levels 16, 8 and 4, two shuffles add levels 2 and 1.
    float4* row = reinterpret_cast<float4*>(part + (t * kExpertsPerCta + eLocal) * kMaxTokens);
    row[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
    row[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
    __syncthreads();
    int const pair = threadIdx.x / 4;
    int const sub = threadIdx.x % 4;
    int const eL = pair / kMaxTokens;
    int const m = pair % kMaxTokens;
    auto v = [&](int tt) { return part[(tt * kExpertsPerCta + eL) * kMaxTokens + m]; };
    float const s1a = v(sub) + v(sub + 16);
    float const s1b = v(sub + 4) + v(sub + 20);
    float const s1c = v(sub + 8) + v(sub + 24);
    float const s1d = v(sub + 12) + v(sub + 28);
    float sum = (s1a + s1c) + (s1b + s1d);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);
    if (sub == 0 && m < kNumTokens)
    {
        emit(eL, m, sum);
    }
}

template <int kNumTokens>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
__global__ void __launch_bounds__(kNumThreads, 1) routingFusedRouterKernel(FusedRouterParams const __grid_constant__ p)
{
    extern __shared__ __align__(128) uint8_t smemSlice[];
    __shared__ __align__(8) uint8_t smemKIdx[kNumExperts * kKIdxRowBytes];
    __shared__ int16_t smemTopKExpert[kMaxTokens * kTopK];
    __shared__ float smemBias[kNumExperts];
    __shared__ __align__(16) uint32_t smemSelected[kNumExperts / 32];

    auto const& params = p.routing;
    float* const buffers = reinterpret_cast<float*>(p.workspace);
    uint32_t* const epochPtr = reinterpret_cast<uint32_t*>(p.workspace + 2 * kExchangeFloats);

    if (blockIdx.x < kNumGemvCtas)
    {
        int const r = blockIdx.x % kSlices;
        int const g = blockIdx.x / kSlices;
        int const warp = threadIdx.x / WarpSize;
        int const lane = threadIdx.x % WarpSize;
        int const t = 8 * (warp % 4) + lane % 8;
        int const eLocal = 4 * (warp / 4) + lane / 8;
        // The weights are not written by the primary grid: load them before the wait. A shared-memory store of their
        // XOR ahead of a barrier makes the loads complete before the wait instead of being scheduled after it.
        uint4 w[kKIters];
        __nv_bfloat16 const* wRow
            = p.routerWeight + (kExpertsPerCta * g + eLocal) * kHiddenDim + (WarpSize * r + t) * kEltsPerVec;
        uint32_t loaded = 0;
#pragma unroll
        for (int ki = 0; ki < kKIters; ki++)
        {
            w[ki] = loadWeight16(wRow + ki * kKIterElts);
            loaded ^= w[ki].x ^ w[ki].y ^ w[ki].z ^ w[ki].w;
        }
        reinterpret_cast<uint32_t*>(smemSlice + kActBytes)[threadIdx.x] = loaded;
        __syncthreads();
        gridDependencyWait();
        // The previous launch on this workspace precedes the primary grid in the stream, so it has completed.
        float* const exchange = buffers + (loadVolatile(epochPtr) & 1u) * kExchangeFloats;
        sliceGemv<kNumTokens>(smemSlice, p.hiddenStates, w, r, t, eLocal,
            [&](int eL, int m, float sliceSum)
            {
                storeRelaxed(&exchange[(m * kNumExperts + kExpertsPerCta * g + eL) * kSlices + r],
                    isNegZero(sliceSum) ? 0.f : sliceSum);
            });
        // Exit without a trigger: the dependents launch once the routing CTA triggers.
        return;
    }

    // Routing CTA. Independent of the primary grid: clear this expert's row of the byte map, stage its bias.
    *reinterpret_cast<uint2*>(&smemKIdx[threadIdx.x * kKIdxRowBytes]) = uint2{0xFFFFFFFFu, 0xFFFFFFFFu};
    auto const& pre = params.mExpertSelectParams.mPreprocessParams;
    smemBias[threadIdx.x] = loadScalar(pre.ptrRoutingBias, threadIdx.x, pre.dtypeBias);
    if (threadIdx.x < kNumExperts / 32)
    {
        smemSelected[threadIdx.x] = 0u;
    }
    __syncthreads();
    gridDependencyWait();
    uint32_t const epoch = loadVolatile(epochPtr);
    float* const exchange = buffers + (epoch & 1u) * kExchangeFloats;
    {
        // The buffer of the previous launch, which its routing CTA has consumed: idle again for the next launch.
        float4* other = reinterpret_cast<float4*>(buffers + ((epoch + 1u) & 1u) * kExchangeFloats);
        for (int i = threadIdx.x; i < kExchangeFloats / 4; i += kNumThreads)
        {
            other[i] = make_float4(-0.f, -0.f, -0.f, -0.f);
        }
    }

    // ----- Phase A: one warp per token, as in routingIndicesSmallTokenKernel -----
    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
    int32_t const laneIdx = cutlass::arch::LaneId();
    auto block = cg::this_thread_block();
    auto warp = cg::tiled_partition<WarpSize>(block);
    if (warpIdx < kNumTokens)
    {
        constexpr int VecSize = kNumExperts / WarpSize;
        float4 slices[VecSize];
        bool ready;
        do
        {
            ready = true;
#pragma unroll
            for (int i = 0; i < VecSize; ++i)
            {
                slices[i] = loadVolatile4(exchange + (warpIdx * kNumExperts + i * WarpSize + laneIdx) * kSlices);
                ready = ready && !isNegZero(slices[i].x) && !isNegZero(slices[i].y) && !isNegZero(slices[i].z)
                    && !isNegZero(slices[i].w);
            }
        } while (!ready);

        float score[VecSize];
        int32_t idx[VecSize];
#pragma unroll
        for (int i = 0; i < VecSize; ++i)
        {
            idx[i] = i * WarpSize + laneIdx;
            float logit = 0.0f;
            logit += slices[i].x;
            logit += slices[i].y;
            logit += slices[i].z;
            logit += slices[i].w;
            score[i] = sigmoid_accurate(logit) + smemBias[idx[i]];
        }
        float topScore[kTopK];
        int32_t topExpert[kTopK];
        topk::reduceTopK(warp, topScore, topExpert, score, idx, float{-INFINITY}, kTopK);

        // Lane k's top-K entry, the value the small-token kernel reads as warpTopKScore[laneIdx].
        float myScore = 0.f;
        int32_t myExpert = 0;
#pragma unroll
        for (int k = 0; k < kTopK; ++k)
        {
            myScore = laneIdx == k ? topScore[k] : myScore;
            myExpert = laneIdx == k ? topExpert[k] : myExpert;
        }
        bool const isTop = laneIdx < kTopK;
        float const sigmoidScore = isTop ? (myScore - smemBias[myExpert]) : 0.f;
        // cg::reduce's butterfly (xor 16, 8, 4, 2, 1) adds +0.0 to lanes 0-7 in its first two levels, which leaves
        // them unchanged (no sigmoidScore is -0.0); levels 4, 2 and 1 then pair lanes 0-7 as here.
        float sum = sigmoidScore;
        sum += __shfl_xor_sync(0xffffffff, sum, 4);
        sum += __shfl_xor_sync(0xffffffff, sum, 2);
        sum += __shfl_xor_sync(0xffffffff, sum, 1);
        if (isTop)
        {
            auto const& post = params.mExpertSelectParams.mPostprocessParams;
            float const weight = sigmoidScore * post.routeScale / (sum + post.sumEpsilon);
            smemKIdx[myExpert * kKIdxRowBytes + warpIdx] = static_cast<uint8_t>(laneIdx);
            smemTopKExpert[warpIdx * kTopK + laneIdx] = static_cast<int16_t>(myExpert);
            atomicOr(&smemSelected[myExpert / 32], 1u << (myExpert % 32));
            params.mPtrTopKWeights[warpIdx * kTopK + laneIdx] = __nv_bfloat16{weight};
        }
    }
    __syncthreads();

    // ----- Metadata: tile index = rank among the selected local experts -----
    int32_t const localBegin = params.mLocalExpertsStartIdx;
    int32_t const localEnd = params.mLocalExpertsStartIdx + params.mNumLocalExperts;
    uint4 const lo = *reinterpret_cast<uint4 const*>(&smemSelected[0]);
    uint4 const hi = *reinterpret_cast<uint4 const*>(&smemSelected[4]);
    uint32_t selected[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
    if (localBegin != 0 || localEnd < kNumExperts)
    {
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            int32_t const b0 = max(localBegin - 32 * i, 0);
            int32_t const b1 = min(localEnd - 32 * i, 32);
            uint32_t const upTo = b1 >= 32 ? 0xFFFFFFFFu : ((1u << max(b1, 0)) - 1u);
            uint32_t const below = b0 >= 32 ? 0xFFFFFFFFu : ((1u << b0) - 1u);
            selected[i] &= b1 <= b0 ? 0u : (upTo & ~below);
        }
    }
    int32_t numSelected = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i)
    {
        numSelected += __popc(selected[i]);
    }
    // (rank, is selected and local) of an expert, from masked popcounts of every word: indexing the words by the expert
    // would place them in local memory.
    auto rankOf = [&](int32_t e, bool& isSelected)
    {
        int32_t const word = e >> 5;
        uint32_t const bit = 1u << (e & 31);
        int32_t rank = 0;
        int32_t hit = 0;
#pragma unroll
        for (int i = 0; i < 8; ++i)
        {
            rank += __popc(selected[i] & (i < word ? 0xFFFFFFFFu : (i == word ? bit - 1u : 0u)));
            hit += __popc(selected[i] & (i == word ? bit : 0u));
        }
        isSelected = hit != 0;
        return rank;
    };
    auto tileRows = [&](int32_t n)
    { return params.mIsPow2 ? mulLog2<int32_t>(n, params.mPaddingLog2) : mulTileN<int32_t>(n, params.mTileTokensDim); };

    {
        int32_t const expert = threadIdx.x;
        bool isSelected;
        int32_t const rank = rankOf(expert, isSelected);
        if (isSelected)
        {
            uint2 const kRow = *reinterpret_cast<uint2 const*>(&smemKIdx[expert * kKIdxRowBytes]);
            int32_t const count = __popc(__vsetne4(kRow.x, 0xFFFFFFFFu)) + __popc(__vsetne4(kRow.y, 0xFFFFFFFFu));
            params.mPtrCtaIdxXyToBatchIdx[rank] = expert - localBegin;
            params.mPtrCtaIdxXyToMnLimit[rank] = min(tileRows(rank + 1), tileRows(rank) + count);
        }
    }
    if (threadIdx.x == 0)
    {
        params.mPtrPermutedIdxSize[0] = tileRows(numSelected);
        params.mPtrNumNonExitingCtas[0] = numSelected;
    }

    // Permutation, one (token, k) pair per thread. offset-within-expert = number of earlier tokens that also selected
    // the expert (its byte-map row below byte t), the small-token kernel's order.
    if (static_cast<int32_t>(threadIdx.x) < kNumTokens * kTopK)
    {
        int32_t const expandedIdx = threadIdx.x;
        int32_t const tokenIdx = expandedIdx / kTopK;
        int32_t const pairExpert = smemTopKExpert[expandedIdx];
        bool pairIsLocal;
        int32_t const rank = rankOf(pairExpert, pairIsLocal);
        uint2 const row = *reinterpret_cast<uint2 const*>(&smemKIdx[pairExpert * kKIdxRowBytes]);
        uint64_t const present = (static_cast<uint64_t>(__vsetne4(row.y, 0xFFFFFFFFu)) << 32)
            | static_cast<uint64_t>(__vsetne4(row.x, 0xFFFFFFFFu));
        uint64_t const earlier = tokenIdx == 0 ? 0ull : (present & ((1ull << (8 * tokenIdx)) - 1));
        int32_t const permutedIdx = pairIsLocal ? tileRows(rank) + __popcll(earlier) : int32_t{-1};
        if (params.mPtrExpandedIdxToPermutedIdx != nullptr)
        {
            params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = permutedIdx;
        }
        if (params.mPtrPermutedIdxToExpandedIdx != nullptr && pairIsLocal)
        {
            params.mPtrPermutedIdxToExpandedIdx[permutedIdx] = expandedIdx;
        }
        if (params.mPtrPermutedIdxToTokenIdx != nullptr && pairIsLocal)
        {
            params.mPtrPermutedIdxToTokenIdx[permutedIdx] = tokenIdx;
        }
    }
    if (threadIdx.x == 0)
    {
        *epochPtr = epoch + 1u;
    }
    // After all routing outputs are written.
    cudaTriggerProgrammaticLaunchCompletion();
}
#else
__global__ void __launch_bounds__(kNumThreads, 1)
    routingFusedRouterKernel(FusedRouterParams const __grid_constant__ /* p */)
{
    assert(false && "routingFusedRouterKernel is only supported on SM100+ architectures");
}
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))

// ---------------------------------------------------------------------------------------------------------------------
// 9-24 tokens: a router GEMV front end followed in the same grid by a routing cluster.
//
// The front ends compute the logits of the production router kernel at that token count, bit for bit:
// - 9-16 tokens: a GEMV whose logits are router_gemm_kernel's (glmRouterGemm.cu): clusters
//   of 4 CTAs of 256 threads, one per K slice of 8 experts; the final sum is added to 0.0f in slice order.
// - 17-24 tokens: router_gemm_mma_kernel's tensor-core GEMM (glmRouterGemm.cu): clusters of 8
//   CTAs of 128 threads, one per K slice of 768 for 16 experts; rank 0 adds the 8 slices in rank order.
// Both load their weights before the grid-dependency wait. Instead of a logits tensor, each cluster's rank 0 stores the
// logits into the exchange buffer [token][expert] of the workspace (by the parity of the epoch word), whose idle value
// is -0.0. A -0.0 logit is stored as +0.0, which gives the same score (sigmoid of either is 0.5); the sliced front end
// never produces one, since its final sum starts from 0.0f.
//
// Routing cluster (the grid's last cluster): warp w of rank r >= 1 routes token W (r - 1) + w (W warps per CTA). It
// polls the token's 256 logits until none is -0.0 and runs routingIndicesSmallTokenKernel's phase A (sigmoid + bias,
// reduceTopK, scaled sum normalization, with the bias staged in shared memory), which computes the values of the
// routing kernels' TopK. It writes the top-K weights and stores the top-K experts into rank 0's shared memory, whose
// slots rank 0 marked empty before the cluster's first barrier phase. Once every slot is filled, rank 0 writes
// routingIndicesDynBlockKernel's metadata: an expert with n local tokens owns ceil(n / tile) CTA tiles, the tiles
// follow expert order, and an expert's rows follow token order (the cluster routing kernel used at 17+ tokens writes
// the same tiles, with the rows of an expert in the order of its shared-memory atomics). Both the token warps and
// rank 0 first run their code once on dummy data in shared memory and registers, before the grid-dependency wait
// (while the primary grid runs): the live pass then executes from a warm instruction cache. The cluster also resets
// the previous launch's buffer and rank 0 advances the epoch, as the 1-8 token kernel does; both kernels keep the
// workspace in the same state.
namespace wide
{
// Sliced front end, 9-16 tokens.
constexpr int kSliceMinTokens = 9;
constexpr int kSliceMaxTokens = 16;
constexpr int kSliceCluster = 4;     // K slices of the 9-16-token GEMV
constexpr int kSliceThreads = 256;
constexpr int kSliceStageGroups = 3; // cp.async groups of 2 k-iterations
constexpr int kSliceKItersPerGroup = kKIters / kSliceStageGroups;
constexpr int kSliceRowBytes = 16;

__host__ __device__ constexpr int slicePaddedTokens(int numTokens)
{
    return (numTokens + 3) / 4 * 4;
}

// GEMV CTAs: activations [tokens][kKIters][32] 16-byte vectors, reused for the butterfly staging [32][8][paddedTokens]
// fp32, then the slice sums [kSliceCluster][8][tokens] fp32 received by rank 0 (router_gemm_sliced_kernel's layout).
__host__ __device__ constexpr int sliceWorkBytes(int numTokens)
{
    return numTokens * kKIters * WarpSize * 16 > WarpSize * kExpertsPerCta * slicePaddedTokens(numTokens) * 4
        ? numTokens * kKIters * WarpSize * 16
        : WarpSize * kExpertsPerCta * slicePaddedTokens(numTokens) * 4;
}

__host__ __device__ constexpr int sliceGemvSmemBytes(int numTokens)
{
    return sliceWorkBytes(numTokens) + kSliceCluster * kExpertsPerCta * numTokens * 4;
}

// Tensor-core front end, 17-24 tokens (router_gemm_mma_kernel's routerMma constants).
constexpr int kMmaMinTokens = 17;
constexpr int kMmaMaxTokens = 24;
constexpr int kMmaExpertsPerCta = 16;
constexpr int kMmaCluster = 8;               // K slices of 768
constexpr int kMmaSliceK = kHiddenDim / kMmaCluster;
constexpr int kMmaRowElems = kMmaSliceK + 8; // the 8 rows of an ldmatrix start in distinct banks
constexpr int kMmaWarps = 4;
constexpr int kMmaThreads = kMmaWarps * WarpSize;
constexpr int kMmaKStepsPerWarp = kMmaSliceK / 16 / kMmaWarps;
constexpr int kMmaChunks = kMmaSliceK / 8; // 16-byte chunks per row of a slice
constexpr int kMmaGemvCtas = kNumExperts / kMmaExpertsPerCta * kMmaCluster;
constexpr int kMmaRowBytes = 32;

__host__ __device__ constexpr int mmaPaddedTokens(int numTokens)
{
    return (numTokens + 7) / 8 * 8;
}

// Weight tile and activation rows (bf16), the 4 warp tiles and the 8 slice tiles received by rank 0 (fp32).
__host__ __device__ constexpr int mmaGemvSmemBytes(int numTokens)
{
    return (kMmaExpertsPerCta + mmaPaddedTokens(numTokens)) * kMmaRowElems * 2
        + (kMmaWarps + kMmaCluster) * kMmaExpertsPerCta * mmaPaddedTokens(numTokens) * 4;
}

// Routing CTAs: bias [256] fp32, top-K experts [kTokenCapacity][8] int32, byte map [256][kRowBytes] (one byte per
// token, 0xFF = not selected), CTA offsets [256] int32, warp totals [8] int32.
template <int kTokenCapacity, int kRowBytes>
struct RoutingSmem
{
    static constexpr int kBias = 0;
    static constexpr int kTopKExperts = kBias + kNumExperts * 4;
    static constexpr int kKIdx = kTopKExperts + kTokenCapacity * kTopK * 4;
    static constexpr int kCtaOffset = kKIdx + kNumExperts * kRowBytes;
    static constexpr int kWarpTotals = kCtaOffset + kNumExperts * 4;
    static constexpr int kBytes = kWarpTotals + 8 * 4;
    static_assert(kKIdx % 16 == 0, "byte map rows are read as 16-byte vectors");
};

using SliceRoutingSmem = RoutingSmem<kSliceMaxTokens, kSliceRowBytes>;
using MmaRoutingSmem = RoutingSmem<kMmaMaxTokens, kMmaRowBytes>;

__host__ __device__ constexpr int sliceSmemBytes(int numTokens)
{
    return sliceGemvSmemBytes(numTokens) > SliceRoutingSmem::kBytes ? sliceGemvSmemBytes(numTokens)
                                                                    : SliceRoutingSmem::kBytes;
}

__host__ __device__ constexpr int mmaSmemBytes(int numTokens)
{
    return mmaGemvSmemBytes(numTokens) > MmaRoutingSmem::kBytes ? mmaGemvSmemBytes(numTokens) : MmaRoutingSmem::kBytes;
}

static_assert(kMmaMaxTokens * kNumExperts <= kExchangeFloats, "a buffer holds [kMmaMaxTokens][256] logits");
static_assert(kSliceCluster == kSlices && kNumGemvCtas == kNumExperts / kExpertsPerCta * kSliceCluster, "GEMV grid");
} // namespace wide

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
__device__ __forceinline__ void clusterArriveRelaxed()
{
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

__device__ __forceinline__ void clusterWait()
{
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

__device__ __forceinline__ void clusterArriveRelease()
{
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}

__device__ __forceinline__ void clusterWaitAcquire()
{
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ uint32_t mapToClusterRank(void const* ptr, uint32_t rank)
{
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(smemAddr(ptr)), "r"(rank));
    return remote;
}

// Stores v at the same shared-memory offset as ptr in the CTA of the given rank of this cluster.
__device__ __forceinline__ void storeToClusterRank(float* ptr, uint32_t rank, float v)
{
    asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(mapToClusterRank(ptr, rank)), "f"(v) : "memory");
}

__device__ __forceinline__ void storeRelaxedToClusterRank(int32_t* ptr, uint32_t rank, int32_t v)
{
    asm volatile("st.relaxed.cluster.shared::cluster.s32 [%0], %1;" ::"r"(mapToClusterRank(ptr, rank)), "r"(v)
                 : "memory");
}

// A slot of this CTA's shared memory that other CTAs of the cluster store to.
__device__ __forceinline__ int32_t loadRelaxedCluster(int32_t const* ptr)
{
    int32_t v;
    asm volatile("ld.relaxed.cluster.shared::cta.s32 %0, [%1];" : "=r"(v) : "r"(smemAddr(ptr)) : "memory");
    return v;
}

__device__ __forceinline__ void copyAsync16(void* dst, void const* src)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smemAddr(dst)), "l"(src) : "memory");
}

template <int kPending>
__device__ __forceinline__ void copyAsyncWait()
{
    asm volatile("cp.async.wait_group %0;" ::"n"(kPending) : "memory");
}

__device__ __forceinline__ void ldmatrixX4(uint32_t (&r)[4], void const* ptr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smemAddr(ptr)));
}

__device__ __forceinline__ void ldmatrixX2(uint32_t (&r)[2], void const* ptr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(smemAddr(ptr)));
}

__device__ __forceinline__ void mmaBf16(float (&c)[4], uint32_t const (&a)[4], uint32_t const (&b)[2])
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// The number of tokens before token beforeToken (all tokens if beforeToken >= kRowBytes) in an expert's byte map row.
template <int kRowBytes>
__device__ __forceinline__ int32_t countTokensBefore(uint8_t const* smemKIdx, int32_t expert, int32_t beforeToken)
{
    uint4 const* row = reinterpret_cast<uint4 const*>(smemKIdx + expert * kRowBytes);
    int32_t n = 0;
#pragma unroll
    for (int v = 0; v < kRowBytes / 16; ++v)
    {
        uint4 const w4 = row[v];
        uint32_t const words[4] = {w4.x, w4.y, w4.z, w4.w};
#pragma unroll
        for (int q = 0; q < 4; ++q)
        {
            int32_t const below = beforeToken - (16 * v + 4 * q); // tokens of this word before beforeToken
            uint32_t const mask = below >= 4 ? 0xFFFFFFFFu : (below <= 0 ? 0u : (1u << (8 * below)) - 1u);
            n += __popc(__vsetne4(words[q], 0xFFFFFFFFu) & mask);
        }
    }
    return n;
}

#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
// The routing cluster of the 9-24 token kernels (see above): kThreads per CTA, kClusterSize CTAs.
template <int kNumTokens, int kThreads, int kClusterSize, int kTokenCapacity, int kRowBytes>
__device__ __forceinline__ void routingClusterTail(FusedRouterParams const& p, uint8_t* smem, uint32_t rank,
    int32_t warpIdx, int32_t laneIdx, float* buffers, uint32_t* epochPtr)
{
    using Layout = wide::RoutingSmem<kTokenCapacity, kRowBytes>;
    constexpr int kWarpsPerCta = kThreads / WarpSize;
    constexpr int kExpertsPerThread = kNumExperts / kThreads;
    static_assert(kNumExperts % kThreads == 0, "whole experts per thread in rank 0");
    static_assert(kNumTokens <= kTokenCapacity && kTokenCapacity <= kRowBytes && kRowBytes % 16 == 0, "byte map rows");
    static_assert((kClusterSize - 1) * kWarpsPerCta >= kNumTokens, "one warp per token in ranks 1 and up");
    static_assert(kWarpsPerCta <= 8, "warp totals");
    auto const& params = p.routing;
    float* const smemBias = reinterpret_cast<float*>(smem + Layout::kBias);
    int32_t* const smemTopKExpert = reinterpret_cast<int32_t*>(smem + Layout::kTopKExperts);
    uint8_t* const smemKIdx = smem + Layout::kKIdx;
    int32_t* const smemCtaOffset = reinterpret_cast<int32_t*>(smem + Layout::kCtaOffset);
    int32_t* const smemWarpTotals = reinterpret_cast<int32_t*>(smem + Layout::kWarpTotals);
    // Independent of the primary grid: stage the routing bias (the value the preprocess computes per element) and, in
    // rank 0, clear the byte map and mark the top-K slots empty (-1).
    auto const& pre = params.mExpertSelectParams.mPreprocessParams;
    for (int e = threadIdx.x; e < kNumExperts; e += kThreads)
    {
        smemBias[e] = loadScalar(pre.ptrRoutingBias, e, pre.dtypeBias);
    }
    if (rank == 0)
    {
        for (int i = threadIdx.x; i < kNumExperts * kRowBytes / 16; i += kThreads)
        {
            reinterpret_cast<uint4*>(smemKIdx)[i] = make_uint4(0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu, 0xFFFFFFFFu);
        }
        for (int i = threadIdx.x; i < kNumTokens * kTopK; i += kThreads)
        {
            smemTopKExpert[i] = -1;
        }
    }
    __syncthreads();
    // First phase of the cluster barrier, after rank 0 has marked the top-K slots empty: the other ranks store to them
    // only after the wait.
    clusterArriveRelaxed();
    // Each loop below runs a pass 0 on dummy data in shared memory and registers only, before the grid-dependency
    // wait (it overlaps the primary grid), so that pass 1 runs from a warm instruction cache. Pass 1 starts with the
    // wait, the epoch, the reset of the previous launch's buffer and the wait of the cluster barrier's first phase.
    uint32_t epoch = 0;
    float const* exchange = nullptr;
    auto const enterLivePass = [&]()
    {
        gridDependencyWait();
        epoch = loadVolatile(epochPtr);
        exchange = buffers + (epoch & 1u) * kExchangeFloats;
        // The buffer of the previous launch, which its routing cluster has consumed: idle again for the next launch.
        float4* other = reinterpret_cast<float4*>(buffers + ((epoch + 1u) & 1u) * kExchangeFloats);
        for (int i = static_cast<int>(rank) * kThreads + threadIdx.x; i < kExchangeFloats / 4;
             i += kClusterSize * kThreads)
        {
            other[i] = make_float4(-0.f, -0.f, -0.f, -0.f);
        }
        clusterWait();
    };

    if (rank != 0)
    {
        // ----- Phase A: one warp per token, as in routingIndicesSmallTokenKernel -----
        // Lane l holds experts 8 l .. 8 l + 7 (2 16-byte loads); the top-K is independent of how the experts are spread
        // over the lanes.
        int32_t const tokenIdx = (static_cast<int32_t>(rank) - 1) * kWarpsPerCta + warpIdx;
        bool const hasToken = tokenIdx < kNumTokens;
        auto block = cg::this_thread_block();
        auto warp = cg::tiled_partition<WarpSize>(block);
        constexpr int VecSize = kNumExperts / WarpSize;
        static_assert(VecSize == 8, "two 16-byte loads per lane");
#pragma unroll 1
        for (int pass = 0; pass < 2; ++pass)
        {
            bool const live = pass == 1;
            if (live)
            {
                enterLivePass();
            }
            if (hasToken)
            {
                float score[VecSize];
                int32_t idx[VecSize];
                if (live)
                {
                    float const* const row = exchange + tokenIdx * kNumExperts + laneIdx * VecSize;
                    bool ready;
                    do
                    {
                        float4 const lo = loadVolatile4(row);
                        float4 const hi = loadVolatile4(row + 4);
                        score[0] = lo.x;
                        score[1] = lo.y;
                        score[2] = lo.z;
                        score[3] = lo.w;
                        score[4] = hi.x;
                        score[5] = hi.y;
                        score[6] = hi.z;
                        score[7] = hi.w;
                        ready = true;
#pragma unroll
                        for (int i = 0; i < VecSize; ++i)
                        {
                            ready = ready && !isNegZero(score[i]);
                        }
                    } while (!ready);
                }
                else
                {
#pragma unroll
                    for (int i = 0; i < VecSize; ++i)
                    {
                        score[i] = 0.f;
                    }
                }
                // SigmoidBiasPreprocess::apply + reduceTopK + ScaledSumNormalizePostprocess::apply, with the bias read
                // from shared memory: the same operations on the same values in the same order.
#pragma unroll
                for (int i = 0; i < VecSize; ++i)
                {
                    idx[i] = laneIdx * VecSize + i;
                    float const s = sigmoid_accurate(score[i]);
                    score[i] = s + smemBias[idx[i]];
                }
                float topScore[kTopK];
                int32_t topExpert[kTopK];
                topk::reduceTopK(warp, topScore, topExpert, score, idx, float{-INFINITY}, kTopK);
                // Lane k's top-K entry.
                float myScore = 0.f;
                int32_t myExpert = 0;
#pragma unroll
                for (int k = 0; k < kTopK; ++k)
                {
                    myScore = laneIdx == k ? topScore[k] : myScore;
                    myExpert = laneIdx == k ? topExpert[k] : myExpert;
                }
                bool const isTop = laneIdx < kTopK;
                auto const& post = params.mExpertSelectParams.mPostprocessParams;
                float const biasVal = isTop ? smemBias[myExpert] : 0.f;
                float const sigmoidScore = isTop ? (myScore - biasVal) : 0.f;
                float const sum = cg::reduce(warp, sigmoidScore, cg::plus<float>());
                if (live && isTop)
                {
                    float const weight = sigmoidScore * post.routeScale / (sum + post.sumEpsilon);
                    params.mPtrTopKWeights[tokenIdx * kTopK + laneIdx] = __nv_bfloat16{weight};
                    storeRelaxedToClusterRank(&smemTopKExpert[tokenIdx * kTopK + laneIdx], 0, myExpert);
                }
            }
        }
        // Rank 0 polls its top-K slots; this CTA is done.
        return;
    }

    // ----- Metadata (rank 0), as routingIndicesDynBlockKernel -----
    // Pass 0 runs on the cleared byte map and expert 0, without global stores.
    int32_t const localBegin = params.mLocalExpertsStartIdx;
    int32_t const localEnd = params.mLocalExpertsStartIdx + params.mNumLocalExperts;
    auto const tileRows = [&](int32_t n)
    { return params.mIsPow2 ? mulLog2<int32_t>(n, params.mPaddingLog2) : mulTileN<int32_t>(n, params.mTileTokensDim); };
#pragma unroll 1
    for (int pass = 0; pass < 2; ++pass)
    {
        bool const live = pass == 1;
        if (live)
        {
            enterLivePass();
            // Every (token, k) slot is stored once by the token's warp; wait for it.
            for (int pr = threadIdx.x; pr < kNumTokens * kTopK; pr += kThreads)
            {
                int32_t expert;
                do
                {
                    expert = loadRelaxedCluster(&smemTopKExpert[pr]);
                } while (expert < 0);
                smemKIdx[expert * kRowBytes + pr / kTopK] = static_cast<uint8_t>(pr % kTopK);
            }
        }
        __syncthreads();
        // Thread t owns experts kExpertsPerThread t + j; an expert's CTA tiles follow those of the experts before it.
        int32_t count[kExpertsPerThread];
        int32_t numCta[kExpertsPerThread];
        int32_t threadCtas = 0;
#pragma unroll
        for (int j = 0; j < kExpertsPerThread; ++j)
        {
            int32_t const expert = threadIdx.x * kExpertsPerThread + j;
            count[j] = countTokensBefore<kRowBytes>(smemKIdx, expert, kRowBytes);
            bool const isLocal = expert >= localBegin && expert < localEnd;
            numCta[j] = !isLocal ? 0
                                 : (params.mIsPow2 ? divUpLog2<int32_t>(count[j], params.mPaddingLog2)
                                                   : divUpTileN<int32_t>(count[j], params.mTileTokensDim));
            threadCtas += numCta[j];
        }
        int32_t inclusive = threadCtas;
#pragma unroll
        for (int offset = 1; offset < WarpSize; offset *= 2)
        {
            int32_t const v = __shfl_up_sync(0xffffffff, inclusive, offset);
            inclusive += laneIdx >= offset ? v : 0;
        }
        if (laneIdx == WarpSize - 1)
        {
            smemWarpTotals[warpIdx] = inclusive;
        }
        __syncthreads();
        int32_t warpPrefix = 0;
        int32_t numNonExitingCtas = 0;
#pragma unroll
        for (int w = 0; w < kWarpsPerCta; ++w)
        {
            int32_t const total = smemWarpTotals[w];
            warpPrefix += w < warpIdx ? total : 0;
            numNonExitingCtas += total;
        }
        int32_t ctaOffset = warpPrefix + inclusive - threadCtas;
#pragma unroll
        for (int j = 0; j < kExpertsPerThread; ++j)
        {
            int32_t const expert = threadIdx.x * kExpertsPerThread + j;
            smemCtaOffset[expert] = ctaOffset;
            for (int32_t cta = 0; cta < numCta[j]; ++cta)
            {
                params.mPtrCtaIdxXyToBatchIdx[ctaOffset + cta] = expert - localBegin;
                params.mPtrCtaIdxXyToMnLimit[ctaOffset + cta]
                    = min(tileRows(ctaOffset + cta + 1), tileRows(ctaOffset) + count[j]);
            }
            ctaOffset += numCta[j];
        }
        if (live && threadIdx.x == 0)
        {
            params.mPtrPermutedIdxSize[0] = tileRows(numNonExitingCtas);
            params.mPtrNumNonExitingCtas[0] = numNonExitingCtas;
        }
        __syncthreads();

        // Permutation, one (token, k) pair per thread and pass of the loop: rows of an expert in token order.
        for (int pr = threadIdx.x; pr < kNumTokens * kTopK; pr += kThreads)
        {
            int32_t const expandedIdx = pr;
            int32_t const pairToken = expandedIdx / kTopK;
            // The warm-up pass reads expert 0's (cleared) row: smemTopKExpert is written by the other ranks.
            int32_t const pairExpert = live ? smemTopKExpert[expandedIdx] : 0;
            bool const pairIsLocal = pairExpert >= localBegin && pairExpert < localEnd;
            int32_t const earlier = countTokensBefore<kRowBytes>(smemKIdx, pairExpert, pairToken);
            int32_t const permutedIdx = pairIsLocal ? tileRows(smemCtaOffset[pairExpert]) + earlier : int32_t{-1};
            if (live && params.mPtrExpandedIdxToPermutedIdx != nullptr)
            {
                params.mPtrExpandedIdxToPermutedIdx[expandedIdx] = permutedIdx;
            }
            if (live && params.mPtrPermutedIdxToExpandedIdx != nullptr && pairIsLocal)
            {
                params.mPtrPermutedIdxToExpandedIdx[permutedIdx] = expandedIdx;
            }
            if (live && params.mPtrPermutedIdxToTokenIdx != nullptr && pairIsLocal)
            {
                params.mPtrPermutedIdxToTokenIdx[permutedIdx] = pairToken;
            }
        }
    }
    if (threadIdx.x == 0)
    {
        *epochPtr = epoch + 1u;
    }
    // After all routing outputs are written.
    cudaTriggerProgrammaticLaunchCompletion();
}
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))

template <int kNumTokens>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
__global__ void __launch_bounds__(wide::kSliceThreads, 1)
    routingFusedRouterWideKernel(FusedRouterParams const __grid_constant__ p)
{
    using namespace wide;
    extern __shared__ __align__(128) uint8_t smemWide[];
    uint8_t* const smem = smemWide;
    float* const buffers = reinterpret_cast<float*>(p.workspace);
    uint32_t* const epochPtr = reinterpret_cast<uint32_t*>(p.workspace + 2 * kExchangeFloats);
    uint32_t const rank = blockIdx.x % kSliceCluster;
    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
    int32_t const laneIdx = cutlass::arch::LaneId();

    if (blockIdx.x < kNumGemvCtas)
    {
        // First phase of the cluster barrier: arrive now, wait before the first remote store, so that every CTA of the
        // cluster has started by then.
        clusterArriveRelaxed();
        constexpr int kPadded = slicePaddedTokens(kNumTokens);
        constexpr int kPairs = kExpertsPerCta * kNumTokens;
        uint4* act = reinterpret_cast<uint4*>(smem);
        float* fold = reinterpret_cast<float*>(smem);
        float* sliceSums = reinterpret_cast<float*>(smem + sliceWorkBytes(kNumTokens));
        int const slice = static_cast<int>(rank);
        int const expertBase = blockIdx.x / kSliceCluster * kExpertsPerCta;
        // Position t of the slice (router_gemm_kernel's lane) and expert of this thread; the 4 experts of a warp read
        // the same 128 bytes of activations.
        int const t = 8 * (warpIdx % 4) + laneIdx % 8;
        int const eLocal = 4 * (warpIdx / 4) + laneIdx / 8;
        __nv_bfloat16 const* wRow = p.routerWeight + (expertBase + eLocal) * kHiddenDim + (WarpSize * slice + t) * 8;
        // The weights are not written by the primary grid: load them before the wait. A shared-memory store of their
        // XOR ahead of a barrier keeps the loads before the wait.
        uint4 w[kKIters];
        uint32_t loaded = 0;
#pragma unroll
        for (int ki = 0; ki < kKIters; ki++)
        {
            w[ki] = loadWeight16(wRow + ki * kKIterElts);
            loaded ^= w[ki].x ^ w[ki].y ^ w[ki].z ^ w[ki].w;
        }
        reinterpret_cast<uint32_t*>(fold)[threadIdx.x] = loaded;
        __syncthreads();
        gridDependencyWait();
        // The previous launch on this workspace precedes the primary grid in the stream, so it has completed.
        uint32_t const epoch = slice == 0 ? loadVolatile(epochPtr) : 0u;
        // Stage the slice: vector (m, ki, pos) holds token m's elements 1024 ki + 8 (32 slice + pos) + 0..7.
#pragma unroll
        for (int group = 0; group < kSliceStageGroups; group++)
        {
            constexpr int kVecs = kNumTokens * kSliceKItersPerGroup * WarpSize;
            for (int c = threadIdx.x; c < kVecs; c += kSliceThreads)
            {
                int const m = c / (kSliceKItersPerGroup * WarpSize);
                int const rem = c - m * (kSliceKItersPerGroup * WarpSize);
                int const ki = group * kSliceKItersPerGroup + rem / WarpSize;
                int const pos = rem % WarpSize;
                copyAsync16(&act[(m * kKIters + ki) * WarpSize + pos],
                    p.hiddenStates + static_cast<size_t>(m) * kHiddenDim + ki * kKIterElts
                        + (WarpSize * slice + pos) * 8);
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
        }
        // The per-thread chains of router_gemm_kernel's thread 32 slice + t, for one expert and every token.
        float acc[kPadded];
#pragma unroll
        for (int m = 0; m < kPadded; m++)
        {
            acc[m] = 0.f;
        }
#pragma unroll
        for (int group = 0; group < kSliceStageGroups; group++)
        {
            if (group == 0)
            {
                copyAsyncWait<kSliceStageGroups - 1>();
            }
            else if (group == 1)
            {
                copyAsyncWait<kSliceStageGroups - 2>();
            }
            else
            {
                copyAsyncWait<0>();
            }
            __syncthreads();
#pragma unroll
            for (int kk = 0; kk < kSliceKItersPerGroup; kk++)
            {
                int const ki = group * kSliceKItersPerGroup + kk;
#pragma unroll
                for (int m = 0; m < kNumTokens; m++)
                {
                    fmaBf16x8(acc[m], act[(m * kKIters + ki) * WarpSize + t], w[ki]);
                }
            }
        }
        // Every thread is done with the activations; the butterfly staging reuses their space.
        __syncthreads();
        {
            float4* row = reinterpret_cast<float4*>(fold + (t * kExpertsPerCta + eLocal) * kPadded);
#pragma unroll
            for (int q = 0; q < kPadded / 4; q++)
            {
                row[q] = make_float4(acc[4 * q], acc[4 * q + 1], acc[4 * q + 2], acc[4 * q + 3]);
            }
        }
        __syncthreads();
        clusterWait();
        // The butterfly of router_gemm_kernel's warp: 4 threads per (expert, token) fold the positions
        // {sub, sub + 4, ..., sub + 28} through levels 16, 8 and 4 from shared memory, two shuffles add levels 2, 1.
#pragma unroll
        for (int base = 0; base < kPairs; base += kSliceThreads / 4)
        {
            int const pairRaw = base + threadIdx.x / 4;
            bool const valid = pairRaw < kPairs;
            int const pair = valid ? pairRaw : 0;
            int const sub = threadIdx.x % 4;
            int const eL = pair / kNumTokens;
            int const m = pair % kNumTokens;
            auto const v = [&](int position) { return fold[(position * kExpertsPerCta + eL) * kPadded + m]; };
            float const s1a = v(sub) + v(sub + 16);
            float const s1b = v(sub + 4) + v(sub + 20);
            float const s1c = v(sub + 8) + v(sub + 24);
            float const s1d = v(sub + 12) + v(sub + 28);
            float sum = (s1a + s1c) + (s1b + s1d);
            sum += __shfl_xor_sync(0xffffffff, sum, 2);
            sum += __shfl_xor_sync(0xffffffff, sum, 1);
            if (valid && sub == 0)
            {
                storeToClusterRank(&sliceSums[slice * kPairs + pair], 0, sum);
            }
        }
        // Second phase: rank 0 waits for the slice sums; the other CTAs are done (nothing reads their shared memory).
        clusterArriveRelease();
        if (slice != 0)
        {
            return;
        }
        clusterWaitAcquire();
        float* const exchange = buffers + (epoch & 1u) * kExchangeFloats;
        for (int pr = threadIdx.x; pr < kPairs; pr += kSliceThreads)
        {
            float finalSum = 0.0f;
#pragma unroll
            for (int s = 0; s < kSliceCluster; s++)
            {
                finalSum += sliceSums[s * kPairs + pr];
            }
            storeRelaxed(&exchange[(pr % kNumTokens) * kNumExperts + expertBase + pr / kNumTokens], finalSum);
        }
        // Exit without a trigger: the dependents launch once the routing cluster's rank 0 triggers.
        return;
    }
    routingClusterTail<kNumTokens, kSliceThreads, kSliceCluster, kSliceMaxTokens, kSliceRowBytes>(
        p, smem, rank, warpIdx, laneIdx, buffers, epochPtr);
}
#else
__global__ void __launch_bounds__(wide::kSliceThreads, 1)
    routingFusedRouterWideKernel(FusedRouterParams const __grid_constant__ /* p */)
{
    assert(false && "routingFusedRouterWideKernel is only supported on SM100+ architectures");
}
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))

template <int kNumTokens>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
__global__ void __launch_bounds__(wide::kMmaThreads, 1)
    routingFusedRouterMmaKernel(FusedRouterParams const __grid_constant__ p)
{
    using namespace wide;
    extern __shared__ __align__(128) uint8_t smemMmaFused[];
    uint8_t* const smem = smemMmaFused;
    float* const buffers = reinterpret_cast<float*>(p.workspace);
    uint32_t* const epochPtr = reinterpret_cast<uint32_t*>(p.workspace + 2 * kExchangeFloats);
    uint32_t const rank = blockIdx.x % kMmaCluster;
    int32_t const warpIdx = __shfl_sync(0xffffffff, threadIdx.x / WarpSize, 0);
    int32_t const laneIdx = cutlass::arch::LaneId();

    if (blockIdx.x < kMmaGemvCtas)
    {
        // First phase of the cluster barrier: arrive now, wait before the first remote store, so that every CTA of the
        // cluster has started by then.
        clusterArriveRelaxed();
        // router_gemm_mma_kernel, with the weight tile copied before the wait.
        constexpr int kPadded = mmaPaddedTokens(kNumTokens);
        constexpr int kTokenTiles = kPadded / 8;
        constexpr int kTile = kMmaExpertsPerCta * kPadded;
        // The weight tile [16][kMmaRowElems] and the activation rows [kPadded][kMmaRowElems] (bf16), the warp tiles
        // [kMmaWarps][16][kPadded] and, used in rank 0, the slice tiles [kMmaCluster][16][kPadded] (fp32).
        __nv_bfloat16* weightTile = reinterpret_cast<__nv_bfloat16*>(smem);
        __nv_bfloat16* actTile = weightTile + kMmaExpertsPerCta * kMmaRowElems;
        float* warpSums = reinterpret_cast<float*>(actTile + kPadded * kMmaRowElems);
        float* sliceSums = warpSums + kMmaWarps * kTile;
        int const slice = static_cast<int>(rank);
        int const expertBase = blockIdx.x / kMmaCluster * kMmaExpertsPerCta;
        // The primary grid does not write the weights.
        for (int c = threadIdx.x; c < kMmaExpertsPerCta * kMmaChunks; c += kMmaThreads)
        {
            int const row = c / kMmaChunks;
            int const chunk = c % kMmaChunks;
            copyAsync16(weightTile + row * kMmaRowElems + chunk * 8,
                p.routerWeight + static_cast<size_t>(expertBase + row) * kHiddenDim + slice * kMmaSliceK + chunk * 8);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        // Rows past the last token are zero; they only feed outputs that are not stored.
        for (int c = threadIdx.x; c < (kPadded - kNumTokens) * kMmaChunks; c += kMmaThreads)
        {
            *reinterpret_cast<uint4*>(actTile + (kNumTokens + c / kMmaChunks) * kMmaRowElems + (c % kMmaChunks) * 8)
                = make_uint4(0, 0, 0, 0);
        }
        gridDependencyWait();
        // The previous launch on this workspace precedes the primary grid in the stream, so it has completed.
        uint32_t const epoch = slice == 0 ? loadVolatile(epochPtr) : 0u;
        for (int c = threadIdx.x; c < kNumTokens * kMmaChunks; c += kMmaThreads)
        {
            int const row = c / kMmaChunks;
            int const chunk = c % kMmaChunks;
            copyAsync16(actTile + row * kMmaRowElems + chunk * 8,
                p.hiddenStates + static_cast<size_t>(row) * kHiddenDim + slice * kMmaSliceK + chunk * 8);
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        copyAsyncWait<0>();
        __syncthreads();

        float acc[kTokenTiles][4] = {};
#pragma unroll
        for (int s = 0; s < kMmaKStepsPerWarp; s++)
        {
            int const k0 = (warpIdx * kMmaKStepsPerWarp + s) * 16;
            uint32_t a[4];
            ldmatrixX4(a, weightTile + (laneIdx % 16) * kMmaRowElems + k0 + (laneIdx / 16) * 8);
#pragma unroll
            for (int j = 0; j < kTokenTiles; j++)
            {
                uint32_t b[2];
                ldmatrixX2(b, actTile + (j * 8 + laneIdx % 8) * kMmaRowElems + k0 + ((laneIdx / 8) % 2) * 8);
                mmaBf16(acc[j], a, b);
            }
        }
        // m16n8 accumulators: acc[j][0..1] at expert lane / 4, tokens 8 j + 2 (lane % 4) + 0..1; acc[j][2..3] 8 experts
        // on.
#pragma unroll
        for (int j = 0; j < kTokenTiles; j++)
        {
            float* row = warpSums + (warpIdx * kMmaExpertsPerCta + laneIdx / 4) * kPadded + j * 8 + 2 * (laneIdx % 4);
            *reinterpret_cast<float2*>(row) = make_float2(acc[j][0], acc[j][1]);
            *reinterpret_cast<float2*>(row + 8 * kPadded) = make_float2(acc[j][2], acc[j][3]);
        }
        __syncthreads();
        clusterWait();
        for (int i = threadIdx.x; i < kTile; i += kMmaThreads)
        {
            if (i % kPadded < kNumTokens)
            {
                float const sum
                    = ((warpSums[i] + warpSums[kTile + i]) + warpSums[2 * kTile + i]) + warpSums[3 * kTile + i];
                storeToClusterRank(&sliceSums[slice * kTile + i], 0, sum);
            }
        }
        // Second phase: rank 0 waits for the slices; the other CTAs are done (nothing reads their shared memory).
        clusterArriveRelease();
        if (slice != 0)
        {
            return;
        }
        clusterWaitAcquire();
        float* const exchange = buffers + (epoch & 1u) * kExchangeFloats;
        for (int pr = threadIdx.x; pr < kMmaExpertsPerCta * kNumTokens; pr += kMmaThreads)
        {
            int const e = pr / kNumTokens;
            int const m = pr % kNumTokens;
            float sum = sliceSums[e * kPadded + m];
#pragma unroll
            for (int r = 1; r < kMmaCluster; r++)
            {
                sum += sliceSums[r * kTile + e * kPadded + m];
            }
            storeRelaxed(&exchange[m * kNumExperts + expertBase + e], isNegZero(sum) ? 0.f : sum);
        }
        // Exit without a trigger: the dependents launch once the routing cluster's rank 0 triggers.
        return;
    }
    routingClusterTail<kNumTokens, kMmaThreads, kMmaCluster, kMmaMaxTokens, kMmaRowBytes>(
        p, smem, rank, warpIdx, laneIdx, buffers, epochPtr);
}
#else
__global__ void __launch_bounds__(wide::kMmaThreads, 1)
    routingFusedRouterMmaKernel(FusedRouterParams const __grid_constant__ /* p */)
{
    assert(false && "routingFusedRouterMmaKernel is only supported on SM100+ architectures");
}
#endif // if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))

template <typename Kernel>
void launchClusterGrid(Kernel kernel, FusedRouterParams const& params, int gridDim, int blockDim, int clusterDim,
    int smemBytes, bool usePdl, cudaStream_t stream)
{
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smemBytes));
    cudaLaunchConfig_t config{};
    config.gridDim = gridDim;
    config.blockDim = blockDim;
    config.dynamicSmemBytes = smemBytes;
    config.stream = stream;
    cudaLaunchAttribute attributes[2] = {};
    attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attributes[0].val.programmaticStreamSerializationAllowed = int(usePdl);
    attributes[1].id = cudaLaunchAttributeClusterDimension;
    attributes[1].val.clusterDim.x = clusterDim;
    attributes[1].val.clusterDim.y = 1;
    attributes[1].val.clusterDim.z = 1;
    config.attrs = attributes;
    config.numAttrs = 2;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, params));
}

template <int kNumTokens>
void launchFusedRouterWide(FusedRouterParams const& params, bool usePdl, cudaStream_t stream)
{
    // 128 GEMV CTAs and one routing cluster.
    launchClusterGrid(routingFusedRouterWideKernel<kNumTokens>, params, kNumGemvCtas + wide::kSliceCluster,
        wide::kSliceThreads, wide::kSliceCluster, wide::sliceSmemBytes(kNumTokens), usePdl, stream);
}

template <int kNumTokens>
void launchFusedRouterMma(FusedRouterParams const& params, bool usePdl, cudaStream_t stream)
{
    // 128 GEMV CTAs and one routing cluster.
    launchClusterGrid(routingFusedRouterMmaKernel<kNumTokens>, params, wide::kMmaGemvCtas + wide::kMmaCluster,
        wide::kMmaThreads, wide::kMmaCluster, wide::mmaSmemBytes(kNumTokens), usePdl, stream);
}

template <int kNumTokens>
void launchFusedRouter(FusedRouterParams const& params, bool usePdl, cudaStream_t stream)
{
    cudaLaunchConfig_t config{};
    config.gridDim = kNumGemvCtas + 1;
    config.blockDim = kNumThreads;
    config.dynamicSmemBytes = kSmemBytes;
    config.stream = stream;
    cudaLaunchAttribute attributes[1] = {};
    attributes[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attributes[0].val.programmaticStreamSerializationAllowed = int(usePdl);
    config.attrs = attributes;
    config.numAttrs = 1;
    TLLM_CUDA_CHECK(cudaLaunchKernelEx(&config, routingFusedRouterKernel<kNumTokens>, params));
}

bool supportsFusedRouter(Data const& data, int32_t hiddenDim)
{
    // 1-8 tokens: at most one CTA tile per expert. 9-24 tokens: any tile size.
    bool const tokensOk
        = (data.mNumTokens >= 1 && data.mNumTokens <= kMaxTokens && data.mNumTokens <= data.mTileTokensDim)
        || (data.mNumTokens >= wide::kSliceMinTokens && data.mNumTokens <= wide::kSliceMaxTokens)
        || (data.mNumTokens >= wide::kMmaMinTokens && data.mNumTokens <= wide::kMmaMaxTokens);
    return hiddenDim == kHiddenDim && data.mNumExperts == kNumExperts && data.mTopK == kTopK && tokensOk
        && data.mLocalExpertsStrideLog2 == 0 && data.mPreprocessType == RoutingPreprocessType::SigmoidBias
        && data.mPostprocessType == RoutingPostprocessType::ScaledSumNormalize
        && data.mDtypeOutput == tg::Dtype::Bfloat16 && data.mPtrTopKWeights != nullptr
        && data.mPtrRoutingBias != nullptr && data.mPtrTopKIds == nullptr;
}

void runFusedRouter(Data const& data, void const* hiddenStates, void const* routerWeight, int32_t hiddenDim,
    int32_t* workspace, void* stream)
{
    TLLM_CHECK_WITH_INFO(supportsFusedRouter(data, hiddenDim),
        "The fused router + routing kernel supports 1-8 tokens (at most the tile size) or 9-24 tokens, 256 experts, "
        "top-8, hidden size %d and SigmoidBias + ScaledSumNormalize routing; got %d tokens, tile %d, %d experts, "
        "top-%d, hidden %d",
        kHiddenDim, data.mNumTokens, data.mTileTokensDim, data.mNumExperts, data.mTopK, hiddenDim);
    FusedRouterParams params;
    params.routing = FusedKernelParams::setKernelParams(data);
    params.hiddenStates = static_cast<__nv_bfloat16 const*>(hiddenStates);
    params.routerWeight = static_cast<__nv_bfloat16 const*>(routerWeight);
    params.workspace = workspace;
    auto const s = static_cast<cudaStream_t>(stream);
    switch (data.mNumTokens)
    {
    case 1: launchFusedRouter<1>(params, data.mUsePdl, s); break;
    case 2: launchFusedRouter<2>(params, data.mUsePdl, s); break;
    case 3: launchFusedRouter<3>(params, data.mUsePdl, s); break;
    case 4: launchFusedRouter<4>(params, data.mUsePdl, s); break;
    case 5: launchFusedRouter<5>(params, data.mUsePdl, s); break;
    case 6: launchFusedRouter<6>(params, data.mUsePdl, s); break;
    case 7: launchFusedRouter<7>(params, data.mUsePdl, s); break;
    case 8: launchFusedRouter<8>(params, data.mUsePdl, s); break;
    case 9: launchFusedRouterWide<9>(params, data.mUsePdl, s); break;
    case 10: launchFusedRouterWide<10>(params, data.mUsePdl, s); break;
    case 11: launchFusedRouterWide<11>(params, data.mUsePdl, s); break;
    case 12: launchFusedRouterWide<12>(params, data.mUsePdl, s); break;
    case 13: launchFusedRouterWide<13>(params, data.mUsePdl, s); break;
    case 14: launchFusedRouterWide<14>(params, data.mUsePdl, s); break;
    case 15: launchFusedRouterWide<15>(params, data.mUsePdl, s); break;
    case 16: launchFusedRouterWide<16>(params, data.mUsePdl, s); break;
    case 17: launchFusedRouterMma<17>(params, data.mUsePdl, s); break;
    case 18: launchFusedRouterMma<18>(params, data.mUsePdl, s); break;
    case 19: launchFusedRouterMma<19>(params, data.mUsePdl, s); break;
    case 20: launchFusedRouterMma<20>(params, data.mUsePdl, s); break;
    case 21: launchFusedRouterMma<21>(params, data.mUsePdl, s); break;
    case 22: launchFusedRouterMma<22>(params, data.mUsePdl, s); break;
    case 23: launchFusedRouterMma<23>(params, data.mUsePdl, s); break;
    default: launchFusedRouterMma<24>(params, data.mUsePdl, s); break;
    }
}

} // namespace
} // namespace routingCustom
} // namespace moe::dev::routing

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{

void invokeFusedRouter(void const* routerHiddenStates, void const* routerWeight, int32_t hiddenDim,
    int32_t* routerWorkspace, void* routingBias, batchedGemm::trtllm::gen::Dtype dtypeRoutingBias, int32_t numTokens,
    int32_t numExperts, int32_t topK, int32_t localExpertOffset, int32_t localNumExperts, float routedScalingFactor,
    int32_t tileTokensDim, int32_t* permutedIdxSize, int32_t* expandedIdxToPermutedIdx, int32_t* permutedIdxToTokenIdx,
    void* expertWeights, int32_t* ctaIdxXyToBatchIdx, int32_t* ctaIdxXyToMnLimit, int32_t* numNonExitingCtas,
    cudaStream_t stream)
{
    int32_t paddingLog2 = 0;
    while ((1 << paddingLog2) < tileTokensDim)
    {
        ++paddingLog2;
    }
    TLLM_CHECK_WITH_INFO(
        (1 << paddingLog2) == tileTokensDim, "tileTokensDim must be a power of 2, got %d", tileTokensDim);

    // The routing configuration of the TRTLLM-Gen MoE runner's DeepSeek-V3 routing without groups.
    ::moe::dev::routing::routingCustom::Data routingData;
    routingData.mDtypeOutput = batchedGemm::trtllm::gen::Dtype::Bfloat16;
    routingData.mDtypeInput = batchedGemm::trtllm::gen::Dtype::Fp32;
    routingData.mUsePdl = tensorrt_llm::common::getEnvEnablePDL();
    routingData.mPreprocessType = ::moe::dev::routing::RoutingPreprocessType::SigmoidBias;
    routingData.mPostprocessType = ::moe::dev::routing::RoutingPostprocessType::ScaledSumNormalize;
    routingData.mPtrRoutingBias = routingBias;
    routingData.mDtypeBias = dtypeRoutingBias;
    routingData.mRouteScale = routedScalingFactor;
    routingData.mPtrPermutedIdxSize = permutedIdxSize;
    routingData.mPtrExpandedIdxToPermutedIdx = expandedIdxToPermutedIdx;
    routingData.mPtrPermutedIdxToExpandedIdx = nullptr;
    routingData.mPtrPermutedIdxToTokenIdx = permutedIdxToTokenIdx;
    routingData.mPtrTopKWeights = expertWeights;
    routingData.mPtrCtaIdxXyToBatchIdx = ctaIdxXyToBatchIdx;
    routingData.mPtrCtaIdxXyToMnLimit = ctaIdxXyToMnLimit;
    routingData.mPtrNumNonExitingCtas = numNonExitingCtas;
    routingData.mNumTokens = numTokens;
    routingData.mNumExperts = numExperts;
    routingData.mTopK = topK;
    routingData.mPaddingLog2 = paddingLog2;
    routingData.mTileTokensDim = tileTokensDim;
    routingData.mLocalExpertsStartIdx = localExpertOffset;
    routingData.mLocalExpertsStrideLog2 = 0;
    routingData.mNumLocalExperts = localNumExperts;

    ::moe::dev::routing::routingCustom::runFusedRouter(
        routingData, routerHiddenStates, routerWeight, hiddenDim, routerWorkspace, stream);
}

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
