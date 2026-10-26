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
// Logits. Every logit is the value dsv3RouterGemm.cu's router_gemm_kernel computes (bit for bit). There, thread t
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
#include "tensorrt_llm/kernels/glmMoeDsa/glmMoeFusedRouter.h"
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
    return hiddenDim == kHiddenDim && data.mNumExperts == kNumExperts && data.mTopK == kTopK && data.mNumTokens >= 1
        && data.mNumTokens <= kMaxTokens && data.mNumTokens <= data.mTileTokensDim && data.mLocalExpertsStrideLog2 == 0
        && data.mPreprocessType == RoutingPreprocessType::SigmoidBias
        && data.mPostprocessType == RoutingPostprocessType::ScaledSumNormalize
        && data.mDtypeOutput == tg::Dtype::Bfloat16 && data.mPtrTopKWeights != nullptr
        && data.mPtrRoutingBias != nullptr && data.mPtrTopKIds == nullptr;
}

void runFusedRouter(Data const& data, void const* hiddenStates, void const* routerWeight, int32_t hiddenDim,
    int32_t* workspace, void* stream)
{
    TLLM_CHECK_WITH_INFO(supportsFusedRouter(data, hiddenDim),
        "The fused router + routing kernel supports 1-8 tokens (at most the tile size), 256 experts, top-8, hidden "
        "size %d and SigmoidBias + ScaledSumNormalize routing; got %d tokens, tile %d, %d experts, top-%d, hidden %d",
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
    default: launchFusedRouter<8>(params, data.mUsePdl, s); break;
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
