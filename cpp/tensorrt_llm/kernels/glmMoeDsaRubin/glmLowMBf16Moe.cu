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

#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmLowMBf16Moe.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/moeTopKFuncs.cuh"

#include <algorithm>
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstring>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{
namespace
{
namespace cg = cooperative_groups;

using Bf16 = __nv_bfloat16;

constexpr int kNumExperts = kLowMBf16MoeNumExperts;
constexpr int kTopK = kLowMBf16MoeTopK;
static_assert(kNumExperts % 32 == 0, "routing gives each lane of one warp kNumExperts / 32 candidates");

// FC1: 8 consumer warps and one producer warp per CTA. Consumer thread t reads the 16-byte pieces t,
// t + kFc1ConsumerThreads, ... of x and of every weight row, so the hidden size is a multiple of kHiddenGranule and at
// most kMaxPiecesPerThread of them.
constexpr int kFc1ConsumerWarps = 8;
constexpr int kFc1ConsumerThreads = kFc1ConsumerWarps * 32;
constexpr int kFc1Threads = kFc1ConsumerThreads + 32;
constexpr int kPieceElems = 8;
constexpr int kHiddenGranule = kFc1ConsumerThreads * kPieceElems;
constexpr int kMaxPiecesPerThread = 4;
// The FC1 ring holds whole weight rows, as many as fit in kFc1RingBytes and at most kMaxStages. At hidden size 6144
// and interSize 256 that is 7 rows of 12,288 B, and an FC1 CTA (ring, row sums, barriers and the 1,024 B the runtime
// reserves per CTA) takes 87,840 B, so one FC2 CTA (138,312 B) fits beside it in the 233,472 B of an SM.
constexpr int kFc1RingBytes = 7 * 6144 * static_cast<int>(sizeof(Bf16));
constexpr int kMaxStages = 8;

// FC2: a warp per routed expert.
constexpr int kFc2Threads = 256;
constexpr int kFc2Warps = kFc2Threads / 32;
static_assert(kTopK == kFc2Warps, "the FC2 kernel gives each routed expert one warp");

// FC2: mma.sync m16n8k16 tiles have 16 matrix rows; one 32-element K step is two mma calls (see mmaStep). A warp per
// routed expert walks the intermediate dimension in pairs of K steps, one per accumulator.
constexpr int kMmaRows = 16;
constexpr int kStepK = 32;
constexpr int kFc2StepPairK = 2 * kStepK;
// The largest intermediate size per rank whose FC2 working set fits in shared memory (kTiles = 1).
constexpr int kMaxInterSize = 512;

// The routed scaling factor as the FC2 kernel takes it: the double value of noaux_tc_op's FP64 weight expression, its
// float value, and whether routingWeight's FP32 path applies to it (decided on the host, see makeRoutingScale).
struct RoutingScale
{
    double value;
    float valueFloat;
    int fastPath;
};

// Multi-token calls (invokeLowMBf16MoeMulti). The FC1 kernel keeps x of every token in shared memory, of which a
// consumer thread reads its at most kMultiMaxPiecesPerThread pieces per token (hidden size 6144), and forms the SwiGLU
// outputs every kMultiChunkUnits units from per-warp row sums held in two alternating buffers. Its ring has between
// kMultiMinFc1Stages (where the shared memory of the SM allows) and kMultiMaxFc1Stages rows.
constexpr int kMultiMaxTokens = kLowMBf16MoeMaxTokens;
constexpr int kMultiMaxPiecesPerThread = 3;
constexpr int kMultiChunkUnits = 8;
constexpr int kMultiChunkRows = 2 * kMultiChunkUnits;
constexpr int kMultiMinFc1Stages = 7;
constexpr int kMultiMaxFc1Stages = 12;
// The multi-token FC1 kernel's warps: the consumer warps of lowMBf16MoeFc1Kernel, its producer warp, and an epilogue
// warp that forms the SwiGLU outputs of each chunk while the consumer warps go on with the next.
constexpr int kMultiFc1EpilogueWarp = kFc1ConsumerWarps + 1;
constexpr int kMultiFc1Threads = kFc1ConsumerThreads + 64;
// FC2: kMultiFc2Tiles 16-row tiles per CTA, one consumer warp per (union expert, tile) and one producer warp that
// streams the union experts' weight slices through a ring of at most kMultiFc2MaxStages stages. One tile per CTA makes
// CTAs small enough that one fits beside an FC1 CTA and two share an SM once the FC1 CTAs exit.
constexpr int kMultiFc2Tiles = 1;
constexpr int kMultiFc2Rows = kMultiFc2Tiles * kMmaRows;
constexpr int kMultiFc2ConsumerWarps = 8;
constexpr int kMultiFc2ConsumerThreads = kMultiFc2ConsumerWarps * 32;
constexpr int kMultiFc2Threads = kMultiFc2ConsumerThreads + 32;
constexpr int kMultiFc2MaxStages = 8;
// An m16n8k16 mma of the FC2 kernel takes four tokens, two B columns each (see lowMBf16MoeMultiFc2Kernel).
constexpr int kMultiTokensPerMma = 4;
static_assert(kMultiFc2ConsumerWarps % kMultiFc2Tiles == 0, "the consumer warps take whole union experts");
static_assert(kMultiMaxTokens <= 2 * kMultiTokensPerMma, "the FC2 kernel takes at most two mma token groups");

// The routing of a multi-token call as both multi-token kernels hold it in shared memory: the experts each token
// selected, and their union in ascending expert order with, per union expert, the tokens that selected it and its slot
// in each of their lists.
template <int kTokens>
struct MultiRouting
{
    // Bit m: token m selected the expert.
    uint32_t expertTokens[kNumExperts];
    // Union position of each selected expert.
    uint8_t unionPos[kNumExperts];
    int32_t topExperts[kTokens][kTopK];
    int32_t unionExperts[kTokens * kTopK];
    uint32_t unionTokens[kTokens * kTopK];
    // Slot of the union expert in token m's list, where bit m of its unionTokens is set.
    uint8_t unionSlots[kTokens * kTopK][kTokens];
    int32_t numUnion;
};

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
constexpr int kCandidates = kNumExperts / 32;
constexpr int kFc1ProducerWarp = kFc1ConsumerWarps;
// Named barrier at which the FC1 consumer warps wait while the producer selects the experts and issues the first rows.
constexpr int kFc1ParkBarrier = 1;
// The FC2 kernel holds its routing computation until this long after its entry; see lowMBf16MoeFc2Kernel.
constexpr uint64_t kFc2RouteHoldNs = 1000;

using TopKKey = reduce_topk::TopKRedType<float>;

// Same expression as sigmoid_accurate in noAuxTcKernels.cu.
__device__ __forceinline__ float sigmoidAccurate(float x)
{
    return 0.5f * tanhf(0.5f * x) + 0.5f;
}

__device__ __forceinline__ uint32_t warpMax(uint32_t value)
{
    uint32_t result;
    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;\n" : "=r"(result) : "r"(value));
    return result;
}

// The correction bias of the experts a lane holds (i * 32 + lane) as float. A constant, so the kernels may read it
// before their grid-dependency wait.
template <typename BiasT>
__device__ __forceinline__ void loadBias(BiasT const* bias, int lane, float (&values)[kCandidates])
{
#pragma unroll
    for (int i = 0; i < kCandidates; ++i)
    {
        values[i] = static_cast<float>(bias[i * 32 + lane]);
    }
}

// The routing keys of noaux_tc_op for n_group == 1 (deepseek_v3_topk_kernel, single group) of the experts a lane
// holds: the sigmoid score plus the correction bias, with the expressions of that kernel, packed with the expert index
// as reduceTopK (moeTopKFuncs.cuh) packs them. Stores the sigmoid scores when sigmoidScores is not null. The logits
// are read through L2.
__device__ __forceinline__ void buildKeys(float const* routerLogits, float const (&biasValues)[kCandidates], int lane,
    TopKKey (&keys)[kCandidates], float* sigmoidScores)
{
#pragma unroll
    for (int i = 0; i < kCandidates; ++i)
    {
        int const expert = i * 32 + lane;
        float const sigmoid = sigmoidAccurate(__ldcg(routerLogits + expert));
        if (sigmoidScores != nullptr)
        {
            sigmoidScores[expert] = sigmoid;
        }
        keys[i] = TopKKey{sigmoid + biasValues[i], expert};
    }
}

// The top kTopK of one full warp's keys in the order of reduceTopK: descending score, on equal scores the lower index
// first. The per-lane sort is reduceTopK's; each round then takes the warp maximum of the lanes' leading keys with two
// 32-bit reductions, of the score half and then of the index half among the lanes holding the maximum score, which is
// the maximum of reduceTopK's 64-bit reduction. So the experts and their order are noaux_tc_op's. Returns the k-th
// expert on lane k < kTopK, 0 on the other lanes.
__device__ __forceinline__ int32_t selectTopK(TopKKey (&keys)[kCandidates], int lane)
{
    reduce_topk::Sort<kCandidates, TopKKey>::run(keys);

    int32_t laneExpert = 0;
    uint64_t packedMax = 0;
#pragma unroll
    for (int k = 0; k < kTopK; ++k)
    {
        if (k > 0 && packedMax == keys[0].compValIdx)
        {
            // This lane held the previous maximum: drop it, as reduceTopK does.
#pragma unroll
            for (int i = 0; i + 1 < kCandidates; ++i)
            {
                keys[i] = keys[i + 1];
            }
            keys[kCandidates - 1] = TopKKey{-INFINITY, (kCandidates - 1) * 32 + lane};
        }
        uint32_t const scoreBits = static_cast<uint32_t>(keys[0].compValIdx >> 32);
        uint32_t const maxScoreBits = warpMax(scoreBits);
        uint32_t const indexBits = warpMax(scoreBits == maxScoreBits ? static_cast<uint32_t>(keys[0].compValIdx) : 0U);
        packedMax = (static_cast<uint64_t>(maxScoreBits) << 32) | indexBits;
        if (lane == k)
        {
            laneExpert = TopKKey::kMaxIdx - static_cast<int32_t>(indexBits & 0xFFFF);
        }
    }
    return laneExpert;
}

// noaux_tc_op's routing weight, static_cast<float>(s * c / (S + 1e-20)) in double with s the expert's sigmoid score,
// S the sum of the selected scores and c the routed scaling factor, without FP64 instructions on the common path: FP64
// in the FC2 CTAs drops the SM clock for the whole FC1 window (measured on Rubin: 2.38 -> 1.6 GHz). When c is a float
// with at most 20 significant bits (fastPath) and S >= 2^-13, the double expression is the float rounding of the exact
// quotient s * c / S: s * c is exact in double, S + 1e-20 rounds to S, and rounding to double first cannot move a
// quotient of these operands across a float rounding boundary except at an exact midpoint, which rounds the same
// either way. The quotient here comes from an exact two-float product, a correctly rounded division and its exact
// remainder; it is returned when it lies more than 2^-20 ulp inside its rounding interval (and is not a power of two,
// where the interval is lopsided). Otherwise, and for scores below 2^-60 (where the FP32 intermediates could lose
// bits), the FP64 expression runs, out of line so that none of it is scheduled on the common path.
__device__ __noinline__ float routingWeightFp64(float s, float S, double c)
{
    return static_cast<float>(s * c / (S + 1e-20));
}

__device__ __forceinline__ float routingWeight(float s, float S, RoutingScale const& scale)
{
    if (scale.fastPath != 0 && S >= 0x1p-13f && s >= 0x1p-60f)
    {
        float const c = scale.valueFloat;
        float const pHi = __fmul_rn(s, c);
        float const pLo = __fmaf_rn(s, c, -pHi);
        float const q = __fdiv_rn(pHi, S);
        float const r = __fmaf_rn(-q, S, pHi);
        float const e = __fdiv_rn(__fadd_rn(r, pLo), S);
        float const y = __fadd_rn(q, e);
        uint32_t const bits = __float_as_uint(y);
        if (y > 0.F && (bits & 0x7FFFFFU) != 0 && (bits & 0x7F800000U) != 0x7F800000U)
        {
            float const ulp = __uint_as_float(bits & 0x7F800000U) * 0x1p-23f;
            float const g = __fsub_rn(e, __fsub_rn(y, q));
            if (fabsf(g) < 0.5F * ulp - 0x1p-20f * ulp)
            {
                return y;
            }
        }
    }
    return routingWeightFp64(s, S, scale.value);
}

__device__ __forceinline__ uint32_t smemAddress(void const* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void mbarrierInit(uint64_t* bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(smemAddress(bar)), "r"(count) : "memory");
}

// Makes the barrier initialization visible to the bulk-copy (async) proxy.
__device__ __forceinline__ void fenceBarrierInit()
{
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

__device__ __forceinline__ void mbarrierArriveExpectTx(uint64_t* bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(smemAddress(bar)), "r"(bytes)
                 : "memory");
}

// Arrives on the barrier with release semantics: this thread's earlier shared-memory reads are done before the phase
// can complete.
__device__ __forceinline__ void mbarrierArrive(uint64_t* bar)
{
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(smemAddress(bar)) : "memory");
}

// Whether the barrier's phase with the given parity has completed (one try, which may suspend the thread for a
// while).
__device__ __forceinline__ bool mbarrierTryWait(uint64_t* bar, uint32_t parity)
{
    uint32_t done = 0;
    asm volatile(
        "{\n"
        ".reg .pred complete;\n"
        "mbarrier.try_wait.parity.shared::cta.b64 complete, [%1], %2;\n"
        "selp.u32 %0, 1, 0, complete;\n"
        "}\n"
        : "=r"(done)
        : "r"(smemAddress(bar)), "r"(parity)
        : "memory");
    return done != 0;
}

__device__ __forceinline__ void mbarrierWait(uint64_t* bar, uint32_t parity)
{
    while (!mbarrierTryWait(bar, parity))
    {
    }
}

// Waits at named barrier `id` until `threads` threads (whole warps) have arrived.
__device__ __forceinline__ void namedBarrierSync(int id, int threads)
{
    asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(threads) : "memory");
}

__device__ __forceinline__ uint64_t globalTimerNs()
{
    uint64_t now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    return now;
}

// Streamed weights are read once per call: mark their L2 lines evict-first.
__device__ __forceinline__ uint64_t evictFirstPolicy()
{
    uint64_t policy;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(policy));
    return policy;
}

// 1D bulk copy global -> shared that completes `bytes` transactions on `bar`.
__device__ __forceinline__ void bulkCopyToShared(
    void* dst, void const* src, uint32_t bytes, uint64_t* bar, uint64_t policy)
{
    uint64_t const srcGlobal = static_cast<uint64_t>(__cvta_generic_to_global(src));
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
        :
        : "r"(smemAddress(dst)), "l"(srcGlobal), "r"(bytes), "r"(smemAddress(bar)), "l"(policy)
        : "memory");
}

__device__ __forceinline__ void mmaBf16(
    float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// One 32-element K step of a 16-row tile times a vector. Lane (g = lane / 4, t = lane % 4) passes elements [8t, 8t + 8)
// of the step from tile row g (lo), tile row g + 8 (hi) and the vector (vec). The fragment's K columns
// {2t, 2t + 1, 2t + 8, 2t + 9} take the lane's elements {0, 1, 2, 3} in the first call and {4, 5, 6, 7} in the second:
// one K permutation for the matrix and the vector, which leaves the dot products unchanged. Every column of the B
// fragment is the vector, so all accumulator elements hold dot products: elements 0 and 1 of row g, 2 and 3 of row
// g + 8.
__device__ __forceinline__ void mmaStep(float (&acc)[4], uint4 const& lo, uint4 const& hi, uint4 const& vec)
{
    mmaBf16(acc, lo.x, hi.x, lo.y, hi.y, vec.x, vec.y);
    mmaBf16(acc, lo.z, hi.z, lo.w, hi.w, vec.z, vec.w);
}

__device__ __forceinline__ float roundToBf16(float value)
{
    return __bfloat162float(__float2bfloat16_rn(value));
}

// The FP32 values of the eight BF16 elements of a 16-byte piece, in memory order.
__device__ __forceinline__ void unpackPiece(uint4 const& piece, float (&values)[kPieceElems])
{
    uint32_t const words[4] = {piece.x, piece.y, piece.z, piece.w};
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        values[2 * i] = __uint_as_float(words[i] << 16);
        values[2 * i + 1] = __uint_as_float(words[i] & 0xFFFF0000U);
    }
}

// Dot product of a 16-byte piece of a BF16 weight row with the FP32 values of the same piece of x, in element order
// with FP32 fused multiply-adds.
__device__ __forceinline__ float dotPiece(uint4 const& piece, float const (&values)[kPieceElems])
{
    float weights[kPieceElems];
    unpackPiece(piece, weights);
    float dot = weights[0] * values[0];
#pragma unroll
    for (int e = 1; e < kPieceElems; ++e)
    {
        dot = fmaf(weights[e], values[e], dot);
    }
    return dot;
}

// The warp sums of two values, each by an xor butterfly over offsets 16, 8, 4, 2, 1, the two shuffle chains
// interleaved.
__device__ __forceinline__ void warpSum2(float& a, float& b)
{
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
    {
        a += __shfl_xor_sync(0xffffffffU, a, offset);
        b += __shfl_xor_sync(0xffffffffU, b, offset);
    }
}

constexpr int kMultiParkBarrier = 1;
constexpr int kMultiRouteBarrier = 2;
constexpr int kMultiConsumerBarrier = 3;

// Arrives at named barrier `id` of `threads` threads (whole warps) without waiting; the barrier orders this thread's
// earlier shared-memory writes before the waiting threads' later reads.
__device__ __forceinline__ void namedBarrierArrive(int id, int threads)
{
    asm volatile("bar.arrive %0, %1;\n" ::"r"(id), "r"(threads) : "memory");
}

// warpSum2's sums with half its shuffles: after the first exchange lanes 0-15 hold a's pairs and lanes 16-31 b's, and
// the next four levels run within each half. Lane 0 ends with warpSum2's a of lane 0 and lane 16 with its b of lane 0:
// the same additions in the same tree (each first pair added in the other operand order, which gives the same sum).
__device__ __forceinline__ float warpSum2Split(float a, float b, int lane)
{
    bool const upper = lane >= 16;
    float const received = __shfl_xor_sync(0xffffffffU, upper ? a : b, 16);
    float sum = upper ? b + received : a + received;
#pragma unroll
    for (int offset = 8; offset > 0; offset /= 2)
    {
        sum += __shfl_xor_sync(0xffffffffU, sum, offset);
    }
    return sum;
}

// dotPiece with x given as a packed piece of eight BF16 values (their FP32 values are exact).
__device__ __forceinline__ float dotPiecePacked(uint4 const& piece, uint4 const& xPiece)
{
    float values[kPieceElems];
    unpackPiece(xPiece, values);
    return dotPiece(piece, values);
}

// One warp builds the union of the tokens' selections (routing.expertTokens and routing.topExperts complete): the
// union experts in ascending expert order, the tokens of each and its slot in each of their lists.
template <int kTokens>
__device__ __forceinline__ void buildUnion(MultiRouting<kTokens>& routing, int lane)
{
    constexpr int kPerLane = kNumExperts / 32;
    uint32_t tokens[kPerLane];
    int count = 0;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i)
    {
        tokens[i] = routing.expertTokens[lane * kPerLane + i];
        count += tokens[i] != 0 ? 1 : 0;
    }
    // Inclusive prefix sum of the counts over the lanes.
    int prefix = count;
#pragma unroll
    for (int offset = 1; offset < 32; offset *= 2)
    {
        int const value = __shfl_up_sync(0xffffffffU, prefix, offset);
        if (lane >= offset)
        {
            prefix += value;
        }
    }
    int pos = prefix - count;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i)
    {
        if (tokens[i] != 0)
        {
            int const expert = lane * kPerLane + i;
            routing.unionExperts[pos] = expert;
            routing.unionTokens[pos] = tokens[i];
            routing.unionPos[expert] = static_cast<uint8_t>(pos);
            ++pos;
        }
    }
    if (lane == 31)
    {
        routing.numUnion = prefix;
    }
    __syncwarp();
    for (int p = lane; p < kTokens * kTopK; p += 32)
    {
        int const token = p / kTopK;
        int const slot = p - token * kTopK;
        routing.unionSlots[routing.unionPos[routing.topExperts[token][slot]]][token] = static_cast<uint8_t>(slot);
    }
    __syncwarp();
}
#endif

// FC1 + SwiGLU, one CTA per SM (the grid is the SM count). The work is kTopK * interSize units; unit k * interSize + j
// is intermediate index j of routed slot k, that is the expert's gate row (W1, w3w1 row interSize + j) and up row (W3,
// row j). CTA b takes units [b * U / G, (b + 1) * U / G) and streams their rows, gate then up, each as one bulk copy
// into a ring of numStages row buffers. The producer warp selects the experts and issues the rows; it refills a buffer
// as soon as every consumer warp has read its pieces of it, with no CTA-wide barrier per row. The consumer warps wait
// at a named barrier while the producer selects the experts and issues the first rows. Each consumer thread keeps its
// pieces of x in registers and adds its shares of the dot products of two rows at a time, so that the latency chains of
// the two overlap; each consumer warp reduces its shares with shuffles into shared memory, and each unit's SwiGLU adds
// the 8 warp sums in a fixed order.
template <typename BiasT>
__global__ void __launch_bounds__(kFc1Threads, 2)
    lowMBf16MoeFc1Kernel(Bf16* __restrict__ act, Bf16 const* __restrict__ x, float const* __restrict__ routerLogits,
        BiasT const* __restrict__ bias, Bf16 const* __restrict__ w3w1, int hiddenSize, int interSize, int numStages)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ __align__(128) uint8_t smem[];
    // rowFull[s]: buffer s holds its next row (the producer's arrival plus the copy's bytes). rowFree[s]: every
    // consumer warp has read its pieces of buffer s.
    __shared__ uint64_t rowFull[kMaxStages];
    __shared__ uint64_t rowFree[kMaxStages];
    __shared__ int32_t topExperts[kTopK];

    int const warp = threadIdx.x / 32;
    int const lane = threadIdx.x % 32;
    int const rowBytes = hiddenSize * static_cast<int>(sizeof(Bf16));
    int const numUnits = kTopK * interSize;
    int const unitBegin = static_cast<int>(static_cast<int64_t>(blockIdx.x) * numUnits / gridDim.x);
    int const unitEnd = static_cast<int>(static_cast<int64_t>(blockIdx.x + 1) * numUnits / gridDim.x);
    int const numRows = 2 * (unitEnd - unitBegin);
    uint8_t* ring = smem;
    float* rowSums = reinterpret_cast<float*>(smem + numStages * rowBytes);

    if (threadIdx.x == 0)
    {
        for (int s = 0; s < numStages; ++s)
        {
            mbarrierInit(&rowFull[s], 1);
            mbarrierInit(&rowFree[s], kFc1ConsumerWarps);
        }
        fenceBarrierInit();
    }
    // The correction bias is a constant: the producer reads it before the grid-dependency wait.
    float biasValues[kCandidates] = {};
    if (warp == kFc1ProducerWarp)
    {
        loadBias(bias, lane, biasValues);
    }
    __syncthreads();

    // x and the router logits are written by earlier kernels.
    cudaGridDependencySynchronize();
    // The FC2 kernel reads only the logits, the bias and its weights before its own wait, and the logits are complete
    // since the wait above returned. Released here, its CTAs take the room beside this kernel's and stage their
    // weights while this kernel streams.
    cudaTriggerProgrammaticLaunchCompletion();

    if (warp == kFc1ProducerWarp)
    {
        TopKKey keys[kCandidates];
        buildKeys(routerLogits, biasValues, lane, keys, nullptr);
        int32_t const expert = selectTopK(keys, lane);
        if (lane < kTopK)
        {
            topExperts[lane] = expert;
        }
        __syncwarp();

        // Lane 0 streams the rows. The next row is of unit (slot, index), the gate row first, into buffer s, which it
        // fills for the pass-th time.
        uint64_t const policy = evictFirstPolicy();
        int64_t const expertStride = 2 * static_cast<int64_t>(interSize) * hiddenSize;
        int slot = unitBegin / interSize;
        int index = unitBegin % interSize;
        bool gateRow = true;
        Bf16 const* expertRows = w3w1 + topExperts[slot] * expertStride;
        int s = 0;
        uint32_t pass = 0;
        auto issueRow = [&]()
        {
            int64_t const rowInExpert = gateRow ? interSize + index : index;
            mbarrierArriveExpectTx(&rowFull[s], rowBytes);
            bulkCopyToShared(ring + s * rowBytes, expertRows + rowInExpert * hiddenSize, rowBytes, &rowFull[s], policy);
            if (!gateRow && ++index == interSize)
            {
                // The up row closed the slot's last unit.
                index = 0;
                if (++slot < kTopK)
                {
                    expertRows = w3w1 + topExperts[slot] * expertStride;
                }
            }
            gateRow = !gateRow;
            if (++s == numStages)
            {
                s = 0;
                ++pass;
            }
        };
        int const firstRows = numStages < numRows ? numStages : numRows;
        if (lane == 0)
        {
            for (int row = 0; row < firstRows; ++row)
            {
                issueRow();
            }
        }
        // Release the consumer warps.
        __syncwarp();
        namedBarrierSync(kFc1ParkBarrier, kFc1Threads);
        if (lane == 0)
        {
            for (int row = firstRows; row < numRows; ++row)
            {
                // Buffer s still holds row - numStages until every consumer warp has read it.
                mbarrierWait(&rowFree[s], (pass - 1) & 1U);
                issueRow();
            }
        }
    }
    else
    {
        float xValues[kMaxPiecesPerThread][kPieceElems] = {};
        int const piecesPerThread = hiddenSize / kHiddenGranule;
#pragma unroll
        for (int i = 0; i < kMaxPiecesPerThread; ++i)
        {
            if (i < piecesPerThread)
            {
                unpackPiece(
                    __ldcg(reinterpret_cast<uint4 const*>(x) + threadIdx.x + i * kFc1ConsumerThreads), xValues[i]);
            }
        }
        // Wait at the barrier, without polling the ring, while the producer selects the experts and issues the first
        // rows.
        namedBarrierSync(kFc1ParkBarrier, kFc1Threads);
        int s0 = 0;
        uint32_t pass0 = 0;
        // Rows row and row + 1 (numRows is even) are in buffers s0 and s1 (numStages >= 2), filled for the pass0-th and
        // pass1-th time.
        for (int row = 0; row < numRows; row += 2)
        {
            int s1 = s0 + 1;
            uint32_t pass1 = pass0;
            if (s1 == numStages)
            {
                s1 = 0;
                ++pass1;
            }
            mbarrierWait(&rowFull[s0], pass0 & 1U);
            mbarrierWait(&rowFull[s1], pass1 & 1U);
            uint4 const* pieces0 = reinterpret_cast<uint4 const*>(ring + s0 * rowBytes);
            uint4 const* pieces1 = reinterpret_cast<uint4 const*>(ring + s1 * rowBytes);
            float dot0 = dotPiece(pieces0[threadIdx.x], xValues[0]);
            float dot1 = dotPiece(pieces1[threadIdx.x], xValues[0]);
#pragma unroll
            for (int i = 1; i < kMaxPiecesPerThread; ++i)
            {
                if (i < piecesPerThread)
                {
                    dot0 += dotPiece(pieces0[threadIdx.x + i * kFc1ConsumerThreads], xValues[i]);
                    dot1 += dotPiece(pieces1[threadIdx.x + i * kFc1ConsumerThreads], xValues[i]);
                }
            }
            // This warp is done with both buffers (the arrival releases its lanes' reads, ordered before it by the
            // warp barrier); the producer refills a buffer once every consumer warp is.
            __syncwarp();
            if (lane == 0)
            {
                mbarrierArrive(&rowFree[s0]);
                mbarrierArrive(&rowFree[s1]);
            }
            warpSum2(dot0, dot1);
            if (lane == 0)
            {
                rowSums[row * kFc1ConsumerWarps + warp] = dot0;
                rowSums[(row + 1) * kFc1ConsumerWarps + warp] = dot1;
            }
            s0 = s1 + 1;
            pass0 = pass1;
            if (s0 == numStages)
            {
                s0 = 0;
                ++pass0;
            }
        }
    }
    __syncthreads();

    for (int unit = static_cast<int>(threadIdx.x); unit < numRows / 2; unit += kFc1Threads)
    {
        float gate = 0.F;
        float up = 0.F;
        for (int w = 0; w < kFc1ConsumerWarps; ++w)
        {
            gate += rowSums[2 * unit * kFc1ConsumerWarps + w];
            up += rowSums[(2 * unit + 1) * kFc1ConsumerWarps + w];
        }
        // The CUTLASS FC1 GEMM writes its output in BF16 and the activation kernel reads it back.
        gate = roundToBf16(gate);
        up = roundToBf16(up);
        // silu(gate) * up in FP32 as the CUTLASS activation (GLUAdaptor<SiLu>) computes it, then BF16.
        float const silu = gate * (1.F / (1.F + expf(-gate)));
        act[unitBegin + unit] = __float2bfloat16_rn(silu * up);
    }
#endif
}

// FC2 + the weighted sum over the routed experts. Grid: one CTA per kTiles * 16 output rows; warp k computes routed
// expert k's terms from its weight slice, staged in shared memory before the grid-dependency wait. Warp 0 routes before
// that wait, from the router logits and the bias staged by bulk copies, without FP64 instructions. After the wait every
// warp stages its expert's SwiGLU outputs itself.
template <typename BiasT, int kTiles>
__global__ void __launch_bounds__(kFc2Threads, 2)
    lowMBf16MoeFc2Kernel(Bf16* __restrict__ out, Bf16 const* __restrict__ act, float const* __restrict__ routerLogits,
        BiasT const* __restrict__ bias, Bf16 const* __restrict__ w2, int hiddenSize, int interSize, RoutingScale scale)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    constexpr int kRows = kTiles * kMmaRows;
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(16) float logitsShared[kNumExperts];
    __shared__ __align__(16) BiasT biasShared[kNumExperts];
    __shared__ uint64_t inputsFull;
    __shared__ uint64_t slicesFull;
    __shared__ float sigmoidScores[kNumExperts];
    __shared__ int32_t topExperts[kTopK];
    __shared__ float topWeights[kTopK];
    __shared__ float terms[kTopK][kRows];

    uint64_t const entryNs = globalTimerNs();

    int const sliceElems = kRows * interSize;
    Bf16* slices = reinterpret_cast<Bf16*>(smem);
    Bf16* actShared = slices + kTopK * sliceElems;

    int const warp = threadIdx.x / 32;
    int const lane = threadIdx.x % 32;
    int const row0 = blockIdx.x * kRows;

    if (threadIdx.x == 0)
    {
        mbarrierInit(&inputsFull, 1);
        mbarrierInit(&slicesFull, 1);
        fenceBarrierInit();
    }
    __syncthreads();

    if (warp == 0)
    {
        // Before the grid-dependency wait this kernel reads only the router logits, the bias and the FC2 weights. The
        // logits come from the router GEMM, which completed before the FC1 kernel returned from its own wait and
        // released this grid; the bias and the weights are constants. The logits and the bias come in by bulk copies:
        // plain loads issued now queue behind the FC1 kernel's bulk copies on this SM for microseconds (measured).
        bool const bulkInputs
            = ((reinterpret_cast<uintptr_t>(routerLogits) | reinterpret_cast<uintptr_t>(bias)) % 16) == 0;
        if (bulkInputs)
        {
            if (lane == 0)
            {
                uint64_t const policy = evictFirstPolicy();
                mbarrierArriveExpectTx(&inputsFull, sizeof(logitsShared) + sizeof(biasShared));
                bulkCopyToShared(logitsShared, routerLogits, sizeof(logitsShared), &inputsFull, policy);
                bulkCopyToShared(biasShared, bias, sizeof(biasShared), &inputsFull, policy);
            }
            mbarrierWait(&inputsFull, 0);
        }
        else
        {
            // Each lane stages, and below reads back, only the experts it holds.
#pragma unroll
            for (int i = 0; i < kCandidates; ++i)
            {
                logitsShared[i * 32 + lane] = __ldcg(routerLogits + i * 32 + lane);
                biasShared[i * 32 + lane] = bias[i * 32 + lane];
            }
        }
        // The FC1 kernel's producer warps select the same experts on these SMs at about this time, on the FC1
        // kernel's critical path, and routing here concurrently makes their selection about twice as long (measured
        // 0.5 -> 1.1 us). So the routing waits until kFc2RouteHoldNs after this CTA's entry; the weight copies are
        // still issued long before the FC1 kernel ends.
        while (globalTimerNs() < entryNs + kFc2RouteHoldNs)
        {
            __nanosleep(64);
        }
        // The FC1 kernel's routing (the keys of buildKeys, selectTopK) on the staged inputs, and the weights from
        // routingWeight: the experts, their order and the FP32 weights of noaux_tc_op.
        TopKKey keys[kCandidates];
#pragma unroll
        for (int i = 0; i < kCandidates; ++i)
        {
            int const expert = i * 32 + lane;
            float const sigmoid = sigmoidAccurate(logitsShared[expert]);
            sigmoidScores[expert] = sigmoid;
            keys[i] = TopKKey{sigmoid + static_cast<float>(biasShared[expert]), expert};
        }
        int32_t const expert = selectTopK(keys, lane);
        __syncwarp();
        auto const tile = cg::tiled_partition<32>(cg::this_thread_block());
        float const scoreNorm = lane < kTopK ? sigmoidScores[expert] : 0.F;
        float const sumNorm = cg::reduce(tile, scoreNorm, cg::plus<float>{});
        if (lane < kTopK)
        {
            topExperts[lane] = expert;
            topWeights[lane] = routingWeight(scoreNorm, sumNorm, scale);
        }
        __syncwarp();
        if (lane == 0)
        {
            mbarrierArriveExpectTx(&slicesFull, kTopK * sliceElems * static_cast<int>(sizeof(Bf16)));
        }
        __syncwarp();
        if (lane < kTopK)
        {
            bulkCopyToShared(slices + lane * sliceElems,
                w2 + (static_cast<int64_t>(topExperts[lane]) * hiddenSize + row0) * interSize,
                sliceElems * static_cast<int>(sizeof(Bf16)), &slicesFull, evictFirstPolicy());
        }
    }
    __syncthreads();

    // The SwiGLU outputs are written by the FC1 kernel: every warp waits for it, then stages its expert's outputs.
    cudaGridDependencySynchronize();
    Bf16 const* slice = slices + warp * sliceElems;
    Bf16* expertAct = actShared + warp * interSize;
    for (int i = lane; i < interSize / 8; i += 32)
    {
        reinterpret_cast<uint4*>(expertAct)[i] = __ldcg(reinterpret_cast<uint4 const*>(act + warp * interSize) + i);
    }
    __syncwarp();
    mbarrierWait(&slicesFull, 0);

    // Lane (g, t) of warp k accumulates the dot products of slice rows tile * 16 + g and tile * 16 + g + 8 with
    // act[k, :], over the two 32-element halves of each 64-element K step in two accumulators. Lanes of odd g take the
    // two halves in the other order, so that a warp's loads of 8 rows spread over all 32 banks: column n of the B
    // fragments then holds the vector in the order of the rows of n's parity, and the accumulators of those columns
    // hold the same products in the same order as with one order for all (which of the two holds which half differs).
    int const g = lane / 4;
    int const t = lane % 4;
    int const swap = g & 1;
    float acc[kTiles][2][4] = {};
    for (int j0 = 0; j0 < interSize; j0 += kFc2StepPairK)
    {
#pragma unroll
        for (int half = 0; half < 2; ++half)
        {
            int const j = j0 + (half ^ swap) * kStepK + 8 * t;
            uint4 const av = *reinterpret_cast<uint4 const*>(expertAct + j);
#pragma unroll
            for (int tileIdx = 0; tileIdx < kTiles; ++tileIdx)
            {
                uint4 const lo = *reinterpret_cast<uint4 const*>(slice + (tileIdx * kMmaRows + g) * interSize + j);
                uint4 const hi = *reinterpret_cast<uint4 const*>(slice + (tileIdx * kMmaRows + g + 8) * interSize + j);
                mmaStep(acc[tileIdx][half], lo, hi, av);
            }
        }
    }

    // The CUTLASS FC2 epilogue rounds each expert's scaled output to BF16 before adding it into the output. Lane (g, 0)
    // holds columns 0 and 1 of rows g and g + 8, of which column g & 1 has their dot products.
    float const weight = topWeights[warp];
    if (t == 0)
    {
#pragma unroll
        for (int tileIdx = 0; tileIdx < kTiles; ++tileIdx)
        {
            float const lo0 = swap ? acc[tileIdx][0][1] : acc[tileIdx][0][0];
            float const lo1 = swap ? acc[tileIdx][1][1] : acc[tileIdx][1][0];
            float const hi0 = swap ? acc[tileIdx][0][3] : acc[tileIdx][0][2];
            float const hi1 = swap ? acc[tileIdx][1][3] : acc[tileIdx][1][2];
            terms[warp][tileIdx * kMmaRows + g] = roundToBf16(weight * (lo0 + lo1));
            terms[warp][tileIdx * kMmaRows + g + 8] = roundToBf16(weight * (hi0 + hi1));
        }
    }
    __syncthreads();
    if (threadIdx.x < kRows)
    {
        float sum = 0.F;
#pragma unroll
        for (int k = 0; k < kTopK; ++k)
        {
            sum += terms[k][threadIdx.x];
        }
        out[row0 + threadIdx.x] = __float2bfloat16_rn(sum);
    }
#endif
}

// FC1 + SwiGLU of a multi-token call, one CTA per SM. Consumer warp m < kTokens routes token m as lowMBf16MoeFc1Kernel
// routes its token, the producer warp builds the union of the selections, and the work is the numUnion * interSize
// units of the union: unit u * interSize + j is intermediate index j of union expert u, its gate and up rows. CTA b
// takes units [b * U / G, (b + 1) * U / G) and streams their rows through a ring as lowMBf16MoeFc1Kernel does. x of
// every token is in shared memory, each consumer thread holding there the pieces it multiplies with. Per pair of rows
// a consumer thread loads its pieces of both rows, releases the two buffers, and for each token that routed to the
// rows' expert adds its shares of the two dot products in the order of lowMBf16MoeFc1Kernel. The warp sums of every
// kMultiChunkUnits units go to one of two chunk buffers, from which the epilogue warp forms those units' SwiGLU
// outputs, again in that kernel's order, while the consumer warps fill the other buffer.
template <typename BiasT, int kTokens>
__global__ void __launch_bounds__(kMultiFc1Threads, 2) lowMBf16MoeMultiFc1Kernel(Bf16* __restrict__ act,
    Bf16 const* __restrict__ x, float const* __restrict__ routerLogits, BiasT const* __restrict__ bias,
    Bf16 const* __restrict__ w3w1, int hiddenSize, int interSize, int numStages)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ __align__(128) uint8_t smem[];
    // rowFull[s]: buffer s holds its next row (the producer's arrival plus the copy's bytes). rowFree[s]: every
    // consumer warp has read its pieces of buffer s.
    __shared__ uint64_t rowFull[kMultiMaxFc1Stages];
    __shared__ uint64_t rowFree[kMultiMaxFc1Stages];
    // chunkFull[b]: every consumer warp has stored its sums of chunk buffer b. chunkFree[b]: the epilogue warp has read
    // them.
    __shared__ uint64_t chunkFull[2];
    __shared__ uint64_t chunkFree[2];
    __shared__ MultiRouting<kTokens> routing;

    int const warp = threadIdx.x / 32;
    int const lane = threadIdx.x % 32;
    int const rowBytes = hiddenSize * static_cast<int>(sizeof(Bf16));
    int const piecesPerRow = hiddenSize / kPieceElems;
    // ring[numStages][hiddenSize], xShared[kTokens][hiddenSize], rowSums[2][kMultiChunkRows][kTokens][consumer warp]
    uint8_t* ring = smem;
    uint4* xShared = reinterpret_cast<uint4*>(smem + numStages * rowBytes);
    float* rowSums = reinterpret_cast<float*>(smem + (numStages + kTokens) * rowBytes);

    if (threadIdx.x == 0)
    {
        for (int s = 0; s < numStages; ++s)
        {
            mbarrierInit(&rowFull[s], 1);
            mbarrierInit(&rowFree[s], kFc1ConsumerWarps);
        }
        for (int b = 0; b < 2; ++b)
        {
            mbarrierInit(&chunkFull[b], kFc1ConsumerWarps);
            mbarrierInit(&chunkFree[b], 1);
        }
        fenceBarrierInit();
    }
    for (int e = static_cast<int>(threadIdx.x); e < kNumExperts; e += kMultiFc1Threads)
    {
        routing.expertTokens[e] = 0;
    }
    // The correction bias is a constant: the routing warps read it before the grid-dependency wait.
    float biasValues[kCandidates] = {};
    if (warp < kTokens)
    {
        loadBias(bias, lane, biasValues);
    }
    __syncthreads();

    // x and the router logits are written by earlier kernels.
    cudaGridDependencySynchronize();
    // The FC2 kernel reads only the logits, the bias and its weights before its own wait, and the logits are complete
    // since the wait above returned.
    cudaTriggerProgrammaticLaunchCompletion();

    if (warp == kFc1ProducerWarp)
    {
        namedBarrierSync(kMultiRouteBarrier, kMultiFc1Threads);
        buildUnion(routing, lane);
        int const numUnion = routing.numUnion;
        int const numUnits = numUnion * interSize;
        int const unitBegin = static_cast<int>(static_cast<int64_t>(blockIdx.x) * numUnits / gridDim.x);
        int const unitEnd = static_cast<int>(static_cast<int64_t>(blockIdx.x + 1) * numUnits / gridDim.x);
        int const numRows = 2 * (unitEnd - unitBegin);

        // Lane 0 streams the rows. The next row is of unit (union expert u, index), the gate row first, into buffer s,
        // which it fills for the pass-th time.
        uint64_t const policy = evictFirstPolicy();
        int64_t const expertStride = 2 * static_cast<int64_t>(interSize) * hiddenSize;
        int u = unitBegin / interSize;
        int index = unitBegin % interSize;
        bool gateRow = true;
        Bf16 const* expertRows = w3w1 + routing.unionExperts[u] * expertStride;
        int s = 0;
        uint32_t pass = 0;
        auto issueRow = [&]()
        {
            int64_t const rowInExpert = gateRow ? interSize + index : index;
            mbarrierArriveExpectTx(&rowFull[s], rowBytes);
            bulkCopyToShared(ring + s * rowBytes, expertRows + rowInExpert * hiddenSize, rowBytes, &rowFull[s], policy);
            if (!gateRow && ++index == interSize)
            {
                // The up row closed the union expert's last unit.
                index = 0;
                if (++u < numUnion)
                {
                    expertRows = w3w1 + routing.unionExperts[u] * expertStride;
                }
            }
            gateRow = !gateRow;
            if (++s == numStages)
            {
                s = 0;
                ++pass;
            }
        };
        int const firstRows = numStages < numRows ? numStages : numRows;
        if (lane == 0)
        {
            for (int row = 0; row < firstRows; ++row)
            {
                issueRow();
            }
        }
        // Release the consumer and epilogue warps: the union is in shared memory.
        __syncwarp();
        namedBarrierArrive(kMultiParkBarrier, kMultiFc1Threads);
        if (lane == 0)
        {

            for (int row = firstRows; row < numRows; ++row)
            {
                // Buffer s still holds row - numStages until every consumer warp has read it.
                mbarrierWait(&rowFree[s], (pass - 1) & 1U);
                issueRow();
            }
        }
    }
    else if (warp == kMultiFc1EpilogueWarp)
    {
        namedBarrierArrive(kMultiRouteBarrier, kMultiFc1Threads);
        namedBarrierSync(kMultiParkBarrier, kMultiFc1Threads);

        int const numUnits = routing.numUnion * interSize;
        int const unitBegin = static_cast<int>(static_cast<int64_t>(blockIdx.x) * numUnits / gridDim.x);
        int const unitEnd = static_cast<int>(static_cast<int64_t>(blockIdx.x + 1) * numUnits / gridDim.x);
        int const numChunks = (unitEnd - unitBegin + kMultiChunkUnits - 1) / kMultiChunkUnits;
        for (int chunk = 0; chunk < numChunks; ++chunk)
        {
            int const buffer = chunk & 1;
            mbarrierWait(&chunkFull[buffer], (chunk >> 1) & 1U);
            // Unit c of the chunk and token m: the gate row's 8 warp sums, then the up row's, in warp order as
            // lowMBf16MoeFc1Kernel adds them.
            int const chunkUnitBegin = unitBegin + chunk * kMultiChunkUnits;
            int const chunkUnits
                = unitEnd - chunkUnitBegin < kMultiChunkUnits ? unitEnd - chunkUnitBegin : kMultiChunkUnits;
            float const* bufferSums = rowSums + buffer * kMultiChunkRows * kTokens * kFc1ConsumerWarps;
            for (int p = lane; p < chunkUnits * kTokens; p += 32)
            {
                int const c = p / kTokens;
                int const m = p - c * kTokens;
                int const unit = chunkUnitBegin + c;
                int const unitExpert = unit / interSize;
                if ((routing.unionTokens[unitExpert] >> m) & 1U)
                {
                    float const* gateSums = bufferSums + (2 * c * kTokens + m) * kFc1ConsumerWarps;
                    float const* upSums = bufferSums + ((2 * c + 1) * kTokens + m) * kFc1ConsumerWarps;
                    float gate = 0.F;
                    float up = 0.F;
                    for (int w = 0; w < kFc1ConsumerWarps; ++w)
                    {
                        gate += gateSums[w];
                        up += upSums[w];
                    }
                    // The CUTLASS FC1 GEMM writes its output in BF16 and the activation kernel reads it back.
                    gate = roundToBf16(gate);
                    up = roundToBf16(up);
                    // silu(gate) * up in FP32 as the CUTLASS activation (GLUAdaptor<SiLu>) computes it, then BF16.
                    float const silu = gate * (1.F / (1.F + expf(-gate)));
                    int const slot = routing.unionSlots[unitExpert][m];
                    act[(m * kTopK + slot) * interSize + unit - unitExpert * interSize]
                        = __float2bfloat16_rn(silu * up);
                }
            }
            // The warp's reads of the buffer are done before lane 0 frees it.
            __syncwarp();
            if (lane == 0)
            {
                mbarrierArrive(&chunkFree[buffer]);
            }
        }
    }
    else
    {
        if (warp < kTokens)
        {
            TopKKey keys[kCandidates];
            buildKeys(routerLogits + warp * kNumExperts, biasValues, lane, keys, nullptr);
            int32_t const expert = selectTopK(keys, lane);
            if (lane < kTopK)
            {
                routing.topExperts[warp][lane] = expert;
                atomicOr(&routing.expertTokens[expert], 1U << warp);
            }
        }
        // The routing results are in shared memory; the producer warp builds the union from them.
        namedBarrierArrive(kMultiRouteBarrier, kMultiFc1Threads);
        // Each thread stores its pieces of every token's x; it alone reads them back.
        int const piecesPerThread = hiddenSize / kHiddenGranule;
        for (int m = 0; m < kTokens; ++m)
        {
#pragma unroll
            for (int i = 0; i < kMultiMaxPiecesPerThread; ++i)
            {
                if (i < piecesPerThread)
                {
                    int const piece = m * piecesPerRow + threadIdx.x + i * kFc1ConsumerThreads;
                    xShared[piece] = __ldcg(reinterpret_cast<uint4 const*>(x) + piece);
                }
            }
        }
        // Wait at the barrier, without polling the ring, while the producer builds the union and issues the first
        // rows.
        namedBarrierSync(kMultiParkBarrier, kMultiFc1Threads);
        int const numUnits = routing.numUnion * interSize;
        int const unitBegin = static_cast<int>(static_cast<int64_t>(blockIdx.x) * numUnits / gridDim.x);
        int const unitEnd = static_cast<int>(static_cast<int64_t>(blockIdx.x + 1) * numUnits / gridDim.x);
        int const numRows = 2 * (unitEnd - unitBegin);
        int u = unitBegin / interSize;
        int index = unitBegin % interSize;

        int s0 = 0;
        uint32_t pass0 = 0;
        int chunk = 0;
        int chunkRow = 0;
        // Rows row and row + 1 are the gate and up rows of unit (u, index), in buffers s0 and s1 (numStages >= 2),
        // filled for the pass0-th and pass1-th time.
        for (int row = 0; row < numRows; row += 2)
        {
            int s1 = s0 + 1;
            uint32_t pass1 = pass0;
            if (s1 == numStages)
            {
                s1 = 0;
                ++pass1;
            }
            uint32_t const tokens = routing.unionTokens[u];
            mbarrierWait(&rowFull[s0], pass0 & 1U);
            mbarrierWait(&rowFull[s1], pass1 & 1U);
            uint4 const* pieces0 = reinterpret_cast<uint4 const*>(ring + s0 * rowBytes);
            uint4 const* pieces1 = reinterpret_cast<uint4 const*>(ring + s1 * rowBytes);
            uint4 rowPieces0[kMultiMaxPiecesPerThread];
            uint4 rowPieces1[kMultiMaxPiecesPerThread];
#pragma unroll
            for (int i = 0; i < kMultiMaxPiecesPerThread; ++i)
            {
                if (i < piecesPerThread)
                {
                    rowPieces0[i] = pieces0[threadIdx.x + i * kFc1ConsumerThreads];
                    rowPieces1[i] = pieces1[threadIdx.x + i * kFc1ConsumerThreads];
                }
            }
            // This warp is done with both buffers (the arrival releases its lanes' reads, ordered before it by the
            // warp barrier); the producer refills a buffer once every consumer warp is.
            __syncwarp();
            if (lane == 0)
            {
                mbarrierArrive(&rowFree[s0]);
                mbarrierArrive(&rowFree[s1]);
            }
            if (chunkRow == 0 && chunk >= 2)
            {
                // The chunk's buffer still holds the sums of chunk - 2 until the epilogue warp has read them.
                mbarrierWait(&chunkFree[chunk & 1], ((chunk >> 1) - 1) & 1U);
            }
            float* chunkSums = rowSums + ((chunk & 1) * kMultiChunkRows + chunkRow) * kTokens * kFc1ConsumerWarps;
            for (uint32_t pending = tokens; pending != 0; pending &= pending - 1)
            {
                int const m = __ffs(static_cast<int>(pending)) - 1;
                uint4 const* xPieces = xShared + m * piecesPerRow + threadIdx.x;
                uint4 xPiece = xPieces[0];
                float dot0 = dotPiecePacked(rowPieces0[0], xPiece);
                float dot1 = dotPiecePacked(rowPieces1[0], xPiece);
#pragma unroll
                for (int i = 1; i < kMultiMaxPiecesPerThread; ++i)
                {
                    if (i < piecesPerThread)
                    {
                        xPiece = xPieces[i * kFc1ConsumerThreads];
                        dot0 += dotPiecePacked(rowPieces0[i], xPiece);
                        dot1 += dotPiecePacked(rowPieces1[i], xPiece);
                    }
                }
                // Lane 0 ends with the gate row's warp sum and lane 16 with the up row's.
                float const sum = warpSum2Split(dot0, dot1, lane);
                if (lane % 16 == 0)
                {
                    chunkSums[((lane / 16) * kTokens + m) * kFc1ConsumerWarps + warp] = sum;
                }
            }
            s0 = s1 + 1;
            pass0 = pass1;
            if (s0 == numStages)
            {
                s0 = 0;
                ++pass0;
            }
            if (++index == interSize)
            {
                index = 0;
                ++u;
            }
            chunkRow += 2;
            if (chunkRow == kMultiChunkRows || row + 2 == numRows)
            {
                // Lanes 0 and 16 stored this warp's sums of the chunk (ordered before lane 0's arrival by the warp
                // barrier); the epilogue warp takes the buffer from here.
                __syncwarp();
                if (lane == 0)
                {
                    mbarrierArrive(&chunkFull[chunk & 1]);
                }
                chunkRow = 0;
                ++chunk;
            }
        }
    }
#endif
}

// FC2 + the weighted sum of a multi-token call. Grid: one CTA per kMultiFc2Rows output rows. Before the grid-dependency
// wait, consumer warp m < kTokens routes token m as lowMBf16MoeFc2Kernel routes its token (same experts, order and
// weights), from the router logits and the bias staged by bulk copies; the producer warp builds the union and starts
// streaming the union experts' weight slices (kMultiFc2Rows rows each) into a ring of numStages stages. After the wait
// the consumer warps stage every token's SwiGLU outputs and take the (union expert, tile) items in turn: the mma of
// lowMBf16MoeFc2Kernel with B column 2 * q + p holding token q's SwiGLU outputs of the expert in the K order of rows of
// parity p, so that accumulator column 2 * q + (g & 1) of row g holds that row's dot product with token q, the same
// products in the same order as lowMBf16MoeFc2Kernel computes for that token alone. Lane (g, t) then holds token t's
// terms of rows g and g + 8 and rounds them as that kernel does; each output row adds its token's 8 terms in slot
// order.
template <typename BiasT, int kTokens>
__global__ void __launch_bounds__(kMultiFc2Threads, 2) lowMBf16MoeMultiFc2Kernel(Bf16* __restrict__ out,
    Bf16 const* __restrict__ act, float const* __restrict__ routerLogits, BiasT const* __restrict__ bias,
    Bf16 const* __restrict__ w2, int hiddenSize, int interSize, int numStages, RoutingScale scale)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    constexpr int kGroups = (kTokens + kMultiTokensPerMma - 1) / kMultiTokensPerMma;
    constexpr int kProducerWarp = kMultiFc2ConsumerWarps;
    extern __shared__ __align__(128) uint8_t smem[];
    __shared__ __align__(16) float logitsShared[kTokens * kNumExperts];
    __shared__ __align__(16) BiasT biasShared[kNumExperts];
    __shared__ uint64_t inputsFull;
    // sliceFull[s]: stage s holds its next weight slice. sliceFree[s]: both tile warps have read it. issuedSlices: the
    // union experts whose slices the producer warp has issued, in union order.
    __shared__ uint64_t sliceFull[kMultiFc2MaxStages];
    __shared__ uint64_t sliceFree[kMultiFc2MaxStages];
    __shared__ int32_t issuedSlices;
    __shared__ MultiRouting<kTokens> routing;
    __shared__ float topWeights[kTokens][kTopK];

    int const sliceElems = kMultiFc2Rows * interSize;
    // slices[numStages][kMultiFc2Rows][interSize], actShared[kTokens][kTopK][interSize],
    // terms[kTokens][kTopK][kMultiFc2Rows]
    Bf16* slices = reinterpret_cast<Bf16*>(smem);
    Bf16* actShared = slices + numStages * sliceElems;
    float* terms = reinterpret_cast<float*>(actShared + kTokens * kTopK * interSize);

    int const warp = threadIdx.x / 32;
    int const lane = threadIdx.x % 32;
    int const row0 = blockIdx.x * kMultiFc2Rows;

    if (threadIdx.x == 0)
    {
        mbarrierInit(&inputsFull, 1);
        for (int s = 0; s < numStages; ++s)
        {
            mbarrierInit(&sliceFull[s], 1);
            mbarrierInit(&sliceFree[s], kMultiFc2Tiles);
        }
        fenceBarrierInit();
    }
    for (int e = static_cast<int>(threadIdx.x); e < kNumExperts; e += kMultiFc2Threads)
    {
        routing.expertTokens[e] = 0;
    }
    __syncthreads();

    // Before the grid-dependency wait this kernel reads only the router logits, the bias and the FC2 weights. The
    // logits come from the router GEMM, which completed before the FC1 kernel returned from its own wait and released
    // this grid; the bias and the weights are constants. The logits and the bias come in by bulk copies, as in
    // lowMBf16MoeFc2Kernel.
    bool const bulkInputs = ((reinterpret_cast<uintptr_t>(routerLogits) | reinterpret_cast<uintptr_t>(bias)) % 16) == 0;
    if (warp == kProducerWarp)
    {
        uint64_t const policy = evictFirstPolicy();
        if (bulkInputs && lane == 0)
        {
            mbarrierArriveExpectTx(&inputsFull, sizeof(logitsShared) + sizeof(biasShared));
            bulkCopyToShared(logitsShared, routerLogits, sizeof(logitsShared), &inputsFull, policy);
            bulkCopyToShared(biasShared, bias, sizeof(biasShared), &inputsFull, policy);
        }
        namedBarrierSync(kMultiRouteBarrier, kMultiFc2Threads);
        buildUnion(routing, lane);
        int const numUnion = routing.numUnion;
        int const sliceBytes = sliceElems * static_cast<int>(sizeof(Bf16));
        auto issueSlice = [&](int unionIdx, int stage)
        {
            mbarrierArriveExpectTx(&sliceFull[stage], sliceBytes);
            bulkCopyToShared(slices + stage * sliceElems,
                w2 + (static_cast<int64_t>(routing.unionExperts[unionIdx]) * hiddenSize + row0) * interSize, sliceBytes,
                &sliceFull[stage], policy);
        };
        int const firstSlices = numStages < numUnion ? numStages : numUnion;
        if (lane == 0)
        {
            for (int idx = 0; idx < firstSlices; ++idx)
            {
                issueSlice(idx, idx);
            }
            issuedSlices = firstSlices;
        }
        // Release the consumer warps: the union is in shared memory.
        __syncwarp();
        namedBarrierArrive(kMultiParkBarrier, kMultiFc2Threads);
        // No stage frees up before the consumer warps return from their grid-dependency wait; waiting here instead of
        // polling the stages leaves the SM's issue slots to the FC1 CTA beside this one.
        cudaGridDependencySynchronize();
        if (lane == 0)
        {
            for (int idx = numStages; idx < numUnion; ++idx)
            {
                // Stage idx % numStages still holds union expert idx - numStages until both its tile warps read it.
                int const stage = idx % numStages;
                mbarrierWait(&sliceFree[stage], ((idx / numStages) - 1) & 1U);
                issueSlice(idx, stage);
                *reinterpret_cast<int32_t volatile*>(&issuedSlices) = idx + 1;
            }
        }
    }
    else
    {
        if (warp < kTokens)
        {
            float const* tokenLogits = routerLogits + warp * kNumExperts;
            if (bulkInputs)
            {
                mbarrierWait(&inputsFull, 0);
            }
            TopKKey keys[kCandidates];
#pragma unroll
            for (int i = 0; i < kCandidates; ++i)
            {
                int const expert = i * 32 + lane;
                float const logit
                    = bulkInputs ? logitsShared[warp * kNumExperts + expert] : __ldcg(tokenLogits + expert);
                float const biasValue
                    = bulkInputs ? static_cast<float>(biasShared[expert]) : static_cast<float>(bias[expert]);
                keys[i] = TopKKey{sigmoidAccurate(logit) + biasValue, expert};
            }
            int32_t const expert = selectTopK(keys, lane);
            __syncwarp();
            // The weights from routingWeight on the selected experts' sigmoid scores, as lowMBf16MoeFc2Kernel forms
            // them.
            auto const tile = cg::tiled_partition<32>(cg::this_thread_block());
            float const scoreNorm = lane < kTopK
                ? sigmoidAccurate(bulkInputs ? logitsShared[warp * kNumExperts + expert] : __ldcg(tokenLogits + expert))
                : 0.F;
            float const sumNorm = cg::reduce(tile, scoreNorm, cg::plus<float>{});
            if (lane < kTopK)
            {
                routing.topExperts[warp][lane] = expert;
                topWeights[warp][lane] = routingWeight(scoreNorm, sumNorm, scale);
                atomicOr(&routing.expertTokens[expert], 1U << warp);
            }
        }
        // The routing results are in shared memory; the producer warp builds the union from them.
        namedBarrierArrive(kMultiRouteBarrier, kMultiFc2Threads);
        namedBarrierSync(kMultiParkBarrier, kMultiFc2Threads);

        // The SwiGLU outputs are written by the FC1 kernel.
        cudaGridDependencySynchronize();
        int const actPieces = kTokens * kTopK * interSize / kPieceElems;
        for (int i = static_cast<int>(threadIdx.x); i < actPieces; i += kMultiFc2ConsumerThreads)
        {
            reinterpret_cast<uint4*>(actShared)[i] = __ldcg(reinterpret_cast<uint4 const*>(act) + i);
        }
        namedBarrierSync(kMultiConsumerBarrier, kMultiFc2ConsumerThreads);

        int const g = lane / 4;
        int const t = lane % 4;
        int const swap = g & 1;
        int const numUnion = routing.numUnion;
        for (int item = warp; item < numUnion * kMultiFc2Tiles; item += kMultiFc2ConsumerWarps)
        {
            int const unionIdx = item / kMultiFc2Tiles;
            int const tileIdx = item - unionIdx * kMultiFc2Tiles;
            int const stage = unionIdx % numStages;
            uint32_t const tokens = routing.unionTokens[unionIdx];
            // B column g of mma group q is token q * kMultiTokensPerMma + g / 2; a token that did not route to this
            // expert reads any SwiGLU row, and its accumulator columns are not used.
            Bf16 const* actRows[kGroups];
#pragma unroll
            for (int q = 0; q < kGroups; ++q)
            {
                int const m = q * kMultiTokensPerMma + g / 2;
                bool const routed = m < kTokens && ((tokens >> m) & 1U);
                actRows[q] = actShared + (routed ? (m * kTopK + routing.unionSlots[unionIdx][m]) * interSize : 0);
            }
            // The items of a stage go to different warps, so a warp may reach a stage's next fill before the fill
            // ahead of it has landed, and a parity wait cannot tell those two fills apart. Once the slice is issued,
            // its stage's previous fill has been read, and the parity wait is for this fill.
            while (*reinterpret_cast<int32_t volatile*>(&issuedSlices) <= unionIdx)
            {
                __nanosleep(32);
            }
            mbarrierWait(&sliceFull[stage], (unionIdx / numStages) & 1U);
            Bf16 const* slice = slices + stage * sliceElems + tileIdx * kMmaRows * interSize;
            float acc[kGroups][2][4] = {};
            for (int j0 = 0; j0 < interSize; j0 += kFc2StepPairK)
            {
#pragma unroll
                for (int half = 0; half < 2; ++half)
                {
                    int const j = j0 + (half ^ swap) * kStepK + 8 * t;
                    uint4 const lo = *reinterpret_cast<uint4 const*>(slice + g * interSize + j);
                    uint4 const hi = *reinterpret_cast<uint4 const*>(slice + (g + 8) * interSize + j);
#pragma unroll
                    for (int q = 0; q < kGroups; ++q)
                    {
                        uint4 const av = *reinterpret_cast<uint4 const*>(actRows[q] + j);
                        mmaStep(acc[q][half], lo, hi, av);
                    }
                }
            }
            // This warp is done with the stage; the producer refills it once both tile warps are.
            __syncwarp();
            if (lane == 0)
            {
                mbarrierArrive(&sliceFree[stage]);
            }
#pragma unroll
            for (int q = 0; q < kGroups; ++q)
            {
                int const m = q * kMultiTokensPerMma + t;
                if (m < kTokens && ((tokens >> m) & 1U))
                {
                    int const slot = routing.unionSlots[unionIdx][m];
                    float const weight = topWeights[m][slot];
                    // Column 2 * t + (g & 1) of rows g and g + 8, as lowMBf16MoeFc2Kernel's lane (g, 0) takes column
                    // g & 1.
                    float const lo0 = swap ? acc[q][0][1] : acc[q][0][0];
                    float const lo1 = swap ? acc[q][1][1] : acc[q][1][0];
                    float const hi0 = swap ? acc[q][0][3] : acc[q][0][2];
                    float const hi1 = swap ? acc[q][1][3] : acc[q][1][2];
                    float* tokenTerms = terms + (m * kTopK + slot) * kMultiFc2Rows + tileIdx * kMmaRows;
                    tokenTerms[g] = roundToBf16(weight * (lo0 + lo1));
                    tokenTerms[g + 8] = roundToBf16(weight * (hi0 + hi1));
                }
            }
        }
        // Every term is in shared memory: each output row adds its token's 8 terms in slot order.
        namedBarrierSync(kMultiConsumerBarrier, kMultiFc2ConsumerThreads);
        for (int i = static_cast<int>(threadIdx.x); i < kTokens * kMultiFc2Rows; i += kMultiFc2ConsumerThreads)
        {
            int const m = i / kMultiFc2Rows;
            int const r = i - m * kMultiFc2Rows;
            float sum = 0.F;
#pragma unroll
            for (int k = 0; k < kTopK; ++k)
            {
                sum += terms[(m * kTopK + k) * kMultiFc2Rows + r];
            }
            out[m * hiddenSize + row0 + r] = __float2bfloat16_rn(sum);
        }
    }
#endif
}

// Lets a kernel use smemBytes of dynamic shared memory and asks for the largest shared-memory carveout for both
// kernels, so that an SM running a CTA of one can take a CTA of the other.
template <typename Kernel>
void setSharedMemoryAttributes(Kernel* kernel, size_t smemBytes)
{
    TLLM_CUDA_CHECK(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smemBytes)));
    TLLM_CUDA_CHECK(cudaFuncSetAttribute(
        kernel, cudaFuncAttributePreferredSharedMemoryCarveout, static_cast<int>(cudaSharedmemCarveoutMaxShared)));
}

// routingWeight's FP32 path applies when the scaling factor is a float with at most 20 significant bits (the low 4
// mantissa bits zero) between 2^-20 and 2^20.
RoutingScale makeRoutingScale(double routedScalingFactor)
{
    float const valueFloat = static_cast<float>(routedScalingFactor);
    uint32_t bits = 0;
    std::memcpy(&bits, &valueFloat, sizeof(bits));
    bool const fastPath = static_cast<double>(valueFloat) == routedScalingFactor && (bits & 0xFU) == 0
        && valueFloat >= 0x1p-20f && valueFloat <= 0x1p20f;
    return RoutingScale{routedScalingFactor, valueFloat, fastPath ? 1 : 0};
}

template <typename BiasT, int kTiles>
void launchFc2(Bf16* out, Bf16 const* act, float const* routerLogits, BiasT const* bias, Bf16 const* w2, int hiddenSize,
    int interSize, double routedScalingFactor, cudaStream_t stream)
{
    int const rows = kTiles * kMmaRows;
    size_t const smemBytes = static_cast<size_t>(kTopK) * (rows + 1) * interSize * sizeof(Bf16);
    auto* kernel = &lowMBf16MoeFc2Kernel<BiasT, kTiles>;
    setSharedMemoryAttributes(kernel, smemBytes);
    common::launchWithPdlWhenEnabled("lowMBf16MoeFc2", kernel, dim3(hiddenSize / rows), dim3(kFc2Threads), smemBytes,
        stream, out, act, routerLogits, bias, w2, hiddenSize, interSize, makeRoutingScale(routedScalingFactor));
}

// The shared memory of an SM, the per-CTA reservation and the static shared memory of the two multi-token kernels.
struct MultiSmemLimits
{
    size_t perSm;
    size_t reservedPerCta;
    size_t fc1Static;
    size_t fc2Static;
};

template <typename Fc1Kernel, typename Fc2Kernel>
MultiSmemLimits queryMultiSmemLimits(Fc1Kernel* fc1, Fc2Kernel* fc2)
{
    int device = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    int perSm = 0;
    int reserved = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&perSm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, device));
    cudaFuncAttributes fc1Attributes{};
    cudaFuncAttributes fc2Attributes{};
    TLLM_CUDA_CHECK(cudaFuncGetAttributes(&fc1Attributes, fc1));
    TLLM_CUDA_CHECK(cudaFuncGetAttributes(&fc2Attributes, fc2));
    return MultiSmemLimits{static_cast<size_t>(perSm), static_cast<size_t>(reserved), fc1Attributes.sharedSizeBytes,
        fc2Attributes.sharedSizeBytes};
}

template <typename BiasT, int kTokens>
void launchLowMBf16MoeMulti(Bf16* out, Bf16* act, Bf16 const* x, float const* routerLogits, BiasT const* bias,
    Bf16 const* w3w1, Bf16 const* w2, int hiddenSize, int interSize, double routedScalingFactor, cudaStream_t stream)
{
    auto* fc1 = &lowMBf16MoeMultiFc1Kernel<BiasT, kTokens>;
    auto* fc2 = &lowMBf16MoeMultiFc2Kernel<BiasT, kTokens>;
    static int const smCount = common::getMultiProcessorCount();
    static MultiSmemLimits const limits = queryMultiSmemLimits(fc1, fc2);

    // Shared memory per CTA as an SM accounts it: the FC1 CTA without its ring, and the FC2 CTA without its stages.
    int const rowBytes = hiddenSize * static_cast<int>(sizeof(Bf16));
    size_t const sliceBytes = static_cast<size_t>(kMultiFc2Rows) * interSize * sizeof(Bf16);
    size_t const fc1Fixed = static_cast<size_t>(kTokens) * rowBytes
        + static_cast<size_t>(2 * kMultiChunkRows * kTokens * kFc1ConsumerWarps) * sizeof(float);
    size_t const fc2Fixed = static_cast<size_t>(kTokens * kTopK) * interSize * sizeof(Bf16)
        + static_cast<size_t>(kTokens * kTopK * kMultiFc2Rows) * sizeof(float);
    size_t const fc1Base = limits.fc1Static + limits.reservedPerCta + fc1Fixed;
    size_t const fc2Base = limits.fc2Static + limits.reservedPerCta + fc2Fixed;

    // FC1: one CTA per SM (each with at least one unit: the union has at least kTopK experts). Its ring takes enough
    // rows that two FC1 CTAs never share an SM (the static unit split would give that SM twice the rows), and at most
    // as many as leave room for an FC2 CTA with two stages beside it.
    size_t const halfSm = limits.perSm / 2 + 1;
    int const oneCtaStages
        = halfSm > fc1Base ? static_cast<int>((halfSm - fc1Base + static_cast<size_t>(rowBytes) - 1) / rowBytes) : 0;
    size_t const fc2Min = fc2Base + 2 * sliceBytes;
    int const roomStages
        = limits.perSm > fc1Base + fc2Min ? static_cast<int>((limits.perSm - fc1Base - fc2Min) / rowBytes) : 0;
    int numStages = std::min(kMultiMaxFc1Stages, std::max(kMultiMinFc1Stages, oneCtaStages));
    if (roomStages >= 2)
    {
        numStages = std::min(numStages, roomStages);
    }
    int const fc1Grid = std::min(smCount, kTopK * interSize);
    size_t const fc1SmemBytes = static_cast<size_t>(numStages) * rowBytes + fc1Fixed;
    setSharedMemoryAttributes(fc1, fc1SmemBytes);
    common::launchWithPdlWhenEnabled("lowMBf16MoeMultiFc1", fc1, dim3(fc1Grid), dim3(kMultiFc1Threads), fc1SmemBytes,
        stream, act, x, routerLogits, bias, w3w1, hiddenSize, interSize, numStages);

    // FC2: as many weight stages as fit beside the FC1 CTA and leave room for two FC2 CTAs per SM once the FC1 CTAs
    // exit, at least 2.
    size_t const fc1Total = fc1Base + static_cast<size_t>(numStages) * rowBytes;
    size_t const fc2Room = std::min(limits.perSm - std::min(limits.perSm, fc1Total), limits.perSm / 2);
    int const fitStages = fc2Room > fc2Base ? static_cast<int>((fc2Room - fc2Base) / sliceBytes) : 0;
    int const fc2Stages = std::max(2, std::min(kMultiFc2MaxStages, fitStages));
    size_t const fc2SmemBytes = static_cast<size_t>(fc2Stages) * sliceBytes + fc2Fixed;
    setSharedMemoryAttributes(fc2, fc2SmemBytes);
    common::launchWithPdlWhenEnabled("lowMBf16MoeMultiFc2", fc2, dim3(hiddenSize / kMultiFc2Rows),
        dim3(kMultiFc2Threads), fc2SmemBytes, stream, out, act, routerLogits, bias, w2, hiddenSize, interSize,
        fc2Stages, makeRoutingScale(routedScalingFactor));
}

} // namespace

bool lowMBf16MoeSupportsShape(int64_t hiddenSize, int64_t interSize)
{
    return hiddenSize > 0 && hiddenSize % kHiddenGranule == 0 && hiddenSize <= kMaxPiecesPerThread * kHiddenGranule
        && interSize > 0 && interSize % kFc2StepPairK == 0 && interSize <= kMaxInterSize;
}

template <typename BiasT>
void invokeLowMBf16Moe(Bf16* out, Bf16* act, Bf16 const* x, float const* routerLogits, BiasT const* bias,
    Bf16 const* w3w1, Bf16 const* w2, int hiddenSize, int interSize, double routedScalingFactor, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(lowMBf16MoeSupportsShape(hiddenSize, interSize),
        "lowMBf16Moe: unsupported hidden size %d or intermediate size %d", hiddenSize, interSize);

    // FC1: one CTA per SM, each with an even share of the units and a slot per row and consumer warp for the sums.
    static int const smCount = common::getMultiProcessorCount();
    int const numUnits = kTopK * interSize;
    int const fc1Grid = std::min(smCount, numUnits);
    int const rowBytes = hiddenSize * static_cast<int>(sizeof(Bf16));
    int const numStages = std::min(kMaxStages, kFc1RingBytes / rowBytes);
    int const maxRows = 2 * ((numUnits + fc1Grid - 1) / fc1Grid);
    size_t const fc1SmemBytes
        = static_cast<size_t>(numStages) * rowBytes + static_cast<size_t>(maxRows) * kFc1ConsumerWarps * sizeof(float);
    auto* fc1 = &lowMBf16MoeFc1Kernel<BiasT>;
    setSharedMemoryAttributes(fc1, fc1SmemBytes);
    common::launchWithPdlWhenEnabled("lowMBf16MoeFc1", fc1, dim3(fc1Grid), dim3(kFc1Threads), fc1SmemBytes, stream, act,
        x, routerLogits, bias, w3w1, hiddenSize, interSize, numStages);

    // Two 16-row tiles per FC2 CTA where the 8 weight slices fit in shared memory, else one.
    if (interSize <= kMaxInterSize / 2)
    {
        launchFc2<BiasT, 2>(out, act, routerLogits, bias, w2, hiddenSize, interSize, routedScalingFactor, stream);
    }
    else
    {
        launchFc2<BiasT, 1>(out, act, routerLogits, bias, w2, hiddenSize, interSize, routedScalingFactor, stream);
    }
}

bool lowMBf16MoeMultiSupportsShape(int64_t numTokens, int64_t hiddenSize, int64_t interSize)
{
    return numTokens >= 2 && numTokens <= kMultiMaxTokens && hiddenSize > 0 && hiddenSize % kHiddenGranule == 0
        && hiddenSize <= kMultiMaxPiecesPerThread * kHiddenGranule && interSize > 0 && interSize % kFc2StepPairK == 0
        && interSize <= kMaxInterSize;
}

template <typename BiasT>
void invokeLowMBf16MoeMulti(Bf16* out, Bf16* act, Bf16 const* x, float const* routerLogits, BiasT const* bias,
    Bf16 const* w3w1, Bf16 const* w2, int numTokens, int hiddenSize, int interSize, double routedScalingFactor,
    cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(lowMBf16MoeMultiSupportsShape(numTokens, hiddenSize, interSize),
        "lowMBf16MoeMulti: unsupported token count %d, hidden size %d or intermediate size %d", numTokens, hiddenSize,
        interSize);
    switch (numTokens)
    {
    case 2:
        launchLowMBf16MoeMulti<BiasT, 2>(
            out, act, x, routerLogits, bias, w3w1, w2, hiddenSize, interSize, routedScalingFactor, stream);
        break;
    case 3:
        launchLowMBf16MoeMulti<BiasT, 3>(
            out, act, x, routerLogits, bias, w3w1, w2, hiddenSize, interSize, routedScalingFactor, stream);
        break;
    case 4:
        launchLowMBf16MoeMulti<BiasT, 4>(
            out, act, x, routerLogits, bias, w3w1, w2, hiddenSize, interSize, routedScalingFactor, stream);
        break;
    case 5:
        launchLowMBf16MoeMulti<BiasT, 5>(
            out, act, x, routerLogits, bias, w3w1, w2, hiddenSize, interSize, routedScalingFactor, stream);
        break;
    default:
        launchLowMBf16MoeMulti<BiasT, 6>(
            out, act, x, routerLogits, bias, w3w1, w2, hiddenSize, interSize, routedScalingFactor, stream);
        break;
    }
}

#define INSTANTIATE_LOW_M_BF16_MOE(BiasT)                                                                              \
    template void invokeLowMBf16Moe<BiasT>(Bf16 * out, Bf16 * act, Bf16 const* x, float const* routerLogits,           \
        BiasT const* bias, Bf16 const* w3w1, Bf16 const* w2, int hiddenSize, int interSize,                            \
        double routedScalingFactor, cudaStream_t stream);

template void invokeLowMBf16MoeMulti<float>(Bf16*, Bf16*, Bf16 const*, float const*, float const*, Bf16 const*,
    Bf16 const*, int, int, int, double, cudaStream_t);
template void invokeLowMBf16MoeMulti<__nv_bfloat16>(Bf16*, Bf16*, Bf16 const*, float const*, __nv_bfloat16 const*,
    Bf16 const*, Bf16 const*, int, int, int, double, cudaStream_t);

INSTANTIATE_LOW_M_BF16_MOE(float);
INSTANTIATE_LOW_M_BF16_MOE(__nv_bfloat16);

#undef INSTANTIATE_LOW_M_BF16_MOE

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
