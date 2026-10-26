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

#include "tensorrt_llm/kernels/glmMoeDsa/glmLowMBf16Moe.h"

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

#define INSTANTIATE_LOW_M_BF16_MOE(BiasT)                                                                              \
    template void invokeLowMBf16Moe<BiasT>(Bf16 * out, Bf16 * act, Bf16 const* x, float const* routerLogits,           \
        BiasT const* bias, Bf16 const* w3w1, Bf16 const* w2, int hiddenSize, int interSize,                            \
        double routedScalingFactor, cudaStream_t stream);

INSTANTIATE_LOW_M_BF16_MOE(float);
INSTANTIATE_LOW_M_BF16_MOE(__nv_bfloat16);

#undef INSTANTIATE_LOW_M_BF16_MOE

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
