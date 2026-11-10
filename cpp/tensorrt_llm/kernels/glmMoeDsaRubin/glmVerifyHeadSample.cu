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

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmVerifyHeadSample.h"
#include <climits>
#include <cmath>
#include <cub/cub.cuh>
#include <curand_kernel.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa
{
namespace
{
constexpr int kSampleBlockSize = 512;
constexpr int kSampleVecSize = 4;
constexpr int kSampleTileElts = kSampleBlockSize * kSampleVecSize;
constexpr int kSampleTilesPerChunk = 4;
// One CTA per chunk of a row in the multi-CTA passes.
constexpr int kSampleChunkElts = kSampleTileElts * kSampleTilesPerChunk;
constexpr int kPivotRadixBits = 8;
constexpr int kPivotRadix = 1 << kPivotRadixBits;
constexpr int kWarpSize = 32;
constexpr unsigned kFullWarpMask = 0xFFFFFFFFU;

using SampleBlockReduce = cub::BlockReduce<float, kSampleBlockSize, cub::BLOCK_REDUCE_WARP_REDUCTIONS>;
using SampleBlockScan = cub::BlockScan<float, kSampleBlockSize, cub::BLOCK_SCAN_WARP_SCANS>;

int numSampleChunks(int vocabSize)
{
    return (vocabSize + kSampleChunkElts - 1) / kSampleChunkElts;
}

// IEEE-754 pivot add/multiply without flush to zero.
__device__ __forceinline__ float ieeeAdd(float a, float b)
{
    float r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __forceinline__ float ieeeMul(float a, float b)
{
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

// Order-preserving float-to-unsigned map for radix top-k.
__device__ __forceinline__ uint32_t toOrderedKey(float value)
{
    uint32_t const bits = __float_as_uint(value);
    return (bits & 0x80000000U) ? ~bits : (bits ^ 0x80000000U);
}

__device__ __forceinline__ float fromOrderedKey(uint32_t key)
{
    uint32_t const bits = (key & 0x80000000U) ? (key ^ 0x80000000U) : ~key;
    return __uint_as_float(bits);
}

__device__ __forceinline__ float inverseTemperature(float temperature)
{
    return temperature > 0.F ? 1.F / temperature : 1.F / 1.e-6F;
}

// The value the softmax sees: the top-k mask keeps logits >= the k-th largest, then the logit is scaled by 1 / T.
__device__ __forceinline__ float maskedScaledLogit(float logit, bool masked, float pivot, float invTemp)
{
    float const kept = (masked && !(logit >= pivot)) ? -INFINITY : logit;
    return __fmul_rn(kept, invTemp);
}

struct FloatMax
{
    __device__ __forceinline__ float operator()(float a, float b) const
    {
        return fmaxf(a, b);
    }
};

__device__ __forceinline__ float sampleWarpMax(float value)
{
#pragma unroll
    for (int mask = kWarpSize / 2; mask > 0; mask >>= 1)
    {
        value = fmaxf(value, __shfl_xor_sync(kFullWarpMask, value, mask));
    }
    return value;
}

__device__ __forceinline__ float sampleWarpSum(float value)
{
#pragma unroll
    for (int mask = kWarpSize / 2; mask > 0; mask >>= 1)
    {
        value += __shfl_xor_sync(kFullWarpMask, value, mask);
    }
    return value;
}

// Counts this CTA in the row's counter once every thread's stores are done. Returns true in the last CTA of the row
// to arrive, which resets the counter and then sees the other CTAs' stores (read them with ld.cg).
__device__ __forceinline__ bool arriveLastOfRow(int32_t* rowCounter, int numChunks)
{
    __shared__ bool isLast;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0)
    {
        isLast = atomicAdd(rowCounter, 1) == numChunks - 1;
        if (isLast)
        {
            *rowCounter = 0;
        }
    }
    __syncthreads();
    if (isLast)
    {
        __threadfence();
    }
    return isLast;
}

// Direct sampling: the tokens of the kernels above drawn from per-segment softmax partials, without the probability
// tensor. A warp handles a segment of 512 logits (lane l holds elements 128 q + 4 l + j, q < 4, j < 4); a CTA of the
// first kernel handles a chunk of 16 segments. The second kernel draws on the first CTA of each row, whose threads
// hold up to kSegsPerThread segments each (1 up to 512 segments, else kDirectMaxSegsPerThread); with top_p it has one
// CTA per chunk, which only rows with the top-p test and without the top-k mask use.
constexpr int kDirectSegVecs = 4;
constexpr int kDirectSegElts = kWarpSize * kSampleVecSize * kDirectSegVecs;
constexpr int kDirectWarps = kSampleBlockSize / kWarpSize;
constexpr int kDirectMaxSegsPerThread = 4;
// Segments a warp loads before it reduces them in the per-row passes of the second kernel, and logit vectors a
// thread loads before it counts them in the top-k pivot passes.
constexpr int kDirectSegBatch = 4;
constexpr int kDirectPivotBatch = 8;
// Logits of a top-k row that the pivot search selects from instead of the whole row.
constexpr int kDirectPivotCandidates = 4096;
static_assert(kDirectSegElts == kSampleBlockSize, "the draw rescans a segment with one logit per thread");
static_assert(kDirectSegElts * kDirectWarps == kSampleChunkElts, "a first-kernel CTA covers one sampling chunk");
static_assert(kSampleBlockSize * kDirectMaxSegsPerThread * kDirectSegElts == (1 << 20),
    "the draw holds at most kDirectMaxSegsPerThread segments per thread");

struct IntMax
{
    __device__ __forceinline__ int operator()(int a, int b) const
    {
        return max(a, b);
    }
};

struct IntMin
{
    __device__ __forceinline__ int operator()(int a, int b) const
    {
        return min(a, b);
    }
};

using SampleBlockReduceInt = cub::BlockReduce<int, kSampleBlockSize, cub::BLOCK_REDUCE_WARP_REDUCTIONS>;
using DirectPivotBlockScan = cub::BlockScan<uint32_t, kSampleBlockSize, cub::BLOCK_SCAN_WARP_SCANS>;

union DirectTempStorage
{
    SampleBlockReduce::TempStorage reduce;
    SampleBlockReduceInt::TempStorage reduceInt;
    SampleBlockScan::TempStorage scan;
};

struct DirectKernelParams
{
    float const* logits;
    float const* temperatures;
    int32_t const* topK;
    float const* topP;
    float const* minP;
    bool perRowRng;
    int64_t const* seed;
    int64_t const* offset;
    int32_t* rowCounters; // [numRows], zero before and after the call: the CTAs of a row's first top-p test
    int32_t* tokens;
    float* segMax;        // [numRows, numSegs] max of v = masked logit / T over a segment
    float* segSum;        // [numRows, numSegs] sum of exp(v - segMax) over the segment
    float* segAbove0;     // [numRows, numSegs] sum of raw weights above the first pivot of the last top-p test
    float* segAbove1;     // [numRows, numSegs] sum of raw weights above its second pivot
    float* laneMax;       // [numRows, numSegs * 32] largest logit of each lane's 16 (rows with the top-k mask)
    float* pivotPool;     // [numRows, kDirectPivotCandidates] logits the top-k pivot is selected from
    int vocabSize;
    int numSegs;
    int numChunks; // CTAs per row of the first kernel, 16 segments each
};

// The sampling parameters of a row. They are step inputs written before the forward pass, not by the kernel that
// precedes the sampler, so the direct-sampling kernels read them before the PDL wait.
struct DirectRow
{
    float invTemp;
    float topP;
    float minP;
    float weightFloor;
    int topK;
    bool masked;  // 0 < top_k < vocab: the top-k mask applies
    bool nucleus; // top_p < 1: the top-p test applies (with top_p 1 the test passes on exact sums)
};

__device__ __forceinline__ DirectRow loadDirectRow(DirectKernelParams const& params, int row)
{
    DirectRow r;
    r.invTemp = inverseTemperature(params.temperatures[row]);
    r.topK = params.topK != nullptr ? params.topK[row] : 0;
    r.topP = params.topP != nullptr ? params.topP[row] : 1.F;
    r.masked = r.topK > 0 && r.topK < params.vocabSize;
    r.nucleus = r.topP < 1.F;
    r.minP = fminf(fmaxf(params.minP[row], 0.F), 1.F);
    r.weightFloor = r.minP;
    return r;
}

__device__ __forceinline__ void initDirectRowRng(
    curandStatePhilox4_32_10_t& state, DirectKernelParams const& params, int row)
{
    if (params.perRowRng)
    {
        curand_init(static_cast<unsigned long long>(params.seed[row]),
            (1ULL << 63) | static_cast<unsigned long long>(params.offset[row]), 0, &state);
    }
    else
    {
        curand_init(static_cast<unsigned long long>(params.seed[0]), static_cast<unsigned long long>(row),
            static_cast<unsigned long long>(params.offset[0]), &state);
    }
}

// All membership tests use raw weights relative to the row maximum. Normalizing individual
// weights can merge distinct FP32 weights into ties, changing FULL top-p/top-k support.
__device__ __forceinline__ float directWeight(float logit, DirectRow const& r, float rowMax)
{
    float const w = __expf(__fmul_rn(logit, r.invTemp) - rowMax);
    return w >= r.weightFloor ? w : 0.F;
}

// The 16 logits of segment `seg` that lane `lane` holds; valid[q] is false for the vectors past the row (or seg < 0).
__device__ __forceinline__ void loadSegment(float const* rowLogits, int vocabSize, int seg, int lane,
    float4 (&x)[kDirectSegVecs], bool (&valid)[kDirectSegVecs])
{
#pragma unroll
    for (int q = 0; q < kDirectSegVecs; ++q)
    {
        int const idx = seg * kDirectSegElts + (q * kWarpSize + lane) * kSampleVecSize;
        valid[q] = seg >= 0 && idx < vocabSize;
        x[q] = valid[q] ? *reinterpret_cast<float4 const*>(rowLogits + idx)
                        : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
    }
}

// Max and sum of exp(v - max) of a loaded segment, v = masked logit / T (warp-wide); lane 0 stores them. A segment
// whose maximum is infinite adds nothing, as a chunk of the multi-CTA softmax.
__device__ __forceinline__ void storeSegmentPartials(DirectKernelParams const& params, int row, int seg, float invTemp,
    bool masked, float pivot, float4 const (&x)[kDirectSegVecs], bool const (&valid)[kDirectSegVecs])
{
    float values[kDirectSegVecs][kSampleVecSize];
    float maxValue = -INFINITY;
#pragma unroll
    for (int q = 0; q < kDirectSegVecs; ++q)
    {
        float const xs[kSampleVecSize] = {x[q].x, x[q].y, x[q].z, x[q].w};
#pragma unroll
        for (int j = 0; j < kSampleVecSize; ++j)
        {
            values[q][j] = valid[q] ? maskedScaledLogit(xs[j], masked, pivot, invTemp) : -INFINITY;
            maxValue = fmaxf(maxValue, values[q][j]);
        }
    }
    maxValue = sampleWarpMax(maxValue);
    bool const empty = isinf(maxValue);
    float sum = 0.F;
    if (!empty)
    {
#pragma unroll
        for (int q = 0; q < kDirectSegVecs; ++q)
        {
#pragma unroll
            for (int j = 0; j < kSampleVecSize; ++j)
            {
                sum += __expf(values[q][j] - maxValue);
            }
        }
    }
    sum = sampleWarpSum(sum);
    if (threadIdx.x % kWarpSize == 0)
    {
        int64_t const idx = static_cast<int64_t>(row) * params.numSegs + seg;
        params.segMax[idx] = empty ? -INFINITY : maxValue;
        params.segSum[idx] = empty ? 0.F : sum;
    }
}

// The segments [segBegin, segEnd) of `row` that `take` selects, by the warps of one CTA: warp w tests segments
// segBegin + w, segBegin + w + 16, ... on their max of v (a lane each) and runs fn(seg, x, valid) on the selected
// ones, kDirectSegBatch loaded at a time; skip(seg) runs on the lane that tested a segment `take` rejected.
template <typename Take, typename Fn, typename Skip>
__device__ __forceinline__ void forEachSegment(
    DirectKernelParams const& params, int row, int segBegin, int segEnd, Take take, Fn fn, Skip skip)
{
    int const lane = static_cast<int>(threadIdx.x) % kWarpSize;
    float const* rowLogits = params.logits + static_cast<int64_t>(row) * params.vocabSize;
    int64_t const rowSegs = static_cast<int64_t>(row) * params.numSegs;
    for (int base = segBegin + static_cast<int>(threadIdx.x) / kWarpSize; base < segEnd;
         base += kDirectWarps * kWarpSize)
    {
        int const tested = base + kDirectWarps * lane;
        bool selected = false;
        if (tested < segEnd)
        {
            selected = take(params.segMax[rowSegs + tested]);
            if (!selected)
            {
                skip(tested);
            }
        }
        unsigned pending = __ballot_sync(kFullWarpMask, selected);
        while (pending != 0U)
        {
            int segs[kDirectSegBatch];
            float4 x[kDirectSegBatch][kDirectSegVecs];
            bool valid[kDirectSegBatch][kDirectSegVecs];
#pragma unroll
            for (int b = 0; b < kDirectSegBatch; ++b)
            {
                segs[b] = -1;
                if (pending != 0U)
                {
                    segs[b] = base + kDirectWarps * (__ffs(static_cast<int>(pending)) - 1);
                    pending &= pending - 1U;
                }
                loadSegment(rowLogits, params.vocabSize, segs[b], lane, x[b], valid[b]);
            }
#pragma unroll
            for (int b = 0; b < kDirectSegBatch; ++b)
            {
                if (segs[b] >= 0)
                {
                    fn(segs[b], x[b], valid[b]);
                }
            }
        }
    }
}

// Save the original global maximum before filtered partials overwrite the unfiltered ones.
__device__ float originalRowMax(DirectKernelParams const& params, int row, DirectTempStorage& tempStorage)
{
    __shared__ float result;
    float local = -INFINITY;
    for (int seg = threadIdx.x; seg < params.numSegs; seg += kSampleBlockSize)
    {
        local = fmaxf(local, params.segMax[static_cast<int64_t>(row) * params.numSegs + seg]);
    }
    float const maximum = SampleBlockReduce(tempStorage.reduce).Reduce(local, FloatMax{});
    if (threadIdx.x == 0)
        result = maximum;
    __syncthreads();
    float const value = result;
    __syncthreads();
    return value;
}

// min-p and top-k compose in weight space before top-p. The original maximum always
// survives valid finite-input filters, including min_p=1 and ties at the kth weight.
__device__ void storeFilteredPartials(DirectKernelParams const& params, int row, DirectRow const& r, float rowMax)
{
    int64_t const rowSegs = static_cast<int64_t>(row) * params.numSegs;
    forEachSegment(
        params, row, 0, params.numSegs, [&](float maximum) { return __expf(maximum - rowMax) >= r.weightFloor; },
        [&](int seg, float4 const(&x)[kDirectSegVecs], bool const(&valid)[kDirectSegVecs])
        {
            float values[kDirectSegVecs][kSampleVecSize];
            float maximum = -INFINITY;
#pragma unroll
            for (int q = 0; q < kDirectSegVecs; ++q)
            {
                float const xs[kSampleVecSize] = {x[q].x, x[q].y, x[q].z, x[q].w};
#pragma unroll
                for (int j = 0; j < kSampleVecSize; ++j)
                {
                    float const v = __fmul_rn(xs[j], r.invTemp);
                    bool const keep = valid[q] && __expf(v - rowMax) >= r.weightFloor;
                    values[q][j] = keep ? v : -INFINITY;
                    maximum = fmaxf(maximum, values[q][j]);
                }
            }
            maximum = sampleWarpMax(maximum);
            float sum = 0.F;
            if (isfinite(maximum))
            {
#pragma unroll
                for (int q = 0; q < kDirectSegVecs; ++q)
                {
#pragma unroll
                    for (int j = 0; j < kSampleVecSize; ++j)
                        sum += __expf(values[q][j] - maximum);
                }
            }
            sum = sampleWarpSum(sum);
            if (threadIdx.x % kWarpSize == 0)
            {
                params.segMax[rowSegs + seg] = maximum;
                params.segSum[rowSegs + seg] = sum;
            }
        },
        [&](int seg)
        {
            params.segMax[rowSegs + seg] = -INFINITY;
            params.segSum[rowSegs + seg] = 0.F;
        });
}

// A bounded-retry/rounding fallback. Logits may include masked -inf; a finite scaled maximum
// always return an argmax, which survives all supported filters. All nonfinite rows
// have no specified sampling distribution; return the first maximum when possible.
__device__ int rowArgmax(DirectKernelParams const& params, int row, DirectRow const& r)
{
    __shared__ SampleBlockReduce::TempStorage maxStorage;
    __shared__ SampleBlockReduceInt::TempStorage minStorage;
    __shared__ float maximum;
    __shared__ int result;
    float const* logits = params.logits + static_cast<int64_t>(row) * params.vocabSize;
    float local = -INFINITY;
    for (int i = threadIdx.x; i < params.vocabSize; i += kSampleBlockSize)
        local = fmaxf(local, __fmul_rn(logits[i], r.invTemp));
    float const blockMax = SampleBlockReduce(maxStorage).Reduce(local, FloatMax{});
    if (threadIdx.x == 0)
        maximum = blockMax;
    __syncthreads();
    int candidate = INT_MAX;
    for (int i = threadIdx.x; i < params.vocabSize; i += kSampleBlockSize)
        if (__fmul_rn(logits[i], r.invTemp) == maximum)
            candidate = min(candidate, i);
    int const first = SampleBlockReduceInt(minStorage).Reduce(candidate, IntMin{});
    if (threadIdx.x == 0)
        result = first == INT_MAX ? 0 : first;
    __syncthreads();
    return result;
}

// The k-th largest logit of a row (0 < k < vocabSize) by radix select over the ordered keys, on one CTA. Each thread
// loads kDirectPivotBatch vectors before it counts them.
__device__ float directTopKPivot(float const* rowLogits, int vocabSize, int k)
{
    __shared__ uint32_t histogram[kPivotRadix];
    __shared__ DirectPivotBlockScan::TempStorage scanStorage;
    __shared__ uint32_t selectedDigit;
    __shared__ uint32_t countAbove;
    int const tid = static_cast<int>(threadIdx.x);
    auto const* vecs = reinterpret_cast<float4 const*>(rowLogits);
    int const numVecs = vocabSize / kSampleVecSize;
    uint32_t prefix = 0U;
    uint32_t prefixMask = 0U;
    uint32_t remaining = static_cast<uint32_t>(k);
#pragma unroll 1
    for (int shift = 32 - kPivotRadixBits; shift >= 0; shift -= kPivotRadixBits)
    {
        for (int i = tid; i < kPivotRadix; i += kSampleBlockSize)
        {
            histogram[i] = 0U;
        }
        __syncthreads();
        for (int base = tid; base < numVecs; base += kSampleBlockSize * kDirectPivotBatch)
        {
            float4 x[kDirectPivotBatch];
#pragma unroll
            for (int b = 0; b < kDirectPivotBatch; ++b)
            {
                int const v = base + b * kSampleBlockSize;
                x[b] = v < numVecs ? vecs[v] : make_float4(0.F, 0.F, 0.F, 0.F);
            }
#pragma unroll
            for (int b = 0; b < kDirectPivotBatch; ++b)
            {
                if (base + b * kSampleBlockSize < numVecs)
                {
                    float const elts[kSampleVecSize] = {x[b].x, x[b].y, x[b].z, x[b].w};
#pragma unroll
                    for (int j = 0; j < kSampleVecSize; ++j)
                    {
                        uint32_t const key = toOrderedKey(elts[j]);
                        if ((key & prefixMask) == prefix)
                        {
                            atomicAdd(&histogram[(key >> shift) & (kPivotRadix - 1)], 1U);
                        }
                    }
                }
            }
        }
        __syncthreads();
        // Digits in descending order: the selected digit is the first whose running count reaches `remaining`.
        uint32_t const count = tid < kPivotRadix ? histogram[kPivotRadix - 1 - tid] : 0U;
        uint32_t inclusive;
        DirectPivotBlockScan(scanStorage).InclusiveSum(count, inclusive);
        if (tid < kPivotRadix && inclusive >= remaining && inclusive - count < remaining)
        {
            selectedDigit = kPivotRadix - 1 - tid;
            countAbove = inclusive - count;
        }
        __syncthreads();
        prefix |= selectedDigit << shift;
        prefixMask |= static_cast<uint32_t>(kPivotRadix - 1) << shift;
        remaining -= countAbove;
        __syncthreads();
    }
    return fromOrderedKey(prefix);
}

// The k-th largest logit of a row with the top-k mask (0 < k < vocabSize), on one CTA. One pass stores the largest of
// each lane's 16 logits of a segment (radix order). T, the k-th largest of these lane maxima, is at most the pivot (k
// distinct logits reach it), and every logit >= T lies in a lane whose maximum is >= T: the pivot is the k-th largest
// of the logits >= T, gathered and selected from when there are at most kDirectPivotCandidates of them, else selected
// over the whole row.
__device__ float directPivot(DirectKernelParams const& params, int row, int k)
{
    __shared__ int numCandidates;
    int const tid = static_cast<int>(threadIdx.x);
    int const lane = tid % kWarpSize;
    float const* rowLogits = params.logits + static_cast<int64_t>(row) * params.vocabSize;
    int const numLanes = params.numSegs * kWarpSize;
    if (k <= numLanes)
    {
        float* candidates = params.pivotPool + static_cast<int64_t>(row) * kDirectPivotCandidates;
        float* rowLaneMax = params.laneMax + static_cast<int64_t>(row) * numLanes;
        forEachSegment(
            params, row, 0, params.numSegs, [](float) { return true; },
            [&](int seg, float4 const(&x)[kDirectSegVecs], bool const(&valid)[kDirectSegVecs])
            {
                // A lane past the row holds the smallest key.
                uint32_t laneKey = 0U;
#pragma unroll
                for (int q = 0; q < kDirectSegVecs; ++q)
                {
                    if (valid[q])
                    {
                        laneKey = max(laneKey, toOrderedKey(x[q].x));
                        laneKey = max(laneKey, toOrderedKey(x[q].y));
                        laneKey = max(laneKey, toOrderedKey(x[q].z));
                        laneKey = max(laneKey, toOrderedKey(x[q].w));
                    }
                }
                rowLaneMax[seg * kWarpSize + lane] = fromOrderedKey(laneKey);
            },
            [](int) {});
        __syncthreads();
        uint32_t const boundKey = toOrderedKey(directTopKPivot(rowLaneMax, numLanes, k));
        if (tid == 0)
        {
            numCandidates = 0;
        }
        __syncthreads();
        for (int laneIdx = tid; laneIdx < numLanes; laneIdx += kSampleBlockSize)
        {
            if (toOrderedKey(rowLaneMax[laneIdx]) < boundKey)
            {
                continue;
            }
            int const laneSeg = laneIdx / kWarpSize;
            int const laneInSeg = laneIdx % kWarpSize;
#pragma unroll
            for (int q = 0; q < kDirectSegVecs; ++q)
            {
                int const idx = laneSeg * kDirectSegElts + (q * kWarpSize + laneInSeg) * kSampleVecSize;
                if (idx < params.vocabSize)
                {
                    float4 const x = *reinterpret_cast<float4 const*>(rowLogits + idx);
                    float const xs[kSampleVecSize] = {x.x, x.y, x.z, x.w};
#pragma unroll
                    for (int j = 0; j < kSampleVecSize; ++j)
                    {
                        if (toOrderedKey(xs[j]) >= boundKey)
                        {
                            int const slot = atomicAdd(&numCandidates, 1);
                            if (slot < kDirectPivotCandidates)
                            {
                                candidates[slot] = xs[j];
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
        int const count = numCandidates;
        if (count <= kDirectPivotCandidates)
        {
            // Pad to whole vectors with the smallest key: the k-th largest of count >= k candidates stays.
            int const padded = (count + kSampleVecSize - 1) / kSampleVecSize * kSampleVecSize;
            for (int i = count + tid; i < padded; i += kSampleBlockSize)
            {
                candidates[i] = fromOrderedKey(0U);
            }
            __syncthreads();
            return directTopKPivot(candidates, padded, k);
        }
    }
    return directTopKPivot(rowLogits, params.vocabSize, k);
}

// The last index of segments fromSeg, fromSeg - 1, ..., 0 whose exp(v - rowMax) is > 0 (searched backwards, one
// logit per thread) and that value; -1 when there is none.
__device__ int lastPositiveToken(DirectKernelParams const& params, int row, DirectRow const& r, float rowMax,
    int fromSeg, float& tokenExp, DirectTempStorage& tempStorage)
{
    __shared__ int lastIdx;
    __shared__ float lastExp;
    float const* rowLogits = params.logits + static_cast<int64_t>(row) * params.vocabSize;
    for (int seg = fromSeg; seg >= 0; --seg)
    {
        int const idx = seg * kDirectSegElts + static_cast<int>(threadIdx.x);
        float const e = idx < params.vocabSize ? directWeight(rowLogits[idx], r, rowMax) : 0.F;
        int const blockLast = SampleBlockReduceInt(tempStorage.reduceInt).Reduce(e > 0.F ? idx : -1, IntMax{});
        if (threadIdx.x == 0)
        {
            lastIdx = blockLast;
        }
        __syncthreads();
        int const found = lastIdx;
        if (found >= 0 && idx == found)
        {
            lastExp = e;
        }
        __syncthreads();
        if (found >= 0)
        {
            tokenExp = lastExp;
            return found;
        }
    }
    tokenExp = 0.F;
    return -1;
}

struct DirectDraw
{
    int token;      // -1 when no probability is > 0
    float rowMax;   // max of v over the row
    float rowSum;   // sum of exp(v - rowMax) over the row
    float tokenExp; // exp(v - rowMax) of the token
};

// Inverse-CDF draw for `row` from its segment partials on one CTA: the first token
// whose running sum of exp(v - rowMax) exceeds u * rowSum, which is the first whose running probability exceeds u.
// The segment where the running sum passes u * rowSum comes from the partials, the token from that segment's 512
// logits. u at or above the running sum gives the last positive-weight token. *u is in
// shared memory; it is read after the first barrier. draw.tokenExp is set when needTokenExp.
template <int kSegsPerThread>
__device__ DirectDraw drawFromSegments(DirectKernelParams const& params, int row, DirectRow const& r, float const* u,
    bool needTokenExp, DirectTempStorage& tempStorage)
{
    __shared__ float sharedMax;
    __shared__ int crossSeg;
    __shared__ float crossBefore;
    __shared__ int sampled;
    __shared__ float sampledExp;
    int const tid = static_cast<int>(threadIdx.x);
    int const numSegs = params.numSegs;
    int const perThread = (numSegs + kSampleBlockSize - 1) / kSampleBlockSize;
    int64_t const rowSegs = static_cast<int64_t>(row) * numSegs;
    // Thread t holds segments [t * perThread, (t + 1) * perThread).
    float partMax[kSegsPerThread];
    float partSum[kSegsPerThread];
    float threadMax = -INFINITY;
#pragma unroll
    for (int i = 0; i < kSegsPerThread; ++i)
    {
        int const seg = tid * perThread + i;
        bool const valid = i < perThread && seg < numSegs;
        partMax[i] = valid ? params.segMax[rowSegs + seg] : -INFINITY;
        partSum[i] = valid ? params.segSum[rowSegs + seg] : 0.F;
        threadMax = fmaxf(threadMax, partMax[i]);
    }
    if (tid == 0)
    {
        crossSeg = INT_MAX;
        sampled = INT_MAX;
    }
    float const blockMax = SampleBlockReduce(tempStorage.reduce).Reduce(threadMax, FloatMax{});
    if (tid == 0)
    {
        sharedMax = blockMax;
    }
    __syncthreads();
    DirectDraw draw;
    draw.rowMax = sharedMax;
    draw.token = -1;
    draw.tokenExp = 0.F;
    float weight[kSegsPerThread];
    float threadSum = 0.F;
#pragma unroll
    for (int i = 0; i < kSegsPerThread; ++i)
    {
        weight[i] = isinf(partMax[i]) ? 0.F : partSum[i] * __expf(partMax[i] - draw.rowMax);
        threadSum += weight[i];
    }
    float before;
    SampleBlockScan(tempStorage.scan).ExclusiveSum(threadSum, before, draw.rowSum);
    if (!(draw.rowSum > 0.F))
    {
        // No probability is > 0: every logit masked or -inf, or NaN.
        return draw;
    }
    float const target = *u * draw.rowSum;
    float running = before;
    int cross = INT_MAX;
    float crossRunning = 0.F;
#pragma unroll
    for (int i = 0; i < kSegsPerThread; ++i)
    {
        if (cross == INT_MAX && i < perThread && weight[i] > 0.F && running + weight[i] > target)
        {
            cross = tid * perThread + i;
            crossRunning = running;
        }
        running += weight[i];
    }
    if (cross != INT_MAX)
    {
        atomicMin(&crossSeg, cross);
    }
    __syncthreads();
    int const seg = crossSeg;
    if (seg != INT_MAX && cross == seg)
    {
        crossBefore = crossRunning;
    }
    __syncthreads();
    int lastFrom = numSegs - 1;
    if (seg != INT_MAX)
    {
        // One logit of the segment per thread, in index order.
        float const* rowLogits = params.logits + static_cast<int64_t>(row) * params.vocabSize;
        int const idx = seg * kDirectSegElts + tid;
        float const e = idx < params.vocabSize ? directWeight(rowLogits[idx], r, draw.rowMax) : 0.F;
        float prefix;
        SampleBlockScan(tempStorage.scan).ExclusiveSum(e, prefix);
        if (e > 0.F && crossBefore + prefix + e > target)
        {
            atomicMin(&sampled, idx);
        }
        __syncthreads();
        if (sampled != INT_MAX)
        {
            draw.token = sampled;
            if (needTokenExp)
            {
                if (idx == sampled)
                {
                    sampledExp = e;
                }
                __syncthreads();
                draw.tokenExp = sampledExp;
            }
            return draw;
        }
        // The segment's sum passed u * rowSum but its scan did not (rounding): its last token with probability > 0.
        lastFrom = seg;
    }
    draw.token = lastPositiveToken(params, row, r, draw.rowMax, lastFrom, draw.tokenExp, tempStorage);
    return draw;
}

// The per-segment sums of raw weights above pivot0 and above pivot1 for the segments [segBegin, segEnd) of
// `row`, stored in segAbove0 / segAbove1 by the warps of one CTA. A segment whose largest probability is at most
// min(pivot0, pivot1) has none above either pivot (the probability is monotone in the logit) and stores 0 unread.
__device__ void storeAboveSums(DirectKernelParams const& params, int row, DirectRow const& r, float rowMax,
    float pivot0, float pivot1, int segBegin, int segEnd)
{
    int const lane = static_cast<int>(threadIdx.x) % kWarpSize;
    int64_t const rowSegs = static_cast<int64_t>(row) * params.numSegs;
    float const threshold = fminf(pivot0, pivot1);
    forEachSegment(
        params, row, segBegin, segEnd, [&](float segMaxV) { return __expf(segMaxV - rowMax) > threshold; },
        [&](int seg, float4 const(&x)[kDirectSegVecs], bool const(&valid)[kDirectSegVecs])
        {
            float sum0 = 0.F;
            float sum1 = 0.F;
#pragma unroll
            for (int q = 0; q < kDirectSegVecs; ++q)
            {
                if (valid[q])
                {
                    float const xs[kSampleVecSize] = {x[q].x, x[q].y, x[q].z, x[q].w};
#pragma unroll
                    for (int j = 0; j < kSampleVecSize; ++j)
                    {
                        float const p = directWeight(xs[j], r, rowMax);
                        sum0 += p > pivot0 ? p : 0.F;
                        sum1 += p > pivot1 ? p : 0.F;
                    }
                }
            }
            sum0 = sampleWarpSum(sum0);
            sum1 = sampleWarpSum(sum1);
            if (lane == 0)
            {
                params.segAbove0[rowSegs + seg] = sum0;
                params.segAbove1[rowSegs + seg] = sum1;
            }
        },
        [&](int seg)
        {
            params.segAbove0[rowSegs + seg] = 0.F;
            params.segAbove1[rowSegs + seg] = 0.F;
        });
}

// The sums of segAbove0 and segAbove1 over the row on one CTA, thread t adding its segments [t * perThread,
// (t + 1) * perThread) in order. The sums may come from other CTAs of the row (ld.cg).
template <int kSegsPerThread>
__device__ void aboveTotals(
    DirectKernelParams const& params, int row, float& total0, float& total1, DirectTempStorage& tempStorage)
{
    __shared__ float totals[2];
    int const tid = static_cast<int>(threadIdx.x);
    int const numSegs = params.numSegs;
    int const perThread = (numSegs + kSampleBlockSize - 1) / kSampleBlockSize;
    int64_t const rowSegs = static_cast<int64_t>(row) * numSegs;
    float sum0 = 0.F;
    float sum1 = 0.F;
#pragma unroll
    for (int i = 0; i < kSegsPerThread; ++i)
    {
        int const seg = tid * perThread + i;
        if (i < perThread && seg < numSegs)
        {
            sum0 += __ldcg(&params.segAbove0[rowSegs + seg]);
            sum1 += __ldcg(&params.segAbove1[rowSegs + seg]);
        }
    }
    float const blockSum0 = SampleBlockReduce(tempStorage.reduce).Sum(sum0);
    __syncthreads();
    float const blockSum1 = SampleBlockReduce(tempStorage.reduce).Sum(sum1);
    if (tid == 0)
    {
        totals[0] = blockSum0;
        totals[1] = blockSum1;
    }
    __syncthreads();
    total0 = totals[0];
    total1 = totals[1];
    __syncthreads();
}

// One retry inverse-CDF draw on one CTA, for the tokens with raw weight > low (the
// weights of the original draw), from their per-segment sums segSums (stored by the top-p test that
// chose low): the segment where the running sum passes u, then the token among its 512 logits. u at or above the
// sum gives the last token with probability > low. Returns -1 when none has.
template <int kSegsPerThread>
__device__ int sampleAboveLow(DirectKernelParams const& params, int row, DirectRow const& r, float rowMax, float low,
    float const* segSums, float u, DirectTempStorage& tempStorage)
{
    __shared__ int crossSeg;
    __shared__ float crossBefore;
    __shared__ int lastSeg;
    __shared__ int sampledIdx;
    int const tid = static_cast<int>(threadIdx.x);
    int const numSegs = params.numSegs;
    int const perThread = (numSegs + kSampleBlockSize - 1) / kSampleBlockSize;
    if (tid == 0)
    {
        crossSeg = INT_MAX;
        sampledIdx = INT_MAX;
    }
    // Thread t holds segments [t * perThread, (t + 1) * perThread).
    float sums[kSegsPerThread];
    float threadSum = 0.F;
    int threadLast = -1;
#pragma unroll
    for (int i = 0; i < kSegsPerThread; ++i)
    {
        int const seg = tid * perThread + i;
        sums[i] = i < perThread && seg < numSegs ? __ldcg(&segSums[seg]) : 0.F;
        threadSum += sums[i];
        if (sums[i] > 0.F)
        {
            threadLast = seg;
        }
    }
    float before;
    SampleBlockScan(tempStorage.scan).ExclusiveSum(threadSum, before);
    float running = before;
    int cross = INT_MAX;
    float crossRunning = 0.F;
#pragma unroll
    for (int i = 0; i < kSegsPerThread; ++i)
    {
        if (cross == INT_MAX && sums[i] > 0.F && running + sums[i] > u)
        {
            cross = tid * perThread + i;
            crossRunning = running;
        }
        running += sums[i];
    }
    __syncthreads();
    int const blockLast = SampleBlockReduceInt(tempStorage.reduceInt).Reduce(threadLast, IntMax{});
    if (cross != INT_MAX)
    {
        atomicMin(&crossSeg, cross);
    }
    if (tid == 0)
    {
        lastSeg = blockLast;
    }
    __syncthreads();
    int const seg = crossSeg;
    if (seg != INT_MAX && cross == seg)
    {
        crossBefore = crossRunning;
    }
    // Without a crossing segment (u at or above the sum) the result is the last token with probability > low.
    int const scanSeg = seg != INT_MAX ? seg : lastSeg;
    __syncthreads();
    if (scanSeg < 0)
    {
        return -1;
    }
    // One logit of the segment per thread, in index order.
    float const* rowLogits = params.logits + static_cast<int64_t>(row) * params.vocabSize;
    int const idx = scanSeg * kDirectSegElts + tid;
    float const p = idx < params.vocabSize ? directWeight(rowLogits[idx], r, rowMax) : 0.F;
    bool const kept = p > low;
    if (seg != INT_MAX)
    {
        float prefix;
        SampleBlockScan(tempStorage.scan).ExclusiveSum(kept ? p : 0.F, prefix);
        if (kept && crossBefore + prefix + p > u)
        {
            atomicMin(&sampledIdx, idx);
        }
        __syncthreads();
        if (sampledIdx != INT_MAX)
        {
            int const result = sampledIdx;
            __syncthreads();
            return result;
        }
        // The segment's sum passed u but its scan did not (rounding): its last token with probability > low.
    }
    int const segLast = SampleBlockReduceInt(tempStorage.reduceInt).Reduce(kept ? idx : -1, IntMax{});
    if (tid == 0)
    {
        sampledIdx = segLast;
    }
    __syncthreads();
    int const result = sampledIdx;
    __syncthreads();
    return result;
}

// FULL top-p uses raw weights and a fixed target based on the original filtered mass.
// Strictly greater weight defines the mass above a candidate, preserving complete tie groups.
// Every retry uses the same RNG mapping and support domain as the initial draw.
template <int kSegsPerThread>
__device__ int directTopPRounds(DirectKernelParams const& params, int row, DirectRow const& r, DirectDraw const& draw,
    float pivot0, float pivot1, float aggregate0, float aggregate1, DirectTempStorage& tempStorage)
{
    int token = draw.token;
    float const target = r.topP * draw.rowSum;
    if (aggregate0 <= target)
    {
        return token;
    }
    int64_t const rowSegs = static_cast<int64_t>(row) * params.numSegs;
    float low = 0.F;
    float high = 1.F;
    float q = draw.rowSum;
    // low is the first pivot of the last test (its sums are in segAbove0), else its second pivot (segAbove1).
    bool lowIsPivot0 = true;
    if (aggregate1 <= target)
    {
        low = pivot0;
        high = pivot1;
        q = aggregate0;
    }
    else
    {
        low = pivot1;
        q = aggregate1;
        lowIsPivot0 = false;
    }
    curandStatePhilox4_32_10_t state;
    initDirectRowRng(state, params, row);
    curand_uniform(&state);
    float const* rowLogits = params.logits + static_cast<int64_t>(row) * params.vocabSize;
    for (int round = 0; round < 32 && low < high; ++round)
    {
        float const u = curand_uniform(&state) * q;
        float const* lowSums = (lowIsPivot0 ? params.segAbove0 : params.segAbove1) + rowSegs;
        token = sampleAboveLow<kSegsPerThread>(params, row, r, draw.rowMax, low, lowSums, u, tempStorage);
        if (token < 0)
        {
            return rowArgmax(params, row, r);
        }
        float const roundPivot0 = directWeight(rowLogits[token], r, draw.rowMax);
        float const roundPivot1 = ieeeMul(ieeeAdd(roundPivot0, high), 0.5F);
        storeAboveSums(params, row, r, draw.rowMax, roundPivot0, roundPivot1, 0, params.numSegs);
        __syncthreads();
        float roundAbove0;
        float roundAbove1;
        aboveTotals<kSegsPerThread>(params, row, roundAbove0, roundAbove1, tempStorage);
        if (roundAbove0 <= target)
        {
            return token;
        }
        if (roundAbove1 <= target)
        {
            low = roundPivot0;
            high = roundPivot1;
            q = roundAbove0;
            lowIsPivot0 = true;
        }
        else
        {
            low = roundPivot1;
            q = roundAbove1;
            lowIsPivot0 = false;
        }
    }
    return rowArgmax(params, row, r);
}

// First direct-sampling kernel, CTA blockIdx.x of row blockIdx.y: max and sum of exp(v - max) of v = logit / T for
// its 16 segments, a warp each. A row with the top-k mask gets its masked partials in the second kernel.
__global__ void __launch_bounds__(kSampleBlockSize) directPartialsKernel(DirectKernelParams params)
{
    int const cta = static_cast<int>(blockIdx.x);
    int const row = static_cast<int>(blockIdx.y);
    // The temperature is a step input, not written by the preceding kernel.
    float const invTemp = inverseTemperature(params.temperatures[row]);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // The logits are written by the preceding kernel. Dependents read the outputs only after their own wait.
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    int const seg = cta * kDirectWarps + static_cast<int>(threadIdx.x) / kWarpSize;
    if (seg >= params.numSegs)
    {
        return;
    }
    float4 x[kDirectSegVecs];
    bool valid[kDirectSegVecs];
    loadSegment(params.logits + static_cast<int64_t>(row) * params.vocabSize, params.vocabSize, seg,
        static_cast<int>(threadIdx.x) % kWarpSize, x, valid);
    storeSegmentPartials(params, row, seg, invTemp, false, 0.F, x, valid);
}

// Second direct-sampling kernel, CTA (row blockIdx.x, chunk blockIdx.y): the draw from the segment partials. A row
// with the top-k mask first finds the k-th largest logit and its masked partials. A row with top_p < 1 then runs the
// top-p test and its later rounds; without the mask, its first test is spread over one CTA per chunk of 16 segments
// (each repeats the draw), and the last CTA of the row to finish runs the rounds. Every other row runs on its first
// CTA alone.
template <int kSegsPerThread>
__global__ void __launch_bounds__(kSampleBlockSize) directDrawKernel(DirectKernelParams params)
{
    __shared__ DirectTempStorage tempStorage;
    __shared__ float drawU;
    int const row = static_cast<int>(blockIdx.x);
    int const chunk = static_cast<int>(blockIdx.y);
    DirectRow r = loadDirectRow(params, row);
    bool const split = r.nucleus && !r.masked && r.minP == 0.F;
    if (chunk != 0 && !split)
    {
        return;
    }
    if (threadIdx.x == 0)
    {
        // The first draw of the row's Philox stream: the seed and the offset are step inputs.
        curandStatePhilox4_32_10_t state;
        initDirectRowRng(state, params, row);
        drawU = curand_uniform(&state);
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // The partials are written by the preceding kernel.
    cudaGridDependencySynchronize();
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    float pivot = 0.F;
    if (r.masked || r.minP > 0.F)
    {
        float const maximum = originalRowMax(params, row, tempStorage);
        if (r.masked)
        {
            pivot = directPivot(params, row, r.topK);
            r.weightFloor = fmaxf(r.minP, __expf(__fmul_rn(pivot, r.invTemp) - maximum));
        }
        storeFilteredPartials(params, row, r, maximum);
        __syncthreads();
    }
    DirectDraw const draw = drawFromSegments<kSegsPerThread>(params, row, r, &drawU, r.nucleus, tempStorage);
    if (draw.token < 0 || !r.nucleus)
    {
        int const result = draw.token < 0 ? rowArgmax(params, row, r) : draw.token;
        if (chunk == 0 && threadIdx.x == 0)
        {
            params.tokens[row] = result;
        }
        return;
    }
    float const pivot0 = draw.tokenExp;
    float const pivot1 = ieeeMul(ieeeAdd(pivot0, 1.F), 0.5F);
    if (split)
    {
        int const segBegin = chunk * kDirectWarps;
        storeAboveSums(
            params, row, r, draw.rowMax, pivot0, pivot1, segBegin, min(segBegin + kDirectWarps, params.numSegs));
        if (!arriveLastOfRow(&params.rowCounters[row], params.numChunks))
        {
            return;
        }
    }
    else
    {
        storeAboveSums(params, row, r, draw.rowMax, pivot0, pivot1, 0, params.numSegs);
        __syncthreads();
    }
    float aggregate0;
    float aggregate1;
    aboveTotals<kSegsPerThread>(params, row, aggregate0, aggregate1, tempStorage);
    int const token
        = directTopPRounds<kSegsPerThread>(params, row, r, draw, pivot0, pivot1, aggregate0, aggregate1, tempStorage);
    if (threadIdx.x == 0)
    {
        params.tokens[row] = token;
    }
}

} // namespace

size_t getGlmVerifyHeadSampleWorkspaceSize(int numRows, int vocabSize)
{
    size_t const numSegs = static_cast<size_t>((vocabSize + kDirectSegElts - 1) / kDirectSegElts);
    return ((4 + kWarpSize) * numSegs + kDirectPivotCandidates) * static_cast<size_t>(numRows) * sizeof(float);
}

void invokeGlmVerifyHeadSample(GlmVerifyHeadSampleParams const& params)
{
    TLLM_CHECK_WITH_INFO(params.vocabSize == 154880 && params.numRows >= 1 && params.numRows <= 24,
        "GLM verify sampling requires 1..24 rows and vocabulary 154880");
    TLLM_CHECK(params.rowCounters != nullptr && params.minP != nullptr && params.topP != nullptr);
    int const numSegs = (params.vocabSize + kDirectSegElts - 1) / kDirectSegElts;
    int const numChunks = numSampleChunks(params.vocabSize);
    size_t const rowSegs = static_cast<size_t>(params.numRows) * static_cast<size_t>(numSegs);
    auto* base = static_cast<float*>(params.workspace);

    DirectKernelParams kernelParams;
    kernelParams.logits = params.logits;
    kernelParams.temperatures = params.temperatures;
    kernelParams.topK = params.topK;
    kernelParams.topP = params.topP;
    kernelParams.minP = params.minP;
    kernelParams.perRowRng = params.perRowRng;
    kernelParams.seed = params.seed;
    kernelParams.offset = params.offset;
    kernelParams.rowCounters = params.rowCounters;
    kernelParams.tokens = params.tokens;
    kernelParams.segMax = base;
    kernelParams.segSum = kernelParams.segMax + rowSegs;
    kernelParams.segAbove0 = kernelParams.segSum + rowSegs;
    kernelParams.segAbove1 = kernelParams.segAbove0 + rowSegs;
    kernelParams.laneMax = kernelParams.segAbove1 + rowSegs;
    kernelParams.pivotPool = kernelParams.laneMax + kWarpSize * rowSegs;
    kernelParams.vocabSize = params.vocabSize;
    kernelParams.numSegs = numSegs;
    kernelParams.numChunks = numChunks;

    tensorrt_llm::common::launchWithPdlWhenEnabled("glmVerifyHeadPartials", directPartialsKernel,
        dim3(static_cast<uint32_t>(numChunks), static_cast<uint32_t>(params.numRows)), dim3(kSampleBlockSize), 0,
        params.stream, kernelParams);
    // With top_p, a row without the top-k mask may spread its first top-p test over numChunks CTAs; the first CTA of
    // every row comes first in launch order.
    dim3 const drawGrid(
        static_cast<uint32_t>(params.numRows), static_cast<uint32_t>(params.topP != nullptr ? numChunks : 1));
    tensorrt_llm::common::launchWithPdlWhenEnabled(
        "glmVerifyHeadDraw", directDrawKernel<1>, drawGrid, dim3(kSampleBlockSize), 0, params.stream, kernelParams);
}

} // namespace kernels::glm_moe_dsa

TRTLLM_NAMESPACE_END
