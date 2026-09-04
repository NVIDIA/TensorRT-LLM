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
#pragma once

#include "fusedSamplingKernels.h"
#include "fusedSamplingKernelsInternal.h"

#include <cfloat>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>

namespace tensorrt_llm
{
namespace kernels
{

namespace fusedSampling
{

//! Digits per radix pass of the threshold search. 8 bits -> 256 buckets, 4 passes to pin
//! a float exactly.
constexpr int kRadixBits = 8;
constexpr int kRadixBuckets = 1 << kRadixBits;
constexpr int kRadixPasses = 32 / kRadixBits;
//! Private histogram copies, one per group of warps, spreading the few buckets a softmax
//! actually lands in across separate addresses.
//!
//! The descent's histogram scatter is bound by shared-atomic contention rather than by
//! bandwidth, so this count governs its cost. 16 copies and a 2048-entry gather buffer
//! together sit just under the 48 KB static shared-memory limit; raising either means
//! lowering the other.
constexpr int kHistCopies = 16;
//! How many survivors the shared gather buffer holds (8 KB). Past this the descent falls
//! back to re-reading the row, so the cap is a performance knob, never a correctness one.
constexpr int kCandCap = 2048;
//! Guards 1/T for a zero temperature. Matches decodingCommon.cu's EPSILON for float.
constexpr float kTempEpsilon = 1e-6f;
//! Rejection rounds a tokens-only row may spend before it settles for the argmax. Each
//! round drops the rejected candidate and everything at or below its weight, so the
//! support collapses geometrically; the cap exists because that is a statement about the
//! expectation, not the worst case. The argmax belongs to every kept set this kernel can
//! build, so exhausting the budget still answers with a token inside the support -- it
//! perturbs the distribution rather than leaving it.
constexpr int kMaxRejectRounds = 32;

__device__ inline float toFloat(float v)
{
    return v;
}

__device__ inline float toFloat(__half v)
{
    return __half2float(v);
}

__device__ inline float toFloat(__nv_bfloat16 v)
{
    return __bfloat162float(v);
}

//! Bytes moved by one vector load. 16 is the widest a single instruction issues.
constexpr int kVecBytes = 16;

//! \brief Sweep one row, 16 bytes per thread per step, calling ``fn(index, logit)``.
//!
//! Every stage of this kernel is a full sweep of [vocabSize], so how efficiently one
//! sweep reads memory multiplies through all of them, and scalar 4-byte loads leave most
//! of the achievable bandwidth unused.
//!
//! Falls back to scalar when the row is not 16-byte aligned or not a whole number of
//! vectors. Both are properties of the base pointer and vocabSize, so the branch is
//! uniform across the block -- no divergence.
template <typename T, typename Fn>
__device__ inline void forEachLogit(T const* row, int vocabSize, Fn fn)
{
    constexpr int kWidth = kVecBytes / sizeof(T);
    if ((reinterpret_cast<uintptr_t>(row) % kVecBytes) != 0)
    {
        for (int i = threadIdx.x; i < vocabSize; i += blockDim.x)
        {
            fn(i, toFloat(row[i]));
        }
        return;
    }

    int const vecCount = vocabSize / kWidth;
    for (int v = threadIdx.x; v < vecCount; v += blockDim.x)
    {
        int4 const packed = reinterpret_cast<int4 const*>(row)[v];
        T const* elems = reinterpret_cast<T const*>(&packed);
#pragma unroll
        for (int j = 0; j < kWidth; ++j)
        {
            fn(v * kWidth + j, toFloat(elems[j]));
        }
    }
    for (int i = vecCount * kWidth + threadIdx.x; i < vocabSize; i += blockDim.x)
    {
        fn(i, toFloat(row[i]));
    }
}

//! Row-local view of one request's sampling parameters, resolved once per block.
struct RowParams
{
    float tempInv;
    float minP;
    float topP;
    int32_t topK;
    bool needTopK;
    bool needTopP;
    bool needMinP;
};

__device__ inline RowParams loadRowParams(FusedSamplingParams const& p, int row)
{
    RowParams r;
    float const temperature = p.temperatures != nullptr ? p.temperatures[row] : 1.0f;
    r.tempInv = 1.0f / (temperature + kTempEpsilon);

    r.minP = p.minPs != nullptr ? p.minPs[row] : 0.0f;
    r.topP = p.topPs != nullptr ? p.topPs[row] : 1.0f;
    r.topK = p.topKs != nullptr ? p.topKs[row] : 0;

    // Every "disabled" spelling the callers use collapses here, so the rest of the
    // kernel tests one boolean instead of re-deriving the sentinel convention:
    // top_k <= 0 means "all logits" per SamplingParams, and INT32_MAX is the
    // one-model path's explicit disable value -- both clamp to vocabSize.
    r.needTopK = r.topK > 0 && r.topK < p.vocabSize;
    r.needTopP = r.topP < 1.0f;
    r.needMinP = r.minP > 0.0f;
    return r;
}

//! The temperature-scaled logit. __fmul_rn, not `*`, and deliberately so: the compiler is
//! free to contract `l * tempInv - maxScaled` into an FMA, which rounds once instead of
//! twice. maxScaled was produced by a separate multiply, so the contracted form does not
//! cancel to exactly 0 at the argmax -- it lands a few ULP below, w comes out just under
//! 1.0, and `min_p == 1.0` (keep only p == p_max) then keeps nothing at all and divides by
//! a zero mass. Forcing the same rounding in both places is what makes that cancellation
//! exact.
template <typename T>
__device__ inline float scaledLogit(T const* rowLogits, int idx, float tempInv)
{
    return __fmul_rn(toFloat(rowLogits[idx]), tempInv);
}

//! w = exp(l/T - maxScaledLogit), i.e. p / p_max before normalization. Recomputed from
//! the logits on each pass rather than staged in a workspace: the read is the same
//! bandwidth either way, and a [numRows, vocabSize] scratch buffer would have to be
//! allocated (and CUDA-graph-pinned) for the tokens-only calls that never want probs.
template <typename T>
__device__ inline float weightOf(T const* rowLogits, int idx, float tempInv, float maxScaled)
{
    return __expf(scaledLogit(rowLogits, idx, tempInv) - maxScaled);
}

//! Running (max, mass, argmax) of an online softmax: ``sum`` is the mass of everything
//! seen so far, expressed relative to ``max``, so it stays finite whatever the logits are.
struct OnlineSoftmax
{
    float max;
    float sum;
    int argmax;
};

constexpr int kOnlineSoftmaxFloats = (sizeof(OnlineSoftmax) + sizeof(float) - 1) / sizeof(float);
constexpr int kSplitStatsFloats = kOnlineSoftmaxFloats * kSmallBatchSplits;
constexpr int kMergedStatsOffset = kSplitStatsFloats;
constexpr int kSplitMassOffset = kMergedStatsOffset + kOnlineSoftmaxFloats;
constexpr int kSmallBatchWorkspaceFloats = kSplitMassOffset + kSmallBatchSplits;
constexpr int kSplitHistogramOffset = kSmallBatchWorkspaceFloats;
constexpr int kSplitHistogramStride = 2 * kRadixBuckets;
static_assert(sizeof(OnlineSoftmax) % sizeof(float) == 0);
static_assert(kSplitHistogramOffset + kSmallBatchSplits * kSplitHistogramStride <= kSmallBatchSplitMinVocab);

//! Merge two partial online-softmax states by rebasing the smaller max onto the larger.
//! Associative and commutative, so a block reduce over it is order-independent -- which is
//! what keeps two TP ranks holding identical logits in agreement.
__device__ inline OnlineSoftmax combineOnlineSoftmax(OnlineSoftmax a, OnlineSoftmax b)
{
    if (a.max >= b.max)
    {
        return OnlineSoftmax{a.max, a.sum + b.sum * __expf(b.max - a.max), a.argmax};
    }
    return OnlineSoftmax{b.max, b.sum + a.sum * __expf(a.max - b.max), b.argmax};
}

struct OnlineSoftmaxOp
{
    __device__ inline OnlineSoftmax operator()(OnlineSoftmax const& a, OnlineSoftmax const& b) const
    {
        return combineOnlineSoftmax(a, b);
    }
};

//! Max and argmax without the online mass. The tokens-only rejection path recomputes
//! that mass in the scan which draws its candidate, so evaluating exp while finding the
//! max would be duplicate work.
struct MaxArg
{
    float max;
    int argmax;
};

struct MaxArgOp
{
    __device__ inline MaxArg operator()(MaxArg const& a, MaxArg const& b) const
    {
        return a.max >= b.max ? a : b;
    }
};

//! How much sits strictly above a candidate weight: how many entries, and how much mass.
//! Both of the rank/mass criteria are statements about exactly this, so the rejection test
//! costs one sweep and one reduction rather than one of each per criterion. ``count`` is a
//! float because the reduction is shared with ``mass`` and a vocabulary never approaches
//! 2^24, where a float stops counting exactly.
struct CountMass
{
    float count;
    float mass;
};

struct CountMassOp
{
    __device__ inline CountMass operator()(CountMass const& a, CountMass const& b) const
    {
        return CountMass{a.count + b.count, a.mass + b.mass};
    }
};

//! \brief Find the cutoff value for ONE rank-or-mass criterion, over the weights that
//!        survived the filters already applied.
//!
//! ``byCount``: fire when the running count reaches ``targetCount`` (top-k).
//! otherwise:   fire when the running mass exceeds ``targetMass``    (top-p).
//!
//! ``floor`` excludes what earlier filters already removed, so the count and the mass are
//! both taken over the surviving set. That is what makes this composable, and it is the
//! whole reason top-k and top-p need two calls rather than one -- see the caller.
//!
//! MSB-first radix over the float bit pattern, which is monotone for non-negative floats
//! (w is an exp, so it never is). Four 8-bit passes pin the boundary value exactly.
//!
//! Returns 0 when the criterion never fires (this filter keeps everything).
template <typename T>
__device__ float findThreshold(T const* rowLogits, int vocabSize, float tempInv, float maxScaled, float floorValue,
    long long targetCount, float targetMass, bool byCount, int* sCount, float* sMass, float* sCand, int* sCandCount,
    int* sBucketCount, int* sChosen, long long* sCountHi, float* sMassHi, bool* sFired,
    bool firstHistogramReady = false)
{
    uint32_t prefix = 0u;
    uint32_t fixedMask = 0u;
    long long countHi = 0;
    float massHi = 0.0f;

    int const tid = threadIdx.x;
    //! Once the survivors have been gathered into shared memory the remaining passes read
    //! them from there, and the row is not touched again.
    bool useShared = false;

    for (int pass = 0; pass < kRadixPasses; ++pass)
    {
        int const shift = 32 - kRadixBits * (pass + 1);
        if (tid == 0)
        {
            *sFired = false;
        }

        if (!(firstHistogramReady && pass == 0))
        {
            for (int b = tid; b < kRadixBuckets * kHistCopies; b += blockDim.x)
            {
                sCount[b] = 0;
                sMass[b] = 0.0f;
            }
            __syncthreads();

            // Which private copy this thread accumulates into. Contention, not bandwidth, is
            // what makes a shared-memory histogram slow here: w lies in (0, 1], so the top 8
            // bits only ever take ~64 of the 256 values, and a softmax concentrates most of a
            // 128k-element row into a handful of those. Every element issues two atomics, so a
            // single shared histogram serializes the whole pass on a few addresses. Splitting
            // by warp group divides that contention by kHistCopies at the cost of one cheap
            // fold afterwards.
            int const copy = (tid >> 5) & (kHistCopies - 1);
            int* const myCount = sCount + copy * kRadixBuckets;
            float* const myMass = sMass + copy * kRadixBuckets;

            if (useShared)
            {
                int const n = *sCandCount;
                for (int i = tid; i < n; i += blockDim.x)
                {
                    float const w = sCand[i];
                    uint32_t const bits = __float_as_uint(w);
                    if ((bits & fixedMask) != prefix)
                    {
                        continue;
                    }
                    uint32_t const digit = (bits >> shift) & (kRadixBuckets - 1);
                    atomicAdd(&myCount[digit], 1);
                    if (!byCount)
                    {
                        atomicAdd(&myMass[digit], w);
                    }
                }
            }
            else
            {
                forEachLogit(rowLogits, vocabSize,
                    [&](int, float logit)
                    {
                        float const w = __expf(__fmul_rn(logit, tempInv) - maxScaled);
                        if (w < floorValue)
                        {
                            return;
                        }
                        uint32_t const bits = __float_as_uint(w);
                        if ((bits & fixedMask) != prefix)
                        {
                            return;
                        }
                        uint32_t const digit = (bits >> shift) & (kRadixBuckets - 1);
                        atomicAdd(&myCount[digit], 1);
                        if (!byCount)
                        {
                            atomicAdd(&myMass[digit], w);
                        }
                    });
            }
            __syncthreads();

            // Fold the private copies into copy 0, one bucket per thread, keeping the raw
            // per-bucket totals in the space copy 1 occupied -- the suffix scan below runs in
            // place and the chosen bucket's own count is still needed afterwards.
            for (int b = tid; b < kRadixBuckets; b += blockDim.x)
            {
                int totalCount = sCount[b];
                float totalMass = byCount ? 0.0f : sMass[b];
#pragma unroll
                for (int c = 1; c < kHistCopies; ++c)
                {
                    totalCount += sCount[c * kRadixBuckets + b];
                    if (!byCount)
                    {
                        totalMass += sMass[c * kRadixBuckets + b];
                    }
                }
                sCount[b] = totalCount;
                sCount[kRadixBuckets + b] = totalCount;
                if (!byCount)
                {
                    sMass[b] = totalMass;
                    sMass[kRadixBuckets + b] = totalMass;
                }
            }
            __syncthreads();
        }

        // Descending suffix sums, Hillis-Steele: 8 parallel steps instead of the 256
        // dependent shared-memory reads a serial walk costs.
        //
        // This walk was originally one thread stepping bucket 255 down to 0 while the
        // other 1023 idled, four times per descent. At 64 rows the kernel moves ~32 MB
        // total -- microseconds of bandwidth -- yet took ~75us, so it was never
        // bandwidth-bound there; this dependent chain was the reason.
#pragma unroll
        for (int offset = 1; offset < kRadixBuckets; offset <<= 1)
        {
            int addCount = 0;
            float addMass = 0.0f;
            bool const active = tid < kRadixBuckets && tid + offset < kRadixBuckets;
            if (active)
            {
                addCount = sCount[tid + offset];
                addMass = byCount ? 0.0f : sMass[tid + offset];
            }
            __syncthreads();
            if (active)
            {
                sCount[tid] += addCount;
                if (!byCount)
                {
                    sMass[tid] += addMass;
                }
            }
            __syncthreads();
        }

        // sCount[b] is now the count of everything in buckets >= b, so "the criterion has
        // fired by the time the walk reaches b" is a per-bucket predicate. It holds for a
        // suffix of small b, so the bucket the serial walk would have stopped at is simply
        // the largest b where it holds.
        if (tid == 0)
        {
            *sChosen = -1;
        }
        __syncthreads();
        if (tid < kRadixBuckets)
        {
            bool const fired = byCount ? countHi + sCount[tid] >= targetCount : massHi + sMass[tid] > targetMass;
            if (fired)
            {
                atomicMax(sChosen, tid);
            }
        }
        __syncthreads();

        if (tid == 0)
        {
            int const chosen = *sChosen;
            *sFired = chosen >= 0;
            int const b = chosen < 0 ? 0 : chosen;
            *sChosen = b;
            // Everything strictly above the chosen bucket, i.e. what the serial walk had
            // accumulated before it stopped.
            *sCountHi = countHi + (b + 1 < kRadixBuckets ? sCount[b + 1] : 0);
            *sMassHi = byCount ? 0.0f : massHi + (b + 1 < kRadixBuckets ? sMass[b + 1] : 0.0f);
            // How many survivors the next pass has to look at. Read from the raw copy,
            // since sCount now holds suffix sums.
            *sBucketCount = sCount[kRadixBuckets + b];
        }
        __syncthreads();

        // Only the first pass can legitimately not fire: it walks the whole range, so
        // "never fired" means the filters keep everything (top_k >= the number of
        // survivors; top_p always fires while topP < 1). A later pass descends into a
        // bucket that is known to contain the crossing, and a non-firing walk there just
        // means the crossing sits at the bucket's floor, which chosen == 0 already says.
        if (pass == 0 && !*sFired)
        {
            return 0.0f;
        }
        prefix |= static_cast<uint32_t>(*sChosen) << shift;
        fixedMask |= static_cast<uint32_t>(kRadixBuckets - 1) << shift;
        countHi = *sCountHi;
        massHi = *sMassHi;
        __syncthreads();

        // Everything still in play now lies in one bucket, and after the first pass that
        // is a tiny fraction of the row -- yet each further pass was re-reading the whole
        // row to histogram it. Gather the survivors once and the remaining passes never
        // touch global memory again: 4 sweeps become 2.
        //
        // Skipped when the bucket is too wide to fit, which keeps this an optimization and
        // not a correctness condition: the fallback is the original full-row pass.
        if (!useShared && pass + 1 < kRadixPasses && *sBucketCount <= kCandCap)
        {
            if (tid == 0)
            {
                *sCandCount = 0;
            }
            __syncthreads();
            forEachLogit(rowLogits, vocabSize,
                [&](int, float logit)
                {
                    float const w = __expf(__fmul_rn(logit, tempInv) - maxScaled);
                    if (w < floorValue)
                    {
                        return;
                    }
                    uint32_t const bits = __float_as_uint(w);
                    if ((bits & fixedMask) != prefix)
                    {
                        return;
                    }
                    int const slot = atomicAdd(sCandCount, 1);
                    if (slot < kCandCap)
                    {
                        sCand[slot] = w;
                    }
                });
            __syncthreads();
            // sBucketCount is exact, so the cap cannot have been exceeded -- but a
            // gathered set that does not match it would mean the later passes silently saw
            // fewer elements, so it is checked rather than assumed.
            useShared = *sCandCount <= kCandCap;
            __syncthreads();
        }
    }

    return __uint_as_float(prefix);
}

template <int BLOCK>
struct FusedSamplingShared
{
    using BlockReduceF = cub::BlockReduce<float, BLOCK>;
    using BlockReduceOnline = cub::BlockReduce<OnlineSoftmax, BLOCK>;
    using BlockReduceMaxArg = cub::BlockReduce<MaxArg, BLOCK>;
    using BlockReduceCountMass = cub::BlockReduce<CountMass, BLOCK>;
    using BlockScanF = cub::BlockScan<float, BLOCK>;

    union
    {
        typename BlockReduceF::TempStorage reduceF;
        typename BlockReduceOnline::TempStorage reduceOnline;
        typename BlockReduceMaxArg::TempStorage reduceMaxArg;
        typename BlockReduceCountMass::TempStorage reduceCountMass;
        typename BlockScanF::TempStorage scanF;
    } temp;

    int count[kRadixBuckets * kHistCopies];
    float mass[kRadixBuckets * kHistCopies];
    float cand[kCandCap];
    int candCount;
    int bucketCount;
    int chosen;
    long long countHi;
    float massHi;
    bool fired;
    float maxScaled;
    float totalMass;
    float keptMass;
    float survivingMass;
    float target;
    int argmax;
    int token;
    float pivot;
    float candWeight;
    float massTarget;
    int candIdx;
    curandStatePhilox4_32_10_t rejectRng;
};

//! \brief The kernel body, shared by both entry points below.
//!
//! A __device__ function rather than the __global__ itself because only ONE of the six
//! instantiations wants an occupancy bound, and __launch_bounds__ lives on the entry
//! point. Writing it as a template-dependent expression on a single __global__ does not
//! work: naming any bound changes what ptxas does even where the expression is the
//! architectural default: `__launch_bounds__(1024, 1)` on an instantiation that wants no
//! bound is measurably worse than no attribute at all. So the bounded case gets its own
//! entry point and every other instantiation keeps none.
//! \brief Fused temperature + min-p + top-k + top-p + (probs | sampling), one block per row.
template <typename T, int BLOCK, bool NEED_TOKENS, bool NEED_PROBS, bool PRECOMPUTED_STATS = false>
__device__ void fusedSamplingBody(FusedSamplingParams const& params, FusedSamplingShared<BLOCK>& shared)
{
    int const row = blockIdx.x;
    int const tid = threadIdx.x;
    int const vocabSize = params.vocabSize;
    T const* rowLogits = static_cast<T const*>(params.logits) + static_cast<size_t>(row) * vocabSize;
    float* rowProbs = NEED_PROBS ? params.outputProbs + static_cast<size_t>(row) * vocabSize : nullptr;

    RowParams const rp = loadRowParams(params, row);

    using BlockReduceF = cub::BlockReduce<float, BLOCK>;
    using BlockReduceOnline = cub::BlockReduce<OnlineSoftmax, BLOCK>;
    using BlockReduceMaxArg = cub::BlockReduce<MaxArg, BLOCK>;
    using BlockReduceCountMass = cub::BlockReduce<CountMass, BLOCK>;
    using BlockScanF = cub::BlockScan<float, BLOCK>;

    auto& temp = shared.temp;
    int* const sCount = shared.count;
    float* const sMass = shared.mass;
    float* const sCand = shared.cand;
    int& sCandCount = shared.candCount;
    int& sBucketCount = shared.bucketCount;
    int& sChosen = shared.chosen;
    long long& sCountHi = shared.countHi;
    float& sMassHi = shared.massHi;
    bool& sFired = shared.fired;
    float& sMaxScaled = shared.maxScaled;
    float& sTotalMass = shared.totalMass;
    float& sKeptMass = shared.keptMass;
    float& sSurvivingMass = shared.survivingMass;
    float& sTarget = shared.target;
    int& sArgMax = shared.argmax;
    int& sToken = shared.token;
    // Rejection-path state. Distinct names rather than reuse of the above: the descent
    // path's scalars each mean one thing for one stage, and an earlier version of this
    // kernel shipped a bug where the sampler's target clobbered the mass it was derived
    // from.
    float& sPivot = shared.pivot;
    float& sCandWeight = shared.candWeight;
    float& sMassTarget = shared.massTarget;
    int& sCandIdx = shared.candIdx;
    curandStatePhilox4_32_10_t& sRejectRng = shared.rejectRng;

    // --- Pass 1: max, total mass and argmax in one read (Milakov & Gimelshein).
    //
    // The tokens-only rejection path has two variants. When no earlier filter changes its
    // support, retain each thread's online-softmax partial and rebase it onto the block max;
    // the first inverse-CDF scan can consume that mass without reading the row again. A
    // min-p or top-k support cannot reuse the unfiltered partial, so those rows specialize
    // this pass down to max + argmax and compute their useful mass in the rejection scan.
    bool const deferMassToRejection
        = NEED_TOKENS && !NEED_PROBS && (!rp.needTopK || (rp.needTopP && params.numRows <= 8));
    bool const reuseThreadMass = deferMassToRejection && !rp.needTopK && !rp.needMinP;
    OnlineSoftmax cachedThreadStats{-FLT_MAX, 0.0f, 0};
    if constexpr (PRECOMPUTED_STATS)
    {
        static_assert(NEED_PROBS);
        if (tid == 0)
        {
            OnlineSoftmax const rowStats = *reinterpret_cast<OnlineSoftmax const*>(rowProbs + kMergedStatsOffset);
            sMaxScaled = rowStats.max;
            sArgMax = rowStats.argmax;
            sTotalMass = rowStats.sum;
        }
    }
    else if constexpr (NEED_TOKENS && !NEED_PROBS)
    {
        if (deferMassToRejection)
        {
            if (reuseThreadMass)
            {
                forEachLogit(rowLogits, vocabSize,
                    [&](int i, float logit) {
                        cachedThreadStats = combineOnlineSoftmax(
                            cachedThreadStats, OnlineSoftmax{__fmul_rn(logit, rp.tempInv), 1.0f, i});
                    });
                OnlineSoftmax const rowStats
                    = BlockReduceOnline(temp.reduceOnline).Reduce(cachedThreadStats, OnlineSoftmaxOp());
                if (tid == 0)
                {
                    sMaxScaled = rowStats.max;
                    sArgMax = rowStats.argmax;
                    sTotalMass = rowStats.sum;
                }
            }
            else
            {
                MaxArg local{-FLT_MAX, 0};
                forEachLogit(rowLogits, vocabSize,
                    [&](int i, float logit) {
                        local = MaxArgOp()(local, MaxArg{__fmul_rn(logit, rp.tempInv), i});
                    });
                MaxArg const rowStats = BlockReduceMaxArg(temp.reduceMaxArg).Reduce(local, MaxArgOp());
                if (tid == 0)
                {
                    sMaxScaled = rowStats.max;
                    sArgMax = rowStats.argmax;
                }
            }
        }
        else
        {
            OnlineSoftmax local{-FLT_MAX, 0.0f, 0};
            forEachLogit(rowLogits, vocabSize,
                [&](int i, float logit) {
                    local = combineOnlineSoftmax(local, OnlineSoftmax{__fmul_rn(logit, rp.tempInv), 1.0f, i});
                });
            OnlineSoftmax const rowStats = BlockReduceOnline(temp.reduceOnline).Reduce(local, OnlineSoftmaxOp());
            if (tid == 0)
            {
                sMaxScaled = rowStats.max;
                sArgMax = rowStats.argmax;
                sTotalMass = rowStats.sum;
            }
        }
    }
    else
    {
        OnlineSoftmax local{-FLT_MAX, 0.0f, 0};
        forEachLogit(rowLogits, vocabSize,
            [&](int i, float logit) {
                local = combineOnlineSoftmax(local, OnlineSoftmax{__fmul_rn(logit, rp.tempInv), 1.0f, i});
            });
        OnlineSoftmax const rowStats = BlockReduceOnline(temp.reduceOnline).Reduce(local, OnlineSoftmaxOp());
        if (tid == 0)
        {
            sMaxScaled = rowStats.max;
            sArgMax = rowStats.argmax;
            sTotalMass = rowStats.sum;
        }
    }
    __syncthreads();
    float const maxScaled = sMaxScaled;
    float const cachedLocalMass
        = reuseThreadMass ? cachedThreadStats.sum * __expf(cachedThreadStats.max - maxScaled) : 0.0f;

    // --- Pass 2: only min-p needs one, and only because its cutoff is relative to the max
    //     -- which pass 1 does not know until it ends, so the filtered mass cannot be
    //     accumulated there. Every other row already has its total.
    //
    //     min-p itself stays free: w == p / p_max is a value the softmax already produced,
    //     so the filter is one comparison, not a pass. This pass exists to re-total, not
    //     to filter.
    if (rp.needMinP && !deferMassToRejection)
    {
        float localMass = 0.0f;
        forEachLogit(rowLogits, vocabSize,
            [&](int, float logit)
            {
                float const w = __expf(__fmul_rn(logit, rp.tempInv) - maxScaled);
                if (w >= rp.minP)
                {
                    localMass += w;
                }
            });
        float const filteredMass = BlockReduceF(temp.reduceF).Sum(localMass);
        if (tid == 0)
        {
            sTotalMass = filteredMass;
        }
        __syncthreads();
    }

    // A small tokens-only top-k + top-p row must solve the rank cutoff before entering
    // rejection. Keep that work inside the hybrid branch; moving every top-k row here
    // perturbs the hot pure-top-k code layout measurably even though its algorithm is the
    // same.
    float threshold = rp.needMinP ? rp.minP : 0.0f;
    if (deferMassToRejection && rp.needTopK)
    {
        float const topKThreshold = findThreshold<T>(rowLogits, vocabSize, rp.tempInv, maxScaled, threshold,
            static_cast<long long>(rp.topK), 0.0f, /*byCount=*/true, sCount, sMass, sCand, &sCandCount, &sBucketCount,
            &sChosen, &sCountHi, &sMassHi, &sFired);
        threshold = fmaxf(threshold, topKThreshold);
    }

    // --- Tokens-only fast path: draw the token by rejection instead of solving for the
    //     cutoff that defines the kept set.
    //
    // The exact cutoff exists to *materialize* the filtered distribution. A token drawn
    // from that distribution does not need it: sample from any superset of the kept set,
    // test whether the draw landed inside, and on a rejection shrink the superset to what
    // the test just measured. Each round conditions on "inside the kept set", so what
    // survives is the kept distribution exactly -- the standard rejection argument, and it
    // stays valid across rounds because every support this loop builds still contains the
    // kept set.
    //
    // The descent dominates a filtered tokens call, so removing it outright is worth more
    // than tuning it.
    //
    // Pure top-k stays out of here because rejection has no bound on how much mass k
    // entries keep: on a flat 131k row, drawing one of the top 50 would almost never
    // succeed. Top-k + top-p is different. Pass 3a has already paid for the rank cutoff,
    // so rejection starts from the exact post-top-k support and only decides top-p. Its
    // acceptance rate is then the caller's top_p, normally close to one. This hybrid is
    // limited to at most eight rows: as the grid grows, the slowest CTA is increasingly
    // likely to need an extra rejection round, and the deterministic descent wins again.
    if constexpr (NEED_TOKENS && !NEED_PROBS)
    {
        if (deferMassToRejection)
        {
            // Both criteria are thresholds on what lies strictly above a candidate, and
            // both are taken over the support left by min-p and Pass 3a. The first
            // inverse-CDF scan below computes that support's mass. `<` for the count and
            // `<=` for the mass mirror findThreshold's firing rules (count *reaches* k,
            // mass *exceeds* its target), so this test and the descent accept exactly the
            // same set.
            float const supportFloor = threshold;
            float const countTarget = static_cast<float>(rp.topK);

            if (tid == 0)
            {
                int const rngIdx = params.perRowRng ? row : 0;
                uint64_t const seed = params.seed != nullptr ? params.seed[rngIdx] : 0ull;
                uint64_t const offset = params.offset != nullptr ? params.offset[rngIdx] : 0ull;
                curand_init(seed, static_cast<uint64_t>(row), offset, &sRejectRng);
                // -1 admits every weight: w is an exp, so it is never negative.
                sPivot = -1.0f;
                sToken = -1;
            }
            __syncthreads();

            for (int round = 0; round < kMaxRejectRounds; ++round)
            {
                float const pivot = sPivot;

                // Inverse-CDF draw over the current support. Round 0 can consume the
                // retained online-softmax partial; a narrowed retry recomputes its mass.
                // Either way the scan aggregates the same per-thread partials that the
                // sampling walk uses, so the target cannot fall outside that walk.
                float localMass = round == 0 && reuseThreadMass ? cachedLocalMass : 0.0f;
                if (!(round == 0 && reuseThreadMass))
                {
                    forEachLogit(rowLogits, vocabSize,
                        [&](int, float logit)
                        {
                            float const w = __expf(__fmul_rn(logit, rp.tempInv) - maxScaled);
                            if (w >= supportFloor && w > pivot)
                            {
                                localMass += w;
                            }
                        });
                }
                float base = 0.0f;
                float supportMass = 0.0f;
                BlockScanF(temp.scanF).ExclusiveSum(localMass, base, supportMass);
                __syncthreads();
                if (tid == 0)
                {
                    if (round == 0)
                    {
                        sMassTarget = rp.topP * supportMass;
                    }
                    sTarget = curand_uniform(&sRejectRng) * supportMass;
                    sCandIdx = -1;
                }
                __syncthreads();

                float const target = sTarget;
                if (target >= base && target < base + localMass)
                {
                    // Same traversal as the accumulation above, so the two float sums
                    // agree term for term and the crossing lands on the same element.
                    float running = base;
                    bool found = false;
                    forEachLogit(rowLogits, vocabSize,
                        [&](int i, float logit)
                        {
                            if (found)
                            {
                                return;
                            }
                            float const w = __expf(__fmul_rn(logit, rp.tempInv) - maxScaled);
                            if (!(w >= supportFloor && w > pivot))
                            {
                                return;
                            }
                            running += w;
                            if (running > target)
                            {
                                sCandIdx = i;
                                sCandWeight = w;
                                found = true;
                            }
                        });
                }
                __syncthreads();

                if (sCandIdx < 0)
                {
                    // Rounding left the target past the last element of the support.
                    // The argmax below is the answer, as on the descent path.
                    break;
                }
                float const candWeight = sCandWeight;

                // Nothing can reject: all active filters are already built into the
                // support, so the first draw is the answer. Skipping the sweep is what
                // keeps a neutral or min-p-only row at the two passes it had before this
                // path existed.
                if (!rp.needTopP)
                {
                    if (tid == 0)
                    {
                        sToken = sCandIdx;
                    }
                    __syncthreads();
                    break;
                }

                // One sweep decides membership for both criteria.
                CountMass local{0.0f, 0.0f};
                forEachLogit(rowLogits, vocabSize,
                    [&](int, float logit)
                    {
                        float const w = __expf(__fmul_rn(logit, rp.tempInv) - maxScaled);
                        if (w >= supportFloor && w > candWeight)
                        {
                            local.count += 1.0f;
                            local.mass += w;
                        }
                    });
                CountMass const above = BlockReduceCountMass(temp.reduceCountMass).Reduce(local, CountMassOp());
                __syncthreads();

                if (tid == 0)
                {
                    // Pass 3a's support floor already implies the top-k arm. Keeping the
                    // explicit predicate documents and checks that invariant at the same
                    // boundary where top-p is decided.
                    bool const keep
                        = (!rp.needTopK || above.count < countTarget) && (!rp.needTopP || above.mass <= sMassTarget);
                    if (keep)
                    {
                        sToken = sCandIdx;
                    }
                    else
                    {
                        // The kept set is downward-closed in w, so a rejected candidate
                        // rules out everything at or below its weight -- and the mass of
                        // what remains is the mass just measured. Narrowing is free, and
                        // it always drops at least the candidate, so the loop terminates.
                        sPivot = candWeight;
                    }
                }
                __syncthreads();

                if (sToken >= 0)
                {
                    break;
                }
            }

            if (tid == 0)
            {
                params.outputTokens[row] = sToken >= 0 ? sToken : sArgMax;
            }
            return;
        }
    }

    // --- Pass 3: rank/mass thresholds for rows which did not return through rejection.
    if (rp.needTopK)
    {
        float const topKThreshold = findThreshold<T>(rowLogits, vocabSize, rp.tempInv, maxScaled, threshold,
            static_cast<long long>(rp.topK), 0.0f, /*byCount=*/true, sCount, sMass, sCand, &sCandCount, &sBucketCount,
            &sChosen, &sCountHi, &sMassHi, &sFired);
        threshold = fmaxf(threshold, topKThreshold);
    }

    // Output shapes which must materialize the kept distribution still need the exact
    // top-p cutoff. Tokens-only rows carrying top-p either returned above or deliberately
    // took this deterministic fallback because the batch was too wide for rejection.
    //
    // Order matters, and only min-p and top-k are order-free. Both of those are invariant
    // under renormalization -- min-p thresholds p / p_max, top-k thresholds rank, and
    // scaling a row by a constant changes neither -- so they compose into one cutoff.
    // top-p does NOT: its cutoff is a fraction of the mass of whatever survived before it,
    // and every earlier filter shrinks that denominator. Running it against the raw
    // softmax mass keeps far too much (measurably: ~0.2 of L1 mass against the
    // TorchSampler reference at top_k=50, top_p=0.9). Hence a second descent, against the
    // surviving mass, in the documented min-p -> top-k -> top-p order.
    if (rp.needTopP)
    {
        // The mass top-p takes its fraction of: post-min-p, post-top-k.
        //
        // When neither of those ran the threshold is still 0, nothing has been removed,
        // and that mass is the total pass 1 already produced -- so the sweep is pure
        // waste, which is exactly what a top-p-only row would pay.
        if (threshold > 0.0f)
        {
            float localSurviving = 0.0f;
            forEachLogit(rowLogits, vocabSize,
                [&](int, float logit)
                {
                    float const w = __expf(__fmul_rn(logit, rp.tempInv) - maxScaled);
                    if (w >= threshold)
                    {
                        localSurviving += w;
                    }
                });
            float const survivingMass = BlockReduceF(temp.reduceF).Sum(localSurviving);
            if (tid == 0)
            {
                sSurvivingMass = survivingMass;
            }
        }
        else if (tid == 0)
        {
            sSurvivingMass = sTotalMass;
        }
        __syncthreads();

        float const topPThreshold = findThreshold<T>(rowLogits, vocabSize, rp.tempInv, maxScaled, threshold, 0,
            rp.topP * sSurvivingMass, /*byCount=*/false, sCount, sMass, sCand, &sCandCount, &sBucketCount, &sChosen,
            &sCountHi, &sMassHi, &sFired);
        threshold = fmaxf(threshold, topPThreshold);
    }

    // --- Pass 4: kept mass. Skipped when no filter can have removed anything, in which
    //     case pass 2's total already is the kept mass.
    //
    //     The per-thread partial is kept in a register rather than discarded: the sampling
    //     pass below needs exactly this sum, over exactly this traversal, to seed its
    //     scan. Recomputing it there cost a second full sweep of the row on every
    //     filtered tokens call -- which is precisely the case that was missing its gate.
    float keptMass = sTotalMass;
    float localKept = 0.0f;
    bool haveLocalKept = false;
    if (threshold > 0.0f)
    {
        forEachLogit(rowLogits, vocabSize,
            [&](int, float logit)
            {
                float const w = __expf(__fmul_rn(logit, rp.tempInv) - maxScaled);
                if (w >= threshold)
                {
                    localKept += w;
                }
            });
        haveLocalKept = true;
        float const blockKept = BlockReduceF(temp.reduceF).Sum(localKept);
        if (tid == 0)
        {
            sKeptMass = blockKept;
        }
        __syncthreads();
        keptMass = sKeptMass;
    }
    // A degenerate row (every weight filtered away by a pathological threshold) would
    // divide by zero; fall back to the unfiltered mass rather than emit NaNs.
    if (!(keptMass > 0.0f))
    {
        keptMass = sTotalMass;
        threshold = 0.0f;
        // The cached partial was summed against the threshold just abandoned, so it no
        // longer describes the set the sampler is about to walk.
        localKept = 0.0f;
        haveLocalKept = false;
    }

    // --- Pass 5: the renormalized distribution.
    if (NEED_PROBS)
    {
        // Read vectorized, and write vectorized too: probs is float32 and as wide as the
        // logits, so the store side is just as much traffic as the load side.
        float const scale = 1.0f / keptMass;
        int const probVecCount = vocabSize / 4;
        bool const probsAligned = (reinterpret_cast<uintptr_t>(rowProbs) % kVecBytes) == 0;
        // A neutral tokens+probs row otherwise evaluates every exp a third time in the
        // sampling scan below. For aligned float logits this loop visits exactly the same
        // indices, in exactly the same per-thread order, as forEachLogit, so its partial
        // sum can seed that scan without changing a rounding decision. Half and BF16
        // vectorize eight logits per thread rather than four and deliberately retain the
        // old path; so do uncommon unaligned rows.
        bool const canReuseOutputMass = NEED_TOKENS && !haveLocalKept && sizeof(T) == sizeof(float) && probsAligned
            && (reinterpret_cast<uintptr_t>(rowLogits) % kVecBytes) == 0;
        float outputLocalMass = 0.0f;
        if (probsAligned)
        {
            for (int v = tid; v < probVecCount; v += blockDim.x)
            {
                float4 out;
                float* outElems = reinterpret_cast<float*>(&out);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    float const w = weightOf(rowLogits, v * 4 + j, rp.tempInv, maxScaled);
                    outElems[j] = w >= threshold ? w * scale : 0.0f;
                    if (canReuseOutputMass && w >= threshold)
                    {
                        outputLocalMass += w;
                    }
                }
                reinterpret_cast<float4*>(rowProbs)[v] = out;
            }
            for (int i = probVecCount * 4 + tid; i < vocabSize; i += blockDim.x)
            {
                float const w = weightOf(rowLogits, i, rp.tempInv, maxScaled);
                rowProbs[i] = w >= threshold ? w * scale : 0.0f;
                if (canReuseOutputMass && w >= threshold)
                {
                    outputLocalMass += w;
                }
            }
        }
        else
        {
            for (int i = tid; i < vocabSize; i += blockDim.x)
            {
                float const w = weightOf(rowLogits, i, rp.tempInv, maxScaled);
                rowProbs[i] = w >= threshold ? w * scale : 0.0f;
            }
        }
        if (canReuseOutputMass)
        {
            localKept = outputLocalMass;
            haveLocalKept = true;
        }
    }

    // --- Pass 6: inverse-CDF sample over the kept weights.
    if (NEED_TOKENS)
    {
        if (tid == 0)
        {
            int const rngIdx = params.perRowRng ? row : 0;
            uint64_t const seed = params.seed != nullptr ? params.seed[rngIdx] : 0ull;
            uint64_t const offset = params.offset != nullptr ? params.offset[rngIdx] : 0ull;
            curandStatePhilox4_32_10_t state;
            // The row index is the subsequence, so rows draw independent streams from a
            // shared seed -- and a seeded request stays reproducible via its own offset.
            curand_init(seed, static_cast<uint64_t>(row), offset, &state);
            sTarget = curand_uniform(&state) * keptMass;
            sToken = -1;
        }
        __syncthreads();
        float const target = sTarget;

        // Each thread sums its own strided elements, an exclusive scan gives it the mass
        // below it, and then only the thread whose span contains the target re-walks --
        // in the same order, so the two accumulations agree.
        // Already summed by pass 4 whenever a filter ran; only an unfiltered row still
        // owes this sweep.
        if (!haveLocalKept)
        {
            forEachLogit(rowLogits, vocabSize,
                [&](int, float logit)
                {
                    float const w = __expf(__fmul_rn(logit, rp.tempInv) - maxScaled);
                    if (w >= threshold)
                    {
                        localKept += w;
                    }
                });
        }
        float base = 0.0f;
        BlockScanF(temp.scanF).ExclusiveSum(localKept, base);
        __syncthreads();

        if (target >= base && target < base + localKept)
        {
            // Same traversal as the accumulation above -- forEachLogit, not a raw strided
            // loop -- because the two float sums must agree term for term. A different
            // order here would put the crossing at a different element.
            float running = base;
            bool found = false;
            forEachLogit(rowLogits, vocabSize,
                [&](int i, float logit)
                {
                    if (found)
                    {
                        return;
                    }
                    float const w = __expf(__fmul_rn(logit, rp.tempInv) - maxScaled);
                    if (w < threshold)
                    {
                        return;
                    }
                    running += w;
                    if (running > target)
                    {
                        sToken = i;
                        found = true;
                    }
                });
        }
        __syncthreads();

        if (tid == 0)
        {
            // Rounding can leave the target just past the last kept element; the argmax
            // is always in the kept set, so it is a safe answer rather than a wrong one.
            params.outputTokens[row] = sToken >= 0 ? sToken : sArgMax;
        }
    }
}

} // namespace fusedSampling
} // namespace kernels
} // namespace tensorrt_llm
