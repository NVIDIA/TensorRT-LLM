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

// This file is compiled two ways and must stay identical under both: by CMake as part of
// the wheel, and by torch.utils.cpp_extension during development. So it depends only on
// CUDA, CUB and curand -- no TensorRT-LLM headers beyond its own.

#include "universalSamplingKernels.h"

#include <cfloat>
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>

namespace tensorrt_llm
{
namespace kernels
{

namespace
{

//! One block owns one row, so the grid is only as wide as the batch, and the best block
//! size depends on which of the two regimes that puts us in:
//!
//! * **Few rows.** At 64 rows on an H200's 132 SMs every block already has an SM to
//!   itself; what goes unused is the warp slots inside it. A wide block shortens the row's
//!   critical path. Measured: neutral probs 1.32x -> 0.86x, top_p probs 1.35x -> 0.96x.
//! * **Many rows.** Blocks now queue for SMs, and a wide block just makes each one longer
//!   while fitting fewer per SM. Measured going the other way: top_k_top_p tokens
//!   1.37x -> 1.82x at 256 rows.
//!
//! So it is chosen per launch rather than fixed. The crossover is between 64 and 256 rows
//! in the sweep; 128 is the midpoint and neither regime is sharp there.
constexpr int kNarrowBlock = 512;
constexpr int kWideBlock = 1024;
constexpr int kWideBlockMaxRows = 128;
//! Digits per radix pass of the threshold search. 8 bits -> 256 buckets, 4 passes to pin
//! a float exactly.
constexpr int kRadixBits = 8;
constexpr int kRadixBuckets = 1 << kRadixBits;
constexpr int kRadixPasses = 32 / kRadixBits;
//! Private histogram copies, one per group of warps, to cut shared-atomic contention.
//! 8 copies is 16 KB of shared memory -- affordable, and enough to spread the few buckets
//! a softmax actually lands in across separate addresses.
constexpr int kHistCopies = 8;
//! How many survivors the shared gather buffer holds (16 KB). Past this the descent falls
//! back to re-reading the row, so the cap is a performance knob, never a correctness one.
constexpr int kCandCap = 4096;
//! Guards 1/T for a zero temperature. Matches decodingCommon.cu's EPSILON for float.
constexpr float kTempEpsilon = 1e-6f;

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
//! sweep reads memory multiplies through all of them. Scalar 4-byte loads leave most of
//! the achievable bandwidth on the table; measurement put a two-pass neutral row at ~1.5x
//! of flashinfer's two-pass softmax, which is a per-pass efficiency gap, not a pass-count
//! one.
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

__device__ inline RowParams loadRowParams(UniversalSamplingParams const& p, int row)
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
    int* sBucketCount, int* sChosen, long long* sCountHi, float* sMassHi, bool* sFired)
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
                atomicAdd(&myMass[digit], w);
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
                    atomicAdd(&myMass[digit], w);
                });
        }
        __syncthreads();

        // Fold the private copies into copy 0, one bucket per thread. 256 buckets against
        // a vocabSize-long histogram pass -- not what this costs.
        for (int b = tid; b < kRadixBuckets; b += blockDim.x)
        {
            int totalCount = sCount[b];
            float totalMass = sMass[b];
#pragma unroll
            for (int c = 1; c < kHistCopies; ++c)
            {
                totalCount += sCount[c * kRadixBuckets + b];
                totalMass += sMass[c * kRadixBuckets + b];
            }
            sCount[b] = totalCount;
            sMass[b] = totalMass;
        }
        __syncthreads();

        // The bucket walk is serial on one thread: 256 steps against vocabSize/512
        // steps of the histogram pass above, so it is not what this costs.
        if (tid == 0)
        {
            long long c = countHi;
            float m = massHi;
            int chosen = 0;
            for (int b = kRadixBuckets - 1; b >= 0; --b)
            {
                long long const c2 = c + sCount[b];
                float const m2 = m + sMass[b];
                bool const fired = byCount ? c2 >= targetCount : m2 > targetMass;
                if (fired)
                {
                    chosen = b;
                    *sFired = true;
                    break;
                }
                c = c2;
                m = m2;
            }
            *sChosen = chosen;
            *sCountHi = c;
            *sMassHi = m;
            // How many survivors the next pass has to look at. Captured now, before the
            // next iteration zeroes the histogram.
            *sBucketCount = sCount[chosen];
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

//! \brief Fused temperature + min-p + top-k + top-p + (probs | sampling), one block per row.
template <typename T, int BLOCK, bool NEED_TOKENS, bool NEED_PROBS>
__global__ void universalSamplingKernel(UniversalSamplingParams params)
{
    int const row = blockIdx.x;
    int const tid = threadIdx.x;
    int const vocabSize = params.vocabSize;
    T const* rowLogits = static_cast<T const*>(params.logits) + static_cast<size_t>(row) * vocabSize;
    float* rowProbs = NEED_PROBS ? params.outputProbs + static_cast<size_t>(row) * vocabSize : nullptr;

    RowParams const rp = loadRowParams(params, row);

    using BlockReduceF = cub::BlockReduce<float, BLOCK>;
    using BlockReduceOnline = cub::BlockReduce<OnlineSoftmax, BLOCK>;
    using BlockScanF = cub::BlockScan<float, BLOCK>;

    __shared__ union
    {
        typename BlockReduceF::TempStorage reduceF;
        typename BlockReduceOnline::TempStorage reduceOnline;
        typename BlockScanF::TempStorage scanF;
    } temp;

    __shared__ int sCount[kRadixBuckets * kHistCopies];
    __shared__ float sMass[kRadixBuckets * kHistCopies];
    __shared__ float sCand[kCandCap];
    __shared__ int sCandCount;
    __shared__ int sBucketCount;
    __shared__ int sChosen;
    __shared__ long long sCountHi;
    __shared__ float sMassHi;
    __shared__ bool sFired;
    __shared__ float sMaxScaled;
    __shared__ float sTotalMass;
    __shared__ float sKeptMass;
    __shared__ float sSurvivingMass;
    __shared__ float sTarget;
    __shared__ int sArgMax;
    __shared__ int sToken;

    // --- Pass 1: online softmax -- max, total mass and argmax in a SINGLE read of the
    //     row (Milakov & Gimelshein). The naive form needs two, one for the max and one
    //     for the sum, and measurement showed that second pass is exactly what put a
    //     neutral `probs` row at ~1.5x of flashinfer's two-pass softmax. The argmax rides
    //     along for free and is the fallback if the inverse-CDF walk falls off the end.
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
    __syncthreads();
    float const maxScaled = sMaxScaled;

    // --- Pass 2: only min-p needs one, and only because its cutoff is relative to the max
    //     -- which pass 1 does not know until it ends, so the filtered mass cannot be
    //     accumulated there. Every other row already has its total.
    //
    //     min-p itself stays free: w == p / p_max is a value the softmax already produced,
    //     so the filter is one comparison, not a pass. This pass exists to re-total, not
    //     to filter.
    if (rp.needMinP)
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

    // --- Pass 3: the rank/mass thresholds, skipped wholesale when neither top-k nor
    //     top-p is active. This is the only expensive stage, and the only one a neutral
    //     row -- or a neutral row in a mixed batch -- does not run at all.
    //
    // Order matters, and only min-p and top-k are order-free. Both of those are invariant
    // under renormalization -- min-p thresholds p / p_max, top-k thresholds rank, and
    // scaling a row by a constant changes neither -- so they compose into one cutoff.
    // top-p does NOT: its cutoff is a fraction of the mass of whatever survived before it,
    // and every earlier filter shrinks that denominator. Running it against the raw
    // softmax mass keeps far too much (measurably: ~0.2 of L1 mass against the
    // TorchSampler reference at top_k=50, top_p=0.9). Hence a second descent, against the
    // surviving mass, in the documented min-p -> top-k -> top-p order.
    float threshold = rp.needMinP ? rp.minP : 0.0f;
    if (rp.needTopK)
    {
        float const topKThreshold = findThreshold<T>(rowLogits, vocabSize, rp.tempInv, maxScaled, threshold,
            static_cast<long long>(rp.topK), 0.0f, /*byCount=*/true, sCount, sMass, sCand, &sCandCount, &sBucketCount,
            &sChosen, &sCountHi, &sMassHi, &sFired);
        threshold = fmaxf(threshold, topKThreshold);
    }
    if (rp.needTopP)
    {
        // The mass top-p takes its fraction of: post-min-p, post-top-k.
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
                }
                reinterpret_cast<float4*>(rowProbs)[v] = out;
            }
            for (int i = probVecCount * 4 + tid; i < vocabSize; i += blockDim.x)
            {
                float const w = weightOf(rowLogits, i, rp.tempInv, maxScaled);
                rowProbs[i] = w >= threshold ? w * scale : 0.0f;
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

} // namespace

//! Launch one (block size, output shape) specialization. Both are compile-time so the
//! kernel is emitted without the half it does not need and with cub sized correctly.
template <typename T, int BLOCK>
void launchUniversalSampling(UniversalSamplingParams const& params, cudaStream_t stream)
{
    dim3 const grid(params.numRows);
    dim3 const block(BLOCK);
    bool const needTokens = params.outputTokens != nullptr;
    bool const needProbs = params.outputProbs != nullptr;

    if (needTokens && needProbs)
    {
        universalSamplingKernel<T, BLOCK, true, true><<<grid, block, 0, stream>>>(params);
    }
    else if (needTokens)
    {
        universalSamplingKernel<T, BLOCK, true, false><<<grid, block, 0, stream>>>(params);
    }
    else if (needProbs)
    {
        universalSamplingKernel<T, BLOCK, false, true><<<grid, block, 0, stream>>>(params);
    }
}

template <typename T>
void invokeUniversalSampling(UniversalSamplingParams const& params, cudaStream_t stream)
{
    // See kNarrowBlock / kWideBlock: a small batch wants a wide block to shorten each
    // row's critical path, a large one wants narrow blocks so more fit per SM.
    if (params.numRows <= kWideBlockMaxRows)
    {
        launchUniversalSampling<T, kWideBlock>(params, stream);
    }
    else
    {
        launchUniversalSampling<T, kNarrowBlock>(params, stream);
    }
}

template void invokeUniversalSampling<float>(UniversalSamplingParams const&, cudaStream_t);
template void invokeUniversalSampling<__half>(UniversalSamplingParams const&, cudaStream_t);
template void invokeUniversalSampling<__nv_bfloat16>(UniversalSamplingParams const&, cudaStream_t);

} // namespace kernels
} // namespace tensorrt_llm
