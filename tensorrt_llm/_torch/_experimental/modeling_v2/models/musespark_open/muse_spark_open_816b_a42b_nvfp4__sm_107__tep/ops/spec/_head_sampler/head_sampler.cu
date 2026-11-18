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

// The MuseSparkOpen target head's token sampler, split over two kernels, and the
// head's vocabulary gather folded into the first of them (ops/spec/head_sampler.py).
//
// The built-in head rounds the LM-head GEMM to BF16, all-gathers the eight vocabulary
// shards, concatenates them, widens them to FP32, applies the soft cap in four
// elementwise kernels (x * m, * fp32(1/cap), tanh, cap * t) and samples each row with
// fusedSamplingKernel<float, 1024, true, false> (one CTA of 1024 threads per row).
// For a row with no top-k / top-p / min-p filter, that kernel folds thread t's
// elements 4 (t + 1024 j) + {0, 1, 2, 3} into one online-softmax state in that order,
// block-reduces the 1024 states, rebases each onto the row max, scans them, draws a
// Philox uniform and lets the thread whose span holds the target walk its elements
// again until the running sum passes it.
//
// Here kernel A (splitFoldKernel) runs the same 1024 per-thread folds, with 128 of them
// per CTA and 8 CTAs per row, reading BF16 logits and applying the soft cap in
// registers with the torch chain's rounding statements; kernel B (splitDrawKernel), one
// 1024-thread CTA per row, runs the shipped block reduce, scan, Philox draw and walk on
// those states. Each virtual thread folds the same elements in the same order with the
// same operations, so every state, every reduction and every token is the built-in's.
// A row that enables a filter runs the shipped kernel body inside kernel B on the BF16
// logits, reading them through CapBf16 (the soft cap applied on every read).
//
// Above kWideBlockMaxRows (128) rows the shipped sampler runs its 512-thread block
// (fusedSamplingTokensNarrowKernel: thread t folds the elements 4 (t + 512 j) + {0..3}).
// Both kernels are templated on that virtual block (VB): a call of more than 128 rows
// runs 512 virtual threads per row (4 CTAs of kernel A per row, a 512-thread kernel B),
// statement for statement the shipped narrow block's first round; a call of at most 128
// rows runs the 1024-thread instances, unchanged.
//
// With the gather, each rank's LM-head GEMM writes its [rows, shard] BF16 output into
// its slot of a multicast buffer whose every word holds a sentinel (0xFFFF in both
// BF16 halves, a NaN no GEMM epilogue produces) between calls. Kernel A polls this
// rank's copy of the buffer, word by word, copies each published vector into the
// [rows, vocab] BF16 row that kernel B and the step's logits output read, and resets
// the word to the sentinel. No -0 canonicalization: the gathered row is the GEMM's
// bits.

#include "draft_path_sample.cuh"
#include "fusedSamplingKernelsCommon.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <climits>
#include <cstdint>
#include <optional>
#include <tuple>

namespace tensorrt_llm
{
namespace kernels
{
namespace fusedSampling
{

//! The soft cap's three FP32 constants: multiplier m, 1/cap (rounded once to FP32, as
//! ATen's division by a scalar multiplies by it) and cap.
struct SplitCap
{
    float multiplier;
    float invCap;
    float cap;
};

//! The torch chain logits.float() * m, / cap, tanh, cap * t, statement for statement.
__device__ __forceinline__ float capOf(float x, SplitCap const& c)
{
    float y = __fmul_rn(x, c.multiplier);
    y = __fmul_rn(y, c.invCap);
    y = tanhf(y);
    return __fmul_rn(c.cap, y);
}

//! The constants for reads through CapBf16 inside the shipped body: kernel B stores its
//! argument here before anything reads a logit.
__shared__ SplitCap sSplitCap;

//! A BF16 LM-head logit read as the soft-capped FP32 logit.
struct CapBf16
{
    __nv_bfloat16 v;
};

__device__ inline float toFloat(CapBf16 v)
{
    return capOf(__bfloat162float(v.v), sSplitCap);
}

//! forEachLogit<float>'s traversal of a row of soft-capped logits held as BF16: vector v
//! (4 elements) on thread v % blockDim.x, in increasing v, read as 8 bytes instead of
//! 16. The FP32 rows the built-in samples are 16-byte aligned, so the shipped sweep takes
//! its vector path; this one always does.
template <typename Fn>
__device__ inline void forEachLogit(CapBf16 const* row, int vocabSize, Fn fn)
{
    int const vecCount = vocabSize / 4;
    uint2 const* vrow = reinterpret_cast<uint2 const*>(row);
    for (int v = threadIdx.x; v < vecCount; v += blockDim.x)
    {
        uint2 const packed = vrow[v];
        __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(&packed);
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            fn(v * 4 + j, capOf(__bfloat162float(elems[j]), sSplitCap));
        }
    }
    for (int i = vecCount * 4 + threadIdx.x; i < vocabSize; i += blockDim.x)
    {
        fn(i, toFloat(row[i]));
    }
}

//! The shipped single-CTA block for at most kWideBlockMaxRows rows; kernel A spreads
//! its threads, kernel B is one of it per row.
constexpr int kSplitBlock = kWideBlock;
//! The shipped single-CTA block above kWideBlockMaxRows rows.
constexpr int kSplitNarrowBlock = kNarrowBlock;
//! Most rows one call takes (the states scratch and the grids are sized by the caller).
constexpr int kSplitMaxRows = 1024;
//! Loads kernel A issues before folding them, per virtual thread.
constexpr int kSplitPrefetch = 16;
//! Vectors of one virtual thread kernel B's walk stages in shared memory (1024-thread block).
constexpr int kSplitWalkVectors = 64;
//! The same for the 512-thread block: twice the vectors per thread, the same vocabulary bound.
constexpr int kSplitNarrowWalkVectors = 128;

//! The virtual block of a call of `numRows` rows: the shipped sampler's block for that call.
__host__ __device__ constexpr int splitVirtualBlock(int numRows)
{
    return numRows <= kWideBlockMaxRows ? kSplitBlock : kSplitNarrowBlock;
}

//! Two BF16 halves of 0xFFFF: the value of every unpublished word of the inbox.
constexpr uint32_t kSplitSentinel = 0xFFFFFFFFu;
//! Status bit: a poll waited past the timeout (the kernel then traps).
constexpr int kSplitPollTimeout = 1;

struct SplitFoldParams
{
    //! Row params (temperature), as FusedSamplingParams; logits unused with the gather.
    FusedSamplingParams fs;
    //! Without the gather: the BF16 [numRows, vocabSize] logits.
    uint2 const* plain{nullptr};
    //! With the gather: this rank's copy of the multicast inbox, [ranks][slotRows][shardVecs].
    uint2* inbox{nullptr};
    //! With the gather: the gathered BF16 [numRows, vocabSize] row, written here.
    uint2* gathered{nullptr};
    //! Per-virtual-thread online-softmax states, [numRows][splitVirtualBlock(numRows)].
    OnlineSoftmax* states{nullptr};
    int* status{nullptr};
    SplitCap cap{};
    int shardVecs{0};
    int slotRows{0};
    unsigned long long timeoutNs{0};
};

__device__ __forceinline__ void gridDependencyWait()
{
    asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void gridDependencyTrigger()
{
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__device__ __forceinline__ uint2 loadRelaxedSys(uint2 const* address)
{
    uint2 value;
    asm volatile("ld.relaxed.sys.global.v2.b32 {%0, %1}, [%2];"
                 : "=r"(value.x), "=r"(value.y)
                 : "l"(address)
                 : "memory");
    return value;
}

__device__ __forceinline__ void storeRelaxedSys(uint2* address, uint2 value)
{
    asm volatile("st.relaxed.sys.global.v2.b32 [%0], {%1, %2};" ::"l"(address), "r"(value.x), "r"(value.y) : "memory");
}

__device__ __forceinline__ unsigned long long globalTimerNs()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

//! Whether every BF16 half of the 8-byte word has been published (none is the sentinel).
__device__ __forceinline__ bool published(uint2 word)
{
    return (__vcmpeq2(word.x, kSplitSentinel) | __vcmpeq2(word.y, kSplitSentinel)) == 0u;
}

//! This rank's inbox word of (row, vector v) of the full vocabulary.
__device__ __forceinline__ uint2* inboxAt(SplitFoldParams const& p, int row, int v)
{
    int const rank = v / p.shardVecs;
    return p.inbox + (static_cast<size_t>(rank) * p.slotRows + row) * p.shardVecs + (v - rank * p.shardVecs);
}

//! Spin until the word at `address` is published. A wait past the timeout is a protocol
//! failure (a peer that never published): record it and trap rather than return data.
__device__ __noinline__ uint2 awaitPublished(uint2* address, int* status, unsigned long long timeoutNs)
{
    unsigned long long const start = globalTimerNs();
    for (int spins = 0;; ++spins)
    {
        uint2 const word = loadRelaxedSys(address);
        if (published(word))
        {
            return word;
        }
        if (spins >= 64)
        {
            __nanosleep(64);
            if (globalTimerNs() - start > timeoutNs)
            {
                atomicOr(status, kSplitPollTimeout);
                __threadfence_system();
                __trap();
            }
        }
    }
}

//! Kernel A: (optionally) gather, then (optionally) fold. Grid (numRows, VB / THREADS);
//! block THREADS; virtual thread vt = blockIdx.y * THREADS + threadIdx.x of the shipped
//! VB-thread block (VB = splitVirtualBlock(numRows)).
//!
//! PDL: waits on its predecessor before its first read of anything, so every input,
//! the inbox included, is read after the predecessor grid (the LM-head GEMM) completed.
//! Peer slots are published by the peers' GEMMs and are read through the Lamport
//! protocol, not through that edge.
template <bool GATHER, bool FOLD, int THREADS, int VB>
__global__ void __launch_bounds__(THREADS) splitFoldKernel(SplitFoldParams p)
{
    gridDependencyWait();
    int const row = blockIdx.x;
    int const vt = blockIdx.y * THREADS + threadIdx.x;
    int const vecCount = p.fs.vocabSize / 4;
    float tempInv = 1.0f;
    if constexpr (FOLD)
    {
        tempInv = loadRowParams(p.fs, row).tempInv;
    }
    OnlineSoftmax st{-FLT_MAX, 0.0f, 0};
    uint2 const* plainRow = GATHER ? nullptr : p.plain + static_cast<size_t>(row) * vecCount;
    uint2* gatheredRow = GATHER ? p.gathered + static_cast<size_t>(row) * vecCount : nullptr;
    for (int v0 = vt; v0 < vecCount; v0 += kSplitPrefetch * VB)
    {
        uint2 buf[kSplitPrefetch];
#pragma unroll
        for (int u = 0; u < kSplitPrefetch; ++u)
        {
            int const v = v0 + u * VB;
            if (v < vecCount)
            {
                if constexpr (GATHER)
                {
                    buf[u] = loadRelaxedSys(inboxAt(p, row, v));
                }
                else
                {
                    buf[u] = __ldg(plainRow + v);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < kSplitPrefetch; ++u)
        {
            int const v = v0 + u * VB;
            if (v < vecCount)
            {
                if constexpr (GATHER)
                {
                    uint2* const slot = inboxAt(p, row, v);
                    if (!published(buf[u]))
                    {
                        buf[u] = awaitPublished(slot, p.status, p.timeoutNs);
                    }
                    gatheredRow[v] = buf[u];
                    storeRelaxedSys(slot, make_uint2(kSplitSentinel, kSplitSentinel));
                }
                if constexpr (FOLD)
                {
                    __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(&buf[u]);
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                    {
                        float const logit = capOf(__bfloat162float(elems[j]), p.cap);
                        st = combineOnlineSoftmax(st, OnlineSoftmax{__fmul_rn(logit, tempInv), 1.0f, v * 4 + j});
                    }
                }
            }
        }
    }
    if constexpr (FOLD)
    {
        p.states[static_cast<size_t>(row) * VB + vt] = st;
    }
    gridDependencyTrigger();
}

// ---- The multi-CTA top-p test (head_sampler_coop_fold TOPP_ROUNDS) -------------------------
// A row with top-p and neither top-k nor min-p takes the shipped body's tokens-only rejection
// path (deferMassToRejection, reuseThreadMass): round 0 draws a candidate from kernel A's
// states (kernel B's unfiltered first round, statement for statement), then every round sweeps
// the whole row once for the CountMass of the weights strictly above the candidate's
// (accepted iff that mass <= top_p x the round-0 support mass) and, on a rejection, once more
// for the next round's per-thread support mass, all inside one 1024-thread CTA per row. Here
// round 0 stops after its draw and leaves a record; the test's sweep runs over VB / 128 CTAs
// per row (splitTopPCountKernel: each virtual thread's CountMass, its elements in the shipped
// order with the shipped statements), and splitTopPDecideKernel reduces the VB partials with
// the shipped cub::BlockReduce. A rejected round's next support mass per thread is that same
// partial's mass (the shipped next-round sweep sums the same elements -- w >= 0 && w > the
// rejected candidate's weight -- in the same order), so the next draw needs no sweep. The last
// decide kernel finishes its rows in the CTA (one sweep per further round).

//! Bytes of one row's record slot (``head_sampler.TOPP_RECORD_WORDS`` x 4).
constexpr int kTopPRecordBytes = 128;

//! A pending top-p row between the kernels of the test (one 128-byte slot per row).
struct alignas(kTopPRecordBytes) SplitTopPRecord
{
    //! The row's Philox state after the round's draw.
    curandStatePhilox4_32_10_t rng;
    float maxScaled;
    //! The round's candidate and its weight w.
    float candWeight;
    //! top_p x the round-0 support mass.
    float massTarget;
    int cand;
    int argmax;
    int round;
    //! 1 while the row's token is undecided; every kernel B row writes it.
    int pending;
};

static_assert(sizeof(SplitTopPRecord) == kTopPRecordBytes, "one record slot per row");
//! Threads per CTA of the test's sweep (VB / kTopPCountThreads CTAs per row).
constexpr int kTopPCountThreads = 128;
//! Most test rounds one call spreads over CTAs (the last decide kernel finishes in-CTA).
constexpr int kTopPMaxRounds = 4;

//! Whether a row takes the multi-CTA test: top-p, and no filter before it.
__device__ __forceinline__ bool splitTopPOnly(RowParams const& rp)
{
    return rp.needTopP && !rp.needTopK && !rp.needMinP;
}

//! Relative margin of the sweep's skip test (1 + 2^-10).
constexpr float kTopPSkipMargin = 1.0009765625f;

//! Whether virtual thread vt's sweep provably adds nothing (its CountMass stays {0, 0}, the
//! shipped thread's value bit for bit), from its kernel A state `st`: no element of vt has
//! w > candWeight. st.max is the exact maximum of vt's scaled logits x (the fold keeps the
//! larger max; a NaN element makes st.sum NaN, which disables the skip), x - maxScaled is
//! monotone in x, and __expf (ex2.approx of a monotone product) errs by far less than the
//! margin, so every element's w is at most __expf(st.max - maxScaled) x the margin.
__device__ __forceinline__ bool splitTopPSkips(OnlineSoftmax const& st, float maxScaled, float candWeight)
{
    return st.sum == st.sum && __expf(st.max - maxScaled) * kTopPSkipMargin < candWeight;
}

// ---- Cooperative kernel A (head_sampler_coop_fold) ----------------------------------------
//! Kernel A on a plain BF16 row with the cap computed by every thread of the CTA.
//!
//! Grid (numRows, kSplitBlock / VTS); block THREADS >= VTS. The CTA owns virtual threads
//! vt0 .. vt0 + VTS - 1 (vt0 = blockIdx.y * VTS), i.e. the vectors v = vt0 + t + 1024 j.
//! Phase 1: all THREADS threads read those vectors (coalesced: VTS consecutive vectors per
//! j) and store each element's `__fmul_rn(capOf(logit), tempInv)` -- the value kernel A
//! hands to combineOnlineSoftmax -- in shared memory at [j][t]. Phase 2: thread t < VTS
//! folds its vectors j = 0, 1, ... in increasing j, each vector's 4 elements in order,
//! with the same combineOnlineSoftmax calls as splitFoldKernel. Every state is
//! splitFoldKernel's bit for bit: the same function of the same inputs in the same order;
//! only who computes the (pure, per-element) cap and where it waits changes.
//!
//! PDL: waits on its predecessor before its first global read; with EARLY_TRIGGER it
//! releases its dependent (kernel B, which itself waits for this grid's completion before
//! reading the states) right after that wait, so kernel B's launch overlaps this kernel.
template <int VTS, int THREADS, bool EARLY_TRIGGER>
__global__ void __launch_bounds__(THREADS) splitFoldCoopKernel(SplitFoldParams p)
{
    static_assert(THREADS >= VTS && THREADS % VTS == 0 && kSplitBlock % VTS == 0, "CTA shape");
    extern __shared__ float4 sCapped[];
    gridDependencyWait();
    if constexpr (EARLY_TRIGGER)
    {
        gridDependencyTrigger();
    }
    int const row = blockIdx.x;
    int const vt0 = blockIdx.y * VTS;
    int const vecCount = p.fs.vocabSize / 4;
    float const tempInv = loadRowParams(p.fs, row).tempInv;
    uint2 const* plainRow = p.plain + static_cast<size_t>(row) * vecCount;
    // Vectors of the CTA's first virtual thread: the most any of its threads folds.
    int const steps = vecCount > vt0 ? (vecCount - vt0 + kSplitBlock - 1) / kSplitBlock : 0;
    for (int i = threadIdx.x; i < steps * VTS; i += THREADS)
    {
        int const j = i / VTS;
        int const v = vt0 + (i - j * VTS) + j * kSplitBlock;
        if (v < vecCount)
        {
            uint2 const packed = __ldg(plainRow + v);
            __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(&packed);
            float4 x;
            x.x = __fmul_rn(capOf(__bfloat162float(elems[0]), p.cap), tempInv);
            x.y = __fmul_rn(capOf(__bfloat162float(elems[1]), p.cap), tempInv);
            x.z = __fmul_rn(capOf(__bfloat162float(elems[2]), p.cap), tempInv);
            x.w = __fmul_rn(capOf(__bfloat162float(elems[3]), p.cap), tempInv);
            sCapped[i] = x;
        }
    }
    __syncthreads();
    if (threadIdx.x < VTS)
    {
        // combineOnlineSoftmax(st, {x, 1, i}) in two passes over the thread's own slots.
        // Pass 1 carries only the running max (and argmax): `keep = st.max >= x` picks the
        // branch, and the branch's weight __expf(x - st.max) (keep) or __expf(st.max - x)
        // depends on the max before the element only, so it is computed here, off the sum's
        // chain, and stored over x with the branch in its sign bit (the weight is >= 0 or
        // NaN; negation flips only the sign bit). Pass 2 runs the sum's chain alone:
        // keep: st.sum + 1.0f * w; otherwise 1.0f + st.sum * w -- the two statements of
        // combineOnlineSoftmax with b.sum = 1, in the same order.
        int const vt = vt0 + threadIdx.x;
        int const mine = vecCount > vt ? (vecCount - vt + kSplitBlock - 1) / kSplitBlock : 0;
        float runMax = -FLT_MAX;
        int runArg = 0;
        for (int j = 0; j < mine; ++j)
        {
            int const base = (vt + j * kSplitBlock) * 4;
            float4 x = sCapped[j * VTS + threadIdx.x];
            float* xs = reinterpret_cast<float*>(&x);
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                bool const keep = runMax >= xs[e];
                float const w = keep ? __expf(xs[e] - runMax) : __expf(runMax - xs[e]);
                runArg = keep ? runArg : base + e;
                runMax = keep ? runMax : xs[e];
                xs[e] = keep ? w : -w;
            }
            sCapped[j * VTS + threadIdx.x] = x;
        }
        float runSum = 0.0f;
        for (int j = 0; j < mine; ++j)
        {
            float4 const x = sCapped[j * VTS + threadIdx.x];
            float const xs[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                bool const keep = (__float_as_uint(xs[e]) >> 31) == 0u;
                float const w = fabsf(xs[e]);
                runSum = keep ? runSum + 1.0f * w : 1.0f + runSum * w;
            }
        }
        p.states[static_cast<size_t>(row) * kSplitBlock + vt] = OnlineSoftmax{runMax, runSum, runArg};
    }
    if constexpr (!EARLY_TRIGGER)
    {
        __syncthreads();
        gridDependencyTrigger();
    }
}

//! Kernel B: one VB-thread CTA per row (VB = splitVirtualBlock(numRows), the shipped
//! block of the call). `params.logits` is the BF16 row (gathered or not); the soft cap is
//! applied on every read.
//!
//! A row without top-k / top-p / min-p takes the shipped tokens-only rejection path's
//! first round from the states kernel A wrote: cub BlockReduce of the VB states, each
//! thread's mass rebased onto the row max, cub BlockScan, tid 0's Philox draw, and the
//! walk of the thread whose span holds the target -- its elements' weights computed in
//! parallel, then summed by one thread in the original order (when two spans overlap
//! around the target, every claimant walks, as in the shipped kernel). Any other row
//! runs the shipped body itself.
//!
//! PDL: waits on kernel A before reading anything. With `kTrigger` (``TargetHead.verify_pdl``)
//! every CTA releases its dependent right after that wait; the dependent waits for this grid
//! before it reads the tokens.
//!
//! With `kTopPMulti` (the multi-CTA top-p test's first kernel) a top-p-only row
//! (splitTopPOnly) takes the unfiltered first round -- the shipped rejection path's round 0,
//! which draws from the same per-thread states and fixes the same mass target -- and, when it
//! found a candidate, leaves `records[row]` pending instead of writing its token; every row
//! writes its record's `pending`. The CTA releases its dependent at its end (not early).
template <int VB, int WALK, bool kTrigger, bool kTopPMulti = false>
__global__ void __launch_bounds__(VB, VB == kSplitNarrowBlock ? 2 : 1)
    splitDrawKernel(FusedSamplingParams params, OnlineSoftmax const* states, SplitCap cap, SplitTopPRecord* records)
{
    __shared__ FusedSamplingShared<VB> shared;
    __shared__ __align__(16) float wbuf[4 * WALK];
    __shared__ int sWalker;
    __shared__ float sBase;
    __shared__ unsigned long long sLastCrossing;
    int const row = blockIdx.x;
    int const tid = threadIdx.x;
    if (tid == 0)
    {
        sSplitCap = cap;
    }
    __syncthreads();
    gridDependencyWait();
    if constexpr (kTrigger && !kTopPMulti)
    {
        gridDependencyTrigger();
    }
    int const vocabSize = params.vocabSize;
    RowParams const rp = loadRowParams(params, row);
    if (rp.needTopK || rp.needTopP || rp.needMinP)
    {
        if constexpr (kTopPMulti)
        {
            if (!splitTopPOnly(rp))
            {
                if (tid == 0)
                {
                    records[row].pending = 0;
                }
                fusedSamplingBody<CapBf16, VB, true, false>(params, shared);
                return;
            }
        }
        else
        {
            fusedSamplingBody<CapBf16, VB, true, false>(params, shared);
            return;
        }
    }
    using BlockReduceOnline = cub::BlockReduce<OnlineSoftmax, VB>;
    using BlockScanF = cub::BlockScan<float, VB>;
    OnlineSoftmax const cachedThreadStats = states[static_cast<size_t>(row) * VB + tid];
    OnlineSoftmax const rowStats
        = BlockReduceOnline(shared.temp.reduceOnline).Reduce(cachedThreadStats, OnlineSoftmaxOp());
    if (tid == 0)
    {
        shared.maxScaled = rowStats.max;
        shared.argmax = rowStats.argmax;
        shared.totalMass = rowStats.sum;
        sWalker = -1;
    }
    __syncthreads();
    float const maxScaled = shared.maxScaled;
    float const cachedLocalMass = cachedThreadStats.sum * __expf(cachedThreadStats.max - maxScaled);
    if (tid == 0)
    {
        initRowRng(params, row, &shared.rejectRng);
        shared.pivot = -1.0f;
        shared.token = -1;
    }
    __syncthreads();
    float const localMass = cachedLocalMass;
    float base = 0.0f;
    float supportMass = 0.0f;
    BlockScanF(shared.temp.scanF).ExclusiveSum(localMass, base, supportMass);
    __syncthreads();
    if (tid == 0)
    {
        shared.massTarget = rp.topP * supportMass;
        shared.target = curand_uniform(&shared.rejectRng) * supportMass;
        shared.candIdx = -1;
        sLastCrossing = 0ull;
    }
    __syncthreads();
    float const target = shared.target;
    // The shipped kernel lets every thread whose span holds the target walk. Usually that
    // is one thread; it is more when two spans of the float scan overlap (an exclusive
    // prefix and the previous prefix plus its mass are rounded differently) and the target
    // lies in the overlap.
    bool const claims = target >= base && target < base + localMass;
    if (claims)
    {
        sWalker = tid;
        sBase = base;
    }
    int const claimants = __syncthreads_count(claims);
    int const vecCount = vocabSize / 4;
    __nv_bfloat16 const* rowLogits
        = static_cast<__nv_bfloat16 const*>(params.logits) + static_cast<size_t>(row) * vocabSize;
    if (claimants == 1)
    {
        // One walker (it alone wrote sWalker and sBase): its elements' weights in
        // parallel, summed by one thread in the shipped walk's order.
        int const walker = sWalker;
        int const walkVectors = (vecCount - walker + VB - 1) / VB;
        if (tid < walkVectors)
        {
            uint2 const packed = reinterpret_cast<uint2 const*>(rowLogits)[walker + tid * VB];
            __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(&packed);
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                wbuf[tid * 4 + j] = __expf(__fmul_rn(capOf(__bfloat162float(elems[j]), cap), rp.tempInv) - maxScaled);
            }
        }
        __syncthreads();
        if (tid == 0)
        {
            // Four weights per shared load, still added one at a time in order.
            float running = sBase;
            int cand = -1;
            float candWeight = 0.0f;
            float4 const* w4 = reinterpret_cast<float4 const*>(wbuf);
            for (int q = 0; q < walkVectors && cand < 0; ++q)
            {
                float4 const quad = w4[q];
                float const ws[4] = {quad.x, quad.y, quad.z, quad.w};
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    float const w = ws[j];
                    // supportFloor 0 and pivot -1 in the shipped walk's membership test.
                    if (cand < 0 && w >= 0.0f && w > -1.0f)
                    {
                        running += w;
                        if (running > target)
                        {
                            cand = (walker + q * VB) * 4 + j;
                            candWeight = w;
                        }
                    }
                }
            }
            if (cand >= 0)
            {
                shared.candIdx = cand;
                shared.candWeight = candWeight;
            }
        }
    }
    else if (claimants > 1)
    {
        // Every claimant walks its own elements from its own base, as in the shipped
        // kernel, where each walk that crosses the target writes the candidate and the
        // last write stands: the walk that crosses at the later position. Here that
        // choice is made deterministically -- the later crossing position, then the
        // higher thread -- instead of by the order the writes land.
        int crossIdx = -1;
        float crossWeight = 0.0f;
        unsigned long long crossing = 0ull;
        if (claims)
        {
            float running = base;
            int position = 0;
            for (int v = tid; v < vecCount && crossIdx < 0; v += VB)
            {
                uint2 const packed = reinterpret_cast<uint2 const*>(rowLogits)[v];
                __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(&packed);
                for (int j = 0; j < 4; ++j)
                {
                    float const w = __expf(__fmul_rn(capOf(__bfloat162float(elems[j]), cap), rp.tempInv) - maxScaled);
                    ++position;
                    if (!(w >= 0.0f && w > -1.0f))
                    {
                        continue;
                    }
                    running += w;
                    if (running > target)
                    {
                        crossIdx = v * 4 + j;
                        crossWeight = w;
                        crossing = (static_cast<unsigned long long>(position) << 32)
                            | static_cast<unsigned long long>(tid + 1);
                        break;
                    }
                }
            }
            if (crossIdx >= 0)
            {
                atomicMax(&sLastCrossing, crossing);
            }
        }
        __syncthreads();
        if (crossIdx >= 0 && crossing == sLastCrossing)
        {
            shared.candIdx = crossIdx;
            shared.candWeight = crossWeight;
        }
    }
    __syncthreads();
    if (tid == 0)
    {
        int const cand = shared.candIdx;
        if constexpr (kTopPMulti)
        {
            // A top-p row with a candidate: the shipped loop tests it next (the test kernels).
            // Without one (the target fell past the support) the shipped loop breaks to the
            // argmax, as the unfiltered row does.
            SplitTopPRecord* const rec = records + row;
            bool const pending = rp.needTopP && cand >= 0;
            if (pending)
            {
                rec->rng = shared.rejectRng;
                rec->maxScaled = maxScaled;
                rec->candWeight = shared.candWeight;
                rec->massTarget = shared.massTarget;
                rec->cand = cand;
                rec->argmax = shared.argmax;
                rec->round = 0;
            }
            else
            {
                params.outputTokens[row] = cand >= 0 ? cand : shared.argmax;
            }
            rec->pending = pending ? 1 : 0;
        }
        else
        {
            params.outputTokens[row] = cand >= 0 ? cand : shared.argmax;
        }
    }
    if constexpr (kTopPMulti)
    {
        gridDependencyTrigger();
    }
}

//! The test's sweep of one round over VB / kTopPCountThreads CTAs per row: virtual thread vt's
//! CountMass of its elements w with w >= 0 && w > the record's candidate weight -- the shipped
//! sweep's statements (`w = __expf(__fmul_rn(cap(x), tempInv) - maxScaled)`, count += 1, mass
//! += w) over its elements in the shipped order (forEachLogit<CapBf16>: vectors vt + VB j in
//! increasing j, elements 0..3), so each partial is the shipped thread vt's `local` bit for
//! bit. A virtual thread whose kernel A state shows no element above the candidate's weight
//! (splitTopPSkips) stores {0, 0} without reading its row. Rows whose record is not pending
//! exit. Grid (numRows, VB / kTopPCountThreads).
//!
//! PDL: waits on its predecessor before reading anything; releases its dependent at its end.
template <int VB>
__global__ void __launch_bounds__(kTopPCountThreads) splitTopPCountKernel(FusedSamplingParams params,
    OnlineSoftmax const* states, SplitTopPRecord const* records, CountMass* partials, SplitCap cap)
{
    static_assert(VB % kTopPCountThreads == 0, "whole CTAs of virtual threads");
    gridDependencyWait();
    int const row = blockIdx.x;
    SplitTopPRecord const* const rec = records + row;
    if (rec->pending == 0)
    {
        return;
    }
    int const vt = blockIdx.y * kTopPCountThreads + threadIdx.x;
    float const tempInv = loadRowParams(params, row).tempInv;
    float const maxScaled = rec->maxScaled;
    float const candWeight = rec->candWeight;
    int const vecCount = params.vocabSize / 4;
    uint2 const* const rowVecs = static_cast<uint2 const*>(params.logits) + static_cast<size_t>(row) * vecCount;
    CountMass local{0.0f, 0.0f};
    bool const skip = splitTopPSkips(states[static_cast<size_t>(row) * VB + vt], maxScaled, candWeight);
    for (int v0 = vt; !skip && v0 < vecCount; v0 += kSplitPrefetch * VB)
    {
        uint2 buf[kSplitPrefetch];
#pragma unroll
        for (int u = 0; u < kSplitPrefetch; ++u)
        {
            int const v = v0 + u * VB;
            if (v < vecCount)
            {
                buf[u] = __ldg(rowVecs + v);
            }
        }
#pragma unroll
        for (int u = 0; u < kSplitPrefetch; ++u)
        {
            int const v = v0 + u * VB;
            if (v < vecCount)
            {
                __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(&buf[u]);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    float const w = __expf(__fmul_rn(capOf(__bfloat162float(elems[j]), cap), tempInv) - maxScaled);
                    if (w >= 0.0f && w > candWeight)
                    {
                        local.count += 1.0f;
                        local.mass += w;
                    }
                }
            }
        }
    }
    partials[static_cast<size_t>(row) * VB + vt] = local;
    gridDependencyTrigger();
}

//! Shared memory of the test's decide kernel.
template <int VB>
struct SplitTopPShared
{
    union
    {
        typename cub::BlockReduce<CountMass, VB>::TempStorage reduceCountMass;
        typename cub::BlockScan<float, VB>::TempStorage scanF;
    } temp;

    curandStatePhilox4_32_10_t rejectRng;
    float target;
    float candWeight;
    int candIdx;
    int keep;
    int walker;
    float walkerBase;
    unsigned long long lastCrossing;
};

//! A draw of the shipped rejection loop at a round >= 1, from every thread's support mass
//! `localMass` (its elements w with w >= 0 && w > pivot, summed in the sweep order): the
//! block scan, tid 0's Philox uniform from shared.rejectRng times the support mass, and the
//! claimants' walk with the shipped membership test (w >= 0 && w > pivot) -- one claimant's
//! weights computed in parallel and summed by one thread in its order (kernel B's walk); several
//! claimants walk their own elements and the later crossing position, then the higher thread,
//! stands (kernel B's rule for the shipped kernel's racing writes). Leaves the candidate in
//! shared.candIdx / shared.candWeight (-1: the target fell past the support). `wbuf` holds
//! 4 x WALK floats; a walker's vectors fit (drawTopP checks vocab / 4 <= WALK x VB).
template <int VB, int WALK>
__device__ void splitTopPDraw(SplitTopPShared<VB>& shared, float* wbuf, uint2 const* rowVecs, int vecCount,
    SplitCap const& cap, float tempInv, float maxScaled, float pivot, float localMass)
{
    int const tid = threadIdx.x;
    float base = 0.0f;
    float supportMass = 0.0f;
    cub::BlockScan<float, VB>(shared.temp.scanF).ExclusiveSum(localMass, base, supportMass);
    __syncthreads();
    if (tid == 0)
    {
        shared.target = curand_uniform(&shared.rejectRng) * supportMass;
        shared.candIdx = -1;
        shared.walker = -1;
        shared.lastCrossing = 0ull;
    }
    __syncthreads();
    float const target = shared.target;
    bool const claims = target >= base && target < base + localMass;
    if (claims)
    {
        shared.walker = tid;
        shared.walkerBase = base;
    }
    int const claimants = __syncthreads_count(claims);
    if (claimants == 1)
    {
        int const walker = shared.walker;
        int const walkVectors = (vecCount - walker + VB - 1) / VB;
        if (tid < walkVectors)
        {
            uint2 const packed = rowVecs[walker + tid * VB];
            __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(&packed);
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                wbuf[tid * 4 + j] = __expf(__fmul_rn(capOf(__bfloat162float(elems[j]), cap), tempInv) - maxScaled);
            }
        }
        __syncthreads();
        if (tid == 0)
        {
            float running = shared.walkerBase;
            int cand = -1;
            float candWeight = 0.0f;
            float4 const* w4 = reinterpret_cast<float4 const*>(wbuf);
            for (int q = 0; q < walkVectors && cand < 0; ++q)
            {
                float4 const quad = w4[q];
                float const ws[4] = {quad.x, quad.y, quad.z, quad.w};
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    float const w = ws[j];
                    if (cand < 0 && w >= 0.0f && w > pivot)
                    {
                        running += w;
                        if (running > target)
                        {
                            cand = (walker + q * VB) * 4 + j;
                            candWeight = w;
                        }
                    }
                }
            }
            if (cand >= 0)
            {
                shared.candIdx = cand;
                shared.candWeight = candWeight;
            }
        }
    }
    else if (claimants > 1)
    {
        int crossIdx = -1;
        float crossWeight = 0.0f;
        unsigned long long crossing = 0ull;
        if (claims)
        {
            float running = base;
            int position = 0;
            for (int v = tid; v < vecCount && crossIdx < 0; v += VB)
            {
                uint2 const packed = rowVecs[v];
                __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(&packed);
                for (int j = 0; j < 4; ++j)
                {
                    float const w = __expf(__fmul_rn(capOf(__bfloat162float(elems[j]), cap), tempInv) - maxScaled);
                    ++position;
                    if (!(w >= 0.0f && w > pivot))
                    {
                        continue;
                    }
                    running += w;
                    if (running > target)
                    {
                        crossIdx = v * 4 + j;
                        crossWeight = w;
                        crossing = (static_cast<unsigned long long>(position) << 32)
                            | static_cast<unsigned long long>(tid + 1);
                        break;
                    }
                }
            }
            if (crossIdx >= 0)
            {
                atomicMax(&shared.lastCrossing, crossing);
            }
        }
        __syncthreads();
        if (crossIdx >= 0 && crossing == shared.lastCrossing)
        {
            shared.candIdx = crossIdx;
            shared.candWeight = crossWeight;
        }
    }
    __syncthreads();
}

//! The test's decision of one round: one VB-thread CTA per pending row reduces the VB partials
//! with the shipped cub::BlockReduce<CountMass, VB> (thread t holds virtual thread t's partial,
//! so the block value is the shipped `above` bit for bit) and keeps the candidate iff
//! above.mass <= the mass target (top-k is off). On a rejection the candidate's weight becomes
//! the pivot and the next round draws from each thread's partial mass (splitTopPDraw); without
//! `kFinal` the new candidate goes back into the record for the next sweep kernel, with it
//! the CTA sweeps the row itself (the shipped sweep) until the row is decided. The shipped
//! round budget (kMaxRejectRounds) and its fallbacks (the argmax) hold.
//!
//! PDL: waits on the sweep kernel before reading anything. With `kTrigger` (the last kernel of
//! the call under ``TargetHead.verify_pdl``) every CTA releases its dependent right after that
//! wait; without `kFinal` at its end.
template <int VB, int WALK, bool kFinal, bool kTrigger>
__global__ void __launch_bounds__(VB, VB == kSplitNarrowBlock ? 2 : 1) splitTopPDecideKernel(FusedSamplingParams params,
    OnlineSoftmax const* states, SplitTopPRecord* records, CountMass const* partials, SplitCap cap)
{
    __shared__ SplitTopPShared<VB> shared;
    __shared__ __align__(16) float wbuf[4 * WALK];
    gridDependencyWait();
    if constexpr (kTrigger)
    {
        gridDependencyTrigger();
    }
    int const row = blockIdx.x;
    int const tid = threadIdx.x;
    SplitTopPRecord* const rec = records + row;
    if (rec->pending == 0)
    {
        return;
    }
    float const maxScaled = rec->maxScaled;
    float const massTarget = rec->massTarget;
    int const argmax = rec->argmax;
    float candWeight = rec->candWeight;
    int cand = rec->cand;
    int round = rec->round;
    if (tid == 0)
    {
        shared.rejectRng = rec->rng;
    }
    float const tempInv = loadRowParams(params, row).tempInv;
    int const vecCount = params.vocabSize / 4;
    uint2 const* const rowVecs = static_cast<uint2 const*>(params.logits) + static_cast<size_t>(row) * vecCount;
    CountMass local = partials[static_cast<size_t>(row) * VB + tid];
    // Every thread has read the record before tid 0 may rewrite it.
    __syncthreads();
    for (;;)
    {
        CountMass const above
            = cub::BlockReduce<CountMass, VB>(shared.temp.reduceCountMass).Reduce(local, CountMassOp());
        __syncthreads();
        if (tid == 0)
        {
            shared.keep = above.mass <= massTarget ? 1 : 0;
        }
        __syncthreads();
        if (shared.keep)
        {
            if (tid == 0)
            {
                params.outputTokens[row] = cand;
                rec->pending = 0;
            }
            break;
        }
        // Rejected: the kept set is downward-closed in w, so the candidate and everything at
        // or below its weight leave the support.
        float const pivot = candWeight;
        if (++round >= kMaxRejectRounds)
        {
            if (tid == 0)
            {
                params.outputTokens[row] = argmax;
                rec->pending = 0;
            }
            break;
        }
        splitTopPDraw<VB, WALK>(shared, wbuf, rowVecs, vecCount, cap, tempInv, maxScaled, pivot, local.mass);
        cand = shared.candIdx;
        candWeight = shared.candWeight;
        if (cand < 0)
        {
            if (tid == 0)
            {
                params.outputTokens[row] = argmax;
                rec->pending = 0;
            }
            break;
        }
        if constexpr (!kFinal)
        {
            if (tid == 0)
            {
                rec->rng = shared.rejectRng;
                rec->candWeight = candWeight;
                rec->cand = cand;
                rec->round = round;
            }
            break;
        }
        else
        {
            // The next test's sweep in this CTA: the shipped sweep (thread tid's vectors), skipped
            // where it provably adds nothing (splitTopPSkips).
            local = CountMass{0.0f, 0.0f};
            bool const skip = splitTopPSkips(states[static_cast<size_t>(row) * VB + tid], maxScaled, candWeight);
            for (int v = tid; !skip && v < vecCount; v += VB)
            {
                uint2 const packed = rowVecs[v];
                __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(&packed);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    float const w = __expf(__fmul_rn(capOf(__bfloat162float(elems[j]), cap), tempInv) - maxScaled);
                    if (w >= 0.0f && w > candWeight)
                    {
                        local.count += 1.0f;
                        local.mass += w;
                    }
                }
            }
        }
    }
    if constexpr (!kFinal)
    {
        gridDependencyTrigger();
    }
}

//! Test reference: the shipped pass 1 of a neutral tokens-only row, verbatim, on FP32
//! soft-capped logits; stores each of the VB threads' online-softmax state.
template <int VB>
__global__ void __launch_bounds__(VB, 1) splitRefFoldKernel(FusedSamplingParams params, OnlineSoftmax* states)
{
    int const row = blockIdx.x;
    int const tid = threadIdx.x;
    float const* rowLogits = static_cast<float const*>(params.logits) + static_cast<size_t>(row) * params.vocabSize;
    RowParams const rp = loadRowParams(params, row);
    OnlineSoftmax cachedThreadStats{-FLT_MAX, 0.0f, 0};
    forEachLogit(rowLogits, params.vocabSize,
        [&](int i, float logit) {
            cachedThreadStats
                = combineOnlineSoftmax(cachedThreadStats, OnlineSoftmax{__fmul_rn(logit, rp.tempInv), 1.0f, i});
        });
    states[static_cast<size_t>(row) * VB + tid] = cachedThreadStats;
}

//! Test reference: capOf over a BF16 tensor.
__global__ void splitCapKernel(__nv_bfloat16 const* x, float* y, int64_t n, SplitCap cap)
{
    int64_t const i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    if (i < n)
    {
        y[i] = capOf(__bfloat162float(x[i]), cap);
    }
}

//! Test probe: records, per 32-bit word of `buf`, the first value it sees with both BF16
//! halves published, while a producer kernel on another stream writes `buf`. Exits when
//! every word was seen or after `timeoutNs`.
__global__ void splitWatchKernel(uint32_t const* buf, uint32_t* firstSeen, int64_t words, unsigned long long timeoutNs)
{
    int64_t const stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    int64_t const start = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    unsigned long long const t0 = globalTimerNs();
    for (;;)
    {
        bool pending = false;
        for (int64_t i = start; i < words; i += stride)
        {
            if (firstSeen[i] != kSplitSentinel)
            {
                continue;
            }
            uint32_t v;
            asm volatile("ld.relaxed.sys.global.b32 %0, [%1];" : "=r"(v) : "l"(buf + i) : "memory");
            if (__vcmpeq2(v, kSplitSentinel) == 0u)
            {
                firstSeen[i] = v;
            }
            else
            {
                pending = true;
            }
        }
        if (!pending || globalTimerNs() - t0 > timeoutNs)
        {
            return;
        }
    }
}

// ----------------------------------------------------------------------------------------
// The vocabulary-parallel head (vp_*): each rank folds only its own LM-head shard.
//
// Virtual thread vt of the shipped VB-thread block owns the vectors v = vt (mod VB) of the
// whole vocabulary; rank r holds the vectors [r * shardVecs, (r + 1) * shardVecs), so vt's
// elements in vocabulary order are rank 0's portion, then rank 1's, and so on. The shipped
// fold and walk are sequential over those elements (combineOnlineSoftmax and the walk's
// running sum are not associative in FP32), so the ranks continue them in rank order instead
// of combining independent partials: vp_fold on rank r > 0 waits for rank r - 1's state of
// (row, vt) and folds its own portion onto it, with the shipped per-element statements, and
// publishes the result to every rank; rank R - 1's state is the shipped state of vt, bit for
// bit. vp_draw runs the shipped first round on those states (block reduce, scan, Philox draw,
// claimants); for each claimant, rank 0 walks its portion from the claimant's base and every
// later rank continues from the running sum the rank before it published, so each rank's
// crossing (or none) is the shipped walk's. vp_decide takes, per claimant, the first rank in
// rank order with a crossing, and kernel B's rule across claimants. Greedy batches: vp_fold
// reduces torch.argmax's comparator over each CTA and vp_argmax over the ranks' CTAs (an
// exact max, any order).
//
// Exchange buffer (one copy per rank, every 32-bit word holds kSplitSentinel between
// calls): states [ranks][stateStride] x 16 B (rank r's slot: vt's state after ranks 0..r),
// greedy words [ranks][greedyStride] x 8 B at greedyOffset, walk records [ranks][candStride]
// x 8 B at candOffset (the crossing or kVpNone / kVpUpstream, and the running sum at the end
// of the rank's portion). Each value is stored through every publish target (the multicast
// address in service; every rank's copy in the single-GPU emulation); every rank reads and
// resets its own copy.

//! Ranks the vocabulary-parallel head takes.
constexpr int kVpMaxRanks = 8;
//! Claimants of a row whose crossings the head exchanges (in thread order).
constexpr int kVpClaimants = 4;
//! A candidate word: no crossing up to the end of this rank's portion.
constexpr uint32_t kVpNone = 0x7FFFFFFEu;
//! A candidate word: an earlier rank's portion holds the crossing (this rank did not walk).
constexpr uint32_t kVpUpstream = 0x7FFFFFFDu;
//! Most vectors of one virtual thread in one shard (the walk's shared-memory stage).
constexpr int kVpWalkVectors = 64;
//! Threads per CTA of vp_fold.
constexpr int kVpFoldThreads = 128;
//! Greedy words per (rank, row): one per vp_fold CTA, at most kSplitBlock / kVpFoldThreads.
constexpr int kVpGreedySlots = kSplitBlock / kVpFoldThreads;
//! vp_draw's per-row record for vp_decide: [claimants or -1 (filtered), argmax, claimant tids].
constexpr int kVpInfoWords = 2 + kVpClaimants;

struct VpParams
{
    //! Row parameters; vocabSize is the whole vocabulary; logits is the pending
    //! [numRows, vocabSize] BF16 row (written by vp_draw for filtered rows only).
    FusedSamplingParams fs;
    //! This rank's [numRows][shardVecs] local LM-head output.
    uint2 const* local{nullptr};
    //! This rank's staging rows [rows][shardVecs] (fabric memory; filtered rows only).
    uint2* staging{nullptr};
    //! Every rank's staging buffer, in rank order (peer addresses).
    uint2 const* stagingOf[kVpMaxRanks]{};
    //! Exchange-buffer bases the values are stored through.
    char* publish[kVpMaxRanks]{};
    int numPublish{0};
    //! This rank's copy of the exchange buffer.
    char* copy{nullptr};
    int* info{nullptr};
    int* status{nullptr};
    int64_t* greedyTokens{nullptr};
    SplitCap cap{};
    int rank{0};
    int ranks{0};
    int shardVecs{0};
    long long stateStride{0};
    long long greedyOffset{0};
    long long greedyStride{0};
    long long candOffset{0};
    long long candStride{0};
    unsigned long long timeoutNs{0};
    //! Sampling rows: 1 continues the fold and the walk in rank order (each rank's state is the
    //! single-rank sampler's, bit for bit; vocab_parallel_head__RANK_CHAIN); 0 folds independent
    //! per-rank partials and combines them in rank order in vp_draw, each rank walking from the
    //! claimant's base plus the earlier ranks' masses (the partial-combine form).
    int chain{1};
};

__device__ __forceinline__ uint4* vpState(char* base, VpParams const& p, int rank, int row, int vt, int vb)
{
    return reinterpret_cast<uint4*>(base) + rank * p.stateStride + static_cast<long long>(row) * vb + vt;
}

__device__ __forceinline__ uint2* vpGreedy(char* base, VpParams const& p, int rank, int row, int slot)
{
    return reinterpret_cast<uint2*>(base + p.greedyOffset) + rank * p.greedyStride
        + static_cast<long long>(row) * kVpGreedySlots + slot;
}

//! The walk record of (rank, row, claimant c): {candidate word, running-sum bits}.
__device__ __forceinline__ uint2* vpCand(char* base, VpParams const& p, int rank, int row, int c)
{
    return reinterpret_cast<uint2*>(base + p.candOffset) + rank * p.candStride
        + static_cast<long long>(row) * kVpClaimants + c;
}

__device__ __forceinline__ uint4 loadRelaxedSys4(uint4 const* address)
{
    uint4 v;
    asm volatile("ld.relaxed.sys.global.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(address)
                 : "memory");
    return v;
}

__device__ __forceinline__ void storeRelaxedSys4(uint4* address, uint4 v)
{
    asm volatile(
        "st.relaxed.sys.global.v4.b32 [%0], {%1, %2, %3, %4};" ::"l"(address), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
        : "memory");
}

__device__ __forceinline__ uint32_t loadRelaxedSys1(uint32_t const* address)
{
    uint32_t v;
    asm volatile("ld.relaxed.sys.global.b32 %0, [%1];" : "=r"(v) : "l"(address) : "memory");
    return v;
}

__device__ __forceinline__ void storeRelaxedSys1(uint32_t* address, uint32_t v)
{
    asm volatile("st.relaxed.sys.global.b32 [%0], %1;" ::"l"(address), "r"(v) : "memory");
}

//! Records a poll that waited past the timeout and traps (a peer that never published).
__device__ __noinline__ void vpPollTimedOut(int* status)
{
    atomicOr(status, kSplitPollTimeout);
    __threadfence_system();
    __trap();
}

//! Spin until none of the first `words` 32-bit words at `address` is the sentinel.
template <int WORDS>
__device__ __forceinline__ uint4 vpAwait(void const* address, int* status, unsigned long long timeoutNs)
{
    unsigned long long start = 0;
    for (int spins = 0;; ++spins)
    {
        uint4 v{kSplitSentinel, kSplitSentinel, kSplitSentinel, kSplitSentinel};
        if constexpr (WORDS == 1)
        {
            v.x = loadRelaxedSys1(static_cast<uint32_t const*>(address));
        }
        else if constexpr (WORDS == 2)
        {
            uint2 const w = loadRelaxedSys(static_cast<uint2 const*>(address));
            v.x = w.x;
            v.y = w.y;
        }
        else
        {
            v = loadRelaxedSys4(static_cast<uint4 const*>(address));
        }
        bool ready = v.x != kSplitSentinel;
        if constexpr (WORDS >= 2)
        {
            ready = ready && v.y != kSplitSentinel;
        }
        if constexpr (WORDS >= 3)
        {
            ready = ready && v.z != kSplitSentinel;
        }
        if (ready)
        {
            return v;
        }
        if (spins == 64)
        {
            start = globalTimerNs();
        }
        if (spins >= 64)
        {
            __nanosleep(64);
            if (globalTimerNs() - start > timeoutNs)
            {
                vpPollTimedOut(status);
            }
        }
    }
}

//! torch.argmax's comparator (GreaterOrNan): whether (a, ia) is chosen over (b, ib).
__device__ __forceinline__ bool vpArgmaxBetter(float a, int ia, float b, int ib)
{
    if (isnan(a))
    {
        return isnan(b) ? ia < ib : true;
    }
    return a == b ? ia < ib : a > b;
}

struct VpMaxArg
{
    float value;
    int index;
};

struct VpMaxArgOp
{
    __device__ __forceinline__ VpMaxArg operator()(VpMaxArg const& a, VpMaxArg const& b) const
    {
        return vpArgmaxBetter(a.value, a.index, b.value, b.index) ? a : b;
    }
};

//! The first vector of virtual thread vt in the shard starting at global vector `first`, as
//! an offset into the shard.
__device__ __forceinline__ int vpLead(int vt, int first, int vb)
{
    return ((vt - first) % vb + vb) % vb;
}

//! The state of (row, vt) that rank `rank` published (vt's fold over ranks 0..rank).
__device__ __forceinline__ OnlineSoftmax vpAwaitState(VpParams const& p, int rank, int row, int vt, int vb)
{
    uint4 const w = vpAwait<3>(vpState(p.copy, p, rank, row, vt, vb), p.status, p.timeoutNs);
    return OnlineSoftmax{__uint_as_float(w.x), __uint_as_float(w.y), static_cast<int>(w.z)};
}

//! vp_fold: grid (numRows, VB / kVpFoldThreads), kVpFoldThreads threads. PDL: waits on
//! the LM-head GEMM before reading anything; releases its dependent after publishing.
//! Sampling (not GREEDY) on rank r > 0: the fold starts from rank r - 1's published state of
//! (row, vt), read from this rank's copy once the first loads are issued.
template <bool GREEDY, int VB>
__global__ void __launch_bounds__(kVpFoldThreads) vpFoldKernel(VpParams p)
{
    gridDependencyWait();
    int const row = blockIdx.x;
    int const vt = blockIdx.y * kVpFoldThreads + threadIdx.x;
    int const sv = p.shardVecs;
    int const first = p.rank * sv;
    int const lead = vpLead(vt, first, VB);
    uint2 const* localRow = p.local + static_cast<size_t>(row) * sv;
    float tempInv = 1.0f;
    bool stage = false;
    if constexpr (!GREEDY)
    {
        RowParams const rp = loadRowParams(p.fs, row);
        tempInv = rp.tempInv;
        stage = rp.needTopK || rp.needTopP || rp.needMinP;
    }
    uint2* stagingRow = p.staging + static_cast<size_t>(row) * sv;
    OnlineSoftmax st{-FLT_MAX, 0.0f, 0};
    // Whether st holds the fold of the earlier ranks' portions (rank 0: the empty fold). The
    // partial-combine form (!p.chain) folds this rank's portion alone.
    bool seeded = GREEDY || p.rank == 0 || !p.chain;
    VpMaxArg best{-INFINITY, INT_MAX};
    for (int l0 = lead; l0 < sv; l0 += kSplitPrefetch * VB)
    {
        uint2 buf[kSplitPrefetch];
#pragma unroll
        for (int u = 0; u < kSplitPrefetch; ++u)
        {
            int const l = l0 + u * VB;
            if (l < sv)
            {
                buf[u] = __ldg(localRow + l);
            }
        }
        if (!seeded)
        {
            st = vpAwaitState(p, p.rank - 1, row, vt, VB);
            seeded = true;
        }
#pragma unroll
        for (int u = 0; u < kSplitPrefetch; ++u)
        {
            int const l = l0 + u * VB;
            if (l < sv)
            {
                if (stage)
                {
                    stagingRow[l] = buf[u];
                }
                __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(&buf[u]);
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    float const logit = capOf(__bfloat162float(elems[j]), p.cap);
                    int const index = (first + l) * 4 + j;
                    if constexpr (GREEDY)
                    {
                        if (vpArgmaxBetter(logit, index, best.value, best.index))
                        {
                            best = VpMaxArg{logit, index};
                        }
                    }
                    else
                    {
                        st = combineOnlineSoftmax(st, OnlineSoftmax{__fmul_rn(logit, tempInv), 1.0f, index});
                    }
                }
            }
        }
    }
    if constexpr (GREEDY)
    {
        using BlockReduceMaxArg = cub::BlockReduce<VpMaxArg, kVpFoldThreads>;
        __shared__ typename BlockReduceMaxArg::TempStorage temp;
        VpMaxArg const cta = BlockReduceMaxArg(temp).Reduce(best, VpMaxArgOp());
        if (threadIdx.x == 0)
        {
            uint2 const word = make_uint2(__float_as_uint(cta.value), static_cast<uint32_t>(cta.index));
            for (int i = 0; i < p.numPublish; ++i)
            {
                *vpGreedy(p.publish[i], p, p.rank, row, blockIdx.y) = word;
            }
        }
    }
    else
    {
        if (!seeded)
        {
            // No vector of vt in this shard: the state passes through.
            st = vpAwaitState(p, p.rank - 1, row, vt, VB);
        }
        if (stage)
        {
            // The staged row reaches the peers before this thread's state does.
            __threadfence_system();
        }
        uint4 const word
            = make_uint4(__float_as_uint(st.max), __float_as_uint(st.sum), static_cast<uint32_t>(st.argmax), 0u);
        for (int i = 0; i < p.numPublish; ++i)
        {
            *vpState(p.publish[i], p, p.rank, row, vt, VB) = word;
        }
    }
    gridDependencyTrigger();
}

//! vp_draw: one VB-thread CTA per row. PDL: waits on vp_fold before reading anything.
template <int VB>
__global__ void __launch_bounds__(VB, VB == kSplitNarrowBlock ? 2 : 1) vpDrawKernel(VpParams p)
{
    __shared__ FusedSamplingShared<VB> shared;
    __shared__ typename cub::BlockScan<int, VB>::TempStorage scanClaims;
    __shared__ __align__(16) float wbuf[4 * kVpWalkVectors];
    __shared__ int sClaimTid[kVpClaimants];
    __shared__ float sClaimStart[kVpClaimants];
    __shared__ uint32_t sCand[kVpClaimants];
    int const row = blockIdx.x;
    int const tid = threadIdx.x;
    if (tid == 0)
    {
        sSplitCap = p.cap;
    }
    __syncthreads();
    gridDependencyWait();
    RowParams const rp = loadRowParams(p.fs, row);
    bool const filtered = rp.needTopK || rp.needTopP || rp.needMinP;
    // This thread's state: the last rank's slot (the fold over every rank's portion); in the
    // partial-combine form (!p.chain) the R partial states, combined in rank order (a claimant
    // reads the earlier ranks' again for its walk offset). Every slot is awaited, so each has
    // landed in this copy before the reset below.
    OnlineSoftmax st{-FLT_MAX, 0.0f, 0};
    for (int r = 0; r < p.ranks; ++r)
    {
        OnlineSoftmax const part = vpAwaitState(p, r, row, tid, VB);
        st = (p.chain || r == 0) ? part : combineOnlineSoftmax(st, part);
    }
    uint4 const sentinel4 = make_uint4(kSplitSentinel, kSplitSentinel, kSplitSentinel, kSplitSentinel);
    int const sv = p.shardVecs;
    int const vecCount = p.fs.vocabSize / 4;
    int* const info = p.info + static_cast<size_t>(row) * kVpInfoWords;
    if (filtered)
    {
        // Every rank staged this row before publishing its states (vp_fold); gather it
        // into the pending row and run the shipped body on it, as kernel B does.
        __threadfence_system();
        __syncthreads();
        uint2* const pendingRow
            = reinterpret_cast<uint2*>(const_cast<void*>(p.fs.logits)) + static_cast<size_t>(row) * vecCount;
        for (int v = tid; v < vecCount; v += VB)
        {
            int const r = v / sv;
            pendingRow[v] = loadRelaxedSys(p.stagingOf[r] + static_cast<size_t>(row) * sv + (v - r * sv));
        }
        __syncthreads();
        fusedSamplingBody<CapBf16, VB, true, false>(p.fs, shared);
        for (int r = 0; r < p.ranks; ++r)
        {
            storeRelaxedSys4(vpState(p.copy, p, r, row, tid, VB), sentinel4);
        }
        if (tid < kVpClaimants)
        {
            for (int i = 0; i < p.numPublish; ++i)
            {
                *vpCand(p.publish[i], p, p.rank, row, tid) = make_uint2(kVpNone, 0u);
            }
        }
        if (tid == 0)
        {
            info[0] = -1;
        }
        gridDependencyTrigger();
        return;
    }
    using BlockReduceOnline = cub::BlockReduce<OnlineSoftmax, VB>;
    using BlockScanF = cub::BlockScan<float, VB>;
    OnlineSoftmax const rowStats = BlockReduceOnline(shared.temp.reduceOnline).Reduce(st, OnlineSoftmaxOp());
    if (tid == 0)
    {
        shared.maxScaled = rowStats.max;
        shared.argmax = rowStats.argmax;
    }
    __syncthreads();
    float const maxScaled = shared.maxScaled;
    float const localMass = st.sum * __expf(st.max - maxScaled);
    if (tid == 0)
    {
        initRowRng(p.fs, row, &shared.rejectRng);
    }
    __syncthreads();
    float base = 0.0f;
    float supportMass = 0.0f;
    BlockScanF(shared.temp.scanF).ExclusiveSum(localMass, base, supportMass);
    __syncthreads();
    if (tid == 0)
    {
        shared.target = curand_uniform(&shared.rejectRng) * supportMass;
    }
    __syncthreads();
    float const target = shared.target;
    bool const claims = target >= base && target < base + localMass;
    int claimIdx = 0;
    int claimants = 0;
    cub::BlockScan<int, VB>(scanClaims).ExclusiveSum(claims ? 1 : 0, claimIdx, claimants);
    if (claims && claimIdx < kVpClaimants)
    {
        sClaimTid[claimIdx] = tid;
        if (p.chain)
        {
            sClaimStart[claimIdx] = base;
        }
        else
        {
            // The partial-combine form: the masses of this thread's portions on the ranks before this one.
            float earlier = 0.0f;
            for (int r = 0; r < p.rank; ++r)
            {
                uint4 const w = loadRelaxedSys4(vpState(p.copy, p, r, row, tid, VB));
                earlier += __uint_as_float(w.y) * __expf(__uint_as_float(w.x) - maxScaled);
            }
            sClaimStart[claimIdx] = base + earlier;
        }
    }
    for (int r = 0; r < p.ranks; ++r)
    {
        storeRelaxedSys4(vpState(p.copy, p, r, row, tid, VB), sentinel4);
    }
    __syncthreads();
    int const first = p.rank * sv;
    uint2 const* localRow = p.local + static_cast<size_t>(row) * sv;
    int const walkers = min(claimants, kVpClaimants);
    for (int c = 0; c < walkers; ++c)
    {
        int const walker = sClaimTid[c];
        int const lead = vpLead(walker, first, VB);
        int const walkVectors = lead < sv ? (sv - lead + VB - 1) / VB : 0;
        if (tid < walkVectors)
        {
            uint2 const packed = __ldg(localRow + lead + tid * VB);
            __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(&packed);
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                wbuf[tid * 4 + j] = __expf(__fmul_rn(capOf(__bfloat162float(elems[j]), p.cap), rp.tempInv) - maxScaled);
            }
        }
        __syncthreads();
        if (tid == 0)
        {
            // The shipped walk's running sum at the start of this rank's portion: the base on
            // rank 0, else where the rank before ended (or its crossing already decided).
            float running = sClaimStart[c];
            uint32_t cand = kVpNone;
            if (p.chain && p.rank > 0)
            {
                uint4 const prev = vpAwait<2>(vpCand(p.copy, p, p.rank - 1, row, c), p.status, p.timeoutNs);
                if (prev.x != kVpNone)
                {
                    cand = kVpUpstream;
                }
                running = __uint_as_float(prev.y);
            }
            float4 const* w4 = reinterpret_cast<float4 const*>(wbuf);
            for (int q = 0; q < walkVectors && cand == kVpNone; ++q)
            {
                float4 const quad = w4[q];
                float const ws[4] = {quad.x, quad.y, quad.z, quad.w};
#pragma unroll
                for (int j = 0; j < 4; ++j)
                {
                    float const w = ws[j];
                    if (cand == kVpNone && w >= 0.0f && w > -1.0f)
                    {
                        running += w;
                        if (running > target)
                        {
                            cand = static_cast<uint32_t>((first + lead + q * VB) * 4 + j);
                        }
                    }
                }
            }
            // Published at once: the next rank's walk of this claimant waits for it.
            uint2 const record = make_uint2(cand, __float_as_uint(running));
            for (int i = 0; i < p.numPublish; ++i)
            {
                *vpCand(p.publish[i], p, p.rank, row, c) = record;
            }
        }
        __syncthreads();
    }
    if (tid >= walkers && tid < kVpClaimants)
    {
        for (int i = 0; i < p.numPublish; ++i)
        {
            *vpCand(p.publish[i], p, p.rank, row, tid) = make_uint2(kVpNone, 0u);
        }
    }
    if (tid == 0)
    {
        info[0] = claimants;
        info[1] = shared.argmax;
        for (int c = 0; c < kVpClaimants; ++c)
        {
            info[2 + c] = c < walkers ? sClaimTid[c] : -1;
        }
    }
    gridDependencyTrigger();
}

//! vp_decide: one warp per row. PDL: waits on vp_draw before reading anything.
__global__ void __launch_bounds__(32) vpDecideKernel(VpParams p, int vb)
{
    gridDependencyWait();
    int const row = blockIdx.x;
    int const lane = threadIdx.x;
    uint32_t mine = kVpNone;
    if (lane < p.ranks * kVpClaimants)
    {
        uint2* const slot = vpCand(p.copy, p, lane / kVpClaimants, row, lane % kVpClaimants);
        mine = vpAwait<2>(slot, p.status, p.timeoutNs).x;
        storeRelaxedSys(slot, make_uint2(kSplitSentinel, kSplitSentinel));
    }
    uint32_t words[kVpMaxRanks * kVpClaimants];
#pragma unroll
    for (int i = 0; i < kVpMaxRanks * kVpClaimants; ++i)
    {
        words[i] = __shfl_sync(0xffffffffu, mine, i);
    }
    if (lane == 0)
    {
        int const* info = p.info + static_cast<size_t>(row) * kVpInfoWords;
        int const claimants = info[0];
        if (claimants >= 0)
        {
            int token = info[1];
            unsigned long long bestKey = 0ull;
            int const walkers = min(claimants, kVpClaimants);
            for (int c = 0; c < walkers; ++c)
            {
                // The first rank in rank order with a crossing; later ranks hold kVpUpstream.
                uint32_t cand = kVpNone;
                for (int r = 0; r < p.ranks && cand == kVpNone; ++r)
                {
                    cand = words[r * kVpClaimants + c];
                }
                if (cand == kVpNone || cand == kVpUpstream)
                {
                    continue;
                }
                int const walker = info[2 + c];
                int const v = static_cast<int>(cand) / 4;
                int const position = ((v - walker) / vb) * 4 + static_cast<int>(cand) % 4 + 1;
                unsigned long long const key
                    = (static_cast<unsigned long long>(position) << 32) | static_cast<unsigned long long>(walker + 1);
                if (walkers == 1 || key > bestKey)
                {
                    bestKey = key;
                    token = static_cast<int>(cand);
                }
            }
            p.fs.outputTokens[row] = token;
        }
    }
    gridDependencyTrigger();
}

//! vp_argmax: one warp per row; the greedy token as int64. PDL: waits on vp_fold first.
__global__ void __launch_bounds__(32) vpArgmaxKernel(VpParams p, int slots)
{
    gridDependencyWait();
    int const row = blockIdx.x;
    int const lane = threadIdx.x;
    VpMaxArg best{-INFINITY, INT_MAX};
    for (int i = lane; i < p.ranks * slots; i += 32)
    {
        uint2* const slot = vpGreedy(p.copy, p, i / slots, row, i % slots);
        uint4 const w = vpAwait<2>(slot, p.status, p.timeoutNs);
        storeRelaxedSys(slot, make_uint2(kSplitSentinel, kSplitSentinel));
        VpMaxArg const cand{__uint_as_float(w.x), static_cast<int>(w.y)};
        best = VpMaxArgOp()(best, cand);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        VpMaxArg const other{
            __shfl_xor_sync(0xffffffffu, best.value, offset), __shfl_xor_sync(0xffffffffu, best.index, offset)};
        best = VpMaxArgOp()(best, other);
    }
    if (lane == 0)
    {
        p.greedyTokens[row] = static_cast<int64_t>(best.index);
    }
    gridDependencyTrigger();
}

// ----------------------------------------------------------------------------------------
// The distributed verify sampler (ds_sample; head_sampler_coop_fold DIST_MIN_ROWS): the split sampler's tokens
// from the ranks' own LM-head shards, with no logits all-gather. One kernel, one 1024-thread CTA per row on every
// rank.
//
// Kernel A's virtual thread vt folds the global vectors v = vt + 1024 j. Rank r takes the virtual threads
// vt = L r + a (a < L = 1024 / world), whose vectors v = 1024 j + L r + a (shared-memory slot i = L j + a) are
// 1 / world of the row in runs of L contiguous vectors; it pulls exactly those from the ranks' staged GEMM outputs
// (vector v lives on rank v / shardVecs of the contiguous column-parallel shards), keeps them in shared memory as
// kernel A's scaled logits
// x = __fmul_rn(capOf(logit), tempInv), and folds each of its virtual threads with kernel A's statements in kernel
// A's order: every state is kernel A's bit for bit. The ranks exchange the states; every rank then runs kernel B's
// block reduce, rebase, scan and Philox draw on all 1024 states (thread t holds virtual thread t, so the cub
// reductions are kernel B's), the claimant's owner walks it from shared memory and publishes the candidate, and
// every rank takes it. A top-p-only row then runs the shipped rejection loop (splitTopPDecideKernel's form): each
// rank sums its virtual threads' CountMass of the weights above the candidate's (the shipped sweep's statements,
// skipped where splitTopPSkips proves the sum is {0, 0}), the ranks exchange them, every rank reduces the 1024
// partials with the shipped cub::BlockReduce and keeps the candidate iff the mass above is at most the mass target;
// a rejected round draws from the partials' masses (the shipped next sweep's sums). Every rank decides alike, so
// every rank writes the same token. A row with top-k or min-p gathers its whole row from the staged shards into
// this rank's pending row and runs the shipped body (kernel B's filtered path on the same bits).
//
// Transport: the s2 sampler's LL elements ((flag << 32) | word, written as 16-byte stores through every publish
// target -- the multicast address in service -- and polled in this rank's copy with relaxed system-scope loads
// until both elements carry the expected flag). flag = (epoch << 5) | round. The staged shards are plain fabric
// memory read by the peers with relaxed system-scope loads after a ready flag (vp_draw's staging protocol:
// stores, __threadfence_system, flag; poll, __threadfence_system, loads).
// - Writers: ready (callParity, row, r), state (callParity, row, r, a), candidate (callParity, roundParity, row, r)
//   and partial (callParity, roundParity, row, r, a) are written only by rank r's CTA `row`, once per (call,
//   round); staging (callParity, row) of rank r only by that CTA. Readers: rank R's CTA `row` reads only rank R's
//   copy (and every rank's staging of its row).
// - Epochs: CTA `row` keeps counters[row] (this rank's device memory, only that CTA's): a call uses counter + 1
//   (counter + 3 when that value's low 27 bits are 0, since the areas start zeroed: the parity still alternates)
//   and stores it back at its end. Every rank makes the same calls with the same rows, so the CTAs of a row agree
//   on every call's epoch, and consecutive calls of a row always have opposite parities.
// - Reuse across calls: call e uses parity e & 1 everywhere. A rank writes a parity-(e & 1) area or staging again
//   only in its call e + 2 of that row, after its call-(e + 1) CTA polled every rank's call-(e + 1) ready flag,
//   which each rank publishes in its call-(e + 1) kernel, i.e. after its call-e kernel completed (stream order: the
//   kernel waits for its predecessor before its first access, so its predecessors have completed transitively).
// - Reuse across rounds of a call: round k uses round parity k & 1. A rank publishes round k + 2 only after it
//   polled every rank's round-(k + 1) partials, which each rank publishes after its reads of round k.
// - Residency: a CTA waits only for the same row's CTAs on the other ranks, so every rank's grid must be resident
//   at once (rows <= the SM count at one CTA per SM; the emulation runs every emulated rank's CTAs in one grid).
//
// The flag has 27 epoch bits: a stale element matches only 2^27 calls of its row later.

//! Most ranks the distributed sampler takes.
constexpr int kDsMaxRanks = 8;
//! Its block: one thread per virtual thread of kernel A's 1024-thread block.
constexpr int kDsThreads = kSplitBlock;
//! Bytes of a state (two LL pairs: {max, sum}, {argmax, 0}), a pair (ready flag, partial) and a candidate entry
//! (two LL pairs: {index, weight}, {key, 0}).
constexpr unsigned long long kDsStateBytes = 32;
constexpr unsigned long long kDsPairBytes = 16;
constexpr unsigned long long kDsSlotBytes = 32;
//! The round in the flag's low bits.
constexpr int kDsRoundBits = 5;
static_assert(kMaxRejectRounds <= (1 << kDsRoundBits), "the round fits the flag");
//! Staged vectors one thread loads before using them.
constexpr int kDsBatch = 8;
//! Polls a thread spins before it backs off.
constexpr int kDsSpins = 64;
//! globaltimer stamps per (rank here, row) of the instrumented build (tests / benches): entry, row staged,
//! ready polled, row pulled, folded + published, states reduced, first draw's candidate, end.
constexpr int kDsStamps = 8;

struct DsParams
{
    //! Row params, seed / offset, the vocabulary; outputTokens [ranksHere][numRows]; logits the pending
    //! [ranksHere][numRows][vocab] BF16 rows (written for top-k / min-p rows only).
    FusedSamplingParams fs;
    //! [ranksHere][numRows][shardVecs]: this process's ranks' LM-head GEMM outputs.
    uint2 const* local{nullptr};
    //! Every rank's staging buffer, [2][maxRows][shardVecs] vectors (peer addresses; the emulation's tensors).
    uint2* staging[kDsMaxRanks]{};
    //! LL area bases the values are stored through (the multicast address; every emulated rank's copy).
    unsigned long long publish[kDsMaxRanks]{};
    int numPublish{0};
    //! The reading rank's LL area copy, by rank here (in service [0]).
    unsigned long long copy[kDsMaxRanks]{};
    //! [ranksHere][maxRows] epoch counters.
    uint32_t* counters{nullptr};
    int* status{nullptr};
    SplitCap cap{};
    int rank{0};
    int world{1};
    //! 1: every emulated rank's CTAs in one grid (blockIdx.y is the rank).
    int emulated{0};
    int shardVecs{0};
    int maxRows{0};
    unsigned long long readyOffset{0};
    unsigned long long statesOffset{0};
    unsigned long long candOffset{0};
    unsigned long long partialOffset{0};
    unsigned long long timeoutNs{0};
    //! The instrumented build only: [ranksHere][maxRows][kDsStamps] globaltimer values.
    unsigned long long* stamps{nullptr};
};

__device__ __forceinline__ uint32_t dsFlag(uint32_t epoch, int round)
{
    return (epoch << kDsRoundBits) | static_cast<uint32_t>(round);
}

__device__ __forceinline__ void dsStorePair(unsigned long long address, unsigned long long a, unsigned long long b)
{
    asm volatile("st.global.v2.b64 [%0], {%1, %2};" ::"l"(address), "l"(a), "l"(b) : "memory");
}

__device__ __forceinline__ void dsLoadPair(unsigned long long address, unsigned long long& a, unsigned long long& b)
{
    asm volatile("ld.relaxed.sys.global.v2.b64 {%0, %1}, [%2];" : "=l"(a), "=l"(b) : "l"(address) : "memory");
}

//! One LL element (the fabric variant's third state word).
__device__ __forceinline__ void dsStoreOne(unsigned long long address, unsigned long long a)
{
    asm volatile("st.global.b64 [%0], %1;" ::"l"(address), "l"(a) : "memory");
}

__device__ __forceinline__ unsigned long long dsLoadOne(unsigned long long address)
{
    unsigned long long a;
    asm volatile("ld.relaxed.sys.global.b64 %0, [%1];" : "=l"(a) : "l"(address) : "memory");
    return a;
}

__device__ __forceinline__ bool dsFresh(unsigned long long a, unsigned long long b, uint32_t flag)
{
    return static_cast<uint32_t>(a >> 32) == flag && static_cast<uint32_t>(b >> 32) == flag;
}

//! Records a poll that waited past the timeout and traps (a peer that never published).
__device__ __noinline__ void dsTimedOut(int* status)
{
    atomicOr(status, kSplitPollTimeout);
    __threadfence_system();
    __trap();
}

//! Spin until both elements of the pair at `address` carry `flag`.
__device__ __forceinline__ void dsAwaitPair(unsigned long long address, uint32_t flag, int* status,
    unsigned long long timeoutNs, unsigned long long& a, unsigned long long& b)
{
    dsLoadPair(address, a, b);
    if (dsFresh(a, b, flag))
    {
        return;
    }
    unsigned long long const start = globalTimerNs();
    for (int spins = 0;; ++spins)
    {
        if (spins >= kDsSpins)
        {
            __nanosleep(32);
        }
        if (globalTimerNs() - start > timeoutNs)
        {
            dsTimedOut(status);
        }
        dsLoadPair(address, a, b);
        if (dsFresh(a, b, flag))
        {
            return;
        }
    }
}

//! Spin until the element at `address` carries `flag`.
__device__ __forceinline__ unsigned long long dsAwaitOne(
    unsigned long long address, uint32_t flag, int* status, unsigned long long timeoutNs)
{
    unsigned long long a = dsLoadOne(address);
    if (static_cast<uint32_t>(a >> 32) == flag)
    {
        return a;
    }
    unsigned long long const start = globalTimerNs();
    for (int spins = 0;; ++spins)
    {
        if (spins >= kDsSpins)
        {
            __nanosleep(32);
        }
        if (globalTimerNs() - start > timeoutNs)
        {
            dsTimedOut(status);
        }
        a = dsLoadOne(address);
        if (static_cast<uint32_t>(a >> 32) == flag)
        {
            return a;
        }
    }
}

__device__ __forceinline__ void dsPublishOne(DsParams const& p, unsigned long long offset, unsigned long long a)
{
    for (int i = 0; i < p.numPublish; ++i)
    {
        dsStoreOne(p.publish[i] + offset, a);
    }
}

//! Store a pair through every publish target.
__device__ __forceinline__ void dsPublish(
    DsParams const& p, unsigned long long offset, unsigned long long a, unsigned long long b)
{
    for (int i = 0; i < p.numPublish; ++i)
    {
        dsStorePair(p.publish[i] + offset, a, b);
    }
}

__device__ __forceinline__ unsigned long long dsTag(uint32_t flag)
{
    return static_cast<unsigned long long>(flag) << 32;
}

//! The distributed sampler: one CTA per row (x the emulated ranks). Dynamic shared memory: L float4 per
//! 1024-vector stripe of the row (this rank's slots).
//! PDL: waits on its predecessor (the LM-head GEMM) before its first global access; with `kTrigger`
//! (``TargetHead.verify_pdl``) releases its dependent right after.
//! kFabric (DIST_FABRIC): the staging is LL -- each rank writes vector v of its row as the pair {flag | word0,
//! flag | word1} (16 bytes, call parity as before) and the peers' pulls check the flags as they load, so there is no
//! system fence, no ready flag and no wait for one (the ready phase was 13 / 27 us at 16 / 128 rows on 8 ranks);
//! a state travels as three LL elements (24 bytes: the {max, sum} pair + {argmax}) instead of four. The same values
//! in the same places, so the same tokens. Staging holds 16 bytes per vector (``DistLayout(ll_staging=True)``).
template <bool kTrigger, bool kStamp = false, bool kFabric = false>
__global__ void __launch_bounds__(kDsThreads, 1) dsSampleKernel(DsParams const p)
{
    extern __shared__ float4 sx[];
    using BlockReduceOnline = cub::BlockReduce<OnlineSoftmax, kDsThreads>;
    using BlockReduceCountMass = cub::BlockReduce<CountMass, kDsThreads>;
    using BlockScanF = cub::BlockScan<float, kDsThreads>;
    __shared__ FusedSamplingShared<kDsThreads> shared;
    __shared__ __align__(16) float wbuf[4 * kSplitWalkVectors];
    __shared__ uint32_t sEpoch;
    __shared__ int sWalker;
    __shared__ float sBase;
    __shared__ uint32_t sOwners;
    __shared__ unsigned long long sLastCrossing;
    __shared__ int sCrossIdx;
    __shared__ float sCrossWeight;
    __shared__ int sCand;
    __shared__ float sCandWeight;
    __shared__ int sKeepToken;

    int const row = blockIdx.x;
    int const tid = threadIdx.x;
    int const rank = p.emulated ? static_cast<int>(blockIdx.y) : p.rank;
    int const here = p.emulated ? rank : 0;
    int const world = p.world;
    int const localThreads = kDsThreads / world;
    int const sv = p.shardVecs;
    int const vecCount = p.fs.vocabSize / 4;
    auto stamp = [&](int k)
    {
        if constexpr (kStamp)
        {
            if (tid == 0)
            {
                p.stamps[(static_cast<size_t>(here) * p.maxRows + row) * kDsStamps + k] = globalTimerNs();
            }
        }
    };
    if (tid == 0)
    {
        sSplitCap = p.cap;
    }
    gridDependencyWait();
    if constexpr (kTrigger)
    {
        gridDependencyTrigger();
    }
    stamp(0);
    uint32_t* const counter = p.counters + static_cast<size_t>(here) * p.maxRows + row;
    if (tid == 0)
    {
        uint32_t epoch = *counter + 1u;
        if ((epoch & 0x07FFFFFFu) == 0u)
        {
            epoch += 2u; // never flag 0 (the zeroed areas); the parity keeps alternating
        }
        sEpoch = epoch;
    }
    __syncthreads();
    uint32_t const epoch = sEpoch;
    unsigned long long const callParity = epoch & 1u;
    unsigned long long const copy = p.copy[here];
    FusedSamplingParams fs = p.fs;
    fs.outputTokens = p.fs.outputTokens + static_cast<size_t>(here) * p.fs.numRows;
    fs.logits = static_cast<__nv_bfloat16 const*>(p.fs.logits)
        + static_cast<size_t>(here) * p.fs.numRows * static_cast<size_t>(p.fs.vocabSize);
    RowParams const rp = loadRowParams(fs, row);
    uint32_t const flag0 = dsFlag(epoch, 0);
    size_t const stagedRow = (static_cast<size_t>(callParity) * p.maxRows + row) * static_cast<size_t>(sv);

    // ---- 1. Stage this rank's row of the GEMM output and publish its ready flag; poll every rank's.
    {
        uint2 const* const localRow = p.local + (static_cast<size_t>(here) * p.fs.numRows + row) * sv;
        if constexpr (kFabric)
        {
            unsigned long long const mineLL = reinterpret_cast<unsigned long long>(p.staging[rank])
                + static_cast<unsigned long long>(stagedRow) * 16u;
            unsigned long long const tag = dsTag(flag0);
            for (int v = tid; v < sv; v += kDsThreads)
            {
                uint2 const w = __ldg(localRow + v);
                dsStorePair(mineLL + static_cast<unsigned long long>(v) * 16u, tag | w.x, tag | w.y);
            }
        }
        else
        {
            uint2* const mine = p.staging[rank] + stagedRow;
            for (int v = tid; v < sv; v += kDsThreads)
            {
                mine[v] = __ldg(localRow + v);
            }
        }
    }
    __syncthreads();
    stamp(1);
    if constexpr (!kFabric)
    {
        // The CTA's staging stores, then (one thread) a system fence and the flag: the stores reach the peers first.
        unsigned long long const readyRow
            = p.readyOffset + (callParity * p.maxRows + row) * static_cast<unsigned long long>(world) * kDsPairBytes;
        if (tid == 0)
        {
            __threadfence_system();
            dsPublish(p, readyRow + static_cast<unsigned long long>(rank) * kDsPairBytes, dsTag(flag0) | 1u,
                dsTag(flag0) | 1u);
        }
        if (tid < world)
        {
            unsigned long long a, b;
            dsAwaitPair(copy + readyRow + static_cast<unsigned long long>(tid) * kDsPairBytes, flag0, p.status,
                p.timeoutNs, a, b);
            __threadfence_system();
        }
        __syncthreads();
    }
    stamp(2);
    // kFabric: the LL staging pair of global vector v (on rank v / shardVecs).
    auto llAt = [&](int v)
    {
        int const q = v / sv;
        return reinterpret_cast<unsigned long long>(p.staging[q])
            + (static_cast<unsigned long long>(stagedRow) + static_cast<unsigned long long>(v - q * sv)) * 16u;
    };

    bool const filtered = rp.needTopK || rp.needTopP || rp.needMinP;
    if (filtered && !splitTopPOnly(rp))
    {
        // ---- The exact fallback: the row in vocabulary order into this rank's pending row, the shipped body.
        uint2* const dst = reinterpret_cast<uint2*>(const_cast<void*>(fs.logits)) + static_cast<size_t>(row) * vecCount;
        if constexpr (kFabric)
        {
            for (int v0 = tid; v0 < vecCount; v0 += kDsThreads * kDsBatch)
            {
                unsigned long long a[kDsBatch], b[kDsBatch];
#pragma unroll
                for (int u = 0; u < kDsBatch; ++u)
                {
                    int const v = v0 + u * kDsThreads;
                    if (v < vecCount)
                    {
                        dsLoadPair(llAt(v), a[u], b[u]);
                    }
                }
#pragma unroll
                for (int u = 0; u < kDsBatch; ++u)
                {
                    int const v = v0 + u * kDsThreads;
                    if (v < vecCount)
                    {
                        if (!dsFresh(a[u], b[u], flag0))
                        {
                            dsAwaitPair(llAt(v), flag0, p.status, p.timeoutNs, a[u], b[u]);
                        }
                        dst[v] = make_uint2(static_cast<uint32_t>(a[u]), static_cast<uint32_t>(b[u]));
                    }
                }
            }
        }
        else
            for (int v0 = tid; v0 < vecCount; v0 += kDsThreads * kDsBatch)
            {
                uint2 buf[kDsBatch];
#pragma unroll
                for (int u = 0; u < kDsBatch; ++u)
                {
                    int const v = v0 + u * kDsThreads;
                    if (v < vecCount)
                    {
                        int const q = v / sv;
                        buf[u] = loadRelaxedSys(p.staging[q] + stagedRow + (v - q * sv));
                    }
                }
#pragma unroll
                for (int u = 0; u < kDsBatch; ++u)
                {
                    int const v = v0 + u * kDsThreads;
                    if (v < vecCount)
                    {
                        dst[v] = buf[u];
                    }
                }
            }
        __syncthreads();
        fusedSamplingBody<CapBf16, kDsThreads, true, false>(fs, shared);
        __syncthreads();
        stamp(7);
        if (tid == 0)
        {
            *counter = epoch;
        }
        return;
    }

    // ---- 2. This rank's vectors of the row (slot i = L j + a: global vector v = 1024 j + L rank + a, runs of L
    //         contiguous vectors) as kernel A's scaled logits.
    int const slots = (vecCount + kDsThreads - 1) / kDsThreads * localThreads;
    auto slotVector = [&](int i) { return (i / localThreads) * kDsThreads + localThreads * rank + i % localThreads; };
    auto capped = [&](uint32_t lo, uint32_t hi)
    {
        uint2 const packed = make_uint2(lo, hi);
        __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(&packed);
        float4 x;
        x.x = __fmul_rn(capOf(__bfloat162float(elems[0]), p.cap), rp.tempInv);
        x.y = __fmul_rn(capOf(__bfloat162float(elems[1]), p.cap), rp.tempInv);
        x.z = __fmul_rn(capOf(__bfloat162float(elems[2]), p.cap), rp.tempInv);
        x.w = __fmul_rn(capOf(__bfloat162float(elems[3]), p.cap), rp.tempInv);
        return x;
    };
    if constexpr (kFabric)
    {
        for (int i0 = tid; i0 < slots; i0 += kDsThreads * kDsBatch)
        {
            unsigned long long a[kDsBatch], b[kDsBatch];
#pragma unroll
            for (int u = 0; u < kDsBatch; ++u)
            {
                int const i = i0 + u * kDsThreads;
                int const v = slotVector(i);
                if (i < slots && v < vecCount)
                {
                    dsLoadPair(llAt(v), a[u], b[u]);
                }
            }
#pragma unroll
            for (int u = 0; u < kDsBatch; ++u)
            {
                int const i = i0 + u * kDsThreads;
                int const v = slotVector(i);
                if (i < slots && v < vecCount)
                {
                    if (!dsFresh(a[u], b[u], flag0))
                    {
                        dsAwaitPair(llAt(v), flag0, p.status, p.timeoutNs, a[u], b[u]);
                    }
                    sx[i] = capped(static_cast<uint32_t>(a[u]), static_cast<uint32_t>(b[u]));
                }
            }
        }
    }
    else
    {
        for (int i0 = tid; i0 < slots; i0 += kDsThreads * kDsBatch)
        {
            uint2 buf[kDsBatch];
#pragma unroll
            for (int u = 0; u < kDsBatch; ++u)
            {
                int const i = i0 + u * kDsThreads;
                int const v = slotVector(i);
                if (i < slots && v < vecCount)
                {
                    int const q = v / sv;
                    buf[u] = loadRelaxedSys(p.staging[q] + stagedRow + (v - q * sv));
                }
            }
#pragma unroll
            for (int u = 0; u < kDsBatch; ++u)
            {
                int const i = i0 + u * kDsThreads;
                if (i < slots && slotVector(i) < vecCount)
                {
                    __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(&buf[u]);
                    float4 x;
                    x.x = __fmul_rn(capOf(__bfloat162float(elems[0]), p.cap), rp.tempInv);
                    x.y = __fmul_rn(capOf(__bfloat162float(elems[1]), p.cap), rp.tempInv);
                    x.z = __fmul_rn(capOf(__bfloat162float(elems[2]), p.cap), rp.tempInv);
                    x.w = __fmul_rn(capOf(__bfloat162float(elems[3]), p.cap), rp.tempInv);
                    sx[i] = x;
                }
            }
        }
    }
    __syncthreads();
    stamp(3);

    // ---- 3. Fold this rank's virtual threads (vt = L rank + tid: slots i = tid + L j, vectors vt + 1024 j, in
    //         increasing j) with kernel A's statements, publish the states, poll every rank's.
    OnlineSoftmax own{-FLT_MAX, 0.0f, 0};
    unsigned long long const statesRow = p.statesOffset + (callParity * p.maxRows + row) * kDsThreads * kDsStateBytes;
    if (tid < localThreads)
    {
        // combineOnlineSoftmax(st, {x, 1, index}) element by element, in splitFoldCoopKernel's / the s2 fold's
        // FORM 1 statements (its two branches as selects; the branch weight __expf off the sum's chain): the same
        // operations on the same values in the same order as kernel A, so the same state bit for bit.
        int const vt = localThreads * rank + tid;
        float runMax = -FLT_MAX;
        float runSum = 0.0f;
        int runArg = 0;
        for (int j = 0; vt + j * kDsThreads < vecCount; ++j)
        {
            float4 const x = sx[tid + localThreads * j];
            float const xs[4] = {x.x, x.y, x.z, x.w};
            int const base = (vt + j * kDsThreads) * 4;
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                bool const keep = runMax >= xs[e];
                float const w = keep ? __expf(xs[e] - runMax) : __expf(runMax - xs[e]);
                runArg = keep ? runArg : base + e;
                runMax = keep ? runMax : xs[e];
                runSum = keep ? runSum + 1.0f * w : 1.0f + runSum * w;
            }
        }
        own = OnlineSoftmax{runMax, runSum, runArg};
        unsigned long long const at
            = statesRow + (static_cast<unsigned long long>(rank) * localThreads + tid) * kDsStateBytes;
        dsPublish(p, at, dsTag(flag0) | __float_as_uint(own.max), dsTag(flag0) | __float_as_uint(own.sum));
        if constexpr (kFabric)
        {
            dsPublishOne(p, at + 16u, dsTag(flag0) | static_cast<uint32_t>(own.argmax));
        }
        else
        {
            dsPublish(p, at + 16u, dsTag(flag0) | static_cast<uint32_t>(own.argmax), dsTag(flag0));
        }
    }
    stamp(4);
    OnlineSoftmax cachedThreadStats;
    {
        // Virtual thread tid: rank tid / L's slot tid % L, i.e. entry tid.
        unsigned long long const at = copy + statesRow + static_cast<unsigned long long>(tid) * kDsStateBytes;
        unsigned long long e0, e1, e2;
        dsAwaitPair(at, flag0, p.status, p.timeoutNs, e0, e1);
        if constexpr (kFabric)
        {
            e2 = dsAwaitOne(at + 16u, flag0, p.status, p.timeoutNs);
        }
        else
        {
            unsigned long long e3;
            dsAwaitPair(at + 16u, flag0, p.status, p.timeoutNs, e2, e3);
        }
        cachedThreadStats = OnlineSoftmax{__uint_as_float(static_cast<uint32_t>(e0)),
            __uint_as_float(static_cast<uint32_t>(e1)), static_cast<int>(static_cast<uint32_t>(e2))};
    }

    // ---- 4. Kernel B's first round on every rank (block reduce, rebase, Philox), then the draws.
    OnlineSoftmax const rowStats
        = BlockReduceOnline(shared.temp.reduceOnline).Reduce(cachedThreadStats, OnlineSoftmaxOp());
    if (tid == 0)
    {
        shared.maxScaled = rowStats.max;
        shared.argmax = rowStats.argmax;
        shared.totalMass = rowStats.sum;
    }
    __syncthreads();
    stamp(5);
    float const maxScaled = shared.maxScaled;
    int const argmax = shared.argmax;
    float const cachedLocalMass = cachedThreadStats.sum * __expf(cachedThreadStats.max - maxScaled);
    if (tid == 0)
    {
        initRowRng(fs, row, &shared.rejectRng);
    }
    __syncthreads();

    // One draw of the rejection loop at `round` from every thread's support mass (w >= 0 && w > pivot): kernel
    // B's scan, tid 0's Philox uniform (round 0 also fixes the mass target), the claimants' walk on the ranks that
    // hold them (kernel B's statements on the shared-memory copy), the candidates through the slot area, kernel B's
    // rule across several claimants. Leaves sCand / sCandWeight (-1: the target fell past the support).
    auto drawRound = [&](float localMass, float pivot, int round)
    {
        float base = 0.0f;
        float supportMass = 0.0f;
        BlockScanF(shared.temp.scanF).ExclusiveSum(localMass, base, supportMass);
        __syncthreads();
        if (tid == 0)
        {
            if (round == 0)
            {
                shared.massTarget = rp.topP * supportMass;
            }
            shared.target = curand_uniform(&shared.rejectRng) * supportMass;
            sWalker = -1;
            sOwners = 0u;
            sLastCrossing = 0ull;
            sCrossIdx = -1;
            sCrossWeight = 0.0f;
            sCand = -1;
            sCandWeight = 0.0f;
        }
        __syncthreads();
        float const target = shared.target;
        bool const claims = target >= base && target < base + localMass;
        if (claims)
        {
            sWalker = tid;
            sBase = base;
            atomicOr(&sOwners, 1u << (tid / localThreads));
        }
        int const claimants = __syncthreads_count(claims);
        uint32_t const flag = dsFlag(epoch, round);
        unsigned long long const tag = dsTag(flag);
        unsigned long long const slotRow = p.candOffset
            + ((callParity * 2u + static_cast<unsigned long long>(round & 1)) * p.maxRows + row)
                * static_cast<unsigned long long>(world) * kDsSlotBytes;
        if (claimants == 1)
        {
            int const walker = sWalker;
            int const owner = walker / localThreads;
            if (owner == rank)
            {
                // Kernel B's single-walker walk: the weights in parallel, summed by one thread in order.
                int const a = walker % localThreads;
                int const walkVectors = (vecCount - walker + kDsThreads - 1) / kDsThreads;
                if (tid < walkVectors)
                {
                    float4 const x = sx[a + localThreads * tid];
                    wbuf[tid * 4 + 0] = __expf(x.x - maxScaled);
                    wbuf[tid * 4 + 1] = __expf(x.y - maxScaled);
                    wbuf[tid * 4 + 2] = __expf(x.z - maxScaled);
                    wbuf[tid * 4 + 3] = __expf(x.w - maxScaled);
                }
                __syncthreads();
                if (tid == 0)
                {
                    float running = sBase;
                    int cand = -1;
                    float candWeight = 0.0f;
                    float4 const* w4 = reinterpret_cast<float4 const*>(wbuf);
                    for (int q = 0; q < walkVectors && cand < 0; ++q)
                    {
                        float4 const quad = w4[q];
                        float const ws[4] = {quad.x, quad.y, quad.z, quad.w};
#pragma unroll
                        for (int j = 0; j < 4; ++j)
                        {
                            float const w = ws[j];
                            if (cand < 0 && w >= 0.0f && w > pivot)
                            {
                                running += w;
                                if (running > target)
                                {
                                    cand = (walker + q * kDsThreads) * 4 + j;
                                    candWeight = w;
                                }
                            }
                        }
                    }
                    unsigned long long const at = slotRow + static_cast<unsigned long long>(rank) * kDsSlotBytes;
                    dsPublish(p, at, tag | static_cast<uint32_t>(cand), tag | __float_as_uint(candWeight));
                    dsPublish(p, at + 16u, tag, tag);
                    sCand = cand;
                    sCandWeight = candWeight;
                }
            }
            else if (tid == 0)
            {
                unsigned long long a, b;
                dsAwaitPair(copy + slotRow + static_cast<unsigned long long>(owner) * kDsSlotBytes, flag, p.status,
                    p.timeoutNs, a, b);
                sCand = static_cast<int>(static_cast<uint32_t>(a));
                sCandWeight = __uint_as_float(static_cast<uint32_t>(b));
            }
        }
        else if (claimants > 1)
        {
            // Kernel B's several-claimants walk, each claimant on the rank that holds its vectors; each rank
            // publishes its latest crossing with key = position * 2048 + tid + 1 (the order of kernel B's
            // (position << 32) | (tid + 1)), every rank takes the largest key.
            int crossIdx = -1;
            float crossWeight = 0.0f;
            unsigned long long crossing = 0ull;
            if (claims && tid / localThreads == rank)
            {
                float running = base;
                int position = 0;
                int const a = tid % localThreads;
                for (int q = 0; tid + q * kDsThreads < vecCount && crossIdx < 0; ++q)
                {
                    float4 const x = sx[a + localThreads * q];
                    float const xs[4] = {x.x, x.y, x.z, x.w};
                    for (int j = 0; j < 4; ++j)
                    {
                        float const w = __expf(xs[j] - maxScaled);
                        ++position;
                        if (!(w >= 0.0f && w > pivot))
                        {
                            continue;
                        }
                        running += w;
                        if (running > target)
                        {
                            crossIdx = (tid + q * kDsThreads) * 4 + j;
                            crossWeight = w;
                            crossing = (static_cast<unsigned long long>(position) << 32)
                                | static_cast<unsigned long long>(tid + 1);
                            break;
                        }
                    }
                }
                if (crossIdx >= 0)
                {
                    atomicMax(&sLastCrossing, crossing);
                }
            }
            __syncthreads();
            if (crossIdx >= 0 && crossing == sLastCrossing)
            {
                sCrossIdx = crossIdx;
                sCrossWeight = crossWeight;
            }
            __syncthreads();
            if (tid == 0)
            {
                uint32_t const owners = sOwners;
                if ((owners >> rank) & 1u)
                {
                    unsigned long long const best = sLastCrossing;
                    uint32_t const key
                        = best == 0ull ? 0u : static_cast<uint32_t>((best >> 32) * 2048ull + (best & 0xFFFFFFFFull));
                    int const idx = best == 0ull ? -1 : sCrossIdx;
                    float const w = best == 0ull ? 0.0f : sCrossWeight;
                    unsigned long long const at = slotRow + static_cast<unsigned long long>(rank) * kDsSlotBytes;
                    dsPublish(p, at, tag | static_cast<uint32_t>(idx), tag | __float_as_uint(w));
                    dsPublish(p, at + 16u, tag | key, tag);
                }
                uint32_t bestKey = 0u;
                int token = -1;
                float tokenWeight = 0.0f;
                for (int r = 0; r < world; ++r)
                {
                    if (((owners >> r) & 1u) == 0u)
                    {
                        continue;
                    }
                    unsigned long long a, b, c, d;
                    unsigned long long const at = copy + slotRow + static_cast<unsigned long long>(r) * kDsSlotBytes;
                    dsAwaitPair(at, flag, p.status, p.timeoutNs, a, b);
                    dsAwaitPair(at + 16u, flag, p.status, p.timeoutNs, c, d);
                    int const idx = static_cast<int>(static_cast<uint32_t>(a));
                    uint32_t const key = static_cast<uint32_t>(c);
                    if (idx >= 0 && key > bestKey)
                    {
                        bestKey = key;
                        token = idx;
                        tokenWeight = __uint_as_float(static_cast<uint32_t>(b));
                    }
                }
                sCand = token;
                sCandWeight = tokenWeight;
            }
        }
        __syncthreads();
    };

    drawRound(cachedLocalMass, -1.0f, 0);
    stamp(6);
    int cand = sCand;
    float candWeight = sCandWeight;
    int token = cand >= 0 ? cand : argmax;
    if (rp.needTopP && cand >= 0)
    {
        // ---- 5. The shipped rejection loop's tests (top-p only), the sweep spread over the ranks.
        for (int round = 0;;)
        {
            uint32_t const flag = dsFlag(epoch, round);
            unsigned long long const partRow = p.partialOffset
                + ((callParity * 2u + static_cast<unsigned long long>(round & 1)) * p.maxRows + row) * kDsThreads
                    * kDsPairBytes;
            if (tid < localThreads)
            {
                CountMass mine{0.0f, 0.0f};
                if (!splitTopPSkips(own, maxScaled, candWeight))
                {
                    int const vt = localThreads * rank + tid;
                    for (int j = 0; vt + j * kDsThreads < vecCount; ++j)
                    {
                        float4 const x = sx[tid + localThreads * j];
                        float const xs[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
                        for (int j = 0; j < 4; ++j)
                        {
                            float const w = __expf(xs[j] - maxScaled);
                            if (w >= 0.0f && w > candWeight)
                            {
                                mine.count += 1.0f;
                                mine.mass += w;
                            }
                        }
                    }
                }
                dsPublish(p, partRow + (static_cast<unsigned long long>(rank) * localThreads + tid) * kDsPairBytes,
                    dsTag(flag) | __float_as_uint(mine.count), dsTag(flag) | __float_as_uint(mine.mass));
            }
            CountMass part;
            {
                unsigned long long a, b;
                dsAwaitPair(copy + partRow + static_cast<unsigned long long>(tid) * kDsPairBytes, flag, p.status,
                    p.timeoutNs, a, b);
                part = CountMass{__uint_as_float(static_cast<uint32_t>(a)), __uint_as_float(static_cast<uint32_t>(b))};
            }
            CountMass const above = BlockReduceCountMass(shared.temp.reduceCountMass).Reduce(part, CountMassOp());
            __syncthreads();
            if (tid == 0)
            {
                sKeepToken = above.mass <= shared.massTarget ? 1 : 0;
            }
            __syncthreads();
            if (sKeepToken)
            {
                token = cand;
                break;
            }
            // Rejected: the candidate and everything at or below its weight leave the support; the next
            // round's per-thread support mass is this test's partial mass (the shipped sweep's same sum).
            float const pivot = candWeight;
            if (++round >= kMaxRejectRounds)
            {
                token = argmax;
                break;
            }
            drawRound(part.mass, pivot, round);
            cand = sCand;
            candWeight = sCandWeight;
            if (cand < 0)
            {
                token = argmax;
                break;
            }
        }
    }
    if (tid == 0)
    {
        fs.outputTokens[row] = token;
    }
    __syncthreads();
    stamp(7);
    if (tid == 0)
    {
        *counter = epoch;
    }
}

} // namespace fusedSampling
} // namespace kernels
} // namespace tensorrt_llm

namespace
{

namespace fs = tensorrt_llm::kernels::fusedSampling;
namespace tk = tensorrt_llm::kernels;

template <typename Kernel, typename... Args>
void launch(Kernel kernel, dim3 grid, dim3 block, bool pdl, Args... args)
{
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    config.attrs = attribute;
    config.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, args...));
}

void checkRowVector(at::Tensor const& t, char const* name, at::ScalarType dtype, int64_t rows)
{
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == dtype, name, " has the wrong dtype");
    TORCH_CHECK(t.dim() == 1 && t.size(0) >= rows, name, " must be a 1D tensor of at least numRows entries");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

//! The shipped op's parameter checks (thop/fusedSamplingOp.cpp buildParams), on a BF16 row.
tk::FusedSamplingParams rowParams(at::Tensor const& logits, at::Tensor const& temperatures, at::Tensor const& topKs,
    at::Tensor const& topPs, at::Tensor const& minPs, std::optional<at::Tensor> const& seed,
    std::optional<at::Tensor> const& offset)
{
    TORCH_CHECK(logits.dim() == 2, "logits must be [numRows, vocabSize]");
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "logits must be a contiguous CUDA tensor");
    auto const numRows = logits.size(0);
    TORCH_CHECK(numRows >= 1 && numRows <= fs::kSplitMaxRows, "1..", fs::kSplitMaxRows, " rows");
    TORCH_CHECK(logits.size(1) % 4 == 0, "the vocabulary must be a whole number of 4-element vectors");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(logits.data_ptr()) % 8 == 0, "logits must be 8-byte aligned");
    checkRowVector(temperatures, "temperatures", at::kFloat, numRows);
    checkRowVector(topKs, "top_ks", at::kInt, numRows);
    checkRowVector(topPs, "top_ps", at::kFloat, numRows);
    checkRowVector(minPs, "min_ps", at::kFloat, numRows);
    tk::FusedSamplingParams params;
    params.logits = logits.data_ptr();
    params.temperatures = temperatures.data_ptr<float>();
    params.topKs = topKs.data_ptr<int32_t>();
    params.topPs = topPs.data_ptr<float>();
    params.minPs = minPs.data_ptr<float>();
    params.numRows = static_cast<int32_t>(numRows);
    params.vocabSize = static_cast<int32_t>(logits.size(1));
    TORCH_CHECK(seed.has_value() == offset.has_value(), "seed and offset must be provided together");
    if (seed.has_value())
    {
        auto const& s = seed.value();
        auto const& o = offset.value();
        TORCH_CHECK(s.is_cuda() && o.is_cuda(), "seed and offset must be CUDA tensors");
        TORCH_CHECK(s.scalar_type() == at::kLong && o.scalar_type() == at::kLong, "seed and offset must be int64");
        TORCH_CHECK(s.numel() == o.numel(), "seed and offset must have the same length");
        TORCH_CHECK(s.numel() == 1 || s.numel() >= numRows, "seed/offset must hold 1 or numRows entries");
        params.seed = reinterpret_cast<uint64_t const*>(s.data_ptr<int64_t>());
        params.offset = reinterpret_cast<uint64_t const*>(o.data_ptr<int64_t>());
        params.perRowRng = s.numel() >= numRows;
    }
    return params;
}

fs::SplitCap softCap(double multiplier, double invCap, double cap)
{
    // Python floats holding FP32 values: the conversions are exact.
    return fs::SplitCap{static_cast<float>(multiplier), static_cast<float>(invCap), static_cast<float>(cap)};
}

void checkStates(at::Tensor const& states, int64_t rows)
{
    TORCH_CHECK(states.is_cuda() && states.is_contiguous() && states.scalar_type() == at::kInt,
        "states must be a contiguous int32 CUDA tensor");
    TORCH_CHECK(rows >= 1 && rows <= fs::kSplitMaxRows, "1..", fs::kSplitMaxRows, " rows");
    TORCH_CHECK(states.numel() >= rows * fs::splitVirtualBlock(static_cast<int>(rows)) * 3,
        "states holds 3 words per virtual thread");
}

template <bool GATHER, bool FOLD, int VB>
void launchFoldBlock(fs::SplitFoldParams const& p, int64_t threads, bool pdl)
{
    if (threads == 128)
    {
        launch(fs::splitFoldKernel<GATHER, FOLD, 128, VB>, dim3(p.fs.numRows, VB / 128), dim3(128), pdl, p);
    }
    else
    {
        TORCH_CHECK(threads == 64, "kernel A runs 64 or 128 threads per CTA");
        launch(fs::splitFoldKernel<GATHER, FOLD, 64, VB>, dim3(p.fs.numRows, VB / 64), dim3(64), pdl, p);
    }
}

template <bool GATHER, bool FOLD>
void launchFold(fs::SplitFoldParams const& p, int64_t threads, bool pdl)
{
    if (fs::splitVirtualBlock(p.fs.numRows) == fs::kSplitBlock)
    {
        launchFoldBlock<GATHER, FOLD, fs::kSplitBlock>(p, threads, pdl);
    }
    else
    {
        launchFoldBlock<GATHER, FOLD, fs::kSplitNarrowBlock>(p, threads, pdl);
    }
}

//! Kernel A on a plain BF16 row (no gather).
void foldPlain(at::Tensor const& logits, at::Tensor const& temperatures, at::Tensor const& states, double multiplier,
    double invCap, double cap, int64_t threads, bool pdl)
{
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16, "logits must be BF16");
    at::cuda::CUDAGuard guard(logits.device());
    fs::SplitFoldParams p;
    TORCH_CHECK(logits.dim() == 2 && logits.is_contiguous() && logits.size(1) % 4 == 0, "logits [rows, vocab]");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(logits.data_ptr()) % 8 == 0, "logits must be 8-byte aligned");
    checkRowVector(temperatures, "temperatures", at::kFloat, logits.size(0));
    checkStates(states, logits.size(0));
    p.fs.temperatures = temperatures.data_ptr<float>();
    p.fs.numRows = static_cast<int32_t>(logits.size(0));
    p.fs.vocabSize = static_cast<int32_t>(logits.size(1));
    p.plain = reinterpret_cast<uint2 const*>(logits.data_ptr());
    p.cap = softCap(multiplier, invCap, cap);
    p.states = reinterpret_cast<fs::OnlineSoftmax*>(states.data_ptr());
    launchFold<false, true>(p, threads, pdl);
}

// ---- Cooperative kernel A launch (head_sampler_coop_fold) ---------------------------------
template <int VTS, int THREADS, bool EARLY>
void launchFoldCoopShape(fs::SplitFoldParams const& p, bool pdl)
{
    auto kernel = fs::splitFoldCoopKernel<VTS, THREADS, EARLY>;
    int const vecCount = p.fs.vocabSize / 4;
    size_t const smem = static_cast<size_t>((vecCount + fs::kSplitBlock - 1) / fs::kSplitBlock) * VTS * sizeof(float4);
    // Raised once per instantiation to the widest row this op admits (the walk's limit).
    static bool const raised = []
    {
        size_t const most = static_cast<size_t>(fs::kSplitWalkVectors) * VTS * sizeof(float4);
        C10_CUDA_CHECK(cudaFuncSetAttribute(fs::splitFoldCoopKernel<VTS, THREADS, EARLY>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(most)));
        return true;
    }();
    (void) raised;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(p.fs.numRows, fs::kSplitBlock / VTS);
    config.blockDim = dim3(THREADS);
    config.dynamicSmemBytes = smem;
    config.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    config.attrs = attribute;
    config.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, p));
}

template <bool EARLY>
void launchFoldCoop(fs::SplitFoldParams const& p, int64_t vts, int64_t threads, bool pdl)
{
#define FOLD_COOP_SHAPE(V, T)                                                                                          \
    if (vts == V && threads == T)                                                                                      \
    {                                                                                                                  \
        launchFoldCoopShape<V, T, EARLY>(p, pdl);                                                                      \
        return;                                                                                                        \
    }
    FOLD_COOP_SHAPE(128, 1024)
    FOLD_COOP_SHAPE(128, 512)
    FOLD_COOP_SHAPE(64, 1024)
    FOLD_COOP_SHAPE(64, 512)
    FOLD_COOP_SHAPE(64, 256)
    FOLD_COOP_SHAPE(32, 512)
    FOLD_COOP_SHAPE(32, 256)
#undef FOLD_COOP_SHAPE
    TORCH_CHECK(
        false, "cooperative kernel A: unsupported (virtual threads per CTA, threads) = (", vts, ", ", threads, ")");
}

//! Cooperative kernel A on a plain BF16 row: the states of foldPlain, bit for bit.
void foldPlainCoop(at::Tensor const& logits, at::Tensor const& temperatures, at::Tensor const& states,
    double multiplier, double invCap, double cap, int64_t vts, int64_t threads, bool earlyTrigger, bool pdl)
{
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16, "logits must be BF16");
    at::cuda::CUDAGuard guard(logits.device());
    fs::SplitFoldParams p;
    TORCH_CHECK(logits.dim() == 2 && logits.is_contiguous() && logits.size(1) % 4 == 0, "logits [rows, vocab]");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(logits.data_ptr()) % 8 == 0, "logits must be 8-byte aligned");
    TORCH_CHECK(logits.size(1) / 4 <= fs::kSplitWalkVectors * fs::kSplitBlock, "vocabulary too wide");
    // The cooperative kernel writes the states of the kSplitBlock virtual block; calls above
    // kWideBlockMaxRows rows take the narrow block (splitVirtualBlock) and run foldPlain.
    TORCH_CHECK(fs::splitVirtualBlock(static_cast<int>(logits.size(0))) == fs::kSplitBlock,
        "cooperative kernel A takes at most ", fs::kWideBlockMaxRows, " rows");
    checkRowVector(temperatures, "temperatures", at::kFloat, logits.size(0));
    checkStates(states, logits.size(0));
    p.fs.temperatures = temperatures.data_ptr<float>();
    p.fs.numRows = static_cast<int32_t>(logits.size(0));
    p.fs.vocabSize = static_cast<int32_t>(logits.size(1));
    p.plain = reinterpret_cast<uint2 const*>(logits.data_ptr());
    p.cap = softCap(multiplier, invCap, cap);
    p.states = reinterpret_cast<fs::OnlineSoftmax*>(states.data_ptr());
    if (earlyTrigger)
    {
        launchFoldCoop<true>(p, vts, threads, pdl);
    }
    else
    {
        launchFoldCoop<false>(p, vts, threads, pdl);
    }
}

//! Kernel A with the gather: poll this rank's inbox copy, write `gathered`, reset the
//! inbox; fold too when `fold`.
void foldGather(at::Tensor const& inbox, at::Tensor const& gathered, at::Tensor const& temperatures,
    at::Tensor const& states, at::Tensor const& status, double multiplier, double invCap, double cap, int64_t shardVecs,
    int64_t slotRows, int64_t timeoutS, bool fold, int64_t threads, bool pdl)
{
    TORCH_CHECK(gathered.scalar_type() == at::kBFloat16 && gathered.dim() == 2 && gathered.is_contiguous(),
        "gathered must be a contiguous BF16 [rows, vocab] tensor");
    auto const rows = gathered.size(0);
    auto const vocab = gathered.size(1);
    TORCH_CHECK(vocab % 4 == 0 && (vocab / 4) % shardVecs == 0, "the vocabulary is a whole number of shards");
    auto const ranks = (vocab / 4) / shardVecs;
    TORCH_CHECK(rows >= 1 && rows <= slotRows, "rows must fit the inbox slots");
    TORCH_CHECK(inbox.is_cuda() && inbox.is_contiguous() && inbox.device() == gathered.device(), "inbox on device");
    TORCH_CHECK(inbox.numel() * inbox.element_size() >= ranks * slotRows * shardVecs * 8, "inbox too small");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(inbox.data_ptr()) % 8 == 0, "inbox must be 8-byte aligned");
    TORCH_CHECK(status.is_cuda() && status.scalar_type() == at::kInt && status.numel() >= 1, "status int32[1]");
    at::cuda::CUDAGuard guard(gathered.device());
    fs::SplitFoldParams p;
    p.fs.numRows = static_cast<int32_t>(rows);
    p.fs.vocabSize = static_cast<int32_t>(vocab);
    if (fold)
    {
        checkRowVector(temperatures, "temperatures", at::kFloat, rows);
        checkStates(states, rows);
        p.fs.temperatures = temperatures.data_ptr<float>();
        p.states = reinterpret_cast<fs::OnlineSoftmax*>(states.data_ptr());
    }
    p.inbox = reinterpret_cast<uint2*>(inbox.data_ptr());
    p.gathered = reinterpret_cast<uint2*>(gathered.data_ptr());
    p.status = status.data_ptr<int32_t>();
    p.cap = softCap(multiplier, invCap, cap);
    p.shardVecs = static_cast<int>(shardVecs);
    p.slotRows = static_cast<int>(slotRows);
    p.timeoutNs = static_cast<unsigned long long>(timeoutS) * 1000000000ull;
    if (fold)
    {
        launchFold<true, true>(p, threads, pdl);
    }
    else
    {
        launchFold<true, false>(p, threads, pdl);
    }
}

//! Kernel B. `logits` is the BF16 row kernel A read (or wrote, with the gather).
at::Tensor draw(at::Tensor const& logits, at::Tensor const& temperatures, at::Tensor const& topKs,
    at::Tensor const& topPs, at::Tensor const& minPs, std::optional<at::Tensor> const& seed,
    std::optional<at::Tensor> const& offset, at::Tensor const& states, double multiplier, double invCap, double cap,
    bool pdl, bool earlyTrigger)
{
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16, "logits must be BF16");
    at::cuda::CUDAGuard guard(logits.device());
    auto params = rowParams(logits, temperatures, topKs, topPs, minPs, seed, offset);
    TORCH_CHECK(params.vocabSize / 4 <= fs::kSplitWalkVectors * fs::kSplitBlock
            && params.vocabSize / 4 <= fs::kSplitNarrowWalkVectors * fs::kSplitNarrowBlock,
        "vocabulary too wide for the walk");
    checkStates(states, logits.size(0));
    auto tokens = at::empty({logits.size(0)}, logits.options().dtype(at::kInt));
    params.outputTokens = tokens.data_ptr<int32_t>();
    auto const* threadStates = reinterpret_cast<fs::OnlineSoftmax const*>(states.data_ptr());
    fs::SplitTopPRecord* const noRecords = nullptr;
    if (fs::splitVirtualBlock(params.numRows) == fs::kSplitBlock)
    {
        if (earlyTrigger)
        {
            launch(fs::splitDrawKernel<fs::kSplitBlock, fs::kSplitWalkVectors, true>, dim3(params.numRows),
                dim3(fs::kSplitBlock), pdl, params, threadStates, softCap(multiplier, invCap, cap), noRecords);
        }
        else
        {
            launch(fs::splitDrawKernel<fs::kSplitBlock, fs::kSplitWalkVectors, false>, dim3(params.numRows),
                dim3(fs::kSplitBlock), pdl, params, threadStates, softCap(multiplier, invCap, cap), noRecords);
        }
    }
    else if (earlyTrigger)
    {
        launch(fs::splitDrawKernel<fs::kSplitNarrowBlock, fs::kSplitNarrowWalkVectors, true>, dim3(params.numRows),
            dim3(fs::kSplitNarrowBlock), pdl, params, threadStates, softCap(multiplier, invCap, cap), noRecords);
    }
    else
    {
        launch(fs::splitDrawKernel<fs::kSplitNarrowBlock, fs::kSplitNarrowWalkVectors, false>, dim3(params.numRows),
            dim3(fs::kSplitNarrowBlock), pdl, params, threadStates, softCap(multiplier, invCap, cap), noRecords);
    }
    return tokens;
}

//! Kernel B with the multi-CTA top-p test (head_sampler_coop_fold TOPP_ROUNDS), on calls of at
//! most kWideBlockMaxRows rows: kernel B on every row (a top-p-only row stops after its first
//! draw and leaves its record), then `rounds` pairs of the test's sweep (VB / 128 CTAs per row)
//! and decision (one CTA per row; the last one finishes every row). `records` holds
//! kTopPRecordBytes per row, `partials` 2 words per (row, virtual thread). The tokens are
//! draw()'s (bitwise: the same per-thread partials, reductions, Philox draws and walks).
at::Tensor drawTopP(at::Tensor const& logits, at::Tensor const& temperatures, at::Tensor const& topKs,
    at::Tensor const& topPs, at::Tensor const& minPs, std::optional<at::Tensor> const& seed,
    std::optional<at::Tensor> const& offset, at::Tensor const& states, at::Tensor const& records,
    at::Tensor const& partials, double multiplier, double invCap, double cap, int64_t rounds, bool pdl,
    bool earlyTrigger)
{
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16, "logits must be BF16");
    at::cuda::CUDAGuard guard(logits.device());
    auto params = rowParams(logits, temperatures, topKs, topPs, minPs, seed, offset);
    TORCH_CHECK(fs::splitVirtualBlock(params.numRows) == fs::kSplitBlock, "the multi-CTA top-p test takes at most ",
        fs::kWideBlockMaxRows, " rows");
    TORCH_CHECK(params.vocabSize / 4 <= fs::kSplitWalkVectors * fs::kSplitBlock, "vocabulary too wide for the walk");
    TORCH_CHECK(rounds >= 1 && rounds <= fs::kTopPMaxRounds, "1..", fs::kTopPMaxRounds, " test rounds");
    checkStates(states, logits.size(0));
    TORCH_CHECK(records.is_cuda() && records.is_contiguous() && records.scalar_type() == at::kInt
            && records.device() == logits.device(),
        "records must be a contiguous int32 tensor on the logits' device");
    TORCH_CHECK(records.numel() * 4 >= static_cast<int64_t>(params.numRows) * fs::kTopPRecordBytes,
        "records holds kTopPRecordBytes per row");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(records.data_ptr()) % fs::kTopPRecordBytes == 0,
        "records must be aligned to its 128-byte slots");
    TORCH_CHECK(partials.is_cuda() && partials.is_contiguous() && partials.scalar_type() == at::kInt
            && partials.device() == logits.device(),
        "partials must be a contiguous int32 tensor on the logits' device");
    TORCH_CHECK(partials.numel() >= static_cast<int64_t>(params.numRows) * fs::kSplitBlock * 2,
        "partials holds 2 words per (row, virtual thread)");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(partials.data_ptr()) % 8 == 0, "partials must be 8-byte aligned");
    auto tokens = at::empty({logits.size(0)}, logits.options().dtype(at::kInt));
    params.outputTokens = tokens.data_ptr<int32_t>();
    auto const* threadStates = reinterpret_cast<fs::OnlineSoftmax const*>(states.data_ptr());
    auto* const recs = reinterpret_cast<fs::SplitTopPRecord*>(records.data_ptr());
    auto* const parts = reinterpret_cast<fs::CountMass*>(partials.data_ptr());
    auto const capv = softCap(multiplier, invCap, cap);
    dim3 const rowsGrid(static_cast<unsigned>(params.numRows));
    launch(fs::splitDrawKernel<fs::kSplitBlock, fs::kSplitWalkVectors, false, true>, rowsGrid, dim3(fs::kSplitBlock),
        pdl, params, threadStates, capv, recs);
    for (int64_t r = 0; r < rounds; ++r)
    {
        launch(fs::splitTopPCountKernel<fs::kSplitBlock>,
            dim3(static_cast<unsigned>(params.numRows), fs::kSplitBlock / fs::kTopPCountThreads),
            dim3(fs::kTopPCountThreads), pdl, params, threadStates, static_cast<fs::SplitTopPRecord const*>(recs),
            parts, capv);
        auto const* const done = static_cast<fs::CountMass const*>(parts);
        if (r + 1 < rounds)
        {
            launch(fs::splitTopPDecideKernel<fs::kSplitBlock, fs::kSplitWalkVectors, false, false>, rowsGrid,
                dim3(fs::kSplitBlock), pdl, params, threadStates, recs, done, capv);
        }
        else if (earlyTrigger)
        {
            launch(fs::splitTopPDecideKernel<fs::kSplitBlock, fs::kSplitWalkVectors, true, true>, rowsGrid,
                dim3(fs::kSplitBlock), pdl, params, threadStates, recs, done, capv);
        }
        else
        {
            launch(fs::splitTopPDecideKernel<fs::kSplitBlock, fs::kSplitWalkVectors, true, false>, rowsGrid,
                dim3(fs::kSplitBlock), pdl, params, threadStates, recs, done, capv);
        }
    }
    return tokens;
}

void refFold(at::Tensor const& capped, at::Tensor const& temperatures, at::Tensor const& states)
{
    TORCH_CHECK(capped.scalar_type() == at::kFloat && capped.dim() == 2 && capped.is_contiguous(), "FP32 rows");
    at::cuda::CUDAGuard guard(capped.device());
    checkRowVector(temperatures, "temperatures", at::kFloat, capped.size(0));
    checkStates(states, capped.size(0));
    tk::FusedSamplingParams params;
    params.logits = capped.data_ptr();
    params.temperatures = temperatures.data_ptr<float>();
    params.numRows = static_cast<int32_t>(capped.size(0));
    params.vocabSize = static_cast<int32_t>(capped.size(1));
    auto* threadStates = reinterpret_cast<fs::OnlineSoftmax*>(states.data_ptr());
    if (fs::splitVirtualBlock(params.numRows) == fs::kSplitBlock)
    {
        launch(fs::splitRefFoldKernel<fs::kSplitBlock>, dim3(params.numRows), dim3(fs::kSplitBlock), false, params,
            threadStates);
    }
    else
    {
        launch(fs::splitRefFoldKernel<fs::kSplitNarrowBlock>, dim3(params.numRows), dim3(fs::kSplitNarrowBlock), false,
            params, threadStates);
    }
}

at::Tensor capRows(at::Tensor const& x, double multiplier, double invCap, double cap)
{
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.is_cuda() && x.is_contiguous(), "BF16 CUDA tensor");
    at::cuda::CUDAGuard guard(x.device());
    auto y = at::empty(x.sizes(), x.options().dtype(at::kFloat));
    int64_t const n = x.numel();
    if (n > 0)
    {
        launch(fs::splitCapKernel, dim3(static_cast<unsigned>((n + 255) / 256)), dim3(256), false,
            reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr()), y.data_ptr<float>(), n,
            softCap(multiplier, invCap, cap));
    }
    return y;
}

void watch(at::Tensor const& buf, at::Tensor const& firstSeen, int64_t blocks, int64_t timeoutMs)
{
    TORCH_CHECK(buf.is_cuda() && buf.is_contiguous() && firstSeen.is_cuda() && firstSeen.is_contiguous(), "CUDA");
    TORCH_CHECK(firstSeen.scalar_type() == at::kInt && firstSeen.numel() * 4 == buf.numel() * buf.element_size(),
        "first_seen is one int32 per 32-bit word of buf");
    at::cuda::CUDAGuard guard(buf.device());
    launch(fs::splitWatchKernel, dim3(static_cast<unsigned>(blocks)), dim3(256), false,
        reinterpret_cast<uint32_t const*>(buf.data_ptr()), reinterpret_cast<uint32_t*>(firstSeen.data_ptr()),
        firstSeen.numel(), static_cast<unsigned long long>(timeoutMs) * 1000000ull);
}

//! The vocabulary-parallel head's common parameters: rank geometry and the exchange buffer.
fs::VpParams vpLayout(int64_t rank, int64_t ranks, int64_t shardVecs, int64_t copy, at::IntArrayRef publish,
    int64_t stateStride, int64_t greedyOffset, int64_t greedyStride, int64_t candOffset, int64_t candStride,
    int64_t timeoutS)
{
    TORCH_CHECK(ranks >= 1 && ranks <= fs::kVpMaxRanks && rank >= 0 && rank < ranks, "rank / ranks out of range");
    TORCH_CHECK(publish.size() >= 1 && publish.size() <= static_cast<size_t>(fs::kVpMaxRanks), "1..8 publish targets");
    TORCH_CHECK(shardVecs >= 1, "empty shard");
    fs::VpParams p;
    p.rank = static_cast<int>(rank);
    p.ranks = static_cast<int>(ranks);
    p.shardVecs = static_cast<int>(shardVecs);
    p.copy = reinterpret_cast<char*>(copy);
    for (size_t i = 0; i < publish.size(); ++i)
    {
        TORCH_CHECK(publish[i] % 16 == 0, "publish targets must be 16-byte aligned");
        p.publish[i] = reinterpret_cast<char*>(publish[i]);
    }
    p.numPublish = static_cast<int>(publish.size());
    TORCH_CHECK(copy % 16 == 0 && greedyOffset % 16 == 0 && candOffset % 16 == 0, "exchange regions 16-byte aligned");
    p.stateStride = stateStride;
    p.greedyOffset = greedyOffset;
    p.greedyStride = greedyStride;
    p.candOffset = candOffset;
    p.candStride = candStride;
    p.timeoutNs = static_cast<unsigned long long>(timeoutS) * 1000000000ull;
    return p;
}

void checkVpRows(
    int64_t rows, int64_t shardVecs, int64_t ranks, int64_t stateStride, int64_t greedyStride, int64_t candStride)
{
    TORCH_CHECK(rows >= 1 && rows <= fs::kSplitMaxRows, "1..", fs::kSplitMaxRows, " rows");
    int const vb = fs::splitVirtualBlock(static_cast<int>(rows));
    TORCH_CHECK((shardVecs + vb - 1) / vb <= fs::kVpWalkVectors, "shard too wide for the walk stage");
    TORCH_CHECK(rows * vb <= stateStride, "rows exceed the state slots");
    TORCH_CHECK(rows * fs::kVpGreedySlots <= greedyStride, "rows exceed the greedy slots");
    TORCH_CHECK(rows * fs::kVpClaimants <= candStride, "rows exceed the candidate slots");
    TORCH_CHECK(ranks * shardVecs * 4 <= INT32_MAX, "vocabulary too large");
}

//! vp_fold: this rank's shard folded per (row, virtual thread) onto the previous rank's states
//! (read from `copy`, this rank's exchange copy), published to every rank.
void vpFold(at::Tensor const& local, at::Tensor const& temperatures, at::Tensor const& topKs, at::Tensor const& topPs,
    at::Tensor const& minPs, int64_t staging, at::Tensor const& status, int64_t copy, at::IntArrayRef publish,
    int64_t rank, int64_t ranks, int64_t stateStride, int64_t greedyOffset, int64_t greedyStride, int64_t candStride,
    int64_t timeoutS, bool greedy, double multiplier, double invCap, double cap, bool pdl, bool chain)
{
    TORCH_CHECK(local.scalar_type() == at::kBFloat16 && local.dim() == 2 && local.is_contiguous() && local.is_cuda(),
        "local must be a contiguous CUDA BF16 [rows, shard] tensor");
    TORCH_CHECK(local.size(1) % 4 == 0, "the shard must be a whole number of 4-element vectors");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(local.data_ptr()) % 8 == 0, "local must be 8-byte aligned");
    auto const rows = local.size(0);
    auto const shardVecs = local.size(1) / 4;
    checkVpRows(rows, shardVecs, ranks, stateStride, greedyStride, candStride);
    at::cuda::CUDAGuard guard(local.device());
    TORCH_CHECK(status.is_cuda() && status.scalar_type() == at::kInt && status.numel() >= 1, "status int32[1]");
    auto p = vpLayout(
        rank, ranks, shardVecs, copy, publish, stateStride, greedyOffset, greedyStride, 0, candStride, timeoutS);
    p.status = status.data_ptr<int32_t>();
    p.chain = chain ? 1 : 0;
    p.fs.numRows = static_cast<int32_t>(rows);
    p.fs.vocabSize = static_cast<int32_t>(ranks * shardVecs * 4);
    if (!greedy)
    {
        checkRowVector(temperatures, "temperatures", at::kFloat, rows);
        checkRowVector(topKs, "top_ks", at::kInt, rows);
        checkRowVector(topPs, "top_ps", at::kFloat, rows);
        checkRowVector(minPs, "min_ps", at::kFloat, rows);
        p.fs.temperatures = temperatures.data_ptr<float>();
        p.fs.topKs = topKs.data_ptr<int32_t>();
        p.fs.topPs = topPs.data_ptr<float>();
        p.fs.minPs = minPs.data_ptr<float>();
        TORCH_CHECK(staging % 8 == 0, "staging must be 8-byte aligned");
    }
    p.local = reinterpret_cast<uint2 const*>(local.data_ptr());
    p.staging = reinterpret_cast<uint2*>(staging);
    p.cap = softCap(multiplier, invCap, cap);
    int const vb = fs::splitVirtualBlock(static_cast<int>(rows));
    dim3 const grid(static_cast<unsigned>(rows), vb / fs::kVpFoldThreads);
    if (vb == fs::kSplitBlock)
    {
        if (greedy)
        {
            launch(fs::vpFoldKernel<true, fs::kSplitBlock>, grid, dim3(fs::kVpFoldThreads), pdl, p);
        }
        else
        {
            launch(fs::vpFoldKernel<false, fs::kSplitBlock>, grid, dim3(fs::kVpFoldThreads), pdl, p);
        }
    }
    else if (greedy)
    {
        launch(fs::vpFoldKernel<true, fs::kSplitNarrowBlock>, grid, dim3(fs::kVpFoldThreads), pdl, p);
    }
    else
    {
        launch(fs::vpFoldKernel<false, fs::kSplitNarrowBlock>, grid, dim3(fs::kVpFoldThreads), pdl, p);
    }
}

//! vp_draw: reduce, scan, draw, walk this rank's portions after the previous rank's; publishes walk records.
//! Returns the token tensor vp_decide completes (filtered rows are written here).
at::Tensor vpDraw(at::Tensor const& pending, at::Tensor const& local, at::Tensor const& temperatures,
    at::Tensor const& topKs, at::Tensor const& topPs, at::Tensor const& minPs, std::optional<at::Tensor> const& seed,
    std::optional<at::Tensor> const& offset, at::Tensor const& info, at::Tensor const& status, int64_t copy,
    at::IntArrayRef stagingOf, at::IntArrayRef publish, int64_t rank, int64_t ranks, int64_t stateStride,
    int64_t greedyStride, int64_t candOffset, int64_t candStride, int64_t timeoutS, double multiplier, double invCap,
    double cap, bool pdl, bool chain)
{
    TORCH_CHECK(pending.scalar_type() == at::kBFloat16, "pending must be BF16");
    TORCH_CHECK(
        local.scalar_type() == at::kBFloat16 && local.dim() == 2 && local.is_contiguous(), "local [rows, shard]");
    at::cuda::CUDAGuard guard(pending.device());
    auto fsParams = rowParams(pending, temperatures, topKs, topPs, minPs, seed, offset);
    auto const rows = pending.size(0);
    auto const shardVecs = local.size(1) / 4;
    TORCH_CHECK(local.size(0) == rows && local.size(1) % 4 == 0 && pending.size(1) == ranks * local.size(1),
        "pending is [rows, ranks * shard] of local [rows, shard]");
    checkVpRows(rows, shardVecs, ranks, stateStride, greedyStride, candStride);
    TORCH_CHECK(static_cast<int64_t>(stagingOf.size()) == ranks, "one staging address per rank");
    TORCH_CHECK(info.is_cuda() && info.scalar_type() == at::kInt && info.is_contiguous()
            && info.numel() >= rows * fs::kVpInfoWords,
        "info holds kVpInfoWords int32 per row");
    TORCH_CHECK(status.is_cuda() && status.scalar_type() == at::kInt && status.numel() >= 1, "status int32[1]");
    auto p = vpLayout(
        rank, ranks, shardVecs, copy, publish, stateStride, 0, greedyStride, candOffset, candStride, timeoutS);
    p.fs = fsParams;
    for (int64_t r = 0; r < ranks; ++r)
    {
        p.stagingOf[r] = reinterpret_cast<uint2 const*>(stagingOf[r]);
    }
    p.local = reinterpret_cast<uint2 const*>(local.data_ptr());
    p.info = info.data_ptr<int32_t>();
    p.status = status.data_ptr<int32_t>();
    p.chain = chain ? 1 : 0;
    p.cap = softCap(multiplier, invCap, cap);
    auto tokens = at::empty({rows}, pending.options().dtype(at::kInt));
    p.fs.outputTokens = tokens.data_ptr<int32_t>();
    if (fs::splitVirtualBlock(static_cast<int>(rows)) == fs::kSplitBlock)
    {
        launch(fs::vpDrawKernel<fs::kSplitBlock>, dim3(static_cast<unsigned>(rows)), dim3(fs::kSplitBlock), pdl, p);
    }
    else
    {
        launch(fs::vpDrawKernel<fs::kSplitNarrowBlock>, dim3(static_cast<unsigned>(rows)), dim3(fs::kSplitNarrowBlock),
            pdl, p);
    }
    return tokens;
}

//! vp_decide: every rank's candidates -> the token of each unfiltered row, into `tokens`.
void vpDecide(at::Tensor const& tokens, at::Tensor const& info, at::Tensor const& status, int64_t copy, int64_t rank,
    int64_t ranks, int64_t shardVecs, int64_t stateStride, int64_t greedyStride, int64_t candOffset, int64_t candStride,
    int64_t timeoutS, bool pdl)
{
    TORCH_CHECK(tokens.is_cuda() && tokens.scalar_type() == at::kInt && tokens.dim() == 1, "tokens int32 [rows]");
    auto const rows = tokens.size(0);
    checkVpRows(rows, shardVecs, ranks, stateStride, greedyStride, candStride);
    TORCH_CHECK(info.is_cuda() && info.scalar_type() == at::kInt && info.numel() >= rows * fs::kVpInfoWords, "info");
    TORCH_CHECK(status.is_cuda() && status.scalar_type() == at::kInt && status.numel() >= 1, "status int32[1]");
    at::cuda::CUDAGuard guard(tokens.device());
    int64_t const self[1] = {copy};
    auto p = vpLayout(rank, ranks, shardVecs, copy, at::IntArrayRef(self, 1), stateStride, 0, greedyStride, candOffset,
        candStride, timeoutS);
    p.fs.numRows = static_cast<int32_t>(rows);
    p.fs.outputTokens = tokens.data_ptr<int32_t>();
    p.info = info.data_ptr<int32_t>();
    p.status = status.data_ptr<int32_t>();
    launch(fs::vpDecideKernel, dim3(static_cast<unsigned>(rows)), dim3(32), pdl, p,
        fs::splitVirtualBlock(static_cast<int>(rows)));
}

//! vp_argmax: every rank's greedy words -> torch.argmax's int64 token per row.
at::Tensor vpArgmax(at::Tensor const& status, int64_t rows, int64_t copy, int64_t rank, int64_t ranks,
    int64_t shardVecs, int64_t stateStride, int64_t greedyOffset, int64_t greedyStride, int64_t candStride,
    int64_t timeoutS, bool pdl)
{
    checkVpRows(rows, shardVecs, ranks, stateStride, greedyStride, candStride);
    TORCH_CHECK(status.is_cuda() && status.scalar_type() == at::kInt && status.numel() >= 1, "status int32[1]");
    at::cuda::CUDAGuard guard(status.device());
    int64_t const self[1] = {copy};
    auto p = vpLayout(rank, ranks, shardVecs, copy, at::IntArrayRef(self, 1), stateStride, greedyOffset, greedyStride,
        0, candStride, timeoutS);
    p.status = status.data_ptr<int32_t>();
    auto tokens = at::empty({rows}, status.options().dtype(at::kLong));
    p.greedyTokens = tokens.data_ptr<int64_t>();
    launch(fs::vpArgmaxKernel, dim3(static_cast<unsigned>(rows)), dim3(32), pdl, p,
        fs::splitVirtualBlock(static_cast<int>(rows)) / fs::kVpFoldThreads);
    return tokens;
}

//! The dynamic shared memory a dsSampleKernel block may use on `device` (the device's opt-in limit less the
//! kernel's static shared memory), raised once per instantiation and device.
template <bool kTrigger, bool kStamp, bool kFabric>
size_t dsRaiseSmem(int device)
{
    constexpr int kDevices = 64;
    static size_t limits[kDevices] = {};
    TORCH_CHECK(device >= 0 && device < kDevices, "device index out of range");
    if (limits[device] == 0)
    {
        cudaFuncAttributes attributes{};
        C10_CUDA_CHECK(cudaFuncGetAttributes(&attributes, fs::dsSampleKernel<kTrigger, kStamp, kFabric>));
        int optin = 0;
        C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        size_t const limit = static_cast<size_t>(optin) - attributes.sharedSizeBytes;
        C10_CUDA_CHECK(cudaFuncSetAttribute(fs::dsSampleKernel<kTrigger, kStamp, kFabric>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(limit)));
        limits[device] = limit;
    }
    return limits[device];
}

//! ds_sample: the distributed sampler's tokens (int32 [ranksHere * rows]; ranksHere = world when `emulated`, else
//! 1). `local` is [ranksHere * rows, shard] BF16 (this process's ranks' LM-head GEMM outputs, rank-major),
//! `pending` [ranksHere * rows, world * shard] BF16 (written for top-k / min-p rows only); the row params are
//! [rows]. `staging` holds every rank's staging address ([2][maxRows][shard / 4] vectors), `publish` the LL area
//! bases the values are stored through, `copies` the LL area copy of each rank here.
at::Tensor dsSample(at::Tensor const& local, at::Tensor const& pending, at::Tensor const& temperatures,
    at::Tensor const& topKs, at::Tensor const& topPs, at::Tensor const& minPs, std::optional<at::Tensor> const& seed,
    std::optional<at::Tensor> const& offset, at::IntArrayRef staging, at::IntArrayRef publish, at::IntArrayRef copies,
    at::Tensor const& counters, at::Tensor const& status, int64_t rank, int64_t world, bool emulated, int64_t maxRows,
    int64_t readyOffset, int64_t statesOffset, int64_t candOffset, int64_t partialOffset, int64_t timeoutS,
    double multiplier, double invCap, double cap, bool pdl, bool earlyTrigger, std::optional<at::Tensor> const& stamps,
    bool fabric)
{
    TORCH_CHECK(world >= 1 && world <= fs::kDsMaxRanks && fs::kDsThreads % world == 0, "1, 2, 4 or 8 ranks");
    TORCH_CHECK(rank >= 0 && rank < world, "rank out of range");
    int64_t const ranksHere = emulated ? world : 1;
    TORCH_CHECK(local.scalar_type() == at::kBFloat16 && local.dim() == 2 && local.is_contiguous() && local.is_cuda(),
        "local must be a contiguous CUDA BF16 [ranks here * rows, shard] tensor");
    TORCH_CHECK(local.size(0) % ranksHere == 0, "local holds every rank here's rows");
    int64_t const rows = local.size(0) / ranksHere;
    int64_t const shard = local.size(1);
    TORCH_CHECK(shard % 4 == 0, "the shard must be a whole number of 4-element vectors");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(local.data_ptr()) % 8 == 0, "local must be 8-byte aligned");
    TORCH_CHECK(rows >= 1 && rows <= maxRows, "1..max_rows rows");
    // Above kWideBlockMaxRows rows the split sampler runs the 512-thread virtual block: not this kernel's.
    TORCH_CHECK(maxRows <= fs::kWideBlockMaxRows, "max_rows above ", fs::kWideBlockMaxRows);
    TORCH_CHECK(pending.scalar_type() == at::kBFloat16 && pending.dim() == 2 && pending.is_contiguous()
            && pending.size(0) == ranksHere * rows && pending.size(1) == world * shard,
        "pending must be a contiguous BF16 [ranks here * rows, world * shard] tensor");
    at::cuda::CUDAGuard guard(local.device());
    auto params = rowParams(pending.narrow(0, 0, rows), temperatures, topKs, topPs, minPs, seed, offset);
    TORCH_CHECK(params.vocabSize / 4 <= fs::kSplitWalkVectors * fs::kDsThreads, "vocabulary too wide for the walk");
    TORCH_CHECK(static_cast<int64_t>(staging.size()) == world, "one staging address per rank");
    TORCH_CHECK(publish.size() >= 1 && publish.size() <= static_cast<size_t>(fs::kDsMaxRanks), "1..8 publish targets");
    TORCH_CHECK(static_cast<int64_t>(copies.size()) == ranksHere, "one LL area copy per rank here");
    TORCH_CHECK(counters.is_cuda() && counters.scalar_type() == at::kInt && counters.is_contiguous()
            && counters.numel() >= ranksHere * maxRows,
        "counters holds one int32 per (rank here, row)");
    TORCH_CHECK(status.is_cuda() && status.scalar_type() == at::kInt && status.numel() >= 1, "status int32[1]");
    TORCH_CHECK(pending.device() == local.device() && counters.device() == local.device()
            && status.device() == local.device() && temperatures.device() == local.device(),
        "every tensor on the local shard's device");
    int device = 0;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    int sms = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    // Every CTA waits for the same row's CTAs of the other ranks: the grid must be resident at once.
    TORCH_CHECK(rows * ranksHere <= sms, "ds_sample: ", rows * ranksHere, " CTAs exceed the ", sms,
        " SMs (one resident CTA each)");
    fs::DsParams p;
    p.fs = params;
    p.fs.logits = pending.data_ptr();
    auto tokens = at::empty({ranksHere * rows}, local.options().dtype(at::kInt));
    p.fs.outputTokens = tokens.data_ptr<int32_t>();
    p.local = reinterpret_cast<uint2 const*>(local.data_ptr());
    for (int64_t r = 0; r < world; ++r)
    {
        TORCH_CHECK(staging[r] % 16 == 0, "staging addresses must be 16-byte aligned");
        p.staging[r] = reinterpret_cast<uint2*>(staging[r]);
    }
    for (size_t i = 0; i < publish.size(); ++i)
    {
        TORCH_CHECK(publish[i] % 16 == 0, "publish targets must be 16-byte aligned");
        p.publish[i] = static_cast<unsigned long long>(publish[i]);
    }
    p.numPublish = static_cast<int>(publish.size());
    for (int64_t r = 0; r < ranksHere; ++r)
    {
        TORCH_CHECK(copies[r] % 16 == 0, "LL area copies must be 16-byte aligned");
        p.copy[r] = static_cast<unsigned long long>(copies[r]);
    }
    TORCH_CHECK(readyOffset % 16 == 0 && statesOffset % 16 == 0 && candOffset % 16 == 0 && partialOffset % 16 == 0,
        "LL regions 16-byte aligned");
    p.counters = reinterpret_cast<uint32_t*>(counters.data_ptr<int32_t>());
    p.status = status.data_ptr<int32_t>();
    p.cap = softCap(multiplier, invCap, cap);
    p.rank = static_cast<int>(rank);
    p.world = static_cast<int>(world);
    p.emulated = emulated ? 1 : 0;
    p.shardVecs = static_cast<int>(shard / 4);
    p.maxRows = static_cast<int>(maxRows);
    p.readyOffset = static_cast<unsigned long long>(readyOffset);
    p.statesOffset = static_cast<unsigned long long>(statesOffset);
    p.candOffset = static_cast<unsigned long long>(candOffset);
    p.partialOffset = static_cast<unsigned long long>(partialOffset);
    p.timeoutNs = static_cast<unsigned long long>(timeoutS) * 1000000000ull;
    // This rank's slots: L per 1024-vector stripe of the row.
    size_t const smem = static_cast<size_t>((params.vocabSize / 4 + fs::kDsThreads - 1) / fs::kDsThreads)
        * (fs::kDsThreads / world) * sizeof(float4);
    bool const stamped = stamps.has_value();
    if (stamped)
    {
        auto const& t = stamps.value();
        TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kLong && t.is_contiguous() && t.device() == local.device()
                && t.numel() >= ranksHere * maxRows * fs::kDsStamps,
            "stamps: int64 [ranks here * max_rows * kDsStamps] on the local shard's device");
        p.stamps = reinterpret_cast<unsigned long long*>(t.data_ptr<int64_t>());
    }
    // One instantiation per (trigger, stamps, fabric); the served ones are <trigger, false, fabric>.
    size_t (*raise)(int) = nullptr;
    void (*kernel)(fs::DsParams const) = nullptr;
#define DS_PICK(T, S, F)                                                                                               \
    if (earlyTrigger == T && stamped == S && fabric == F)                                                              \
    {                                                                                                                  \
        raise = &dsRaiseSmem<T, S, F>;                                                                                 \
        kernel = &fs::dsSampleKernel<T, S, F>;                                                                         \
    }
    DS_PICK(true, true, true)
    DS_PICK(true, true, false)
    DS_PICK(true, false, true)
    DS_PICK(true, false, false)
    DS_PICK(false, true, true)
    DS_PICK(false, true, false)
    DS_PICK(false, false, true)
    DS_PICK(false, false, false)
#undef DS_PICK
    TORCH_CHECK(raise != nullptr && kernel != nullptr, "ds_sample: no instantiation");
    size_t const most = raise(device);
    TORCH_CHECK(smem <= most, "ds_sample: a shard of ", shard, " needs ", smem, " bytes of shared memory, ", most,
        " available");
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(static_cast<unsigned>(rows), static_cast<unsigned>(ranksHere));
    config.blockDim = dim3(fs::kDsThreads);
    config.dynamicSmemBytes = smem;
    config.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    config.attrs = attribute;
    config.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, p));
    return tokens;
}

//! The DFlash 2 sampled path walk (draft_path_sample.cuh): the [B, K] int32 draft tokens and the [K, B, top_k]
//! distributions the positions were drawn from. With draft_probs, also writes each position's distribution into
//! its request's slot row at the candidates' vocabulary ids (the caller zeroes those rows first).
std::tuple<at::Tensor, at::Tensor> draftPathSample(at::Tensor const& edges, at::Tensor const& unary,
    at::Tensor const& ids, at::Tensor const& temperatures, at::Tensor const& topKs, at::Tensor const& topPs,
    at::Tensor const& minPs, at::Tensor const& seed, at::Tensor const& offset, int64_t slotBase, int64_t slotSpan,
    std::optional<at::Tensor> const& draftProbs, std::optional<at::Tensor> const& slots)
{
    TORCH_CHECK(unary.is_cuda() && unary.dim() == 3 && unary.scalar_type() == at::kFloat && unary.is_contiguous(),
        "unary must be a contiguous FP32 [B, K, top_k] CUDA tensor");
    int64_t const rows = unary.size(0);
    int64_t const steps = unary.size(1);
    int64_t const topK = unary.size(2);
    TORCH_CHECK(rows >= 1 && rows <= INT32_MAX && steps >= 1 && topK >= 1 && topK <= fs::kNarrowBlock,
        "unary shape out of range");
    TORCH_CHECK(edges.scalar_type() == at::kFloat && edges.is_contiguous()
            && edges.sizes() == at::IntArrayRef({rows, steps, topK, topK}),
        "edges must be a contiguous FP32 [B, K, top_k, top_k] tensor");
    TORCH_CHECK(ids.scalar_type() == at::kLong && ids.is_contiguous() && ids.sizes() == unary.sizes(),
        "candidate ids must be a contiguous int64 [B, K, top_k] tensor");
    TORCH_CHECK(edges.device() == unary.device() && ids.device() == unary.device(),
        "edges and candidate ids must be on the device of unary");
    checkRowVector(temperatures, "temperatures", at::kFloat, rows);
    checkRowVector(topKs, "top_ks", at::kInt, rows);
    checkRowVector(topPs, "top_ps", at::kFloat, rows);
    checkRowVector(minPs, "min_ps", at::kFloat, rows);
    checkRowVector(seed, "seed", at::kLong, rows);
    checkRowVector(offset, "offset", at::kLong, rows);
    TORCH_CHECK(draftProbs.has_value() == slots.has_value(), "draft_probs and slots go together");
    at::cuda::CUDAGuard guard(unary.device());

    auto const options = unary.options();
    auto rowScratch = at::empty({steps, rows, topK}, options);
    auto probs = at::empty({steps, rows, topK}, options);
    auto offsets = at::empty({steps, rows}, options.dtype(at::kLong));
    auto picks = at::empty({steps, rows}, options.dtype(at::kInt));
    auto tokens = at::empty({rows, steps}, options.dtype(at::kInt));

    fs::DraftPathSampleParams p;
    p.edges = edges.data_ptr<float>();
    p.unary = unary.data_ptr<float>();
    p.ids = ids.data_ptr<int64_t>();
    p.temperatures = temperatures.data_ptr<float>();
    p.topKs = topKs.data_ptr<int32_t>();
    p.topPs = topPs.data_ptr<float>();
    p.minPs = minPs.data_ptr<float>();
    p.seed = seed.data_ptr<int64_t>();
    p.offset = offset.data_ptr<int64_t>();
    p.slotBase = slotBase;
    p.slotSpan = slotSpan;
    p.rows = rowScratch.data_ptr<float>();
    p.probs = probs.data_ptr<float>();
    p.offsets = offsets.data_ptr<int64_t>();
    p.picks = picks.data_ptr<int32_t>();
    p.tokens = tokens.data_ptr<int32_t>();
    if (draftProbs.has_value())
    {
        auto const& dp = draftProbs.value();
        auto const& sl = slots.value();
        TORCH_CHECK(
            dp.is_cuda() && dp.scalar_type() == at::kFloat && dp.dim() == 3 && dp.stride(2) == 1 && dp.size(1) >= steps,
            "draft_probs must be an FP32 [slots, >= K, vocab] tensor with unit vocabulary stride");
        checkRowVector(sl, "slots", at::kLong, rows);
        TORCH_CHECK(dp.device() == unary.device() && sl.device() == unary.device(),
            "draft_probs and slots must be on the device of unary");
        p.draftProbs = dp.data_ptr<float>();
        p.slots = sl.data_ptr<int64_t>();
        p.draftProbsSlotStride = dp.stride(0);
        p.draftProbsStepStride = dp.stride(1);
    }
    p.numRows = static_cast<int32_t>(rows);
    p.numSteps = static_cast<int32_t>(steps);
    p.topK = static_cast<int32_t>(topK);
    // The shipped launcher's block: 1024 threads up to kWideBlockMaxRows rows, else 512.
    if (rows <= fs::kWideBlockMaxRows)
    {
        launch(fs::draftPathSampleKernel<fs::kWideBlock>, dim3(static_cast<unsigned>(rows)), dim3(fs::kWideBlock),
            false, p);
    }
    else
    {
        launch(fs::draftPathSampleKernel<fs::kNarrowBlock>, dim3(static_cast<unsigned>(rows)), dim3(fs::kNarrowBlock),
            false, p);
    }
    return {tokens, probs};
}

} // namespace

TORCH_LIBRARY(musespark_open_sm107_head_sampler, m)
{
    m.def(
        "fold_plain(Tensor logits, Tensor temperatures, Tensor(a!) states, float multiplier, float inv_cap, "
        "float cap, int threads, bool pdl) -> ()");
    m.def(
        "fold_gather(Tensor(a!) inbox, Tensor(b!) gathered, Tensor temperatures, Tensor(c!) states, "
        "Tensor(d!) status, float multiplier, float inv_cap, float cap, int shard_vecs, int slot_rows, "
        "int timeout_s, bool fold, int threads, bool pdl) -> ()");
    m.def(
        "draw(Tensor logits, Tensor temperatures, Tensor top_ks, Tensor top_ps, Tensor min_ps, Tensor? seed, "
        "Tensor? offset, Tensor states, float multiplier, float inv_cap, float cap, bool pdl, "
        "bool early_trigger=False) -> Tensor");
    m.def(
        "draw_topp(Tensor logits, Tensor temperatures, Tensor top_ks, Tensor top_ps, Tensor min_ps, Tensor? seed, "
        "Tensor? offset, Tensor states, Tensor(a!) records, Tensor(b!) partials, float multiplier, float inv_cap, "
        "float cap, int rounds, bool pdl, bool early_trigger=False) -> Tensor");
    m.def(
        "fold_plain_coop(Tensor logits, Tensor temperatures, Tensor(a!) states, float multiplier, float inv_cap, "
        "float cap, int vts, int threads, bool early_trigger, bool pdl) -> ()");
    m.def("ref_fold(Tensor capped, Tensor temperatures, Tensor(a!) states) -> ()");
    m.def("cap(Tensor x, float multiplier, float inv_cap, float cap) -> Tensor");
    m.def("watch(Tensor buf, Tensor(a!) first_seen, int blocks, int timeout_ms) -> ()");
    m.def(
        "vp_fold(Tensor local, Tensor temperatures, Tensor top_ks, Tensor top_ps, Tensor min_ps, int staging, "
        "Tensor(a!) status, int copy, int[] publish, int rank, int ranks, int state_stride, int greedy_offset, "
        "int greedy_stride, int cand_stride, int timeout_s, bool greedy, float multiplier, float inv_cap, float cap, "
        "bool pdl, bool chain=True) -> ()");
    m.def(
        "vp_draw(Tensor(a!) pending, Tensor local, Tensor temperatures, Tensor top_ks, Tensor top_ps, "
        "Tensor min_ps, Tensor? seed, Tensor? offset, Tensor(b!) info, Tensor(c!) status, int copy, "
        "int[] staging_of, int[] publish, int rank, int ranks, int state_stride, int greedy_stride, "
        "int cand_offset, int cand_stride, int timeout_s, float multiplier, float inv_cap, float cap, "
        "bool pdl, bool chain=True) -> Tensor");
    m.def(
        "vp_decide(Tensor(a!) tokens, Tensor info, Tensor(b!) status, int copy, int rank, int ranks, "
        "int shard_vecs, int state_stride, int greedy_stride, int cand_offset, int cand_stride, int timeout_s, "
        "bool pdl) -> ()");
    m.def(
        "vp_argmax(Tensor(a!) status, int rows, int copy, int rank, int ranks, int shard_vecs, int state_stride, "
        "int greedy_offset, int greedy_stride, int cand_stride, int timeout_s, bool pdl) -> Tensor");
    m.def(
        "ds_sample(Tensor local, Tensor(a!) pending, Tensor temperatures, Tensor top_ks, Tensor top_ps, "
        "Tensor min_ps, Tensor? seed, Tensor? offset, int[] staging, int[] publish, int[] copies, "
        "Tensor(b!) counters, Tensor(c!) status, int rank, int world, bool emulated, int max_rows, "
        "int ready_offset, int states_offset, int cand_offset, int partial_offset, int timeout_s, "
        "float multiplier, float inv_cap, float cap, bool pdl, bool early_trigger=False, Tensor? stamps=None, "
        "bool fabric=False) -> Tensor");
    m.def(
        "draft_path_sample(Tensor edges, Tensor unary, Tensor ids, Tensor temperatures, Tensor top_ks, "
        "Tensor top_ps, Tensor min_ps, Tensor seed, Tensor offset, int slot_base, int slot_span, "
        "Tensor(a!)? draft_probs, Tensor? slots) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(musespark_open_sm107_head_sampler, CUDA, m)
{
    m.impl("fold_plain", &foldPlain);
    m.impl("fold_gather", &foldGather);
    m.impl("draw", &draw);
    m.impl("draw_topp", &drawTopP);
    m.impl("fold_plain_coop", &foldPlainCoop);
    m.impl("ref_fold", &refFold);
    m.impl("cap", &capRows);
    m.impl("watch", &watch);
    m.impl("vp_fold", &vpFold);
    m.impl("vp_draw", &vpDraw);
    m.impl("vp_decide", &vpDecide);
    m.impl("vp_argmax", &vpArgmax);
    m.impl("ds_sample", &dsSample);
    m.impl("draft_path_sample", &draftPathSample);
}
