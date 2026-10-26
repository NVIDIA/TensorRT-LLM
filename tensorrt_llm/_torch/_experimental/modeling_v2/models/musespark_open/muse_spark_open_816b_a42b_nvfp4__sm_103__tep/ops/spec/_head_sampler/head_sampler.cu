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

#include "fusedSamplingKernelsCommon.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <climits>
#include <cstdint>
#include <optional>

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
//! PDL: waits on kernel A before reading anything.
template <int VB, int WALK>
__global__ void __launch_bounds__(VB, VB == kSplitNarrowBlock ? 2 : 1)
    splitDrawKernel(FusedSamplingParams params, OnlineSoftmax const* states, SplitCap cap)
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
    int const vocabSize = params.vocabSize;
    RowParams const rp = loadRowParams(params, row);
    if (rp.needTopK || rp.needTopP || rp.needMinP)
    {
        fusedSamplingBody<CapBf16, VB, true, false>(params, shared);
        return;
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
        params.outputTokens[row] = cand >= 0 ? cand : shared.argmax;
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
// whole vocabulary; rank r holds the vectors [r * shardVecs, (r + 1) * shardVecs). vp_fold
// folds, per (row, vt), the vectors of vt that lie in this rank's shard, in increasing
// order, with the shipped per-element statements, and publishes that partial state to
// every rank. vp_draw combines the R partials of its vt in rank order (the vocabulary
// order of vt's vectors), then runs the shipped first round (block reduce, scan, Philox
// draw, claimants); every rank walks its own portion of each claimant's elements, from
// the claimant's base plus the masses of the earlier ranks' portions, and publishes the
// crossing it finds (or none). vp_decide takes, per claimant, the first rank in rank order
// with a crossing, and kernel B's rule across claimants. Greedy batches: vp_fold reduces
// torch.argmax's comparator over each CTA and vp_argmax over the ranks' CTAs.
//
// Exchange buffer (one copy per rank, every 32-bit word holds kSplitSentinel between
// calls): states [ranks][stateStride] x 16 B, greedy words [ranks][greedyStride] x 8 B at
// greedyOffset, candidates [ranks][candStride] x 4 B at candOffset. Each value is stored
// through every publish target (the multicast address in service; every rank's copy in
// the single-GPU emulation); every rank reads and resets its own copy.

//! Ranks the vocabulary-parallel head takes.
constexpr int kVpMaxRanks = 8;
//! Claimants of a row whose crossings the head exchanges (in thread order).
constexpr int kVpClaimants = 4;
//! A candidate word: no crossing in this rank's portion.
constexpr uint32_t kVpNone = 0x7FFFFFFEu;
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

__device__ __forceinline__ uint32_t* vpCand(char* base, VpParams const& p, int rank, int row, int c)
{
    return reinterpret_cast<uint32_t*>(base + p.candOffset) + rank * p.candStride
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

//! vp_fold: grid (numRows, VB / kVpFoldThreads), kVpFoldThreads threads. PDL: waits on
//! the LM-head GEMM before reading anything; releases its dependent after publishing.
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
    // This thread's R partial states, combined in rank order. They stay in the copy until
    // the end of the call (a claimant reads the earlier ranks' again for its walk offset).
    OnlineSoftmax st{-FLT_MAX, 0.0f, 0};
    for (int r = 0; r < p.ranks; ++r)
    {
        uint4 const w = vpAwait<3>(vpState(p.copy, p, r, row, tid, VB), p.status, p.timeoutNs);
        OnlineSoftmax const part{__uint_as_float(w.x), __uint_as_float(w.y), static_cast<int>(w.z)};
        st = r == 0 ? part : combineOnlineSoftmax(st, part);
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
                *vpCand(p.publish[i], p, p.rank, row, tid) = kVpNone;
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
        // The masses of this thread's portions on the ranks before this one.
        float earlier = 0.0f;
        for (int r = 0; r < p.rank; ++r)
        {
            uint4 const w = loadRelaxedSys4(vpState(p.copy, p, r, row, tid, VB));
            earlier += __uint_as_float(w.y) * __expf(__uint_as_float(w.x) - maxScaled);
        }
        sClaimTid[claimIdx] = tid;
        sClaimStart[claimIdx] = base + earlier;
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
            float running = sClaimStart[c];
            uint32_t cand = kVpNone;
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
            sCand[c] = cand;
        }
        __syncthreads();
    }
    if (tid < kVpClaimants)
    {
        uint32_t const cand = tid < walkers ? sCand[tid] : kVpNone;
        for (int i = 0; i < p.numPublish; ++i)
        {
            *vpCand(p.publish[i], p, p.rank, row, tid) = cand;
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
        uint32_t* const slot = vpCand(p.copy, p, lane / kVpClaimants, row, lane % kVpClaimants);
        mine = vpAwait<1>(slot, p.status, p.timeoutNs).x;
        storeRelaxedSys1(slot, kSplitSentinel);
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
                uint32_t cand = kVpNone;
                for (int r = 0; r < p.ranks && cand == kVpNone; ++r)
                {
                    cand = words[r * kVpClaimants + c];
                }
                if (cand == kVpNone)
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
    bool pdl)
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
    if (fs::splitVirtualBlock(params.numRows) == fs::kSplitBlock)
    {
        launch(fs::splitDrawKernel<fs::kSplitBlock, fs::kSplitWalkVectors>, dim3(params.numRows), dim3(fs::kSplitBlock),
            pdl, params, threadStates, softCap(multiplier, invCap, cap));
    }
    else
    {
        launch(fs::splitDrawKernel<fs::kSplitNarrowBlock, fs::kSplitNarrowWalkVectors>, dim3(params.numRows),
            dim3(fs::kSplitNarrowBlock), pdl, params, threadStates, softCap(multiplier, invCap, cap));
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

//! vp_fold: this rank's shard folded per (row, virtual thread), published to every rank.
void vpFold(at::Tensor const& local, at::Tensor const& temperatures, at::Tensor const& topKs, at::Tensor const& topPs,
    at::Tensor const& minPs, int64_t staging, at::IntArrayRef publish, int64_t rank, int64_t ranks, int64_t stateStride,
    int64_t greedyOffset, int64_t greedyStride, int64_t candStride, bool greedy, double multiplier, double invCap,
    double cap, bool pdl)
{
    TORCH_CHECK(local.scalar_type() == at::kBFloat16 && local.dim() == 2 && local.is_contiguous() && local.is_cuda(),
        "local must be a contiguous CUDA BF16 [rows, shard] tensor");
    TORCH_CHECK(local.size(1) % 4 == 0, "the shard must be a whole number of 4-element vectors");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(local.data_ptr()) % 8 == 0, "local must be 8-byte aligned");
    auto const rows = local.size(0);
    auto const shardVecs = local.size(1) / 4;
    checkVpRows(rows, shardVecs, ranks, stateStride, greedyStride, candStride);
    at::cuda::CUDAGuard guard(local.device());
    auto p = vpLayout(
        rank, ranks, shardVecs, publish[0], publish, stateStride, greedyOffset, greedyStride, 0, candStride, 0);
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

//! vp_draw: combine, reduce, scan, draw, walk this rank's portions; publishes candidates.
//! Returns the token tensor vp_decide completes (filtered rows are written here).
at::Tensor vpDraw(at::Tensor const& pending, at::Tensor const& local, at::Tensor const& temperatures,
    at::Tensor const& topKs, at::Tensor const& topPs, at::Tensor const& minPs, std::optional<at::Tensor> const& seed,
    std::optional<at::Tensor> const& offset, at::Tensor const& info, at::Tensor const& status, int64_t copy,
    at::IntArrayRef stagingOf, at::IntArrayRef publish, int64_t rank, int64_t ranks, int64_t stateStride,
    int64_t greedyStride, int64_t candOffset, int64_t candStride, int64_t timeoutS, double multiplier, double invCap,
    double cap, bool pdl)
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

} // namespace

TORCH_LIBRARY(musespark_open_head_sampler, m)
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
        "Tensor? offset, Tensor states, float multiplier, float inv_cap, float cap, bool pdl) -> Tensor");
    m.def("ref_fold(Tensor capped, Tensor temperatures, Tensor(a!) states) -> ()");
    m.def("cap(Tensor x, float multiplier, float inv_cap, float cap) -> Tensor");
    m.def("watch(Tensor buf, Tensor(a!) first_seen, int blocks, int timeout_ms) -> ()");
    m.def(
        "vp_fold(Tensor local, Tensor temperatures, Tensor top_ks, Tensor top_ps, Tensor min_ps, int staging, "
        "int[] publish, int rank, int ranks, int state_stride, int greedy_offset, int greedy_stride, "
        "int cand_stride, bool greedy, float multiplier, float inv_cap, float cap, bool pdl) -> ()");
    m.def(
        "vp_draw(Tensor(a!) pending, Tensor local, Tensor temperatures, Tensor top_ks, Tensor top_ps, "
        "Tensor min_ps, Tensor? seed, Tensor? offset, Tensor(b!) info, Tensor(c!) status, int copy, "
        "int[] staging_of, int[] publish, int rank, int ranks, int state_stride, int greedy_stride, "
        "int cand_offset, int cand_stride, int timeout_s, float multiplier, float inv_cap, float cap, "
        "bool pdl) -> Tensor");
    m.def(
        "vp_decide(Tensor(a!) tokens, Tensor info, Tensor(b!) status, int copy, int rank, int ranks, "
        "int shard_vecs, int state_stride, int greedy_stride, int cand_offset, int cand_stride, int timeout_s, "
        "bool pdl) -> ()");
    m.def(
        "vp_argmax(Tensor(a!) status, int rows, int copy, int rank, int ranks, int shard_vecs, int state_stride, "
        "int greedy_offset, int greedy_stride, int cand_stride, int timeout_s, bool pdl) -> Tensor");
}

TORCH_LIBRARY_IMPL(musespark_open_head_sampler, CUDA, m)
{
    m.impl("fold_plain", &foldPlain);
    m.impl("fold_gather", &foldGather);
    m.impl("draw", &draw);
    m.impl("ref_fold", &refFold);
    m.impl("cap", &capRows);
    m.impl("watch", &watch);
    m.impl("vp_fold", &vpFold);
    m.impl("vp_draw", &vpDraw);
    m.impl("vp_decide", &vpDecide);
    m.impl("vp_argmax", &vpArgmax);
}
