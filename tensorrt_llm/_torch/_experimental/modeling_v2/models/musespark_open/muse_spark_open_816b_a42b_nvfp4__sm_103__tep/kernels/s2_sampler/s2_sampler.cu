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

// The target head's sampler fold on an interleaved vocabulary shard, with the online-softmax states exchanged
// instead of the logits (``distributed_head_sampler``; kernels/s2_sampler/sampler.py).
//
// The served sampler (ops/spec/_head_sampler/head_sampler.cu) folds a gathered BF16 row of V = 4 * vecCount logits:
// virtual thread vt (0 .. 1023) folds the vectors v = vt + 1024 j (4 logits each), in increasing j, each vector's
// four logits in order, into one OnlineSoftmax {max, sum, argmax}; kernel B reduces / scans the 1024 states. When
// rank r's shard holds the vectors v = world * m + r (m = 0 .. vecCount / world - 1; local row 4m + q is global row
// 4(world m + r) + q), virtual thread vt = world * a + r folds exactly the local vectors m = a + (1024 / world) j,
// all on rank r. So each rank folds its 1024 / world virtual threads per row from its own shard, with the served
// cooperative fold's statements (splitFoldCoopKernel: the soft cap of every element by the whole CTA into shared
// memory, then one thread per virtual thread), and every state is the served state bit for bit: the same function
// of the same elements in the same order. The argmax a state carries is the global element index 4 (vt + 1024 j) + e.
//
// Transport of the states: LL elements over the NVLink multicast fabric, no reset (the one-shot logits gather's
// protocol, kernels/pv_logits/pv_logits_gather.cu). A state travels as four 64-bit elements (epoch << 32) | word with
// the words {max, sum, argmax, 0}, written with two 16-byte multicast stores to (row, rank, a) of the area of the
// call's epoch parity, i.e. into every rank's copy; a reader polls its own copy with 16-byte relaxed system-scope loads
// until the four elements of a state carry the call's epoch. A 64-bit element is written and read as one access, so a
// matching epoch half implies the data half of the same call.
//
// Ownership and progress.
// - Writers: element (row, r, a, w) of rank R's copy is written only by rank r's fold kernel, CTA (row, a / VTS),
//   thread a % VTS, through the multicast address, once per call. Readers: only rank R's poll kernel or the draw
//   kernel (a call runs one of the two) reads rank R's copy. Nothing stores sentinels into fabric memory.
// - Epochs: the fold kernel's CTA b keeps foldCounters[b] and the poll kernel's CTA row keeps pollCounters[row] (this
//   rank's device memory, each read and written only by its own CTA): a call uses counter + 1 and stores it back at
//   the CTA's end. Grids are fixed per instance (rows x 1024 / world / VTS fold CTAs, rows poll CTAs), every call runs
//   every CTA of both kernels once, and every rank runs the same sequence of calls, so all CTAs of all ranks agree
//   on a call's epoch. The areas start zeroed (epoch 0 is never used); a stale element carries an older epoch.
// - Reuse: call e uses area e & 1. A peer writes the area rank R reads in call e next in its call e + 2. The peer's
//   fold of call e + 2 runs after its poll of call e + 1 completed (stream order through the kernels between them,
//   each of which waits for its predecessor before it completes), and that poll completed only after it saw rank R's
//   call-(e + 1) states, which rank R's fold publishes only after its own poll of call e completed (same argument).
//   So calls may follow each other with no other collective in between.
// - Counter reads: the poll kernel of call e is launched on the fold's trigger, which the fold issues after its
//   griddepcontrol.wait, so its predecessors (the poll of call e - 1 among them) have completed; the poll kernel
//   itself waits for the fold grid (griddepcontrol.wait) before it completes, which keeps that chain transitive.
// - PDL: the fold waits before its first global access and releases its dependent right after (as the served
//   cooperative fold with EARLY_TRIGGER), so the poll kernel is resident while the fold runs.
//
// The draw (s2DrawKernel) stands in for the served kernel B (head_sampler.cu splitDrawKernel), one 1024-thread
// CTA per row on every rank. A neutral row (no top-k / top-p / min-p): thread vt polls state (row, vt % world,
// vt / world) instead of loading it, then kernel B's code runs verbatim (block reduce, rebase, scan, Philox draw,
// claimants). The walk runs only on the rank that holds the walker's vectors (walker % world: its local vectors
// walker / world + localThreads q, the global vectors walker + 1024 q), with kernel B's statements; that rank
// publishes its result into its entry of a per-row token slot and every rank polls the entry. With several
// claimants (two scan spans overlapping around the target), every rank that holds claimants walks them and
// publishes its latest crossing as key = position * 2048 + tid + 1 (the order of kernel B's 64-bit key
// (position << 32) | (tid + 1), tid + 1 <= 1024), and every rank takes the largest key over those ranks -- kernel B's
// rule. A row whose params enable a filter: the fold kernel also publishes that row's local vectors into the row
// area (LL, as the states), the draw polls every rank's vectors into a [rows, vocab] BF16 scratch row in vocabulary
// order and runs the shipped body on it (kernel B's filtered path on the same bits).
// - Token slot and row area ownership: entry (row, r) of rank R's copy is written only by rank r's draw CTA row
//   (slot) or rank r's fold CTAs of the row (row area), through the multicast address, once per call, with the call's
//   epoch; only rank R's draw reads its copy. Reuse follows the states' argument: every draw CTA polls data of every
//   rank (states, or the row area) published in the same call, so a peer reaches its call e + 2 only after this
//   rank's call-(e + 1) fold, which runs after this rank's draw of call e completed.
// - The draw keeps one epoch counter per row (its CTA's), read at entry and stored at its end; it is launched on the
//   fold's trigger, after the fold's wait, so the previous call's draw has completed. It waits for the fold grid
//   (griddepcontrol.wait) before it completes. With `trigger` it releases its dependent at entry (the dependent
//   waits for this grid before it reads the tokens).

#include "fusedSamplingKernelsCommon.cuh"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cstdint>
#include <optional>

namespace tensorrt_llm
{
namespace kernels
{
namespace fusedSampling
{

// ---- Verbatim copies of head_sampler.cu's soft-capped BF16 row reads (SplitCap, capOf, sSplitCap, CapBf16, toFloat,
// forEachLogit): the shipped body instantiated on CapBf16 here reads the scratch row exactly as the served kernel B's
// filtered path reads the gathered row.

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

//! The constants for reads through CapBf16 inside the shipped body: the draw stores its
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

// ---- end of the copies.

} // namespace fusedSampling
} // namespace kernels
} // namespace tensorrt_llm

namespace
{

namespace fs = tensorrt_llm::kernels::fusedSampling;
namespace tk = tensorrt_llm::kernels;

//! Virtual threads of the served sampler's row (its 1024-thread block, head_sampler.cu kSplitBlock).
constexpr int kVirtualThreads = fs::kWideBlock;
//! Vectors one virtual thread folds at most (head_sampler.cu kSplitWalkVectors, the walk's shared-memory stage).
constexpr int kWalkVectors = 64;
//! 64-bit LL elements per state: max, sum, argmax, pad.
constexpr int kStateElems = 4;
//! Poll kernel block: one thread per virtual thread of a row.
constexpr int kPollThreads = kVirtualThreads;

//! The soft cap's FP32 constants (head_sampler.cu SplitCap).
struct Cap
{
    float multiplier;
    float invCap;
    float cap;
};

//! The torch chain logits.float() * m, / cap, tanh, cap * t, statement for statement (head_sampler.cu capOf).
__device__ __forceinline__ float capOf(float x, Cap const& c)
{
    float y = __fmul_rn(x, c.multiplier);
    y = __fmul_rn(y, c.invCap);
    y = tanhf(y);
    return __fmul_rn(c.cap, y);
}

struct FoldParams
{
    //! Row params (temperature; with the row area also top-k / top-p / min-p, which decide whether a row is filtered).
    tk::FusedSamplingParams fs;
    //! This rank's interleaved shard of the BF16 logits, [rows][localVecs] vectors of 4.
    uint2 const* local{nullptr};
    //! Multicast address of the two state areas, each [rows][world][localThreads][kStateElems] 64-bit elements.
    uint64_t mcArea{0};
    uint64_t areaBytes{0};
    //! Optional (0: off): multicast address of the two row areas, each [rows][world][localVecs] pairs of 64-bit
    //! elements; a filtered row's local vectors are published there.
    uint64_t mcRows{0};
    uint64_t rowAreaBytes{0};
    uint32_t* counters{nullptr};
    //! Optional (tests): this rank's own states, [rows][kVirtualThreads] at their global virtual thread.
    fs::OnlineSoftmax* ownStates{nullptr};
    //! Optional (bench): [slots][4] globaltimer stamps; slot `slot`: [0] min post-wait, [1] max publish end.
    unsigned long long* stamps{nullptr};
    int slot{0};
    Cap cap{};
    int rank{0};
    int world{1};
    int localVecs{0};
    int localThreads{0};
};

struct PollParams
{
    uint64_t ucArea{0};
    uint64_t areaBytes{0};
    uint32_t* counters{nullptr};
    //! [rows][kVirtualThreads] states at their virtual thread (the layout the served kernel B reads).
    fs::OnlineSoftmax* states{nullptr};
    //! Optional (bench): slot `slot`: [2] max time a CTA saw all its states, [3] max poll-kernel entry.
    unsigned long long* stamps{nullptr};
    int slot{0};
    int world{1};
    int localThreads{0};
    unsigned long long timeoutNs{0};
};

__device__ __forceinline__ void gridDependencyWait()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

__device__ __forceinline__ void gridDependencyTrigger()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

__device__ __forceinline__ unsigned long long globalTimer()
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void publishPair(uint64_t address, unsigned long long a, unsigned long long b)
{
    asm volatile("st.global.v2.b64 [%0], {%1, %2};" ::"l"(address), "l"(a), "l"(b) : "memory");
}

__device__ __forceinline__ void pollPair(uint64_t address, unsigned long long& a, unsigned long long& b)
{
    asm volatile("ld.relaxed.sys.global.v2.b64 {%0, %1}, [%2];" : "=l"(a), "=l"(b) : "l"(address) : "memory");
}

__device__ __forceinline__ bool fresh(unsigned long long a, unsigned long long b, uint32_t epoch)
{
    return static_cast<uint32_t>(a >> 32) == epoch && static_cast<uint32_t>(b >> 32) == epoch;
}

//! Byte offset of state (row, r, a) in one area.
__device__ __forceinline__ uint64_t stateOffset(int row, int r, int a, int world, int localThreads)
{
    return (static_cast<uint64_t>(row * world + r) * localThreads + a) * (kStateElems * 8u);
}

//! The fold of this rank's virtual threads and the publish of their states.
//!
//! Grid (rows, localThreads / VTS); block THREADS >= VTS. The CTA owns the local virtual threads a0 .. a0 + VTS - 1
//! (a0 = blockIdx.y * VTS; global vt = world * a + rank), i.e. the local vectors m = a0 + t + localThreads * j. Phase
//! 1 (all threads, coalesced over t) stores each element's __fmul_rn(capOf(logit), tempInv) at [j][t] in shared
//! memory; phase 2 (thread t < VTS) folds its slots j = 0, 1, ... with splitFoldCoopKernel's statements. FORM 0 runs
//! the coop kernel's two passes as written; FORM 1 runs them (running max / branch weight, then the sum chain) in one
//! loop: the same operations on the same values, each in the same order, only interleaved; FORM 2 is FORM 1 with phase
//! 1's loads all issued before its first cap and phase 2's next slot read from shared memory one slot ahead (again
//! the same values in the same order).
template <int VTS, int THREADS, int FORM>
__global__ void __launch_bounds__(THREADS) s2FoldPublishKernel(FoldParams const p)
{
    static_assert(THREADS >= VTS && THREADS % VTS == 0, "CTA shape");
    static_assert(FORM >= 0 && FORM <= 2, "fold form");
    //! FORM 2: slots one thread loads at most (a virtual thread folds at most kWalkVectors vectors).
    constexpr int kPer = (kWalkVectors * VTS + THREADS - 1) / THREADS;
    extern __shared__ float4 sCapped[];
    __shared__ uint32_t sEpoch;
    gridDependencyWait();
    gridDependencyTrigger();
    if (p.stamps != nullptr && threadIdx.x == 0)
    {
        atomicMin(p.stamps + p.slot * 4 + 0, globalTimer());
    }
    int const row = blockIdx.x;
    int const cta = blockIdx.x * gridDim.y + blockIdx.y;
    if (threadIdx.x == 0)
    {
        sEpoch = p.counters[cta] + 1u;
    }
    int const a0 = blockIdx.y * VTS;
    int const localVecs = p.localVecs;
    int const lt = p.localThreads;
    fs::RowParams const rp = fs::loadRowParams(p.fs, row);
    float const tempInv = rp.tempInv;
    uint2 const* localRow = p.local + static_cast<size_t>(row) * localVecs;
    // Slots of the CTA's first virtual thread: the most any of its threads folds.
    int const steps = localVecs > a0 ? (localVecs - a0 + lt - 1) / lt : 0;
    if constexpr (FORM == 2)
    {
        uint2 packed[kPer];
#pragma unroll
        for (int k = 0; k < kPer; ++k)
        {
            int const i = threadIdx.x + k * THREADS;
            int const j = i / VTS;
            int const m = a0 + (i - j * VTS) + j * lt;
            if (i < steps * VTS && m < localVecs)
            {
                packed[k] = __ldg(localRow + m);
            }
        }
#pragma unroll
        for (int k = 0; k < kPer; ++k)
        {
            int const i = threadIdx.x + k * THREADS;
            int const j = i / VTS;
            int const m = a0 + (i - j * VTS) + j * lt;
            if (i < steps * VTS && m < localVecs)
            {
                __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(&packed[k]);
                float4 x;
                x.x = __fmul_rn(capOf(__bfloat162float(elems[0]), p.cap), tempInv);
                x.y = __fmul_rn(capOf(__bfloat162float(elems[1]), p.cap), tempInv);
                x.z = __fmul_rn(capOf(__bfloat162float(elems[2]), p.cap), tempInv);
                x.w = __fmul_rn(capOf(__bfloat162float(elems[3]), p.cap), tempInv);
                sCapped[i] = x;
            }
        }
    }
    else
    {
        for (int i = threadIdx.x; i < steps * VTS; i += THREADS)
        {
            int const j = i / VTS;
            int const m = a0 + (i - j * VTS) + j * lt;
            if (m < localVecs)
            {
                uint2 const packed = __ldg(localRow + m);
                __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(&packed);
                float4 x;
                x.x = __fmul_rn(capOf(__bfloat162float(elems[0]), p.cap), tempInv);
                x.y = __fmul_rn(capOf(__bfloat162float(elems[1]), p.cap), tempInv);
                x.z = __fmul_rn(capOf(__bfloat162float(elems[2]), p.cap), tempInv);
                x.w = __fmul_rn(capOf(__bfloat162float(elems[3]), p.cap), tempInv);
                sCapped[i] = x;
            }
        }
    }
    __syncthreads();
    uint32_t const epoch = sEpoch;
    if (p.mcRows != 0 && (rp.needTopK || rp.needTopP || rp.needMinP))
    {
        // A filtered row: the draw runs the shipped body on the full row, so publish this CTA's local vectors
        // (the CTA's virtual threads' vectors; the row's CTAs together cover the shard) into every rank's row area.
        unsigned long long const tag = static_cast<unsigned long long>(epoch) << 32;
        uint64_t const dst = p.mcRows + static_cast<uint64_t>(epoch & 1u) * p.rowAreaBytes
            + static_cast<uint64_t>(row * p.world + p.rank) * localVecs * 16u;
        for (int i = threadIdx.x; i < steps * VTS; i += THREADS)
        {
            int const j = i / VTS;
            int const m = a0 + (i - j * VTS) + j * lt;
            if (m < localVecs)
            {
                uint2 const packed = __ldg(localRow + m);
                publishPair(dst + static_cast<uint64_t>(m) * 16u, tag | packed.x, tag | packed.y);
            }
        }
    }
    if (threadIdx.x < VTS)
    {
        int const a = a0 + threadIdx.x;
        int const vt = a * p.world + p.rank;
        int const mine = localVecs > a ? (localVecs - a + lt - 1) / lt : 0;
        float runMax = -FLT_MAX;
        int runArg = 0;
        float runSum = 0.0f;
        if constexpr (FORM == 2)
        {
            float4 next = mine > 0 ? sCapped[threadIdx.x] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            for (int j = 0; j < mine; ++j)
            {
                int const base = (vt + j * kVirtualThreads) * 4;
                float4 const x = next;
                if (j + 1 < mine)
                {
                    next = sCapped[(j + 1) * VTS + threadIdx.x];
                }
                float const xs[4] = {x.x, x.y, x.z, x.w};
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
        }
        else if constexpr (FORM == 1)
        {
            for (int j = 0; j < mine; ++j)
            {
                // Global vector vt + 1024 j = world (a + lt j) + rank: its first element's index.
                int const base = (vt + j * kVirtualThreads) * 4;
                float4 const x = sCapped[j * VTS + threadIdx.x];
                float const xs[4] = {x.x, x.y, x.z, x.w};
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
        }
        else
        {
            for (int j = 0; j < mine; ++j)
            {
                int const base = (vt + j * kVirtualThreads) * 4;
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
        }
        unsigned long long const tag = static_cast<unsigned long long>(epoch) << 32;
        uint64_t const dst
            = p.mcArea + static_cast<uint64_t>(epoch & 1u) * p.areaBytes + stateOffset(row, p.rank, a, p.world, lt);
        publishPair(dst, tag | __float_as_uint(runMax), tag | __float_as_uint(runSum));
        publishPair(dst + 16u, tag | static_cast<uint32_t>(runArg), tag);
        if (p.ownStates != nullptr)
        {
            p.ownStates[static_cast<size_t>(row) * kVirtualThreads + vt] = fs::OnlineSoftmax{runMax, runSum, runArg};
        }
    }
    if (p.stamps != nullptr)
    {
        __syncthreads();
        if (threadIdx.x == 0)
        {
            atomicMax(p.stamps + p.slot * 4 + 1, globalTimer());
        }
    }
    if (threadIdx.x == 0)
    {
        p.counters[cta] = epoch;
    }
}

//! Every rank's states of one row, polled from this rank's copy: thread vt (one CTA per row) waits until the four
//! elements of state (row, vt % world, vt / world) carry the call's epoch, then stores the state at [row][vt].
__global__ void __launch_bounds__(kPollThreads, 1) s2PollStatesKernel(PollParams const p)
{
    __shared__ uint32_t sEpoch;
    int const row = blockIdx.x;
    int const vt = threadIdx.x;
    if (p.stamps != nullptr && threadIdx.x == 0)
    {
        atomicMax(p.stamps + p.slot * 4 + 3, globalTimer());
    }
    if (threadIdx.x == 0)
    {
        sEpoch = p.counters[row] + 1u;
    }
    __syncthreads();
    uint32_t const epoch = sEpoch;
    uint64_t const src = p.ucArea + static_cast<uint64_t>(epoch & 1u) * p.areaBytes
        + stateOffset(row, vt % p.world, vt / p.world, p.world, p.localThreads);
    unsigned long long e0, e1, e2, e3;
    pollPair(src, e0, e1);
    pollPair(src + 16u, e2, e3);
    if (!(fresh(e0, e1, epoch) && fresh(e2, e3, epoch)))
    {
        unsigned long long const start = globalTimer();
        do
        {
            if (globalTimer() - start > p.timeoutNs)
            {
                __trap();
            }
            pollPair(src, e0, e1);
            pollPair(src + 16u, e2, e3);
        } while (!(fresh(e0, e1, epoch) && fresh(e2, e3, epoch)));
    }
    if (p.stamps != nullptr)
    {
        __syncthreads();
        if (threadIdx.x == 0)
        {
            atomicMax(p.stamps + p.slot * 4 + 2, globalTimer());
        }
    }
    // Completion implies the fold grid's (see the header: counter reads of the next call).
    gridDependencyWait();
    gridDependencyTrigger();
    p.states[static_cast<size_t>(row) * kVirtualThreads + vt]
        = fs::OnlineSoftmax{__uint_as_float(static_cast<uint32_t>(e0)), __uint_as_float(static_cast<uint32_t>(e1)),
            static_cast<int>(static_cast<uint32_t>(e2))};
    if (threadIdx.x == 0)
    {
        p.counters[row] = epoch;
    }
}

struct DrawParams
{
    //! Row params, seed / offset and outputTokens of the served kernel B; `logits` is the [rows][vocab] BF16 scratch
    //! row the filtered path fills (and the shipped body then reads).
    tk::FusedSamplingParams fs;
    //! This rank's interleaved shard of the BF16 logits, [rows][localVecs] vectors of 4.
    uint2 const* local{nullptr};
    //! This rank's copy of the two state areas.
    uint64_t ucStates{0};
    uint64_t areaBytes{0};
    //! The two token-slot areas, each [rows][world] entries of two 64-bit elements: this rank's copy, multicast.
    uint64_t ucSlot{0};
    uint64_t mcSlot{0};
    uint64_t slotBytes{0};
    //! This rank's copy of the two row areas (see FoldParams::mcRows).
    uint64_t ucRows{0};
    uint64_t rowAreaBytes{0};
    //! One epoch counter per row (the draw CTA's).
    uint32_t* counters{nullptr};
    //! Optional (bench): slot `slot`: [2] max time a CTA saw all its states, [3] max time a CTA stored its token.
    unsigned long long* stamps{nullptr};
    int slot{0};
    //! Tests only: a neutral row with one claimant takes the several-claimants path (the same token).
    int forceMulti{0};
    Cap cap{};
    int rank{0};
    int world{1};
    int localVecs{0};
    int localThreads{0};
    unsigned long long timeoutNs{0};
};

//! Spin until the 64-bit pair at `address` carries `epoch` in both elements; a wait past the timeout traps.
__device__ __forceinline__ void awaitPair(
    uint64_t address, uint32_t epoch, unsigned long long timeoutNs, unsigned long long& a, unsigned long long& b)
{
    pollPair(address, a, b);
    if (!fresh(a, b, epoch))
    {
        unsigned long long const start = globalTimer();
        do
        {
            if (globalTimer() - start > timeoutNs)
            {
                __trap();
            }
            pollPair(address, a, b);
        } while (!fresh(a, b, epoch));
    }
}

//! The draw of one row on every rank (see the file header). Grid rows; block kVirtualThreads.
template <bool kTrigger>
__global__ void __launch_bounds__(kVirtualThreads, 1) s2DrawKernel(DrawParams const p)
{
    using BlockReduceOnline = cub::BlockReduce<fs::OnlineSoftmax, kVirtualThreads>;
    using BlockScanF = cub::BlockScan<float, kVirtualThreads>;
    __shared__ fs::FusedSamplingShared<kVirtualThreads> shared;
    __shared__ __align__(16) float wbuf[4 * kWalkVectors];
    __shared__ int sWalker;
    __shared__ float sBase;
    __shared__ unsigned long long sLastCrossing;
    __shared__ int sCrossIdx;
    __shared__ uint32_t sOwners;
    __shared__ uint32_t sEpoch;
    int const row = blockIdx.x;
    int const tid = threadIdx.x;
    if constexpr (kTrigger)
    {
        gridDependencyTrigger();
    }
    if (tid == 0)
    {
        fs::sSplitCap = fs::SplitCap{p.cap.multiplier, p.cap.invCap, p.cap.cap};
        sEpoch = p.counters[row] + 1u;
    }
    __syncthreads();
    uint32_t const epoch = sEpoch;
    uint64_t const parity = epoch & 1u;
    int const world = p.world;
    int const vocabSize = p.fs.vocabSize;
    int const vecCount = vocabSize / 4;
    fs::RowParams const rp = fs::loadRowParams(p.fs, row);
    if (rp.needTopK || rp.needTopP || rp.needMinP)
    {
        // Every rank's local vectors of this row, from the row area, into the scratch row in vocabulary order:
        // global vector v is rank (v % world)'s local vector v / world.
        uint64_t const src
            = p.ucRows + parity * p.rowAreaBytes + static_cast<uint64_t>(row) * world * p.localVecs * 16u;
        uint2* const dst
            = reinterpret_cast<uint2*>(const_cast<void*>(p.fs.logits)) + static_cast<size_t>(row) * vecCount;
        for (int v = tid; v < vecCount; v += kVirtualThreads)
        {
            unsigned long long a, b;
            awaitPair(
                src + (static_cast<uint64_t>(v % world) * p.localVecs + v / world) * 16u, epoch, p.timeoutNs, a, b);
            dst[v] = make_uint2(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
        }
        __syncthreads();
        fs::fusedSamplingBody<fs::CapBf16, kVirtualThreads, true, false>(p.fs, shared);
    }
    else
    {
        // Kernel B's state load, as a poll of this rank's copy of the state areas.
        unsigned long long e0, e1, e2, e3;
        uint64_t const src
            = p.ucStates + parity * p.areaBytes + stateOffset(row, tid % world, tid / world, world, p.localThreads);
        pollPair(src, e0, e1);
        pollPair(src + 16u, e2, e3);
        if (!(fresh(e0, e1, epoch) && fresh(e2, e3, epoch)))
        {
            unsigned long long const start = globalTimer();
            do
            {
                if (globalTimer() - start > p.timeoutNs)
                {
                    __trap();
                }
                pollPair(src, e0, e1);
                pollPair(src + 16u, e2, e3);
            } while (!(fresh(e0, e1, epoch) && fresh(e2, e3, epoch)));
        }
        fs::OnlineSoftmax const cachedThreadStats{__uint_as_float(static_cast<uint32_t>(e0)),
            __uint_as_float(static_cast<uint32_t>(e1)), static_cast<int>(static_cast<uint32_t>(e2))};
        if (p.stamps != nullptr)
        {
            __syncthreads();
            if (tid == 0)
            {
                atomicMax(p.stamps + p.slot * 4 + 2, globalTimer());
            }
        }
        // ---- From here to the walk: kernel B's statements (head_sampler.cu splitDrawKernel), in its order.
        fs::OnlineSoftmax const rowStats
            = BlockReduceOnline(shared.temp.reduceOnline).Reduce(cachedThreadStats, fs::OnlineSoftmaxOp());
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
            fs::initRowRng(p.fs, row, &shared.rejectRng);
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
            sCrossIdx = -1;
            sOwners = 0u;
        }
        __syncthreads();
        float const target = shared.target;
        bool const claims = target >= base && target < base + localMass;
        if (claims)
        {
            sWalker = tid;
            sBase = base;
            atomicOr(&sOwners, 1u << (tid % world));
        }
        int const claimants = __syncthreads_count(claims);
        // ---- The walk: on the rank holding the walker's vectors; the result through the token slot.
        uint2 const* const localRow = p.local + static_cast<size_t>(row) * p.localVecs;
        uint64_t const slotRow = parity * p.slotBytes + static_cast<uint64_t>(row) * world * 16u;
        unsigned long long const tag = static_cast<unsigned long long>(epoch) << 32;
        int token = -1;
        if (claimants == 1 && p.forceMulti == 0)
        {
            int const walker = sWalker;
            int const owner = walker % world;
            if (owner == p.rank)
            {
                // Kernel B's single-walker walk on the local copy of the walker's vectors.
                int const walkVectors = (vecCount - walker + kVirtualThreads - 1) / kVirtualThreads;
                if (tid < walkVectors)
                {
                    uint2 const packed = localRow[walker / world + tid * p.localThreads];
                    __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(&packed);
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                    {
                        wbuf[tid * 4 + j]
                            = __expf(__fmul_rn(capOf(__bfloat162float(elems[j]), p.cap), rp.tempInv) - maxScaled);
                    }
                }
                __syncthreads();
                if (tid == 0)
                {
                    float running = sBase;
                    int cand = -1;
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
                                    cand = (walker + q * kVirtualThreads) * 4 + j;
                                }
                            }
                        }
                    }
                    publishPair(p.mcSlot + slotRow + static_cast<uint64_t>(p.rank) * 16u,
                        tag | static_cast<uint32_t>(cand), tag);
                    token = cand;
                }
            }
            else if (tid == 0)
            {
                unsigned long long a, b;
                awaitPair(p.ucSlot + slotRow + static_cast<uint64_t>(owner) * 16u, epoch, p.timeoutNs, a, b);
                token = static_cast<int>(static_cast<uint32_t>(a));
            }
        }
        else if (claimants >= 1)
        {
            // Kernel B's several-claimants walk, each claimant on the rank holding its vectors.
            int crossIdx = -1;
            unsigned long long crossing = 0ull;
            if (claims && tid % world == p.rank)
            {
                float running = base;
                int position = 0;
                for (int q = 0; tid + q * kVirtualThreads < vecCount && crossIdx < 0; ++q)
                {
                    int const v = tid + q * kVirtualThreads;
                    uint2 const packed = localRow[tid / world + q * p.localThreads];
                    __nv_bfloat16 const* elems = reinterpret_cast<__nv_bfloat16 const*>(&packed);
                    for (int j = 0; j < 4; ++j)
                    {
                        float const w
                            = __expf(__fmul_rn(capOf(__bfloat162float(elems[j]), p.cap), rp.tempInv) - maxScaled);
                        ++position;
                        if (!(w >= 0.0f && w > -1.0f))
                        {
                            continue;
                        }
                        running += w;
                        if (running > target)
                        {
                            crossIdx = v * 4 + j;
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
            }
            __syncthreads();
            if (tid == 0)
            {
                uint32_t const owners = sOwners;
                if ((owners >> p.rank) & 1u)
                {
                    unsigned long long const best = sLastCrossing;
                    uint32_t const key
                        = best == 0ull ? 0u : static_cast<uint32_t>((best >> 32) * 2048ull + (best & 0xFFFFFFFFull));
                    int const idx = best == 0ull ? -1 : sCrossIdx;
                    publishPair(p.mcSlot + slotRow + static_cast<uint64_t>(p.rank) * 16u,
                        tag | static_cast<uint32_t>(idx), tag | key);
                }
                uint32_t bestKey = 0u;
                for (int r = 0; r < world; ++r)
                {
                    if (((owners >> r) & 1u) == 0u)
                    {
                        continue;
                    }
                    unsigned long long a, b;
                    awaitPair(p.ucSlot + slotRow + static_cast<uint64_t>(r) * 16u, epoch, p.timeoutNs, a, b);
                    int const idx = static_cast<int>(static_cast<uint32_t>(a));
                    uint32_t const key = static_cast<uint32_t>(b);
                    if (idx >= 0 && key > bestKey)
                    {
                        bestKey = key;
                        token = idx;
                    }
                }
            }
        }
        if (tid == 0)
        {
            p.fs.outputTokens[row] = token >= 0 ? token : shared.argmax;
            if (p.stamps != nullptr)
            {
                atomicMax(p.stamps + p.slot * 4 + 3, globalTimer());
            }
        }
    }
    // Completion implies the fold grid's (the next call's counter reads).
    gridDependencyWait();
    if (tid == 0)
    {
        p.counters[row] = epoch;
    }
}

//! Bench helper: one thread stores the globaltimer into stamps[index] after its PDL wait.
__global__ void s2StampKernel(unsigned long long* stamps, int index)
{
    gridDependencyWait();
    stamps[index] = globalTimer();
    gridDependencyTrigger();
}

__global__ void s2DryKernel() {}

template <typename Kernel, typename... Args>
void launchEx(Kernel kernel, dim3 grid, dim3 block, size_t smem, bool pdl, Args... args)
{
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem;
    config.stream = at::cuda::getCurrentCUDAStream();
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    config.attrs = attribute;
    config.numAttrs = 1;
    C10_CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, args...));
}

template <int VTS, int THREADS, int FORM>
void launchFoldShape(FoldParams const& p, int rows, bool pdl)
{
    auto kernel = s2FoldPublishKernel<VTS, THREADS, FORM>;
    int const steps = (p.localVecs + p.localThreads - 1) / p.localThreads;
    size_t const smem = static_cast<size_t>(steps) * VTS * sizeof(float4);
    static bool const raised = []
    {
        size_t const most = static_cast<size_t>(kWalkVectors) * VTS * sizeof(float4);
        C10_CUDA_CHECK(cudaFuncSetAttribute(s2FoldPublishKernel<VTS, THREADS, FORM>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(most)));
        return true;
    }();
    (void) raised;
    launchEx(kernel, dim3(rows, p.localThreads / VTS), dim3(THREADS), smem, pdl, p);
}

template <int FORM>
void launchFold(FoldParams const& p, int rows, int64_t vts, int64_t threads, bool pdl)
{
#define S2_FOLD_SHAPE(V, T)                                                                                            \
    if (vts == V && threads == T)                                                                                      \
    {                                                                                                                  \
        launchFoldShape<V, T, FORM>(p, rows, pdl);                                                                     \
        return;                                                                                                        \
    }
    S2_FOLD_SHAPE(2, 128)
    S2_FOLD_SHAPE(4, 64)
    S2_FOLD_SHAPE(4, 128)
    S2_FOLD_SHAPE(4, 256)
    S2_FOLD_SHAPE(8, 128)
    S2_FOLD_SHAPE(8, 256)
    S2_FOLD_SHAPE(16, 256)
    S2_FOLD_SHAPE(8, 512)
    S2_FOLD_SHAPE(16, 512)
    S2_FOLD_SHAPE(32, 512)
#undef S2_FOLD_SHAPE
    TORCH_CHECK(false, "s2 fold: unsupported (virtual threads per CTA, threads) = (", vts, ", ", threads, ")");
}

int64_t foldCtas(int64_t rows, int64_t world, int64_t vts)
{
    return rows * (kVirtualThreads / world) / vts;
}

void checkCounters(at::Tensor const& counters, int64_t need, char const* name)
{
    TORCH_CHECK(counters.is_cuda() && counters.scalar_type() == at::kInt && counters.is_contiguous()
            && counters.numel() >= need,
        "s2: ", name, " must be a contiguous int32 CUDA tensor of at least ", need, " entries");
}

void checkStamps(std::optional<at::Tensor> const& stamps, int64_t slot)
{
    if (stamps.has_value())
    {
        auto const& s = stamps.value();
        TORCH_CHECK(s.is_cuda() && s.scalar_type() == at::kLong && s.is_contiguous() && s.dim() == 2 && s.size(1) == 4
                && slot >= 0 && slot < s.size(0),
            "s2: stamps must be a contiguous int64 [slots, 4] CUDA tensor and slot inside it");
    }
}

void checkRowParam(at::Tensor const& t, char const* name, at::ScalarType dtype, int64_t rows)
{
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == dtype && t.dim() == 1 && t.size(0) >= rows && t.is_contiguous(),
        "s2: ", name, " must be a contiguous 1-D CUDA tensor of the right dtype with at least numRows entries");
}

//! Bytes of one row area: every rank's local vectors of `rows` rows, one pair of 64-bit elements per vector.
int64_t rowAreaNeed(int64_t rows, int64_t world, int64_t localVecs)
{
    return rows * world * localVecs * 16;
}

//! This rank's fold of its virtual threads and the publish of their states (see s2FoldPublishKernel).
void foldPublish(at::Tensor const& local, at::Tensor const& temperatures, at::Tensor const& counters, int64_t mcArea,
    int64_t areaBytes, int64_t rank, int64_t world, double multiplier, double invCap, double cap, int64_t vts,
    int64_t threads, int64_t form, bool pdl, std::optional<at::Tensor> const& ownStates,
    std::optional<at::Tensor> const& stamps, int64_t slot, std::optional<at::Tensor> const& topKs,
    std::optional<at::Tensor> const& topPs, std::optional<at::Tensor> const& minPs, int64_t mcRows,
    int64_t rowAreaBytes)
{
    at::cuda::CUDAGuard guard(local.device());
    TORCH_CHECK(local.is_cuda() && local.scalar_type() == at::kBFloat16 && local.dim() == 2 && local.is_contiguous(),
        "s2 fold: local must be a contiguous BF16 [rows, shard] CUDA tensor");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(local.data_ptr()) % 8 == 0, "s2 fold: local must be 8-byte aligned");
    int64_t const rows = local.size(0);
    int64_t const shard = local.size(1);
    TORCH_CHECK(rows >= 1 && rows <= fs::kWideBlockMaxRows, "s2 fold: 1..", fs::kWideBlockMaxRows, " rows");
    TORCH_CHECK(world >= 1 && kVirtualThreads % world == 0 && rank >= 0 && rank < world, "s2 fold: bad rank / world");
    TORCH_CHECK(shard % 4 == 0, "s2 fold: the shard is a whole number of 4-element vectors");
    int64_t const localThreads = kVirtualThreads / world;
    int64_t const localVecs = shard / 4;
    TORCH_CHECK((localVecs + localThreads - 1) / localThreads <= kWalkVectors, "s2 fold: vocabulary too wide");
    TORCH_CHECK(vts >= 1 && localThreads % vts == 0, "s2 fold: virtual threads per CTA must divide ", localThreads);
    TORCH_CHECK(temperatures.is_cuda() && temperatures.scalar_type() == at::kFloat && temperatures.dim() == 1
            && temperatures.size(0) >= rows && temperatures.is_contiguous(),
        "s2 fold: temperatures must be a contiguous FP32 [>= rows] CUDA tensor");
    checkCounters(counters, foldCtas(rows, world, vts), "fold counters");
    TORCH_CHECK(mcArea % 16 == 0, "s2 fold: misaligned area");
    TORCH_CHECK(areaBytes >= rows * kVirtualThreads * kStateElems * 8, "s2 fold: area too small");
    if (ownStates.has_value())
    {
        auto const& s = ownStates.value();
        TORCH_CHECK(
            s.is_cuda() && s.scalar_type() == at::kInt && s.is_contiguous() && s.numel() >= rows * kVirtualThreads * 3,
            "s2 fold: own_states holds 3 words per virtual thread");
    }
    checkStamps(stamps, slot);
    FoldParams p;
    if (mcRows != 0)
    {
        TORCH_CHECK(topKs.has_value() && topPs.has_value() && minPs.has_value(),
            "s2 fold: the row area needs top_ks / top_ps / min_ps (they decide which rows are filtered)");
        checkRowParam(topKs.value(), "top_ks", at::kInt, rows);
        checkRowParam(topPs.value(), "top_ps", at::kFloat, rows);
        checkRowParam(minPs.value(), "min_ps", at::kFloat, rows);
        TORCH_CHECK(mcRows % 16 == 0 && rowAreaBytes >= rowAreaNeed(rows, world, shard / 4),
            "s2 fold: misaligned or too small row area");
        p.fs.topKs = topKs.value().data_ptr<int32_t>();
        p.fs.topPs = topPs.value().data_ptr<float>();
        p.fs.minPs = minPs.value().data_ptr<float>();
        p.mcRows = static_cast<uint64_t>(mcRows);
        p.rowAreaBytes = static_cast<uint64_t>(rowAreaBytes);
    }
    p.fs.temperatures = temperatures.data_ptr<float>();
    p.fs.numRows = static_cast<int32_t>(rows);
    p.fs.vocabSize = static_cast<int32_t>(shard * world);
    p.local = reinterpret_cast<uint2 const*>(local.data_ptr());
    p.mcArea = static_cast<uint64_t>(mcArea);
    p.areaBytes = static_cast<uint64_t>(areaBytes);
    p.counters = reinterpret_cast<uint32_t*>(counters.data_ptr());
    p.ownStates = ownStates.has_value() ? reinterpret_cast<fs::OnlineSoftmax*>(ownStates.value().data_ptr()) : nullptr;
    p.stamps = stamps.has_value() ? reinterpret_cast<unsigned long long*>(stamps.value().data_ptr()) : nullptr;
    p.slot = static_cast<int>(slot);
    p.cap = Cap{static_cast<float>(multiplier), static_cast<float>(invCap), static_cast<float>(cap)};
    p.rank = static_cast<int>(rank);
    p.world = static_cast<int>(world);
    p.localVecs = static_cast<int>(localVecs);
    p.localThreads = static_cast<int>(localThreads);
    if (form == 2)
    {
        launchFold<2>(p, static_cast<int>(rows), vts, threads, pdl);
    }
    else if (form == 1)
    {
        launchFold<1>(p, static_cast<int>(rows), vts, threads, pdl);
    }
    else
    {
        TORCH_CHECK(form == 0, "s2 fold: form 0, 1 or 2");
        launchFold<0>(p, static_cast<int>(rows), vts, threads, pdl);
    }
}

//! Every rank's states of `rows` rows into `states` ([rows][1024] OnlineSoftmax, 3 int32 words each).
void pollStates(at::Tensor const& states, at::Tensor const& counters, int64_t ucArea, int64_t areaBytes, int64_t rows,
    int64_t world, int64_t timeoutNs, bool pdl, std::optional<at::Tensor> const& stamps, int64_t slot)
{
    at::cuda::CUDAGuard guard(states.device());
    TORCH_CHECK(states.is_cuda() && states.scalar_type() == at::kInt && states.is_contiguous()
            && states.numel() >= rows * kVirtualThreads * 3,
        "s2 poll: states holds 3 words per virtual thread");
    TORCH_CHECK(rows >= 1 && rows <= fs::kWideBlockMaxRows, "s2 poll: 1..", fs::kWideBlockMaxRows, " rows");
    TORCH_CHECK(world >= 1 && kVirtualThreads % world == 0, "s2 poll: bad world");
    checkCounters(counters, rows, "poll counters");
    TORCH_CHECK(ucArea % 16 == 0, "s2 poll: misaligned area");
    TORCH_CHECK(areaBytes >= rows * kVirtualThreads * kStateElems * 8, "s2 poll: area too small");
    checkStamps(stamps, slot);
    PollParams p;
    p.ucArea = static_cast<uint64_t>(ucArea);
    p.areaBytes = static_cast<uint64_t>(areaBytes);
    p.counters = reinterpret_cast<uint32_t*>(counters.data_ptr());
    p.states = reinterpret_cast<fs::OnlineSoftmax*>(states.data_ptr());
    p.stamps = stamps.has_value() ? reinterpret_cast<unsigned long long*>(stamps.value().data_ptr()) : nullptr;
    p.slot = static_cast<int>(slot);
    p.world = static_cast<int>(world);
    p.localThreads = static_cast<int>(kVirtualThreads / world);
    p.timeoutNs = static_cast<unsigned long long>(timeoutNs);
    launchEx(s2PollStatesKernel, dim3(rows), dim3(kPollThreads), 0, pdl, p);
}

//! The draw of `rows` rows on this rank (see s2DrawKernel); the int32 [rows] tokens. `scratch` is the
//! [rows, world * shard] BF16 row the filtered path fills. The row-parameter and seed / offset checks are the served
//! draw's (head_sampler.cu rowParams).
at::Tensor draw(at::Tensor const& local, at::Tensor const& scratch, at::Tensor const& temperatures,
    at::Tensor const& topKs, at::Tensor const& topPs, at::Tensor const& minPs, std::optional<at::Tensor> const& seed,
    std::optional<at::Tensor> const& offset, at::Tensor const& counters, int64_t ucStates, int64_t areaBytes,
    int64_t ucSlot, int64_t mcSlot, int64_t slotBytes, int64_t ucRows, int64_t rowAreaBytes, int64_t rank,
    int64_t world, double multiplier, double invCap, double cap, int64_t timeoutNs, bool pdl, bool trigger,
    std::optional<at::Tensor> const& stamps, int64_t slot, int64_t forceMulti)
{
    at::cuda::CUDAGuard guard(local.device());
    TORCH_CHECK(local.is_cuda() && local.scalar_type() == at::kBFloat16 && local.dim() == 2 && local.is_contiguous(),
        "s2 draw: local must be a contiguous BF16 [rows, shard] CUDA tensor");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(local.data_ptr()) % 8 == 0, "s2 draw: local must be 8-byte aligned");
    int64_t const rows = local.size(0);
    int64_t const shard = local.size(1);
    TORCH_CHECK(rows >= 1 && rows <= fs::kWideBlockMaxRows, "s2 draw: 1..", fs::kWideBlockMaxRows, " rows");
    TORCH_CHECK(world >= 1 && kVirtualThreads % world == 0 && rank >= 0 && rank < world, "s2 draw: bad rank / world");
    TORCH_CHECK(world <= 32, "s2 draw: the claimant mask holds 32 ranks");
    TORCH_CHECK(shard % 4 == 0, "s2 draw: the shard is a whole number of 4-element vectors");
    int64_t const localThreads = kVirtualThreads / world;
    int64_t const localVecs = shard / 4;
    TORCH_CHECK((localVecs + localThreads - 1) / localThreads <= kWalkVectors, "s2 draw: vocabulary too wide");
    TORCH_CHECK(scratch.is_cuda() && scratch.scalar_type() == at::kBFloat16 && scratch.dim() == 2
            && scratch.is_contiguous() && scratch.size(0) >= rows && scratch.size(1) == shard * world,
        "s2 draw: scratch must be a contiguous BF16 [>= rows, world * shard] CUDA tensor");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(scratch.data_ptr()) % 8 == 0, "s2 draw: scratch must be 8-byte aligned");
    checkRowParam(temperatures, "temperatures", at::kFloat, rows);
    checkRowParam(topKs, "top_ks", at::kInt, rows);
    checkRowParam(topPs, "top_ps", at::kFloat, rows);
    checkRowParam(minPs, "min_ps", at::kFloat, rows);
    checkCounters(counters, rows, "draw counters");
    TORCH_CHECK(ucStates % 16 == 0 && ucSlot % 16 == 0 && mcSlot % 16 == 0 && ucRows % 16 == 0, "s2 draw: misaligned");
    TORCH_CHECK(areaBytes >= rows * kVirtualThreads * kStateElems * 8, "s2 draw: state area too small");
    TORCH_CHECK(slotBytes >= rows * world * 16, "s2 draw: token slot too small");
    TORCH_CHECK(rowAreaBytes >= rowAreaNeed(rows, world, localVecs), "s2 draw: row area too small");
    checkStamps(stamps, slot);
    DrawParams p;
    p.fs.logits = scratch.data_ptr();
    p.fs.temperatures = temperatures.data_ptr<float>();
    p.fs.topKs = topKs.data_ptr<int32_t>();
    p.fs.topPs = topPs.data_ptr<float>();
    p.fs.minPs = minPs.data_ptr<float>();
    p.fs.numRows = static_cast<int32_t>(rows);
    p.fs.vocabSize = static_cast<int32_t>(shard * world);
    TORCH_CHECK(seed.has_value() == offset.has_value(), "seed and offset must be provided together");
    if (seed.has_value())
    {
        auto const& s = seed.value();
        auto const& o = offset.value();
        TORCH_CHECK(s.is_cuda() && o.is_cuda(), "seed and offset must be CUDA tensors");
        TORCH_CHECK(s.scalar_type() == at::kLong && o.scalar_type() == at::kLong, "seed and offset must be int64");
        TORCH_CHECK(s.numel() == o.numel(), "seed and offset must have the same length");
        TORCH_CHECK(s.numel() == 1 || s.numel() >= rows, "seed/offset must hold 1 or numRows entries");
        p.fs.seed = reinterpret_cast<uint64_t const*>(s.data_ptr<int64_t>());
        p.fs.offset = reinterpret_cast<uint64_t const*>(o.data_ptr<int64_t>());
        p.fs.perRowRng = s.numel() >= rows;
    }
    auto tokens = at::empty({rows}, local.options().dtype(at::kInt));
    p.fs.outputTokens = tokens.data_ptr<int32_t>();
    p.local = reinterpret_cast<uint2 const*>(local.data_ptr());
    p.ucStates = static_cast<uint64_t>(ucStates);
    p.areaBytes = static_cast<uint64_t>(areaBytes);
    p.ucSlot = static_cast<uint64_t>(ucSlot);
    p.mcSlot = static_cast<uint64_t>(mcSlot);
    p.slotBytes = static_cast<uint64_t>(slotBytes);
    p.ucRows = static_cast<uint64_t>(ucRows);
    p.rowAreaBytes = static_cast<uint64_t>(rowAreaBytes);
    p.counters = reinterpret_cast<uint32_t*>(counters.data_ptr());
    p.stamps = stamps.has_value() ? reinterpret_cast<unsigned long long*>(stamps.value().data_ptr()) : nullptr;
    p.slot = static_cast<int>(slot);
    p.forceMulti = static_cast<int>(forceMulti);
    p.cap = Cap{static_cast<float>(multiplier), static_cast<float>(invCap), static_cast<float>(cap)};
    p.rank = static_cast<int>(rank);
    p.world = static_cast<int>(world);
    p.localVecs = static_cast<int>(localVecs);
    p.localThreads = static_cast<int>(localThreads);
    p.timeoutNs = static_cast<unsigned long long>(timeoutNs);
    if (trigger)
    {
        launchEx(s2DrawKernel<true>, dim3(rows), dim3(kVirtualThreads), 0, pdl, p);
    }
    else
    {
        launchEx(s2DrawKernel<false>, dim3(rows), dim3(kVirtualThreads), 0, pdl, p);
    }
    return tokens;
}

void stamp(at::Tensor const& stamps, int64_t index, bool pdl)
{
    at::cuda::CUDAGuard guard(stamps.device());
    TORCH_CHECK(stamps.is_cuda() && stamps.scalar_type() == at::kLong && stamps.is_contiguous() && index >= 0
            && index < stamps.numel(),
        "s2 stamp: contiguous int64 CUDA stamps and an index inside them");
    launchEx(s2StampKernel, dim3(1), dim3(1), 0, pdl, reinterpret_cast<unsigned long long*>(stamps.data_ptr()),
        static_cast<int>(index));
}

void dry(at::Tensor const& anchor)
{
    at::cuda::CUDAGuard guard(anchor.device());
    launchEx(s2DryKernel, dim3(1), dim3(1), 0, false);
}

} // namespace

TORCH_LIBRARY(musespark_open_s2_sampler, m)
{
    m.def(
        "fold_publish(Tensor local, Tensor temperatures, Tensor(a!) counters, int mc_area, int area_bytes, int rank, "
        "int world, float multiplier, float inv_cap, float cap, int vts, int threads, int form, bool pdl, "
        "Tensor(b!)? own_states, Tensor(c!)? stamps, int slot, Tensor? top_ks=None, Tensor? top_ps=None, "
        "Tensor? min_ps=None, int mc_rows=0, int row_area_bytes=0) -> ()");
    m.def(
        "poll_states(Tensor(a!) states, Tensor(b!) counters, int uc_area, int area_bytes, int rows, int world, "
        "int timeout_ns, bool pdl, Tensor(c!)? stamps, int slot) -> ()");
    m.def(
        "draw(Tensor local, Tensor(a!) scratch, Tensor temperatures, Tensor top_ks, Tensor top_ps, Tensor min_ps, "
        "Tensor? seed, Tensor? offset, Tensor(b!) counters, int uc_states, int area_bytes, int uc_slot, int mc_slot, "
        "int slot_bytes, int uc_rows, int row_area_bytes, int rank, int world, float multiplier, float inv_cap, "
        "float cap, int timeout_ns, bool pdl, bool trigger, Tensor(c!)? stamps, int slot, int force_multi=0) "
        "-> Tensor");
    m.def("stamp(Tensor(a!) stamps, int index, bool pdl) -> ()");
    m.def("dry(Tensor anchor) -> ()");
}

TORCH_LIBRARY_IMPL(musespark_open_s2_sampler, CUDA, m)
{
    m.impl("fold_publish", &foldPublish);
    m.impl("poll_states", &pollStates);
    m.impl("draw", &draw);
    m.impl("stamp", &stamp);
    m.impl("dry", &dry);
}
