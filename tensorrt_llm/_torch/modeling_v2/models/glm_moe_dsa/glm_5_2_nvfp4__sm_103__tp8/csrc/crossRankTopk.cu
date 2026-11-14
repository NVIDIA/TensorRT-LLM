/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/csrc/crossRankTopk.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace cross_rank_topk
{
namespace
{

constexpr int32_t kBins = 2048;
static_assert(kBins >= kTopK, "the output row is staged in the histogram");
constexpr int32_t kWarps = kThreads / 32;
//! Per-warp key statistics kept in shared memory (see KeyStats).
constexpr int32_t kNumKeyStats = 5;
//! Tokens of each page one source holds.
constexpr int32_t kBlockTokens = kPageTokens / kSourcesPerRank;
//! Static shared memory of the kernel (round-2 segment tables), kept out of the dynamic budget.
constexpr size_t kStaticSmemBytes = 1024;

// Scalar slots in shared memory.
enum ScalarSlot : int32_t
{
    kSlotBin = 0,
    kSlotAbove,
    kSlotBinCount,
    kSlotTotal,
    kSlotCounter2,
    kSlotSrcCount,                               // [kMaxSources]
    kSlotSrcBound = kSlotSrcCount + kMaxSources, // [kMaxSources]
    kSlotSrcLen = kSlotSrcBound + kMaxSources,   // [kMaxSources]
    kSlotSrcOffset = kSlotSrcLen + kMaxSources,  // [kMaxSources + 1]
    kNumScalarSlots = kSlotSrcOffset + kMaxSources + 1
};

__host__ __device__ inline int32_t numSources(int32_t world)
{
    return world * kSourcesPerRank;
}

// Even share of kTopK per source.
__host__ __device__ inline int32_t evenShare(int32_t world)
{
    return (kTopK + numSources(world) - 1) / numSources(world);
}

// Most keys a source sends in round 1 (an even number).
__host__ __device__ inline int32_t roundCap(int32_t world)
{
    return min(kSourceCapacity, 4 * evenShare(world));
}

// Keys one source can hold: one block of every local page.
__host__ __device__ inline int32_t chunkCapacity(int32_t localCapacity)
{
    int32_t const pages = (localCapacity + kPageTokens - 1) / kPageTokens;
    return pages * kBlockTokens;
}

__host__ __device__ inline int32_t mergeCapacity(int32_t world)
{
    return numSources(world) * roundCap(world);
}

__host__ __device__ inline size_t smemBytes(int32_t localCapacity, int32_t world)
{
    size_t words = 2 * size_t{kBins} + 2 * size_t{kWarps} + size_t{kNumKeyStats} * kWarps + size_t{kNumScalarSlots};
    words = (words + 3) / 4 * 4;
    words += static_cast<size_t>(chunkCapacity(localCapacity)) + 2 * static_cast<size_t>(mergeCapacity(world));
    return words * sizeof(uint32_t);
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
// The kernel targets SM90+ (multicast Lamport exchange).
__global__ void __launch_bounds__(kThreads, 1) cpTopkKernel(CpTopkParams p)
{
    __trap();
}
#else

//! Bits of one radix-select digit (kBins bins).
constexpr int32_t kDigitBits = 11;
//! Order-preserving key of +0.0: the keys of non-negative values are at least this, those of negative values below.
//! Real indexer logits are mostly negative, and the cut of a row (and of many sources) often lies among small
//! negative values while the largest keys are positive: one key range across zero spans ~2^31 key units, where a
//! radix digit has no resolution. The selects therefore work on one sign at a time: with fewer than `need` keys at
//! or above kPositiveKey these are all taken and the cut is searched among the negative keys only, else among the
//! non-negative keys only; each range is narrow enough for one or two passes.
constexpr uint32_t kPositiveKey = 0x80000000u;
//! Independent 16-byte candidate loads one merge thread issues before it waits on any of them.
constexpr int32_t kPollBatch = 4;
//! Polls of a still-empty Lamport word between two reads of the global timer.
constexpr int32_t kPollsPerTimerRead = 256;

// Keys a saturated source aims to send in round 1 (kSourceBudget at 8 ranks).
__device__ __forceinline__ int32_t roundNeed(int32_t world)
{
    return min(kSourceCapacity, 3 * evenShare(world));
}

// Tokens of `rank` among the global tokens [0, n).
__host__ __device__ __forceinline__ int32_t localLength(int32_t n, int32_t rank, int32_t world)
{
    if (n <= 0)
    {
        return 0;
    }
    int32_t const fullPages = n / kPageTokens;
    int32_t const rem = n % kPageTokens;
    int32_t const ownedFull = (fullPages - rank + world - 1) / world;
    int32_t len = ownedFull * kPageTokens;
    if (rem > 0 && fullPages % world == rank)
    {
        len += rem;
    }
    return len;
}

// Source `src` of a rank holds one block of kBlockTokens consecutive tokens of every local page: block
// (src + page) % kSourcesPerRank. A fully selected page is thus spread over all sources of its rank.
// Keys of source `src` among `localLen` local columns (only the last local page can be partial).
__host__ __device__ __forceinline__ int32_t sourceLength(int32_t localLen, int32_t src)
{
    int32_t const pages = (localLen + kPageTokens - 1) / kPageTokens;
    if (pages == 0)
    {
        return 0;
    }
    int32_t const last = pages - 1;
    int32_t const tail = localLen - last * kPageTokens - ((src + last) % kSourcesPerRank) * kBlockTokens;
    return last * kBlockTokens + min(max(tail, 0), kBlockTokens);
}

// Global token of a local column under the page-interleaved partition.
__device__ __forceinline__ uint32_t localToGlobal(int32_t col, int32_t rank, int32_t world)
{
    int32_t const page = col / kPageTokens;
    return static_cast<uint32_t>((page * world + rank) * kPageTokens + (col % kPageTokens));
}

struct SmemLayout
{
    uint32_t* hist;     // [2 * kBins] radix histograms of alternate passes; the first one stages the output row
    uint32_t* warpSums; // [2 * kWarps]
    uint32_t* stats;    // [kNumKeyStats * kWarps] per-warp key statistics
    uint32_t* scal;     // [kNumScalarSlots]
    uint32_t* chunk;    // [chunkCapacity] order-preserving keys of this source
    uint32_t* mkey;     // [mergeCapacity] staged round-1 pairs, then the merge candidates (merge CTA)
    uint32_t* midx;     // [mergeCapacity]
};

__device__ inline SmemLayout carveSmem(uint32_t* base, int32_t localCapacity, int32_t world)
{
    SmemLayout s;
    uint32_t* p = base;
    s.hist = p;
    p += 2 * kBins;
    s.warpSums = p;
    p += 2 * kWarps;
    s.stats = p;
    p += kNumKeyStats * kWarps;
    s.scal = p;
    p += kNumScalarSlots;
    p = base + ((p - base) + 3) / 4 * 4;
    s.chunk = p;
    p += chunkCapacity(localCapacity);
    s.mkey = p;
    p += mergeCapacity(world);
    s.midx = p;
    return s;
}

__device__ __forceinline__ uint32_t orderedKey(float v)
{
    uint32_t const b = __float_as_uint(v);
    uint32_t const k = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    // NaN keys above +inf are clamped so that no key, and no exclusive bound key + 1, equals the empty marker.
    return k < kEmptyWord - 1u ? k : kEmptyWord - 2u;
}

// Bits of x (0 for 0): the digit range of keys in [base, base + x].
__device__ __forceinline__ int32_t bitLength(uint32_t x)
{
    return x == 0u ? 0 : 32 - __clz(x);
}

// Statistics of a set of keys: extremes overall and per sign, and the count of non-negative keys.
struct KeyStats
{
    uint32_t maxKey = 0u;         // largest key (0 when none)
    uint32_t minKey = kEmptyWord; // smallest key (kEmptyWord when none)
    uint32_t positives = 0u;      // keys >= kPositiveKey
    uint32_t maxNeg = 0u;         // largest key below kPositiveKey (0 when none)
    uint32_t minPos = kEmptyWord; // smallest key >= kPositiveKey (kEmptyWord when none)

    __device__ __forceinline__ void add(uint32_t k)
    {
        bool const pos = k >= kPositiveKey;
        maxKey = max(maxKey, k);
        minKey = min(minKey, k);
        positives += pos ? 1u : 0u;
        maxNeg = pos ? maxNeg : max(maxNeg, k);
        minPos = pos ? min(minPos, k) : minPos;
    }
};

// Reduce a thread's statistics over its warp; lane 0 stores them in the per-warp array (read with readStats after
// the next block barrier).
__device__ __forceinline__ void storeWarpStats(KeyStats const& t, uint32_t* stats)
{
    uint32_t const v0 = __reduce_max_sync(0xFFFFFFFFu, t.maxKey);
    uint32_t const v1 = __reduce_min_sync(0xFFFFFFFFu, t.minKey);
    uint32_t const v2 = __reduce_add_sync(0xFFFFFFFFu, t.positives);
    uint32_t const v3 = __reduce_max_sync(0xFFFFFFFFu, t.maxNeg);
    uint32_t const v4 = __reduce_min_sync(0xFFFFFFFFu, t.minPos);
    if (threadIdx.x % 32 == 0)
    {
        int32_t const w = threadIdx.x / 32;
        stats[w] = v0;
        stats[kWarps + w] = v1;
        stats[2 * kWarps + w] = v2;
        stats[3 * kWarps + w] = v3;
        stats[4 * kWarps + w] = v4;
    }
}

// Block statistics from the per-warp array (every warp reduces it; no barrier).
__device__ __forceinline__ KeyStats readStats(uint32_t const* stats)
{
    int32_t const lane = threadIdx.x % 32;
    KeyStats b;
    b.maxKey = __reduce_max_sync(0xFFFFFFFFu, stats[lane]);
    b.minKey = __reduce_min_sync(0xFFFFFFFFu, stats[kWarps + lane]);
    b.positives = __reduce_add_sync(0xFFFFFFFFu, stats[2 * kWarps + lane]);
    b.maxNeg = __reduce_max_sync(0xFFFFFFFFu, stats[3 * kWarps + lane]);
    b.minPos = __reduce_min_sync(0xFFFFFFFFu, stats[4 * kWarps + lane]);
    return b;
}

// Digit setup of a select of the need-th largest key: over the non-negative keys when there are at least `need` of
// them, else over the negative keys with every non-negative key counted as taken (`aboveHi`).
struct SignRange
{
    uint32_t lo;      // smallest key of the range (also the digit base)
    uint64_t hi;      // exclusive end of the histogrammed range
    int32_t bits;     // digit range of the range
    uint32_t aboveHi; // keys at or above hi, all taken
    bool negative;    // the cut is among the negative keys
};

__device__ __forceinline__ SignRange signRange(KeyStats const& st, uint32_t need)
{
    if (st.positives >= need || st.positives == 0u)
    {
        uint32_t const lo = st.positives >= need ? st.minPos : st.minKey;
        return SignRange{lo, uint64_t{kEmptyWord}, bitLength(st.maxKey - lo), 0u, st.positives == 0u};
    }
    return SignRange{st.minKey, uint64_t{kPositiveKey}, bitLength(st.maxNeg - st.minKey), st.positives, true};
}

__device__ __forceinline__ int64_t round1Offset(int32_t buf, int32_t src, int32_t row)
{
    return int64_t{buf} * kBufferWords + (int64_t{src} * kMaxRows + row) * kRound1SlotWords;
}

__device__ __forceinline__ int64_t round2Offset(int32_t buf, int32_t src, int32_t row)
{
    return int64_t{buf} * kBufferWords + int64_t{kMaxSources} * kMaxRows * kRound1SlotWords
        + (int64_t{src} * kMaxRows + row) * kRound2SlotWords;
}

//! Word offset of the candidates of a slot (after the 16-byte header).
constexpr int32_t kSlotHeaderWords = 4;

__device__ __forceinline__ uint2 loadVolatile2(uint32_t const* ptr)
{
    uint2 v;
    asm volatile("ld.volatile.global.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(ptr) : "memory");
    return v;
}

__device__ __forceinline__ uint4 loadVolatile4(uint32_t const* ptr)
{
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(ptr)
                 : "memory");
    return v;
}

__device__ __forceinline__ void store2(uint32_t* ptr, uint32_t a, uint32_t b)
{
    asm volatile("st.global.v2.u32 [%0], {%1, %2};" ::"l"(ptr), "r"(a), "r"(b) : "memory");
}

__device__ __forceinline__ void store4(uint32_t* ptr, uint4 v)
{
    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

__device__ __forceinline__ bool isEmpty(uint2 v)
{
    return v.x == kEmptyWord || v.y == kEmptyWord;
}

// Whether the first pair of a 16-byte candidate vector, and the second one when `second`, were written.
__device__ __forceinline__ bool vectorReady(uint4 v, bool second)
{
    return v.x != kEmptyWord && v.y != kEmptyWord && (!second || (v.z != kEmptyWord && v.w != kEmptyWord));
}

__device__ __forceinline__ uint64_t globalTimerNs()
{
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// What a polling thread needs to give up: the error word, the timeout and this rank (for the report).
struct WaitCtx
{
    int32_t* state;
    uint64_t timeoutNs;
    int32_t rank;
};

// A peer did not deliver within the timeout. The selection cannot be completed correctly any more, and the peer's
// late writes could land in a buffer this rank resets or reuses, so record which wait failed and abort the kernel:
// the resulting CUDA error fails the run instead of letting any rank continue with a partial selection.
__device__ __noinline__ void abortOnTimeout(WaitCtx const& w, int32_t what)
{
    bool const first = atomicExch(&w.state[3], what) == 0;
    __threadfence_system();
    if (first)
    {
        printf(
            "[glm_moe_dsa] crossRankTopk: rank %d, row %d: wait %d for a peer exceeded %llu ms; aborting the kernel\n",
            w.rank, static_cast<int32_t>(blockIdx.y), what, static_cast<unsigned long long>(w.timeoutNs / 1000000ull));
    }
    __trap();
}

// Spin until both words of a Lamport pair were written; aborts the kernel after the timeout.
__device__ uint2 pollPair(uint32_t const* ptr, WaitCtx const& w, int32_t what)
{
    uint2 v = loadVolatile2(ptr);
    uint64_t start = 0;
    for (int32_t polls = 1; isEmpty(v); ++polls)
    {
        if (polls % kPollsPerTimerRead == 0)
        {
            uint64_t const now = globalTimerNs();
            if (start == 0)
            {
                start = now;
            }
            else if (now - start > w.timeoutNs)
            {
                abortOnTimeout(w, what);
            }
        }
        v = loadVolatile2(ptr);
    }
    return v;
}

// Spin until a 16-byte candidate vector was written (see vectorReady); aborts the kernel after the timeout.
__device__ uint4 pollVector(uint32_t const* ptr, bool second, WaitCtx const& w, int32_t what)
{
    uint4 v = loadVolatile4(ptr);
    uint64_t start = 0;
    for (int32_t polls = 1; !vectorReady(v, second); ++polls)
    {
        if (polls % kPollsPerTimerRead == 0)
        {
            uint64_t const now = globalTimerNs();
            if (start == 0)
            {
                start = now;
            }
            else if (now - start > w.timeoutNs)
            {
                abortOnTimeout(w, what);
            }
        }
        v = loadVolatile4(ptr);
    }
    return v;
}

// Block-wide inclusive scan of one value per thread. Returns the inclusive prefix; `total` gets the block sum.
__device__ uint32_t blockInclusiveScan(uint32_t v, uint32_t* warpSums, uint32_t& total)
{
    int32_t const lane = threadIdx.x % 32;
    int32_t const warp = threadIdx.x / 32;
    uint32_t x = v;
#pragma unroll
    for (int32_t o = 1; o < 32; o <<= 1)
    {
        uint32_t const y = __shfl_up_sync(0xFFFFFFFFu, x, o);
        if (lane >= o)
        {
            x += y;
        }
    }
    if (lane == 31)
    {
        warpSums[warp] = x;
    }
    __syncthreads();
    if (warp == 0)
    {
        uint32_t w = warpSums[lane];
#pragma unroll
        for (int32_t o = 1; o < 32; o <<= 1)
        {
            uint32_t const y = __shfl_up_sync(0xFFFFFFFFu, w, o);
            if (lane >= o)
            {
                w += y;
            }
        }
        warpSums[kWarps + lane] = w;
    }
    __syncthreads();
    total = warpSums[2 * kWarps - 1];
    uint32_t const before = warp > 0 ? warpSums[kWarps + warp - 1] : 0u;
    return x + before;
}

// Exclusive block-wide scan with a single barrier: every warp scans the per-warp sums itself. Returns this
// thread's exclusive prefix; `total` gets the block sum. warpSums must not be rewritten before the next barrier.
__device__ uint32_t blockExclusiveScan(uint32_t v, uint32_t* warpSums, uint32_t& total)
{
    int32_t const lane = threadIdx.x % 32;
    int32_t const warp = threadIdx.x / 32;
    uint32_t x = v;
#pragma unroll
    for (int32_t o = 1; o < 32; o <<= 1)
    {
        uint32_t const y = __shfl_up_sync(0xFFFFFFFFu, x, o);
        if (lane >= o)
        {
            x += y;
        }
    }
    if (lane == 31)
    {
        warpSums[warp] = x;
    }
    __syncthreads();
    uint32_t const w = warpSums[lane];
    uint32_t y = w;
#pragma unroll
    for (int32_t o = 1; o < 32; o <<= 1)
    {
        uint32_t const z = __shfl_up_sync(0xFFFFFFFFu, y, o);
        if (lane >= o)
        {
            y += z;
        }
    }
    total = __shfl_sync(0xFFFFFFFFu, y, 31);
    return __shfl_sync(0xFFFFFFFFu, y - w, warp) + x - v;
}

// Find, counting from the top bin, the bin of `hist` that holds the target-th largest key, with two barriers:
// per-warp sums, then each warp scans them and only the warp holding the target scans its own bins. Returns the
// total count (every thread); when target <= total, the bin, the count above it and its own count go to the
// scalar slots.
__device__ uint32_t findBin(SmemLayout const& s, uint32_t const* hist, int32_t nbins, uint32_t target)
{
    int32_t const lane = threadIdx.x % 32;
    int32_t const warp = threadIdx.x / 32;
    int32_t const perThread = (nbins + kThreads - 1) / kThreads;
    int32_t const top = nbins - 1 - perThread * static_cast<int32_t>(threadIdx.x);
    uint32_t mine = 0;
    for (int32_t i = 0; i < perThread && top - i >= 0; ++i)
    {
        mine += hist[top - i];
    }
    uint32_t const warpTotal = __reduce_add_sync(0xFFFFFFFFu, mine);
    if (lane == 0)
    {
        s.warpSums[warp] = warpTotal;
    }
    __syncthreads();
    uint32_t const w = s.warpSums[lane];
    uint32_t x = w;
#pragma unroll
    for (int32_t o = 1; o < 32; o <<= 1)
    {
        uint32_t const y = __shfl_up_sync(0xFFFFFFFFu, x, o);
        if (lane >= o)
        {
            x += y;
        }
    }
    uint32_t const total = __shfl_sync(0xFFFFFFFFu, x, 31);
    uint32_t const warpBefore = __shfl_sync(0xFFFFFFFFu, x - w, warp);
    if (warpBefore < target && target <= warpBefore + warpTotal)
    {
        uint32_t t = mine;
#pragma unroll
        for (int32_t o = 1; o < 32; o <<= 1)
        {
            uint32_t const y = __shfl_up_sync(0xFFFFFFFFu, t, o);
            if (lane >= o)
            {
                t += y;
            }
        }
        uint32_t acc = warpBefore + t - mine;
        if (acc < target && target <= acc + mine)
        {
            for (int32_t i = 0; i < perThread && top - i >= 0; ++i)
            {
                uint32_t const c = hist[top - i];
                if (acc + c >= target)
                {
                    s.scal[kSlotBin] = static_cast<uint32_t>(top - i);
                    s.scal[kSlotAbove] = acc;
                    s.scal[kSlotBinCount] = c;
                    break;
                }
                acc += c;
            }
        }
    }
    __syncthreads();
    return total;
}

struct SelectOut
{
    uint32_t edge;     // lower edge of the selected bin (the exact key after the last digit)
    uint32_t above;    // predicate keys strictly above the selected bin
    uint32_t binCount; // predicate keys in the selected bin
    uint32_t total;    // predicate keys
    bool all;          // take every predicate key (fewer than needed, or all fit the capacity)
    bool full;         // exact mode: the keys at or above `edge` are exactly the `need` largest
    int32_t passes;    // radix passes run
};

// Candidate keys and token indices held in shared memory (16-byte aligned, readable in groups of four up to the
// next multiple of four).
struct SmemKeys
{
    static constexpr bool kVector = true;
    uint32_t const* keys;
    uint32_t const* idx;

    __device__ uint4 keys4(int32_t j) const
    {
        return *reinterpret_cast<uint4 const*>(keys + j);
    }

    __device__ uint4 index4(int32_t j) const
    {
        return *reinterpret_cast<uint4 const*>(idx + j);
    }
};

// Merge candidates in shared memory followed by round-2 candidates read from their global Lamport slots.
struct MergeKeys
{
    static constexpr bool kVector = false;
    uint32_t const* smemKeys;
    uint32_t const* smemIdx;
    int32_t smemCount;
    uint32_t const* const* slots; // round-2 slot of each segment
    uint32_t const* segBegin;     // [numSegments + 1] candidate offsets
    int32_t numSegments;

    __device__ uint32_t const* pair(int32_t k) const
    {
        // Last segment whose begin is <= k.
        int32_t lo = 0;
        int32_t hi = numSegments - 1;
        while (lo < hi)
        {
            int32_t const mid = (lo + hi + 1) / 2;
            if (static_cast<int32_t>(segBegin[mid]) <= k)
            {
                lo = mid;
            }
            else
            {
                hi = mid - 1;
            }
        }
        return slots[lo] + kSlotHeaderWords + 2 * (k - static_cast<int32_t>(segBegin[lo]));
    }

    __device__ uint32_t operator()(int32_t j) const
    {
        return j < smemCount ? smemKeys[j] : pair(j - smemCount)[0];
    }

    __device__ uint32_t index(int32_t j) const
    {
        return j < smemCount ? smemIdx[j] : pair(j - smemCount)[1];
    }
};

// Debug slots of this CTA starting at `offset` (nullptr without a debug buffer).
__device__ __forceinline__ int64_t* debugSlots(CpTopkParams const& p, int32_t offset)
{
    if (p.debugTimes == nullptr)
    {
        return nullptr;
    }
    return p.debugTimes + (static_cast<int64_t>(blockIdx.y) * kSourcesPerRank + blockIdx.x) * kDebugSlots + offset;
}

// clock64 of thread 0 into a debug slot (no-op without a debug buffer).
__device__ __forceinline__ void stamp(int64_t* dbg, int32_t slot)
{
    if (dbg != nullptr && threadIdx.x == 0)
    {
        dbg[slot] = clock64();
    }
}

// A value into a debug slot (thread 0; no-op without a debug buffer).
__device__ __forceinline__ void putDebug(int64_t* dbg, int32_t slot, int64_t value)
{
    if (dbg != nullptr && threadIdx.x == 0)
    {
        dbg[slot] = value;
    }
}

// Phase timestamp for diagnosis (thread 0 of the CTA; no-op without a debug buffer).
__device__ __forceinline__ void markPhase(CpTopkParams const& p, int32_t phase)
{
    if (p.debugTimes != nullptr && threadIdx.x == 0)
    {
        uint64_t t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        int64_t* d = p.debugTimes + (static_cast<int64_t>(blockIdx.y) * kSourcesPerRank + blockIdx.x) * kDebugSlots;
        d[phase] = static_cast<int64_t>(t);
        d[16 + phase] = static_cast<int64_t>(clock64());
    }
}

// Histogram update of one key of a radix pass.
__device__ __forceinline__ void histogramKey(uint32_t* hist, bool inRange, uint32_t k, uint32_t lo, uint64_t hi,
    uint32_t base, uint32_t prefMask, uint32_t prefVal, int32_t shift, uint32_t binMask)
{
    uint32_t const d = k - base;
    if (inRange && k >= lo && static_cast<uint64_t>(k) < hi && (d & prefMask) == prefVal)
    {
        atomicAdd(&hist[(d >> shift) & binMask], 1u);
    }
}

// Block radix select of the need-th largest of the n keys at or above lo, of which `aboveHi` (< need) lie at or
// above hi and are counted as taken without being histogrammed. Every key in [lo, hi) is at least `base`, and
// key - base < 2^topBits; digits of up to kDigitBits bits of key - base are refined from the top. With cap > 0 the
// select stops at the first bin edge whose keys at or above it number at most `cap`. With cap = 0 it runs to the
// exact key, or stops early when the selected bin is wholly taken (`full`). Alternate passes use alternate
// histograms, so the next one is cleared while the current one is searched. Counts in the result include aboveHi.
template <typename KeyAt>
__device__ SelectOut blockSelect(SmemLayout const& s, KeyAt const& keyAt, int32_t n, uint32_t lo, uint64_t hi,
    uint32_t need, uint32_t cap, uint32_t base, int32_t topBits, int64_t* dbg = nullptr, uint32_t aboveHi = 0u)
{
    if (dbg != nullptr && threadIdx.x == 0)
    {
        dbg[0] = n;
    }
    SelectOut out{lo, 0u, 0u, 0u, false, false, 0};
    uint32_t prefMask = 0;
    uint32_t prefVal = 0;
    uint32_t above = aboveHi;
    int32_t remaining = max(topBits, 1);
    uint32_t* hist = s.hist;
    uint32_t* histNext = s.hist + kBins;
    for (int32_t b = threadIdx.x; b < (1 << min(kDigitBits, remaining)); b += kThreads)
    {
        hist[b] = 0;
    }
    __syncthreads();
#pragma unroll 1
    for (int32_t pass = 0; remaining > 0; ++pass)
    {
        int32_t const width = min(kDigitBits, remaining);
        int32_t const shift = remaining - width;
        int32_t const nbins = 1 << width;
        uint32_t const binMask = static_cast<uint32_t>(nbins - 1);
        out.passes = pass + 1;
        if (dbg != nullptr && threadIdx.x == 0 && pass < 3)
        {
            dbg[1 + 3 * pass] = clock64();
        }
        if constexpr (KeyAt::kVector)
        {
            for (int32_t j0 = 4 * static_cast<int32_t>(threadIdx.x); j0 < n; j0 += 4 * kThreads)
            {
                uint4 const q = keyAt.keys4(j0);
                histogramKey(hist, true, q.x, lo, hi, base, prefMask, prefVal, shift, binMask);
                histogramKey(hist, j0 + 1 < n, q.y, lo, hi, base, prefMask, prefVal, shift, binMask);
                histogramKey(hist, j0 + 2 < n, q.z, lo, hi, base, prefMask, prefVal, shift, binMask);
                histogramKey(hist, j0 + 3 < n, q.w, lo, hi, base, prefMask, prefVal, shift, binMask);
            }
        }
        else
        {
            for (int32_t j = threadIdx.x; j < n; j += kThreads)
            {
                histogramKey(hist, true, keyAt(j), lo, hi, base, prefMask, prefVal, shift, binMask);
            }
        }
        __syncthreads();
        if (dbg != nullptr && threadIdx.x == 0 && pass < 3)
        {
            dbg[2 + 3 * pass] = clock64();
        }
        if (shift > 0)
        {
            for (int32_t b = threadIdx.x; b < (1 << min(kDigitBits, shift)); b += kThreads)
            {
                histNext[b] = 0;
            }
        }
        // Find the bin holding the (need - above)-th largest key, counting from the top bin.
        uint32_t const target = need - above;
        uint32_t const total = findBin(s, hist, nbins, target);
        if (dbg != nullptr && threadIdx.x == 0 && pass < 3)
        {
            dbg[3 + 3 * pass] = clock64();
        }
        if (pass == 0)
        {
            out.total = total + aboveHi;
            if (total + aboveHi <= cap || target > total)
            {
                out.all = true;
                out.above = total + aboveHi;
                return out;
            }
        }
        // The scalars are next written after two more barriers (the next histogram and scan).
        uint32_t const bin = s.scal[kSlotBin];
        above += s.scal[kSlotAbove];
        out.binCount = s.scal[kSlotBinCount];
        prefVal |= bin << shift;
        prefMask |= binMask << shift;
        out.edge = base + prefVal;
        out.above = above;
        remaining = shift;
        uint32_t* const t = hist;
        hist = histNext;
        histNext = t;
        if (cap > 0 && above + out.binCount <= cap)
        {
            return out;
        }
        if (cap == 0 && remaining > 0 && above + out.binCount == need)
        {
            out.full = true;
            return out;
        }
    }
    return out;
}

// Send one source's selected keys (keys in [lo, hi) of the chunk) to its slot; returns the count sent.
// Takes every predicate key at or above the edge when they fit `cap`. Otherwise (more than `cap` keys tie at the
// exact edge) it takes every key above the edge, plus, with kPartialTies, the first keys equal to it in source
// order up to `cap` (any subset of equal keys is a valid tie-aware choice). Round 1 sends no partial ties, so the
// keys a second round requests from a source (at most the edge) were never sent before.
// The header carries the count and an exclusive upper bound of the keys left unsent (0 when none).
// With `stage` (round 1, at most kSourceCapacity keys), the pairs are gathered in shared memory first and written
// with coalesced 16-byte stores; otherwise each pair is stored where it is found.
template <bool kPartialTies>
__device__ uint32_t sendSource(SmemLayout const& s, SelectOut const& sel, int32_t n, uint32_t lo, uint64_t hi,
    uint32_t cap, uint32_t* mcSlot, int32_t src, int32_t rank, int32_t world, uint2* stage, int64_t* dbg = nullptr)
{
    bool const strict = !sel.all && sel.above + sel.binCount > cap;
    uint32_t const threshold = sel.all ? lo : sel.edge;
    // Keys equal to the edge that fit after the keys above it (sel.above < need <= cap when strict).
    uint32_t const tieBudget = kPartialTies && strict ? cap - sel.above : 0u;
    uint32_t const sent = sel.all ? sel.total : (strict ? sel.above + tieBudget : sel.above + sel.binCount);
    uint32_t const bound = sel.all ? lo : (strict ? sel.edge + 1u : sel.edge);
    stamp(dbg, 42);
    // One page block (kBlockTokens keys) per thread; positions from a block scan, so the slot holds the keys in
    // source order: first those taken outright, then (strict only) the admitted keys equal to the edge.
    int32_t const groups = (n + kBlockTokens - 1) / kBlockTokens;
    uint32_t takenMain = 0;
    uint32_t takenTies = 0;
    uint32_t const tieBase = strict ? sel.above : 0u;
    for (int32_t g0 = 0; g0 < groups; g0 += kThreads)
    {
        int32_t const g = g0 + static_cast<int32_t>(threadIdx.x);
        uint32_t k[kBlockTokens];
        uint32_t mask = 0;
        uint32_t tieMask = 0;
        if (g < groups)
        {
            uint4 const a = *reinterpret_cast<uint4 const*>(s.chunk + kBlockTokens * g);
            uint4 const b = *reinterpret_cast<uint4 const*>(s.chunk + kBlockTokens * g + 4);
            k[0] = a.x;
            k[1] = a.y;
            k[2] = a.z;
            k[3] = a.w;
            k[4] = b.x;
            k[5] = b.y;
            k[6] = b.z;
            k[7] = b.w;
#pragma unroll
            for (int32_t i = 0; i < kBlockTokens; ++i)
            {
                bool const in = kBlockTokens * g + i < n && k[i] >= lo && static_cast<uint64_t>(k[i]) < hi;
                bool const take = in && (strict ? k[i] > threshold : k[i] >= threshold);
                bool const tie = kPartialTies && in && strict && k[i] == threshold;
                mask |= (take ? 1u : 0u) << i;
                tieMask |= (tie ? 1u : 0u) << i;
            }
        }
        // At most kThreads * kBlockTokens keys per pass: both counts fit 16 bits.
        uint32_t const c = static_cast<uint32_t>(__popc(mask)) | (static_cast<uint32_t>(__popc(tieMask)) << 16);
        uint32_t chunkTaken;
        uint32_t const before = blockExclusiveScan(c, s.warpSums, chunkTaken);
        uint32_t pos = takenMain + (before & 0xFFFFu);
        uint32_t tiePos = takenTies + (before >> 16);
        takenMain += chunkTaken & 0xFFFFu;
        takenTies += chunkTaken >> 16;
        if ((mask | tieMask) != 0u)
        {
            int32_t const colBase = g * kPageTokens + ((src + g) % kSourcesPerRank) * kBlockTokens;
#pragma unroll
            for (int32_t i = 0; i < kBlockTokens; ++i)
            {
                uint32_t slot = kEmptyWord;
                if ((mask >> i) & 1u)
                {
                    slot = pos++;
                }
                else if constexpr (kPartialTies)
                {
                    if ((tieMask >> i) & 1u)
                    {
                        slot = tiePos < tieBudget ? tieBase + tiePos : kEmptyWord;
                        ++tiePos;
                    }
                }
                if (slot != kEmptyWord)
                {
                    uint32_t const token = localToGlobal(colBase + i, rank, world);
                    if (stage != nullptr)
                    {
                        stage[slot] = make_uint2(k[i], token);
                    }
                    else
                    {
                        store2(mcSlot + kSlotHeaderWords + 2 * static_cast<int64_t>(slot), k[i], token);
                    }
                }
            }
        }
        // The staged pairs are complete, and the scan's warp sums may be rewritten.
        __syncthreads();
    }
    stamp(dbg, 43);
    if (stage != nullptr)
    {
        // Two pairs per 16-byte store; an odd count leaves the last half of the last vector empty.
        int32_t const vectors = static_cast<int32_t>(sent + 1u) / 2;
        for (int32_t v = threadIdx.x; v < vectors; v += kThreads)
        {
            uint2 const a = stage[2 * v];
            uint2 const b
                = 2 * v + 1 < static_cast<int32_t>(sent) ? stage[2 * v + 1] : make_uint2(kEmptyWord, kEmptyWord);
            store4(mcSlot + kSlotHeaderWords + 4 * static_cast<int64_t>(v), make_uint4(a.x, a.y, b.x, b.y));
        }
    }
    // Receivers check every candidate word, so the header needs no ordering against the candidates.
    if (threadIdx.x == kThreads - 1)
    {
        store2(mcSlot, sent, bound);
    }
    stamp(dbg, 44);
    return sent;
}

// Load one source's keys (one block of kBlockTokens tokens of every local page) from the local logits into the
// chunk; returns the number of keys. Consecutive threads read consecutive keys (whole 32-byte blocks per eight
// threads), and each thread issues kLoadBatch loads before it uses any. With `withStats`, the per-warp key
// statistics are stored for readStats after the next block barrier.
__device__ int32_t loadSource(
    SmemLayout const& s, float const* rowLogits, int32_t localLen, int32_t src, bool withStats)
{
    constexpr int32_t kLoadBatch = 8;
    int32_t const n = sourceLength(localLen, src);
    KeyStats st;
    for (int32_t j0 = threadIdx.x; j0 < n; j0 += kLoadBatch * kThreads)
    {
        float v[kLoadBatch];
#pragma unroll
        for (int32_t i = 0; i < kLoadBatch; ++i)
        {
            int32_t const j = j0 + i * kThreads;
            int32_t const page = j / kBlockTokens;
            int32_t const col = page * kPageTokens + ((src + page) % kSourcesPerRank) * kBlockTokens + j % kBlockTokens;
            v[i] = j < n ? rowLogits[col] : 0.F;
        }
#pragma unroll
        for (int32_t i = 0; i < kLoadBatch; ++i)
        {
            int32_t const j = j0 + i * kThreads;
            if (j < n)
            {
                uint32_t const k = orderedKey(v[i]);
                s.chunk[j] = k;
                st.add(k);
            }
        }
    }
    if (withStats)
    {
        storeWarpStats(st, s.stats);
    }
    return n;
}

// Whether source q withheld keys that may lie above the merged k-th key.
__device__ __forceinline__ bool sourceOverflows(SmemLayout const& s, int32_t q, uint32_t kth)
{
    uint32_t const cnt = s.scal[kSlotSrcCount + q];
    uint64_t const bound = s.scal[kSlotSrcBound + q];
    return cnt < s.scal[kSlotSrcLen + q] && bound > uint64_t{kth} + 1u;
}

// Emit the top-kTopK of the merge candidates. Keys above `kth` are all taken; ties at `kth` are taken in candidate
// order when only some of them fit (deterministic on every rank), otherwise all of them. With `takeAll`, every
// candidate is taken and the row is padded with -1. The row is assembled in shared memory (the first histogram)
// and written with coalesced stores.
template <typename KeyAt>
__device__ bool emitSelection(SmemLayout const& s, KeyAt const& keys, int32_t total, uint32_t kth, bool takeAll,
    int32_t* outRow, int32_t overflowSources, int64_t* dbg = nullptr)
{
    uint32_t const want = static_cast<uint32_t>(kTopK);
    // Positions from one block scan over contiguous candidate ranges (no contended counter); the order is the
    // candidate order, identical on every rank. Shared-memory candidates are read four at a time.
    int32_t per = (total + kThreads - 1) / kThreads;
    if constexpr (KeyAt::kVector)
    {
        per = (per + 3) / 4 * 4;
    }
    int32_t const j0 = min(total, static_cast<int32_t>(threadIdx.x) * per);
    int32_t const j1 = min(total, j0 + per);
    uint32_t cGt = 0;
    uint32_t cEq = 0;
    if constexpr (KeyAt::kVector)
    {
        for (int32_t j = j0; j < j1; j += 4)
        {
            uint4 const q = keys.keys4(j);
            uint32_t const kk[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
            for (int32_t i = 0; i < 4; ++i)
            {
                bool const in = j + i < j1;
                cGt += (in && (takeAll || kk[i] > kth)) ? 1u : 0u;
                cEq += (in && !takeAll && kk[i] == kth) ? 1u : 0u;
            }
        }
    }
    else
    {
        for (int32_t j = j0; j < j1; ++j)
        {
            uint32_t const k = keys(j);
            cGt += (takeAll || k > kth) ? 1u : 0u;
            cEq += (!takeAll && k == kth) ? 1u : 0u;
        }
    }
    // The first `overflowSources` threads flag a source that withheld keys above the threshold (the flag slot is
    // zero on entry); with a flag set, nothing is written and the caller runs the second round.
    if (static_cast<int32_t>(threadIdx.x) < overflowSources && sourceOverflows(s, threadIdx.x, kth))
    {
        atomicOr(&s.scal[kSlotCounter2], 1u);
    }
    uint32_t allGt;
    uint32_t allEq;
    uint32_t posGt;
    uint32_t posEq;
    if (total < 0x10000)
    {
        // Both counts fit 16 bits: one scan of the packed pair.
        uint32_t packedTotal;
        uint32_t const packed = cGt | (cEq << 16);
        uint32_t const before = blockExclusiveScan(packed, s.warpSums, packedTotal);
        allGt = packedTotal & 0xFFFFu;
        allEq = packedTotal >> 16;
        posGt = before & 0xFFFFu;
        posEq = allGt + (before >> 16);
    }
    else
    {
        posGt = blockInclusiveScan(cGt, s.warpSums, allGt) - cGt;
        __syncthreads();
        posEq = blockInclusiveScan(cEq, s.warpSums, allEq) - cEq;
        posEq += allGt;
    }
    stamp(dbg, 59);
    if (overflowSources > 0 && s.scal[kSlotCounter2] != 0u)
    {
        return true;
    }
    uint32_t* row = s.hist;
    auto place = [&](uint32_t k, uint32_t idx)
    {
        if (takeAll || k > kth)
        {
            if (posGt < want)
            {
                row[posGt] = idx;
            }
            ++posGt;
        }
        else if (k == kth)
        {
            if (posEq < want)
            {
                row[posEq] = idx;
            }
            ++posEq;
        }
    };
    if constexpr (KeyAt::kVector)
    {
        for (int32_t j = j0; j < j1; j += 4)
        {
            uint4 const q = keys.keys4(j);
            uint4 const x = keys.index4(j);
            uint32_t const kk[4] = {q.x, q.y, q.z, q.w};
            uint32_t const xx[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
            for (int32_t i = 0; i < 4; ++i)
            {
                if (j + i < j1)
                {
                    place(kk[i], xx[i]);
                }
            }
        }
    }
    else
    {
        for (int32_t j = j0; j < j1; ++j)
        {
            place(keys(j), keys.index(j));
        }
    }
    __syncthreads();
    stamp(dbg, 60);
    uint32_t const selected = min(allGt + allEq, want);
    for (uint32_t j = threadIdx.x; j < want; j += kThreads)
    {
        outRow[j] = j < selected ? static_cast<int32_t>(row[j]) : -1;
    }
    stamp(dbg, 61);
    return false;
}

// Threshold of an exact (cap = 0) merge select: keys above `kth` are taken, ties at `kth` while they fit.
__device__ __forceinline__ uint32_t mergeThreshold(SelectOut const& m, bool& takeAll)
{
    takeAll = m.all || (m.full && m.edge == 0u);
    if (takeAll)
    {
        return 0u;
    }
    // A wholly taken bin: the keys at or above its edge are exactly kTopK.
    return m.full ? m.edge - 1u : m.edge;
}

// Reset the slots of Lamport buffer `buf` on this rank's copy, using one warp per slot. A slot was written iff its
// header was; the header's count bounds the candidates to reset.
__device__ void clearBuffer(CpTopkParams const& p, int32_t buf, int32_t clearer, int32_t numClearers)
{
    int32_t const warp = threadIdx.x / 32;
    int32_t const lane = threadIdx.x % 32;
    int32_t const round1Slots = kMaxSources * kMaxRows;
    uint4 const empty = make_uint4(kEmptyWord, kEmptyWord, kEmptyWord, kEmptyWord);
    for (int32_t slot = clearer * kWarps + warp; slot < 2 * round1Slots; slot += numClearers * kWarps)
    {
        bool const second = slot >= round1Slots;
        int32_t const qr = second ? slot - round1Slots : slot;
        uint32_t* base = p.ucWorkspace
            + (second ? round2Offset(buf, qr / kMaxRows, qr % kMaxRows)
                      : round1Offset(buf, qr / kMaxRows, qr % kMaxRows));
        uint2 h = make_uint2(kEmptyWord, kEmptyWord);
        if (lane == 0)
        {
            h = loadVolatile2(base);
        }
        h.x = __shfl_sync(0xFFFFFFFFu, h.x, 0);
        h.y = __shfl_sync(0xFFFFFFFFu, h.y, 0);
        if (h.x == kEmptyWord && h.y == kEmptyWord)
        {
            continue;
        }
        uint32_t const capacity = second ? kRound2Capacity : kSourceCapacity;
        uint32_t const count = h.x == kEmptyWord ? capacity : min(h.x, capacity);
        int32_t const vectors = static_cast<int32_t>(count + 1u) / 2;
        for (int32_t v = lane; v < vectors; v += 32)
        {
            store4(base + kSlotHeaderWords + 4 * static_cast<int64_t>(v), empty);
        }
        if (lane == 0)
        {
            store2(base, kEmptyWord, kEmptyWord);
        }
    }
}

// Merge of one row on this rank (one CTA): gathers every source's round-1 candidates, selects the exact
// top-kTopK, runs the second round when a source withheld keys above the threshold, and writes the row.
__device__ void mergeRow(CpTopkParams const& p, SmemLayout const& s, uint32_t const** segSlots, uint32_t* segBegin,
    int32_t buf, int32_t row, int32_t rowLen, int32_t localLen, float const* rowLogits, int32_t* outRow,
    WaitCtx const& wait)
{
    int32_t const world = p.worldSize;
    int32_t const rank = p.rank;
    int32_t const sources = numSources(world);
    uint32_t const cap = static_cast<uint32_t>(roundCap(world));
    __syncthreads();
    if (threadIdx.x == 0)
    {
        s.scal[kSlotCounter2] = 0u;
    }
    if (static_cast<int32_t>(threadIdx.x) < sources)
    {
        // Keys each source holds, computed while its header is in flight.
        int32_t const q = threadIdx.x;
        s.scal[kSlotSrcLen + q] = static_cast<uint32_t>(
            sourceLength(min(localLength(rowLen, q / kSourcesPerRank, world), p.localCapacity), q % kSourcesPerRank));
        uint2 const h = pollPair(p.ucWorkspace + round1Offset(buf, threadIdx.x, row), wait, kWaitRound1Header);
        s.scal[kSlotSrcCount + threadIdx.x] = min(h.x, cap);
        s.scal[kSlotSrcBound + threadIdx.x] = h.y;
    }
    __syncthreads();
    if (threadIdx.x < 32)
    {
        // Exclusive prefix of the source counts (at most 64 sources: two per lane).
        uint32_t const a = static_cast<int32_t>(threadIdx.x) < sources ? s.scal[kSlotSrcCount + threadIdx.x] : 0u;
        uint32_t const b
            = static_cast<int32_t>(threadIdx.x) + 32 < sources ? s.scal[kSlotSrcCount + threadIdx.x + 32] : 0u;
        uint32_t x = a;
        uint32_t z = b;
#pragma unroll
        for (int32_t o = 1; o < 32; o <<= 1)
        {
            uint32_t const y = __shfl_up_sync(0xFFFFFFFFu, x, o);
            uint32_t const w = __shfl_up_sync(0xFFFFFFFFu, z, o);
            if (static_cast<int32_t>(threadIdx.x) >= o)
            {
                x += y;
                z += w;
            }
        }
        uint32_t const totalA = __shfl_sync(0xFFFFFFFFu, x, 31);
        s.scal[kSlotSrcOffset + threadIdx.x] = x - a;
        s.scal[kSlotSrcOffset + 32 + threadIdx.x] = totalA + z - b;
        if (threadIdx.x == 31)
        {
            s.scal[kSlotSrcOffset + kMaxSources] = totalA + z;
        }
    }
    __syncthreads();
    int32_t const total = static_cast<int32_t>(s.scal[kSlotSrcOffset + kMaxSources]);
    markPhase(p, 4);

    // Candidates: padded (source, vector) positions; issue a batch of independent 16-byte loads, then re-poll the
    // ones not yet written. The key statistics are tracked for the merge select.
    int32_t const vectorsPerSource = static_cast<int32_t>(cap) / 2;
    int32_t const padded = sources * vectorsPerSource;
    KeyStats st;
#pragma unroll 1
    for (int32_t v0 = 0; v0 < padded; v0 += kThreads * kPollBatch)
    {
        uint4 v[kPollBatch];
#pragma unroll
        for (int32_t i = 0; i < kPollBatch; ++i)
        {
            int32_t const jv = v0 + i * kThreads + static_cast<int32_t>(threadIdx.x);
            int32_t const q = jv / vectorsPerSource;
            int32_t const kv = jv % vectorsPerSource;
            v[i] = make_uint4(0u, 0u, 0u, 0u);
            if (jv < padded && 2 * kv < static_cast<int32_t>(s.scal[kSlotSrcCount + q]))
            {
                v[i] = loadVolatile4(p.ucWorkspace + round1Offset(buf, q, row) + kSlotHeaderWords + 4 * kv);
            }
        }
#pragma unroll
        for (int32_t i = 0; i < kPollBatch; ++i)
        {
            int32_t const jv = v0 + i * kThreads + static_cast<int32_t>(threadIdx.x);
            int32_t const q = jv / vectorsPerSource;
            int32_t const kv = jv % vectorsPerSource;
            if (jv < padded && 2 * kv < static_cast<int32_t>(s.scal[kSlotSrcCount + q]))
            {
                bool const second = 2 * kv + 1 < static_cast<int32_t>(s.scal[kSlotSrcCount + q]);
                if (!vectorReady(v[i], second))
                {
                    v[i] = pollVector(p.ucWorkspace + round1Offset(buf, q, row) + kSlotHeaderWords + 4 * kv, second,
                        wait, kWaitRound1Candidates);
                }
                int32_t const j = static_cast<int32_t>(s.scal[kSlotSrcOffset + q]) + 2 * kv;
                s.mkey[j] = v[i].x;
                s.midx[j] = v[i].y;
                st.add(v[i].x);
                if (second)
                {
                    s.mkey[j + 1] = v[i].z;
                    s.midx[j + 1] = v[i].w;
                    st.add(v[i].z);
                }
            }
        }
    }
    stamp(debugSlots(p, 0), 45);
    storeWarpStats(st, s.stats);
    __syncthreads();
    markPhase(p, 5);

    // Digits of the merge select span the candidates of the sign the cut lies in (see kPositiveKey).
    SignRange const r = total > 0 ? signRange(readStats(s.stats), static_cast<uint32_t>(kTopK))
                                  : SignRange{0u, uint64_t{kEmptyWord}, 32, 0u, false};
    SmemKeys const merged{s.mkey, s.midx};
    SelectOut const m = blockSelect(
        s, merged, total, r.lo, r.hi, static_cast<uint32_t>(kTopK), 0u, r.lo, r.bits, debugSlots(p, 48), r.aboveHi);
    putDebug(debugSlots(p, 0), 62, total);
    putDebug(debugSlots(p, 0), 63, m.passes + (r.negative ? 10 : 0));
    bool takeAll = false;
    uint32_t const kth = mergeThreshold(m, takeAll);
    markPhase(p, 6);
    stamp(debugSlots(p, 0), 58);
    if (!emitSelection(s, merged, total, kth, takeAll, outRow, sources, debugSlots(p, 0)))
    {
        return;
    }

    // ---- Round 2 (rare; every rank takes it): withheld keys in [kth, bound) of overflowing sources ----------
    __syncthreads();
    // 1. This rank's overflowing sources send their top-kRound2Capacity withheld keys.
    for (int32_t c = 0; c < kSourcesPerRank; ++c)
    {
        int32_t const q = rank * kSourcesPerRank + c;
        if (!sourceOverflows(s, q, kth))
        {
            continue;
        }
        uint64_t const b = s.scal[kSlotSrcBound + q];
        int32_t const nq = loadSource(s, rowLogits, localLen, c, false);
        __syncthreads();
        SelectOut const sel2 = blockSelect(
            s, SmemKeys{s.chunk, nullptr}, nq, kth, b, static_cast<uint32_t>(kRound2Capacity), kRound2Capacity, 0u, 32);
        (void) sendSource<true>(
            s, sel2, nq, kth, b, kRound2Capacity, p.mcWorkspace + round2Offset(buf, q, row), c, rank, world, nullptr);
        __syncthreads();
    }
    // 2. Gather the round-2 segments of every overflowing source (all ranks).
    if (threadIdx.x == 0)
    {
        int32_t segs = 0;
        uint32_t off = 0;
        for (int32_t q = 0; q < sources; ++q)
        {
            if (!sourceOverflows(s, q, kth))
            {
                continue;
            }
            uint32_t const* slot = p.ucWorkspace + round2Offset(buf, q, row);
            uint2 const h = pollPair(slot, wait, kWaitRound2Header);
            segSlots[segs] = slot;
            segBegin[segs] = off;
            off += min(h.x, static_cast<uint32_t>(kRound2Capacity));
            ++segs;
        }
        segBegin[segs] = off;
        s.scal[kSlotTotal] = static_cast<uint32_t>(segs);
    }
    __syncthreads();
    int32_t const segs = static_cast<int32_t>(s.scal[kSlotTotal]);
    int32_t const total2 = static_cast<int32_t>(segBegin[segs]);
    MergeKeys const keys{s.mkey, s.midx, total, segSlots, segBegin, segs};
    for (int32_t k = threadIdx.x; k < total2; k += kThreads)
    {
        (void) pollPair(keys.pair(k), wait, kWaitRound2Candidates);
    }
    __syncthreads();
    SelectOut const m2
        = blockSelect(s, keys, total + total2, 0u, uint64_t{kEmptyWord}, static_cast<uint32_t>(kTopK), 0u, 0u, 32);
    bool takeAll2 = false;
    uint32_t const kth2 = mergeThreshold(m2, takeAll2);
    (void) emitSelection(s, keys, total + total2, kth2, takeAll2, outRow, 0);
    if (threadIdx.x == 0)
    {
        atomicAdd(&p.state[2], 1);
    }
}

__global__ void __launch_bounds__(kThreads, 1) cpTopkKernel(CpTopkParams p)
{
    markPhase(p, 0);
    extern __shared__ __align__(16) uint32_t smemRaw[];
    __shared__ uint32_t const* segSlots[kMaxSources];
    __shared__ uint32_t segBegin[kMaxSources + 1];
    int32_t const world = p.worldSize;
    int32_t const rank = p.rank;
    int32_t const src = blockIdx.x;
    int32_t const row = blockIdx.y;
    SmemLayout const s = carveSmem(smemRaw, p.localCapacity, world);
    int32_t* outRow = p.out + static_cast<int64_t>(row) * kTopK;
    // Every global read below (lengths, epoch, logits) comes after the wait, so a programmatic early launch is safe
    // whatever the preceding kernel writes.
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int32_t const kvLen = p.kvLens[0];
    int32_t const epoch = *reinterpret_cast<int32_t volatile*>(&p.state[0]);
    int32_t const buf = epoch % kNumLamportBuffers;
    int32_t const rowLen = max(0, kvLen - p.nextN + row + 1);
    // Identity launch (every row has at most kTopK tokens; the same decision on every rank): no exchange, and
    // neither the workspace nor the epoch is touched, so identity launches never shorten the reuse chain of the
    // Lamport buffers (see crossRankTopk.h).
    bool const exchange = kvLen - p.nextN + p.numRows > kTopK;
    __syncthreads();

    if (rowLen <= kTopK)
    {
        // Every rank agrees on this branch: identity selection, no exchange for this row.
        if (src == 0)
        {
            for (int32_t j = threadIdx.x; j < kTopK; j += kThreads)
            {
                outRow[j] = j < rowLen ? j : -1;
            }
        }
    }
    else
    {
        int32_t const localLen = min(localLength(rowLen, rank, world), p.localCapacity);
        float const* rowLogits = p.logits + static_cast<int64_t>(row) * p.logitsStride;

        // ---- Round 1: this source's local top keys ---------------------------------------------------------
        int32_t const n = loadSource(s, rowLogits, localLen, src, true);
        __syncthreads();
        markPhase(p, 1);
        uint32_t const cap = static_cast<uint32_t>(roundCap(world));
        uint32_t const need = static_cast<uint32_t>(roundNeed(world));
        SelectOut sel{0u, 0u, 0u, static_cast<uint32_t>(n), true, false, 0};
        uint32_t selLo = 0u;
        int32_t passInfo = 0;
        if (n > static_cast<int32_t>(cap))
        {
            // The cut lies among the keys of one sign (see kPositiveKey); the digits span that sign's key range.
            SignRange const r = signRange(readStats(s.stats), need);
            selLo = r.lo;
            SmemKeys const keys{s.chunk, nullptr};
            sel = blockSelect(s, keys, n, r.lo, r.hi, need, cap, r.lo, r.bits, debugSlots(p, 32), r.aboveHi);
            passInfo = sel.passes + (r.negative ? 10 : 0);
        }
        markPhase(p, 2);
        uint32_t const sent = sendSource<false>(s, sel, n, selLo, uint64_t{kEmptyWord}, cap,
            p.mcWorkspace + round1Offset(buf, rank * kSourcesPerRank + src, row), src, rank, world,
            reinterpret_cast<uint2*>(s.mkey), debugSlots(p, 0));
        putDebug(debugSlots(p, 0), 46, sent);
        putDebug(debugSlots(p, 0), 47, passInfo);
        markPhase(p, 3);

        // ---- Merge: one CTA per row and rank ---------------------------------------------------------------
        if (src == 0)
        {
            WaitCtx const wait{p.state, p.timeoutNs, rank};
            mergeRow(p, s, segSlots, segBegin, buf, row, rowLen, localLen, rowLogits, outRow, wait);
            markPhase(p, 7);
        }
    }
    if (!exchange)
    {
        return;
    }

    // The other CTAs reset the buffer of the previous exchange launch, whose candidates every rank has consumed: a
    // peer writes it again only two exchange launches later, after this launch completed on this rank.
    if (src != 0)
    {
        clearBuffer(p, (epoch + kNumLamportBuffers - 1) % kNumLamportBuffers, row * (kSourcesPerRank - 1) + src - 1,
            static_cast<int32_t>(gridDim.y) * (kSourcesPerRank - 1));
    }

    // Advance the epoch once every CTA of this launch has read it.
    __syncthreads();
    markPhase(p, 8);
    if (threadIdx.x == 0)
    {
        __threadfence();
        int32_t const numCtas = static_cast<int32_t>(gridDim.x * gridDim.y);
        int32_t const arrived = atomicAdd(&p.state[1], 1);
        if (arrived == numCtas - 1)
        {
            p.state[1] = 0;
            __threadfence();
            atomicAdd(&p.state[0], 1);
        }
    }
}

#endif

} // namespace

size_t cpTopkSmemBytes(int32_t localCapacity, int32_t worldSize)
{
    return smemBytes(localCapacity, worldSize);
}

int32_t cpTopkMaxLocalCapacity(int32_t worldSize)
{
    static int32_t cached[kMaxRanks + 1] = {};
    if (worldSize < 1 || worldSize > kMaxRanks)
    {
        return 0;
    }
    if (cached[worldSize] == 0)
    {
        int device = 0;
        cudaGetDevice(&device);
        int maxSmemOptin = 0;
        cudaDeviceGetAttribute(&maxSmemOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
        int32_t cap = 0;
        for (int32_t c = 1 << 22; c >= kPageTokens * kSourcesPerRank; c -= kPageTokens * kSourcesPerRank)
        {
            if (smemBytes(c, worldSize) + kStaticSmemBytes <= static_cast<size_t>(maxSmemOptin))
            {
                cap = c;
                break;
            }
        }
        cached[worldSize] = cap;
    }
    return cached[worldSize];
}

int64_t cpTopkDefaultTimeoutMs()
{
    static int64_t const timeoutMs = []()
    {
        char const* env = std::getenv("TRTLLM_STAIRCASE_GLM_LL_CROSS_RANK_TOPK_TIMEOUT_MS");
        if (env == nullptr || env[0] == '\0')
        {
            return kDefaultTimeoutMs;
        }
        char* end = nullptr;
        long long const v = std::strtoll(env, &end, 10);
        if (end == env || *end != '\0' || v <= 0)
        {
            throw std::invalid_argument(
                std::string("glm_moe_dsa_cross_rank_topk: bad TRTLLM_STAIRCASE_GLM_LL_CROSS_RANK_TOPK_TIMEOUT_MS=")
                + env);
        }
        return static_cast<int64_t>(v);
    }();
    return timeoutMs;
}

void invokeCpTopk(CpTopkParams const& inParams, cudaStream_t stream)
{
    if (inParams.numRows <= 0)
    {
        return;
    }
    CpTopkParams params = inParams;
    if (params.timeoutNs == 0)
    {
        params.timeoutNs = static_cast<uint64_t>(cpTopkDefaultTimeoutMs()) * 1000000ull;
    }
    if (params.numRows > kMaxRows || params.worldSize <= 0 || params.worldSize > kMaxRanks || params.rank < 0
        || params.rank >= params.worldSize)
    {
        throw std::invalid_argument("glm_moe_dsa_cross_rank_topk: unsupported rows/world/rank");
    }
    size_t const smem = smemBytes(params.localCapacity, params.worldSize);
    // The attribute is per function and device; raise it only when a larger footprint is requested.
    static thread_local size_t configuredSmem[64] = {};
    int device = 0;
    cudaGetDevice(&device);
    cudaError_t err = cudaSuccess;
    if (device >= 64 || configuredSmem[device] < smem)
    {
        err = cudaFuncSetAttribute(cpTopkKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("glm_moe_dsa_cross_rank_topk: smem attribute: ") + cudaGetErrorString(err));
        }
        if (device < 64)
        {
            configuredSmem[device] = smem;
        }
    }
    // Programmatic dependent launch as for the other TensorRT-LLM kernels (on unless TRTLLM_ENABLE_PDL=0).
    static bool const enablePdl = []()
    {
        char const* env = std::getenv("TRTLLM_ENABLE_PDL");
        return !(env != nullptr && env[0] == '0' && env[1] == '\0');
    }();
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(kSourcesPerRank, params.numRows);
    config.blockDim = dim3(kThreads);
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = enablePdl ? 1 : 0;
    config.attrs = attrs;
    config.numAttrs = 1;
    err = cudaLaunchKernelEx(&config, cpTopkKernel, params);
    if (err == cudaSuccess)
    {
        err = cudaGetLastError();
    }
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("glm_moe_dsa_cross_rank_topk: launch: ") + cudaGetErrorString(err));
    }
}

} // namespace cross_rank_topk

TRTLLM_MV2_NAMESPACE_END
