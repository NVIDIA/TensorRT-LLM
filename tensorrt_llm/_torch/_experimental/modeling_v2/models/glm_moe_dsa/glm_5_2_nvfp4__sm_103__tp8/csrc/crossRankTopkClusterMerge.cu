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

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_103__tp8/csrc/crossRankTopkClusterMerge.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <stdexcept>
#include <string>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace cross_rank_topk_cluster_merge
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
//! CTAs of one row's cluster.
constexpr int32_t kClusterCtas = kSourcesPerRank;
//! Histogram bins per group of the cluster bin search (one warp lane per bin).
constexpr int32_t kGroupBins = 32;
constexpr int32_t kMaxGroups = kBins / kGroupBins;
//! Static shared memory of the kernel (round-2 segment tables and the cluster exchange records), kept out of the
//! dynamic budget.
constexpr size_t kStaticSmemBytes = 8192;

// Records the CTAs of a row's cluster send each other with st.async (static, so every CTA has them at the same
// address). Exchange x of a row completes on barrier bar[x % 2], phase x / 2; the record sets alternate by pass
// parity and are written again only after every CTA has completed the exchange in between.
struct ClusterRecords
{
    //! Transaction barriers of the exchanges (arrival count 1: this CTA's expect-tx).
    uint64_t bar[2];
    //! [pass parity][source CTA][group] histogram totals per group of kGroupBins bins.
    uint32_t groups[2][kClusterCtas][kMaxGroups];
    //! [pass parity][source CTA][bin] counts of the selected group's bins, ascending.
    uint32_t bins[2][kClusterCtas][kGroupBins];
};

// Scalar slots in shared memory.
enum ScalarSlot : int32_t
{
    kSlotBin = 0,
    kSlotAbove,
    kSlotBinCount,
    kSlotTotal,
    kSlotCounter2,
    kSlotSrcCount,                                       // [kMaxSources]
    kSlotSrcBound = kSlotSrcCount + kMaxSources,         // [kMaxSources]
    kSlotSrcLen = kSlotSrcBound + kMaxSources,           // [kMaxSources]
    kSlotSrcOffset = kSlotSrcLen + kMaxSources,          // [kMaxSources + 1]
    kSlotCountBefore = kSlotSrcOffset + kMaxSources + 1, // candidates of the cluster CTAs before this one
    kSlotCountMine,                                      // candidates of this cluster CTA
    kSlotCountTotal,                                     // candidates of the row
    kSlotPositivesBefore,                                // candidates >= +0.0 of the cluster CTAs before this one
    kNumScalarSlots
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
// The kernel targets SM90+ (multicast Lamport exchange, thread-block clusters).
__global__ void __launch_bounds__(kThreads, 1) cpTopkZ9Kernel(CpTopkParams p)
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
//! Warps of the cluster merge: warps [0, kPollWarps) poll the round-1 candidates of this CTA's sources (one 16-byte
//! vector per thread: at most 520 vectors per CTA, at world sizes 5 to 7), the next kHeaderWarps warps poll the
//! round-1 headers of every source (one per thread), and the remaining kResetWarps warps reset the previous exchange
//! launch's buffer meanwhile.
constexpr int32_t kPollWarps = 17;
constexpr int32_t kHeaderWarps = 2;
constexpr int32_t kFirstResetWarp = kPollWarps + kHeaderWarps;
constexpr int32_t kResetWarps = kWarps - kFirstResetWarp;
static_assert(kHeaderWarps * 32 >= kMaxSources, "one header per header-warp thread");

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
            "[glm_moe_dsa] crossRankTopkClusterMerge: rank %d, row %d: wait %d for a peer exceeded %llu ms; aborting "
            "the kernel\n",
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
// The header carries the count and an exclusive upper bound of the keys left unsent (0 when none); in round 1
// (!kPartialTies) also the key statistics of the keys sent (see kSlotHeaderWords).
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
    if constexpr (kPartialTies)
    {
        if (threadIdx.x == kThreads - 1)
        {
            store2(mcSlot, sent, bound);
        }
    }
    else
    {
        // The statistics of the keys sent, from the staged pairs (round 1 always stages), by the last warp, which
        // stores no vector (at most kSourceCapacity / 2), while the first warps store the vectors.
        static_assert(kSourceCapacity / 2 <= kThreads - 32, "the last warp stores no candidate vector");
        if (threadIdx.x >= kThreads - 32)
        {
            int32_t const lane = threadIdx.x % 32;
            KeyStats t;
#pragma unroll
            for (int32_t j = 0; j < kSourceCapacity / 32; ++j)
            {
                int32_t const i = lane + 32 * j;
                if (i < static_cast<int32_t>(sent))
                {
                    t.add(stage[i].x);
                }
            }
            uint32_t const maxKey = __reduce_max_sync(0xFFFFFFFFu, t.maxKey);
            uint32_t const minKey = __reduce_min_sync(0xFFFFFFFFu, t.minKey);
            uint32_t const positives = __reduce_add_sync(0xFFFFFFFFu, t.positives);
            uint32_t const maxNeg = __reduce_max_sync(0xFFFFFFFFu, t.maxNeg);
            uint32_t const minPos = __reduce_min_sync(0xFFFFFFFFu, t.minPos);
            if (lane == 31)
            {
                store4(mcSlot, make_uint4(sent, bound, maxKey, minKey + 1u));
                store4(mcSlot + 4, make_uint4(positives, maxNeg, minPos + 1u, 0u));
            }
        }
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

// Reset of the previous exchange launch's buffer on this rank's copy. Its 2 * kMaxSources * kMaxRows slots (round 1,
// then round 2) are spread over the kResetWarps reset warps (the warps from kFirstResetWarp on) of every CTA of the
// launch, at most ten consecutive slots per warp, one per lane: each lane fetches its slot's header once the epoch is
// known (fetchResetHeaders, after the grid dependency wait and the first block barrier; an asynchronous copy into
// shared memory that completes during the local select and the send) and the warp resets the written slots while the
// poll warps wait for the candidates (resetSlots). A slot was written iff its header was; the header's count bounds
// the candidates to reset.
constexpr int32_t kResetSlots = 2 * kMaxSources * kMaxRows;

// This CTA's index among the launch's clearers, and their number (from the grid; recomputed where used).
__device__ __forceinline__ int32_t resetClearer()
{
    return static_cast<int32_t>(blockIdx.y) * kSourcesPerRank + static_cast<int32_t>(blockIdx.x);
}

__device__ __forceinline__ int32_t resetClearers()
{
    return static_cast<int32_t>(gridDim.y) * kSourcesPerRank;
}

__device__ __forceinline__ int32_t resetSlotsPerWarp(int32_t numClearers)
{
    int32_t const warps = numClearers * kResetWarps;
    return (kResetSlots + warps - 1) / warps;
}

// First slot of this (reset) warp.
__device__ __forceinline__ int32_t resetFirstSlot(int32_t clearer, int32_t numClearers)
{
    int32_t const warp = static_cast<int32_t>(threadIdx.x) / 32 - kFirstResetWarp;
    return (clearer * kResetWarps + warp) * resetSlotsPerWarp(numClearers);
}

__device__ __forceinline__ int64_t resetSlotOffset(int32_t buf, int32_t slot)
{
    int32_t const round1Slots = kMaxSources * kMaxRows;
    bool const second = slot >= round1Slots;
    int32_t const qr = second ? slot - round1Slots : slot;
    return second ? round2Offset(buf, qr / kMaxRows, qr % kMaxRows) : round1Offset(buf, qr / kMaxRows, qr % kMaxRows);
}

__device__ __forceinline__ uint32_t* resetSlotBase(CpTopkParams const& p, int32_t buf, int32_t slot)
{
    return p.ucWorkspace + resetSlotOffset(buf, slot);
}

//! Most slots one reset warp resets (one row: the fewest reset warps).
constexpr int32_t kMaxResetSlotsPerWarp
    = (kResetSlots + kSourcesPerRank * kResetWarps - 1) / (kSourcesPerRank * kResetWarps);
static_assert(kMaxResetSlotsPerWarp <= 32, "one slot header per lane");

// Shared-memory home of the reset warps' slot headers (16 bytes per slot: count, bound and two further header words),
// in the merge candidate area beyond the round-1 staging of the send; it is free from the first block barrier until
// the reset.
constexpr int32_t kResetHeaderOffset = 2 * kSourceCapacity;
static_assert(kResetHeaderOffset + 4 * kMaxResetSlotsPerWarp * kResetWarps <= kSourcesPerRank * kSourceCapacity,
    "the reset headers fit the smallest merge candidate area");

__device__ __forceinline__ uint4* resetHeaderHome(SmemLayout const& s)
{
    int32_t const warp = static_cast<int32_t>(threadIdx.x) / 32 - kFirstResetWarp;
    return reinterpret_cast<uint4*>(s.mkey + kResetHeaderOffset) + warp * kMaxResetSlotsPerWarp;
}

// Word offset, within one Lamport buffer, of the slot whose header this lane fetches for the reset; -1 for the lanes
// and warps without one. Arithmetic on the grid only (no memory access).
__device__ __forceinline__ int32_t resetHeaderOffset()
{
    int32_t const numClearers = resetClearers();
    int32_t const lane = threadIdx.x % 32;
    if (static_cast<int32_t>(threadIdx.x) / 32 < kFirstResetWarp || lane >= resetSlotsPerWarp(numClearers))
    {
        return -1;
    }
    int32_t const slot = resetFirstSlot(resetClearer(), numClearers) + lane;
    if (slot >= kResetSlots)
    {
        return -1;
    }
    static_assert(kBufferWords <= int64_t{1} << 31, "slot offsets within a buffer fit int32");
    return static_cast<int32_t>(resetSlotOffset(0, slot));
}

// Starts the copy of this lane's reset slot header of buffer `buf` (offset from resetHeaderOffset) into the warp's
// home with an asynchronous copy (cp.async, generic proxy, completed in resetSlots): a pending copy holds no register
// and no block barrier waits for it. Lanes without a slot do nothing.
__device__ __forceinline__ void fetchResetHeader(
    CpTopkParams const& p, SmemLayout const& s, int32_t buf, int32_t offset)
{
    if (offset >= 0)
    {
        uint32_t const dst = static_cast<uint32_t>(
            __cvta_generic_to_shared(resetHeaderHome(s) + static_cast<int32_t>(threadIdx.x % 32)));
        uint32_t const* const src = p.ucWorkspace + int64_t{buf} * kBufferWords + offset;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
    }
}

// Reset this reset warp's slots of buffer `buf` (headers fetched by fetchResetHeaders); other warps return.
__device__ void resetSlots(CpTopkParams const& p, SmemLayout const& s, int32_t buf)
{
    if (static_cast<int32_t>(threadIdx.x) / 32 < kFirstResetWarp)
    {
        return;
    }
    int32_t const clearer = resetClearer();
    int32_t const numClearers = resetClearers();
    // Each lane completes its own header copy; the warp barrier then makes every lane's header visible to the warp.
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncwarp();
    int32_t const per = resetSlotsPerWarp(numClearers);
    int32_t const lane = threadIdx.x % 32;
    int32_t const first = resetFirstSlot(clearer, numClearers);
    uint4 const* const headers = resetHeaderHome(s);
    uint4 const empty = make_uint4(kEmptyWord, kEmptyWord, kEmptyWord, kEmptyWord);
    for (int32_t i = 0; i < per && first + i < kResetSlots; ++i)
    {
        uint4 const h = headers[i];
        if (h.x == kEmptyWord && h.y == kEmptyWord)
        {
            continue;
        }
        int32_t const slot = first + i;
        uint32_t* base = resetSlotBase(p, buf, slot);
        uint32_t const capacity = slot >= kMaxSources * kMaxRows ? kRound2Capacity : kSourceCapacity;
        uint32_t const count = h.x == kEmptyWord ? capacity : min(h.x, capacity);
        int32_t const vectors = static_cast<int32_t>(count + 1u) / 2;
        for (int32_t v = lane; v < vectors; v += 32)
        {
            store4(base + kSlotHeaderWords + 4 * static_cast<int64_t>(v), empty);
        }
        if (lane < 2)
        {
            store4(base + 4 * lane, empty);
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

// ---- Thread-block cluster primitives (one cluster of kClusterCtas CTAs per row; cluster rank = blockIdx.x) ----------

__device__ __forceinline__ uint32_t smemAddr(void const* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Shared-memory window address of `ptr` (this CTA's shared memory) in CTA `rank` of the cluster.
__device__ __forceinline__ uint32_t clusterAddr(void const* ptr, int32_t rank)
{
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(smemAddr(ptr)), "r"(rank));
    return remote;
}

// Initialize the exchange barriers and announce it to the cluster (relaxed arrive; clusterInitWait completes it
// before the first exchange). Touches shared memory only.
__device__ __forceinline__ void clusterInit(ClusterRecords& rec)
{
    if (threadIdx.x == 0)
    {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smemAddr(&rec.bar[0])) : "memory");
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smemAddr(&rec.bar[1])) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

__device__ __forceinline__ void clusterInitWait()
{
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

// This CTA's single arrival on exchange barrier `bar`, expecting `bytes` from the cluster's st.async stores.
__device__ __forceinline__ void expectBytes(uint64_t* bar, uint32_t bytes)
{
    asm volatile(
        "{\n\t.reg .b64 st;\n\tmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n\t}" ::"r"(smemAddr(bar)),
        "r"(bytes)
        : "memory");
}

// Wait until the phase of exchange barrier `bar` with parity `parity` completed (every expected byte arrived).
__device__ __forceinline__ void waitExchange(uint64_t* bar, uint32_t parity)
{
    uint32_t done;
    do
    {
        asm volatile(
            "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n\t"
            "selp.u32 %0, 1, 0, p;\n\t}"
            : "=r"(done)
            : "r"(smemAddr(bar)), "r"(parity)
            : "memory");
    } while (done == 0u);
}

// Store 16 / 8 bytes at `ptr` (a record of this CTA's layout) in CTA `rank`, completing them on its barrier `bar`.
__device__ __forceinline__ void sendTo(int32_t rank, void const* ptr, uint64_t const* bar, uint4 v)
{
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(
                     clusterAddr(ptr, rank)),
                 "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(clusterAddr(bar, rank))
                 : "memory");
}

__device__ __forceinline__ void sendTo(int32_t rank, void const* ptr, uint64_t const* bar, uint2 v)
{
    asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [%0], {%1, %2}, [%3];" ::"r"(
                     clusterAddr(ptr, rank)),
                 "r"(v.x), "r"(v.y), "r"(clusterAddr(bar, rank))
                 : "memory");
}

// Exchange sequence of a row: per pass x = 2 * pass group totals and x = 2 * pass + 1 the selected group's bins.
// Every CTA sends the records of exchange x + 1 only after its own exchange x completed, so a barrier phase (and an
// alternating record set) is reused only after every CTA consumed the previous one.
__device__ __forceinline__ uint64_t* exchangeBar(ClusterRecords& rec, uint32_t x)
{
    return &rec.bar[x & 1u];
}

__device__ __forceinline__ uint32_t exchangeParity(uint32_t x)
{
    return (x >> 1) & 1u;
}

// Inclusive scan over the lanes of a warp.
__device__ __forceinline__ uint32_t warpInclusiveScan(uint32_t v)
{
    int32_t const lane = threadIdx.x % 32;
#pragma unroll
    for (int32_t o = 1; o < 32; o <<= 1)
    {
        uint32_t const y = __shfl_up_sync(0xFFFFFFFFu, v, o);
        if (lane >= o)
        {
            v += y;
        }
    }
    return v;
}

// Tallies of the cluster select over the candidates of the cluster CTAs before this one (in candidate order), kept
// by warp 0 and published in shared memory: those strictly above the selected bins of all passes so far, and those
// in the selected bin of the last pass.
struct CtaTally
{
    uint32_t aboveBefore;
    uint32_t binBefore;
};

// Cluster radix select of the need-th largest key among the candidates of every CTA of the cluster (this thread
// holds k0 and k1; kEmptyWord is no candidate, as it lies above every range). It computes exactly what blockSelect
// with cap = 0 computes over the union of the candidates: the same digits (from `base` and topBits), the same bins
// and counts from the cluster-summed histograms, and the same stop rules. Per pass: every CTA histograms its own
// candidates and sends its totals per group of kGroupBins bins to every CTA (st.async, two groups per warp); warp 0
// of every CTA finds, from the top, the group holding the target, sends its own counts of that group's bins to every
// CTA, and finds the bin from the eight CTAs' counts. From the same records, warp 0 also tallies the candidates of
// the CTAs before this one above the selected bins and in the last selected bin (`tally`), which give this CTA's
// emission offsets without another exchange.
__device__ SelectOut clusterSelect(SmemLayout const& s, ClusterRecords& rec, CtaTally& tally, uint32_t k0, uint32_t k1,
    uint32_t lo, uint64_t hi, uint32_t need, uint32_t base, int32_t topBits, uint32_t aboveHi, int64_t* dbg)
{
    int32_t const me = blockIdx.x;
    int32_t const lane = threadIdx.x % 32;
    int32_t const warp = threadIdx.x / 32;
    SelectOut out{lo, 0u, 0u, 0u, false, false, 0};
    uint32_t prefMask = 0;
    uint32_t prefVal = 0;
    uint32_t above = aboveHi;
    int32_t remaining = max(topBits, 1);
    uint32_t* hist = s.hist;
    uint32_t* histNext = s.hist + kBins;
    // Warp 0: the running tallies of the CTAs before this one.
    uint32_t aboveBefore = 0u;
    uint32_t binBefore = 0u;
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
        int32_t const groups = (nbins + kGroupBins - 1) / kGroupBins;
        int32_t const pairs = (groups + 1) / 2;
        uint32_t const xg = 2u * pass;
        uint32_t const xb = xg + 1u;
        out.passes = pass + 1;
        if (dbg != nullptr && threadIdx.x == 0 && pass < 3)
        {
            dbg[1 + 3 * pass] = clock64();
        }
        if (threadIdx.x == 0)
        {
            expectBytes(exchangeBar(rec, xg), kClusterCtas * pairs * 8);
        }
        // The next pass's histogram: its last reads (the previous pass's bins sent by warp 0) are complete.
        if (shift > 0)
        {
            for (int32_t b = threadIdx.x; b < (1 << min(kDigitBits, shift)); b += kThreads)
            {
                histNext[b] = 0;
            }
        }
        histogramKey(hist, true, k0, lo, hi, base, prefMask, prefVal, shift, binMask);
        histogramKey(hist, true, k1, lo, hi, base, prefMask, prefVal, shift, binMask);
        __syncthreads();
        uint32_t(*const groupRec)[kMaxGroups] = rec.groups[pass & 1];
        uint32_t(*const binRec)[kGroupBins] = rec.bins[pass & 1];
        if (warp < pairs)
        {
            int32_t const b0 = 2 * warp * kGroupBins + lane;
            int32_t const b1 = b0 + kGroupBins;
            uint32_t const t0 = __reduce_add_sync(0xFFFFFFFFu, b0 < nbins ? hist[b0] : 0u);
            uint32_t const t1 = __reduce_add_sync(0xFFFFFFFFu, b1 < nbins ? hist[b1] : 0u);
            if (lane < kClusterCtas)
            {
                sendTo(lane, &groupRec[me][2 * warp], exchangeBar(rec, xg), make_uint2(t0, t1));
            }
        }
        uint32_t const target = need - above;
        if (warp == 0)
        {
            waitExchange(exchangeBar(rec, xg), exchangeParity(xg));
            if (dbg != nullptr && lane == 0 && pass < 3)
            {
                dbg[2 + 3 * pass] = clock64();
            }
            // Lane l sums the groups at positions 2l and 2l + 1 from the top over the cluster.
            int32_t const ga = groups - 1 - 2 * lane;
            int32_t const gb = ga - 1;
            uint32_t a = 0;
            uint32_t b = 0;
            uint32_t beforeA = 0;
            uint32_t beforeB = 0;
#pragma unroll
            for (int32_t c = 0; c < kClusterCtas; ++c)
            {
                uint32_t const x = ga >= 0 ? groupRec[c][ga] : 0u;
                uint32_t const y = gb >= 0 ? groupRec[c][gb] : 0u;
                a += x;
                b += y;
                beforeA += c < me ? x : 0u;
                beforeB += c < me ? y : 0u;
            }
            uint32_t const incl = warpInclusiveScan(a + b);
            uint32_t const before = incl - a - b;
            uint32_t const total = __shfl_sync(0xFFFFFFFFu, incl, 31);
            bool const holds = before < target && target <= incl;
            uint32_t const holder = __ballot_sync(0xFFFFFFFFu, holds);
            if (lane == 0)
            {
                s.scal[kSlotTotal] = total;
            }
            if (holder != 0u)
            {
                int32_t const hl = __ffs(holder) - 1;
                bool const first = target <= before + a;
                int32_t const gSel = __shfl_sync(0xFFFFFFFFu, first ? ga : gb, hl);
                uint32_t const gAbove = __shfl_sync(0xFFFFFFFFu, first ? before : before + a, hl);
                if (lane == 0)
                {
                    expectBytes(exchangeBar(rec, xb), kClusterCtas * kGroupBins * 4);
                }
                // This CTA's counts of the selected group's bins to every CTA: lane l sends vector l % 8 to CTAs
                // l / 8 and l / 8 + 4.
                int32_t const v = lane % 8;
                int32_t const b0 = gSel * kGroupBins + 4 * v;
                uint4 const q = make_uint4(b0 < nbins ? hist[b0] : 0u, b0 + 1 < nbins ? hist[b0 + 1] : 0u,
                    b0 + 2 < nbins ? hist[b0 + 2] : 0u, b0 + 3 < nbins ? hist[b0 + 3] : 0u);
                sendTo(lane / 8, &binRec[me][4 * v], exchangeBar(rec, xb), q);
                sendTo(lane / 8 + 4, &binRec[me][4 * v], exchangeBar(rec, xb), q);
                // While the bins travel: the earlier CTAs' candidates in the groups above the selected one (both
                // groups of the lanes before the holder, and the holder's first group when it selected the second).
                bool const selFirst = __shfl_sync(0xFFFFFFFFu, first, hl);
                aboveBefore += __reduce_add_sync(
                    0xFFFFFFFFu, lane < hl ? beforeA + beforeB : (lane == hl && !selFirst ? beforeA : 0u));
                waitExchange(exchangeBar(rec, xb), exchangeParity(xb));
                // Bins of the selected group from the top: lane l holds bin 32 * gSel + 31 - l.
                int32_t const bin = gSel * kGroupBins + kGroupBins - 1 - lane;
                uint32_t c = 0;
                uint32_t cBefore = 0;
#pragma unroll
                for (int32_t d = 0; d < kClusterCtas; ++d)
                {
                    uint32_t const x = binRec[d][kGroupBins - 1 - lane];
                    c += x;
                    cBefore += d < me ? x : 0u;
                }
                uint32_t const y = warpInclusiveScan(c);
                uint32_t const acc = gAbove + y - c;
                bool const found = acc < target && target <= acc + c;
                if (found)
                {
                    s.scal[kSlotBin] = static_cast<uint32_t>(bin);
                    s.scal[kSlotAbove] = acc;
                    s.scal[kSlotBinCount] = c;
                }
                // The earlier CTAs' candidates in the bins above the selected one (lanes before the finder) and in
                // the selected bin (the finder's).
                int32_t const finder = __ffs(__ballot_sync(0xFFFFFFFFu, found)) - 1;
                aboveBefore += __reduce_add_sync(0xFFFFFFFFu, lane < finder ? cBefore : 0u);
                binBefore = __shfl_sync(0xFFFFFFFFu, cBefore, finder);
            }
            else if (lane == 0)
            {
                // Every key is taken (pass 0 only, the same decision in every CTA): the bin exchange still
                // completes, with no bytes, so that the exchange sequence stays aligned.
                expectBytes(exchangeBar(rec, xb), 0u);
                waitExchange(exchangeBar(rec, xb), exchangeParity(xb));
            }
            if (lane == 0)
            {
                tally.aboveBefore = aboveBefore;
                tally.binBefore = binBefore;
            }
        }
        __syncthreads();
        if (dbg != nullptr && threadIdx.x == 0 && pass < 3)
        {
            dbg[3 + 3 * pass] = clock64();
        }
        uint32_t const total = s.scal[kSlotTotal];
        if (pass == 0)
        {
            out.total = total + aboveHi;
            if (total + aboveHi == 0u || target > total)
            {
                out.all = true;
                out.above = total + aboveHi;
                return out;
            }
        }
        // The scalars are next written by warp 0 after the next histogram's block barrier.
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
        if (remaining > 0 && above + out.binCount == need)
        {
            out.full = true;
            return out;
        }
    }
    return out;
}

// Spin until the header of a round-1 slot was written and, when the slot's count covers candidate vector kv, that
// vector too; returns the header's count (clamped to `cap`) and bound. Both are polled together, so neither waits
// for the other; aborts the kernel after the timeout.
__device__ uint2 pollSlotVector(uint32_t const* slot, int32_t kv, uint32_t cap, uint4& v, WaitCtx const& w)
{
    uint32_t const* vec = slot + kSlotHeaderWords + 4 * kv;
    uint2 h = loadVolatile2(slot);
    v = loadVolatile4(vec);
    uint64_t start = 0;
    for (int32_t polls = 1;; ++polls)
    {
        bool const headerReady = !isEmpty(h);
        uint32_t const cnt = headerReady ? min(h.x, cap) : 0u;
        if (headerReady && (2u * kv >= cnt || vectorReady(v, 2u * kv + 1u < cnt)))
        {
            return make_uint2(cnt, h.y);
        }
        if (polls % kPollsPerTimerRead == 0)
        {
            uint64_t const now = globalTimerNs();
            if (start == 0)
            {
                start = now;
            }
            else if (now - start > w.timeoutNs)
            {
                abortOnTimeout(w, headerReady ? kWaitRound1Candidates : kWaitRound1Header);
            }
        }
        if (!headerReady)
        {
            h = loadVolatile2(slot);
        }
        v = loadVolatile4(vec);
    }
}

// Spin until the whole 32-byte round-1 header of a slot was written; aborts the kernel after the timeout.
__device__ void pollHeader(uint32_t const* slot, uint4& a, uint4& b, WaitCtx const& w)
{
    a = loadVolatile4(slot);
    b = loadVolatile4(slot + 4);
    uint64_t start = 0;
    for (int32_t polls = 1; !(vectorReady(a, true) && vectorReady(b, true)); ++polls)
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
                abortOnTimeout(w, kWaitRound1Header);
            }
        }
        a = loadVolatile4(slot);
        b = loadVolatile4(slot + 4);
    }
}

// Merge of one row by its cluster. CTA c gathers the round-1 candidates of sources [c * world, (c + 1) * world) (one
// 16-byte vector, two candidate slots, per thread of the poll warps, kept in registers: thread order is crossRankTopk's
// candidate order) while its header warps read every source's header (counts, bounds and the key statistics of the keys
// sent, so every CTA has the row's statistics and can decide every source's overflow without an exchange). The cluster
// runs crossRankTopk's merge select on the summed histograms (clusterSelect), whose records also give every CTA's
// number of keys above the threshold and of ties, and every CTA writes its selected candidates at crossRankTopk's
// positions: keys above the threshold in candidate order, then ties in candidate order while they fit, then -1. Returns
// true, writing nothing, when a source withheld keys that may lie above the threshold (the same decision on every CTA
// and rank); the row is then merged by mergeRow.
__device__ bool clusterMergeRow(CpTopkParams const& p, SmemLayout const& s, ClusterRecords& rec, CtaTally& tally,
    int32_t buf, int32_t row, int32_t rowLen, int32_t* outRow, WaitCtx const& wait)
{
    int32_t const world = p.worldSize;
    int32_t const sources = numSources(world);
    int32_t const me = blockIdx.x;
    int32_t const tid = threadIdx.x;
    int32_t const lane = tid % 32;
    int32_t const warp = tid / 32;
    uint32_t const cap = static_cast<uint32_t>(roundCap(world));
    int32_t const vectorsPerSource = static_cast<int32_t>(cap) / 2;
    int32_t const q0 = me * world;
    int64_t* const dbg = debugSlots(p, 0);
    // This thread's candidates (kEmptyWord: none; never a key).
    uint32_t k0 = kEmptyWord;
    uint32_t k1 = kEmptyWord;
    if (tid == 0)
    {
        s.scal[kSlotCounter2] = 0u;
    }
    if (tid < world * vectorsPerSource)
    {
        int32_t const q = q0 + tid / vectorsPerSource;
        int32_t const kv = tid % vectorsPerSource;
        uint4 v;
        uint2 const h = pollSlotVector(p.ucWorkspace + round1Offset(buf, q, row), kv, cap, v, wait);
        if (2u * kv < h.x)
        {
            k0 = v.x;
            if (2u * kv + 1u < h.x)
            {
                k1 = v.z;
            }
        }
        // The tokens wait in shared memory until the emission (s.midx is free in the cluster merge).
        *reinterpret_cast<uint2*>(s.midx + 2 * tid) = make_uint2(v.y, v.w);
    }
    else if (warp >= kPollWarps && warp < kFirstResetWarp)
    {
        // Header warps: thread h reads source h's header; per source its count (clamped to the round-1 capacity),
        // bound and length, and per header warp the reduced key statistics.
        int32_t const q = tid - kPollWarps * 32;
        KeyStats st;
        if (q < sources)
        {
            uint4 a;
            uint4 b;
            pollHeader(p.ucWorkspace + round1Offset(buf, q, row), a, b, wait);
            s.scal[kSlotSrcCount + q] = min(a.x, cap);
            s.scal[kSlotSrcBound + q] = a.y;
            s.scal[kSlotSrcLen + q] = static_cast<uint32_t>(sourceLength(
                min(localLength(rowLen, q / kSourcesPerRank, world), p.localCapacity), q % kSourcesPerRank));
            st.maxKey = a.z;
            st.minKey = a.w - 1u;
            st.positives = b.x;
            st.maxNeg = b.y;
            st.minPos = b.z - 1u;
        }
        uint32_t const cnt = q < sources ? s.scal[kSlotSrcCount + q] : 0u;
        // Candidates and keys >= +0.0 of the cluster CTAs before this one (CTA c merges sources
        // [c * world, (c + 1) * world)), of this one and of the row.
        int32_t const c = q < sources ? q / world : kClusterCtas;
        uint32_t const nBefore = __reduce_add_sync(0xFFFFFFFFu, c < me ? cnt : 0u);
        uint32_t const nMine = __reduce_add_sync(0xFFFFFFFFu, c == me ? cnt : 0u);
        uint32_t const nAll = __reduce_add_sync(0xFFFFFFFFu, cnt);
        uint32_t const pBefore = __reduce_add_sync(0xFFFFFFFFu, c < me ? st.positives : 0u);
        if (lane == 0)
        {
            atomicAdd(&s.scal[kSlotCountBefore], nBefore);
            atomicAdd(&s.scal[kSlotCountMine], nMine);
            atomicAdd(&s.scal[kSlotCountTotal], nAll);
            atomicAdd(&s.scal[kSlotPositivesBefore], pBefore);
        }
        uint32_t const v0 = __reduce_max_sync(0xFFFFFFFFu, st.maxKey);
        uint32_t const v1 = __reduce_min_sync(0xFFFFFFFFu, st.minKey);
        uint32_t const v2 = __reduce_add_sync(0xFFFFFFFFu, st.positives);
        uint32_t const v3 = __reduce_max_sync(0xFFFFFFFFu, st.maxNeg);
        uint32_t const v4 = __reduce_min_sync(0xFFFFFFFFu, st.minPos);
        if (lane == 0)
        {
            // In the scan scratch (s.stats may still be read by the send's last warp; s.warpSums is next written
            // by the emission scan, after the reads below).
            uint32_t* const out = s.warpSums + kNumKeyStats * (warp - kPollWarps);
            out[0] = v0;
            out[1] = v1;
            out[2] = v2;
            out[3] = v3;
            out[4] = v4;
        }
    }
    stamp(dbg, 45);
    __syncthreads();
    markPhase(p, 4);

    // Statistics of the row's candidates from every source's header (identical in every CTA and rank).
    KeyStats g;
#pragma unroll
    for (int32_t w = 0; w < kHeaderWarps; ++w)
    {
        uint32_t const* const in = s.warpSums + kNumKeyStats * w;
        g.maxKey = max(g.maxKey, in[0]);
        g.minKey = min(g.minKey, in[1]);
        g.positives += in[2];
        g.maxNeg = max(g.maxNeg, in[3]);
        g.minPos = min(g.minPos, in[4]);
    }
    uint32_t const total = s.scal[kSlotCountTotal];
    stamp(dbg, 61);
    markPhase(p, 5);

    SignRange const r
        = total > 0u ? signRange(g, static_cast<uint32_t>(kTopK)) : SignRange{0u, uint64_t{kEmptyWord}, 32, 0u, false};
    clusterInitWait();
    SelectOut const m = clusterSelect(
        s, rec, tally, k0, k1, r.lo, r.hi, static_cast<uint32_t>(kTopK), r.lo, r.bits, r.aboveHi, debugSlots(p, 48));
    putDebug(debugSlots(p, 48), 0, s.scal[kSlotCountMine]);
    putDebug(dbg, 62, total);
    putDebug(dbg, 63, m.passes + (r.negative ? 10 : 0));
    bool takeAll = false;
    uint32_t const kth = mergeThreshold(m, takeAll);
    markPhase(p, 6);

    // Keys above the threshold and ties of the cluster CTAs before this one and of the row (identical in every CTA),
    // from the select's tallies; with the threshold below a wholly taken bin the ties fall beyond kTopK and are taken
    // as keys above it.
    uint32_t gtBefore;
    uint32_t eqBefore;
    uint32_t allGt;
    uint32_t allEq;
    if (takeAll)
    {
        gtBefore = s.scal[kSlotCountBefore];
        eqBefore = 0u;
        allGt = total;
        allEq = 0u;
    }
    else
    {
        uint32_t const aboveBefore = tally.aboveBefore + (r.aboveHi > 0u ? s.scal[kSlotPositivesBefore] : 0u);
        gtBefore = m.full ? aboveBefore + tally.binBefore : aboveBefore;
        eqBefore = m.full ? 0u : tally.binBefore;
        allGt = m.full ? m.above + m.binCount : m.above;
        allEq = m.full ? 0u : m.binCount;
    }
    // This CTA's positions, scanned in thread order; a source of the row that withheld keys above the threshold
    // sends the row to the second round.
    bool const gt0 = k0 != kEmptyWord && (takeAll || k0 > kth);
    bool const eq0 = k0 != kEmptyWord && !takeAll && k0 == kth;
    bool const gt1 = k1 != kEmptyWord && (takeAll || k1 > kth);
    bool const eq1 = k1 != kEmptyWord && !takeAll && k1 == kth;
    if (tid < sources && sourceOverflows(s, tid, kth))
    {
        atomicOr(&s.scal[kSlotCounter2], 1u);
    }
    // Exclusive prefix in thread order (k0 before k1): within the warp from ballots, across warps from the warp
    // totals (one block barrier).
    uint32_t const below = (1u << lane) - 1u;
    uint32_t const bGt0 = __ballot_sync(0xFFFFFFFFu, gt0);
    uint32_t const bGt1 = __ballot_sync(0xFFFFFFFFu, gt1);
    uint32_t const bEq0 = __ballot_sync(0xFFFFFFFFu, eq0);
    uint32_t const bEq1 = __ballot_sync(0xFFFFFFFFu, eq1);
    if (lane == 0)
    {
        s.warpSums[warp] = static_cast<uint32_t>(__popc(bGt0) + __popc(bGt1))
            + (static_cast<uint32_t>(__popc(bEq0) + __popc(bEq1)) << 16);
    }
    __syncthreads();
    uint32_t const warpsBefore = __reduce_add_sync(0xFFFFFFFFu, lane < warp ? s.warpSums[lane] : 0u);
    uint32_t const before = warpsBefore + static_cast<uint32_t>(__popc(bGt0 & below) + __popc(bGt1 & below))
        + (static_cast<uint32_t>(__popc(bEq0 & below) + __popc(bEq1 & below)) << 16);
    stamp(dbg, 58);
    if (s.scal[kSlotCounter2] != 0u)
    {
        return true;
    }
    uint32_t const want = static_cast<uint32_t>(kTopK);
    uint32_t posGt = gtBefore + (before & 0xFFFFu);
    uint32_t posEq = allGt + eqBefore + (before >> 16);
    uint32_t x0 = 0u;
    uint32_t x1 = 0u;
    if (k0 != kEmptyWord)
    {
        uint2 const x = *reinterpret_cast<uint2 const*>(s.midx + 2 * tid);
        x0 = x.x;
        x1 = x.y;
    }
    if (gt0 && posGt < want)
    {
        outRow[posGt] = static_cast<int32_t>(x0);
    }
    posGt += gt0 ? 1u : 0u;
    if (eq0 && posEq < want)
    {
        outRow[posEq] = static_cast<int32_t>(x0);
    }
    posEq += eq0 ? 1u : 0u;
    if (gt1 && posGt < want)
    {
        outRow[posGt] = static_cast<int32_t>(x1);
    }
    if (eq1 && posEq < want)
    {
        outRow[posEq] = static_cast<int32_t>(x1);
    }
    uint32_t const selected = min(allGt + allEq, want);
    for (uint32_t j = selected + static_cast<uint32_t>(me * kThreads + tid); j < want; j += kClusterCtas * kThreads)
    {
        outRow[j] = -1;
    }
    stamp(dbg, 60);
    return false;
}

__global__ void __launch_bounds__(kThreads, 1) cpTopkZ9Kernel(CpTopkParams p)
{
    markPhase(p, 0);
    extern __shared__ __align__(16) uint32_t smemRaw[];
    __shared__ uint32_t const* segSlots[kMaxSources];
    __shared__ uint32_t segBegin[kMaxSources + 1];
    __shared__ __align__(16) ClusterRecords rec;
    __shared__ CtaTally tally;
    __shared__ int32_t epochShared;
    int32_t const world = p.worldSize;
    int32_t const rank = p.rank;
    int32_t const src = blockIdx.x;
    int32_t const row = blockIdx.y;
    SmemLayout const s = carveSmem(smemRaw, p.localCapacity, world);
    int32_t* outRow = p.out + static_cast<int64_t>(row) * kTopK;
    clusterInit(rec);
    if (threadIdx.x <= kSlotPositivesBefore - kSlotCountBefore)
    {
        s.scal[kSlotCountBefore + threadIdx.x] = 0u;
    }
    int32_t const resetOffset = resetHeaderOffset();
    // Every global read (lengths, logits, the epoch word, the workspace) comes after the grid dependency wait, so a
    // programmatic early launch is safe whatever the preceding kernels write.
    asm volatile("griddepcontrol.wait;" ::: "memory");
    markPhase(p, 10);
    // Thread 0 and the reset warps read this CTA's epoch word, written only by the previous exchange launch of this
    // op, before the key loads; nothing waits for the value until the key loads are done. Thread 0 publishes it to the
    // CTA at the block barrier that follows the key loads, and the reset warps start the copies of the previous
    // buffer's headers right before that barrier (slot offsets computed before the wait, from the grid only).
    int32_t* const epochWord = p.state + kEpochBase + row * kSourcesPerRank + src;
    int32_t epochRead = 0;
    if (threadIdx.x == 0 || static_cast<int32_t>(threadIdx.x) >= kFirstResetWarp * 32)
    {
        epochRead = *reinterpret_cast<int32_t volatile*>(epochWord);
    }
    int32_t const prevBufRead = (epochRead + kNumLamportBuffers - 1) % kNumLamportBuffers;
    int32_t const kvLen = p.kvLens[0];
    int32_t const rowLen = max(0, kvLen - p.nextN + row + 1);
    // Identity launch (every row has at most kTopK tokens; the same decision on every rank): no exchange, and
    // neither the workspace nor the epoch words are touched, so identity launches never shorten the reuse chain of
    // the Lamport buffers (see crossRankTopkClusterMerge.h).
    bool const exchange = kvLen - p.nextN + p.numRows > kTopK;
    // Thread 0 advances this CTA's epoch word (the CTAs of the last row also those of the rows this launch does not
    // have, so every word counts every exchange launch), after a block barrier that follows every read of the word in
    // this CTA. Nothing else of this launch reads the epoch words, and the next launch reads them only after its grid
    // dependency wait.
    auto advanceEpoch = [&]()
    {
        if (exchange && threadIdx.x == 0)
        {
            int32_t const rows = row == static_cast<int32_t>(gridDim.y) - 1 ? kMaxRows - row : 1;
            for (int32_t r = 0; r < rows; ++r)
            {
                epochWord[r * kSourcesPerRank] = epochRead + 1;
            }
        }
    };

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
        if (exchange)
        {
            fetchResetHeader(p, s, prevBufRead, resetOffset);
            resetSlots(p, s, prevBufRead);
            markPhase(p, 9);
        }
        __syncthreads();
        advanceEpoch();
        clusterInitWait();
    }
    else
    {
        int32_t const localLen = min(localLength(rowLen, rank, world), p.localCapacity);
        float const* rowLogits = p.logits + static_cast<int64_t>(row) * p.logitsStride;

        // ---- Round 1: this source's local top keys ---------------------------------------------------------
        int32_t const n = loadSource(s, rowLogits, localLen, src, true);
        if (threadIdx.x == 0)
        {
            epochShared = epochRead;
        }
        // The reset warps start fetching the headers of their share of the previous exchange launch's buffer (an
        // exchange row implies an exchange launch); the values are first used by the reset after the send.
        fetchResetHeader(p, s, prevBufRead, resetOffset);
        __syncthreads();
        markPhase(p, 1);
        advanceEpoch();
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
        int32_t const sendBuf = epochShared % kNumLamportBuffers;
        uint32_t const sent = sendSource<false>(s, sel, n, selLo, uint64_t{kEmptyWord}, cap,
            p.mcWorkspace + round1Offset(sendBuf, rank * kSourcesPerRank + src, row), src, rank, world,
            reinterpret_cast<uint2*>(s.mkey), debugSlots(p, 0));
        putDebug(debugSlots(p, 0), 46, sent);
        putDebug(debugSlots(p, 0), 47, passInfo);
        markPhase(p, 3);

        // The reset warps reset the previous exchange launch's buffer, whose candidates every rank has consumed (a
        // peer writes it again only two exchange launches later, after this launch completed on this rank), while
        // the poll and header warps wait for this launch's candidates.
        resetSlots(p, s, (epochShared + kNumLamportBuffers - 1) % kNumLamportBuffers);
        int32_t const buf = epochShared % kNumLamportBuffers;
        if (threadIdx.x == kFirstResetWarp * 32)
        {
            int64_t* const d = debugSlots(p, 0);
            if (d != nullptr)
            {
                d[9] = static_cast<int64_t>(globalTimerNs());
            }
        }

        // ---- Merge: the row's cluster -----------------------------------------------------------------------
        WaitCtx const wait{p.state, p.timeoutNs, rank};
        if (clusterMergeRow(p, s, rec, tally, buf, row, rowLen, outRow, wait) && src == 0)
        {
            // A source withheld keys above the threshold: crossRankTopk's merge, with its second round, on CTA 0.
            mergeRow(p, s, segSlots, segBegin, buf, row, rowLen, localLen, rowLogits, outRow, wait);
            int64_t* const dbg = debugSlots(p, 0);
            putDebug(dbg, 63, dbg != nullptr ? dbg[63] + 100 : 0);
        }
        markPhase(p, 7);
    }
    markPhase(p, 8);
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

int32_t cpTopkMaxActiveClusters(int32_t worldSize)
{
    // Queried once per world size and device (the value + 1; 0 = not queried yet).
    constexpr int32_t kMaxDevices = 64;
    static int32_t cached[kMaxDevices][kMaxRanks + 1] = {};
    if (worldSize < 1 || worldSize > kMaxRanks)
    {
        return 0;
    }
    int device = 0;
    cudaGetDevice(&device);
    if (device >= 0 && device < kMaxDevices && cached[device][worldSize] > 0)
    {
        return cached[device][worldSize] - 1;
    }
    int32_t const capacity = cpTopkMaxLocalCapacity(worldSize);
    if (capacity == 0)
    {
        return 0;
    }
    size_t const smem = smemBytes(capacity, worldSize);
    if (cudaFuncSetAttribute(cpTopkZ9Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem))
        != cudaSuccess)
    {
        (void) cudaGetLastError();
        return 0;
    }
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(kSourcesPerRank, kMaxRows);
    config.blockDim = dim3(kThreads);
    config.dynamicSmemBytes = smem;
    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeClusterDimension;
    attr.val.clusterDim.x = kSourcesPerRank;
    attr.val.clusterDim.y = 1;
    attr.val.clusterDim.z = 1;
    config.attrs = &attr;
    config.numAttrs = 1;
    int clusters = 0;
    if (cudaOccupancyMaxActiveClusters(&clusters, cpTopkZ9Kernel, &config) != cudaSuccess)
    {
        (void) cudaGetLastError();
        return 0;
    }
    if (device >= 0 && device < kMaxDevices)
    {
        cached[device][worldSize] = clusters + 1;
    }
    return clusters;
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
            throw std::invalid_argument(std::string("glm_moe_dsa_cross_rank_topk_cluster_merge: bad "
                                                    "TRTLLM_STAIRCASE_GLM_LL_CROSS_RANK_TOPK_TIMEOUT_MS=")
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
        throw std::invalid_argument("glm_moe_dsa_cross_rank_topk_cluster_merge: unsupported rows/world/rank");
    }
    size_t const smem = smemBytes(params.localCapacity, params.worldSize);
    // The attribute is per function and device; raise it only when a larger footprint is requested.
    static thread_local size_t configuredSmem[64] = {};
    int device = 0;
    cudaGetDevice(&device);
    cudaError_t err = cudaSuccess;
    if (device >= 64 || configuredSmem[device] < smem)
    {
        err = cudaFuncSetAttribute(cpTopkZ9Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
        if (err != cudaSuccess)
        {
            throw std::runtime_error(
                std::string("glm_moe_dsa_cross_rank_topk_cluster_merge: smem attribute: ") + cudaGetErrorString(err));
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
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = kSourcesPerRank;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = enablePdl ? 1 : 0;
    config.attrs = attrs;
    config.numAttrs = 2;
    err = cudaLaunchKernelEx(&config, cpTopkZ9Kernel, params);
    if (err == cudaSuccess)
    {
        err = cudaGetLastError();
    }
    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            std::string("glm_moe_dsa_cross_rank_topk_cluster_merge: launch: ") + cudaGetErrorString(err));
    }
}

} // namespace cross_rank_topk_cluster_merge

TRTLLM_MV2_NAMESPACE_END
