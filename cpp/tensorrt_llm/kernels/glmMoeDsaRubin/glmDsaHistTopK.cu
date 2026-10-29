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

#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmDsaHistTopK.h"

#include "tensorrt_llm/common/envUtils.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <string>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::glm_moe_dsa::hist_topk
{
namespace
{

constexpr int32_t kCtas = kCtasPerRow;
constexpr int32_t kThreads = 1024;
static_assert(kThreads / 32 == 32, "the block scans assume 32 warps");
constexpr uint32_t kFull = 0xFFFFFFFFu;
constexpr int32_t kMaxGroups = kBins / kGroupBins;
constexpr int32_t kFineWords = kBins / 2;
static_assert(kHistWords == kFineWords + kMaxGroups, "histogram words: bin-count pairs, then group totals");
constexpr uint32_t kEmpty = 0xFFFFFFFFu;
//! Exchanged bin counts saturate here; a group holds at most one saturated bin.
constexpr uint32_t kSat = 0xFFFFu;
static_assert(2 * int64_t{kSat} > kMaxShard, "two saturated bins would hold more keys than a shard");
//! float4 of logits per thread (the keys are staged in shared memory).
constexpr int32_t kVecs = 3;
static_assert(kMaxShard == kCtas * kThreads * 4 * kVecs, "three float4 per thread at the largest shard");
constexpr int32_t kKeys = 4 * kVecs;
constexpr int32_t kOutPerCta = kTopK / kCtas;
constexpr int32_t kCtasPerLaunch = kCtas * kMaxRows;
//! Boundary sub-bins of at most this many pairs are ranked directly.
constexpr int32_t kSmallRank = 64;
constexpr int64_t kHistRegionWords = int64_t{kMaxRows} * kMaxRanks * kHistWords;
constexpr int64_t kOutRegionWords = int64_t{kMaxRows} * kTopK;

enum StateSlot : int32_t
{
    kStateCount = 0, // 64-bit: CTAs of exchange launches (each adds one once it read the count)
    kStateError = 2,
    kStateRecords = 4
};

constexpr int32_t kRecordWords = 4;
static_assert(kStateRecords + kMaxRows * kRecordWords == kStateWords, "state layout");

enum Mode : int32_t
{
    kModeIdentity = 0, // no exchange for the row
    kModeFull = 1,     // the boundary bin is taken whole
    kModeTies = 2,     // the exact key: its ties are taken by count, the higher ranks first
    kModeCand = 3,     // the boundary pairs are exchanged and resolved on every rank
    kModeNext = 4      // the boundary bin is refined with the next digit
};

enum DebugSlot : int32_t
{
    // %globaltimer stamps
    kDbgEntry = 0,
    kDbgWaited = 1,
    kDbgLoaded = 2,
    kDbgLocalHist = 3,
    kDbgB0 = 4,
    kDbgSent0 = 5,
    kDbgSuffix = 6,
    kDbgBucketed = 7,
    kDbgCoarse = 8,
    kDbgGroup = 9,
    kDbgFine = 10,
    kDbgCut = 11,
    kDbgPeer = 12,
    kDbgClassify = 13,
    kDbgStaged = 14,
    kDbgFinalSent = 15,
    kDbgOutCopied = 16,
    kDbgBndPolled = 17,
    kDbgPass1 = 18,
    kDbgThreshold = 19,
    kDbgResolved = 20,
    kDbgEnd = 21,
    kDbgCleared = 22,
    kDbgPreWaited = 24,
    kDbgTokSent = 25,
    kDbgClockEntry = 26, // clock64 (not %globaltimer)
    kDbgClockEnd = 27,
    // statistics
    kDbgCutGroup = 32,
    kDbgCutBin = 33,
    kDbgOwnAbove = 34,
    kDbgLevels = 35,
    kDbgMode = 36,
    kDbgBoundary = 37,
    kDbgNeed = 38,
    kDbgOutOff = 39,
    kDbgBndOff = 40,
    kDbgSelCount = 41,
    kDbgBndCount = 42,
    kDbgLocalCut = 43
};

static_assert(kDbgLocalCut < kDebugSlots, "debug slots");

// Shared memory (32-bit words). Peers never read this CTA's shared memory: the cluster's exchanges are pushes
// (st.async) that complete on the receiver's barriers.
constexpr int32_t kSmemHist = 0;                                    // [kBins] local histogram of the current level
constexpr int32_t kSmemSuff = kSmemHist + kBins;                    // [kBins + 4] its suffix sums
constexpr int32_t kSmemKeys = kSmemSuff + kBins + 4;                // [kVecs][kThreads] uint4 keys
constexpr int32_t kSmemStageArea = kSmemKeys + kKeys * kThreads;    // [kStageWords] buckets or stage (see below)
constexpr int32_t kStageWords = kKeys * kThreads / 2;
constexpr int32_t kSmemRHist = kSmemStageArea + kStageWords;        // [kBins] bucket fill, then the resolve histogram
constexpr int32_t kSmemSlots = kSmemRHist + kBins;                  // [kCtas][kBins / kCtas] peers' counts of our bins
constexpr int32_t kSmemCoarse = kSmemSlots + kBins;                 // [kMaxRanks][kMaxGroups]
constexpr int32_t kSmemFine = kSmemCoarse + kMaxRanks * kMaxGroups; // [kMaxRanks][kGroupBins]
constexpr int32_t kSmemGsum = kSmemFine + kMaxRanks * kGroupBins;   // [2][kMaxGroups] group totals (all, lower ranks)
constexpr int32_t kPreRow = kGroupBins + 4; // the group's kGroupBins + 1 suffix sums, padded to 16-byte pushes
constexpr int32_t kSmemPre = kSmemGsum + 2 * kMaxGroups;               // [kLevels][kCtas][kPreRow] CTAs' suffix sums
constexpr int32_t kSmemScratch = kSmemPre + kLevels * kCtas * kPreRow; // [256] scan scratch
constexpr int32_t kSmemCut = kSmemScratch + 256;                       // [32] the cut of the current level
constexpr int32_t kSmemOffs = kSmemCut + 32;                           // [8] this CTA's offsets (warp 0)
constexpr int32_t kSmemMisc = kSmemOffs + 8;                           // [16]
constexpr int32_t kSmemSmall = kSmemMisc + 16;                         // [2 * kSmallRank]
constexpr int32_t kSmemBar = kSmemSmall + 2 * kSmallRank;              // [2 * kLevels] uint64 barriers
constexpr int32_t kSmemDbg = kSmemBar + 4 * kLevels;                   // [kDebugSlots] int64 (debug only)
constexpr int32_t kSmemWords = kSmemDbg + 2 * kDebugSlots;
constexpr size_t kSmemBytes = size_t{kSmemWords} * sizeof(uint32_t);
// The stage area holds, for a cut on the first digit, the level-0 buckets: 16-bit key indices of this CTA (the key at
// index i is keys[i / 4] component i % 4, its token base + cbase + i); for a later cut, the selected tokens.
constexpr int32_t kSmemBucket = kSmemStageArea; // [kKeys * kThreads] uint16
constexpr int32_t kSmemStage = kSmemStageArea;  // [kTopK] selected tokens
static_assert(kTopK <= kStageWords && kKeys * kThreads <= 65536, "16-bit bucket indices, the stage in the buckets");
static_assert(kSmemKeys % 4 == 0 && kSmemStage % 4 == 0 && kSmemSlots % 4 == 0 && kSmemFine % 4 == 0
        && kSmemPre % 4 == 0 && kSmemScratch % 4 == 0 && kSmemBar % 2 == 0 && kSmemDbg % 2 == 0,
    "aligned arrays");
// Below the 132 KB carveout: L1 keeps the register spills of 1024 threads.
static_assert(kSmemBytes <= 131 * 1024, "shared memory");

// kSmemCut slots.
enum CutSlot : int32_t
{
    kCutBin = 0,
    kCutAboveAll = 1,      // all ranks' keys above the cut bin (at this level, within the active prefix)
    kCutBinAll = 2,        // all ranks' keys in the cut bin
    kCutAboveLow = 3,      // the lower ranks' keys above the cut bin
    kCutGroup = 4,         // the group of the cut bin
    kCutGroupAbove = 5,    // all ranks' keys in the groups above it
    kCutGroupAboveLow = 6, // the lower ranks' keys in the groups above it
    kCutBndRankOff = 7,    // the lower ranks' keys in the cut bin
    kCutTiesOff = 8,       // the exact key's ties the lower ranks take (when the cut is the exact key)
    kCutTakeRank = 9,      // the ties this rank takes
    kCutRank = 16          // [kMaxRanks] every rank's count in the cut bin
};

// kSmemOffs slots.
enum OffsSlot : int32_t
{
    kOffsOut = 0,     // this CTA's first word in the output stage
    kOffsBnd = 1,     // this CTA's first pair in the boundary stage
    kOffsTake = 2,    // the exact-key ties this CTA takes (kModeTies)
    kOffsBndCta = 3,  // this CTA's keys in the boundary bin
    kOffsAboveCta = 4 // this CTA's keys above the boundary bin
};

enum MiscSlot : int32_t
{
    kMiscLevelBin = 0,  // [kLevels] the cut bin of each level
    kMiscSmallCount = 4,
    kMiscBucketBin = 5, // the lowest bin of the level-0 buckets
    kMiscRec = 6,       // [3] the previous exchange launch's record of the row (levels, output words, boundary pairs)
    kMiscLevelGroup = 9 // [kLevels] the group of each level's cut bin
};

__device__ __forceinline__ int32_t levelShift(int32_t level)
{
    return level == 0 ? 21 : (level == 1 ? 10 : 0);
}

__device__ __forceinline__ int32_t levelBits(int32_t level)
{
    return level == 2 ? 10 : 11;
}

__device__ __forceinline__ uint32_t orderedKey(float v)
{
    uint32_t const b = __float_as_uint(v);
    uint32_t const k = (b & 0x80000000u) ? ~b : (b | 0x80000000u);
    // NaN keys above +inf are clamped so that no key equals the empty word.
    return k < kEmpty ? k : kEmpty - 1u;
}

__device__ __forceinline__ uint32_t nibbleCount(uint32_t mask, int32_t k)
{
    return static_cast<uint32_t>(__popc((mask >> (4 * k)) & 0xFu));
}

__device__ __forceinline__ uint4 ldVolatile4(uint32_t const* p)
{
    uint4 v;
    asm volatile("ld.volatile.global.v4.u32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p)
                 : "memory");
    return v;
}

__device__ __forceinline__ void st1(uint32_t* p, uint32_t v)
{
    asm volatile("st.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}

__device__ __forceinline__ void st4(uint32_t* p, uint4 v)
{
    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w)
                 : "memory");
}

__device__ __forceinline__ uint64_t globalTimer()
{
    uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// The debug slots of this CTA live in shared memory and are copied to HistTopKParams::debug at the exit: a stamp adds
// no global store (a later release would wait for it).
__device__ __forceinline__ int64_t* debugSlots()
{
    extern __shared__ __align__(16) uint32_t dynSmem[];
    return reinterpret_cast<int64_t*>(dynSmem + kSmemDbg);
}

__device__ __forceinline__ void stamp(bool dbg, int32_t slot)
{
    if (dbg && threadIdx.x == 0)
    {
        debugSlots()[slot] = static_cast<int64_t>(globalTimer());
    }
}

__device__ __forceinline__ void putDebug(bool dbg, int32_t slot, int64_t v)
{
    if (dbg && threadIdx.x == 0)
    {
        debugSlots()[slot] = v;
    }
}

// Thread 0: this CTA's debug slots to HistTopKParams::debug.
__device__ __forceinline__ void flushDebug(int64_t* debug, int32_t row, int32_t c)
{
    if (debug != nullptr && threadIdx.x == 0)
    {
        int64_t* const g = debug + (int64_t{row} * kCtas + c) * kDebugSlots;
        for (int32_t i = 0; i < kDebugSlots; ++i)
        {
            g[i] = debugSlots()[i];
        }
    }
}

__device__ __forceinline__ void clusterArrive()
{
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
}

// An arrive without release semantics: the arriving thread does not wait for its own stores to be performed.
__device__ __forceinline__ void clusterArriveRelaxed()
{
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}

__device__ __forceinline__ void clusterWait()
{
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ void clusterSync()
{
    __syncwarp();
    clusterArrive();
    clusterWait();
}

__device__ __forceinline__ uint32_t smemAddr(void const* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// The shared::cluster address of the same shared variable in CTA `rank` of the cluster.
__device__ __forceinline__ uint32_t peerAddr(uint32_t addr, int32_t rank)
{
    uint32_t r;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(addr), "r"(rank));
    return r;
}

__device__ __forceinline__ void barInit(uint32_t bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

// The phase's one arrival, expecting `bytes` of asynchronous stores before it completes.
__device__ __forceinline__ void barArriveExpect(uint32_t bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}

// Spins on the phase with test_wait (try_wait may suspend the thread for longer than the phase takes to complete).
__device__ __forceinline__ void barWait(uint32_t bar, uint32_t parity)
{
    asm volatile(
        "{\n"
        ".reg .pred done;\n"
        "BAR_WAIT:\n"
        "mbarrier.test_wait.parity.shared::cta.b64 done, [%0], %1;\n"
        "@!done bra BAR_WAIT;\n"
        "}\n" ::"r"(bar),
        "r"(parity)
        : "memory");
}

// Orders the barrier initializations before the cluster arrive that follows (peers then push to them).
__device__ __forceinline__ void fenceBarInit()
{
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
}

// 16 bytes into a peer's shared memory, completing on the peer's barrier.
__device__ __forceinline__ void pushV4(uint32_t remote, uint4 v, uint32_t remoteBar)
{
    asm volatile(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.u32 [%0], {%1, %2, %3, %4}, [%5];" ::"r"(remote),
        "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w), "r"(remoteBar)
        : "memory");
}

__device__ __forceinline__ void st2(uint32_t* p, uint32_t a, uint32_t b)
{
    asm volatile("st.global.v2.u32 [%0], {%1, %2};" ::"l"(p), "r"(a), "r"(b) : "memory");
}

__device__ __forceinline__ int64_t histOff(int32_t buf, int32_t level, int32_t row, int32_t rank)
{
    return int64_t{buf} * kBufferWords + (int64_t{level} * kMaxRows + row) * kMaxRanks * kHistWords
        + int64_t{rank} * kHistWords;
}

__device__ __forceinline__ int64_t outOff(int32_t buf, int32_t row)
{
    return int64_t{buf} * kBufferWords + kLevels * kHistRegionWords + int64_t{row} * kTopK;
}

__device__ __forceinline__ int64_t bndOff(int32_t buf, int32_t row)
{
    return int64_t{buf} * kBufferWords + kLevels * kHistRegionWords + kOutRegionWords + int64_t{row} * kBoundaryCap * 2;
}

struct WaitCtx
{
    int32_t* state;
    uint64_t timeoutNs;
    int32_t rank;
};

// what: 1..3 the histograms of a level, 10 the output stage, 11 the boundary stage, 50 the previous launch's words
// before clearing; 200 + level / 300 + level: the ranks' histograms disagree; 400: the shard exceeds the logits.
__device__ __noinline__ void abortKernel(WaitCtx const& w, int32_t what)
{
    if (atomicExch(&w.state[kStateError], what) == 0)
    {
        printf("[glm_dsa_hist_topk] rank %d, row %d: %s (%d); aborting the kernel\n", w.rank,
            static_cast<int32_t>(blockIdx.y), what < 200 ? "a wait for a peer exceeded the timeout" : "bad input",
            what);
    }
    __threadfence_system();
    __trap();
}

struct PollClock
{
    uint64_t start = 0;
    int32_t iter = 0;

    __device__ __forceinline__ void tick(WaitCtx const& w, int32_t what)
    {
        if ((++iter & 255) == 0)
        {
            uint64_t const now = globalTimer();
            if (start == 0)
            {
                start = now;
            }
            else if (now - start > w.timeoutNs)
            {
                abortKernel(w, what);
            }
        }
    }
};

__device__ __forceinline__ bool anyEmpty(uint4 v, int32_t valid)
{
    return (valid > 0 && v.x == kEmpty) || (valid > 1 && v.y == kEmpty) || (valid > 2 && v.z == kEmpty)
        || (valid > 3 && v.w == kEmpty);
}

// The first `valid` words at p once a peer wrote them.
__device__ __forceinline__ uint4 pollVec(uint32_t const* p, int32_t valid, WaitCtx const& wc, int32_t what)
{
    uint4 v = ldVolatile4(p);
    PollClock clk;
    while (anyEmpty(v, valid))
    {
        clk.tick(wc, what);
        v = ldVolatile4(p);
    }
    return v;
}

// Exclusive prefixes of N words in thread order and their block totals. One barrier; ws[0, 32 N) must not be
// rewritten before the next barrier.
template <int32_t N>
__device__ __forceinline__ void blockExclusiveScanN(
    uint32_t const (&v)[N], uint32_t* ws, uint32_t (&pre)[N], uint32_t (&tot)[N])
{
    int32_t const lane = threadIdx.x & 31;
    int32_t const warp = threadIdx.x >> 5;
    uint32_t x[N];
#pragma unroll
    for (int32_t n = 0; n < N; ++n)
    {
        x[n] = v[n];
    }
#pragma unroll
    for (int32_t o = 1; o < 32; o <<= 1)
    {
#pragma unroll
        for (int32_t n = 0; n < N; ++n)
        {
            uint32_t const y = __shfl_up_sync(kFull, x[n], o);
            if (lane >= o)
            {
                x[n] += y;
            }
        }
    }
    if (lane == 31)
    {
#pragma unroll
        for (int32_t n = 0; n < N; ++n)
        {
            ws[32 * n + warp] = x[n];
        }
    }
    __syncthreads();
    uint32_t w[N];
    uint32_t y[N];
#pragma unroll
    for (int32_t n = 0; n < N; ++n)
    {
        w[n] = ws[32 * n + lane];
        y[n] = w[n];
    }
#pragma unroll
    for (int32_t o = 1; o < 32; o <<= 1)
    {
#pragma unroll
        for (int32_t n = 0; n < N; ++n)
        {
            uint32_t const z = __shfl_up_sync(kFull, y[n], o);
            if (lane >= o)
            {
                y[n] += z;
            }
        }
    }
#pragma unroll
    for (int32_t n = 0; n < N; ++n)
    {
        tot[n] = __shfl_sync(kFull, y[n], 31);
        pre[n] = __shfl_sync(kFull, y[n] - w[n], warp) + x[n] - v[n];
    }
}

// Exclusive suffix of v in thread order (the sum over the higher threads) and the block total. One barrier; ws[0, 32)
// as above.
__device__ __forceinline__ uint32_t blockExclusiveSuffix(uint32_t v, uint32_t* ws, uint32_t& total)
{
    int32_t const lane = threadIdx.x & 31;
    int32_t const warp = threadIdx.x >> 5;
    uint32_t x = v;
#pragma unroll
    for (int32_t o = 1; o < 32; o <<= 1)
    {
        uint32_t const y = __shfl_down_sync(kFull, x, o);
        if (lane + o < 32)
        {
            x += y;
        }
    }
    if (lane == 0)
    {
        ws[warp] = x;
    }
    __syncthreads();
    uint32_t const w = ws[lane];
    uint32_t y = w;
#pragma unroll
    for (int32_t o = 1; o < 32; o <<= 1)
    {
        uint32_t const z = __shfl_down_sync(kFull, y, o);
        if (lane + o < 32)
        {
            y += z;
        }
    }
    total = __shfl_sync(kFull, y, 0);
    return __shfl_sync(kFull, y - w, warp) + x - v;
}

__device__ __forceinline__ uint32_t blockExclusiveSuffix(uint32_t v, uint32_t* ws)
{
    uint32_t total = 0;
    return blockExclusiveSuffix(v, ws, total);
}

// Position bases, in position order (float4 k major, then thread, then lane j: increasing column), of the keys
// flagged in mask a and in mask b (bit 4 k + j): the position of this thread's first flagged key of float4 k is
// ba[k] (bb[k]), the next ones follow. Also the block counts of both. One block scan (16-bit fields: a CTA flags at
// most 4 keys per thread and float4).
__device__ __forceinline__ void positionBases(
    uint32_t ma, uint32_t mb, uint32_t* ws, uint32_t (&ba)[kVecs], uint32_t (&bb)[kVecs], uint32_t& na, uint32_t& nb)
{
    static_assert(kVecs == 3, "three float4 counts per mask");
    uint32_t const v[3] = {nibbleCount(ma, 0) | (nibbleCount(ma, 1) << 16),
        nibbleCount(ma, 2) | (nibbleCount(mb, 0) << 16), nibbleCount(mb, 1) | (nibbleCount(mb, 2) << 16)};
    uint32_t pre[3];
    uint32_t tot[3];
    blockExclusiveScanN<3>(v, ws, pre, tot);
    uint32_t const ta0 = tot[0] & 0xFFFFu;
    uint32_t const ta1 = tot[0] >> 16;
    uint32_t const ta2 = tot[1] & 0xFFFFu;
    uint32_t const tb0 = tot[1] >> 16;
    uint32_t const tb1 = tot[2] & 0xFFFFu;
    uint32_t const tb2 = tot[2] >> 16;
    ba[0] = pre[0] & 0xFFFFu;
    ba[1] = ta0 + (pre[0] >> 16);
    ba[2] = ta0 + ta1 + (pre[1] & 0xFFFFu);
    bb[0] = pre[1] >> 16;
    bb[1] = tb0 + (pre[2] & 0xFFFFu);
    bb[2] = tb0 + tb1 + (pre[2] >> 16);
    na = ta0 + ta1 + ta2;
    nb = tb0 + tb1 + tb2;
}

// The bin of a local histogram (nbins <= 2 * kThreads, two adjacent bins per thread) holding the need-th largest
// item counting from the top bin: (bin, count above, bin count) into cut[0, 3). Two barriers.
__device__ __forceinline__ void findLocalCut(
    uint32_t const* h, int32_t nbins, uint32_t need, uint32_t* ws, uint32_t* cut)
{
    int32_t const t = threadIdx.x;
    bool const has = 2 * t < nbins;
    uint32_t const c0 = has ? h[2 * t] : 0u;
    uint32_t const c1 = has ? h[2 * t + 1] : 0u;
    uint32_t const above1 = blockExclusiveSuffix(c0 + c1, ws);
    uint32_t const above0 = above1 + c1;
    if (has)
    {
        if (above1 < need && need <= above1 + c1)
        {
            cut[0] = 2 * t + 1;
            cut[1] = above1;
            cut[2] = c1;
        }
        else if (above0 < need && need <= above0 + c0)
        {
            cut[0] = 2 * t;
            cut[1] = above0;
            cut[2] = c0;
        }
    }
    __syncthreads();
}

// Every CTA of the row owns nbins / kCtas bins of the level's histogram. This CTA pushes its local counts of each
// peer's bins into that peer's slots (st.async, completing on the peer's barrier `bar`), waits for the peers' counts of
// its own bins, and stores the rank's counts of its bins through the multicast address: two per word, saturated at
// kSat, and its 32-bin group totals. The local histogram must be complete (a CTA barrier before). red: >= 16 + 128
// words of 16-byte aligned scratch. One barrier; the multicast stores are issued by the last two warps.
__device__ __forceinline__ void pushAndSend(
    uint32_t const* hist, int32_t nbins, uint32_t* slots, uint32_t bar, uint32_t* red, uint32_t* mc)
{
    int32_t const t = threadIdx.x;
    int32_t const perCta = nbins / kCtas;             // 256 or 128 bins: whole warps, whole groups
    int32_t const groupsPerCta = perCta / kGroupBins; // 8 or 4
    int32_t const c = static_cast<int32_t>(blockIdx.x);
    int32_t const vecs = perCta / 4;
    for (int32_t i = t; i < kCtas * vecs; i += kThreads)
    {
        int32_t const q = i / vecs;
        if (q != c)
        {
            int32_t const k = i - q * vecs;
            pushV4(peerAddr(smemAddr(slots + c * perCta + 4 * k), q),
                *reinterpret_cast<uint4 const*>(hist + q * perCta + 4 * k), peerAddr(bar, q));
        }
    }
    if (t < perCta)
    {
        barWait(bar, 0);
        uint32_t s = hist[c * perCta + t];
#pragma unroll
        for (int32_t q = 0; q < kCtas; ++q)
        {
            s += q != c ? slots[q * perCta + t] : 0u;
        }
        uint32_t const group = __reduce_add_sync(kFull, s);
        if ((t & 31) == 0)
        {
            red[t / 32] = group;
        }
        // Lane 8 m packs the bins 8 m .. 8 m + 7 of the warp: four words of (even, odd) bins.
        uint32_t const w = min(s, kSat) | (min(__shfl_down_sync(kFull, s, 1), kSat) << 16);
        uint32_t const w2 = __shfl_down_sync(kFull, w, 2);
        uint32_t const w4 = __shfl_down_sync(kFull, w, 4);
        uint32_t const w6 = __shfl_down_sync(kFull, w, 6);
        if ((t & 7) == 0)
        {
            *reinterpret_cast<uint4*>(red + 16 + t / 2) = make_uint4(w, w2, w4, w6);
        }
    }
    __syncthreads();
    int32_t const u = t - (kThreads - 64);
    if (u >= 0 && u < perCta / 8)
    {
        st4(mc + (c * perCta) / 2 + 4 * u, *reinterpret_cast<uint4 const*>(red + 16 + 4 * u));
    }
    else if (u >= 32 && 4 * (u - 32) < groupsPerCta)
    {
        st4(mc + kFineWords + c * groupsPerCta + 4 * (u - 32), *reinterpret_cast<uint4 const*>(red + 4 * (u - 32)));
    }
}

// Suffix sums of a local histogram: suff[b] = keys in bins >= b, suff[nbins] = 0 (read by the cluster's CTAs). With
// bucketBin: the highest bin b with suff[b] >= min(kTopK, suff[0]) into *bucketBin (nbins without keys). The cut of
// all ranks cannot be above it: this CTA has fewer than kTopK keys above that cut.
__device__ __forceinline__ void localSuffix(
    uint32_t const* hist, int32_t nbins, uint32_t* suff, uint32_t* ws, uint32_t* bucketBin)
{
    int32_t const t = threadIdx.x;
    bool const has = 2 * t < nbins;
    uint32_t const c0 = has ? hist[2 * t] : 0u;
    uint32_t const c1 = has ? hist[2 * t + 1] : 0u;
    uint32_t total = 0;
    uint32_t const above = blockExclusiveSuffix(c0 + c1, ws, total);
    if (has)
    {
        suff[2 * t + 1] = above + c1;
        suff[2 * t] = above + c1 + c0;
        uint32_t const thr = min(static_cast<uint32_t>(kTopK), total);
        if (bucketBin != nullptr && thr > 0u && above < thr)
        {
            if (above + c1 >= thr)
            {
                *bucketBin = static_cast<uint32_t>(2 * t + 1);
            }
            else if (above + c1 + c0 >= thr)
            {
                *bucketBin = static_cast<uint32_t>(2 * t);
            }
        }
    }
    if (t == 0)
    {
        suff[nbins] = 0;
        if (bucketBin != nullptr && total == 0u)
        {
            *bucketBin = static_cast<uint32_t>(nbins);
        }
    }
    __syncthreads();
}

// Clear, on this rank's copy, what the previous exchange launch wrote into buffer `buf` for this row. That launch read
// only the group totals and one group of each histogram, so every histogram word is first read written (no peer
// store of that launch can land after the clear), then cleared; the output and boundary stages were read whole.
__device__ __forceinline__ void clearPrevious(HistTopKParams const& p, int32_t buf, int32_t row, int32_t levels,
    int32_t outWords, int32_t bndPairs, WaitCtx const& wc, int32_t ctaIndex, int32_t ctaCount)
{
    int32_t const c = ctaIndex;
    int32_t const t = threadIdx.x;
    uint4 const empty = make_uint4(kEmpty, kEmpty, kEmpty, kEmpty);
    for (int32_t level = 0; level < levels; ++level)
    {
        // Written per rank at this level: nbins / 2 bin-count words and nbins / 32 group totals.
        int32_t const nbins = 1 << levelBits(level);
        int32_t const fineVecs = nbins / 8;
        int32_t const rankVecs = fineVecs + nbins / (4 * kGroupBins);
        uint32_t* base = p.ucWorkspace + histOff(buf, level, row, 0);
        for (int32_t i = c * kThreads + t; i < p.worldSize * rankVecs; i += ctaCount * kThreads)
        {
            int32_t const r = i / rankVecs;
            int32_t const j = i % rankVecs;
            uint32_t* w = base + int64_t{r} * kHistWords + (j < fineVecs ? 4 * j : kFineWords + 4 * (j - fineVecs));
            (void) pollVec(w, 4, wc, 50);
            st4(w, empty);
        }
    }
    uint32_t* o = p.ucWorkspace + outOff(buf, row);
    for (int32_t i = c * kThreads + t; i < outWords; i += ctaCount * kThreads)
    {
        st1(o + i, kEmpty);
    }
    uint32_t* b = p.ucWorkspace + bndOff(buf, row);
    for (int32_t i = c * kThreads + t; i < 2 * bndPairs; i += ctaCount * kThreads)
    {
        st1(b + i, kEmpty);
    }
}

// The lowest set bit of a non-zero mask.
__device__ __forceinline__ int32_t lowBit(uint32_t m)
{
    return static_cast<int32_t>(__ffs(static_cast<int32_t>(m))) - 1;
}

// Warp 0, after the final cut: every CTA's keys above it and in its bin (lane q < kCtas: CTA q's suffix sums pushed
// into pre at each level, waited for on preBar0 + 8 lv), then this CTA's offsets in the output and boundary stages
// (the rank's table from receiveCut, then the lower CTAs). Into offs[] (read after the next barrier).
__device__ __forceinline__ void cutOffsets(uint32_t const* misc, uint32_t const* pre, uint32_t preBar0,
    uint32_t const* cut, int32_t level, int32_t mode, uint32_t lowTot, int32_t c, uint32_t* offs, bool dbg)
{
    int32_t const lane = static_cast<int32_t>(threadIdx.x & 31);
    for (int32_t lv = 0; lv <= level; ++lv)
    {
        barWait(preBar0 + 8 * lv, 0);
    }
    stamp(dbg, kDbgPreWaited);
    uint32_t aboveQ = 0;
    uint32_t bndQ = 0;
    if (lane < kCtas)
    {
        for (int32_t lv = 0; lv < level; ++lv)
        {
            uint32_t const j = misc[kMiscLevelBin + lv] - misc[kMiscLevelGroup + lv] * kGroupBins;
            aboveQ += pre[(lv * kCtas + lane) * kPreRow + j + 1];
        }
        uint32_t const* const pq = pre + (level * kCtas + lane) * kPreRow;
        uint32_t const j = cut[kCutBin] - cut[kCutGroup] * kGroupBins;
        aboveQ += pq[j + 1];
        bndQ = pq[j] - pq[j + 1];
    }
    uint32_t rankOff = lowTot;
    uint32_t bndRankOff = 0;
    uint32_t takeQ = 0;
    if (mode == kModeTies)
    {
        // The rank's ties go to the higher CTAs first (the larger tokens).
        rankOff += cut[kCutTiesOff];
        int32_t const takeRank = static_cast<int32_t>(cut[kCutTakeRank]);
        uint32_t xq = bndQ;
#pragma unroll
        for (int32_t o = 1; o < kCtas; o <<= 1)
        {
            uint32_t const y = __shfl_down_sync(kFull, xq, o);
            if (lane + o < kCtas)
            {
                xq += y;
            }
        }
        int32_t const higherQ = static_cast<int32_t>(xq - bndQ);
        takeQ = static_cast<uint32_t>(max(0, min(takeRank - higherQ, static_cast<int32_t>(bndQ))));
    }
    else
    {
        bndRankOff = cut[kCutBndRankOff];
        if (mode == kModeFull)
        {
            rankOff += bndRankOff;
            takeQ = bndQ;
        }
    }
    uint32_t const outOffC = rankOff + __reduce_add_sync(kFull, lane < c ? aboveQ + takeQ : 0u);
    uint32_t const bndOffC = bndRankOff + __reduce_add_sync(kFull, lane < c ? bndQ : 0u);
    uint32_t const takeC = __shfl_sync(kFull, takeQ, c);
    uint32_t const bndC = __shfl_sync(kFull, bndQ, c);
    uint32_t const aboveC = __shfl_sync(kFull, aboveQ, c);
    if (lane == 0)
    {
        offs[kOffsOut] = outOffC;
        offs[kOffsBnd] = bndOffC;
        offs[kOffsTake] = takeC;
        offs[kOffsBndCta] = bndC;
        offs[kOffsAboveCta] = aboveC;
    }
}

// Store n words from shared memory through the multicast address: 16-byte stores where dst is aligned.
__device__ __forceinline__ void storeWords(uint32_t* dst, uint32_t const* src, int32_t n)
{
    int32_t const t = threadIdx.x;
    int32_t const head
        = min(n, static_cast<int32_t>((4u - static_cast<uint32_t>((reinterpret_cast<uintptr_t>(dst) >> 2) & 3u)) & 3u));
    int32_t const vecs = (n - head) / 4;
    int32_t const tail = n - head - 4 * vecs;
    if (t < head)
    {
        st1(dst + t, src[t]);
    }
    for (int32_t i = t; i < vecs; i += kThreads)
    {
        int32_t const j = head + 4 * i;
        st4(dst + j, make_uint4(src[j], src[j + 1], src[j + 2], src[j + 3]));
    }
    if (t >= 64 && t < 64 + tail)
    {
        int32_t const j = head + 4 * vecs + (t - 64);
        st1(dst + j, src[j]);
    }
}

// Every CTA of the row: the cut of one level from all ranks' histograms, read through the group totals (4 KB) and the
// crossing group's bins (1 KB), with every rank's count in the cut bin and the rank's offsets that follow from them.
// Warps 2..8 meanwhile push this CTA's suffix sums over the crossing group's bins and the bin above them into row c
// of every CTA's pre rows of this level (completing on their barriers preBar0 + 8 level); when this is the final cut,
// warp 0 then computes this CTA's offsets (cutOffsets). gsum must be zero; lowTotPrev: the lower ranks' keys above
// the cuts of the earlier levels. Results in cut[] and offs[] (every thread can read them after the last barrier).
__device__ void receiveCut(HistTopKParams const& p, uint32_t const* ucLevel, uint32_t const* suffLevel, int32_t nbins,
    int32_t level, uint32_t need, uint32_t lowTotPrev, uint32_t* coarse, uint32_t* fine, uint32_t* gsum,
    uint32_t const* misc, uint32_t* pre, uint32_t preBar0, uint32_t* cut, uint32_t* offs, WaitCtx const& wc,
    int32_t what, bool dbg)
{
    uint32_t* const preL = pre + level * kCtas * kPreRow;
    uint32_t const preBar = preBar0 + 8 * level;
    int32_t const t = threadIdx.x;
    int32_t const world = p.worldSize;
    int32_t const rank = p.rank;
    int32_t const ngroups = nbins / kGroupBins;  // 64 or 32
    int32_t const vecsPerRank = ngroups / 4;     // 16 or 8
    uint32_t* const gAllSum = gsum;              // [kMaxGroups] all ranks
    uint32_t* const gLowSum = gsum + kMaxGroups; // [kMaxGroups] the lower ranks
    if (t == 0)
    {
        cut[kCutGroup] = kEmpty;
        cut[kCutBin] = kEmpty;
    }
    // 1. The group totals of every rank, summed per group over all ranks and over the lower ranks.
    if (t < kMaxRanks * vecsPerRank)
    {
        int32_t const r = t / vecsPerRank;
        int32_t const vi = t % vecsPerRank;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (r < world)
        {
            v = pollVec(ucLevel + int64_t{r} * kHistWords + kFineWords + 4 * vi, 4, wc, what);
            uint32_t const vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int32_t j = 0; j < 4; ++j)
            {
                atomicAdd(&gAllSum[4 * vi + j], vv[j]);
                if (r < rank)
                {
                    atomicAdd(&gLowSum[4 * vi + j], vv[j]);
                }
            }
        }
        coarse[r * kMaxGroups + 4 * vi] = v.x;
        coarse[r * kMaxGroups + 4 * vi + 1] = v.y;
        coarse[r * kMaxGroups + 4 * vi + 2] = v.z;
        coarse[r * kMaxGroups + 4 * vi + 3] = v.w;
    }
    __syncthreads();
    stamp(dbg, kDbgCoarse);
    // 2. The crossing group (warp 0): lane l holds groups 2 l and 2 l + 1.
    if (t < 32)
    {
        uint32_t gAll[2];
        uint32_t gLow[2];
#pragma unroll
        for (int32_t k = 0; k < 2; ++k)
        {
            int32_t const g = 2 * t + k;
            gAll[k] = g < ngroups ? gAllSum[g] : 0u;
            gLow[k] = g < ngroups ? gLowSum[g] : 0u;
        }
        // Exclusive suffixes over the lanes of the per-lane totals.
        uint32_t xa = gAll[0] + gAll[1];
        uint32_t xl = gLow[0] + gLow[1];
#pragma unroll
        for (int32_t o = 1; o < 32; o <<= 1)
        {
            uint32_t const ya = __shfl_down_sync(kFull, xa, o);
            uint32_t const yl = __shfl_down_sync(kFull, xl, o);
            if (t + o < 32)
            {
                xa += ya;
                xl += yl;
            }
        }
        uint32_t const aAbove1 = xa - gAll[0] - gAll[1];
        uint32_t const lAbove1 = xl - gLow[0] - gLow[1];
        int32_t which = -1;
        if (aAbove1 < need && need <= aAbove1 + gAll[1])
        {
            which = 1;
        }
        else if (aAbove1 + gAll[1] < need && need <= aAbove1 + gAll[1] + gAll[0])
        {
            which = 0;
        }
        if (which >= 0)
        {
            cut[kCutGroup] = 2 * t + which;
            cut[kCutGroupAbove] = which == 1 ? aAbove1 : aAbove1 + gAll[1];
            cut[kCutGroupAboveLow] = which == 1 ? lAbove1 : lAbove1 + gLow[1];
        }
    }
    __syncthreads();
    int32_t const g = static_cast<int32_t>(cut[kCutGroup]);
    if (g < 0 || g >= ngroups)
    {
        // No group holds the need-th key: the ranks' inputs disagree, nothing exact can follow.
        abortKernel(wc, 200 + what);
    }
    uint32_t const needInGroup = need - cut[kCutGroupAbove];
    stamp(dbg, kDbgGroup);
    // Warps 2..8 (idle in steps 3 and 4): this CTA's suffix sums over the group's bins and the bin above them
    // (vectors 0..8 of four, the last three words padding) to every CTA of the row, one destination per warp: the 9
    // lanes' 16-byte pushes go to consecutive addresses of one peer, and the warps push at once.
    if (t >= 64 && t < 64 + 32 * (kCtas - 1))
    {
        int32_t const c = static_cast<int32_t>(blockIdx.x);
        int32_t const lane = t & 31;
        int32_t const qi = (t >> 5) - 2;
        int32_t const q = qi < c ? qi : qi + 1;
        if (lane < kPreRow / 4)
        {
            uint32_t const* const src = suffLevel + g * kGroupBins + 4 * lane;
            uint4 const val = make_uint4(src[0], src[1], src[2], src[3]);
            uint4* const slot = reinterpret_cast<uint4*>(preL + c * kPreRow) + lane;
            if (qi == 0)
            {
                *slot = val; // this CTA's own row
            }
            pushV4(peerAddr(smemAddr(slot), q), val, peerAddr(preBar, q));
        }
    }
    // 3. The group's 32 bins of every rank (threads 4 r .. 4 r + 3: 8 bins each), a saturated bin restored from the
    //    group total.
    if (t < kMaxRanks * 4)
    {
        int32_t const r = t / 4;
        int32_t const vi = t % 4;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (r < world)
        {
            v = pollVec(ucLevel + int64_t{r} * kHistWords + (kGroupBins / 2) * g + 4 * vi, 4, wc, what);
        }
        uint32_t const ww[4] = {v.x, v.y, v.z, v.w};
        uint32_t x[8];
        uint32_t others = 0;
        bool sat = false;
#pragma unroll
        for (int32_t j = 0; j < 4; ++j)
        {
            x[2 * j] = ww[j] & 0xFFFFu;
            x[2 * j + 1] = ww[j] >> 16;
        }
#pragma unroll
        for (int32_t j = 0; j < 8; ++j)
        {
            if (x[j] == kSat)
            {
                sat = true;
            }
            else
            {
                others += x[j];
            }
        }
        others += __shfl_xor_sync(kFull, others, 1);
        others += __shfl_xor_sync(kFull, others, 2);
        if (sat)
        {
            uint32_t const total = coarse[r * kMaxGroups + g];
#pragma unroll
            for (int32_t j = 0; j < 8; ++j)
            {
                if (x[j] == kSat)
                {
                    x[j] = total - others;
                }
            }
        }
        *reinterpret_cast<uint4*>(fine + r * kGroupBins + 8 * vi) = make_uint4(x[0], x[1], x[2], x[3]);
        *reinterpret_cast<uint4*>(fine + r * kGroupBins + 8 * vi + 4) = make_uint4(x[4], x[5], x[6], x[7]);
        // Warps 0 and 1 only: warps 2..8 may still push.
        asm volatile("bar.sync 1, 64;" ::: "memory");
    }
    stamp(dbg, kDbgFine);
    // 4. The cut bin (warp 0: lane = bin of the group), every rank's count in it and the rank's offsets.
    if (t < 32)
    {
        uint32_t all = 0;
        uint32_t low = 0;
#pragma unroll
        for (int32_t r = 0; r < kMaxRanks; ++r)
        {
            uint32_t const x = fine[r * kGroupBins + t];
            all += x;
            low += r < rank ? x : 0u;
        }
        // Exclusive suffix over the 32 bins of the group.
        uint32_t xa = all;
        uint32_t xl = low;
#pragma unroll
        for (int32_t o = 1; o < 32; o <<= 1)
        {
            uint32_t const ya = __shfl_down_sync(kFull, xa, o);
            uint32_t const yl = __shfl_down_sync(kFull, xl, o);
            if (t + o < 32)
            {
                xa += ya;
                xl += yl;
            }
        }
        uint32_t const aAbove = xa - all;
        uint32_t const lAbove = xl - low;
        uint32_t const holder = __ballot_sync(kFull, aAbove < needInGroup && needInGroup <= aAbove + all);
        if (holder != 0u)
        {
            int32_t const w = lowBit(holder);
            uint32_t const aAboveW = __shfl_sync(kFull, aAbove, w);
            uint32_t const allW = __shfl_sync(kFull, all, w);
            uint32_t const lAboveW = __shfl_sync(kFull, lAbove, w);
            uint32_t const bc = t < kMaxRanks ? fine[t * kGroupBins + w] : 0u;
            if (t < kMaxRanks)
            {
                cut[kCutRank + t] = bc;
            }
            uint32_t const bndRankOff = __reduce_add_sync(kFull, t < rank ? bc : 0u);
            // When the bin is the exact key: rank r takes what the higher ranks leave of the bin's need.
            uint32_t const needBin = needInGroup - aAboveW;
            uint32_t xs = bc;
#pragma unroll
            for (int32_t o = 1; o < kMaxRanks; o <<= 1)
            {
                uint32_t const y = __shfl_down_sync(kFull, xs, o);
                if (t + o < kMaxRanks)
                {
                    xs += y;
                }
            }
            int32_t const higher = static_cast<int32_t>(xs - bc);
            int32_t const take = max(0, min(static_cast<int32_t>(needBin) - higher, static_cast<int32_t>(bc)));
            uint32_t const tiesOff = __reduce_add_sync(kFull, t < rank ? static_cast<uint32_t>(take) : 0u);
            uint32_t const takeRank = __shfl_sync(kFull, static_cast<uint32_t>(take), rank);
            uint32_t const aboveAll = cut[kCutGroupAbove] + aAboveW;
            uint32_t const aboveLow = cut[kCutGroupAboveLow] + lAboveW;
            if (t == 0)
            {
                cut[kCutBin] = static_cast<uint32_t>(g * kGroupBins + w);
                cut[kCutAboveAll] = aboveAll;
                cut[kCutBinAll] = allW;
                cut[kCutAboveLow] = aboveLow;
                cut[kCutBndRankOff] = bndRankOff;
                cut[kCutTiesOff] = tiesOff;
                cut[kCutTakeRank] = takeRank;
            }
            __syncwarp();
            stamp(dbg, kDbgCut);
            // When this is the final cut (as the main loop decides it), this CTA's offsets at once.
            uint32_t const needAfter = need - aboveAll;
            int32_t const mode = needAfter == allW
                ? kModeFull
                : (level == kLevels - 1 ? kModeTies
                                        : (allW <= static_cast<uint32_t>(kBoundaryCap) ? kModeCand : kModeNext));
            if (mode != kModeNext)
            {
                cutOffsets(misc, pre, preBar0, cut, level, mode, lowTotPrev + aboveLow,
                    static_cast<int32_t>(blockIdx.x), offs, dbg);
            }
        }
    }
    __syncthreads();
    stamp(dbg, kDbgPeer);
}

// Store the tokens of n bucket entries (key indices) through the multicast address: 16-byte stores where dst is
// aligned.
__device__ __forceinline__ void storeBucketTokens(uint32_t* dst, uint16_t const* idx, uint32_t tokenBase, int32_t n)
{
    int32_t const t = threadIdx.x;
    int32_t const head
        = min(n, static_cast<int32_t>((4u - static_cast<uint32_t>((reinterpret_cast<uintptr_t>(dst) >> 2) & 3u)) & 3u));
    int32_t const vecs = (n - head) / 4;
    int32_t const tail = n - head - 4 * vecs;
    if (t < head)
    {
        st1(dst + t, tokenBase + idx[t]);
    }
    for (int32_t i = t; i < vecs; i += kThreads)
    {
        int32_t const j = head + 4 * i;
        st4(dst + j,
            make_uint4(tokenBase + idx[j], tokenBase + idx[j + 1], tokenBase + idx[j + 2], tokenBase + idx[j + 3]));
    }
    if (t >= 64 && t < 64 + tail)
    {
        int32_t const j = head + 4 * vecs + (t - 64);
        st1(dst + j, tokenBase + idx[j]);
    }
}

// Store n bucket entries as (key, token) pairs through the multicast address: 16-byte stores (two pairs) where dst
// is aligned (dst is 8-byte aligned). From the second half of the CTA, beside storeBucketTokens.
__device__ __forceinline__ void storeBucketPairs(
    uint32_t* dst, uint16_t const* idx, uint32_t const* keys, uint32_t tokenBase, int32_t n)
{
    int32_t const t = (static_cast<int32_t>(threadIdx.x) + kThreads / 2) % kThreads;
    int32_t const head = min(n, ((reinterpret_cast<uintptr_t>(dst) >> 3) & 1u) != 0u ? 1 : 0);
    int32_t const vecs = (n - head) / 2;
    if (t < head)
    {
        st1(dst, keys[idx[0]]);
        st1(dst + 1, tokenBase + idx[0]);
    }
    for (int32_t i = t; i < vecs; i += kThreads)
    {
        int32_t const j = head + 2 * i;
        uint32_t const a = idx[j];
        uint32_t const b = idx[j + 1];
        st4(dst + 2 * j, make_uint4(keys[a], tokenBase + a, keys[b], tokenBase + b));
    }
    if (t == 64 && head + 2 * vecs < n)
    {
        uint32_t const a = idx[n - 1];
        st1(dst + 2 * (n - 1), keys[a]);
        st1(dst + 2 * (n - 1) + 1, tokenBase + a);
    }
}

// The resolver: among all ranks' boundary pairs (thread t holds pairs 2 t, 2 t + 1, 2 kThreads + 2 t and
// 2 kThreads + 2 t + 1 in pk/pt, valid as flagged in pv; every key shares its bits from `shift` up with keyTop),
// select the `need` largest by (key, token) and write their tokens, in pair order, to outRow[base, base + need).
__device__ void resolveBoundary(uint32_t const (&pk)[4], uint32_t const (&pt)[4], uint32_t pv, uint32_t need,
    int32_t shift, uint32_t keyTop, uint32_t* rh, uint32_t* ws, uint32_t* cut, uint32_t* small, uint32_t* smallCount,
    int32_t* outRow, int32_t base, bool dbg)
{
    int32_t const t = threadIdx.x;
    // The next 11 key bits (rh and smallCount are zero: cleared at the kernel start).
    int32_t const w1 = shift > 11 ? 11 : shift;
    int32_t const sh1 = shift - w1;
    uint32_t const mask1 = (1u << w1) - 1u;
#pragma unroll
    for (int32_t i = 0; i < 4; ++i)
    {
        if ((pv >> i) & 1u)
        {
            atomicAdd(&rh[(pk[i] >> sh1) & mask1], 1u);
        }
    }
    __syncthreads();
    findLocalCut(rh, 1 << w1, need, ws, cut);
    uint32_t const d1 = cut[0];
    uint32_t const need1 = need - cut[1];
    uint32_t const cnt1 = cut[2];
    stamp(dbg, kDbgPass1);
    uint32_t tKey = 0;
    uint32_t tTok = 0;
    if (cnt1 <= static_cast<uint32_t>(kSmallRank))
    {
        // Few pairs in that sub-bin: rank them by (key, token) directly.
#pragma unroll
        for (int32_t i = 0; i < 4; ++i)
        {
            if (((pv >> i) & 1u) && ((pk[i] >> sh1) & mask1) == d1)
            {
                uint32_t const slot = atomicAdd(smallCount, 1u);
                small[slot] = pk[i];
                small[kSmallRank + slot] = pt[i];
            }
        }
        __syncthreads();
        if (t < static_cast<int32_t>(cnt1))
        {
            uint32_t const k = small[t];
            uint32_t const tok = small[kSmallRank + t];
            uint32_t higher = 0;
            for (uint32_t j = 0; j < cnt1; ++j)
            {
                uint32_t const kj = small[j];
                uint32_t const tj = small[kSmallRank + j];
                higher += (kj > k || (kj == k && tj > tok)) ? 1u : 0u;
            }
            if (higher == need1 - 1)
            {
                cut[0] = k;
                cut[1] = tok;
            }
        }
        __syncthreads();
        tKey = cut[0];
        tTok = cut[1];
    }
    else
    {
        // The remaining key bits, then the token bits of the threshold key's ties (the larger tokens first).
        uint32_t needR = need1;
        uint32_t prefix = d1;
        uint32_t tie = cnt1;
        if (sh1 > 0)
        {
            __syncthreads();
            for (int32_t i = t; i < (1 << sh1); i += kThreads)
            {
                rh[i] = 0;
            }
            __syncthreads();
            uint32_t const mask2 = (1u << sh1) - 1u;
#pragma unroll
            for (int32_t i = 0; i < 4; ++i)
            {
                if (((pv >> i) & 1u) && ((pk[i] >> sh1) & mask1) == d1)
                {
                    atomicAdd(&rh[pk[i] & mask2], 1u);
                }
            }
            __syncthreads();
            findLocalCut(rh, 1 << sh1, needR, ws, cut);
            prefix = (d1 << sh1) | cut[0];
            needR -= cut[1];
            tie = cut[2];
            __syncthreads();
        }
        uint32_t const kLow = prefix;
        uint32_t const lowMask = (1u << shift) - 1u;
        uint32_t tokPrefix = 0;
        if (needR != tie)
        {
            int32_t rem = 21;
            while (rem > 0)
            {
                int32_t const width = rem > 11 ? 11 : rem;
                int32_t const sh = rem - width;
                uint32_t const matchMask = (1u << (21 - rem)) - 1u;
                for (int32_t i = t; i < (1 << width); i += kThreads)
                {
                    rh[i] = 0;
                }
                __syncthreads();
#pragma unroll
                for (int32_t i = 0; i < 4; ++i)
                {
                    if (((pv >> i) & 1u) && (pk[i] & lowMask) == kLow && ((pt[i] >> rem) & matchMask) == tokPrefix)
                    {
                        atomicAdd(&rh[(pt[i] >> sh) & ((1u << width) - 1u)], 1u);
                    }
                }
                __syncthreads();
                findLocalCut(rh, 1 << width, needR, ws, cut);
                tokPrefix = (tokPrefix << width) | cut[0];
                needR -= cut[1];
                rem = sh;
                __syncthreads();
            }
        }
        tKey = keyTop | kLow;
        tTok = tokPrefix;
    }
    stamp(dbg, kDbgThreshold);
    // The pairs at or above (tKey, tTok), in pair order.
    uint32_t win = 0;
#pragma unroll
    for (int32_t i = 0; i < 4; ++i)
    {
        if (((pv >> i) & 1u) && (pk[i] > tKey || (pk[i] == tKey && pt[i] >= tTok)))
        {
            win |= 1u << i;
        }
    }
    uint32_t const v[1] = {static_cast<uint32_t>(__popc(win & 3u)) | (static_cast<uint32_t>(__popc(win >> 2)) << 16)};
    uint32_t pre[1];
    uint32_t tot[1];
    blockExclusiveScanN<1>(v, ws, pre, tot);
    uint32_t p0 = pre[0] & 0xFFFFu;
    uint32_t p1 = (tot[0] & 0xFFFFu) + (pre[0] >> 16);
#pragma unroll
    for (int32_t i = 0; i < 4; ++i)
    {
        if ((win >> i) & 1u)
        {
            uint32_t& pos = i < 2 ? p0 : p1;
            outRow[base + static_cast<int32_t>(pos)] = static_cast<int32_t>(pt[i]);
            ++pos;
        }
    }
}

__global__ void __launch_bounds__(kThreads, 1) dsaHistTopKKernel(HistTopKParams p)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    extern __shared__ __align__(16) uint32_t smem[];
    uint32_t* const hist = smem + kSmemHist;
    uint32_t* const suff = smem + kSmemSuff;
    uint4* const keys = reinterpret_cast<uint4*>(smem + kSmemKeys);
    uint32_t* const stageTok = smem + kSmemStage;
    uint16_t* const bucket = reinterpret_cast<uint16_t*>(smem + kSmemBucket);
    uint32_t* const rhist = smem + kSmemRHist;
    uint32_t* const cursor = rhist; // the level-0 bucket fill; its bins are cleared again for the resolver
    uint32_t* const slots = smem + kSmemSlots;
    uint32_t* const coarse = smem + kSmemCoarse;
    uint32_t* const fine = smem + kSmemFine;
    uint32_t* const gsum = smem + kSmemGsum;
    uint32_t* const pre = smem + kSmemPre;
    uint32_t* const ws = smem + kSmemScratch;
    uint32_t* const cut = smem + kSmemCut;
    uint32_t* const offs = smem + kSmemOffs;
    uint32_t* const misc = smem + kSmemMisc;
    uint32_t* const small = smem + kSmemSmall;
    // Barriers: [kLevels] the peers' histogram counts, then [kLevels] the peers' suffix sums at the cut.
    uint32_t const histBar0 = smemAddr(smem + kSmemBar);
    uint32_t const preBar0 = histBar0 + 8 * kLevels;

    int32_t const c = static_cast<int32_t>(blockIdx.x);
    int32_t const row = static_cast<int32_t>(blockIdx.y);
    int32_t const t = threadIdx.x;
    bool const dbg = p.debug != nullptr;
    if (dbg && t == 0)
    {
        for (int32_t i = 0; i < kDebugSlots; ++i)
        {
            debugSlots()[i] = 0;
        }
    }
    stamp(dbg, kDbgEntry);
    putDebug(dbg, kDbgClockEntry, static_cast<int64_t>(clock64()));

    // The request's KV length is written before the logits of this call (the forward's metadata).
    int32_t const kvLen = p.kvLens[0];
    int32_t const world = p.worldSize;
    int32_t const rank = p.rank;
    int32_t const perRank = (max(kvLen, 0) + world - 1) / world;
    int32_t const S = (perRank + kShardAlign - 1) / kShardAlign * kShardAlign;
    int32_t const base = rank * S;
    int32_t const n = max(0, min(kvLen - base, S));
    int32_t const rowWin = kvLen - p.nextN + row + 1;
    bool const rowActive = row < p.numRows;
    bool const exchangeLaunch = kvLen - p.nextN + p.numRows > kTopK;
    bool const rowExchange = rowActive && rowWin > kTopK;
    int32_t const v = max(0, min(rowWin - base, n));
    int32_t const chunk = S / kCtas;
    int32_t const cbase = c * chunk;

    for (int32_t i = t; i < kBins; i += kThreads)
    {
        hist[i] = 0;
        rhist[i] = 0;
    }
    if (t == 0)
    {
        misc[kMiscSmallCount] = 0;
        if (rowExchange)
        {
            // One arrival per phase (this thread, now), completing when the peers' bytes arrived.
            for (int32_t lv = 0; lv < kLevels; ++lv)
            {
                barInit(histBar0 + 8 * lv, 1);
                barInit(preBar0 + 8 * lv, 1);
                barArriveExpect(histBar0 + 8 * lv, (kCtas - 1) * (1 << levelBits(lv)) / kCtas * 4);
                barArriveExpect(preBar0 + 8 * lv, (kCtas - 1) * kPreRow * 4);
            }
            fenceBarInit();
        }
    }
    __syncthreads(); // the zeroed histograms before any thread counts into them, the barriers before their waits
    if (rowExchange)
    {
        // Every CTA's barriers are initialized before any peer pushes to them (waited for after the logits load).
        __syncwarp();
        clusterArriveRelaxed();
    }

    asm volatile("griddepcontrol.wait;" ::: "memory");
    stamp(dbg, kDbgWaited);

    int32_t* const outRow = p.out + int64_t{row} * kTopK;
    if (rowActive && !rowExchange && t < kOutPerCta)
    {
        int32_t const pos = c * kOutPerCta + t;
        outRow[pos] = pos < rowWin ? pos : -1;
    }
    if (!exchangeLaunch)
    {
        flushDebug(p.debug, row, c);
        return;
    }
    WaitCtx const wc{p.state, p.timeoutNs, rank};
    if (S > p.localCapacity || S > kMaxShard)
    {
        abortKernel(wc, 400);
    }

    // This CTA's chunk of the row's shard logits: keys into shared memory (each thread reads back only its own).
    float const* rowLogits = p.logits + int64_t{row} * p.logitsStride + cbase;
    float4 f[kVecs];
#pragma unroll
    for (int32_t k = 0; k < kVecs; ++k)
    {
        int32_t const q = t + k * kThreads;
        f[k] = make_float4(0.F, 0.F, 0.F, 0.F);
        if (rowExchange && 4 * q < chunk)
        {
            // Read once: not cached in L1, which then keeps the register spills.
            f[k] = __ldcg(reinterpret_cast<float4 const*>(rowLogits + 4 * q));
        }
    }
    // The epoch: every CTA of an exchange launch adds one to the count once it has read it (off the critical path:
    // after its final send), and this launch started after the previous one completed, so a CTA reads base + k with
    // base a multiple of kCtasPerLaunch and k < kCtasPerLaunch: count / kCtasPerLaunch is the number of earlier
    // exchange launches.
    unsigned long long* const counter = reinterpret_cast<unsigned long long*>(p.state + kStateCount);
    uint64_t const arrivals = *reinterpret_cast<unsigned long long volatile*>(counter);
    int32_t const buf = static_cast<int32_t>((arrivals / kCtasPerLaunch) % kNumBuffers);
    int32_t const prevBuf = (buf + kNumBuffers - 1) % kNumBuffers;
    int32_t const* rec = p.state + kStateRecords + row * kRecordWords;
    int32_t const recLevels = rec[0];
    int32_t const recOut = rec[1];
    int32_t const recBnd = rec[2];
    if (t == 0)
    {
        misc[kMiscRec] = static_cast<uint32_t>(recLevels);
        misc[kMiscRec + 1] = static_cast<uint32_t>(recOut);
        misc[kMiscRec + 2] = static_cast<uint32_t>(recBnd);
    }
    if (rowExchange)
    {
        __syncwarp();
        clusterWait();
    }

    uint32_t validMask = 0;
#pragma unroll
    for (int32_t k = 0; k < kVecs; ++k)
    {
        int32_t const q = t + k * kThreads;
        if (rowExchange && 4 * q < chunk)
        {
            keys[q] = make_uint4(orderedKey(f[k].x), orderedKey(f[k].y), orderedKey(f[k].z), orderedKey(f[k].w));
            int32_t const col = cbase + 4 * q;
#pragma unroll
            for (int32_t j = 0; j < 4; ++j)
            {
                validMask |= (col + j < v ? 1u : 0u) << (4 * k + j);
            }
        }
    }
    stamp(dbg, kDbgLoaded);

    int32_t mode = kModeIdentity;
    int32_t levelsUsed = 0;
    uint32_t totalSel = 0;
    uint32_t nbFinal = 0;

    if (!rowExchange)
    {
        clearPrevious(p, prevBuf, row, recLevels, recOut, recBnd, wc, c, kCtas);
        if (t == 0)
        {
            atomicAdd(counter, 1ull);
        }
        clusterSync();
    }
    else
    {
        uint32_t need = kTopK;
        uint32_t aboveAllTot = 0;
        uint32_t lowTot = 0;
        uint32_t prefix = 0;
        int32_t level = 0;
        for (;; ++level)
        {
            int32_t const nbins = 1 << levelBits(level);
            int32_t const shiftPrev = level == 0 ? 0 : levelShift(level - 1);
            int32_t const shiftNow = levelShift(level);
            uint32_t const maskNow = (1u << levelBits(level)) - 1u;
#pragma unroll
            for (int32_t k = 0; k < kVecs; ++k)
            {
                uint32_t const vm = (validMask >> (4 * k)) & 0xFu;
                if (vm != 0u)
                {
                    uint4 const kv = keys[t + k * kThreads];
                    uint32_t const kk[4] = {kv.x, kv.y, kv.z, kv.w};
#pragma unroll
                    for (int32_t j = 0; j < 4; ++j)
                    {
                        if (((vm >> j) & 1u) && (level == 0 || (kk[j] >> shiftPrev) == prefix))
                        {
                            atomicAdd(&hist[(kk[j] >> shiftNow) & maskNow], 1u);
                        }
                    }
                }
            }
            if (level == 0)
            {
                stamp(dbg, kDbgLocalHist);
            }
            __syncthreads(); // this CTA's local histogram of this level is complete
            if (level == 0)
            {
                stamp(dbg, kDbgB0);
            }
            uint32_t* const ucLevel = p.ucWorkspace + histOff(buf, level, row, 0);
            pushAndSend(hist, nbins, slots, histBar0 + 8 * level, ws, p.mcWorkspace + histOff(buf, level, row, rank));
            if (level == 0)
            {
                stamp(dbg, kDbgSent0);
            }
            if (t < 2 * kMaxGroups)
            {
                gsum[t] = 0; // receiveCut's sums, ordered before it by localSuffix's barrier
            }
            localSuffix(hist, nbins, suff, ws + 192, level == 0 ? misc + kMiscBucketBin : nullptr);
            stamp(dbg, kDbgSuffix);
            // The next level's local histogram starts from zero (receiveCut's barriers order this before it).
            for (int32_t i = t; i < kBins; i += kThreads)
            {
                hist[i] = 0;
            }
            if (level == 0)
            {
                // This CTA's keys in the bins the cut cannot be above, by bin from the top (bin b at
                // [suff[b + 1], suff[b])): the keys of a level-0 cut are then prefixes of the buckets.
                uint32_t const bucketBin = misc[kMiscBucketBin];
#pragma unroll
                for (int32_t k = 0; k < kVecs; ++k)
                {
                    uint32_t const vm = (validMask >> (4 * k)) & 0xFu;
                    if (vm != 0u)
                    {
                        uint4 const kv = keys[t + k * kThreads];
                        uint32_t const kk[4] = {kv.x, kv.y, kv.z, kv.w};
#pragma unroll
                        for (int32_t j = 0; j < 4; ++j)
                        {
                            uint32_t const b = kk[j] >> shiftNow;
                            if (((vm >> j) & 1u) && b >= bucketBin)
                            {
                                uint32_t const pos = suff[b + 1] + atomicAdd(&cursor[b], 1u);
                                bucket[pos] = static_cast<uint16_t>(4 * (t + k * kThreads) + j);
                            }
                        }
                    }
                }
                stamp(dbg, kDbgBucketed);
            }
            receiveCut(p, ucLevel, suff, nbins, level, need, lowTot, coarse, fine, gsum, misc, pre, preBar0, cut, offs,
                wc, 1 + level, dbg);
            uint32_t const bin = cut[kCutBin];
            if (bin == kEmpty)
            {
                abortKernel(wc, 300 + level);
            }
            putDebug(dbg, kDbgCutGroup, cut[kCutGroup]);
            putDebug(dbg, kDbgCutBin, bin);
            uint32_t const aboveAll = cut[kCutAboveAll];
            uint32_t const binAll = cut[kCutBinAll];
            uint32_t const aboveLow = cut[kCutAboveLow];
            if (t == 0)
            {
                misc[kMiscLevelBin + level] = bin;
                misc[kMiscLevelGroup + level] = cut[kCutGroup];
            }
            aboveAllTot += aboveAll;
            lowTot += aboveLow;
            need -= aboveAll;
            prefix = level == 0 ? bin : ((prefix << levelBits(level)) | bin);
            if (need == binAll)
            {
                mode = kModeFull;
            }
            else if (level == kLevels - 1)
            {
                mode = kModeTies;
            }
            else if (binAll <= static_cast<uint32_t>(kBoundaryCap))
            {
                mode = kModeCand;
            }
            else
            {
                mode = kModeNext;
            }
            if (mode != kModeNext)
            {
                break;
            }
        }
        levelsUsed = level + 1;
        nbFinal = cut[kCutBinAll];
        putDebug(dbg, kDbgLevels, levelsUsed);
        putDebug(dbg, kDbgMode, mode);
        putDebug(dbg, kDbgBoundary, nbFinal);
        putDebug(dbg, kDbgNeed, need);

        // offs[]: this CTA's offsets, computed by warp 0 in receiveCut at the final cut.

        totalSel = mode == kModeCand ? aboveAllTot : static_cast<uint32_t>(kTopK);
        int32_t const shiftF = levelShift(level);
        uint32_t selCount = 0;
        uint32_t bndCount = 0;
        uint32_t const outOffC = offs[kOffsOut];
        uint32_t const bndOffC = offs[kOffsBnd];
        uint32_t* const mcOut = p.mcWorkspace + outOff(buf, row) + outOffC;
        uint32_t* const mcBnd = p.mcWorkspace + bndOff(buf, row) + 2 * int64_t{bndOffC};
        if (level == 0)
        {
            // The selected tokens and the boundary pairs are prefixes of the buckets (kModeFull or kModeCand).
            uint32_t const nAbove = offs[kOffsAboveCta];
            bndCount = offs[kOffsBndCta];
            selCount = mode == kModeFull ? nAbove + bndCount : nAbove;
            stamp(dbg, kDbgStaged);
            uint32_t const tokenBase = static_cast<uint32_t>(base + cbase);
            storeBucketTokens(mcOut, bucket, tokenBase, static_cast<int32_t>(selCount));
            stamp(dbg, kDbgTokSent);
            if (mode == kModeCand)
            {
                storeBucketPairs(mcBnd, bucket + nAbove, reinterpret_cast<uint32_t const*>(keys), tokenBase,
                    static_cast<int32_t>(bndCount));
            }
        }
        else
        {
            // This CTA's keys above the final cut and in its bin.
            uint32_t aboveMask = 0;
            uint32_t bndMask = 0;
#pragma unroll
            for (int32_t k = 0; k < kVecs; ++k)
            {
                uint32_t const vm = (validMask >> (4 * k)) & 0xFu;
                if (vm != 0u)
                {
                    uint4 const kv = keys[t + k * kThreads];
                    uint32_t const kk[4] = {kv.x, kv.y, kv.z, kv.w};
#pragma unroll
                    for (int32_t j = 0; j < 4; ++j)
                    {
                        uint32_t const s = kk[j] >> shiftF;
                        if ((vm >> j) & 1u)
                        {
                            aboveMask |= (s > prefix ? 1u : 0u) << (4 * k + j);
                            bndMask |= (s == prefix ? 1u : 0u) << (4 * k + j);
                        }
                    }
                }
            }
            stamp(dbg, kDbgClassify);
            uint32_t bndBase[kVecs];
            uint32_t selBase[kVecs];
            uint32_t selMask = aboveMask;
            if (mode != kModeTies)
            {
                if (mode == kModeFull)
                {
                    selMask |= bndMask;
                }
                positionBases(bndMask, selMask, ws, bndBase, selBase, bndCount, selCount);
            }
            else
            {
                // The CTA's last offs[kOffsTake] keys of the exact key: the larger tokens.
                uint32_t unusedBase[kVecs];
                uint32_t unusedCount = 0;
                positionBases(bndMask, 0u, ws, bndBase, unusedBase, bndCount, unusedCount);
                uint32_t const firstTaken = offs[kOffsBndCta] - offs[kOffsTake];
#pragma unroll
                for (int32_t k = 0; k < kVecs; ++k)
                {
                    uint32_t pos = bndBase[k];
#pragma unroll
                    for (int32_t j = 0; j < 4; ++j)
                    {
                        uint32_t const bit = 1u << (4 * k + j);
                        if (bndMask & bit)
                        {
                            selMask |= pos >= firstTaken ? bit : 0u;
                            ++pos;
                        }
                    }
                }
                positionBases(selMask, 0u, ws + 128, selBase, unusedBase, selCount, unusedCount);
            }
            stamp(dbg, kDbgStaged);

            // The selected tokens into every rank's output stage, staged here so that the multicast stores are 16
            // bytes wide; in kModeCand the boundary pairs straight into every rank's boundary stage (8 bytes each).
            uint32_t const pairMask = mode == kModeCand ? bndMask : 0u;
#pragma unroll
            for (int32_t k = 0; k < kVecs; ++k)
            {
                if ((((selMask | pairMask) >> (4 * k)) & 0xFu) != 0u)
                {
                    uint4 const kv = keys[t + k * kThreads];
                    uint32_t const kk[4] = {kv.x, kv.y, kv.z, kv.w};
                    uint32_t sp = selBase[k];
                    uint32_t bp = bndBase[k];
#pragma unroll
                    for (int32_t j = 0; j < 4; ++j)
                    {
                        uint32_t const bit = 1u << (4 * k + j);
                        uint32_t const token = static_cast<uint32_t>(base + cbase + 4 * (t + k * kThreads) + j);
                        if (selMask & bit)
                        {
                            stageTok[sp] = token;
                            ++sp;
                        }
                        else if (pairMask & bit)
                        {
                            st2(mcBnd + 2 * bp, kk[j], token);
                        }
                        bp += (bndMask & bit) ? 1u : 0u;
                    }
                }
            }
            __syncthreads();
            storeWords(mcOut, stageTok, static_cast<int32_t>(selCount));
        }
        stamp(dbg, kDbgFinalSent);
        if (t == 0)
        {
            atomicAdd(counter, 1ull); // the epoch (see above), while the stage words travel
        }
        putDebug(dbg, kDbgSelCount, selCount);
        putDebug(dbg, kDbgBndCount, bndCount);
        putDebug(dbg, kDbgOutOff, outOffC);
        putDebug(dbg, kDbgBndOff, bndOffC);
        putDebug(dbg, kDbgOwnAbove, offs[kOffsAboveCta]);

        // The resolver requests all ranks' boundary pairs (nb <= kBoundaryCap: at most two 16-byte words per
        // thread) before the output stage is copied.
        int32_t const nb = static_cast<int32_t>(nbFinal);
        bool const resolver = mode == kModeCand && c == kCtas - 1;
        uint32_t const* ucBnd = p.ucWorkspace + bndOff(buf, row);
        uint32_t const* ucOut = p.ucWorkspace + outOff(buf, row);
        uint4 bv[2] = {make_uint4(0, 0, 0, 0), make_uint4(0, 0, 0, 0)};
        int32_t bvalid[2] = {0, 0};
        if (resolver)
        {
            int32_t const vecs = (2 * nb + 3) / 4;
#pragma unroll
            for (int32_t k = 0; k < 2; ++k)
            {
                int32_t const vi = t + k * kThreads;
                bvalid[k] = vi < vecs ? min(4, 2 * nb - 4 * vi) : 0;
                if (bvalid[k] > 0)
                {
                    bv[k] = ldVolatile4(ucBnd + 4 * vi);
                }
            }
            // The level-0 bucket fill counters back to zero: they are the resolve histogram.
            for (int32_t i = static_cast<int32_t>(misc[kMiscBucketBin]) + t; i < kBins; i += kThreads)
            {
                cursor[i] = 0;
            }
            __syncthreads();
        }
        else
        {
            // What the previous exchange launch wrote into buffer prevBuf for this row, cleared by the CTAs other
            // than the resolver while the peers' stage words travel.
            clearPrevious(p, prevBuf, row, static_cast<int32_t>(misc[kMiscRec]),
                static_cast<int32_t>(misc[kMiscRec + 1]), static_cast<int32_t>(misc[kMiscRec + 2]), wc, c,
                mode == kModeCand ? kCtas - 1 : kCtas);
            stamp(dbg, kDbgCleared);
        }
        // The output stage of every rank into the output row: this CTA's kOutPerCta positions.
        if (t < kOutPerCta / 4)
        {
            int32_t const pos = c * kOutPerCta + 4 * t;
            int32_t const valid = min(4, max(0, static_cast<int32_t>(totalSel) - pos));
            if (valid > 0)
            {
                uint4 const w = pollVec(ucOut + pos, valid, wc, 10);
                if (valid == 4)
                {
                    *reinterpret_cast<int4*>(outRow + pos) = make_int4(static_cast<int32_t>(w.x),
                        static_cast<int32_t>(w.y), static_cast<int32_t>(w.z), static_cast<int32_t>(w.w));
                }
                else
                {
                    uint32_t const ww[4] = {w.x, w.y, w.z, w.w};
                    for (int32_t j = 0; j < valid; ++j)
                    {
                        outRow[pos + j] = static_cast<int32_t>(ww[j]);
                    }
                }
            }
        }
        stamp(dbg, kDbgOutCopied);
        if (resolver)
        {
            PollClock clk;
            while (anyEmpty(bv[0], bvalid[0]) || anyEmpty(bv[1], bvalid[1]))
            {
                clk.tick(wc, 11);
#pragma unroll
                for (int32_t k = 0; k < 2; ++k)
                {
                    if (anyEmpty(bv[k], bvalid[k]))
                    {
                        bv[k] = ldVolatile4(ucBnd + 4 * (t + k * kThreads));
                    }
                }
            }
            stamp(dbg, kDbgBndPolled);
            uint32_t const pk[4] = {bv[0].x, bv[0].z, bv[1].x, bv[1].z};
            uint32_t const pt[4] = {bv[0].y, bv[0].w, bv[1].y, bv[1].w};
            uint32_t const pv = (bvalid[0] >= 2 ? 1u : 0u) | (bvalid[0] == 4 ? 2u : 0u) | (bvalid[1] >= 2 ? 4u : 0u)
                | (bvalid[1] == 4 ? 8u : 0u);
            resolveBoundary(pk, pt, pv, need, shiftF, prefix << shiftF, rhist, ws + 192, cut, small,
                misc + kMiscSmallCount, outRow, static_cast<int32_t>(totalSel), dbg);
            stamp(dbg, kDbgResolved);
        }
    }

    // This launch's record for the next exchange launch's clearing (every CTA of the row read the previous record
    // before it pushed its level-0 counts, which CTA 0 waited for; a row without an exchange passed its cluster
    // barrier).
    if (t == 0 && c == 0)
    {
        int32_t* recW = p.state + kStateRecords + row * kRecordWords;
        recW[0] = levelsUsed;
        recW[1] = static_cast<int32_t>(totalSel);
        recW[2] = mode == kModeCand ? static_cast<int32_t>(nbFinal) : 0;
    }
    stamp(dbg, kDbgEnd);
    putDebug(dbg, kDbgClockEnd, static_cast<int64_t>(clock64()));
    flushDebug(p.debug, row, c);
#else
    (void) p;
#endif
}

__global__ void dsaHistShardMetaKernel(int32_t const* kvLens, int32_t const* blockTable, int32_t blockTableWidth,
    int32_t worldSize, int32_t rank, int32_t tokensPerBlock, int32_t* shardCtx, int32_t* shardBlockTable,
    int32_t shardBlockTableWidth)
{
    int32_t const kvLen = kvLens[0];
    int32_t const perRank = (max(kvLen, 0) + worldSize - 1) / worldSize;
    int32_t const size = (perRank + kShardAlign - 1) / kShardAlign * kShardAlign;
    int32_t const base = rank * size;
    int32_t const n = min(max(kvLen - base, 0), size);
    if (threadIdx.x == 0)
    {
        shardCtx[0] = max(n, 1);
    }
    int32_t const firstPage = base / tokensPerBlock;
    int32_t const pages = (n + tokensPerBlock - 1) / tokensPerBlock;
    for (int32_t j = threadIdx.x; j < shardBlockTableWidth; j += blockDim.x)
    {
        shardBlockTable[j] = (j < pages && firstPage + j < blockTableWidth) ? blockTable[firstPage + j] : 0;
    }
}

int64_t defaultTimeoutNs()
{
    return kDefaultTimeoutMs * 1000000;
}

} // namespace

void invokeDsaHistTopK(HistTopKParams const& inParams, cudaStream_t stream)
{
    HistTopKParams params = inParams;
    if (params.numRows <= 0 || params.numRows > kMaxRows || params.worldSize <= 0 || params.worldSize > kMaxRanks
        || params.rank < 0 || params.rank >= params.worldSize || params.localCapacity > kMaxShard
        || params.logitsStride % 4 != 0 || reinterpret_cast<uintptr_t>(params.logits) % 16 != 0
        || reinterpret_cast<uintptr_t>(params.state) % 8 != 0)
    {
        throw std::invalid_argument("glm_dsa_hist_topk: unsupported rows, group, shard capacity or alignment");
    }
    if (params.timeoutNs == 0)
    {
        params.timeoutNs = static_cast<uint64_t>(defaultTimeoutNs());
    }
    static thread_local bool configured[64] = {};
    int device = 0;
    cudaGetDevice(&device);
    cudaError_t err = cudaSuccess;
    if (device >= 64 || !configured[device])
    {
        err = cudaFuncSetAttribute(
            dsaHistTopKKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmemBytes));
        if (err != cudaSuccess)
        {
            throw std::runtime_error(std::string("glm_dsa_hist_topk: smem attribute: ") + cudaGetErrorString(err));
        }
        if (device < 64)
        {
            configured[device] = true;
        }
    }
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(kCtas, kMaxRows);
    config.blockDim = dim3(kThreads);
    config.dynamicSmemBytes = kSmemBytes;
    config.stream = stream;
    cudaLaunchAttribute attrs[2];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = kCtas;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[1].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    config.attrs = attrs;
    config.numAttrs = 2;
    err = cudaLaunchKernelEx(&config, dsaHistTopKKernel, params);
    if (err == cudaSuccess)
    {
        err = cudaGetLastError();
    }
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("glm_dsa_hist_topk: launch: ") + cudaGetErrorString(err));
    }
}

void invokeDsaHistShardMeta(int32_t const* kvLens, int32_t const* blockTable, int32_t blockTableWidth,
    int32_t worldSize, int32_t rank, int32_t tokensPerBlock, int32_t* shardCtx, int32_t* shardBlockTable,
    int32_t shardBlockTableWidth, cudaStream_t stream)
{
    if (worldSize <= 0 || worldSize > kMaxRanks || rank < 0 || rank >= worldSize || tokensPerBlock <= 0
        || kShardAlign % tokensPerBlock != 0)
    {
        throw std::invalid_argument("glm_dsa_hist_shard_meta: unsupported group or page size");
    }
    dsaHistShardMetaKernel<<<1, 256, 0, stream>>>(kvLens, blockTable, blockTableWidth, worldSize, rank, tokensPerBlock,
        shardCtx, shardBlockTable, shardBlockTableWidth);
    cudaError_t const err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("glm_dsa_hist_shard_meta: launch: ") + cudaGetErrorString(err));
    }
}

} // namespace kernels::glm_moe_dsa::hist_topk

TRTLLM_NAMESPACE_END
