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

// Device route and statistics of the adaptive expert load balancer. Both read fixed-address tables and write
// caller-owned buffers on the given stream, so CUDA graphs can capture them.

#include "tensorrt_llm/kernels/moe/loadBalance/adaptiveEplbKernels.h"

#include "tensorrt_llm/common/envUtils.h"

#include <mutex>
#include <unordered_map>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

namespace
{

constexpr int kSplitBlock = 1024; // elements per split block, the block size of moeComputeRouteKernel (256 threads x 4)
constexpr int kTileThreads = 256; // threads of one route block, which routes kTileThreads elements of a split block
constexpr int kTilesPerSplitBlock = kSplitBlock / kTileThreads;
constexpr int kWarpsPerTile = kTileThreads / 32;
constexpr int kWarpWordsPerExpert = kWarpsPerTile / 4; // words of the per-warp byte counts of one expert
// Table entries every thread loads before the first shared-memory store: the loads of a table of up to
// kTableEntriesPerThread * kTileThreads = 512 entries (its experts and its slots, e.g. 384 experts in 512 slots) share
// one memory round trip. A longer table is routed alike; it costs one further round trip per kTileThreads entries,
// and a larger value would cost every launch four more registers per entry.
constexpr int kTableEntriesPerThread = 2;
// The route's offset of a split block: kBlockMultiplier is floor(2^32 / golden ratio), so consecutive blocks' offsets
// stay far apart; kExpertMultiplier and kRankMultiplier, odd constants, separate the experts of one block and the
// ranks, for any number of split blocks in a launch.
constexpr uint32_t kBlockMultiplier = 0x9E3779B9u;
constexpr uint32_t kExpertMultiplier = 0x632BE5ABu;
constexpr uint32_t kRankMultiplier = 0x85EBCA6Bu;
constexpr int kRecordThreads = 1024;
constexpr size_t kDefaultSharedBytes = 48 * 1024;

// Raise a kernel's dynamic shared memory limit on the current device when a launch needs more than the default, once
// per device and size.
template <auto Kernel>
void allowSharedBytes(size_t bytes)
{
    if (bytes <= kDefaultSharedBytes)
    {
        return;
    }
    int device = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));
    static std::mutex mutex;
    static std::unordered_map<int, size_t> allowed;
    std::lock_guard<std::mutex> lock(mutex);
    size_t& limit = allowed.try_emplace(device, kDefaultSharedBytes).first->second;
    if (bytes > limit)
    {
        TLLM_CUDA_CHECK(
            cudaFuncSetAttribute(Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes)));
        limit = bytes;
    }
}

static_assert(kSplitBlock % kTileThreads == 0 && kTileThreads % 32 == 0, "tiles are whole warps of one split block");
static_assert(kWarpsPerTile % 4 == 0, "the per-warp counts of an expert fill whole words");

template <typename IdT>
__device__ __forceinline__ int sanitizeExpert(IdT raw, int numExperts)
{
    return (raw < 0 || raw >= numExperts) ? numExperts : static_cast<int>(raw);
}

// The id an element is counted under when its expert's elements are spread over copies: the expert if it has several
// copies, else numExperts, which no spreading uses.
__device__ __forceinline__ int spreadId(int expert, int const* counts, int numExperts)
{
    return (expert < numExperts && counts[expert] > 1) ? expert : numExperts;
}

// floor(2^32 / n) for n > 1 by 32-bit divisions: with 2^32 = q n + r, floor((2^32 - 1) / n) is q - 1 when r == 0
// (and then (2^32 - 1) mod n == n - 1), else q. 2^32 - 1 for n <= 1.
__device__ __forceinline__ uint32_t pointStep(int n)
{
    if (n <= 1)
    {
        return 0xFFFFFFFFu;
    }
    uint32_t const un = static_cast<uint32_t>(n);
    uint32_t const q = 0xFFFFFFFFu / un;
    return q + ((0xFFFFFFFFu - q * un) == un - 1u ? 1u : 0u);
}

__device__ __forceinline__ int weightedSlot(int expert, int ordinal, uint32_t step, int block, int epRank,
    int const* counts, int const* starts, int const* slots, uint32_t const* cumulativeShares, int numExperts,
    int numSlots)
{
    if (expert < 0 || expert >= numExperts)
    {
        return numSlots;
    }
    int const copies = counts[expert];
    int const start = starts[expert];
    if (copies <= 0 || start < 0 || start + copies > numSlots)
    {
        return numSlots;
    }
    int k = 0;
    if (copies > 1)
    {
        uint32_t const point = static_cast<uint32_t>(expert) * kExpertMultiplier
            + static_cast<uint32_t>(block) * kBlockMultiplier + static_cast<uint32_t>(epRank) * kRankMultiplier
            + static_cast<uint32_t>(ordinal) * step;
        while (k < copies - 1 && point >= cumulativeShares[start + k])
        {
            ++k;
        }
    }
    return slots[start + k];
}

// A table without replicated experts routes every expert to its only copy.
__device__ __forceinline__ int firstCopySlot(
    int expert, int const* counts, int const* starts, int const* slots, int numExperts, int numSlots)
{
    if (expert < 0 || expert >= numExperts)
    {
        return numSlots;
    }
    int const copies = counts[expert];
    int const start = starts[expert];
    if (copies <= 0 || start < 0 || start >= numSlots)
    {
        return numSlots;
    }
    return slots[start];
}

size_t routeSharedBytes(int numExperts, int numSlots)
{
    return (4 * static_cast<size_t>(numExperts) + 2 * static_cast<size_t>(numSlots)) * sizeof(int)
        + static_cast<size_t>(kWarpsPerTile) * numExperts;
}

// Store element f: its slot and gate weight while f is a routed element, slot -1 and weight 0 in the rest of the
// staged bucket, nothing past the bucket.
template <typename OutT>
__device__ __forceinline__ void storeRoute(
    int64_t f, int64_t count, int slot, OutT* out, AdaptiveEplbStaging const& staging)
{
    if (f < count)
    {
        out[f] = static_cast<OutT>(slot);
        if (staging.weightsOut != nullptr)
        {
            staging.weightsOut[(f / staging.topK) * staging.ldWeights + f % staging.topK] = staging.weightsIn[f];
        }
    }
    else if (f < staging.total)
    {
        out[f] = static_cast<OutT>(-1);
        staging.weightsOut[(f / staging.topK) * staging.ldWeights + f % staging.topK] = 0.0f;
    }
}

// The ids of the live rows among count ids in rows of topK (adaptiveEplbRecord); every id without a word.
__device__ __forceinline__ int64_t liveIdCount(int const* liveRowWord, int64_t count, int topK)
{
    if (liveRowWord == nullptr)
    {
        return count;
    }
    int const word = __ldg(liveRowWord);
    int const real = word & 0xFFFF;
    int const requests = (word >> 16) & 0xFFFF;
    int64_t const rows = count / topK;
    if (requests == 0)
    {
        return 0;
    }
    return real >= requests || rows % requests != 0 ? count : rows / requests * real * topK;
}

__device__ __forceinline__ void addCounts(int const* counts, int numExperts, float* rows, int64_t* forwards, bool first)
{
    for (int e = threadIdx.x; e < numExperts; e += blockDim.x)
    {
        rows[e] += static_cast<float>(counts[e]);
    }
    if (threadIdx.x == 0 && first)
    {
        forwards[0] += 1;
    }
}

// Count ids [begin, end) of a block into the shared histogram. The enable flag and the thread's first id are loaded
// before the histogram is zeroed, so the flag's read costs no round trip of its own; the block returns false while
// the flag is clear.
template <typename IdT>
__device__ __forceinline__ bool countIds(
    IdT const* ids, int64_t begin, int64_t end, int numExperts, int const* enabled, int* histogram)
{
    int const on = __ldg(enabled);
    int64_t i = begin + threadIdx.x;
    IdT const firstId = i < end ? ids[i] : static_cast<IdT>(-1);
    for (int e = threadIdx.x; e < numExperts; e += blockDim.x)
    {
        histogram[e] = 0;
    }
    __syncthreads();
    if (on == 0)
    {
        return false;
    }
    if (firstId >= 0 && firstId < numExperts)
    {
        atomicAdd_block(histogram + static_cast<int>(firstId), 1);
    }
    for (i += blockDim.x; i < end; i += blockDim.x)
    {
        IdT const e = ids[i];
        if (e >= 0 && e < numExperts)
        {
            atomicAdd_block(histogram + e, 1);
        }
    }
    __syncthreads();
    return true;
}

// One block per tile of kTileThreads elements; thread t routes element t of the tile. An element's ordinal j among
// its expert's elements in the split block needs the counts of every spread id there, so every block counts its whole
// split block in shared memory: per id in the split block, in the tiles before its own, and in each warp of its own
// tile, where __match_any_sync ranks the lanes of equal ids. A tile past the routed elements only writes the staged
// bucket's rest; with statistics, the launch's last block counts all the ids and updates them.
// Under PDL the launch may start while the stream's earlier kernel drains; completion is reported when the blocks
// exit, so a dependent that waits on this grid reads whole routes without a fence of this kernel's own.
template <typename IdT, typename OutT>
__global__ void __launch_bounds__(kTileThreads)
    adaptiveEplbRouteKernel(IdT const* __restrict__ ids, int64_t count, int epRank, AdaptiveEplbRouteTables tables,
        OutT* __restrict__ out, AdaptiveEplbStaging staging, AdaptiveEplbStats stats)
{
    extern __shared__ __align__(16) int shared[];
#if __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize(); // launched early under PDL: the ids and the tables are the stream's earlier work
#endif
    int const numExperts = tables.numExperts;
    int const numSlots = tables.numSlots;
    if (stats.rows != nullptr && blockIdx.x == gridDim.x - 1)
    {
        if (countIds(ids, 0, liveIdCount(stats.liveRowWord, count, stats.topK), numExperts, stats.enabled, shared))
        {
            addCounts(shared, numExperts, stats.rows, stats.forwards, stats.first);
        }
        return;
    }
    if (static_cast<int64_t>(blockIdx.x) * kTileThreads >= count)
    {
        storeRoute(static_cast<int64_t>(blockIdx.x) * kTileThreads + threadIdx.x, count, 0, out, staging);
        return;
    }
    // Shared memory: the tables, and per spread id its elements in the split block (total), in the tiles before this
    // one (before) and in each warp of this tile (warpCount, at most 32 each).
    int* counts = shared;
    int* starts = counts + numExperts;
    int* total = starts + numExperts;
    int* before = total + numExperts;
    int* slots = before + numExperts;
    uint32_t* cumulativeShares = reinterpret_cast<uint32_t*>(slots + numSlots);
    auto* warpCount = reinterpret_cast<unsigned char*>(cumulativeShares + numSlots);

    int const t = threadIdx.x;
    int const lane = t & 31;
    int const warp = t >> 5;
    int const block = blockIdx.x / kTilesPerSplitBlock;
    int const tile = blockIdx.x % kTilesPerSplitBlock;
    int64_t const base = static_cast<int64_t>(block) * kSplitBlock;
    // The ids and the first kTableEntriesPerThread entries of every table go to registers before the first shared
    // store, so that their reads overlap in one memory round trip.
    IdT raw[kTilesPerSplitBlock];
#pragma unroll
    for (int i = 0; i < kTilesPerSplitBlock; ++i)
    {
        int64_t const f = base + t + kTileThreads * i;
        raw[i] = f < count ? ids[f] : static_cast<IdT>(numExperts);
    }
    int copiesOf[kTableEntriesPerThread];
    int startOf[kTableEntriesPerThread];
    int slotOf[kTableEntriesPerThread];
    uint32_t sharesOf[kTableEntriesPerThread];
#pragma unroll
    for (int k = 0; k < kTableEntriesPerThread; ++k)
    {
        int const x = t + kTileThreads * k;
        copiesOf[k] = x < numExperts ? __ldg(tables.counts + x) : 0;
        startOf[k] = x < numExperts ? __ldg(tables.starts + x) : 0;
        slotOf[k] = x < numSlots ? __ldg(tables.slots + x) : 0;
        sharesOf[k] = x < numSlots ? __ldg(tables.cumulativeShares + x) : 0u;
    }
    int expert[kTilesPerSplitBlock];
#pragma unroll
    for (int i = 0; i < kTilesPerSplitBlock; ++i)
    {
        expert[i] = sanitizeExpert(raw[i], numExperts);
    }
    auto* warpWords = reinterpret_cast<unsigned*>(warpCount); // kWarpWordsPerExpert * numExperts words
    int replicated = 0;
#pragma unroll
    for (int k = 0; k < kTableEntriesPerThread; ++k)
    {
        int const x = t + kTileThreads * k;
        if (x < numExperts)
        {
            counts[x] = copiesOf[k];
            starts[x] = startOf[k];
            total[x] = 0;
            before[x] = 0;
#pragma unroll
            for (int w = 0; w < kWarpWordsPerExpert; ++w)
            {
                warpWords[w * numExperts + x] = 0u;
            }
            replicated |= copiesOf[k] > 1 ? 1 : 0;
        }
        if (x < numSlots)
        {
            slots[x] = slotOf[k];
            cumulativeShares[x] = sharesOf[k];
        }
    }
    // Longer tables: the remaining entries after the stores, one round trip per kTileThreads entries.
#pragma unroll 1
    for (int x = t + kTableEntriesPerThread * kTileThreads; x < numExperts; x += kTileThreads)
    {
        int const copies = __ldg(tables.counts + x);
        counts[x] = copies;
        starts[x] = __ldg(tables.starts + x);
        total[x] = 0;
        before[x] = 0;
        for (int w = 0; w < kWarpWordsPerExpert; ++w)
        {
            warpWords[w * numExperts + x] = 0u;
        }
        replicated |= copies > 1 ? 1 : 0;
    }
#pragma unroll 1
    for (int x = t + kTableEntriesPerThread * kTileThreads; x < numSlots; x += kTileThreads)
    {
        slots[x] = __ldg(tables.slots + x);
        cumulativeShares[x] = __ldg(tables.cumulativeShares + x);
    }
    int ownExpert = expert[0];
#pragma unroll
    for (int i = 1; i < kTilesPerSplitBlock; ++i)
    {
        ownExpert = tile == i ? expert[i] : ownExpert;
    }
    int64_t const f = base + static_cast<int64_t>(tile) * kTileThreads + t;
    if (!__syncthreads_or(replicated))
    {
        storeRoute(f, count, firstCopySlot(ownExpert, counts, starts, slots, numExperts, numSlots), out, staging);
        return;
    }
    int ownId = numExperts;
    int ownLaneOrdinal = 0;
#pragma unroll
    for (int i = 0; i < kTilesPerSplitBlock; ++i)
    {
        // Iteration i: this warp holds the elements of the same warp in tile i of the split block.
        int const id = spreadId(expert[i], counts, numExperts);
        unsigned const peers = __match_any_sync(0xFFFFFFFFu, id);
        int const laneOrdinal = __popc(peers & ((1u << lane) - 1u));
        if (laneOrdinal == 0 && id < numExperts)
        {
            int const k = __popc(peers);
            atomicAdd(&total[id], k);
            if (i < tile)
            {
                atomicAdd(&before[id], k);
            }
            else if (i == tile)
            {
                warpCount[warp * numExperts + id] = static_cast<unsigned char>(k);
            }
        }
        if (i == tile)
        {
            ownId = id;
            ownLaneOrdinal = laneOrdinal;
        }
    }
    __syncthreads();
    int ordinal = 0;
    uint32_t step = 0u;
    if (ownId < numExperts)
    {
        ordinal = before[ownId] + ownLaneOrdinal;
        for (int v = 0; v < warp; ++v)
        {
            ordinal += warpCount[v * numExperts + ownId];
        }
        step = pointStep(total[ownId]);
    }
    storeRoute(f, count,
        weightedSlot(
            ownExpert, ordinal, step, block, epRank, counts, starts, slots, cumulativeShares, numExperts, numSlots),
        out, staging);
}

// One block counts all the ids and updates the statistics.
template <typename IdT>
__global__ void __launch_bounds__(kRecordThreads)
    adaptiveEplbRecordKernel(IdT const* __restrict__ ids, int64_t count, int numExperts, float* __restrict__ rows,
        int64_t* __restrict__ forwards, int const* __restrict__ enabled, bool first, int const* liveRowWord, int topK)
{
    extern __shared__ int histogram[];
    if (countIds(ids, 0, liveIdCount(liveRowWord, count, topK), numExperts, enabled, histogram))
    {
        addCounts(histogram, numExperts, rows, forwards, first);
    }
}

} // namespace

template <typename IdT, typename OutT>
void adaptiveEplbRoute(IdT const* ids, int64_t count, int epRank, AdaptiveEplbRouteTables tables, OutT* out,
    AdaptiveEplbStaging staging, AdaptiveEplbStats stats, cudaStream_t stream)
{
    int64_t const extent = staging.weightsOut != nullptr ? staging.total : count;
    auto blocks = static_cast<unsigned>((extent + kTileThreads - 1) / kTileThreads) + (stats.rows != nullptr ? 1u : 0u);
    if (blocks > 0)
    {
        size_t const bytes = routeSharedBytes(tables.numExperts, tables.numSlots);
        allowSharedBytes<adaptiveEplbRouteKernel<IdT, OutT>>(bytes);
        common::launchWithPdlWhenEnabled("adaptiveEplbRoute", adaptiveEplbRouteKernel<IdT, OutT>, dim3(blocks),
            dim3(kTileThreads), bytes, stream, ids, count, epRank, tables, out, staging, stats);
    }
}

template <typename IdT>
void adaptiveEplbRecord(IdT const* ids, int64_t count, int numExperts, float* rows, int64_t* forwards,
    int const* enabled, bool first, int const* liveRowWord, int topK, cudaStream_t stream)
{
    size_t const bytes = static_cast<size_t>(numExperts) * sizeof(int);
    allowSharedBytes<adaptiveEplbRecordKernel<IdT>>(bytes);
    adaptiveEplbRecordKernel<IdT><<<1, kRecordThreads, bytes, stream>>>(
        ids, count, numExperts, rows, forwards, enabled, first, liveRowWord, topK);
}

#define INSTANTIATE_ADAPTIVE_EPLB_ROUTE(IdT, OutT)                                                                     \
    template void adaptiveEplbRoute<IdT, OutT>(IdT const*, int64_t, int, AdaptiveEplbRouteTables, OutT*,               \
        AdaptiveEplbStaging, AdaptiveEplbStats, cudaStream_t);

INSTANTIATE_ADAPTIVE_EPLB_ROUTE(int32_t, int32_t)
INSTANTIATE_ADAPTIVE_EPLB_ROUTE(int32_t, int64_t)
INSTANTIATE_ADAPTIVE_EPLB_ROUTE(int64_t, int32_t)
INSTANTIATE_ADAPTIVE_EPLB_ROUTE(int64_t, int64_t)

#define INSTANTIATE_ADAPTIVE_EPLB_RECORD(IdT)                                                                          \
    template void adaptiveEplbRecord<IdT>(                                                                             \
        IdT const*, int64_t, int, float*, int64_t*, int const*, bool, int const*, int, cudaStream_t);

INSTANTIATE_ADAPTIVE_EPLB_RECORD(int32_t)
INSTANTIATE_ADAPTIVE_EPLB_RECORD(int64_t)

#undef INSTANTIATE_ADAPTIVE_EPLB_ROUTE
#undef INSTANTIATE_ADAPTIVE_EPLB_RECORD

} // namespace kernels

TRTLLM_NAMESPACE_END
