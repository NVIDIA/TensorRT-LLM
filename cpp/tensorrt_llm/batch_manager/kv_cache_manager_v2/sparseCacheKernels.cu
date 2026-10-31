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

#include "sparseCacheKernels.h"

#include "exceptions.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"

#include <cassert>

#include <cub/cub.cuh>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{

//! Enforce a precondition that correct use cannot violate. A debug build names the condition.
#define KVCM2_SPARSE_REQUIRE(cond)                                                                                     \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(cond))                                                                                                   \
        {                                                                                                              \
            assert(!"KVCM2 sparse invariant violated: " #cond);                                                        \
            __trap();                                                                                                  \
        }                                                                                                              \
    } while (0)

//! Threads a copy, out-index or flush-list CTA takes. The resolve is templated on its own
//! width instead; see resolveThreads().
constexpr int kThreads = 256;
//! The two widths the resolve is instantiated at. Nothing wider was ever faster at any
//! shape measured, and the narrow one must still cover the eviction histogram.
constexpr int kResolveThreadsNarrow = 256;
constexpr int kResolveThreadsWide = 512;
constexpr uint32_t kWarpSize = 32;
constexpr int32_t kWarpsPerCopyCta = kThreads / static_cast<int>(kWarpSize);
//! kBadPageIndex as a plain int32, since device code cannot use the strong typedef.
constexpr int32_t kBadPageIndexValue = kBadPageIndex.value();

//! A lease's metadata block: one SparseEntry word per entry.
//!
//! Every lease of a life cycle is the same size, so the block is arithmetic from the
//! arena's base -- the same addressing cells already use, which is why a row publishes an
//! index rather than the pointers it would imply.
__device__ inline uint32_t* leaseMetadata(SparseArenaView const& arena, SparseLeaseIndex lease)
{
    return reinterpret_cast<uint32_t*>(arena.metadataBase + static_cast<uint64_t>(lease.value()) * arena.leaseStride);
}

//! What the resolve knows about an entry while it is choosing victims.
constexpr uint8_t kFlagEvictable = 0; //!< Not selected this step and holding no in-flight unit.
constexpr uint8_t kFlagMatched = 1;   //!< Selected this step, so its unit stays and its age resets.
constexpr uint8_t kFlagHeld = 2;      //!< Holding in-flight data, which only this entry has.

//! Buckets the eviction histogram takes, which is the whole range of SparseEntry::staleness.
constexpr int32_t kAgeBuckets = 256;
//! The bucket a free entry falls in. No resident entry reaches it, so the free ones are
//! always given away before any unit is evicted.
constexpr int32_t kStalest = kAgeBuckets - 1;

using BlockScan = cub::BlockScan<int32_t, kThreads>;

__host__ __device__ inline size_t alignUp16(size_t bytes) noexcept
{
    return (bytes + 15U) & ~static_cast<size_t>(15U);
}

//! Byte offsets of the shared-memory regions one resolve CTA carves. Computed identically
//! on the host, which sizes the launch, and in the kernel, which carves it. Every region is
//! 16-byte aligned: flag is a byte array, so an unaligned int32 region after it faults.
struct SmemLayout
{
    size_t table = 0;
    size_t entry = 0;
    size_t flag = 0;
    size_t missIdx = 0;
    size_t missOrd = 0;
    size_t hist = 0;
    size_t total = 0;
};

//! @param required Length of the virtual list the probe walks: in-flight units first,
//! then the selection.
__host__ __device__ inline SmemLayout smemLayout(int32_t tableSize, int32_t entries, int32_t required) noexcept
{
    SmemLayout layout;
    size_t offset = 0;
    layout.table = offset;
    offset = alignUp16(offset + sizeof(uint32_t) * static_cast<size_t>(tableSize));
    layout.entry = offset;
    offset = alignUp16(offset + sizeof(uint32_t) * static_cast<size_t>(entries));
    layout.flag = offset;
    offset = alignUp16(offset + sizeof(uint8_t) * static_cast<size_t>(entries));
    layout.missIdx = offset;
    offset = alignUp16(offset + sizeof(int32_t) * static_cast<size_t>(required));
    layout.missOrd = offset;
    offset = alignUp16(offset + sizeof(int32_t) * static_cast<size_t>(required));
    layout.hist = offset;
    offset = alignUp16(offset + sizeof(int32_t) * static_cast<size_t>(kAgeBuckets));
    layout.total = offset;
    return layout;
}

[[nodiscard]] constexpr int32_t nextPow2(int32_t value) noexcept
{
    int32_t result = 1;
    while (result < value)
    {
        result <<= 1;
    }
    return result;
}

//! Programmatic dependent launch, where the hardware has it.
//!
//! Both calls are hints, not contracts: a wait with no launch dependency returns at once,
//! and a release with no dependent kernel does nothing. So a kernel carrying them is correct
//! whether or not its launch enrolled in PDL, and compiling them out below Hopper only falls
//! back to ordinary stream ordering.
__device__ inline void gridDependencyWait()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaGridDependencySynchronize();
#endif
}

//! Let a dependent grid start. Only sound once everything it reads from this one is written.
__device__ inline void gridDependencyRelease()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

//! Fibonacci hashing. Unit ordinals are small and dense, and identity hashing turns a dense
//! run into one long probe chain.
__device__ inline uint32_t slotOf(int32_t key, uint32_t mask) noexcept
{
    return (static_cast<uint32_t>(key) * 0x9E3779B1U) & mask;
}

//! Bits of a probe-index position given to the entry holding the keyed ordinal, leaving the
//! rest for the ordinal itself. An ordinal of every one bit would be indistinguishable from
//! an empty position, and a slot of every one bit is what "keyed, but not resident" is, so
//! neither field reaches its own maximum.
constexpr uint32_t kTableSlotBits = 14;
constexpr uint32_t kTableSlotMask = (1U << kTableSlotBits) - 1U;
constexpr uint32_t kTableEmpty = 0xFFFFFFFFU;
constexpr int32_t kMaxTableOrdinal = static_cast<int32_t>((1U << (32 - kTableSlotBits)) - 2U);
constexpr int32_t kMaxTableSlot = static_cast<int32_t>(kTableSlotMask) - 1;

//! The resolve's open-addressed probe index, one word per position.
//!
//! Packed rather than two arrays because every pass over it is a shared-memory round trip
//! per probe step, and the publish pass walks the whole resident set: halving the bytes a
//! position takes halves the traffic and leaves twice the index in one cache line.
//!
//! Insert races; publish and lookup do not, because they run between syncs that separate
//! them from the inserts. A position is written once by insert and at most once by publish,
//! each by the single thread owning the ordinal.
struct QueryTable
{
    uint32_t* word;

    __device__ explicit QueryTable(unsigned char* base)
        : word(reinterpret_cast<uint32_t*>(base))
    {
    }

    __device__ void clear(int32_t position) const
    {
        word[position] = kTableEmpty;
    }

    __device__ uint32_t find(int32_t ordinal, uint32_t mask, uint32_t& position) const
    {
        position = slotOf(ordinal, mask);
        uint32_t current = word[position];
        while (current != kTableEmpty && (current >> kTableSlotBits) != static_cast<uint32_t>(ordinal))
        {
            position = (position + 1U) & mask;
            current = word[position];
        }
        return current;
    }

    //! Key `ordinal`, with no entry named yet.
    __device__ void insert(int32_t ordinal, uint32_t mask) const
    {
        uint32_t const want = (static_cast<uint32_t>(ordinal) << kTableSlotBits) | kTableSlotMask;
        uint32_t position = slotOf(ordinal, mask);
        uint32_t previous = atomicCAS(&word[position], kTableEmpty, want);
        while (previous != kTableEmpty && (previous >> kTableSlotBits) != static_cast<uint32_t>(ordinal))
        {
            position = (position + 1U) & mask;
            previous = atomicCAS(&word[position], kTableEmpty, want);
        }
    }

    //! Name `slot` as the entry holding `ordinal`, if the ordinal was keyed. Returns whether
    //! it was, which is what says the entry is selected this step.
    __device__ bool publish(int32_t ordinal, int32_t slot, uint32_t mask) const
    {
        uint32_t position = 0;
        if (find(ordinal, mask, position) == kTableEmpty)
        {
            return false;
        }
        word[position] = (static_cast<uint32_t>(ordinal) << kTableSlotBits) | static_cast<uint32_t>(slot);
        return true;
    }

    //! The entry holding `ordinal`, or kNoSparseSlot when nothing does.
    __device__ int32_t lookup(int32_t ordinal, uint32_t mask) const
    {
        uint32_t position = 0;
        uint32_t const current = find(ordinal, mask, position);
        if (current == kTableEmpty)
        {
            return kNoSparseSlot;
        }
        uint32_t const slot = current & kTableSlotMask;
        return slot == kTableSlotMask ? kNoSparseSlot : static_cast<int32_t>(slot);
    }
};

//! Bits of the prepare-write reduction key given to the entry index, the rest to the
//! staleness plus one -- so that zero is a key no candidate produces and can stand for a
//! block that found none.
constexpr uint32_t kVictimSlotBits = 23;
constexpr uint32_t kVictimSlotMask = (1U << kVictimSlotBits) - 1U;

__device__ inline uint32_t victimKey(uint32_t staleness, int32_t slot) noexcept
{
    return ((staleness + 1U) << kVictimSlotBits) | static_cast<uint32_t>(slot);
}

//! Reserve one in-flight unit without building a selection probe index.
__global__ __launch_bounds__(kThreads) void sparsePrepareWriteKernel(SparseResolveArgs args)
{
    int32_t const row = static_cast<int32_t>(blockIdx.x);
    SparseLeaseIndex const lease = args.leases[row];
    if (lease == kNoSparseLease)
    {
        return;
    }

    gridDependencyWait();
    int32_t const historyUnits = args.historyTokens[row] / args.unitTokens;
    int32_t const inFlight = args.capacityTokens[row] / args.unitTokens - historyUnits;
    KVCM2_SPARSE_REQUIRE(inFlight >= 0 && inFlight <= 1);
    // The same bound the resolve's probe index imposes, since this ordinal is resolved
    // through it on every later step.
    KVCM2_SPARSE_REQUIRE(historyUnits <= kMaxTableOrdinal);
    if (threadIdx.x == 0)
    {
        args.missCount[row] = 0;
        args.inFlightSlot[row] = kNoSparseSlot;
    }
    if (inFlight == 0)
    {
        return;
    }

    int32_t const entries = args.arena.entriesPerSelection;
    auto const stepAge = static_cast<uint32_t>(*args.stepAge);
    bool const rebase = SparseEntry::rebaseDue(stepAge);
    auto* entry = leaseMetadata(args.arena, lease) + static_cast<ptrdiff_t>(args.selectionSlot) * entries;
    using Reduce = cub::BlockReduce<uint32_t, kThreads>;
    __shared__ typename Reduce::TempStorage temp;

    // The entry already holding the in-flight unit, if one does, and the stalest of those
    // that could be given to it otherwise -- free entries first, then history by age. An
    // entry above the watermark is another unit's only copy and is never a candidate.
    //
    // Both reductions take a key that is zero where a thread saw no candidate, which is
    // below every key one did produce.
    uint32_t match = 0;
    uint32_t stalest = 0;
    for (int32_t s = static_cast<int32_t>(threadIdx.x); s < entries; s += kThreads)
    {
        uint32_t word = entry[s];
        if (rebase)
        {
            word = SparseEntry::rebase(word, stepAge);
            entry[s] = word;
        }
        int32_t const ordinal = SparseEntry::ordinal(word);
        if (ordinal == historyUnits)
        {
            match = static_cast<uint32_t>(s) + 1U;
        }
        else if (ordinal < historyUnits)
        {
            auto const staleness
                = ordinal < 0 ? static_cast<uint32_t>(kStalest) : SparseEntry::staleness(word, stepAge);
            stalest = max(stalest, victimKey(staleness, s));
        }
    }
    uint32_t const victim = Reduce(temp).Reduce(stalest, cuda::maximum<>{});
    __syncthreads();
    uint32_t const matched = Reduce(temp).Reduce(match, cuda::maximum<>{});
    if (threadIdx.x == 0)
    {
        // The reservation covers a full selection plus every in-flight unit, so a candidate
        // exists whenever the unit is not already held.
        KVCM2_SPARSE_REQUIRE(matched != 0 || victim != 0);
        auto const slot = static_cast<int32_t>(matched != 0 ? matched - 1U : (victim & kVictimSlotMask));
        entry[slot]
            = SparseEntry::pack(SparseEntry::usedByte(stepAge, matched != 0), static_cast<uint32_t>(historyUnits));
        args.inFlightSlot[row] = slot;
    }
}

//! One CTA per request row of the resolved selection.
//!
//! Shared memory is carved by hand because the probe index is sized at launch; see
//! SmemLayout for the regions and their alignment.
//!
//! `kResolveThreads` is chosen per shape by resolveThreads(): every pass but the probe is a
//! stride loop over the resident set, so a wider CTA shortens the kernel wherever a
//! multiprocessor has the room to hold more warps of it.
template <int kResolveThreads>
__global__ __launch_bounds__(kResolveThreads) void sparseResolveKernel(SparseResolveArgs args, int32_t tableSize)
{
    // The threshold search below gives one thread to each histogram bucket.
    static_assert(kResolveThreads >= kAgeBuckets, "A resolve CTA must cover the eviction histogram");
    using Scan = cub::BlockScan<int32_t, kResolveThreads>;
    extern __shared__ unsigned char smem[];
    int32_t const selection = args.selectionSlot;
    int32_t const row = static_cast<int32_t>(blockIdx.x);
    SparseLeaseIndex const lease = args.leases[row];
    if (lease == kNoSparseLease)
    {
        // A free row, or a member that has suspended and given its reservation back. Its
        // outputs keep whatever the caller initialised them to.
        return;
    }
    int32_t const entries = args.arena.entriesPerSelection;
    int32_t const width = args.selectionWidth;
    uint32_t const mask = static_cast<uint32_t>(tableSize) - 1U;
    int32_t const tid = static_cast<int32_t>(threadIdx.x);
    auto const stepAge = static_cast<uint32_t>(*args.stepAge);
    bool const rebase = SparseEntry::rebaseDue(stepAge);

    int32_t const historyUnits = args.historyTokens[row] / args.unitTokens;
    int32_t const inFlight = args.capacityTokens[row] / args.unitTokens - historyUnits;
    // resize() rejects a gap wider than the cache can pin, so the check cannot fire.
    KVCM2_SPARSE_REQUIRE(inFlight >= 0 && inFlight <= args.maxInFlightUnits);
    // The probe walks one virtual list: the in-flight units, which must be resident before
    // the selection takes any victim, then the selection itself.
    int32_t const required = inFlight + width;

    SmemLayout const layout = smemLayout(tableSize, entries, args.maxInFlightUnits + width);
    QueryTable const table(smem + layout.table);
    auto* entryCache = reinterpret_cast<uint32_t*>(smem + layout.entry);
    auto* flag = reinterpret_cast<uint8_t*>(smem + layout.flag);
    auto* missIdx = reinterpret_cast<int32_t*>(smem + layout.missIdx);
    auto* missOrd = reinterpret_cast<int32_t*>(smem + layout.missOrd);
    auto* hist = reinterpret_cast<int32_t*>(smem + layout.hist);

    __shared__ typename Scan::TempStorage temp;
    __shared__ int32_t numMiss;
    __shared__ int32_t numCopy;
    __shared__ int32_t numResolved;
    __shared__ int32_t copyBase;
    // The staleness at which the misses run out, and how many of the entries holding
    // exactly that staleness are wanted. A threshold past the histogram evicts nothing.
    __shared__ int32_t thresholdStaleness;
    __shared__ int32_t neededAtThreshold;
    __shared__ int32_t takenAtThreshold;
    __shared__ int32_t victimCounter;

    // The lease's own words belong to this row's allocation, so only the selection indexes
    // them.
    uint32_t* entry = leaseMetadata(args.arena, lease) + static_cast<ptrdiff_t>(selection) * entries;
    // The batch-wide arrays are indexed by row alone: the call is one selection, and every
    // buffer resolving within it shares the single result.
    auto const selectionBase = static_cast<ptrdiff_t>(row) * width;
    auto const inFlightBase = static_cast<ptrdiff_t>(row) * args.maxInFlightUnits;
    int32_t const* selected = args.selected + selectionBase;
    int32_t* outSlot = args.outSlot + selectionBase;
    int32_t* inFlightSlot = args.inFlightSlot + inFlightBase;

    if (tid == 0)
    {
        numMiss = 0;
        numCopy = 0;
        numResolved = 0;
        thresholdStaleness = kAgeBuckets;
        neededAtThreshold = 0;
        takenAtThreshold = 0;
        victimCounter = 0;
    }
    for (int32_t i = tid; i < tableSize; i += kResolveThreads)
    {
        table.clear(i);
    }
    for (int32_t b = tid; b < kAgeBuckets; b += kResolveThreads)
    {
        hist[b] = 0;
    }
    __syncthreads();

    // The selection may be produced by the preceding grid. Only shared-memory
    // initialization is independent of that producer.
    gridDependencyWait();

    // Index the required ordinals, then match the resident set against that index.
    // The index size follows the selection width rather than the cache capacity.
    for (int32_t i = tid; i < required; i += kResolveThreads)
    {
        int32_t const ord = i < inFlight ? historyUnits + i : selected[i - inFlight];
        if (ord < 0)
        {
            continue;
        }
        // A position of the index holds the ordinal beside the entry naming it, so an
        // ordinal past this would be truncated into a different one.
        KVCM2_SPARSE_REQUIRE(ord <= kMaxTableOrdinal);
        table.insert(ord, mask);
    }

    // Slots past the in-flight span say "no unit in flight", so a consumer can read this
    // array without also being told how many units there are.
    for (int32_t i = tid + inFlight; i < args.maxInFlightUnits; i += kResolveThreads)
    {
        inFlightSlot[i] = kNoSparseSlot;
    }
    __syncthreads();

    // Classify the resident set against the index, and build the eviction histogram over
    // what the step may give away. The words land in shared memory on the way past: two
    // passes below read them again, and each read would otherwise cross to global memory.
    for (int32_t s = tid; s < entries; s += kResolveThreads)
    {
        uint32_t word = entry[s];
        if (rebase)
        {
            word = SparseEntry::rebase(word, stepAge);
            entry[s] = word;
        }
        entryCache[s] = word;
        int32_t const ordinal = SparseEntry::ordinal(word);
        bool const matched = ordinal >= 0 && table.publish(ordinal, s, mask);
        bool const held = ordinal >= historyUnits;
        uint8_t const cls = matched ? kFlagMatched : (held ? kFlagHeld : kFlagEvictable);
        flag[s] = cls;
        if (cls == kFlagEvictable)
        {
            auto const staleness
                = ordinal < 0 ? static_cast<uint32_t>(kStalest) : SparseEntry::staleness(word, stepAge);
            atomicAdd(&hist[staleness], 1);
        }
    }
    __syncthreads();

    // Probe once per required unit. A hit resolves immediately; a miss is recorded against
    // its position in the virtual list and resolved once a victim is chosen.
    int32_t resolvedMine = 0;
    for (int32_t i = tid; i < required; i += kResolveThreads)
    {
        int32_t const ord = i < inFlight ? historyUnits + i : selected[i - inFlight];
        if (ord < 0)
        {
            outSlot[i - inFlight] = kNoSparseSlot;
            continue;
        }
        resolvedMine += i < inFlight ? 0 : 1;
        int32_t const slot = table.lookup(ord, mask);
        if (slot != kNoSparseSlot)
        {
            (i < inFlight ? inFlightSlot[i] : outSlot[i - inFlight]) = slot;
        }
        else
        {
            int32_t const j = atomicAdd(&numMiss, 1);
            missOrd[j] = ord;
            missIdx[j] = i;
        }
    }
    if (resolvedMine > 0)
    {
        atomicAdd(&numResolved, resolvedMine);
    }
    __syncthreads();

    // Claim this row's run in the compacted list now that the miss set is final, so the
    // eviction loop below can write straight into it. Only a miss below the watermark is
    // copied, which is the same test the loop applies, so counting it here costs one pass
    // over a list already in shared memory and saves staging the entries anywhere.
    {
        int32_t mine = 0;
        for (int32_t j = tid; j < numMiss; j += kResolveThreads)
        {
            mine += missOrd[j] < historyUnits ? 1 : 0;
        }
        int32_t scanned = 0;
        int32_t claimed = 0;
        Scan(temp).ExclusiveSum(mine, scanned, claimed);
        if (tid == 0)
        {
            copyBase = claimed > 0 ? atomicAdd(args.work.total, claimed) : 0;
            // The list is sized for every row missing its whole selection, so a claim can
            // only run past it if the caller sized it from a different width.
            KVCM2_SPARSE_REQUIRE(copyBase + claimed <= args.work.capacity);
            if (args.hitMiss != nullptr)
            {
                atomicAdd(&args.hitMiss->resolved, static_cast<unsigned long long>(numResolved));
                atomicAdd(&args.hitMiss->misses, static_cast<unsigned long long>(claimed));
            }
        }
    }
    __syncthreads();

    // Where the misses run out, walking the histogram down from its stalest bucket. One
    // thread per bucket, so the scan that totals them is the whole search.
    {
        int32_t const bucket = tid < kAgeBuckets ? hist[kStalest - tid] : 0;
        int32_t inclusive = 0;
        Scan(temp).InclusiveSum(bucket, inclusive);
        int32_t const before = inclusive - bucket;
        if (tid < kAgeBuckets && numMiss > 0 && before < numMiss && inclusive >= numMiss)
        {
            thresholdStaleness = kStalest - tid;
            neededAtThreshold = numMiss - before;
        }
    }
    __syncthreads();
    // The reservation covers a full selection plus every in-flight unit, and the in-flight
    // ones are resolved first, so there is always a cell for each remaining miss.
    KVCM2_SPARSE_REQUIRE(numMiss == 0 || thresholdStaleness < kAgeBuckets);

    // Give the misses the stalest entries. Everything staler than the threshold goes; the
    // bucket at it holds more entries than are left to fill, so those race for the places.
    for (int32_t s = tid; s < entries; s += kResolveThreads)
    {
        uint32_t const word = entryCache[s];
        uint8_t const cls = flag[s];
        if (cls == kFlagMatched)
        {
            entry[s] = SparseEntry::pack(SparseEntry::usedByte(stepAge, true), word);
            continue;
        }
        if (cls != kFlagEvictable)
        {
            continue;
        }
        auto const staleness = static_cast<int32_t>(
            SparseEntry::ordinal(word) < 0 ? static_cast<uint32_t>(kStalest) : SparseEntry::staleness(word, stepAge));
        bool victim = staleness > thresholdStaleness;
        if (!victim && staleness == thresholdStaleness)
        {
            victim = atomicAdd(&takenAtThreshold, 1) < neededAtThreshold;
        }
        if (!victim)
        {
            continue;
        }
        int32_t const j = atomicAdd(&victimCounter, 1);
        int32_t const ordinal = missOrd[j];
        int32_t const position = missIdx[j];
        entry[s] = SparseEntry::pack(SparseEntry::usedByte(stepAge, false), static_cast<uint32_t>(ordinal));
        (position < inFlight ? inFlightSlot[position] : outSlot[position - inFlight]) = s;
        // Only history has anything to fetch. An in-flight unit's host page is not
        // written until the watermark passes it, so copying from it would read stale
        // bytes over the data the model is about to write.
        if (ordinal < historyUnits)
        {
            auto const c = copyBase + atomicAdd(&numCopy, 1);
            args.work.row[c] = row;
            args.work.ordinal[c] = ordinal;
            args.work.slot[c] = s;
        }
    }
    __syncthreads();

    // The work list and the entry indices are final, so a copy may start.
    gridDependencyRelease();

    if (tid == 0)
    {
        args.missCount[row] = numCopy;
    }
}

//! Copy `bytes` from `src` to `dst` with one warp, `sizeof(Vec)` at a time.
//!
//! The width is the launcher's choice, made from the layout and the addresses it produces,
//! so there is no alignment test here and no tail: `bytes` divides by `sizeof(Vec)` and both
//! pointers are aligned to it.
template <typename Vec>
__device__ inline void copyChunkWarp(char* dst, char const* src, uint32_t bytes, uint32_t lane)
{
    auto* out = reinterpret_cast<Vec*>(dst);
    auto const* in = reinterpret_cast<Vec const*>(src);
    uint32_t const count = bytes / static_cast<uint32_t>(sizeof(Vec));
    for (uint32_t i = lane; i < count; i += kWarpSize)
    {
        out[i] = in[i];
    }
}

//! One pool's work, with a vector width fixed at compile time.
//!
//! Split from the kernel so the width is switched on once per block: one launch covers
//! every pool, so there is no per-pool launch to template.
template <typename Vec>
__device__ void copySparseUnits(SparseUnitCopyArgs const& args, SparseCopyDesc const& desc, int32_t list, int32_t first)
{
    uint32_t const lane = threadIdx.x % kWarpSize;
    int32_t const stride = static_cast<int32_t>(gridDim.x) * kWarpsPerCopyCta;
    auto const listBase = static_cast<ptrdiff_t>(list) * args.work.capacity;
    int32_t const count = args.work.total[list];

    for (int32_t j = first; j < count; j += stride)
    {
        auto const index = listBase + j;
        int32_t const row = args.work.row[index];
        int32_t const ordinal = args.work.ordinal[index];
        int32_t const slot = args.work.slot[index];

        // Read per item rather than per warp: the list is dense across rows, so consecutive
        // entries need not share one. A producer's CTA claims a contiguous run, so in
        // practice a warp's items mostly do.
        SparseLeaseIndex const lease = args.leases[row];

        // A unit ordinal names a unit within a block; a cache entry names a unit within a
        // sub-slot. Both divide by the same units-per-page.
        int32_t const block = ordinal / args.unitsPerPage;
        int32_t const srcUnit = ordinal % args.unitsPerPage;
        int32_t const dstUnit = slot % args.unitsPerPage;
        if (block >= static_cast<int32_t>(desc.numBlocks))
        {
            // Outside the extent this call was given; the warp's other items may not be.
            continue;
        }

        // The descriptor's pointer already stands at this buffer's entry, so the table
        // index is the row's run of blocks and nothing else.
        auto const hostPage
            = static_cast<uint64_t>(desc.pageTable[static_cast<ptrdiff_t>(row) * desc.pageTableStride + block]);

        // The cold side is reached by ordinal through the page table, the hot side by
        // arithmetic from the lease: the arena hands out a contiguous run, so there is no
        // page to look up, and the buffer's own slot in the pool is already in deviceBase.
        // Direction only chooses which of the two is read.
        auto* cold = reinterpret_cast<char*>(
            desc.hostBase + hostPage * desc.hostPageStride + static_cast<uint64_t>(srcUnit) * desc.layout.unitStride);
        auto* hot = reinterpret_cast<char*>(desc.deviceBase + static_cast<uint64_t>(lease.value()) * desc.leaseStride
            + static_cast<uint64_t>(slot / args.unitsPerPage) * desc.cellBytes
            + static_cast<uint64_t>(dstUnit) * desc.layout.unitStride);
        bool const fetching = args.direction == SparseCopyDirection::kFetch;
        char const* src = fetching ? cold : hot;
        char* dst = fetching ? hot : cold;

        for (uint32_t c = 0; c < desc.layout.numChunks; ++c)
        {
            auto const offset = static_cast<uint64_t>(c) * desc.layout.chunkStride;
            copyChunkWarp<Vec>(dst + offset, src + offset, desc.layout.chunkBytes, lane);
        }
    }
}

//! One warp per work item, walking the compacted list with a grid-stride loop. `blockIdx.y`
//! is the buffer, `blockIdx.x` the work.
__global__ __launch_bounds__(kThreads) void sparseUnitCopyKernel(SparseUnitCopyArgs args)
{
    // Broadcast from lane zero before anything can diverge: threadIdx.x varies within the
    // warp, so the compiler cannot otherwise see that this quotient does not, and every lane
    // must still be present for the shuffle.
    uint32_t warpInCta = threadIdx.x / kWarpSize;
    warpInCta = __shfl_sync(0xffffffffU, warpInCta, 0);
    int32_t const globalWarp = static_cast<int32_t>(blockIdx.x) * kWarpsPerCopyCta + static_cast<int32_t>(warpInCta);

    // Everything down to the dependency wait is the producer's business only in that it
    // names where to put the result: the descriptors are built once with the Batch, and the
    // buffer and list follow from the block's own coordinates. Resolving it here is what the
    // launch dependency is for -- the block is resident and addressed before the kernel it
    // waits on has finished.
    int32_t const buffer = args.bufferBase + static_cast<int32_t>(blockIdx.y);
    SparseCopyDesc const desc = args.copyDesc[buffer];
    // The buffer is this block's, so its layer -- and therefore its list -- is uniform and
    // known before any entry is read. A buffer's layer is dense from zero within its group,
    // and a multi-list caller has one list per layer of that group in the same order, so the
    // layer is the list. The index is zero whenever the caller passed one list, which is what
    // a fetch does: it resolves a single layer.
    int32_t const list = args.work.numLists > 1 ? args.arena.bufferLayer[buffer] : 0;

    // The work list and its total are the producer's, so this is as far as the block gets
    // before the kernel that fills them has run.
    gridDependencyWait();

    // Nothing launched after this reads a cell, so a successor need not wait for the copy
    // itself -- only for the block to have got this far. Releasing here is what keeps the
    // index kernel that follows on the stream from being ordered behind a copy it does not
    // depend on.
    gridDependencyRelease();

    switch (desc.width)
    {
    case SparseCopyWidth::kWord4: copySparseUnits<uint4>(args, desc, list, globalWarp); break;
    case SparseCopyWidth::kWord2: copySparseUnits<uint2>(args, desc, list, globalWarp); break;
    case SparseCopyWidth::kWord: copySparseUnits<uint32_t>(args, desc, list, globalWarp); break;
    case SparseCopyWidth::kByte: copySparseUnits<char>(args, desc, list, globalWarp); break;
    }
}

//! One CTA per (row, buffer).
__global__ __launch_bounds__(kThreads) void sparseOutIndexKernel(SparseOutIndexArgs args)
{
    int32_t const row = static_cast<int32_t>(blockIdx.y);
    int32_t const buffer = args.bufferBase + static_cast<int32_t>(blockIdx.x);
    auto* out = args.out
        + (static_cast<ptrdiff_t>(row) * args.outBuffers + args.outBase + blockIdx.x)
            * static_cast<ptrdiff_t>(args.selectionWidth);

    SparseLeaseIndex const lease = args.leases[row];
    bool const reserved = lease != kNoSparseLease;
    auto const base = static_cast<ptrdiff_t>(row) * args.selectionWidth;
    int32_t const slotStride = args.outSlotStride > 0 ? args.outSlotStride : args.selectionWidth;
    auto const slotBase = static_cast<ptrdiff_t>(row) * slotStride;

    // Every slot is written, so a stale index from a previous step can never be read back.
    // One pass does it, because the output is the selection's own shape: a padded slot and
    // an unreserved row both just take the sentinel here rather than needing a prior fill.
    for (int32_t i = static_cast<int32_t>(threadIdx.x); i < args.selectionWidth; i += kThreads)
    {
        int32_t const slot = reserved ? args.outSlot[slotBase + i] : kNoSparseSlot;
        // Without a selection array the slot alone says whether the entry is live, which is
        // what the write-index path passes: its in-flight list is already self-describing.
        if ((args.selected != nullptr && args.selected[base + i] < 0) || slot < 0)
        {
            out[i] = kBadPageIndexValue;
            continue;
        }

        int32_t const cell = lease.value() * args.cellsPerLease
            + args.arena.bufferSlotInPool[buffer] * args.cellsPerBuffer + slot / args.unitsPerPage;
        out[i] = cell * args.unitsPerPage + slot % args.unitsPerPage;
    }
}

//! One CTA per (layer, row). The units wanted form a contiguous ordinal range, so a scan of the
//! reverse map finds them without a probe index.
__global__ __launch_bounds__(kThreads) void sparseFlushListKernel(SparseFlushListArgs args)
{
    int32_t const row = static_cast<int32_t>(blockIdx.y);
    SparseLeaseIndex const lease = args.leases[row];
    if (lease == kNoSparseLease)
    {
        // A free row, or a member that has suspended: no reservation to write back from.
        return;
    }
    int32_t const layer = static_cast<int32_t>(blockIdx.x);
    int32_t const entries = args.arena.entriesPerSelection;

    // The row's view is staged with the watermarks and is nobody's output here, so a block
    // can address itself before waiting. The range itself is uploaded by the call that asked
    // for this write-back, so reading it is where the block has to stop.
    gridDependencyWait();

    int32_t const from = args.fromUnits[row];
    int32_t const to = args.toUnits[row];
    if (from >= to)
    {
        // This row's watermark did not move, so nothing of its crossed, but the grid is
        // sized for the widest row.
        if (threadIdx.x == 0)
        {
            args.outCount[static_cast<ptrdiff_t>(row) * args.arena.numLayers + layer] = 0;
        }
        return;
    }

    __shared__ typename BlockScan::TempStorage temp;
    // The entries scanned are the layer's selection's, which several layers may share; the
    // list claimed below is the layer's own, because the copy that consumes it walks the
    // layer's buffers and each holds its own mirror of a shared entry.
    uint32_t const* entry
        = leaseMetadata(args.arena, lease) + static_cast<ptrdiff_t>(args.arena.layerSelection[layer]) * entries;
    auto const ordinalAt = [&](int32_t slot) { return SparseEntry::ordinal(entry[slot]); };
    auto const crossed = [&](int32_t slot)
    {
        int32_t const ord = ordinalAt(slot);
        return ord >= from && ord < to;
    };

    // Two passes over the resident set, because the run cannot be claimed until its length
    // is known and the entries are too many to stage in shared memory: a prefilling row
    // crosses as many units as its prompt holds. The set is the per-request cache, which is
    // small enough that scanning it twice costs less than staging it once.
    __shared__ int32_t count;
    __shared__ int32_t base;
    if (threadIdx.x == 0)
    {
        count = 0;
    }
    __syncthreads();

    {
        int32_t mine = 0;
        for (int32_t s = static_cast<int32_t>(threadIdx.x); s < entries; s += kThreads)
        {
            mine += crossed(s) ? 1 : 0;
        }
        int32_t scanned = 0;
        int32_t claimed = 0;
        BlockScan(temp).ExclusiveSum(mine, scanned, claimed);
        if (threadIdx.x == 0)
        {
            base = claimed > 0 ? atomicAdd(args.work.total + layer, claimed) : 0;
            // Sized from the step's token budget, which bounds how far every row's watermark
            // can move together. Running past it means the budget the Batch was given is not
            // the one the caller honoured.
            KVCM2_SPARSE_REQUIRE(base + claimed <= args.work.capacity);
            args.outCount[static_cast<ptrdiff_t>(row) * args.arena.numLayers + layer] = claimed;
        }
    }
    __syncthreads();

    auto const listBase = static_cast<ptrdiff_t>(layer) * args.work.capacity + base;
    for (int32_t s = static_cast<int32_t>(threadIdx.x); s < entries; s += kThreads)
    {
        if (!crossed(s))
        {
            continue;
        }
        auto const j = listBase + atomicAdd(&count, 1);
        args.work.row[j] = row;
        args.work.ordinal[j] = ordinalAt(s);
        args.work.slot[j] = s;
    }

    // The list is the last thing this kernel writes, so the release only brings the copy's
    // launch forward.
    __syncthreads();
    gridDependencyRelease();
}

} // namespace

//! Throw unless the work list is fully described.
void checkWorkList(SparseWorkList const& work, char const* api)
{
    TLLM_CHECK_WITH_INFO(
        work.row != nullptr && work.ordinal != nullptr && work.slot != nullptr && work.total != nullptr,
        std::string(api) + ": the work list's device arrays must be provided");
    TLLM_CHECK_WITH_INFO(work.capacity > 0 && work.numLists > 0,
        std::string(api) + ": the work list's capacity and list count must be positive");
}

void launchSparseOutIndices(SparseOutIndexArgs const& args, CUstream stream)
{
    TLLM_CHECK_WITH_INFO(args.numRows > 0 && args.numBuffers > 0 && args.selectionWidth > 0,
        "launchSparseOutIndices: every extent must be positive");
    TLLM_CHECK_WITH_INFO(args.leases != nullptr && args.outSlot != nullptr && args.out != nullptr,
        "launchSparseOutIndices: every device array must be provided");
    TLLM_CHECK_WITH_INFO(
        args.cellsPerBuffer > 0 && args.cellsPerLease > 0 && args.unitsPerPage > 0 && args.bufferBase >= 0,
        "launchSparseOutIndices: the cell geometry must be positive");
    TLLM_CHECK_WITH_INFO(args.outBase >= 0 && args.outBase + args.numBuffers <= args.outBuffers,
        "launchSparseOutIndices: this call's buffers must fit the output row");
    TLLM_CHECK_WITH_INFO(
        args.outSlotStride >= 0 && (args.outSlotStride == 0 || args.outSlotStride >= args.selectionWidth),
        "launchSparseOutIndices: the slot stride must cover what this call reads");

    dim3 const grid(static_cast<unsigned>(args.numBuffers), static_cast<unsigned>(args.numRows));
    sparseOutIndexKernel<<<grid, kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(args);
    TLLM_CUDA_CHECK(cudaGetLastError());
}

void launchSparseFlushList(SparseFlushListArgs const& args, CUstream stream)
{
    TLLM_CHECK_WITH_INFO(
        args.numRows > 0 && args.arena.numLayers > 0, "launchSparseFlushList: rows and layers must be positive");
    TLLM_CHECK_WITH_INFO(args.arena.layerSelection != nullptr,
        "launchSparseFlushList: the scan reads each layer's selection, so the arena must name it");
    TLLM_CHECK_WITH_INFO(
        args.leases != nullptr && args.fromUnits != nullptr && args.toUnits != nullptr && args.outCount != nullptr,
        "launchSparseFlushList: every device array must be provided");
    checkWorkList(args.work, "launchSparseFlushList");
    TLLM_CHECK_WITH_INFO(args.work.numLists == args.arena.numLayers,
        "launchSparseFlushList: a write-back needs one work list per layer");
    dim3 const grid(static_cast<unsigned>(args.arena.numLayers), static_cast<unsigned>(args.numRows));
    tensorrt_llm::common::launchWithPdlWhenEnabled(
        "sparseFlushList", sparseFlushListKernel, grid, kThreads, 0, reinterpret_cast<cudaStream_t>(stream), args);
}

SparseCopyWidth sparseCopyWidth(SelectionUnitLayout const& layout, uint64_t hostPageStride, uint64_t bufferBytes)
{
    // Sizes only. Every offset a lane adds to a base is a sum of these, so a width dividing
    // all of them leaves every access aligned relative to the base -- and the bases
    // themselves are checked once, on the host, where a misaligned pool is a configuration
    // error rather than a case to carry into the kernel.
    auto const fits = [&](uint64_t width)
    {
        auto const divides = [width](uint64_t v) { return v % width == 0; };
        return divides(hostPageStride) && divides(bufferBytes) && divides(layout.unitStride)
            && divides(layout.chunkStride) && divides(layout.chunkBytes);
    };
    if (fits(sizeof(uint4)))
    {
        return SparseCopyWidth::kWord4;
    }
    if (fits(sizeof(uint2)))
    {
        return SparseCopyWidth::kWord2;
    }
    if (fits(sizeof(uint32_t)))
    {
        return SparseCopyWidth::kWord;
    }
    return SparseCopyWidth::kByte;
}

size_t sparseCopyWidthBytes(SparseCopyWidth width) noexcept
{
    switch (width)
    {
    case SparseCopyWidth::kWord4: return sizeof(uint4);
    case SparseCopyWidth::kWord2: return sizeof(uint2);
    case SparseCopyWidth::kWord: return sizeof(uint32_t);
    case SparseCopyWidth::kByte: break;
    }
    return 1;
}

void launchSparseUnitCopy(SparseUnitCopyArgs const& args, CUstream stream)
{
    TLLM_CHECK_WITH_INFO(
        args.unitsPerPage > 0 && args.numRows > 0 && args.gridCtas > 0 && args.numBuffers > 0 && args.bufferBase >= 0,
        "launchSparseUnitCopy: rows, buffers, grid CTAs and units-per-page must be positive");
    TLLM_CHECK_WITH_INFO(args.copyDesc != nullptr && args.leases != nullptr,
        "launchSparseUnitCopy: every device array must be provided");
    checkWorkList(args.work, "launchSparseUnitCopy");
    TLLM_CHECK_WITH_INFO(args.work.numLists <= 1 || args.arena.bufferLayer != nullptr,
        "launchSparseUnitCopy: a per-layer work list needs the buffer-to-layer map");

    dim3 const grid(static_cast<unsigned>(args.gridCtas), static_cast<unsigned>(args.numBuffers));
    tensorrt_llm::common::launchWithPdlWhenEnabled(
        "sparseUnitCopy", sparseUnitCopyKernel, grid, kThreads, 0, reinterpret_cast<cudaStream_t>(stream), args);
}

int32_t sparseCopyMultiprocessors(CUdevice device)
{
    int smCount = 0;
    cuCheck(cuDeviceGetAttribute(&smCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    return std::max(smCount, 1);
}

int32_t sparseCopyGridCtas(int32_t workCapacity, int32_t smCount) noexcept
{
    constexpr int32_t kCtasPerSm = 4;

    int32_t const full = std::max(smCount, 1) * kCtasPerSm;
    int32_t const needed = (workCapacity + kWarpsPerCopyCta - 1) / kWarpsPerCopyCta;
    return std::max(1, std::min(full, needed));
}

size_t sparseResolveSharedBytes(int32_t entriesPerSelection, int32_t selectionWidth, int32_t maxInFlightUnits)
{
    int32_t const required = maxInFlightUnits + selectionWidth;
    return smemLayout(nextPow2(2 * required), entriesPerSelection, required).total;
}

namespace
{

//! Dynamic shared memory a block of the resolve kernel may take, which it is opted in to.
//!
//! The device is asked once and each instantiation opted in at that maximum once: the
//! opt-in is a ceiling, and a block's occupancy follows the bytes its launch asks for
//! rather than the ceiling, so granting the most the device allows costs nothing.
//!
//! The narrower allowance of the two is what a caller is told, since the shape is sized
//! before the width that will run it is chosen.
int32_t sparseResolveSharedLimit()
{
    static int32_t const limit = []
    {
        CUdevice device = 0;
        cuCheck(cuCtxGetDevice(&device));
        int deviceMax = 0;
        cuCheck(cuDeviceGetAttribute(&deviceMax, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, device));
        int allowed = deviceMax;
        for (void const* function : {reinterpret_cast<void const*>(&sparseResolveKernel<kResolveThreadsNarrow>),
                 reinterpret_cast<void const*>(&sparseResolveKernel<kResolveThreadsWide>)})
        {
            // The opt-in names dynamic bytes, and what an instantiation declares statically
            // is spent from the same per-block budget, so only the rest of it can be asked
            // for. The two differ: cub's scan storage follows the thread count.
            cudaFuncAttributes attr{};
            TLLM_CUDA_CHECK(cudaFuncGetAttributes(&attr, function));
            int const value = deviceMax - static_cast<int>(attr.sharedSizeBytes);
            TLLM_CUDA_CHECK(cudaFuncSetAttribute(function, cudaFuncAttributeMaxDynamicSharedMemorySize, value));
            allowed = std::min(allowed, value);
        }
        return static_cast<int32_t>(allowed);
    }();
    return limit;
}

//! Threads a resolve CTA of this shape takes.
//!
//! Every pass but the probe is a stride loop over the resident set, so doubling the CTA
//! halves the entries a thread walks. A CTA's work is one row and does not grow with its
//! width, so that only pays where the wider block raises the warps a multiprocessor holds;
//! where it does not, the wider CTA costs residency and the kernel loses more to waves than
//! the shorter loops win. The device's own occupancy answer for each instantiation -- at
//! the dynamic shared memory this shape asks for, which is what drives the count -- is
//! therefore the whole decision, and a tie goes to the narrow one, whose smaller blocks
//! leave a smaller tail on a grid that does not fill the device.
//!
//! The three extents are the life cycle's, so the answer never changes within one; it is
//! held rather than asked per launch.
int resolveThreads(int32_t entriesPerSelection, int32_t selectionWidth, int32_t maxInFlightUnits, size_t shared)
{
    using Shape = std::tuple<int32_t, int32_t, int32_t>;
    static std::mutex mutex;
    static std::map<Shape, int> decided;

    Shape const shape{entriesPerSelection, selectionWidth, maxInFlightUnits};
    std::lock_guard<std::mutex> const guard(mutex);
    auto const found = decided.find(shape);
    if (found != decided.end())
    {
        return found->second;
    }
    // The query adds the instantiation's own static shared memory to what is passed here,
    // so the figure it works from is the block's whole footprint.
    auto residentBlocks = [shared](void const* kernel, int threads)
    {
        int blocksPerSm = 0;
        TLLM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSm, kernel, threads, shared));
        return blocksPerSm;
    };
    int const narrowWarps = residentBlocks(reinterpret_cast<void const*>(&sparseResolveKernel<kResolveThreadsNarrow>),
                                kResolveThreadsNarrow)
        * (kResolveThreadsNarrow / static_cast<int>(kWarpSize));
    int const wideWarps
        = residentBlocks(reinterpret_cast<void const*>(&sparseResolveKernel<kResolveThreadsWide>), kResolveThreadsWide)
        * (kResolveThreadsWide / static_cast<int>(kWarpSize));

    // A CTA's work is one row and does not grow with its width, so a wider one is worth
    // taking only where it raises the warps a multiprocessor holds. Where both widths reach
    // the same occupancy the narrow one wins on grid quantisation, since a row count that
    // does not fill the device leaves a smaller tail.
    int const threads = wideWarps > narrowWarps ? kResolveThreadsWide : kResolveThreadsNarrow;
    decided.emplace(shape, threads);
    return threads;
}

} // namespace

size_t checkedSparseResolveSharedBytes(int32_t entriesPerSelection, int32_t selectionWidth, int32_t maxInFlightUnits)
{
    size_t const shared = sparseResolveSharedBytes(entriesPerSelection, selectionWidth, maxInFlightUnits);
    int32_t const limit = sparseResolveSharedLimit();
    TLLM_CHECK_WITH_INFO(shared <= static_cast<size_t>(limit),
        "Sparse resolve: the cache needs " + std::to_string(shared)
            + " bytes of shared memory, above this device's limit of " + std::to_string(limit)
            + "; reduce max_cached_units or num_selected_blocks, or raise the selection unit size");
    return shared;
}

void launchSparseResolve(SparseResolveArgs const& args, CUstream stream)
{
    TLLM_CHECK_WITH_INFO(args.arena.numSelections > 0 && args.arena.entriesPerSelection > 0 && args.selectionWidth >= 0
            && args.numRows > 0,
        "launchSparseResolve: rows, selections and entries must be positive");
    // A zero-width resolve is the write-index path: it makes the in-flight unit resident and
    // nothing else, so it needs neither a selection to read nor a miss list to fill.
    TLLM_CHECK_WITH_INFO(args.selectionWidth != 0 || args.maxInFlightUnits != 0,
        "launchSparseResolve: a resolve with no selection must have an in-flight span");
    TLLM_CHECK_WITH_INFO(args.selectionSlot >= 0 && args.selectionSlot < args.arena.numSelections,
        "launchSparseResolve: selectionSlot must name a selection of the life cycle");
    bool const selectionArrays = args.selectionWidth == 0 || (args.selected != nullptr && args.outSlot != nullptr);
    TLLM_CHECK_WITH_INFO(args.leases != nullptr && selectionArrays && args.missCount != nullptr
            && args.historyTokens != nullptr && args.capacityTokens != nullptr
            && (args.maxInFlightUnits <= 0 || args.inFlightSlot != nullptr),
        "launchSparseResolve: every device array must be provided");
    TLLM_CHECK_WITH_INFO(args.maxInFlightUnits >= 0, "launchSparseResolve: maxInFlightUnits must not be negative");
    TLLM_CHECK_WITH_INFO(args.stepAge != nullptr,
        "launchSparseResolve: the eviction clock is device-resident, so a captured graph reads the step's own value");
    if (args.selectionWidth > 0)
    {
        checkWorkList(args.work, "launchSparseResolve");
        TLLM_CHECK_WITH_INFO(
            args.work.numLists == 1, "launchSparseResolve: a fetch resolves one selection, so it fills one list");
    }
    TLLM_CHECK_WITH_INFO(args.unitTokens > 0, "launchSparseResolve: unitTokens must be positive");
    // A position of the probe index holds an entry index beside the ordinal keyed there, so
    // a cache wider than the field would resolve two entries onto one position.
    TLLM_CHECK_WITH_INFO(args.arena.entriesPerSelection <= kMaxTableSlot,
        "Sparse resolve: the cache holds " + std::to_string(args.arena.entriesPerSelection)
            + " entries per selection, above the " + std::to_string(kMaxTableSlot)
            + " its probe index can name; reduce max_cached_units");

    dim3 const grid(static_cast<unsigned>(args.numRows));
    if (args.selectionWidth == 0 && args.maxInFlightUnits == 1)
    {
        tensorrt_llm::common::launchWithPdlWhenEnabled("sparsePrepareWrite", sparsePrepareWriteKernel, grid,
            dim3(kThreads), 0, reinterpret_cast<cudaStream_t>(stream), args);
        return;
    }

    // entriesPerSelection belongs to the life cycle, so one size and one grid serve every row.
    int32_t const tableSize = nextPow2(2 * (args.maxInFlightUnits + args.selectionWidth));
    size_t const shared
        = checkedSparseResolveSharedBytes(args.arena.entriesPerSelection, args.selectionWidth, args.maxInFlightUnits);
    if (resolveThreads(args.arena.entriesPerSelection, args.selectionWidth, args.maxInFlightUnits, shared)
        == kResolveThreadsWide)
    {
        tensorrt_llm::common::launchWithPdlWhenEnabled("sparseResolve", sparseResolveKernel<kResolveThreadsWide>, grid,
            dim3(kResolveThreadsWide), shared, reinterpret_cast<cudaStream_t>(stream), args, tableSize);
        return;
    }
    tensorrt_llm::common::launchWithPdlWhenEnabled("sparseResolve", sparseResolveKernel<kResolveThreadsNarrow>, grid,
        dim3(kResolveThreadsNarrow), shared, reinterpret_cast<cudaStream_t>(stream), args, tableSize);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
