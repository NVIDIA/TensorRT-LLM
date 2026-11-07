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

#include "kv_cache_manager_v2/common.h"
#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/storageManager.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

//! Defined in batch.h, which this header does not pull in.
struct GpuCoalescedPageTable;

//! Where a host-tier allocation's pages ended up, and under which policy.
struct HostMemPlacement
{
    //! NUMA node of the GPU current at the time of the probe.
    int gpuNumaNode;
    //! True when the range carries MPOL_BIND rather than MPOL_PREFERRED, read
    //! back from the kernel so it reports what took effect rather than what was
    //! asked. Absent where the policy cannot be queried: get_mempolicy sits
    //! behind CAP_SYS_NICE in a container's default seccomp profile, which also
    //! means mbind did not apply and the placement below came from first touch.
    std::optional<bool> strictBinding;
    //! Resident pages per NUMA node, as reported by /proc/self/numa_maps. That
    //! is an ordinary proc read rather than a gated syscall, so placement stays
    //! observable even where the policy is not.
    std::vector<std::pair<int, size_t>> nodePageCounts;
};

class KvCacheIntrospection
{
public:
    using ActivePageStats = std::tuple<TypedVec<CacheLevel, int>, TypedVec<CacheLevel, int>>;

    static ActivePageStats activePageStats(KvCache const& kvCache);

    // Whether the sequence's page at (ordinal, lcId) still points at a tree block;
    // nullopt when the slot is empty or holds an uncommitted page. Test hook for the
    // back-pointer invariant Block::replacePage() maintains.
    static std::optional<bool> committedPageIsLinked(KvCache const& kvCache, int ordinal, int lcId);
    static bool allTreePagesDroppable(KvCacheManager& manager);

    // The device-resident state of a request's sparse history cache for
    // one life cycle. Synchronises and copies back, so it is test-only. Empty when the
    // life cycle has no cache reserved.
    static std::vector<int32_t> sparseCacheResidentOrdinals(KvCache const& kvCache, int lcId);

    // The first cell of this request's lease on the life cycle's arena;
    // nullopt when it holds none. A lease is one contiguous run, so its first cell
    // identifies it.
    static std::optional<int> sparseCacheBaseCell(KvCache const& kvCache, int lcId, int pool);

    // Units the last write-back moved per buffer, -1 where none has run.
    static std::vector<int32_t> sparseFlushCounts(KvCache const& kvCache, int lcId);

    // A Batch's published device page table for one layer group, row-major
    // [max_batch][max_blocks_per_seq]. Synchronises.
    static std::vector<int32_t> batchPageTable(Batch const& batch, int layerGroupId);

    //! (table, entry) serving one buffer. A consumer names a table by the
    //! spec it asked for and slices it at the buffer's position in that spec; this is the
    //! same answer reached from the buffer, which is what the tests compare.
    static std::pair<int, int> batchIndexTableSlot(Batch const& batch, int layerId, DataRole const& role);

    //! What a converted index for one buffer is added to.
    static MemAddress batchIndexTableBase(Batch const& batch, int layerId, DataRole const& role);

    //! One converted index table, row-major [max_batch][entries][blocks_per_seq].
    //! Reads the device copy, so it shows what the last publish() converted. Synchronises.
    static std::vector<int32_t> batchIndexTable(Batch const& batch, int tableId);

    //! One layer group's coalesced page table on the device.
    //!
    //! The view, not a copy: unlike the hooks around it this neither synchronises nor
    //! reads back. Its callers check the DLPack device-array contract and that the device
    //! pointer is stable across resizes, both of which a host copy would erase. The host
    //! table, which is what a consumer actually reads, is Batch::coalescedPageTable().
    static GpuCoalescedPageTable batchGpuCoalescedPageTable(Batch const& batch, int layerGroupId);

    // entriesPerLayer of the lease each row holds, zero where it holds none. Non-zero
    // exactly where a member holds a reservation for that layer group. Synchronises.
    static std::vector<int32_t> batchSparseEntriesPerRow(Batch const& batch, int layerGroupId);

    // A Batch's published per-row watermarks as (history_length, capacity).
    // Synchronises.
    static std::pair<std::vector<int32_t>, std::vector<int32_t>> batchWatermarks(Batch const& batch);

    // Run the sparse resolve kernel over caller-supplied state, so the
    // probe index, victim choice and recency rebuild can be exercised without a request.
    // Synchronises. Inputs are [layer][entry] and [layer][selectionWidth] row-major.
    struct SparseResolveProbe
    {
        std::vector<int32_t> residentOrdinal; //!< [layer][entry] after the resolve
        std::vector<int32_t> lruOrder;        //!< [layer][entry] after the resolve
        std::vector<int32_t> outSlot;         //!< [layer][width] entry per selection
        //! The compacted work list, one per layer: [layer][rows * width], of which the
        //! first missTotal[layer] are valid, in whatever order the CTAs raced to.
        std::vector<int32_t> missRow;
        std::vector<int32_t> missOrdinal;
        std::vector<int32_t> missSlot;
        std::vector<int32_t> missTotal;    //!< [layer] entries the list holds
        std::vector<int32_t> missCount;    //!< [layer][row] entries that row contributed
        std::vector<int32_t> inFlightSlot; //!< [layer][maxInFlightUnits]
    };

    static SparseResolveProbe sparseResolve(std::vector<int32_t> const& residentOrdinal,
        std::vector<int32_t> const& lruOrder, int numLayers, int entriesPerLayer, std::vector<int32_t> const& selected,
        int selectionWidth, std::vector<int32_t> const& historyUnits, std::vector<int32_t> const& capacityUnits,
        int maxInFlightUnits, std::vector<int32_t> const& inactiveRows);

    // Run the sparse unit copy over caller-supplied pools, so the address
    // arithmetic can be checked against a known byte pattern. Returns the destination pool
    // after the copy. Synchronises.
    static std::vector<uint8_t> sparseUnitCopy(std::vector<uint8_t> const& hostPool, uint64_t hostPageStride,
        std::vector<int32_t> const& pageTable, std::vector<uint8_t> const& deviceArena, int baseCell, int numBlocks,
        int cellsPerBuffer, uint64_t bufferBytes, uint32_t numChunks, uint32_t chunkBytes, uint32_t chunkStride,
        uint32_t unitStride, int unitsPerPage, std::vector<int32_t> const& missOrdinal,
        std::vector<int32_t> const& missSlot, std::vector<int32_t> const& missCount, int selectionWidth,
        std::vector<int32_t> const& bufferLayer, bool writeBack, int gridCtas);

    // The entries holding units a watermark move has just passed.
    // Synchronises.
    struct SparseFlushList
    {
        //! The compacted list, one per layer: [layer][capacity], of which the first
        //! total[layer] are valid, in whatever order the CTAs raced to.
        std::vector<int32_t> row;
        std::vector<int32_t> ordinal;
        std::vector<int32_t> slot;
        std::vector<int32_t> total; //!< [layer]
        std::vector<int32_t> count; //!< [layer] entries this row contributed
    };

    static SparseFlushList sparseFlushList(std::vector<int32_t> const& residentOrdinal, int numLayers,
        int entriesPerLayer, int fromUnits, int toUnits, int maxFlushUnits);

    // White-box hook: minimum per-pool-group slot counts to support a BatchDesc.
    // Reaches StorageManager::computePoolGroupSlotsForBatch() (private) via friendship.
    static TypedVec<PoolGroupIndex, SlotCount> computeSlotsForBatch(KvCacheManager& manager, BatchDesc const& batch,
        int tokensPerBlock, std::optional<SwaScratchReuseConfig> const& swaScratchReuse);

    // White-box test hooks: mutate auto-tuner state so accuracy tests can force a
    // pool rebalance. Reach KvCacheManager's private members via friendship.
    static void setNumSampledKvCaches(KvCacheManager& manager, int value);
    static void setLastAdjustmentTime(KvCacheManager& manager, double value);
    static void setTargetRatioListGpu(KvCacheManager& manager, TypedVec<PoolGroupIndex, float> value);

    //! Allocates `size` bytes of host-tier memory the way the host tier would
    //! and reports where its pages landed.
    //!
    //! The commit runs on a worker bound to the GPU's node, as HostMem's fill
    //! does, so the result reflects the arrangement in production rather than
    //! mbind in isolation: placement holds through first touch when mbind is
    //! unavailable, and through the range policy when it is.
    //!
    //! Returns nullopt only where the question does not apply -- no libnuma, or
    //! a single NUMA node.
    //!
    //! Uses the current device, so the caller selects which GPU the placement is
    //! relative to. A CUDA context must already exist on it: the backing
    //! selection queries device attributes and will not create one.
    [[nodiscard]] static std::optional<HostMemPlacement> probeHostMemPlacement(
        size_t size, bool allowRemoteNumaFallback);
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
