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

#include "kv_cache_manager_v2/introspection.h"

#include "kv_cache_manager_v2/batch.h"
#include "kv_cache_manager_v2/blockRadixTree.h"

#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/page.h"
#include "kv_cache_manager_v2/sparseCacheKernels.h"
#include "kv_cache_manager_v2/stagingBuffer.h"
#include "kv_cache_manager_v2/storageManager.h"
#include "kv_cache_manager_v2/utils/hostMemBacking.h"
#include "kv_cache_manager_v2/utils/math.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/runtime/moeLoadBalancer/topologyDetector.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <numa.h>
#include <numaif.h>
#include <numeric>
#include <sstream>
#include <thread>
#include <utility>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{

bool allBlockPagesDroppable(Block const& block)
{
    for (auto const* page : block.storage)
    {
        if (page != nullptr && page->status() != PageStatus::DROPPABLE)
        {
            return false;
        }
    }

    for (auto const& [_, child] : block.next)
    {
        if (!allBlockPagesDroppable(*child))
        {
            return false;
        }
    }
    return true;
}

} // namespace

KvCacheIntrospection::ActivePageStats KvCacheIntrospection::activePageStats(KvCache const& kvCache)
{
    auto& storageMgr = kvCache.manager().storage();
    CacheLevel const numTiers = storageMgr.numCacheLevels();
    TypedVec<CacheLevel, int> counts(numTiers, 0);
    TypedVec<CacheLevel, int> unscheduledEvictable(numTiers, 0);

    for (auto const& activePage : kvCache._activePages())
    {
        auto page = kvCache._page(activePage.ordinal, activePage.beamIdx, activePage.lcId);
        if (!page)
        {
            continue;
        }

        CacheLevel const level = page->cacheLevel;
        counts.at(level) += 1;
        if (storageMgr.isEvictable(*page) && !page->scheduledForEviction())
        {
            unscheduledEvictable.at(level) += 1;
        }
    }

    return {std::move(counts), std::move(unscheduledEvictable)};
}

std::optional<bool> KvCacheIntrospection::committedPageIsLinked(KvCache const& kvCache, int ordinal, int lcId)
{
    auto page = kvCache._page(BlockOrdinal{ordinal}, kDefaultBeamIndex, LifeCycleId{lcId});
    if (!page)
    {
        return std::nullopt;
    }
    auto committed = dynamicPointerCast<CommittedPage>(page);
    if (!committed)
    {
        return std::nullopt;
    }
    return committed->block != nullptr;
}

TypedVec<PoolGroupIndex, SlotCount> KvCacheIntrospection::computeSlotsForBatch(KvCacheManager& manager,
    BatchDesc const& batch, int tokensPerBlock, std::optional<SwaScratchReuseConfig> const& swaScratchReuse)
{
    return manager.storage().computePoolGroupSlotsForBatch(batch, tokensPerBlock, swaScratchReuse);
}

std::vector<int32_t> KvCacheIntrospection::sparseCacheResidentOrdinals(KvCache const& kvCache, int lcId)
{
    LifeCycleId const id{lcId};
    // The arena belongs to the Batch the lease was taken from, so an unseated cache has
    // nothing resident to report.
    if (id >= kvCache.mManager->numLifeCycles() || kvCache.mBatch == nullptr)
    {
        return {};
    }
    return kvCache.mBatch->_sparseArena(id).peekResidentOrdinals(kvCache.mBatch->leaseOf(*kvCache.mBatchRow));
}

std::optional<int> KvCacheIntrospection::sparseCacheBaseCell(KvCache const& kvCache, int lcId, int pool)
{
    LifeCycleId const id{lcId};
    if (id >= kvCache.mManager->numLifeCycles() || kvCache.mBatch == nullptr
        || kvCache.mBatch->_sparseArena(id).empty())
    {
        return std::nullopt;
    }
    // Relative to the pool's own arena region, which is what getSparseCacheBaseAddress
    // reports for a buffer drawing on it.
    auto const& arena = kvCache.mBatch->_sparseArena(id);
    return kvCache.mBatch->leaseOf(*kvCache.mBatchRow).value() * arena.cellsPerLease(PoolIndex{pool});
}

std::vector<int32_t> KvCacheIntrospection::batchPageTable(Batch const& batch, int layerGroupId)
{
    // The Batch's own stream, not the null one: it is created non-blocking, so a null-stream
    // read is not ordered against the conversion that fills what is read here.
    CUstream const stream = batch.cudaStream();
    auto const view = batch.gpuCoalescedPageTable(LifeCycleId{layerGroupId});
    std::vector<int32_t> out(static_cast<size_t>(view.numRows) * view.blocksPerRow);
    cuCheck(cuMemcpyDtoHAsync(out.data(), view.data, sizeof(int32_t) * out.size(), stream));
    cuCheck(cuStreamSynchronize(stream));
    return out;
}

std::pair<int, int> KvCacheIntrospection::batchIndexTableSlot(Batch const& batch, int layerId, DataRole const& role)
{
    auto const [table, entry] = batch.indexTableSlot(LayerId{layerId}, role);
    return {table.value(), entry};
}

MemAddress KvCacheIntrospection::batchIndexTableBase(Batch const& batch, int layerId, DataRole const& role)
{
    return batch.indexTableBase(LayerId{layerId}, role);
}

std::vector<int32_t> KvCacheIntrospection::batchIndexTable(Batch const& batch, int tableId)
{
    // The Batch's own stream, not the null one: it is created non-blocking, so a null-stream
    // read is not ordered against the conversion that fills what is read here.
    CUstream const stream = batch.cudaStream();
    IndexTableId const typed{tableId};
    auto const numBuffers = static_cast<size_t>(batch.indexTable(typed).front().buffers.size());
    auto const perEntry = static_cast<size_t>(batch.maxBatchSize()) * static_cast<size_t>(batch.maxBlocksPerSeq());
    std::vector<int32_t> out(numBuffers * perEntry);
    cuCheck(cuMemcpyDtoHAsync(out.data(), batch.indexTableAddress(typed), sizeof(int32_t) * out.size(), stream));
    cuCheck(cuStreamSynchronize(stream));
    return out;
}

GpuCoalescedPageTable KvCacheIntrospection::batchGpuCoalescedPageTable(Batch const& batch, int layerGroupId)
{
    return batch.gpuCoalescedPageTable(LifeCycleId{layerGroupId});
}

std::vector<int32_t> KvCacheIntrospection::batchSparseEntriesPerRow(Batch const& batch, int layerGroupId)
{
    // The Batch's own stream, not the null one: it is created non-blocking, so a null-stream
    // read is not ordered against the conversion that fills what is read here.
    CUstream const stream = batch.cudaStream();
    auto const rows = static_cast<size_t>(batch.maxBatchSize());
    std::vector<int32_t> leases(rows);
    cuCheck(cuMemcpyDtoHAsync(leases.data(), batch.sparseLeases(), sizeof(int32_t) * rows, stream));
    cuCheck(cuStreamSynchronize(stream));

    // Every lease of a life cycle has the same shape, so the arena answers for all of them.
    auto const entriesPerLayer = batch._sparseArena(LifeCycleId{layerGroupId}).view().entriesPerSelection;
    std::vector<int32_t> out;
    out.reserve(rows);
    for (auto const lease : leases)
    {
        out.push_back(lease == kNoSparseLease.value() ? 0 : entriesPerLayer);
    }
    return out;
}

std::pair<std::vector<int32_t>, std::vector<int32_t>> KvCacheIntrospection::batchWatermarks(Batch const& batch)
{
    // The Batch's own stream, not the null one: it is created non-blocking, so a null-stream
    // read is not ordered against the conversion that fills what is read here.
    CUstream const stream = batch.cudaStream();
    auto const view = batch.watermarks();
    std::vector<int32_t> history(static_cast<size_t>(view.numRows));
    std::vector<int32_t> capacity(static_cast<size_t>(view.numRows));
    cuCheck(cuMemcpyDtoHAsync(history.data(), view.historyLength, sizeof(int32_t) * history.size(), stream));
    cuCheck(cuMemcpyDtoHAsync(capacity.data(), view.capacity, sizeof(int32_t) * capacity.size(), stream));
    cuCheck(cuStreamSynchronize(stream));
    return {std::move(history), std::move(capacity)};
}

std::vector<int32_t> KvCacheIntrospection::sparseFlushCounts(KvCache const& kvCache, int lcId)
{
    LifeCycleId const id{lcId};
    if (id >= kvCache.mManager->numLifeCycles() || kvCache.mBatch == nullptr)
    {
        return {};
    }
    // The counters are written on the Batch's stream, which a caller may make non-blocking;
    // a null-stream read would not be ordered against it.
    CUstream const stream = kvCache.mBatch->cudaStream();
    // The lists are the step's, not the request's, so this reads the Batch's row for it.
    auto const& scratch = kvCache.mBatch->sparseStepScratch(LifeCycleId{lcId});
    auto const layers = static_cast<size_t>(scratch.numLayers);
    std::vector<int32_t> out(layers);
    cuCheck(cuMemcpyDtoHAsync(out.data(),
        reinterpret_cast<CUdeviceptr>(scratch.flushCount + toSizeT(*kvCache.mBatchRow) * layers),
        sizeof(int32_t) * layers, stream));
    cuCheck(cuStreamSynchronize(stream));
    return out;
}

namespace
{

CudaUniqPtr<> deviceCopy(std::vector<int32_t> const& values)
{
    CUstream const stream = nullptr;
    // Null for an empty array, which cuMemAlloc refuses and the kernels accept: a probe may
    // declare a width of zero, and every array sized from it is then empty.
    auto owned = allocateDeviceMemory(sizeof(int32_t) * values.size());
    if (values.empty())
    {
        return owned;
    }
    cuCheck(cuMemcpyHtoDAsync(
        reinterpret_cast<CUdeviceptr>(owned.get()), values.data(), sizeof(int32_t) * values.size(), stream));
    cuCheck(cuStreamSynchronize(stream));
    return owned;
}

std::vector<int32_t> readBack(CudaUniqPtr<> const& buffer, size_t count)
{
    CUstream const stream = nullptr;
    std::vector<int32_t> out(count);
    if (count == 0)
    {
        return out;
    }
    cuCheck(
        cuMemcpyDtoHAsync(out.data(), reinterpret_cast<CUdeviceptr>(buffer.get()), sizeof(int32_t) * count, stream));
    cuCheck(cuStreamSynchronize(stream));
    return out;
}

} // namespace

KvCacheIntrospection::SparseResolveProbe KvCacheIntrospection::sparseResolve(
    std::vector<int32_t> const& residentOrdinal, std::vector<int32_t> const& lruOrder, int numLayers,
    int entriesPerLayer, std::vector<int32_t> const& selected, int selectionWidth,
    std::vector<int32_t> const& historyUnits, std::vector<int32_t> const& capacityUnits, int maxInFlightUnits,
    std::vector<int32_t> const& inactiveRows)
{
    CUstream const stream = nullptr;
    auto const numRows = static_cast<int>(historyUnits.size());
    auto const perRowEntries = static_cast<size_t>(numLayers) * entriesPerLayer;
    auto const entryCount = perRowEntries * static_cast<size_t>(numRows);
    auto const perRowSelection = static_cast<size_t>(numLayers) * selectionWidth;
    auto const selectionCount = perRowSelection * static_cast<size_t>(numRows);
    TLLM_CHECK_WITH_INFO(maxInFlightUnits >= 0, "sparseResolve: maxInFlightUnits counts units and cannot be negative");
    TLLM_CHECK_WITH_INFO(numRows != 0 && capacityUnits.size() == historyUnits.size()
            && residentOrdinal.size() == entryCount && lruOrder.size() == entryCount
            && selected.size() == selectionCount,
        "sparseResolve: input sizes must match the declared shape");

    auto selection = deviceCopy(selected);
    auto outSlot = deviceCopy(std::vector<int32_t>(selectionCount, 0));
    auto missRow = deviceCopy(std::vector<int32_t>(selectionCount, kNoSparseSlot));
    auto missOrdinal = deviceCopy(std::vector<int32_t>(selectionCount, kNoSparseSlot));
    auto missSlot = deviceCopy(std::vector<int32_t>(selectionCount, kNoSparseSlot));
    auto missTotal = deviceCopy(std::vector<int32_t>(static_cast<size_t>(numLayers), 0));
    auto const countCount = static_cast<size_t>(numLayers) * static_cast<size_t>(numRows);
    auto missCount = deviceCopy(std::vector<int32_t>(countCount, 0));
    auto history = deviceCopy(historyUnits);
    auto capacity = deviceCopy(capacityUnits);
    auto const inFlightCount = countCount * static_cast<size_t>(maxInFlightUnits);
    auto inFlight = deviceCopy(std::vector<int32_t>(inFlightCount, kNoSparseSlot));

    // One metadata block per lease, its resident ordinals then its recency order, which is
    // how the arena lays a lease out. The probe takes and returns each array whole, so they
    // are interleaved on the way in and separated on the way out.
    auto const leaseInts = 2 * perRowEntries;
    std::vector<int32_t> metadataHost(leaseInts * static_cast<size_t>(numRows));
    for (int row = 0; row < numRows; ++row)
    {
        auto const from = perRowEntries * static_cast<size_t>(row);
        auto const to = leaseInts * static_cast<size_t>(row);
        std::copy_n(residentOrdinal.begin() + from, perRowEntries, metadataHost.begin() + to);
        std::copy_n(lruOrder.begin() + from, perRowEntries, metadataHost.begin() + to + perRowEntries);
    }
    auto metadata = deviceCopy(metadataHost);

    // Row r takes lease r, except an inactive row, which holds none for the kernel to skip.
    std::vector<int32_t> leaseHost(static_cast<size_t>(numRows));
    for (int row = 0; row < numRows; ++row)
    {
        leaseHost[static_cast<size_t>(row)]
            = std::find(inactiveRows.begin(), inactiveRows.end(), row) != inactiveRows.end() ? kNoSparseLease.value()
                                                                                             : row;
    }
    auto leases = deviceCopy(leaseHost);

    SparseArenaView arena;
    arena.metadataBase = metadata.get();
    arena.leaseStride = sizeof(int32_t) * leaseInts;
    arena.lruOffset = static_cast<int32_t>(perRowEntries);
    arena.numLayers = numLayers;
    // One metadata row per layer: this probe drives a model where every layer selects for
    // itself, which is what a resolve sees whenever no selection_id is shared.
    arena.numSelections = numLayers;
    arena.entriesPerSelection = entriesPerLayer;
    // This probe drives the metadata, not the addressing.
    arena.bufferSlotInPool = nullptr;
    arena.bufferLayer = nullptr;
    arena.layerSelection = nullptr;

    // One launch per layer, as a step does: a layer's selection is not known until the
    // previous layer has run. The arrays are therefore layer-major, so each launch sees a
    // contiguous [row][...] slice of its own. Same stream, so the layers stay ordered.
    for (int layer = 0; layer < numLayers; ++layer)
    {
        auto const selectionOffset = static_cast<size_t>(layer) * numRows * selectionWidth;
        auto const countOffset = static_cast<size_t>(layer) * numRows;

        SparseResolveArgs args;
        args.leases = reinterpret_cast<SparseLeaseIndex const*>(leases.get());
        args.arena = arena;
        args.numRows = numRows;
        args.selectionSlot = layer;
        args.selected = reinterpret_cast<int32_t const*>(selection.get()) + selectionOffset;
        args.selectionWidth = selectionWidth;
        // The hook works in units, so one token is one unit.
        args.historyTokens = reinterpret_cast<int32_t const*>(history.get());
        args.capacityTokens = reinterpret_cast<int32_t const*>(capacity.get());
        args.unitTokens = 1;
        args.inFlightSlot
            = reinterpret_cast<int32_t*>(inFlight.get()) + countOffset * static_cast<size_t>(maxInFlightUnits);
        args.maxInFlightUnits = maxInFlightUnits;
        args.outSlot = reinterpret_cast<int32_t*>(outSlot.get()) + selectionOffset;
        // One list per layer, each the width of the whole batch's selection: the worst case
        // is every row missing everything it asked for.
        args.work.row = reinterpret_cast<int32_t*>(missRow.get()) + selectionOffset;
        args.work.ordinal = reinterpret_cast<int32_t*>(missOrdinal.get()) + selectionOffset;
        args.work.slot = reinterpret_cast<int32_t*>(missSlot.get()) + selectionOffset;
        args.work.total = reinterpret_cast<int32_t*>(missTotal.get()) + layer;
        args.work.capacity = numRows * selectionWidth;
        args.work.numLists = 1;
        args.missCount = reinterpret_cast<int32_t*>(missCount.get()) + countOffset;

        launchSparseResolve(args, stream);
    }
    cuCheck(cuStreamSynchronize(stream));

    SparseResolveProbe result;
    auto const metadataOut = readBack(metadata, leaseInts * static_cast<size_t>(numRows));
    result.residentOrdinal.resize(entryCount);
    result.lruOrder.resize(entryCount);
    for (int row = 0; row < numRows; ++row)
    {
        auto const from = leaseInts * static_cast<size_t>(row);
        auto const to = perRowEntries * static_cast<size_t>(row);
        std::copy_n(metadataOut.begin() + from, perRowEntries, result.residentOrdinal.begin() + to);
        std::copy_n(metadataOut.begin() + from + perRowEntries, perRowEntries, result.lruOrder.begin() + to);
    }
    result.outSlot = readBack(outSlot, selectionCount);
    result.missRow = readBack(missRow, selectionCount);
    result.missOrdinal = readBack(missOrdinal, selectionCount);
    result.missSlot = readBack(missSlot, selectionCount);
    result.missTotal = readBack(missTotal, static_cast<size_t>(numLayers));
    result.missCount = readBack(missCount, countCount);
    result.inFlightSlot = inFlightCount > 0 ? readBack(inFlight, inFlightCount) : std::vector<int32_t>{};
    return result;
}

std::vector<uint8_t> KvCacheIntrospection::sparseUnitCopy(std::vector<uint8_t> const& hostPool, uint64_t hostPageStride,
    std::vector<int32_t> const& pageTable, std::vector<uint8_t> const& deviceArena, int baseCell, int numBlocks,
    int cellsPerBuffer, uint64_t bufferBytes, uint32_t numChunks, uint32_t chunkBytes, uint32_t chunkStride,
    uint32_t unitStride, int unitsPerPage, std::vector<int32_t> const& missOrdinal,
    std::vector<int32_t> const& missSlot, std::vector<int32_t> const& missCount, int selectionWidth,
    std::vector<int32_t> const& bufferLayer, bool writeBack, int gridCtas)
{
    // The row's lease is what says whether it holds a reservation, so a negative one asks
    // the kernel to skip the very copy this probe exists to run.
    TLLM_CHECK_WITH_INFO(
        baseCell >= 0, "KvCacheIntrospection::sparseUnitCopy: baseCell names an arena cell and cannot be negative");
    CUstream const stream = nullptr;
    auto source = allocateDeviceMemory(hostPool.size());
    auto const hostPtr = reinterpret_cast<CUdeviceptr>(source.get());
    cuCheck(cuMemcpyHtoDAsync(hostPtr, hostPool.data(), hostPool.size(), stream));

    auto destination = allocateDeviceMemory(deviceArena.size());
    auto const devicePtr = reinterpret_cast<CUdeviceptr>(destination.get());
    cuCheck(cuMemcpyHtoDAsync(devicePtr, deviceArena.data(), deviceArena.size(), stream));

    auto table = deviceCopy(pageTable);
    // The caller describes work as [list][selectionWidth] with a count per list, which is
    // the shape a test can write by hand. Compact it here, because that is what the kernel
    // consumes: a dense run per list, with the row carried by the entry.
    auto const numLists = bufferLayer.empty() ? 1 : bufferLayer.back() + 1;
    auto const perList = static_cast<size_t>(std::max(selectionWidth, 1));
    std::vector<int32_t> flatRow;
    std::vector<int32_t> flatOrdinal;
    std::vector<int32_t> flatSlot;
    std::vector<int32_t> totals(static_cast<size_t>(numLists), 0);
    flatRow.reserve(missOrdinal.size());
    flatOrdinal.reserve(missOrdinal.size());
    flatSlot.reserve(missOrdinal.size());
    for (int list = 0; list < numLists; ++list)
    {
        auto const count = static_cast<size_t>(missCount.at(static_cast<size_t>(list)));
        for (size_t j = 0; j < perList; ++j)
        {
            auto const index = static_cast<size_t>(list) * perList + j;
            // The hook drives one request, so every entry names row zero.
            flatRow.push_back(0);
            flatOrdinal.push_back(j < count ? missOrdinal.at(index) : kNoSparseSlot);
            flatSlot.push_back(j < count ? missSlot.at(index) : kNoSparseSlot);
        }
        totals[static_cast<size_t>(list)] = static_cast<int32_t>(count);
    }

    auto workRowBuf = deviceCopy(flatRow);
    auto missOrd = deviceCopy(flatOrdinal);
    auto missSlotBuf = deviceCopy(flatSlot);
    auto totalBuf = deviceCopy(totals);
    auto bufferLayerBuf = deviceCopy(bufferLayer);

    // The kernel reads the lease and the per-buffer maps, so the hook assembles the arena a
    // single request's cells would be addressed within. The copy reads no metadata block, so
    // the arena names none.
    std::vector<int32_t> slotInPool(bufferLayer.size());
    std::iota(slotInPool.begin(), slotInPool.end(), 0);
    auto slotInPoolBuf = deviceCopy(slotInPool);

    // The probe drives one pool, so a lease of one cell and a buffer-slot identity reproduce
    // the address the production path computes from a lease index and a pool slot.
    auto leases = deviceCopy(std::vector<int32_t>(1, baseCell));
    SparseArenaView arena;
    arena.bufferSlotInPool = reinterpret_cast<int32_t const*>(slotInPoolBuf.get());
    arena.bufferLayer = reinterpret_cast<int32_t const*>(bufferLayerBuf.get());
    // This probe drives the addressing, not the metadata.
    arena.metadataBase = nullptr;
    arena.leaseStride = 0;
    arena.lruOffset = 0;
    arena.numLayers = 0;
    arena.numSelections = 0;
    arena.entriesPerSelection = 0;
    arena.layerSelection = nullptr;

    // One descriptor per buffer, as the real path builds them: the hook drives the
    // addressing, so every buffer shares a pool and differs only by its slot in it.
    std::vector<SparseCopyDesc> descs(bufferLayer.size());
    for (size_t i = 0; i < descs.size(); ++i)
    {
        auto& desc = descs[i];
        desc.hostBase = hostPtr;
        desc.hostPageStride = static_cast<uint32_t>(hostPageStride);
        desc.pageTable = reinterpret_cast<int32_t const*>(table.get()) + i * static_cast<size_t>(numBlocks);
        desc.pageTableStride = static_cast<uint32_t>(numBlocks);
        desc.numBlocks = static_cast<uint32_t>(numBlocks);
        desc.deviceBase = devicePtr + static_cast<uint64_t>(i) * static_cast<uint64_t>(cellsPerBuffer) * bufferBytes;
        desc.leaseStride = static_cast<uint32_t>(bufferBytes);
        desc.cellBytes = static_cast<uint32_t>(bufferBytes);
        desc.layout.numChunks = numChunks;
        desc.layout.chunkBytes = chunkBytes;
        desc.layout.chunkStride = chunkStride;
        desc.layout.unitStride = unitStride;
        desc.width = sparseCopyWidth(desc.layout, desc.hostPageStride, bufferBytes);
    }
    auto descOwner = allocateDeviceMemory(sizeof(SparseCopyDesc) * descs.size());
    auto const descPtr = reinterpret_cast<CUdeviceptr>(descOwner.get());
    cuCheck(cuMemcpyHtoDAsync(descPtr, descs.data(), sizeof(SparseCopyDesc) * descs.size(), stream));

    SparseUnitCopyArgs args;
    args.leases = reinterpret_cast<SparseLeaseIndex const*>(leases.get());
    args.arena = arena;
    args.numRows = 1;
    args.copyDesc = reinterpret_cast<SparseCopyDesc const*>(descPtr);
    args.bufferBase = 0;
    args.numBuffers = static_cast<int32_t>(descs.size());
    args.direction = writeBack ? SparseCopyDirection::kWriteBack : SparseCopyDirection::kFetch;
    args.unitsPerPage = unitsPerPage;
    args.work.row = reinterpret_cast<int32_t*>(workRowBuf.get());
    args.work.ordinal = reinterpret_cast<int32_t*>(missOrd.get());
    args.work.slot = reinterpret_cast<int32_t*>(missSlotBuf.get());
    args.work.total = reinterpret_cast<int32_t*>(totalBuf.get());
    args.work.capacity = static_cast<int32_t>(flatOrdinal.size());
    args.work.numLists = numLists;
    args.gridCtas = gridCtas;

    launchSparseUnitCopy(args, stream);
    cuCheck(cuStreamSynchronize(stream));

    // Write-back moves into the cold pool, so that is the side worth returning.
    CUdeviceptr const result = writeBack ? hostPtr : devicePtr;
    size_t const resultBytes = writeBack ? hostPool.size() : deviceArena.size();
    std::vector<uint8_t> out(resultBytes);
    cuCheck(cuMemcpyDtoHAsync(out.data(), result, resultBytes, stream));
    cuCheck(cuStreamSynchronize(stream));
    return out;
}

KvCacheIntrospection::SparseFlushList KvCacheIntrospection::sparseFlushList(std::vector<int32_t> const& residentOrdinal,
    int numLayers, int entriesPerLayer, int fromUnits, int toUnits, int maxFlushUnits)
{
    CUstream const stream = nullptr;
    auto const entryCount = static_cast<size_t>(numLayers) * entriesPerLayer;
    TLLM_CHECK_WITH_INFO(
        residentOrdinal.size() == entryCount, "sparseFlushList: input size must match the declared shape");
    auto const listCount = static_cast<size_t>(numLayers) * maxFlushUnits;

    auto from = deviceCopy(std::vector<int32_t>(1, fromUnits));
    auto to = deviceCopy(std::vector<int32_t>(1, toUnits));
    auto outRow = deviceCopy(std::vector<int32_t>(listCount, kNoSparseSlot));
    auto outOrdinal = deviceCopy(std::vector<int32_t>(listCount, kNoSparseSlot));
    auto outSlot = deviceCopy(std::vector<int32_t>(listCount, kNoSparseSlot));
    auto outTotal = deviceCopy(std::vector<int32_t>(static_cast<size_t>(numLayers), 0));
    auto outCount = deviceCopy(std::vector<int32_t>(static_cast<size_t>(numLayers), 0));

    // The scan spans the batch, so the hook stands one row in for a single request, holding
    // one lease laid out as the arena lays one: its resident ordinals, then its recency
    // order, which the scan does not read.
    std::vector<int32_t> metadataHost(2 * entryCount, 0);
    std::copy(residentOrdinal.begin(), residentOrdinal.end(), metadataHost.begin());
    auto metadata = deviceCopy(metadataHost);
    auto leases = deviceCopy(std::vector<int32_t>(1, 0));

    // A layer per metadata row, in order: the probe drives unshared selections.
    std::vector<int32_t> layerSelectionHost(static_cast<size_t>(numLayers));
    std::iota(layerSelectionHost.begin(), layerSelectionHost.end(), 0);
    auto layerSelection = deviceCopy(layerSelectionHost);

    SparseArenaView arena;
    arena.metadataBase = metadata.get();
    arena.leaseStride = sizeof(int32_t) * 2 * entryCount;
    arena.lruOffset = static_cast<int32_t>(entryCount);
    arena.numLayers = numLayers;
    arena.numSelections = numLayers;
    arena.entriesPerSelection = entriesPerLayer;
    // This probe drives the metadata, not the addressing.
    arena.bufferSlotInPool = nullptr;
    arena.bufferLayer = nullptr;
    arena.layerSelection = reinterpret_cast<int32_t const*>(layerSelection.get());

    SparseFlushListArgs args;
    args.leases = reinterpret_cast<SparseLeaseIndex const*>(leases.get());
    args.arena = arena;
    args.numRows = 1;
    args.fromUnits = reinterpret_cast<int32_t const*>(from.get());
    args.toUnits = reinterpret_cast<int32_t const*>(to.get());
    args.work.row = reinterpret_cast<int32_t*>(outRow.get());
    args.work.ordinal = reinterpret_cast<int32_t*>(outOrdinal.get());
    args.work.slot = reinterpret_cast<int32_t*>(outSlot.get());
    args.work.total = reinterpret_cast<int32_t*>(outTotal.get());
    args.work.capacity = maxFlushUnits;
    args.work.numLists = numLayers;
    args.outCount = reinterpret_cast<int32_t*>(outCount.get());

    launchSparseFlushList(args, stream);
    cuCheck(cuStreamSynchronize(stream));

    SparseFlushList result;
    result.row = readBack(outRow, listCount);
    result.ordinal = readBack(outOrdinal, listCount);
    result.slot = readBack(outSlot, listCount);
    result.total = readBack(outTotal, static_cast<size_t>(numLayers));
    result.count = readBack(outCount, static_cast<size_t>(numLayers));
    return result;
}

bool KvCacheIntrospection::allTreePagesDroppable(KvCacheManager& manager)
{
    for (auto const& [_, root] : manager.radixTree().roots())
    {
        for (auto const& [__, block] : root->next)
        {
            if (!allBlockPagesDroppable(*block))
            {
                return false;
            }
        }
    }
    return true;
}

void KvCacheIntrospection::setNumSampledKvCaches(KvCacheManager& manager, int value)
{
    manager.mNumSampledKvCaches = value;
}

void KvCacheIntrospection::setLastAdjustmentTime(KvCacheManager& manager, double value)
{
    manager.mLastAdjustmentTime = value;
}

void KvCacheIntrospection::setTargetRatioListGpu(KvCacheManager& manager, TypedVec<PoolGroupIndex, float> value)
{
    manager.mTargetRatioListHot = std::move(value);
}

namespace
{

//! Resident pages per NUMA node for the VMA containing `base`.
//!
//! Read from /proc/self/numa_maps rather than queried with get_mempolicy or
//! move_pages: those are gated behind CAP_SYS_NICE in a container's default
//! seccomp profile, while this is an ordinary file read. The line for a mapping
//! carries "N<node>=<pages>" fields, which is what this picks out.
std::vector<std::pair<int, size_t>> residentPagesPerNode(MemAddress base)
{
    std::ifstream maps("/proc/self/numa_maps");
    if (!maps)
    {
        return {};
    }
    std::ostringstream prefix;
    prefix << std::hex << base;
    std::string const wanted = prefix.str();

    std::string line;
    while (std::getline(maps, line))
    {
        if (line.rfind(wanted, 0) != 0)
        {
            continue;
        }
        std::vector<std::pair<int, size_t>> counts;
        std::istringstream fields(line);
        std::string field;
        while (fields >> field)
        {
            if (field.size() < 2 || field[0] != 'N' || std::isdigit(field[1]) == 0)
            {
                continue;
            }
            auto const eq = field.find('=');
            if (eq == std::string::npos)
            {
                continue;
            }
            counts.emplace_back(std::stoi(field.substr(1, eq - 1)), std::stoull(field.substr(eq + 1)));
        }
        return counts;
    }
    return {};
}

} // namespace

std::optional<HostMemPlacement> KvCacheIntrospection::probeHostMemPlacement(size_t size, bool allowRemoteNumaFallback)
{
    TLLM_CHECK_WITH_INFO(size > 0, "probeHostMemPlacement needs a non-zero size");
    if (numa_available() < 0 || numa_max_node() < 1)
    {
        return std::nullopt;
    }

    HostMemBackingOptions options;
    options.allowRemoteNumaFallback = allowRemoteNumaFallback;
    size_t const span = roundUp(size, hostMemSizeGranularity());
    auto backing = std::make_unique<HostMemBacking>(span, options);

    HostMemPlacement placement{};
    placement.gpuNumaNode = runtime::TopologyDetector::getInstance().getCurrentGpuNumaId();

    MemAddress const base = backing->address();

    int mode = -1;
    if (::get_mempolicy(&mode, nullptr, 0, reinterpret_cast<void*>(base), MPOL_F_ADDR) == 0)
    {
        placement.strictBinding = mode == MPOL_BIND;
    }

    // Commit and fault on a worker bound to the GPU's node, which is what
    // HostMem's fill does. Placement then holds through first touch even where
    // mbind was refused, so this reports the production arrangement rather than
    // mbind alone.
    int device = 0;
    TLLM_CUDA_CHECK(cudaGetDevice(&device));

    std::exception_ptr failure;
    std::thread worker(
        [&]
        {
            try
            {
                // A new thread starts on device 0 whatever the caller selected,
                // so the device has to be reselected before binding by GPU.
                TLLM_CUDA_CHECK(cudaSetDevice(device));
                runtime::TopologyDetector::getInstance().bindThreadByCurrentGpu();
                backing->resizeTo(span);
                std::memset(reinterpret_cast<void*>(base), 0, span);
            }
            catch (...)
            {
                failure = std::current_exception();
            }
        });
    worker.join();
    if (failure)
    {
        std::rethrow_exception(failure);
    }

    placement.nodePageCounts = residentPagesPerNode(base);
    return placement;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
