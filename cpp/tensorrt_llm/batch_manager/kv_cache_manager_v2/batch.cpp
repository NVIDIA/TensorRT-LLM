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

#include "kv_cache_manager_v2/batch.h"

#include "kv_cache_manager_v2/batchedPageCopy.h"

#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/indexConversionKernels.h"
#include "kv_cache_manager_v2/kvCache.h"
#include "kv_cache_manager_v2/kvCacheManager.h"
#include "kv_cache_manager_v2/sparseCacheKernels.h"
#include "kv_cache_manager_v2/storageManager.h"
#include "kv_cache_manager_v2/utils/math.h"

#include "tensorrt_llm/common/cudaUtils.h"

#include <algorithm>
#include <array>
#include <numeric>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{

//! Largest batch and longest sequence the declared constraints describe, or zero when none
//! were declared.
std::pair<int, int> deriveBounds(KVCacheManagerConfig const& config)
{
    int rows = 0;
    int capacity = 0;
    for (auto const& desc : config.constraints)
    {
        rows = std::max(rows, static_cast<int>(desc.kvCaches.size()));
        for (auto const& kvCache : desc.kvCaches)
        {
            capacity = std::max(capacity, kvCache.capacity);
        }
    }
    return {rows, capacity == 0 ? 0 : divUp(capacity, config.tokensPerBlock)};
}

} // namespace

Batch::Batch(KvCacheManager& manager, CUstream stream, std::vector<IndexTableSpec> indexTables,
    std::optional<int> maxBatchSize, std::optional<int> maxBlocksPerSeq, std::optional<PageIndexMode> mode,
    std::optional<int> maxTotalInputTokens)
    : mManager(&manager)
    , mCudaStream(stream)
    , mMaxTotalInputTokens(maxTotalInputTokens)
{
    TLLM_CHECK_WITH_INFO(!maxTotalInputTokens.has_value() || *maxTotalInputTokens > 0,
        "Batch: maxTotalInputTokens must be positive when given");
    auto const [derivedRows, derivedBlocks] = deriveBounds(manager.config());
    mMaxBatchSize = maxBatchSize.value_or(derivedRows);
    mMaxBlocksPerSeq = maxBlocksPerSeq.value_or(derivedBlocks);
    TLLM_CHECK_WITH_INFO(mMaxBatchSize > 0 && mMaxBlocksPerSeq > 0,
        "Batch needs a row and block bound; pass them explicitly when the manager declares no constraints");

    mRows = TypedVec<BatchRow, std::shared_ptr<KvCache>>(BatchRow{mMaxBatchSize}, nullptr);

    _buildSparseArenas();

    // Prefer SHARED: one table per life cycle rather than one per layer. The manager
    // answers from configuration alone, so this cannot drift with what a member is doing.
    mIndexModes = TypedVec<LifeCycleId, PageIndexMode>(mManager->numLifeCycles(), PageIndexMode::PER_LAYER);
    mIndexModeNamed = mode.has_value();
    for (LifeCycleId lcId{0}; lcId < mManager->numLifeCycles(); ++lcId)
    {
        mIndexModes[lcId]
            = mode.value_or(manager.supportsIndexMode(PageIndexMode::SHARED, lcId) == std::optional<bool>(true)
                    ? PageIndexMode::SHARED
                    : PageIndexMode::PER_LAYER);
    }

    // After the modes, which decide where a life cycle puts its base, and before the staging
    // block, whose tail is sized by how many scratch slots the tables say a row can hold.
    _buildIndexTables(indexTables);
    _checkSparseRunsAreAddressable();
    auto tables = _layoutIndexTables();
    auto const scratchElements = tables.value().value_or(0);
    _layoutBufs(scratchElements);
    mUploadBytes = static_cast<size_t>(mBuf[BufName::kScratch].end);

    // Page tables and watermarks are published together, so a kernel cannot read one step's
    // table beside another step's lengths -- an out-of-range read rather than a stale
    // answer.
    mStaging = StaticHostMem(mUploadBytes);
    auto* staging = reinterpret_cast<int32_t*>(mStaging.address());
    // A row that no member occupies must still read as "no such block" rather than as slot
    // zero, since the fetch has no other way to tell them apart.
    std::ranges::fill(_buf<int32_t>(BufName::kPageTables), kBadPageIndex.value());
    // Watermarks of a free row are zero: a request with nothing eligible.
    for (auto const name : {BufName::kHistory, BufName::kCapacity})
    {
        std::ranges::fill(_buf<int32_t>(name), 0);
    }
    // No row holds a lease yet, which is what an unpublished batch should read as. Zero
    // would name arena slot zero.
    std::ranges::fill(_buf<SparseLeaseIndex>(BufName::kLeases), kNoSparseLease);

    // The scratch slot metadata is the block's tail, so one transfer per step carries it
    // along with the page tables rather than needing a second.
    mScratchStaging = _buf<int32_t>(BufName::kScratch).data();
    if (scratchElements > 0)
    {
        // A zero range leaves every block on the plain path, which is what a member holding
        // no scratch looks like. The slots behind the ranges hold the sentinel instead,
        // since zero there would name slot zero.
        auto const ranges
            = 2 * static_cast<size_t>(mManager->numLifeCycles().value()) * static_cast<size_t>(mMaxBatchSize);
        std::fill(mScratchStaging, mScratchStaging + ranges, 0);
        std::fill(mScratchStaging + ranges, mScratchStaging + scratchElements, kBadPageIndex.value());
    }

    // The device staging takes the whole block, so the host-to-device crossing is one
    // linear copy that can overlap the previous step. The mirror takes only the tail a
    // captured kernel reads: the page tables and the scratch are read from the staging, by
    // the conversion, which runs fresh each publish and so needs no snapshot of its own.
    mMirror = allocateDeviceMemory(_mirrorBytes());
    mDeviceStaging = allocateDeviceMemory(mUploadBytes);
    auto* const deviceStaging = reinterpret_cast<std::byte*>(mDeviceStaging.get());
    cuCheck(cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(mMirror.get()), _buf<int32_t>(BufName::kHistory).data(),
        _mirrorBytes(), cudaStream()));
    cuCheck(cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(deviceStaging), staging, mUploadBytes, cudaStream()));

    // Hands the layout its slice of the block above and lets it build the descriptors,
    // which name both that block's head and its scratch tail.
    tables.send(reinterpret_cast<int32_t*>(_deviceBufAt(BufName::kScratch)));
    TLLM_CHECK_WITH_INFO(tables.done(), "The index table layout asked for memory twice");

    {
        // Each generator yields what it needs and then fills the slice it is handed, so
        // one allocation serves them all even though the last count is unknown until the
        // last generator has run.
        std::array<LayoutGenerator, 2> layouts{_layoutStepScratch(), _layoutFlushRanges()};
        size_t total = 0;
        for (auto& layout : layouts)
        {
            total += layout.value().value_or(0);
        }
        mDeviceArena = allocateDeviceMemory<int32_t>(total);
        auto* cursor = mDeviceArena.get();
        for (auto& layout : layouts)
        {
            auto const elements = layout.value().value_or(0);
            layout.send(cursor);
            cursor += elements;
            // One round each: a body that asked again would be destroyed suspended, with
            // the rest of its layout silently unrun.
            TLLM_CHECK_WITH_INFO(layout.done(), "A layout generator asked for memory twice");
        }
    }

    // Last: a descriptor names where in the index tables a buffer's converted indices sit.
    _allocateCopyDesc();

    // Everything above is staged on the Batch's stream. A consumer reading this Batch from
    // another stream has nothing ordering it against that, so the state is made visible here.
    cuCheck(cuStreamSynchronize(cudaStream()));
}

void Batch::_allocateCopyDesc()
{
    // One descriptor per buffer, serving both directions, for every life cycle. Small
    // enough to share an allocation and a lifetime.
    std::vector<SparseCopyDesc> host;
    std::vector<size_t> offset(static_cast<size_t>(mManager->numLifeCycles().value()), 0);

    for (LifeCycleId lcId{0}; lcId < mManager->numLifeCycles(); ++lcId)
    {
        auto& scratch = mScratch[lcId];
        if (scratch.selectionWidth == 0)
        {
            continue;
        }
        auto const runs = _sparseCopyTemplates(lcId);
        auto const plan = mManager->_sparseBufferPlan(lcId);
        scratch.numBuffers = static_cast<int32_t>(plan.size());
        offset[static_cast<size_t>(lcId.value())] = host.size();

        for (auto const& run : runs)
        {
            // A run is consecutive buffers of one pool, which can span layers: a pool holds
            // every layer's buffer of one size. An index table spec covers one layer's
            // run of a pool (sparseIndexTableSpecs), so each buffer names its own table,
            // already offset to its entry, and the base that table's indices are relative to.
            for (int32_t i = 0; i < run.numBuffers; ++i)
            {
                auto const buffer = run.bufferBase + i;
                auto const& id = plan[static_cast<size_t>(buffer)].id;
                auto const table = sparseIndexRun(id.layerId, id.role);
                SparseCopyDesc desc;
                desc.hostBase = static_cast<CUdeviceptr>(table.base);
                desc.hostPageStride = run.hostPageStride;
                // Pre-offset to this buffer's entry, so the kernel indexes the row alone.
                desc.pageTable = reinterpret_cast<int32_t const*>(table.table);
                desc.pageTableStride = static_cast<uint32_t>(table.rowStride);
                // Not the row stride: a row now spans every entry of the table, while the
                // blocks this run may address are still one row's worth of one entry.
                desc.numBlocks = static_cast<uint32_t>(mMaxBlocksPerSeq);
                // The buffer's own slot in the pool, folded in so the kernel need not read
                // it through a view it loads per work item.
                desc.deviceBase = run.deviceBase
                    + static_cast<uint64_t>(_sparseArena(lcId).shape().bufferSlotInPool[static_cast<size_t>(buffer)])
                        * run.cellsPerBuffer * run.bufferBytes;
                desc.leaseStride = static_cast<uint32_t>(run.cellsPerLease) * run.bufferBytes;
                desc.cellBytes = run.bufferBytes;
                desc.layout = run.layout;
                desc.width = run.width;

                auto const width = static_cast<uint64_t>(sparseCopyWidthBytes(desc.width));
                TLLM_CHECK_WITH_INFO(desc.hostBase % width == 0 && desc.deviceBase % width == 0,
                    "A sparse buffer's bases must be aligned to the %llu bytes its layout copies at a time; host "
                    "base %llu, device base %llu",
                    static_cast<unsigned long long>(width), static_cast<unsigned long long>(desc.hostBase),
                    static_cast<unsigned long long>(desc.deviceBase));
                host.push_back(desc);
            }
        }
    }

    if (host.empty())
    {
        return;
    }
    mCopyDesc = allocateDeviceMemory<SparseCopyDesc>(host.size());
    auto const* device = mCopyDesc.get();
    cuCheck(cuMemcpyHtoDAsync(
        reinterpret_cast<CUdeviceptr>(device), host.data(), sizeof(SparseCopyDesc) * host.size(), cudaStream()));
    for (LifeCycleId lcId{0}; lcId < mManager->numLifeCycles(); ++lcId)
    {
        auto& scratch = mScratch[lcId];
        if (scratch.selectionWidth != 0)
        {
            scratch.copyDesc = device + offset[static_cast<size_t>(lcId.value())];
        }
    }
    // `host` does not outlive this scope, so the copy out of it has to have landed.
    cuCheck(cuStreamSynchronize(cudaStream()));
}

int32_t Batch::_flushCapacity(LifeCycleId lifeCycle) const
{
    // A row advancing `t` tokens crosses at most `t / unitTokens + 1` units -- the `+1` for
    // the partial unit its watermark starts inside -- so the batch crosses at most
    // `maxTotalInputTokens / unitTokens + maxBatchSize`. The bound is an aggregate one, and
    // usable only because the list is compacted: indexed by row, every row's slice would
    // have to allow for that row taking the whole budget.
    int const perPage = mManager->sparseCacheUnitsPerPage(lifeCycle);
    int const unitTokens = perPage > 0 ? mManager->config().tokensPerBlock / perPage : 0;
    if (mMaxTotalInputTokens.has_value() && unitTokens > 0)
    {
        return *mMaxTotalInputTokens / unitTokens + mMaxBatchSize;
    }
    // Without a budget, a step is assumed to advance capacity by the in-flight headroom in
    // every row at once, which is what a decode step does.
    return mMaxBatchSize * mManager->_sparseCacheMaxInFlightUnits(lifeCycle);
}

Batch::LayoutGenerator Batch::_layoutStepScratchForLifecycle(LifeCycleId lcId)
{
    int const width = mManager->_sparseCacheSelectionWidth(lcId);
    if (!mManager->hasSparseFetchPath(lcId))
    {
        co_return;
    }
    auto const rows = static_cast<size_t>(mMaxBatchSize);
    auto& scratch = mScratch[lcId];
    scratch.selectionWidth = width;
    scratch.maxInFlightUnits = mManager->_sparseCacheMaxInFlightUnits(lcId);
    scratch.numLayers = mManager->_sparseCacheLayerCount(lcId);
    auto const layers = static_cast<size_t>(scratch.numLayers);
    // One list for the fetch, holding the worst case of every row missing its whole
    // selection; one per layer for the write-back, each holding the units the step's token
    // budget can carry across a watermark. The fetch's is the life cycle's, not the
    // selection's: every layer of the selection just resolved reads it, and the next
    // resolve of any selection here overwrites it.
    scratch.fetchWork.capacity = static_cast<int32_t>(rows) * width;
    scratch.fetchWork.numLists = 1;
    scratch.flushWork.capacity = _flushCapacity(lcId);
    scratch.flushWork.numLists = scratch.numLayers;

    auto const listEntries = [](SparseWorkList const& work)
    { return static_cast<size_t>(work.capacity) * static_cast<size_t>(work.numLists); };
    auto const fetchEntries = listEntries(scratch.fetchWork);
    auto const flushEntries = listEntries(scratch.flushWork);
    auto const outSlot = rows * static_cast<size_t>(width);
    auto const inFlightSlot = rows * static_cast<size_t>(scratch.maxInFlightUnits);
    auto const flushCount = rows * layers;

    int32_t* cursor = co_yield outSlot + rows + inFlightSlot + 3 * fetchEntries + 3 * flushEntries + flushCount;

    auto const carve = [&cursor](size_t count)
    {
        auto* ptr = cursor;
        cursor += count;
        return ptr;
    };
    scratch.outSlot = carve(outSlot);
    scratch.missCount = carve(rows);
    scratch.inFlightSlot = carve(inFlightSlot);
    auto const carveList = [&carve](SparseWorkList& work, size_t entries)
    {
        work.row = carve(entries);
        work.ordinal = carve(entries);
        work.slot = carve(entries);
    };
    carveList(scratch.fetchWork, fetchEntries);
    carveList(scratch.flushWork, flushEntries);
    scratch.flushCount = carve(flushCount);
    // -1 so that "no write-back has run" stays distinguishable from "one ran and found
    // nothing to move".
    cuCheck(cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(scratch.flushCount), 0xFFFFFFFFU, flushCount, cudaStream()));
}

Batch::LayoutGenerator Batch::_layoutStepScratch()
{
    mScratch = TypedVec<LifeCycleId, SparseStepScratch>(mManager->numLifeCycles(), SparseStepScratch{});

    // The counters stay here rather than in the leaves: they lead the block so that one
    // clear covers every life cycle's, which a per-life-cycle slice would scatter.
    std::vector<std::pair<LifeCycleId, LayoutGenerator>> leaves;
    size_t mains = 0;
    for (LifeCycleId lcId{0}; lcId < mManager->numLifeCycles(); ++lcId)
    {
        auto leaf = _layoutStepScratchForLifecycle(lcId);
        auto const needed = leaf.value();
        if (!needed.has_value())
        {
            continue;
        }
        mains += *needed;
        // A counter per layer for each of the two lists, even though the fetch's list is
        // one: the list is reused from selection to selection, but the counter has to read
        // zero when a resolve starts. Sized by layers because selections cannot outnumber
        // them.
        mCounterElements += 2 * static_cast<size_t>(mScratch[lcId].numLayers);
        leaves.emplace_back(lcId, std::move(leaf));
    }
    if (leaves.empty())
    {
        co_return;
    }

    int32_t* base = co_yield mains + mCounterElements;

    // Its own allocation: the step's clear covers mCounters, and these have to survive it.
    mHitMiss = allocateDeviceMemory<SparseHitMiss>(1);
    cuCheck(cuMemsetD32Async(
        reinterpret_cast<CUdeviceptr>(mHitMiss.get()), 0U, sizeof(SparseHitMiss) / sizeof(uint32_t), cudaStream()));

    mCounters = base;
    auto* counter = base;
    auto* cursor = base + mCounterElements;
    for (auto& [lcId, leaf] : leaves)
    {
        auto& scratch = mScratch[lcId];
        auto const layers = static_cast<size_t>(scratch.numLayers);
        // The fetch takes one counter per selection against its single list, so a resolve
        // names its own by offsetting this; the write-back's counters match its lists one
        // to one.
        scratch.fetchWork.total = counter;
        counter += layers;
        scratch.flushWork.total = counter;
        counter += layers;
        scratch.hitMiss = mHitMiss.get();

        auto const elements = *leaf.value();
        leaf.send(cursor);
        cursor += elements;
    }
}

SparseStepScratch const& Batch::sparseStepScratch(LifeCycleId lcId) const
{
    TLLM_CHECK_WITH_INFO(lcId < mScratch.size(), "Batch::sparseStepScratch: no such layer group");
    return mScratch[lcId];
}

void Batch::setCudaStream(CUstream stream)
{
    if (stream == cudaStream())
    {
        return;
    }
    // The Batch's own queued work -- the staging copy, the index conversion, the counter
    // resets -- is ordered on the old stream alone, and a Batch with no members still has
    // it. Each member's move below syncs that member, which is a different question.
    CachedCudaEvent handover(reinterpret_cast<CudaStream>(cudaStream()));
    handover.waitInStream(reinterpret_cast<CudaStream>(stream));
    mCudaStream = stream;
    for (auto const& kvCache : mRows)
    {
        if (kvCache != nullptr)
        {
            // The private form: the public one refuses to move a member at all.
            kvCache->_setCudaStream(stream);
        }
    }
}

Batch::~Batch()
{
    // Both buffers are about to go, and transfers on the upload stream still name them.
    _drain();
    KVCM2_POISON_ON_EXCEPT(
        [this]
        {
            for (auto const& kvCache : mRows)
            {
                if (kvCache != nullptr)
                {
                    removeKvCache(*kvCache);
                }
            }
        });
}

size_t Batch::_rowElements() const noexcept
{
    return static_cast<size_t>(mMaxBatchSize) * static_cast<size_t>(mMaxBlocksPerSeq);
}

PageIndexMode Batch::indexMode(LifeCycleId lifeCycleId) const
{
    TLLM_CHECK_WITH_INFO(
        lifeCycleId.value() >= 0 && lifeCycleId < mManager->numLifeCycles(), "Batch::indexMode: unknown layer group");
    return mIndexModes[lifeCycleId];
}

void Batch::_buildIndexTables(std::vector<IndexTableSpec> const& specs)
{
    // The offset of a buffer relative to a chosen base, in page-index units. Mirrors the
    // runtime's kv_offset: a base difference divided by the page stride, which must divide
    // exactly or the two buffers do not sit a whole number of pages apart.
    auto pageOffset = [this](BufferId const& buffer, MemAddress base)
    {
        auto const bufferBase = mManager->getMemPoolBaseAddress(buffer.layerId, buffer.role, PageIndexMode::SHARED);
        auto const stride = static_cast<int64_t>(mManager->getPageStride(buffer.layerId, buffer.role));
        TLLM_CHECK_WITH_INFO(stride > 0, "Page stride must be positive to offset a buffer within its slot");
        auto const delta = static_cast<int64_t>(bufferBase) - static_cast<int64_t>(base);
        TLLM_CHECK_WITH_INFO(delta % stride == 0,
            "Buffers sharing an index table must sit a whole number of pages apart; they cannot share one");
        return static_cast<int>(delta / stride);
    };

    for (auto const& spec : specs)
    {
        TLLM_CHECK_WITH_INFO(!spec.buffers.empty(), "An index table spec names no buffers");

        auto const& first = spec.buffers.front();
        auto const lifeCycle = mManager->_bufferAttr(first.layerId, first.role).lifeCycleId;
        // Substitution replaces one layer's page, so every layer of the life cycle can no
        // longer read one table. SHARED is exactly the claim that they can.
        // Derived SHARED has to be provable from configuration, because nothing else would
        // check it. A named one is the caller's claim about this batch's members, which
        // publish() tests against each of them, so it is allowed to outrun what the manager
        // can prove about requests in general.
        bool const mayHoldScratch = !mManager->supportsIndexMode(PageIndexMode::SHARED, lifeCycle).has_value();
        TLLM_CHECK_WITH_INFO(!mayHoldScratch || mIndexModes[lifeCycle] == PageIndexMode::PER_LAYER || mIndexModeNamed,
            "A life cycle that can hold scratch slots needs PER_LAYER index tables; its layers stop sharing one "
            "set of converted indices as soon as a scratch slot is substituted. Name SHARED on the Batch to "
            "assert no member of it will hold scratch there -- as a decode-only batch does -- and publish() "
            "will hold you to it");
        auto const scale = mManager->getPageIndexScale(first.layerId, first.role);
        auto const mode = mIndexModes[lifeCycle];

        // Under SHARED the base carries the per-layer offset, so the spec's own offsets
        // are relative to its first buffer. Under PER_LAYER the base is the pool group's
        // and carries none, so every offset is absolute within the slot. Derived the same
        // way every reader derives it, so the offsets below and the base a caller is
        // handed later cannot disagree.
        auto const base = _specBase(spec);

        std::vector<int> offsets;
        offsets.reserve(spec.buffers.size());
        for (auto const& buffer : spec.buffers)
        {
            auto const& attr = mManager->_bufferAttr(buffer.layerId, buffer.role);
            // The reference converter emits `expansion` entries per block. Nothing in the
            // tree sets tokensPerBlockOverride, and every attention backend refuses an
            // expansion of anything but one.
            TLLM_CHECK_WITH_INFO(attr.expansion == 1,
                "Index conversion does not implement page index expansion, so a buffer with a tokens-per-block "
                "override cannot have converted index tables");
            TLLM_CHECK_WITH_INFO(attr.lifeCycleId == lifeCycle,
                "An index table spec spans life cycles; their slot ids differ, so no one table can serve it");
            TLLM_CHECK_WITH_INFO(mManager->getPageIndexScale(buffer.layerId, buffer.role) == scale,
                "An index table spec spans page index scales; their converted indices differ, so no one table "
                "can serve it");
            offsets.push_back(pageOffset(buffer, base));
        }

        // Specs coincide when their entries do. Under PER_LAYER the table is per layer,
        // so specs for different layers never merge however alike their entries look.
        auto const sameTable = [&](IndexTable const& table)
        {
            if (table.lifeCycle != lifeCycle || table.scale != scale || table.concatOffsets != offsets)
            {
                return false;
            }
            return mode == PageIndexMode::SHARED || table.served.front().buffers.front().layerId == first.layerId;
        };

        auto found = std::ranges::find_if(mIndexTables, sameTable);
        if (found == mIndexTables.end())
        {
            mIndexTables.push_back(IndexTable{{}, lifeCycle, scale, offsets});
            found = std::prev(mIndexTables.end());
        }
        IndexTableId const tableId{static_cast<int>(std::distance(mIndexTables.begin(), found))};
        found->served.push_back(spec);

        for (size_t slot = 0; slot < spec.buffers.size(); ++slot)
        {
            auto const& buffer = spec.buffers[slot];
            BufferSlot const resolved{tableId, static_cast<int>(slot)};
            auto const [it, inserted] = mBufferSlots.emplace(buffer, resolved);
            // The base is not compared because it cannot differ here: two specs reach one
            // table only when their concatOffsets agree, and an offset is measured from
            // the spec's own base, so equal offsets at equal entries mean equal bases.
            TLLM_CHECK_WITH_INFO(inserted || (it->second.table == resolved.table && it->second.entry == resolved.entry),
                "A buffer appears in two index table specs that resolved differently. Specs may "
                "not overlap: name a buffer in one spec only, and read a subset of a table by "
                "slicing it at the entry indexTableSlot() gives that buffer");
        }
    }
}

Batch::LayoutGenerator Batch::_layoutFlushRanges()
{
    mFlushedHistory = TypedVec<BatchRow, int>(BatchRow{mMaxBatchSize}, 0);

    bool anySparse = false;
    for (LifeCycleId lcId{0}; lcId < mManager->numLifeCycles(); ++lcId)
    {
        anySparse = anySparse || mManager->sparseCacheMaxUnits(lcId) > 0;
    }
    if (!anySparse)
    {
        co_return;
    }

    auto const elements
        = 2 * static_cast<size_t>(mMaxBatchSize) * static_cast<size_t>(mManager->numLifeCycles().value());
    mFlushRanges = co_yield elements;
    // An empty range asks for nothing, which is what a row that never flushes should read.
    cuCheck(cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(mFlushRanges), 0, elements, cudaStream()));
    mFlushRangeStaging.assign(elements, 0);
}

int32_t const* Batch::flushFromUnits(LifeCycleId lifeCycle) const
{
    auto* base = mFlushRanges;
    return base + 2 * static_cast<size_t>(mMaxBatchSize) * static_cast<size_t>(lifeCycle.value());
}

int32_t const* Batch::flushToUnits(LifeCycleId lifeCycle) const
{
    return flushFromUnits(lifeCycle) + mMaxBatchSize;
}

int Batch::flushedHistory(BatchRow row) const
{
    TLLM_CHECK_WITH_INFO(row >= 0 && row < mMaxBatchSize, "Batch::flushedHistory: row is beyond max_batch_size");
    return mFlushedHistory[row];
}

void Batch::flushSparseUnits(std::vector<int> const& newHistoryLengths)
{
    TLLM_CHECK_WITH_INFO(
        newHistoryLengths.size() == mRows.stdSize(), "Batch::flushSparseUnits: one history length per row");
    // A history past the row's capacity names units the row never wrote, so there is no cell
    // to write back from. Checked for every row before anything is recorded, and ahead of the
    // vacuous path, so the record a row leaves behind is one it can reach.
    for (BatchRow row{0}; row < mRows.size(); ++row)
    {
        auto const& kvCache = mRows[row];
        if (kvCache != nullptr)
        {
            TLLM_CHECK_WITH_INFO(newHistoryLengths[toSizeT(row)] <= kvCache->capacity(),
                "Batch::flushSparseUnits: row %d was given history %d against capacity %d; flush a history the "
                "row can take up",
                row.value(), newHistoryLengths[toSizeT(row)], kvCache->capacity());
        }
    }
    if (!mFlushRanges)
    {
        // No sparse life cycle, so nothing is ever in flight and the record is vacuous.
        for (BatchRow row{0}; row < mRows.size(); ++row)
        {
            mFlushedHistory[row] = newHistoryLengths[toSizeT(row)];
        }
        return;
    }

    // Every row is checked before any work is issued, so a move no reservation could cover
    // leaves the batch as it was.
    bool anyWork = false;
    for (LifeCycleId lcId{0}; lcId < mManager->numLifeCycles(); ++lcId)
    {
        int const unitsPerPage = mManager->sparseCacheUnitsPerPage(lcId);
        if (!mManager->hasSparseFetchPath(lcId))
        {
            continue;
        }
        int const unitTokens = mManager->config().tokensPerBlock / unitsPerPage;
        auto const base = static_cast<size_t>(2 * mMaxBatchSize * lcId.value());

        // Bounded in aggregate rather than per row, which is what the compacted list allows
        // and what in-flight batching needs: one prefilling row may cross the whole step's
        // worth of units while the rest cross none.
        int64_t crossed = 0;

        for (BatchRow row{0}; row < mRows.size(); ++row)
        {
            auto const& kvCache = mRows[row];
            int fromUnit = 0;
            int toUnit = 0;
            if (kvCache != nullptr)
            {
                TLLM_CHECK_WITH_INFO(newHistoryLengths[toSizeT(row)] >= kvCache->historyLength(),
                    "Batch::flushSparseUnits: history length cannot be decreased");
                fromUnit = kvCache->historyLength() / unitTokens;
                toUnit = newHistoryLengths[toSizeT(row)] / unitTokens;

                crossed += toUnit - fromUnit;
                anyWork = anyWork || toUnit > fromUnit;
            }
            mFlushRangeStaging[base + toSizeT(row)] = fromUnit;
            mFlushRangeStaging[base + static_cast<size_t>(mMaxBatchSize) + toSizeT(row)] = toUnit;
        }

        TLLM_CHECK_WITH_INFO(crossed <= mScratch[lcId].flushWork.capacity,
            "Batch::flushSparseUnits: the batch crossed more units than the step's token budget allows, so "
            "the write-back list could not name them all; construct the Batch with maxTotalInputTokens");
    }

    for (BatchRow row{0}; row < mRows.size(); ++row)
    {
        mFlushedHistory[row] = newHistoryLengths[toSizeT(row)];
    }
    if (!anyWork)
    {
        // Nothing crossed anywhere, so the ranges on the device are stale but unread.
        return;
    }

    auto* const upload = mUploadStream.handle();
    if (!mFlushRead.isClosed())
    {
        // The previous step's write-back read these same ranges; its reads must retire
        // before the copy below overwrites them.
        mFlushRead.waitInStream(reinterpret_cast<CudaStream>(upload));
        mFlushRead.close();
    }
    copyToDeviceSyncSource(
        reinterpret_cast<CUdeviceptr>(mFlushRanges), mFlushRangeStaging.data(), mFlushRangeStaging.size(), upload);
    auto staged = mUploadStream.recordEvent();
    staged.waitInStream(reinterpret_cast<CudaStream>(cudaStream()));

    for (LifeCycleId lcId{0}; lcId < mManager->numLifeCycles(); ++lcId)
    {
        if (!mManager->hasSparseFetchPath(lcId))
        {
            continue;
        }
        if (mFlushCountersDirty)
        {
            // Narrower than publish()'s clear, which is the one a step normally runs on:
            // only this life cycle's write-back totals, and only when something is left in
            // them. Ordered on the Batch's stream behind whatever last read them.
            auto const& scratch = mScratch[lcId];
            cuCheck(cuMemsetD32Async(reinterpret_cast<CUdeviceptr>(scratch.flushWork.total), 0U,
                static_cast<size_t>(scratch.flushWork.numLists), cudaStream()));
        }
        writeBackSparseUnits(lcId, cudaStream());
    }
    mFlushCountersDirty = true;
    mFlushRead = CachedCudaEvent{reinterpret_cast<CudaStream>(cudaStream())};
}

void Batch::_checkSparseRunsAreAddressable() const
{
    for (LifeCycleId lcId{0}; lcId < mManager->numLifeCycles(); ++lcId)
    {
        auto const plan = mManager->_sparseBufferPlan(lcId);
        if (plan.empty())
        {
            continue;
        }
        for (size_t begin = 0; begin < plan.size();)
        {
            size_t end = begin;
            while (end < plan.size() && plan[end].poolIndex == plan[begin].poolIndex
                && plan[end].layerRow == plan[begin].layerRow)
            {
                ++end;
            }

            auto const slotOf = [this](SparseBufferPlan const& entry)
            {
                auto const found = mBufferSlots.find(entry.id);
                TLLM_CHECK_WITH_INFO(found != mBufferSlots.end(),
                    "A sparse buffer has no index table spec; the fetch reads converted indices, so every "
                    "sparse buffer needs one");
                return found->second;
            };

            auto const baseOf = [this](SparseBufferPlan const& entry)
            { return _specBase(_specOf(entry.id.layerId, entry.id.role, "Batch::_checkSparseRunsAreAddressable")); };

            auto const first = slotOf(plan[begin]);
            auto const firstBase = baseOf(plan[begin]);
            for (size_t i = begin + 1; i < end; ++i)
            {
                auto const slot = slotOf(plan[i]);
                // Both, because neither implies the other: a merged table serves several
                // specs, each with its own base, so one table is not one base pointer.
                TLLM_CHECK_WITH_INFO(slot.table == first.table && baseOf(plan[i]) == firstBase,
                    "Sparse buffers sharing a pool are copied in one launch, from one base pointer, so they must "
                    "share one index table spec");
                TLLM_CHECK_WITH_INFO(slot.entry == first.entry + static_cast<int>(i - begin),
                    "Sparse buffers sharing a pool must sit at consecutive entries of their request, in the order "
                    "the pool holds them");
            }
            begin = end;
        }
    }
}

Batch::LayoutGenerator Batch::_layoutIndexTables()
{
    if (mIndexTables.empty())
    {
        co_return;
    }

    // Scratch-capable life cycles sort last so that one launch covers the plain tables and
    // one the substituting ones, rather than a kernel branching per table.
    auto const mayHoldScratch = [this](IndexTable const& table)
    { return !mManager->supportsIndexMode(PageIndexMode::SHARED, table.lifeCycle).has_value(); };

    std::vector<size_t> order(mIndexTables.size());
    std::iota(order.begin(), order.end(), size_t{0});
    std::ranges::stable_partition(order, [&](size_t index) { return !mayHoldScratch(mIndexTables[index]); });

    std::vector<IndexTable> sorted;
    std::vector<int> remap(mIndexTables.size());
    sorted.reserve(mIndexTables.size());
    for (size_t position = 0; position < order.size(); ++position)
    {
        remap[order[position]] = static_cast<int>(position);
        sorted.push_back(std::move(mIndexTables[order[position]]));
    }
    mIndexTables = std::move(sorted);
    for (auto& [key, slot] : mBufferSlots)
    {
        slot.table = IndexTableId{remap[static_cast<size_t>(slot.table.value())]};
    }
    mNumPlainTables = static_cast<int>(std::count_if(
        mIndexTables.begin(), mIndexTables.end(), [&](IndexTable const& table) { return !mayHoldScratch(table); }));

    // The most slots a request could need for one life cycle, over this batch's own tables:
    // the stride every (row, life cycle) of the scratch staging gets. Zero where no member's
    // step can outrun the window, and so adds no already-stale block.
    for (size_t index = static_cast<size_t>(mNumPlainTables); index < mIndexTables.size(); ++index)
    {
        auto const slots
            = mManager->_scratchSlotsFor(mIndexTables[index].lifeCycle, mMaxBlocksPerSeq, mMaxTotalInputTokens);
        mMaxScratchSlots = std::max(mMaxScratchSlots, slots);
    }

    // With no slot any member can reach there is nothing to substitute, so every table takes
    // the plain path and the scratch kernel never runs over a tail that is not there.
    if (mMaxScratchSlots == 0)
    {
        mNumPlainTables = static_cast<int>(mIndexTables.size());
    }

    // Everything above only sized the scratch tail; everything below names addresses within
    // the block the caller allocates from it.
    mScratchDevice = co_yield _scratchElements();

    auto const rowBlocks = static_cast<size_t>(mMaxBatchSize) * static_cast<size_t>(mMaxBlocksPerSeq);
    size_t elements = 0;
    for (auto& table : mIndexTables)
    {
        table.elementOffset = elements;
        elements += table.concatOffsets.size() * rowBlocks;
        mMaxTableEntries = std::max(mMaxTableEntries, static_cast<int>(table.concatOffsets.size()));
    }

    mIndexTableMemory = allocateDeviceMemory<int32_t>(elements);
    auto const memory = reinterpret_cast<CUdeviceptr>(mIndexTableMemory.get());
    // A table a publish has not reached yet must read as absent rather than as slot zero.
    cuCheck(cuMemsetD32Async(memory, static_cast<unsigned int>(kBadPageIndex.value()), elements, cudaStream()));

    // The descriptors and the offsets they point at share one allocation: the offsets are a
    // handful of ints per table and would otherwise be an allocation each.
    size_t offsetElements = 0;
    for (auto const& table : mIndexTables)
    {
        offsetElements += table.concatOffsets.size();
    }
    size_t const descBytes = sizeof(IndexTableDesc) * mIndexTables.size();
    size_t const offsetsBytes = sizeof(int32_t) * offsetElements;

    mIndexTableDescs = allocateDeviceMemory(descBytes + offsetsBytes);
    auto const descs = reinterpret_cast<CUdeviceptr>(mIndexTableDescs.get());

    std::vector<int32_t> offsetHost;
    offsetHost.reserve(offsetElements);
    std::vector<IndexTableDesc> descHost;
    descHost.reserve(mIndexTables.size());

    auto const offsetsBase = descs + descBytes;
    auto const rawBase = _deviceBufAt(BufName::kPageTables);
    for (auto const& table : mIndexTables)
    {
        IndexTableDesc desc;
        // The raw slot ids of this table's life cycle, in the staging the upload filled.
        desc.raw = reinterpret_cast<int32_t const*>(
            rawBase + sizeof(int32_t) * _rowElements() * static_cast<size_t>(table.lifeCycle.value()));
        desc.converted = reinterpret_cast<int32_t*>(memory + sizeof(int32_t) * table.elementOffset);
        desc.offsets = reinterpret_cast<int32_t const*>(offsetsBase + sizeof(int32_t) * offsetHost.size());
        desc.numBuffers = static_cast<int32_t>(table.concatOffsets.size());
        desc.scale = table.scale;
        if (static_cast<int>(descHost.size()) >= mNumPlainTables)
        {
            auto const row = static_cast<size_t>(mMaxBatchSize);
            auto const lc = static_cast<size_t>(table.lifeCycle.value());
            auto const begs = reinterpret_cast<CUdeviceptr>(mScratchDevice);
            auto const groups = static_cast<size_t>(mManager->numLifeCycles().value());
            desc.scratchBegs = reinterpret_cast<int32_t const*>(begs + sizeof(int32_t) * lc * row);
            desc.scratchEnds = reinterpret_cast<int32_t const*>(begs + sizeof(int32_t) * (groups * row + lc * row));
            desc.scratchSlots = reinterpret_cast<int32_t const*>(
                begs + sizeof(int32_t) * (2 * groups * row + lc * row * static_cast<size_t>(mMaxScratchSlots)));
            desc.scratchSlotStride = mMaxScratchSlots;
            auto const& firstBuffer = table.served.front().buffers.front();
            desc.scratchPagesPerBlock
                = mManager->getPageIndexConverter(firstBuffer.layerId, firstBuffer.role).scratchPagesPerBlock;
        }
        descHost.push_back(desc);
        offsetHost.insert(offsetHost.end(), table.concatOffsets.begin(), table.concatOffsets.end());
    }

    cuCheck(cuMemcpyHtoDAsync(descs, descHost.data(), descBytes, cudaStream()));
    cuCheck(cuMemcpyHtoDAsync(offsetsBase, offsetHost.data(), offsetsBytes, cudaStream()));
    // Neither host vector outlives this scope, so the copies out of them have to have landed.
    cuCheck(cuStreamSynchronize(cudaStream()));
}

void Batch::_clearIndexRow(BatchRow row)
{
    if (mIndexTables.empty())
    {
        return;
    }
    auto const* const descs = reinterpret_cast<IndexTableDesc const*>(mIndexTableDescs.get());
    launchClearIndexRow(descs, static_cast<int32_t>(mIndexTables.size()), mMaxTableEntries, row.value(),
        mMaxBlocksPerSeq, mManager->_smCount(), cudaStream());
}

void Batch::_convertIndices(CUstream stream)
{
    if (mIndexTables.empty())
    {
        return;
    }
    auto const* const descs = reinterpret_cast<IndexTableDesc const*>(mIndexTableDescs.get());
    auto const* const capacity = reinterpret_cast<int32_t const*>(_mirrorAt(BufName::kCapacity));
    int32_t const tokensPerBlock = mManager->config().tokensPerBlock;
    launchIndexConversion(descs, mNumPlainTables, mMaxTableEntries, mMaxBatchSize, mMaxBlocksPerSeq,
        static_cast<int32_t>(mMaxBlocksPerSeq), capacity, tokensPerBlock, mManager->_smCount(), stream);
    launchIndexConversionWithScratch(descs + mNumPlainTables,
        static_cast<int32_t>(mIndexTables.size()) - mNumPlainTables, mMaxTableEntries, mMaxBatchSize, mMaxBlocksPerSeq,
        static_cast<int32_t>(mMaxBlocksPerSeq), capacity, tokensPerBlock, mManager->_smCount(), stream);
}

CUdeviceptr Batch::indexTableAddress(IndexTableId tableId) const
{
    TLLM_CHECK_WITH_INFO(tableId.value() >= 0 && tableId < numIndexTables(), "Batch::indexTableAddress: unknown table");
    return reinterpret_cast<CUdeviceptr>(mIndexTableMemory.get())
        + sizeof(int32_t) * mIndexTables[static_cast<size_t>(tableId.value())].elementOffset;
}

ConvertedIndexTable Batch::convertedIndexTable(IndexTableId tableId) const
{
    TLLM_CHECK_WITH_INFO(
        tableId.value() >= 0 && tableId < numIndexTables(), "Batch::convertedIndexTable: unknown table");
    ConvertedIndexTable view;
    view.data = indexTableAddress(tableId);
    view.numRows = mMaxBatchSize;
    // A table's entries are its requests' buffers: the offsets it converts with, one each.
    view.numBuffers = static_cast<int32_t>(mIndexTables[static_cast<size_t>(tableId.value())].concatOffsets.size());
    view.blocksPerRow = mMaxBlocksPerSeq;
    return view;
}

Batch::SparseIndexRun Batch::sparseIndexRun(LayerId layerId, DataRole const& role) const
{
    auto const& slot = _bufferSlot(layerId, role, "Batch::sparseIndexRun");
    auto const tableId = slot.table;
    SparseIndexRun run;
    // Row-major in the entries, so a row costs the whole table's entries and an entry costs
    // one row of blocks.
    auto const numBuffers = static_cast<int>(mIndexTables[static_cast<size_t>(tableId.value())].concatOffsets.size());
    run.bufferStride = mMaxBlocksPerSeq;
    run.rowStride = numBuffers * mMaxBlocksPerSeq;
    run.table = indexTableAddress(tableId) + sizeof(int32_t) * static_cast<size_t>(slot.entry) * run.bufferStride;
    run.base = _specBase(_specOf(layerId, role, "Batch::sparseIndexRun"));
    return run;
}

size_t Batch::_scratchElements() const noexcept
{
    // No table takes the substituting path, so nothing would read a range or a slot.
    if (mMaxScratchSlots == 0)
    {
        return 0;
    }
    auto const rows = static_cast<size_t>(mMaxBatchSize);
    auto const groups = static_cast<size_t>(mManager->numLifeCycles().value());
    // [begs][ends][slots], each indexed by life cycle then row.
    return 2 * groups * rows + groups * rows * static_cast<size_t>(mMaxScratchSlots);
}

void Batch::_stageScratch()
{
    if (mMaxScratchSlots == 0)
    {
        return;
    }
    auto const rows = static_cast<size_t>(mMaxBatchSize);
    auto const groups = static_cast<size_t>(mManager->numLifeCycles().value());
    auto* const begs = mScratchStaging;
    auto* const ends = begs + groups * rows;

    // Only the ranges: a member's slot ids reach its row as they change, through the
    // buffer bound when it joined.
    for (BatchRow row{0}; row < mRows.size(); ++row)
    {
        auto const& kvCache = mRows[row];
        for (LifeCycleId lcId{0}; lcId < mManager->numLifeCycles(); ++lcId)
        {
            auto const flat = static_cast<size_t>(lcId.value()) * rows + toSizeT(row);

            // A free row, a suspended member, or one holding no scratch here all leave an
            // empty range, which the kernel reads as "every block takes the plain path".
            auto const range = kvCache != nullptr && kvCache->isActive() ? kvCache->getSwaScratchRange(lcId)
                                                                         : HalfOpenRange<BlockOrdinal>{};
            begs[flat] = range.beg.value();
            ends[flat] = range.end.value();
            if (range.empty())
            {
                continue;
            }
            TLLM_CHECK_DEBUG_WITH_INFO(range.end.value() - range.beg.value()
                    <= mManager->_maxScratchBlocks(lcId, mMaxBlocksPerSeq, mMaxTotalInputTokens),
                "A request's scratch range outruns the blocks the batch was sized for");
        }
    }
}

IndexTableId Batch::numIndexTables() const noexcept
{
    return IndexTableId{static_cast<int>(mIndexTables.size())};
}

std::vector<IndexTableSpec> const& Batch::indexTable(IndexTableId tableId) const
{
    TLLM_CHECK_WITH_INFO(tableId.value() >= 0 && tableId < numIndexTables(), "Batch::indexTable: unknown table");
    return mIndexTables[static_cast<size_t>(tableId.value())].served;
}

Batch::BufferSlot const& Batch::_bufferSlot(LayerId layerId, DataRole const& role, char const* api) const
{
    auto const found = mBufferSlots.find(BufferId{layerId, role});
    TLLM_CHECK_WITH_INFO(found != mBufferSlots.end(), std::string(api) + ": no index table spec named this buffer");
    return found->second;
}

IndexTableId Batch::indexTableOf(IndexTableSpec const& spec) const
{
    for (size_t i = 0; i < mIndexTables.size(); ++i)
    {
        for (auto const& served : mIndexTables[i].served)
        {
            if (served.buffers == spec.buffers)
            {
                return IndexTableId{static_cast<int>(i)};
            }
        }
    }
    throw std::out_of_range("Batch::indexTableOf: no index table was built for this spec");
}

IndexTableSpec const& Batch::_specOf(LayerId layerId, DataRole const& role, char const* api) const
{
    auto const& slot = _bufferSlot(layerId, role, api);
    BufferId const wanted{layerId, role};
    for (auto const& served : mIndexTables[static_cast<size_t>(slot.table.value())].served)
    {
        if (std::ranges::find(served.buffers, wanted) != served.buffers.end())
        {
            return served;
        }
    }
    // Unreachable while construction records a slot only for a buffer a spec names.
    throw LogicError(std::string(api) + ": a buffer's table does not serve the spec that named it");
}

MemAddress Batch::_specBase(IndexTableSpec const& spec) const
{
    auto const& first = spec.buffers.front();
    auto const lifeCycle = mManager->_bufferAttr(first.layerId, first.role).lifeCycleId;
    return mManager->getMemPoolBaseAddress(first.layerId, first.role, mIndexModes[lifeCycle]);
}

PoolSpec Batch::poolSpecFor(IndexTableSpec const& spec) const
{
    // Membership first: a spec the Batch never built has no base.
    static_cast<void>(indexTableOf(spec));

    auto const& first = spec.buffers.front();
    PoolSpec out;
    out.base = _specBase(spec);
    out.pageStride = mManager->getPageStride(first.layerId, first.role);
    for (auto const& buffer : spec.buffers)
    {
        // One stride for the spec, because a reader has only one: the entries carry each
        // buffer's offset in pages, which is a count of this stride and of no other.
        TLLM_CHECK_WITH_INFO(
            static_cast<size_t>(mManager->getPageStride(buffer.layerId, buffer.role)) == out.pageStride,
            "An index table spec spans page strides, so one base cannot address its buffers");
    }
    return out;
}

std::pair<IndexTableId, int> Batch::indexTableSlot(LayerId layerId, DataRole const& role) const
{
    auto const& slot = _bufferSlot(layerId, role, "Batch::indexTableSlot");
    return {slot.table, slot.entry};
}

MemAddress Batch::indexTableBase(LayerId layerId, DataRole const& role) const
{
    return _specBase(_specOf(layerId, role, "Batch::indexTableBase"));
}

void Batch::_drain() noexcept
{
    KVCM2_POISON_ON_EXCEPT(
        [this]
        {
            waitPublished();
            if (!mDeviceStagingConsumed.isClosed())
            {
                mDeviceStagingConsumed.synchronize();
                mDeviceStagingConsumed.close();
            }
        });
}

void Batch::_layoutBufs(size_t scratchElements) noexcept
{
    auto const rows = static_cast<size_t>(mMaxBatchSize);
    auto const groups = static_cast<size_t>(mManager->numLifeCycles().value());

    struct Spec
    {
        BufName name;
        size_t bytes;
        size_t align;
    };

    std::array<Spec, kBufCount> const specs{{
        {BufName::kPageTables, sizeof(int32_t) * _rowElements() * groups, alignof(int32_t)},
        {BufName::kHistory, sizeof(int32_t) * rows, alignof(int32_t)},
        {BufName::kCapacity, sizeof(int32_t) * rows, alignof(int32_t)},
        {BufName::kLeases, sizeof(SparseLeaseIndex) * rows, alignof(SparseLeaseIndex)},
        {BufName::kScratch, sizeof(int32_t) * scratchElements, alignof(int32_t)},
    }};

    ptrdiff_t offset = 0;
    for (auto const& spec : specs)
    {
        auto const align = static_cast<ptrdiff_t>(std::max(spec.align, kBufMinAlign));
        offset = (offset + align - 1) / align * align;
        mBuf[spec.name] = HalfOpenRange<ptrdiff_t>{offset, offset + static_cast<ptrdiff_t>(spec.bytes)};
        offset = mBuf[spec.name].end;
    }
}

SparseLeaseIndex const* Batch::_leaseMirror() const noexcept
{
    return reinterpret_cast<SparseLeaseIndex const*>(sparseLeases());
}

size_t Batch::_mirrorBytes() const noexcept
{
    // From the histories through the leases: the page tables ahead of them are read from the
    // staging, and the scratch behind them is the device staging's own tail.
    return static_cast<size_t>(mBuf[BufName::kLeases].end - mBuf[BufName::kHistory].beg);
}

CUdeviceptr Batch::_deviceBufAt(BufName name) const noexcept
{
    return reinterpret_cast<CUdeviceptr>(mDeviceStaging.get()) + static_cast<size_t>(mBuf[name].beg);
}

CUdeviceptr Batch::_mirrorAt(BufName name) const noexcept
{
    // The mirror carries the block from the histories on, so a buffer addresses it rebased.
    return reinterpret_cast<CUdeviceptr>(mMirror.get())
        + static_cast<size_t>(mBuf[name].beg - mBuf[BufName::kHistory].beg);
}

void Batch::_stageRowState()
{
    auto const history = _buf<int32_t>(BufName::kHistory);
    auto const capacity = _buf<int32_t>(BufName::kCapacity);
    auto const leases = _buf<SparseLeaseIndex>(BufName::kLeases);

    // A row's page table is mMaxBlocksPerSeq entries wide, and the index conversion walks it
    // as far as the capacity published here reaches.
    auto const capacityLimit = static_cast<int64_t>(mMaxBlocksPerSeq) * mManager->config().tokensPerBlock;

    for (BatchRow row{0}; row < mRows.size(); ++row)
    {
        auto const& kvCache = mRows[row];
        history[toSizeT(row)] = kvCache == nullptr ? 0 : kvCache->historyLength();
        capacity[toSizeT(row)] = kvCache == nullptr ? 0 : kvCache->capacity();
        TLLM_CHECK_WITH_INFO(capacity[toSizeT(row)] <= capacityLimit,
            "Batch: row %d has capacity %d tokens, past the %lld its page table holds", row.value(),
            capacity[toSizeT(row)], static_cast<long long>(capacityLimit));

        // A free row, or a member holding no reservation, publishes no lease, which the
        // kernels read as "nothing to resolve here". One slot serves every life cycle.
        leases[toSizeT(row)] = kvCache == nullptr ? kNoSparseLease : leaseOf(row);
    }
}

void Batch::_seedRowWatermarks(BatchRow row, int historyLength, int capacity)
{
    auto const index = toSizeT(row);
    _buf<int32_t>(BufName::kHistory)[index] = historyLength;
    _buf<int32_t>(BufName::kCapacity)[index] = capacity;
}

int32_t* Batch::_stagingRow(LifeCycleId lifeCycleId, BatchRow row) const
{
    auto* base = _buf<int32_t>(BufName::kPageTables).data();
    return base + static_cast<int64_t>(_rowElements()) * lifeCycleId.value()
        + static_cast<int64_t>(mMaxBlocksPerSeq) * row.value();
}

int32_t* Batch::_scratchSlotRow(LifeCycleId lifeCycleId, BatchRow row) const
{
    auto const rows = static_cast<size_t>(mMaxBatchSize);
    auto const groups = static_cast<size_t>(mManager->numLifeCycles().value());
    auto const flat = static_cast<size_t>(lifeCycleId.value()) * rows + toSizeT(row);
    return mScratchStaging + 2 * groups * rows + flat * static_cast<size_t>(mMaxScratchSlots);
}

BatchRow Batch::addKvCache(KvCache& kvCache, std::optional<BatchRow> requestedRow)
{
    if (kvCache.mBatch != nullptr)
    {
        throw LogicError("A KvCache belongs to at most one Batch");
    }
    if (kvCache.mManager.get() != mManager)
    {
        // Rows, arena leases and index tables are all this manager's. A member from another
        // would be addressed through them and read memory that was never its own.
        throw LogicError("A KvCache may only join a Batch of the manager that created it");
    }
    if (kvCache.beamWidth() != BeamIndex{1})
    {
        // Only beam 0 is mirrored below, so a wider request would publish one beam's pages
        // and have every beam read them.
        throw LogicError("Batch does not support beam search yet; a member must have beam width 1");
    }
    if (!kvCache.isActive())
    {
        // Membership means the member is being stepped, which is what lets its arena lease,
        // its row and its active state be one thing rather than three that can disagree.
        throw LogicError("A KvCache must be resumed before it joins a Batch");
    }
    // Binding a row writes it, so the last publish must be done reading it.
    waitPublished();
    BatchRow row{0};
    if (!requestedRow.has_value())
    {
        auto const free = std::find(mRows.begin(), mRows.end(), nullptr);
        if (free == mRows.end())
        {
            throw LogicError("Batch is full; it was sized for max_batch_size rows");
        }
        row = BatchRow{static_cast<int>(std::distance(mRows.begin(), free))};
    }
    else
    {
        row = *requestedRow;
        TLLM_CHECK_WITH_INFO(
            row != kNoBatchRow && row < mMaxBatchSize, "Batch::addKvCache: row is beyond max_batch_size");
        if (mRows[row] != nullptr)
        {
            throw LogicError("Batch::addKvCache: row is already occupied");
        }
    }

    // The bind copies the current indices in and keeps writing there until the member
    // leaves, so the row needs no refresh of its own and a step that only advances its
    // members has nothing to rebind.
    for (LifeCycleId lc{0}; lc < mManager->numLifeCycles(); ++lc)
    {
        // Beam 0 only; addKvCache rejects anything wider.
        kvCache.setBasePageIndexBuf(kDefaultBeamIndex, lc, _stagingRow(lc, row), mMaxBlocksPerSeq);
        if (mMaxScratchSlots > 0)
        {
            kvCache.setScratchSlotBuf(lc, _scratchSlotRow(lc, row), mMaxScratchSlots);
        }
    }

    _seedRowWatermarks(row, kvCache.historyLength(), kvCache.capacity());
    // A seating member brings no in-flight units: its lease is fresh, so everything its
    // history names already lives in the pages behind it. Seeding from the history is what
    // lets the row be reused, since the departing member leaves this equal to its own.
    mFlushedHistory[row] = kvCache.historyLength();

    // Joining is a stream change, so it carries the same cross-stream sync a direct move
    // would: work the member already enqueued has to retire before the Batch's stream
    // reaches anything that reads it.
    kvCache._setCudaStream(cudaStream());

    // Owning from here: the member cannot be destroyed under the Batch, so a caller may
    // keep only the Batch and get the reference back when it leaves.
    mRows[row] = kvCache.shared_from_this();
    kvCache.mBatch = this;
    kvCache.mBatchRow = row;
    return row;
}

std::shared_ptr<KvCache> const& Batch::operator[](BatchRow row) const
{
    return mRows.at(row);
}

std::shared_ptr<KvCache> Batch::removeKvCache(BatchRow row)
{
    TLLM_CHECK_WITH_INFO(row >= 0 && row < mMaxBatchSize, "Batch::removeKvCache: row is beyond max_batch_size");
    auto seated = mRows[row];
    return seated ? removeKvCache(*seated) : nullptr;
}

std::shared_ptr<KvCache> Batch::removeKvCache(KvCache& kvCache)
{
    if (kvCache.mBatch != this)
    {
        throw LogicError("KvCache is not a member of this Batch");
    }
    waitPublished();
    BatchRow const row = *kvCache.mBatchRow;
    // Checked before anything is touched, so a rejected removal leaves the batch as it was.
    //
    // Flushing a history the member never takes up is pointless, so the two agreeing is what
    // a member that has written back everything it means to keep looks like. The leases
    // released below discard the GPU cache, and the next occupant seeds its own watermark
    // from its own history, so anything still pending here has no later chance to be noticed.
    if (mManager->hasSparseCache())
    {
        TLLM_CHECK_WITH_INFO(kvCache.historyLength() == mFlushedHistory[row],
            "A sparse KvCache leaving row %d has history %d against %d written back; call "
            "Batch::flushSparseUnits() with the current history before removing it",
            row.value(), kvCache.historyLength(), mFlushedHistory[row]);
    }
    // Held until the end: clearing the row below drops the Batch's reference, and the rest
    // of this runs against the member.
    auto leaving = std::move(mRows[row]);
    // Membership ends first, so the unbinding below goes through the same setter a caller
    // would use. Nothing binds a seated member's buffer, including the Batch.
    kvCache.mBatch = nullptr;
    kvCache.mBatchRow = std::nullopt;
    for (LifeCycleId lc{0}; lc < mManager->numLifeCycles(); ++lc)
    {
        // Hands the indices back to the cache's own storage before the row is reused.
        kvCache.setBasePageIndexBuf(kDefaultBeamIndex, lc, nullptr, 0);
        if (mMaxScratchSlots > 0)
        {
            kvCache.setScratchSlotBuf(lc, nullptr, 0);
            auto* slots = _scratchSlotRow(lc, row);
            std::fill(slots, slots + mMaxScratchSlots, kBadPageIndex.value());
        }
        auto* staged = _stagingRow(lc, row);
        std::fill(staged, staged + mMaxBlocksPerSeq, kBadPageIndex.value());
    }
    _clearIndexRow(row);
    // The row keeps its lease and hands it back clean, as it does the converted indices
    // above: flushSparseUnits() has written back anything worth keeping, so what is left
    // names units no later occupant has any claim on.
    if (mManager->hasSparseCache())
    {
        for (LifeCycleId lc{0}; lc < mManager->numLifeCycles(); ++lc)
        {
            _sparseArena(lc).initialiseLease(leaseOf(row), cudaStream());
        }
    }
    _seedRowWatermarks(row, 0, 0);
    mRows[row] = nullptr;
    return leaving;
}

void Batch::move(BatchRow dstRow, BatchRow srcRow)
{
    TLLM_CHECK_WITH_INFO(dstRow >= 0 && dstRow < mMaxBatchSize && srcRow >= 0 && srcRow < mMaxBatchSize,
        "Batch::move: row is beyond max_batch_size");
    // Checked before anything is touched, so a rejected move leaves the batch as it was.
    // dstRow == srcRow fails here too: a row cannot be occupied and empty at once.
    if (mRows[srcRow] == nullptr)
    {
        throw LogicError("Batch::move: source row holds no member");
    }
    if (mRows[dstRow] != nullptr)
    {
        throw LogicError("Batch::move: destination row is already occupied");
    }

    waitPublished();

    for (LifeCycleId lc{0}; lc < mManager->numLifeCycles(); ++lc)
    {
        auto* const from = _stagingRow(lc, srcRow);
        auto* const to = _stagingRow(lc, dstRow);
        // Swapped rather than copied so the vacated row ends up holding what the empty
        // destination held: every entry kBadPageIndex, which is what a free row must publish.
        std::swap_ranges(from, from + mMaxBlocksPerSeq, to);
        if (mMaxScratchSlots > 0)
        {
            // The scratch slots are a second per-row table the member writes through, and
            // _stageScratch() publishes them from the row it occupies. They travel with it
            // for the same reason its page indices do.
            auto* const scratchFrom = _scratchSlotRow(lc, srcRow);
            auto* const scratchTo = _scratchSlotRow(lc, dstRow);
            std::swap_ranges(scratchFrom, scratchFrom + mMaxScratchSlots, scratchTo);
        }
    }
    std::swap(mRows[dstRow], mRows[srcRow]);
    std::swap(mFlushedHistory[dstRow], mFlushedHistory[srcRow]);
    // The member keeps the cells it already filled, so its lease travels with it rather
    // than its contents being copied into the destination row's.
    std::swap(mRowLease[dstRow], mRowLease[srcRow]);

    auto& kvCache = *mRows[dstRow];
    kvCache.mBatchRow = dstRow;
    for (LifeCycleId lc{0}; lc < mManager->numLifeCycles(); ++lc)
    {
        // The row's contents have already moved. Rebind its sole writer without copying from
        // the old row or ending membership -- mBatch and the arena lease are untouched.
        kvCache.mBasePageIndices[kDefaultBeamIndex][lc] = Span<int>{_stagingRow(lc, dstRow), mMaxBlocksPerSeq};
        if (mMaxScratchSlots > 0)
        {
            kvCache.setScratchSlotBuf(lc, _scratchSlotRow(lc, dstRow), mMaxScratchSlots);
        }
    }
    _seedRowWatermarks(dstRow, kvCache.historyLength(), kvCache.capacity());
    _seedRowWatermarks(srcRow, 0, 0);
    _clearIndexRow(srcRow);
}

void Batch::_detach(KvCache& kvCache) noexcept
{
    if (kvCache.mBatch != this || !kvCache.mBatchRow.has_value())
    {
        return;
    }
    // A seated member holds a reference from mRows, so its destructor cannot run while it
    // is one. Reaching here means a row was cleared without unbinding the member, leaving a
    // row that publishes from an object about to stop existing.
    KVCM2_CHECK_FATAL_WITH_INFO(
        false, "A KvCache was destroyed while seated in a Batch, so its row outlived the cache it names");
}

std::vector<uint8_t> Batch::resize(
    std::vector<std::optional<int>> const& capacities, std::vector<std::optional<int>> const& historyLengths)
{
    TLLM_CHECK_WITH_INFO(capacities.size() == mRows.stdSize() && historyLengths.size() == mRows.stdSize(),
        "Batch::resize: one capacity and history length per row");

    // Every row is checked before any is touched, so a bad request leaves the batch as it
    // was rather than part-way through.
    for (BatchRow row{0}; row < mRows.size(); ++row)
    {
        auto const& kvCache = mRows[row];
        if (kvCache == nullptr || !historyLengths[toSizeT(row)].has_value())
        {
            continue;
        }
        int const newHist = *historyLengths[toSizeT(row)];
        TLLM_CHECK_WITH_INFO(newHist >= kvCache->historyLength(), "Batch::resize: history length cannot be decreased");
        // Against the capacity that already existed, not the one being asked for. Capacity
        // this call grants makes room for tokens written after it, so history can only
        // cover tokens whose space was there while they were being written.
        TLLM_CHECK_WITH_INFO(newHist <= kvCache->capacity(),
            "Batch::resize: history length cannot exceed the capacity the row already had");
    }

    // Before the first member is touched, since each resize writes its staging row.
    waitPublished();

    std::vector<uint8_t> results(mRows.stdSize(), 0);
    for (BatchRow row{0}; row < mRows.size(); ++row)
    {
        auto const& kvCache = mRows[row];
        if (kvCache == nullptr)
        {
            continue;
        }
        if (kvCache->resize(capacities[toSizeT(row)], historyLengths[toSizeT(row)]))
        {
            results[toSizeT(row)] = 1;
            continue;
        }
        // The growth failed and rolled itself back, but the history advance is still both
        // legal -- it was checked against the capacity this row kept -- and free: the stale
        // range and the scratch range are each monotone in history, so raising it can only
        // release slots. Taking it anyway is what lets the units below it be written back
        // and their cells released, which is the pressure that made the growth fail.
        if (capacities[toSizeT(row)].has_value() && historyLengths[toSizeT(row)].has_value())
        {
            kvCache->resize(std::nullopt, historyLengths[toSizeT(row)]);
        }
    }
    // Once, after every member has settled: a partial failure leaves the rolled-back state,
    // which is what the table should describe.
    publish();
    return results;
}

void Batch::publish()
{
    // A step's work-list counters start here, all of them at once: per layer and per
    // direction they are independent, and one clear at the head of a step keeps a
    // non-kernel node out from between every resolve and whatever ran before it.
    //
    // On the Batch's stream, like the staging below it. A fetch issued on another stream
    // therefore has to be ordered after this call -- but it already did, because it reads
    // the page tables and watermarks this same call stages.
    if (mCounterElements > 0)
    {
        TLLM_CUDA_CHECK(cudaMemsetAsync(
            mCounters, 0, sizeof(int32_t) * mCounterElements, reinterpret_cast<cudaStream_t>(cudaStream())));
    }
    mFlushCountersDirty = false;

    // Refusing rather than waiting, because by this point a wait would be too late to help.
    // Everything that writes the staged rows has to be separated from the copy reading them:
    // the members' own mutations, which happened before this call, and the watermarks and
    // views staged just below. Waiting here would cover only the second, and would do it
    // while quietly hiding that the first had already raced.
    TLLM_CHECK_WITH_INFO(mPublished.isClosed(),
        "Batch::publish: the previous publish is still reading the staged rows. Call waitPublished() before "
        "whatever wrote them -- a direct resize(), a commit(), a beam fork, an SWA recycle, anything that locks "
        "or unlocks a page -- and not merely before this call: moving the wait here silences this check while "
        "leaving that earlier write racing the copy. Batch's own mutators, resize() and move() among them, "
        "already wait.");

    // A SHARED life cycle gives every layer the same converted indices, which only holds
    // while no member substitutes scratch slots there. Prefill acquires those and decode
    // does not, so a derived mode would have been PER_LAYER and reaching here means the
    // phases were mixed after it was fixed. A named SHARED makes this the check that holds
    // the caller to the claim: it is what lets a decode-only batch take the smaller table
    // on a life cycle the manager cannot clear for requests in general.
    // Life cycle outermost: the rows are walked only for one the manager left undecided,
    // since every other answer passes for every member.
    for (LifeCycleId lcId{0}; lcId < mManager->numLifeCycles(); ++lcId)
    {
        if (mManager->supportsIndexMode(mIndexModes[lcId], lcId) == std::optional<bool>(true))
        {
            continue;
        }
        for (auto const& kvCache : mRows)
        {
            TLLM_CHECK_WITH_INFO(kvCache == nullptr || kvCache->supportsIndexMode(mIndexModes[lcId], lcId),
                "Batch::publish: a member holds scratch slots in a SHARED life cycle; its layers no longer share "
                "one converted index table");
        }
    }

    // The members write their own page indices, but not their watermarks; collect those
    // now so the single transfer below carries a matched pair.
    _stageRowState();
    _stageScratch();

    auto const staging = reinterpret_cast<CUdeviceptr>(mDeviceStaging.get());
    auto* const upload = mUploadStream.handle();

    // The fill below overwrites what the previous step's copy read into the final tables.
    // Those run on different streams, so nothing but this event orders them.
    if (!mDeviceStagingConsumed.isClosed())
    {
        mDeviceStagingConsumed.waitInStream(reinterpret_cast<CudaStream>(upload));
        mDeviceStagingConsumed.close();
    }

    // One transfer for the whole block rather than one per changed row, or one per buffer:
    // a single driver call on the upload stream overlaps compute, where the alternatives
    // cost a launch apiece on the step's critical path.
    cuCheck(cuMemcpyHtoDAsync(staging, reinterpret_cast<void*>(mStaging.address()), mUploadBytes, upload));
    mPublished = mUploadStream.recordEvent();

    // The staging now holds this step's tables, so carrying the mirror across is all that
    // is left on the Batch's stream -- one copy, because the mirror is its tail verbatim.
    mPublished.waitInStream(reinterpret_cast<CudaStream>(cudaStream()));
    cuCheck(cuMemcpyDtoDAsync(
        reinterpret_cast<CUdeviceptr>(mMirror.get()), _deviceBufAt(BufName::kHistory), _mirrorBytes(), cudaStream()));
    _convertIndices(cudaStream());
    mDeviceStagingConsumed = CachedCudaEvent{reinterpret_cast<CudaStream>(cudaStream())};
}

void Batch::waitPublished()
{
    if (mPublished.isClosed())
    {
        return;
    }
    mPublished.synchronize();
    // Closing returns the event to the pool and makes every later call free.
    mPublished.close();
}

CoalescedPageTable Batch::coalescedPageTable(LifeCycleId lifeCycleId) const
{
    TLLM_CHECK_WITH_INFO(lifeCycleId.value() >= 0 && lifeCycleId < mManager->numLifeCycles(),
        "Batch::coalescedPageTable: unknown layer group");
    CoalescedPageTable view;
    view.data = _stagingRow(lifeCycleId, BatchRow{0});
    view.numRows = mMaxBatchSize;
    view.blocksPerRow = mMaxBlocksPerSeq;
    return view;
}

GpuCoalescedPageTable Batch::gpuCoalescedPageTable(LifeCycleId lifeCycleId) const
{
    TLLM_CHECK_WITH_INFO(lifeCycleId.value() >= 0 && lifeCycleId < mManager->numLifeCycles(),
        "Batch::gpuCoalescedPageTable: unknown layer group");
    GpuCoalescedPageTable view;
    // From the staging: the mirror carries only what a captured kernel reads, and the raw
    // indices are read there by the conversion rather than copied on.
    view.data = _deviceBufAt(BufName::kPageTables)
        + sizeof(int32_t) * _rowElements() * static_cast<size_t>(lifeCycleId.value());
    view.numRows = mMaxBatchSize;
    view.blocksPerRow = mMaxBlocksPerSeq;
    return view;
}

CUdeviceptr Batch::sparseLeases() const
{
    return _mirrorAt(BufName::kLeases);
}

WatermarkView Batch::watermarks() const
{
    WatermarkView view;
    view.historyLength = _mirrorAt(BufName::kHistory);
    view.capacity = _mirrorAt(BufName::kCapacity);
    view.numRows = mMaxBatchSize;
    return view;
}

SparseHitMiss Batch::peekHitMissCounter() const
{
    if (!mHitMiss)
    {
        return {};
    }
    // A temporary stream, synced alone: the Batch's own stream carries the step, and
    // waiting on that would turn a statistics read into a pipeline stall.
    SparseHitMiss host;
    TemporaryCudaStream temp{{}};
    cuCheck(cuMemcpyDtoHAsync(&host, reinterpret_cast<CUdeviceptr>(mHitMiss.get()), sizeof(host), temp.get()));
    cuCheck(cuStreamSynchronize(temp.get()));
    return host;
}

void Batch::resetHitMissCounter()
{
    if (mHitMiss)
    {
        TLLM_CUDA_CHECK(
            cudaMemsetAsync(mHitMiss.get(), 0, sizeof(SparseHitMiss), reinterpret_cast<cudaStream_t>(cudaStream())));
    }
}

size_t Batch::sparseArenaBytes() const
{
    size_t total = 0;
    for (LifeCycleId lc{0}; lc < mManager->numLifeCycles(); ++lc)
    {
        total += _sparseArena(lc).numBytes();
    }
    return total;
}

int Batch::numKvCaches() const noexcept
{
    return static_cast<int>(
        std::count_if(mRows.begin(), mRows.end(), [](std::shared_ptr<KvCache> const& row) { return row != nullptr; }));
}

bool Batch::isContiguous() const noexcept
{
    // Nothing occupied past the first free row is the same thing as a prefix.
    auto const firstFree = std::find(mRows.begin(), mRows.end(), nullptr);
    return std::none_of(firstFree, mRows.end(), [](std::shared_ptr<KvCache> const& row) { return row != nullptr; });
}

std::optional<BatchRow> Batch::rowOf(KvCache const& kvCache) const
{
    return kvCache.mBatch == this ? kvCache.mBatchRow : std::nullopt;
}

void Batch::prepareSparseWriteIndices(
    std::vector<BufferId> const& buffers, int32_t* out, std::optional<CUstream> streamOpt)
{
    CUstream const stream = streamOpt.value_or(cudaStream());
    TLLM_CHECK_WITH_INFO(out != nullptr, "prepareSparseWriteIndices: out must be a device pointer");

    auto const span = mManager->_sparseLayerSpan(buffers, "prepareSparseWriteIndices");
    LifeCycleId const lcId = span.lifeCycle;

    auto const& scratch = sparseStepScratch(lcId);
    TLLM_CHECK_WITH_INFO(
        scratch.inFlightSlot != nullptr, "prepareSparseWriteIndices: this layer group carries no sparse cache");

    int const unitsPerPage = mManager->sparseCacheUnitsPerPage(lcId);
    auto const leases = _leaseMirror();
    auto const arena = _sparseArena(lcId).view();
    auto const marks = watermarks();
    auto const numRows = maxBatchSize();

    // A resolve with no selection. It makes the in-flight unit resident -- reusing its entry
    // when a previous step already pinned it -- and does nothing else: an in-flight unit's
    // host page is not written until the watermark passes it, so there is nothing to copy.
    //
    // The selection and miss arrays are therefore left null, which the resolve tolerates only
    // because it records a miss against the copy list under `ordinal < historyUnits`, and an
    // in-flight ordinal never satisfies it. A change that made that write unconditional would
    // fault here rather than quietly copying stale bytes over what the model wrote.
    SparseResolveArgs resolve;
    resolve.leases = leases;
    resolve.arena = arena;
    resolve.numRows = numRows;
    resolve.selectionSlot = span.selectionSlot;
    resolve.selectionWidth = 0;
    resolve.historyTokens = reinterpret_cast<int32_t const*>(marks.historyLength);
    resolve.capacityTokens = reinterpret_cast<int32_t const*>(marks.capacity);
    resolve.unitTokens = mManager->config().tokensPerBlock / unitsPerPage;
    resolve.inFlightSlot = scratch.inFlightSlot;
    resolve.maxInFlightUnits = scratch.maxInFlightUnits;
    resolve.missCount = scratch.missCount;
    // No selection on this path, so there is nothing to resolve into a slot.
    resolve.selected = nullptr;
    resolve.outSlot = nullptr;
    launchSparseResolve(resolve, stream);

    // The in-flight list stands in for a selection: the resolve writes kNoSparseSlot past the
    // span, so a row with nothing in flight needs no separate signal.
    //
    // Shared the same way the fetch shares, and for the same reason: buffers whose arena
    // indices coincide are resolved once.
    auto const slots = sparseOutSlots(buffers);
    std::vector<bool> written(static_cast<size_t>(span.numBuffers), false);
    for (auto const& copy : _sparseCopyTemplates(lcId))
    {
        auto const runBase = std::max(copy.bufferBase, span.bufferBase);
        auto const runEnd = std::min(copy.bufferBase + copy.numBuffers, span.bufferBase + span.numBuffers);
        if (runBase >= runEnd)
        {
            continue;
        }
        int const firstSlot = slots[static_cast<size_t>(runBase - span.bufferBase)];
        if (written[static_cast<size_t>(firstSlot)])
        {
            continue;
        }
        for (int32_t buffer = runBase; buffer < runEnd; ++buffer)
        {
            written[static_cast<size_t>(slots[static_cast<size_t>(buffer - span.bufferBase)])] = true;
        }
        SparseOutIndexArgs indices;
        indices.leases = leases;
        indices.arena = arena;
        indices.numRows = numRows;
        indices.selected = nullptr;
        indices.outSlot = scratch.inFlightSlot;
        // One entry per request: resize() bounds capacity - history to a single token, so at
        // most one unit is ever in flight. The scratch row is the reservation's headroom,
        // which is wider, hence the separate stride.
        indices.selectionWidth = 1;
        indices.outSlotStride = scratch.maxInFlightUnits;
        indices.out = out;
        // The row stride is the caller's: `out` keeps one entry per buffer, and an entry
        // no buffer maps to is never written.
        indices.outBuffers = span.numBuffers;
        indices.outBase = firstSlot;
        indices.bufferBase = runBase;
        indices.numBuffers = runEnd - runBase;
        indices.cellsPerBuffer = copy.cellsPerBuffer;
        indices.cellsPerLease = copy.cellsPerLease;
        indices.unitsPerPage = unitsPerPage;
        launchSparseOutIndices(indices, stream);
    }
}

void Batch::writeBackSparseUnits(LifeCycleId lifeCycle, CUstream stream)
{
    auto const leases = _leaseMirror();
    auto const arena = _sparseArena(lifeCycle).view();
    auto const numRows = maxBatchSize();

    // Which units each row crossed. One scan covers every (layer, row); a row whose
    // watermark did not move reports none.
    auto const& scratch = sparseStepScratch(lifeCycle);
    SparseFlushListArgs listArgs;
    listArgs.leases = leases;
    listArgs.arena = arena;
    listArgs.numRows = numRows;
    listArgs.fromUnits = flushFromUnits(lifeCycle);
    listArgs.toUnits = flushToUnits(lifeCycle);
    listArgs.work = scratch.flushWork;
    listArgs.outCount = scratch.flushCount;
    launchSparseFlushList(listArgs, stream);

    // One launch for every pool and every layer: the descriptors the Batch built already
    // say where each pool's buffers and index tables are, and a flush list is per layer.
    SparseUnitCopyArgs copy;
    copy.leases = leases;
    copy.arena = arena;
    copy.numRows = numRows;
    copy.copyDesc = scratch.copyDesc;
    copy.bufferBase = 0;
    copy.numBuffers = scratch.numBuffers;
    copy.direction = SparseCopyDirection::kWriteBack;
    copy.unitsPerPage = mManager->sparseCacheUnitsPerPage(lifeCycle);
    // The work list is every layer of this row, which is what a write-back crosses at once,
    // unlike a fetch, which resolves one layer across the whole batch.
    copy.work = scratch.flushWork;
    copy.gridCtas = sparseCopyGridCtas(scratch.flushWork.capacity, mManager->_smCount());
    launchSparseUnitCopy(copy, stream);
}

std::vector<int> Batch::sparseOutSlots(std::vector<BufferId> const& buffers) const
{
    auto const span = mManager->_sparseLayerSpan(buffers, "sparseOutSlots");
    auto const runs = _sparseCopyTemplates(span.lifeCycle);
    auto const& shape = _sparseArena(span.lifeCycle).shape();

    // What decides whether two buffers sharing a dense table also share an arena index.
    // cellsPerBuffer is one value for the whole shape, so a lease spans the same cells in
    // two pools exactly when they hold the same number of buffers.
    auto geometry = [&](int32_t buffer)
    {
        for (auto const& run : runs)
        {
            if (buffer >= run.bufferBase && buffer < run.bufferBase + run.numBuffers)
            {
                return std::tuple<int, int, int>{static_cast<int>(run.cellsPerLease),
                    static_cast<int>(run.cellsPerBuffer), shape.bufferSlotInPool[static_cast<size_t>(buffer)]};
            }
        }
        throw LogicError("sparseOutSlots: a sparse buffer belongs to no pool run");
    };

    std::vector<std::pair<int, int>> tableEntries;
    std::vector<std::tuple<int, int, int>> geometries;
    std::vector<int> slots;
    slots.reserve(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i)
    {
        auto const [tableId, entry] = indexTableSlot(buffers[i].layerId, buffers[i].role);
        auto const key = std::make_pair(tableId.value(), entry);
        auto const geo = geometry(span.bufferBase + static_cast<int32_t>(i));
        auto const found = std::find(tableEntries.begin(), tableEntries.end(), key);
        if (found == tableEntries.end())
        {
            tableEntries.push_back(key);
            geometries.push_back(geo);
            slots.push_back(static_cast<int>(tableEntries.size()) - 1);
            continue;
        }
        auto const at = static_cast<size_t>(std::distance(tableEntries.begin(), found));
        // The dense tables merge on the hot pool's scale and concat offsets; the arena has
        // its own geometry and has to agree separately. It normally does, but the two are
        // not the same quantity -- a pool carrying only some of the group's layers has the
        // same scale and a different cellsPerLease -- and taking the dense answer there
        // would alias two different cells onto one slice.
        TLLM_CHECK_WITH_INFO(geometries[at] == geo,
            "sparseOutSlots: buffers share an index table entry but not their arena geometry, so one "
            "slice cannot serve both");
        slots.push_back(static_cast<int>(at));
    }
    return slots;
}

void Batch::fetchSparsePages(std::vector<BufferId> const& buffers, int32_t const* selected, int selectionWidth,
    int32_t* out, std::optional<CUstream> streamOpt)
{
    auto const span = mManager->_sparseLayerSpan(buffers, "fetchSparsePages");
    TLLM_CHECK_WITH_INFO(mManager->_selectionLayerCount(span.lifeCycle, span.selectionSlot) == 1,
        "fetchSparsePages: this layer shares its selection_id, so its selection is resolved once for every "
        "layer naming it; call resolveSparseSelection() then fetchResolvedPages()");
    resolveSparseSelection(buffers, selected, selectionWidth, streamOpt);
    fetchResolvedPages(buffers, selected, selectionWidth, out, streamOpt);
}

void Batch::resolveSparseSelection(std::vector<BufferId> const& buffers, int32_t const* selected, int selectionWidth,
    std::optional<CUstream> streamOpt)
{
    CUstream const stream = streamOpt.value_or(cudaStream());
    TLLM_CHECK_WITH_INFO(selected != nullptr, "resolveSparseSelection: selected must be a device pointer");

    auto const span = mManager->_sparseLayerSpan(buffers, "resolveSparseSelection");
    LifeCycleId const lcId = span.lifeCycle;

    auto const& scratch = sparseStepScratch(lcId);
    TLLM_CHECK_WITH_INFO(
        scratch.outSlot != nullptr, "resolveSparseSelection: this layer group carries no sparse cache");
    TLLM_CHECK_WITH_INFO(selectionWidth == scratch.selectionWidth,
        "resolveSparseSelection: selectionWidth must match the configured numSelectedBlocks");

    int const unitsPerPage = mManager->sparseCacheUnitsPerPage(lcId);
    auto const marks = watermarks();

    SparseResolveArgs resolve;
    resolve.leases = _leaseMirror();
    resolve.arena = _sparseArena(lcId).view();
    resolve.numRows = maxBatchSize();
    resolve.selectionSlot = span.selectionSlot;
    resolve.selected = selected;
    resolve.selectionWidth = selectionWidth;
    resolve.historyTokens = reinterpret_cast<int32_t const*>(marks.historyLength);
    resolve.capacityTokens = reinterpret_cast<int32_t const*>(marks.capacity);
    resolve.unitTokens = mManager->config().tokensPerBlock / unitsPerPage;
    resolve.outSlot = scratch.outSlot;
    resolve.inFlightSlot = scratch.inFlightSlot;
    resolve.maxInFlightUnits = scratch.maxInFlightUnits;
    resolve.work = _sparseFetchWork(scratch, span.selectionSlot);
    resolve.missCount = scratch.missCount;
    // Every row adds its own hits and misses as it resolves them.
    resolve.hitMiss = scratch.hitMiss;
    launchSparseResolve(resolve, stream);
}

void Batch::fetchResolvedPages(std::vector<BufferId> const& buffers, int32_t const* selected, int selectionWidth,
    int32_t* out, std::optional<CUstream> streamOpt)
{
    CUstream const stream = streamOpt.value_or(cudaStream());
    TLLM_CHECK_WITH_INFO(
        selected != nullptr && out != nullptr, "fetchResolvedPages: selected and out must be device pointers");

    auto const span = mManager->_sparseLayerSpan(buffers, "fetchResolvedPages");
    LifeCycleId const lcId = span.lifeCycle;
    auto const bufferBase = span.bufferBase;
    auto const numBuffers = span.numBuffers;

    auto const& scratch = sparseStepScratch(lcId);
    TLLM_CHECK_WITH_INFO(scratch.outSlot != nullptr, "fetchResolvedPages: this layer group carries no sparse cache");
    TLLM_CHECK_WITH_INFO(selectionWidth == scratch.selectionWidth,
        "fetchResolvedPages: selectionWidth must match the configured numSelectedBlocks");

    int const unitsPerPage = mManager->sparseCacheUnitsPerPage(lcId);
    auto const leases = _leaseMirror();
    auto const arena = _sparseArena(lcId).view();
    auto const numRows = maxBatchSize();
    auto const layerWork = _sparseFetchWork(scratch, span.selectionSlot);

    // The misses the selection produced, applied to each of this layer's buffers. Layers
    // sharing the selection each run this and each fill their own mirror of the entry.
    //
    // One launch for every pool of this layer. The descriptors the Batch built already
    // clip each pool to the layer's buffers and name its index table, so nothing here
    // depends on which layer is being fetched beyond choosing the buffer range.
    SparseUnitCopyArgs copy;
    copy.leases = leases;
    copy.arena = arena;
    copy.numRows = numRows;
    // The same descriptors as the write-back: a run never spans layers, so it already
    // names this layer's index table. Only the buffer range below differs.
    copy.copyDesc = scratch.copyDesc;
    copy.bufferBase = bufferBase;
    copy.numBuffers = numBuffers;
    copy.direction = SparseCopyDirection::kFetch;
    copy.unitsPerPage = unitsPerPage;
    copy.work = layerWork;
    copy.gridCtas = sparseCopyGridCtas(scratch.fetchWork.capacity, mManager->_smCount());
    launchSparseUnitCopy(copy, stream);

    // The index table the attention kernel reads is derived from the resolve's slots alone:
    // it neither reads a cell nor waits for one to be filled. The copy releases its launch
    // dependency as soon as it is addressed, so this follows it on the stream without
    // waiting for it to finish.
    auto const plan = mManager->_sparseBufferPlan(lcId);
    // Buffers whose indices coincide share a slice, so a run that only repeats slices an
    // earlier one already wrote is not launched at all. A layer's data and scale pools
    // mirror, which is exactly that case.
    auto const slots = sparseOutSlots(buffers);
    std::vector<bool> written(static_cast<size_t>(numBuffers), false);
    for (auto const& pool : _sparseCopyTemplates(lcId))
    {
        auto const runBase = std::max(pool.bufferBase, bufferBase);
        auto const runEnd = std::min(pool.bufferBase + pool.numBuffers, bufferBase + numBuffers);
        if (runBase >= runEnd)
        {
            continue;
        }
        // A run's buffers are consecutive entries of one table, so their slices are
        // consecutive too and the first one says where the run writes.
        int const firstSlot = slots[static_cast<size_t>(runBase - bufferBase)];
        if (written[static_cast<size_t>(firstSlot)])
        {
            continue;
        }
        for (int32_t buffer = runBase; buffer < runEnd; ++buffer)
        {
            written[static_cast<size_t>(slots[static_cast<size_t>(buffer - bufferBase)])] = true;
        }
        SparseOutIndexArgs indices;
        indices.leases = leases;
        indices.arena = arena;
        indices.numRows = numRows;
        indices.selected = selected;
        indices.outSlot = scratch.outSlot;
        indices.selectionWidth = selectionWidth;
        indices.out = out;
        // The row stride is the caller's, not the slice count: `out` keeps one entry per
        // buffer, and a slice no buffer maps to is simply never written.
        indices.outBuffers = numBuffers;
        indices.outBase = firstSlot;
        indices.bufferBase = runBase;
        indices.numBuffers = runEnd - runBase;
        indices.cellsPerBuffer = pool.cellsPerBuffer;
        indices.cellsPerLease = pool.cellsPerLease;
        indices.unitsPerPage = unitsPerPage;
        launchSparseOutIndices(indices, stream);
    }
}

void Batch::_buildSparseArenas()
{
    LifeCycleId const numLc = mManager->numLifeCycles();
    mSparseArenas = TypedVec<LifeCycleId, SparseCellArena>(numLc);

    // One lease per row this Batch can seat, which is the concurrency the arenas serve: a
    // row's member is stepped, and stepping is what reads the cells.
    int const maxLeases = mMaxBatchSize;

    // Every row owns a lease for the Batch's lifetime. move() permutes the mapping, so a
    // member carries its cells between rows without them being copied.
    mRowLease.resize(BatchRow{maxLeases});
    for (int row = 0; row < maxLeases; ++row)
    {
        mRowLease[BatchRow{row}] = SparseLeaseIndex{row};
    }

    for (LifeCycleId lcId{0}; lcId < numLc; ++lcId)
    {
        if (!mManager->hasSparseFetchPath(lcId))
        {
            continue;
        }
        auto const plan = mManager->_sparseBufferPlan(lcId);
        TLLM_CHECK(!plan.empty());

        auto const cellsPerBuffer = mManager->_sparseCacheSubSlotsPerBuffer(lcId);
        SparseCacheInit shape{
            .numLayers = mManager->_sparseCacheLayerCount(lcId),
            .numSelections = mManager->_sparseCacheGroupCount(lcId),
            .numBuffers = mManager->sparseCacheBufferCount(lcId),
            .entriesPerSelection = cellsPerBuffer * mManager->sparseCacheUnitsPerPage(lcId),
            .cellsPerBuffer = cellsPerBuffer,
            .layerSelection = mManager->_sparseLayerSelection(lcId),
        };
        (void) checkedSparseResolveSharedBytes(shape.entriesPerSelection, mManager->_sparseCacheSelectionWidth(lcId),
            mManager->_sparseCacheMaxInFlightUnits(lcId));

        // The arena mirrors the hot pool group: one region per pool the life cycle's sparse
        // buffers draw on, sized by that pool's cell and the buffers sharing it.
        size_t numPools = 0;
        for (auto const& entry : plan)
        {
            numPools = std::max(numPools, static_cast<size_t>(entry.poolIndex.value()) + 1);
        }
        shape.pools.assign(numPools, {0, 0});
        for (auto const& entry : plan)
        {
            shape.bufferLayer.push_back(entry.layerRow);
            shape.bufferSlotInPool.push_back(entry.slotInPool);
            auto& [cellBytes, buffers] = shape.pools[static_cast<size_t>(entry.poolIndex.value())];
            cellBytes = entry.size;
            ++buffers;
        }
        // A pool index the life cycle's sparse buffers never use keeps its place in the list,
        // because the arena is indexed by the storage pool index rather than by a dense
        // numbering of its own. It stays {0, 0}: holding no cells, it takes no space, and its
        // base is simply where the next pool's region starts.
        mSparseArenas[lcId] = SparseCellArena(shape, maxLeases);
    }
}

std::vector<SparseCopyRun> Batch::_sparseCopyTemplates(LifeCycleId lifeCycle) const
{
    auto const plan = mManager->_sparseBufferPlan(lifeCycle);
    TLLM_CHECK(!plan.empty());
    auto const& arena = mSparseArenas[lifeCycle];

    std::vector<SparseCopyRun> templates;
    for (size_t begin = 0; begin < plan.size();)
    {
        auto const pool = plan[begin].poolIndex;
        size_t end = begin;
        while (end < plan.size() && plan[end].poolIndex == pool)
        {
            ++end;
        }

        // A pool holds buffers of one size, but not necessarily of one unit layout, and the
        // copy walks a run with one set of strides.
        auto const& layout = plan[begin].layout;
        for (size_t i = begin; i < end; ++i)
        {
            auto const& other = plan[i].layout;
            if (other.numChunks != layout.numChunks || other.chunkBytes != layout.chunkBytes
                || other.chunkStride != layout.chunkStride || other.unitStride != layout.unitStride)
            {
                throw LogicError("Sparse buffers sharing a pool must share a selection unit layout");
            }
        }

        SparseCopyRun args;
        // The page side is the life cycle's own hot pool: host memory, registered device-mapped,
        // so the kernel addresses it directly.
        //
        // A converted index counts whole buffers, not coalesced slots, so the stride is one
        // buffer and the pool's slot size never enters. The base belongs to the index table
        // the run was declared in, which only the caller's Batch knows, so it is left unset
        // here.
        args.hostPageStride = static_cast<uint32_t>(plan[begin].size);
        // The hot side is the cache arena, not a storage pool.
        args.deviceBase = arena.base(pool);
        args.bufferBytes = static_cast<uint32_t>(arena.bufferBytes(pool));
        args.cellsPerBuffer = mManager->_sparseCacheSubSlotsPerBuffer(lifeCycle);
        args.cellsPerLease = arena.cellsPerLease(pool);
        args.layout = layout;
        args.bufferBase = static_cast<int32_t>(begin);
        args.numBuffers = static_cast<int32_t>(end - begin);
        args.width = sparseCopyWidth(args.layout, args.hostPageStride, args.bufferBytes);
        templates.push_back(args);
        begin = end;
    }
    return templates;
}

MemAddress Batch::getSparseCacheBaseAddress(LayerId layerId, DataRole const& role) const
{
    LifeCycleId const lcId = mManager->getLayerGroupId(layerId);
    auto const& arena = mSparseArenas[lcId];
    auto const plan = mManager->_sparseBufferPlan(lcId);
    auto const it = std::ranges::find_if(
        plan, [&](SparseBufferPlan const& entry) { return entry.id.layerId == layerId && entry.id.role == role; });
    TLLM_CHECK_WITH_INFO(
        !arena.empty() && it != plan.end(), "getSparseCacheBaseAddress: this buffer has no sparse fetch path");
    return static_cast<MemAddress>(arena.base(it->poolIndex));
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
