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

#include "kv_cache_manager_v2/stagingBuffer.h"
#include "kv_cache_manager_v2/storage/config.h"
#include "kv_cache_manager_v2/utils/typedIndex.h"

#include <cstdint>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// SparseLeaseIndex and SparseArenaView — the device-resident state of one request's sparse
// history cache for one life cycle, as the fetch kernel sees it: a slot per row, and the
// geometry that slot is read within.
//
// A reserved slot is a row of buffer-sized cells, and any layer's unit may occupy any cell;
// a cell holds `unitsPerPage` units when a selection unit is smaller than a buffer. An entry
// is identified by (layer, entry index) and resolved to a cell arithmetically, so nothing
// ties a layer to a fixed position.
//
// The metadata is laid out [selection][entry]. It is per SELECTION, not per buffer or even
// per layer: everything selecting the same units resolves to the same residency, so one
// resident set and one recency order serve them all, and the entry index the resolve
// produces addresses each buffer's own cell. A layer selecting for itself is the singleton
// case of that; layers sharing a selection_id share the row.
// ---------------------------------------------------------------------------
//! The arena slot a row's member holds, or kNoSparseLease when the row carries none.
//!
//! This is the whole of a row's published cache state: everything else a kernel needs is
//! either arithmetic from this index -- a lease's cells and its metadata both sit at a
//! fixed stride from the arena's bases, see "Why the cache is a flat array of cells" in
//! SPARSE_ATTENTION_DESIGN.md -- or a constant of the life cycle, which SparseArenaView
//! carries once per launch rather than once per row.
using SparseLeaseIndex = StrongIndex<int32_t, struct SparseLeaseIndexTag, -1>;
inline constexpr SparseLeaseIndex kNoSparseLease{};

//! One life cycle's arena, as a kernel addresses it. Fixed for the arena's lifetime, so a
//! launch takes it by value and no row carries a copy.
struct SparseArenaView
{
    //! Lease i's metadata begins at `metadataBase + i * leaseStride`: first its resident
    //! unit ordinal per entry, kBadBlockOrdinal where the entry is free, then at
    //! `lruOffset` its entry indices in recency order, most-recently-used first.
    //!
    //! The ordinals are dense rather than keyed: a fetch builds a hash over them in shared
    //! memory to answer "is this ordinal resident, and where".
    std::byte* metadataBase;
    uint64_t leaseStride;
    int32_t lruOffset;

    //! [numBuffers] each buffer's index among those sharing its pool: where its cells sit
    //! inside that pool's arena region.
    int32_t const* bufferSlotInPool;

    //! [numBuffers] the layer each buffer belongs to, i.e. which work list carries it.
    //! Non-decreasing: the buffers of a layer are consecutive, which is what lets a
    //! one-layer call name its buffers as a range rather than a set.
    int32_t const* bufferLayer;

    //! [numLayers] the metadata row each layer resolves within. The identity where every
    //! layer selects for itself; layers sharing a selection share an entry here, which is
    //! the whole of what sharing one means to a kernel.
    int32_t const* layerSelection;

    //! Layers in this life cycle. Not the metadata's first dimension -- it is the extent of
    //! the write-back's grid and of its per-(row, layer) counts.
    int32_t numLayers;

    //! Selections those layers resolve between them: the metadata's first dimension.
    int32_t numSelections;
    int32_t entriesPerSelection;

    [[nodiscard]] constexpr int32_t numEntries() const noexcept
    {
        return numSelections * entriesPerSelection;
    }
};

// ---------------------------------------------------------------------------
// SparseCacheInit — everything a request's cache state is built from, all of it
// fixed for the life of the reservation.
// ---------------------------------------------------------------------------
struct SparseCacheInit
{
    int32_t numLayers;
    //! Metadata rows those layers share between them, each the residency of one selection.
    int32_t numSelections;
    int32_t numBuffers;
    int32_t entriesPerSelection;
    //! Cells one buffer gets, i.e. entriesPerSelection / unitsPerPage. A lease is this many
    //! cells per buffer, laid out buffer-major.
    int32_t cellsPerBuffer;
    //! [numBuffers] the layer each buffer belongs to, non-decreasing.
    std::vector<int32_t> bufferLayer;
    //! [numLayers] the metadata row each layer resolves within, every row named by at
    //! least one layer.
    std::vector<int32_t> layerSelection;
    //! [numBuffers] each buffer's index among those sharing its pool.
    std::vector<int32_t> bufferSlotInPool;
    //! One entry per pool of the life cycle: cell size, and buffers drawing on it.
    std::vector<std::pair<uint64_t, int32_t>> pools;
};

class SparseCellArena;

// ---------------------------------------------------------------------------
// SparseCellArena — one life cycle's GPU cells, and the metadata that describes them.
//
// A dedicated allocation rather than storage slots, so the life cycle's hot pool group can
// be host memory without the cache following it there. Every request of a life cycle needs
// the same number of cells, so this is a fixed-size-block free list.
//
// One block holds the cells of every lease followed by their metadata. Cells come first so
// that a cell index is relative to the arena's base, which is what the fetch reports and
// what get_sparse_cache_base_address returns for a sparse buffer.
// ---------------------------------------------------------------------------
class SparseCellArena
{
public:
    SparseCellArena() = default;

    //! @param shape     per-request metadata and cell geometry, fixed by configuration,
    //!                   including one entry per pool of the life cycle.
    //! @param maxLeases  concurrent requests this life cycle can serve.
    //!
    //! One region per pool, mirroring the hot pool group: pools partition the life cycle's
    //! buffers by size, and a cell is one buffer, so cells cannot be uniform across them.
    SparseCellArena(SparseCacheInit const& shape, int maxLeases);

    ~SparseCellArena() = default;
    SparseCellArena(SparseCellArena&&) noexcept = default;
    SparseCellArena& operator=(SparseCellArena&&) noexcept = default;
    SparseCellArena(SparseCellArena const&) = delete;
    SparseCellArena& operator=(SparseCellArena const&) = delete;

    [[nodiscard]] bool empty() const noexcept
    {
        return mMemory == nullptr;
    }

    //! Base of one pool's cell region: the base the fetch's indices for a buffer are relative to.
    [[nodiscard]] CUdeviceptr base(PoolIndex pool) const
    {
        return mPoolBase.at(pool);
    }

    [[nodiscard]] uint64_t bufferBytes(PoolIndex pool) const
    {
        return mShape.pools.at(static_cast<size_t>(pool.value())).first;
    }

    //! Cells one lease takes in this pool's region: its buffers times the cells each gets.
    [[nodiscard]] int cellsPerLease(PoolIndex pool) const
    {
        return mShape.pools.at(static_cast<size_t>(pool.value())).second * mShape.cellsPerBuffer;
    }

    //! Device bytes this arena holds: every pool's cell region, the state its leases share,
    //! and one metadata block per lease -- the whole allocation.
    [[nodiscard]] size_t numBytes() const noexcept
    {
        return mNumBytes;
    }

    //! Clear `slot`'s metadata on `stream`, which is what taking it means here: every
    //! ordinal becomes kBadBlockOrdinal and the recency order the identity permutation.
    //! The slot is the Batch row's own, and one row's slot serves every life cycle.
    void initialiseLease(SparseLeaseIndex slot, CUstream stream);

    //! Copy a slot's resident ordinals back to the host, for debugging and introspection
    //! only. Not on any serving path: it synchronises the device to pick up whatever a
    //! resolve left on the Batch's stream, which no stream of the caller's orders it
    //! against, and so reports a moment rather than a state the caller can act on.
    [[nodiscard]] std::vector<int32_t> peekResidentOrdinals(SparseLeaseIndex slot) const;

    //! How a kernel addresses this arena's leases. Constant once constructed, so a launch
    //! carries one of these rather than one descriptor per row.
    [[nodiscard]] SparseArenaView view() const noexcept;

    [[nodiscard]] SparseCacheInit const& shape() const noexcept
    {
        return mShape;
    }

private:
    CudaUniqPtr<> mMemory;
    SparseCacheInit mShape;
    //! Slots this arena was sized for, which every lease index must fall inside.
    int mMaxLeases = 0;
    size_t mMetadataBytes = 0;
    size_t mNumBytes = 0;
    TypedVec<PoolIndex, CUdeviceptr> mPoolBase;
    std::byte* mMetadataBase = nullptr;

    //! Shared by every lease of the life cycle, so a resume copies none of them.
    int32_t* mSharedLayer = nullptr;
    int32_t* mSharedSlotInPool = nullptr;
    int32_t* mSharedLayerSelection = nullptr;
    int32_t* mIdentity = nullptr;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
