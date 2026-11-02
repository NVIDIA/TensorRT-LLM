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
#include "kv_cache_manager_v2/config.h"
#include "kv_cache_manager_v2/lifeCycleRegistry.h"

#include "tensorrt_llm/common/assert.h"
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// Index types (mirrors _storage/_core.py)
// ---------------------------------------------------------------------------
// Index of a pool group (= set of pools with mirrored allocation).
using PoolGroupIndex = StrongIndex<int, struct PoolGroupIndexTag>;
// Index of a pool within a pool group.
using PoolIndex = StrongIndex<int, struct PoolIndexTag>;
// Plain count of slots when the value is not itself a slot id.
using SlotCount = std::int64_t;
// Slot index within a pool.
using SlotId = StrongIndex<SlotCount, struct SlotIdTag>;

[[nodiscard]] inline SlotCount slotCountValueFromSize(std::size_t count) noexcept
{
    TLLM_CHECK_DEBUG(count <= static_cast<std::size_t>(std::numeric_limits<SlotCount>::max()));
    return static_cast<SlotCount>(count);
}

[[nodiscard]] inline std::size_t slotCountToSizeT(SlotCount count) noexcept
{
    TLLM_CHECK_DEBUG_WITH_INFO(count >= 0, "Slot count must be non-negative for size_t conversion");
    return static_cast<std::size_t>(count);
}

[[nodiscard]] inline int slotIdToPageIndexValue(SlotId slotId)
{
    if (slotId.value() < 0 || slotId.value() > std::numeric_limits<int>::max())
    {
        throw std::overflow_error("SlotId does not fit in int page-index storage");
    }
    return static_cast<int>(slotId.value());
}

// ---------------------------------------------------------------------------
// BufferId — (layer_id, role) pair identifying one buffer in a layer.
// Mirrors _storage/_config.py::BufferId.
// ---------------------------------------------------------------------------
struct BufferId
{
    LayerId layerId = 0;
    DataRole role;

    bool operator==(BufferId const& o) const noexcept
    {
        return layerId == o.layerId && role == o.role;
    }

    bool operator<(BufferId const& o) const noexcept
    {
        if (layerId != o.layerId)
            return layerId < o.layerId;
        return role < o.role;
    }
};

//! Buffers a consumer must address from one table, under one base pointer.
//!
//! Some attention ops take a single block table covering K and V rather than one each.
//! Order defines the table's buffer dimension: a kernel reads buffer `i` at entry `i`.
//!
//! All buffers must share a life cycle and a page-index scale; one table cannot serve
//! buffers whose converted indices differ. A spec spanning either is refused when the
//! Batch is constructed.
struct IndexTableSpec
{
    std::vector<BufferId> buffers;
};

struct BufferIdHash
{
    size_t operator()(BufferId const& id) const noexcept
    {
        size_t h = std::hash<int>{}(id.layerId);
        h ^= std::hash<std::string>{}(id.role) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// ---------------------------------------------------------------------------
// CoalescedBuffer — several buffers of the same size and life cycle,
// laid out contiguously within one slot.
// Mirrors _storage/_config.py::CoalescedBuffer.
// ---------------------------------------------------------------------------
struct CoalescedBuffer
{
    size_t singleBufferSize = 0;
    std::vector<BufferId> bufferIds;

    //! Whether this pool's pages live in the sparse GPU cache: host memory, device-mapped,
    //! and not charged to the GPU tier's quota.
    //!
    //! Per pool, not per life cycle -- one layer may hold both a sparse buffer and a dense one.
    bool isSparse = false;

    [[nodiscard]] size_t size() const noexcept
    {
        return singleBufferSize * bufferIds.size();
    }

    [[nodiscard]] int numBuffers() const noexcept
    {
        return static_cast<int>(bufferIds.size());
    }
};

// ---------------------------------------------------------------------------
// SlotDescVariant — one life cycle's view of a pool group.
// Mirrors _storage/_config.py::SlotDescVariant.
// ---------------------------------------------------------------------------
struct SlotDescVariant
{
    LifeCycleId lifeCycleId{0};
    TypedVec<PoolIndex, CoalescedBuffer> coalescedBuffers; // sorted size desc

    [[nodiscard]] bool isSparse(PoolIndex poolIdx) const
    {
        return coalescedBuffers.at(poolIdx).isSparse;
    }

    // Slot size for each pool in this group.
    [[nodiscard]] TypedVec<PoolIndex, size_t> slotSizeList() const
    {
        TypedVec<PoolIndex, size_t> out;
        out.reserve(coalescedBuffers.size());
        for (auto const& cb : coalescedBuffers)
            out.push_back(cb.size());
        // Coalesced buffers must be sorted in descending size order.
        TLLM_CHECK_DEBUG(std::is_sorted(out.begin(), out.end(), std::greater<>()));
        return out;
    }
};

// ---------------------------------------------------------------------------
// SlotDesc — a pool group descriptor. Variants share the same slotSizeList
// but may differ in which buffers they contain.
// Mirrors _storage/_config.py::SlotDesc.
// ---------------------------------------------------------------------------
struct SlotDesc
{
    std::vector<SlotDescVariant> variants; // different life cycles sharing this pool group

    [[nodiscard]] TypedVec<PoolIndex, size_t> slotSizeList() const
    {
        return getUniformAttribute(variants, [](auto const& v) { return v.slotSizeList(); });
    }

    //! Life cycles share a pool group only when their pools agree position-wise on
    //! sparsity as well as size, so this is uniform across variants.
    [[nodiscard]] bool isSparse(PoolIndex poolIdx) const
    {
        return getUniformAttribute(variants, [poolIdx](auto const& v) { return v.isSparse(poolIdx); });
    }
};

// ---------------------------------------------------------------------------
// BufferAttr — metadata for one buffer within storage.
// Mirrors _storage/_config.py::BufferAttr.
// ---------------------------------------------------------------------------
struct BufferAttr
{
    LifeCycleId lifeCycleId{0};
    PoolIndex poolIndex{0};
    size_t offset = 0;            // byte offset within the slot
    size_t size = 0;              // expanded size of the buffer (after expansion)
    int expansion = 1;            // expansion factor (tokens_per_block / tokens_per_block_override)
    int scratchPagesPerBlock = 1; // buffers of this layer in the same coalesced slot
};

// ---------------------------------------------------------------------------
// Rational — exact fraction for slot utilization computations.
// ---------------------------------------------------------------------------
struct Rational
{
    int num = 0;
    int den = 1;

    // ceil(n * num / den). Uses int64_t intermediates to avoid overflow.
    [[nodiscard]] int ceilMul(int value) const noexcept
    {
        auto n = static_cast<int64_t>(value) * num;
        return static_cast<int>((n + den - 1) / den);
    }

    bool operator>(Rational const& other) const noexcept
    {
        return static_cast<int64_t>(num) * other.den > static_cast<int64_t>(other.num) * den;
    }

    bool operator==(Rational const& other) const noexcept
    {
        return static_cast<int64_t>(num) * other.den == static_cast<int64_t>(other.num) * den;
    }
};

// ---------------------------------------------------------------------------
// StorageConfig — complete storage layout for the KV cache.
// Mirrors _storage/_config.py::StorageConfig.
// ---------------------------------------------------------------------------
struct StorageConfig
{
    TypedVec<CacheLevel, CacheTierConfig> cacheTiers;
    TypedVec<PoolGroupIndex, SlotDesc> slotDescList;

    //! Host bytes the GPU tier's host-backed pools may take, or unset for unbounded.
    std::optional<size_t> hostSparsePoolQuota;

    // Expansion factor per buffer (for heterogeneous tokens_per_block).
    std::unordered_map<BufferId, int, BufferIdHash> expansion;

    // Map each LifeCycleId → its PoolGroupIndex.
    TypedVec<LifeCycleId, PoolGroupIndex> lifeCycleGrouping() const;

    // Attribute map for each buffer.
    std::map<BufferId, BufferAttr> bufferAttributes() const;

    // Map LayerId → the lifecycles of its buffers, in ascending order. A layer spans several
    // lifecycles when its buffers use different reuse policies.
    std::unordered_map<LayerId, std::vector<LifeCycleId>> layerToLifeCycleIds() const;

    LifeCycleId numLifeCycles() const;
};

// ---------------------------------------------------------------------------
// Factory: create StorageConfig from KVCacheManagerConfig.
// Mirrors _storage/_config.py::create_storage_config.
// ---------------------------------------------------------------------------
StorageConfig createStorageConfig(KVCacheManagerConfig const& config);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
