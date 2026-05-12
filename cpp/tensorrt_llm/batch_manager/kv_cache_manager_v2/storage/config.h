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

#include <map>
#include <unordered_map>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// Index types (mirrors _storage/_core.py)
// ---------------------------------------------------------------------------
using PoolGroupIndex = int; // index of a pool group (= set of pools with mirrored allocation)
using PoolIndex = int;      // index of a pool within a pool group
using SlotId = int;         // slot index within a pool

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

    size_t size() const noexcept
    {
        return singleBufferSize * bufferIds.size();
    }

    int numBuffers() const noexcept
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
    LifeCycleId lifeCycleId = 0;
    std::vector<CoalescedBuffer> coalescedBuffers; // indexed by PoolIndex, sorted size desc

    // Slot size for each pool in this group.
    std::vector<size_t> slotSizeList() const
    {
        std::vector<size_t> out;
        out.reserve(coalescedBuffers.size());
        for (auto const& cb : coalescedBuffers)
            out.push_back(cb.size());
        // Coalesced buffers must be sorted in descending size order.
        assert(std::is_sorted(out.begin(), out.end(), std::greater<>()));
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

    std::vector<size_t> slotSizeList() const
    {
        return getUniformAttribute(variants, [](auto const& v) { return v.slotSizeList(); });
    }
};

// ---------------------------------------------------------------------------
// BufferAttr — metadata for one buffer within storage.
// Mirrors _storage/_config.py::BufferAttr.
// ---------------------------------------------------------------------------
struct BufferAttr
{
    LifeCycleId lifeCycleId = 0;
    PoolIndex poolIndex = 0;
    size_t offset = 0; // byte offset within the slot
    size_t size = 0;   // expanded size of the buffer (after expansion)
    int expansion = 1; // expansion factor (tokens_per_block / tokens_per_block_override)
};

// ---------------------------------------------------------------------------
// Rational — exact fraction for slot utilization computations.
// Mirrors Python's fractions.Fraction used in _storage/_config.py::LayerAttr.
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
// LayerAttr — per-layer storage attributes for scratch slot management.
// Mirrors _storage/_config.py::LayerAttr.
// ---------------------------------------------------------------------------
struct LayerAttr
{
    LifeCycleId lifeCycleId = 0;

    // Number of sub-pages within a single coalesced slot that belong to this layer,
    // per pool within the pool group.
    std::vector<int> slotUtil;

    // Fraction of slot_util to total number of buffers in the slot, max over all pools.
    Rational slotUtilFracMax;
};

// ---------------------------------------------------------------------------
// StorageConfig — complete storage layout for the KV cache.
// Mirrors _storage/_config.py::StorageConfig.
// ---------------------------------------------------------------------------
struct StorageConfig
{
    std::vector<CacheTierConfig> cacheTiers; // from KVCacheManagerConfig
    std::vector<SlotDesc> slotDescList;      // indexed by PoolGroupIndex

    // Expansion factor per buffer (for heterogeneous tokens_per_block).
    std::unordered_map<BufferId, int, BufferIdHash> expansion;

    // Map each LifeCycleId → its PoolGroupIndex.
    std::vector<PoolGroupIndex> lifeCycleGrouping() const;

    // Attribute map for each buffer.
    std::map<BufferId, BufferAttr> bufferAttributes() const;

    // Map LayerId → LifeCycleId.
    std::unordered_map<LayerId, LifeCycleId> layerToLifeCycleIds() const;

    // Per-layer storage attributes (slot utilization within coalesced slots).
    // Mirrors _storage/_config.py::StorageConfig.layer_attributes().
    std::map<LayerId, LayerAttr> layerAttributes() const;

    int numLifeCycles() const;
};

// ---------------------------------------------------------------------------
// Factory: create StorageConfig from KVCacheManagerConfig.
// Mirrors _storage/_config.py::create_storage_config.
// ---------------------------------------------------------------------------
StorageConfig createStorageConfig(KVCacheManagerConfig const& config);

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
