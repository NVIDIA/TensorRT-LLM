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
#include "kv_cache_manager_v2/utils/math.h"
#include "tensorrt_llm/common/assert.h"

#include <algorithm>
#include <map>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// AttnLifeCycle — lifecycle for attention layers (SWA + sink blocks).
// Mirrors _life_cycle_registry.py::AttnLifeCycle.
// ---------------------------------------------------------------------------
struct AttnLifeCycle
{
    std::optional<int> windowSize; // nullopt = no sliding window
    int numSinkBlocks = 0;         // divUp(numSinkTokens, tokensPerBlock)

    HalfOpenRange<BlockOrdinal> getStaleRange(int historyLength, int tokensPerBlock) const
    {
        int numBlocks = divUp(historyLength, tokensPerBlock);
        BlockOrdinal start{std::min(numBlocks, numSinkBlocks)};
        if (!windowSize.has_value())
            return {start, start};
        BlockOrdinal windowStart{(historyLength + 1 - *windowSize) / tokensPerBlock};
        return {start, std::max(start, windowStart)};
    }

    bool operator==(AttnLifeCycle const& o) const noexcept
    {
        return windowSize == o.windowSize && numSinkBlocks == o.numSinkBlocks;
    }

    bool operator<(AttnLifeCycle const& o) const noexcept
    {
        if (windowSize != o.windowSize)
            return windowSize < o.windowSize;
        return numSinkBlocks < o.numSinkBlocks;
    }

    static AttnLifeCycle make(std::optional<int> ws, std::optional<int> numSinkTokens, int tokensPerBlock)
    {
        TLLM_CHECK_DEBUG(tokensPerBlock > 0);
        TLLM_CHECK_DEBUG(!ws.has_value() || *ws > 0);
        TLLM_CHECK_DEBUG(!numSinkTokens.has_value() || *numSinkTokens >= 0);
        TLLM_CHECK_DEBUG((!numSinkTokens.has_value() || *numSinkTokens == 0) || ws.has_value());
        int sinkBlocks = divUp(numSinkTokens.value_or(0), tokensPerBlock);
        return AttnLifeCycle{ws, sinkBlocks};
    }
};

// ---------------------------------------------------------------------------
// SsmLifeCycle — lifecycle for SSM (State Space Model) layers.
// All blocks before the last full block are stale (recurrent state).
// Singleton: all SSM layers share the same lifecycle.
// ---------------------------------------------------------------------------
struct SsmLifeCycle
{
    HalfOpenRange<BlockOrdinal> getStaleRange(int historyLength, int tokensPerBlock) const
    {
        return {BlockOrdinal{0}, BlockOrdinal{historyLength / tokensPerBlock}};
    }

    bool operator==(SsmLifeCycle const&) const noexcept
    {
        return true;
    }

    bool operator<(SsmLifeCycle const&) const noexcept
    {
        return false;
    }
};

// ---------------------------------------------------------------------------
// LifeCycle — variant of attention or SSM lifecycle.
// ---------------------------------------------------------------------------
using LifeCycle = std::variant<AttnLifeCycle, SsmLifeCycle>;

// Free function: compute stale range via std::visit.
inline HalfOpenRange<BlockOrdinal> getStaleRange(LifeCycle const& lc, int historyLength, int tokensPerBlock)
{
    return std::visit([&](auto const& v) { return v.getStaleRange(historyLength, tokensPerBlock); }, lc);
}

// Compute the range of blocks that should use scratch (shared) slots during SWA prefill.
// Scratch = stale_at_capacity ∩ input_blocks, where:
//   stale_at_capacity: blocks out-of-window when all capacity tokens become history.
//   input_blocks: [divUp(historyLength, tpb), divUp(capacity, tpb)) — new blocks
//     for the current chunk.
// Mirrors _life_cycle_registry.py::compute_scratch_range().
inline HalfOpenRange<BlockOrdinal> computeScratchRange(
    LifeCycle const& lc, int historyLength, int capacity, int tokensPerBlock, int maxRewindLen)
{
    auto const* attn = std::get_if<AttnLifeCycle>(&lc);
    if (!attn || !attn->windowSize.has_value())
    {
        return {BlockOrdinal{0}, BlockOrdinal{0}};
    }
    int const nonRewindableCapacity = std::max(0, capacity - maxRewindLen);
    auto capStale = attn->getStaleRange(nonRewindableCapacity, tokensPerBlock);
    HalfOpenRange<BlockOrdinal> inputRange{divUp(historyLength, tokensPerBlock), divUp(capacity, tokensPerBlock)};
    return intersect(capStale, inputRange);
}

// Factory: create a LifeCycle from a LayerConfig variant.
LifeCycle makeLifeCycle(LayerConfig const& layer, int tokensPerBlock);

// Integer id assigned to each unique LifeCycle.
using LifeCycleId = StrongIndex<int, struct LifeCycleIdTag>;

// Alias for public exposure (same meaning as LifeCycleId).
using LayerGroupId = LifeCycleId;

// ---------------------------------------------------------------------------
// LifeCycleRegistry — deduplicates LifeCycle objects and assigns integer ids.
// Mirrors _life_cycle_registry.py::LifeCycleRegistry.
// ---------------------------------------------------------------------------
class LifeCycleRegistry
{
public:
    explicit LifeCycleRegistry(KVCacheManagerConfig const& config);

    // Look up a LifeCycle by id.
    LifeCycle const& operator[](LifeCycleId id) const;
    LifeCycle const& getLifeCycle(LifeCycleId id) const;

    // Look up the id for a LifeCycle (throws if not found).
    LifeCycleId getId(LifeCycle const& lc) const;

    // Number of unique life cycles.
    LifeCycleId size() const noexcept;

    // Iteration over LifeCycles in registration order.
    TypedVec<LifeCycleId, LifeCycle> const& getAll() const noexcept
    {
        return mLifeCycleList;
    }

    bool contains(LifeCycle const& lc) const noexcept
    {
        return mLifeCycleIdMap.count(lc) > 0;
    }

    // SSM helpers.
    std::optional<LifeCycleId> ssmLifeCycleId() const noexcept
    {
        return mSsmLifeCycleId;
    }

    bool hasSSM() const noexcept
    {
        return mSsmLifeCycleId.has_value();
    }

    // Return (id, AttnLifeCycle*) pairs for attention lifecycles only. Used by _setupForReuse.
    std::vector<std::pair<LifeCycleId, AttnLifeCycle const*>> attentionLifeCycles() const
    {
        std::vector<std::pair<LifeCycleId, AttnLifeCycle const*>> result;
        for (LifeCycleId lcId{0}; lcId < mLifeCycleList.size(); ++lcId)
        {
            if (auto const* attn = std::get_if<AttnLifeCycle>(&mLifeCycleList[lcId]))
                result.emplace_back(lcId, attn);
        }
        return result;
    }

    // Iterate: (id, lifecycle) pairs — all entries.
    struct Item
    {
        LifeCycleId id;
        LifeCycle const& lc;
    };

    class ItemIterator
    {
    public:
        ItemIterator(TypedVec<LifeCycleId, LifeCycle> const& list, LifeCycleId pos)
            : mList(&list)
            , mPos(pos)
        {
        }

        Item operator*() const
        {
            return {mPos, (*mList)[mPos]};
        }

        ItemIterator& operator++()
        {
            ++mPos;
            return *this;
        }

        bool operator!=(ItemIterator const& o) const
        {
            return mPos != o.mPos;
        }

    private:
        TypedVec<LifeCycleId, LifeCycle> const* mList;
        LifeCycleId mPos;
    };

    ItemIterator begin() const
    {
        return {mLifeCycleList, LifeCycleId{0}};
    }

    ItemIterator end() const
    {
        return {mLifeCycleList, mLifeCycleList.size()};
    }

private:
    void check() const;

    TypedVec<LifeCycleId, LifeCycle> mLifeCycleList;
    std::map<LifeCycle, LifeCycleId> mLifeCycleIdMap;
    std::optional<LifeCycleId> mSsmLifeCycleId;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
