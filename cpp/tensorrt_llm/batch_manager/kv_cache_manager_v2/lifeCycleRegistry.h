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

    HalfOpenRange getStaleRange(int historyLength, int tokensPerBlock) const
    {
        int numBlocks = divUp(historyLength, tokensPerBlock);
        int start = std::min(numBlocks, numSinkBlocks);
        if (!windowSize.has_value())
            return {start, start};
        int windowStart = (historyLength + 1 - *windowSize) / tokensPerBlock;
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
    HalfOpenRange getStaleRange(int historyLength, int tokensPerBlock) const
    {
        return {0, historyLength / tokensPerBlock};
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
inline HalfOpenRange getStaleRange(LifeCycle const& lc, int historyLength, int tokensPerBlock)
{
    return std::visit([&](auto const& v) { return v.getStaleRange(historyLength, tokensPerBlock); }, lc);
}

// Factory: create a LifeCycle from a LayerConfig variant.
LifeCycle makeLifeCycle(LayerConfig const& layer, int tokensPerBlock);

// Integer id assigned to each unique LifeCycle.
using LifeCycleId = int;

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
    std::vector<LifeCycle> const& getAll() const noexcept
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
        for (size_t i = 0; i < mLifeCycleList.size(); ++i)
        {
            if (auto const* attn = std::get_if<AttnLifeCycle>(&mLifeCycleList[i]))
                result.emplace_back(static_cast<LifeCycleId>(i), attn);
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
        ItemIterator(std::vector<LifeCycle> const& list, size_t pos)
            : mList(&list)
            , mPos(pos)
        {
        }

        Item operator*() const
        {
            return {static_cast<LifeCycleId>(mPos), (*mList)[mPos]};
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
        std::vector<LifeCycle> const* mList;
        size_t mPos;
    };

    ItemIterator begin() const
    {
        return {mLifeCycleList, 0};
    }

    ItemIterator end() const
    {
        return {mLifeCycleList, mLifeCycleList.size()};
    }

private:
    std::vector<LifeCycle> mLifeCycleList; // indexed by LifeCycleId
    std::map<LifeCycle, LifeCycleId> mLifeCycleIdMap;
    std::optional<LifeCycleId> mSsmLifeCycleId;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
