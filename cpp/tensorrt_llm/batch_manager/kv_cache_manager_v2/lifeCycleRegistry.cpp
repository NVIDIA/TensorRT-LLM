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

#include "kv_cache_manager_v2/lifeCycleRegistry.h"

#include "tensorrt_llm/common/assert.h"
#include <stdexcept>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// makeLifeCycle — factory dispatching on LayerConfig variant.
// ---------------------------------------------------------------------------

LifeCycle makeLifeCycle(LayerConfig const& layer, BufferConfig const& buffer, int tokensPerBlock)
{
    TLLM_CHECK_WITH_INFO(!buffer.allocationGroupId.has_value(),
        "A buffer with allocation_group_id must resolve its declared allocation group");
    return std::visit(
        [&](auto const& cfg) -> LifeCycle
        {
            auto const policy = buffer.effectiveReusePolicy(cfg.reusePolicy);
            using T = std::decay_t<decltype(cfg)>;
            if constexpr (std::is_same_v<T, SsmLayerConfig>)
            {
                return SsmLifeCycle{policy};
            }
            else
            {
                return AttnLifeCycle::make(cfg.slidingWindowSize, cfg.numSinkTokens, tokensPerBlock, policy);
            }
        },
        layer);
}

LifeCycle makeLifeCycle(AllocationGroupConfig const& group, int tokensPerBlock)
{
    group.validate();
    if (group.allocationUnit == AllocationUnit::kFixed)
    {
        return SsmLifeCycle{group.reusePolicy};
    }
    return AttnLifeCycle::make(group.slidingWindowSize, group.numSinkTokens, tokensPerBlock, group.reusePolicy);
}

// ---------------------------------------------------------------------------
// LifeCycleRegistry
// ---------------------------------------------------------------------------

LifeCycleRegistry::LifeCycleRegistry(KVCacheManagerConfig const& config)
{
    config.validateAllocationGroups();
    std::map<AllocationGroupId, AllocationGroupConfig const*> explicitGroups;
    for (auto const& group : config.allocationGroups)
    {
        explicitGroups.emplace(group.allocationGroupId, &group);
    }
    auto const appendLifeCycle = [this](LifeCycle const& lc, bool initiallyActive = true,
                                     SuspendPolicy suspendPolicy = SuspendPolicy::kHold, int alignment = 1)
    {
        LifeCycleId const id = mLifeCycleList.size();
        mLifeCycleList.push_back(lc);
        mInitiallyActive.push_back(initiallyActive);
        mSuspendPolicies.push_back(suspendPolicy);
        mReuseAlignments.push_back(alignment);
        if (std::holds_alternative<SsmLifeCycle>(lc))
        {
            mSsmLifeCycleIds.push_back(id);
        }
        return id;
    };
    auto const registerLifeCycle = [this, &appendLifeCycle](LifeCycle const& lc)
    {
        if (mLifeCycleIdMap.find(lc) == mLifeCycleIdMap.end())
        {
            check();
            mLifeCycleIdMap[lc] = appendLifeCycle(lc);
        }
    };
    for (auto const& layer : config.layers)
    {
        std::visit([](auto const& cfg) { cfg.validate(); }, layer);
        for (auto const& buffer : layerBuffers(layer))
        {
            if (buffer.allocationGroupId.has_value())
            {
                AllocationGroupId const id = *buffer.allocationGroupId;
                if (mAllocationGroupIdMap.find(id) == mAllocationGroupIdMap.end())
                {
                    auto const& group = *explicitGroups.at(id);
                    mAllocationGroupIdMap[id] = appendLifeCycle(makeLifeCycle(group, config.tokensPerBlock),
                        group.initiallyActive, group.suspendPolicy, group.reuseAlignment);
                }
            }
            else
            {
                registerLifeCycle(makeLifeCycle(layer, buffer, config.tokensPerBlock));
            }
        }
    }
    check();
}

LifeCycle const& LifeCycleRegistry::operator[](LifeCycleId id) const
{
    return mLifeCycleList.at(id);
}

LifeCycle const& LifeCycleRegistry::getLifeCycle(LifeCycleId id) const
{
    return (*this)[id];
}

LifeCycleId LifeCycleRegistry::getId(LifeCycle const& lc) const
{
    auto it = mLifeCycleIdMap.find(lc);
    if (it == mLifeCycleIdMap.end())
    {
        throw std::out_of_range("LifeCycleRegistry::getId: life cycle not found");
    }
    return it->second;
}

LifeCycleId LifeCycleRegistry::getId(AllocationGroupId id) const
{
    auto const found = mAllocationGroupIdMap.find(id);
    TLLM_CHECK_WITH_INFO(found != mAllocationGroupIdMap.end(), "Unknown allocation_group_id %d", id.value());
    return found->second;
}

LifeCycleRegistry::ActiveGroups LifeCycleRegistry::activeGroups(
    std::vector<AllocationGroupId> const& additionalGroups) const
{
    auto result = mInitiallyActive;
    for (auto const id : additionalGroups)
    {
        result[getId(id)] = true;
    }
    return result;
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
