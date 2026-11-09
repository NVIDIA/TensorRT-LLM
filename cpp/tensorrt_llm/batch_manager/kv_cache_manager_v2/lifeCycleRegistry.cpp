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

LifeCycle makeLifeCycle(RegionConfig const& region, int regionIndex)
{
    // RegionConfig::validate() admits request regions only; a token-block region would map
    // to an AttnLifeCycle with the region's own window.
    TLLM_CHECK_WITH_INFO(region.allocationUnit == RegionAllocationUnit::kRequest,
        "Region '%s': only request regions are supported", region.name.c_str());
    return SsmLifeCycle{region.reusePolicy, regionIndex};
}

// ---------------------------------------------------------------------------
// LifeCycleRegistry
// ---------------------------------------------------------------------------

LifeCycleRegistry::LifeCycleRegistry(KVCacheManagerConfig const& config)
{
    auto const registerLifeCycle = [this](LifeCycle const& lc)
    {
        if (mLifeCycleIdMap.find(lc) == mLifeCycleIdMap.end())
        {
            check();
            LifeCycleId id = mLifeCycleList.size();
            mLifeCycleList.push_back(lc);
            mLifeCycleIdMap[lc] = id;
            if (std::holds_alternative<SsmLifeCycle>(lc))
                mSsmLifeCycleIds.push_back(id);
        }
    };
    for (auto const& layer : config.layers)
    {
        std::visit([](auto const& cfg) { cfg.validate(); }, layer);
        for (auto const& buffer : layerBuffers(layer))
        {
            registerLifeCycle(makeLifeCycle(layer, buffer, config.tokensPerBlock));
        }
    }
    // The region index is part of a region's lifecycle, so each region registers a new one.
    mRegionLifeCycleIds.reserve(config.regions.size());
    for (size_t regionIndex = 0; regionIndex < config.regions.size(); ++regionIndex)
    {
        auto const& region = config.regions[regionIndex];
        region.validate();
        auto const lc = makeLifeCycle(region, static_cast<int>(regionIndex));
        registerLifeCycle(lc);
        auto const id = getId(lc);
        mRegionLifeCycleIds.push_back(id);
        TLLM_CHECK_WITH_INFO(mRegionLifeCycleIdByName.emplace(region.name, id).second,
            "Region name '%s' is declared twice", region.name.c_str());
    }
    check();
}

LifeCycleId LifeCycleRegistry::regionLifeCycleId(std::string const& name) const
{
    auto const it = mRegionLifeCycleIdByName.find(name);
    if (it == mRegionLifeCycleIdByName.end())
    {
        throw std::out_of_range("LifeCycleRegistry: no region named '" + name + "'");
    }
    return it->second;
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

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
