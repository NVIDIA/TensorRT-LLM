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

LifeCycle makeLifeCycle(LayerConfig const& layer, int tokensPerBlock)
{
    return std::visit(
        [&](auto const& cfg) -> LifeCycle
        {
            cfg.validate();
            using T = std::decay_t<decltype(cfg)>;
            if constexpr (std::is_same_v<T, SsmLayerConfig>)
                return SsmLifeCycle{};
            else
                return AttnLifeCycle::make(cfg.slidingWindowSize, cfg.numSinkTokens, tokensPerBlock);
        },
        layer);
}

// ---------------------------------------------------------------------------
// LifeCycleRegistry
// ---------------------------------------------------------------------------

LifeCycleRegistry::LifeCycleRegistry(KVCacheManagerConfig const& config)
{
    for (auto const& layer : config.layers)
    {
        LifeCycle lc = makeLifeCycle(layer, config.tokensPerBlock);
        if (mLifeCycleIdMap.find(lc) == mLifeCycleIdMap.end())
        {
            check();
            LifeCycleId id = mLifeCycleList.size();
            mLifeCycleList.push_back(lc);
            mLifeCycleIdMap[lc] = id;
            if (std::holds_alternative<SsmLifeCycle>(lc))
                mSsmLifeCycleId = id;
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

LifeCycleId LifeCycleRegistry::size() const noexcept
{
    check();
    return mLifeCycleList.size();
}

inline void LifeCycleRegistry::check() const
{
    TLLM_CHECK_DEBUG_WITH_INFO(
        toSizeT(mLifeCycleList.size()) == mLifeCycleIdMap.size(), "corrupted life cycle registry");
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
