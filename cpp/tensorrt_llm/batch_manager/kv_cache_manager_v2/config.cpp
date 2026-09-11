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

#include "kv_cache_manager_v2/config.h"
#include "kv_cache_manager_v2/exceptions.h"

#include <cmath>
#include <filesystem>
#include <set>
#include <stdexcept>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

void DiskCacheTierConfig::assertValid() const
{
    if (quota == 0)
    {
        throw std::invalid_argument("DiskCacheTierConfig: quota must be > 0");
    }
    if (!std::filesystem::is_directory(path))
    {
        throw std::invalid_argument("DiskCacheTierConfig: path '" + path + "' is not a directory");
    }
}

void KVCacheManagerConfig::validate() const
{
    if (swaScratchReuse.has_value())
    {
        swaScratchReuse->validate();
    }

    if (rebalanceMinSampledKvCaches < 0)
    {
        throw std::invalid_argument("KVCacheManagerConfig: rebalance_min_sampled_kv_caches must be non-negative");
    }
    if (!std::isfinite(rebalanceCooldownSecs) || rebalanceCooldownSecs < 0.0)
    {
        throw std::invalid_argument("KVCacheManagerConfig: rebalance_cooldown_secs must be finite and non-negative");
    }
    if (rebalanceTargetRatioUpdateInterval <= 0)
    {
        throw std::invalid_argument("KVCacheManagerConfig: rebalance_target_ratio_update_interval must be positive");
    }
    if (!std::isfinite(rebalanceRatioThreshold) || rebalanceRatioThreshold <= 1.0f)
    {
        throw std::invalid_argument("KVCacheManagerConfig: rebalance_ratio_threshold must be finite and > 1.0");
    }
    if (rebalanceMovingAverageDecay <= 0.0 || rebalanceMovingAverageDecay >= 1.0)
    {
        throw std::invalid_argument("KVCacheManagerConfig: rebalance_moving_average_decay must be in (0, 1)");
    }

    // These mirror Python's KVCacheManagerConfig.__post_init__ asserts, so they
    // throw AssertionError (translated in the binding layer) rather than ValueError.
    if (cacheTiers.empty() || cacheTierOf(cacheTiers[0]) != CacheTier::GPU_MEM)
    {
        throw AssertionError("KVCacheManagerConfig: first cache tier must be GPU memory");
    }

    // Check for duplicate layer ids.
    std::set<LayerId> seenLayerIds;
    for (auto const& layer : layers)
    {
        std::visit(
            [&](auto const& cfg)
            {
                if (!seenLayerIds.insert(cfg.layerId).second)
                {
                    throw AssertionError("KVCacheManagerConfig: duplicate layer id");
                }
                for (auto const& buf : cfg.buffers)
                {
                    if (buf.tokensPerBlockOverride.has_value()
                        && (*buf.tokensPerBlockOverride <= 0 || tokensPerBlock % *buf.tokensPerBlockOverride != 0))
                    {
                        throw AssertionError(
                            "KVCacheManagerConfig: tokensPerBlockOverride must be a divisor of "
                            "tokensPerBlock");
                    }
                }
            },
            layer);
    }

    // SSM-specific validation.
    bool hasSSM = false;
    for (auto const& layer : layers)
    {
        if (std::holds_alternative<SsmLayerConfig>(layer))
        {
            hasSSM = true;
            break;
        }
    }
    if (hasSSM)
    {
        if (!commitMinSnapshot)
            throw AssertionError("KVCacheManagerConfig: commit_min_snapshot must be True when SSM layers are present");
    }
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
