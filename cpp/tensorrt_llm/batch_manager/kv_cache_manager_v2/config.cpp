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
    if (cacheTiers.empty() || cacheTierOf(cacheTiers[0]) != CacheTier::GPU_MEM)
    {
        throw std::invalid_argument("KVCacheManagerConfig: first cache tier must be GPU memory");
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
                    throw std::invalid_argument("KVCacheManagerConfig: duplicate layer id");
                }
                for (auto const& buf : cfg.buffers)
                {
                    if (buf.tokensPerBlockOverride.has_value() && tokensPerBlock % *buf.tokensPerBlockOverride != 0)
                    {
                        throw std::invalid_argument(
                            "KVCacheManagerConfig: tokensPerBlockOverride must be a divisor of "
                            "tokensPerBlock");
                    }
                }
            },
            layer);
    }
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
