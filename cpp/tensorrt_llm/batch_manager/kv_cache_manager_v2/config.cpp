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

#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>
#include <variant>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

void DiskCacheTierConfig::assertValid() const
{
    if (quota == 0)
    {
        throw std::invalid_argument("DiskCacheTierConfig: quota must be > 0");
    }
    if (store.has_value())
    {
        // A store level keeps no files: its slots are objects of the store the IO table talks to.
        if (!path.empty())
        {
            throw std::invalid_argument("DiskCacheTierConfig: a store tier takes no path, got '" + path + "'");
        }
        return;
    }
    if (!std::filesystem::is_directory(path))
    {
        throw std::invalid_argument("DiskCacheTierConfig: path '" + path + "' is not a directory");
    }
}

void RemoteHostTierConfig::assertValid() const
{
    if (chunkBytes == 0)
    {
        throw std::invalid_argument("RemoteHostTierConfig: chunk_bytes must be > 0");
    }
    if (fabricHandles.empty())
    {
        throw std::invalid_argument("RemoteHostTierConfig: no fabric handles");
    }
    for (auto const& handle : fabricHandles)
    {
        if (handle.size() != kFabricHandleBytes)
        {
            throw std::invalid_argument("RemoteHostTierConfig: a fabric handle must be "
                + std::to_string(kFabricHandleBytes) + " bytes, got " + std::to_string(handle.size()));
        }
    }
}

void HostCacheTierConfig::assertValid() const
{
    if (quota == 0)
    {
        throw std::invalid_argument("HostCacheTierConfig: quota must be > 0");
    }
    if (remote.has_value())
    {
        remote->assertValid();
        if (quota > remote->capacity())
        {
            throw std::invalid_argument("HostCacheTierConfig: remote quota " + std::to_string(quota)
                + " exceeds the lent capacity " + std::to_string(remote->capacity()));
        }
    }
}

void KVCacheManagerConfig::validate() const
{
    if (swaScratchReuse.has_value())
    {
        swaScratchReuse->validate();
    }
    for (auto const& batch : constraints)
    {
        batch.validate();
    }
    if (typicalStep.has_value())
    {
        typicalStep->validate();
    }

    // These mirror Python's KVCacheManagerConfig.__post_init__ asserts, so they
    // throw AssertionError (translated in the binding layer) rather than ValueError.
    if (cacheTiers.empty() || cacheTierOf(cacheTiers[0]) != CacheTier::GPU_MEM)
    {
        throw AssertionError("KVCacheManagerConfig: first cache tier must be GPU memory");
    }

    // A store level holds cold pages in objects of an external store, written from and read into the local host level.
    // It must be the single last level, directly below one local host level, and replaces both a disk level and a
    // remote host level (they cannot be combined).
    for (size_t level = 0; level < cacheTiers.size(); ++level)
    {
        auto const* disk = std::get_if<DiskCacheTierConfig>(&cacheTiers[level]);
        if (disk == nullptr || !disk->isStore())
        {
            continue;
        }
        disk->assertValid();
        if (level + 1 != cacheTiers.size())
        {
            throw std::invalid_argument("KVCacheManagerConfig: the store tier must be the last cache tier");
        }
        for (auto const& tier : cacheTiers)
        {
            auto const* host = std::get_if<HostCacheTierConfig>(&tier);
            if (host != nullptr && host->isRemote())
            {
                throw std::invalid_argument(
                    "KVCacheManagerConfig: the remote host tier and the store tier are mutually exclusive");
            }
        }
        for (size_t other = 0; other < cacheTiers.size(); ++other)
        {
            if (other != level && std::holds_alternative<DiskCacheTierConfig>(cacheTiers[other]))
            {
                throw std::invalid_argument("KVCacheManagerConfig: the store tier cannot be combined with a disk tier");
            }
        }
        auto const* localHost = level >= 2 ? std::get_if<HostCacheTierConfig>(&cacheTiers[1]) : nullptr;
        if (level != 2 || localHost == nullptr || localHost->isRemote())
        {
            throw std::invalid_argument(
                "KVCacheManagerConfig: the store tier must follow a GPU tier and one local host tier");
        }
    }

    // A remote host level holds lent memory that only the GPU can reach. It must be the single last level, below a
    // local host level, so every page reaches it from local memory and nothing below it needs CPU access.
    for (size_t level = 0; level < cacheTiers.size(); ++level)
    {
        auto const* host = std::get_if<HostCacheTierConfig>(&cacheTiers[level]);
        if (host == nullptr || !host->isRemote())
        {
            continue;
        }
        host->assertValid();
        if (level + 1 != cacheTiers.size())
        {
            throw std::invalid_argument("KVCacheManagerConfig: the remote host tier must be the last cache tier");
        }
        auto const* localHost = level >= 2 ? std::get_if<HostCacheTierConfig>(&cacheTiers[1]) : nullptr;
        if (level != 2 || localHost == nullptr || localHost->isRemote())
        {
            throw std::invalid_argument(
                "KVCacheManagerConfig: the remote host tier must follow a GPU tier and one local host tier");
        }
    }

    // Check for duplicate layer ids.
    std::set<LayerId> seenLayerIds;
    for (auto const& layer : layers)
    {
        std::visit(
            [&](auto const& cfg)
            {
                cfg.validate();
                if (!seenLayerIds.insert(cfg.layerId).second)
                {
                    throw AssertionError("KVCacheManagerConfig: duplicate layer id");
                }
                for (auto const& buf : cfg.buffers)
                {
                    if (buf.isSparse && (cacheTiers.size() < 2 || cacheTierOf(cacheTiers[1]) != CacheTier::HOST_MEM))
                    {
                        throw std::invalid_argument("Sparse buffers require cache level 1 to use HOST_MEM");
                    }
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
