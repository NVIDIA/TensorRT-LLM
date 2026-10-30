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

#include <algorithm>
#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

void SelectionUnitLayout::validate(size_t unitSize) const
{
    TLLM_CHECK_WITH_INFO(
        numChunks != 0 && chunkBytes != 0, "SelectionUnitLayout: numChunks and chunkBytes must be non-zero");
    TLLM_CHECK_WITH_INFO(static_cast<size_t>(numChunks) * chunkBytes == unitSize,
        "SelectionUnitLayout: numChunks * chunkBytes must equal one unit's bytes");
    TLLM_CHECK_WITH_INFO(numChunks <= 1 || chunkStride >= chunkBytes,
        "SelectionUnitLayout: chunkStride must not overlap consecutive chunks");
    TLLM_CHECK_WITH_INFO(
        unitStride >= chunkBytes, "SelectionUnitLayout: unitStride must not overlap consecutive units");
}

void BufferConfig::validate() const
{
    TLLM_CHECK_WITH_INFO(
        isSparse || sparseUnitChunks == 1, "BufferConfig: sparseUnitChunks set on a non-sparse buffer");
    // Both would name a unit count, and nothing reconciles them.
    TLLM_CHECK_WITH_INFO(!isSparse || !tokensPerBlockOverride.has_value(),
        "BufferConfig: tokensPerBlockOverride is not supported on a sparse buffer; use the layer's "
        "sparse_cache.unit_tokens");
    TLLM_CHECK_WITH_INFO(sparseUnitChunks > 0, "BufferConfig: sparseUnitChunks must be positive");
}

void SparseCacheConfig::validate(int tokensPerBlock) const
{
    TLLM_CHECK_WITH_INFO(numSelectedBlocks > 0 && maxCachedUnits > 0,
        "SparseCacheConfig: numSelectedBlocks and maxCachedUnits must be positive; omit sparse_cache "
        "entirely for a layer with no fetch path");
    // Equality leaves no cell for the unit being written, so the request could not take a
    // single token. The cache holds a full selection plus everything in flight, and those
    // populations are disjoint: selection is below the watermark, in flight is above it.
    TLLM_CHECK_WITH_INFO(maxCachedUnits > numSelectedBlocks,
        "SparseCacheConfig: maxCachedUnits must exceed numSelectedBlocks, to leave room for units in "
        "flight");
    TLLM_CHECK_WITH_INFO(unitTokens > 0 && tokensPerBlock % unitTokens == 0,
        "SparseCacheConfig: unitTokens must be a divisor of tokensPerBlock");
}

void AttentionLayerConfig::validate() const
{
    detail::validateNoDuplicateBufferRoles(buffers);
    for (auto const& buf : buffers)
    {
        buf.validate();
    }
    TLLM_CHECK_WITH_INFO(!sparseCache.has_value()
            || std::any_of(buffers.begin(), buffers.end(), [](BufferConfig const& b) { return b.isSparse; }),
        "AttentionLayerConfig: sparse_cache is set on a layer with no sparse buffer to serve");
}

void DiskCacheTierConfig::validate() const
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

namespace
{

//! What a layer's selection goes by, or nullopt where the layer has no sparse cache.
std::optional<std::pair<int, SparseCacheConfig const*>> selectionOf(LayerConfig const& layer)
{
    return std::visit(
        [](auto const& cfg) -> std::optional<std::pair<int, SparseCacheConfig const*>>
        {
            if constexpr (std::is_same_v<std::decay_t<decltype(cfg)>, AttentionLayerConfig>)
            {
                if (cfg.sparseCache.has_value())
                {
                    return std::pair{cfg.sparseCache->effectiveSelectionId(cfg.layerId), &*cfg.sparseCache};
                }
            }
            return std::nullopt;
        },
        layer);
}

//! An unset id reads as the layer's own, so the two namespaces are one and a layer that
//! named nothing can be named by someone else. Reject that, and require the layers of one
//! selection to agree on what they are selecting.
void _validateSelectionIds(std::vector<LayerConfig> const& layers)
{
    std::map<int, SparseCacheConfig const*> byId;
    std::set<int> silent;

    for (auto const& layer : layers)
    {
        auto const selection = selectionOf(layer);
        if (!selection.has_value())
        {
            continue;
        }
        auto const [id, cache] = *selection;
        if (!cache->selectionId.has_value())
        {
            silent.insert(id);
        }

        auto const [it, fresh] = byId.try_emplace(id, cache);
        if (fresh)
        {
            continue;
        }
        auto const& first = *it->second;
        if (first.numSelectedBlocks != cache->numSelectedBlocks || first.maxCachedUnits != cache->maxCachedUnits
            || first.unitTokens != cache->unitTokens)
        {
            throw AssertionError(
                "KVCacheManagerConfig: layers sharing a selection_id must agree on "
                "num_selected_blocks, max_cached_units and unit_tokens");
        }
    }

    for (auto const& layer : layers)
    {
        auto const selection = selectionOf(layer);
        if (!selection.has_value() || !selection->second->selectionId.has_value())
        {
            continue;
        }
        // Naming a layer that named nothing would share its selection without its consent.
        if (silent.count(*selection->second->selectionId) != 0)
        {
            throw AssertionError(
                "KVCacheManagerConfig: selection_id names a layer that set none of its own; "
                "a layer that does not name a selection never shares one");
        }
    }
}

} // namespace

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

    poolRebalance.validate();

    // These mirror Python's KVCacheManagerConfig.__post_init__ asserts, so they
    // throw AssertionError (translated in the binding layer) rather than ValueError.
    if (cacheTiers.empty() || cacheTierOf(cacheTiers[0]) != CacheTier::GPU_MEM)
    {
        throw AssertionError("KVCacheManagerConfig: first cache tier must be GPU memory");
    }

    for (auto const& tier : cacheTiers)
    {
        std::visit([](auto const& cfg) { cfg.validate(); }, tier);
    }

    // Zero asks for host-backed pools with no room for a single slot. It is not a way to turn
    // them off: whether a pool is host-backed is a property of the buffer, not of this quota.
    if (hostSparsePoolQuota.has_value() && *hostSparsePoolQuota == 0)
    {
        throw AssertionError("KVCacheManagerConfig: host_sparse_pool_quota must be positive when set");
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
                    if (buf.tokensPerBlockOverride.has_value()
                        && (*buf.tokensPerBlockOverride <= 0 || tokensPerBlock % *buf.tokensPerBlockOverride != 0))
                    {
                        throw AssertionError(
                            "KVCacheManagerConfig: tokensPerBlockOverride must be a divisor of "
                            "tokensPerBlock");
                    }
                }
                if constexpr (std::is_same_v<std::decay_t<decltype(cfg)>, AttentionLayerConfig>)
                {
                    if (cfg.sparseCache.has_value())
                    {
                        cfg.sparseCache->validate(tokensPerBlock);
                        int const unitsPerPage = cfg.sparseCache->unitsPerPage(tokensPerBlock);
                        for (auto const& buf : cfg.buffers)
                        {
                            // Reject an unusable layout now, not at the first fetch, when
                            // serving has begun. A zero-size buffer has no layout to be
                            // unusable, and is dropped before the storage layer sees it,
                            // which is how a dense one of no size is treated too.
                            if (buf.isSparse && buf.size != 0)
                            {
                                static_cast<void>(buf.sparseLayout(unitsPerPage));
                            }
                        }
                    }
                }
            },
            layer);
    }

    _validateSelectionIds(layers);

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
