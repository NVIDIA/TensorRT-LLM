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
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
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

void RegionConfig::validate() const
{
    if (name.empty())
    {
        throw std::invalid_argument("RegionConfig: name must not be empty");
    }
    auto const fail
        = [this](std::string const& what) { throw std::invalid_argument("RegionConfig '" + name + "': " + what); };
    if (allocationUnit != RegionAllocationUnit::kRequest)
    {
        fail("only request regions are supported");
    }
    // A request region with any other policy would be a second snapshotted per-request life
    // cycle beside the SSM layers' own, and reuse matching assumes there is one.
    if (reusePolicy != ReusePolicy::PRIVATE)
    {
        fail("a request region must use the PRIVATE reuse policy");
    }
    if (buffers.empty())
    {
        fail("must declare at least one buffer");
    }
    if (fixedSlotCount.has_value() && *fixedSlotCount <= 0)
    {
        fail("fixed_slot_count must be positive when set");
    }
    std::set<std::pair<LayerId, DataRole>> seen;
    for (auto const& [layerId, buf] : buffers)
    {
        buf.validate();
        if (!seen.emplace(layerId, buf.role).second)
        {
            fail("declares buffer '" + buf.role + "' of layer " + std::to_string(layerId) + " twice");
        }
        if (buf.reusePolicy.has_value())
        {
            fail("buffer '" + buf.role + "' sets its own reuse policy; a region's buffers use the region's");
        }
        if (buf.tokensPerBlockOverride.has_value() || buf.isSparse)
        {
            fail("buffer '" + buf.role + "' sets tokensPerBlockOverride or isSparse, which request regions do not "
                "support");
        }
    }
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
    TLLM_CHECK_WITH_INFO(unitTokens > 0 && tokensPerBlock > 0 && tokensPerBlock % unitTokens == 0,
        "SparseCacheConfig: unitTokens must be a divisor of positive tokensPerBlock");
    // The arena rounds capacity up to whole pages. Bound it by the largest whole-page
    // capacity the probe supports, without rounding an arbitrarily large input upward.
    int const units = unitsPerPage(tokensPerBlock);
    int const maxSupportedUnits = (kMaxCacheEntries / units) * units;
    TLLM_CHECK_WITH_INFO(maxCachedUnits <= maxSupportedUnits,
        "SparseCacheConfig: maxCachedUnits=%d exceeds the supported capacity of %d units for %d units per page "
        "(sparse probe limit %d); reduce gpu_cache_factor or max_cached_units",
        maxCachedUnits, maxSupportedUnits, units, kMaxCacheEntries);
}

void AttentionLayerConfig::validate() const
{
    detail::validateNoDuplicateBufferRoles(buffers);
    for (auto const& buf : buffers)
    {
        buf.validate();
        // Sinks required, sliding window optional: the policy needs both a sliding window and sink tokens.
        if (buf.effectiveReusePolicy(reusePolicy) == ReusePolicy::SINK_REQUIRED_ONLY
            && (!slidingWindowSize.has_value() || numSinkTokens.value_or(0) == 0))
        {
            throw std::invalid_argument("SINK_REQUIRED_ONLY requires a sliding window with sink tokens");
        }
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

void RemoteCacheTierConfig::validate() const
{
    TLLM_CHECK_WITH_INFO(quota > 0, "RemoteCacheTierConfig: quota must be > 0");
    TLLM_CHECK_WITH_INFO(std::filesystem::is_directory(path),
        "RemoteCacheTierConfig: rendezvous directory '%s' is not a directory", path.c_str());
    TLLM_CHECK_WITH_INFO(0 < waitTimeoutMs && waitTimeoutMs <= kMaxRemoteTimeoutMs,
        "RemoteCacheTierConfig: wait_timeout_ms (donor_wait_timeout_s) must be in (0, %lld], got %lld",
        static_cast<long long>(kMaxRemoteTimeoutMs), static_cast<long long>(waitTimeoutMs));
    // The fabric transport's copies are stream-ordered and never timed.
    TLLM_CHECK_WITH_INFO(transfer.transport != RemoteTransport::kNixl
            || (0 < transfer.timeoutMs && transfer.timeoutMs <= kMaxRemoteTimeoutMs),
        "RemoteCacheTierConfig: the NIXL transfer timeout_ms (transfer_timeout_s) must be in (0, %lld], got %lld",
        static_cast<long long>(kMaxRemoteTimeoutMs), static_cast<long long>(transfer.timeoutMs));
    TLLM_CHECK_WITH_INFO(std::all_of(poolRatio.begin(), poolRatio.end(),
                             [](double weight) { return std::isfinite(weight) && weight > 0.0; }),
        "RemoteCacheTierConfig: pool ratio entries must be positive and finite");
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

//! Region checks that need the layers: every region buffer sits on an SSM layer, and each
//! (layer, role) pair belongs to one layer config or one region.
void _validateRegions(std::vector<LayerConfig> const& layers, std::vector<RegionConfig> const& regions)
{
    std::set<LayerId> ssmLayerIds;
    std::set<std::pair<LayerId, DataRole>> owned;
    for (auto const& layer : layers)
    {
        std::visit(
            [&](auto const& cfg)
            {
                if constexpr (std::is_same_v<std::decay_t<decltype(cfg)>, SsmLayerConfig>)
                {
                    ssmLayerIds.insert(cfg.layerId);
                }
                for (auto const& buf : cfg.buffers)
                {
                    owned.emplace(cfg.layerId, buf.role);
                }
            },
            layer);
    }
    std::set<std::string> names;
    for (auto const& region : regions)
    {
        region.validate();
        if (!names.insert(region.name).second)
        {
            throw std::invalid_argument("KVCacheManagerConfig: region name '" + region.name + "' is declared twice");
        }
        for (auto const& [layerId, buf] : region.buffers)
        {
            if (ssmLayerIds.count(layerId) == 0)
            {
                throw std::invalid_argument("KVCacheManagerConfig: region '" + region.name + "' places buffer '"
                    + buf.role + "' on layer " + std::to_string(layerId)
                    + ", which has no SsmLayerConfig; request regions need one");
            }
            if (!owned.emplace(layerId, buf.role).second)
            {
                throw std::invalid_argument("KVCacheManagerConfig: buffer '" + buf.role + "' of layer "
                    + std::to_string(layerId) + " belongs to region '" + region.name
                    + "' and to a layer config or another region");
            }
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

    // Remote levels move pages through the page staging buffer, which cannot reach a file-backed disk level, and share
    // one transport, so that their copies are either all stream-ordered or all host-synchronous.
    std::optional<RemoteTransport> remoteTransport;
    bool hasFileDisk = false;
    for (auto const& tier : cacheTiers)
    {
        if (std::holds_alternative<DiskCacheTierConfig>(tier))
        {
            hasFileDisk = true;
        }
        else if (auto const* remote = std::get_if<RemoteCacheTierConfig>(&tier))
        {
            if (remoteTransport.value_or(remote->transfer.transport) != remote->transfer.transport)
            {
                throw AssertionError("KVCacheManagerConfig: remote cache levels must share one transport");
            }
            remoteTransport = remote->transfer.transport;
        }
    }
    if (remoteTransport.has_value() && hasFileDisk)
    {
        throw AssertionError(
            "KVCacheManagerConfig: a remote cache level cannot be combined with a file-backed disk level");
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

    _validateRegions(layers, regions);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
