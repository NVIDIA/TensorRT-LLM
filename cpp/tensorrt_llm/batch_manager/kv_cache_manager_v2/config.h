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

#include "tensorrt_llm/common/assert.h"
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <variant>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

// ---------------------------------------------------------------------------
// Cache tier configuration structs (mirrors _config.py)
// ---------------------------------------------------------------------------

struct GpuCacheTierConfig
{
    size_t quota = 0; // bytes

    CacheTier tier() const noexcept
    {
        return CacheTier::GPU_MEM;
    }

    void assertValid() const
    {
        if (quota == 0)
            throw std::invalid_argument("GpuCacheTierConfig: quota must be > 0");
    }
};

struct HostCacheTierConfig
{
    size_t quota = 0; // bytes

    CacheTier tier() const noexcept
    {
        return CacheTier::HOST_MEM;
    }

    void assertValid() const
    {
        if (quota == 0)
            throw std::invalid_argument("HostCacheTierConfig: quota must be > 0");
    }
};

struct DiskCacheTierConfig
{
    size_t quota = 0; // bytes
    std::string path; // directory for temp files

    CacheTier tier() const noexcept
    {
        return CacheTier::DISK;
    }

    void assertValid() const;
};

// Variant holding any tier config.
using CacheTierConfig = std::variant<GpuCacheTierConfig, HostCacheTierConfig, DiskCacheTierConfig>;

// Helper to extract tier from a variant.
inline CacheTier cacheTierOf(CacheTierConfig const& cfg)
{
    return std::visit([](auto const& c) { return c.tier(); }, cfg);
}

inline size_t cacheTierQuota(CacheTierConfig const& cfg)
{
    return std::visit([](auto const& c) { return c.quota; }, cfg);
}

// ---------------------------------------------------------------------------
// Buffer configuration (one KV buffer inside an attention layer).
// ---------------------------------------------------------------------------

struct BufferConfig
{
    DataRole role;
    size_t size = 0; // bytes per page (without expansion)

    // If set, overrides tokens_per_block for this buffer.
    // Must be a divisor of KVCacheManagerConfig::tokensPerBlock.
    std::optional<int> tokensPerBlockOverride;
};

// ---------------------------------------------------------------------------
// Layer type discriminator.
// ---------------------------------------------------------------------------

enum class LayerType : int
{
    ATTENTION = 0,
    SSM = 1,
};

// ---------------------------------------------------------------------------
// Attention layer configuration.
// ---------------------------------------------------------------------------

namespace detail
{

inline void validateNoDuplicateBufferRoles(std::vector<BufferConfig> const& buffers)
{
    std::unordered_set<DataRole> roles;
    for (auto const& buf : buffers)
    {
        if (!roles.insert(buf.role).second)
            throw std::invalid_argument("duplicate buffer role");
    }
}

} // namespace detail

struct AttentionLayerConfig
{
    static constexpr LayerType type = LayerType::ATTENTION;

    LayerId layerId = 0;
    std::vector<BufferConfig> buffers;

    // nullopt = no sliding window.
    std::optional<int> slidingWindowSize;

    // nullopt or 0 = no sink tokens.
    std::optional<int> numSinkTokens;

    std::optional<int> windowSize() const noexcept
    {
        return slidingWindowSize;
    }

    void validate() const
    {
        detail::validateNoDuplicateBufferRoles(buffers);
    }
};

// ---------------------------------------------------------------------------
// SSM (State Space Model) layer configuration.
// ---------------------------------------------------------------------------

struct SsmLayerConfig
{
    static constexpr LayerType type = LayerType::SSM;

    LayerId layerId = 0;
    std::vector<BufferConfig> buffers;

    void validate() const
    {
        detail::validateNoDuplicateBufferRoles(buffers);
        for (auto const& buf : buffers)
        {
            if (buf.tokensPerBlockOverride.has_value())
                throw std::invalid_argument("tokensPerBlockOverride not supported for SSM layers");
        }
    }
};

using LayerConfig = std::variant<AttentionLayerConfig, SsmLayerConfig>;

// ---------------------------------------------------------------------------
// KVCacheDesc — describes one KV cache request's capacity and history length.
// Mirrors _config.py::KVCacheDesc.
// ---------------------------------------------------------------------------
struct KVCacheDesc
{
    int capacity = 0;
    int historyLength = 0;

    void validate() const
    {
        TLLM_CHECK_DEBUG(0 <= historyLength && historyLength <= capacity);
    }
};

// ---------------------------------------------------------------------------
// BatchDesc — a batch of requests that the KVCacheManager must support.
// Mirrors _config.py::BatchDesc.
// ---------------------------------------------------------------------------
struct BatchDesc
{
    std::vector<KVCacheDesc> kvCaches;
    int systemPromptLength = 0; // tokens shared by all requests (0 if no reuse)

    void validate() const
    {
        TLLM_CHECK_DEBUG(systemPromptLength >= 0);
    }
};

// ---------------------------------------------------------------------------
// Helix (disaggregated serving) configuration — unsupported yet.
// ---------------------------------------------------------------------------
struct HelixConfig
{
    int helixGroupSize = 1;
    int helixGpuRank = 0;
    int helixShardSize = 0;
    int sharedCommPort = 0;
};

// ---------------------------------------------------------------------------
// SWA scratch reuse configuration.
// ---------------------------------------------------------------------------
struct SwaScratchReuseConfig
{
    int maxRewindLen = 0;

    void validate() const
    {
        if (maxRewindLen < 0)
        {
            throw std::invalid_argument("SwaScratchReuseConfig: max_rewind_len must be non-negative");
        }
    }
};

// ---------------------------------------------------------------------------
// Top-level KV cache manager configuration (mirrors _config.py::KVCacheManagerConfig).
// ---------------------------------------------------------------------------

struct KVCacheManagerConfig
{
    int tokensPerBlock = 0;

    // Ordered from warm (GPU) to cold (disk). First must be GPU memory.
    std::vector<CacheTierConfig> cacheTiers;

    // Layer configs (attention or SSM). Layer IDs must be unique.
    std::vector<LayerConfig> layers;

    // Suspend/resume threshold: if utilization > this, resuming will fail.
    float maxUtilForResume = 0.97f;

    // Try to reuse tokens from partially matched blocks.
    bool enablePartialReuse = true;

    // Constraint-based memory partitioning.
    std::vector<BatchDesc> constraints;   // batches that must always be supportable
    std::optional<BatchDesc> typicalStep; // typical step for initial ratio computation

    // Interval (in tokens) at which SSM state is snapshotted for prefix reuse.
    // Must be a positive multiple of tokensPerBlock. Only takes effect when SSM layers are present.
    int ssmReuseInterval = 512;

    // When set, SWA layers reuse physical pages for out-of-window blocks during prefill.
    // Scratch blocks share coalesced slot sub-pages across blocks for the currently executing
    // layer, reducing peak memory. Trade-off: KV cache reuse is degraded because scratch blocks
    // have no preserved data after the step.
    std::optional<SwaScratchReuseConfig> swaScratchReuse;

    bool enableSwaScratchReuse() const noexcept
    {
        return swaScratchReuse.has_value();
    }

    std::optional<HelixConfig> helixConfig; // unsupported yet

    void validate() const;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
