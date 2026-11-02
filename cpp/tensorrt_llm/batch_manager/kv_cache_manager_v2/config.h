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
#include <cmath>
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

    [[nodiscard]] CacheTier tier() const noexcept
    {
        return CacheTier::GPU_MEM;
    }

    void validate() const
    {
        if (quota == 0)
            throw std::invalid_argument("GpuCacheTierConfig: quota must be > 0");
    }
};

struct HostCacheTierConfig
{
    size_t quota = 0; // bytes

    //! Upper bound this tier may ever be resized to, in bytes.
    //!
    //! The address range is reserved once at this size so that growing the tier
    //! only commits more of it and never moves the base address. Reserving costs
    //! address space, not memory. A resize past this bound is rejected.
    //!
    //! Defaults to the host memory the OS reports, which no allocation can
    //! exceed anyway.
    std::optional<size_t> maxQuota;

    //! Whether a page may come from a NUMA node other than the one the GPU
    //! attaches to, when the local node cannot satisfy it.
    //!
    //! False keeps every page local, at the cost of capping the tier at one
    //! node's memory: a quota sized from total system memory cannot then be
    //! filled. True accepts a remote page, which the GPU reaches more slowly,
    //! rather than failing the allocation.
    //!
    //! False additionally requires that a memory policy be settable. Where the
    //! host tier is mmap-backed, a container without CAP_SYS_NICE cannot set
    //! one, and the tier refuses to start rather than fall back to first touch
    //! -- which prefers the local node but takes a remote page instead of
    //! failing, and so would not be the strictness that was asked for.
    bool allowRemoteNumaFallback = true;

    [[nodiscard]] CacheTier tier() const noexcept
    {
        return CacheTier::HOST_MEM;
    }

    void validate() const
    {
        if (quota == 0)
            throw std::invalid_argument("HostCacheTierConfig: quota must be > 0");
        if (maxQuota.has_value() && *maxQuota < quota)
            throw std::invalid_argument("HostCacheTierConfig: maxQuota must be >= quota");
    }
};

struct DiskCacheTierConfig
{
    size_t quota = 0; // bytes
    std::string path; // directory for temp files

    [[nodiscard]] CacheTier tier() const noexcept
    {
        return CacheTier::DISK;
    }

    void validate() const;
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

//! The configured ceiling on this tier's quota, where it has one. Only the host
//! tier reserves address space up front and so carries a maximum; the others are
//! bounded by the device or the filesystem.
inline std::optional<size_t> cacheTierMaxQuota(CacheTierConfig const& cfg)
{
    auto const* host = std::get_if<HostCacheTierConfig>(&cfg);
    return host != nullptr ? host->maxQuota : std::nullopt;
}

//! Controls whether a lifecycle participates in prefix matching and page reuse.
enum class ReusePolicy
{
    //! Commit reusable pages and require live-window coverage when matching a prefix.
    REQUIRED = 0,
    //! Keep request-local working pages; never publish or reuse them through the radix tree.
    PRIVATE = 1,
    //! Commit reusable pages without constraining the REQUIRED prefix match. The caller selects
    //! available groups at resume; missing or unselected groups use fresh pages for recomputation.
    //! These pages may be evicted independently without removing the REQUIRED prefix.
    OPTIONAL = 2,
};

// ---------------------------------------------------------------------------
// Buffer configuration (one buffer inside an attention or SSM layer).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// SelectionUnitLayout — where one selection unit's bytes sit inside a buffer.
//
// Buffer-relative, carrying no page offset: it describes the internal shape of the buffer,
// which is the part the model integration knows. The manager supplies the page offset from
// BufferAttr.
//
// Describes both sides of the copy: the fetch reads hot pages in ordinary slot layout and
// writes cells with the same structure, so one set of strides serves both, only the unit
// index differing. Codecs do not apply -- a codec transforms pages leaving the hot tier.
// ---------------------------------------------------------------------------
struct SelectionUnitLayout
{
    uint32_t numChunks;   //!< One per head for a head-major page; 1 otherwise.
    uint32_t chunkBytes;  //!< Contiguous bytes of one unit within one chunk.
    uint32_t chunkStride; //!< Distance between consecutive chunks.
    uint32_t unitStride;  //!< Distance between consecutive selection units.

    //! Derive the layout of a densely packed page.
    //!
    //! @param pageBytes One buffer's bytes for a whole page, i.e. BufferConfig::size.
    //! @param unitsPerPage Selection units the page holds.
    //! @param numChunks Pieces a unit is gathered from; see BufferConfig::sparseUnitChunks.
    static SelectionUnitLayout derive(size_t pageBytes, int unitsPerPage, int numChunks)
    {
        auto const chunkStride = static_cast<uint32_t>(pageBytes / static_cast<size_t>(numChunks));
        auto const chunkBytes = chunkStride / static_cast<uint32_t>(unitsPerPage);
        return SelectionUnitLayout{static_cast<uint32_t>(numChunks), chunkBytes, chunkStride, chunkBytes};
    }

    //! @param unitSize Bytes of one selection unit, i.e. size / unitsPerPage.
    void validate(size_t unitSize) const;
};

struct BufferConfig
{
    DataRole role;
    size_t size = 0; // bytes per page

    // If set, overrides tokens_per_block for this buffer.
    // Must be a divisor of KVCacheManagerConfig::tokensPerBlock.
    std::optional<int> tokensPerBlockOverride;

    bool isSparse = false; //!< Whether history uses the sparse attention lifecycle.

    //! Buffers may coalesce only when their group and expanded size match.
    //! This does not change their lifecycle or shared slot allocation.
    int coalescingGroup = 0;
    //! Pieces a selection unit is gathered from within its page. Sparse buffers only.
    //!
    //!   - **1** for a sequence-major page (`[token][head][dim]`), where a unit is one
    //!     contiguous range; also 1 for a single-head page, where the two orders coincide.
    //!   - **numHeads** for a head-major page (`[head][token][dim]`), where each head
    //!     contributes its own contiguous run.
    //!
    //! The rest of the layout follows from this, the layer's unit size and the page being
    //! densely packed, so a padded page is not expressible.
    int sparseUnitChunks = 1;

    //! Reuse policy of this buffer; nullopt inherits the layer's policy. Overrides can
    //! place buffers of one attention or SSM layer in different lifecycle groups.
    std::optional<ReusePolicy> reusePolicy;

    [[nodiscard]] ReusePolicy effectiveReusePolicy(ReusePolicy layerPolicy) const noexcept
    {
        return reusePolicy.value_or(layerPolicy);
    }

    void validate() const;

    //! Where this buffer's selection units sit inside a page, given how many the layer
    //! divides a page into. Throws when the page cannot be divided as declared.
    [[nodiscard]] SelectionUnitLayout sparseLayout(int unitsPerPage) const
    {
        auto const pieces = static_cast<size_t>(sparseUnitChunks) * static_cast<size_t>(unitsPerPage);
        TLLM_CHECK_WITH_INFO(size != 0 && pieces != 0 && size % pieces == 0,
            "BufferConfig: size must divide evenly into sparseUnitChunks * units-per-page pieces");
        return SelectionUnitLayout::derive(size, unitsPerPage, sparseUnitChunks);
    }
};

// ---------------------------------------------------------------------------
// SparseCacheConfig — one layer's sparse fetch path.
//
// Every sparse buffer of a layer shares one resolve, one resident set and one recency
// order. Each Batch row's lease holds [selection][entry] metadata: layers sharing a
// selectionId share this metadata while keeping separate data cells; otherwise each
// layer has its own selection.
//
// Its presence is what configures the fetch path. A sparse buffer in a layer without it
// still gets sparse storage and locking; it simply has no GPU cache to read through.
// ---------------------------------------------------------------------------
struct SparseCacheConfig
{
    //! Selection width: units a sparse-attention step selects for this layer.
    int numSelectedBlocks;

    //! Cells one buffer gets in a request's GPU cache, counting both populations: the
    //! history it caches and every unit still in flight.
    //!
    //! Two to four times numSelectedBlocks is the usual choice: a cache only as wide as the
    //! selection retains nothing between steps, so what it holds above the selection is
    //! what survives to be hit next step. Only one unit of it is ever in flight, because a
    //! sparse request contributes one query token per step, so the rest is all reuse.
    int maxCachedUnits;

    //! Original tokens one selection unit spans. A whole page is `tokensPerBlock`.
    //!
    //! Must divide KVCacheManagerConfig::tokensPerBlock; the quotient is how many units a
    //! page holds. It sets the sparse cache's granularity only: pages, page indices and
    //! index spaces all keep the manager's tokensPerBlock.
    int unitTokens;

    //! Layers whose selection is one selection, named so they can say so.
    //!
    //! A model that runs one indexer across several layers hands them all the same units
    //! every step. Naming that lets those layers share one resident set and one resolve,
    //! where otherwise each repeats the other's answer against its own copy of it.
    //!
    //! Unset means this layer selects for itself, and is read as the layer's own id -- so a
    //! layer that says nothing never shares, and no explicit id may name a layer that said
    //! nothing. Layers that do share must agree on the rest of this config.
    std::optional<int> selectionId;

    //! The id this layer's selection goes by, which is its own when it named none.
    [[nodiscard]] int effectiveSelectionId(LayerId layerId) const noexcept
    {
        return selectionId.value_or(layerId);
    }

    //! Selection units one page holds, given the manager's tokensPerBlock.
    [[nodiscard]] int unitsPerPage(int tokensPerBlock) const
    {
        return tokensPerBlock / unitTokens;
    }

    void validate(int tokensPerBlock) const;
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
        TLLM_CHECK(roles.insert(buf.role).second);
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

    // Private working pages never constrain prefix matching or enter the tree.
    // Every buffer of an attention layer uses this policy.
    ReusePolicy reusePolicy = ReusePolicy::REQUIRED;
    //! The sparse fetch path for this layer's sparse buffers, if it has one.
    std::optional<SparseCacheConfig> sparseCache;

    [[nodiscard]] std::optional<int> windowSize() const noexcept
    {
        return slidingWindowSize;
    }

    void validate() const;
};

// ---------------------------------------------------------------------------
// SSM (State Space Model) layer configuration.
// ---------------------------------------------------------------------------

struct SsmLayerConfig
{
    static constexpr LayerType type = LayerType::SSM;

    LayerId layerId = 0;
    std::vector<BufferConfig> buffers;
    // Default policy of the layer's buffers; BufferConfig::reusePolicy may override it per buffer.
    ReusePolicy reusePolicy = ReusePolicy::REQUIRED;

    void validate() const
    {
        detail::validateNoDuplicateBufferRoles(buffers);
        for (auto const& buf : buffers)
        {
            if (buf.tokensPerBlockOverride.has_value())
            {
                throw std::invalid_argument("tokensPerBlockOverride not supported for SSM layers");
            }
            if (buf.isSparse)
            {
                throw std::invalid_argument("Sparse buffers are only supported for attention layers");
            }
            buf.validate();
        }
    }
};

using LayerConfig = std::variant<AttentionLayerConfig, SsmLayerConfig>;

inline std::vector<BufferConfig> const& layerBuffers(LayerConfig const& layer)
{
    return std::visit([](auto const& cfg) -> std::vector<BufferConfig> const& { return cfg.buffers; }, layer);
}

// ---------------------------------------------------------------------------
// KVCacheDesc — describes one KV cache request's capacity and history length.
// Mirrors _config.py::KVCacheDesc.
// ---------------------------------------------------------------------------
struct KVCacheDesc
{
    int capacity = 0;
    int historyLength = 0;
    // Beam search does not replicate the whole cache. Blocks that lie entirely
    // inside the prompt are committed and canonicalized to beam 0 (see
    // KvCache::_appendBeams(), which skips ordinals below promptLength /
    // tokensPerBlock and every committed block); only the prompt tail onward is
    // replicated. Sizing that scales everything by beamWidth would cancel out in
    // the normalized pool ratio and leave the skew between life cycles in place,
    // so the split has to be modelled explicitly.
    int beamWidth = 1;
    int promptLength = 0;

    void validate() const
    {
        TLLM_CHECK(0 <= historyLength && historyLength <= capacity);
        TLLM_CHECK(beamWidth >= 1);
        TLLM_CHECK(0 <= promptLength);
    }

    // Value equality, mirroring the Python @dataclass(frozen=True) semantics the
    // bindings replace. Required so tests can compare descs by value.
    bool operator==(KVCacheDesc const& other) const noexcept
    {
        return capacity == other.capacity && historyLength == other.historyLength && beamWidth == other.beamWidth
            && promptLength == other.promptLength;
    }

    bool operator!=(KVCacheDesc const& other) const noexcept
    {
        return !(*this == other);
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
        TLLM_CHECK(systemPromptLength >= 0);
        for (auto const& desc : kvCaches)
        {
            desc.validate();
        }
    }

    // Value equality, mirroring the Python @dataclass(frozen=True) semantics the
    // bindings replace. Uses KVCacheDesc::operator== elementwise.
    bool operator==(BatchDesc const& other) const noexcept
    {
        return systemPromptLength == other.systemPromptLength && kvCaches == other.kvCaches;
    }

    bool operator!=(BatchDesc const& other) const noexcept
    {
        return !(*this == other);
    }
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
// Pool rebalancing configuration (mirrors _config.py::PoolRebalanceConfig).
// Tunes the auto-tuner driven by KvCacheManager::needAdjustment / adjust.
// ---------------------------------------------------------------------------
struct PoolRebalanceConfig
{
    // Minimum number of sampled (closed) KvCaches observed before rebalancing is considered.
    int minSampledKvCaches = 2000;

    // Minimum time, in seconds, between successive rebalancing adjustments.
    double cooldownSecs = 120.0;

    // Number of newly sampled KvCaches between recomputations of the target pool ratios.
    int targetRatioUpdateInterval = 100;

    // How far a pool group's current ratio may drift from its target before a rebalance is
    // triggered, as a fraction of the smaller of the two. The comparison is symmetric, so
    // 0.25 means "rebalance once one of the two exceeds the other by more than 25%".
    float ratioTolerance = 0.25f;

    // Decay factor for the exponential moving averages (reused length, capacity, history
    // length) used to compute target pool ratios. Higher values react more slowly to change.
    double movingAverageDecay = 0.9999;

    void validate() const
    {
        if (minSampledKvCaches < 0)
        {
            throw std::invalid_argument("PoolRebalanceConfig: min_sampled_kv_caches must be non-negative");
        }
        if (!std::isfinite(cooldownSecs) || cooldownSecs < 0.0)
        {
            throw std::invalid_argument("PoolRebalanceConfig: cooldown_secs must be finite and non-negative");
        }
        if (targetRatioUpdateInterval <= 0)
        {
            throw std::invalid_argument("PoolRebalanceConfig: target_ratio_update_interval must be positive");
        }
        if (!std::isfinite(ratioTolerance) || ratioTolerance <= 0.0f)
        {
            throw std::invalid_argument("PoolRebalanceConfig: ratio_tolerance must be finite and > 0");
        }
        if (!std::isfinite(movingAverageDecay) || movingAverageDecay <= 0.0 || movingAverageDecay >= 1.0)
        {
            throw std::invalid_argument("PoolRebalanceConfig: moving_average_decay must be finite and in (0, 1)");
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

    // Tokens dropped from the tail of every prefix match.
    //
    // For a pool whose KV at position i is a function of tokens [0, i] this is 0: a match of
    // m tokens proves all m are reusable. Set it to D when the pool also holds state that
    // reads D tokens ahead -- one-model speculative decoding draft layers -- where a match
    // of m only describes the first m - D positions.
    //
    // Applied inside the match so a single tree walk yields the usable depth.
    int reuseMatchBackoff = 0;

    // Constraint-based memory partitioning.
    std::vector<BatchDesc> constraints;   // batches that must always be supportable
    std::optional<BatchDesc> typicalStep; // typical step for initial ratio computation
    // One normalized hot-tier byte-quota weight per layer group. Cold initialization preserves the implied
    // layer-group slot-count proportions while accounting for cold page sizes.
    std::optional<std::vector<float>> initialPoolRatio; // overrides inferred sizing inputs

    // When set, SWA layers reuse physical pages for out-of-window blocks during prefill.
    // Scratch blocks share coalesced slot sub-pages across blocks for the currently executing
    // layer, reducing peak memory. Trade-off: KV cache reuse is degraded because scratch blocks
    // have no preserved data after the step.
    std::optional<SwaScratchReuseConfig> swaScratchReuse;

    // If true, commit() records only the minimum cache snapshot reusable at the post-call
    // numCommittedTokens. Only the minimum amount of pages required for such reuse will be
    // preserved. Required when SSM layers are present.
    bool commitMinSnapshot = false;

    // Collect V2 KV cache allocation, reuse, and transfer statistics.
    bool enableStats = true;

    // Tuning for the pool rebalancing auto-tuner. Only consulted when the executor opts in.
    PoolRebalanceConfig poolRebalance;

    // Deployment-level guarantee that no request carries multi-modal content, so token
    // sequences never contain digests. Lets block-key hashing take the digest-free fast
    // path without scanning. A per-KvCache text_only override may only tighten this
    // (a text-only deployment forbids a request claiming otherwise). Default false.
    bool textOnly = false;

    // Host memory the GPU tier's host-backed pools may take, in bytes. Unset, the storage
    // manager sets it at construction to 80% of the host memory the host tiers' quotas leave,
    // and keeps that value.
    //
    // A sparse buffer's pages are host memory, so the device quota does not size them: a slot
    // spans every pool of its group, so their row count follows from the device-backed pools.
    // This only ever lowers a slot count, and pool groups share it by the same ratio that
    // divides the device quota.
    //
    // Honored exactly, as is the feasibility floor `constraints` implies; the pool ratio is
    // what gives way between them. They can only conflict if the floors alone exceed this
    // budget, which construction rejects.
    std::optional<size_t> hostSparsePoolQuota;

    // Publish a finalized partial block for reuse when committing stops.
    // Beam search disables this while retaining full-block reuse.
    bool enablePartialCommit = true;

    // Unreferenced GPU pages below this priority are dropped instead of offloaded.
    Priority coldTierOffloadMinPriority = kDefaultColdTierOffloadMinPriority;

    bool enableSwaScratchReuse() const noexcept
    {
        return swaScratchReuse.has_value();
    }

    void validate() const;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
