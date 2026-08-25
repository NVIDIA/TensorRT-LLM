/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kv_cache_manager_v2/coldPageCodec.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <vector>

namespace tensorrt_llm::kv_cache_compression
{

namespace kv = batch_manager::kv_cache_manager_v2;

//! One hot buffer resolved from KVCM's authoritative pool descriptors.
struct ResolvedColdPageBuffer
{
    std::uintptr_t rawBase = 0;
    std::size_t rawSlotBytes = 0;
    std::size_t rawBytes = 0;
};

using ResolvedColdPageLayer = std::map<kv::DataRole, ResolvedColdPageBuffer>;
using ResolvedColdPageLifecycle = std::map<kv::LayerId, ResolvedColdPageLayer>;

//! Storage properties produced while a backend prepares one lifecycle.
struct ColdPageLifecycleConfig
{
    std::size_t coldPageBytes = 0;
    kv::PageIndexLocation pageIndexLocation = kv::PageIndexLocation::kBadLocation;
};

//! Native algorithm backend retained by the generic KVCM codec adapter.
class IColdPageCodecBackend
{
public:
    virtual ~IColdPageCodecBackend() = default;

    [[nodiscard]] virtual std::set<kv::LayerId> const& getLayerIds() const noexcept = 0;

    virtual std::vector<ColdPageLifecycleConfig> configure(std::vector<ResolvedColdPageLifecycle> const& lifecycles)
        = 0;

    virtual void encode(std::size_t lifecycleIndex, void* coldBase, kv::PageIndexPair const* pageIndices,
        std::size_t numPages, cudaStream_t stream)
        = 0;

    virtual void decode(std::size_t lifecycleIndex, void const* coldBase, kv::PageIndexPair const* pageIndices,
        std::size_t numPages, cudaStream_t stream)
        = 0;
};

//! Resolves KVCM layouts, routes lifecycles, and owns one native backend.
class NativeColdPageCodec final : public kv::IKvCacheColdPageCodec
{
public:
    explicit NativeColdPageCodec(std::unique_ptr<IColdPageCodecBackend> backend);

    bool configure(kv::PoolGroupDesc const* gpuDescs, kv::PoolGroupIndex numGpuDescs) noexcept override;

    [[nodiscard]] std::size_t queryColdPageBytes(kv::LayerGroupId layerGroupId) const noexcept override;

    [[nodiscard]] kv::LayerGroupId getBatchingLayerGroupId(kv::LayerGroupId layerGroupId) const noexcept override;

    [[nodiscard]] kv::PageIndexLocation queryPageIndexLocation(kv::LayerGroupId layerGroupId) const noexcept override;

    bool encode(kv::LayerGroupId layerGroupId, void* dstBasePtr, kv::PageIndexPair const* pageIndices,
        std::size_t numBasePages, cudaStream_t stream) noexcept override;

    bool decode(kv::LayerGroupId layerGroupId, void const* srcBasePtr, kv::PageIndexPair const* pageIndices,
        std::size_t numBasePages, cudaStream_t stream) noexcept override;

private:
    struct LayerGroupState
    {
        std::optional<std::size_t> backendIndex;
        std::size_t coldPageBytes = 0;
        kv::PageIndexLocation pageIndexLocation = kv::PageIndexLocation::kBadLocation;
    };

    [[nodiscard]] LayerGroupState const* findLayerGroup(kv::LayerGroupId layerGroupId) const noexcept;

    std::unique_ptr<IColdPageCodecBackend> mBackend;
    std::map<kv::LayerGroupId, LayerGroupState> mLayerGroups;
    std::unique_ptr<kv::IKvCacheColdPageCodec> mLosslessCodec;
};

} // namespace tensorrt_llm::kv_cache_compression
