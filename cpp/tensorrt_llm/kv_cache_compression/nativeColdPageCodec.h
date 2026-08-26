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
struct ResolvedHotBuffer
{
    std::uintptr_t rawBase = 0;
    std::size_t rawSlotBytes = 0;
    std::size_t rawBytes = 0;
};

using ResolvedHotLayer = std::map<kv::DataRole, ResolvedHotBuffer>;

//! One KVCM lifecycle resolved into its hot buffers.
struct ResolvedHotLifecycle
{
    kv::LifeCycleId lifeCycleId{-1};
    std::map<kv::LayerId, ResolvedHotLayer> layers;
};

//! Storage properties produced while an algorithm prepares one lifecycle.
struct ColdPageLifecycleProperties
{
    std::size_t coldPageBytes = 0;
    kv::PageIndexLocation pageIndexLocation = kv::PageIndexLocation::kBadLocation;
};

//! Resolves KVCM layouts and routes lifecycles for one native compression method.
class NativeColdPageCodec : public kv::IKvCacheColdPageCodec
{
public:
    bool configure(kv::PoolGroupDesc const* gpuDescs, kv::PoolGroupIndex numGpuDescs) noexcept final;

    [[nodiscard]] std::size_t queryColdPageBytes(kv::LayerGroupId layerGroupId) const noexcept final;

    [[nodiscard]] kv::LayerGroupId getBatchingLayerGroupId(kv::LayerGroupId layerGroupId) const noexcept final;

    [[nodiscard]] kv::PageIndexLocation queryPageIndexLocation(kv::LayerGroupId layerGroupId) const noexcept final;

    bool encode(kv::LayerGroupId layerGroupId, void* dstBasePtr, kv::PageIndexPair const* pageIndices,
        std::size_t numBasePages, cudaStream_t stream) noexcept final;

    bool decode(kv::LayerGroupId layerGroupId, void const* srcBasePtr, kv::PageIndexPair const* pageIndices,
        std::size_t numBasePages, cudaStream_t stream) noexcept final;

private:
    [[nodiscard]] virtual std::set<kv::LayerId> const& getLayerIds() const noexcept = 0;

    virtual std::vector<ColdPageLifecycleProperties> configureAlgorithm(
        std::vector<ResolvedHotLifecycle> const& lifecycles)
        = 0;

    //! Enqueue only on stream; drain earlier partial submissions before throwing.
    virtual void encodeAlgorithm(std::size_t planIndex, void* coldBase, kv::PageIndexPair const* pageIndices,
        std::size_t numPages, cudaStream_t stream)
        = 0;

    virtual void decodeAlgorithm(std::size_t planIndex, void const* coldBase, kv::PageIndexPair const* pageIndices,
        std::size_t numPages, cudaStream_t stream)
        = 0;

    struct LayerGroupState
    {
        std::optional<std::size_t> planIndex;
        std::size_t coldPageBytes = 0;
        kv::PageIndexLocation pageIndexLocation = kv::PageIndexLocation::kBadLocation;
    };

    [[nodiscard]] LayerGroupState const* findLayerGroup(kv::LayerGroupId layerGroupId) const noexcept;

    std::map<kv::LayerGroupId, LayerGroupState> mLayerGroups;
    std::unique_ptr<kv::IKvCacheColdPageCodec> mLosslessCodec;
};

} // namespace tensorrt_llm::kv_cache_compression
