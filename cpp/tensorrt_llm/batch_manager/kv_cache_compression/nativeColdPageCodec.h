/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
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
    explicit NativeColdPageCodec(std::set<kv::LayerId> layerIds);

    bool configure(kv::PoolGroupDesc const* gpuDescs, kv::PoolGroupIndex numGpuDescs) noexcept final;

    [[nodiscard]] std::size_t queryColdPageBytes(kv::LayerGroupId layerGroupId) const noexcept final;

    [[nodiscard]] kv::LayerGroupId getBatchingLayerGroupId(kv::LayerGroupId layerGroupId) const noexcept final;

    [[nodiscard]] kv::PageIndexLocation queryPageIndexLocation(kv::LayerGroupId layerGroupId) const noexcept final;

    //! Both forward to the embedded lossless codec so fallback lifecycles keep the batched-copy
    //! registration-boundary workaround.
    [[nodiscard]] bool needsHostMemRegistration() const noexcept final;

    void registerHostMem(kv::HostMem const* memory) final;

    bool encode(kv::LayerGroupId layerGroupId, void* dstBasePtr, kv::PageIndexPair const* pageIndices,
        std::size_t numBasePages, cudaStream_t stream) noexcept final;

    bool decode(kv::LayerGroupId layerGroupId, void const* srcBasePtr, kv::PageIndexPair const* pageIndices,
        std::size_t numBasePages, cudaStream_t stream) noexcept final;

private:
    virtual std::vector<ColdPageLifecycleProperties> configureProvider(
        std::vector<ResolvedHotLifecycle> const& lifecycles)
        = 0;

    //! Enqueue only on stream; this codec drains partial submissions after a throw.
    virtual void encodeProvider(std::size_t lifecycleIndex, void* coldBase, kv::PageIndexPair const* pageIndices,
        std::size_t numPages, cudaStream_t stream)
        = 0;

    virtual void decodeProvider(std::size_t lifecycleIndex, void const* coldBase, kv::PageIndexPair const* pageIndices,
        std::size_t numPages, cudaStream_t stream)
        = 0;

    struct LayerGroupState
    {
        std::optional<std::size_t> lifecycleIndex;
        std::size_t coldPageBytes = 0;
        kv::PageIndexLocation pageIndexLocation = kv::PageIndexLocation::kBadLocation;
    };

    [[nodiscard]] LayerGroupState const* findLayerGroup(kv::LayerGroupId layerGroupId) const noexcept;

    std::set<kv::LayerId> mLayerIds;
    std::map<kv::LayerGroupId, LayerGroupState> mLayerGroups;
    std::unique_ptr<kv::IKvCacheColdPageCodec> mLosslessCodec;
};

} // namespace tensorrt_llm::kv_cache_compression
