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

#include "kv_cache_manager_v2/coldPageCodec.h"
#include "tensorrt_llm/kernels/nvfp4BoundaryKernels.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

namespace tensorrt_llm::kv_cache_compression
{

namespace kv = batch_manager::kv_cache_manager_v2;

//! Per-layer geometry and calibration for NVFP4 cold Pages.
struct Nvfp4ColdPageLayerConfig
{
    kv::LayerId layerId = 0;
    kernels::Nvfp4BoundaryRuntimeType runtimeType = kernels::Nvfp4BoundaryRuntimeType::kFloat16;
    std::int32_t numKvHeads = 0;
    std::int32_t tokensPerPage = 0;
    std::int32_t headDim = 0;
    std::array<float, 2> nvfp4ScaleOrigQuant{};
    std::array<float, 2> nvfp4ScaleQuantOrig{};
    std::array<float, 2> fp8ScaleOrigQuant{1.0F, 1.0F};
    std::array<float, 2> fp8ScaleQuantOrig{1.0F, 1.0F};
};

//! NVFP4 codec for compact Attention records with lossless side-buffer spans.
class Nvfp4ColdPageCodec final : public kv::IKvCacheColdPageCodec
{
public:
    explicit Nvfp4ColdPageCodec(std::vector<Nvfp4ColdPageLayerConfig> layerConfigs);

    bool configure(kv::PoolGroupDesc const* gpuDescs, kv::PoolGroupIndex numGpuDescs) noexcept override;

    [[nodiscard]] std::size_t queryColdPageBytes(kv::LayerGroupId layerGroupId) const noexcept override;

    [[nodiscard]] kv::LayerGroupId getBatchingLayerGroupId(kv::LayerGroupId layerGroupId) const noexcept override;

    [[nodiscard]] kv::PageIndexLocation queryPageIndexLocation(kv::LayerGroupId layerGroupId) const noexcept override;

    bool encode(kv::LayerGroupId layerGroupId, void* dstBasePtr, kv::PageIndexPair const* pageIndices,
        std::size_t numBasePages, cudaStream_t stream) noexcept override;

    bool decode(kv::LayerGroupId layerGroupId, void const* srcBasePtr, kv::PageIndexPair const* pageIndices,
        std::size_t numBasePages, cudaStream_t stream) noexcept override;

private:
    enum class Transform
    {
        kNvfp4Attention,
        kLosslessConcat,
    };

    struct LayerGroupState
    {
        Transform transform = Transform::kLosslessConcat;
        kernels::Nvfp4BoundaryPreparedPlan preparedPlan;
        std::size_t coldPageBytes = 0;
    };

    [[nodiscard]] LayerGroupState const* findLayerGroup(kv::LayerGroupId layerGroupId) const noexcept;

    std::map<kv::LayerId, Nvfp4ColdPageLayerConfig> mLayerConfigs;
    std::map<kv::LayerGroupId, LayerGroupState> mLayerGroups;
    std::unique_ptr<kv::IKvCacheColdPageCodec> mLosslessCodec;
};

} // namespace tensorrt_llm::kv_cache_compression
