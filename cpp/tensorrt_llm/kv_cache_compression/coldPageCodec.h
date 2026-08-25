/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kv_cache_manager_v2/coldPageCodec.h"
#include "tensorrt_llm/kernels/nvfp4ColdPageKernels.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace tensorrt_llm::kv_cache_compression
{

namespace kv = batch_manager::kv_cache_manager_v2;

//! Native transform selected for one buffer in a planned cold-page record.
enum class ColdPageTransformKind : std::uint8_t
{
    kLosslessCopy,
    kNvfp4,
};

//! NVFP4 parameters for one independently scaled buffer.
struct Nvfp4ColdPageParams
{
    kernels::Nvfp4ColdPageRuntimeType runtimeType = kernels::Nvfp4ColdPageRuntimeType::kFloat16;
    std::int32_t numKvHeads = 0;
    std::int32_t tokensPerPage = 0;
    std::int32_t headDim = 0;
    float nvfp4ScaleOrigQuant = 1.0F;
    float nvfp4ScaleQuantOrig = 1.0F;
    float fp8ScaleOrigQuant = 1.0F;
    float fp8ScaleQuantOrig = 1.0F;
};

//! Python-authored transform and layer-relative cold offsets for one buffer.
struct ColdPageBufferPlan
{
    kv::DataRole role;
    ColdPageTransformKind transform = ColdPageTransformKind::kLosslessCopy;
    std::size_t rawBytes = 0;
    std::size_t coldDataOffset = 0;
    std::size_t coldScaleOffset = 0;
    std::optional<Nvfp4ColdPageParams> nvfp4Params;
};

//! Python-authored fixed cold record for one layer.
struct ColdPageLayerPlan
{
    kv::LayerId layerId = 0;
    std::size_t coldPageBytes = 0;
    std::size_t coldPaddingOffset = 0;
    std::size_t coldPaddingBytes = 0;
    std::vector<ColdPageBufferPlan> buffers;
};

//! Resolves declarative layer plans against KVCM's authoritative hot-pool
// descriptors.
class PlannedColdPageCodec final : public kv::IKvCacheColdPageCodec
{
public:
    explicit PlannedColdPageCodec(std::vector<ColdPageLayerPlan> layerPlans);

    bool configure(kv::PoolGroupDesc const* gpuDescs, kv::PoolGroupIndex numGpuDescs) noexcept override;

    [[nodiscard]] std::size_t queryColdPageBytes(kv::LayerGroupId layerGroupId) const noexcept override;

    [[nodiscard]] kv::LayerGroupId getBatchingLayerGroupId(kv::LayerGroupId layerGroupId) const noexcept override;

    [[nodiscard]] kv::PageIndexLocation queryPageIndexLocation(kv::LayerGroupId layerGroupId) const noexcept override;

    bool encode(kv::LayerGroupId layerGroupId, void* dstBasePtr, kv::PageIndexPair const* pageIndices,
        std::size_t numBasePages, cudaStream_t stream) noexcept override;

    bool decode(kv::LayerGroupId layerGroupId, void const* srcBasePtr, kv::PageIndexPair const* pageIndices,
        std::size_t numBasePages, cudaStream_t stream) noexcept override;

private:
    enum class ExecutionKind : std::uint8_t
    {
        kLossless,
        kPlanned,
    };

    struct LayerGroupState
    {
        ExecutionKind execution = ExecutionKind::kLossless;
        kernels::Nvfp4ColdPagePreparedPlan preparedPlan;
        std::size_t coldPageBytes = 0;
    };

    [[nodiscard]] LayerGroupState const* findLayerGroup(kv::LayerGroupId layerGroupId) const noexcept;

    std::map<kv::LayerId, ColdPageLayerPlan> mLayerPlans;
    std::map<kv::LayerGroupId, LayerGroupState> mLayerGroups;
    std::unique_ptr<kv::IKvCacheColdPageCodec> mLosslessCodec;
};

} // namespace tensorrt_llm::kv_cache_compression
