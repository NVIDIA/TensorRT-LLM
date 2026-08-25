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
#include <memory>
#include <optional>
#include <vector>

namespace tensorrt_llm::kv_cache_compression
{

namespace kv = batch_manager::kv_cache_manager_v2;

//! Per-buffer global scales used by the NVFP4 cold-page kernels.
struct Nvfp4ColdPageScales
{
    float nvfp4ScaleOrigQuant = 1.0F;
    float nvfp4ScaleQuantOrig = 1.0F;
    float fp8ScaleOrigQuant = 1.0F;
    float fp8ScaleQuantOrig = 1.0F;
};

//! Algorithm and layer-relative cold offsets for one buffer.
struct Nvfp4ColdPageBufferLayout
{
    kv::DataRole role;
    std::size_t coldDataOffset = 0;
    std::size_t coldScaleOffset = 0;
    std::optional<Nvfp4ColdPageScales> scales;
};

//! Python-authored NVFP4 record layout for one Attention layer.
struct Nvfp4ColdPageLayerLayout
{
    kv::LayerId layerId = 0;
    kernels::Nvfp4ColdPageRuntimeType runtimeType = kernels::Nvfp4ColdPageRuntimeType::kFloat16;
    std::int32_t numKvHeads = 0;
    std::int32_t tokensPerPage = 0;
    std::int32_t headDim = 0;
    std::size_t coldPageBytes = 0;
    std::size_t coldPaddingOffset = 0;
    std::vector<Nvfp4ColdPageBufferLayout> buffers;
};

//! Create an owning native codec configured by NVFP4 layer layouts.
[[nodiscard]] std::unique_ptr<kv::IKvCacheColdPageCodec> createNvfp4ColdPageCodec(
    std::vector<Nvfp4ColdPageLayerLayout> layerLayouts);

} // namespace tensorrt_llm::kv_cache_compression
