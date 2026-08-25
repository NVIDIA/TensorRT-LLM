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

#include "tensorrt_llm/common/config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

//! Active GPU representation encoded into cold Pages.
enum class Nvfp4ColdPageRuntimeType : std::uint8_t
{
    kFloat16,
    kBfloat16,
    kFp8E4m3,
};

//! One Base Page selected for GPU-to-Host transformation.
struct Nvfp4ColdPageOffloadPageTask
{
    std::int32_t gpuPageIndex;
    std::int32_t coldPageIndex;
};

//! One Base Page selected for Host-to-GPU transformation.
struct Nvfp4ColdPageOnboardPageTask
{
    std::int32_t gpuPageIndex;
    std::int32_t coldPageIndex;
};

//! Per-buffer geometry and scales for one NVFP4 record in HND order.
//! `headDim` is a multiple of 16; `*OrigQuant` encodes and `*QuantOrig` decodes this buffer.
struct Nvfp4ColdPageKernelParams
{
    std::int32_t numKvHeads;
    std::int32_t tokensPerPage;
    std::int32_t headDim;
    float nvfp4ScaleOrigQuant;
    float nvfp4ScaleQuantOrig;
    float fp8ScaleOrigQuant;
    float fp8ScaleQuantOrig;
};

//! Transformation applied to one independently addressed hot buffer.
enum class Nvfp4ColdPageTransform : std::uint8_t
{
    kNvfp4,
    //! Byte-exact copy for an Attention side buffer such as DSA index_key.
    kLosslessCopy,
};

//! Immutable transform plan for one hot buffer and its fixed-offset cold record.
struct Nvfp4ColdPageBufferPlan
{
    std::uintptr_t rawBase;
    std::size_t rawSlotBytes;
    std::size_t rawBytes;
    std::size_t coldDataOffset;
    std::size_t coldScaleOffset;
    std::size_t coldPaddingOffset;
    std::uint32_t coldPaddingBytes;
    Nvfp4ColdPageTransform transform;
    Nvfp4ColdPageKernelParams params;
};

inline constexpr std::uint32_t kNvfp4ColdPageMaxBuffersPerLaunch = 256;

//! Configure-time launch plan for one Attention lifecycle.
struct Nvfp4ColdPagePreparedPlan
{
    std::array<Nvfp4ColdPageBufferPlan, kNvfp4ColdPageMaxBuffersPerLaunch> buffers{};
    std::uint32_t numBuffers = 0;
    std::uint32_t maxHalfGroupsPerTile = 0;
    std::size_t coldPageBytes = 0;
    Nvfp4ColdPageRuntimeType runtimeType = Nvfp4ColdPageRuntimeType::kFloat16;
};

//! Validate and freeze one lifecycle's cold-page transform plan.
[[nodiscard]] Nvfp4ColdPagePreparedPlan prepareNvfp4ColdPagePlan(std::vector<Nvfp4ColdPageBufferPlan> const& buffers,
    std::size_t coldPageBytes, Nvfp4ColdPageRuntimeType runtimeType);

//! Compress GPU Pages into mapped-Host NVFP4 records.
void invokeNvfp4ColdPageEncode(std::vector<Nvfp4ColdPageOffloadPageTask> const& pages,
    Nvfp4ColdPagePreparedPlan const& plan, void* coldBase, cudaStream_t stream);

//! Restore mapped-Host NVFP4 records into GPU Pages.
void invokeNvfp4ColdPageDecode(std::vector<Nvfp4ColdPageOnboardPageTask> const& pages,
    Nvfp4ColdPagePreparedPlan const& plan, void const* coldBase, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
