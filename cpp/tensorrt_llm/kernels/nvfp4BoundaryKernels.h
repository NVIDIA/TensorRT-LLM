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

//! Active GPU representation at the cold-page boundary.
enum class Nvfp4BoundaryRuntimeType : std::uint8_t
{
    kFloat16,
    kBfloat16,
    kFp8E4m3,
};

//! One Base Page selected for GPU-to-Host transformation.
struct Nvfp4BoundaryOffloadPageTask
{
    std::int32_t gpuPageIndex;
    std::int32_t coldPageIndex;
};

//! One Base Page selected for Host-to-GPU transformation.
struct Nvfp4BoundaryOnboardPageTask
{
    std::int32_t gpuPageIndex;
    std::int32_t coldPageIndex;
};

//! Per-buffer geometry and scales for one NVFP4 record in HND order.
//! `headDim` is a multiple of 16; `*OrigQuant` encodes and `*QuantOrig` decodes this buffer.
struct Nvfp4BoundaryKernelParams
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
enum class Nvfp4BoundaryTransform : std::uint8_t
{
    kNvfp4,
    kLossless,
};

//! Immutable transform plan for one hot buffer and its fixed-offset cold record.
struct Nvfp4BoundaryBufferPlan
{
    std::uintptr_t rawBase;
    std::size_t rawSlotBytes;
    std::size_t rawBytes;
    std::size_t coldDataOffset;
    std::size_t coldScaleOffset;
    std::size_t coldPaddingOffset;
    std::uint32_t coldPaddingBytes;
    Nvfp4BoundaryTransform transform;
    Nvfp4BoundaryKernelParams params;
};

inline constexpr std::uint32_t kNvfp4BoundaryMaxBuffersPerLaunch = 256;

//! Configure-time launch plan for one Attention lifecycle.
struct Nvfp4BoundaryPreparedPlan
{
    std::array<Nvfp4BoundaryBufferPlan, kNvfp4BoundaryMaxBuffersPerLaunch> buffers{};
    std::uint32_t numBuffers = 0;
    std::uint32_t maxTileHalfGroups = 0;
    std::size_t coldPageBytes = 0;
    Nvfp4BoundaryRuntimeType runtimeType = Nvfp4BoundaryRuntimeType::kFloat16;
};

//! Validate and freeze one lifecycle's boundary-transform plan.
[[nodiscard]] Nvfp4BoundaryPreparedPlan prepareNvfp4BoundaryPlan(std::vector<Nvfp4BoundaryBufferPlan> const& buffers,
    std::size_t coldPageBytes, Nvfp4BoundaryRuntimeType runtimeType);

//! Compress GPU Pages into mapped-Host NVFP4 records.
void invokeNvfp4BoundaryOffloadCompress(std::vector<Nvfp4BoundaryOffloadPageTask> const& pages,
    Nvfp4BoundaryPreparedPlan const& plan, void* coldBase, cudaStream_t stream);

//! Restore mapped-Host NVFP4 records into GPU Pages.
void invokeNvfp4BoundaryOnboardDecompress(std::vector<Nvfp4BoundaryOnboardPageTask> const& pages,
    Nvfp4BoundaryPreparedPlan const& plan, void const* coldBase, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
