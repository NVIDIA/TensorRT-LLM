/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/runtime/common.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace w4a16_nvfp4
{

using SizeType32 = tensorrt_llm::runtime::SizeType32;

static constexpr SizeType32 kScaleGranularity = 16;
static constexpr SizeType32 kScaleRowsPerTile = 128;
static constexpr SizeType32 kPackedScaleColsPerTile = 4;
static constexpr SizeType32 kScaleTileElements = 512;

__host__ __device__ inline SizeType32 getScaleIndex(SizeType32 rowIdx, SizeType32 scaleColIdx, SizeType32 k)
{
    SizeType32 const numScaleCols = k / kScaleGranularity;
    SizeType32 const numScaleColTiles = (numScaleCols + kPackedScaleColsPerTile - 1) / kPackedScaleColsPerTile;
    SizeType32 const tileOffset
        = ((rowIdx / kScaleRowsPerTile) * numScaleColTiles + scaleColIdx / kPackedScaleColsPerTile)
        * kScaleTileElements;
    return tileOffset + (rowIdx % 32) * 16 + ((rowIdx % kScaleRowsPerTile) / 32) * 4
        + scaleColIdx % kPackedScaleColsPerTile;
}

} // namespace w4a16_nvfp4
} // namespace kernels

TRTLLM_NAMESPACE_END
