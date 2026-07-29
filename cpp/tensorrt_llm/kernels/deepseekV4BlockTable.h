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

#include <cstdint>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

void invokeDeepseekV4ComputeSlidingBlockTables(int32_t const* blockOffsets, int32_t const* copyIdx,
    int64_t const* poolIds, bool const* validPool, int32_t const* scales, int32_t const* layerOffsets, int32_t* output,
    int32_t numPools, int32_t copyIdxCapacity, int32_t numLayers, int32_t numAttnTypes, int32_t numTables,
    int32_t maxBlocksPerSeq, cudaStream_t stream);

void invokeDeepseekV4ComputeSlidingBlockTablesWithScratch(int32_t const* blockOffsets, int32_t const* copyIdx,
    int64_t const* poolIds, bool const* validPool, int32_t const* scales, int32_t const* layerOffsets,
    int32_t const* scratchPages, int32_t const* scratchBegs, int32_t const* scratchEnds, int32_t const* scratchSlots,
    int32_t const* numContexts, int32_t* output, int32_t numPools, int32_t copyIdxCapacity, int32_t numLayers,
    int32_t numAttnTypes, int32_t numTables, int32_t maxBlocksPerSeq, int32_t scratchCapacity, int32_t maxScratchSlots,
    cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
