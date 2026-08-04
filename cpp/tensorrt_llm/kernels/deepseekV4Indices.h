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
#include <cuda_runtime_api.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Builds the DeepSeek-V4 SWA / compressed local index tables and the per-ratio
// sparse-MLA topk lengths from `tokenPositions`. Any of the three topkLens
// pointers may be null when that compression ratio is not configured.
void invokeDeepseekV4ComputeIndices(int32_t const* tokenPositions, int32_t* swaLocalIndices,
    int32_t* compressedLocalIndices, int32_t* topkLensRatio1, int32_t* topkLensRatio4, int32_t* topkLensRatio128,
    int32_t numTokens, int32_t windowSize, int32_t maxCompressedIndices, int32_t sparseMlaTopk, int32_t swaStride,
    int32_t compressedStride, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
