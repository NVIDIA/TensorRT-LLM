/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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
    int32_t* compressedLocalIndices, int32_t* topkLensRatio1, int32_t* topkLensRatio4,
    int32_t* topkLensRatio128, int32_t numTokens, int32_t windowSize, int32_t maxCompressedIndices,
    int32_t sparseMlaTopk, int32_t swaStride, int32_t compressedStride, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
