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
#include "tensorrt_llm/common/cudaUtils.h"

#include <cstddef>
#include <cstdint>
#include <cuda_fp8.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

void invokeNvFp4MlaKvCacheGather(uint8_t const* dataPool, __nv_fp8_e4m3 const* scalePool, int32_t const* globalIndices,
    __nv_fp8_e4m3* output, int32_t* compactIndices, float const* globalDequantScale, int32_t numRows, int32_t topK,
    int32_t headDim, int32_t residualDim, int64_t numPoolTokens, cudaStream_t stream = 0);

size_t getNvFp4MlaContextKvCacheGatherWorkspaceSize(int32_t totalKvTokens, cudaStream_t stream = 0);

void invokeNvFp4MlaContextKvCacheGather(uint8_t const* dataPool, __nv_fp8_e4m3 const* scalePool,
    int32_t const* localTopKIndices, int32_t const* queryReqIndices, int32_t const* blockTable,
    int64_t const* cuKvLengths, __nv_fp8_e4m3* output, int32_t* compactIndices, float const* globalDequantScale,
    void* workspace, size_t workspaceSize, int32_t numQueryRows, int32_t topK, int32_t numRequests,
    int32_t maxBlocksPerRequest, int32_t totalKvTokens, int32_t outputCapacity, int32_t tokensPerBlock,
    int32_t pageStride, int32_t layerId, int32_t headDim, int32_t residualDim, int64_t numPoolTokens,
    cudaStream_t stream = 0);

} // namespace kernels

TRTLLM_NAMESPACE_END
